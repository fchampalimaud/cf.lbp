"""
Smoke tests for the 2D simulator core.

Run from simulation/2d/:
    pytest tests/test_smoke.py -v

Three groups:
  1. Layer math contracts  — LeakyLayer does what it claims
  2. Serialisation round-trip — save → load preserves params
  3. Visualiser build     — NetworkVisualizerWindow.build() doesn't crash
                            (requires pytest-qt; skipped otherwise)
"""
import sys, os, json
import numpy as np
import pytest
import torch

# Both simulation/2d/ and simulation/2d/src/ must be on the path.
# LBPSimulator.py does this at startup; tests must do it manually.
_HERE    = os.path.dirname(__file__)
_SIM2D   = os.path.abspath(os.path.join(_HERE, '..'))
_SRC     = os.path.join(_SIM2D, 'src')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
for _p in (_SRC, _SIM2D):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_circuit():
    """Minimal circuit: one GradientSensor → one LeakyLayer."""
    from sensors import GradientSensor
    from neurons import LeakyLayer
    from circuit_model import CircuitModel, Connection

    s  = GradientSensor(name='light', n=2)
    l  = LeakyLayer(name='l1', n=2, tau_rise=0.1, tau_decay=0.1)
    W  = np.ones((2, 2), dtype=np.float32)
    c  = Connection(src='light', tgt='l1', W=W)
    return CircuitModel(sensors=[s], layers=[l], connections=[c])


# ── 1. Layer math contracts ───────────────────────────────────────────────────

def test_leaky_decays_to_zero():
    """With zero input, output should decay toward zero."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=2, tau_rise=0.05, tau_decay=0.05)
    layer._ensure_n(2)
    layer.output = torch.ones(2)
    for _ in range(200):
        layer.step(torch.zeros(2), dt=0.01)
    assert layer.output.abs().max().item() < 0.01


def test_leaky_converges_to_input():
    """With constant input and linear activation, output must converge to that input."""
    from neurons import LeakyLayer
    # Use linear activation to test dynamics independently of the nonlinearity.
    # (Default relu would clamp negative targets to 0 — tested separately below.)
    layer = LeakyLayer(name='t', n=2, tau_rise=0.05, tau_decay=0.05, activation='linear')
    layer._ensure_n(2)
    target = torch.tensor([1.0, -0.5])
    for _ in range(200):
        layer.step(target, dt=0.01)
    assert torch.allclose(layer.output, target, atol=0.01)


def test_leaky_relu_clamps_negative():
    """With relu activation (the default), negative inputs must produce zero output."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=1, tau_rise=0.05, tau_decay=0.05)  # default: relu
    layer._ensure_n(1)
    for _ in range(200):
        layer.step(torch.tensor([-1.0]), dt=0.01)
    assert layer.output.item() == pytest.approx(0.0)


def test_hard_sigmoid_matches_torch_and_numpy_paths():
    """hard_sigmoid must match PyTorch's own F.hardsigmoid on the torch path
    (clip(x/6 + 0.5, 0, 1)) and agree exactly with the numpy path used by
    sensors — the two branches of _activate must never silently diverge."""
    import torch.nn.functional as F
    from neurons_base import _activate

    x_t = torch.tensor([-5.0, -3.0, 0.0, 3.0, 5.0])
    x_n = np.array([-5.0, -3.0, 0.0, 3.0, 5.0])

    torch_out = _activate(x_t, 'hard_sigmoid')
    numpy_out = _activate(x_n, 'hard_sigmoid')

    assert torch.allclose(torch_out, F.hardsigmoid(x_t))
    assert torch.allclose(torch_out, torch.as_tensor(numpy_out, dtype=torch.float32))


def test_leaky_tau_decay_none_holds_value():
    """tau_decay=None must disable decay entirely: value holds when input drops."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=1, tau_rise=0.05, tau_decay=None, activation='linear')
    layer._ensure_n(1)
    for _ in range(200):
        layer.step(torch.tensor([1.0]), dt=0.01)
    assert layer.output.item() == pytest.approx(1.0, abs=0.01)
    for _ in range(200):
        layer.step(torch.tensor([0.0]), dt=0.01)
    assert layer.output.item() == pytest.approx(1.0, abs=0.01)   # held, not decayed


def test_leaky_asymmetric_tau_rise_and_decay_differ():
    """tau_rise != tau_decay (both set, neither None) must keep using distinct
    per-regime time constants — regression test for the tau_rise==tau_decay
    fast path added to DynamicsBase._apply_leaky (a perf fix: most layers in
    practice use symmetric tau, so the torch.where/torch.full_like branch was
    pure overhead in the common case). A fast rise / slow decay layer must
    reach its target quickly but relax back to zero much more slowly, and the
    two convergence rates must differ by roughly the tau ratio."""
    from neurons import LeakyLayer

    fast_rise_slow_decay = LeakyLayer(name='t', n=1, tau_rise=0.02, tau_decay=0.2,
                                       activation='linear')
    fast_rise_slow_decay._ensure_n(1)
    for _ in range(50):   # 0.5s @ dt=0.01, several tau_rise time constants
        fast_rise_slow_decay.step(torch.tensor([1.0]), dt=0.01)
    assert fast_rise_slow_decay.output.item() == pytest.approx(1.0, abs=0.01)

    fast_rise_slow_decay.step(torch.tensor([0.0]), dt=0.01)
    after_one_decay_step = fast_rise_slow_decay.output.item()
    # tau_decay=0.2 is 10x tau_rise=0.02 -> after one dt=0.01 step (half of
    # tau_decay), the value should have dropped only modestly, not rushed to 0.
    assert 0.9 < after_one_decay_step < 1.0

    for _ in range(500):   # let it fully relax on the slow decay branch
        fast_rise_slow_decay.step(torch.tensor([0.0]), dt=0.01)
    assert fast_rise_slow_decay.output.item() == pytest.approx(0.0, abs=0.01)


def test_layer_output_mode_derivative_is_rate_of_change():
    """output_mode='derivative' is the generic DynamicsBase mechanism shared by
    every layer/sensor (see DynamicsBase._apply_output_mode) — it replaces the
    raw summed input u with a finite difference between consecutive ticks'
    raw input, BEFORE the leaky filter/activation run. tau_rise=0 bypasses
    leaky filtering (x=u directly) and activation='linear' is a passthrough,
    so the final output equals the transformed u exactly, isolating this
    contract from the rest of the pipeline. Layers (unlike sensors) don't
    special-case the first tick: _prev_out starts at a zero buffer, so the
    first call's "previous" value is 0 — a documented quirk, not something
    this test is trying to hide.
    """
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=1, tau_rise=0.0, activation='linear', output_mode='derivative')
    layer._ensure_n(1)
    layer.step(torch.tensor([1.0]), dt=0.1)
    out2 = layer.step(torch.tensor([3.0]), dt=0.1)
    assert out2.item() == pytest.approx(20.0)   # (3.0 - 1.0) / 0.1


def test_layer_output_mode_integral_accumulates():
    """output_mode='integral' replaces the raw summed input u with the running
    ∫u dt (forward-Euler) before the leaky filter/activation run. tau_rise=0
    and activation='linear' make the final output equal that integral exactly."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=1, tau_rise=0.0, activation='linear', output_mode='integral')
    layer._ensure_n(1)
    for _ in range(5):
        layer.step(torch.tensor([2.0]), dt=0.1)
    assert layer.output.item() == pytest.approx(1.0)   # 2.0 * 0.1 * 5


def test_output_mode_transforms_raw_input_not_leaky_filtered_output():
    """Regression for the original design mistake: output_mode used to
    transform the layer's final (leaky-filtered, activated) output, so an
    integral kept growing during the leaky filter's own decay tail even after
    the true raw input returned to zero — it never touched the raw signal.
    Now output_mode runs on the raw summed input u BEFORE the leaky filter
    (tau_rise/tau_decay=0.1, a real lag, unlike the tau_rise=0 isolation used
    above), so the internal _integral buffer must stop growing the instant
    the raw input hits zero, regardless of how far the leaky filter's own
    state x still has to fall.
    """
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=1, tau_rise=0.1, tau_decay=0.1,
                       activation='linear', output_mode='integral')
    layer._ensure_n(1)
    layer.step(torch.tensor([2.0]), dt=0.1)
    layer.step(torch.tensor([2.0]), dt=0.1)
    assert layer._integral.item() == pytest.approx(0.4)   # 2.0 * 0.1 * 2

    layer.step(torch.tensor([0.0]), dt=0.1)
    assert layer._integral.item() == pytest.approx(0.4)   # unchanged: raw input was 0


def test_output_mode_invalid_raises():
    """DynamicsBase._init_dynamics validates output_mode against OUTPUT_MODES,
    mirroring _activate's unknown-activation error."""
    from neurons import LeakyLayer
    with pytest.raises(ValueError, match='output_mode'):
        LeakyLayer(name='bad', n=1, output_mode='bogus')


def test_sensor_output_mode_derivative_and_integral():
    """BaseSensor._apply_output_mode (invoked via _process()) supports the
    same none/derivative/integral contract as layers, but with sensor-specific
    'zero on first tick' semantics for derivative (no artificial first-tick
    jump, unlike the layer-side zero-buffer quirk tested above)."""
    from sensors import GradientSensor
    from sim_config import SimConfig

    cfg = SimConfig()
    deriv = GradientSensor(name='s', n=1, output_mode='derivative')
    out1 = deriv._process(np.array([1.0]), cfg)
    assert out1[0] == pytest.approx(0.0)
    out2 = deriv._process(np.array([3.0]), cfg)
    assert out2[0] == pytest.approx((3.0 - 1.0) / cfg.dt)

    integ = GradientSensor(name='s2', n=1, output_mode='integral')
    out = None
    for _ in range(5):
        out = integ._process(np.array([2.0]), cfg)
    assert out[0] == pytest.approx(2.0 * cfg.dt * 5)


def test_modulator_row_transform_derivative():
    """LayerBase._transform_modulator_value's 'derivative' mode computes the
    rate of change of a raw modulator-bus reading, keyed by (name, mode) in
    self._mod_row_state — independent of the layer's own output_mode state
    (_prev_out/_integral). Zero on the first call for a given key, matching
    _apply_output_mode's own first-tick convention."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=1)
    key = ('dopamine', 'derivative')
    v1 = layer._transform_modulator_value(key, 'derivative', 1.0, dt=0.1)
    assert v1 == pytest.approx(0.0)
    v2 = layer._transform_modulator_value(key, 'derivative', 3.0, dt=0.1)
    assert v2 == pytest.approx(20.0)   # (3.0 - 1.0) / 0.1


def test_modulator_row_transform_integral():
    """'integral' mode accumulates the raw modulator reading over time
    (forward-Euler), same math as _apply_output_mode's integral branch but
    tracked per-(name, mode) key instead of a single per-layer buffer."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=1)
    key = ('dopamine', 'integral')
    total = None
    for _ in range(5):
        total = layer._transform_modulator_value(key, 'integral', 2.0, dt=0.1)
    assert total == pytest.approx(1.0)   # 2.0 * 0.1 * 5


def test_modulator_row_state_independent_per_key():
    """Two different (name, mode) subscriptions on the same layer must not
    share state, and modulator-row state must not collide with the layer's
    own output_mode state — both are 'derivative' here, on purpose."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=1, tau_rise=0.0, activation='linear', output_mode='derivative')
    layer._ensure_n(1)
    layer.step(torch.tensor([100.0]), dt=0.1)   # drives the layer's OWN _prev_out to 100

    key_a = ('dopamine', 'derivative')
    key_b = ('serotonin', 'derivative')
    v1 = layer._transform_modulator_value(key_a, 'derivative', 1.0, dt=0.1)
    assert v1 == pytest.approx(0.0)          # first call for key_a: not influenced by _prev_out=100
    v2 = layer._transform_modulator_value(key_b, 'derivative', 1.0, dt=0.1)
    assert v2 == pytest.approx(0.0)          # first call for key_b too: independent of key_a's state
    v3 = layer._transform_modulator_value(key_a, 'derivative', 5.0, dt=0.1)
    assert v3 == pytest.approx(40.0)         # (5.0 - 1.0) / 0.1 — key_a's own history, untouched by key_b


def test_learning_layer_drives_plasticity_row_gates_on_threshold():
    """A `modulators` row flagged drives_plasticity=True only contributes to
    reward on ticks where its mode-transformed value crosses `threshold` —
    the generalized replacement for the old always-on reward_modulator field.
    DeltaLayer's rule (delta = r - V) moves weight from V=0 whenever r != 0,
    isolating the threshold gate from any confound with V needing to be
    nonzero first (as ThreeFactorLayer's r*V rule would require)."""
    from network_runner import step_network
    from neurons import ConstantLayer, DeltaLayer
    from circuit_model import Connection
    from brain_base import BaseBrain

    dan = ConstantLayer(name='dan', value=0.0, n=1, neuromodulator_transmitter='dopamine')
    src = ConstantLayer(name='src', value=1.0, n=1)
    learn = DeltaLayer(name='learn', n=1, alpha_pos=1.0,
                       modulators=[('dopamine', 1.0, 'none', 'absolute', True, 0.5)])
    conn = Connection(src='src', tgt='learn', W=np.zeros((1, 1), dtype=np.float32))

    brain = BaseBrain()
    brain.sensors     = []
    brain.layers      = [dan, src, learn]
    brain.connections = [conn]
    brain.src         = src   # network_runner reads connection sources via getattr(brain, conn.src)

    dan.output = torch.tensor([0.2])   # below threshold 0.5
    step_network(brain, dt=0.01)       # default dt: learning rate applies as-is
    assert conn.W[0, 0] == pytest.approx(0.0)

    dan.output = torch.tensor([0.9])   # above threshold 0.5
    step_network(brain, dt=0.01)
    assert conn.W[0, 0] == pytest.approx(0.9)   # alpha_pos=1.0 * (0.9 - 0) * s(1.0)


def test_learning_layer_reward_modulator_backcompat_default_is_none():
    """reward_modulator now defaults to None (not 'dopamine') so a freshly
    constructed learning layer never silently picks up a stray 'dopamine'
    channel via the legacy field — only an explicit drives_plasticity row,
    or an explicitly-set reward_modulator (e.g. from old saved JSON, which
    always wrote this field explicitly before it left param_defs()), should
    drive plasticity."""
    from neurons import DeltaLayer
    layer = DeltaLayer(name='learn', n=1)
    assert layer.reward_modulator is None


def test_learning_layer_reward_modulator_explicit_value_still_works():
    """Back-compat: a layer constructed with an explicit reward_modulator
    (as happens when loading old saved JSON, which always serialized this
    field) still drives plasticity via that legacy channel, additively with
    any drives_plasticity rows — network_runner.py sums both."""
    from network_runner import step_network
    from neurons import ConstantLayer, DeltaLayer
    from circuit_model import Connection
    from brain_base import BaseBrain

    dan = ConstantLayer(name='dan', value=0.7, n=1, neuromodulator_transmitter='dopamine')
    src = ConstantLayer(name='src', value=1.0, n=1)
    learn = DeltaLayer(name='learn', n=1, alpha_pos=1.0, reward_modulator='dopamine')
    conn = Connection(src='src', tgt='learn', W=np.zeros((1, 1), dtype=np.float32))

    brain = BaseBrain()
    brain.sensors     = []
    brain.layers      = [dan, src, learn]
    brain.connections = [conn]
    brain.src         = src   # network_runner reads connection sources via getattr(brain, conn.src)

    dan.output = torch.tensor([0.7])
    step_network(brain, dt=0.01)   # DeltaLayer credits this tick's input (s = 1.0)
    assert conn.W[0, 0] == pytest.approx(0.7)   # no threshold on the legacy path — always applied


def _make_snapshot_circuit(threshold=0.5):
    """cpu4 (teach source) + gate → mem (SnapshotLayer) → cpu1 (readout).

    cpu4 → mem is a real, ordinary connection (not a name-lookup) — mem's
    step_td identifies it by conn.src == teach_source and reads its raw
    value, excluding it from mem's own output.
    """
    from neurons import ConstantLayer, SnapshotLayer, LeakyLayer
    from circuit_model import Connection
    from brain_base import BaseBrain

    cpu4 = ConstantLayer(name='cpu4', value=[0.0, 0.0, 0.0], n=3)
    gate = ConstantLayer(name='gate', value=0.0, n=1)
    dan  = ConstantLayer(name='dan', value=0.0, n=1, neuromodulator_transmitter='dopamine')
    mem  = SnapshotLayer(name='mem', n=1, teach_source='cpu4', tau_rise=0.0, activation='linear',
                         modulators=[('dopamine', 1.0, 'none', 'absolute', True, threshold)])
    cpu1 = LeakyLayer(name='cpu1', n=3, tau_rise=0.0, activation='linear')

    teach_conn = Connection(src='cpu4', tgt='mem', W=np.ones((1, 3), dtype=np.float32))
    gate_conn  = Connection(src='gate', tgt='mem', W=np.array([[1.0]], dtype=np.float32))
    out_conn   = Connection(src='mem',  tgt='cpu1', W=np.zeros((3, 1), dtype=np.float32))

    brain = BaseBrain()
    brain.sensors     = []
    brain.layers      = [cpu4, gate, dan, mem, cpu1]
    brain.connections = [teach_conn, gate_conn, out_conn]
    brain.cpu4        = cpu4
    brain.gate        = gate
    brain.mem         = mem
    return brain, cpu4, gate, dan, mem, out_conn


def test_snapshot_layer_teach_connection_excluded_from_output():
    """The teach connection's value never reaches mem's own output — only
    gate connections do, regardless of how large cpu4's value is."""
    from network_runner import step_network

    brain, cpu4, gate, dan, mem, out_conn = _make_snapshot_circuit()

    cpu4.output = torch.tensor([5.0, 5.0, 5.0])
    gate.output = torch.tensor([0.0])
    step_network(brain, dt=0.1)
    assert mem.output.item() == pytest.approx(0.0)   # gate is 0 — cpu4's huge value has no effect

    gate.output = torch.tensor([1.0])
    step_network(brain, dt=0.1)
    assert mem.output.item() == pytest.approx(1.0)   # driven only by gate, not by cpu4


def test_snapshot_layer_write_overwrites_outgoing_weights_exactly():
    """On a drives_plasticity trigger, every outgoing connection's weights
    are hard-set to -teach_val (Le Moël Eq. 14) — not nudged."""
    from network_runner import step_network

    brain, cpu4, gate, dan, mem, out_conn = _make_snapshot_circuit(threshold=0.5)

    cpu4.output = torch.tensor([2.0, -3.0, 1.0])
    dan.output  = torch.tensor([0.9])   # above threshold
    step_network(brain, dt=0.1)

    assert out_conn.W[:, 0] == pytest.approx([-2.0, 3.0, -1.0])


def test_snapshot_layer_outgoing_weights_frozen_between_triggers():
    """Once written, outgoing weights stay exactly as they were — they do
    not track cpu4's value on ticks where the reward doesn't cross threshold,
    no matter how much cpu4 changes in the meantime."""
    from network_runner import step_network

    brain, cpu4, gate, dan, mem, out_conn = _make_snapshot_circuit(threshold=0.5)

    cpu4.output = torch.tensor([2.0, -3.0, 1.0])
    dan.output  = torch.tensor([0.9])
    step_network(brain, dt=0.1)
    written = out_conn.W.copy()

    dan.output  = torch.tensor([0.0])            # below threshold now
    cpu4.output = torch.tensor([100.0, 100.0, 100.0])   # cpu4 changes drastically
    step_network(brain, dt=0.1)

    assert out_conn.W == pytest.approx(written)   # unchanged despite cpu4's drift


def test_connection_kind_classifies_teach_before_td():
    """connection_kind must return TEACH for the src==teach_source connection
    into a SnapshotLayer, and TD for that same layer's other (gate) incoming
    connection — TEACH is checked with higher priority than TD since a
    SnapshotLayer is itself a LearningLayerBase. connection_kind only reads
    class-level _CK_*/_DENSE_THRESHOLD constants, so a real Qt widget (and
    pytest-qt) isn't needed — __new__ skips __init__ entirely."""
    from network_viz import NetworkVisualizerWindow
    from network_viz_layout import LayoutEngine
    from neurons import ConstantLayer, SnapshotLayer

    cpu4 = ConstantLayer(name='cpu4', value=[0.0, 0.0, 0.0], n=3)
    gate = ConstantLayer(name='gate', value=0.0, n=1)
    mem  = SnapshotLayer(name='mem', n=1, teach_source='cpu4')

    win = NetworkVisualizerWindow.__new__(NetworkVisualizerWindow)
    engine = LayoutEngine(win)
    teach_kind = engine.connection_kind(cpu4, mem, np.ones((1, 3)), 3, 1)
    gate_kind  = engine.connection_kind(gate, mem, np.ones((1, 1)), 1, 1)
    assert teach_kind == win._CK_TEACH
    assert gate_kind == win._CK_TD


def test_reichardt_integer_shift_matches_bilinear():
    """Reichardt2dLayer's fast integer-pixel _shift (used for axis-aligned
    direction sets, e.g. n_directions=1/2/4 with a multiple-of-90° init_direction)
    must produce identical output to the general bilinear grid_sample path.

    Regression test for the perf fix: _shift used to rebuild a grid_sample
    grid from scratch every tick even for exact integer offsets, dominating
    per-tick cost for any brain using this layer.
    """
    import torch.nn.functional as F
    from neurons_vision import Reichardt2dLayer

    def bilinear_shift(img, dy, dx):
        H, W = img.shape[-2], img.shape[-1]
        ys = torch.arange(H, dtype=torch.float32) + dy
        xs = torch.arange(W, dtype=torch.float32) + dx
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        norm_x = grid_x / max(W - 1, 1) * 2 - 1
        norm_y = grid_y / max(H - 1, 1) * 2 - 1
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
        sampled = F.grid_sample(img.unsqueeze(0), grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        return sampled.squeeze(0)

    layer = Reichardt2dLayer(n_directions=4, offset=1, in_ch=1, frame_h=12, frame_w=16,
                             name='r')
    torch.manual_seed(0)
    img = torch.rand(1, 12, 16)
    for dy, dx in layer._direction_vectors():   # exact integers for n_directions=4
        fast = layer._shift(img, dy, dx)
        ref  = bilinear_shift(img, dy, dx)
        assert torch.allclose(fast, ref, atol=1e-5)

    # Fractional (non-axis-aligned) direction sets must still fall back correctly.
    layer8 = Reichardt2dLayer(n_directions=8, offset=1, in_ch=1, frame_h=12, frame_w=16,
                              name='r8')
    for dy, dx in layer8._direction_vectors():
        fast = layer8._shift(img, dy, dx)
        ref  = bilinear_shift(img, dy, dx)
        assert torch.allclose(fast, ref, atol=1e-5)


def test_reset_zeroes_state():
    """`reset()` must bring output back to zero regardless of prior activity."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=2, tau_rise=0.05, tau_decay=0.05)
    layer._ensure_n(2)
    for _ in range(50):
        layer.step(torch.ones(2), dt=0.01)
    assert layer.output.abs().max().item() > 0.1   # confirm it charged up
    layer.reset()
    assert torch.all(layer.output == 0)


def test_ensure_n_raises_on_mismatch():
    """`_ensure_n` must raise when called with a different n than declared."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=2)
    with pytest.raises(ValueError):
        layer._ensure_n(5)


def test_ensure_n_initialises_when_none():
    """`_ensure_n` must initialise buffers when n was not set at construction."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t')     # n=None
    assert layer.output is None
    layer._ensure_n(3)
    assert layer.output is not None
    assert layer.output.shape == (3,)


def test_x0_seeds_initial_state():
    """x0 seeds _x (and, independently, the output_mode='integral' accumulator)
    at construction — both immediately (n given up front) and via the deferred
    _ensure_n path (n inferred later from a connection). Needed for bounded
    integrator neurons whose resting point isn't the edge of their range (e.g.
    Stone et al. 2017's CPU4 path-integration memory, clipped to [0, 1], must
    start at 0.5 to move both up and down)."""
    from neurons import LeakyLayer
    immediate = LeakyLayer(name='t1', n=2, x0=0.5)
    assert torch.equal(immediate._x, torch.full((2,), 0.5))
    assert torch.equal(immediate._integral, torch.full((2,), 0.5))

    deferred = LeakyLayer(name='t2', x0=0.5)   # n=None
    deferred._ensure_n(3)
    assert torch.equal(deferred._x, torch.full((3,), 0.5))


def test_x0_per_neuron_list():
    """x0 also accepts a per-neuron list, mirroring ConstantLayer's value handling."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=3, x0=[0.1, 0.2, 0.3])
    assert torch.allclose(layer._x, torch.tensor([0.1, 0.2, 0.3]))


def test_x0_wrong_length_raises():
    from neurons import LeakyLayer
    with pytest.raises(ValueError, match='x0'):
        LeakyLayer(name='t', n=3, x0=[0.1, 0.2])


def test_x0_restored_on_reset_not_zeroed():
    """reset() must bring _x back to x0, not to a hardcoded zero — otherwise a
    bounded integrator (x0=0.5) would come back from reset at the wrong end
    of its range every time the user resets the simulation."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=1, tau_rise=0.05, tau_decay=0.05, x0=0.5)
    layer._ensure_n(1)
    for _ in range(50):
        layer.step(torch.tensor([0.0]), dt=0.01)   # drive x away from x0
    assert layer._x.item() < 0.4
    layer.reset()
    assert layer._x.item() == pytest.approx(0.5)


def test_x0_seeds_integral_accumulator_not_bias():
    """x0 sets the one-time starting point of the output_mode='integral'
    accumulator; unlike bias (added every tick), a zero raw input must leave
    the accumulator exactly at x0 rather than drifting."""
    from neurons import LeakyLayer
    layer = LeakyLayer(name='t', n=1, tau_rise=0.0, activation='linear',
                       output_mode='integral', x0=0.5)
    layer._ensure_n(1)
    out = layer.step(torch.tensor([0.0]), dt=0.1)
    assert out.item() == pytest.approx(0.5)
    out = layer.step(torch.tensor([2.0]), dt=0.1)
    assert out.item() == pytest.approx(0.7)   # 0.5 + 2.0*0.1


def test_collision_sensor_silent_when_never_hit():
    """noise_std > 0 must not produce any output while the sensor never registers a hit.

    Regression test: CollisionSensor.sample() already adds its own hit-gated noise;
    _process()'s generic noise stage used to add a second, ungated pass on top of it,
    so the sensor looked noisy/active even with zero hits.
    """
    from sensors import CollisionSensor

    class _FakeSimCfg:
        body_radius = 0.15
        arena_scale = 10.0   # nothing within reach — never hits
        dt = 0.02

    class _FakeWorld:
        arena_round = True
        objects = []
        walls = []

    s = CollisionSensor(n=4, noise_std=0.5, name='touch')
    sim_cfg = _FakeSimCfg()
    world = _FakeWorld()
    for _ in range(200):
        out = s.sample(0.0, 0.0, 0.0, world, sim_cfg)
        assert np.abs(out).max() == 0.0


def test_collision_sensor_vectorized_matches_reference_loop():
    """CollisionSensor._detect_hits (numpy-vectorized) must exactly match the
    original nested-Python-loop geometry it replaced, across round/square
    arenas and 2-point/triangle/square-polygon walls.

    Regression test for the O(sectors x probes x objects x walls) perf fix
    (TODO.md Performance) — verified bit-identical against the pre-fix
    implementation over 6000 random positions during development; this test
    pins a smaller but representative sample of that check.
    """
    from sensors import CollisionSensor
    from world import World

    def reference_detect_hits(sensor, x, y, theta, world, sim_cfg):
        r = sim_cfg.body_radius * sensor.radius
        limit = sim_cfg.arena_scale
        n_pts = max(5, int(np.degrees(sensor.arc_angle) / 5))
        wall_threshold = max(0.01, sim_cfg.body_radius * (sensor.radius - 0.9))
        vals = []
        for center_a in sensor._sensor_centers():
            abs_center = theta + center_a
            half = sensor.arc_angle / 2
            angles = np.linspace(abs_center - half, abs_center + half, n_pts)
            hit = False
            for a in angles:
                px = x + r * np.cos(a)
                py = y + r * np.sin(a)
                if world.arena_round:
                    if np.hypot(px, py) > limit - wall_threshold:
                        hit = True; break
                else:
                    if abs(px) > limit - wall_threshold or abs(py) > limit - wall_threshold:
                        hit = True; break
                for obj in world.objects:
                    if np.hypot(px - obj['x'], py - obj['y']) <= obj['r'] + 0.005:
                        hit = True; break
                if not hit:
                    for wall in getattr(world, 'walls', []):
                        pts = wall['points']
                        for i in range(len(pts)):
                            ax_, ay_ = pts[i]
                            bx_, by_ = pts[(i + 1) % len(pts)]
                            ddx, ddy = bx_ - ax_, by_ - ay_
                            len2_ = ddx * ddx + ddy * ddy
                            if len2_ < 1e-12:
                                continue
                            tt = np.clip(((px - ax_) * ddx + (py - ay_) * ddy) / len2_, 0.0, 1.0)
                            if np.hypot(px - (ax_ + tt * ddx), py - (ay_ + tt * ddy)) < wall_threshold:
                                hit = True; break
                        if hit:
                            break
                if hit:
                    break
            vals.append(1.0 if hit else 0.0)
        return np.array(vals)

    class _Cfg:
        arena_scale = 5.0
        body_radius = 0.15
        dt = 0.02

    cfg = _Cfg()
    world = World(cfg)
    rng = np.random.RandomState(7)
    for _ in range(12):
        ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 4.0)
        world.objects.append({'x': dist * np.cos(ang), 'y': dist * np.sin(ang), 'r': 0.25})
    world.walls.append({'points': [[1.0, 1.0], [1.5, 1.5], [1.0, 2.0]]})   # triangle
    world.walls.append({'points': [[-2.0, -2.0], [-1.5, -2.5]]})          # 2-point segment
    world.walls.append({'points': [[-3, -3], [-3, 3], [3, 3], [3, -3]]})  # square polygon

    sensor = CollisionSensor(n=6, angle_spread=120.0, arc_angle=40.0, radius=1.3, name='touch')

    for arena_round in (True, False):
        world.arena_round = arena_round
        for _ in range(300):
            if rng.rand() < 0.5:
                obj = world.objects[rng.randint(len(world.objects))]
                x = obj['x'] + rng.uniform(-0.6, 0.6)
                y = obj['y'] + rng.uniform(-0.6, 0.6)
            else:
                x, y = rng.uniform(-4.5, 4.5), rng.uniform(-4.5, 4.5)
            theta = rng.uniform(0, 2 * np.pi)
            new_out = sensor._detect_hits(x, y, theta, world, cfg)
            old_out = reference_detect_hits(sensor, x, y, theta, world, cfg)
            assert np.array_equal(new_out, old_out), (x, y, theta, new_out, old_out)


def test_distance_sensor_detects_other_agent():
    """other_agents (circles for other agents' bodies, built per-tick by
    sim_engine.step_agents) must be treated as
    obstacles exactly like world.objects — inter-agent distance sensing."""
    from sensors import DistanceSensor
    from world import World

    class _Cfg:
        arena_scale = 10.0
        body_radius = 0.15
        dt = 0.02

    cfg = _Cfg()
    world = World(cfg)
    sensor = DistanceSensor(n=1, max_range=1.0, name='dist')

    out = sensor.sample(0.0, 0.0, 0.0, world, cfg)
    assert out[0] == pytest.approx(0.0)   # nothing ahead

    other_agents = [{'x': 0.5, 'y': 0.0, 'r': cfg.body_radius}]
    out = sensor.sample(0.0, 0.0, 0.0, world, cfg, other_agents=other_agents)
    expected_d = 0.5 - cfg.body_radius
    assert out[0] == pytest.approx(1.0 - expected_d / sensor.max_range)


def test_collision_sensor_detects_other_agent():
    """other_agents must be checked the same way as world.objects — a sector
    fires on contact with another agent's body, not just walls/objects."""
    from sensors import CollisionSensor
    from world import World

    class _Cfg:
        arena_scale = 10.0
        body_radius = 0.15
        dt = 0.02

    cfg = _Cfg()
    world = World(cfg)
    sensor = CollisionSensor(n=1, angle_spread=0.0, arc_angle=10.0, radius=1.0, name='touch')

    hit = sensor._detect_hits(0.0, 0.0, 0.0, world, cfg)
    assert hit[0] == 0.0

    other_agents = [{'x': 0.25, 'y': 0.0, 'r': cfg.body_radius}]
    hit = sensor._detect_hits(0.0, 0.0, 0.0, world, cfg, other_agents=other_agents)
    assert hit[0] == 1.0


def test_step_network_size_reconciliation_is_gated():
    """`step_network`'s size-reconciliation pass must only re-run when the
    connections list identity changes, not on every tick.

    Regression test for the O(connections) caching fix (TODO.md Performance
    review): the pass used to rescan every connection unconditionally each
    tick to keep layer sizes in sync with connection weight shapes. It is now
    gated by the same `conn_id = id(connections)` check already used for
    `_w_cache`/`_conn_by_tgt`.
    """
    from network_runner import step_network
    from sensors import GradientSensor
    from neurons import LeakyLayer
    from circuit_model import Connection
    from brain_base import BaseBrain

    sensor = GradientSensor(name='light', n=2)
    layer  = LeakyLayer(name='l1', tau_rise=0.1, tau_decay=0.1)   # n=None: sized on first tick
    conn   = Connection(src='light', tgt='l1', W=np.ones((2, 2), dtype=np.float32))

    brain = BaseBrain()
    brain.sensors     = [sensor]
    brain.layers      = [layer]
    brain.connections = [conn]
    brain.light       = np.zeros(2, dtype=np.float32)

    step_network(brain, dt=0.01)
    assert layer.n == 2

    # Mutate W to an incompatible shape in place, on the *same* connection and
    # *same* connections list object. If the reconciliation pass re-ran, it
    # would call `_ensure_n(5)` on a layer already declared at n=2, which
    # raises ValueError (see test_ensure_n_raises_on_mismatch) — so this call
    # must NOT raise, proving the pass was skipped.
    conn.W = np.ones((5, 2), dtype=np.float32)
    step_network(brain, dt=0.01)
    assert layer.n == 2

    # Replacing the connections list object (mirrors what network_viz_dialogs.py's
    # sensor/layer edit-mode handlers now do on every property edit) must bump
    # conn_id and force the pass to re-run — which now hits the same mismatch
    # and raises.
    brain.connections = list(brain.connections)
    with pytest.raises(ValueError):
        step_network(brain, dt=0.01)


def test_product_layer_multiplies_connections():
    """ProductLayer combines incoming connections by elementwise product, not
    sum — the core contract behind the multiplicative-fan-in feature (see
    network_runner.step_network's _is_product special case). tau_rise=0.0
    bypasses the (otherwise identical to LeakyLayer) leaky filtering so a
    single step reads the raw product. Also checks that a ProductLayer with
    zero connections outputs all-ones (not all-zeros), since an empty
    product is 1, unlike SumLayer's empty sum.
    """
    from network_runner import step_network
    from neurons import ConstantLayer, ProductLayer
    from circuit_model import Connection
    from brain_base import BaseBrain

    const_a = ConstantLayer(name='const_a', value=3.0, n=1)
    const_b = ConstantLayer(name='const_b', value=5.0, n=1)
    prod    = ProductLayer(name='prod', n=1, tau_rise=0.0)
    conn_a  = Connection(src='const_a', tgt='prod', W=np.array([[2.0]], dtype=np.float32))
    conn_b  = Connection(src='const_b', tgt='prod', W=np.array([[0.5]], dtype=np.float32))

    brain = BaseBrain()
    brain.sensors     = []
    brain.layers      = [const_a, const_b, prod]
    brain.connections = [conn_a, conn_b]
    brain.const_a     = const_a
    brain.const_b     = const_b

    step_network(brain, dt=0.01)

    # 2.0*3.0 = 6.0, 0.5*5.0 = 2.5 -> product 15.0 (a sum would give 8.5)
    assert prod.output.item() == pytest.approx(15.0)

    # Zero connections: n must be declared up front since it's normally
    # inferred from the first incoming connection.
    orphan = ProductLayer(name='orphan', n=2, tau_rise=0.0)
    brain2 = BaseBrain()
    brain2.sensors     = []
    brain2.layers      = [orphan]
    brain2.connections = []
    step_network(brain2, dt=0.01)
    assert torch.equal(orphan.output, torch.ones(2))


def test_product_layer_has_leaky_dynamics():
    """ProductLayer must obey the same DynamicsBase principles as every other
    layer type (tau_rise/tau_decay/noise/bias/activation/scale) — it is not
    an instantaneous SumLayer-style combinator. With the default tau_rise,
    one tiny-dt step must land strictly between the starting output (0) and
    the fully-converged product, proving the leaky filter is actually active
    rather than passing the product straight through.
    """
    from neurons import DynamicsBase, ConstantLayer, ProductLayer, LeakyLayer
    from network_runner import step_network
    from circuit_model import Connection
    from brain_base import BaseBrain

    assert isinstance(ProductLayer(name='p'), DynamicsBase)
    assert issubclass(ProductLayer, LeakyLayer)

    const_a = ConstantLayer(name='const_a', value=2.0, n=1)
    const_b = ConstantLayer(name='const_b', value=3.0, n=1)
    prod    = ProductLayer(name='prod', n=1)   # default tau_rise=0.1, activation='relu'
    conn_a  = Connection(src='const_a', tgt='prod', W=np.array([[1.0]], dtype=np.float32))
    conn_b  = Connection(src='const_b', tgt='prod', W=np.array([[1.0]], dtype=np.float32))

    brain = BaseBrain()
    brain.sensors     = []
    brain.layers      = [const_a, const_b, prod]
    brain.connections = [conn_a, conn_b]
    brain.const_a     = const_a
    brain.const_b     = const_b

    step_network(brain, dt=0.001)   # dt << tau_rise: leaky filter should barely move x
    out = prod.output.item()
    assert 0.0 < out < 6.0   # product is 2*3=6; a fully-converged/instantaneous read would equal 6.0


def test_bonsai_export_rejects_product_layer():
    """LBP.Torch's JoinAdditive only sums fan-in — there is no multiplicative
    join node. Exporting a circuit containing a ProductLayer must fail loudly
    rather than silently generating XML that behaves like a SumLayer.
    """
    from bonsai_exporter import generate_bonsai_xml
    from neurons import ProductLayer
    from circuit_model import CircuitModel

    circuit = CircuitModel()
    circuit.layers = [ProductLayer(name='prod', n=1)]

    with pytest.raises(ValueError, match='ProductLayer'):
        generate_bonsai_xml(circuit)


# ── 2. Serialisation round-trip ───────────────────────────────────────────────

def test_layer_from_dict_backward_compat_derivative():
    """Old JSON saved LeakyLayer/ProductLayer/Leaky2dLayer with their own
    boolean 'derivative' field (the retired x-vs-u novelty mode);
    _layer_from_dict must map it onto the new generic output_mode."""
    from brain_serializer import _layer_from_dict
    layer_true  = _layer_from_dict({'type': 'LeakyLayer', 'name': 'l',  'n': 2, 'derivative': True})
    layer_false = _layer_from_dict({'type': 'LeakyLayer', 'name': 'l2', 'n': 2, 'derivative': False})
    assert layer_true.output_mode  == 'derivative'
    assert layer_false.output_mode == 'none'


def test_sensor_from_dict_backward_compat_differential():
    """Old JSON saved any sensor with a boolean 'differential' field;
    _sensor_from_dict must map it onto the new generic output_mode."""
    from brain_serializer import _sensor_from_dict
    sensor = _sensor_from_dict({'type': 'GradientSensor', 'name': 's', 'differential': True})
    assert sensor.output_mode == 'derivative'


def test_layer_to_dict_persists_auto_injected_output_mode():
    """AdaptiveLayer doesn't list output_mode in its own param_defs() — it's
    only reachable via NetworkDialogs.layer_dialog (network_viz_dialogs.py), which
    auto-injects any DynamicsBase._dynamics_param_defs() entry a layer's own
    param_defs() omits. _layer_to_dict must persist it too via the same
    merge, or a value set through that auto-injected dialog field would
    silently revert to 'none' on the next save/reload."""
    from brain_serializer import _layer_to_dict
    from neurons import AdaptiveLayer
    layer = AdaptiveLayer(name='a', n=2, output_mode='integral')
    d = _layer_to_dict(layer)
    assert d['output_mode'] == 'integral'


def test_serialisation_round_trip():
    """save → load must preserve layer params and connection weight shape."""
    from brain_serializer import serialize_network_json, load_network_json
    circuit = _make_circuit()

    data = serialize_network_json(
        circuit.sensors, circuit.layers, circuit.connections,
        hidden_containers=set(), disabled_containers=set(), container_labels={},
    )
    sensors2, layers2, conns2, *_ = load_network_json(data)

    assert len(sensors2) == len(circuit.sensors)
    assert len(layers2)  == len(circuit.layers)
    assert len(conns2)   == len(circuit.connections)
    assert layers2[0].name == 'l1'
    assert layers2[0].tau_rise == pytest.approx(0.1)
    assert conns2[0].W.shape == (2, 2)


# ── 3. Visualiser smoke test ──────────────────────────────────────────────────

try:
    import pytestqt as _pytestqt
    _HAS_PYTEST_QT = True
except ImportError:
    _HAS_PYTEST_QT = False


class _FakeGui:
    """Minimal stand-in for SimulatorApp — only what NetworkVisualizerWindow reads."""
    _net_viz = None
    brain    = None

    def __init__(self, circuit):
        self.circuit = circuit

    def tracked_osc_items(self):
        return []

    def rebuild_channels(self):
        pass

    def notify_closed(self):
        pass


@pytest.mark.skipif(not _HAS_PYTEST_QT, reason='pytest-qt not installed')
def test_build_no_crash(qtbot):
    """NetworkVisualizerWindow.build() on a minimal circuit must not raise."""
    from network_viz import NetworkVisualizerWindow

    gui = _FakeGui(_make_circuit())
    win = NetworkVisualizerWindow(gui)
    qtbot.addWidget(win)
    win.build()   # synchronous; raises on any unhandled exception
    win.close()


@pytest.mark.skipif(not _HAS_PYTEST_QT, reason='pytest-qt not installed')
def test_weight_matrix_cosine_pattern_is_circulant(qtbot):
    """The Cosine weight pattern must produce a circulant (ring/rotation-
    invariant) matrix — constant along i-j diagonals — like its Gaussian and
    Mexican-hat siblings, which both explicitly key off |i-j|. Regression: the
    formula used to compute cos(source_angle + target_phase) (a sum), banding
    along i+j=const anti-diagonals instead of i-j=const diagonals — visually
    "rotated perpendicular" relative to a real ring kernel.
    """
    from network_viz_dialogs import WeightMatrixDialog

    n = 8
    dlg = WeightMatrixDialog(None, 'src', 'tgt', n, n, np.zeros((n, n)))
    qtbot.addWidget(dlg)
    dlg._pattern_cb.setCurrentIndex(1)   # Cosine
    dlg._cos_amp.setValue(1.0)
    dlg._cos_ph0.setValue(10.0)
    dlg._cos_step.setValue(45.0)         # default for n=8: 360/n
    W = dlg._compute_W()

    for i in range(1, n):
        assert np.allclose(W[i], np.roll(W[i - 1], 1)), \
            f"row {i} is not the previous row rotated by one column"
    dlg.close()


@pytest.mark.skipif(not _HAS_PYTEST_QT, reason='pytest-qt not installed')
def test_activation_panel_pin_and_update(qtbot):
    """Pinning a layer to the activation panel (mirrors the weight panel's
    pin/refresh contract) and refreshing must populate its bar chart with the
    layer's current per-neuron output, indexed 0..n-1."""
    from network_viz import NetworkVisualizerWindow

    circuit = _make_circuit()
    layer = circuit.layers[0]   # 'l1', n=2
    layer.output = torch.tensor([0.3, -0.7])

    gui = _FakeGui(circuit)
    win = NetworkVisualizerWindow(gui)
    qtbot.addWidget(win)
    win.build()

    assert 'l1' not in win._activation_pinned
    win._toggle_activation_entry('l1')
    assert 'l1' in win._activation_pinned
    # isVisibleTo (not isVisible) reflects the local setVisible(True) flag
    # without needing the top-level window itself to be shown.
    assert win._activation_panel.isVisibleTo(win)

    win.renderer.update_activation_panel()
    entry = win._activation_pinned['l1']
    assert list(entry._bars.opts['height']) == pytest.approx([0.3, -0.7])

    win._toggle_activation_entry('l1')
    assert 'l1' not in win._activation_pinned

    win.close()


@pytest.mark.skipif(not _HAS_PYTEST_QT, reason='pytest-qt not installed')
def test_side_view_drag_preserves_sensor_z(qtbot):
    """Dragging a sensor container in the side view must update its z, not just
    its column — otherwise it can never win a z-cut competition in its new
    column and its connections get silently dropped on the next rebuild.

    Regression test: side_view.py's drag/drop/insert handlers used to set
    sensor.layer but never sensor.z (layers got both).
    """
    from sensors import GradientSensor
    from neurons import LeakyLayer
    from circuit_model import CircuitModel, Connection
    from network_viz import NetworkVisualizerWindow
    from side_view import SideViewWindow

    sensor = GradientSensor(name='s', n=1)
    sensor.lateralized = True   # exercise the _L/_R split path
    sensor.layer = 1
    sensor.z     = 2            # distinct starting z, so a stale value is obvious
    competitor = LeakyLayer(name='comp', n=1, tau_rise=0.1, tau_decay=0.1)
    competitor.layer = 0
    competitor.z = 1
    target = LeakyLayer(name='tgt', n=1, tau_rise=0.1, tau_decay=0.1)
    target.layer = 2
    W = np.ones((1, 1), dtype=np.float32)
    conns = [Connection(src='s_L', tgt='tgt', W=W), Connection(src='s_R', tgt='tgt', W=W)]
    circuit = CircuitModel(sensors=[sensor], layers=[competitor, target], connections=conns)

    gui = _FakeGui(circuit)
    win = NetworkVisualizerWindow(gui)
    qtbot.addWidget(win)
    win.build()

    sv = SideViewWindow(win)
    qtbot.addWidget(sv)
    sv.refresh()
    old_dv, old_z = sensor.layer, (sensor.z or 0)
    old_ci = sv._ordered_containers.index(old_dv)
    old_zi = sv._z_levels.index(old_z)
    # Cross-column drag into competitor's column, at an empty z (0) there.
    sv._on_container_dropped(old_dv, old_z, old_ci, old_zi,
                             sv._ordered_containers.index(0), sv._z_levels.index(0))

    assert sensor.layer == 0
    assert sensor.z == 0   # used to stay frozen at the old value (2)

    win.build()
    # Dropped below competitor (z=0 < 1) -> legitimately subsumed; connections
    # through it are hidden. This is correct given where it was actually placed.
    assert 's_L' not in win._lay.active_names

    # Same-column drag (still dv=0) raising z to 1, tying with the competitor.
    sv.refresh()
    old_zi2 = sv._z_levels.index(0)
    sv._on_container_dropped(0, 0, sv._ordered_containers.index(0), old_zi2,
                             sv._ordered_containers.index(0), sv._z_levels.index(1))

    assert sensor.z == 1   # used to stay frozen; same-column branch never touched sensor.z

    win.build()
    assert 's_L' in win._lay.active_names and 's_R' in win._lay.active_names
    assert not win._lay.subsumed_by


@pytest.mark.skipif(not _HAS_PYTEST_QT, reason='pytest-qt not installed')
def test_paste_selection_bumps_connections_identity(qtbot):
    """_paste_selection appends new connections via .append() on the existing
    connections list object — id(circuit.connections) must still change
    afterward, or network_runner's conn_id-gated caches (_w_cache,
    _conn_by_tgt, size-reconciliation, shape-metadata propagation) will
    silently miss the pasted connection until an unrelated edit happens to
    replace the list.

    Regression test for the append-only conn_id gap (TODO.md Performance):
    _insert_motif and the RingAttractorLayer default self-recurrent
    connection (network_viz_dialogs.py) hit the exact same pattern and got
    the identical one-line fix; this test exercises the simplest of the
    three call sites directly.
    """
    from PySide6.QtWidgets import QApplication
    from brain_serializer import serialize_network_json
    from neurons import LeakyLayer
    from circuit_model import CircuitModel, Connection
    from network_viz import NetworkVisualizerWindow

    a    = LeakyLayer(name='a', n=1, tau_rise=0.1, tau_decay=0.1)
    b    = LeakyLayer(name='b', n=1, tau_rise=0.1, tau_decay=0.1)
    conn = Connection(src='a', tgt='b', W=np.ones((1, 1), dtype=np.float32))
    data = serialize_network_json(
        [], [a, b], [conn],
        hidden_containers=set(), disabled_containers=set(), container_labels={},
    )

    circuit = CircuitModel(sensors=[], layers=[], connections=[])
    gui = _FakeGui(circuit)
    win = NetworkVisualizerWindow(gui)
    qtbot.addWidget(win)
    win.build()

    QApplication.clipboard().setText(json.dumps(
        {'layers': data['layers'], 'connections': data['connections']}))
    before_id = id(circuit.connections)
    win.editing.paste_selection()

    assert len(circuit.connections) == 1
    assert id(circuit.connections) != before_id


# ── 4. World / texture round-trip ─────────────────────────────────────────────

def test_world_serializer_round_trip():
    """save_world_file → load_world_file must preserve all world state, including
    per-object/per-wall/floor texture assignments."""
    from world import World
    from world_serializer import save_world_file, load_world_file

    class _FakeCfg:
        arena_scale = 2.0
        body_radius = 0.15

    w = World(_FakeCfg())
    w.objects = [{'x': 0, 'y': 0, 'r': 0.2, 'color': [1, 0, 0],
                  'texture': 'checkerboard.png', 'external': True}]
    w.walls = [{'points': [[1, 1], [1, 2]], 'color': [0.5, 0.5, 0.5],
                'texture': 'grid.png', 'external': True}]
    w.arena_round   = True
    w.sky           = {'enabled': True, 'angle': 1.2}
    w.floor_texture = 'noise.png'

    path = os.path.join(_SIM2D, 'worlds', '_test_round_trip.json')
    try:
        save_world_file(w, path)
        w2 = World(_FakeCfg())
        load_world_file(path, w2)
        assert w2.objects == w.objects
        assert w2.walls == w.walls
        assert w2.arena_round == w.arena_round
        assert w2.sky == w.sky
        assert w2.floor_texture == w.floor_texture
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_session_mount_survives_agent_id_remint_on_reload():
    """A gradient patch's mounted_on must keep following the same agent across
    a session save/reload, even when that agent isn't the very first one.

    Regression test: RobotAgent.id comes from an ever-incrementing counter
    (AgentRegistry._next_agent_id) that's never reused, but
    _load_session_agents only reuses the very first agent's original id —
    every other agent (any later group, or any extra member within a group)
    gets a brand-new id on every reload. A saved raw agent id would silently
    go stale, leaving the patch static (the reported bug). _patches_for_save/
    _resolve_mounted_patches translate mounted_on to/from a stable positional
    index across that boundary instead."""
    from sim_app_session import _SessionMixin

    class _FakeAgent:
        def __init__(self, id):
            self.id = id

    class _FakeWorld:
        def __init__(self, patches):
            self.patches = patches

    class _FakeSimCtrl:
        def __init__(self, agents):
            from types import SimpleNamespace
            self.registry = SimpleNamespace(agents=agents)

    class Host(_SessionMixin):
        pass

    host = Host()
    host._sim_ctrl = _FakeSimCtrl([_FakeAgent(10), _FakeAgent(11), _FakeAgent(12)])
    host.world = _FakeWorld([
        {'x': 0.0, 'y': 0.0, 'r': 0.3, 'mounted_on': 11},   # mounted on the SECOND agent
        {'x': 1.0, 'y': 1.0, 'r': 0.2},                     # unmounted patch — must pass through
    ])

    saved = host._patches_for_save()
    assert saved[0]['mounted_on'] == 1        # positional index, not the raw id
    assert 'mounted_on' not in saved[1]

    # Simulate a reload: every agent but the first gets a brand-new id
    # (mirrors _load_session_agents rebuilding groups/members from scratch).
    host._sim_ctrl = _FakeSimCtrl([_FakeAgent(10), _FakeAgent(50), _FakeAgent(51)])
    host.world = _FakeWorld([dict(p) for p in saved])   # as if freshly loaded from JSON
    host._resolve_mounted_patches()

    assert host.world.patches[0]['mounted_on'] == 50   # new id of the agent now at position 1
    assert 'mounted_on' not in host.world.patches[1]


def test_session_mount_dropped_when_agent_no_longer_exists():
    """If the mounted-on agent was removed before saving, or the saved index
    is out of range after a reload with fewer agents, mounted_on must be
    dropped cleanly rather than resolving to the wrong agent or crashing."""
    from sim_app_session import _SessionMixin

    class _FakeAgent:
        def __init__(self, id):
            self.id = id

    class _FakeWorld:
        def __init__(self, patches):
            self.patches = patches

    class _FakeSimCtrl:
        def __init__(self, agents):
            from types import SimpleNamespace
            self.registry = SimpleNamespace(agents=agents)

    class Host(_SessionMixin):
        pass

    host = Host()
    host._sim_ctrl = _FakeSimCtrl([_FakeAgent(10)])
    host.world = _FakeWorld([{'x': 0.0, 'y': 0.0, 'r': 0.3, 'mounted_on': 999}])  # unknown id
    saved = host._patches_for_save()
    assert 'mounted_on' not in saved[0]

    host._sim_ctrl = _FakeSimCtrl([_FakeAgent(10)])   # only one agent after reload
    host.world = _FakeWorld([{'x': 0.0, 'y': 0.0, 'r': 0.3, 'mounted_on': 2}])   # index out of range
    host._resolve_mounted_patches()
    assert 'mounted_on' not in host.world.patches[0]


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('mujoco') is None,
    reason='mujoco not installed')
def test_mounted_gradient_snaps_to_agent_immediately_on_reset():
    """A gradient patch mounted on an agent must already reflect that agent's
    reset position the instant reset() returns, not just after the first tick.

    Regression test for a user report: clicking Run after driving the agent
    away from spawn showed the mounted gradient rendered at its stale
    pre-reset position. The per-tick mount->robot sync used to live inline in
    SimController._tick only, so ArenaWidget._rebuild_gradient() -- called by
    _setup_world() right after reset() repositions agents -- drew the patch
    wherever it was before Stop was clicked, not where the agent was just
    reset to. _sync_mounted_patches() is now also called at the end of
    reset(), closing that one-tick gap."""
    from sim_controller import SimController
    from circuit_model import CircuitModel
    from rigid_body import RigidBody
    from world import World
    from brain_base import BaseBrain
    from sim_config import SimConfig

    class _FakeArena:
        def setup_sensors(self, *a, **k): pass
        def sync_agents(self, *a, **k): pass
        def sync_robot_items(self, *a, **k): pass

    class _FakeOscCtrl:
        channels = []
        _osc_items = set()
        channel_colors = {}
        def reset_trace(self): pass
        def update_osc(self): pass

    class _FakeLogger:
        def log(self, *a, **k): pass

    class _TestBrain(BaseBrain):
        def setup(self):
            pass
        def loop(self, *sensors):
            return 20.0, 20.0
        def plots(self):
            return []

    sim_cfg = SimConfig()
    circuit = CircuitModel()
    circuit.bodies = [RigidBody('root', 'root', sim_cfg.body_radius)]
    world = World(sim_cfg)

    ctrl = SimController(circuit, sim_cfg, world, _FakeArena(), _FakeOscCtrl(), _FakeLogger(),
                          get_trail_visible=lambda: False, get_motor_override=lambda: None)

    agent = ctrl.registry.agents[0]
    agent.brain = _TestBrain()
    agent.brain.sensors, agent.brain.layers, agent.brain.connections = [], [], []
    agent.circuit = circuit
    ctrl.registry.brain = agent.brain

    world.patches.append({'x': 0.0, 'y': 0.0, 'r': 0.3, 'mounted_on': agent.id})
    ok, err = ctrl.mujoco.start()   # MuJoCo moves the agents
    assert ok, err

    for _ in range(5):   # drive the agent away from spawn
        ctrl._tick()
    assert agent.bot_pos[0] != 0.0

    ctrl.stop()
    ctrl.reset()

    # No tick has run since reset() -- the patch must already match the RESET
    # position, not the position it tracked to just before Stop was clicked.
    assert world.patches[0]['x'] == pytest.approx(sim_cfg.init_x)
    assert world.patches[0]['y'] == pytest.approx(sim_cfg.init_y)
    assert world.patches[0]['x'] == pytest.approx(agent.bot_pos[0])
    assert world.patches[0]['y'] == pytest.approx(agent.bot_pos[1])


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('mujoco') is None,
    reason='mujoco not installed')
def test_mujoco_build_xml_textures():
    """_build_xml must: dedupe repeated texture files into one <texture>/<material>
    pair, use material= (not rgba=) for textured objects/walls, keep rgba= for
    untextured ones, and leave output byte-identical when no textures are used."""
    import mujoco
    from sim_engine_mujoco import MuJoCoEngine

    class _FakeCfg:
        arena_scale = 2.0
        body_radius = 0.15
        dt          = 0.02

    class _WorldWithTextures:
        arena_round = False
        objects = [
            {'x': 0.5, 'y': 0.5, 'r': 0.2, 'color': [1, 0, 0],
             'external': True, 'texture': 'checkerboard.png'},
            {'x': -0.5, 'y': 0.5, 'r': 0.2, 'color': [0, 1, 0],
             'external': True, 'texture': 'checkerboard.png'},   # shares the same file
            {'x': 0.0, 'y': -0.5, 'r': 0.2, 'color': [0, 0, 1], 'external': True},
        ]
        walls = [{'points': [[1, 1], [1, 1.5], [1.5, 1.5], [1.5, 1]],
                  'color': [0.5, 0.5, 0.5], 'texture': 'grid.png'}]
        floor_texture = 'noise.png'

    xml, assets = MuJoCoEngine._build_xml(_WorldWithTextures(), _FakeCfg(), n_agents=1)
    assert set(assets) == {'checkerboard.png', 'grid.png', 'noise.png'}   # deduped
    assert xml.count('rgba="0.000 0.000 1.000 1"') == 1   # the one untextured object
    model = mujoco.MjModel.from_xml_string(xml, assets)   # must actually compile
    assert model.ngeom > 0

    class _WorldNoTextures:
        arena_round   = False
        objects       = [{'x': 0, 'y': 0, 'r': 0.2, 'color': [1, 0, 0], 'external': True}]
        walls         = []
        floor_texture = None

    xml2, assets2 = MuJoCoEngine._build_xml(_WorldNoTextures(), _FakeCfg(), n_agents=1)
    assert assets2 == {}
    assert 'builtin="checker"' in xml2   # default floor unchanged when untextured
    mujoco.MjModel.from_xml_string(xml2, assets2)


def test_mujoco_collision_eligible_gating():
    """_mujoco_collision_eligible must accept any root-mounted CollisionSensor,
    regardless of radius — dedicated per-sector geometry is built at exactly
    that sensor's own probe radius (see _build_xml), so there's no lookahead
    limit to check, unlike the earlier margin-based design. Only a non-root
    mount (e.g. a whisker joint pair) still needs the analytic path, since
    there's no per-body-id geometry built for those."""
    from sensors import CollisionSensor
    from sim_engine_mujoco import _mujoco_collision_eligible

    assert _mujoco_collision_eligible(CollisionSensor(n=4, radius=1.0, name='touch'))
    assert _mujoco_collision_eligible(CollisionSensor(n=4, name='touch'))          # radius=1.2 default
    assert _mujoco_collision_eligible(CollisionSensor(n=4, radius=5.0, name='touch'))  # any radius now

    off_root = CollisionSensor(n=4, radius=1.0, name='touch')
    off_root.body_ids = ['body3_L', 'body3_R']
    assert not _mujoco_collision_eligible(off_root)


def test_log_collision_sensor_routing_reports_correct_path(capsys):
    """log_collision_sensor_routing must print one line per CollisionSensor
    naming the path it actually uses, so the routing decision is visible from
    the console instead of only inferable from reading the code — this is
    the only way to confirm from the running app which sensors got the
    MuJoCo speedup and which fell back to analytic."""
    from sensors import CollisionSensor
    from sim_engine_mujoco import log_collision_sensor_routing

    class _Circuit:
        def __init__(self, sensors):
            self.sensors = sensors

    class _Agent:
        def __init__(self, sensors):
            self.circuit = _Circuit(sensors)

    eligible = CollisionSensor(n=2, radius=1.2, name='bumper')      # matches 04 - CollisionBrain.json
    off_root = CollisionSensor(n=4, radius=1.0, name='whisker_touch')
    off_root.body_ids = ['body3_L', 'body3_R']

    agents = [_Agent([eligible, off_root])]
    log_collision_sensor_routing(agents)

    out = capsys.readouterr().out
    assert "'bumper'" in out and 'MuJoCo contact path' in out
    assert "'whisker_touch'" in out and 'analytic path' in out and 'not mounted on root body' in out


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('mujoco') is None,
    reason='mujoco not installed')
def test_mujoco_collision_sensor_fires_on_contact_not_before():
    """A MuJoCo-eligible CollisionSensor sampled through sim_engine.step_agents
    must fire exactly at its own configured lookahead distance: a literal touch
    sensor (radius=1.0) fires only on actual penetration, while a lookahead
    sensor (radius=1.2, the default) fires strictly earlier — matching the
    analytic path's own behavior, without running any analytic geometry.
    Each sensor's own dedicated per-sector geometry (see _build_xml) is what
    makes this exact rather than an approximation.

    Regression test for the collision-detection MuJoCo offload (TODO.md
    Performance): see _scratch/measure_collision_backends.py for the
    benchmark that motivated this (~65x cheaper than the analytic check,
    since MuJoCo already computes contacts every tick for physics)."""
    from types import SimpleNamespace
    from sim_engine_mujoco import MuJoCoEngine
    from sim_engine import step_agents
    from circuit_model import CircuitModel
    from sensors import CollisionSensor
    from world import World
    from brain_base import BaseBrain

    class _Cfg:
        arena_scale  = 5.0
        body_radius  = 0.15
        dt           = 0.02
        motor_gain   = 1.0
        fixate_robot = 0.0
        toggle_stim  = False

    class _TestBrain(BaseBrain):
        def loop(self, *sensors):
            return 0.0, 0.0

        def plots(self):
            return []

    cfg = _Cfg()
    world = World(cfg)
    world.arena_round = True
    world.objects.append({'x': 1.0, 'y': 0.0, 'r': 0.25, 'color': [1, 0, 0], 'external': True})

    touch     = CollisionSensor(n=4, angle_spread=90.0, arc_angle=45.0, radius=1.0, name='touch')
    lookahead = CollisionSensor(n=4, angle_spread=90.0, arc_angle=45.0, name='lookahead')  # radius=1.2 default
    brain = _TestBrain()
    brain.sensors, brain.layers, brain.connections = [touch, lookahead], [], []
    brain.touch = np.zeros(4)
    brain.lookahead = np.zeros(4)

    engine = MuJoCoEngine(world, cfg, n_agents=1, agent_sensors=[[touch, lookahead]])

    def hit(sensor_name):
        return bool(np.any(getattr(brain, sensor_name) > 0))

    # Sum of radii = body_radius(0.15) + object.r(0.25) = 0.40 -> touch fires
    # for distance < 0.40. Lookahead adds body_radius*(1.2-1.0)=0.03, firing
    # for distance < 0.43 -- strictly earlier than touch, never later.
    cases = [
        (-2.0, False, False),   # far: neither fires
        (0.615, True, True),    # distance=0.385: both fire (past touch's threshold)
        (0.90, True, True),     # deep overlap: both fire
    ]
    for x, expect_touch, expect_lookahead in cases:
        engine.reset([[x, 0.0, 0.0]])
        agent = SimpleNamespace(bot_pos=[x, 0.0, 0.0], brain=brain,
                                circuit=CircuitModel(sensors=[touch, lookahead]))
        step_agents([agent], world, cfg, engine, 0.0)
        assert hit('touch') == expect_touch, f'x={x}: touch expected {expect_touch}, got {brain.touch}'
        assert hit('lookahead') == expect_lookahead, f'x={x}: lookahead expected {expect_lookahead}, got {brain.lookahead}'


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('mujoco') is None,
    reason='mujoco not installed')
def test_step_agents_distance_sensor_sees_other_agent():
    """step_agents' other_agents wiring (built once per step as a pre-step
    position snapshot, sliced to exclude self) must reach each agent's
    DistanceSensor — the exact call site sim_controller._tick uses."""
    from types import SimpleNamespace
    from sim_engine_mujoco import MuJoCoEngine
    from sim_engine import step_agents
    from circuit_model import CircuitModel
    from sensors import DistanceSensor
    from world import World
    from brain_base import BaseBrain

    class _Cfg:
        arena_scale  = 5.0
        body_radius  = 0.15
        dt           = 0.02
        motor_gain   = 1.0
        fixate_robot = 0.0
        toggle_stim  = False

    class _TestBrain(BaseBrain):
        def loop(self, *sensors):
            return 0.0, 0.0

        def plots(self):
            return []

    cfg = _Cfg()
    world = World(cfg)
    world.arena_round = True

    dist0, dist1 = DistanceSensor(n=1, max_range=1.0, name='dist'), DistanceSensor(n=1, max_range=1.0, name='dist')
    brain0, brain1 = _TestBrain(), _TestBrain()
    for b, s in ((brain0, dist0), (brain1, dist1)):
        b.sensors, b.layers, b.connections = [s], [], []
        b.dist = np.zeros(1)

    engine = MuJoCoEngine(world, cfg, n_agents=2, agent_sensors=[[], []])
    engine.reset([[0.0, 0.0, 0.0], [0.5, 0.0, np.pi]])   # facing each other, 0.5 apart

    agents = [
        SimpleNamespace(bot_pos=[0.0, 0.0, 0.0], brain=brain0, circuit=CircuitModel(sensors=[dist0])),
        SimpleNamespace(bot_pos=[0.5, 0.0, np.pi], brain=brain1, circuit=CircuitModel(sensors=[dist1])),
    ]
    step_agents(agents, world, cfg, engine, 0.0)

    expected_d = 0.5 - cfg.body_radius   # gap between the two body circles
    expected_out = 1.0 - expected_d / dist0.max_range
    assert brain0.dist[0] == pytest.approx(expected_out, abs=1e-3)
    assert brain1.dist[0] == pytest.approx(expected_out, abs=1e-3)
def test_mujoco_collision_matches_analytic_randomized():
    """The MuJoCo per-sector-geometry path must agree with the analytic
    reference (CollisionSensor._detect_hits) across randomized scenes —
    objects, walls, varying n/angle_spread/arc_angle/radius, round and square
    arenas. Both approximate the same underlying geometry differently (real
    MuJoCo collision geometry vs discrete point-probes), so this allows a
    small tolerance rather than requiring bit-for-bit equality; it must not
    regress below that tolerance.

    Regression test for the CollisionSensor MuJoCo redesign (TODO.md
    Performance) — see the conversation history / _scratch/ for the earlier,
    less accurate bearing-only design this replaced."""
    from types import SimpleNamespace
    from sim_engine_mujoco import MuJoCoEngine
    from sim_engine import step_agents
    from circuit_model import CircuitModel
    from sensors import CollisionSensor
    from world import World
    from brain_base import BaseBrain

    class _Cfg:
        arena_scale  = 5.0
        body_radius  = 0.15
        dt           = 0.02
        motor_gain   = 1.0
        fixate_robot = 0.0
        toggle_stim  = False

    class _TestBrain(BaseBrain):
        def loop(self, *sensors):
            return 0.0, 0.0

        def plots(self):
            return []

    cfg = _Cfg()
    rng = np.random.RandomState(11)
    mismatches = 0
    n_trials = 80
    for _ in range(n_trials):
        world = World(cfg)
        world.arena_round = bool(rng.rand() < 0.5)
        for _ in range(rng.randint(0, 4)):
            ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 1.0)
            world.objects.append({'x': dist * np.cos(ang), 'y': dist * np.sin(ang),
                                   'r': rng.uniform(0.05, 0.3), 'color': [1, 0, 0], 'external': True})
        for _ in range(rng.randint(0, 2)):
            cx, cy = rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6)
            a = rng.uniform(0, np.pi)
            dx, dy = 0.3 * np.cos(a), 0.3 * np.sin(a)
            world.walls.append({'points': [[cx - dx, cy - dy], [cx + dx, cy + dy]], 'color': [0.5, 0.5, 0.5]})

        n      = int(rng.choice([1, 2, 4]))
        spread = float(rng.choice([60, 90, 120, 180]))
        arc    = float(rng.choice([20, 30, 45, 60]))
        radius = float(rng.choice([1.0, 1.2, 1.5]))
        sensor = CollisionSensor(n=n, angle_spread=spread, arc_angle=arc, radius=radius, name='touch')
        theta  = rng.uniform(0, 2 * np.pi)
        x, y   = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)

        analytic = sensor._detect_hits(x, y, theta, world, cfg)

        brain = _TestBrain()
        brain.sensors, brain.layers, brain.connections = [sensor], [], []
        brain.touch = np.zeros(n)
        engine = MuJoCoEngine(world, cfg, n_agents=1, agent_sensors=[[sensor]])
        engine.reset([[x, y, theta]])
        agent = SimpleNamespace(bot_pos=[x, y, theta], brain=brain, circuit=CircuitModel(sensors=[sensor]))
        step_agents([agent], world, cfg, engine, 0.0)

        if not np.array_equal(analytic, brain.touch):
            mismatches += 1

    # Measured ~3% on a similar distribution during development, concentrated at
    # exact detection-boundary cases (discretization) and wall contacts (MuJoCo's
    # walls have real thickness; the analytic path idealizes them as thin lines).
    assert mismatches / n_trials < 0.10, f'{mismatches}/{n_trials} mismatches — regression?'


# ── Simulation step (sense → think → motors → act) ───────────────────────────

class _FakeEngine:
    """Stand-in for MuJoCoEngine: records calls, owns cameras, moves nothing."""
    def __init__(self):
        self.calls = []
        self.commands = None

    @staticmethod
    def owns_sensor(sensor):
        from sensors import CameraSensor
        return isinstance(sensor, CameraSensor)

    def sample_collision_sensors(self, brain, sensors, sim_cfg, agent_idx=0):
        self.calls.append('contacts')

    def render_cameras(self, brain, cam_sensors, agent_idx, t, sim_dt):
        if cam_sensors:
            self.calls.append(('cameras', [s.name for s in cam_sensors], t))

    def move_agents(self, bot_positions, commands, sim_cfg):
        self.calls.append('move')
        self.commands = commands


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('mujoco') is None,
    reason='mujoco not installed')
def test_engines_satisfy_engine_protocol():
    """MuJoCoEngine and the test stand-in both provide everything step_agents
    needs (sim_engine.Engine) — keeps the stand-in from drifting."""
    from sim_engine import Engine
    from sim_engine_mujoco import MuJoCoEngine
    assert issubclass(MuJoCoEngine, Engine)
    assert isinstance(_FakeEngine(), Engine)


class _StepCfg:
    arena_scale  = 5.0
    body_radius  = 0.15
    dt           = 0.01
    motor_gain   = 1.0
    fixate_robot = 0.0
    toggle_stim  = False


def test_step_agents_senses_everything_before_the_brain_runs():
    """All sensing (2-D fields, MuJoCo contacts, due cameras) happens before
    brain.loop, and movement happens after it — so the brain never sees a
    contact one step late."""
    from types import SimpleNamespace
    from sim_engine import step_agents
    from circuit_model import CircuitModel
    from world import World
    from brain_base import BaseBrain

    engine = _FakeEngine()

    class _Brain(BaseBrain):
        def loop(self, dt):
            engine.calls.append('think')
            return 10.0, 20.0

    agent = SimpleNamespace(bot_pos=[0.0, 0.0, 0.0], brain=_Brain(), circuit=CircuitModel())
    raws = step_agents([agent], World(_StepCfg()), _StepCfg(), engine, 0.0)

    assert engine.calls == ['contacts', 'think', 'move']
    assert engine.commands == [(10.0, 20.0)]
    assert raws[0]['mL'] == 10.0 and raws[0]['mR'] == 20.0


def test_step_agents_keyboard_command_replaces_wheels_and_shows_in_motor_layer():
    """A keyboard/network command is the wheel command for that agent; the
    brain still runs, and the command is written into the motor layer so the
    visualizer/oscilloscope show what drives the robot (rules/motor_commands.md)."""
    from types import SimpleNamespace
    from sim_engine import step_agents
    from circuit_model import CircuitModel
    from neurons import MotorLayer
    from world import World
    from brain_base import BaseBrain

    ran = []

    class _Brain(BaseBrain):
        def loop(self, dt):
            ran.append(dt)
            return 5.0, 5.0

    motor = MotorLayer(n=2, name='motor')
    motor.output = torch.zeros(2)
    agent = SimpleNamespace(bot_pos=[0.0, 0.0, 0.0], brain=_Brain(),
                            circuit=CircuitModel(layers=[motor]))
    engine = _FakeEngine()
    raws = step_agents([agent], World(_StepCfg()), _StepCfg(), engine, 0.0,
                       motor_for=lambda a: (30.0, -30.0))

    assert ran, 'brain must run even when another motor source drives'
    assert engine.commands == [(30.0, -30.0)]
    assert raws[0]['mL'] == 30.0 and raws[0]['mR'] == -30.0
    assert motor.output.tolist() == [30.0, -30.0]


def test_camera_fps_schedule_runs_on_sim_time():
    """fps > 0: one frame per 1/fps simulated seconds, regardless of step size.
    fps == 0: free-running — due whenever sim time advanced since the last frame."""
    from sensors import GrayCameraSensor

    cam = GrayCameraSensor(width=4, height=2, fps=10.0, name='cam')
    frame = np.zeros((2, 4, 3), dtype=np.float32)
    rendered = []
    for k in range(25):                      # 0.00 … 0.24 s at dt = 0.01
        t = k * 0.01
        if cam.frame_due(t):
            cam.process_frame(frame, t, 0.01)
            rendered.append(round(t, 2))
    assert rendered == [0.0, 0.1, 0.2]

    free = GrayCameraSensor(width=4, height=2, fps=0.0, name='cam0')
    assert free.frame_due(0.0)
    free.process_frame(frame, 0.0, 0.01)
    assert not free.frame_due(0.0)           # no sim time passed
    assert free.frame_due(0.05)

    cam.reset()
    assert cam.frame_due(0.0)                # reset restarts the schedule


def test_camera_process_frame_output_formats():
    """Gray: non-lateralized output is the centre row; RGB: full CHW frame.
    Lateralized halves are raw pixels in the same layout."""
    from sensors import GrayCameraSensor, RGBCameraSensor

    H, W = 3, 4
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[1, :, :] = 0.9                       # bright centre row
    rgb[:, :, 0] += 0.05                     # distinguish channels

    gray = GrayCameraSensor(width=W, height=H, name='g')
    out = gray.process_frame(rgb, 0.0, 0.01)
    assert out.shape == (W,)
    np.testing.assert_allclose(out, rgb[1].mean(axis=-1), atol=1e-6)

    col = RGBCameraSensor(width=W, height=H, lateralized=True, name='c')
    out = col.process_frame(rgb, 0.0, 0.01)
    np.testing.assert_allclose(out, rgb.transpose(2, 0, 1).reshape(-1), atol=1e-6)
    np.testing.assert_allclose(col._left_output,
                               rgb[:, :W // 2, :].transpose(2, 0, 1).reshape(-1), atol=1e-6)


def test_robot_mode_runs_brain_loop_and_sends_its_command():
    """Robot mode uses the same think/motors code as the simulator: the
    brain's loop() runs (so code-only brains work), and the wheel command it
    returns is sent to every motor layer's robot_address. The keyboard takes
    priority and is written back into the motor layer."""
    from robot_mode_controller import RobotModeController
    from circuit_model import CircuitModel
    from neurons import MotorLayer
    from brain_base import BaseBrain

    class _CodeOnlyBrain(BaseBrain):
        def loop(self, dt):
            self.calls = getattr(self, 'calls', 0) + 1
            return 12.0, -7.0

    class _Osc:
        channels = ['mL', 'mR']
        _osc_items = {'mL', 'mR'}

    motor = MotorLayer(n=2, name='motor', robot_address='10.0.0.2:2390/wheels')
    motor.output = torch.zeros(2)
    circuit = CircuitModel(layers=[motor])
    brain = _CodeOnlyBrain()
    keyboard = [None]
    rm = RobotModeController(_StepCfg(), lambda: keyboard[0])

    values = rm.tick(circuit, brain, _Osc())
    assert brain.calls == 1
    assert values == {'mL': 12.0, 'mR': -7.0}
    assert rm.get_motor_commands(circuit, brain) == [('10.0.0.2', 2390, '/wheels', 12.0, -7.0)]

    keyboard[0] = (40.0, 40.0)
    rm.tick(circuit, brain, _Osc())
    assert brain.calls == 2, 'brain still runs while the keyboard drives'
    assert rm.get_motor_commands(circuit, brain) == [('10.0.0.2', 2390, '/wheels', 40.0, 40.0)]
    assert motor.output.tolist() == [40.0, 40.0]


def test_simulation_step_ticks_task_once_with_all_agents():
    """Simulation.step ticks the task once per step and gives it every agent's
    position, not only the selected agent's."""
    from simulation import Simulation
    from agent_registry import AgentRegistry
    from circuit_model import CircuitModel
    from brain_manager import BrainManager
    from world import World
    from brain_base import BaseBrain

    class _Brain(BaseBrain):
        def loop(self, dt):
            return 0.0, 0.0

    cfg = _StepCfg()
    cfg.init_x = cfg.init_y = 0.0
    c0, c1 = CircuitModel(), CircuitModel()
    registry = AgentRegistry(cfg, c0, BrainManager(c0, cfg))
    registry.add_agent(c1, BrainManager(c1, cfg))
    for a in registry.agents:
        a.brain = _Brain()

    seen = []

    class _Task:
        def tick(self, world, bot_positions, sim_cfg, dt):
            seen.append([list(p) for p in bot_positions])

    sim = Simulation(World(cfg), cfg, registry)
    sim.engine = _FakeEngine()
    sim.task = _Task()
    sim.step()
    sim.step()
    assert len(seen) == 2
    assert all(len(positions) == 2 for positions in seen)


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('mujoco') is None,
    reason='mujoco not installed')
def test_headless_session_loads_all_agents_and_runs():
    """session_loader.build_simulation builds every agent group of a saved
    session without the GUI, and the result can be stepped."""
    from session_loader import build_simulation

    cwd = os.getcwd()
    os.chdir(_SIM2D)   # session paths (brains/, networks/) are relative to simulation/2d
    try:
        sim = build_simulation(os.path.join('configs', 'Tutorials', 'T08 - MultiAgentMultiSession.json'))
        start = [list(a.bot_pos) for a in sim.agents]
        for _ in range(50):
            raws = sim.step()
        assert len(sim.agents) == 3 and len(raws) == 3
        assert all(a.brain is not None for a in sim.agents)
        assert any(a.bot_pos[:2] != s[:2] for a, s in zip(sim.agents, start))
        assert sim.sim_time == pytest.approx(50 * sim.sim_cfg.dt)
        sim.close()
    finally:
        os.chdir(cwd)


# ── Circuit editor (edit transactions / undo) ────────────────────────────────

def _editor_fixture():
    from types import SimpleNamespace
    from circuit_model import CircuitModel, Connection
    from circuit_editor import CircuitEditor
    from neurons import LeakyLayer
    from sensors import GradientSensor
    from brain_base import BaseBrain

    class _Brain(BaseBrain):
        def loop(self, dt):
            self.step_network(dt)
            return 0.0, 0.0

    light = GradientSensor(name='light', n=2)
    l1 = LeakyLayer(name='l1', n=2, tau_rise=0.1, tau_decay=0.1)
    circuit = CircuitModel(sensors=[light], layers=[l1],
                           connections=[Connection('light', 'l1', np.eye(2))])
    meta = SimpleNamespace(_hidden_containers=set(), _disabled_containers=set(),
                           _container_labels={}, _container_notes={}, _weight_params={})
    brain = _Brain()
    editor = CircuitEditor(circuit, brain, meta=meta)
    editor.sync_brain()
    return editor, circuit, brain, meta


def test_editor_undo_restores_state_from_before_the_edit():
    """The snapshot is taken when the transaction begins, so changes made
    early in an edit (here: weight params, then the weights) are all undone."""
    from dataclasses import replace
    editor, circuit, brain, meta = _editor_fixture()
    with editor.edit():
        meta._weight_params[('light', 'l1')] = {'pattern': 'uniform'}
        circuit.connections = [replace(circuit.connections[0], W=2 * np.eye(2))]
    assert editor.can_undo()
    assert editor.undo()
    assert meta._weight_params == {}
    np.testing.assert_allclose(circuit.connections[0].W, np.eye(2))
    assert not editor.can_undo()


def test_editor_no_change_records_no_undo_step_and_nesting_is_one_step():
    from neurons import LeakyLayer
    editor, circuit, brain, meta = _editor_fixture()
    with editor.edit():
        pass                                   # e.g. a cancelled dialog
    assert not editor.can_undo()
    with editor.edit():
        circuit.layers.append(LeakyLayer(name='l2', n=1))
        with editor.edit():                    # nested call joins the outer edit
            circuit.layers.append(LeakyLayer(name='l3', n=1))
    assert len(circuit.history.undo_stack) == 1
    editor.undo()
    assert [l.name for l in circuit.layers] == ['l1']


def test_editor_undo_restores_bodies_and_joints():
    from rigid_body import RigidBody, Joint
    editor, circuit, brain, meta = _editor_fixture()
    circuit.bodies = [RigidBody('root', 'root', 0.15)]
    with editor.edit():
        circuit.bodies.append(RigidBody('arm', 'arm', 0.05))
        circuit.joints.append(Joint(parent_id='root', child_id='arm', attach_dist=0.2,
                                    attach_angle=0.0, angle_min=-1.0, angle_max=1.0,
                                    motor_layer_name='arm', motor_output_idx=0))
    editor.undo()
    assert [b.id for b in circuit.bodies] == ['root']
    assert circuit.joints == []


def test_editor_syncs_brain_after_add_and_rename():
    """After an edit the brain sees every layer by name (renamed and added
    layers too, stale names removed), and step_network rebuilds its caches
    even when a layer was appended to the same list in place."""
    from neurons import LeakyLayer
    from circuit_model import Connection
    editor, circuit, brain, meta = _editor_fixture()
    brain.light = np.array([1.0, 0.5])
    brain.loop(0.01)                                       # warm the caches
    with editor.edit():
        circuit.layers[0].name = 'renamed'
        circuit.connections[0].tgt = 'renamed'
        circuit.layers.append(LeakyLayer(name='added', n=2, tau_rise=0.1, tau_decay=0.1))
        circuit.connections.append(Connection('renamed', 'added', np.eye(2)))
    assert 'l1' not in brain.__dict__
    assert brain.renamed is circuit.layers[0] and brain.added is circuit.layers[1]
    for _ in range(5):
        brain.loop(0.01)                                   # used to raise KeyError
    assert float(np.asarray(brain.added.output).sum()) > 0


def test_editor_history_is_per_circuit_bounded_and_clearable():
    from circuit_editor import CircuitEditor, clear_history, MAX_UNDO
    from neurons import LeakyLayer
    editor, circuit, brain, meta = _editor_fixture()
    for i in range(MAX_UNDO + 5):
        with editor.edit():
            circuit.layers.append(LeakyLayer(name=f'x{i}', n=1))
    assert len(circuit.history.undo_stack) == MAX_UNDO
    # A fresh editor on the same circuit sees the same history (e.g. window reopened).
    assert CircuitEditor(circuit, brain, meta=meta).can_undo()
    clear_history(circuit)
    assert not editor.can_undo()


@pytest.mark.skipif(not _HAS_PYTEST_QT, reason='pytest-qt not installed')
def test_network_window_remove_layer_undo_keeps_brain_runnable(qtbot):
    """Through the real network window: removing a layer and undoing it
    restores the circuit, re-syncs the brain, enables/disables the Undo
    button, and the brain keeps running after each edit."""
    from network_viz import NetworkVisualizerWindow
    from neurons import LeakyLayer
    from circuit_model import Connection

    editor, circuit, brain, meta = _editor_fixture()
    circuit.layers.append(LeakyLayer(name='l2', n=2, tau_rise=0.1, tau_decay=0.1))
    circuit.connections.append(Connection('l1', 'l2', np.eye(2)))
    gui = _FakeGui(circuit)
    gui.brain = brain
    editor.sync_brain()
    brain.light = np.array([1.0, 0.5])

    win = NetworkVisualizerWindow(gui)
    qtbot.addWidget(win)
    win.build()
    assert not win._btn_undo.isEnabled()

    win.editing.sel.node = 'l2'
    win.editing.remove_selected_layer()
    assert [l.name for l in circuit.layers] == ['l1']
    assert 'l2' not in brain.__dict__ and win._btn_undo.isEnabled()
    brain.loop(0.01)

    win._undo()
    assert [l.name for l in circuit.layers] == ['l1', 'l2']
    assert brain.l2 is circuit.layers[1]
    assert not win._btn_undo.isEnabled()
    for _ in range(3):
        brain.loop(0.01)


def _lateral_fixture():
    """Lateralized camera → conv_L/conv_R pair → shared 'out' layer, plus a
    plain layer that merely ends in _L."""
    from circuit_model import CircuitModel, Connection
    from neurons import LeakyLayer
    from sensors import GrayCameraSensor
    cam = GrayCameraSensor(width=8, height=4, lateralized=True, name='cam')
    conv_L = LeakyLayer(name='conv_L', n=2); conv_L.lateral_pair = 'conv_R'
    conv_R = LeakyLayer(name='conv_R', n=2); conv_R.lateral_pair = 'conv_L'
    out = LeakyLayer(name='out', n=2)
    solo_L = LeakyLayer(name='solo_L', n=2)
    circuit = CircuitModel(sensors=[cam], layers=[conv_L, conv_R, out, solo_L], connections=[
        Connection('cam_L', 'conv_L', np.ones((2, 16))),
        Connection('cam_R', 'conv_R', np.ones((2, 16))),
        Connection('cam_L', 'out', np.ones((2, 16))),
        Connection('cam_R', 'out', np.ones((2, 16))),
        Connection('solo_L', 'out', np.eye(2)),
        Connection('conv_L', 'out', np.eye(2)),
    ])
    return circuit


def test_session_brain_entry_code_and_network_formats():
    """Session files say whether a group runs a code brain or a network; the old
    form (network brain module + project/file params) still reads the same."""
    from session_io import brain_to_file, brain_from_file, NETWORK_BRAIN_MODULE
    net = brain_to_file(NETWORK_BRAIN_MODULE,
                        {'network_project': 'Tutorials', 'network_file': 'T06 - FeedingBrain.json'})
    assert net == {'mode': 'network', 'network': 'Tutorials/T06 - FeedingBrain.json'}
    assert brain_from_file(net) == (NETWORK_BRAIN_MODULE,
                                    {'network_project': 'Tutorials',
                                     'network_file': 'T06 - FeedingBrain.json'})
    old = {'module_name': NETWORK_BRAIN_MODULE,
           'brain_params': {'network_project': 'Tutorials', 'network_file': 'T06 - FeedingBrain.json'}}
    assert brain_from_file(old) == brain_from_file(net)
    code = brain_to_file('BrainARS', {'speed': 50.0})
    assert code == {'mode': 'code', 'module_name': 'BrainARS', 'brain_params': {'speed': 50.0}}
    assert brain_from_file(code) == ('BrainARS', {'speed': 50.0})
    no_project = brain_to_file(NETWORK_BRAIN_MODULE, {'network_project': '', 'network_file': 'a.json'})
    assert no_project['network'] == 'a.json'
    assert brain_from_file(no_project)[1] == {'network_project': '', 'network_file': 'a.json'}


def test_find_mirror_follows_lateral_pairs_only():
    from circuit_editor import find_mirror
    c = _lateral_fixture()
    assert find_mirror(c, 'cam_L', 'conv_L') == ('cam_R', 'conv_R')    # lat → lat pair
    assert find_mirror(c, 'cam_L', 'out') == ('cam_R', 'out')          # halves → shared target
    assert find_mirror(c, 'solo_L', 'out') is None                    # just a name ending in _L
    assert find_mirror(c, 'conv_L', 'out') is None                    # no mirror wired


def test_rename_layer_renames_lateral_pair_and_weight_settings():
    from types import SimpleNamespace
    from circuit_editor import rename_layer
    c = _lateral_fixture()
    meta = SimpleNamespace(_weight_params={('cam_R', 'conv_R'): {'pattern': 'x'}})
    conv_L = c.layers[0]
    renames = rename_layer(c, conv_L, 'edges', meta=meta)   # suffix kept automatically
    assert renames == {'conv_L': 'edges_L', 'conv_R': 'edges_R'}
    assert [l.name for l in c.layers[:2]] == ['edges_L', 'edges_R']
    assert c.layers[0].lateral_pair == 'edges_R' and c.layers[1].lateral_pair == 'edges_L'
    assert ('cam_R', 'edges_R') in {(x.src, x.tgt) for x in c.connections}
    assert ('edges_L', 'out') in {(x.src, x.tgt) for x in c.connections}
    assert meta._weight_params == {('cam_R', 'edges_R'): {'pattern': 'x'}}


def test_unpair_layer_turns_pair_back_into_one_layer():
    """Switching lateralized off: the edited half is kept under the base name,
    the partner and its connections / weight settings go."""
    from types import SimpleNamespace
    from circuit_editor import unpair_layer
    c = _lateral_fixture()
    meta = SimpleNamespace(_weight_params={('cam_R', 'conv_R'): {'p': 1}, ('conv_L', 'out'): {'p': 2}})
    removed, renames = unpair_layer(c, c.layers[0], meta=meta)
    assert removed == 'conv_R' and renames == {'conv_L': 'conv'}
    assert [l.name for l in c.layers] == ['conv', 'out', 'solo_L']
    assert c.layers[0].lateral_pair is None
    pairs = {(x.src, x.tgt) for x in c.connections}
    assert ('cam_L', 'conv') in pairs and ('conv', 'out') in pairs
    assert not any('conv_R' in p for p in pairs)
    assert meta._weight_params == {('conv', 'out'): {'p': 2}}


@pytest.mark.skipif(not _HAS_PYTEST_QT, reason='pytest-qt not installed')
def test_layer_edit_lateralized_off_removes_partner(qtbot):
    from network_viz import NetworkVisualizerWindow
    c = _lateral_fixture()
    win = NetworkVisualizerWindow(_FakeGui(c))
    qtbot.addWidget(win)
    win.build()
    conv_L = c.layers[0]
    conv_L.lateralized = True
    win.dialogs._apply_layer_edit(conv_L, [], {'lateralized': False}, '', 0, None, None, [])
    assert [l.name for l in c.layers] == ['conv', 'out', 'solo_L']
    assert not any('conv_R' in (x.src, x.tgt) for x in c.connections)


@pytest.mark.skipif(not _HAS_PYTEST_QT, reason='pytest-qt not installed')
def test_column_renumbering_keeps_hidden_flags_and_set_identity(qtbot):
    """Hidden / disabled columns follow their layers when columns are
    renumbered (side-view insert, compaction), and the sets are updated in
    place — the app holds and saves the same set objects."""
    from network_viz import NetworkVisualizerWindow
    c = _lateral_fixture()
    win = NetworkVisualizerWindow(_FakeGui(c))
    qtbot.addWidget(win)
    hidden, disabled = set(), set()            # the app's sets, shared with the window
    win._hidden_containers, win._disabled_containers = hidden, disabled
    win.build()
    out_col = win._lay.node_container_map['out_0']
    hidden.add(out_col)
    disabled.add(out_col)

    win._toggle_side_view()
    sv = win._side_view
    sv.refresh()
    solo_col = win._lay.node_container_map['solo_L_0']
    sv._on_insert_container(solo_col, 0, 0, 0)   # move solo_L into a new column after the first
    win.build()
    assert win._hidden_containers is hidden and win._disabled_containers is disabled
    new_out = win._lay.node_container_map['out_0']
    assert hidden == {new_out} and disabled == {new_out}
    sv.close()


@pytest.mark.skipif(not _HAS_PYTEST_QT, reason='pytest-qt not installed')
def test_network_window_delete_connection_removes_mirror_and_column_disable(qtbot):
    """Deleting cam_L → out also deletes its mirror cam_R → out; disabling one
    column mutes only that column's layers (it used to mute every layer)."""
    from network_viz import NetworkVisualizerWindow
    c = _lateral_fixture()
    win = NetworkVisualizerWindow(_FakeGui(c))
    qtbot.addWidget(win)
    win.build()

    win.editing.sel.edge = ('cam_L', 'out')
    win.editing.remove_selected_connection()
    pairs = {(x.src, x.tgt) for x in c.connections}
    assert ('cam_L', 'out') not in pairs and ('cam_R', 'out') not in pairs
    assert ('cam_L', 'conv_L') in pairs

    out_container = win._lay.node_container_map['out_0']
    win._on_container_disable(out_container, True)
    muted = {l.name for l in c.layers if getattr(l, 'muted', False)}
    assert 'out' in muted and 'conv_L' not in muted


@pytest.mark.skipif(not _HAS_PYTEST_QT, reason='pytest-qt not installed')
def test_new_layer_type_needs_only_its_own_file(qtbot):
    """A layer type defined entirely here — declaring its params and nothing
    else — saves/loads, runs in step_network and renders in the network window
    without any other module knowing about it (rules/network_elements.md §4)."""
    from neurons_base import DynamicsBase, LayerBase
    from circuit_model import CircuitModel, Connection
    from sensors import GradientSensor
    from brain_serializer import serialize_network_json, load_network_json
    from network_runner import step_network
    from network_viz import NetworkVisualizerWindow
    from brain_base import BaseBrain

    class TestHalfWaveLayer(DynamicsBase, LayerBase):
        """Test-only layer: output = gain * relu(input)."""
        def __init__(self, n=2, gain=2.0, name='halfwave', **kwargs):
            super().__init__(name=name, **kwargs)
            self.n    = n
            self.gain = float(gain)
            self._init_dynamics_buffers(n)
            self.output = torch.zeros(n)

        @classmethod
        def param_defs(cls):
            return [('n', int, 2, 'neurons'), ('gain', float, 2.0, 'output gain')]

        def step(self, input_vec, dt):
            self.output = torch.relu(torch.as_tensor(input_vec, dtype=torch.float32)) * self.gain
            return self.output

        def reset(self):
            self._reset_dynamics()
            self.output = torch.zeros(self.n)

    light = GradientSensor(name='light', n=2)
    hw = TestHalfWaveLayer(name='hw', gain=3.0)
    circuit = CircuitModel(sensors=[light], layers=[hw],
                           connections=[Connection('light', 'hw', np.eye(2))])

    # Save → load keeps the type and its own param.
    data = serialize_network_json(circuit.sensors, circuit.layers, circuit.connections,
                                  set(), set(), {})
    _s, layers, *_ = load_network_json(json.loads(json.dumps(data)))
    assert type(layers[0]) is TestHalfWaveLayer and layers[0].gain == 3.0
    assert not layers[0].accepts_image and not layers[0].is_image_node   # capability defaults

    # Runs in the forward pass.
    class _Brain(BaseBrain):
        def loop(self, dt):
            return 0.0, 0.0
    brain = _Brain()
    brain.sensors, brain.layers, brain.connections = circuit.sensors, circuit.layers, circuit.connections
    brain.hw = hw
    brain.light = np.array([0.5, -1.0], dtype=np.float32)
    step_network(brain, 0.01)
    np.testing.assert_allclose(hw.output.numpy(), [1.5, 0.0], atol=1e-6)

    # Renders in the network window.
    win = NetworkVisualizerWindow(_FakeGui(circuit))
    qtbot.addWidget(win)
    win.build()
    assert 'hw_0' in win._lay.positions


def test_lateral_helpers():
    from lateral import (side_of, base_name, mirror_name, half_names, partner_layer,
                         parent_sensor, is_camera_half, is_body_pair_half, is_lateral_half)
    c = _lateral_fixture()
    assert side_of('cam_L') == 'L' and side_of('cam') is None
    assert base_name('conv_R') == 'conv' and base_name('out') == 'out'
    assert mirror_name('conv_L') == 'conv_R' and mirror_name('out') is None
    assert half_names('cam') == ('cam_L', 'cam_R')
    assert partner_layer(c.layers, c.layers[0]) is c.layers[1]
    assert partner_layer(c.layers, c.layers[3]) is None            # solo_L has no partner
    assert parent_sensor(c.sensors, 'cam_R') is c.sensors[0]
    assert parent_sensor(c.sensors, 'cam') is None
    assert is_camera_half(c.sensors, 'cam_L') and not is_body_pair_half(c.sensors, 'cam_L')
    assert is_lateral_half(c, 'conv_L') and not is_lateral_half(c, 'solo_L')


def test_load_syncs_every_lateral_pair_param():
    """On load, an L/R pair's _R side takes the _L side's params — for every
    pair-capable layer type (Leaky2dLayer used to be skipped)."""
    from neurons import Leaky2dLayer
    from brain_serializer import serialize_network_json, load_network_json
    l = Leaky2dLayer(name='img_L', lateralized=True, tau_rise=0.2, tau_decay=0.3)
    r = Leaky2dLayer(name='img_R', lateralized=True, tau_rise=0.9, tau_decay=0.9)
    l.lateral_pair, r.lateral_pair = 'img_R', 'img_L'
    data = serialize_network_json([], [l, r], [], set(), set(), {})
    _s, layers, *_ = load_network_json(json.loads(json.dumps(data)))
    loaded = {x.name: x for x in layers}
    assert loaded['img_R'].tau_rise == 0.2 and loaded['img_R'].tau_decay == 0.3


def test_freshness_check_satisfied_after_resave():
    """A sensor param whose value is None (GradientSensor.gradient = all labels)
    is not written to the file — the freshness check must not report it as
    missing, or the 'Network file outdated' prompt reappears after every save.
    A genuinely missing param is still reported."""
    from sensors import GradientSensor
    from brain_serializer import serialize_network_json, load_network_json, check_network_freshness
    s = GradientSensor(name='sensor1', n=2)            # gradient=None
    assert s.gradient is None
    data = json.loads(json.dumps(serialize_network_json([s], [], [], set(), set(), {})))
    sensors, layers, *_ = load_network_json(data)
    assert check_network_freshness(data, sensors, layers) == []

    del data['sensors'][0]['scale']                    # an old file without 'scale'
    issues = check_network_freshness(data, sensors, layers)
    assert issues and issues[0]['missing'] == ['scale']


def test_sensors_and_layers_share_dynamics():
    """Sensors and layers use the same leaky filter and tau rules
    (neurons_base.leaky_step): a sensor and a LeakyLayer fed the same input
    produce the same output; tau_decay unset = rise-and-hold; tau_rise unset =
    no filtering (even with tau_decay set)."""
    from types import SimpleNamespace
    from sensors import GradientSensor
    from neurons import LeakyLayer
    cfg = SimpleNamespace(dt=0.01)
    inputs = [1.0] * 30 + [0.0] * 30

    def run_sensor(**kw):
        s = GradientSensor(name='s', n=1, scale=1.0, **kw)
        return [float(s._process(np.array([u]), cfg)[0]) for u in inputs]

    def run_layer(**kw):
        l = LeakyLayer(name='l', n=1, activation='linear', **kw)
        return [float(l.step(torch.tensor([u]), 0.01)[0]) for u in inputs]

    np.testing.assert_allclose(run_sensor(tau_rise=0.05, tau_decay=0.2),
                               run_layer(tau_rise=0.05, tau_decay=0.2), atol=1e-6)
    hold = run_sensor(tau_rise=0.05, tau_decay=None)
    assert hold[-1] == pytest.approx(hold[29]) and hold[29] > 0.9   # rises, then holds
    np.testing.assert_allclose(hold, run_layer(tau_rise=0.05, tau_decay=None), atol=1e-6)
    assert run_sensor(tau_rise=None, tau_decay=0.2) == inputs          # no filtering


def test_derivative_output_mode_is_zero_on_first_step():
    """output_mode='derivative' outputs 0 on the first step (no previous
    value) for layers as for sensors — layers used to spike value/dt."""
    from types import SimpleNamespace
    from sensors import GradientSensor
    from neurons import LeakyLayer
    l = LeakyLayer(name='l', n=1, tau_rise=0, activation='linear', output_mode='derivative')
    assert float(l.step(torch.tensor([5.0]), 0.01)[0]) == 0.0
    assert float(l.step(torch.tensor([6.0]), 0.01)[0]) == pytest.approx(100.0)
    l.reset()
    assert float(l.step(torch.tensor([7.0]), 0.01)[0]) == 0.0        # again after reset
    s = GradientSensor(name='s', n=1, scale=1.0, output_mode='derivative')
    assert float(s._process(np.array([5.0]), SimpleNamespace(dt=0.01))[0]) == 0.0


def test_simulation_runs_without_qt():
    """The simulation core (Simulation + step) must not import Qt, so it can
    run headless. Checked in a fresh interpreter so other tests' imports
    don't hide a regression."""
    import subprocess
    code = (
        "import sys; sys.path[:0] = [%r, %r]\n"
        "import simulation, sim_engine, agent_registry, session_loader, headless\n"
        "bad = [m for m in sys.modules if m.startswith(('PySide6', 'pyqtgraph'))]\n"
        "print('QT:' + ','.join(bad))\n" % (_SRC, _SIM2D)
    )
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=_SIM2D)
    assert out.returncode == 0, out.stderr
    assert 'QT:\n' in out.stdout or out.stdout.strip().endswith('QT:'), out.stdout


def test_step_agents_renders_due_cameras_only():
    """Cameras with fps > 0 render inside the step on their own schedule;
    fps == 0 cameras are left to the display loop."""
    from types import SimpleNamespace
    from sim_engine import step_agents
    from circuit_model import CircuitModel
    from sensors import GrayCameraSensor
    from world import World
    from brain_base import BaseBrain

    class _Brain(BaseBrain):
        def loop(self, dt):
            return 0.0, 0.0

    fixed = GrayCameraSensor(width=4, height=2, fps=50.0, name='fixed')
    free  = GrayCameraSensor(width=4, height=2, fps=0.0,  name='free')
    agent = SimpleNamespace(bot_pos=[0.0, 0.0, 0.0], brain=_Brain(),
                            circuit=CircuitModel(sensors=[fixed, free]))
    engine = _FakeEngine()
    step_agents([agent], World(_StepCfg()), _StepCfg(), engine, 0.0)
    assert ('cameras', ['fixed'], 0.0) in engine.calls
    assert not any(c[0] == 'cameras' and 'free' in c[1] for c in engine.calls if isinstance(c, tuple))


def _one_learning_step(layer_cls, dt, ticks=1, **kw):
    """W after `ticks` steps of a learning layer fed by a constant input 1.0
    and a constant reward 0.5 (legacy reward_modulator path)."""
    from network_runner import step_network
    from neurons import ConstantLayer
    from circuit_model import Connection
    from brain_base import BaseBrain
    dan = ConstantLayer(name='dan', value=0.5, n=1, neuromodulator_transmitter='dopamine')
    src = ConstantLayer(name='src', value=1.0, n=1)
    learn = layer_cls(name='learn', n=1, alpha_pos=0.1, alpha_neg=0.1,
                      reward_modulator='dopamine', **kw)
    conn = Connection(src='src', tgt='learn', W=np.zeros((1, 1), dtype=np.float32))
    brain = BaseBrain()
    brain.sensors, brain.layers, brain.connections = [], [dan, src, learn], [conn]
    brain.src = src
    for _ in range(ticks):
        dan.output = torch.tensor([0.5])
        step_network(brain, dt=dt)
    return float(conn.W[0, 0])


def test_learning_rate_scales_with_dt():
    """Learning rates are per tick at the default dt 0.01 and scale with dt
    (TODO 1.4): one tick at dt=0.02 learns twice what one tick at 0.01 does,
    and the default dt applies alpha unchanged."""
    from neurons import DeltaLayer
    w_ref = _one_learning_step(DeltaLayer, 0.01)
    assert w_ref == pytest.approx(0.1 * 0.5)            # alpha · (r − V) · s
    assert _one_learning_step(DeltaLayer, 0.02) == pytest.approx(2 * w_ref)
    assert _one_learning_step(DeltaLayer, 0.010000000000000002) == w_ref   # GUI's dt: exact


def test_td_credits_previous_input_delta_current():
    """TD's δ compares V_t with V_{t-1}, so it credits the previous tick's input
    (nothing on the first tick); Delta / ThreeFactor credit this tick's input
    (TODO 1.5)."""
    from neurons import DeltaLayer, TDLayer
    assert _one_learning_step(TDLayer, 0.01) == 0.0
    assert _one_learning_step(TDLayer, 0.01, ticks=2) != 0.0
    assert _one_learning_step(DeltaLayer, 0.01) != 0.0


def test_fast_taus_reports_only_taus_shorter_than_dt():
    """TODO 1.1: taus shorter than dt are reported (never changed); 0 / blank
    taus are 'off' and not reported."""
    from neurons import LeakyLayer
    from neurons_base import fast_taus, fast_tau_warning
    ok   = LeakyLayer(name='ok', n=1, tau_rise=0.05, tau_decay=0.0)
    fast = LeakyLayer(name='fast', n=1, tau_rise=0.004, tau_decay=0.02)
    hits = fast_taus([ok, fast], 0.01)
    assert hits == [('fast', 'tau_rise', 0.004)]
    assert fast.tau_rise == 0.004                        # untouched
    assert 'blow up' in fast_tau_warning(hits, 0.01)     # dt/tau = 2.5 > 2
    assert fast_tau_warning([], 0.01) == ''


def test_mute_keeps_layer_state_and_constant_comes_back():
    """TODO 2.1: muting zeroes a layer's output without touching its state —
    an AccumulatorLayer resumes from what it had accumulated, and a
    ConstantLayer's value is back after unmute."""
    from network_runner import step_network
    from neurons import ConstantLayer, AccumulatorLayer
    from circuit_model import Connection
    from brain_base import BaseBrain
    src = ConstantLayer(name='src', value=1.0, n=1)
    acc = AccumulatorLayer(name='acc', n=1, rate=1.0, zero_center=False)
    brain = BaseBrain()
    brain.sensors, brain.layers = [], [src, acc]
    brain.connections = [Connection(src='src', tgt='acc', W=np.ones((1, 1), dtype=np.float32))]
    brain.src, brain.acc = src, acc
    for _ in range(10):
        step_network(brain, dt=0.01)
    stored = float(acc.output[0])
    assert stored == pytest.approx(0.1)
    acc.muted = True
    step_network(brain, dt=0.01)
    assert float(acc.output[0]) == 0.0                     # silent while muted
    acc.muted = False
    step_network(brain, dt=0.01)
    assert float(acc.output[0]) == pytest.approx(stored + 0.01)   # resumed, not restarted
    src.muted = True
    step_network(brain, dt=0.01)
    src.muted = False
    step_network(brain, dt=0.01)
    assert float(np.asarray(src.output)[0]) == 1.0         # constant is back


def test_learning_layer_bias_integral_and_late_tau_rise():
    """TODO 2.2 / 2.3: a learning layer adds bias even with tau_rise = 0, runs
    with output_mode='integral', and keeps working when tau_rise is raised
    after construction (as the edit dialog does)."""
    from neurons import DeltaLayer
    biased = DeltaLayer(name='b', n=1, bias=0.5)
    biased.reset()
    assert float(biased.step_td([], 0.01)[0]) == 0.0          # no inputs → silent
    src = torch.ones(1)
    W = torch.zeros(1, 1)
    out = biased.step_td([(src, W, 0, type('C', (), {'W': None})())], 0.01)
    assert float(out[0]) == pytest.approx(0.5)                 # bias applied at tau_rise = 0
    integ = DeltaLayer(name='i', n=1, output_mode='integral')
    integ.reset()
    integ.step_td([(src, torch.ones(1, 1), 0, type('C', (), {'W': None})())], 0.01)
    late = DeltaLayer(name='l', n=1)
    late.tau_rise = 0.05
    late.reset()
    late.step_td([(src, torch.ones(1, 1), 0, type('C', (), {'W': None})())], 0.01)


def test_modulator_rows_sharing_mode_advance_once_per_tick():
    """TODO 2.5: a 'pre' and a 'post' row on the same modulator and mode share
    their derivative state; it must advance once per tick, so both rows see
    the same (non-zero) derivative."""
    from network_runner import step_network
    from neurons import ConstantLayer, LeakyLayer
    from circuit_model import Connection
    from brain_base import BaseBrain
    dan = ConstantLayer(name='dan', value=0.0, n=1, neuromodulator_transmitter='dopamine')
    src = ConstantLayer(name='src', value=1.0, n=1)
    seen = []
    tgt = LeakyLayer(name='tgt', n=1, tau_rise=0.0, activation='linear',
                     modulators=[('dopamine', 1.0, 'pre', 'derivative'),
                                 ('dopamine', 1.0, 'post', 'derivative')])
    orig = tgt._transform_modulator_value
    tgt._transform_modulator_value = lambda *a: seen.append(orig(*a)) or seen[-1]
    brain = BaseBrain()
    brain.sensors, brain.layers = [], [dan, src, tgt]
    brain.connections = [Connection(src='src', tgt='tgt', W=np.ones((1, 1), dtype=np.float32))]
    brain.src, brain.tgt = src, tgt
    for v in (0.0, 0.5):
        dan.output = np.array([v])
        step_network(brain, dt=0.01)
    assert len(seen) == 2                       # one transform per tick, not one per row
    assert seen[-1] == pytest.approx(50.0)      # (0.5 − 0) / 0.01


def test_codegen_keeps_every_non_default_param():
    """TODO 2.6: init_code_parts (export as a Python brain) round-trips the
    shared dynamics params for Matsuoka / Pulse / RingAttractor layers."""
    from neurons import MatsuokaLayer, PulseLayer, RingAttractorLayer
    extra = dict(activation='tanh', scale=2.0, noise_std=0.1, noise_tau=0.2, x0=0.3, alpha=0.5)
    for cls, kw in ((MatsuokaLayer, dict(tau_decay=0.4)), (PulseLayer, dict(n=2)),
                    (RingAttractorLayer, dict(n=8))):
        layer = cls(name='x', **kw, **extra)
        code = ', '.join(layer.init_code_parts())
        rebuilt = cls(**eval(f'dict({code})'))
        for k in list(extra) + list(kw):
            assert getattr(rebuilt, k) == getattr(layer, k), (cls.__name__, k)


def test_every_dynamics_layer_applies_noise_and_scale():
    """TODO 2.4: noise and scale come from the shared DynamicsBase pipeline
    (_input / _filter / _emit), so they work on every layer — including the
    ones whose step() used to skip them (Pulse / Conv2d / learning noise,
    RingAttractor scale)."""
    from neurons import PulseLayer, RingAttractorLayer, Conv2dLayer, DeltaLayer

    def run(layer, steps=5):
        layer.reset()
        torch.manual_seed(0)
        out = None
        for _ in range(steps):
            if hasattr(layer, 'step_td'):
                out = layer.step_td([(torch.ones(2), torch.eye(2), 0, type('C', (), {'W': None})())], 0.01)
            else:
                out = layer.step(torch.ones(layer.n), 0.01)
        return out.clone()

    for cls, kw in ((PulseLayer, dict(n=2)), (Conv2dLayer, dict(n=2)), (DeltaLayer, dict(n=2))):
        quiet = run(cls(name='q', activation='linear', **kw))
        noisy = run(cls(name='n', activation='linear', noise_std=0.5, **kw))
        assert not torch.allclose(quiet, noisy), f'{cls.__name__}: noise_std has no effect'
    base   = run(RingAttractorLayer(name='r1', n=8, tau_rise=0.05))
    scaled = run(RingAttractorLayer(name='r2', n=8, tau_rise=0.05, scale=3.0))
    assert torch.allclose(scaled, 3.0 * base), 'RingAttractorLayer: scale has no effect'
