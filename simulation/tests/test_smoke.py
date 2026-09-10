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
    step_network(brain, dt=0.1)
    assert conn.W[0, 0] == pytest.approx(0.0)

    dan.output = torch.tensor([0.9])   # above threshold 0.5
    step_network(brain, dt=0.1)
    assert conn.W[0, 0] == pytest.approx(0.9)   # alpha_pos=1.0 * (0.9 - 0) * s_prev(1.0)


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
    step_network(brain, dt=0.1)   # first tick: warms up _src_prev (LearningLayerBase.step_td
                                   # uses the PREVIOUS tick's presynaptic value, empty on tick 1 —
                                   # ΔW is always 0 on a layer's very first step_td call regardless
                                   # of reward, see LearningLayerBase.step_td's src_prev lookup)
    assert conn.W[0, 0] == pytest.approx(0.0)
    step_network(brain, dt=0.1)   # second tick: now sees src_prev=1.0 from tick 1
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
    """_connection_kind must return TEACH for the src==teach_source connection
    into a SnapshotLayer, and TD for that same layer's other (gate) incoming
    connection — TEACH is checked with higher priority than TD since a
    SnapshotLayer is itself a LearningLayerBase. _connection_kind only reads
    class-level _CK_*/_DENSE_THRESHOLD constants, so a real Qt widget (and
    pytest-qt) isn't needed — __new__ skips __init__ entirely."""
    from network_viz import NetworkVisualizerWindow
    from neurons import ConstantLayer, SnapshotLayer

    cpu4 = ConstantLayer(name='cpu4', value=[0.0, 0.0, 0.0], n=3)
    gate = ConstantLayer(name='gate', value=0.0, n=1)
    mem  = SnapshotLayer(name='mem', n=1, teach_source='cpu4')

    win = NetworkVisualizerWindow.__new__(NetworkVisualizerWindow)
    teach_kind = win._connection_kind(cpu4, mem, np.ones((1, 3)), 3, 1)
    gate_kind  = win._connection_kind(gate, mem, np.ones((1, 1)), 1, 1)
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
    sim_controller._tick / MuJoCoEngine.tick_physics_batch) must be treated as
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
    only reachable via network_viz_dialogs.py's _layer_dialog, which
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

    win._update_activation_panel()
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
    assert 's_L' not in win._active_names

    # Same-column drag (still dv=0) raising z to 1, tying with the competitor.
    sv.refresh()
    old_zi2 = sv._z_levels.index(0)
    sv._on_container_dropped(0, 0, sv._ordered_containers.index(0), old_zi2,
                             sv._ordered_containers.index(0), sv._z_levels.index(1))

    assert sensor.z == 1   # used to stay frozen; same-column branch never touched sensor.z

    win.build()
    assert 's_L' in win._active_names and 's_R' in win._active_names
    assert not win._subsumed_by


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
    win._paste_selection()

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
            self._agents = agents

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
            self._agents = agents

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
    ctrl.brain = agent.brain

    world.patches.append({'x': 0.0, 'y': 0.0, 'r': 0.3, 'mounted_on': agent.id})

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
    """A MuJoCo-eligible CollisionSensor sampled through MuJoCoEngine.tick_physics
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
    from sim_engine_mujoco import MuJoCoEngine
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
        engine.tick_physics([x, 0.0, 0.0], brain, [touch, lookahead], world, cfg)
        assert hit('touch') == expect_touch, f'x={x}: touch expected {expect_touch}, got {brain.touch}'
        assert hit('lookahead') == expect_lookahead, f'x={x}: lookahead expected {expect_lookahead}, got {brain.lookahead}'


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('mujoco') is None,
    reason='mujoco not installed')
def test_mujoco_tick_physics_batch_distance_sensor_sees_other_agent():
    """tick_physics_batch's other_agents wiring (built once per tick as a
    pre-tick position snapshot, sliced to exclude self) must reach each
    agent's DistanceSensor — the exact call site sim_controller._tick uses
    for the MuJoCo multi-agent path."""
    from sim_engine_mujoco import MuJoCoEngine
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

    agent_list = [
        ([0.0, 0.0, 0.0], brain0, [dist0], None),
        ([0.5, 0.0, np.pi], brain1, [dist1], None),
    ]
    engine.tick_physics_batch(agent_list, world, cfg)

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
    from sim_engine_mujoco import MuJoCoEngine
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
        engine.tick_physics([x, y, theta], brain, [sensor], world, cfg)

        if not np.array_equal(analytic, brain.touch):
            mismatches += 1

    # Measured ~3% on a similar distribution during development, concentrated at
    # exact detection-boundary cases (discretization) and wall contacts (MuJoCo's
    # walls have real thickness; the analytic path idealizes them as thin lines).
    assert mismatches / n_trials < 0.10, f'{mismatches}/{n_trials} mismatches — regression?'
