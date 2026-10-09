import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = ['relu', 'sigmoid', 'tanh', 'linear', 'heaviside', 'hard_sigmoid', 'elu']
OUTPUT_MODES = ['none', 'derivative', 'integral']


def _activate(x, name: str, alpha: float = 1.0):
    """Activation — dispatches on torch.Tensor vs numpy array so sensors.py stays unchanged.

    alpha only affects 'elu' (negative-side saturation level); every other
    activation ignores it.
    """
    if isinstance(x, torch.Tensor):
        if name == 'relu':         return F.relu(x)
        if name == 'sigmoid':      return torch.sigmoid(x)
        if name == 'tanh':         return torch.tanh(x)
        if name == 'linear':       return x
        if name == 'heaviside':    return (x > 0).float()
        if name == 'hard_sigmoid': return F.hardsigmoid(x)
        if name == 'elu':          return F.elu(x, alpha=alpha)
    else:
        x = np.asarray(x, dtype=float)
        if name == 'relu':         return np.maximum(0.0, x)
        if name == 'sigmoid':      return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        if name == 'tanh':         return np.tanh(x)
        if name == 'linear':       return x
        if name == 'heaviside':    return (x > 0).astype(float)
        if name == 'hard_sigmoid': return np.clip(x / 6.0 + 0.5, 0.0, 1.0)
        if name == 'elu':          return np.where(x > 0, x, alpha * np.expm1(np.clip(x, -500, 0)))
    raise ValueError(f"Unknown activation '{name}'. Choose from: {ACTIVATIONS}")


# ── Shared dynamics math ───────────────────────────────────────────────────────
# One implementation for layers (torch tensors, in nn.Module buffers) and
# sensors (numpy arrays). Each function takes the current state and returns
# the new state; the caller decides where the state lives.

def leaky_step(x, u, tau_rise, tau_decay, dt):
    """Asymmetric leaky integration: x moves toward u with tau_rise while rising
    and tau_decay while falling. Either tau unset (None or 0) means x snaps
    straight to u on that side instead of easing toward it — e.g. tau_rise
    unset + tau_decay set gives a fast-attack/slow-decay envelope follower.
    Callers skip this entirely when both taus are unset (no state to track,
    output just follows input). Every tau follows the same rule: 0 or blank
    switches its own dynamic off (instant), never divides by zero."""
    if not tau_rise:
        tau_rise = None
    if not tau_decay:
        tau_decay = None
    if isinstance(x, torch.Tensor):
        if tau_rise == tau_decay:          # both None (full passthrough) or both equal (symmetric)
            return u if tau_rise is None else x + (u - x) / tau_rise * dt
        rising = u > x
        if tau_rise is None:
            delta = torch.where(rising, u - x, (u - x) / tau_decay * dt)
        elif tau_decay is None:
            delta = torch.where(rising, (u - x) / tau_rise * dt, u - x)
        else:
            tau = torch.where(rising, tau_rise, tau_decay)   # scalars: no temporary tensors
            delta = (u - x) / tau * dt
        return x + delta
    rising = u > x
    if tau_rise is None and tau_decay is None:
        return u
    if tau_rise is None:
        return x + np.where(rising, u - x, (u - x) / tau_decay * dt)
    if tau_decay is None:
        return x + np.where(rising, (u - x) / tau_rise * dt, u - x)
    tau = np.where(rising, tau_rise, tau_decay)
    return x + (u - x) / tau * dt


# The time step every network was tuned at (SimConfig's default dt). Per-tick
# quantities that should not depend on dt — correlated-noise kicks, learning
# rates — are defined at this dt and rescaled at any other dt.
DT_REF = 0.01


def dt_ratio(dt):
    """dt / DT_REF, but exactly 1.0 at the default time step so runs there are
    bit-identical. Tolerance, not ==: the GUI's dt arrives as
    0.010000000000000002 (slider rounding)."""
    return 1.0 if abs(dt - DT_REF) < 1e-9 else dt / DT_REF


# Every time constant integrated with explicit Euler (x += dt/tau · …).
TAU_ATTRS = ('tau_rise', 'tau_decay', 'tau_a', 'tau_hold', 'noise_tau')


def fast_taus(elements, dt):
    """(element name, tau name, value) for every tau shorter than the time step.
    Explicit Euler overshoots when dt/tau > 1 and blows up when dt/tau > 2.
    Only reported, never corrected — the user decides (TODO 1.1)."""
    hits = []
    for el in elements:
        for attr in TAU_ATTRS:
            try:
                tau = float(getattr(el, attr, None) or 0.0)
            except (TypeError, ValueError):
                continue
            if tau > 0 and dt / tau > 1:
                hits.append((el.name, attr, tau))
    return hits


def fast_tau_warning(hits, dt):
    """Text for the user about fast_taus() hits, or '' if there are none."""
    if not hits:
        return ''
    worst = max(dt / tau for _n, _a, tau in hits)
    items = ', '.join(f'{n}.{a} = {tau:g}' for n, a, tau in hits[:6])
    more = f' (+{len(hits) - 6} more)' if len(hits) > 6 else ''
    effect = 'will blow up (NaN / huge values)' if worst > 2 else 'will overshoot and ring'
    return (f'⚠ Time constant shorter than the time step dt = {dt:g} s: {items}{more}. '
            f'The integration {effect}. Use a tau ≥ dt, or a smaller dt (Physics tab).')


def ou_noise_step(buf, noise_std, noise_tau, dt):
    """One Ornstein-Uhlenbeck step of correlated noise (correlation time noise_tau).

    Random kicks add up as √(number of steps), so the kick scales with √dt
    (Euler–Maruyama): kick = noise_std · √(dt · DT_REF) · N(0, 1). At
    dt = DT_REF that is exactly noise_std · dt · N(0, 1) — the original
    formula, so saved networks behave identically — and at any other dt the
    noise keeps the same size instead of shrinking with dt. Stationary std of
    the noise: noise_std · √(noise_tau · DT_REF / 2)."""
    k = 1.0 / math.sqrt(dt_ratio(dt))
    if isinstance(buf, torch.Tensor):
        return buf + (-buf / noise_tau + noise_std * k * torch.randn_like(buf)) * dt
    return buf + (-buf / noise_tau + noise_std * k * np.random.randn(*np.shape(buf))) * dt


def transform_modulator_value(state, key, mode, value, dt):
    """Apply a modulator response mode to a raw scalar reading from the mod bus.

    'absolute'   — pass through unchanged.
    'derivative' — rate of change since the last call with this same key
                   (zero on the first call — no previous value yet).
    'integral'   — running accumulation over time (forward-Euler): each call
                   adds `value * dt`, not `value` itself, so the accumulator
                   grows more slowly at a smaller `dt` for the same reading.

    State is tracked per `key` (conventionally `(modulator_name, mode)`) in the
    dict *state*, independent of the owner's own output_mode state and of every
    other modulator row, so multiple subscriptions never collide.
    """
    if mode == 'derivative':
        row = state.setdefault(key, {'prev': value})
        prev = row['prev']
        row['prev'] = value
        return (value - prev) / max(float(dt), 1e-9)
    if mode == 'integral':
        row = state.setdefault(key, {'integral': 0.0})
        row['integral'] += value * dt
        return row['integral']
    return value


class DynamicsBase:
    """
    Mixin providing shared leaky dynamics, adaptation, and noise parameters and helpers.

    Layer subclasses call super().__init__(name=name, **kwargs) — DynamicsBase.__init__
    calls _init_dynamics() automatically. They then call _init_dynamics_buffers(n)
    themselves, because n depends on layer-specific setup.

    Sensor subclasses (BaseSensor) call _init_dynamics() directly and do NOT use the
    cooperative __init__ chain.

    All three state buffers (_x, _a, _noise_buf) are always registered when
    _init_dynamics_buffers is called. Helper methods guard on params before use.
    """

    def __init__(self, *, tau_rise=0.1, tau_decay=None, activation='relu',
                 bias=0.0, scale=1.0, noise_std=0.0, noise_tau=0.0,
                 tau_a=0.0, beta=0.0, output_mode='none', x0=0.0, alpha=1.0, **kwargs):
        super().__init__(**kwargs)  # → LayerBase → nn.Module
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay,
                            activation=activation, bias=bias, scale=scale,
                            tau_a=tau_a, beta=beta,
                            noise_std=noise_std, noise_tau=noise_tau,
                            output_mode=output_mode, x0=x0, alpha=alpha)

    def _init_dynamics(self, tau_rise=0.1, tau_decay=None, activation='relu',
                       bias=0.0, scale=1.0, tau_a=0.0, beta=0.0,
                       noise_std=0.0, noise_tau=0.0, output_mode='none', x0=0.0, alpha=1.0):
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"Unknown output_mode '{output_mode}'. Choose from: {OUTPUT_MODES}")
        self.tau_rise     = float(tau_rise) if tau_rise is not None else None
        self.tau_decay    = float(tau_decay) if tau_decay is not None else None
        self.activation   = activation
        self.bias         = float(bias)
        self.scale        = float(scale)
        self.tau_a        = float(tau_a)
        self.beta         = float(beta)
        self.noise_std    = float(noise_std)
        self.noise_tau    = float(noise_tau)
        self.output_mode  = output_mode
        self.x0           = x0
        self.alpha        = float(alpha)

    def _x0_tensor(self, n):
        """Broadcast self.x0 (scalar or per-neuron list) to a length-n tensor.

        Used to seed _x/_integral so a layer's persistent state starts at a
        biologically-meaningful resting point (e.g. a bounded integrator that
        needs to move both up and down from a mid-range value) instead of
        always starting at 0.
        """
        t = torch.as_tensor(getattr(self, 'x0', 0.0), dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            return t.expand(n).clone()
        if t.numel() != n:
            raise ValueError(f"x0 has length {t.numel()} but n={n}")
        return t.clone()

    def _init_dynamics_buffers(self, n):
        """Register nn.Module buffers for all dynamics state. Only call from layer classes.

        _x and _integral seed from x0 (the layer's declared initial state);
        _a/_noise_buf/_prev_out have no meaningful nonzero rest state (no
        adaptation/noise/history yet) and always start at zero.
        """
        self.register_buffer('_x',         self._x0_tensor(n))
        self.register_buffer('_a',         torch.zeros(n))
        self.register_buffer('_noise_buf', torch.zeros(n))
        self.register_buffer('_prev_out',  torch.zeros(n))
        self.register_buffer('_integral',  self._x0_tensor(n))
        self._prev_valid = False   # _prev_out holds no real value until the first step

    def _reset_dynamics(self):
        self._prev_valid = False
        for attr in ('_a', '_noise_buf', '_prev_out'):
            buf = getattr(self, attr, None)
            if buf is not None:
                buf.zero_()
        for attr in ('_x', '_integral'):
            buf = getattr(self, attr, None)
            if buf is not None:
                buf.copy_(self._x0_tensor(buf.numel()).reshape(buf.shape))
        mod_state = getattr(self, '_mod_row_state', None)
        if mod_state is not None:
            mod_state.clear()

    def _apply_output_mode(self, out, dt):
        """Transform *out* per self.output_mode:

        'none'       — pass through unchanged.
        'derivative' — dOut/dt via backward finite difference (zero on the
                       first tick, since there is no previous value yet).
        'integral'   — running ∫Out dt via forward-Euler accumulation.

        Each tick adds `out * dt` to the running total, not `out` itself —
        the same per-tick drive accumulates more slowly at a smaller `dt`,
        so the number of ticks needed to reach a given accumulated value
        scales with 1/dt, not with the drive alone.

        Both non-'none' modes keep the stored history buffer detached (a
        constant from the current step's perspective) while letting gradient
        flow through the current *out* — the same single-step-detach
        convention as _apply_leaky/_apply_adaptation_pre.
        """
        mode = getattr(self, 'output_mode', 'none')
        if mode == 'derivative':
            prev = self._prev_out.detach().clone()
            self._prev_out.copy_(out.detach())
            if not getattr(self, '_prev_valid', False):
                self._prev_valid = True
                return torch.zeros_like(out)      # no previous value on the first step
            return (out - prev) / max(float(dt), 1e-9)
        if mode == 'integral':
            prev_integral = self._integral.detach().clone()
            self._integral.copy_(prev_integral + out.detach() * dt)
            return prev_integral + out * dt
        return out

    def _apply_noise(self, u, dt):
        """Add noise to u. Returns u unchanged if noise_std == 0."""
        if not self.noise_std:
            return u
        if self.noise_tau > 0:
            self._noise_buf.copy_(ou_noise_step(self._noise_buf.detach(),
                                                self.noise_std, self.noise_tau, dt))
            return u + self._noise_buf
        return u + self.noise_std * torch.randn_like(u)

    # ── The shared pipeline ────────────────────────────────────────────────────
    # Every DynamicsBase layer runs  bias → noise → output_mode → adaptation →
    # leaky filter → activation → scale. step() calls these three stages and
    # adds only what is specific to the layer (Matsuoka's mutual inhibition,
    # Pulse's plateau, Reichardt's motion detectors, a learning layer's weight
    # update), so every dialog parameter works on every layer.

    def _input(self, u, dt):
        """Input stage: bias → noise → output_mode. Returns the driven input."""
        u = torch.as_tensor(u, dtype=torch.float32)
        if self.bias:
            u = u + self.bias
        u = self._apply_noise(u, dt)
        return self._apply_output_mode(u, dt)

    def _filter(self, u, dt):
        """State stage: adaptation (if tau_a and beta are set) → leaky filter
        (if tau_rise or tau_decay is set). Returns the state x."""
        return self._apply_leaky(self._apply_adaptation_pre(u), dt)

    def _emit(self, x, activate=True):
        """Output stage: activation → scale. Never returns the state buffer
        itself, so the caller can't alias (and later overwrite) the state."""
        out = _activate(x, self.activation, alpha=self.alpha) if activate else x
        if self.scale != 1.0:
            out = out * self.scale
        elif out is x:
            out = out.clone()
        return out

    def _apply_leaky(self, u, dt):
        """Asymmetric leaky integration (leaky_step). Updates _x and returns it;
        returns u unchanged only when both taus are unset (no filtering at all).

        Either tau unset makes that side instantaneous: tau_rise unset + tau_decay
        set gives a fast-attack/slow-decay envelope follower; tau_rise set +
        tau_decay unset snaps straight down to u as soon as it falls.

        Updates _x in place via .copy_() rather than `self._x = ...` — nn.Module's
        __setattr__ silently calls the full register_buffer() machinery on every
        plain reassignment of an already-registered buffer, which is dramatically
        more expensive than an in-place tensor write and dominates per-tick cost
        for any layer calling this on every step().
        """
        if not self.tau_rise and not self.tau_decay:
            return u
        self._x.copy_(leaky_step(self._x.detach(), u, self.tau_rise, self.tau_decay, dt))
        return self._x

    def _apply_adaptation_pre(self, u):
        """Subtract adaptation variable from u. Call before leaky integration."""
        if self.tau_a > 0 and self.beta > 0:
            return u - self.beta * self._a.detach()
        return u

    def _update_adaptation(self, out, dt):
        """Update adaptation variable from output. Call after integration."""
        if self.tau_a > 0 and self.beta > 0:
            a = self._a.detach()
            self._a.copy_(a + (out - a) / self.tau_a * dt)

    @classmethod
    def _dynamics_param_defs(cls):
        """Canonical param_defs entries for universal dynamics parameters.

        Only params shared by ALL DynamicsBase subclasses belong here.
        Adaptation-specific params (tau_a, beta) stay in the per-layer
        param_defs() that use them (only layers with real adaptation state
        expose them). `NetworkDialogs.layer_dialog` (network_viz_dialogs.py) auto-appends
        any of these entries a layer's own param_defs() doesn't already
        declare, so `output_mode`/`x0` become available on every DynamicsBase
        layer for free without each one needing to list it explicitly.
        """
        return [
            ('tau_rise',   float, '0.1',  'leaky rise τ (s; 0 / blank = instant rise)'),
            ('tau_decay',  float, '0.1',  'leaky decay τ (s; 0 / blank = instant decay)'),
            ('x0',         float, '0.0',  'initial value of the internal state (x at t=0)'),
            ('activation', str,   'relu',  'nonlinearity', ACTIVATIONS),
            ('bias',       float, '0.0',  'constant added to input sum'),
            ('scale',      float, '1.0',  'output scale factor'),
            ('output_mode', str,  'none', 'output transform: none / derivative (dOutput/dt) / integral (∫Output dt)', OUTPUT_MODES),
            ('noise_std',  float, '0.0',  'noise amplitude (0 = off)'),
            ('noise_tau',  float, '0.0',  'noise correlation τ (0 = white noise)'),
            ('alpha',      float, '1.0',  'ELU negative-side saturation level (only used when activation=elu)'),
        ]

    def _dyn_code_parts(self, activation_default='relu'):
        """Code-gen kwarg strings for shared optional dynamics attrs (omits defaults)."""
        parts = []
        if self.bias != 0.0:
            parts.append(f'bias={self.bias}')
        if self.activation != activation_default:
            parts.append(f"activation='{self.activation}'")
        if getattr(self, 'noise_std', 0.0):
            parts.append(f'noise_std={self.noise_std}')
        if getattr(self, 'noise_tau', 0.0):
            parts.append(f'noise_tau={self.noise_tau}')
        if getattr(self, 'scale', 1.0) != 1.0:
            parts.append(f'scale={self.scale}')
        if getattr(self, 'output_mode', 'none') != 'none':
            parts.append(f"output_mode='{self.output_mode}'")
        if getattr(self, 'x0', 0.0) != 0.0:
            parts.append(f'x0={self.x0!r}')
        if getattr(self, 'alpha', 1.0) != 1.0:
            parts.append(f'alpha={self.alpha}')
        return parts

    def _ensure_n(self, n):
        """Deferred buffer init — called when n is first inferred from a connection.

        Subclasses with fixed n or non-standard init (RingAttractorLayer, Conv2dLayer,
        PulseLayer, SumLayer, SineLayer, ConstantLayer …) override this.
        """
        if self.n is None:
            self.n = n
            self._init_dynamics_buffers(n)
            self.output = torch.zeros(n)
        elif self.n != n:
            raise ValueError(
                f"{self.__class__.__name__} '{self.name}': declared n={self.n} but connection implies n={n}")

    def reset(self):
        if self.n is None:
            return
        self._reset_dynamics()
        self.output = torch.zeros(self.n)


class LayerBase(nn.Module):
    """Mixin carrying the 7 display / neuromodulation attrs shared by every layer type."""
    _registry: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        LayerBase._registry[cls.__name__] = cls

    def __setattr__(self, name, value):
        # `output` is reassigned every step and is never a parameter, buffer or
        # submodule — skip nn.Module's registration checks for it (they cost
        # more than the layer's arithmetic).
        if name == 'output':
            self.__dict__['output'] = value
        else:
            super().__setattr__(name, value)

    def __init__(self, name='', color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None,
                 lateral_pair=None, **kwargs):
        super().__init__()  # nn.Module; all kwargs consumed upstream
        self.name                       = name
        self.color                      = color
        self.layer                      = layer
        self.modulators                 = modulators or []
        self.neuromodulator_transmitter = neuromodulator_transmitter
        self.neuromodulator_color       = neuromodulator_color
        self.lateral_pair               = lateral_pair  # str name of partner layer, or None
        self.z                          = 0
        self.span                       = 1
        # Per-(name, mode) derivative/integral state for modulator subscriptions.
        # Deliberately NOT stored on/inside `self.modulators` itself: lateralized
        # L/R pair sync copies that list by reference (network_viz_dialogs.py),
        # which would silently share this mutable state across partners.
        self._mod_row_state             = {}

    def on_unmute(self):
        """Called once when a muted layer runs again. Its state was kept while
        muted (only its output was zeroed); layers whose output isn't rebuilt
        by step() restore it here."""

    def is_lateralized(self) -> bool:
        """True when this layer is one half of a lateralized L/R pair."""
        return self.lateral_pair is not None

    def _transform_modulator_value(self, key, mode, value, dt):
        """Modulator response mode (absolute / derivative / integral) for one
        subscription row — see transform_modulator_value."""
        return transform_modulator_value(self._mod_row_state, key, mode, value, dt)

    # ── Capabilities ───────────────────────────────────────────────────────────
    # Declared by each layer class; the runner, serializer and editor ask these
    # instead of checking for concrete classes, so a new layer type only has to
    # set what applies to it. See rules/network_elements.md.

    is_image_node           = False  # output is a 2-D image (thumbnail node; valid image source)
    signed_image            = False  # that image is signed (displayed around mid-grey)
    accepts_image           = False  # input is a flat 2-D image (camera or image layer)
    needs_camera_input      = False  # ...which must come straight from a camera
    kernel_weights          = False  # incoming weights are 4-D conv kernels (n_filters, in_ch, kH, kW)
    passthrough_input       = False  # input arrives through a 1-D ones weight (no weight matrix)
    supports_lateral        = False  # can be split into an L/R pair (lateralized=True)
    is_learning             = False  # learns its incoming weights (TD / delta / three-factor)
    combines_by_product     = False  # incoming connections multiply instead of summing
    has_outgoing_plasticity = False  # plasticity is on its outgoing connections (teacher readout)
    saved_state             = {}     # extra runtime state to save: {json key: attribute name};
                                     # the json keys must be constructor kwargs

    @property
    def n_follows_input(self):
        """True when n is the incoming pixel count (an unpooled image layer)."""
        return False

    @classmethod
    def all_param_defs(cls):
        """param_defs() plus the shared dynamics params (DynamicsBase layers) not
        already listed — everything the edit dialog shows and the serializer saves."""
        defs = list(cls.param_defs()) if hasattr(cls, 'param_defs') else []
        if issubclass(cls, DynamicsBase):
            existing = {p[0] for p in defs}
            defs += [p for p in DynamicsBase._dynamics_param_defs() if p[0] not in existing]
        return defs

    @classmethod
    def lateral_sync_params(cls):
        """Params kept identical across an L/R pair — it is one layer split spatially."""
        return [p[0] for p in cls.all_param_defs() if p[0] != 'lateralized']

    # ── Visualization protocol ─────────────────────────────────────────────────

    def thumbnail_frames(self, disp_h=32):
        """Yield (key, uint8_data) tuples for image thumbnail display. Default: nothing."""
        yield from ()

    @property
    def viz_color(self):
        """Color used for the node in the network visualizer."""
        return self.color

    # ── Code-generation protocol ───────────────────────────────────────────────

    def _base_code_parts(self):
        """kwarg strings for name, color, layer (shared by every layer type)."""
        parts = [f"name='{self.name}'"]
        if getattr(self, 'color', None) is not None:
            parts.append(f"color='{self.color}'")
        if getattr(self, 'layer', None) is not None:
            parts.append(f'layer={self.layer}')
        return parts

    def _mod_code_parts(self):
        """kwarg strings for neuromodulation attrs (omits empties)."""
        parts = []
        if getattr(self, 'modulators', None):
            parts.append(f'modulators={self.modulators!r}')
        if getattr(self, 'neuromodulator_transmitter', None):
            parts.append(f'neuromodulator_transmitter={self.neuromodulator_transmitter!r}')
        if getattr(self, 'neuromodulator_color', None):
            parts.append(f'neuromodulator_color={self.neuromodulator_color!r}')
        return parts

    def internal_edges(self):
        """Return a list of (src_idx, tgt_idx, weight) internal recurrent edges.

        Used by the network visualizer. Default: no internal edges.
        Override only when the layer has meaningful internal recurrent structure
        (e.g. AdaptiveLayer with mutual inhibition).
        """
        return []

    def init_code_parts(self):
        """Return a list of kwarg strings that reconstruct this layer.

        Subclasses override this to produce the full constructor argument list.
        The default fallback just yields name/color/layer + neuromod attrs.
        brain_serializer.generate_layer_code() calls this and joins with ', '.
        """
        return self._base_code_parts() + self._mod_code_parts()


