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

    def _reset_dynamics(self):
        for attr in ('_a', '_noise_buf', '_prev_out'):
            buf = getattr(self, attr, None)
            if buf is not None:
                buf.zero_()
        for attr in ('_x', '_integral'):
            buf = getattr(self, attr, None)
            if buf is not None:
                buf.copy_(self._x0_tensor(buf.numel()).reshape(buf.shape))

    def _apply_output_mode(self, out, dt):
        """Transform *out* per self.output_mode:

        'none'       — pass through unchanged.
        'derivative' — dOut/dt via backward finite difference (zero on the
                       first tick, since there is no previous value yet).
        'integral'   — running ∫Out dt via forward-Euler accumulation.

        Both non-'none' modes keep the stored history buffer detached (a
        constant from the current step's perspective) while letting gradient
        flow through the current *out* — the same single-step-detach
        convention as _apply_leaky/_apply_adaptation_pre.
        """
        mode = getattr(self, 'output_mode', 'none')
        if mode == 'derivative':
            prev = self._prev_out.detach().clone()
            self._prev_out.copy_(out.detach())
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
            nb = self._noise_buf.detach()
            self._noise_buf.copy_(
                nb + (-nb / self.noise_tau + self.noise_std * torch.randn_like(nb)) * dt
            )
            return u + self._noise_buf
        return u + self.noise_std * torch.randn_like(u)

    def _apply_leaky(self, u, dt):
        """Asymmetric leaky integration. Updates _x and returns it; returns u when tau_rise==0.

        tau_decay=None disables the decay branch entirely: x only moves toward u
        while rising, and holds its value when u drops (rise-and-hold integrator).

        Updates _x in place via .copy_() rather than `self._x = ...` — nn.Module's
        __setattr__ silently calls the full register_buffer() machinery on every
        plain reassignment of an already-registered buffer, which is dramatically
        more expensive than an in-place tensor write and dominates per-tick cost
        for any layer calling this on every step().
        """
        if not self.tau_rise:
            return u
        x = self._x.detach()
        rising = u > x
        if self.tau_decay is None:
            delta = torch.where(rising, (u - x) / self.tau_rise * dt,
                                torch.zeros_like(x))
        else:
            tau = torch.where(rising,
                              torch.full_like(x, self.tau_rise),
                              torch.full_like(x, self.tau_decay))
            delta = (u - x) / tau * dt
        self._x.copy_(x + delta)
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
        expose them). `network_viz_dialogs.py`'s `_layer_dialog` auto-appends
        any of these entries a layer's own param_defs() doesn't already
        declare, so `output_mode`/`x0` become available on every DynamicsBase
        layer for free without each one needing to list it explicitly.
        """
        return [
            ('tau_rise',   float, '0.1',  'leaky rise τ (s)'),
            ('tau_decay',  float, '0.1',  'leaky decay τ (s; blank/None = no decay, holds value)'),
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

    def is_lateralized(self) -> bool:
        """True when this layer is one half of a lateralized L/R pair."""
        return self.lateral_pair is not None

    # ── Visualization protocol ─────────────────────────────────────────────────

    is_image_node = False  # overridden to True by image-displaying layers

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


