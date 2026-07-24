import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = ['relu', 'sigmoid', 'tanh', 'linear', 'heaviside']


def _activate(x, name: str):
    """Activation — dispatches on torch.Tensor vs numpy array so sensors.py stays unchanged."""
    if isinstance(x, torch.Tensor):
        if name == 'relu':      return F.relu(x)
        if name == 'sigmoid':   return torch.sigmoid(x)
        if name == 'tanh':      return torch.tanh(x)
        if name == 'linear':    return x
        if name == 'heaviside': return (x > 0).float()
    else:
        x = np.asarray(x, dtype=float)
        if name == 'relu':      return np.maximum(0.0, x)
        if name == 'sigmoid':   return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        if name == 'tanh':      return np.tanh(x)
        if name == 'linear':    return x
        if name == 'heaviside': return (x > 0).astype(float)
    raise ValueError(f"Unknown activation '{name}'. Choose from: {ACTIVATIONS}")


class DynamicsBase:
    """
    Mixin providing shared leaky dynamics, adaptation, and noise parameters and helpers.

    Layer subclasses call _init_dynamics() then _init_dynamics_buffers(n) to set up torch
    buffers for state. Sensor subclasses call _init_dynamics() only; their runtime dynamics
    are handled by BaseSensor._process() using numpy arrays.

    All three state buffers (_x, _a, _noise_buf) are always registered when
    _init_dynamics_buffers is called. Helper methods guard on params before use.
    """

    def _init_dynamics(self, tau_rise=0.1, tau_decay=None, activation='relu',
                       bias=0.0, scale=1.0, tau_a=0.0, beta=0.0,
                       noise_std=0.0, noise_tau=0.0, differential=False):
        self.tau_rise     = float(tau_rise) if tau_rise is not None else None
        self.tau_decay    = float(tau_decay) if tau_decay is not None else self.tau_rise
        self.activation   = activation
        self.bias         = float(bias)
        self.scale        = float(scale)
        self.tau_a        = float(tau_a)
        self.beta         = float(beta)
        self.noise_std    = float(noise_std)
        self.noise_tau    = float(noise_tau)
        self.differential = bool(differential)

    def _init_dynamics_buffers(self, n):
        """Register nn.Module buffers for all dynamics state. Only call from layer classes."""
        self.register_buffer('_x',         torch.zeros(n))
        self.register_buffer('_a',         torch.zeros(n))
        self.register_buffer('_noise_buf', torch.zeros(n))
        self.register_buffer('_prev_out',  torch.zeros(n))

    def _reset_dynamics(self):
        for attr in ('_x', '_a', '_noise_buf', '_prev_out'):
            buf = getattr(self, attr, None)
            if buf is not None:
                buf.zero_()

    def _apply_differential(self, out, dt):
        """Return dV/dt if self.differential, else return out unchanged."""
        if not getattr(self, 'differential', False):
            return out
        prev = self._prev_out.detach()
        self._prev_out = out.detach().clone()
        return (out - prev) / max(float(dt), 1e-9)

    def _apply_noise(self, u, dt):
        """Add noise to u. Returns u unchanged if noise_std == 0."""
        if not self.noise_std:
            return u
        self._noise_buf = self._noise_buf.detach()
        if self.noise_tau > 0:
            self._noise_buf = self._noise_buf + (
                -self._noise_buf / self.noise_tau +
                self.noise_std * torch.randn_like(self._noise_buf)
            ) * dt
            return u + self._noise_buf
        return u + self.noise_std * torch.randn_like(u)

    def _apply_leaky(self, u, dt):
        """Asymmetric leaky integration. Updates _x and returns it; returns u when tau_rise==0."""
        if not self.tau_rise:
            return u
        self._x = self._x.detach()
        tau = torch.where(u > self._x,
                          torch.full_like(self._x, self.tau_rise),
                          torch.full_like(self._x, self.tau_decay))
        self._x = self._x + (u - self._x) / tau * dt
        return self._x

    def _apply_adaptation_pre(self, u):
        """Subtract adaptation variable from u. Call before leaky integration."""
        if self.tau_a > 0 and self.beta > 0:
            self._a = self._a.detach()
            return u - self.beta * self._a
        return u

    def _update_adaptation(self, out, dt):
        """Update adaptation variable from output. Call after integration."""
        if self.tau_a > 0 and self.beta > 0:
            self._a = self._a + (out - self._a) / self.tau_a * dt

    @classmethod
    def _dynamics_param_defs(cls):
        """Canonical param_defs entries for universal dynamics parameters.

        Only params shared by ALL DynamicsBase subclasses belong here.
        Adaptation-specific params (tau_a, beta) and output-mode params
        (differential) stay in the per-layer param_defs() that use them.
        """
        return [
            ('tau_rise',   float, '0.1',  'leaky rise τ (s)'),
            ('tau_decay',  float, '0.1',  'leaky decay τ (s; defaults to tau_rise)'),
            ('activation', str,   'relu',  'nonlinearity', ACTIVATIONS),
            ('bias',       float, '0.0',  'constant added to input sum'),
            ('scale',      float, '1.0',  'output scale factor'),
            ('noise_std',  float, '0.0',  'noise amplitude (0 = off)'),
            ('noise_tau',  float, '0.0',  'noise correlation τ (0 = white noise)'),
        ]


class LayerBase(nn.Module):
    """Mixin carrying the 7 display / neuromodulation attrs shared by every layer type."""
    _registry: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        LayerBase._registry[cls.__name__] = cls

    def __init__(self, name='', color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None,
                 lateral_pair=None):
        super().__init__()
        self.name                       = name
        self.color                      = color
        self.layer                      = layer
        self.modulators                 = modulators or []
        self.neuromodulator_transmitter = neuromodulator_transmitter
        self.neuromodulator_color       = neuromodulator_color
        self.lateral_pair               = lateral_pair  # str name of partner layer, or None

    def is_lateralized(self) -> bool:
        """True when this layer is one half of a lateralized L/R pair."""
        return self.lateral_pair is not None


class LeakyLayer(DynamicsBase, LayerBase):
    """
    N first-order low-pass filter neurons (leaky integrators).

    Each neuron tracks its input with exponential smoothing:

        dx/dt = (u - x) / tau

    where tau switches between tau_rise (input increasing) and tau_decay
    (input decreasing), allowing asymmetric filtering — e.g. fast rise /
    slow decay for a memory trace, or slow rise / fast decay for a
    transient detector.

    derivative mode
        When derivative=True the output is relu(x - u) instead of relu(x).
        Because x lags behind u, x > u only when u has recently fallen below
        its own filtered history — i.e. the neuron fires on a *decrease* of
        its input, not an increase. This is a disappointment / novelty signal.
        tau_decay controls how long the memory of the previous level persists.

    noise
        noise_std > 0 adds zero-mean Gaussian noise to the input each step.
        noise_tau > 0 replaces white noise with an Ornstein-Uhlenbeck process
        of that correlation time, producing slow, correlated fluctuations.

    scale
        Multiplies the final output. Use scale=-1 with derivative=True to
        detect increases instead of decreases (inverts the sign before relu,
        which then clips the negative part).

    Parameters
    ----------
    tau_rise  : float   Rise time constant (s). Default 0.1.
    tau_decay : float   Decay time constant (s). Defaults to tau_rise.
    bias      : float   Constant added to input before filtering.
    activation: str     Output nonlinearity: relu / sigmoid / tanh / linear.
    derivative: bool    If True, output tracks the *fall* of the input signal.
    noise_std : float   Noise amplitude (std dev).
    noise_tau : float   OU correlation time; 0 = white noise each step.
    scale     : float   Output multiplier applied after activation.
    n         : int     Number of neurons (inferred from connections if None).
    """

    help_text = """\
## LeakyLayer — low-pass filter neurons

**Parameters:**
- `n` — number of neurons
- `tau_rise` (τ_rise) — rise time constant (s); default 0.1
- `tau_decay` (τ_decay) — decay time constant (s); defaults to `tau_rise`
- `bias` (b) — constant added to each input sum; default 0.0
- `activation` (f) — nonlinearity: `relu`, `sigmoid`, `tanh`, `linear`; default `relu`
- `scale` (s) — output multiplier; default 1.0
- `derivative` — if True, outputs x − u instead of x (detects input drops); default False
- `noise_std` (σ) — Gaussian noise std added to u each step; default 0.0
- `noise_tau` — Ornstein-Uhlenbeck time constant (0 = white noise); default 0.0

**Dynamics** (u = Σ inputs + b):

$$\\frac{dx}{dt} = \\frac{u - x}{\\tau}, \\quad \\tau = \\begin{cases} \\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x \\end{cases}$$

$$\\text{output} = f(x) \\times s$$

- `tau_rise = tau_decay` — symmetric smoothing.
- `tau_rise < tau_decay` — fast rise, slow decay (memory trace).
- `tau_rise > tau_decay` — slow rise, fast decay (transient detector).

---

**Derivative mode** (`derivative=True`):

$$\\text{output} = f(x - u) \\times s$$

x lags u, so x − u > 0 only when u has recently *fallen* — fires on input **decrease**.
Set `scale = -1` to detect *increases* instead.

---

**Noise:** `noise_std > 0` → Gaussian on u each step.
`noise_tau > 0` → Ornstein-Uhlenbeck correlated fluctuations.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published to the bus each tick and can modulate any layer that lists it in `modulators`.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site)` triples:
  - `site="pre"`: multiplies the input sum by `1 + scale × signal` before integration.
  - `site="post"`: multiplies the output by `1 + scale × signal` after integration.
  - `site="none"`: declares the neuromodulator for learning/visualization only — no signal amplification.
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
  - Feedback paths use the previous tick's value (one-step delay).
"""

    def __init__(self, tau_rise=0.1, tau_decay=None,
                 bias=0.0, activation='relu', derivative=False,
                 noise_std=0.0, noise_tau=0.0, scale=1.0,
                 name='leaky', n=None, color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        LayerBase.__init__(self, name=name, color=color, layer=layer,
                           modulators=modulators,
                           neuromodulator_transmitter=neuromodulator_transmitter,
                           neuromodulator_color=neuromodulator_color)
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay, activation=activation,
                            bias=bias, scale=scale, noise_std=noise_std, noise_tau=noise_tau)
        self.derivative = bool(derivative)
        self.n = n
        if n is not None:
            self._init_dynamics_buffers(n)
        else:
            self.register_buffer('_x',         None)
            self.register_buffer('_a',         None)
            self.register_buffer('_noise_buf', None)
        self.output = torch.zeros(n) if n is not None else None

    @classmethod
    def param_defs(cls):
        return [
            ('tau_rise',   float, '0.1',   'rise τ'),
            ('tau_decay',  float, '0.1',   'decay τ'),
            ('bias',       float, '0.0',   'constant added to input each step'),
            ('activation', str,   'relu',  'output nonlinearity', ACTIVATIONS),
            ('n',          int,   '2',     'number of neurons'),
            ('derivative', bool,  False,   'fires when input falls below baseline'),
            ('noise_std',  float, '0.0',   'noise amplitude'),
            ('noise_tau',  float, '0.0',   'noise correlation time (0 = white noise)'),
            ('scale',      float, '1.0',   'output multiplier'),
        ]

    def _ensure_n(self, n):
        if self.n is None:
            self.n = n
            self._init_dynamics_buffers(n)
            self.output = torch.zeros(n)
        elif self.n != n:
            raise ValueError(f"LeakyLayer '{self.name}': declared n={self.n} but connection implies n={n}")

    def reset(self):
        if self.n is None:
            return
        self._reset_dynamics()
        self.output = torch.zeros(self.n)

    def step(self, input_vec, dt):
        u = torch.as_tensor(input_vec, dtype=torch.float32) + self.bias
        u = self._apply_noise(u, dt)
        x = self._apply_leaky(u, dt)
        if self.derivative:
            out = _activate(x - u, self.activation) * self.scale
        else:
            out = _activate(x, self.activation) * self.scale
        self.output = self._apply_differential(out, dt).detach()
        return self.output

    def internal_edges(self):
        return []


class AdaptiveLayer(DynamicsBase, LayerBase):
    """
    Leaky integrator with spike-frequency adaptation and optional half-centre
    oscillation via mutual inhibition.

    Membrane dynamics
        dx/dt = (u_eff - x) / tau,   output = activation(x) * scale
        u_eff = u + bias + noise - beta*a - w*output_other

    Adaptation
        A slow variable a tracks the neuron's own output:
            da/dt = (output - a) / tau_a
        It feeds back negatively (beta * a), suppressing sustained firing
        (burst-then-adapt). Larger beta or smaller tau_a → faster adaptation.

    Oscillation (w > 0, n = 2)
        Each neuron inhibits the other by w * its own output. Combined with
        adaptation this produces half-centre (Matsuoka) oscillation. The
        period is approximately 2 * tau_a; tune tau_a to set the frequency.
        Minimum drive and inhibition strength (w) required for oscillation:
        empirically w ≳ 1 with beta ≈ 2–3 and sufficient tonic drive.

    Noise
        noise_std > 0 adds noise to the effective input each step.
        noise_tau > 0 gives an Ornstein-Uhlenbeck process, which randomises
        the oscillation period cycle-by-cycle without disrupting the mean.

    Parameters
    ----------
    tau_rise  : float   Membrane rise time constant (s).
    tau_decay : float   Membrane decay time constant (s). Defaults to tau_rise.
    tau_a     : float   Adaptation time constant (s). Sets oscillation period ≈ 2*tau_a.
    beta      : float   Adaptation strength. Higher → shorter burst, faster oscillation.
    w         : float   Mutual inhibition weight (n=2 only). 0 = no oscillation.
    bias      : float   Tonic drive added to input each step.
    activation: str     Output nonlinearity: relu / sigmoid / tanh / linear.
    noise_std : float   Noise amplitude.
    noise_tau : float   OU correlation time; 0 = white noise each step.
    scale     : float   Output multiplier applied after activation.
    n         : int     Number of neurons (inferred from connections if None).
    """

    help_text = """\
## AdaptiveLayer — leaky integrator with spike-frequency adaptation

**Parameters:**
- `n` — number of neurons; default 2
- `tau_rise` (τ) — membrane rise time constant (s); default 0.1
- `tau_decay` — membrane decay time constant (s); defaults to `tau_rise`
- `tau_a` (τ_a) — adaptation time constant (s); default 0.5
- `beta` (β) — adaptation strength; default 1.0
- `w` — mutual inhibition weight (CPG mode, n=2); default 0.0
- `bias` (b) — constant added to each input sum; default 0.0
- `activation` (f) — nonlinearity: `relu`, `sigmoid`, `tanh`, `linear`; default `relu`
- `scale` — output multiplier; default 1.0
- `noise_std` / `noise_tau` — same as LeakyLayer

**Dynamics** (u = Σ inputs + b + noise):

$$u_{eff} = u - \\beta a - w \\cdot o_{other}$$

$$\\frac{dx}{dt} = \\frac{u_{eff} - x}{\\tau}, \\quad \\frac{da}{dt} = \\frac{f(x) - a}{\\tau_a}$$

$$\\text{output} = f(x) \\times \\text{scale}$$

**Adaptation** (`beta > 0`): a tracks output and inhibits it.
Larger β → shorter burst. Smaller τ_a → faster adaptation.

---

**CPG oscillation** (`n=2`, `w > 0`): neurons mutually inhibit.

$$\\text{Period} \\approx 2\\,\\tau_a$$

Requirements: `w >= 1`, `beta ~ 2–3`, `bias > 0` (tonic drive).

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published to the bus each tick and can modulate any layer that lists it in `modulators`.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site)` triples:
  - `site="pre"`: multiplies the input sum by `1 + scale × signal` before integration.
  - `site="post"`: multiplies the output by `1 + scale × signal` after integration.
  - `site="none"`: declares the neuromodulator for learning/visualization only — no signal amplification.
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
  - Feedback paths use the previous tick's value (one-step delay).
"""

    def __init__(self, tau_rise=0.1, tau_decay=None, tau_a=0.5, beta=1.0,
                 w=0.0, bias=0.0, activation='relu',
                 noise_std=0.0, noise_tau=0.0, scale=1.0,
                 name='adaptive', n=None, color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        LayerBase.__init__(self, name=name, color=color, layer=layer,
                           modulators=modulators,
                           neuromodulator_transmitter=neuromodulator_transmitter,
                           neuromodulator_color=neuromodulator_color)
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay, activation=activation,
                            bias=bias, scale=scale, tau_a=tau_a, beta=beta,
                            noise_std=noise_std, noise_tau=noise_tau)
        self.w = float(w)
        self.n = n
        if n is not None:
            self._init_dynamics_buffers(n)
        else:
            self.register_buffer('_x',         None)
            self.register_buffer('_a',         None)
            self.register_buffer('_noise_buf', None)
        self.output = torch.zeros(n) if n is not None else None

    @classmethod
    def param_defs(cls):
        return [
            ('tau_rise',  float, '0.1',  'rise τ (membrane)'),
            ('tau_decay', float, '0.1',  'decay τ (membrane)'),
            ('tau_a',     float, '0.5',  'adaptation time constant'),
            ('beta',      float, '1.0',  'adaptation strength'),
            ('w',         float, '0.0',  'mutual inhibition (n=2 only)'),
            ('bias',      float, '0.0',  'constant added to input each step'),
            ('activation',str,   'relu', 'output nonlinearity', ACTIVATIONS),
            ('noise_std', float, '0.0',  'noise amplitude'),
            ('noise_tau', float, '0.0',  'noise correlation time (0 = white noise)'),
            ('scale',     float, '1.0',  'output multiplier'),
            ('n',         int,   '2',    'number of neurons'),
        ]

    def _ensure_n(self, n):
        if self.n is None:
            self.n = n
            self._init_dynamics_buffers(n)
            self.output = torch.zeros(n)
        elif self.n != n:
            raise ValueError(f"AdaptiveLayer '{self.name}': declared n={self.n} but connection implies n={n}")

    def reset(self):
        if self.n is None:
            return
        self._reset_dynamics()
        self.output = torch.zeros(self.n)

    def step(self, input_vec, dt):
        u = torch.as_tensor(input_vec, dtype=torch.float32) + self.bias
        u = self._apply_noise(u, dt)
        if self.w != 0.0 and self.n == 2:
            prev = torch.as_tensor(self.output, dtype=torch.float32)
            u = u - self.w * prev[[1, 0]]
        u = self._apply_adaptation_pre(u)
        x = self._apply_leaky(u, dt)
        out = _activate(x, self.activation) * self.scale
        self._update_adaptation(out, dt)
        self.output = self._apply_differential(out, dt).detach()
        return self.output

    def internal_edges(self):
        if self.w != 0.0 and self.n == 2:
            return [(0, 1, -self.w), (1, 0, -self.w)]
        return []


class MatsuokaLayer(AdaptiveLayer):
    """
    Deprecated — use AdaptiveLayer with w > 0 and n = 2 instead.

    Thin wrapper around AdaptiveLayer kept for loading old JSON networks that
    contain "type": "MatsuokaLayer". Exposes the original tauM / tauA parameter
    names and hard-codes n=2 with a small asymmetric initial state to seed the
    oscillation.
    """

    help_text = """\
## MatsuokaLayer — half-centre oscillator *(deprecated — use AdaptiveLayer)*

Thin wrapper around **AdaptiveLayer** with `n=2` fixed. Use `AdaptiveLayer` for new networks.

**Parameters:**
- `tau_rise` (τ_M) — membrane time constant (s); default 0.3
- `tau_a` (τ_A) — adaptation time constant (s); default 1.2
- `beta` (β) — adaptation strength; default 2.5
- `w` — mutual inhibition weight; default 2.5
- `bias` (b) — tonic drive; default 0.0

**Dynamics (each neuron):**

$$\\tau_M \\frac{dx}{dt} = -x + u - \\beta a - w \\cdot o_{other}$$

$$\\tau_A \\frac{da}{dt} = -a + \\text{relu}(x), \\quad \\text{output} = \\text{relu}(x)$$

$$\\text{Period} \\approx 2\\,\\tau_A$$

**Oscillation conditions:**
1. `tau_a > tau_rise` — adaptation slower than membrane (required)
2. `w > 1` — mutual inhibition strong enough
3. `beta > 0` — adaptation must engage
4. `bias > 0` — tonic drive needed

If neurons lock (both fire / both silent): increase `w` or `bias`.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published to the bus each tick and can modulate any layer that lists it in `modulators`.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site)` triples:
  - `site="pre"`: multiplies the input sum by `1 + scale × signal` before integration.
  - `site="post"`: multiplies the output by `1 + scale × signal` after integration.
  - `site="none"`: declares the neuromodulator for learning/visualization only — no signal amplification.
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
  - Feedback paths use the previous tick's value (one-step delay).
"""

    def __init__(self, tauM=None, tauA=None, tau_rise=0.3, tau_decay=None, tau_a=1.2,
                 beta=2.5, w=2.5, bias=0.0, name='matsuoka', **kwargs):
        if tauM is not None:   # backward compat — old JSON / brain files use tauM
            tau_rise = tauM
        if tauA is not None:
            tau_a = tauA
        super().__init__(tau_rise=tau_rise,
                         tau_decay=tau_decay if tau_decay is not None else tau_rise,
                         tau_a=tau_a, beta=beta, w=w, bias=bias, n=2, name=name, **kwargs)
        self._x[0] = 0.1  # asymmetry to seed oscillation

    @classmethod
    def param_defs(cls):
        overrides = {
            'tau_rise':  ('tau_rise',  float, '0.3', 'membrane rise τ (s)'),
            'tau_decay': ('tau_decay', float, '',    'membrane decay τ (s; empty = same as tau_rise)'),
            'tau_a':     ('tau_a',     float, '1.2', 'adaptation time constant'),
            'beta':      ('beta',      float, '2.5', 'adaptation suppression gain'),
            'w':         ('w',         float, '2.5', 'mutual inhibition weight'),
            'bias':      ('bias',      float, '0.0', 'tonic drive added each step'),
        }
        return [overrides.get(p[0], p) for p in AdaptiveLayer.param_defs() if p[0] != 'n']

    @property
    def tauM(self):
        return self.tau_rise

    @tauM.setter
    def tauM(self, v):
        self.tau_rise = v

    @property
    def tauA(self):
        return self.tau_a

    @tauA.setter
    def tauA(self, v):
        self.tau_a = v

    def _ensure_n(self, n):
        if n != 2:
            raise ValueError(f"MatsuokaLayer '{self.name}': n is always 2, got n={n} from connection")

    def reset(self):
        super().reset()
        self._x[0] = 0.1


class ConstantLayer(LayerBase):
    """
    Outputs a fixed constant vector every step, ignoring any input.

    Used as a tonic (always-on) drive source. Connect it to other layers to
    provide baseline excitation — e.g. a ConstantLayer → motor connection sets
    the robot's cruising speed; a ConstantLayer → AdaptiveLayer drives the
    oscillator independently of sensory input.

    The value can be a scalar (broadcast to all n neurons) or a list of length n
    for per-neuron values.

    Parameters
    ----------
    value : float or list   Constant output. Scalar is broadcast to all neurons.
    n     : int             Number of neurons (inferred from value length if omitted).
    """

    help_text = """\
## ConstantLayer — fixed tonic drive

Outputs a fixed value every step, ignoring incoming connections.

**Parameters:**
- `n` — number of neurons; default 2
- `value` (v) — constant output (scalar or list of length n); default 1.0
- `noise_std` (σ) — Gaussian noise std added each step; default 0.0

**Output:**

$$\\text{output}_i = v + \\varepsilon_i, \\quad \\varepsilon_i \\sim \\mathcal{N}(0,\\,\\sigma)$$

`noise_std = 0` — pure constant. `value` can be a per-neuron list.

**Typical use:** tonic drive for downstream layers.
Keep `value` small (0.1–1.0) relative to downstream activation thresholds.
For ring attractors: use a **one-to-one** connection so each ring neuron gets
exactly `value`, not `value × n`.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published each tick.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site)` triples. Only `site="post"` has effect (multiplies output by `1 + scale × signal`); `site="pre"` does nothing because this layer ignores incoming connections.
"""

    def __init__(self, value=1.0, n=None, noise_std=0.0, noise=None,
                 name='const', color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        super().__init__(name=name, color=color, layer=layer,
                         modulators=modulators,
                         neuromodulator_transmitter=neuromodulator_transmitter,
                         neuromodulator_color=neuromodulator_color)
        self.noise_std = float(noise if noise is not None else noise_std)
        self._value = np.asarray(value, dtype=float).ravel()
        if n is not None:
            self.n      = n
            self.output = self._make_output(n)
        elif self._value.size > 1:
            self.n      = self._value.size
            self.output = self._value.copy()
        else:
            self.n      = None
            self.output = None

    @classmethod
    def param_defs(cls):
        return [
            ('value',     float, '1.0', 'constant output value'),
            ('n',         int,   '2',   'number of neurons'),
            ('noise_std', float, '0.0', 'std dev of Gaussian noise added each step'),
        ]

    @property
    def value(self):
        return float(self._value[0]) if self._value.size == 1 else self._value.tolist()

    @value.setter
    def value(self, v):
        self._value = np.asarray(v, dtype=float).ravel()
        if self.n is not None:
            self.output = self._make_output(self.n)

    def _make_output(self, n):
        if self._value.size == 1:
            return np.full(n, float(self._value[0]))
        return np.asarray(self._value[:n], dtype=float)

    def _ensure_n(self, n):
        if self.n is None:
            self.n      = n
            self.output = self._make_output(n)
        elif self.n != n:
            raise ValueError(
                f"ConstantLayer '{self.name}': declared n={self.n} but connection implies n={n}")

    def reset(self):
        if self.n is not None:
            self.output = self._make_output(self.n)

    def step(self, _input_vec, _dt):
        if self.noise_std > 0.0:
            self.output = self._make_output(self.n) + np.random.normal(0.0, self.noise_std, self.output.shape)
        return self.output

    def internal_edges(self):
        return []


class SumLayer(LayerBase):
    """
    Instantaneous linear combinator — no dynamics, no memory.

    Outputs the weighted sum of all incoming connections in the same step,
    optionally passed through a nonlinearity. Because there is no time constant,
    the output tracks its inputs without lag.

    Useful as an intermediate mixing/summing node before a recurrent layer,
    or as a direct motor output in sim-only networks (use MotorLayer instead
    when robot actuation is required).

    Contrast with LeakyLayer (filtered, with optional derivative mode) and
    AdaptiveLayer (filtered + adaptation + oscillation).

    Parameters
    ----------
    activation : str   Output nonlinearity: relu / sigmoid / tanh / linear.
    n          : int   Number of neurons (inferred from connections if None).
    """

    help_text = """\
## SumLayer — instantaneous linear combinator

No dynamics, no memory. Output tracks input in the **same step**.

**Parameters:**
- `n` — number of neurons; default 2
- `activation` (f) — nonlinearity: `relu`, `sigmoid`, `tanh`, `linear`; default `relu`
- `scale` (s) — output multiplier; default 1.0

**Output:**

$$\\text{output} = f\\!\\left(\\sum_k W_k \\cdot \\text{input}_k\\right) \\times s$$

- `activation='linear'` — pass-through (standard for motor output layer).
- `activation='relu'` — clips negative sums to zero.
- `scale = -1` — invert output without changing the weight matrix.

Also useful as a mixing / summing node before a recurrent layer.
Use **MotorLayer** (a SumLayer subclass) when you need robot actuation.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published to the bus each tick and can modulate any layer that lists it in `modulators`.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site)` triples:
  - `site="pre"`: multiplies the input sum by `1 + scale × signal` before integration.
  - `site="post"`: multiplies the output by `1 + scale × signal` after integration.
  - `site="none"`: declares the neuromodulator for learning/visualization only — no signal amplification.
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
  - Feedback paths use the previous tick's value (one-step delay).
"""

    def __init__(self, activation='relu', scale=1.0, name='sum', n=None, color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        super().__init__(name=name, color=color, layer=layer,
                         modulators=modulators,
                         neuromodulator_transmitter=neuromodulator_transmitter,
                         neuromodulator_color=neuromodulator_color)
        self.activation = activation
        self.scale      = scale
        self.n          = n
        self.output     = torch.zeros(n) if n is not None else None

    @classmethod
    def param_defs(cls):
        return [
            ('activation', str,   'relu', 'output nonlinearity', ACTIVATIONS),
            ('scale',      float, '1.0',  'output multiplier applied after activation'),
            ('n',          int,   '2',    'number of neurons'),
        ]

    def _ensure_n(self, n):
        if self.n is None:
            self.n      = n
            self.output = torch.zeros(n)
        elif self.n != n:
            raise ValueError(f"SumLayer '{self.name}': input size mismatch, expected {self.n} got {n}")

    def reset(self):
        if self.n is not None:
            self.output = torch.zeros(self.n)

    def step(self, input_vec, _dt):
        inp = torch.as_tensor(input_vec, dtype=torch.float32)
        out = _activate(inp, self.activation) * self.scale
        self.output = out.detach()
        return self.output

    def internal_edges(self):
        return []


class MotorLayer(SumLayer):
    """
    Motor output layer — SumLayer computation + physical actuation.

    In sim mode: drives wheel velocity or joint angle exactly like a SumLayer.
    In robot mode: the output is sent as an OSC message to *robot_address*.

    Use this instead of SumLayer whenever a layer directly commands a motor
    (wheels, joints, or any other actuator on real hardware).

    Parameters
    ----------
    activation   : str   Output nonlinearity (default 'linear' for motors).
    n            : int   Number of motor outputs (default 2 for left/right wheels).
    scale        : float Output multiplier.
    robot_address: str   Full target address: ip:port/osc_path
                         e.g. 192.168.0.1:2390/wheels
                         Leave empty to suppress sending in robot mode.
    """

    help_text = """\
## MotorLayer — motor output with robot actuation

SumLayer computation (instantaneous weighted sum) with a **robot_address** that
routes output to physical hardware in real-robot mode.

**Parameters:**
- `n` — number of motor outputs; default 2 (left wheel, right wheel)
- `activation` — nonlinearity; default `linear` (pass-through)
- `scale` — output multiplier; default 1.0
- `robot_address` — full OSC target: `ip:port/osc_path`
  e.g. `192.168.0.1:2390/wheels`
  Leave empty to run in sim-only mode.

**Output:**

$$\\text{output} = f\\!\\left(\\sum_k W_k \\cdot \\text{input}_k\\right) \\times s$$

In **sim mode** the output drives wheel velocity or joint angle via the circuit,
identical to a SumLayer.  In **robot mode** the output values are packed into an
OSC message and sent to `robot_address` each tick.

**Note:** manual-control override also writes through this layer so the circuit
sees what the wheels are actually doing.
"""

    def __init__(self, activation='linear', scale=1.0, name='motor', n=None,
                 color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None,
                 robot_address=''):
        super().__init__(activation=activation, scale=scale, name=name, n=n,
                         color=color, layer=layer,
                         modulators=modulators,
                         neuromodulator_transmitter=neuromodulator_transmitter,
                         neuromodulator_color=neuromodulator_color)
        self.robot_address = robot_address

    @classmethod
    def param_defs(cls):
        return [
            ('activation',    str,   'linear', 'output nonlinearity', ACTIVATIONS),
            ('n',             int,   '2',      'number of motor outputs'),
            ('scale',         float, '1.0',    'output multiplier'),
            ('robot_address', str,   '',
             'ip:port/osc_path for robot mode (e.g. 192.168.0.1:2390/wheels)'),
        ]


class PulseLayer(LayerBase):
    """
    N plateau-potential neurons with sustained activation and inhibitory reset.

    Two coupled state variables model a fast membrane + slow calcium-like plateau:

        _u : fast activation — asymmetric leaky integrator tracking net input
                 du/dt = (input - u) / tau   (tau_rise when input > u, else tau_decay)

        _s : slow sustained variable — CAN-channel / calcium-buffer analogue
                 ds/dt = (relu(u - theta) - s) / tau_hold

    Output
        output = activation(u + w_s * s) * scale

    Plateau behaviour
        While input drives _u above theta, _s charges slowly (time constant tau_hold).
        When input then drops, _u decays quickly but _s stays elevated, keeping output
        active for approximately tau_hold seconds — a plateau potential.

    Silencing
        Strong negative input drives _u below zero.  Because the activation is relu,
        output collapses even while _s is still high.  The ``drain`` parameter
        additionally erodes _s proportionally to the negative input, permanently
        collapsing the plateau so it does not resume when inhibition ends:

            ds -= drain * relu(-input) * dt   (clamped at 0)

        Set drain=0 for a plateau that resumes after transient inhibition; set
        drain>0 for a plateau that is permanently silenced by sustained inhibition.

    Parameters
    ----------
    tau_rise  : float   Fast rise time constant (s). Default 0.05.
    tau_decay : float   Fast decay time constant (s). Defaults to tau_rise.
    tau_hold  : float   Plateau duration — time constant of _s decay (s). Default 2.0.
    theta     : float   Threshold above which _s charges. 0 = hold at any positive input.
    w_s       : float   Gain of sustained variable on output. Default 1.0.
    drain     : float   Rate at which negative input erodes _s (s⁻¹). Default 1.0.
    bias      : float   Constant added to input before filtering.
    activation: str     Output nonlinearity: relu / sigmoid / tanh / linear.
    scale     : float   Output multiplier applied after activation.
    n         : int     Number of neurons (inferred from connections if None).
    """

    help_text = """\
## PulseLayer — plateau-potential neurons

Models calcium-like sustained (working-memory) activity.

**Parameters:**
- `n` — number of neurons; default 2
- `tau_rise` (τ_rise) — fast membrane rise time constant (s); default 0.05
- `tau_decay` (τ_decay) — fast membrane decay time constant (s); defaults to `tau_rise`
- `tau_hold` (τ_hold) — plateau charging/draining time constant (s); default 2.0
- `theta` (θ) — threshold for charging plateau; default 0.0
- `w_s` — gain of plateau variable on output; default 1.0
- `drain` — rate at which sustained inhibition erodes plateau (0 = pure latch); default 1.0
- `bias` (b) — constant added to input; default 0.0
- `activation` (f) — output nonlinearity; default `relu`
- `scale` — output multiplier; default 1.0

**Fast membrane** (u_in = Σ inputs + b):

$$\\frac{du}{dt} = \\frac{u_{in} - u}{\\tau}, \\quad \\tau = \\begin{cases}\\tau_{rise} & u_{in}>u \\\\ \\tau_{decay} & u_{in}\\leq u\\end{cases}$$

**Slow plateau variable** (charges while u > θ):

$$\\frac{ds}{dt} = \\frac{\\text{relu}(u - \\theta) - s}{\\tau_{hold}}$$

If `drain > 0` and input < 0: plateau is also eroded by inhibition.

**Output:**

$$\\text{output} = f(u + w_s \\cdot s) \\times \\text{scale}$$

While u > θ, s charges slowly. When input drops, u decays but s holds
the plateau for ≈ `tau_hold` seconds.

- `drain = 0` — pure latch, resumes after transient inhibition.
- `drain > 0` — permanently collapsed by sustained inhibition.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published to the bus each tick and can modulate any layer that lists it in `modulators`.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site)` triples:
  - `site="pre"`: multiplies the input sum by `1 + scale × signal` before integration.
  - `site="post"`: multiplies the output by `1 + scale × signal` after integration.
  - `site="none"`: declares the neuromodulator for learning/visualization only — no signal amplification.
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
  - Feedback paths use the previous tick's value (one-step delay).
"""

    def __init__(self, tau_rise=0.05, tau_decay=None, tau_hold=2.0,
                 theta=0.0, w_s=1.0, drain=1.0,
                 bias=0.0, activation='relu', scale=1.0,
                 name='pulse', n=None, color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        super().__init__(name=name, color=color, layer=layer,
                         modulators=modulators,
                         neuromodulator_transmitter=neuromodulator_transmitter,
                         neuromodulator_color=neuromodulator_color)
        self.tau_rise   = float(tau_rise)
        self.tau_decay  = float(tau_decay) if tau_decay is not None else self.tau_rise
        self.tau_hold   = float(tau_hold)
        self.theta      = float(theta)
        self.w_s        = float(w_s)
        self.drain      = float(drain)
        self.bias       = float(bias)
        self.activation = activation
        self.scale      = float(scale)
        self.n          = n
        self.register_buffer('_u', torch.zeros(n) if n is not None else None)
        self.register_buffer('_s', torch.zeros(n) if n is not None else None)
        self.output = torch.zeros(n) if n is not None else None

    @classmethod
    def param_defs(cls):
        return [
            ('tau_rise',  float, '0.05', 'fast rise τ (membrane)'),
            ('tau_decay', float, '0.05', 'fast decay τ (membrane)'),
            ('tau_hold',  float, '2.0',  'plateau duration τ (sustained variable)'),
            ('theta',     float, '0.0',  'threshold for charging plateau (0 = any positive input)'),
            ('w_s',       float, '1.0',  'gain of sustained variable on output'),
            ('drain',     float, '1.0',  'rate at which negative input erodes plateau'),
            ('bias',      float, '0.0',  'constant added to input each step'),
            ('activation',str,   'relu', 'output nonlinearity', ACTIVATIONS),
            ('scale',     float, '1.0',  'output multiplier'),
            ('n',         int,   '2',    'number of neurons'),
        ]

    def _ensure_n(self, n):
        if self.n is None:
            self.n  = n
            self.register_buffer('_u', torch.zeros(n))
            self.register_buffer('_s', torch.zeros(n))
            self.output = torch.zeros(n)
        elif self.n != n:
            raise ValueError(
                f"PulseLayer '{self.name}': declared n={self.n} but connection implies n={n}")

    def reset(self):
        if self.n is None:
            return
        self._u.zero_()
        self._s.zero_()
        self.output = torch.zeros(self.n)

    def step(self, input_vec, dt):
        self._u = self._u.detach()
        self._s = self._s.detach()
        u = torch.as_tensor(input_vec, dtype=torch.float32) + self.bias

        tau = torch.where(u > self._u,
                          torch.full_like(self._u, self.tau_rise),
                          torch.full_like(self._u, self.tau_decay))
        self._u = self._u + (u - self._u) / tau * dt
        self._s = self._s + (F.relu(self._u - self.theta) - self._s) / self.tau_hold * dt

        if self.drain > 0.0:
            self._s = self._s - self.drain * F.relu(-u) * dt
            self._s = torch.clamp(self._s, min=0.0)

        out = _activate(self._u + self.w_s * self._s, self.activation) * self.scale
        self.output = out.detach()
        return self.output

    def internal_edges(self):
        return []


class SineLayer(LayerBase):
    """
    Outputs amplitude * sin(2π * frequency * t + phase) to all n neurons.
    Time is tracked internally and resets on reset(). Ignores incoming connections.
    """

    help_text = """\
## SineLayer — autonomous sinusoidal oscillator

Ignores incoming connections. All `n` neurons share the same value.
`t` resets to 0 on `reset()`.

**Parameters:**
- `n` — number of neurons; default 1
- `amplitude` (A) — peak amplitude; default 1.0
- `frequency` (f) — oscillation frequency in Hz; default 1.0
- `phase` (φ) — initial phase offset in radians; default 0.0

**Output:**

$$\\text{output} = A \\sin(2\\pi f\\, t + \\phi)$$

**Typical use:** clock signal, rhythmic drive, CPG rhythm source.
Connect to a `LeakyLayer` to smooth the waveform, or directly to a motor
layer for open-loop sinusoidal motion.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published each tick.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site)` triples. Only `site="post"` has effect (multiplies output by `1 + scale × signal`); `site="pre"` does nothing because this layer ignores incoming connections.
"""

    def __init__(self, amplitude=1.0, frequency=1.0, phase=0.0,
                 name='sine', n=1, color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        super().__init__(name=name, color=color, layer=layer,
                         modulators=modulators,
                         neuromodulator_transmitter=neuromodulator_transmitter,
                         neuromodulator_color=neuromodulator_color)
        self.amplitude  = amplitude
        self.frequency  = frequency
        self.phase      = phase
        self.n          = n
        self.derivative = False
        self._t         = 0.0
        self.output     = np.full(n, amplitude * np.sin(phase)) if n else None

    @classmethod
    def param_defs(cls):
        return [
            ('amplitude', float, '1.0', 'peak amplitude'),
            ('frequency', float, '1.0', 'oscillation frequency in Hz'),
            ('phase',     float, '0.0', 'initial phase offset in radians'),
            ('n',         int,   '1',   'number of neurons'),
        ]

    def reset(self):
        self._t = 0.0
        if self.n:
            self.output = np.full(self.n, self.amplitude * np.sin(self.phase))

    def _ensure_n(self, n):
        if self.n is None:
            self.n = n
            self.output = np.full(n, self.amplitude * np.sin(self.phase))
        elif self.n != n:
            raise ValueError(f"SineLayer '{self.name}': declared n={self.n} but connection implies n={n}")

    def step(self, _input_vec, dt):
        self._t += dt
        val = self.amplitude * np.sin(2 * np.pi * self.frequency * self._t + self.phase)
        self.output = np.full(self.n or 1, val)
        return self.output

    def internal_edges(self):
        return []


class RingAttractorLayer(DynamicsBase, LayerBase):
    """
    N leaky-integrator neurons arranged in a ring.

    Recurrent connectivity lives in circuit.connections as a regular self-connection
    (same layer as both source and target), editable like any other connection.

    Dynamics:
        tau · dx/dt = −x + u      (u = all incoming connections, including self)
        output = activation(x)

    Parameters
    ----------
    n         : int    Number of neurons.
    tau       : float  Membrane time constant (s).
    activation: str    Output nonlinearity (relu recommended).
    """
    viz_layout = 'ring'

    help_text = """\
## RingAttractorLayer — N neurons on a ring

Recurrent connectivity defined by a **self-connection** (use the Mexican hat preset).

**Parameters:**
- `n` — number of neurons on the ring; default 8
- `tau_rise` (τ_rise) — rise time constant (s); default 0.1
- `tau_decay` (τ_decay) — decay time constant (s); defaults to `tau_rise`
- `activation` (f) — nonlinearity: `relu`, `sigmoid`, `tanh`, `linear`; default `relu`
- `bias` (b) — constant tonic drive per neuron (replaces a ConstantLayer); default 0.0
- `noise_std` / `noise_tau` — same as LeakyLayer

**Dynamics** (u = Σ all incoming connections including self-connection):

$$\\frac{dx}{dt} = \\frac{-x + u}{\\tau}, \\quad \\tau = \\begin{cases}\\tau_{rise} & u>x \\\\ \\tau_{decay} & u\\leq x\\end{cases}$$

$$\\text{output} = f(x)$$

`tau_rise = tau_decay` — symmetric (typical for ring attractors).

---

**Requirements for bump formation:**

1. Self-connection kernel: *Mexican hat*. All row sums must be **negative**.
2. Tonic drive: `bias > 0` or a one-to-one `ConstantLayer` (0.1–1.0 per neuron).
3. Excitatory width σ_exc ≥ 1/n. For `n=8`: σ_exc ≥ 0.15.
4. Two bumps: σ_exc too narrow — widen until neighbours are excitatory.

Quick check: if all neurons stay at the same nonzero value, drive is too strong
or the Mexican hat row sums are not sufficiently negative.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published to the bus each tick and can modulate any layer that lists it in `modulators`.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site)` triples:
  - `site="pre"`: multiplies the input sum by `1 + scale × signal` before integration.
  - `site="post"`: multiplies the output by `1 + scale × signal` after integration.
  - `site="none"`: declares the neuromodulator for learning/visualization only — no signal amplification.
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
  - Feedback paths use the previous tick's value (one-step delay).
"""

    def __init__(self, n=8, tau_rise=0.1, tau_decay=None, activation='relu', bias=0.0,
                 noise_std=0.0, noise_tau=0.0,
                 name='ring', color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None,
                 # Legacy params accepted for migration — not used at runtime
                 tau=None, w_exc=None, sigma_exc=None, w_inh=None, sigma_inh=None):
        LayerBase.__init__(self, name=name, color=color, layer=layer,
                           modulators=modulators,
                           neuromodulator_transmitter=neuromodulator_transmitter,
                           neuromodulator_color=neuromodulator_color)
        if tau is not None:
            tau_rise = tau
        self.n = n
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay, activation=activation,
                            bias=bias, noise_std=noise_std, noise_tau=noise_tau)
        self._init_dynamics_buffers(n)
        self.output = torch.zeros(n)
        if w_exc is not None:
            self._legacy_W = self._build_legacy_kernel(
                n,
                w_exc     if w_exc     is not None else 3.0,
                sigma_exc if sigma_exc is not None else 0.6,
                w_inh     if w_inh     is not None else 1.5,
                sigma_inh if sigma_inh is not None else 1.5,
            )
        else:
            self._legacy_W = None

    @staticmethod
    def _build_legacy_kernel(n, w_exc, sigma_exc, w_inh, sigma_inh):
        """Difference-of-Gaussians circulant kernel (used for migration only)."""
        thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
        dists  = np.minimum(thetas, 2 * np.pi - thetas)
        row0   = (w_exc * np.exp(-dists**2 / (2 * sigma_exc**2))
                - w_inh * np.exp(-dists**2 / (2 * sigma_inh**2)))
        row0[0] = 0.0
        W = np.empty((n, n))
        for i in range(n):
            W[i] = np.roll(row0, i)
        return W

    @staticmethod
    def default_kernel(n):
        """Default Mexican-hat kernel used when auto-creating the self-connection."""
        return RingAttractorLayer._build_legacy_kernel(n, 3.0, 0.6, 1.5, 1.5)

    @classmethod
    def param_defs(cls):
        return [
            ('n',          int,   '8',    'number of neurons'),
            ('tau_rise',   float, '0.1',  'rise time constant (s)'),
            ('tau_decay',  float, '0.1',  'decay time constant (s) — defaults to tau_rise'),
            ('activation', str,   'relu', 'output nonlinearity', ACTIVATIONS),
            ('bias',       float, '0.0',  'constant added to input each step — replaces a ConstantLayer drive'),
            ('noise_std',  float, '0.0',  'noise amplitude'),
            ('noise_tau',  float, '0.0',  'noise correlation time (0 = white noise)'),
        ]

    @property
    def tau(self):
        return self.tau_rise

    @tau.setter
    def tau(self, v):
        self.tau_rise  = float(v)
        self.tau_decay = float(v)

    def _ensure_n(self, n):
        if n != self.n:
            raise ValueError(
                f"RingAttractorLayer '{self.name}': n is fixed at {self.n}, connection implies n={n}")

    def reset(self):
        self._reset_dynamics()
        self.output = torch.zeros(self.n)

    def step(self, input_vec, dt):
        u = torch.as_tensor(input_vec, dtype=torch.float32) + self.bias
        u = self._apply_noise(u, dt)
        x = self._apply_leaky(u, dt)
        out = _activate(x, self.activation)
        self.output = self._apply_differential(out, dt).detach()
        return self.output

    def internal_edges(self):
        return []


class Conv2dLayer(DynamicsBase, LayerBase):
    """
    2-D convolution layer for camera image input (H × W, 1 or 3 channels).

    The connection weight is a 4-D array (n_filters, in_ch, kH, kW).
    in_ch is 1 (grayscale) or 3 (RGB), determined by the connected camera's mode.
    Filters are defined in the Filter Stack dialog — n_filters equals the number
    of filters added to the stack.

    With pool='global_avg' or 'global_max' the output is (n_filters,) scalars,
    one per filter — n is known once the first connection is made.
    With pool='none' the full spatial feature map is output as a flat vector.

    Parameters
    ----------
    kernel_size : int   Square kernel size (kH = kW). Default 3.
    stride      : int   Convolution stride. Default 1.
    padding     : str   'same' — zero-pad to preserve H×W. 'valid' — no padding.
    pool        : str   'global_avg', 'global_max', or 'none'.
    activation  : str   Nonlinearity applied to feature maps BEFORE pooling (relu/sigmoid/tanh/linear).
    tau_rise    : float Rise τ for optional leaky dynamics on pooled output (0 = instantaneous).
    tau_decay   : float Decay τ. Defaults to tau_rise.
    bias        : float Constant added to pooled output.
    """

    help_text = """\
## Conv2dLayer — 2-D convolution over camera input

Connect a `CameraSensor` to `Conv2dLayer`, then open the **Filter Stack** dialog.
Each filter is a `(in_ch, kH, kW)` kernel; `in_ch` is inferred from camera mode.

**Parameters:**
- `n_filters` — number of filters (= output neurons with global pool); default 1
- `kernel_size` — square kernel side kH = kW; default 3
- `stride` — convolution stride; default 1
- `padding` — `same` (preserve H×W) or `valid` (no padding); default `same`
- `pool` — `global_avg`, `global_max`, or `none`; default `global_avg`
- `activation` (f) — nonlinearity applied to feature maps before pooling; default `relu`
- `tau_rise` (τ_rise) — leaky dynamics rise τ on pooled output (0 = off); default 0.0
- `tau_decay` (τ_decay) — leaky dynamics decay τ; defaults to `tau_rise`
- `tau_a` (τ_a) — adaptation time constant (0 = off); default 0.0
- `beta` (β) — adaptation strength; 0 = no adaptation; default 0.0
- `bias` (b) — constant added to each pooled output; default 0.0
- `scale` — output multiplier; default 1.0
- `lateralized` — create mirrored _L / _R pair for split-camera input; default False

**Forward pass** (I: in_ch × H × W, W: n_filters × in_ch × kH × kW):

$$M = f(\\text{conv2d}(I,\\, W)), \\quad \\text{pooled} = \\text{pool}(M) + b$$

**Optional leaky dynamics** (when `tau_rise > 0`):

$$\\frac{dx}{dt} = \\frac{\\text{pooled} - x}{\\tau}, \\quad \\text{output} = x \\times \\text{scale}$$

If `tau_rise = 0`: output = pooled × scale directly.

**Optional adaptation** (when `beta > 0` and `tau_a > 0`):

A slow variable *a* tracks the layer's own output and subtracts from the effective input:

$$u_{eff} = \\text{pooled} - \\beta \\, a, \\quad \\frac{da}{dt} = \\frac{\\text{output} - a}{\\tau_a}$$

Larger β → stronger suppression of sustained responses (burst-then-adapt).
Smaller τ_a → faster adaptation, more transient responses.

- `pool='global_avg'` / `'global_max'` — output shape: `(n_filters,)`
- `pool='none'` — output shape: `(n_filters, H_out, W_out)`
- Use `padding='valid'` with zero-sum kernels to avoid edge artifacts.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published to the bus each tick and can modulate any layer that lists it in `modulators`.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site)` triples:
  - `site="pre"`: multiplies the input sum by `1 + scale × signal` before integration.
  - `site="post"`: multiplies the output by `1 + scale × signal` after integration.
  - `site="none"`: declares the neuromodulator for learning/visualization only — no signal amplification.
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
  - Feedback paths use the previous tick's value (one-step delay).
"""

    def __init__(self, n_filters=1, kernel_size=3, stride=1, padding='same',
                 pool='global_avg', activation='relu',
                 tau_rise=0.0, tau_decay=None, tau_a=0.0, beta=0.0,
                 bias=0.0, scale=1.0, lateralized=False,
                 noise_std=0.0, noise_tau=0.0, differential=False,
                 name='conv2d', n=None, color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        LayerBase.__init__(self, name=name, color=color, layer=layer,
                           modulators=modulators,
                           neuromodulator_transmitter=neuromodulator_transmitter,
                           neuromodulator_color=neuromodulator_color)
        self.kernel_size = int(kernel_size)
        self.stride      = max(1, int(stride))
        self.padding     = padding
        self.pool        = pool
        self.n_filters   = max(1, int(n_filters))
        self.lateralized = bool(lateralized)
        if self.pool == 'none' and self.n_filters > 1:
            raise ValueError(
                f"Conv2dLayer '{name}': pool='none' requires n_filters=1 "
                f"(got n_filters={self.n_filters}). Set n_filters=1 to use spatial output mode."
            )
        if self.pool == 'none':
            self.viz_n       = 1
            self._last_frame = None
            self.frame_h     = None
            self.frame_w     = None
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay, activation=activation,
                            bias=bias, scale=scale, tau_a=tau_a, beta=beta,
                            noise_std=noise_std, noise_tau=noise_tau, differential=differential)
        n      = n if n is not None else self.n_filters
        self.n = n
        self._init_dynamics_buffers(n)
        self.output = torch.zeros(n)

    @classmethod
    def param_defs(cls):
        return [
            ('n_filters',   int,   '1',          'number of filters (= output neurons with global pool)'),
            ('kernel_size', int,   '3',          'square kernel size (kH = kW)'),
            ('stride',      int,   '1',           'convolution stride'),
            ('padding',     str,   'same',        'same: preserve H×W   valid: no padding',
             ['same', 'valid']),
            ('pool',        str,   'global_avg',  'how to collapse spatial dims to a scalar per filter',
             ['global_avg', 'global_max', 'none']),
            ('activation',  str,   'relu',        'nonlinearity applied to feature maps before pooling',
             ACTIVATIONS),
            ('tau_rise',    float, '0.0',         'leaky rise τ on pooled output (0 = instantaneous)'),
            ('tau_decay',   float, '0.0',         'leaky decay τ (defaults to tau_rise)'),
            ('tau_a',       float, '0.0',         'adaptation time constant (0 = off)'),
            ('beta',        float, '0.0',         'adaptation strength (0 = off)'),
            ('bias',        float, '0.0',         'constant added to pooled output'),
            ('scale',       float, '1.0',         'output scale factor applied after dynamics'),
            ('lateralized', bool,  False,         'create mirrored _L / _R pair for split-camera input'),
        ]

    def _ensure_n(self, n):
        if self.n != n:
            self.n = n
            # n_filters is set by the weight tensor shape; for lateralized camera n = 2 * n_filters.
            self._init_dynamics_buffers(n)
            self.output = torch.zeros(n)

    def reset(self):
        if self.n is None:
            return
        self._reset_dynamics()
        self.output = torch.zeros(self.n)

    def step(self, input_vec, dt):
        # input_vec has activation applied in _conv_forward before pooling — no activation here.
        u = torch.as_tensor(input_vec, dtype=torch.float32) + self.bias
        u = self._apply_adaptation_pre(u)
        out = self._apply_leaky(u, dt)
        # _a tracks pre-scale output so beta operates in the same units as the input.
        self._update_adaptation(out, dt)
        out = out * self.scale
        self.output = self._apply_differential(out, dt).detach()
        if self.pool == 'none':
            self._update_last_frame(self.output.numpy())
        return self.output

    def _update_last_frame(self, arr):
        """Reshape flat spatial output → (H, W) for display (pool='none' only)."""
        H = getattr(self, 'frame_h', None)
        W = getattr(self, 'frame_w', None)
        if H is None or W is None:
            n = arr.shape[0]
            sq = int(n ** 0.5)
            H = sq
            W = max(1, n // max(sq, 1))
        try:
            self._last_frame = arr.reshape(H, W)
        except ValueError:
            pass

    def internal_edges(self):
        return []


class Leaky2dLayer(DynamicsBase, LayerBase):
    """
    Pixel-wise leaky integrator that preserves the full spatial image structure.

    Applies first-order temporal low-pass filtering independently to each pixel
    of a camera sensor's output (or another Leaky2dLayer's output).  Unlike
    Conv2dLayer — which pools spatial responses down to a feature vector —
    Leaky2dLayer keeps the image dimensions intact so the result can be fed
    directly into a Conv2dLayer or inspected as a filtered image.

    Typical uses
    ────────────
    • Retinal temporal adaptation: slow low-pass filter per pixel.
    • Motion / optic-flow detection: set derivative=True → each pixel fires
      when its value has recently *increased* above its running mean.
      (derivative=True computes activation(x − u) where x lags u, so x > u
      only when u has recently *fallen* — use scale=-1 to detect increases.)

    Connection weight
    ─────────────────
    The connection from a CameraSensor to a Leaky2dLayer uses a 1-D ones
    vector (shape n_pixels) stored as an element-wise passthrough.  No matrix
    multiply is performed; the weight just gates each pixel individually.

    Frame metadata
    ──────────────
    After each step() the layer updates _last_frame (H×W or H×W×3 numpy array)
    so a downstream Conv2dLayer can infer the correct spatial dimensions.
    frame_h / frame_w are set automatically when a camera is connected and are
    also serialized so the shape survives save / load.
    """

    help_text = """\
## Leaky2dLayer — pixel-wise temporal filter (image → image)

Applies the same first-order leaky integration as `LeakyLayer` but independently
to **each pixel** of the input image, producing an output image of the same spatial size.
Downstream `Conv2dLayer` nodes can use this as their source.

**Parameters:**
- `tau_rise` (τ_rise) — rise time constant (s); default 0.1
- `tau_decay` (τ_decay) — decay τ (s); defaults to `tau_rise`
- `bias` (b) — constant added to each pixel before integration; default 0.0
- `activation` (f) — nonlinearity per pixel: `linear`, `relu`, `sigmoid`, `tanh`; default `linear`
- `scale` (s) — output multiplier; default 1.0
- `derivative` — if True, outputs x − u instead of x (fires on pixel *decrease*); default False
- `noise_std` / `noise_tau` — per-pixel noise (same as `LeakyLayer`)

**Dynamics** (u = pixel value + b):

$$\\frac{dx}{dt} = \\frac{u - x}{\\tau}, \\quad \\tau = \\begin{cases} \\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x \\end{cases}$$

$$\\text{output pixel} = f(x) \\times s$$

**Derivative / motion mode** (`derivative=True`):

$$\\text{output pixel} = f(x - u) \\times s$$

x lags u, so x − u > 0 only when u has recently *fallen*.
Use `scale = -1` to detect *increases* (brighter = active).

**Optic flow recipe:**
1. Connect `GrayCameraSensor` → `Leaky2dLayer(tau_rise=0.2, derivative=True, scale=-1, activation='relu')`
2. Connect `Leaky2dLayer` → `Conv2dLayer` to extract spatial motion features.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits.
- `modulators` — list of `(name, scale, site)` triples (pre / post / none).
"""

    viz_n = 1  # show as a single image node in the network visualizer (like a camera)

    def __init__(self, tau_rise=0.1, tau_decay=None, activation='linear',
                 bias=0.0, scale=1.0, derivative=False,
                 noise_std=0.0, noise_tau=0.0,
                 in_ch=1, frame_h=None, frame_w=None,
                 lateralized=False,
                 name='leaky2d', n=None, color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        LayerBase.__init__(self, name=name, color=color, layer=layer,
                           modulators=modulators,
                           neuromodulator_transmitter=neuromodulator_transmitter,
                           neuromodulator_color=neuromodulator_color)
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay, activation=activation,
                            bias=bias, scale=scale, noise_std=noise_std, noise_tau=noise_tau)
        self.lateralized = bool(lateralized)
        self.derivative  = bool(derivative)
        self.in_ch      = int(in_ch)
        self.frame_h    = int(frame_h) if frame_h is not None else None
        self.frame_w    = int(frame_w) if frame_w is not None else None
        self._last_frame = None
        n = n if n is not None else 1  # placeholder so the node is visible before connecting
        self.n = n
        self._init_dynamics_buffers(n)
        self.output = torch.zeros(n)

    @classmethod
    def param_defs(cls):
        return [
            ('lateralized', bool,  False,    'create mirrored _L/_R pair for split-camera input'),
            ('tau_rise',   float, '0.1',    'rise τ (s)'),
            ('tau_decay',  float, '0.1',    'decay τ (s; defaults to tau_rise)'),
            ('activation', str,   'linear', 'per-pixel nonlinearity', ACTIVATIONS),
            ('derivative', bool,  False,    'motion mode: output = activation(x − u) per pixel'),
            ('bias',       float, '0.0',    'constant added to each pixel input'),
            ('scale',      float, '1.0',    'output multiplier'),
            ('noise_std',  float, '0.0',    'per-pixel noise amplitude (0 = off)'),
            ('noise_tau',  float, '0.0',    'noise correlation τ (0 = white noise)'),
        ]

    def _ensure_n(self, n):
        if self.n != n:
            self.n = n
            self._init_dynamics_buffers(n)
            self.output = torch.zeros(n)

    def reset(self):
        self._reset_dynamics()
        self.output = torch.zeros(self.n)

    def step(self, input_vec, dt):
        u = torch.as_tensor(input_vec, dtype=torch.float32) + self.bias
        u = self._apply_noise(u, dt)
        x = self._apply_leaky(u, dt)
        if self.derivative:
            out = _activate(x - u, self.activation) * self.scale
        else:
            out = _activate(x, self.activation) * self.scale
        self.output = out.detach()
        self._update_last_frame(self.output.numpy())
        return self.output

    def _update_last_frame(self, arr):
        """Reshape flat output → (H, W) or (H, W, C) for downstream Conv2d."""
        n = arr.shape[0]
        n_pixels = n // max(self.in_ch, 1)
        H = self.frame_h if self.frame_h else int(n_pixels ** 0.5)
        W = self.frame_w if self.frame_w else max(1, n_pixels // max(H, 1))
        # Guard against non-exact integer divisions (e.g. in_ch wrong after reload).
        if H * W != n_pixels:
            W = n_pixels  # fall back to a 1×n_pixels strip
            H = 1
        try:
            if self.in_ch > 1:
                self._last_frame = arr.reshape(self.in_ch, H, W).transpose(1, 2, 0)
            else:
                self._last_frame = arr.reshape(H, W)
        except ValueError:
            pass  # dimensions still inconsistent; keep previous frame

    def internal_edges(self):
        return []


class LearningLayerBase(DynamicsBase, LayerBase):
    """
    Base class for all reward-driven learning layers.

    Forward pass: V = Σ_conn W @ s  (linear weighted sum over incoming connections)
    Weight update: ΔW = α_eff · δ · s_prev  where α_eff is alpha_pos (δ≥0) or alpha_neg (δ<0)
    Episodic state: _src_prev, _V_prev — cleared on reset(); connection weights survive.

    Subclasses implement _compute_delta(V, r) → δ tensor of shape (n,).
    Optional leaky dynamics on V output via DynamicsBase (_x buffer, tau_rise/tau_decay).
    """

    def __init__(self, n=1, alpha_pos=0.01, alpha_neg=None,
                 tau_rise=0.0, tau_decay=None, activation='linear',
                 bias=0.0, scale=1.0,
                 noise_std=0.0, noise_tau=0.0, differential=False,
                 reward_modulator='dopamine',
                 w_min=None, w_max=None,
                 competition='none', k=1,
                 weight_decay=0.0,
                 name='learning', color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        LayerBase.__init__(self, name=name, color=color, layer=layer,
                           modulators=modulators,
                           neuromodulator_transmitter=neuromodulator_transmitter,
                           neuromodulator_color=neuromodulator_color)
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay, activation=activation,
                            bias=bias, scale=scale,
                            noise_std=noise_std, noise_tau=noise_tau, differential=differential)
        self.n              = int(n)
        self.alpha_pos      = float(alpha_pos)
        self.alpha_neg      = float(alpha_neg) if alpha_neg is not None else self.alpha_pos
        self.reward_modulator = reward_modulator
        self.w_min          = float(w_min) if w_min not in (None, '', 'none') else None
        self.w_max          = float(w_max) if w_max not in (None, '', 'none') else None
        self.competition    = competition
        self.k              = int(k)
        self.weight_decay   = float(weight_decay)
        self._reward        = 0.0
        self._V_prev        = torch.zeros(self.n)
        self._src_prev      = {}
        self.output         = torch.zeros(self.n)
        if tau_rise:
            self._init_dynamics_buffers(self.n)
        else:
            self.register_buffer('_x',         None)
            self.register_buffer('_a',         None)
            self.register_buffer('_noise_buf', None)
            self.register_buffer('_prev_out',  torch.zeros(self.n))

    def _competition_mask(self, V):
        """Return a multiplicative mask applying lateral competition to V."""
        if self.competition == 'none' or self.n == 1:
            return torch.ones_like(V)
        if self.competition == 'wta':
            mask = torch.zeros_like(V)
            mask[torch.topk(V, min(self.k, self.n)).indices] = 1.0
            return mask
        if self.competition == 'softmax':
            return F.softmax(V, dim=0)
        return torch.ones_like(V)

    @classmethod
    def _shared_learning_param_defs(cls):
        return [
            ('weight_decay', float, '0.0',  'passive weight decay rate (per second; 0 = off)'),
            ('w_min',       str,   '',     'min synaptic weight (blank = unbounded)'),
            ('w_max',       str,   '',     'max synaptic weight (blank = unbounded)'),
            ('competition', str,   'none', 'lateral competition: none / softmax / wta',
             ['none', 'softmax', 'wta']),
            ('k',           int,   '1',    'number of winners for wta competition'),
        ]

    def _compute_delta(self, V, r):
        raise NotImplementedError

    def _ensure_n(self, n):
        if n != self.n:
            raise ValueError(
                f"{self.__class__.__name__} '{self.name}': outgoing connection expects n={n} "
                f"but this layer has n={self.n}")

    def reset(self):
        self._V_prev   = torch.zeros(self.n)
        self._src_prev = {}
        self.output    = torch.zeros(self.n)
        self._reset_dynamics()

    def step(self, _inp, _dt):
        self.output = torch.zeros(self.n)
        return self.output

    def step_td(self, src_inputs, dt):
        """Forward pass + weight update. Called by the runner with collected connection data."""
        if not src_inputs:
            self.output = torch.zeros(self.n)
            return self.output

        V = torch.zeros(self.n)
        for src_val, w_cached, conn_idx, conn in src_inputs:
            V = V + w_cached @ src_val

        if self.tau_rise:
            V = self._apply_leaky(V + self.bias, dt)

        r     = self._reward
        delta = self._compute_delta(V, r)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        mask      = self._competition_mask(V)
        alpha_eff = torch.where(delta >= 0,
                                torch.full_like(delta, self.alpha_pos),
                                torch.full_like(delta, self.alpha_neg))
        w_lo = float(self.w_min) if self.w_min not in (None, '', 'none') else -float('inf')
        w_hi = float(self.w_max) if self.w_max not in (None, '', 'none') else  float('inf')
        for src_val, w_cached, conn_idx, conn in src_inputs:
            src_prev = self._src_prev.get(conn_idx, torch.zeros_like(src_val))
            w_cached.add_(torch.outer(alpha_eff * delta * mask, src_prev))
            torch.nan_to_num_(w_cached, nan=0.0, posinf=0.0, neginf=0.0)
            if self.weight_decay > 0:
                w_cached.mul_(1.0 - self.weight_decay * dt)
            torch.clamp_(w_cached, w_lo, w_hi)
            conn.W = w_cached.detach().numpy().copy()

        for src_val, w_cached, conn_idx, conn in src_inputs:
            self._src_prev[conn_idx] = src_val.detach().clone()

        self._V_prev = V.detach()
        out = _activate(V, self.activation) * self.scale * mask
        self.output  = self._apply_differential(out, dt).detach()
        return self.output

    def internal_edges(self):
        return []


class TDLayer(LearningLayerBase):
    help_text = """\
## TDLayer — TD(0) reward-prediction critic

Implements the Schultz/Dayan/Montague (1997) dopamine model: a linear critic
whose **connection weights** W are updated each tick by the temporal-difference
error δ. There are no separate internal weights — the connection matrix IS the
learned weight: W[j, i] is neuron j's weight on input i.

**Parameters:**
- `n` — number of output neurons (parallel critics); default 1
- `alpha_pos` — learning rate for positive δ (acquisition); default 0.01
- `alpha_neg` — learning rate for negative δ (extinction); default = alpha_pos
- `gamma` (γ) — discount factor (0–1); default 0.99
- `reward_modulator` — neuromodulator name carrying the reward signal r; default "dopamine"

**Output:** V(s) ∈ ℝⁿ — predicted future reward per neuron.
Can be wired to motors: higher V → stronger approach drive.

**Learning rule (per tick, per connection i):**

$$V = \\sum_i W_i \\, s_i, \\quad \\delta = r + \\gamma V - V_{\\text{prev}}, \\quad \\Delta W_i = \\alpha_{\\text{eff}} \\, \\delta \\otimes s_{i,\\text{prev}}$$

**Wiring:**
1. Connect any sensory/feature layer → TDLayer. Initialize the connection W to zeros.
2. Declare the reward-carrying layer as a neuromodulator transmitter (e.g. "dopamine").
3. Set `reward_modulator` to that name. The layer reads r from it each tick.
4. Optionally wire TDLayer output → motor layers for direct actor behaviour.

**Why 1-step TD is sufficient in the ecological setting:**
The reward patch and its sensory cue are spatially co-located. When the robot is
inside the patch it is also seeing the cue — s and r are temporally aligned by
the world. There is no credit-assignment delay to bridge.

**Reset behaviour:** ↺ Reset clears V_prev and src_prev (episodic state) but leaves
the connection weights intact — learning survives across episodes.
Use **↺ Reset Weights** in Brain Parameters to zero all incoming connection weights.

**When to use:** Episodic or sequential tasks where value must propagate backward
through a chain of states (temporal credit assignment). For ecological closed-loop
settings with spatially co-located cue and reward, prefer **ThreeFactorLayer**.

**References:**
- Sutton, R. S. & Barto, A. G. (1988). Learning by temporal differences. In
  *Proceedings of the 1988 Connectionist Models Summer School*. Morgan Kaufmann.
- Schultz, W., Dayan, P. & Montague, P. R. (1997). A neural substrate of prediction
  and reward. *Science*, 275(5306), 1593–1599.
"""

    def __init__(self, alpha_pos=0.01, alpha_neg=None,
                 gamma=0.99, reward_modulator='dopamine', n=1,
                 tau_rise=0.0, tau_decay=None, activation='linear', bias=0.0, scale=1.0,
                 w_min=None, w_max=None, competition='none', k=1, weight_decay=0.0,
                 name='td', color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None,
                 alpha=None):  # alpha kept as ignored kwarg for JSON backward compat
        _pos = float(alpha_pos)
        _neg = float(alpha_neg) if alpha_neg is not None else _pos
        super().__init__(n=n, alpha_pos=_pos, alpha_neg=_neg,
                         tau_rise=tau_rise, tau_decay=tau_decay,
                         activation=activation, bias=bias, scale=scale,
                         reward_modulator=reward_modulator,
                         w_min=w_min, w_max=w_max, competition=competition, k=k,
                         weight_decay=weight_decay,
                         name=name, color=color, layer=layer,
                         modulators=modulators,
                         neuromodulator_transmitter=neuromodulator_transmitter,
                         neuromodulator_color=neuromodulator_color)
        self.gamma = float(gamma)

    def _compute_delta(self, V, r):
        return r + self.gamma * V - self._V_prev

    @classmethod
    def param_defs(cls):
        return [
            ('n',                int,   '1',        'number of output neurons (parallel critics)'),
            ('alpha_pos',        float, '0.01',     'learning rate for δ ≥ 0 (acquisition)'),
            ('alpha_neg',        float, '0.01',     'learning rate for δ < 0 (extinction)'),
            ('gamma',            float, '0.99',     'discount factor γ (0–1)'),
            ('reward_modulator', str,   'dopamine', 'neuromodulator name carrying reward r'),
            ('tau_rise',         float, '0.0',      'leaky rise τ on V output (0 = off)'),
            ('tau_decay',        float, '0.0',      'leaky decay τ on V output'),
            ('activation',       str,   'linear',   'output nonlinearity',
             ACTIVATIONS),
            ('bias',             float, '0.0',      'constant added to V before output'),
            ('scale',            float, '1.0',      'output scale factor'),
        ] + cls._shared_learning_param_defs()


class DeltaLayer(LearningLayerBase):
    help_text = """\
## DeltaLayer — Rescorla-Wagner / delta-rule critic

δ = r − V with asymmetric learning rates. Biologically matches the asymmetry
between LTP (fast acquisition) and LTD (slow extinction):

- **alpha_pos** (δ ≥ 0): fast — learn the cue-reward association.
- **alpha_neg** (δ < 0): slow — avoid rapid unlearning during approach to the patch.

The slow negative update gives the animal time to reach the reward before the
association erodes. If reward is genuinely absent across many exposures the small
negative updates accumulate and eventually dissociate cue from reward — matching
the behavioural extinction timescale.

**Parameters:**
- `n` — number of output neurons (parallel critics); default 1
- `alpha_pos` — learning rate for δ ≥ 0 (acquisition); default 0.05
- `alpha_neg` — learning rate for δ < 0 (extinction); default 0.005
- `reward_modulator` — neuromodulator name carrying reward r; default "dopamine"

**Output:** V(s) ∈ ℝⁿ — reward prediction per neuron.
Can be wired to motors for direct approach drive.

**Learning rule (per tick, per connection i):**

$$V = \\sum_i W_i \\, s_i, \\quad \\delta = r - V, \\quad \\Delta W_i = \\alpha_{\\text{eff}} \\, \\delta \\otimes s_{i,\\text{prev}}$$

**Reset behaviour:** same as TDLayer — episodic state cleared, weights survive.

**When to use:** Conditioning paradigms where the cue and reward may be separated
in time but extinction should be slow (asymmetric α). For ecological closed-loop
settings where reward gates all learning, prefer **ThreeFactorLayer**.

**References:**
- Rescorla, R. A. & Wagner, A. R. (1972). A theory of Pavlovian conditioning:
  Variations in the effectiveness of reinforcement and non-reinforcement. In
  *Classical Conditioning II: Current Research and Theory*. Appleton-Century-Crofts.
- Widrow, B. & Hoff, M. E. (1960). Adaptive switching circuits. *IRE WESCON
  Convention Record*, 4, 96–104.
"""

    def __init__(self, alpha_pos=0.05, alpha_neg=0.005,
                 reward_modulator='dopamine', n=1,
                 tau_rise=0.0, tau_decay=None, activation='linear', bias=0.0, scale=1.0,
                 w_min=None, w_max=None, competition='none', k=1, weight_decay=0.0,
                 name='delta', color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        super().__init__(n=n, alpha_pos=alpha_pos, alpha_neg=alpha_neg,
                         tau_rise=tau_rise, tau_decay=tau_decay,
                         activation=activation, bias=bias, scale=scale,
                         reward_modulator=reward_modulator,
                         w_min=w_min, w_max=w_max, competition=competition, k=k,
                         weight_decay=weight_decay,
                         name=name, color=color, layer=layer,
                         modulators=modulators,
                         neuromodulator_transmitter=neuromodulator_transmitter,
                         neuromodulator_color=neuromodulator_color)

    def _compute_delta(self, V, r):
        return r - V

    @classmethod
    def param_defs(cls):
        return [
            ('n',                int,   '1',        'number of output neurons (parallel critics)'),
            ('alpha_pos',        float, '0.05',     'learning rate for δ ≥ 0 (acquisition)'),
            ('alpha_neg',        float, '0.005',    'learning rate for δ < 0 (extinction)'),
            ('reward_modulator', str,   'dopamine', 'neuromodulator name carrying reward r'),
            ('tau_rise',         float, '0.0',      'leaky rise τ on V output (0 = off)'),
            ('tau_decay',        float, '0.0',      'leaky decay τ on V output'),
            ('activation',       str,   'linear',   'output nonlinearity',
             ACTIVATIONS),
            ('bias',             float, '0.0',      'constant added to V before output'),
            ('scale',            float, '1.0',      'output scale factor'),
        ] + cls._shared_learning_param_defs()


class ThreeFactorLayer(LearningLayerBase):
    help_text = """\
## ThreeFactorLayer — reward-gated Hebbian learning

Implements the canonical **three-factor Hebbian rule**: weight changes require the
simultaneous coincidence of presynaptic activity, postsynaptic activity, *and* a
neuromodulatory reward signal. When reward is absent the weights are frozen; only
passive decay (if enabled) erodes them.

**Learning rule (per tick, per connection i → neuron j):**

$$\\Delta W_{ji} = \\alpha_{\\text{eff}} \\cdot r \\cdot V_j \\cdot s_{i,\\text{prev}}$$

- `r` — reward signal from the designated neuromodulator (gates all plasticity)
- `V_j` — postsynaptic activity of neuron j (selects *which* neuron learns)
- `s_prev` — presynaptic activity on the previous tick (selects *which* inputs)
- `α_eff` = `alpha_pos` if r·V ≥ 0, else `alpha_neg`

**Passive forgetting (independent of reward):**

$$W \\leftarrow W \\cdot (1 - \\text{decay} \\cdot dt)$$

Applied every tick regardless of r. Equilibrium weight reflects the balance between
acquisition rate and decay rate — infrequently rewarded associations fade naturally.

**Parameters:**
- `n` — number of output neurons; default 1
- `alpha_pos` — learning rate when r·V ≥ 0; default 0.01
- `alpha_neg` — learning rate when r·V < 0 (punishment); default = alpha_pos
- `reward_modulator` — neuromodulator name carrying r; default "dopamine"
- `weight_decay` — passive decay rate (s⁻¹); default 0.0
- `w_min` / `w_max` — synaptic bounds (blank = unbounded)
- `competition` — lateral competition: `none` / `softmax` / `wta`; default `none`
- `k` — number of winners for `wta`; default 1

**Wiring:**
1. Connect any sensory/feature layer → ThreeFactorLayer (initialise W to zeros).
2. Declare the reward-carrying layer as a neuromodulator transmitter (e.g. "dopamine").
3. Set `reward_modulator` to that name.
4. Optionally wire output → motor layers for direct approach drive.

**Reset behaviour:** ↺ Reset clears episodic state but leaves weights intact.
Use **↺ Reset Weights** to zero all incoming connection weights.

**References:**
- Hebb, D. O. (1949). *The Organization of Behavior*. Wiley.
- Montague, P. R., Dayan, P. & Sejnowski, T. J. (1996). A framework for mesencephalic
  dopamine systems based on predictive Hebbian learning. *J. Neuroscience*, 16(5), 1936–1947.
- Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and
  dopamine signaling. *Cerebral Cortex*, 17(10), 2443–2452.
- Frémaux, N. & Gerstner, W. (2016). Neuromodulated spike-timing-dependent plasticity, and
  theory of three-factor learning rules. *Frontiers in Neural Circuits*, 9, 85.
"""

    def __init__(self, alpha_pos=0.01, alpha_neg=None,
                 reward_modulator='dopamine', n=1,
                 tau_rise=0.0, tau_decay=None, activation='linear', bias=0.0, scale=1.0,
                 w_min=None, w_max=None, competition='none', k=1, weight_decay=0.0,
                 name='3factor', color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        super().__init__(n=n, alpha_pos=alpha_pos,
                         alpha_neg=float(alpha_neg) if alpha_neg is not None else alpha_pos,
                         tau_rise=tau_rise, tau_decay=tau_decay,
                         activation=activation, bias=bias, scale=scale,
                         reward_modulator=reward_modulator,
                         w_min=w_min, w_max=w_max, competition=competition, k=k,
                         weight_decay=weight_decay,
                         name=name, color=color, layer=layer,
                         modulators=modulators,
                         neuromodulator_transmitter=neuromodulator_transmitter,
                         neuromodulator_color=neuromodulator_color)

    def _compute_delta(self, V, r):
        return r * V

    @classmethod
    def param_defs(cls):
        return [
            ('n',                int,   '1',        'number of output neurons'),
            ('alpha_pos',        float, '0.01',     'learning rate for r·V ≥ 0'),
            ('alpha_neg',        float, '0.01',     'learning rate for r·V < 0 (punishment)'),
            ('reward_modulator', str,   'dopamine', 'neuromodulator name carrying reward r'),
            ('tau_rise',         float, '0.0',      'leaky rise τ on output (0 = off)'),
            ('tau_decay',        float, '0.0',      'leaky decay τ on output'),
            ('activation',       str,   'linear',   'output nonlinearity',
             ACTIVATIONS),
            ('bias',             float, '0.0',      'constant added to output'),
            ('scale',            float, '1.0',      'output scale factor'),
        ] + cls._shared_learning_param_defs()


# Registry — populated automatically by LayerBase.__init_subclass__ as each class is defined.
LAYER_REGISTRY = LayerBase._registry
