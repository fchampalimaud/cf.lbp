import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neurons_base import ACTIVATIONS, _activate, DynamicsBase, LayerBase

class LeakyLayer(DynamicsBase, LayerBase):
    """
    N first-order low-pass filter neurons (leaky integrators).

    Each neuron tracks its input with exponential smoothing:

        dx/dt = (u - x) / tau

    where tau switches between tau_rise (input increasing) and tau_decay
    (input decreasing), allowing asymmetric filtering — e.g. fast rise /
    slow decay for a memory trace, or slow rise / fast decay for a
    transient detector.

    noise
        noise_std > 0 adds zero-mean Gaussian noise to the input each step.
        noise_tau > 0 replaces white noise with an Ornstein-Uhlenbeck process
        of that correlation time, producing slow, correlated fluctuations.

    output_mode
        Transforms the raw summed input u (post-bias/noise) BEFORE it reaches
        the leaky filter/activation below — not the layer's own output.
        'none' (default): u passes through unchanged.
        'derivative': u becomes du/dt (finite difference of consecutive raw
        inputs, see DynamicsBase._apply_output_mode), which then still runs
        through the ordinary leaky filter + activation pipeline. Fires on any
        change to the input, both rising and falling.
        'integral': u becomes the running ∫u dt.
        This replaced an earlier design that transformed the final *output*
        instead — that meant a derivative/integral kept responding to the
        leaky filter's own decay tail even after the true input returned to
        zero, since it never touched the raw drive at all.

    scale
        Multiplies the final output. Use scale=-1 with output_mode='derivative'
        to detect increases instead of decreases (inverts the sign before relu,
        which then clips the negative part).

    x0
        Initial value of the internal state x (and, independently, of the
        output_mode='integral' accumulator) — both at construction and every
        reset(). Default 0.0. Needed for bounded integrator neurons whose
        resting point isn't the bottom of their range: e.g. a memory neuron
        clipped to [0, 1] that must be able to move both up and down from a
        neutral starting point needs x0=0.5, not 0 — the classic example is
        the CPU4 path-integration memory in Stone et al. 2017's bee-navigation
        model. Note this is NOT the same as `bias`: bias is added to the
        input on every tick (a permanent per-tick offset, which under
        output_mode='integral' would make the accumulator drift forever),
        while x0 only sets the one-time starting point.

    Parameters
    ----------
    tau_rise    : float  Rise time constant (s). Default 0.1.
    tau_decay   : float  Decay time constant (s). None = no decay (holds value).
    bias        : float  Constant added to input before filtering.
    activation  : str    Output nonlinearity: relu / sigmoid / tanh / linear.
    output_mode : str    'none' / 'derivative' / 'integral' — see DynamicsBase.
    x0          : float  Initial value of x (and the integral accumulator). Default 0.0.
    noise_std   : float  Noise amplitude (std dev).
    noise_tau   : float  OU correlation time; 0 = white noise each step.
    scale       : float  Output multiplier applied after activation.
    n           : int    Number of neurons (inferred from connections if None).
    """

    help_text = """\
## LeakyLayer — low-pass filter neurons

**Parameters:**
- `tau_rise` (τ_rise) — rise time constant (s); default 0.1
- `tau_decay` (τ_decay) — decay time constant (s); blank/None = no decay (holds value)
- `bias` (b) — constant added to each input sum; default 0.0
- `activation` (f) — nonlinearity: `relu`, `sigmoid`, `tanh`, `linear`; default `relu`
- `n` — number of neurons
- `output_mode` — `none` / `derivative` / `integral`, see below; default `none`
- `x0` — initial value of x (and the integral accumulator), both at construction and on reset; default 0.0
- `noise_std` (σ) — Gaussian noise std added to u each step; default 0.0
- `noise_tau` — Ornstein-Uhlenbeck time constant (0 = white noise); default 0.0
- `scale` (s) — output multiplier; default 1.0
**Dynamics** (u = Σ inputs + b, transformed by `output_mode` below before filtering; x(0) = x0):

$$\\frac{dx}{dt} = \\frac{u - x}{\\tau}, \\quad \\tau = \\begin{cases} \\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x \\end{cases}$$

$$\\text{output} = f(x) \\times s$$

- `tau_rise = tau_decay` — symmetric smoothing.
- `tau_rise < tau_decay` — fast rise, slow decay (memory trace).
- `tau_rise > tau_decay` — slow rise, fast decay (transient detector).
- `tau_decay = None` (blank) — no decay: x only moves toward u while rising, holds when u drops.

---

**Output mode** (shared by every `DynamicsBase` layer/sensor — see `DynamicsBase._apply_output_mode`) — applied to the raw input **u**, before the leaky filter and activation above run:

- `output_mode='derivative'`: replaces u with du/dt — finite difference between consecutive ticks' raw input; fires on *any* change to the input. The (now-derivative) signal still passes through the leaky filter and activation normally.
- `output_mode='integral'`: replaces u with the running ∫u dt.

---

**Initial value (`x0`)** — seeds both `x` and the `output_mode='integral'` accumulator, at construction *and* on every reset. Needed for a bounded integrator whose resting point isn't the edge of its range (e.g. a `[0, 1]`-clipped memory neuron that must move both up and down needs `x0=0.5`, not 0 — the CPU4 path-integration memory in Stone et al. 2017's bee model is the canonical example). Not the same as `bias`: bias is added to the input *every tick*, so under `output_mode='integral'` a nonzero bias makes the accumulator drift forever instead of just setting where it starts.

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

    def __init__(self, n=None, name='leaky', **kwargs):
        super().__init__(name=name, **kwargs)
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
            ('noise_std',  float, '0.0',   'noise amplitude'),
            ('noise_tau',  float, '0.0',   'noise correlation time (0 = white noise)'),
            ('scale',      float, '1.0',   'output multiplier'),
        ]

    def step(self, input_vec, dt):
        u = torch.as_tensor(input_vec, dtype=torch.float32) + self.bias
        u = self._apply_noise(u, dt)
        u = self._apply_output_mode(u, dt)
        x = self._apply_leaky(u, dt)
        out = _activate(x, self.activation, alpha=self.alpha) * self.scale
        self.output = out.detach()
        return self.output

    def init_code_parts(self):
        parts = [f'tau_rise={self.tau_rise}', f'tau_decay={self.tau_decay}']
        parts += self._base_code_parts()
        if self.n is not None:
            parts.append(f'n={self.n}')
        parts += self._dyn_code_parts()
        parts += self._mod_code_parts()
        return parts


class ProductLayer(LeakyLayer):
    """
    Same leaky-integrator dynamics as LeakyLayer — the only difference is
    upstream: network_runner.step_network combines this layer's incoming
    connections by elementwise PRODUCT instead of sum before the dynamics
    below ever run (see the `_is_product` special case there).

        u = prod_k(W_k . input_k) + bias
        dx/dt = (u - x) / tau,   tau = tau_rise if rising else tau_decay
        output = f(x) * scale

    Useful for coincidence detection / gating: e.g. an output that requires
    two sensory drives to be active simultaneously (an AND-like unit), or a
    signal that scales another rather than adding to it — filtered through
    the same leaky/noise/output_mode pipeline as every other dynamics-based
    layer.

    Each connection is still an ordinary weighted transform (its own weight
    matrix/pattern) — only the fan-in *across* connections is multiplicative.
    A single incoming connection is just a weighted pass-through. With zero
    connections the product is 1 for every neuron (not 0, unlike SumLayer's
    empty sum), before bias/dynamics/activation/scale.

    Not the same as neuromodulation: `modulators` gain-scales the *aggregate*
    input/output of a layer from a dedicated tagged transmitter signal; this
    multiplies two ordinary afferent connections together at the neuron
    itself.

    Bonsai export is not supported: LBP.Torch's JoinAdditive only sums
    fan-in, so `Copy Bonsai` raises an error if a ProductLayer is present.

    Parameters
    ----------
    tau_rise    : float  Rise time constant (s). Default 0.1.
    tau_decay   : float  Decay time constant (s). None = no decay (holds value).
    bias        : float  Constant added to the product before filtering.
    activation  : str    Output nonlinearity: relu / sigmoid / tanh / linear.
    output_mode : str    'none' / 'derivative' / 'integral' — see DynamicsBase.
    noise_std   : float  Noise amplitude (std dev).
    noise_tau   : float  OU correlation time; 0 = white noise each step.
    scale       : float  Output multiplier applied after activation.
    n           : int    Number of neurons (inferred from connections if None).
    """

    help_text = """\
## ProductLayer — multiplicative-fan-in leaky neurons

Same leaky-integrator dynamics as **LeakyLayer** — the only difference is
upstream: incoming connections combine by **elementwise product** instead of
sum before these dynamics run. Use for coincidence detection / gating
(output requires BOTH inputs active), filtered through the same
tau/noise/output_mode pipeline as every other dynamics-based layer.

**Parameters:**
- `tau_rise` (τ_rise) — rise time constant (s); default 0.1
- `tau_decay` (τ_decay) — decay time constant (s); blank/None = no decay (holds value)
- `bias` (b) — constant added to the product each step; default 0.0
- `activation` (f) — nonlinearity: `relu`, `sigmoid`, `tanh`, `linear`; default `relu`
- `n` — number of neurons
- `output_mode` — `none` / `derivative` / `integral`; default `none`
- `noise_std` (σ) — Gaussian noise std added to u each step; default 0.0
- `noise_tau` — Ornstein-Uhlenbeck time constant (0 = white noise); default 0.0
- `scale` (s) — output multiplier; default 1.0
**Dynamics** (u = ∏ₖ Wₖ·inputₖ + b):

$$\\frac{dx}{dt} = \\frac{u - x}{\\tau}, \\quad \\tau = \\begin{cases} \\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x \\end{cases}$$

$$\\text{output} = f(x) \\times s$$

Each connection is still an ordinary weighted transform (its own weight
matrix/pattern); only the **fan-in across connections** is multiplicative
instead of additive. A single incoming connection is just a weighted
pass-through. With zero connections the product is 1 (not 0, as SumLayer's
empty sum would be), before bias/dynamics/activation/scale.

Contrast with **SumLayer**/**LeakyLayer** (additive fan-in). Not the same as
neuromodulation (`modulators` gain-scales the *aggregate* input/output of a
layer from a tagged transmitter signal) — this multiplies two ordinary
afferent connections together at the neuron itself.

**Bonsai export:** not supported — LBP.Torch's `JoinAdditive` only sums
fan-in; there is no multiplicative join node. `Copy Bonsai` raises an error
if a ProductLayer is present in the circuit.

---

**Output mode** (shared by every `DynamicsBase` layer/sensor) — applied to the raw product **u**, before the leaky filter/activation above run:

- `output_mode='derivative'`: replaces u with du/dt — rate of change of the product itself.
- `output_mode='integral'`: replaces u with the running ∫u dt.

---

**Noise:** `noise_std > 0` → Gaussian on u each step.
`noise_tau > 0` → Ornstein-Uhlenbeck correlated fluctuations.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published to the bus each tick and can modulate any layer that lists it in `modulators`.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site)` triples:
  - `site="pre"`: multiplies the input product by `1 + scale × signal` before integration.
  - `site="post"`: multiplies the output by `1 + scale × signal` after integration.
  - `site="none"`: declares the neuromodulator for learning/visualization only — no signal amplification.
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
  - Feedback paths use the previous tick's value (one-step delay).
"""

    def __init__(self, n=None, name='product', **kwargs):
        super().__init__(n=n, name=name, **kwargs)


class AccumulatorLayer(DynamicsBase, LayerBase):
    """
    Pure integrator with optional leaky decay: x accumulates input over time.

    Primary update (per step):
        x += rate * u_c * dt          (u_c = u - mean(u) if zero_center else u)

    Optional decay toward zero (tau_decay is not None):
        x -= x / tau_decay * dt

    This separates accumulation (rate) from forgetting (tau_decay), unlike
    LeakyLayer which conflates both into a single tau that pulls x toward u.

    zero_center=True subtracts mean(u) before accumulating so only the spatial
    bump integrates, not the DC offset.  Matches the hDelta update in Goulard
    et al. (2023):
        hDc = hDc + rate * (PFN - mean(PFN))

    Parameters
    ----------
    rate        : float  Accumulation rate (output/s). Default 0.001.
    zero_center : bool   Subtract mean(u) before accumulating. Default True.
    clip        : float  Symmetric clamp |x| ≤ clip. None = no clamp.
    tau_decay   : float  Decay toward zero time constant (s). None = no decay.
    n           : int    Number of neurons (inferred from connections if None).
    """

    help_text = """\
## AccumulatorLayer — pure integrator with optional decay

Primary update each step:

$$x \\leftarrow x + r \\cdot u_c \\cdot dt$$

where $u_c = u - \\bar{u}$ when `zero_center=True`, or $u_c = u$ otherwise.

Optional decay toward zero (set `tau_decay`):

$$x \\leftarrow x - \\frac{x}{\\tau_{decay}} \\cdot dt$$

**Parameters:**
- `n` — number of neurons (inferred from connections if not set)
- `tau_decay` — decay-toward-zero time constant (s); 0 = no decay; default 0
- `noise_std`, `noise_tau` — optional noise on input (same as other layers)
- `noise_tau` — noise correlation τ (0 = white noise)
- `rate` (r) — accumulation rate (output per second); default 0.001
- `zero_center` — subtract mean(u) before accumulating; removes DC offset; default True
- `clip` — symmetric clamp |x| ≤ clip after each update; 0 = off; default 0
**Use cases:**
- Path integration (hDelta / hDc in the insect central complex)
- Satiation / energy with digestion (`zero_center=False`, set `tau_decay`)
- Slow evidence accumulation
"""

    @classmethod
    def _dynamics_param_defs(cls):
        return [
            ('tau_decay',  float, '0.0', 'decay-toward-zero τ (s; 0 = no decay)'),
            ('noise_std',  float, '0.0', 'noise amplitude (0 = off)'),
            ('noise_tau',  float, '0.0', 'noise correlation τ (0 = white noise)'),
        ]

    def __init__(self, n=None, tau_decay=None, noise_std=0.0, noise_tau=0.0,
                 rate=0.001, zero_center=True, clip=None, name='accumulator', **kwargs):
        # output_mode doesn't apply: this layer already *is* an integrator by
        # design (step() never calls _apply_output_mode), so exclude it the
        # same way as the other unsupported universal dynamics params below.
        for k in ('tau_rise', 'activation', 'bias', 'scale', 'output_mode'):
            kwargs.pop(k, None)
        super().__init__(name=name, tau_rise=0.0, tau_decay=tau_decay,
                         noise_std=noise_std, noise_tau=noise_tau, **kwargs)
        self.rate        = float(rate)
        self.zero_center = bool(zero_center)
        self.clip        = float(clip) if clip is not None else None
        self.n           = n
        if n is not None:
            self._init_dynamics_buffers(n)
        else:
            self.register_buffer('_x',         None)
            self.register_buffer('_noise_buf', None)
        self.output = torch.zeros(n) if n is not None else None

    @classmethod
    def param_defs(cls):
        return [
            ('n',           int,   '16',    'number of neurons'),
            ('tau_decay',   float, '0.0',   'decay-toward-zero τ (s; 0 = no decay)'),
            ('noise_std',   float, '0.0',   'noise amplitude'),
            ('noise_tau',   float, '0.0',   'noise correlation τ (0 = white noise)'),
            ('rate',        float, '0.001', 'accumulation rate per second'),
            ('zero_center', bool,  True,    'subtract mean(u) before accumulating'),
            ('clip',        float, '0.0',   'symmetric value clamp (0 = off)'),
        ]

    def step(self, input_vec, dt):
        u = torch.as_tensor(input_vec, dtype=torch.float32)
        u = self._apply_noise(u, dt)
        if self.zero_center:
            u = u - u.mean()
        x = self._x.detach()
        x = x + self.rate * u * dt
        if self.tau_decay:
            x = x - x / self.tau_decay * dt
        if self.clip is not None and self.clip > 0.0:
            x = torch.clamp(x, -self.clip, self.clip)
        self._x.copy_(x)
        self.output = self._x.detach()
        return self.output

    def init_code_parts(self):
        parts = [f'rate={self.rate}']
        if not self.zero_center:
            parts.append('zero_center=False')
        if self.clip is not None and self.clip > 0.0:
            parts.append(f'clip={self.clip}')
        if self.tau_decay:
            parts.append(f'tau_decay={self.tau_decay}')
        parts += self._base_code_parts()
        if self.n is not None:
            parts.append(f'n={self.n}')
        parts += self._dyn_code_parts()
        parts += self._mod_code_parts()
        return parts


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
    tau_decay : float   Membrane decay time constant (s). None = no decay (holds value).
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
- `tau_rise` (τ) — membrane rise time constant (s); default 0.1
- `tau_decay` — membrane decay time constant (s); blank/None = no decay (holds value)
- `tau_a` (τ_a) — adaptation time constant (s); default 0.5
- `beta` (β) — adaptation strength; default 1.0
- `w` — mutual inhibition weight (CPG mode, n=2); default 0.0
- `bias` (b) — constant added to each input sum; default 0.0
- `activation` (f) — nonlinearity: `relu`, `sigmoid`, `tanh`, `linear`; default `relu`
- `noise_std` / `noise_tau` — same as LeakyLayer
- `noise_tau` — noise correlation time (0 = white noise)
- `scale` — output multiplier; default 1.0
- `n` — number of neurons; default 2
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

**Output mode** — `output_mode` ∈ `{none, derivative, integral}` (not listed above;
auto-added to this dialog by `DynamicsBase`). Transforms the raw input u
*before* adaptation/leaky filtering/activation run: `derivative` replaces u
with its rate of change; `integral` replaces u with its running accumulation.

**Initial value** — `x0` (not listed above; also auto-added by `DynamicsBase`)
sets the starting value of `x` and of the `output_mode='integral'` accumulator,
both at construction and on every reset. Default 0.0. Not the same as `bias`
(added every tick, not just at the start).

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

    def __init__(self, n=None, w=0.0, name='adaptive', **kwargs):
        super().__init__(name=name, **kwargs)
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

    def step(self, input_vec, dt):
        u = torch.as_tensor(input_vec, dtype=torch.float32) + self.bias
        u = self._apply_noise(u, dt)
        u = self._apply_output_mode(u, dt)
        if self.w != 0.0 and self.n == 2:
            prev = torch.as_tensor(self.output, dtype=torch.float32)
            u = u - self.w * prev[[1, 0]]
        u = self._apply_adaptation_pre(u)
        x = self._apply_leaky(u, dt)
        out = _activate(x, self.activation, alpha=self.alpha) * self.scale
        self._update_adaptation(out, dt)
        self.output = out.detach()
        return self.output

    def internal_edges(self):
        if self.w != 0.0 and self.n == 2:
            return [(0, 1, -self.w), (1, 0, -self.w)]
        return []

    def init_code_parts(self):
        parts = [f'tau_rise={self.tau_rise}', f'tau_decay={self.tau_decay}']
        parts += self._base_code_parts()
        parts.append(f'tau_a={self.tau_a}')
        parts.append(f'beta={self.beta}')
        if self.w != 0.0:
            parts.append(f'w={self.w}')
        if self.n is not None:
            parts.append(f'n={self.n}')
        parts += self._dyn_code_parts()
        parts += self._mod_code_parts()
        return parts


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
- `tau_decay` — membrane decay τ (s; empty = same as tau_rise)
- `tau_a` (τ_A) — adaptation time constant (s); default 1.2
- `beta` (β) — adaptation strength; default 2.5
- `w` — mutual inhibition weight; default 2.5
- `bias` (b) — tonic drive; default 0.0
- `activation` — output nonlinearity
- `noise_std` — noise amplitude
- `noise_tau` — noise correlation time (0 = white noise)
- `scale` — output multiplier
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

    def __init__(self, tau_rise=0.3, tau_decay=None, tau_a=1.2, beta=2.5, w=2.5,
                 bias=0.0, tauM=None, tauA=None, name='matsuoka', **kwargs):
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

    def init_code_parts(self):
        parts = [f'tau_rise={self.tau_rise}', f'tau_a={self.tau_a}',
                 f'beta={self.beta}', f'w={self.w}']
        parts += self._base_code_parts()
        if self.bias != 0.0:
            parts.append(f'bias={self.bias}')
        parts += self._mod_code_parts()
        return parts


class PulseLayer(DynamicsBase, LayerBase):
    """
    N plateau-potential neurons with sustained activation and inhibitory reset.

    Two coupled state variables model a fast membrane + slow calcium-like plateau:

        _x : fast activation — asymmetric leaky integrator tracking net input
                 (shared DynamicsBase buffer/mechanism; see _apply_leaky)

        _s : slow sustained variable — CAN-channel / calcium-buffer analogue
                 ds/dt = (relu(x - theta) - s) / tau_hold

    Output
        output = activation(x + w_s * s) * scale

    Plateau behaviour
        While input drives x above theta, _s charges slowly (time constant tau_hold).
        When input then drops, x decays (or holds, if tau_decay=None) but _s stays
        elevated, keeping output active for approximately tau_hold seconds — a
        plateau potential.

    Silencing
        Strong negative input drives x below zero.  Because the activation is relu,
        output collapses even while _s is still high.  The ``drain`` parameter
        additionally erodes _s proportionally to the negative input, permanently
        collapsing the plateau so it does not resume when inhibition ends:

            ds -= drain * relu(-input) * dt   (clamped at 0)

        Set drain=0 for a plateau that resumes after transient inhibition; set
        drain>0 for a plateau that is permanently silenced by sustained inhibition.

    Parameters
    ----------
    tau_rise  : float   Fast rise time constant (s). Default 0.05.
    tau_decay : float   Fast decay time constant (s). None = no decay (holds value).
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
- `tau_rise` (τ_rise) — fast membrane rise time constant (s); default 0.05
- `tau_decay` (τ_decay) — fast membrane decay time constant (s); blank/None = no decay (holds value)
- `tau_hold` (τ_hold) — plateau charging/draining time constant (s); default 2.0
- `theta` (θ) — threshold for charging plateau; default 0.0
- `w_s` — gain of plateau variable on output; default 1.0
- `drain` — rate at which sustained inhibition erodes plateau (0 = pure latch); default 1.0
- `bias` (b) — constant added to input; default 0.0
- `activation` (f) — output nonlinearity; default `relu`
- `scale` — output multiplier; default 1.0
- `n` — number of neurons; default 2
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
                 n=None, name='pulse', **kwargs):
        super().__init__(name=name, tau_rise=tau_rise, tau_decay=tau_decay, **kwargs)
        self.tau_hold = float(tau_hold)
        self.theta    = float(theta)
        self.w_s      = float(w_s)
        self.drain    = float(drain)
        self.n        = n
        if n is not None:
            self._init_dynamics_buffers(n)
            self.register_buffer('_s', torch.zeros(n))
        else:
            self.register_buffer('_x',         None)
            self.register_buffer('_a',         None)
            self.register_buffer('_noise_buf', None)
            self.register_buffer('_s',         None)
        self.output = torch.zeros(n) if n is not None else None

    @classmethod
    def param_defs(cls):
        return [
            ('tau_rise',  float, '0.05', 'fast rise τ (membrane)'),
            ('tau_decay', float, '0.05', 'fast decay τ (membrane; blank/None = no decay, holds value)'),
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
            self.n = n
            self._init_dynamics_buffers(n)
            self.register_buffer('_s', torch.zeros(n))
            self.output = torch.zeros(n)
        elif self.n != n:
            raise ValueError(
                f"PulseLayer '{self.name}': declared n={self.n} but connection implies n={n}")

    def reset(self):
        if self.n is None:
            return
        self._reset_dynamics()
        self._s.zero_()
        self.output = torch.zeros(self.n)

    def step(self, input_vec, dt):
        u = torch.as_tensor(input_vec, dtype=torch.float32) + self.bias
        u = self._apply_output_mode(u, dt)
        x = self._apply_leaky(u, dt)

        s = self._s.detach()
        s = s + (F.relu(x - self.theta) - s) / self.tau_hold * dt
        if self.drain > 0.0:
            s = s - self.drain * F.relu(-u) * dt
            s = torch.clamp(s, min=0.0)
        self._s.copy_(s)

        out = _activate(x + self.w_s * self._s, self.activation, alpha=self.alpha) * self.scale
        self.output = out.detach()
        return self.output

    def init_code_parts(self):
        parts = [f'tau_rise={self.tau_rise}', f'tau_decay={self.tau_decay}',
                 f'tau_hold={self.tau_hold}']
        parts += self._base_code_parts()
        if self.theta != 0.0:
            parts.append(f'theta={self.theta}')
        if self.w_s != 1.0:
            parts.append(f'w_s={self.w_s}')
        if self.drain != 1.0:
            parts.append(f'drain={self.drain}')
        if self.n is not None:
            parts.append(f'n={self.n}')
        if self.bias != 0.0:
            parts.append(f'bias={self.bias}')
        if self.activation != 'relu':
            parts.append(f"activation='{self.activation}'")
        if getattr(self, 'scale', 1.0) != 1.0:
            parts.append(f'scale={self.scale}')
        if getattr(self, 'output_mode', 'none') != 'none':
            parts.append(f"output_mode='{self.output_mode}'")
        parts += self._mod_code_parts()
        return parts


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
- `amplitude` (A) — peak amplitude; default 1.0
- `frequency` (f) — oscillation frequency in Hz; default 1.0
- `phase` (φ) — initial phase offset in radians; default 0.0
- `n` — number of neurons; default 1
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
                 n=1, name='sine', color=None, layer=None,
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

    def init_code_parts(self):
        parts = [f'amplitude={self.amplitude}', f'frequency={self.frequency}']
        parts += self._base_code_parts()
        if self.phase != 0.0:
            parts.append(f'phase={self.phase}')
        if getattr(self, 'n', 1) != 1:
            parts.append(f'n={self.n}')
        parts += self._mod_code_parts()
        return parts


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
    help_text = """\
## RingAttractorLayer — N neurons on a ring

Recurrent connectivity defined by a **self-connection** (use the Mexican hat preset).

**Parameters:**
- `n` — number of neurons on the ring; default 8
- `tau_rise` (τ_rise) — rise time constant (s); default 0.1
- `tau_decay` (τ_decay) — decay time constant (s); blank/None = no decay (holds value)
- `activation` (f) — nonlinearity: `relu`, `sigmoid`, `tanh`, `linear`; default `relu`
- `bias` (b) — constant tonic drive per neuron (replaces a ConstantLayer); default 0.0
- `noise_std` / `noise_tau` — same as LeakyLayer
- `noise_tau` — noise correlation time (0 = white noise)
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

**Output mode** — `output_mode` ∈ `{none, derivative, integral}` (not listed above;
auto-added to this dialog by `DynamicsBase`). Transforms the raw input u
*before* adaptation/leaky filtering/activation run: `derivative` replaces u
with its rate of change; `integral` replaces u with its running accumulation.

**Initial value** — `x0` (not listed above; also auto-added by `DynamicsBase`)
sets the starting value of `x` and of the `output_mode='integral'` accumulator,
both at construction and on every reset. Default 0.0. Not the same as `bias`
(added every tick, not just at the start).

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

    def __init__(self, n=8, name='ring', viz_layout='ring',
                 # Legacy params accepted for migration — not used at runtime
                 tau=None, w_exc=None, sigma_exc=None, w_inh=None, sigma_inh=None,
                 **kwargs):
        if tau is not None:
            kwargs.setdefault('tau_rise', tau)
        super().__init__(name=name, **kwargs)
        self.n = n
        self.viz_layout = viz_layout
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
            ('n',          int,   '8',      'number of neurons'),
            ('tau_rise',   float, '0.1',    'rise time constant (s)'),
            ('tau_decay',  float, '0.1',    'decay time constant (s); blank/None = no decay (holds value)'),
            ('activation', str,   'relu',   'output nonlinearity', ACTIVATIONS),
            ('bias',       float, '0.0',    'constant added to input each step — replaces a ConstantLayer drive'),
            ('noise_std',  float, '0.0',    'noise amplitude'),
            ('noise_tau',  float, '0.0',    'noise correlation time (0 = white noise)'),
            ('viz_layout', str,   'ring',   'visualizer layout', ['ring', 'linear']),
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

    def step(self, input_vec, dt):
        u = torch.as_tensor(input_vec, dtype=torch.float32) + self.bias
        u = self._apply_noise(u, dt)
        u = self._apply_output_mode(u, dt)
        x = self._apply_leaky(u, dt)
        out = _activate(x, self.activation, alpha=self.alpha)
        self.output = out.detach()
        return self.output

    def init_code_parts(self):
        parts = [f'n={self.n}', f'tau_rise={self.tau_rise}']
        if self.tau_decay != self.tau_rise:
            parts.append(f'tau_decay={self.tau_decay}')
        parts += self._base_code_parts()
        if self.activation != 'relu':
            parts.append(f"activation='{self.activation}'")
        if self.bias != 0.0:
            parts.append(f'bias={self.bias}')
        if self.noise_std != 0.0:
            parts.append(f'noise_std={self.noise_std}')
        if self.noise_tau != 0.0:
            parts.append(f'noise_tau={self.noise_tau}')
        if getattr(self, 'output_mode', 'none') != 'none':
            parts.append(f"output_mode='{self.output_mode}'")
        parts += self._mod_code_parts()
        return parts


