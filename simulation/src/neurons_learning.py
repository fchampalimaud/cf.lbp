import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neurons_base import ACTIVATIONS, _activate, DynamicsBase, LayerBase

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
                 bias=0.0, scale=1.0, noise_std=0.0, noise_tau=0.0, output_mode='none',
                 reward_modulator='dopamine',
                 w_min=None, w_max=None,
                 competition='none', k=1,
                 weight_decay=0.0,
                 name='learning', **kwargs):
        super().__init__(name=name,
                         tau_rise=tau_rise, tau_decay=tau_decay, activation=activation,
                         bias=bias, scale=scale, noise_std=noise_std, noise_tau=noise_tau,
                         output_mode=output_mode, **kwargs)
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
        V = self._apply_output_mode(V, dt)

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
        out = _activate(V, self.activation, alpha=self.alpha) * self.scale * mask
        self.output  = out.detach()
        return self.output

    def _learning_code_parts(self):
        """Common code parts for all LearningLayerBase subclasses."""
        parts = []
        if self.reward_modulator != 'dopamine':
            parts.append(f'reward_modulator={self.reward_modulator!r}')
        parts += self._base_code_parts()
        if self.tau_rise:
            parts.append(f'tau_rise={self.tau_rise}')
            if self.tau_decay != self.tau_rise:
                parts.append(f'tau_decay={self.tau_decay}')
        if self.activation != 'linear':
            parts.append(f"activation='{self.activation}'")
        if self.bias != 0.0:
            parts.append(f'bias={self.bias}')
        if self.scale != 1.0:
            parts.append(f'scale={self.scale}')
        if getattr(self, 'output_mode', 'none') != 'none':
            parts.append(f"output_mode='{self.output_mode}'")
        if self.weight_decay:
            parts.append(f'weight_decay={self.weight_decay}')
        if self.w_min is not None:
            parts.append(f'w_min={self.w_min!r}')
        if self.w_max is not None:
            parts.append(f'w_max={self.w_max!r}')
        if self.competition != 'none':
            parts.append(f'competition={self.competition!r}')
        if self.k != 1:
            parts.append(f'k={self.k}')
        parts += self._mod_code_parts()
        return parts


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
- `tau_rise` — leaky rise τ on V output (0 = off)
- `tau_decay` — leaky decay τ on V output
- `activation` — output nonlinearity
- `bias` — constant added to V before output
- `scale` — output scale factor
- `weight_decay` — passive weight decay rate (per second; 0 = off)
- `w_min` — min synaptic weight (blank = unbounded)
- `w_max` — max synaptic weight (blank = unbounded)
- `competition` — lateral competition: none / softmax / wta
- `k` — number of winners for wta competition
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

    def __init__(self, n=1, alpha_pos=0.01,
                 alpha_neg=None, gamma=0.99, reward_modulator='dopamine',
                 tau_rise=0.0, tau_decay=None, activation='linear', bias=0.0, scale=1.0,
                 noise_std=0.0, noise_tau=0.0,
                 weight_decay=0.0, w_min=None, w_max=None, competition='none', k=1,
                 name='td', alpha=None, **kwargs):  # alpha: legacy pre-alpha_pos/alpha_neg JSON field, ignored (also means ELU's alpha isn't configurable on TDLayer)
        _pos = float(alpha_pos)
        _neg = float(alpha_neg) if alpha_neg is not None else _pos
        super().__init__(n=n, alpha_pos=_pos, alpha_neg=_neg,
                         tau_rise=tau_rise, tau_decay=tau_decay,
                         activation=activation, bias=bias, scale=scale,
                         noise_std=noise_std, noise_tau=noise_tau,
                         reward_modulator=reward_modulator,
                         w_min=w_min, w_max=w_max, competition=competition, k=k,
                         weight_decay=weight_decay, name=name, **kwargs)
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

    def init_code_parts(self):
        return ([f'n={self.n}', f'alpha_pos={self.alpha_pos}',
                 f'alpha_neg={self.alpha_neg}', f'gamma={self.gamma}']
                + self._learning_code_parts())


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
- `tau_rise` — leaky rise τ on V output (0 = off)
- `tau_decay` — leaky decay τ on V output
- `activation` — output nonlinearity
- `bias` — constant added to V before output
- `scale` — output scale factor
- `weight_decay` — passive weight decay rate (per second; 0 = off)
- `w_min` — min synaptic weight (blank = unbounded)
- `w_max` — max synaptic weight (blank = unbounded)
- `competition` — lateral competition: none / softmax / wta
- `k` — number of winners for wta competition
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

    def __init__(self, n=1, alpha_pos=0.05,
                 alpha_neg=0.005, reward_modulator='dopamine',
                 tau_rise=0.0, tau_decay=None, activation='linear', bias=0.0, scale=1.0,
                 noise_std=0.0, noise_tau=0.0,
                 weight_decay=0.0, w_min=None, w_max=None, competition='none', k=1,
                 name='delta', **kwargs):
        super().__init__(n=n, alpha_pos=alpha_pos, alpha_neg=alpha_neg,
                         tau_rise=tau_rise, tau_decay=tau_decay,
                         activation=activation, bias=bias, scale=scale,
                         noise_std=noise_std, noise_tau=noise_tau,
                         reward_modulator=reward_modulator,
                         w_min=w_min, w_max=w_max, competition=competition, k=k,
                         weight_decay=weight_decay, name=name, **kwargs)

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

    def init_code_parts(self):
        return ([f'n={self.n}', f'alpha_pos={self.alpha_pos}',
                 f'alpha_neg={self.alpha_neg}']
                + self._learning_code_parts())


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
- `tau_rise` — leaky rise τ on output (0 = off)
- `tau_decay` — leaky decay τ on output
- `activation` — output nonlinearity
- `bias` — constant added to output
- `scale` — output scale factor
- `weight_decay` — passive decay rate (s⁻¹); default 0.0
- `w_min` / `w_max` — synaptic bounds (blank = unbounded)
- `w_max` — max synaptic weight (blank = unbounded)
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

    def __init__(self, n=1, alpha_pos=0.01,
                 alpha_neg=None, reward_modulator='dopamine',
                 tau_rise=0.0, tau_decay=None, activation='linear', bias=0.0, scale=1.0,
                 noise_std=0.0, noise_tau=0.0,
                 weight_decay=0.0, w_min=None, w_max=None, competition='none', k=1,
                 name='3factor', **kwargs):
        super().__init__(n=n, alpha_pos=alpha_pos,
                         alpha_neg=float(alpha_neg) if alpha_neg is not None else alpha_pos,
                         tau_rise=tau_rise, tau_decay=tau_decay,
                         activation=activation, bias=bias, scale=scale,
                         noise_std=noise_std, noise_tau=noise_tau,
                         reward_modulator=reward_modulator,
                         w_min=w_min, w_max=w_max, competition=competition, k=k,
                         weight_decay=weight_decay, name=name, **kwargs)

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

    def init_code_parts(self):
        return ([f'n={self.n}', f'alpha_pos={self.alpha_pos}',
                 f'alpha_neg={self.alpha_neg}']
                + self._learning_code_parts())


# Registry — populated automatically by LayerBase.__init_subclass__ as each class is defined.
LAYER_REGISTRY = LayerBase._registry
