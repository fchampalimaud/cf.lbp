import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neurons_base import ACTIVATIONS, _activate, DynamicsBase, LayerBase, dt_ratio

class LearningLayerBase(DynamicsBase, LayerBase):
    """
    Base class for all reward-driven learning layers.

    Forward pass: V = Σ_conn W @ s  (linear weighted sum over incoming connections)
    Weight update: ΔW = α_eff · δ · s  where s is the previous tick's input for TD and
    the current input for Delta / ThreeFactor (_credit_previous_input), and α_eff is
    alpha_pos (δ≥0) or alpha_neg (δ<0) — per tick at the default dt, scaled with dt.
    Episodic state: _src_prev, _V_prev — cleared on reset(); connection weights survive.

    Subclasses implement _compute_delta(V, r) → δ tensor of shape (n,).
    Optional leaky dynamics on V output via DynamicsBase (_x buffer, tau_rise/tau_decay).
    """

    is_learning = True   # capability (see LayerBase)

    def __init__(self, n=1, alpha_pos=0.01, alpha_neg=None,
                 tau_rise=0.0, tau_decay=None, activation='linear',
                 bias=0.0, scale=1.0, noise_std=0.0, noise_tau=0.0, output_mode='none',
                 reward_modulator=None,
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
        # Legacy field, superseded by a 'drives_plasticity' row in `modulators`.
        # Defaults to None (not 'dopamine'): network_runner.py applies this
        # additively and unconditionally whenever truthy, so a non-empty
        # default here would silently bypass the new threshold gating for
        # every freshly-created instance. Old saved JSON always has this
        # field explicitly set (it was unconditionally serialized before
        # this change), so existing networks are unaffected.
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
        # Always: output_mode='integral' needs _integral and tau_rise can be
        # raised later in the dialog — state that isn't used just stays at rest.
        self._init_dynamics_buffers(self.n)

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

    # Which presynaptic activity the weight update credits: the previous tick's
    # (TD — δ compares V_t with V_{t-1}) or the current one (δ about V_t = W·s_t).
    _credit_previous_input = True

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

    def step_td(self, src_inputs, dt, outgoing=None):
        """Forward pass + weight update. Called by the runner with collected connection data.

        `outgoing` is unused here — it exists only so the runner can call every
        LearningLayerBase-family layer's step_td with the same signature.
        SnapshotLayer is the one subclass that actually reads it (its
        plasticity lives on outgoing connections, not incoming ones)."""
        if not src_inputs:
            self.output = torch.zeros(self.n)
            return self.output

        V = torch.zeros(self.n)
        for src_val, w_cached, conn_idx, conn in src_inputs:
            V = V + w_cached @ src_val
        V = self._filter(self._input(V, dt), dt)   # same pipeline as every layer

        r     = self._reward
        delta = self._compute_delta(V, r)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        mask      = self._competition_mask(V)
        # Learning rates are per tick at the default dt (DT_REF): scaled with
        # dt so learning per second is the same at any time step.
        rate      = dt_ratio(dt)
        alpha_eff = torch.where(delta >= 0,
                                torch.full_like(delta, self.alpha_pos * rate),
                                torch.full_like(delta, self.alpha_neg * rate))
        w_lo = float(self.w_min) if self.w_min not in (None, '', 'none') else -float('inf')
        w_hi = float(self.w_max) if self.w_max not in (None, '', 'none') else  float('inf')
        for src_val, w_cached, conn_idx, conn in src_inputs:
            # Credit goes to the input that produced the error: the previous
            # tick's input for TD (δ compares V_t with V_{t-1}), the current one
            # for Delta / ThreeFactor (δ is about V_t = W·s_t).
            if self._credit_previous_input:
                pre = self._src_prev.get(conn_idx, torch.zeros_like(src_val))
            else:
                pre = src_val
            w_cached.add_(torch.outer(alpha_eff * delta * mask, pre))
            torch.nan_to_num_(w_cached, nan=0.0, posinf=0.0, neginf=0.0)
            if self.weight_decay > 0:
                w_cached.mul_(1.0 - self.weight_decay * dt)
            torch.clamp_(w_cached, w_lo, w_hi)
            conn.W = w_cached.detach().numpy().copy()

        for src_val, w_cached, conn_idx, conn in src_inputs:
            self._src_prev[conn_idx] = src_val.detach().clone()

        self._V_prev = V.detach()
        out = self._emit(V) * mask
        self.output  = out.detach()
        return self.output

    def _learning_code_parts(self):
        """Common code parts for all LearningLayerBase subclasses."""
        parts = []
        if self.reward_modulator:   # legacy field — default is now None, not 'dopamine'
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

**Order of operations** (`step_td`, per tick):
1. `V = Σ_conn W_conn · s_conn` — weighted sum over incoming connections
2. apply `output_mode` transform to `V` — derivative/integral (if not `none`)
3. `V += bias`, then the optional leaky filter `V = leaky(V)` (only if `tau_rise > 0`)
4. `δ = r + γ·V − V_prev` (`r` = current reward signal from the modulator bus)
5. sanitize `δ` (replace NaN/±Inf with 0)
6. `mask = competition_mask(V)` (`none` → all ones; `wta` → top-k one-hot; `softmax` → softmax(V))
7. `α_eff = alpha_pos` where `δ ≥ 0`, else `alpha_neg`
8. for each incoming connection: `ΔW = outer(α_eff · δ · mask, s_prev)`; `W += ΔW`; sanitize; if `weight_decay > 0`: `W *= (1 − weight_decay·dt)`; clamp `W` to `[w_min, w_max]`
9. store this tick's `s` as `s_prev` and `V` as `V_prev` for next tick
10. `output = activation(V) × scale × mask`

**Wiring:**
1. Connect any sensory/feature layer → TDLayer. Initialize the connection W to zeros.
2. Declare the reward-carrying layer as a neuromodulator transmitter (e.g. "dopamine").
3. In this layer's modulator receptor table, add a row for that name and check
   "Drives Plasticity" — the layer reads r from it each tick.
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
                 alpha_neg=None, gamma=0.99, reward_modulator=None,
                 tau_rise=0.0, tau_decay=None, activation='linear', bias=0.0, scale=1.0,
                 noise_std=0.0, noise_tau=0.0,
                 weight_decay=0.0, w_min=None, w_max=None, competition='none', k=1,
                 name='td', **kwargs):
        # No local 'alpha' param here (unlike the old pre-alpha_pos/alpha_neg
        # shim) — lets a real 'alpha' kwarg flow through to DynamicsBase's
        # ELU alpha instead of being silently swallowed.
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
            ('alpha_pos',        float, '0.01',     'learning rate for δ ≥ 0 (acquisition; per tick at dt 0.01)'),
            ('alpha_neg',        float, '0.01',     'learning rate for δ < 0 (extinction; per tick at dt 0.01)'),
            ('gamma',            float, '0.99',     'discount factor γ (0–1)'),
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

**Order of operations** (`step_td`, per tick):
1. `V = Σ_conn W_conn · s_conn` — weighted sum over incoming connections
2. apply `output_mode` transform to `V` — derivative/integral (if not `none`)
3. `V += bias`, then the optional leaky filter `V = leaky(V)` (only if `tau_rise > 0`)
4. `δ = r − V` (`r` = current reward signal from the modulator bus)
5. sanitize `δ` (replace NaN/±Inf with 0)
6. `mask = competition_mask(V)` (`none` → all ones; `wta` → top-k one-hot; `softmax` → softmax(V))
7. `α_eff = alpha_pos` where `δ ≥ 0`, else `alpha_neg`
8. for each incoming connection: `ΔW = outer(α_eff · δ · mask, s)` with this tick's input `s` — the one that produced `V`; `W += ΔW`; sanitize; if `weight_decay > 0`: `W *= (1 − weight_decay·dt)`; clamp `W` to `[w_min, w_max]`
9. store `V` as `V_prev` (unused by this layer's own `δ`)
10. `output = activation(V) × scale × mask`

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
                 alpha_neg=0.005, reward_modulator=None,
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

    _credit_previous_input = False   # Rescorla-Wagner: δ_t = r − W·s_t → credit s_t

    def _compute_delta(self, V, r):
        return r - V

    @classmethod
    def param_defs(cls):
        return [
            ('n',                int,   '1',        'number of output neurons (parallel critics)'),
            ('alpha_pos',        float, '0.05',     'learning rate for δ ≥ 0 (acquisition; per tick at dt 0.01)'),
            ('alpha_neg',        float, '0.005',    'learning rate for δ < 0 (extinction; per tick at dt 0.01)'),
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

$$\\Delta W_{ji} = \\alpha_{\\text{eff}} \\cdot r \\cdot V_j \\cdot s_i$$

- `r` — reward signal from the designated neuromodulator (gates all plasticity)
- `V_j` — postsynaptic activity of neuron j (selects *which* neuron learns)
- `s_i` — presynaptic activity on the same tick, the input that produced `V_j` (selects *which* inputs)
- `α_eff` = `alpha_pos` if r·V ≥ 0, else `alpha_neg`

**Passive forgetting (independent of reward):**

$$W \\leftarrow W \\cdot (1 - \\text{decay} \\cdot dt)$$

Applied every tick regardless of r. Equilibrium weight reflects the balance between
acquisition rate and decay rate — infrequently rewarded associations fade naturally.

**Order of operations** (`step_td`, per tick):
1. `V = Σ_conn W_conn · s_conn` — weighted sum over incoming connections
2. apply `output_mode` transform to `V` — derivative/integral (if not `none`)
3. `V += bias`, then the optional leaky filter `V = leaky(V)` (only if `tau_rise > 0`)
4. `δ = r · V` (`r` = current reward signal from the modulator bus)
5. sanitize `δ` (replace NaN/±Inf with 0)
6. `mask = competition_mask(V)` (`none` → all ones; `wta` → top-k one-hot; `softmax` → softmax(V))
7. `α_eff = alpha_pos` where `δ ≥ 0`, else `alpha_neg`
8. for each incoming connection: `ΔW = outer(α_eff · δ · mask, s)` with this tick's input `s`; `W += ΔW`; sanitize; then (regardless of `δ`) if `weight_decay > 0`: `W *= (1 − weight_decay·dt)`; clamp `W` to `[w_min, w_max]`
9. (nothing to store — pre and post coincide on the same tick)
10. `output = activation(V) × scale × mask`

**Parameters:**
- `n` — number of output neurons; default 1
- `alpha_pos` — learning rate when r·V ≥ 0; default 0.01
- `alpha_neg` — learning rate when r·V < 0 (punishment); default = alpha_pos
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
3. In this layer's modulator receptor table, add a row for that name and check
   "Drives Plasticity" (with an optional threshold — the row's transformed
   value must cross it for that tick to count).
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
                 alpha_neg=None, reward_modulator=None,
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

    _credit_previous_input = False   # pre and post must coincide: r · V_t · s_t

    def _compute_delta(self, V, r):
        return r * V

    @classmethod
    def param_defs(cls):
        return [
            ('n',                int,   '1',        'number of output neurons'),
            ('alpha_pos',        float, '0.01',     'learning rate for r·V ≥ 0 (per tick at dt 0.01)'),
            ('alpha_neg',        float, '0.01',     'learning rate for r·V < 0 (punishment; per tick at dt 0.01)'),
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


class SnapshotLayer(LearningLayerBase):
    has_outgoing_plasticity = True   # capability (see LayerBase): teacher readout

    help_text = """\
## SnapshotLayer — one-shot vector-memory neuron (Le Moël et al. 2019)

Implements a **hard-overwrite, one-shot** memory: on a reward trigger, its
*outgoing* connection weights are set directly to (minus) a named source
layer's current output — not nudged gradually like `TDLayer`/`DeltaLayer`/
`ThreeFactorLayer`. Recall is then just this neuron's own output (0 or
graded) multiplying those frozen weights back out — a snapshot, held exactly
until the next trigger, regardless of how much the source drifts afterward.

Two categorically different incoming connections, distinguished by `src` name:

- **Teach connection** — `src == teach_source`. A real, ordinary connection
  (not a neuromodulator-style broadcast — the dependency is structured and
  per-channel, e.g. one source column informing one output synapse, not a
  diffuse signal any receptor could pick up). Its raw value is read for the
  write; it does **not** contribute to this neuron's own output.
- **Gate connection(s)** — everything else. Drives this neuron's own output
  normally, exactly like any other layer's inputs.

**Outgoing connections** (this layer → wherever the memory is expressed) are
not learned via a Hebbian rule — their weights *are* the memory itself,
directly overwritten on trigger. This is the one layer type in the
`LearningLayerBase` family whose plasticity lives on its *outgoing* edges,
not its incoming ones.

**Parameters:**
- `n` — number of independent memory units; default 1. At `n=1` this matches
  Le Moël's "single vector-memory neuron" exactly. `n>1` is accepted as a
  structural parameter but every unit currently writes together on any
  trigger (no independent per-slot gating yet — see "When to use").
- `teach_source` — name of the layer whose current output is copied (negated)
  into every outgoing connection's weights on trigger.
- `tau_rise` / `tau_decay` — leaky dynamics on this neuron's own output
  (from gate connections only).
- `activation`, `bias`, `scale` — standard output shaping.
- `weight_decay` / `w_min` / `w_max` — applied to the *outgoing* (stored
  memory) weights every tick, not to any incoming connection: passive
  forgetting and bounds on the stored content.
- `competition` / `k` — lateral competition over this neuron's own output
  (meaningful once `n>1` recall is designed; a no-op at `n=1`).

**Write rule** (Le Moël Eq. 14, on any tick `self._reward` is nonzero — wire
a `modulators` row with `drives_plasticity=True` on this layer, same
mechanism every other `LearningLayerBase` layer already uses for reward):

$$W_{\\text{outgoing}} \\leftarrow -\\,\\text{teach\\_source.output}$$

A hard overwrite, not `ΔW = α·δ·s_prev` — old content is fully discarded
each time, matching the paper's description exactly rather than converging
toward it gradually.

**Output (recall):** `output = activation(gate_contribution) · scale · mask`
— completely ordinary, computed only from non-teach incoming connections.

**Order of operations** (`step_td`, per tick):
1. split incoming connections by `src`: the **teach connection** (`src == teach_source`) vs. everything else (**gate connection(s)**)
2. `V = Σ_conn W_conn · s_conn` over the gate connections only — the teach connection's value is *not* summed into `V`
3. apply `output_mode` transform to `V` — derivative/integral (if not `none`)
4. `V += bias`, then the optional leaky filter `V = leaky(V)` (only if `tau_rise > 0`)
5. `mask = competition_mask(V)`
6. `output = activation(V) × scale × mask`
7. for each outgoing connection: if `self._reward` is nonzero *and* a teach value was found, hard-overwrite `W_outgoing ← −teach_source.output` (every output column set to the same negated vector)
8. if `weight_decay > 0`: `W_outgoing *= (1 − weight_decay·dt)` — applied every tick to the outgoing weights, regardless of whether a write happened
9. sanitize (replace NaN/±Inf with 0) and clamp `W_outgoing` to `[w_min, w_max]`; write back

**Wiring:**
1. Connect the layer to snapshot (e.g. a path-integration accumulator) →
   this layer, with a real weight matrix (e.g. identity/one-to-one if
   preserving per-channel structure matters, as it typically does).
2. Set `teach_source` to that layer's name.
3. Connect a gate/recall driver (e.g. a `ConstantLayer`, or a context
   signal) → this layer — its value becomes this neuron's own output.
4. Connect this layer → wherever the memory should be expressed (its
   outgoing connections' weights will be overwritten on trigger — give
   them any placeholder initial `W` of the correct shape).
5. Declare a reward-carrying layer as a neuromodulator transmitter, and add
   a `modulators` row on *this* layer with `drives_plasticity=True`.

**Reset behaviour:** ↺ Reset clears this neuron's own episodic output state,
but — like every `LearningLayerBase` sibling — leaves connection weights
(including the stored memory on outgoing connections) intact, since the
whole point is that the memory survives across resets until next written.

**When to use:** A single, hard-overwritten spatial or feature memory that
must stay frozen against a live, continuously-changing source until
explicitly rewritten — e.g. an insect path-integration "vector memory"
neuron. For `n>1` independent memory slots (e.g. one per remembered
location, as in trapline foraging), the recall side isn't implemented yet:
every unit currently reads the same trigger and writes together.

**References:**
- Le Moël, F., Stone, T., Lihoreau, M., Wystrach, A. & Webb, B. (2019). The
  Central Complex as a Potential Substrate for Vector Based Navigation.
  *Frontiers in Psychology*, 10:690.
- Goulard, R., Heinze, S. & Webb, B. (2023). Emergent spatial goals in an
  integrative model of the insect central complex. *PLOS Computational
  Biology*, 19(12): e1011480.
"""

    def __init__(self, n=1, teach_source=None,
                 tau_rise=0.0, tau_decay=None, activation='linear', bias=0.0, scale=1.0,
                 noise_std=0.0, noise_tau=0.0,
                 weight_decay=0.0, w_min=None, w_max=None, competition='none', k=1,
                 name='snapshot', **kwargs):
        super().__init__(n=n,
                         tau_rise=tau_rise, tau_decay=tau_decay,
                         activation=activation, bias=bias, scale=scale,
                         noise_std=noise_std, noise_tau=noise_tau,
                         w_min=w_min, w_max=w_max, competition=competition, k=k,
                         weight_decay=weight_decay, name=name, **kwargs)
        self.teach_source = teach_source

    def step_td(self, src_inputs, dt, outgoing=None):
        """Forward pass from gate connections only, plus a one-shot overwrite
        of every outgoing connection's weights when self._reward is nonzero.

        Unlike the base class, this never touches the weights of its own
        incoming connections — the teach connection's weight is irrelevant
        (only its raw src_val is read), and the gate connection(s)' weights
        are ordinary, hand-set, non-plastic values.
        """
        teach_val   = None
        gate_inputs = []
        for src_val, w_cached, conn_idx, conn in src_inputs:
            if conn.src == self.teach_source:
                teach_val = src_val
            else:
                gate_inputs.append((src_val, w_cached, conn_idx, conn))

        V = torch.zeros(self.n)
        for src_val, w_cached, conn_idx, conn in gate_inputs:
            V = V + w_cached @ src_val
        V = self._filter(self._input(V, dt), dt)   # same pipeline as every layer

        mask = self._competition_mask(V)
        out  = self._emit(V) * mask
        self.output = out.detach()

        if outgoing:
            w_lo = float(self.w_min) if self.w_min not in (None, '', 'none') else -float('inf')
            w_hi = float(self.w_max) if self.w_max not in (None, '', 'none') else  float('inf')
            write = bool(self._reward) and teach_val is not None
            for w_cached, conn_idx, conn in outgoing:
                if write and w_cached.shape[0] == teach_val.shape[0]:
                    new_col = -teach_val.detach()
                    w_cached.copy_(new_col.unsqueeze(1).expand(-1, w_cached.shape[1]))
                if self.weight_decay > 0:
                    w_cached.mul_(1.0 - self.weight_decay * dt)
                torch.nan_to_num_(w_cached, nan=0.0, posinf=0.0, neginf=0.0)
                torch.clamp_(w_cached, w_lo, w_hi)
                conn.W = w_cached.detach().numpy().copy()

        return self.output

    @classmethod
    def param_defs(cls):
        return [
            ('n',            int, '1', 'number of independent memory units (n>1: recall/selection not yet independent per unit)'),
            ('teach_source', str, '',  'name of the layer whose output is copied (negated) into outgoing weights on write'),
            ('tau_rise',     float, '0.0',    'leaky rise τ on output (0 = off)'),
            ('tau_decay',    float, '0.0',    'leaky decay τ on output'),
            ('activation',   str,   'linear', 'output nonlinearity', ACTIVATIONS),
            ('bias',         float, '0.0',    'constant added to output'),
            ('scale',        float, '1.0',    'output scale factor'),
        ] + cls._shared_learning_param_defs()

    def init_code_parts(self):
        return ([f'n={self.n}', f'teach_source={self.teach_source!r}']
                + self._learning_code_parts())


# Registry — populated automatically by LayerBase.__init_subclass__ as each class is defined.
LAYER_REGISTRY = LayerBase._registry
