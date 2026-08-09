# LeakyLayer

First-order low-pass filter neurons. Output approaches input with time constant `tau_rise` and decays with `tau_decay`.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of neurons |
| `tau_rise` | Rise time constant (s); default 0.1 |
| `tau_decay` | Decay time constant (s); blank/None = no decay, holds value |
| `bias` | Constant added to each input sum; default 0.0 |
| `activation` | `relu`, `sigmoid`, `tanh`, `linear`; default `relu` |
| `scale` | Output multiplier; default 1.0 |
| `output_mode` | `none` / `derivative` / `integral`; default `none` |
| `x0` | Initial value of `x` (and the `integral` accumulator), at construction and on reset; default 0.0 |
| `noise_std` | Gaussian noise std on u each step; default 0.0 |
| `noise_tau` | Ornstein-Uhlenbeck time constant (0 = white noise); default 0.0 |

## Dynamics

$$\frac{dx}{dt} = \frac{u - x}{\tau}, \quad \tau = \begin{cases} \tau_{rise} & u > x \\ \tau_{decay} & u \leq x \end{cases}, \quad x(0) = x_0$$

$$\text{output} = f(x) \times s$$

| Setting | Behaviour |
|---|---|
| `tau_rise = tau_decay` | Symmetric smoothing |
| `tau_rise < tau_decay` | Fast rise, slow decay — memory trace |
| `tau_rise > tau_decay` | Slow rise, fast decay — transient detector |
| `tau_decay = None` (blank) | No decay — x rises toward u but holds when u drops |

## Output mode

Shared by every `DynamicsBase` layer/sensor (see `DynamicsBase._apply_output_mode`) — applied to the raw input **u**, before the leaky filter and activation above run:

- `output_mode='derivative'`: replaces u with du/dt — finite difference between consecutive ticks' raw input; fires on any change, both rising and falling. Set `scale = -1` to flip which direction reads positive.
- `output_mode='integral'`: replaces u with the running accumulation (∫u dt).

## Initial value (`x0`)

Seeds both `x` and the `output_mode='integral'` accumulator, at construction *and* on every reset. Needed for a bounded integrator whose resting point isn't the edge of its range — e.g. a `[0, 1]`-clipped memory neuron that must move both up and down needs `x0=0.5`, not 0 (the CPU4 path-integration memory in Stone et al. 2017's bee-navigation model is the canonical example). **Not the same as `bias`**: bias is added to the input every tick, so under `output_mode='integral'` a nonzero bias makes the accumulator drift forever instead of just setting where it starts.

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — `(name, scale, site)` triples:
  - `site="pre"` multiplies the input sum; `site="post"` multiplies the output.
  - Feedback paths use the previous tick's value (one-step delay).
