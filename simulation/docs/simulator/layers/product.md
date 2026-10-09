# ProductLayer

Same leaky-integrator dynamics as **LeakyLayer** — the only difference is
upstream: incoming connections combine by **elementwise product** instead of
sum before these dynamics run.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of neurons |
| `tau_rise` | Rise time constant (s); 0 or blank = instant rise; default 0.1 |
| `tau_decay` | Decay time constant (s); 0 or blank = instant decay |
| `bias` | Constant added to the product each step; default 0.0 |
| `activation` | `relu`, `sigmoid`, `tanh`, `linear`; default `relu` |
| `scale` | Output multiplier; default 1.0 |
| `output_mode` | `none` / `derivative` / `integral`; default `none` |
| `x0` | Initial value of `x` (and the `integral` accumulator), at construction and on reset; default 0.0 |
| `noise_std` | Gaussian noise std on u each step; default 0.0 |
| `noise_tau` | Ornstein-Uhlenbeck time constant (0 = white noise); default 0.0 |

## Dynamics

$$u = \prod_k W_k \cdot \text{input}_k + b$$

$$\frac{dx}{dt} = \frac{u - x}{\tau}, \quad \tau = \begin{cases} \tau_{rise} & u > x \\ \tau_{decay} & u \leq x \end{cases}$$

$$\text{output} = f(x) \times s$$

Each connection is still an ordinary weighted transform (its own weight
matrix/pattern); only the **fan-in across connections** is multiplicative
instead of additive. A single incoming connection is just a weighted
pass-through. With zero connections the product is 1 (not 0, as SumLayer's
empty sum would be), before bias/dynamics/activation/scale.

Use for coincidence detection / gating — an output that requires two
sensory drives to be active simultaneously (an AND-like unit), or a signal
that scales another rather than adding to it. Contrast with **SumLayer** /
**LeakyLayer** (additive fan-in). Not the same as neuromodulation
(`modulators` gain-scales the *aggregate* input/output of a layer from a
tagged transmitter signal) — this multiplies two ordinary afferent
connections together at the neuron itself.

**Bonsai export is not supported.** LBP.Torch's `JoinAdditive` only sums
fan-in; there is no multiplicative join node. `Copy Bonsai` raises an error
if a ProductLayer is present in the circuit.

## Output mode

Shared by every `DynamicsBase` layer/sensor — applied to the raw product **u**, before the leaky filter/activation above run: `output_mode='derivative'` replaces u with du/dt (rate of change of the product itself); `output_mode='integral'` replaces u with the running ∫u dt.

## Initial value (`x0`)

Seeds both `x` and the `output_mode='integral'` accumulator, at construction and on every reset. Not the same as `bias` (added every tick — under `output_mode='integral'` a nonzero bias makes the accumulator drift forever instead of just setting where it starts).

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — `(name, scale, site)` triples:
  - `site="pre"` multiplies the input product; `site="post"` multiplies the output.
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
