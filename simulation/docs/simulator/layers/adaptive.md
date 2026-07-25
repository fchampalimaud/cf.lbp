# AdaptiveLayer

Leaky integrator with spike-frequency adaptation. Can function as a burst neuron or, with `n=2` and mutual inhibition, as a half-centre oscillator (CPG).

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of neurons; default 2 |
| `tau_rise` | Membrane rise time constant (s); default 0.1 |
| `tau_decay` | Membrane decay time constant (s); defaults to `tau_rise` |
| `tau_a` | Adaptation time constant (s); default 0.5 |
| `beta` | Adaptation strength; default 1.0 |
| `w` | Mutual inhibition weight (CPG mode, n=2); default 0.0 |
| `bias` | Constant added to each input sum; default 0.0 |
| `activation` | `relu`, `sigmoid`, `tanh`, `linear`; default `relu` |
| `scale` | Output multiplier; default 1.0 |

## Dynamics

$$u_{eff} = u - \beta a - w \cdot o_{other}$$

$$\frac{dx}{dt} = \frac{u_{eff} - x}{\tau}, \quad \frac{da}{dt} = \frac{f(x) - a}{\tau_a}$$

$$\text{output} = f(x) \times \text{scale}$$

**Adaptation** (`beta > 0`): a tracks output and inhibits it. Larger β → shorter burst. Smaller τ_a → faster adaptation.

## CPG oscillation mode

With `n=2` and `w > 0`, neurons mutually inhibit to produce alternating bursts:

$$\text{Period} \approx 2\,\tau_a$$

Requirements: `w >= 1`, `beta ~ 2–3`, `bias > 0` (tonic drive).

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — `(name, scale, site)` triples:
  - `site="pre"` multiplies the input sum; `site="post"` multiplies the output.
  - Feedback paths use the previous tick's value (one-step delay).
