# RingAttractorLayer

N neurons arranged on a ring with recurrent connectivity. A localised activity bump forms and can be shifted by input, making it useful for head-direction and path-integration circuits.

Recurrent connectivity is defined by a **self-connection** — use the Mexican hat preset in the connection editor.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of neurons on the ring; default 8 |
| `tau_rise` | Rise time constant (s); default 0.1 |
| `tau_decay` | Decay time constant (s); defaults to `tau_rise` |
| `activation` | `relu`, `sigmoid`, `tanh`, `linear`; default `relu` |
| `bias` | Constant tonic drive per neuron (replaces a ConstantLayer); default 0.0 |
| `noise_std` / `noise_tau` | Same as LeakyLayer |

## Dynamics

$$\frac{dx}{dt} = \frac{-x + u}{\tau}, \quad \tau = \begin{cases}\tau_{rise} & u>x \\ \tau_{decay} & u\leq x\end{cases}$$

$$\text{output} = f(x)$$

## Requirements for bump formation

1. Self-connection kernel: **Mexican hat**. All row sums must be **negative**.
2. Tonic drive: `bias > 0` or a one-to-one `ConstantLayer` (0.1–1.0 per neuron).
3. Excitatory width σ_exc ≥ 1/n. For `n=8`: σ_exc ≥ 0.15.

!!! tip
    If all neurons stay at the same nonzero value, drive is too strong or the Mexican hat row sums are not sufficiently negative. If you get two bumps, σ_exc is too narrow — widen until neighbours are excitatory.

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — `(name, scale, site)` triples: `site="pre"` / `"post"` / `"none"`.
