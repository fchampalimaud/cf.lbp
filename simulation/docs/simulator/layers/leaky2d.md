# Leaky2dLayer

Pixel-wise temporal filter — applies the same first-order leaky integration as `LeakyLayer` independently to **each pixel**, producing an output image of the same spatial size. Downstream `Conv2dLayer` nodes can use this as their source.

## Parameters

| Parameter | Description |
|---|---|
| `tau_rise` | Rise time constant (s); default 0.1 |
| `tau_decay` | Decay time constant (s); blank/None = no decay, holds value |
| `bias` | Constant added to each pixel before integration; default 0.0 |
| `activation` | Per-pixel nonlinearity: `linear`, `relu`, `sigmoid`, `tanh`; default `linear` |
| `scale` | Output multiplier; default 1.0 |
| `output_mode` | `none` / `derivative` / `integral` (per pixel); default `none` |
| `x0` | Initial value of each pixel's `x` (and the `integral` accumulator); default 0.0 |
| `noise_std` / `noise_tau` | Per-pixel noise (same as LeakyLayer) |

## Dynamics

$$\frac{dx}{dt} = \frac{u - x}{\tau}, \quad \tau = \begin{cases} \tau_{rise} & u > x \\ \tau_{decay} & u \leq x \end{cases}$$

$$\text{output pixel} = f(x) \times s$$

## Motion mode

`output_mode='derivative'` transforms each pixel's raw input into its own rate of change between ticks *before* the leaky filter/activation run. Use `scale = -1` to flip which direction (brightening vs. darkening) reads positive.

## Optic flow recipe

1. `GrayCameraSensor` → `Leaky2dLayer(tau_rise=0.2, output_mode='derivative', scale=-1, activation='relu')`
2. `Leaky2dLayer` → `Conv2dLayer` to extract spatial motion features.

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — `(name, scale, site)` triples: `site="pre"` / `"post"` / `"none"`.
