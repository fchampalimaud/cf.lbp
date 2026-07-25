# Leaky2dLayer

Pixel-wise temporal filter — applies the same first-order leaky integration as `LeakyLayer` independently to **each pixel**, producing an output image of the same spatial size. Downstream `Conv2dLayer` nodes can use this as their source.

## Parameters

| Parameter | Description |
|---|---|
| `tau_rise` | Rise time constant (s); default 0.1 |
| `tau_decay` | Decay time constant (s); defaults to `tau_rise` |
| `bias` | Constant added to each pixel before integration; default 0.0 |
| `activation` | Per-pixel nonlinearity: `linear`, `relu`, `sigmoid`, `tanh`; default `linear` |
| `scale` | Output multiplier; default 1.0 |
| `derivative` | `True` → outputs x − u (fires on pixel *decrease*); default False |
| `noise_std` / `noise_tau` | Per-pixel noise (same as LeakyLayer) |

## Dynamics

$$\frac{dx}{dt} = \frac{u - x}{\tau}, \quad \tau = \begin{cases} \tau_{rise} & u > x \\ \tau_{decay} & u \leq x \end{cases}$$

$$\text{output pixel} = f(x) \times s$$

## Derivative / motion mode

`derivative=True`:

$$\text{output pixel} = f(x - u) \times s$$

x lags u, so x − u > 0 only when u has recently *fallen*. Use `scale = -1` to detect *increases* (brighter = active).

## Optic flow recipe

1. `GrayCameraSensor` → `Leaky2dLayer(tau_rise=0.2, derivative=True, scale=-1, activation='relu')`
2. `Leaky2dLayer` → `Conv2dLayer` to extract spatial motion features.

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — `(name, scale, site)` triples: `site="pre"` / `"post"` / `"none"`.
