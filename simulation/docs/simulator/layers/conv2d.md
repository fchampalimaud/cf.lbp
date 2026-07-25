# Conv2dLayer

2-D convolution over camera input. Connect a `CameraSensor` to `Conv2dLayer`, then open the **Filter Stack** dialog to edit kernels. Each filter is a `(in_ch, kH, kW)` kernel; `in_ch` is inferred from camera mode.

## Parameters

| Parameter | Description |
|---|---|
| `n_filters` | Number of filters (= output neurons with global pool); default 1 |
| `kernel_size` | Square kernel side kH = kW; default 3 |
| `stride` | Convolution stride; default 1 |
| `padding` | `same` (preserve H×W) or `valid` (no padding); default `same` |
| `pool` | `global_avg`, `global_max`, or `none`; default `global_avg` |
| `activation` | Nonlinearity on feature maps before pooling; default `relu` |
| `tau_rise` | Leaky dynamics rise τ on pooled output (0 = off); default 0.0 |
| `tau_decay` | Leaky dynamics decay τ; defaults to `tau_rise` |
| `tau_a` | Adaptation time constant (0 = off); default 0.0 |
| `beta` | Adaptation strength (0 = off); default 0.0 |
| `bias` | Constant added to each pooled output; default 0.0 |
| `scale` | Output multiplier; default 1.0 |
| `lateralized` | Create mirrored _L / _R pair for split-camera input; default False |

## Forward pass

$$M = f(\text{conv2d}(I,\, W)), \quad \text{pooled} = \text{pool}(M) + b$$

**Optional leaky dynamics** (when `tau_rise > 0`):

$$\frac{dx}{dt} = \frac{\text{pooled} - x}{\tau}, \quad \text{output} = x \times \text{scale}$$

**Optional adaptation** (when `beta > 0` and `tau_a > 0`):

$$u_{eff} = \text{pooled} - \beta \, a, \quad \frac{da}{dt} = \frac{\text{output} - a}{\tau_a}$$

## Output shape

| Pool mode | Shape |
|---|---|
| `global_avg` / `global_max` | `(n_filters,)` |
| `none` | `(n_filters, H_out, W_out)` |

!!! tip
    Use `padding='valid'` with zero-sum kernels to avoid edge artifacts.

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — `(name, scale, site)` triples: `site="pre"` / `"post"` / `"none"`.
