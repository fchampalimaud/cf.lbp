# Reichardt2dLayer

Elementary motion detector over camera input. Implements the balanced Reichardt (EM detector) correlator.

For each direction and pixel `(i, j)`:

$$R(i,j) = I_{del}(i,j) \cdot I_{cur}(i{+}\Delta y,\,j{+}\Delta x) - I_{del}(i{+}\Delta y,\,j{+}\Delta x) \cdot I_{cur}(i,j)$$

where $I_{del}$ is an exponential low-pass of the input with time constant `tau_delay`. Positive output → motion in the preferred direction.

## Parameters

| Parameter | Description |
|---|---|
| `n_directions` | Number of evenly-spaced preferred directions; default 4 |
| `init_direction` | Angle of direction 0, degrees (0 = right, increasing counter-clockwise); others follow `init_direction + k * 360 / n_directions`; default 0.0 |
| `offset` | Spatial pixel offset for the correlation; default 1 |
| `tau_delay` | Internal Reichardt delay time constant (s); default 0.1 |
| `pool` | `global_avg`, `global_max`, or `none`; default `global_avg` |
| `activation` | Nonlinearity applied **after** pooling for `global_avg`/`global_max` (see below), or per-pixel for `pool='none'`; default `relu` |
| `tau_rise` / `tau_decay` / `tau_a` / `beta` / `bias` / `scale` / `lateralized` | Same as Conv2dLayer |

Each direction's pixel offset is sampled with bilinear interpolation, so `n_directions`/`init_direction` can be any values — directions no longer need to land on the integer pixel grid (multiples of 90°/45°).

## Forward pass

1. Low-pass filter each pixel with `tau_delay` → $I_{del}$
2. For each direction: compute balanced Reichardt map $R$
3. `global_avg`/`global_max`: pool the raw signed $R$ map first, **then** apply activation to that
   single scalar. `pool='none'`: activation is applied per pixel instead (nothing to pool yet).
4. Add bias → optional leaky dynamics → × scale

**Why pool before activation:** $R$ is signed and spatially balanced — positive pixels vote for
this direction, negative pixels vote for the opposite one, and are meant to cancel out when
averaged into a clean net direction signal. Clipping every pixel with `activation` *before*
averaging (the order Conv2dLayer uses, correct for a feature detector) destroys that
cancellation — two exactly opposite directions end up responding almost identically on a
textured image instead of one being clearly suppressed. `pool='global_max'` is unaffected by
the ordering either way, since every built-in activation is monotonic
(`max(activation(x)) == activation(max(x))`).

## Output shape

| Pool mode | Shape |
|---|---|
| `global_avg` / `global_max` | `(n_directions,)` scalars |
| `none` | `(n_directions × H × W,)` flat motion-field vector, displayed as RGB via `disp_r`/`disp_g`/`disp_b` |

## Tuning tips

- Set `tau_delay ≈ 1 / fps` for one-frame delay (optimal for frame-rate motion).
- `offset = 1` detects smallest shifts; increase for slower / large-scale motion.
- Use `activation='linear'` to preserve sign — negative means opposite-direction motion.
- For push-pull motor drive: `n_directions=2`, wire outputs [0] and [1] with opposite signs.

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — `(name, scale, site)` triples: `site="pre"` / `"post"` / `"none"`.
