# Reichardt2dLayer

Elementary motion detector over camera input. Implements the balanced Reichardt (EM detector) correlator.

For each direction and pixel `(i, j)`:

$$R(i,j) = I_{del}(i,j) \cdot I_{cur}(i{+}\Delta y,\,j{+}\Delta x) - I_{del}(i{+}\Delta y,\,j{+}\Delta x) \cdot I_{cur}(i,j)$$

where $I_{del}$ is an exponential low-pass of the input with time constant `tau_delay`. Positive output → motion in the preferred direction.

## Parameters

| Parameter | Description |
|---|---|
| `n_dirs` | Direction set: 1 (right), 2 (left/right), 4 (LRUD), 8 (LRUD+diagonals); default 4 |
| `offset` | Spatial pixel offset for the correlation; default 1 |
| `tau_delay` | Internal Reichardt delay time constant (s); default 0.1 |
| `pool` | `global_avg`, `global_max`, or `none`; default `global_avg` |
| `activation` | Nonlinearity applied to correlation maps before pooling; default `relu` |
| `tau_rise` / `tau_decay` / `tau_a` / `beta` / `bias` / `scale` / `lateralized` | Same as Conv2dLayer |

## Forward pass

1. Low-pass filter each pixel with `tau_delay` → $I_{del}$
2. For each direction: compute balanced Reichardt map $R$
3. Apply activation → pool → add bias → optional leaky dynamics → × scale

## Output shape

| Pool mode | Shape |
|---|---|
| `global_avg` / `global_max` | `(n_dirs,)` scalars |
| `none` | `(H × W,)` flat spatial map (requires `n_dirs = 1`) |

## Tuning tips

- Set `tau_delay ≈ 1 / fps` for one-frame delay (optimal for frame-rate motion).
- `offset = 1` detects smallest shifts; increase for slower / large-scale motion.
- Use `activation='linear'` to preserve sign — negative means opposite-direction motion.
- For push-pull motor drive: `n_dirs=2`, wire outputs [0] and [1] with opposite signs.

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — `(name, scale, site)` triples: `site="pre"` / `"post"` / `"none"`.
