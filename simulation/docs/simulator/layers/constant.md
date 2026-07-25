# ConstantLayer

Outputs a fixed value every step, ignoring incoming connections.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of neurons; default 2 |
| `value` | Constant output — scalar or list of length n; default 1.0 |
| `noise_std` | Gaussian noise std added each step; default 0.0 |

## Output

$$\text{output}_i = v + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0,\,\sigma)$$

`noise_std = 0` — pure constant. `value` can be a per-neuron list.

**Typical use:** tonic drive for downstream layers. Keep `value` small (0.1–1.0) relative to downstream thresholds. For ring attractors: use a **one-to-one** connection so each ring neuron gets exactly `value`, not `value × n`.

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — only `site="post"` has effect (multiplies output); `site="pre"` does nothing because this layer ignores incoming connections.
