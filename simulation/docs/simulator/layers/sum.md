# SumLayer

No dynamics, no memory. Output tracks input in the **same step**.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of neurons; default 2 |
| `activation` | `relu`, `sigmoid`, `tanh`, `linear`; default `relu` |
| `scale` | Output multiplier; default 1.0 |

## Output

$$\text{output} = f\!\left(\sum_k W_k \cdot \text{input}_k\right) \times s$$

- `activation='linear'` — pass-through (standard for motor output layer).
- `activation='relu'` — clips negative sums to zero.
- `scale = -1` — invert output without changing the weight matrix.

Also useful as a mixing node before a recurrent layer. Use **MotorLayer** when you need robot actuation.

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — `(name, scale, site)` triples:
  - `site="pre"` multiplies the input sum; `site="post"` multiplies the output.
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
