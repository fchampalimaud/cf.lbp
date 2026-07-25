# SineLayer

Autonomous sinusoidal oscillator. Ignores incoming connections. All `n` neurons share the same value. `t` resets to 0 on `reset()`.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of neurons; default 1 |
| `amplitude` | Peak amplitude; default 1.0 |
| `frequency` | Oscillation frequency in Hz; default 1.0 |
| `phase` | Initial phase offset in radians; default 0.0 |

## Output

$$\text{output} = A \sin(2\pi f\, t + \phi)$$

**Typical use:** clock signal, rhythmic drive, CPG rhythm source. Connect to a `LeakyLayer` to smooth the waveform, or directly to a motor layer for open-loop sinusoidal motion.

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — only `site="post"` has effect; `site="pre"` does nothing (layer ignores incoming connections).
