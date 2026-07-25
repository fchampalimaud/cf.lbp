# PulseLayer

Models calcium-like sustained (working-memory) activity. A fast membrane variable triggers a slow plateau that holds the output after the input is removed.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of neurons; default 2 |
| `tau_rise` | Fast membrane rise time constant (s); default 0.05 |
| `tau_decay` | Fast membrane decay time constant (s); defaults to `tau_rise` |
| `tau_hold` | Plateau charging/draining time constant (s); default 2.0 |
| `theta` | Threshold for charging plateau; default 0.0 |
| `w_s` | Gain of plateau variable on output; default 1.0 |
| `drain` | Rate at which sustained inhibition erodes plateau (0 = pure latch); default 1.0 |
| `bias` | Constant added to input; default 0.0 |
| `activation` | Output nonlinearity; default `relu` |
| `scale` | Output multiplier; default 1.0 |

## Dynamics

**Fast membrane** (u_in = Σ inputs + b):

$$\frac{du}{dt} = \frac{u_{in} - u}{\tau}, \quad \tau = \begin{cases}\tau_{rise} & u_{in}>u \\ \tau_{decay} & u_{in}\leq u\end{cases}$$

**Slow plateau** (charges while u > θ):

$$\frac{ds}{dt} = \frac{\text{relu}(u - \theta) - s}{\tau_{hold}}$$

**Output:**

$$\text{output} = f(u + w_s \cdot s) \times \text{scale}$$

While u > θ, s charges slowly. When input drops, u decays but s holds the plateau for ≈ `tau_hold` seconds.

| Setting | Behaviour |
|---|---|
| `drain = 0` | Pure latch — resumes after transient inhibition |
| `drain > 0` | Permanently collapsed by sustained inhibition |

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — `(name, scale, site)` triples: `site="pre"` / `"post"` / `"none"`.
