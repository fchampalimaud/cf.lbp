# LeakyLayer

First-order low-pass filter neurons. Output approaches input with time constant `tau_rise` and decays with `tau_decay`.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of neurons |
| `tau_rise` | Rise time constant (s); default 0.1 |
| `tau_decay` | Decay time constant (s); defaults to `tau_rise` |
| `bias` | Constant added to each input sum; default 0.0 |
| `activation` | `relu`, `sigmoid`, `tanh`, `linear`; default `relu` |
| `scale` | Output multiplier; default 1.0 |
| `derivative` | `True` → outputs x − u (detects input drops); default False |
| `noise_std` | Gaussian noise std on u each step; default 0.0 |
| `noise_tau` | Ornstein-Uhlenbeck time constant (0 = white noise); default 0.0 |

## Dynamics

$$\frac{dx}{dt} = \frac{u - x}{\tau}, \quad \tau = \begin{cases} \tau_{rise} & u > x \\ \tau_{decay} & u \leq x \end{cases}$$

$$\text{output} = f(x) \times s$$

| Setting | Behaviour |
|---|---|
| `tau_rise = tau_decay` | Symmetric smoothing |
| `tau_rise < tau_decay` | Fast rise, slow decay — memory trace |
| `tau_rise > tau_decay` | Slow rise, fast decay — transient detector |

## Derivative mode

`derivative=True`:

$$\text{output} = f(x - u) \times s$$

x lags u, so x − u > 0 only when u has recently *fallen* — fires on input **decrease**. Set `scale = -1` to detect *increases* instead.

## Neuromodulation

- `neuromodulator_transmitter` / `neuromodulator_color` — publish mean output to the bus.
- `modulators` — `(name, scale, site)` triples:
  - `site="pre"` multiplies the input sum; `site="post"` multiplies the output.
  - Feedback paths use the previous tick's value (one-step delay).
