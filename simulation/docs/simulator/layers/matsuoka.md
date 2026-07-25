# MatsuokaLayer

!!! warning "Deprecated — use AdaptiveLayer"
    `MatsuokaLayer` is a thin wrapper around `AdaptiveLayer` with `n=2` fixed. Use `AdaptiveLayer` for all new networks.

## Parameters

| Parameter | Description |
|---|---|
| `tau_rise` | Membrane time constant (s); default 0.3 |
| `tau_a` | Adaptation time constant (s); default 1.2 |
| `beta` | Adaptation strength; default 2.5 |
| `w` | Mutual inhibition weight; default 2.5 |
| `bias` | Tonic drive; default 0.0 |

## Dynamics

$$\tau_M \frac{dx}{dt} = -x + u - \beta a - w \cdot o_{other}$$

$$\tau_A \frac{da}{dt} = -a + \text{relu}(x), \quad \text{output} = \text{relu}(x)$$

$$\text{Period} \approx 2\,\tau_A$$

## Oscillation conditions

1. `tau_a > tau_rise` — adaptation slower than membrane (required)
2. `w > 1` — mutual inhibition strong enough
3. `beta > 0` — adaptation must engage
4. `bias > 0` — tonic drive needed

If neurons lock (both fire / both silent): increase `w` or `bias`.
