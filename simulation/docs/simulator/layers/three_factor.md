# ThreeFactorLayer

Reward-gated Hebbian learning. Weight changes require the simultaneous coincidence of presynaptic activity, postsynaptic activity, *and* a neuromodulatory reward signal. When reward is absent the weights are frozen; only passive decay (if enabled) erodes them.

## Learning rule

$$\Delta W_{ji} = \alpha_{\text{eff}} \cdot r \cdot V_j \cdot s_{i,\text{prev}}$$

| Term | Role |
|---|---|
| `r` | Reward signal from the designated neuromodulator — gates all plasticity |
| `V_j` | Postsynaptic activity of neuron j — selects *which* neuron learns |
| `s_prev` | Presynaptic activity on the previous tick — selects *which* inputs |

## Passive forgetting

$$W \leftarrow W \cdot (1 - \text{decay} \cdot dt)$$

Applied every tick regardless of r. Equilibrium weight reflects the balance between acquisition rate and decay — infrequently rewarded associations fade naturally.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of output neurons; default 1 |
| `alpha_pos` | Learning rate when r·V ≥ 0; default 0.01 |
| `alpha_neg` | Learning rate when r·V < 0 (punishment); default = alpha_pos |
| `reward_modulator` | Neuromodulator name carrying r; default `"dopamine"` |
| `weight_decay` | Passive decay rate (s⁻¹); default 0.0 |
| `w_min` / `w_max` | Synaptic bounds (blank = unbounded) |
| `competition` | Lateral competition: `none` / `softmax` / `wta`; default `none` |
| `k` | Number of winners for `wta`; default 1 |

## Wiring

1. Connect any sensory/feature layer → ThreeFactorLayer (initialise W to zeros).
2. Declare the reward-carrying layer as a neuromodulator transmitter (e.g. `"dopamine"`).
3. Set `reward_modulator` to that name.
4. Optionally wire output → motor layers for direct approach drive.

!!! note "Reset behaviour"
    ↺ Reset clears episodic state but leaves weights intact. Use **↺ Reset Weights** to zero all incoming connection weights.

## References

- Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cerebral Cortex*, 17(10), 2443–2452.
- Frémaux, N. & Gerstner, W. (2016). Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules. *Frontiers in Neural Circuits*, 9, 85.
