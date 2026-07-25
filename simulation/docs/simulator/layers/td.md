# TDLayer

TD(0) reward-prediction critic. Implements the Schultz/Dayan/Montague (1997) dopamine model: a linear critic whose **connection weights** W are updated each tick by the temporal-difference error δ.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of output neurons (parallel critics); default 1 |
| `alpha_pos` | Learning rate for positive δ (acquisition); default 0.01 |
| `alpha_neg` | Learning rate for negative δ (extinction); default = alpha_pos |
| `gamma` | Discount factor (0–1); default 0.99 |
| `reward_modulator` | Neuromodulator name carrying the reward signal r; default `"dopamine"` |

## Learning rule

$$V = \sum_i W_i \, s_i, \quad \delta = r + \gamma V - V_{\text{prev}}, \quad \Delta W_i = \alpha_{\text{eff}} \, \delta \otimes s_{i,\text{prev}}$$

**Output:** V(s) ∈ ℝⁿ — predicted future reward per neuron.

## Wiring

1. Connect any sensory/feature layer → TDLayer. Initialise the connection W to zeros.
2. Declare the reward-carrying layer as a neuromodulator transmitter (e.g. `"dopamine"`).
3. Set `reward_modulator` to that name.
4. Optionally wire TDLayer output → motor layers for direct actor behaviour.

!!! note "Reset behaviour"
    ↺ Reset clears V_prev and src_prev (episodic state) but leaves connection weights intact. Use **↺ Reset Weights** in Brain Parameters to zero all incoming connection weights.

## Reference

Schultz, W., Dayan, P. & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*, 275(5306), 1593–1599.
