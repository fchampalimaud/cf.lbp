# TDLayer

TD(0) reward-prediction critic. Implements the Schultz/Dayan/Montague (1997) dopamine model: a linear critic whose **connection weights** W are updated each tick by the temporal-difference error δ.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of output neurons (parallel critics); default 1 |
| `alpha_pos` | Learning rate for positive δ (acquisition); default 0.01 |
| `alpha_neg` | Learning rate for negative δ (extinction); default = alpha_pos |
| `gamma` | Discount factor (0–1); default 0.99 |

## Learning rule

$$V = \sum_i W_i \, s_i, \quad \delta = r + \gamma V - V_{\text{prev}}, \quad \Delta W_i = \alpha_{\text{eff}} \, \delta \otimes s_{i,\text{prev}}$$

The update credits the previous tick's input `s_prev`, because δ compares this tick's prediction with the previous one.

Like every layer, V goes through the shared pipeline first — bias, noise (`noise_std` / `noise_tau`, see [Noise](leaky.md#noise)), output mode, leaky filter — so noise on V also reaches the weight update.

Learning rates are **per tick at the default time step dt = 0.01 s** and scale with dt, so learning per second is the same at any dt (see [Noise](leaky.md#noise) for why the default dt is the reference).

**Output:** V(s) ∈ ℝⁿ — predicted future reward per neuron.

## Wiring

1. Connect any sensory/feature layer → TDLayer. Initialise the connection W to zeros.
2. Declare the reward-carrying layer as a neuromodulator transmitter (e.g. `"dopamine"`).
3. In this layer's modulator receptor table, add a row for that name and check **Drives Plasticity** — the layer reads r from it each tick.
4. Optionally wire TDLayer output → motor layers for direct actor behaviour.

!!! note "Reset behaviour"
    ↺ Reset clears V_prev and src_prev (episodic state) but leaves connection weights intact. Use **↺ Reset Weights** in Brain Parameters to zero all incoming connection weights.

## Reference

Schultz, W., Dayan, P. & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*, 275(5306), 1593–1599.
