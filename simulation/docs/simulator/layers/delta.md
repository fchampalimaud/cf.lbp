# DeltaLayer

Rescorla-Wagner / delta-rule critic. δ = r − V with asymmetric learning rates. Biologically matches the asymmetry between LTP (fast acquisition) and LTD (slow extinction).

The slow negative update (`alpha_neg`) gives the animal time to reach the reward before the association erodes. If reward is genuinely absent across many exposures the small negative updates accumulate and eventually dissociate cue from reward — matching the behavioural extinction timescale.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of output neurons (parallel critics); default 1 |
| `alpha_pos` | Learning rate for δ ≥ 0 (acquisition); default 0.05 |
| `alpha_neg` | Learning rate for δ < 0 (extinction); default 0.005 |
| `reward_modulator` | Neuromodulator name carrying reward r; default `"dopamine"` |

## Learning rule

$$V = \sum_i W_i \, s_i, \quad \delta = r - V, \quad \Delta W_i = \alpha_{\text{eff}} \, \delta \otimes s_{i,\text{prev}}$$

**Output:** V(s) ∈ ℝⁿ — reward prediction per neuron.

!!! note "Reset behaviour"
    Same as TDLayer — episodic state cleared, weights survive.

## Reference

Rescorla, R. A. & Wagner, A. R. (1972). A theory of Pavlovian conditioning. In *Classical Conditioning II*. Appleton-Century-Crofts.
