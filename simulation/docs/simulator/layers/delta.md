# DeltaLayer

Rescorla-Wagner / delta-rule critic. δ = r − V with asymmetric learning rates. Biologically matches the asymmetry between LTP (fast acquisition) and LTD (slow extinction).

The slow negative update (`alpha_neg`) gives the animal time to reach the reward before the association erodes. If reward is genuinely absent across many exposures the small negative updates accumulate and eventually dissociate cue from reward — matching the behavioural extinction timescale.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of output neurons (parallel critics); default 1 |
| `alpha_pos` | Learning rate for δ ≥ 0 (acquisition); default 0.05 |
| `alpha_neg` | Learning rate for δ < 0 (extinction); default 0.005 |

## Learning rule

$$V = \sum_i W_i \, s_i, \quad \delta = r - V, \quad \Delta W_i = \alpha_{\text{eff}} \, \delta \otimes s_i$$

The update credits this tick's input `s_i` — the one that produced `V`.

Like every layer, V goes through the shared pipeline first — bias, noise (`noise_std` / `noise_tau`, see [Noise](leaky.md#noise)), output mode, leaky filter — so noise on V also reaches the weight update.

Learning rates are **per tick at the default time step dt = 0.01 s** and scale with dt, so learning per second is the same at any dt (see [Noise](leaky.md#noise) for why the default dt is the reference).

**Output:** V(s) ∈ ℝⁿ — reward prediction per neuron.

!!! note "Reset behaviour"
    Same as TDLayer — episodic state cleared, weights survive.

## Reference

Rescorla, R. A. & Wagner, A. R. (1972). A theory of Pavlovian conditioning. In *Classical Conditioning II*. Appleton-Century-Crofts.
