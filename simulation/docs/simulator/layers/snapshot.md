# SnapshotLayer

One-shot vector-memory neuron (Le Moël et al. 2019). On a reward trigger, its **outgoing** connection weights are set directly to (minus) a named source layer's current output — not nudged gradually like `TDLayer`/`DeltaLayer`/`ThreeFactorLayer`. Recall is then just this neuron's own output multiplying those frozen weights back out — a snapshot, held exactly until the next trigger, regardless of how much the source drifts afterward.

## Two kinds of incoming connection

- **Teach connection** — `src == teach_source`. A real, ordinary connection (not a neuromodulator-style broadcast — the dependency is structured and per-channel, e.g. one source column informing one output synapse, not a diffuse signal any receptor could pick up). Its raw value is read for the write; it does **not** contribute to this neuron's own output.
- **Gate connection(s)** — everything else. Drives this neuron's own output normally, exactly like any other layer's inputs.

**Outgoing connections** (this layer → wherever the memory is expressed) are not learned via a Hebbian rule — their weights *are* the memory itself, directly overwritten on trigger. This is the one layer type in the `LearningLayerBase` family whose plasticity lives on its outgoing edges, not its incoming ones.

## Parameters

| Parameter | Description |
|---|---|
| `n` | Number of independent memory units; default 1. At `n=1` this matches Le Moël's "single vector-memory neuron" exactly. `n>1` is accepted as a structural parameter, but every unit currently writes together on any trigger — no independent per-slot gating yet. |
| `teach_source` | Name of the layer whose current output is copied (negated) into every outgoing connection's weights on trigger. |
| `tau_rise` / `tau_decay` | Leaky dynamics on this neuron's own output (from gate connections only). |
| `activation`, `bias`, `scale` | Standard output shaping. |
| `weight_decay` / `w_min` / `w_max` | Applied to the *outgoing* (stored memory) weights every tick, not to any incoming connection — passive forgetting and bounds on the stored content. |
| `competition` / `k` | Lateral competition over this neuron's own output; meaningful once `n>1` recall is designed, a no-op at `n=1`. |

## Write rule

Le Moël Eq. 14, on any tick `self._reward` is nonzero:

$$W_{\text{outgoing}} \leftarrow -\,\text{teach\_source.output}$$

A hard overwrite, not `ΔW = α · δ · s_prev` — old content is fully discarded each time, matching the paper's description exactly rather than converging toward it gradually.

**Output (recall):** `output = activation(gate_contribution) · scale · mask` — computed only from non-teach incoming connections.

## Wiring

1. Connect the layer to snapshot (e.g. a path-integration accumulator) → this layer, with a real weight matrix (e.g. identity/one-to-one if preserving per-channel structure matters, as it typically does).
2. Set `teach_source` to that layer's name.
3. Connect a gate/recall driver (e.g. a `ConstantLayer`, or a context signal) → this layer — its value becomes this neuron's own output.
4. Connect this layer → wherever the memory should be expressed (its outgoing connections' weights will be overwritten on trigger — give them any placeholder initial `W` of the correct shape).
5. Declare a reward-carrying layer as a neuromodulator transmitter, and add a modulator row on *this* layer with **Drives Plasticity** checked.

In the network visualizer, the teach connection is drawn as a solid green line, distinct from the amber styling of an ordinary connection into a learning layer — it isn't a trained weight or a signed synapse, just a fixed structural readout.

!!! note "Reset behaviour"
    ↺ Reset clears this neuron's own episodic output state, but — like every `LearningLayerBase` sibling — leaves connection weights (including the stored memory on outgoing connections) intact, since the whole point is that the memory survives across resets until next written.

## When to use

A single, hard-overwritten spatial or feature memory that must stay frozen against a live, continuously-changing source until explicitly rewritten — e.g. an insect path-integration "vector memory" neuron. For `n>1` independent memory slots (e.g. one per remembered location, as in trapline foraging), the recall side isn't implemented yet: every unit currently reads the same trigger and writes together.

## References

- Le Moël, F., Stone, T., Lihoreau, M., Wystrach, A. & Webb, B. (2019). The Central Complex as a Potential Substrate for Vector Based Navigation. *Frontiers in Psychology*, 10:690.
- Goulard, R., Heinze, S. & Webb, B. (2023). Emergent spatial goals in an integrative model of the insect central complex. *PLOS Computational Biology*, 19(12): e1011480.
