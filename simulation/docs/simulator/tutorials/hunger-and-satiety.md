# Hunger Tunes the Senses — Neuromodulation

A hungry animal and a full one can have exactly the same eyes, nose and wiring, and still behave very differently when they smell food. The difference is not in *where* signals go but in *how strongly* they get through. Internal state turns the volume of the senses up or down.

In this tutorial you will build a vehicle that lives in a round arena with two food patches. It circles the arena, eats at one patch, and then **ignores the second patch** it meets, because a hormone released by its gut is turning down its sense of smell. By the time it comes round to food again, the hormone has worn off. The robot is hungry again and eats. No wire is added or removed between those two moments. Only the gain of the food sensor changes.

**What you will build**

| Step | What happens |
|------|--------------|
| Set up the arena | A round arena with two food patches (gradient `A`) |
| Build an always-hungry vehicle | Constant drive, wall avoidance and food approach, the same wiring as Vehicle III |
| Watch it overeat | It stops at *every* patch it meets |
| Give it a gut | An `InteroceptiveSensor` that fills up while the robot eats and releases **insulin** |
| Put insulin receptors on the nose | The food sensor's gain becomes `1 − insulin` |
| Watch it skip a meal | Fed at one patch, it passes the next one by and eats again once it is hungry |

The finished vehicle ships as the built-in session **D06 - FeedingSession** (network **D06 - FeedingBrain**). Load it at any point to compare with what you built.

---

## Background — satiety turns down the gain at the very first synapse

Neuromodulators such as hormones and neuropeptides are not wired point-to-point like ordinary synapses. They are released into the blood or into tissue and reach every cell that carries the matching **receptor**. Each of those cells changes how it responds. One slow signal can retune a whole pathway at once.

Two studies show this happening at the **first synapse of the olfactory system**, where receptor neurons in the nose hand their signal to the brain:

- **Fruit flies (Root et al., 2011).** When a fly is fed, circulating insulin is high. In the olfactory receptor neurons that detect vinegar-like food odours, insulin signalling keeps the receptor for the neuropeptide sNPF switched down. When the fly starves, insulin drops, those neurons make more sNPF receptor, and their output synapses are **facilitated**: the same odour drives a larger response in the brain. A single type of receptor neuron (the one projecting to glomerulus DM1) was both necessary and sufficient for the starvation-driven improvement in food search. Making just those neurons more excitable made *fed* flies search like starved ones.
- **Mice (Oncul et al., 2026, preprint).** In the mouse olfactory bulb, insulin blocks a potassium channel (Kv1.3) in periglomerular cells, the inhibitory neurons that wrap around the receptor-neuron terminals. These cells become more excitable and inhibit the incoming odour signal more strongly, so **less sensory input reaches the brain**. Fasted mice given glucose (which raises insulin) took about twice as long to find buried food (128 s vs 67 s). They also dug much further from it (9.0 cm vs 1.8 cm from the pellet).

The two species use different molecules but share one principle. **The metabolic state sets the gain of the food-odour channel, right where it enters the brain, and that gain shapes which way the animal steers.** When hungry, the gain is high and food odours dominate. When full, the gain is low and the same odours barely register.

This is not a quirk of two species. Sengupta (2013), in a review aptly titled *The belly rules the nose*, surveys work in *C. elegans* and *Drosophila* showing that feeding state routinely gates both smell and taste at the peripheral sensory neurons, through neuromodulatory signals that translate nutritional state into changes in how those neurons respond.

That is exactly what you will build. Here is how the parts map onto the biology:

| In the papers | In the simulator |
|---|---|
| Olfactory receptor neurons for a food odour | `food`, a `GradientSensor` tuned to gradient `A` |
| Metabolic state (gut content, blood glucose) | `gut`, an `InteroceptiveSensor` that fills while the mouth is on food |
| Insulin released into the circulation | `gut` set as a **transmitter** of `insulin` on the neuromodulator bus |
| Insulin acting on the first olfactory synapse | An `insulin` **receptor** on `food`'s neuron stage that scales its output |

---

## Step 1 — Set up the arena

1. In the **Physics** tab, set **arena_scale** to `4` so the arena is big enough to circle. 
2. In the **World** tab, place two gradient-`A` patches (the red **A** button, then drag from the centre outwards). Put one at around North and the other around East, making sure the patches are far enough from the wall to observe a movement from the robot towards them but close enough to allow the robot to sense them with one of its antenae. These are the two meals.
3. Leave the robot near the middle, slightly left of centre.

---

## Step 2 — Build a wall-following robot

Create a new network in the **Brain** tab (**Network** mode, **+**, call it `Feeding`) and open the visualizer. You'll add three things the robot already knows from the earlier tutorials.

**A reason to move.** Drag a **ConstantLayer** onto the canvas, name it `drive`, `n = 2`, `value = 50`. Wire `drive → motor` with the identity matrix:

|  | `drive_0` | `drive_1` |
|--|--|--|
| `motor_0` | **1** | 0 |
| `motor_1` | 0 | **1** |

**A way to stay off the walls.** Drag a **DistanceSensor**, name it `walls`, `n = 2`, `angle_spread_deg = 60`, `max_range = 1.0`, `scale = 50`. Wire `walls → motor`:

|  | `walls_0` (left) | `walls_1` (right) |
|--|--|--|
| `motor_0` (left wheel) | **+1** | **−1** |
| `motor_1` (right wheel) | **−1** | **+1** |

A wall close on the left speeds up the left wheel and slows the right one, so the robot turns right, away from the wall. In a round arena this keeps the robot sliding along the boundary. Try it!

**Save.**

---

## Step 3 — Build an always-hungry vehicle

**A nose for food.** Drag a **GradientSensor**, name it `food`, `n = 2`, `gradient = A`, `angle_spread_deg = 30`, `scale = 30`. Wire `food → motor` **same-side inhibitory**, the Vehicle III approach wiring:

|  | `food_0` (left) | `food_1` (right) |
|--|--|--|
| `motor_0` (left wheel) | **−1** | 0 |
| `motor_1` (right wheel) | 0 | **−1** |

More food smell on the right slows the right wheel against the constant drive, so the robot curves toward the food (see [Braitenberg Vehicle III](second-braitenberg.md)).

**Save.**

Press **▶ Run**. The robot circles the arena. Each time it passes near a patch, the food smell pulls it in, and it slows and crawls through the patch. Then it continues along the wall to the next patch and does exactly the same thing. **It stops to eat at every patch it meets, every lap, forever.**

This vehicle has no internal state. Its response to food depends only on what it smells right now, never on what it has already eaten. A real animal that did this would spend its whole life at the food it last found.

---

## Step 4 — Give it a gut that releases insulin

1. Drag an **InteroceptiveSensor** onto the canvas and name it `gut`. On the **Sensor** tab set:
    - `gradient` → `A`, so it "tastes" the same food the nose smells
    - `scale` → `1`, `max_val` → `1`, `start_val` → `0` (starts with an empty stomach)
    - `tau_rise` → `1.0`, so it fills relatively fast when the mouth is on food
    - `tau_decay` → `30`, so it empties over about 30 s once the robot leaves
2. Switch to the **Neuron** tab and set:
    - **neuromodulator transmitter** → `insulin`
    - **transmitter color** → `#FF0000`
3. Confirm and **Save**.

Don't draw any connection from `gut`. A transmitter doesn't need one: every tick, `gut` publishes its output as the value of `insulin` on the network's neuromodulator bus, and anything that has an insulin receptor reads it. That's the point of a hormone.

`gut` samples food at the robot's **mouth** (the front of the body). The deeper the robot goes into a patch, the higher the reading, up to `1` at the centre. See [InteroceptiveSensor](../sensors/interoceptive.md) for the full dynamics.

---

## Step 5 — Put insulin receptors on the nose

1. Double-click `food` and switch to its **Neuron** tab.
2. Under **Modulator receptors**, click **Add Receptor** and fill the row in:

    | Modulator | Scale | Site | Mode |
    |---|---|---|---|
    | `insulin` | **−1.0** | `post` | `absolute` |

3. Confirm and **Save**. Small red circles now appear in the `food` neurons signaling they are sensitive to the insuline neuromodulator you created.

Each tick, the receptor multiplies `food`'s output by a gain:

$$\text{gain} = 1 + \text{scale} \times \text{insulin} = 1 - \text{insulin}$$

| `insulin` (gut) | gain | What the food smell does |
|---|---|---|
| 0 (empty) | 1.0 | full strength, strong approach |
| 0.25 | 0.75 | slightly weaker pull |
| 0.5 | 0.5 | half strength, weak pull |
| 1 (full) | 0 | nothing, the robot is blind to food |

Insulin can only turn the food sense *down*, from 100 % when empty to 0 % when full. It never increases the signal, and it never flips its sign.

!!! note "Why `post`, and why −1?"
    `post` scales what the sensor *sends on*, which is the closest match to presynaptic gating of the receptor-neuron terminal. `food`'s neuron stage is pass-through (linear, no time constants), so `pre` would give the same result here. See [Neuromodulators — pre vs post](../neuromodulators.md#wiring-details-pre-vs-post).

    With `gut` capped at `max_val = 1`, a scale of `−1` takes the gain all the way to zero at full satiety and no further, matching the reduced gain reported in both papers. The gain is not clamped, so a scale beyond `−1` (e.g. `−2`) would push it *negative* above half-full. Same-side inhibition would turn into excitation, and food would start to repel the robot. To keep insulin purely inhibitory, keep `|scale| × max_val ≤ 1`.

---

## Step 6 — Run it: skipping a meal

Press **▶ Run**, and open the oscilloscope on `gut` and on `food_0` / `food_1`. You should see a cycle like this:

1. **Hungry.** `gut` is 0 and the gain is 1. The robot is pulled into the first patch it meets.
2. **Eating.** With its mouth in the patch, `gut` climbs over a second or so (`tau_rise = 1`). The gain falls with it, the food signal fades, and the pull that held the robot in the patch weakens. The constant drive and the wall avoidance take over again, and the robot drifts out of the patch and back onto its path along the wall.
3. **Full, passing food by.** The next patch comes up soon after, because the two patches are close together. `gut` has barely started to empty, so the gain is still low. The weak food signal is not enough to pull the robot off its path, and it skirts the patch and keeps going.
4. **Hungry again.** On the longer stretch of wall that follows, `gut` decays (`tau_decay = 30` s) and the gain climbs back towards 1. The next time the robot meets food, it turns in and eats. The cycle repeats every lap.

Now try the receptor scale at `0` (effectively no receptor), `−0.5` and `−1`:

| Receptor scale | Gain when full | Patch it reaches hungry | Patch it reaches just after eating |
|---|---|---|---|
| none (`0`) | 1.0 | eats | eats as well, so the robot stops at every patch |
| `−0.5` | 0.5 | eats | weaker pull, may still turn in |
| `−1` | 0 | eats | **skipped** |

Same sensors, same wiring, same world. A single receptor on the nose, fed by a signal from the gut, decides which food the robot eats.

!!! tip "Which patch gets skipped depends on timing"
    Nothing in the network says "skip this patch". After the robot leaves food, `gut` decays as `gut(t) ≈ gut₀ · e^(−t/30)`. If it leaves with `gut ≈ 0.8`, then 10 s later `gut ≈ 0.57` and the gain is only about 0.43. 60 s later `gut ≈ 0.11` and the gain is back to about 0.89. A patch reached soon after a meal is met with a weak nose, one reached much later with a strong one. Change the patch spacing, the starting position or `tau_decay`, and a different meal gets skipped, or none, or two.

---

## Challenges

### Challenge 1 — Hunger turns the gain *up* (the fly version)

In Root et al. (2011) the logic works the other way round. Starvation *increases* the gain of the food channel, through sNPF facilitation that insulin normally holds down. Build that version.

### Challenge 2 — Only the food channel is tuned

In the fly, only some odour channels were modulated by starvation, and only one (DM1) mattered for food search. Add a second patch type, gradient `B`, as a neutral odour with its own `GradientSensor` and **no** insulin receptor. Wire it for mild approach. Watch what happens to the robot's preference between `A` and `B` as `gut` fills and empties.

<!--
### Challenge 3 — Re-run the mouse experiment

Oncul et al. compared the same animals fasted and after a glucose injection, timing how long each took to find food. Do the same with [Headless Runs](../headless.md). Save two copies of your session, one with `gut`'s `start_val = 0` (fasted) and one with `start_val = 1` (sated). Run each over several `--seed` values and measure, from the CSV, the time until the robot first enters a patch. Is the sated robot slower? How does the answer change with receptor scale `−0.5` vs `−1`?

---
-->

## References

- Root, C. M., Ko, K. I., Jafari, A., & Wang, J. W. (2011). Presynaptic facilitation by neuropeptide signaling mediates odor-driven food search. *Cell*, 145, 133–144. [doi:10.1016/j.cell.2011.02.008](https://doi.org/10.1016/j.cell.2011.02.008)
- Sengupta, P. (2013). The belly rules the nose: feeding state-dependent modulation of peripheral chemosensory responses. *Current Opinion in Neurobiology*, 23, 68–75. [doi:10.1016/j.conb.2012.08.001](https://doi.org/10.1016/j.conb.2012.08.001)
- Oncul, M., Stefens, C., Smith, E., Choudhuri, S., Filippi, B. M., & Johnston, J. (2026). Insulin controls olfactory gain at the first central synapse by regulating periglomerular neuron excitability. *bioRxiv* (preprint, not yet peer reviewed). [doi:10.64898/2026.07.22.740090](https://doi.org/10.64898/2026.07.22.740090)
