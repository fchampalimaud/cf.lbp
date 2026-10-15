# Surviving in Buridan's World

**Buridan's ass** is a centuries-old thought experiment: a donkey stands exactly midway between two identical, equally appetizing bales of hay. With no reason to prefer either one, the story goes, it can never decide — and starves to death standing still.

This tutorial picks up from **[Braitenberg Vehicle III](second-braitenberg.md)** and puts its "love" circuit in exactly that bind: two identical gradient patches, placed as mirror images of each other. You'll see the direct wiring from Vehicle III fail the test outright — it has no way to break a perfect tie. Then you'll build two more vehicles, **Rivalry** and **Discriminator**, each adding one more idea on top of the last to decide with less and less sensor noise to go on.

**What you will build**

| Step | What happens |
|------|--------------|
| Add a second, mirrored patch | Two identical `A` patches, equidistant from the robot — the two bales of hay |
| Widen the steering wiring | Builds **Waverer**: a direct sensor→motor connection that computes a pure left-vs-right difference |
| Watch the tie | With both patches identical, the difference sits at zero and the robot drives straight between them |
| Insert a competing pair | Builds **Rivalry**: an `AdaptiveLayer` wired so its two neurons inhibit each other |
| Watch it decide | The competition amplifies the tiniest random imbalance into a committed turn toward one patch |
| Let the winner reach back | Builds **Discriminator**: two new connections that let the winning side suppress the losing eye's own signal, reinforcing the decision from two directions at once |

---

## Background — a tie needs something to break it

Every circuit you've built so far is purely feedforward: a sensor reading flows straight through to a motor command, instantaneously and deterministically. That's fine as long as the world only ever gives the robot one thing to react to. Put two *identical* things in front of it, symmetric left and right, and a feedforward circuit has nowhere to go — by construction, equal inputs produce equal (or exactly cancelling) outputs, forever.

Breaking a tie like that needs two ingredients: something that amplifies a small difference instead of just passing it through, and something to seed that difference in the first place. The amplifier is **mutual inhibition** — two neurons that each suppress the other — which turns any momentary edge, however small, into a growing one. The seed is ordinary sensor noise, which this tutorial deliberately leaves switched on.

Here's the shape of the whole tutorial before you build any of it. **Waverer** reuses Vehicle III's wiring almost unchanged and has no amplifier at all — no amount of sensor noise ever helps it, it just wavers down the middle forever. **Rivalry** adds a pair of mutually-inhibiting neurons: a little sensor noise is now enough to untie the symmetry and force a choice. But turn that noise down and Rivalry takes longer and longer to decide — below some level, it won't commit within any reasonable time at all. That failure is what motivates **Discriminator**: the same competition, but wired so the winning side reaches back and suppresses the losing eye's own signal at its source, not just its rival neuron downstream. That extra feedback reinforces the decision from two directions at once, so it can still resolve a difference at noise levels where Rivalry stalls. Turn the noise down far enough, though, and even Discriminator eventually runs out of signal to work with.

---

## Step 1 — Reopen your Vehicle III network

Launch the simulator and reopen the network you built in Vehicle III, with `light` (a `GradientSensor`, `n=2`, `gradient='A'`) wired for **approach** — `cruise` (a `ConstantLayer`, value `60`) feeding `motor` directly, and `light` inhibiting `motor` same-side (Vehicle III, [Step 5](second-braitenberg.md#step-5-make-the-sensor-inhibitory)).

!!! note "Skipped Vehicle III?"
    Build that starting point first: a `GradientSensor` named `light` (`n=2`, `gradient='A'`, `scale=60`), a `ConstantLayer` named `cruise` (`n=2`, `value=60`) wired to `motor` with the identity matrix, and `light → motor` wired same-side with weight `−1` on the diagonal. See [Braitenberg Vehicle III](second-braitenberg.md) for the full build.

---

## Step 2 — Give it two identical choices

Right now there's a single `A` patch in the world. You're about to give the robot two, placed as exact mirror images of each other.

1. Go to the **World** tab. If you don't already have an `A` patch placed ahead of the robot, place one now (as in Vehicle II/III).
2. Hold **Ctrl** and drag the patch — this clones it, the same size and colour, following your cursor while the original stays put.
3. Hold **Shift** at the same time to snap both the original and the clone to the grid (set the grid spacing in the Physics tab if you want finer control). Drop the clone so it mirrors the original across the robot's forward axis — if the original sits ahead and to the left, drop the clone the same distance ahead and to the right.

You should now have two identical `A` patches, the same distance from the robot, one on each side of the path straight ahead of it — the two bales of hay, equally far and equally appealing.

!!! tip "Give it room"
    If the arena feels cramped with two patches in it, increase **arena_scale** in the Physics tab so the robot has more distance to react before it reaches either one.

---

## Step 3 — Widen the steering wiring

Vehicle III's same-side inhibitory wiring already produces an approach response, but on its own it leaves a loose end: as the robot nears a patch, *both* wheels can be slowed by their own-side sensor, so the robot can drift to a stop in the middle rather than just failing to turn. To make this demonstration purely about *which way to turn* — not also about whether to stop — add a crossed excitatory term on top of the existing same-side one.

1. In the network visualizer, right-click the `light → motor` connection and choose **Edit weight…**.
2. Switch to **Manual** and enter:

    |  | `light_0` (left) | `light_1` (right) |
    |--|--|--|
    | `motor_0` (left wheel) | **−1** | **+1** |
    | `motor_1` (right wheel) | **+1** | **−1** |

3. Confirm and **Save**.

Writing `Δ = light_0 − light_1` for the difference between the two eyes, this matrix reduces to `left wheel = 60 − Δ`, `right wheel = 60 + Δ`. The two correction terms are now exact opposites, so `left + right = 120` no matter what `Δ` is — the robot's overall speed never changes, only how that speed is split between the wheels. It still turns toward whichever side senses more signal, exactly as in Vehicle III, but the turn is now a pure, speed-preserving comparison between left and right.

This is **Waverer**: a plain comparator, no competition, no memory.

---

## Step 4 — Run it: the tie that never breaks

Press **▶ Run**. With two identical, symmetric patches, `light_0` and `light_1` read almost exactly the same value at every moment — `Δ` sits at (or flickers randomly around) zero. The robot drives essentially straight ahead, passing directly between the two patches without ever committing to either one.

!!! note "It still wobbles a little"
    `light`'s `noise_std` (`0.05`) means `Δ` never sits at *exactly* zero — each eye has its own small independent noise, so the robot wobbles slightly left and right as it drives. But the wobble has no memory: the next noisy sample is as likely to push it back the other way. Watch closely and you'll see it never accumulates into an actual turn. This is Buridan's ass, literally — stuck exactly between the two things it might otherwise want. No amount of noise fixes Waverer — see the table at the end of Step 8.

---

## Step 5 — Insert a competing pair

To actually break the tie, the circuit needs two things it doesn't have yet: a way to amplify a small difference instead of just reporting it, and persistence, so an early (even noise-driven) lead keeps growing instead of resetting every tick. A pair of mutually-inhibiting neurons gives you both.

1. Stop the simulation. Right-click the `light → motor` connection from Step 3 and delete it — the new circuit routes through a middle layer instead, with no direct shortcut.
2. Drag an **AdaptiveLayer** chip onto the canvas, between `light` and `motor`. Name it `choice`, set `n = 2`.
3. In `choice`'s settings, besides the defaults, set:
    - `tau_rise` → `0.1`, `tau_decay` → `0.1`
    - `w` → `1.0` — **this is the one that matters.** It's the only layer type with a built-in cross-inhibition term: each of `choice`'s two neurons subtracts the *other's* output from its own input.
    - `beta` → `0.0` — `AdaptiveLayer` defaults this to `1.0`, which turns on spike-frequency adaptation (the layer's namesake feature). You don't want that here: set it to `0` so the only thing `choice` does is compete, with none of the burst-and-fade behaviour adaptation would add on top.
4. Wire `light → choice`, same-side only:

    |  | `light_0` (left) | `light_1` (right) |
    |--|--|--|
    | `choice_0` | **+1** | 0 |
    | `choice_1` | 0 | **+1** |

5. Wire `choice → motor` — the *exact same matrix* as Vehicle III's original approach wiring, just one synapse further downstream:

    |  | `choice_0` | `choice_1` |
    |--|--|--|
    | `motor_0` (left wheel) | **−1** | 0 |
    | `motor_1` (right wheel) | 0 | **−1** |

6. **Save**.

`choice_0` is excited by the left eye and inhibited by `choice_1`'s output; `choice_1` is the mirror image. Whichever one edges ahead — even from noise alone — suppresses the other a little more than it's suppressed back, which lets it edge ahead a little further: a positive feedback loop. Because `choice`'s activation is `relu`, the losing neuron is floored at zero rather than going negative, so once it loses, it can't recover — the decision sticks. See [AdaptiveLayer](../layers/adaptive.md) for the full dynamics behind the `w` term.

This is **Rivalry**: recurrent mutual inhibition, winner-take-all.

---

## Step 6 — Run it: watch it decide

Press **▶ Run**. Within the first second or so, Rivalry commits to one side and curves firmly toward that patch instead of splitting the difference. Which side it picks is effectively random, seeded by whichever way the sensor noise happened to tip the balance first.

Open the oscilloscope on `choice_0` and `choice_1` while it runs: you'll see both rise together briefly, then rapidly diverge — one heading toward its ceiling, the other clamped at zero. Stop and run it a few more times from the same starting position; it won't always pick the same side.

---

## Step 7 — Let the winner silence the other eye

Rivalry has a weakness: `choice`'s competition only acts *downstream* of the two eyes. Whichever side is winning suppresses its rival neuron in `choice`, but it never touches the losing eye's own reading — `light` keeps reporting exactly what it senses regardless of how the competition is going. At low noise the competition has very little to work with, and nothing shortens that struggle.

This step closes the loop: it lets the winning side reach back and suppress the *losing eye's own output*, not just its rival neuron. No new node is needed — just two new connections on top of the circuit you already built in Step 5.

!!! note "Sensors can be connection targets too"
    Every sensor has its own **neuron stage** sitting after its raw reading — a small filter that the rest of the network actually reads from. You can draw a connection onto a sensor exactly as onto a layer: it lands in that neuron stage, summed with the sensor's own reading. See [Sensors — neuron stage](../sensors/index.md#neuron-stage) for the full picture. `light`'s neuron stage is pass-through by default, so you don't need to change any of its settings here — just draw the connection.

1. Wire `light → motor` directly, **crossed**, on top of the existing `cruise → motor` and `choice → motor` connections:

    |  | `light_0` (left) | `light_1` (right) |
    |--|--|--|
    | `motor_0` (left wheel) | 0 | **+1** |
    | `motor_1` (right wheel) | **+1** | 0 |

2. Wire `choice → light`, **crossed**, inhibitory — drag from `choice` onto `light` itself:

    |  | `choice_0` | `choice_1` |
    |--|--|--|
    | `light_0` | 0 | **−1** |
    | `light_1` | **−1** | 0 |

3. **Save**. Everything from Step 5 — `light → choice` (same-side), `choice → motor` (same-side), `cruise → motor` — stays exactly as it was; this step only adds to the circuit.

Trace what happens once `choice_0` (the left-eye competitor) starts winning: it suppresses `choice_1` as before (Rivalry's own competition), but now it also suppresses `light_1` — the right eye's own output — directly at the source. That shrinks the `+light_1` term feeding `motor_0`, so the left wheel loses a source of excitation just as it's being suppressed by `choice_0`. Meanwhile `choice_1`, now losing, barely suppresses `light_0` at all, so the left eye's full-strength reading keeps flowing into `motor_1` through the direct crossed connection, boosting the right wheel on top of `choice_1`'s fading suppression. Both wheels get pushed further apart than Rivalry's competition alone would manage — a second, independent feedback loop stacked on top of the first, reinforcing the same decision from two directions at once.

This is **Discriminator**.

---

## Step 8 — Run it: deciding with less to go on

Press **▶ Run**. Discriminator should still commit to one patch, same as Rivalry — but because the winning side also starves the losing eye's own signal, it should be able to do so at lower noise levels than Rivalry can. That's worth testing directly rather than taking on faith: the table below is where to record what you find.

### How much noise does each one need?

For a fixed `light.noise_std`, run a vehicle several times and see whether it reliably commits to one patch within a fixed window, or fails to (drives down the middle, or flip-flops without settling). Sweep `noise_std` downward to find roughly where that changes.

| Vehicle | Built in | Mechanism | Noise floor — lowest `light.noise_std` it can still decide at, in time |
|---|---|---|---|
| Waverer | Step 3 | None | *(never decides, at any noise level — see Step 4)* |
| Rivalry | Step 5 | Recurrent mutual inhibition | `TBD` |
| Discriminator | Step 7 | + winner suppresses the losing eye's own signal | `TBD` |

---

## Challenges

Every vehicle so far turns toward or away from something it can sense from a distance. Braitenberg vehicles are just as often defined by what they do once a stimulus can no longer be avoided — contact.

### Challenge 1 — Bump, back up, turn away

Using the same motifs as this tutorial — a constant baseline drive, same-side/crossed sensor wiring, and a dynamics layer with its own time constants — build a vehicle that drives forward by default, and on collision, backs away and turns away from whichever side made contact, then resumes forward once it's clear. Use a `CollisionSensor` (see [CollisionSensor](../sensors/collision.md)) in place of `light`.

!!! tip "Split forward and backward into separate pathways"
    Don't try to make one `motor` connection do both jobs. Give `motor` two separate drives instead: a **forward** pathway that receives *only* a constant (no sensor ever touches it — the vehicle always wants to go forward, the same intrinsic-drive idea as `cruise` in Vehicle III), and a **backward** pathway driven *only* by the collision sensor, wired same-side or crossed depending on which way you want it to turn while backing away.

!!! tip "The sensor goes quiet before the response should"
    A bumper only fires for the instant the robot is actually touching something — by the time it's backed off and turned a little, the sensor already reads zero again. If the backward drive follows the raw sensor directly, the robot stops backing up the moment contact breaks, often before it's actually clear of the obstacle. Insert a dynamics layer between the sensor and the backward pathway (or use the `CollisionSensor`'s own `tau_rise` / `tau_decay`), and set `tau_decay` noticeably longer than `tau_rise`: fast to react, slow to let go.

### Challenge 2 — Take it to the real robot

Once Challenge 1 works in simulation, run the same network on the physical robot — see [Running Networks on the Real Robot](../real-robot.md). No code changes are needed, just a `robot_address` on the sensor and on `motor`. If you built your `CollisionSensor` with `n=2` in Challenge 1, matching the real robot's two bumpers (reported together as a single `/bumpers` message), the same network file should drive the hardware with no rewiring at all.

---

## What to try next

- **[Wiring Brains](../wiring-brains.md)** — more general-purpose layer types and wiring patterns, including other ways dynamics layers reshape a sensor's signal before it reaches the motors.
- **[AdaptiveLayer](../layers/adaptive.md)** — the full layer reference, including the half-centre oscillator mode (`beta > 0` together with `w > 0`) this tutorial deliberately avoided.
- **[Creating Worlds](../creating-worlds.md)** — all the patch, obstacle, and duplication/grid-snap tools used in Step 2.
