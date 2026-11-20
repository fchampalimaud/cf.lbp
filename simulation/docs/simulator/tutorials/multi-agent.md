# A World with Many Vehicles

The simulator can run any number of robots simultaneously, each with its own brain, its own body, and its own sensor readings — all sharing the same arena. This tutorial introduces the **multi-agent architecture**: how agents are organised into groups, how to run several agents with the same network, and how to stage two groups with different behaviours in the same world.

The focus is on the architecture, not the brain. You will use two networks that ship with the simulator — **D01 - AttractionBrain** and **D01 - AvoidanceBrain** — so there is nothing to build before you start.

**What you will do**

| Step | What happens |
|------|--------------|
| Understand groups | One group = one network shared by N agents |
| Scale within a group | A single spinbox clones an agent N times |
| Add a second group | Two groups, two networks, two behaviours in the same arena |
| Position agents | Drag discs in the arena to set starting positions |
| Observe the interaction | Attraction and avoidance vehicles in the same world |

---

## Background — groups and agents

The simulator organises agents into **groups**. Every agent in a group runs the same network — same layers, same weights — but each has its own physics body, its own sensor readings, and its own motor state. Changing a weight or property in the visualizer changes it for every agent in the selected group simultaneously.

A session can have any number of groups, each running a different network. This lets you stage interactions: one group that seeks light, another that flees it; both in the same arena at the same time.

The **Brain tab** shows one row per group:

| Column | Meaning |
|---|---|
| Colour swatch | The group's display colour — click to change |
| Name | Editable label |
| Brain | Which network the group runs |
| N | How many agents are in the group |

---

## Step 1 — Load the attraction network

Launch the simulator. In the **Brain** tab, switch to **Network** mode. Set **Project** to **🔒 Simulator → Demos** and select **D01 - AttractionBrain**. The Brain tab shows one group — **Group 1** — with **N = 1**.

Place a single gradient patch (label **A**) in the arena (World tab → click **A** → click in the arena). Press **▶ Run**: the robot steers toward the patch using crossed excitatory connections. Press **⏸ Pause**.

---

## Step 2 — Scale Group 1 to multiple agents

Look at the **N** spinbox in Group 1's row. Change it from **1** to **3**.

Three robot discs appear. Press **▶ Run**: all three run the identical network independently — same weights, same layers — but each receives its own sensor readings and produces its own motor commands. Watch them converge on the patch from different angles.


!!! note "Agents don't share state"
    Each agent has a completely independent activation state. The group only means they run the same network — it is cloned, not shared.

Press **⏸ Pause**. Set N back to **1** before the next step.

---

## Step 3 — Add a second group

At the bottom of the agent table, click **+**. A second row appears — **Group 2**.

With Group 2 selected, switch its network: in the **Project** selector choose **🔒 Simulator → Demos**, then pick **D01 - AvoidanceBrain**. This network uses the same `light` GradientSensor and `motor` layer as the attraction brain, but wired with same-side excitatory connections — the robot steers away from the patch.

!!! note "The network visualizer follows the selected group"
    Click Group 1's row — the visualizer shows the attraction network. Click Group 2 — it shows avoidance. There is only one visualizer window; it always reflects the selected group.

---

## Step 4 — Position agents in the arena

With two groups, the default start positions overlap. Press **↺ Reset**, then drag each robot disc (*Alt+drag*) to a different position in the arena. The position you drop it on becomes that agent's permanent reset position, saved with the session. 

Place both agents at the same distance from the patch, facing it directly. Starting from identical conditions makes the behavioural difference immediately clear: one will drive toward the patch, the other will veer away — purely because of how their sensors are wired to their motors.

!!! note "Need to rotate the robots?"
    You can rotate the robots using *Alt+Shift+drag*.

---

## Step 5 — Run and observe

Press **▶ Run**. You now have two agents with opposite light responses coexisting in the same arena:

- **Group 1** steers toward the patch and settles near it.
- **Group 2** steers away and orbits the arena perimeter.

Add a second patch (same label **A**, different position) while the simulation is running. Both groups respond to both patches simultaneously — the sensors detect the maximum signal from all patches of the same label, so the approach agent tracks whichever patch it is closest to, and the avoidance agent keeps its distance from both.

---

## Step 6 — Try varying N

Select Group 2 and set **N** to **4**. Four avoidance agents appear. Each is repelled from the patch, but their sensor readings differ by position, so they naturally spread around the arena rather than clustering. This emergent dispersion is invisible when you run only one agent.


