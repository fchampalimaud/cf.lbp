# Watching the Brain While You Drive

The **oscilloscope** is a scrolling time-series plot that shows the live state of any node in your network. This page walks through it using the D01 attraction circuit: first with manual drive, then letting the brain take over, so you can see the difference.

---

## Step 1 — Set up

1. Open the simulator.
2. Click the **Brain** tab, select **Network**, and load **🔒 Simulator → Demos / D01 - AttractionBrain**.
3. Click the **World** tab, press the **A** button, and drag out a gradient patch somewhere in the arena.
4. Tick **⌨ Manual** in the *Options* row.
5. Press **▶ Run**.

The robot sits still — you are in the driving seat.

---

## Step 2 — Open the oscilloscope and watch the motors

Click **Osc** in the *Visualization* row. A scrolling plot appears docked at the bottom, already showing the **motor** channels (`motor_0` left wheel, `motor_1` right wheel) — the motor node is pinned by default.

Drive the robot around with `W` / `S` / `A` / `D` and watch the motor traces respond. Check the **Real Time** checkbox in the top panel if your robot is driving too fast.

!!! tip "Signal too small to see?"
    Each channel has a multiplier spinner next to its label. Increase it to scale the trace up or down. This allows you to vizualize neurons with different activity ranges.

---

## Step 3 — Pin the sensor channels

The motor traces show what the wheels are doing, but not *why*. Add the sensor to see the full picture.

1. Open the **Network Visualizer** (Brain tab → **⬡ Open visualizer**).
2. Right-click the **`light`** node.
3. Choose **Add to oscilloscope**.

Two new channels appear — `light_0` (left ray) and `light_1` (right ray). Now drive the robot toward and away from the gradient patch:

**Approach head-on.** Both channels rise together as the robot closes in; if you come in straight they stay roughly equal.

**Approach from one side.** The ray pointing toward the patch rises; the other stays low.

**Spin in place** next to the patch. One ray sweeps through and spikes, then the other — you can read off exactly where each ray is pointing by watching the signal.

**Drive through the centre.** Both channels peak and drop as the patch slides out behind the robot.

!!! note "The brain is still running"
    Manual drive replaces only the final motor command — where the wheel values come from. Everything else keeps running: the sensor rays cast, the weights multiply, the motor layer computes its output. The oscilloscope reflects this: `light_0` and `light_1` are live sensor readings, and the motor traces show what the brain *would* have commanded even though you are overriding it.

---

## Step 4 — Hand back control

Untick **⌨ Manual** (the robot is still running). The brain now drives the wheels.

Watch the oscilloscope: the sensor channels and motor channels are now coupled — when `light_1` rises, `motor_0` follows (the crossed wiring of the attraction circuit). You can see the full sensorimotor loop unfolding in the plot as the robot homes in on the patch.

---

## Going further

The same technique works for any node in the network. Right-click any sensor, layer, or motor node in the visualizer and choose **Add to oscilloscope** to track it. This is useful for:

- Checking that an intermediate layer is responding as expected before the full circuit is wired.
- Comparing sensor and motor signals side by side to verify the weight signs are correct.
- Watching learning layers change their output over time.
