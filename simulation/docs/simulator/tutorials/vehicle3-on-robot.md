# Circuits Driving a Real Robot

The simulator can drive a live LBP robot using exactly the same network it runs in simulation — no code changes, no rewiring. This tutorial shows you how: you tell each sensor and the motor layer where to find their data on the network, switch to robot mode, and the brain runs on hardware.

You will use the **D01 - AttractionBrain** network that ships with the simulator. It has a `light` GradientSensor and a `motor` layer wired for light-seeking — simple enough to verify easily on a real robot, and a good template for wiring any network to hardware.

**What you will do**

| Step | What happens |
|------|--------------|
| Enable robot mode | Switch the simulator to live hardware |
| Set the motor address | Tell the simulator where to send wheel commands |
| Verify the connection | Confirm the link with manual drive |
| Set the sensor address | Map the robot's IR readings to the `light` node |
| Run the brain live | Light-seeking behaviour on real hardware |

---

## Before you start

Make sure you have the following ready:

- The name of the Wi-Fi network your robot uses.
- Your computer connected to that exact same network.
- The robot's IP address.

The robot communicates over OSC/UDP on port **2390**. The simulator sends and receives on that same port — no extra software is needed.


## Step 2 — Load the attraction brain

In the **Brain** tab, switch to **Network** mode, set the project to **🔒 Simulator → Demos**, and select **D01 - AttractionBrain**.

---

## Step 3 — Enable robot mode

In the left panel, press **🤖 Robot** in the Simulation group. The status bar will indicate that robot mode is active.

In robot mode the simulated body is replaced by the real robot: sensor values arrive as OSC packets from the hardware, and each motor command is sent back over the network. The arena view goes dark — there is no simulated world to draw — and the oscilloscope and network visualizer continue working as normal, now reflecting live hardware data.

---

## Step 4 — Open the network visualizer

In the **Brain** tab, click **⬡ Open visualizer**. You should see the attraction network: a `light` GradientSensor and a `motor` layer connected with crossed excitatory weights.

---

## Step 5 — Set the motor address

Double-click the `motor` node in the network visualizer. The properties dialog has a **Robot address** field. Enter:

```
192.168.0.224:2390/wheels
```

Replace `192.168.0.224` with your robot's actual IP. Confirm.

The simulator will send `/wheels int vleft int vright` to this address each step, clamped to [−100, 100].

---

## Step 6 — Verify the connection with manual drive

With the motor address set, confirm that commands are actually reaching the robot before going further.

1. In the **Simulation group**, press **⌨ Manual** to enable manual mode.
2. Press **▶ Run**.
3. Drive the robot with the keyboard — `W` forward, `S` backward, `A`/`D` to turn, `Space` to stop.

!!! note "What manual mode does"
    Manual mode does **not** bypass the brain — the network keeps running every step. It only changes which wheel command gets sent: yours instead of the brain's output. This is exactly what makes it useful here: the brain is live, the visualizer updates, and you can confirm the OSC link is healthy before committing to full autonomous mode.

If the robot responds to your keys, the connection is working. Press **⌨ Manual** again to turn it off, then **⏸ Pause** before the next step.

---

## Step 7 — Set the sensor address

The robot's light sensors are read from its analog input ports and sent as OSC on `/analogs`. Each `/analogs` packet carries five integers: a packet counter followed by four ADC readings (A1, A2, A5, A6). The light sensors should be connected to **A1** (left) and **A6** (right), which are indices 1 and 4 in the packet.

Double-click the `light` node. Set the **Robot address** field to:

```
192.168.0.224:2390/analogs[iiiii](1,4){x/500}
```

| Part | Meaning |
|---|---|
| `192.168.0.224:2390` | Robot IP and port |
| `/analogs` | The OSC address the robot sends sensor readings on |
| `[iiiii]` | Five integer arguments expected |
| `(1,4)` | Extract indices 1 and 4 (the two IR sensors) |
| `{x/1023}` | Normalise raw ADC range 0–1023 to [0, 1] |

Confirm.

!!! note "Scaling"
    `{x/1023}` normalises the raw ADC value to [0, 1]. The sensor's own **scale** parameter then multiplies that result — so the effective range the brain sees is [0, scale]. Set `scale` on the `light` node to match the motor range you need, exactly as you would in simulation.

---

## Step 8 — Run the brain on the robot

Press **▶ Run**. The simulator begins sending motor commands and receiving sensor data at approximately 60 Hz.

Point the robot toward a light source that your IR sensors can detect (a bright lamp or a projector patch works well in lab settings). With the crossed excitatory wiring, the sensor on the side closer to the source drives the opposite wheel harder — the robot steers toward the light.

Watch the network visualizer while the robot moves: the `light` node's neuron activations update in real time from the hardware, and the motor output reflects what is actually being sent to the wheels.

---

## Step 9 — Tune for the real world

Simulation and hardware rarely behave identically. Here are the most common adjustments:

**Sensor scale doesn't match**

Your IR sensors may saturate at a different distance than the simulator's gradient patch did. If the robot steers too weakly toward the source, increase `light`'s `scale`. If it overshoots and oscillates past the source, reduce it.

**One wheel is faster than the other**

Real motors have small asymmetries. If the robot drifts left or right when both sensors read equal values, adjust the two connection weights slightly — raise the weight on the path feeding the slower wheel.

**Response is too jerky**

The motor command changes every step based on the current sensor reading, with no smoothing. Add a **LeakyLayer** between `light` and `motor` (see [Wiring Brains](../wiring-brains.md#adding-memory-the-leakylayer)) to introduce a time constant — the sensor's influence ramps up and decays gradually, giving the robot a more fluid response.

---

## What to try next

- **Switch to avoidance** — load **D01 - AvoidanceBrain** instead and repeat the steps. The robot will now flee the light source.
- **[Running Networks on the Real Robot](../real-robot.md)** — full reference for the `robot_address` format, all supported sensors, and the packet layout for bumpers, encoders, and gyro.
- **[Wiring Brains](../wiring-brains.md)** — add a LeakyLayer for smooth motor responses, or chain more sensors into the same network.
