# Running Networks on the Real Robot

The simulator can drive a live robot using the same brain you designed in simulation — no code changes required. In **real-robot mode** the real robot takes the place of the simulated world: sensor values come from the robot's actual hardware, and motor commands go out over the network each step.

---

## How it works

Every mode runs the same step: **sense → think → motors → act**. Only sensing and acting change:

| Mode | Sense | Act | `dt` |
|---|---|---|---|
| **Simulation** (default) | MuJoCo + 2-D fields | MuJoCo moves the robot | Fixed (`sim_cfg.dt`) |
| **Real robot** | Latest value from `RobotDriver` threads, per sensor | OSC `/wheels` to the robot | Actual wall-clock elapsed time |

Think and motors are the same code in both modes (`sim_engine.run_brain` / `choose_motor_command`): the brain's `loop(dt)` runs once per step, so network brains, code-only brains and any extra Python logic in `loop()` behave the same on the robot as in simulation.

---

## Architecture

All robot I/O is isolated in `robot_driver.py`. `RobotModeController` (owned by `SimController`) holds one `RobotDriver`:

```
sim_controller.enable_robot_mode(True)
    └── robot_driver.start(sensors)
            └── one thread per unique sensor robot_address

sim_controller._tick_robot()           # called each step instead of _tick()
    ├── sense:  sensor._robot_value → brain.<name>
    ├── think:  brain.loop(real_dt)
    ├── motors: keyboard, else the brain's command → wheel_cmd
    └── act:    MotorThread (~60 Hz) sends wheel_cmd to every motor layer's robot_address

sim_controller.enable_robot_mode(False)
    └── robot_driver.stop()
```

Nothing outside `robot_driver.py` opens a socket or knows about UDP.

---

## Thread model

One background thread is created per unique `robot_address`. Threads are started when robot mode is enabled and joined when it is disabled.

### `OscThread` — bumpers and analog sensors

Binds a local UDP socket on the port from `robot_address`, receives OSC packets, and dispatches each message to the sensor whose `osc_path` matches. Writes a `numpy` array into `sensor._robot_value`.

```
Robot (MKR1010)
  │  OSC/UDP  →  OscThread (local :9998)
  │                  /bumpers  →  CollisionSensor._robot_value
  │                  /analogs2 →  AnalogSensor._robot_value
  └──────────────────────────────────────────────────────────
```

### `CameraThread` — JPEG camera

Connects to the Raspberry Pi Zero camera stream (`send_frames.py` protocol): sends a keepalive UDP packet once per second, then reads incoming JPEG datagrams. Each frame is decoded with Pillow and resized to the sensor's declared `(height, width)` so the brain's conv weights remain valid.

```
Raspberry Pi Zero
  │  JPEG/UDP  ←  keepalive (once per second)
  │            →  CameraThread  →  decode → resize → sensor._robot_value
  │                                                 → sensor._last_frame
  │                                                 → sensor._left/_right_output (if lateralized)
  └────────────────────────────────────────────────────────────────────────────────────
```

Output formats written to `sensor._robot_value`:

| Sensor type | Shape | Range |
|---|---|---|
| `GrayCameraSensor` | `(H × W,)` flat, row-major | `[0, 1]` |
| `RGBCameraSensor` | `(3 × H × W,)` flat, CHW | `[0, 1]` |

---

## Configuring sensors for the real robot

Each sensor has a `robot_address` field (settable in the network editor's sensor properties). Set it to the **remote host:port** that provides its data.

### Camera sensor

```python
GrayCameraSensor(
    width=64, height=32,
    robot_address='192.168.0.190:5002',   # Raspberry Pi Zero
    lateralized=True,
)
```

### Collision / bumper sensor

```python
CollisionSensor(
    n=4,
    robot_address='192.168.0.224:9998',   # MKR1010 OSC port (local listen)
    osc_path='/bumpers',
)
```

`osc_path` is the OSC message address the robot sends. The `OscThread` filters incoming packets by this path and writes matching values to the sensor buffer.

### Sensors without a `robot_address`

Sensors with an empty `robot_address` are ignored by `RobotDriver` and receive a zero array each step in robot mode. This is safe — the network still runs, those inputs are just silent.

---

## Real-time `dt`

In simulation mode `dt` is fixed (`SimConfig.dt`, default 20 ms). In robot mode, `dt` is the actual wall-clock interval between successive `_tick_robot` calls, measured with `time.perf_counter()`. The first tick of a session falls back to the configured `dt`.

This matters for any layer that uses leaky dynamics (`tau_rise`, `tau_decay`): the integration step is `Δx = (u − x) / tau × dt`, so using the real elapsed time keeps the neuron time constants correctly calibrated to wall-clock seconds.

---

## Motor output

Each step's wheel command is what the brain's `loop()` returns (for network brains, the `motor` layer output), clamped to the duty range [−100, 100]. The motor thread rounds it to integers and sends it as a `/wheels` OSC message:

```
/wheels  int vleft  int vright
```

It goes to the `robot_address` of **every motor layer** — the same rule for every brain. A code-only brain (one that computes `mL, mR` in Python) just needs a motor layer with a `robot_address` to drive the robot.

Keyboard control (WASD) is a motor command source exactly as in simulation: while active, its command is sent instead of the brain's, and is written into the motor layer so the visualizer and oscilloscope show what drives the robot. The brain keeps running (see `rules/motor_commands.md`).

---

## Ports and addresses (default setup)

| Sensor | `robot_address` | `osc_path` |
|---|---|---|
| Bumpers (`CollisionSensor`) | `192.168.0.224:9998` | `/bumpers` |
| Pi Zero camera | `192.168.0.190:5002` | *(camera, no OSC path)* |

Motor commands go to `192.168.0.224:2390`.

See the robot network connectivity documentation for the full port reference.

---

## Enabling robot mode from code

`SimController.enable_robot_mode` is the single entry point:

```python
sim_controller.enable_robot_mode(True)    # connects using each sensor's robot_address
sim_controller.enable_robot_mode(False)
```

Addresses are not arguments: sensors read from their own `robot_address`, and motor commands go to each motor layer's `robot_address`.

Calling `enable_robot_mode` while the simulation is running pauses it, reconfigures the driver, and restarts automatically.
