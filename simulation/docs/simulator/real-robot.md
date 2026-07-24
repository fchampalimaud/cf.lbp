# Running Networks on the Real Robot

The simulator can drive a live robot using the same brain network you designed in simulation — no code changes required. In **real-robot mode** the virtual physics are bypassed: sensor values come from the robot's actual hardware, and motor commands go out over the network each step.

---

## How it works

`SimController` has two operating modes:

| Mode | Sensor source | Motor output | `dt` |
|---|---|---|---|
| **Simulation** (default) | Virtual raycasting / world sampling | In-process physics | Fixed (`sim_cfg.dt`) |
| **Real robot** | Live data from `RobotDriver` threads | OSC `/wheels` to the robot | Actual wall-clock elapsed time |

The neural forward pass (`network_runner.step_network`) is identical in both modes. Only the source of sensor values and the destination of motor commands change.

---

## Architecture

All robot I/O is isolated in `robot_driver.py`. `SimController` holds one `RobotDriver` instance and calls three methods:

```
sim_controller.enable_robot_mode(True, host, osc_port, motor_port)
    └── robot_driver.start(sensors, host, osc_port)
            └── one thread per unique sensor robot_address

sim_controller._tick_robot()           # called each frame instead of _tick()
    ├── read sensor._robot_value → brain.<name>
    ├── network_runner.step_network(brain, real_dt)
    └── robot_driver.send_wheels(host, motor_port, mL, mR)

sim_controller.enable_robot_mode(False, ...)
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

Motor values come from the brain's `motor` layer output in the usual way. They are clamped to integers and sent as a `/wheels` OSC message:

```
/wheels  int vleft  int vright
```

The target address is `robot_host:robot_motor_port`. The manual-control override (WASD keys) works in robot mode exactly as in simulation — it replaces the brain's motor output before the OSC send.

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
# Enable
sim_controller.enable_robot_mode(
    True,
    robot_host='192.168.0.224',
    robot_osc_port=9998,     # local port the OSC server binds to
    robot_motor_port=2390,   # port on the robot that receives /wheels
)

# Disable
sim_controller.enable_robot_mode(False)
```

Calling `enable_robot_mode` while the simulation is running pauses it, reconfigures the driver, and restarts automatically.
