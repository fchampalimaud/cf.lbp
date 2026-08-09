# Class Reference

A file-by-file, class-by-class listing of the simulator's Python source. If you haven't yet, read [Architecture](architecture.md) first for the mental model this reference assumes.

---

## Timing

The simulator separates **physics/network stepping** from **display rendering** — and in real-robot mode, also from **sensor I/O** and **motor output**. This holds whether one agent or many are running: the inner loop just does more work per iteration when there are more agents.

### Simulation mode

```mermaid
flowchart TD
    T(["Qt Timer\n~every 20 ms"]) --> L

    subgraph L ["Inner loop — up to 50 ms"]
        P["tick_physics (or _batch)\ndt = 0.01 s, all agents"] --> N[append oscilloscope\ntraces for selected agent]
        N -->|deadline not reached| P
    end

    L -->|deadline reached| R[render arena]
    R --> O[update oscilloscope\ndisplay]
    O --> T
```

Because the inner loop runs many steps before handing control back to the GUI, the **network executes at ~200+ steps/s** while the **display updates at ~14 Hz**.

| Component | Typical rate |
|---|---|
| Qt timer fires | ~14 Hz |
| Physics + network step (all agents) | ~230 Hz |
| Arena repaint | ~14 Hz |
| Oscilloscope update | ~14 Hz |

### Real-robot mode

Three concurrent activities share the process, each at its own rate.

```mermaid
flowchart LR
    subgraph OSC ["OscThread (one per host:port)"]
        direction TB
        O1(["wait for\nUDP packet"]) --> O2[parse OSC\nmessage]
        O2 --> O3["sensor._robot_value\n← new value"]
        O3 --> O1
    end

    subgraph QT ["Qt Timer (~every 20 ms)"]
        direction TB
        Q1(["timer fires"]) --> QL
        subgraph QL ["Inner loop — up to 50 ms"]
            QR["read _robot_value\napply scale · τ · f"] --> QN["step_network\ndt = real elapsed"]
            QN --> QA[append traces]
            QA -->|deadline not reached| QR
        end
        QL -->|deadline| QD["render arena\n+ oscilloscope"]
        QD --> Q1
    end

    subgraph MT ["MotorThread (~60 Hz)"]
        direction TB
        M1(["sleep 16 ms"]) --> M2["read\nbrain.motor.output"]
        M2 --> M3["send /wheels\nOSC to robot"]
        M3 --> M1
    end

    Robot((Robot)) -- "bumpers · analogs\nencoders" --> OSC
    OSC -- _robot_value --> QT
    QT -- motor.output --> MT
    MT -- "/wheels UDP" --> Robot
```

| Thread | Rate | Driven by |
|---|---|---|
| OscThread | robot send rate (~60 Hz) | incoming UDP packets |
| Qt loop — network | ~230 Hz | 50 ms deadline |
| MotorThread | ~60 Hz | fixed 16 ms sleep |

!!! note "Thread safety"
    The simulator relies on Python's GIL rather than explicit locks. `OscThread` writes `sensor._robot_value`; the Qt thread reads it — a stale read is at most one robot transmission cycle old, which is harmless. `MotorThread` reads `brain.motor.output` every 16 ms; the worst case is one network step stale, well within motor latency tolerance.

---

## App shell

### `LBPSimulator.py` — `SimulatorApp`

The Qt main window. Its job is layout and wiring: it creates the panels, instantiates the controller objects, and routes Qt signals to them. Since 2026 it's been split across three files by concern — `SimulatorApp` is declared as `class SimulatorApp(_BrainMixin, _SessionMixin, QMainWindow)` — but `LBPSimulator.py` itself still keeps: window/dock construction (`_build_ui`), the generic slider/spinbox row factory (`_make_param_row`), the **Physics** tab, the **Network** tab (Host/Client mode, port + frame-rate controls, connected-client list), the **Robot** tab (real-robot Hz readouts), the **World** tab, and the network callbacks that turn an incoming client registration into a `RobotAgent` (`_on_client_registered`, `_on_remote_agent_removed`, `_update_net_status`).

### `sim_app_brain.py` — `_BrainMixin`

Owns the agent table (add/remove agent rows), the brain-file combo and reload/new buttons, the brain-parameter panel (`_rebuild_brain_params`), network-file load/save for the selected agent (`_load_data_brain_network`, `_new_network_from_sidebar`), and `_sync_group_network`, which propagates a saved network file to every other member of the same `AgentGroup`.

### `sim_app_session.py` — `_SessionMixin`

Owns the Session/Task/Logger tab builders, `_save_session` / `_load_session`, and `_load_session_agents`, which reconstructs the full multi-agent/group layout from a session JSON's `agents` array.

---

## Simulation core

### `SimController` — the simulation loop

`sim_controller.py`

`SimController` owns everything that changes every tick: the `QTimer`, the list of `RobotAgent`s, the running/paused flag, speed multiplier, real-time mode, manual-control override, the active task, and the `SimLogger`. Each tick it resolves a motor command for every agent, steps physics (batched across agents in MuJoCo mode, one at a time otherwise), and emits a single `sig_frame_ready(agents, selected_agent_id, sim_cfg, world, trail_visible, overhead_rgb)` signal that `ArenaWidget` and `OscChannelManager` both listen to.

**`RobotAgent`** (dataclass) — one robot's state bundle:

| Field | Meaning |
|---|---|
| `id` | Stable identity, assigned once, never reused or shifted when other agents are removed |
| `circuit` | This agent's `CircuitModel` |
| `brain` | Its `BaseBrain`/`DataBrain` instance (or a placeholder, for remote agents) |
| `bot_pos` | `[x, y, theta]` |
| `trail_xy` | Ring buffer of recent positions |
| `brain_mgr` | Its `BrainManager` |
| `name`, `color` | Display identity |
| `remote` | `True` if motors come from a connected `SimNetClient` rather than a local brain |

**`AgentGroup`** (dataclass) — a row in the agent table: `id`, `module` (brain filename, or `None` for a remote group), `color`, `name`, and `member_ids` (agent ids sharing this brain; `member_ids[0]` is the template used when growing the group or saving).

Every cross-reference to a robot — the selection, a gradient's `mounted_on`, a network slot — stores the agent's **id**, resolved through `_agent_by_id` / `index_of_agent`, never a raw list index. Using list position as identity was a real source of bugs (a mounted gradient reattaching to the wrong robot, or the wrong agent getting reselected after a removal); see `TODO.md`'s "stable-agent-id refactor" entry for the history.

### `network_runner` — neural forward pass

`network_runner.py`

`step_network` is the neural forward pass extracted from `BaseBrain`. It reads `layers`, `connections`, and `sensors` from the brain instance, propagates signals through the weight matrices, applies neuromodulation, and mutates `layer.output` in place. Incoming connections are summed by default; a `ProductLayer` target is special-cased to combine them by elementwise product instead.

### `sim_engine` — pure physics step

`sim_engine.py`

A module of pure functions, no Qt imports. `tick_physics(bot_pos, brain, sensors, world, sim_cfg, ...)` samples sensors, calls the brain's `loop()`, and integrates the differential-drive kinematics one timestep forward for a single agent.

### `sim_engine_mujoco.py` — `MuJoCoEngine`

The MuJoCo physics bridge actually wired into `SimController`. Builds a per-tick-rebuildable XML model covering every agent, arena walls, and objects. Key methods: `tick_physics_batch()` (steps **all** agents together in one physics step, so they can collide with each other), `render_overhead()` (top-down image composited into `ArenaWidget`), `render_cameras()` (per-agent camera-sensor rendering), `launch_viewer()` (interactive 3-D window).

!!! warning "Legacy file"
    `mujoco_bridge.py`'s `MuJocoBridge` class is an earlier, single-robot, passive-viewer experiment. It predates `sim_engine_mujoco.MuJoCoEngine` and is no longer imported anywhere — treat it as dead code, not as the current MuJoCo integration.

### `WorldEditor` — arena editing

`world_editor.py`

Owns the draw-mode state machine: `'gradient' | 'object' | 'wall' | 'wall_paint' | 'sky' | 'move'`. Implements the arena mouse handlers that mutate `World` and request a display refresh. Also handles mounting a gradient patch onto a specific robot (it then tracks that robot's pose every tick) and is multi-agent aware for mount-snapping and per-agent cleanup.

### `OscChannelManager` — oscilloscope

`osc_controller.py`

Owns the set of tracked oscilloscope channels, their colours, per-channel multiplier spinboxes, and trace ring-buffers, all for the **currently selected agent only**. Discovers which channels the active brain and circuit expose and rebuilds the plot layout when the brain or the selected agent changes. `on_frame_ready()` is its slot on `SimController.sig_frame_ready`.

### `BrainManager` — plugin loading and circuit wiring

`brain_manager.py`

Discovers Python files in `brains/`, imports them, finds the class that inherits `BaseBrain`, instantiates it, and populates a `CircuitModel`. Also synthesises the motor `SumLayer` for each joint and wires `ProprioceptiveSensor` instances onto articulated bodies automatically.

### `SimLogger` — data recording

`logger.py`

`start(path)` / `stop()` / `log(time_index, bot_pos, raw_signals, world)` — records a run's state to CSV for later analysis, independent of the oscilloscope.

---

## Distributed simulation networking

This is unrelated to the real-robot OSC protocol (see below) — it's a JSON-over-UDP link between two **simulator instances**.

### `SimNetHost`

`sim_net_host.py`

One UDP socket plus one receiver thread. Tracks a `_RemoteSlot` per connected client (motor cache, `ready` flag, Hz stats). All messages are JSON, zlib-compressed on the wire (falls back to reading them as uncompressed for backward compatibility):

| Direction | Message | Payload |
|---|---|---|
| client → host | `register` | `slot` request, client `name`, full circuit JSON |
| client → host | `heartbeat` / `ready` / `disconnect` | keep-alive and lifecycle |
| client → host | `motors` | `{mL, mR}` |
| host → client | `ack` | assigned `slot` |
| host → client | `sensors` | `{slot, dt, data}` |
| host → client | `go` | tick-synchronised start signal |

`prune_stale()` drops clients that go quiet without a clean disconnect.

### `SimNetClient`

`sim_net_client.py`

Mirrors the host: `pop_sensors()` / `send_motors()` for use inside the GUI, plus `run_blocking()` for a **headless CLI client** (`python sim_net_client.py host:port network.json`) with no Qt dependency at all.

### How a remote agent is born

When a client registers, `SimController` parses its circuit JSON (`load_network_json`), builds a `CircuitModel` containing just that client's sensors/bodies/joints (no layers or connections needed host-side), attaches a no-op placeholder brain, and calls `add_agent(..., remote=True)`. The host samples and sends exactly the sensors that client declared — earlier versions used one fixed circuit for every client, which could overrun the UDP packet size for camera-equipped robots.

---

## Rendering & views

### `ArenaWidget` — the main 2-D scene

`arena_widget.py`

Keeps a `RobotItem` and a trail `PlotDataItem` per agent. `add_robot_item` / `remove_robot_item` / `select_robot` / `update_robot_pos` provide cheap updates for non-selected agents; the **selected** agent gets a full update each frame (wheels, sensor rays, child bodies). `sync_agents()` / `on_frame_ready()` is the slot wired to `SimController.sig_frame_ready`. Also owns `set_3d_mode` / `set_overhead_frame` (compositing the MuJoCo top-down image) and the `PolyWallItem` / `CircleItem` world-object graphics.

### `NetworkVisualizerWindow` and friends

`network_viz.py` composes five mixins, organized by responsibility rather than by code-kind:

| File | Mixin | Covers |
|---|---|---|
| `network_viz_layout.py` | `_LayoutMixin` | pure column/depth/position layout math and hit-testing — zero Qt/pyqtgraph mutation |
| `network_viz_render.py` | `_RenderMixin` | all pyqtgraph/Qt graphics item creation and mutation (drawing nodes, edges, panels) |
| `network_viz_dialogs.py` | `_DialogsMixin` | sensor/layer/body creation and edit dialogs; also `WeightMatrixDialog`, `FilterStackDialog`, `PaletteChip` |
| `network_viz_editing.py` | `_EditingMixin` | mouse events, edit-mode toggles, every circuit-mutating user action; also `NetworkViewBox` |
| `network_viz_serialization.py` | `_SerializationMixin` | undo snapshots, save/load JSON, Bonsai/SVG export, motif save |

`network_viz_context.py` holds `NetworkVizContext`, the narrow DI facade bridging the visualizer to `SimulatorApp` — it doesn't fit any of the five responsibility categories above, so it gets its own small file.

### `side_view.py` — network layout editor

Despite the name, this is **not** a physical/sagittal view of the robot. It's a drag-and-drop grid editor — columns are processing depth, rows are subsumption "z-level" — used to arrange the network diagram's layers and sensors. The coordinates it produces feed `net_view_3d.py`.

### `net_view_3d.py` — `NetView3DWindow`

A read-only, rotatable OpenGL (`pyqtgraph.opengl`) 3-D rendering of **the circuit graph** (nodes = sensors/layers, Bezier arcs = connections, colour-coded excitatory/inhibitory) — not a 3-D view of the robot or arena. Its layout comes from `side_view.py`.

---

## World & circuit data

### `World` — the physical environment

`world.py`

Owns everything that is not the robot: gradient patches, solid obstacles, polygon walls, the arena boundary, and the sky (for the compass sensor). Plain data container — no rendering or physics logic.

### `CircuitModel` — shared circuit state

`circuit_model.py`

A plain container holding the four lists that define one agent's circuit: `sensors`, `layers`, `connections`, and `bodies`/`joints`. `connections` is a list of `Connection` dataclass objects (`src`, `tgt`, `W`, `learning`, `lr`).

### `RigidBody` / `Joint` — articulated robot body

`rigid_body.py`

`RigidBody` is a named disk with a radius. Every robot has a root body (the drive disk); extra bodies attach via `Joint`s — for example a passive or motor-driven segment carrying its own sensors.

### `SimConfig` — shared simulation parameters

`sim_config.py`

Holds knobs global to a session: timestep `dt`, arena size, robot body radius, maximum speed, and similar constants. A `BaseConfig` subclass, so any `Param` declared on it automatically generates a GUI slider.

| Parameter | Default | Description |
|---|---|---|
| `dt` | 0.01 s | Simulation timestep |
| `arena_scale` | 5.0 m | Arena half-width |
| `motor_gain` | 1.0 | Motor speed multiplier |
| `body_radius` | 0.2 m | Robot body radius |
| `sense_radius` | 1.0 m | Sensor ray length |
| `init_x`, `init_y` | 0, 0 | Robot start position |
| `stim_radius` | 0.5 m | Radius of new gradient patches |
| `toggle_stim` | on | Show / hide stimulus patches |
| `fixate_robot` | off | Freeze robot position |

---

## Brains

### `BaseBrain` / `DataBrain` — the control law

`brain_base.py`

`BaseBrain` is the base class every brain plugin inherits. Class-level `Param` and `ChoiceParam` descriptors declare tunable knobs that `BaseConfig.__init__` copies to instance attributes; the GUI reads the metadata to build sliders automatically.

`DataBrain` extends `BaseBrain` for brains loaded from a JSON network file — it rebuilds `layers`, `connections`, and `sensors` from the serialised description, so the brain file contains only data, no Python logic.

---

## Sensors — transduction

`sensors.py`

`BaseSensor` defines the contract every sensor satisfies: `sample(x, y, theta, world, sim_cfg)` maps the robot's pose and world state to a numpy array, stored on the brain as `brain.<sensor.name>`. All sensors share an optional output pipeline: Gaussian noise → asymmetric leaky dynamics (`tau_rise` / `tau_decay`) → activation function → `output_mode` (none / derivative / integral).

```mermaid
%%{init: {'themeVariables': {'fontSize': '13px'}}}%%
graph LR
    BS[BaseSensor]
    BS --> GS[GradientSensor]
    BS --> CS[ColorSensor]
    BS --> DS[DistanceSensor]
    BS --> CL[CollisionSensor]
    BS --> WH[WhiskerSensor]
    BS --> GC[GrayCameraSensor]
    BS --> RC[RGBCameraSensor]
    BS --> IN[InteroceptiveSensor]
    BS --> PR[ProprioceptiveSensor]
    BS --> SK[SkyCompassSensor]
```

| Class | What it detects |
|---|---|
| `GradientSensor` | Soft circular gradient patches; casts n rays in a fan, returns field intensity per ray |
| `ColorSensor` | Solid coloured circular objects via ray-circle intersection |
| `DistanceSensor` | Normalised proximity to the nearest wall or obstacle (1 = touching, 0 = at max range) |
| `CollisionSensor` | Contact within n arc sectors around the robot perimeter (1 = contact, 0 = clear) |
| `WhiskerSensor` | Tactile whisker: bending proportion from 0 (no contact) to 1 (contact at base) |
| `GrayCameraSensor` | Wide-angle raycasted image (luminance); output shape `(H × W,)` |
| `RGBCameraSensor` | Wide-angle raycasted image (colour, CHW); output shape `(3 × H × W,)` |
| `InteroceptiveSensor` | Internal gut state: integrates gradient exposure at the mouth over time (scalar) |
| `ProprioceptiveSensor` | Joint angle or angular velocity of articulated body segments |
| `SkyCompassSensor` | Polarised-light sky compass (DRA); encodes heading relative to sun direction |

Both camera sensors support `lateralized=True`, splitting the image at the horizontal midline into `sensor_L` / `sensor_R` halves, each feeding its own `Conv2dLayer`.

---

## Neuron layers — neural dynamics

`neurons.py` re-exports everything below and maintains `LAYER_REGISTRY` for JSON deserialisation. `LayerBase` is a thin `nn.Module` mixin adding display and neuromodulation attributes shared by every layer type (`name`, `color`, `group`, `modulators`, …).

```mermaid
%%{init: {'themeVariables': {'fontSize': '13px'}}}%%
graph LR
    LB[LayerBase]
    LB --> LL[LeakyLayer]
    LL --> PRL[ProductLayer]
    LB --> AL[AdaptiveLayer]
    AL --> ML[MatsuokaLayer *deprecated*]
    LB --> CL[ConstantLayer]
    LB --> SL[SumLayer]
    SL --> MOT[MotorLayer]
    LB --> PL[PulseLayer]
    LB --> SNL[SineLayer]
    LB --> RL[RingAttractorLayer]
    LB --> CV[Conv2dLayer]
    LB --> L2[Leaky2dLayer]
```

| Class | Dynamics |
|---|---|
| `LeakyLayer` | First-order low-pass filter (`dx/dt = (u−x)/τ`); asymmetric rise/decay, derivative mode, OU noise |
| `ProductLayer` | `LeakyLayer` dynamics, but incoming connections combine by elementwise product instead of sum |
| `AdaptiveLayer` | Leaky integrator with spike-frequency adaptation; `w > 0` + `n=2` gives half-centre oscillation |
| `MatsuokaLayer` | *(deprecated — use `AdaptiveLayer`)* Thin wrapper kept for loading old JSON networks |
| `ConstantLayer` | Fixed output; tonic drive source, ignores incoming connections |
| `SumLayer` | Instantaneous weighted sum; no dynamics, no memory |
| `MotorLayer` | `SumLayer` + robot actuation; sends output via OSC to `robot_address` in real-robot mode |
| `PulseLayer` | Plateau-potential neurons with sustained activation and inhibitory reset |
| `SineLayer` | Autonomous sine-wave generator; ignores incoming connections |
| `RingAttractorLayer` | N leaky neurons on a ring; recurrent connectivity via a self-connection (Mexican-hat kernel) |
| `Conv2dLayer` | 2-D convolution over camera input; per-filter global pooling; optional leaky dynamics and adaptation |
| `Leaky2dLayer` | Pixel-wise leaky integrator that preserves full spatial image structure; feeds into `Conv2dLayer` |
| `AccumulatorLayer` | Integrates input over time without decay |
| `DeltaLayer` | Reports the change in its input since the previous step |
| `Reichardt2dLayer` | Elementary motion detector over 2-D input (Reichardt correlator) |
| `TDLayer` | Temporal-difference learning layer |
| `ThreeFactorLayer` | Three-factor (Hebbian + neuromodulator) learning layer |

---

## Persistence & export

### `session_io` — save / load

`session_io.py`

Two pure functions, `save_session` and `load_session`, serialise and deserialise the full simulator state (agents, brain params, world patches, sim config, oscilloscope multipliers) as JSON.

### `brain_serializer` — code generation

`brain_serializer.py`

Pure functions for writing brain `.py` files from a live circuit, and for JSON ↔ circuit serialisation (`serialize_network_json`, `load_network_json`) used both by session saving and by the distributed-networking handshake. Kept separate from the visualiser so the same logic is available from tests or command-line tools.

### `bonsai_exporter` — Bonsai XML export

`bonsai_exporter.py`

Converts a `CircuitModel` into LBP.Torch Bonsai XML that can be pasted directly into a Bonsai workflow. Traces only the layers reachable from the motor output, maps sensor dynamics and activations to their Bonsai equivalents, and generates the input-preparation, graph-construction, and forward-pass branches.

---

## Real-robot I/O

### `RobotDriver`

`robot_driver.py`

Isolates all real-robot communication. Manages one background thread per unique `robot_address` string found among the active sensors — `CameraThread` (UDP JPEG client) for camera sensors, `OscThread` (UDP OSC server) for everything else. Each thread writes decoded data into `sensor._robot_value` so the simulation loop reads it without touching any sockets. `_parse_address()` is a shared helper also used by the Robot tab's Hz display and by `SimController`'s motor dispatch. See [Running on the real robot](real-robot.md) for the full usage guide.

---

## File map

```
LBPSimulator.py           main window (SimulatorApp) — Physics/Network/Robot/World tabs
sim_app_brain.py          _BrainMixin — agent table, brain loading, network sync
sim_app_session.py        _SessionMixin — Session/Task/Logger tabs, save/load
sim_controller.py         simulation loop, RobotAgent, AgentGroup, multi-agent tick
sim_engine.py             pure physics step for one agent (tick_physics)
sim_engine_mujoco.py      MuJoCoEngine — batched multi-agent MuJoCo physics + rendering
mujoco_bridge.py          legacy single-robot viewer — unused, not wired in
network_runner.py         neural forward pass
neurons.py                all layer classes + DynamicsBase
sensors.py                all sensor classes + SENSOR_REGISTRY
circuit_model.py          CircuitModel, Connection
brain_base.py             BaseBrain, DataBrain, Param, ChoiceParam
brain_manager.py          brain discovery, loading, circuit wiring
brain_serializer.py       JSON ↔ circuit serialisation
bonsai_exporter.py        CircuitModel → LBP.Torch Bonsai XML
world.py                  World — patches, objects, walls, sky
world_editor.py           arena draw-mode state machine (multi-agent aware)
rigid_body.py             RigidBody, Joint
osc_controller.py         OscChannelManager (oscilloscope, selected agent only)
session_io.py             save/load session JSON (multi-agent)
robot_driver.py           real-robot OSC + camera threads
sim_net_host.py           SimNetHost — distributed-sim host side
sim_net_client.py         SimNetClient — distributed-sim client side (+ headless CLI)
sim_config.py             SimConfig (global simulation parameters)
sim_widgets.py            reusable Qt widgets
sim_constants.py          shared numeric constants
logger.py                 SimLogger — CSV data recording
arena_widget.py           ArenaWidget, RobotItem — multi-robot 2-D scene
network_viz.py            NetworkVisualizerWindow (composes the mixins below)
network_viz_context.py    NetworkVizContext — DI facade bridging to SimulatorApp
network_viz_layout.py     _LayoutMixin — pure column/depth/position layout math
network_viz_render.py     _RenderMixin — pyqtgraph/Qt graphics item drawing
network_viz_dialogs.py    _DialogsMixin — creation/edit dialogs
network_viz_editing.py    _EditingMixin — mouse events, toolbar, circuit edits
network_viz_serialization.py  _SerializationMixin — undo, save/load, export
side_view.py              network layout editor (NOT a robot body view)
net_view_3d.py            NetView3DWindow — 3-D view of the circuit graph
trajectory_viz.py         post-run trajectory visualiser
export_network_svg.py     export network graph as SVG
brains/                   hot-pluggable brain plugins
networks/                 saved network JSON files
tasks/                    pluggable world dynamics (BaseTask subclasses)
configs/                  saved session configs
```
