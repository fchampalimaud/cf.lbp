# 2D Braitenberg Simulator

PySide6 + PyQtGraph interactive simulator for designing and testing Braitenberg vehicle neural circuits. Circuits are built in a visual network editor, run in real time against a 2D physics world, and can be exported directly to LBP.Torch Bonsai workflows.

Run with:
```bash
cd simulation
python BraitenbergSimulator.py
```

---

## Features

- **Visual network editor** — drag-and-drop layer palette, edit weights and time constants, hide/disable circuit groups, oscilloscope for any layer's output
- **Hot-pluggable brains** — drop a Python file into `brains/` and it appears in the UI immediately
- **JSON network format** — circuits saved as human-readable JSON in `networks/`; version alongside code
- **World tasks** — pluggable arena dynamics (e.g. orbiting object) in `tasks/`
- **Copy Bonsai** — one-click export of the current network to Bonsai/LBP.Torch XML, ready to paste into a Bonsai workflow

---

## Sensor Types

| Sensor | Description |
|---|---|
| `GradientSensor` | Casts n rays in a fan; returns intensity of a named gradient (A–F) |
| `CollisionSensor` | Arc-shaped bumper; detects wall or object contact |
| `DistanceSensor` | Returns distance to nearest obstacle along n rays |
| `TouchSensor` | Point contacts evenly spaced around the robot perimeter |
| `ColorSensor` | Ray-based detection of colored circular objects |
| `InteroceptiveSensor` | Gut/satiation sensor — integrates gradient exposure at the mouth point over time with asymmetric tau_rise / tau_decay; scalar output |

## Layer Types

| Layer | Description |
|---|---|
| `LeakyLayer` | First-order low-pass filter with asymmetric tau_rise / tau_decay and activation |
| `MatsuokaLayer` | Half-centre oscillator — 2 neurons with mutual inhibition |
| `ConstantLayer` | Fixed tonic drive signal |
| `AdaptiveLayer` | Leaky integrator with spike-frequency adaptation |
| `SumLayer` | Instantaneous weighted sum (no temporal dynamics) |

---

## Network JSON Format

Networks are stored in `networks/*.json`:

```json
{
  "version": 1,
  "motor_layer": "motor",
  "sensors": [ { "type": "CollisionSensor", "name": "bumper", ... } ],
  "layers":  [ { "type": "LeakyLayer",      "name": "fw",     ... } ],
  "connections": [ { "src": "bumper", "tgt": "fw", "W": [[...]] } ]
}
```

`motor_layer` names the `SumLayer` whose output drives the wheels (`output[0]` = left motor, `output[1]` = right motor).

---

## Exporting to Bonsai

With a network loaded in the network visualizer, click **Copy Bonsai**. The clipboard receives a complete Bonsai WorkflowBuilder XML fragment with three sections:

1. **Input prep** — `SubscribeSubject("{Sensor}Input") → ToTensor` for each active sensor; `CombineLatest` + `Timer → WithLatestFrom → Item2 → BehaviorSubject("NetworkInput")`.
2. **Graph construction** — `InputLayer → LeakyLayer` per sensor (dynamics inside model); `Linear(W)` per connection; `rx:Zip → JoinAdditive` for fan-in; `CreateTorchModel → BehaviorSubject("ModelCreated")`.
3. **Forward pass** — `SubscribeSubject("NetworkInput") → TorchModelForward(Dt=0.02) → BehaviorSubject(Output) → SubscribeWhen(ModelCreated)`.

Only sensors and layers that actually influence the motor output are included. Sensors with no connections are silently omitted.

---

## Key Files

```
simulation/2d/
├── BraitenbergSimulator.py   # main GUI — run this
├── bonsai_exporter.py        # converts CircuitModel → Bonsai XML
├── neurons.py                # LeakyLayer, MatsuokaLayer, ConstantLayer, AdaptiveLayer, SumLayer
├── sensors.py                # all sensor classes + SENSOR_REGISTRY
├── sim_engine.py             # pure physics step (no Qt)
├── circuit_model.py          # CircuitModel — sensors, layers, connections
├── brain_serializer.py       # JSON save/load for network files
├── network_viz.py            # network visualizer window
├── brains/                   # hot-pluggable brain modules
├── networks/                 # saved network JSON files
├── tasks/                    # pluggable world dynamics (BaseTask subclasses)
└── configs/                  # saved session configs (brain params + world state)
```

---

## Brain Plugin Contract

Drop a `.py` file in `brains/` containing a class that subclasses `BaseBrain`. The class is discovered automatically at startup.

For data-driven (JSON) brains use `DataBrain` (the network visualizer saves to `networks/`). For scripted brains:

```python
from BraitenbergSimulator import BaseBrain, Param
from sensors import GradientSensor
from neurons import LeakyLayer

class MyBrain(BaseBrain):
    sensors = [ GradientSensor(n=2, angle_spread=30, name='light') ]
    layers  = [ LeakyLayer(tau=0.1, name='motor') ]
    connections = [ ('light', 'motor', np.eye(2)) ]
    speed = Param(50.0, 0, 100, step=1.0, desc="Base speed")

    def setup(self): pass
    def loop(self, dt): return self.motor.output[0], self.motor.output[1]
    def plots(self): return []
```
