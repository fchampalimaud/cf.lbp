# 2D Simulator

`simulation/2d/LBPSimulator.py` is a world/brain simulator for studying *the Brain* in an evolutionary way — starting simple and growing progressively more sophisticated. It couples a physics simulator with a neural simulator, with sensors and motors as the interface between them. In its simplest form the physics layer runs a differential-drive robot through a 2D arena of gradient fields and obstacles; the neural layer runs dynamic neural models that read from the robot's sensors and write motor commands back.

## Running it

```bash
cd simulation/2d
python LBPSimulator.py
```

## Key concepts

**The arena** holds gradient patches (circular fields that sensors can detect) and solid obstacles. You edit the world through the GUI — right-click to add patches, drag to move them.

**Sensors** sit on the robot body. A `GradientSensor` casts rays into the arena and returns how much of a patch falls along each ray. Sensors are declared in your brain file.

**The brain** is a Python plugin: a class that inherits from `BaseBrain`, declares its sensors, and implements a `loop(dt)` method that returns `(mL, mR)` — left and right motor speeds in the range `[-100, 100]`.

**Parameters** (`Param`) are class-level descriptors that automatically create sliders in the GUI. You tune them live without restarting.

## Tutorials

- [Tour of the simulator](tour.md) — every panel, button, and control, with linkable anchors
- [Coding Brains](coding-brains.md) — hardwire sensor-to-motor logic directly in `loop()`
- [Wiring Brains](wiring-brains.md) — express the same logic as a weight matrix so you can edit it live

## Vehicle library

- [FeedingBrain](feedingbrain.md) — hunger-driven vehicle with food attraction, satiety neuromodulation, and two-timescale bumper avoidance
- [RingAttractorBrain](ringattractorbrain.md) — heading-direction vehicle with path-integration: EPG ring attractor updated by wheel velocity via PEN shifter neurons

## Deployment

- [Running on the real robot](real-robot.md) — connect the simulator network to live hardware; sensor threads, camera streaming, real-time `dt`
