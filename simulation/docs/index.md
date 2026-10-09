# 2D LBP Simulator

A self-contained Python application for building and running neural-circuit behaviours on a simulated differential-drive robot. Everything runs on your laptop — no hardware required.

## Getting started

Download one file and run it — no git, no manual Python install.

=== "Windows"

    [Download install.bat](https://raw.githubusercontent.com/fchampalimaud/cf.lbp/main/simulation/install.bat){ .md-button }

    Save it anywhere and double-click it.

=== "macOS / Linux"

    [Download install.sh](https://raw.githubusercontent.com/fchampalimaud/cf.lbp/main/simulation/install.sh){ .md-button }

    ```bash
    bash install.sh
    ```

First run downloads the simulator and installs everything it needs (including Python itself) into
an isolated environment. Later runs just relaunch it. See [Installing](simulator/install.md) for
the git-clone alternative or a fully manual install.

## Where to start

- [Tour of the simulator](simulator/tour.md) — every panel, button, and control
- [Coding Brains](simulator/coding-brains.md) — build your first vehicle in five minutes
- [Wiring Brains](simulator/wiring-brains.md) — express the same logic as a neural circuit
- [Headless Runs](simulator/headless.md) — run saved sessions without the window, for long or repeated experiments

## Key concepts

**The arena** holds gradient patches (circular fields that sensors detect) and solid obstacles. Edit the world through the GUI — right-click to add patches, drag to move them.

**Sensors** sit on the robot body and cast rays into the arena. Sensor types include gradient sensors, collision sensors, and camera sensors.

**Neural layers** connect sensors to motor outputs. Weight matrices are editable live in the network editor without restarting the simulation.

**Parameters** (`Param`) are class-level descriptors that automatically create sliders in the GUI.

## Vehicle library

- [FeedingBrain](simulator/feedingbrain.md) — hunger-driven vehicle with food attraction, satiety neuromodulation, and two-timescale bumper avoidance
- [RingAttractorBrain](simulator/ringattractorbrain.md) — heading-direction vehicle with path-integration via an EPG ring attractor

## Reference

- [Architecture](simulator/architecture.md) — class breakdown and data-flow diagrams
- [Timing](simulator/timing.md) — simulation loop and real-time considerations
- [Real Robot](simulator/real-robot.md) — connect the simulator network to live hardware
