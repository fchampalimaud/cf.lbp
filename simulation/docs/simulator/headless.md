# Headless Runs

A **headless run** plays a saved session without opening the simulator window. Nothing is drawn, so it runs as fast as your computer allows — usually much faster than real time — and it can be scripted. Use it when you want to:

- run a long experiment (minutes or hours of simulated time) without watching it,
- repeat the same experiment many times, e.g. with different random seeds,
- compare brains or parameter values side by side (a *parameter sweep*),
- record every step to a file for analysis in Python, a notebook or a spreadsheet.

A headless run uses exactly the same simulation as the app: the same session gives the same result in both, step for step.

---

## Before you start

1. **Build and save a session in the app.** A headless run starts from a session file — the arena, the agents, their brains and the simulation settings — saved with **Session → Save**. Session files live in `configs/` (and its subfolders).
2. **Open a terminal in the simulator folder** (`simulation/2d/`) with the simulator's Python environment active — the same one you use to start the app.

---

## Running a session

```bash
python src/headless.py "configs/Tutorials/T03 - ColisionSession.json" --seconds 60
```

When it finishes it prints a short summary — how many steps ran and where each agent ended up:

```
T03 - ColisionSession.json: 6000 steps, 60.00 s simulated, 1 agent(s)
  agent 0: x=1.284 y=-0.412 theta=2.031
```

Put the session path in quotes if it contains spaces.

### Options

| Option | Meaning |
|---|---|
| `--seconds N` | How many seconds of **simulated** time to run (required). The number of steps is `N / dt`, with `dt` taken from the session (0.01 s by default, so 60 s = 6000 steps). |
| `--seed S` | Fix the random numbers (sensor noise, random initial weights, …) so the run is exactly repeatable. Without it, every run differs slightly whenever the brain or sensors use randomness. |
| `--out FILE.csv` | Save the values of every step to a CSV file (see below). |

---

## The CSV file

With `--out run.csv`, the file has **one row per agent per step**:

| Column | Meaning |
|---|---|
| `step` | Step number, starting at 1 |
| `t` | Simulated time in seconds |
| `agent` | Agent number (0, 1, 2, … in the order of the agent table) |
| `x`, `y`, `theta` | Position (metres) and heading (radians) after the step |
| `mL`, `mR` | Left and right motor commands used in that step |
| `<sensor>_<i>` | Every sensor reading, one column per value — e.g. `bumper_0`, `bumper_1`. Camera pixels are left out. |

In a multi-agent session, agents with different sensors leave each other's sensor columns empty.

Reading it in Python:

```python
import pandas as pd

df = pd.read_csv('run.csv')
agent0 = df[df.agent == 0]
agent0.plot(x='x', y='y')            # the path the robot took
```

---

## Repeatable runs

Two runs with the same session and the same `--seed` produce identical files. (Your own sessions are in your files folder — `~/LBPSimulator/configs/` by default, see [Your files](tour.md#my-files); the built-in ones are in `configs/Tutorials/` and `configs/Demos/`.) That makes it easy to:

- check that a change to your brain really changed the behaviour (and not just the noise),
- run the same experiment over several seeds and average the results:

```bash
python src/headless.py ~/LBPSimulator/configs/experiment_1.json --seconds 120 --seed 1 --out seed1.csv
python src/headless.py ~/LBPSimulator/configs/experiment_1.json --seconds 120 --seed 2 --out seed2.csv
python src/headless.py ~/LBPSimulator/configs/experiment_1.json --seconds 120 --seed 3 --out seed3.csv
```

---

## Scripting experiments in Python

For anything beyond a single run — sweeping a parameter, stopping when a goal is reached, computing a score — load the session yourself and step it in a loop. Run the script from `simulation/2d/`:

```python
import os, sys
sys.path[:0] = ['src', 'brains']

import numpy as np
from headless import seed_everything
from session_loader import build_simulation

for gain in [10, 20, 40, 80]:
    seed_everything(0)
    sim = build_simulation(os.path.expanduser('~/LBPSimulator/configs/experiment_1.json'))
    sim.agents[0].brain.gain_ipsi = gain          # a brain parameter (slider) by name — here BrainBraitenberg's

    distance = 0.0
    for _ in range(int(30 / sim.sim_cfg.dt)):    # 30 simulated seconds
        x0, y0 = sim.agents[0].bot_pos[:2]
        sim.step()
        x1, y1 = sim.agents[0].bot_pos[:2]
        distance += np.hypot(x1 - x0, y1 - y0)
    print(f'gain={gain}: travelled {distance:.2f} m')
    sim.close()
```

Useful pieces:

| Code | What it gives you |
|---|---|
| `build_simulation(path)` | The loaded session, reset to t = 0, ready to step |
| `sim.step()` | Advance every agent by one `dt`; returns one dict per agent with its motor commands (`mL`, `mR`) and sensor readings (`<sensor>_<i>`) |
| `sim.agents[i].bot_pos` | `[x, y, theta]` of agent *i* |
| `sim.agents[i].brain` | Its brain — brain parameters are attributes with the slider's name; network layers are attributes with the layer's name (e.g. `brain.motor.output`) |
| `sim.sim_time`, `sim.time_index` | Simulated seconds and steps since reset |
| `sim.reset()` | Back to t = 0 with the same agents and brains |
| `sim.close()` | Release the 3-D engine when you're done |

If you only need a callback per step, `headless.run(path, seconds, seed=…, out=…, on_step=lambda sim, values: …)` does the loading, stepping and CSV writing for you. Note that `run()` switches the working directory to `simulation/2d/`.

---

## What's different from the app

- **No keyboard, no network, no real robot.** Every agent is driven by its own brain. Manual driving, Host/Client networking and real-robot mode only exist in the app.
- **Cameras with `fps = 0`** ("as fast as possible") render after every step, since there is no screen refresh to pace them. Cameras with a fixed `fps` behave exactly as in the app.
- **Speed settings don't apply.** "×N" and "Real time" only pace the app's display; a headless run always goes as fast as it can.
- **No task.** The Task tab's choice isn't stored in session files, so a headless run starts without one. In a script you can set one: `from tasks import load_task; sim.task = load_task('task_circle_object'); sim.task.setup(sim.world, sim.sim_cfg)` (the name is the file name in `src/tasks/`).
- **Layer outputs aren't in the CSV yet** — only positions, motor commands and sensor readings. To record a layer, read it in a Python loop (see above), e.g. `sim.agents[0].brain.motor.output`.
