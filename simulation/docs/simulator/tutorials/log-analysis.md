# Analysing a Logged Run

The **Logger** in the Session tab records every tick of the simulation to a `.jsonl` file in `logs/` inside your files folder. This page shows how to load one and plot the robots' trajectories together with the arena patches using plain Python and matplotlib.

---

## Log format

Each file is a sequence of JSON lines. The first line is a metadata header; subsequent lines are either agent-registration events or tick rows.

```json
{"_meta": true, "version": 2, "arena_scale": 5.0, "arena_round": false,
 "agents": [{"id": 0, "name": "Agent 1", "color": "#4a7fcb"},
            {"id": 1, "name": "Agent 2", "color": "#cc4444"}]}
{"t": 0, "sel": 0,
 "agents": [{"id": 0, "x": 0.0, "y": 0.0, "theta": 1.57, "mL": 0.0, "mR": 0.0,
             "signals": {"light_L": 0.3, "light_R": 0.1}},
            {"id": 1, "x": 1.0, "y": 0.5, "theta": 0.0, "mL": 0.1, "mR": 0.2,
             "signals": {"light_L": 0.6, "light_R": 0.5}}],
 "patches": [{"x": 2.0, "y": 1.0, "r": 0.5, "label": "food", "intensity": 1.0}],
 "objects": [{"x": -1.0, "y": 0.5, "r": 0.15, "color": "#888888"}]}
...
```

If an agent joins while recording is already running, a registration row is emitted before the first tick that includes it:

```json
{"_agent": true, "id": 2, "name": "Agent 3", "color": "#44cc88"}
```

| Field | Type | Meaning |
|-------|------|---------|
| `_meta.version` | int | Format version (2) |
| `_meta.agents` | list | Agents present when recording started |
| `t` | int | Tick index |
| `sel` | int | Id of the selected agent on this tick |
| `agents` | list | Per-agent state for this tick |
| `agents[i].id` | int | Stable agent id |
| `agents[i].x`, `y` | float | Position (metres) |
| `agents[i].theta` | float | Heading (radians) |
| `agents[i].mL`, `mR` | float | Motor commands |
| `agents[i].signals` | dict | Sensor outputs (scalar channels only) |
| `patches` | list | Gradient patches — same for every row; read from row 0 |
| `objects` | list | Solid objects — same for every row; read from row 0 |

---

## Loading the file

```python
import json
from collections import defaultdict
from pathlib import Path

LOG_FILE = r"C:\Users\you\LBPSimulator\logs\sim_20260101_120000.jsonl"

agent_info = {}   # id → {name, color}
ticks      = []   # list of tick dicts
patches    = []
objects    = []
meta       = {}

with open(LOG_FILE, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if row.get("_meta"):
            meta   = row
            for a in row.get("agents", []):
                agent_info[a["id"]] = {"name": a["name"], "color": a["color"]}
        elif row.get("_agent"):
            agent_info[row["id"]] = {"name": row["name"], "color": row["color"]}
        else:
            if not ticks:
                patches = row.get("patches", [])
                objects = row.get("objects", [])
            ticks.append(row)

arena_scale = meta.get("arena_scale", 5.0)
arena_round = meta.get("arena_round", False)

# Build per-agent arrays
tracks = defaultdict(lambda: {"x": [], "y": [], "theta": [], "t": []})
for row in ticks:
    for a in row.get("agents", []):
        tracks[a["id"]]["x"].append(a["x"])
        tracks[a["id"]]["y"].append(a["y"])
        tracks[a["id"]]["theta"].append(a["theta"])
        tracks[a["id"]]["t"].append(row["t"])
```

---

## Plotting all trajectories

```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ax.set_xlim(-arena_scale, arena_scale)
ax.set_ylim(-arena_scale, arena_scale)
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

# Arena border
if arena_round:
    border = plt.Circle((0, 0), arena_scale, color="#555", fill=False, lw=1.5)
    ax.add_patch(border)
else:
    border = plt.Rectangle((-arena_scale, -arena_scale),
                            2 * arena_scale, 2 * arena_scale,
                            color="#555", fill=False, lw=1.5)
    ax.add_patch(border)

# Gradient patches
for p in patches:
    halo = plt.Circle(
        (p["x"], p["y"]), p["r"],
        color="#44cc88", alpha=0.25 * p.get("intensity", 1.0), zorder=1,
    )
    ax.add_patch(halo)
    ax.text(p["x"], p["y"], p.get("label", ""), color="#aaffcc",
            ha="center", va="center", fontsize=7, zorder=2)

# Solid objects
for o in objects:
    circ = plt.Circle(
        (o["x"], o["y"]), o["r"],
        color=o.get("color") or "#888", alpha=0.9, zorder=3,
    )
    ax.add_patch(circ)

# Trajectories — one per agent, using its recorded color
for aid, track in tracks.items():
    info   = agent_info.get(aid, {})
    color  = info.get("color") or "#4fa3e0"
    xs, ys = np.array(track["x"]), np.array(track["y"])
    ax.plot(xs, ys, color=color, lw=0.8, alpha=0.8, zorder=4,
            label=info.get("name", f"Agent {aid}"))
    ax.scatter(xs[0],  ys[0],  color=color, s=40, zorder=5, marker="o")
    ax.scatter(xs[-1], ys[-1], color=color, s=40, zorder=5, marker="x")

ax.legend(facecolor="#222", labelcolor="white", framealpha=0.6)
ax.set_title("Robot trajectories", color="white")
plt.tight_layout()
out_path = Path(LOG_FILE).with_name("trajectories.png")
plt.savefig(out_path, dpi=150)
plt.show()
```

---

## Colouring a trajectory by motor command

Replace the trajectory section above with this snippet to colour each segment by the signed turn rate (`mR - mL`) for every agent:

```python
from matplotlib.collections import LineCollection
from matplotlib.cm import coolwarm

norm = plt.Normalize(vmin=-1, vmax=1)
last_lc = None

# Build per-agent turn arrays from the tick rows
agent_turns = {}  # id → list of (mR - mL)
for row in ticks:
    for a in row.get("agents", []):
        agent_turns.setdefault(a["id"], []).append(a["mR"] - a["mL"])

for aid, track in tracks.items():
    xs, ys = np.array(track["x"]), np.array(track["y"])
    turn   = np.array(agent_turns.get(aid, []))
    if len(xs) < 2 or len(turn) < 1:
        continue
    segments = np.stack([np.c_[xs[:-1], ys[:-1]],
                         np.c_[xs[1:],  ys[1:]]], axis=1)
    last_lc = LineCollection(segments, cmap=coolwarm, norm=norm, lw=1.2, zorder=4)
    last_lc.set_array(turn[:len(segments)])
    ax.add_collection(last_lc)

if last_lc is not None:
    _ = fig.colorbar(last_lc, ax=ax, label="turn (mR − mL)")
```

Blue segments are left turns, red segments are right turns, white is straight.

---

## Reading signal channels

Each agent's `signals` dict contains its scalar sensor outputs for that tick. To extract a specific channel over time for one agent, select the agent's id and the channel you want to look at:

```python
AID     = 0
CHANNEL = "light_L"

values = []
for row in ticks:
    for a in row.get("agents", []):
        if a["id"] == AID:
            values.append(a["signals"].get(CHANNEL, 0.0))

fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
axes[0].plot(values, label=CHANNEL)
axes[0].set_ylabel(CHANNEL)

mL = [a["mL"] for row in ticks for a in row["agents"] if a["id"] == AID]
mR = [a["mR"] for row in ticks for a in row["agents"] if a["id"] == AID]
axes[1].plot(mL, label="mL")
axes[1].plot(mR, label="mR")
axes[1].set_ylabel("motors")
axes[1].set_xlabel("tick")
for ax in axes:
    ax.legend()
plt.tight_layout()
plt.show()
```
