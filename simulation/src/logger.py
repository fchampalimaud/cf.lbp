"""
logger.py — structured data logger for the 2-D LBP simulator.

Records per-tick state for every agent to a JSON-lines file (v2 format).
Each row is a self-contained dict separated by newlines.

Usage (from SimController):
    logger = SimLogger()
    logger.start('logs/run_001.jsonl', arena_scale=5.0, arena_round=False,
                 agents=[{'id': 0, 'name': 'Agent 1', 'color': '#4a7fcb'}])
    # inside _tick:
    logger.log(time_index, agents, raws, selected_id, world)
    logger.stop()

Format
------
First row — arena + agent metadata:
    {"_meta": true, "version": 2, "arena_scale": 5.0, "arena_round": false,
     "agents": [{"id": 0, "name": "Agent 1", "color": "#4a7fcb"}, ...]}

If an agent joins mid-recording a registration row is emitted before that tick:
    {"_agent": true, "id": 1, "name": "Agent 2", "color": "#cc4444"}

Tick rows:
    {
        "t":     <time step index>,
        "sel":   <selected agent id>,
        "agents": [
            {"id": 0, "x": .., "y": .., "theta": .., "mL": .., "mR": ..,
             "signals": {<name>: <float>, ...}},
            ...
        ],
        "objects": [{"x":…,"y":…,"r":…,"color":…}, …],
        "patches": [{"x":…,"y":…,"r":…,"label":…,"intensity":…}, …]
    }

`signals` contains only scalar-valued sensor outputs; non-scalar values
(e.g. camera arrays) are silently skipped.
"""

import json
from datetime import datetime
from typing import Optional


def _scalar_signals(raw: dict) -> dict:
    """Return only the scalar float-castable entries from a raw signal dict."""
    out = {}
    for k, v in raw.items():
        if k in ('mL', 'mR'):
            continue
        try:
            out[k] = round(float(v), 4)
        except (TypeError, ValueError):
            pass
    return out


class SimLogger:
    def __init__(self):
        self._file      = None
        self._path      = None
        self.running    = False
        self._row_count = 0
        self._known_ids: set = set()

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self, path: Optional[str] = None,
              arena_scale: float = 5.0, arena_round: bool = False,
              agents: list = ()):
        """Open a log file and start recording. Auto-names if path is None.

        agents: list of dicts with keys id, name, color — the registry
                agents present at the moment recording starts.
        """
        if self.running:
            self.stop()
        if path is None:
            import data_paths
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = str(data_paths.user_path('logs', f'sim_{ts}.jsonl'))
        self._path      = path
        self._file      = open(path, 'w', encoding='utf-8')
        self._row_count = 0
        self._known_ids = {a['id'] for a in agents}
        self.running    = True
        meta = {
            '_meta':       True,
            'version':     2,
            'arena_scale': arena_scale,
            'arena_round': arena_round,
            'agents':      list(agents),
        }
        self._file.write(json.dumps(meta) + '\n')
        print(f'[Logger] Recording → {path}')

    def stop(self):
        """Flush and close the current log file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        if self.running:
            print(f'[Logger] Stopped. {self._row_count} rows → {self._path}')
        self.running = False

    # ── Per-tick recording ────────────────────────────────────────────────────

    def log(self, time_index: int, agents: list, raws: list,
            selected_id: int, world):
        """Write one tick row. Call every tick when running.

        agents      : list[RobotAgent]
        raws        : list[dict] aligned with agents (from sim.step)
        selected_id : id of the currently selected agent
        world       : world object with .objects and .patches
        """
        if not self.running or self._file is None:
            return

        # Announce any agent that joined mid-recording
        for agent in agents:
            if agent.id not in self._known_ids:
                reg = {'_agent': True, 'id': agent.id,
                       'name': agent.name, 'color': agent.color}
                self._file.write(json.dumps(reg) + '\n')
                self._known_ids.add(agent.id)

        agent_rows = []
        for agent, raw in zip(agents, raws):
            x, y, theta = agent.bot_pos
            agent_rows.append({
                'id':      agent.id,
                'x':       round(float(x),     6),
                'y':       round(float(y),     6),
                'theta':   round(float(theta), 6),
                'mL':      round(float(raw.get('mL', 0)), 4),
                'mR':      round(float(raw.get('mR', 0)), 4),
                'signals': _scalar_signals(raw),
            })

        row = {
            't':       time_index,
            'sel':     selected_id,
            'agents':  agent_rows,
            'objects': [
                {'x': o['x'], 'y': o['y'], 'r': o['r'],
                 'color': o.get('color', '')}
                for o in world.objects
            ],
            'patches': [
                {'x': p.get('x', 0), 'y': p.get('y', 0),
                 'r': p.get('r', 0), 'label': p.get('label', ''),
                 'intensity': p.get('intensity', 1.0)}
                for p in world.patches
            ],
        }
        self._file.write(json.dumps(row) + '\n')
        self._row_count += 1

    @property
    def row_count(self):
        return self._row_count
