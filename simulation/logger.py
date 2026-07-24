"""
logger.py — structured data logger for the 2-D Braitenberg simulator.

Records per-tick robot state and world landmark state to a JSON-lines file.
Each row is a self-contained dict; rows are separated by newlines.

Usage (from SimulatorApp):
    logger = SimLogger()
    logger.start('logs/run_001.jsonl')
    # inside _tick:
    logger.log(time_index, bot_pos, raw_signals, world)
    logger.stop()

Format per row (JSON):
    {
        "t":     <time step index>,
        "x":     <float>,
        "y":     <float>,
        "theta": <float>,
        "mL":    <float>,
        "mR":    <float>,
        "signals": {<name>: <float>, ...},   # sensor outputs etc.
        "objects": [{"x":…,"y":…,"r":…,"color":…}, …],
        "patches": [{"x":…,"y":…,"r":…,"label":…,"intensity":…}, …]
    }
"""

import json
import os
from datetime import datetime
from typing import Optional


class SimLogger:
    def __init__(self):
        self._file    = None
        self._path    = None
        self.running  = False
        self._row_count = 0

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self, path: Optional[str] = None,
              arena_scale: float = 5.0, arena_round: bool = False):
        """Open a log file and start recording. Auto-names if path is None."""
        if self.running:
            self.stop()
        os.makedirs('logs', exist_ok=True)
        if path is None:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = f'logs/sim_{ts}.jsonl'
        self._path      = path
        self._file      = open(path, 'w', encoding='utf-8')
        self._row_count = 0
        self.running    = True
        # First row: arena metadata (not a data tick)
        meta = {'_meta': True, 'arena_scale': arena_scale, 'arena_round': arena_round}
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

    def log(self, time_index: int, bot_pos: list, raw_signals: dict, world):
        """Write one row. Call every tick when running."""
        if not self.running or self._file is None:
            return
        x, y, theta = bot_pos
        row = {
            't':       time_index,
            'x':       round(float(x),     6),
            'y':       round(float(y),     6),
            'theta':   round(float(theta), 6),
            'mL':      round(float(raw_signals.get('mL', 0)), 4),
            'mR':      round(float(raw_signals.get('mR', 0)), 4),
            'signals': {k: round(float(v), 4)
                        for k, v in raw_signals.items()
                        if k not in ('mL', 'mR')},
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
