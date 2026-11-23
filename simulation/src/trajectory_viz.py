"""
trajectory_viz.py — Trajectory visualiser for the 2-D LBP simulator.

Architecture (each layer is independently importable):

  TrajectoryLoader   – parse a sim .jsonl log; expose arrays + world snapshot
  _make_patch_item   – translucent filled circle (pyqtgraph GraphicsObject)
  TrajectoryWindow   – pyqtgraph dialog matching the arena look exactly

v2 format: each tick row has an "agents" list; v1 (legacy) rows have top-level
x/y/theta. Both are handled transparently. Multi-agent logs produce one
trajectory per agent; single-agent logs produce one trajectory with id=0.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data layer
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryLoader:
    """Parse a sim JSONL log; expose trajectory arrays and world snapshot.

    Patches and objects are read from the first tick row — they don't change.

    Multi-agent (v2) attributes
    ---------------------------
    tracks      : dict[int, dict]  — per-agent id → {'t', 'x', 'y', 'theta'}
                  as numpy arrays
    agent_info  : dict[int, dict]  — per-agent id → {'name', 'color'}

    Single-agent compatibility (always present)
    -------------------------------------------
    xs, ys, thetas, times  — arrays for the first agent in the file
    n_steps                — length of the first agent's track
    """

    def __init__(self, path: str):
        self._path = path
        self._load(path)

    @classmethod
    def from_file(cls, path: str) -> 'TrajectoryLoader':
        return cls(path)

    @classmethod
    def latest_in_dir(cls, log_dir: str = 'logs') -> Optional['TrajectoryLoader']:
        """Return a loader for the most-recently-modified .jsonl in log_dir."""
        d = Path(log_dir)
        if not d.is_dir():
            return None
        files = sorted(d.glob('*.jsonl'), key=lambda p: p.stat().st_mtime)
        return cls(str(files[-1])) if files else None

    def _load(self, path: str):
        rows: list[dict] = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
        if not rows:
            raise ValueError(f"Empty log file: {path}")

        # Separate metadata from data rows
        self._arena_scale: Optional[float] = None
        self._arena_round = False
        self.agent_info: dict[int, dict] = {}  # id → {name, color}

        tick_rows: list[dict] = []
        for row in rows:
            if row.get('_meta'):
                self._arena_scale = float(row.get('arena_scale', 5.0))
                self._arena_round = bool(row.get('arena_round', False))
                for a in row.get('agents', []):
                    self.agent_info[a['id']] = {
                        'name':  a.get('name', f"Agent {a['id']}"),
                        'color': a.get('color', ''),
                    }
            elif row.get('_agent'):
                self.agent_info[row['id']] = {
                    'name':  row.get('name', f"Agent {row['id']}"),
                    'color': row.get('color', ''),
                }
            else:
                tick_rows.append(row)

        if not tick_rows:
            raise ValueError(f"Log file has no data rows: {path}")

        # Build per-agent tracks — works for v2 (agents list) and v1 (top-level x/y)
        accum: dict[int, dict] = {}  # id → lists of t, x, y, theta

        def _ensure(aid):
            if aid not in accum:
                accum[aid] = {'t': [], 'x': [], 'y': [], 'theta': []}
            return accum[aid]

        first = tick_rows[0]
        if 'agents' in first:
            # v2
            for row in tick_rows:
                for a in row.get('agents', []):
                    buf = _ensure(a['id'])
                    buf['t'].append(row['t'])
                    buf['x'].append(a['x'])
                    buf['y'].append(a['y'])
                    buf['theta'].append(a['theta'])
        else:
            # v1 — single agent, synthesise id=0
            self.agent_info.setdefault(0, {'name': 'Agent 1', 'color': ''})
            buf = _ensure(0)
            for row in tick_rows:
                buf['t'].append(row['t'])
                buf['x'].append(row['x'])
                buf['y'].append(row['y'])
                buf['theta'].append(row.get('theta', 0.0))

        self.tracks: dict[int, dict] = {
            aid: {k: np.array(v, dtype=float) for k, v in buf.items()}
            for aid, buf in accum.items()
        }

        # Ensure agent_info has an entry for every track (mid-recording joiners
        # write an _agent row, but very old files may lack one)
        for aid in self.tracks:
            self.agent_info.setdefault(aid, {'name': f'Agent {aid}', 'color': ''})

        # Static world snapshot from the first tick row
        self._patches = first.get('patches', [])
        self._objects = first.get('objects', [])

        # Infer scale from all tracks if the metadata row was absent
        if self._arena_scale is None:
            all_x = np.concatenate([t['x'] for t in self.tracks.values()])
            all_y = np.concatenate([t['y'] for t in self.tracks.values()])
            self._arena_scale = float(np.ceil(
                max(np.max(np.abs(all_x)), np.max(np.abs(all_y))) + 1.0))

    # ── Single-agent compat props (first agent in insertion order) ────────────

    def _first_track(self) -> dict:
        return next(iter(self.tracks.values()))

    @property
    def xs(self) -> np.ndarray:      return self._first_track()['x']
    @property
    def ys(self) -> np.ndarray:      return self._first_track()['y']
    @property
    def thetas(self) -> np.ndarray:  return self._first_track()['theta']
    @property
    def times(self) -> np.ndarray:   return self._first_track()['t']
    @property
    def n_steps(self) -> int:        return len(self._first_track()['x'])

    # ── Other props ───────────────────────────────────────────────────────────

    @property
    def path(self) -> str:            return self._path
    @property
    def patches(self) -> list:        return self._patches
    @property
    def objects(self) -> list:        return self._objects
    @property
    def arena_scale(self) -> float:   return self._arena_scale
    @property
    def arena_round(self) -> bool:    return self._arena_round


# ─────────────────────────────────────────────────────────────────────────────
# Graphics helper — translucent patch circle
# ─────────────────────────────────────────────────────────────────────────────

def _make_patch_item(x: float, y: float, r: float, hex_color: str, alpha: int = 80):
    """Return a pyqtgraph GraphicsObject: a translucent filled circle for a gradient patch."""
    import pyqtgraph as pg
    from PySide6.QtCore import QRectF, QPointF, Qt
    from PySide6.QtGui import QColor, QBrush, QPainter

    _x, _y, _r = x, y, r
    _rect = QRectF(_x - _r, _y - _r, 2 * _r, 2 * _r)
    c = QColor(hex_color)
    c.setAlpha(alpha)
    _brush = QBrush(c)

    class _PatchCircle(pg.GraphicsObject):
        def boundingRect(self):
            return _rect

        def paint(self, p, option, widget):
            p.setRenderHint(QPainter.Antialiasing)
            p.setBrush(_brush)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(_x, _y), _r, _r)

    return _PatchCircle()


# ─────────────────────────────────────────────────────────────────────────────
# Qt viewer
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryWindow:
    """
    pyqtgraph dialog that mirrors the arena background then overlays trajectories.

    One coloured line per agent; a legend is shown when there are multiple agents.

    Usage:
        win = TrajectoryWindow.for_latest(log_dir=str(__import__('data_paths').user_path('logs')),
                                          parent=app_window)
        win.show()          # opens / raises the dialog
    """

    def __init__(self, traj: TrajectoryLoader, parent=None):
        self._traj   = traj
        self._parent = parent
        self._dlg    = None   # built lazily on first show()

    @classmethod
    def for_latest(cls, log_dir: str = 'logs',
                   parent=None) -> Optional['TrajectoryWindow']:
        traj = TrajectoryLoader.latest_in_dir(log_dir)
        return cls(traj, parent=parent) if traj is not None else None

    # ── Public ───────────────────────────────────────────────────────────────

    def show(self):
        if self._dlg is None:
            self._dlg = self._build_dialog()
        self._dlg.show()
        self._dlg.raise_()

    # ── Dialog construction ──────────────────────────────────────────────────

    def _build_dialog(self):
        from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                       QPushButton, QLabel)
        from PySide6.QtGui import QColor
        import pyqtgraph as pg
        from sim_constants import C, _TRAIL_COLOR, GRADIENT_COLORS
        from arena_widget import CircleItem

        self._trail_color      = _TRAIL_COLOR
        self._gradient_colors  = GRADIENT_COLORS
        self._CircleItem       = CircleItem

        dlg = QDialog(self._parent)
        dlg.setWindowTitle(f"Trajectory — {Path(self._traj.path).name}")
        dlg.resize(620, 640)

        root = QVBoxLayout(dlg)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        canvas = pg.GraphicsLayoutWidget()
        canvas.setBackground(QColor(C['bg']))
        self._plot = canvas.addPlot()
        self._plot.setAspectLocked(True)
        self._plot.hideAxis('bottom')
        self._plot.hideAxis('left')
        root.addWidget(canvas)

        bar = QHBoxLayout()
        self._info_lbl = QLabel()
        self._info_lbl.setStyleSheet("color: grey; font-size: 8pt;")
        bar.addWidget(self._info_lbl)
        bar.addStretch()
        btn_open = QPushButton("Open file…")
        btn_open.setFixedHeight(24)
        btn_open.clicked.connect(lambda: self._pick_file(dlg))
        bar.addWidget(btn_open)
        root.addLayout(bar)

        self._draw()
        return dlg

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw(self):
        import pyqtgraph as pg
        from PySide6.QtCore import QRectF

        plot = self._plot
        traj = self._traj
        plot.clear()

        label_hex = {letter: bg for letter, _, _, bg in self._gradient_colors}

        # Gradient patches
        for p in traj.patches:
            col = p.get('color')
            if col and isinstance(col, (list, tuple)) and len(col) >= 3:
                hex_c = f"#{int(col[0]*255):02x}{int(col[1]*255):02x}{int(col[2]*255):02x}"
            else:
                hex_c = label_hex.get(p.get('label', ''), '#DDDDAA')
            item = _make_patch_item(p['x'], p['y'], p['r'], hex_c, alpha=90)
            item.setZValue(1)
            plot.addItem(item)

        # Objects
        for o in traj.objects:
            col = o.get('color', [1.0, 0.0, 0.0])
            if isinstance(col, (list, tuple)) and len(col) >= 3:
                hex_c = f"#{int(col[0]*255):02x}{int(col[1]*255):02x}{int(col[2]*255):02x}"
            elif isinstance(col, str) and col.startswith('#'):
                hex_c = col
            else:
                hex_c = '#888888'
            item = self._CircleItem(o['x'], o['y'], o['r'], hex_c)
            item.setZValue(2)
            plot.addItem(item)

        # Trajectories — one per agent
        multi = len(traj.tracks) > 1
        if multi:
            legend = plot.addLegend(offset=(10, 10))

        for aid, track in traj.tracks.items():
            info  = traj.agent_info.get(aid, {})
            color = info.get('color') or self._trail_color
            name  = info.get('name') or f'Agent {aid}'
            pen   = pg.mkPen(color, width=2)
            line  = pg.PlotDataItem(track['x'], track['y'], pen=pen, name=name)
            line.setZValue(5)
            plot.addItem(line)

        # Arena boundary
        from sim_constants import C as _C
        lim = traj.arena_scale
        if traj.arena_round:
            angles = np.linspace(0, 2 * np.pi, 200)
            bx = lim * np.cos(angles)
            by = lim * np.sin(angles)
        else:
            bx = [-lim, lim, lim, -lim, -lim]
            by = [-lim, -lim, lim, lim, -lim]
        boundary = pg.PlotDataItem(bx, by, pen=pg.mkPen(_C['dark'], width=2))
        boundary.setZValue(3)
        plot.addItem(boundary)

        plot.getViewBox().setRange(
            QRectF(-lim * 1.05, -lim * 1.05, lim * 2.1, lim * 2.1), padding=0)
        plot.getViewBox().disableAutoRange()

        total_steps = sum(len(t['x']) for t in traj.tracks.values())
        n_agents    = len(traj.tracks)
        agent_str   = f"{n_agents} agent{'s' if n_agents != 1 else ''}"
        self._info_lbl.setText(
            f"{Path(traj.path).name}  ·  {agent_str}  ·  {total_steps} steps")

    # ── File picker ──────────────────────────────────────────────────────────

    def _pick_file(self, dlg):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getOpenFileName(
            dlg, "Open trajectory log", str(__import__('data_paths').user_path('logs')), "JSON-lines (*.jsonl)")
        if not path:
            return
        try:
            self._traj = TrajectoryLoader.from_file(path)
            dlg.setWindowTitle(f"Trajectory — {Path(path).name}")
            self._draw()
        except Exception as exc:
            QMessageBox.warning(dlg, "Load error", str(exc))
