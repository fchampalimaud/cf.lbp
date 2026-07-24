"""
trajectory_viz.py — Trajectory visualiser for the 2-D Braitenberg simulator.

Architecture (each layer is independently importable):

  TrajectoryLoader   – parse a sim .jsonl log; expose arrays + world snapshot
  _make_patch_item   – translucent filled circle (pyqtgraph GraphicsObject)
  TrajectoryWindow   – pyqtgraph dialog matching the arena look exactly
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

    Patches and objects are read from the first row — they do not change
    during a recording session.
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

        # First row may be an arena metadata header written by SimLogger.start()
        if rows[0].get('_meta'):
            meta = rows.pop(0)
            self._arena_scale = float(meta.get('arena_scale', 5.0))
            self._arena_round = bool(meta.get('arena_round', False))
        else:
            # Older file without header — infer scale from trajectory extent
            self._arena_round = False
            self._arena_scale = None   # resolved after arrays are built

        if not rows:
            raise ValueError(f"Log file has no data rows: {path}")

        self._xs     = np.array([r['x']          for r in rows], dtype=float)
        self._ys     = np.array([r['y']          for r in rows], dtype=float)
        self._thetas = np.array([r['theta']       for r in rows], dtype=float)
        self._times  = np.array([r['t']           for r in rows], dtype=float)
        self._mL     = np.array([r.get('mL', 0.0) for r in rows], dtype=float)
        self._mR     = np.array([r.get('mR', 0.0) for r in rows], dtype=float)
        first = rows[0]
        self._patches = first.get('patches', [])
        self._objects = first.get('objects', [])

        if self._arena_scale is None:
            self._arena_scale = float(np.ceil(
                max(np.max(np.abs(self._xs)), np.max(np.abs(self._ys))) + 1.0))

    @property
    def path(self) -> str:            return self._path
    @property
    def xs(self) -> np.ndarray:       return self._xs
    @property
    def ys(self) -> np.ndarray:       return self._ys
    @property
    def thetas(self) -> np.ndarray:   return self._thetas
    @property
    def times(self) -> np.ndarray:    return self._times
    @property
    def patches(self) -> list:        return self._patches
    @property
    def objects(self) -> list:        return self._objects
    @property
    def n_steps(self) -> int:         return len(self._xs)
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
    pyqtgraph dialog that mirrors the arena background then overlays a trajectory.

    Usage:
        win = TrajectoryWindow.for_latest(log_dir='logs', parent=app_window)
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

        # Stash constants so _draw() can use them without re-importing
        self._trail_color      = _TRAIL_COLOR
        self._gradient_colors  = GRADIENT_COLORS
        self._CircleItem       = CircleItem

        dlg = QDialog(self._parent)
        dlg.setWindowTitle(f"Trajectory — {Path(self._traj.path).name}")
        dlg.resize(620, 640)

        root = QVBoxLayout(dlg)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # pyqtgraph canvas — same background and style as ArenaWidget
        canvas = pg.GraphicsLayoutWidget()
        canvas.setBackground(QColor(C['bg']))
        self._plot = canvas.addPlot()
        self._plot.setAspectLocked(True)
        self._plot.hideAxis('bottom')
        self._plot.hideAxis('left')
        root.addWidget(canvas)

        # Info bar at bottom
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

        # label → button-background hex; used as fill when RGB color unavailable
        label_hex = {letter: bg for letter, _, _, bg in self._gradient_colors}

        # Gradient patches — translucent filled circles
        for p in traj.patches:
            col = p.get('color')           # saved as [r,g,b] float list if present
            if col and isinstance(col, (list, tuple)) and len(col) >= 3:
                hex_c = f"#{int(col[0]*255):02x}{int(col[1]*255):02x}{int(col[2]*255):02x}"
            else:
                hex_c = label_hex.get(p.get('label', ''), '#DDDDAA')
            item = _make_patch_item(p['x'], p['y'], p['r'], hex_c, alpha=90)
            item.setZValue(1)
            plot.addItem(item)

        # Objects — solid circles, identical to the arena
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

        # Trajectory — thin dark line matching the arena trail color
        line = pg.PlotDataItem(traj.xs, traj.ys,
                               pen=pg.mkPen(self._trail_color, width=2))
        line.setZValue(5)
        plot.addItem(line)

        # Arena boundary — same pen and shape as ArenaWidget
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

        # Set view to match ArenaWidget exactly
        plot.getViewBox().setRange(
            QRectF(-lim * 1.05, -lim * 1.05, lim * 2.1, lim * 2.1), padding=0)
        plot.getViewBox().disableAutoRange()

        self._info_lbl.setText(f"{Path(traj.path).name}  ·  {traj.n_steps} steps")

    # ── File picker ──────────────────────────────────────────────────────────

    def _pick_file(self, dlg):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getOpenFileName(
            dlg, "Open trajectory log", "logs", "JSON-lines (*.jsonl)")
        if not path:
            return
        try:
            self._traj = TrajectoryLoader.from_file(path)
            dlg.setWindowTitle(f"Trajectory — {Path(path).name}")
            self._draw()
        except Exception as exc:
            QMessageBox.warning(dlg, "Load error", str(exc))
