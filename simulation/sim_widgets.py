"""
sim_widgets.py — reusable Qt widget helpers for the LBP simulator.

No simulation logic; pure UI components with no dependency on SimulatorApp.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QDoubleSpinBox, QScrollArea, QWidget, QVBoxLayout,
    QGroupBox, QTabWidget, QHBoxLayout, QLabel,
)
from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtGui import QColor

from sim_constants import C, _CHAN_PALETTE


_MONETARY_SCALE = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]


class _ManualKeyFilter(QObject):
    """Event filter that tracks held WASD/Space keys for manual robot control."""
    def __init__(self, held_keys, parent=None):
        super().__init__(parent)
        self._held = held_keys

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            self._held.add(event.key())
        elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            self._held.discard(event.key())
        return False


class MonetarySpinBox(QDoubleSpinBox):
    """SpinBox that steps through a monetary scale: 0, 0.1, 0.2, 0.5, 1, 2, 5, 10 …"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(2)
        self.setMinimum(0.0)
        self.setMaximum(200.0)
        self.setSingleStep(0.1)

    def stepBy(self, steps):
        cur = self.value()
        nearest = min(range(len(_MONETARY_SCALE)),
                      key=lambda i: abs(_MONETARY_SCALE[i] - cur))
        new_idx = max(0, min(len(_MONETARY_SCALE) - 1, nearest + steps))
        self.setValue(_MONETARY_SCALE[new_idx])


class OscilloscopeWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(QColor(C['surface']))
        self.getPlotItem().showGrid(x=False, y=True, alpha=0.3)
        self.getPlotItem().hideAxis('bottom')
        self.getPlotItem().getAxis('left').setStyle(tickTextOffset=4)
        self._curves  = {}
        self._buf_len = 1000

    def setup_channels(self, channels, channel_colors, buf_len=1000):
        self._buf_len = buf_len
        self.clear()
        self._curves = {}
        for ch in channels:
            color = channel_colors.get(ch, '#888888')
            curve = self.plot([], [], pen=pg.mkPen(color, width=1.5), name=ch,
                              autoDownsample=True, downsampleMethod='peak',
                              clipToView=True, skipFiniteCheck=True,
                              antialias=False)
            self._curves[ch] = curve
        vb = self.getViewBox()
        vb.disableAutoRange()
        vb.setXRange(0, buf_len, padding=0)
        vb.setYRange(-0.11, 0.11, padding=0)
        self._last_y_max = 0.1

    def update_channels(self, trace_data, multipliers, channels):
        cur_max = 0.1
        for k in channels:
            curve = self._curves.get(k)
            if curve is None:
                continue
            mult = multipliers.get(k, 0.1)
            if mult > 0:
                y_vals = np.array(trace_data[k]) * mult
                curve.setData(y=y_vals, skipFiniteCheck=True)
                curve.setVisible(True)
                m = float(np.max(np.abs(y_vals))) if len(y_vals) else 0.0
                if m > cur_max:
                    cur_max = m
            else:
                curve.setVisible(False)
        new_y_max = cur_max * 1.1
        if abs(new_y_max - self._last_y_max) > self._last_y_max * 0.05:
            self.getViewBox().setYRange(-new_y_max, new_y_max, padding=0)
            self._last_y_max = new_y_max


class ControlPanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(260)

        self._inner  = QWidget()
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(6)
        self.setWidget(self._inner)

    def add_group(self, title, target=None):
        gb = QGroupBox(title)
        gb.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {C['border']};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 6px;
                background: {C['surface']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                color: {C['dark']};
            }}
        """)
        vl = QVBoxLayout()
        vl.setSpacing(4)
        gb.setLayout(vl)
        (target or self._layout).addWidget(gb)
        return gb, vl

    def add_tab_widget(self, tab_names):
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {C['border']};
                background: {C['bg']};
                top: -1px;
            }}
            QTabBar::tab {{
                background: {C['surface']};
                color: {C['dark']};
                padding: 5px 10px;
                border: 1px solid {C['border']};
                border-bottom: none;
                border-radius: 3px 3px 0 0;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background: {C['bg']};
                color: {C['primary']};
                font-weight: bold;
            }}
        """)
        vls = []
        for name in tab_names:
            page = QWidget()
            vl   = QVBoxLayout(page)
            vl.setContentsMargins(4, 6, 4, 6)
            vl.setSpacing(6)
            tabs.addTab(page, name)
            vls.append(vl)
        self._layout.addWidget(tabs)
        return vls

    def add_stretch(self):
        self._layout.addStretch()
