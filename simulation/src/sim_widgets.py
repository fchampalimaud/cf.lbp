"""
sim_widgets.py — reusable Qt widget helpers for the LBP simulator.

No simulation logic; pure UI components with no dependency on SimulatorApp.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QDoubleSpinBox, QScrollArea, QWidget, QVBoxLayout,
    QGroupBox, QTabWidget, QHBoxLayout, QLabel, QComboBox, QAbstractButton, QSizePolicy,
)
from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtGui import QColor

from sim_constants import C, _CHAN_PALETTE


_MONETARY_SCALE = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]


class _ManualKeyFilter(QObject):
    """Event filter that tracks held WASD/Space keys for manual robot control.

    Installed on the QApplication instance (not a single window) so manual
    drive keeps working regardless of which top-level window — main sim
    window, network visualizer, ... — currently has keyboard focus. Unlike
    _ArrowKeyFilter, no redelivery dedup is needed: add/discard into a set
    are idempotent, so seeing the same physical key event more than once
    (as Qt redelivers ignored events up the parent chain) is harmless.
    """
    def __init__(self, held_keys, parent=None):
        super().__init__(parent)
        self._held = held_keys

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            self._held.add(event.key())
        elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            self._held.discard(event.key())
        return False


class _ArrowKeyFilter(QObject):
    """App-wide event filter for Up/Down arrow keys.

    Installed on the QApplication instance (not a single window) so it fires
    regardless of which top-level window — main sim window, network
    visualizer, ... — currently has keyboard focus.

    Qt redelivers an *ignored* key event to each widget up the parent chain
    (each redelivery is its own sendEvent/notify() call), so an app-wide
    filter would otherwise see one physical key press multiple times. Dedupe
    on (key, timestamp) — identical for every redelivery of the same event,
    distinct for genuinely separate presses/autorepeats.
    """
    def __init__(self, on_step, parent=None):
        super().__init__(parent)
        self._on_step = on_step
        self._last_seen = None

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                seen = (key, event.timestamp())
                if seen != self._last_seen:
                    self._last_seen = seen
                    self._on_step(1 if key == Qt.Key.Key_Up else -1)
        return False


class _CueKeyFilter(QObject):
    """App-wide event filter driving every `ManualCueSensor`'s held-key state.

    Installed unconditionally (like `_ArrowKeyFilter`, unlike `_ManualKeyFilter`
    which only exists while Manual-drive mode is on) so a faked cue works even
    while the brain/network is actually driving the robot. Matches by character
    text (`event.text()`), the same convention already used for gradient/object
    world-edit letter keys, since `ManualCueSensor.key` is a free-form letter
    rather than a fixed Qt.Key_X constant.

    No redelivery dedup needed: like `_ManualKeyFilter`, calling `on_change`
    more than once for the same physical press/release is harmless (idempotent).
    """
    def __init__(self, on_change, parent=None):
        super().__init__(parent)
        self._on_change = on_change

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and not event.isAutoRepeat():
            letter = event.text().upper()
            if letter:
                self._on_change(letter, event.type() == QEvent.Type.KeyPress)
        return False


class ElidedLabel(QLabel):
    """A label that never widens its panel: it takes whatever width the row
    leaves and shows the text with an ellipsis in the middle (full text in
    the tooltip)."""

    def __init__(self, text='', parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._full = ''
        self.setText(text)

    def setText(self, text):
        self._full = text
        self._elide()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._elide()

    def _elide(self):
        shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, max(self.width(), 10))
        QLabel.setText(self, shown)


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
        self._last_y_max = 0.1

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

    def compact(self):
        """Fit the panel's width: combo boxes shrink instead of growing to their
        longest entry, and rows use tight spacing. Call after the panel is
        built (and on widgets added later via compact_widget). Each tab's
        content is pushed to the top, so a short tab doesn't spread its rows
        over the height of the tallest one."""
        for w in self._inner.findChildren(QWidget):
            self.compact_widget(w)
        for vl in getattr(self, '_tab_layouts', []):
            vl.addStretch()
        self._align_row_labels()

    def _align_row_labels(self):
        """Within each group box, give every row label ("World:", "Arena:", ...)
        the width of the widest one, so the controls start in one column."""
        by_group = {}
        for w in self._inner.findChildren(QWidget):
            lay = w.layout()
            if not isinstance(lay, QHBoxLayout) or not lay.count():
                continue
            first = lay.itemAt(0).widget()
            if not (isinstance(first, QLabel) and first.text().rstrip().endswith(':')):
                continue
            group = w.parentWidget()
            while group is not None and not isinstance(group, QGroupBox):
                group = group.parentWidget()
            by_group.setdefault(id(group), []).append(first)
        for labels in by_group.values():
            width = max(lbl.sizeHint().width() for lbl in labels)
            for lbl in labels:
                lbl.setFixedWidth(width)

    @staticmethod
    def compact_widget(w):
        if isinstance(w, QAbstractButton):
            # Clicked with the mouse only: a focused button would otherwise be
            # pressed by Space / Enter — Space is the manual-drive brake, and
            # it used to pause the run after clicking ▶ Run.
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if isinstance(w, QComboBox):
            w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(6)
        lay = w.layout()
        if isinstance(lay, QHBoxLayout):
            lay.setSpacing(4)
            # Row labels ("World:", "Arena:", ...) in the same bold as
            # "Gradients:" / "Objects:" — unless the label is already styled.
            first = lay.itemAt(0).widget() if lay.count() else None
            if isinstance(first, QLabel) and first.text().rstrip().endswith(':') \
                    and not first.styleSheet():
                first.setStyleSheet(f"color:{C['dark']};font-weight:bold;")
            # A combo takes the row's spare width (instead of the spare width
            # being shared out between labels, leaving gaps), unless the row
            # already says who stretches.
            if not any(lay.stretch(i) for i in range(lay.count())):
                for i in range(lay.count()):
                    if isinstance(lay.itemAt(i).widget(), QComboBox):
                        lay.setStretch(i, 1)

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
                padding: 4px 6px;
                border: 1px solid {C['border']};
                border-bottom: none;
                border-radius: 3px 3px 0 0;
                margin-right: 1px;
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
        self._tab_layouts = getattr(self, '_tab_layouts', []) + vls
        return vls

    def add_stretch(self):
        self._layout.addStretch()
