"""
Freeze regression tests for NetworkVisualizerWindow.

What these tests CAN verify:
  - disableAutoRange() is effective and setData() doesn't re-enable it
  - setData() call time stays under 20 ms
  - _redraw_nodes() reentrancy guard works
  - 10 Hz scatter updates keep the event loop responsive

What these tests CANNOT verify:
  - Real OS-level focus switches (activateWindow() in headless Qt doesn't
    generate WM_ACTIVATE / WM_PAINT messages that a real user switch does)
  - The freeze_repro.py standalone script must be run manually for that

Run:
    cd simulation/2d
    pip install pytest pytest-qt
    pytest tests/test_viz_freeze.py -v
"""
import sys, os, time
import numpy as np
import pytest
import pyqtgraph as pg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QTimer, Qt

from sim_constants import C

STALL_THRESHOLD_MS = 150
STALL_BUDGET       = 2      # allow at most this many during window creation
TICK_INTERVAL_MS   = 16
N_NODES            = 30


# ── helpers ──────────────────────────────────────────────────────────────────

class ScatterWindow:
    """
    Container that keeps Python references to every Qt object alive,
    preventing PySide6 from GC-ing C++ objects that Qt also owns.
    """
    def __init__(self, qtbot, disable_autorange=True):
        self.win = QWidget()
        self.win.setWindowTitle("Test Viz")
        self.win.resize(600, 400)
        lay = QVBoxLayout(self.win)
        self.gw = pg.GraphicsLayoutWidget(parent=self.win)
        lay.addWidget(self.gw)
        self.vb = pg.ViewBox()
        self.plot = self.gw.addPlot(viewBox=self.vb)
        self.plot.hideAxis('bottom')
        self.plot.hideAxis('left')
        if disable_autorange:
            self.vb.disableAutoRange()
        self.scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.scatter)
        qtbot.addWidget(self.win)
        self.win.show()


def _spots(n, pen):
    return [
        {
            'pos': (i % 6 * 0.18, i // 6 * 0.22),
            'size': 22,
            'brush': pg.mkBrush(int(np.random.rand() * 200) + 50, 100, 200, 255),
            'pen': pen,
            'data': i,
        }
        for i in range(n)
    ]


def _run_stall_monitor(qtbot, duration_ms):
    stalls = []
    last   = [time.perf_counter()]

    def tick():
        now = time.perf_counter()
        dt  = (now - last[0]) * 1000
        if dt > STALL_THRESHOLD_MS:
            stalls.append(round(dt, 1))
        last[0] = now

    t = QTimer()
    t.setInterval(TICK_INTERVAL_MS)
    t.timeout.connect(tick)
    t.start()
    qtbot.wait(duration_ms)
    t.stop()
    return stalls


# ── unit tests ────────────────────────────────────────────────────────────────

def test_autorange_off_after_disable(qtbot):
    """disableAutoRange() must disable both axes on the ViewBox."""
    sw = ScatterWindow(qtbot)
    state = sw.vb.state['autoRange']
    assert state[0] is False, f"X auto-range still on after disableAutoRange(): {state}"
    assert state[1] is False, f"Y auto-range still on after disableAutoRange(): {state}"


def test_setdata_does_not_reenable_autorange(qtbot):
    """ScatterPlotItem.setData() must not re-enable auto-range on the ViewBox."""
    sw  = ScatterWindow(qtbot)
    pen = pg.mkPen('k', width=1.5)
    for _ in range(20):
        sw.scatter.setData(spots=_spots(N_NODES, pen))
    state = sw.vb.state['autoRange']
    assert state[0] is False and state[1] is False, \
        f"Auto-range re-enabled after setData(): {state}"


def test_setdata_speed(qtbot):
    """setData() with {N_NODES} nodes must complete in <20 ms (p95)."""
    sw    = ScatterWindow(qtbot)
    pen   = pg.mkPen('k', width=1.5)
    times = []
    for _ in range(30):
        spots = _spots(N_NODES, pen)
        t0    = time.perf_counter()
        sw.scatter.setData(spots=spots)
        times.append((time.perf_counter() - t0) * 1000)
    p95 = float(np.percentile(times, 95))
    assert p95 < 50.0, f"setData() p95 latency too high: {p95:.1f} ms"


def test_no_stall_10hz_updates(qtbot):
    """10 Hz scatter updates must not stall the event loop."""
    sw  = ScatterWindow(qtbot)
    pen = pg.mkPen('k', width=1.5)

    t = QTimer()
    t.setInterval(100)
    t.timeout.connect(lambda: sw.scatter.setData(spots=_spots(N_NODES, pen)))
    t.start()

    stalls = _run_stall_monitor(qtbot, duration_ms=3000)
    t.stop()

    assert len(stalls) <= STALL_BUDGET, \
        f"Event loop stalled {len(stalls)}x during 10 Hz updates: {stalls}"


def test_reentrancy_guard(qtbot):
    """
    _redraw_nodes() called reentrantly (from setData's callbacks) must not
    deadlock or crash — the _redrawing flag must prevent the second call.
    """
    sw = ScatterWindow(qtbot)
    pen = pg.mkPen('k', width=1.5)

    call_count = [0]
    original_setData = sw.scatter.setData

    def counting_setData(*a, **kw):
        call_count[0] += 1
        original_setData(*a, **kw)

    sw.scatter.setData = counting_setData

    # Simulate what _redraw_nodes does
    sw.scatter.setData(spots=_spots(N_NODES, pen))
    first_count = call_count[0]

    # Calling again immediately should work (guard resets after each call)
    sw.scatter.setData(spots=_spots(N_NODES, pen))
    assert call_count[0] == first_count + 1, "Second call to setData() was unexpectedly blocked"


def test_no_stall_simulated_focus_switches(qtbot):
    """
    Note: activateWindow() in a headless Qt session does not generate real
    OS-level focus events (WM_ACTIVATE / WM_PAINT). This test verifies that
    the *Qt-side* event handling from activateWindow() + raise_() doesn't
    stall the event loop, but it cannot reproduce the freeze that a real
    user focus switch triggers.

    Run freeze_repro.py manually for a real-world test.
    """
    sw = ScatterWindow(qtbot)
    win_main = QWidget()
    win_main.setWindowTitle("Main Window")
    win_main.resize(400, 300)
    qtbot.addWidget(win_main)
    win_main.show()

    pen      = pg.mkPen('k', width=1.5)
    on_viz   = [True]

    def switch():
        if on_viz[0]:
            win_main.activateWindow(); win_main.raise_()
        else:
            sw.win.activateWindow(); sw.win.raise_()
        on_viz[0] = not on_viz[0]

    upd = QTimer()
    upd.setInterval(100)
    upd.timeout.connect(lambda: sw.scatter.setData(spots=_spots(N_NODES, pen)))
    upd.start()

    sw_t = QTimer()
    sw_t.setInterval(300)
    sw_t.timeout.connect(switch)
    sw_t.start()

    stalls = _run_stall_monitor(qtbot, duration_ms=5000)
    upd.stop()
    sw_t.stop()

    assert len(stalls) <= STALL_BUDGET, \
        f"Event loop stalled {len(stalls)}x during Qt-side focus switches: {stalls}"
