"""
Standalone freeze reproducer — no pytest needed.

Creates two windows and switches focus between them 15 times while a
ScatterPlotItem is updated at 10 Hz. A 16 ms QTimer measures event-loop
latency. Any tick >150 ms is printed as a STALL.

Usage:
    python tests/freeze_repro.py            # with fix (disableAutoRange)
    python tests/freeze_repro.py --broken   # without fix (shows stalls)

Output: timing summary + PASS / FAIL.
"""
import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import QTimer

N_NODES          = 30
UPDATE_INTERVAL  = 100   # ms — scatter update rate (10 Hz)
SWITCH_INTERVAL  = 350   # ms — focus switch rate
N_SWITCHES       = 15
LATENCY_TICK     = 16    # ms — event-loop monitor rate (~60 Hz)
STALL_THRESHOLD  = 150   # ms — a tick this long = stall


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


def run(disable_autorange: bool):
    app = QApplication.instance() or QApplication(sys.argv)

    label = "WITH fix (disableAutoRange)" if disable_autorange else "WITHOUT fix (autorange ON)"
    print(f"\n{'='*60}")
    print(f"  freeze_repro — {label}")
    print(f"  {N_NODES} nodes, {N_SWITCHES} focus switches at {SWITCH_INTERVAL} ms intervals")
    print(f"{'='*60}")

    # ── Window A: pyqtgraph scatter ──────────────────────────────────────────
    win_viz = QWidget()
    win_viz.setWindowTitle("Network Viz")
    win_viz.resize(600, 400)
    lay_a = QVBoxLayout(win_viz)
    gw = pg.GraphicsLayoutWidget(parent=win_viz)
    lay_a.addWidget(gw)
    vb = pg.ViewBox()
    plot = gw.addPlot(viewBox=vb)
    plot.hideAxis('bottom')
    plot.hideAxis('left')
    if disable_autorange:
        vb.disableAutoRange()
    scatter = pg.ScatterPlotItem()
    plot.addItem(scatter)

    # ── Window B: plain widget ───────────────────────────────────────────────
    win_main = QWidget()
    win_main.setWindowTitle("Main Window")
    win_main.resize(400, 300)
    status = QLabel("switching focus…")
    lay_b = QVBoxLayout(win_main)
    lay_b.addWidget(status)

    win_viz.show()
    win_main.show()

    # ── Scatter updater ──────────────────────────────────────────────────────
    pen = pg.mkPen('k', width=1.5)
    upd_timer = QTimer()
    upd_timer.setInterval(UPDATE_INTERVAL)
    upd_timer.timeout.connect(lambda: scatter.setData(spots=_spots(N_NODES, pen)))
    upd_timer.start()

    # ── Latency monitor ──────────────────────────────────────────────────────
    all_ticks  = []
    stalls     = []
    _last      = [time.perf_counter()]

    def latency_tick():
        now = time.perf_counter()
        dt  = (now - _last[0]) * 1000
        all_ticks.append(dt)
        if dt > STALL_THRESHOLD:
            stalls.append(round(dt, 1))
            print(f"  [STALL] {dt:.0f} ms  (switch #{switch_count[0]})")
        _last[0] = now

    lat_timer = QTimer()
    lat_timer.setInterval(LATENCY_TICK)
    lat_timer.timeout.connect(latency_tick)
    lat_timer.start()

    # ── Focus switcher ───────────────────────────────────────────────────────
    switch_count = [0]
    on_viz       = [True]

    def switch():
        switch_count[0] += 1
        if on_viz[0]:
            win_main.activateWindow()
            win_main.raise_()
        else:
            win_viz.activateWindow()
            win_viz.raise_()
        on_viz[0] = not on_viz[0]
        status.setText(f"Switch #{switch_count[0]} / {N_SWITCHES}  |  stalls so far: {len(stalls)}")
        print(f"  focus → {'main' if not on_viz[0] else 'viz':4s}   "
              f"(switch #{switch_count[0]:2d}   stalls: {len(stalls)})")
        if switch_count[0] >= N_SWITCHES:
            sw_timer.stop()
            QTimer.singleShot(600, finish)

    sw_timer = QTimer()
    sw_timer.setInterval(SWITCH_INTERVAL)
    sw_timer.timeout.connect(switch)
    sw_timer.start()

    # ── Finish & report ──────────────────────────────────────────────────────
    def finish():
        upd_timer.stop()
        lat_timer.stop()
        win_viz.close()
        win_main.close()

        print()
        if all_ticks:
            arr = np.array(all_ticks)
            print(f"  Ticks measured : {len(arr)}")
            print(f"  Mean latency   : {arr.mean():.1f} ms  (target: {LATENCY_TICK} ms)")
            print(f"  95th pct       : {np.percentile(arr, 95):.1f} ms")
            print(f"  Max latency    : {arr.max():.1f} ms")
            print(f"  Stalls >150 ms : {len(stalls)}")
        print()
        if stalls:
            print(f"  FAIL — {len(stalls)} stall(s): {stalls}")
        else:
            print("  PASS — no stalls detected")
        print()
        app.quit()

    app.exec()
    return len(stalls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--broken', action='store_true',
                        help='Run without the fix to confirm stalls occur')
    args = parser.parse_args()
    n_stalls = run(disable_autorange=not args.broken)
    sys.exit(1 if n_stalls else 0)
