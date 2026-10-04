"""
sim_app_robot.py — Robot tab: real-robot mode on/off, the per-address rows and their
live update-rate readouts.
"""

import time

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PySide6.QtCore import QTimer

from sim_constants import C
from neurons import MotorLayer


class _RobotMixin:

    # ── Robot tab ────────────────────────────────────────────────────────────

    def _build_robot_group(self, panel_vl=None):
        target = panel_vl or self._panel._layout

        # Status chip — one compact line
        status_row = QWidget()
        sl = QHBoxLayout(status_row)
        sl.setContentsMargins(2, 2, 2, 2)
        self._robot_status_lbl = QLabel("● Offline")
        self._robot_status_lbl.setStyleSheet("color: gray; font-weight: bold;")
        sl.addWidget(self._robot_status_lbl)
        sl.addStretch()
        target.addWidget(status_row)

        # Dynamic rows container
        self._robot_rows_widget = QWidget()
        self._robot_rows_vl = QVBoxLayout(self._robot_rows_widget)
        self._robot_rows_vl.setContentsMargins(2, 0, 2, 0)
        self._robot_rows_vl.setSpacing(1)
        target.addWidget(self._robot_rows_widget)

        target.addStretch()   # keep rows packed at the top

        self._robot_hz_labels: dict = {}        # osc_path → QLabel  (sensors)
        self._robot_motor_hz_labels: dict = {}  # osc_path → QLabel  (motors)
        self._robot_last_seen: dict = {}        # osc_path → monotonic time of last nonzero Hz
        self._robot_connect_t = None            # monotonic time robot mode was last enabled

        self._robot_tab_timer = QTimer(self)
        self._robot_tab_timer.setInterval(1000)
        self._robot_tab_timer.timeout.connect(self._update_robot_hz)
        self._robot_tab_timer.start()

    def _rebuild_robot_rows(self):
        """Repopulate the Robot tab rows from the current circuit."""
        while self._robot_rows_vl.count():
            item = self._robot_rows_vl.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._robot_hz_labels.clear()
        self._robot_motor_hz_labels.clear()

        circuit = getattr(self._sim_ctrl, 'circuit', None)
        if circuit is None:
            return

        from robot_driver import _parse_address

        for sensor in circuit.sensors:
            addr = getattr(sensor, 'robot_address', '').strip()
            if not addr:
                continue
            _, _, osc_path, *_ = _parse_address(addr)
            hz_lbl = self._add_robot_row('S', sensor.name, osc_path or addr)
            if osc_path:
                self._robot_hz_labels[osc_path] = hz_lbl

        for layer in circuit.layers:
            if not isinstance(layer, MotorLayer):
                continue
            addr = getattr(layer, 'robot_address', '').strip()
            if not addr:
                continue
            _, _, osc_path, *_ = _parse_address(addr)
            hz_lbl = self._add_robot_row('M', layer.name, osc_path or addr)
            if osc_path:
                self._robot_motor_hz_labels[osc_path] = hz_lbl

    def _add_robot_row(self, kind: str, name: str, path: str) -> QLabel:
        """One compact row: [S/M] name  path  Hz. Returns the Hz QLabel."""
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)

        kind_lbl = QLabel(kind)
        kind_lbl.setFixedWidth(12)
        kind_lbl.setStyleSheet(
            "color: #888;" if kind == 'S' else "color: #55a;")
        hl.addWidget(kind_lbl)

        name_lbl = QLabel(name)
        name_lbl.setFixedWidth(80)
        hl.addWidget(name_lbl)

        path_lbl = QLabel(path)
        path_lbl.setStyleSheet("color: #666;")
        hl.addWidget(path_lbl)

        hl.addStretch()

        hz_lbl = QLabel("-- Hz")
        hz_lbl.setFixedWidth(50)
        hl.addWidget(hz_lbl)

        self._robot_rows_vl.addWidget(row)
        return hz_lbl

    _ROBOT_STALE_GRACE = 3.0   # seconds of zero Hz before flagging a sensor row as stale

    def _update_robot_hz(self):
        """Called every second — refresh Hz labels only when robot mode is active."""
        if not self._sim_ctrl.robot.enabled:
            return
        now = time.monotonic()
        sensor_rates = self._sim_ctrl.robot.driver.get_rates()
        stale_count = 0
        for path, lbl in self._robot_hz_labels.items():
            rate = sensor_rates.get(path, 0.0)
            lbl.setText(f"{rate:.0f} Hz")
            if rate > 0:
                self._robot_last_seen[path] = now
                lbl.setStyleSheet("")
            else:
                since = now - self._robot_last_seen.get(path, self._robot_connect_t or now)
                if since > self._ROBOT_STALE_GRACE:
                    lbl.setStyleSheet("color: #cc3333; font-weight: bold;")
                    stale_count += 1
                else:
                    lbl.setStyleSheet("")
        mt = self._sim_ctrl.robot.motor_thread
        if mt:
            motor_rates = mt.send_rates()
            for path, lbl in self._robot_motor_hz_labels.items():
                lbl.setText(f"{motor_rates.get(path, 0.0):.0f} Hz")

        # No sensor has produced a single packet since connecting (or all went
        # silent for longer than the grace period) — the socket can be "Online"
        # (bound fine) while the robot itself is off/unreachable, since UDP
        # gives no OS-level error for that. Surface it as a visible warning
        # instead of leaving the checkbox's static "Online" as the only signal.
        if self._robot_hz_labels and stale_count == len(self._robot_hz_labels):
            self._robot_status_lbl.setText("● No data from robot")
            self._robot_status_lbl.setStyleSheet("color: #cc3333; font-weight: bold;")
        else:
            self._robot_status_lbl.setText("● Online")
            self._robot_status_lbl.setStyleSheet("color: green; font-weight: bold;")

    # ── Real robot ───────────────────────────────────────────────────────────

    def _on_robot_mode_toggle(self, state):
        self._sim_ctrl.enable_robot_mode(bool(state))
        self._rebuild_channels()
        if state:
            self._robot_last_seen = {}
            self._robot_connect_t = time.monotonic()
            self._robot_status_lbl.setText("● Online")
            self._robot_status_lbl.setStyleSheet("color: green; font-weight: bold;")
        else:
            self._robot_last_seen = {}
            self._robot_connect_t = None
            self._robot_status_lbl.setText("● Offline")
            self._robot_status_lbl.setStyleSheet("color: gray; font-weight: bold; font-size: 8pt;")
            for lbl in {**self._robot_hz_labels, **self._robot_motor_hz_labels}.values():
                lbl.setText("-- Hz")
                lbl.setStyleSheet("")
