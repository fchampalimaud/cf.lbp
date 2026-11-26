"""
sim_app_robot.py — Robot tab: real-robot mode on/off, the robot-address table (the one
place sensor / motor-layer robot_address fields are edited) and live update rates.
"""

import time

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QTabWidget
from PySide6.QtCore import QTimer, QObject, QEvent, Qt

from sim_constants import C
from neurons import MotorLayer

_DRIVE_KEYS = {Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D, Qt.Key.Key_Space}


class _DriveKeyBlocker(QObject):
    """Keeps the manual-drive keys out of the address fields while ⌨ Manual is on.
    The app-wide _ManualKeyFilter runs first, so the robot still gets the keys."""
    def __init__(self, app):
        super().__init__(app)
        self._app = app

    def eventFilter(self, obj, event):
        return (event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
                and self._app._manual_active and event.key() in _DRIVE_KEYS)


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
        test_btn = self._make_btn("🔍 Test connection")
        test_btn.setToolTip(
            "Test connection: check the robot addresses, the port, ping, and that every\n"
            "sensor's data arrives — with a fix suggested for anything that fails.\n"
            "Robot mode is paused during the test and restored afterwards.")
        test_btn.clicked.connect(self._on_robot_test)
        sl.addWidget(test_btn)
        target.addWidget(status_row)

        hint = QLabel("Robot address of each sensor and motor layer — "
                      "host:port/path[types](indices){formula}. Empty = not connected.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:8pt;")
        target.addWidget(hint)

        # Dynamic rows container
        self._robot_rows_widget = QWidget()
        self._robot_rows_vl = QVBoxLayout(self._robot_rows_widget)
        self._robot_rows_vl.setContentsMargins(2, 0, 2, 0)
        self._robot_rows_vl.setSpacing(1)
        target.addWidget(self._robot_rows_widget)

        target.addStretch()   # keep rows packed at the top

        self._robot_tab_page = target.parentWidget()
        self._drive_key_blocker = _DriveKeyBlocker(self)
        self._robot_addr_edits: dict = {}       # element name → its address QLineEdit
        self._robot_hz_labels: dict = {}        # osc_path → [QLabel]  (sensors)
        self._robot_motor_hz_labels: dict = {}  # osc_path → [QLabel]  (motors)
        self._robot_last_seen: dict = {}        # osc_path → monotonic time of last nonzero Hz
        self._robot_connect_t = None            # monotonic time robot mode was last enabled

        self._robot_tab_timer = QTimer(self)
        self._robot_tab_timer.setInterval(1000)
        self._robot_tab_timer.timeout.connect(self._update_robot_hz)
        self._robot_tab_timer.start()

    def _rebuild_robot_rows(self):
        """Repopulate the address table from the current circuit: every sensor and
        every motor layer, connected or not."""
        while self._robot_rows_vl.count():
            item = self._robot_rows_vl.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._robot_addr_edits.clear()
        self._robot_hz_labels.clear()
        self._robot_motor_hz_labels.clear()

        circuit = getattr(self._sim_ctrl.registry, 'circuit', None)
        if circuit is None:
            return

        from robot_driver import _parse_address

        elements = [('S', s, self._robot_hz_labels) for s in circuit.sensors]
        elements += [('M', l, self._robot_motor_hz_labels)
                     for l in circuit.layers if isinstance(l, MotorLayer)]
        for kind, obj, hz_map in elements:
            addr = (getattr(obj, 'robot_address', '') or '').strip()
            hz_lbl = self._add_robot_row(kind, obj, addr)
            _, _, osc_path, *_ = _parse_address(addr)
            if osc_path:
                hz_map.setdefault(osc_path, []).append(hz_lbl)

    def _add_robot_row(self, kind: str, obj, addr: str) -> QLabel:
        """Two lines: [S/M] name … Hz, then the editable address. Returns the Hz QLabel."""
        row = QWidget()
        vl = QVBoxLayout(row)
        vl.setContentsMargins(0, 3, 0, 3)
        vl.setSpacing(1)

        head = QHBoxLayout()
        head.setSpacing(4)
        kind_lbl = QLabel(kind)
        kind_lbl.setFixedWidth(12)
        kind_lbl.setToolTip("Sensor" if kind == 'S' else "Motor layer")
        kind_lbl.setStyleSheet("color: #888;" if kind == 'S' else "color: #55a;")
        head.addWidget(kind_lbl)
        head.addWidget(QLabel(obj.name))
        head.addStretch()
        hz_lbl = QLabel("-- Hz" if addr else "")
        hz_lbl.setFixedWidth(50)
        head.addWidget(hz_lbl)
        vl.addLayout(head)

        edit = QLineEdit(addr)
        edit.setPlaceholderText("not connected")
        edit.setToolTip(
            "Sensor: host:port/osc_path[types](indices){formula}\n"
            "  e.g. 192.168.0.224:2390/analogs[iiiii](1,4){x/500}\n"
            "Motor layer: host:port/osc_path, e.g. 192.168.0.224:2390/wheels\n"
            "Press Enter or leave the field to apply. Empty = not connected.")
        edit.editingFinished.connect(lambda o=obj, e=edit: self._set_robot_address(o, e.text()))
        # Enter hands the keyboard back, so driving afterwards doesn't type into the field.
        edit.returnPressed.connect(edit.clearFocus)
        edit.installEventFilter(self._drive_key_blocker)
        vl.addWidget(edit)
        self._robot_addr_edits[obj.name] = edit

        self._robot_rows_vl.addWidget(row)
        return hz_lbl

    def _set_robot_address(self, obj, text):
        """Write one element's robot_address as an undoable network edit."""
        text = text.strip()
        if text == (getattr(obj, 'robot_address', '') or '').strip():
            return
        nv = getattr(self, '_net_viz', None)
        if nv is not None:
            with nv._editor.edit():
                obj.robot_address = text
            nv._on_edit_committed()
            nv.build()          # also rebuilds this table via _rebuild_channels
        else:
            obj.robot_address = text
            QTimer.singleShot(0, self._rebuild_robot_rows)
        if self._sim_ctrl.robot.enabled:
            # The driver reads the addresses when it starts — restart it.
            self._sim_ctrl.enable_robot_mode(True)

    def _show_robot_address(self, name):
        """Switch the side panel to the Robot tab and focus *name*'s address field.
        Deferred so it runs after the caller's edit has rebuilt the table."""
        def _show():
            page = self._robot_tab_page
            tabs = page.parentWidget() if page is not None else None
            while tabs is not None and not isinstance(tabs, QTabWidget):
                tabs = tabs.parentWidget()
            if tabs is not None:
                tabs.setCurrentWidget(page)
            self.raise_()
            self.activateWindow()
            edit = self._robot_addr_edits.get(name)
            if edit is not None:
                edit.setFocus()
                edit.selectAll()
        QTimer.singleShot(0, _show)

    _ROBOT_STALE_GRACE = 3.0   # seconds of zero Hz before flagging a sensor row as stale

    def _update_robot_hz(self):
        """Called every second — refresh Hz labels only when robot mode is active."""
        if not self._sim_ctrl.robot.enabled:
            return
        now = time.monotonic()
        sensor_rates = self._sim_ctrl.robot.driver.get_rates()
        stale_count = 0
        for path, lbls in self._robot_hz_labels.items():
            rate = sensor_rates.get(path, 0.0)
            style = ""
            if rate > 0:
                self._robot_last_seen[path] = now
            else:
                since = now - self._robot_last_seen.get(path, self._robot_connect_t or now)
                if since > self._ROBOT_STALE_GRACE:
                    style = "color: #cc3333; font-weight: bold;"
                    stale_count += 1
            for lbl in lbls:
                lbl.setText(f"{rate:.0f} Hz")
                lbl.setStyleSheet(style)
        mt = self._sim_ctrl.robot.motor_thread
        if mt:
            motor_rates = mt.send_rates()
            for path, lbls in self._robot_motor_hz_labels.items():
                for lbl in lbls:
                    lbl.setText(f"{motor_rates.get(path, 0.0):.0f} Hz")

        # No sensor has produced a single packet since connecting (or all went
        # silent for longer than the grace period) — the socket can be "Online"
        # (bound fine) while the robot itself is off/unreachable, since UDP
        # gives no OS-level error for that. Surface it as a visible warning
        # instead of leaving the checkbox's static "Online" as the only signal.
        if self._robot_hz_labels and stale_count == len(self._robot_hz_labels):
            self._robot_status_lbl.setText("● No data from robot")
            self._robot_status_lbl.setStyleSheet("color: #cc3333; font-weight: bold;")
        elif self._sim_ctrl.robot.enabled:
            self._robot_status_lbl.setText("● Online")
            self._robot_status_lbl.setStyleSheet("color: green; font-weight: bold;")

    # ── Real robot ───────────────────────────────────────────────────────────

    def _on_robot_test(self):
        """Run the connection checklist. Robot mode holds the robot's UDP port, so
        it is switched off (and the run paused) for the test, then restored."""
        circuit = getattr(self._sim_ctrl.registry, 'circuit', None)
        if circuit is None:
            return
        from robot_check_dialog import RobotCheckDialog
        was_running = self._sim_ctrl.running
        was_robot = self._robot_btn.isChecked()
        if was_running:
            self._sim_ctrl.stop()
        if was_robot:
            self._robot_btn.setChecked(False)
        RobotCheckDialog(circuit, self).exec()
        if was_robot:
            self._robot_btn.setChecked(True)
        if was_running:
            self._sim_ctrl.start()

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
            for lbls in [*self._robot_hz_labels.values(), *self._robot_motor_hz_labels.values()]:
                for lbl in lbls:
                    lbl.setText("-- Hz")
                    lbl.setStyleSheet("")
