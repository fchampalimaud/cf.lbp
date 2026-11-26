"""
robot_check_dialog.py — "Test connection" window: runs robot_check.RobotCheck in a
background thread and ticks off each row of the checklist as results arrive.
"""

import threading

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
                               QPushButton, QWidget, QScrollArea, QFrame)

from robot_check import RobotCheck, PENDING, RUNNING, PASS, WARN, FAIL, SKIP

_ICON = {PENDING: ('○', '#8A9AAA'), RUNNING: ('…', '#4A7AB0'), PASS: ('✔', '#2E8B57'),
         WARN: ('⚠', '#C08020'), FAIL: ('✖', '#C03030'), SKIP: ('–', '#8A9AAA')}


class _Worker(QThread):
    result = Signal(str, str, str)

    def __init__(self, check: RobotCheck, cancel: threading.Event):
        super().__init__()
        self._check = check
        self._cancel = cancel

    def run(self):
        self._check.run(self.result.emit, self._cancel)


class RobotCheckDialog(QDialog):
    def __init__(self, circuit, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Robot connection test")
        self.setMinimumWidth(560)
        self._circuit = circuit
        self._worker = None
        self._cancel = threading.Event()

        vl = QVBoxLayout(self)
        intro = QLabel("Checks everything between this computer and the robot. "
                       "The robot receives a stop command (/wheels 0 0) during the test.")
        intro.setWordWrap(True)
        vl.addWidget(intro)

        self._rows_host = QWidget()
        self._grid = QGridLayout(self._rows_host)
        self._grid.setColumnStretch(1, 1)
        self._grid.setVerticalSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._rows_host)
        vl.addWidget(scroll, 1)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("font-weight: bold;")
        vl.addWidget(self._summary)

        bl = QHBoxLayout()
        bl.addStretch()
        self._rerun_btn = QPushButton("Run again")
        self._rerun_btn.clicked.connect(self._start)
        bl.addWidget(self._rerun_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bl.addWidget(close_btn)
        vl.addLayout(bl)

        self.resize(620, 480)
        self._start()

    def _start(self):
        check = RobotCheck(self._circuit)
        self._build_rows(check.checks)
        self._results = {}
        self._rerun_btn.setEnabled(False)
        self._summary.setText("Testing…")
        self._summary.setStyleSheet("font-weight: bold;")
        self._cancel = threading.Event()
        self._worker = _Worker(check, self._cancel)
        self._worker.result.connect(self._on_result)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _build_rows(self, checks):
        while self._grid.count():
            w = self._grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._rows = {}
        for i, (key, title) in enumerate(checks):
            icon = QLabel()
            icon.setFixedWidth(22)
            icon.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            text = QLabel()
            text.setWordWrap(True)
            text.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._grid.addWidget(icon, i, 0)
            self._grid.addWidget(text, i, 1)
            self._rows[key] = (icon, text, title)
            self._set_row(key, PENDING, '')
        self._grid.setRowStretch(len(checks), 1)

    def _set_row(self, key, status, detail):
        icon, text, title = self._rows[key]
        glyph, color = _ICON[status]
        icon.setText(glyph)
        icon.setStyleSheet(f"color:{color}; font-size: 13pt; font-weight: bold;")
        body = f"<b>{_esc(title)}</b>"
        if detail:
            body += "<br>" + _esc(detail).replace('\n', '<br>')
        text.setText(body)

    def _on_result(self, key, status, detail):
        if key in self._rows:
            self._set_row(key, status, detail)
            self._results[key] = status

    def _on_finished(self):
        self._rerun_btn.setEnabled(True)
        if self._cancel.is_set():
            return
        statuses = set(self._results.values())
        if FAIL in statuses:
            msg, color = "Some checks failed — fix the ✖ items above and press Run again.", '#C03030'
        elif WARN in statuses:
            msg, color = "The robot works, with warnings (⚠). You can switch Robot mode on.", '#C08020'
        else:
            msg, color = "All checks passed. Close this window and switch Robot mode on.", '#2E8B57'
        self._summary.setText(msg)
        self._summary.setStyleSheet(f"font-weight: bold; color:{color};")

    def done(self, r):
        # The worker holds the robot's port until it returns; wait so Robot mode
        # can bind it as soon as this window is gone.
        if self._worker is not None and self._worker.isRunning():
            self._cancel.set()
            self._summary.setText("Stopping…")
            self._worker.wait(10000)
        super().done(r)


def _esc(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
