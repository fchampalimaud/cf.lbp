import sys
import os
import time

_splash = None  # replaced by _Splash instance when running as __main__

# Show splash before any heavy imports so the user sees it during loading
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QSplashScreen
    from PySide6.QtGui import (QPixmap, QPainter, QBrush, QColor, QFont, QPen,
                                QLinearGradient)
    from PySide6.QtCore import Qt, QRectF
    from PySide6.QtSvg import QSvgRenderer

    class _Splash(QSplashScreen):
        def __init__(self, base_pix):
            super().__init__(base_pix)
            self._base = base_pix

        def set_progress(self, value):
            pix = self._base.copy()
            W, H = pix.width(), pix.height()
            bx, by, bw, bh = 50, H - 34, W - 100, 7
            p = QPainter(pix)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(Qt.PenStyle.NoPen)
            if value > 0:
                fw = max(bh, int(bw * value / 100))
                grad = QLinearGradient(bx, 0, bx + bw, 0)
                grad.setColorAt(0, QColor('#7a4e28'))
                grad.setColorAt(1, QColor('#b8895a'))
                p.setBrush(QBrush(grad))
                p.drawRoundedRect(bx, by, fw, bh, 3, 3)
            p.end()
            self.setPixmap(pix)
            QApplication.processEvents()

    _app = QApplication(sys.argv)
    _app.setStyle("Fusion")
    _W, _H = 520, 320
    _pix = QPixmap(_W, _H)
    _pix.fill(Qt.GlobalColor.transparent)
    _p = QPainter(_pix)
    _p.setRenderHint(QPainter.RenderHint.Antialiasing)
    # Dark slate background
    _p.setBrush(QBrush(QColor('#181c25')))
    _p.setPen(Qt.PenStyle.NoPen)
    _p.drawRoundedRect(0, 0, _W, _H, 14, 14)
    # Logo
    _logo = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', 'assets', 'logo.svg')
    if os.path.exists(_logo):
        _r = QSvgRenderer(_logo)
        _lh, _lw = 145, int(145 * 130 / 120)
        _r.render(_p, QRectF((_W - _lw) // 2, 16, _lw, _lh))
    # Title
    _f = QFont('Segoe UI', 22, QFont.Weight.Bold)
    _p.setFont(_f)
    _p.setPen(QPen(QColor('#b5855a')))
    _p.drawText(0, 168, _W, 38, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                'Little Brain Project')
    # Accent divider
    _p.setPen(QPen(QColor('#9a6840'), 1))
    _p.drawLine((_W - 140) // 2, 210, (_W + 140) // 2, 210)
    # Subtitle
    _f2 = QFont('Segoe UI', 11)
    _f2.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
    _p.setFont(_f2)
    _p.setPen(QPen(QColor('#8898aa')))
    _p.drawText(0, 215, _W, 28, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                'S I M U L A T O R')
    # Version (read the VERSION file directly — src/ isn't on sys.path yet here)
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VERSION'), encoding='utf-8') as _vf:
            _version = _vf.read().strip() or '0.0.0'
    except OSError:
        _version = '0.0.0'
    _f3 = QFont('Segoe UI', 9)
    _p.setFont(_f3)
    _p.setPen(QPen(QColor('#5a6478')))
    _p.drawText(0, 246, _W, 20, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                f'v{_version}')
    # Progress bar track
    _p.setBrush(QBrush(QColor('#2a3040')))
    _p.setPen(Qt.PenStyle.NoPen)
    _p.drawRoundedRect(50, _H - 34, _W - 100, 7, 3, 3)
    _p.end()
    _splash = _Splash(_pix)
    _splash.show()
    _splash.set_progress(5)

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import glob
from collections import deque

import numpy as np
if _splash: _splash.set_progress(20)
import pyqtgraph as pg
pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')
if _splash: _splash.set_progress(40)

# Force working directory to script location; add src/ and brains/ to import path
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
sys.path.insert(0, os.path.join(_HERE, "src"))
import data_paths
# Public installs from before the user folder existed kept the user's files
# in the app's folders: move them out before anything reads (or an update
# replaces) those folders. No-op without a manifest (the private repo).
_MIGRATED = data_paths.migrate_user_files()
for _p in _MIGRATED:
    print(f"[files] moved to your folder: {_p}")
# Brains are imported by module name from both roots (the user's first).
for _brains in reversed(data_paths.search_dirs('brains')):
    sys.path.insert(0, str(_brains))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSplitter,
    QLineEdit, QRadioButton, QMessageBox, QMenu,
    QDialog, QFormLayout, QDialogButtonBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QColorDialog, QSplashScreen, QGroupBox,
)
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, Signal, QEvent, QMimeData, QObject
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QDrag, QPixmap
if _splash: _splash.set_progress(60)

from rigid_body import RigidBody, Joint, world_poses as _rb_world_poses
from sim_constants import C, _CHAN_PALETTE, _TRAIL_COLOR, GRADIENT_COLORS, OBJECT_COLORS
from circuit_model import CircuitModel, Connection
from circuit_editor import CircuitEditor
from logger import SimLogger
from sim_app_brain import _BrainMixin
from sim_app_session import _SessionMixin
from sim_app_ui import _UiBuilderMixin
from sim_app_agents import _AgentsMixin
from sim_app_network import _NetworkMixin
from sim_app_robot import _RobotMixin
from sim_app_world import _WorldMixin

from neurons import MotorLayer
from sensors import ManualBumpSensor, ManualCueSensor
from brain_base import Param, ChoiceParam, BaseConfig, BaseBrain, DataBrain
from brain_manager import BrainManager
from sim_config import SimConfig
from world import World
from session_io import save_session, load_session
from world_serializer import discover_worlds, save_world_file, load_world_file

from arena_widget import ArenaViewBox, RobotItem, ChildBodyItem, CircleItem
from sim_widgets import MonetarySpinBox, _ManualKeyFilter, _ArrowKeyFilter, _CueKeyFilter
from shortcuts import install_simulation_shortcuts
from world_editor import WorldEditor
from osc_controller import OscChannelManager
from sim_controller import SimController
from app_version import get_app_version
if _splash: _splash.set_progress(80)


# ============================================================
# MAIN WINDOW
# ============================================================
class SimulatorApp(_UiBuilderMixin, _BrainMixin, _SessionMixin, _AgentsMixin,
                   _NetworkMixin, _RobotMixin, _WorldMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"LBP Simulator v{get_app_version()}")
        self.resize(950, 800)
        screen = QApplication.primaryScreen().availableGeometry()
        self.move((screen.width() - 950) // 2, (screen.height() - 800) // 2)

        # ── Shared state (no Qt deps) ─────────────────────────────────────────
        self.sim_cfg  = SimConfig()
        self.world    = World(self.sim_cfg)
        _circuit0 = CircuitModel()
        _circuit0.bodies = [RigidBody('root', 'root', self.sim_cfg.body_radius)]

        self._net_viz           = None
        self._connection_params = {}
        # Network-window column settings — set when a network file loads; empty
        # until then (e.g. a group in Network mode with no file chosen yet).
        self._hidden_containers   = set()
        self._disabled_containers = set()
        self._container_labels    = {}
        self._container_notes     = {}
        self._manual_active     = False
        self._held_keys         = set()
        self._key_filter        = None

        # ── Logger (shared with SimController) ───────────────────────────────
        self._logger = SimLogger()
        self._video_recorders = []

        # ── Brain manager (per-agent; circuit/brain_mgr are proxy properties) ─
        _brain_mgr0 = BrainManager(_circuit0, self.sim_cfg)
        self.brain_files = _brain_mgr0.discover_brains()
        self.network_brain_modules = _brain_mgr0.network_brain_modules
        # Agent-table groups now live on SimController (self._sim_ctrl.registry.groups_ordered()
        # etc.), keyed by stable group/agent ids — created once _sim_ctrl exists below.

        # ── Build UI first so arena/osc exist before controllers ─────────────
        self._build_ui()

        # ── Controllers (need arena and osc widget references) ─────────────
        self._osc_ctrl = OscChannelManager(self._osc, self._osc_mult_layout)

        self._sim_ctrl = SimController(
            circuit            = _circuit0,
            brain_mgr          = _brain_mgr0,
            sim_cfg            = self.sim_cfg,
            world              = self.world,
            arena              = self._arena,
            osc_ctrl           = self._osc_ctrl,
            logger             = self._logger,
            get_trail_visible  = lambda: self._trail_cb.isChecked(),
            get_motor_override = self._get_motor_override,
            parent             = self,
        )
        self._sim_ctrl.sig_status_changed.connect(self._on_status_changed)
        self._sim_ctrl.sig_timing_updated.connect(self._timing_label.setText)
        self._sim_ctrl.network.sig_client_ready.connect(self._update_net_status)
        self._sim_ctrl.network.sig_client_disconnect.connect(self._update_net_status)
        self._sim_ctrl.network.sig_client_registered.connect(
            self._on_client_registered, type=Qt.ConnectionType.QueuedConnection)
        self._sim_ctrl.network.sig_agent_removed.connect(
            self._on_remote_agent_removed, type=Qt.ConnectionType.QueuedConnection)
        self._sim_ctrl.sig_frame_ready.connect(self._arena.on_frame_ready)
        self._sim_ctrl.sig_frame_ready.connect(self._osc_ctrl.on_frame_ready)
        self._sim_ctrl.sig_tick_values.connect(self._osc_ctrl.on_tick_values)
        self._refresh_agent_list()

        # App-wide (not window-scoped) so Up/Down works no matter which
        # top-level window — main sim window, network visualizer, ... — has
        # keyboard focus.
        self._bump_key_filter = _ArrowKeyFilter(self._step_bump_sensors)
        QApplication.instance().installEventFilter(self._bump_key_filter)

        # App-wide and unconditional (not gated behind Manual-drive mode, unlike
        # _ManualKeyFilter) so a faked ManualCueSensor keeps working while the
        # brain/network is actually driving the robot.
        self._cue_key_filter = _CueKeyFilter(self._on_cue_key_change)
        QApplication.instance().installEventFilter(self._cue_key_filter)

        # Run / Step / Reset from any window of the app (shortcuts.py lists all keys).
        self._sim_shortcuts = install_simulation_shortcuts(
            self, self._on_run_stop, lambda: self._sim_ctrl.step(), self._reset,
            self._show_shortcuts)

        self._editor = WorldEditor(
            world          = self.world,
            sim_cfg        = self.sim_cfg,
            arena          = self._arena,
            bot_pos        = self._sim_ctrl.registry.bot_pos,
            setup_world_cb = self._setup_world,
            get_agents     = lambda: self._sim_ctrl.registry.agents,
        )

        # ── Connect arena signals to editor ──────────────────────────────────
        self._arena.sigClick.connect(self._editor.handle_click)
        self._arena.sigClick.connect(self._on_arena_click)
        self._arena.sigDrag.connect(self._editor.handle_drag)
        self._arena.sigHover.connect(self._editor.handle_hover)

        # ── Populate combos and auto-load ─────────────────────────────────────
        self._refresh_brain_list()
        self._refresh_session_dirs()
        if self._load_latest_session():
            pass
        elif self._session_combo.count() > 0:
            self._load_session()
        else:
            if 'BrainGUI' in self.brain_files:
                self.load_brain('BrainGUI')
            self._reset()

        self._start_mujoco()

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def brain(self):
        return self._sim_ctrl.registry.brain

    @brain.setter
    def brain(self, value):
        self._sim_ctrl.registry.brain = value

    @property
    def bot_pos(self):
        return self._sim_ctrl.registry.bot_pos

    @property
    def circuit(self):
        return self._sim_ctrl.registry.circuit

    @property
    def brain_mgr(self):
        return self._sim_ctrl.registry.brain_mgr

    # ── UI Construction ───────────────────────────────────────────────────────
    # _build_ui / _make_btn / _darken / _text_color / _make_param_row /
    # _build_sim_group / _build_physics_group / _build_network_group /
    # _build_robot_group / _add_robot_row / _build_world_group → sim_app_ui._UiBuilderMixin
    # _build_brain_group → sim_app_brain._BrainMixin
    # _build_session_group / _build_task_group / _build_logger_group → sim_app_session._SessionMixin

    # ── Sim param handlers ────────────────────────────────────────────────────

    def _on_sim_param(self, key, val):
        setattr(self.sim_cfg, key, val)
        if key == 'dt':
            self._report_fast_taus()
        if key in ['arena_scale', 'stim_radius', 'body_radius', 'toggle_stim']:
            self._setup_world()

    def _on_toggle_stim(self, state):
        self.sim_cfg.toggle_stim = 1 if state else 0
        self._setup_world()

    def _on_toggle_osc(self, state):
        if state:
            self._osc_dock.setVisible(True)
            self.resize(self.width(), self.height() + self._osc_hidden_height)
        else:
            self._osc_hidden_height = self._osc_dock.height()
            self._osc_dock.setVisible(False)
            self.resize(self.width(), self.height() - self._osc_hidden_height)

    def _on_arena_type_change(self, checked):
        self.world.arena_round = self._arena_round_rb.isChecked()
        self._setup_world()

    def _on_trail_len_change(self, n):
        n   = max(10, n)
        for agent in self._sim_ctrl.registry.agents:
            old = list(agent.trail_xy)[-n:]
            agent.trail_xy = deque(old, maxlen=n)

    def _on_rt_mode_toggled(self, checked):
        self._sim_ctrl.set_rt_mode(checked)
        self._speed_spin.setEnabled(not checked)
        self._speed_spin.setToolTip(
            "Disabled while Real time is on: the simulation then runs exactly as fast "
            "as the wall clock (speed ×1). Untick Real time to set a speed."
            if checked else self._speed_tip)

    def _on_status_changed(self, text, color):
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color:{color};font-weight:bold;padding:0 8px;")
        if text.startswith("●"):
            self._btn_run_stop.setText("⏸ Pause")
            self._btn_run_stop.setStyleSheet(self._make_btn("⏸ Pause", C['warning']).styleSheet())
        else:
            self._btn_run_stop.setText("▶ Run")
            self._btn_run_stop.setStyleSheet(self._make_btn("▶ Run", C['success']).styleSheet())

        if self._video_auto_cb.isChecked():
            if text == "●  RUNNING" and not self._video_recorders:
                self._video_start()
            elif text in ("⏸  PAUSED", "■  STOPPED") and self._video_recorders:
                self._video_stop()

    def _on_run_stop(self):
        """Run / Pause: pausing keeps the robot and the brain where they are and
        Run continues from there; ↺ Reset goes back to the start."""
        if self._sim_ctrl.running:
            self._sim_ctrl.stop()
        else:
            self._sim_ctrl.start()

    # ── Manual control ────────────────────────────────────────────────────────

    def _step_bump_sensors(self, direction):
        for sensor in self.circuit.sensors:
            if isinstance(sensor, ManualBumpSensor):
                sensor.step(direction)

    def _on_cue_key_change(self, letter, pressed):
        for sensor in self.circuit.sensors:
            if isinstance(sensor, ManualCueSensor) and sensor.key == letter:
                sensor.set_pressed(pressed)

    def _get_motor_override(self):
        return self._manual_motors() if self._manual_active else None

    def _toggle_manual_mode(self):
        self._manual_active = not self._manual_active
        self._manual_btn.setChecked(self._manual_active)
        self._manual_hint.setVisible(self._manual_active)
        if self._manual_active:
            self._held_keys.clear()
            if self._key_filter is None:
                self._key_filter = _ManualKeyFilter(self._held_keys, self)
            # App-wide (not window-scoped), so WASD/Space still drive the robot
            # when another top-level window — e.g. the network visualizer — has
            # keyboard focus. Mirrors _ArrowKeyFilter's install below.
            QApplication.instance().installEventFilter(self._key_filter)
        else:
            if self._key_filter is not None:
                QApplication.instance().removeEventFilter(self._key_filter)
            self._held_keys.clear()

    def _manual_motors(self):
        speed = 60.0; mL = mR = 0.0; keys = self._held_keys
        if Qt.Key.Key_Space in keys:
            return 0.0, 0.0
        if Qt.Key.Key_W in keys: mL += speed; mR += speed
        if Qt.Key.Key_S in keys: mL -= speed; mR -= speed
        if Qt.Key.Key_A in keys: mL -= speed; mR += speed
        if Qt.Key.Key_D in keys: mL += speed; mR -= speed
        return mL, mR

    # _apply_task / _logger_* / _open_trajectory_viewer → sim_app_session._SessionMixin

    def _reset(self):
        if self.brain is not None:
            self.brain.layers      = self.circuit.layers
            self.brain.sensors     = self.circuit.sensors
            self.brain.connections = self.circuit.connections
            self.brain_mgr.resolve_joint_sensor_refs()
        self._sim_ctrl.reset()
        self._setup_world()
        self._arena.setup_sensors(self.circuit.sensors, self._osc_ctrl.channel_colors)

    # ── MuJoCo ───────────────────────────────────────────────────────────────

    def _start_mujoco(self):
        """MuJoCo is required: it owns movement, contacts and cameras."""
        ok, err = self._sim_ctrl.mujoco.start()
        if not ok:
            QMessageBox.critical(self, "MuJoCo error",
                                 f"The MuJoCo engine could not start, so the simulation cannot run.\n\n{err}")
            return
        self._mujoco_viewer_btn.setEnabled(True)
        self._view_3d_btn.setEnabled(True)
        self._view_3d_btn.setChecked(True)
        self._on_view_3d_toggle()
        print("[MuJoCo] engine started")

    def _on_view_3d_toggle(self):
        checked = self._view_3d_btn.isChecked()
        self._sim_ctrl.mujoco.set_view_3d(checked)
        if not checked:
            self._setup_world()

    # ── Brain loading ─────────────────────────────────────────────────────────

    # Agent table / add / remove / select / arena click → sim_app_agents._AgentsMixin

    # ── Joint management ──────────────────────────────────────────────────────

    def _add_joint(self):
        import math
        bodies = self.circuit.bodies
        if not bodies:
            return
        dlg = QDialog(self); dlg.setWindowTitle("Add Body")
        form = QFormLayout(dlg)

        parent_combo = QComboBox()
        for b in bodies:
            parent_combo.addItem(b.name, b.id)
        form.addRow("Parent body", parent_combo)
        name_edit = QLineEdit(f"body{len(bodies)}")
        form.addRow("Child name", name_edit)
        radius_spin = QDoubleSpinBox()
        radius_spin.setRange(0.01, 2.0); radius_spin.setSingleStep(0.01); radius_spin.setValue(0.08)
        form.addRow("Child radius", radius_spin)
        attach_dist_spin = QDoubleSpinBox()
        attach_dist_spin.setRange(0.0, 5.0); attach_dist_spin.setSingleStep(0.01)
        attach_dist_spin.setValue(0.18)
        form.addRow("Attach distance", attach_dist_spin)
        attach_angle_spin = QDoubleSpinBox()
        attach_angle_spin.setRange(-180.0, 180.0); attach_angle_spin.setSingleStep(1.0)
        attach_angle_spin.setValue(30.0); attach_angle_spin.setSuffix("°")
        form.addRow("Attach angle", attach_angle_spin)
        angle_min_spin = QDoubleSpinBox()
        angle_min_spin.setRange(-180.0, 0.0); angle_min_spin.setSingleStep(5.0)
        angle_min_spin.setValue(-90.0); angle_min_spin.setSuffix("°")
        form.addRow("Angle min", angle_min_spin)
        angle_max_spin = QDoubleSpinBox()
        angle_max_spin.setRange(0.0, 180.0); angle_max_spin.setSingleStep(5.0)
        angle_max_spin.setValue(90.0); angle_max_spin.setSuffix("°")
        form.addRow("Angle max", angle_max_spin)
        mirror_chk = QCheckBox(); mirror_chk.setChecked(False)
        form.addRow("Mirror on other side", mirror_chk)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        def _add():
            self.brain_mgr.add_joint(
                parent_id       = parent_combo.currentData(),
                layer_name      = name_edit.text().strip() or f'body{len(bodies)}',
                radius          = radius_spin.value(),
                dist            = attach_dist_spin.value(),
                attach_angle_deg= attach_angle_spin.value(),
                amin            = math.radians(angle_min_spin.value()),
                amax            = math.radians(angle_max_spin.value()),
                mirrored        = mirror_chk.isChecked(),
            )
        if self._net_viz:
            # One undoable edit in the network window's history.
            with self._net_viz._editor.edit():
                _add()
            self._net_viz._on_edit_committed()
            self._net_viz.build()
        else:
            _add()
            CircuitEditor(self.circuit, self.brain, brain_mgr=self.brain_mgr).sync_brain()
        poses = _rb_world_poses(self.bot_pos, self.circuit.bodies, self.circuit.joints)
        self._arena.update_child_bodies(poses, self.circuit.bodies, self.sim_cfg)
        if self._net_viz:
            self._net_viz.raise_(); self._net_viz.activateWindow()

    # _save_session / _load_session / _load_session_agents / _refresh_session_list → sim_app_session._SessionMixin

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Delete:
            self._remove_selected_agent()
            return
        if self._manual_active:
            # W / A / D drive the robot then — they must not also pick world tools.
            super().keyPressEvent(ev)
            return
        key = ev.text().upper()
        for letter, name, color, bg in GRADIENT_COLORS:
            if key == letter:
                self._set_gradient_mode(color, letter)
                return
        for letter, _, color, _ in OBJECT_COLORS[:5]:
            if key == letter:
                self._set_object_mode(color, letter)
                return
        super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self._save_latest_session()   # reopened at the next start
        self._sim_ctrl.close()
        super().closeEvent(ev)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # _app and _splash were created at the very top of the file, before heavy
    # imports, so the splash is visible during the entire loading phase
    _app.setStyleSheet(f"""
        QWidget {{ background:{C['bg']}; color:{C['dark']}; font-family:'Segoe UI'; font-size:9pt; }}
        QGroupBox {{ background:{C['surface']}; }}
        QLineEdit {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:3px; padding:2px 4px; }}
        QComboBox {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:3px; padding:2px 4px; }}
        QSpinBox  {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:3px; padding:2px 4px; }}
        QDoubleSpinBox {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:3px; padding:2px 4px; }}
        QScrollBar:vertical {{ background:{C['bg']}; width:8px; }}
        QScrollBar::handle:vertical {{ background:{C['border']}; border-radius:4px; }}
    """)
    win = SimulatorApp()
    _splash.set_progress(90)
    win.show()
    if _MIGRATED:
        win._status_bar.showMessage(
            f"Moved {len(_MIGRATED)} of your files to {data_paths.user_dir()} "
            "(the app folder is replaced by updates)", 15000)
    _splash.finish(win)
    try:
        sys.exit(_app.exec())
    except KeyboardInterrupt:
        pass
