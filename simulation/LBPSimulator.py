import sys
import os

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

# Force working directory to script location and add brains/ to import path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "brains"))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDockWidget, QScrollArea,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QSplitter,
    QStatusBar, QLineEdit, QRadioButton, QMessageBox, QMenu,
    QDialog, QFormLayout, QDialogButtonBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QColorDialog, QSplashScreen,
)
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, Signal, QEvent, QMimeData, QObject
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QDrag, QPixmap
if _splash: _splash.set_progress(60)

try:
    from sim_engine_mujoco import MuJoCoEngine
    _MUJOCO_AVAILABLE = True
except Exception:
    _MUJOCO_AVAILABLE = False

from rigid_body import RigidBody, Joint, world_poses as _rb_world_poses
from sim_constants import C, _CHAN_PALETTE, _TRAIL_COLOR, GRADIENT_COLORS, OBJECT_COLORS
from network_viz import NetworkVisualizerWindow
from circuit_model import CircuitModel, Connection
from tasks import discover_tasks, load_task
from logger import SimLogger

from neurons import MotorLayer
from brain_base import Param, ChoiceParam, BaseConfig, BaseBrain, DataBrain
from brain_manager import BrainManager
from sim_config import SimConfig
from world import World
from session_io import save_session, load_session

from arena_widget import ArenaViewBox, RobotItem, ChildBodyItem, CircleItem, ArenaWidget
from sim_widgets import MonetarySpinBox, OscilloscopeWidget, ControlPanel, _ManualKeyFilter
from world_editor import WorldEditor
from osc_controller import OscChannelManager
from sim_controller import SimController
if _splash: _splash.set_progress(80)


# ============================================================
# MAIN WINDOW
# ============================================================
class SimulatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LBP Simulator")
        self.resize(950, 800)
        screen = QApplication.primaryScreen().availableGeometry()
        self.move((screen.width() - 950) // 2, (screen.height() - 800) // 2)

        if not os.path.exists("configs"):
            os.makedirs("configs")

        # ── Shared state (no Qt deps) ─────────────────────────────────────────
        self.sim_cfg  = SimConfig()
        self.world    = World(self.sim_cfg)
        _circuit0 = CircuitModel()
        _circuit0.bodies = [RigidBody('root', 'root', self.sim_cfg.body_radius)]

        self._net_viz           = None
        self._connection_params = {}
        self._hidden_cols       = set()
        self._disabled_cols     = set()
        self._col_labels        = {}
        self._manual_active     = False
        self._held_keys         = set()
        self._key_filter        = None

        # ── Logger (shared with SimController) ───────────────────────────────
        self._logger = SimLogger()

        # ── Brain manager (per-agent; circuit/brain_mgr are proxy properties) ─
        _brain_mgr0 = BrainManager(_circuit0, self.sim_cfg)
        self.brain_files = _brain_mgr0.discover_brains()
        # One entry per table row; 'indices' lists flat _agents indices in this group
        self._agent_groups = [
            {'module': None, 'n': 1, 'color': '#4a7fcb', 'name': 'Group 1', 'indices': [0]}
        ]

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
        self._refresh_agent_list()

        self._editor = WorldEditor(
            world          = self.world,
            sim_cfg        = self.sim_cfg,
            arena          = self._arena,
            bot_pos        = self._sim_ctrl.bot_pos,
            setup_world_cb = self._setup_world,
            get_agents     = lambda: self._sim_ctrl._agents,
        )

        # ── Connect arena signals to editor ──────────────────────────────────
        self._arena.sigClick.connect(self._editor.handle_click)
        self._arena.sigClick.connect(self._on_arena_click)
        self._arena.sigDrag.connect(self._editor.handle_drag)
        self._arena.sigHover.connect(self._editor.handle_hover)

        # ── Populate combos and auto-load ─────────────────────────────────────
        self._refresh_brain_list()
        self._refresh_session_list()
        if self._session_combo.count() > 0:
            self._load_session()
        else:
            self._reset()

        if _MUJOCO_AVAILABLE:
            self._mujoco_cb.setChecked(True)
            self._view_3d_btn.setChecked(True)
            self._on_view_3d_toggle()

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def brain(self):
        return self._sim_ctrl.brain

    @brain.setter
    def brain(self, value):
        self._sim_ctrl.brain = value

    @property
    def bot_pos(self):
        return self._sim_ctrl.bot_pos

    @property
    def circuit(self):
        return self._sim_ctrl._agent.circuit

    @property
    def brain_mgr(self):
        return self._sim_ctrl._agent.brain_mgr

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self._arena = ArenaWidget()
        self._arena.setMaximumHeight(600)
        self.setCentralWidget(self._arena)

        self._panel = ControlPanel()
        self._left_dock = QDockWidget("Controls", self)
        self._left_dock.setWidget(self._panel)
        self._left_dock.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._left_dock)

        osc_container = QWidget()
        osc_lay = QHBoxLayout(osc_container)
        osc_lay.setContentsMargins(4, 4, 4, 4)
        osc_lay.setSpacing(6)

        self._osc_ctrl_inner  = QWidget()
        self._osc_mult_layout = QVBoxLayout(self._osc_ctrl_inner)
        self._osc_mult_layout.setContentsMargins(2, 2, 2, 2)
        self._osc_mult_layout.setSpacing(2)
        osc_ctrl_scroll = QScrollArea()
        osc_ctrl_scroll.setWidgetResizable(True)
        osc_ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        osc_ctrl_scroll.setFixedWidth(180)
        osc_ctrl_scroll.setWidget(self._osc_ctrl_inner)
        osc_lay.addWidget(osc_ctrl_scroll)

        self._osc = OscilloscopeWidget()
        osc_lay.addWidget(self._osc, 1)

        self._osc_dock = QDockWidget("Oscilloscope", self)
        self._osc_dock.setWidget(osc_container)
        self._osc_dock.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._osc_dock)
        self.setCorner(Qt.BottomLeftCorner,  Qt.BottomDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

        self._build_sim_group()
        tab_brain, tab_world, tab_session, tab_physics, tab_robot = self._panel.add_tab_widget(
            ["Brain", "World", "Session", "Physics", "Robot"])
        self._build_brain_group(tab_brain)
        self._build_world_group(tab_world)
        self._build_session_group(tab_session)
        self._build_task_group(tab_session)
        self._build_logger_group(tab_session)
        self._build_physics_group(tab_physics)
        self._build_robot_group(tab_robot)
        self._panel.add_stretch()

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("■  STOPPED")
        self._status_label.setStyleSheet(f"color:{C['muted']};font-weight:bold;padding:0 8px;")
        self._timing_label = QLabel("")
        self._timing_label.setStyleSheet(f"color:{C['muted']};padding:0 8px;")
        self._status_bar.addWidget(self._status_label)
        self._status_bar.addPermanentWidget(self._timing_label)

        self._osc_hidden_height = 0

        def _set_dock_sizes():
            self.resizeDocks([self._left_dock], [430], Qt.Horizontal)
            self.resizeDocks([self._osc_dock],  [220], Qt.Vertical)
            if not self._osc_cb.isChecked():
                self._osc_hidden_height = 220
                self._osc_dock.setVisible(False)
                self.resize(self.width(), self.height() - 220)
        QTimer.singleShot(0, _set_dock_sizes)

    def _make_btn(self, text, color=None, checkable=False):
        btn = QPushButton(text)
        bg  = color or C['surface']
        btn.setStyleSheet(f"""
            QPushButton {{
                background:{bg}; border:none; border-radius:3px;
                padding:5px 10px; font-weight:bold; color:{self._text_color(bg)};
            }}
            QPushButton:hover {{ background:{self._darken(bg)}; }}
            QPushButton:disabled {{ background:{C['border']}; color:{C['muted']}; }}
            QPushButton:checked {{ background:{self._darken(bg, 0.80)}; }}
        """)
        if checkable:
            btn.setCheckable(True)
        return btn

    @staticmethod
    def _darken(hex_color, factor=0.88):
        h = hex_color.lstrip('#')
        r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
        return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

    @staticmethod
    def _text_color(bg_hex):
        h = bg_hex.lstrip('#')
        r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
        return C['dark'] if (0.299*r + 0.587*g + 0.114*b) / 255 > 0.5 else 'white'

    def _make_param_row(self, parent_layout, label, p_obj, current_val, on_change, desc="",
                        choices=None):
        row = QWidget()
        hl  = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setFixedWidth(110)
        lbl.setStyleSheet(f"color:{C['dark']};font-size:9px;")
        hl.addWidget(lbl)

        if hasattr(p_obj, 'get_choices'):
            combo = QComboBox()
            choices = choices if choices is not None else p_obj.get_choices()
            combo.addItems([''] + choices)
            if current_val in choices:
                combo.setCurrentText(current_val)
            combo.currentTextChanged.connect(lambda v: on_change(v))
            hl.addWidget(combo)
            parent_layout.addWidget(row)
            return combo

        is_int = isinstance(current_val, int) and p_obj.step >= 1
        if is_int:
            spin = QSpinBox()
            spin.setMinimum(int(p_obj.min))
            spin.setMaximum(int(p_obj.max))
            spin.setValue(int(current_val))
            spin.valueChanged.connect(lambda v: on_change(v))
            hl.addWidget(spin)
            parent_layout.addWidget(row)
            return spin
        else:
            slider = QSlider(Qt.Horizontal)
            steps  = max(1, int((p_obj.max - p_obj.min) / p_obj.step))
            slider.setMinimum(0)
            slider.setMaximum(steps)
            slider.setValue(int((current_val - p_obj.min) / p_obj.step))
            slider.setFixedWidth(120)

            val_edit = QLineEdit(f"{current_val:.4g}")
            val_edit.setFixedWidth(60)
            val_edit.setStyleSheet(
                f"background:{C['surface']};border:1px solid {C['border']};border-radius:2px;")

            def _on_slider(v, po=p_obj, edit=val_edit, cb=on_change):
                val = po.min + v * po.step
                edit.setText(f"{val:.4g}")
                cb(val)

            def _on_edit(po=p_obj, sl=slider, edit=val_edit, cb=on_change):
                try:
                    val = float(edit.text())
                    val = max(po.min, min(po.max, val))
                    sl.blockSignals(True)
                    sl.setValue(int((val - po.min) / po.step))
                    sl.blockSignals(False)
                    edit.setText(f"{val:.4g}")
                    cb(val)
                except ValueError:
                    pass

            slider.valueChanged.connect(_on_slider)
            val_edit.returnPressed.connect(_on_edit)
            val_edit.editingFinished.connect(_on_edit)
            hl.addWidget(slider)
            hl.addWidget(val_edit)
            parent_layout.addWidget(row)
            return slider, val_edit

    # ── Panel builders ────────────────────────────────────────────────────────

    def _build_sim_group(self):
        gb, vl = self._panel.add_group("Simulation")

        row1 = QWidget(); hl1 = QHBoxLayout(row1); hl1.setContentsMargins(0, 0, 0, 0)
        self._btn_run_stop = self._make_btn("▶ Run",   C['success'])
        self._btn_step     = self._make_btn("⏭ Step", C['surface'])
        self._btn_reset    = self._make_btn("↺ Reset", C['surface'])
        for b in [self._btn_run_stop, self._btn_step, self._btn_reset]:
            hl1.addWidget(b)
        vl.addWidget(row1)
        self._btn_run_stop.clicked.connect(self._on_run_stop)
        self._btn_step.clicked.connect(lambda: self._sim_ctrl.step())
        self._btn_reset.clicked.connect(self._reset)

        row2 = QWidget(); hl2 = QHBoxLayout(row2); hl2.setContentsMargins(0, 0, 0, 0)
        hl2.addWidget(QLabel("Speed ×"))
        self._speed_spin = QSpinBox()
        self._speed_spin.setMinimum(1); self._speed_spin.setMaximum(500)
        self._speed_spin.setValue(1)
        self._speed_spin.valueChanged.connect(lambda v: self._sim_ctrl.set_speed_mult(v))
        hl2.addWidget(self._speed_spin)
        self._rt_cb = QCheckBox("Real time")
        self._rt_cb.setToolTip("Run only as many physics ticks per frame as wall-clock time demands")
        self._rt_cb.toggled.connect(self._on_rt_mode_toggled)
        hl2.addWidget(self._rt_cb)
        btn_clear = self._make_btn("✕ Clear World", C['muted'])
        btn_clear.clicked.connect(self._clear_world)
        hl2.addWidget(btn_clear)
        vl.addWidget(row2)

        row3 = QWidget(); rl3 = QHBoxLayout(row3); rl3.setContentsMargins(0, 0, 0, 0)
        self._fixate_cb = QCheckBox("Fixate")
        self._fixate_cb.setToolTip("Freeze robot position (physics still runs)")
        self._fixate_cb.setChecked(bool(self.sim_cfg.fixate_robot))
        self._fixate_cb.stateChanged.connect(
            lambda s: setattr(self.sim_cfg, 'fixate_robot', 1.0 if s else 0.0))
        rl3.addWidget(self._fixate_cb)
        self._stim_cb = QCheckBox("Show stimulus")
        self._stim_cb.setChecked(bool(self.sim_cfg.toggle_stim))
        self._stim_cb.stateChanged.connect(self._on_toggle_stim)
        rl3.addWidget(self._stim_cb)
        self._osc_cb = QCheckBox("Oscilloscope")
        self._osc_cb.setChecked(False)
        self._osc_cb.stateChanged.connect(self._on_toggle_osc)
        rl3.addWidget(self._osc_cb)
        self._robot_cb = QCheckBox("Real Robot")
        self._robot_cb.setToolTip(
            "Replace sim physics with live robot I/O.\n"
            "Each sensor's robot_address field specifies its host:port connection.\n"
            "Motor commands are sent as OSC to the motor layer's robot_address.")
        self._robot_cb.stateChanged.connect(self._on_robot_mode_toggle)
        rl3.addWidget(self._robot_cb)
        rl3.addStretch()
        vl.addWidget(row3)

        row4 = QWidget(); rl4 = QHBoxLayout(row4); rl4.setContentsMargins(0, 0, 0, 0)
        move_btn = QPushButton("↖ Move")
        move_btn.setCheckable(True)
        move_btn.setToolTip("Drag gradients, objects, or the robot to a new position")
        move_btn.setStyleSheet(
            f"QPushButton {{ background:{C['surface']}; border:2px solid {C['border']};"
            f" border-radius:3px; padding: 6px 12px; }}"
            f"QPushButton:checked {{ border:2px solid {C['primary']};"
            f" background:#1a3a5c; font-weight:bold; }}")
        move_btn.clicked.connect(self._set_move_mode)
        rl4.addWidget(move_btn)
        self._move_btn = move_btn

        manual_btn = QPushButton("⌨ Manual")
        manual_btn.setCheckable(True)
        manual_btn.setToolTip(
            "Manual control (WASD = steer, Space = stop).\nBrain simulation keeps running.")
        manual_btn.setStyleSheet(
            f"QPushButton {{ background:{C['surface']}; border:2px solid {C['border']};"
            f" border-radius:3px; padding: 6px 12px; }}"
            f"QPushButton:checked {{ border:2px solid {C['warning']};"
            f" background:#3a2a00; font-weight:bold; }}")
        manual_btn.clicked.connect(self._toggle_manual_mode)
        rl4.addWidget(manual_btn)
        self._manual_btn = manual_btn
        rl4.addStretch()
        vl.addWidget(row4)

        self._manual_hint = QLabel("W/S=fwd/back  A/D=turn  Space=stop")
        self._manual_hint.setStyleSheet(f"color:{C['muted']};font-size:8pt;")
        self._manual_hint.setVisible(False)
        vl.addWidget(self._manual_hint)

    def _build_session_group(self, panel_vl=None):
        gb, vl = self._panel.add_group("Sessions", panel_vl)
        save_row = QWidget(); sl = QHBoxLayout(save_row); sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(QLabel("Name:"))
        self._session_name = QLineEdit("experiment_1")
        sl.addWidget(self._session_name)
        btn_save = self._make_btn("Save", C['dark'])
        btn_save.clicked.connect(self._save_session)
        sl.addWidget(btn_save)
        vl.addWidget(save_row)

        load_row = QWidget(); ll = QHBoxLayout(load_row); ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Load:"))
        self._session_combo = QComboBox()
        ll.addWidget(self._session_combo)
        btn_load = self._make_btn("Load", C['muted'])
        btn_load.clicked.connect(self._load_session)
        ll.addWidget(btn_load)
        vl.addWidget(load_row)

    def _build_brain_group(self, panel_vl=None):
        gb, vl = self._panel.add_group("Brain", panel_vl)

        # ── Agent list ───────────────────────────────────────────────────────
        agent_hdr = QWidget(); ahl = QHBoxLayout(agent_hdr)
        ahl.setContentsMargins(0, 0, 0, 0)
        ahl.addWidget(QLabel("Agents"))
        ahl.addStretch()
        btn_add_agent = self._make_btn("+", C['primary'])
        btn_add_agent.setFixedWidth(24)
        btn_add_agent.setToolTip("Add agent")
        btn_add_agent.clicked.connect(self._add_agent)
        btn_rem_agent = self._make_btn("−", C['danger'])
        btn_rem_agent.setFixedWidth(24)
        btn_rem_agent.setToolTip("Remove selected agent")
        btn_rem_agent.clicked.connect(self._remove_agent)
        ahl.addWidget(btn_add_agent)
        ahl.addWidget(btn_rem_agent)
        vl.addWidget(agent_hdr)

        self._agent_table = QTableWidget(0, 4)
        self._agent_table.setHorizontalHeaderLabels(["", "Name", "Brain", "N"])
        hdr = self._agent_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.Fixed)
        self._agent_table.setColumnWidth(0, 28)
        self._agent_table.setColumnWidth(3, 48)
        self._agent_table.verticalHeader().setVisible(False)
        self._agent_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._agent_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._agent_table.setFixedHeight(96)
        self._agent_table.currentCellChanged.connect(
            lambda row, col, pr, pc: self._on_agent_selected(row) if row != pr else None
        )
        self._agent_table.itemChanged.connect(self._on_agent_name_changed)
        vl.addWidget(self._agent_table)

        # ── Brain combo ──────────────────────────────────────────────────────
        row1 = QWidget(); hl1 = QHBoxLayout(row1); hl1.setContentsMargins(0, 0, 0, 0)
        self._brain_combo = QComboBox()
        self._brain_combo.addItems(self.brain_files)
        self._brain_combo.currentTextChanged.connect(self.load_brain)
        btn_reload = self._make_btn("↺", C['warning'])
        btn_reload.setFixedWidth(28)
        btn_reload.setToolTip("Reload brain")
        btn_reload.clicked.connect(self.load_brain)
        btn_new_brain = self._make_btn("+", C['primary'])
        btn_new_brain.setFixedWidth(28)
        btn_new_brain.setToolTip("Scaffold a new brain file")
        btn_new_brain.clicked.connect(self._new_brain)
        hl1.addWidget(self._brain_combo)
        hl1.addWidget(btn_reload)
        hl1.addWidget(btn_new_brain)
        vl.addWidget(row1)
        self._brain_params_group, self._brain_params_layout = \
            self._panel.add_group("Brain Parameters", panel_vl)

    def _build_task_group(self, panel_vl=None):
        gb, vl = self._panel.add_group("Task", panel_vl)
        row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0)
        self._task_combo = QComboBox()
        self._task_combo.addItem("— none —")
        self._task_combo.addItems(discover_tasks())
        hl.addWidget(self._task_combo)
        btn_apply = self._make_btn("Apply", C['primary'])
        btn_apply.clicked.connect(self._apply_task)
        hl.addWidget(btn_apply)
        vl.addWidget(row)

    def _build_logger_group(self, panel_vl=None):
        gb, vl = self._panel.add_group("Logger", panel_vl)
        row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0)
        self._btn_log_start = self._make_btn("● Record", C['danger'])
        self._btn_log_stop  = self._make_btn("■ Stop",   C['muted'])
        self._btn_log_stop.setEnabled(False)
        self._btn_log_start.clicked.connect(self._logger_start)
        self._btn_log_stop.clicked.connect(self._logger_stop)
        hl.addWidget(self._btn_log_start)
        hl.addWidget(self._btn_log_stop)
        vl.addWidget(row)
        self._log_label = QLabel("Idle")
        self._log_label.setStyleSheet("color: grey; font-size: 8pt;")
        vl.addWidget(self._log_label)
        btn_viz = self._make_btn("Visualize trajectories", C['primary'])
        btn_viz.clicked.connect(self._open_trajectory_viewer)
        vl.addWidget(btn_viz)

    def _build_physics_group(self, panel_vl=None):
        gb, vl = self._panel.add_group("Physics", panel_vl)
        self._phys_widgets = {}
        phys_params = ['dt', 'motor_gain', 'body_radius', 'arena_scale',
                       'sense_radius', 'sensor_angle', 'init_x', 'init_y']
        meta = self.sim_cfg.get_param_metadata()
        for k in phys_params:
            if k not in meta:
                continue
            p  = meta[k]
            cv = getattr(self.sim_cfg, k)
            result = self._make_param_row(vl, k, p, cv,
                lambda v, key=k: self._on_sim_param(key, v), desc=p.desc)
            self._phys_widgets[k] = result

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
            _, _, osc_path, _, _ = _parse_address(addr)
            hz_lbl = self._add_robot_row('S', sensor.name, osc_path or addr)
            if osc_path:
                self._robot_hz_labels[osc_path] = hz_lbl

        for layer in circuit.layers:
            if not isinstance(layer, MotorLayer):
                continue
            addr = getattr(layer, 'robot_address', '').strip()
            if not addr:
                continue
            _, _, osc_path, _, _ = _parse_address(addr)
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

    def _update_robot_hz(self):
        """Called every second — refresh Hz labels only when robot mode is active."""
        if not self._sim_ctrl._robot_mode:
            return
        sensor_rates = self._sim_ctrl._robot_driver.get_rates()
        for path, lbl in self._robot_hz_labels.items():
            lbl.setText(f"{sensor_rates.get(path, 0.0):.0f} Hz")
        mt = self._sim_ctrl._motor_thread
        if mt:
            motor_rates = mt.send_rates()
            for path, lbl in self._robot_motor_hz_labels.items():
                lbl.setText(f"{motor_rates.get(path, 0.0):.0f} Hz")

    def _build_world_group(self, panel_vl=None):
        gb, vl = self._panel.add_group("World", panel_vl)

        trail_row = QWidget(); tl = QHBoxLayout(trail_row); tl.setContentsMargins(0, 0, 0, 0)
        self._trail_cb = QCheckBox("Trail")
        self._trail_cb.setChecked(True)
        tl.addWidget(self._trail_cb)
        tl.addWidget(QLabel("Len:"))
        self._trail_len_spin = QSpinBox()
        self._trail_len_spin.setMinimum(10); self._trail_len_spin.setMaximum(5000)
        self._trail_len_spin.setSingleStep(50); self._trail_len_spin.setValue(500)
        self._trail_len_spin.valueChanged.connect(self._on_trail_len_change)
        tl.addWidget(self._trail_len_spin)
        vl.addWidget(trail_row)

        mujoco_row = QWidget(); ml = QHBoxLayout(mujoco_row); ml.setContentsMargins(0, 0, 0, 0)
        self._mujoco_cb = QCheckBox("3D (MuJoCo)")
        self._mujoco_cb.setToolTip(
            "Enable MuJoCo 3D engine.\nCamera sensors render a real 3D perspective image.\n"
            "Physics (collision, gradients) stays 2D.\nWorld edits reload the MuJoCo model.")
        self._mujoco_cb.setEnabled(_MUJOCO_AVAILABLE)
        if not _MUJOCO_AVAILABLE:
            self._mujoco_cb.setToolTip("mujoco package not installed (pip install mujoco)")
        self._mujoco_cb.stateChanged.connect(self._on_mujoco_toggle)
        ml.addWidget(self._mujoco_cb)
        self._mujoco_viewer_btn = QPushButton("Show 3D")
        self._mujoco_viewer_btn.setFixedHeight(22)
        self._mujoco_viewer_btn.setEnabled(False)
        self._mujoco_viewer_btn.setToolTip("Open the MuJoCo 3D viewer window")
        self._mujoco_viewer_btn.clicked.connect(lambda: self._sim_ctrl.show_mujoco_viewer())
        ml.addWidget(self._mujoco_viewer_btn)
        self._view_3d_btn = QPushButton("Top view")
        self._view_3d_btn.setFixedHeight(22)
        self._view_3d_btn.setCheckable(True)
        self._view_3d_btn.setEnabled(False)
        self._view_3d_btn.setToolTip(
            "Show MuJoCo overhead (top-down) render in the arena.\n"
            "Object editing still works — switch back to 2D to see gradients.")
        self._view_3d_btn.clicked.connect(self._on_view_3d_toggle)
        ml.addWidget(self._view_3d_btn)
        ml.addStretch()
        vl.addWidget(mujoco_row)

        arena_row = QWidget(); al = QHBoxLayout(arena_row); al.setContentsMargins(0, 0, 0, 0)
        al.addWidget(QLabel("Arena:"))
        self._arena_square_rb = QRadioButton("Square")
        self._arena_round_rb  = QRadioButton("Round")
        self._arena_square_rb.setChecked(True)
        self._arena_square_rb.toggled.connect(self._on_arena_type_change)
        al.addWidget(self._arena_square_rb); al.addWidget(self._arena_round_rb)
        vl.addWidget(arena_row)

        grad_row = QWidget(); gl = QHBoxLayout(grad_row); gl.setContentsMargins(0, 0, 0, 0)
        grad_lbl = QLabel("Gradients:")
        grad_lbl.setStyleSheet(f"color:{C['dark']};font-weight:bold;")
        gl.addWidget(grad_lbl)
        self._grad_btns = {}
        for letter, name, color, bg in GRADIENT_COLORS:
            btn = QPushButton(letter)
            btn.setFixedSize(26, 26); btn.setCheckable(True)
            btn.setStyleSheet(
                f"background:{bg};border:2px solid {C['border']};border-radius:3px;font-weight:bold;")
            btn.clicked.connect(lambda checked, c=color, l=letter: self._set_gradient_mode(c, l))
            gl.addWidget(btn)
            self._grad_btns[letter] = btn
        self._grad_cont_btn = QPushButton("Cont.")
        self._grad_cont_btn.setFixedHeight(26)
        self._grad_cont_btn.setCheckable(True)
        self._grad_cont_btn.setToolTip("Continuous: flat value inside patch (no gradient falloff)")
        self._grad_cont_btn.setStyleSheet(
            f"QPushButton{{border:2px solid {C['border']};border-radius:3px;font-weight:bold;}}"
            f"QPushButton:checked{{background:#444444;color:white;"
            f"border:2px solid #000;border-radius:3px;font-weight:bold;}}")
        self._grad_cont_btn.clicked.connect(self._toggle_gradient_continuous)
        gl.addWidget(self._grad_cont_btn)
        wall_btn = QPushButton("Wall")
        wall_btn.setFixedHeight(26); wall_btn.setCheckable(True)
        wall_btn.setStyleSheet(
            f"QPushButton {{ background:{C['surface']}; border:2px solid {C['border']};"
            f" border-radius:3px; }}"
            f"QPushButton:checked {{ border:2px solid {C['warning']};"
            f" background:#3a2a00; font-weight:bold; }}")
        wall_btn.clicked.connect(self._set_wall_mode)
        gl.addWidget(wall_btn)
        self._wall_btn = wall_btn
        vl.addWidget(grad_row)

        obj_row = QWidget(); ol = QHBoxLayout(obj_row); ol.setContentsMargins(0, 0, 0, 0)
        obj_lbl = QLabel("Objects:")
        obj_lbl.setStyleSheet(f"color:{C['dark']};font-weight:bold;")
        ol.addWidget(obj_lbl)
        self._obj_btns = {}
        for letter, _, color, bg in OBJECT_COLORS[:5]:
            btn = QPushButton(letter)
            btn.setFixedSize(26, 26); btn.setCheckable(True)
            btn.setStyleSheet(
                f"background:{bg};border:2px solid {C['border']};border-radius:3px;font-weight:bold;")
            btn.clicked.connect(lambda *_, c=color, l=letter: self._set_object_mode(c, l))
            ol.addWidget(btn)
            self._obj_btns[letter] = btn
        self._obj_picker_btn = QPushButton("…")
        self._obj_picker_btn.setFixedSize(26, 26); self._obj_picker_btn.setCheckable(True)
        self._obj_picker_btn.setToolTip("Pick a custom color")
        self._obj_picker_btn.setStyleSheet(
            f"border:2px solid {C['border']};border-radius:3px;font-weight:bold;")
        self._obj_picker_btn.clicked.connect(self._pick_object_color)
        ol.addWidget(self._obj_picker_btn)
        self._obj_wall_btn = QPushButton("Wall")
        self._obj_wall_btn.setFixedHeight(26); self._obj_wall_btn.setCheckable(True)
        self._obj_wall_btn.setStyleSheet(
            f"QPushButton {{ background:{C['surface']}; border:2px solid {C['border']};"
            f" border-radius:3px; }}"
            f"QPushButton:checked {{ border:2px solid {C['warning']};"
            f" background:#3a2a00; font-weight:bold; }}")
        self._obj_wall_btn.clicked.connect(self._set_obj_wall_mode)
        ol.addWidget(self._obj_wall_btn)
        self._poly_ext_btn = QPushButton("Solid")
        self._poly_ext_btn.setFixedHeight(26); self._poly_ext_btn.setCheckable(True)
        self._poly_ext_btn.setChecked(True)
        self._poly_ext_btn.setStyleSheet(
            f"QPushButton{{border:2px solid {C['border']};border-radius:3px;font-weight:bold;}}"
            f"QPushButton:checked{{background:#444444;color:white;"
            f"border:2px solid #000;border-radius:3px;font-weight:bold;}}")
        self._poly_ext_btn.clicked.connect(self._toggle_poly_external)
        ol.addWidget(self._poly_ext_btn)
        vl.addWidget(obj_row)

        sky_row = QWidget(); skl = QHBoxLayout(sky_row); skl.setContentsMargins(0, 0, 0, 0)
        sky_lbl = QLabel("Sky:")
        sky_lbl.setStyleSheet(f"color:{C['dark']};font-weight:bold;")
        skl.addWidget(sky_lbl)
        self._sky_cb = QCheckBox("Polarization field")
        self._sky_cb.setChecked(False)
        self._sky_cb.stateChanged.connect(self._on_sky_toggle)
        skl.addWidget(self._sky_cb)
        sky_btn = QPushButton("↕")
        sky_btn.setFixedSize(26, 26); sky_btn.setCheckable(True)
        sky_btn.setToolTip("Drag to set e-vector direction")
        sky_btn.setStyleSheet(
            f"background:#FFEEAA;border:2px solid {C['border']};border-radius:3px;font-weight:bold;")
        sky_btn.clicked.connect(self._set_sky_mode)
        skl.addWidget(sky_btn); skl.addStretch()
        vl.addWidget(sky_row)
        self._sky_btn = sky_btn

        self._set_gradient_mode(GRADIENT_COLORS[0][2], GRADIENT_COLORS[0][0])
        self._set_object_mode(OBJECT_COLORS[0][2], OBJECT_COLORS[0][0])

    # ── Draw mode button management ───────────────────────────────────────────

    def _clear_mode_buttons(self, keep=None):
        for b in self._grad_btns.values():
            b.setChecked(False)
        for b in self._obj_btns.values():
            b.setChecked(False)
        self._obj_picker_btn.setChecked(False)
        self._wall_btn.setChecked(False)
        self._obj_wall_btn.setChecked(False)
        if self._move_btn:
            self._move_btn.setChecked(False)
        if self._sky_btn:
            self._sky_btn.setChecked(False)

    def _set_gradient_mode(self, color, letter):
        self._editor.set_gradient_mode(color, letter) if hasattr(self, '_editor') else None
        self._clear_mode_buttons()
        self._grad_btns[letter].setChecked(True)

    def _set_object_mode(self, color, letter=None):
        if hasattr(self, '_editor'):
            self._editor.set_object_mode(color)
        self._clear_mode_buttons()
        if letter is not None and letter in self._obj_btns:
            self._obj_btns[letter].setChecked(True)
        else:
            r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
            self._obj_picker_btn.setStyleSheet(
                f"background:#{r:02x}{g:02x}{b:02x};"
                f"border:2px solid {C['border']};border-radius:3px;font-weight:bold;")
            self._obj_picker_btn.setChecked(True)

    def _pick_object_color(self):
        cur = self._editor.object_color if hasattr(self, '_editor') else [1.0, 0.0, 0.0]
        qc = QColor(int(cur[0]*255), int(cur[1]*255), int(cur[2]*255))
        new_qc = QColorDialog.getColor(qc, self, "Object / wall color")
        if new_qc.isValid():
            color = [new_qc.red()/255, new_qc.green()/255, new_qc.blue()/255]
            if hasattr(self, '_editor'):
                self._editor.object_color = color
            for b in self._obj_btns.values():
                b.setChecked(False)
            r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
            self._obj_picker_btn.setStyleSheet(
                f"background:#{r:02x}{g:02x}{b:02x};"
                f"border:2px solid {C['border']};border-radius:3px;font-weight:bold;")
            self._obj_picker_btn.setChecked(True)
        else:
            self._obj_picker_btn.setChecked(False)

    def _set_obj_wall_mode(self):
        if self._obj_wall_btn.isChecked():
            if hasattr(self, '_editor'):
                self._editor.set_wall_paint_mode()
            self._clear_mode_buttons()
            self._obj_wall_btn.setChecked(True)
        else:
            if hasattr(self, '_editor'):
                self._editor.set_object_mode(self._editor.object_color)

    def _toggle_gradient_continuous(self):
        cont = self._grad_cont_btn.isChecked()
        if hasattr(self, '_editor'):
            self._editor.gradient_continuous = cont

    def _set_wall_mode(self):
        if self._wall_btn.isChecked():
            self._editor.set_wall_mode()
            self._clear_mode_buttons()
            self._wall_btn.setChecked(True)
        else:
            self._editor.draw_mode = 'gradient'

    def _set_sky_mode(self):
        self._editor.set_sky_mode()
        self._clear_mode_buttons()
        self._sky_btn.setChecked(True)

    def _set_move_mode(self):
        if self._move_btn.isChecked():
            self._editor.set_move_mode()
            self._clear_mode_buttons()
            self._move_btn.setChecked(True)
        else:
            self._editor.draw_mode = 'gradient'

    def _toggle_poly_external(self):
        is_external = self._editor.toggle_poly_external()
        self._poly_ext_btn.setText("Solid" if is_external else "Room")

    def _on_sky_toggle(self, state):
        self.world.sky["enabled"] = bool(state)
        self._setup_world()

    # ── Sim param handlers ────────────────────────────────────────────────────

    def _on_sim_param(self, key, val):
        setattr(self.sim_cfg, key, val)
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
        for agent in self._sim_ctrl._agents:
            old = list(agent.trail_xy)[-n:]
            agent.trail_xy = deque(old, maxlen=n)

    def _on_rt_mode_toggled(self, checked):
        self._sim_ctrl.set_rt_mode(checked)
        self._speed_spin.setEnabled(not checked)

    def _on_status_changed(self, text, color):
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color:{color};font-weight:bold;padding:0 8px;")
        if text.startswith("●"):
            self._btn_run_stop.setText("■ Stop")
            self._btn_run_stop.setStyleSheet(self._make_btn("■ Stop", C['danger']).styleSheet())
        else:
            self._btn_run_stop.setText("▶ Run")
            self._btn_run_stop.setStyleSheet(self._make_btn("▶ Run", C['success']).styleSheet())

    def _on_run_stop(self):
        if self._sim_ctrl.running:
            self._sim_ctrl.stop()
        else:
            self._reset()
            self._sim_ctrl.start()

    # ── Manual control ────────────────────────────────────────────────────────

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
            self.installEventFilter(self._key_filter)
        else:
            if self._key_filter is not None:
                self.removeEventFilter(self._key_filter)
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

    # ── Task ─────────────────────────────────────────────────────────────────

    def _apply_task(self):
        name = self._task_combo.currentText()
        if name == "— none —":
            self._sim_ctrl._active_task = None
            return
        try:
            task = load_task(name)
            task.setup(self.world, self.sim_cfg)
            self._sim_ctrl._active_task = task
        except Exception as e:
            QMessageBox.critical(self, 'Task error', str(e))

    # ── Logger ────────────────────────────────────────────────────────────────

    def _logger_start(self):
        self._logger.start(arena_scale=self.sim_cfg.arena_scale,
                           arena_round=self.world.arena_round)
        self._btn_log_start.setEnabled(False)
        self._btn_log_stop.setEnabled(True)
        self._log_label.setText(f"Recording → {self._logger._path}")

    def _logger_stop(self):
        self._logger.stop()
        self._btn_log_start.setEnabled(True)
        self._btn_log_stop.setEnabled(False)
        self._log_label.setText(f"Saved {self._logger.row_count} rows")

    def _open_trajectory_viewer(self):
        from trajectory_viz import TrajectoryWindow as TrajectoryViewer
        viewer = TrajectoryViewer.for_latest(log_dir='logs', parent=self)
        if viewer is None:
            QMessageBox.information(self, "No logs",
                                    "No trajectory files found in logs/.\n"
                                    "Record a session first with ● Record.")
            return
        viewer.show()
        self._traj_viewer = viewer   # keep reference so the window stays open

    # ── World setup ──────────────────────────────────────────────────────────

    def _setup_world(self, rebuild=True):
        self._arena.setup(self.sim_cfg, self.world)
        if rebuild:
            self._sim_ctrl.rebuild_mujoco(self.world, self.sim_cfg)
            if self._sim_ctrl._view_3d and not self._sim_ctrl.running:
                self._sim_ctrl.render_mujoco_overhead()

    def _clear_world(self):
        self.world.patches = []
        self.world.objects = []
        self._setup_world()

    def _reset(self):
        if self.brain is not None:
            self.brain.layers      = self.circuit.layers
            self.brain.sensors     = self.circuit.sensors
            self.brain.connections = self.circuit.connections
            self.brain_mgr.resolve_joint_sensor_refs()
        self._sim_ctrl.reset()
        self._setup_world()
        self._arena.setup_sensors(self.circuit.sensors, self._osc_ctrl.channel_colors)

    # ── Real robot ───────────────────────────────────────────────────────────

    def _on_robot_mode_toggle(self, state):
        self._sim_ctrl.enable_robot_mode(bool(state))
        self._rebuild_channels()
        if state:
            self._robot_status_lbl.setText("● Online")
            self._robot_status_lbl.setStyleSheet("color: green; font-weight: bold;")
        else:
            self._robot_status_lbl.setText("● Offline")
            self._robot_status_lbl.setStyleSheet("color: gray; font-weight: bold; font-size: 8pt;")
            for lbl in {**self._robot_hz_labels, **self._robot_motor_hz_labels}.values():
                lbl.setText("-- Hz")

    # ── MuJoCo ───────────────────────────────────────────────────────────────

    def _on_mujoco_toggle(self, state):
        if state and not _MUJOCO_AVAILABLE:
            self._mujoco_cb.setChecked(False)
            QMessageBox.warning(self, "MuJoCo unavailable",
                                "The mujoco package is not installed.\nRun: pip install mujoco")
            return
        ok, err = self._sim_ctrl.enable_mujoco(state, self.world, self.sim_cfg)
        if state and not ok:
            self._mujoco_cb.setChecked(False)
            QMessageBox.warning(self, "MuJoCo error", err)
            return
        has_engine = self._sim_ctrl._mujoco_engine is not None
        self._mujoco_viewer_btn.setEnabled(has_engine)
        self._view_3d_btn.setEnabled(has_engine)
        if not state:
            self._view_3d_btn.setChecked(False)
            self._setup_world()
        print(f"[MuJoCo] engine {'started' if state else 'stopped'}")

    def _on_view_3d_toggle(self):
        checked = self._view_3d_btn.isChecked()
        self._sim_ctrl.set_view_3d(checked)
        if not checked:
            self._setup_world()

    # ── Brain loading ─────────────────────────────────────────────────────────

    # ── Agent management ──────────────────────────────────────────────────────

    def _refresh_agent_list(self):
        """Rebuild the agent table from _agent_groups (one row per group)."""
        self._agent_table.blockSignals(True)
        self._agent_table.setRowCount(0)
        for i, group in enumerate(self._agent_groups):
            self._agent_table.insertRow(i)

            # Col 0: color swatch button (centered in cell)
            btn = QPushButton()
            btn.setFixedSize(18, 18)
            btn.setStyleSheet(
                f"background-color: {group['color']}; border: none; border-radius: 3px;"
            )
            btn.setToolTip("Click to change group color")
            btn.clicked.connect(lambda checked, idx=i: self._pick_agent_color(idx))
            swatch_container = QWidget()
            swatch_layout = QHBoxLayout(swatch_container)
            swatch_layout.addWidget(btn)
            swatch_layout.setAlignment(Qt.AlignCenter)
            swatch_layout.setContentsMargins(0, 0, 0, 0)
            self._agent_table.setCellWidget(i, 0, swatch_container)

            # Col 1: group name (editable)
            name_item = QTableWidgetItem(group['name'])
            self._agent_table.setItem(i, 1, name_item)

            # Col 2: brain module (read-only)
            brain_item = QTableWidgetItem(group['module'] or "")
            brain_item.setFlags(brain_item.flags() & ~Qt.ItemIsEditable)
            self._agent_table.setItem(i, 2, brain_item)

            # Col 3: N spinbox (setValue before connecting to avoid spurious signal)
            sb = QSpinBox()
            sb.setRange(1, 20)
            sb.setValue(group['n'])
            sb.valueChanged.connect(lambda v, g=i: self._on_agent_n_changed(g, v))
            self._agent_table.setCellWidget(i, 3, sb)

        group_row = self._group_of_agent(self._sim_ctrl._selected)
        self._agent_table.setCurrentCell(group_row, 1)
        self._agent_table.blockSignals(False)

    def _add_agent(self, color=None, name=None):
        """Add a new agent group (n=1) with an optional brain load."""
        from arena_widget import _AGENT_COLORS
        new_circuit = CircuitModel()
        new_circuit.bodies = [RigidBody('root', 'root', self.sim_cfg.body_radius)]
        new_brain_mgr = BrainManager(new_circuit, self.sim_cfg)
        if color is None:
            color = _AGENT_COLORS[len(self._agent_groups) % len(_AGENT_COLORS)]
        if name is None:
            name = f'Group {len(self._agent_groups) + 1}'
        next_agent_idx = len(self._sim_ctrl._agents)
        idx = self._sim_ctrl.add_agent(new_circuit, new_brain_mgr,
                                       name=f"Agent {next_agent_idx + 1}", color=color)
        self._arena.add_robot_item(idx, color=color)
        self._agent_groups.append({
            'module': None, 'n': 1, 'color': color, 'name': name, 'indices': [idx]
        })
        self._refresh_agent_list()
        # Select the new group and load the current brain into it
        self._select_agent(idx)
        current_brain = self._brain_combo.currentText()
        if current_brain:
            self.load_brain(current_brain)

    def _remove_agent(self):
        """Remove the currently selected group and all its agents (min 1 agent total)."""
        row = self._agent_table.currentRow()
        if row < 0 or row >= len(self._agent_groups):
            return
        group = self._agent_groups[row]
        # Refuse if this would leave zero agents
        other_total = len(self._sim_ctrl._agents) - group['n']
        if other_total < 1:
            return
        # Remove all agents in this group (highest index first to preserve lower indices)
        for agent_idx in sorted(group['indices'], reverse=True):
            self._arena.remove_robot_item(agent_idx)
            self._sim_ctrl.remove_agent(agent_idx)
            self._decrement_indices_above(agent_idx)
        self._agent_groups.pop(row)
        self._refresh_agent_list()

    def _pick_agent_color(self, group_idx):
        """Open a color dialog and apply the chosen color to all agents in the group."""
        if group_idx >= len(self._agent_groups):
            return
        group = self._agent_groups[group_idx]
        color = QColorDialog.getColor(QColor(group['color']), self)
        if color.isValid():
            group['color'] = color.name()
            for agent_idx in group['indices']:
                if agent_idx < len(self._sim_ctrl._agents):
                    self._sim_ctrl._agents[agent_idx].color = color.name()
                    self._arena.update_robot_color(agent_idx, color.name())
            self._refresh_agent_list()

    def _on_agent_name_changed(self, item):
        """Persist inline name edits back to the group data model."""
        if item.column() == 1:
            row = item.row()
            if 0 <= row < len(self._agent_groups):
                self._agent_groups[row]['name'] = item.text()

    def _on_agent_selected(self, row):
        """Select the first agent in the clicked group row."""
        if row < 0 or row >= len(self._agent_groups):
            return
        group = self._agent_groups[row]
        if not group['indices']:
            return
        self._select_agent(group['indices'][0])

    # ── Group / agent helpers ─────────────────────────────────────────────────

    def _group_of_agent(self, agent_idx):
        """Return the group index that owns agent_idx, or 0 if not found."""
        for i, g in enumerate(self._agent_groups):
            if agent_idx in g['indices']:
                return i
        return 0

    def _decrement_indices_above(self, removed_idx):
        """After removing flat agent at removed_idx, shift all stored indices down by 1."""
        for g in self._agent_groups:
            g['indices'] = [i - 1 if i > removed_idx else i for i in g['indices']]

    def _select_agent(self, agent_idx):
        """Select a specific agent for oscilloscope / network viz, sync group table row."""
        if agent_idx < 0 or agent_idx >= len(self._sim_ctrl._agents):
            return
        self._sim_ctrl.select_agent(agent_idx)
        self._arena.select_robot(agent_idx)
        if hasattr(self, '_editor'):
            self._editor._bot_pos = self._sim_ctrl.bot_pos
        group_idx = self._group_of_agent(agent_idx)
        if group_idx < len(self._agent_groups):
            mod = self._agent_groups[group_idx]['module']
            self._brain_combo.blockSignals(True)
            if mod:
                self._brain_combo.setCurrentText(mod)
            self._brain_combo.blockSignals(False)
        self._rebuild_brain_params()
        self._rebuild_channels()
        self._arena.setup_sensors(self.circuit.sensors, self._osc_ctrl.channel_colors)
        if self._net_viz:
            self._net_viz.build()
        # Sync table row to the owning group (without triggering _on_agent_selected)
        self._agent_table.blockSignals(True)
        self._agent_table.setCurrentCell(group_idx, 1)
        self._agent_table.blockSignals(False)

    def _on_agent_n_changed(self, group_idx, new_n):
        """Spinbox value changed: add or remove agents for the group."""
        if group_idx >= len(self._agent_groups):
            return
        group = self._agent_groups[group_idx]
        old_n = group['n']
        delta = new_n - old_n
        prev_sel = self._sim_ctrl._selected
        if delta > 0:
            for _ in range(delta):
                self._add_agent_to_group(group_idx)
        elif delta < 0:
            for _ in range(-delta):
                self._remove_agent_from_group(group_idx)
        # Restore the agent that was selected before the resize
        restore_idx = min(prev_sel, len(self._sim_ctrl._agents) - 1)
        self._select_agent(restore_idx)

    def _add_agent_to_group(self, group_idx):
        """Spawn one more agent for the given group and load its brain."""
        group = self._agent_groups[group_idx]
        new_circuit = CircuitModel()
        new_circuit.bodies = [RigidBody('root', 'root', self.sim_cfg.body_radius)]
        new_brain_mgr = BrainManager(new_circuit, self.sim_cfg)
        next_total = len(self._sim_ctrl._agents)
        idx = self._sim_ctrl.add_agent(new_circuit, new_brain_mgr,
                                       name=f"Agent {next_total + 1}",
                                       color=group['color'])
        self._arena.add_robot_item(idx, color=group['color'])
        group['indices'].append(idx)
        group['n'] += 1
        if group['module']:
            # Copy brain params from the group's original agent (index 0 in the list,
            # which is always the first agent added to this group). This carries over
            # network_file/network_project for DataBrain so load_brain can find and
            # load the same JSON network for the new agent.
            original_brain = self._sim_ctrl._agents[group['indices'][0]].brain
            params_copy = (
                {k: getattr(original_brain, k) for k in original_brain.get_param_metadata()}
                if original_brain is not None else None
            )
            # Temporarily select the new agent so load_brain targets it correctly
            self._sim_ctrl.select_agent(idx)
            self.load_brain(group['module'], external_params=params_copy)
            # Selection is restored by _on_agent_n_changed after all agents are added

    def _remove_agent_from_group(self, group_idx):
        """Remove the last agent from the group (refuses if it would leave zero total)."""
        if len(self._sim_ctrl._agents) <= 1:
            return
        group = self._agent_groups[group_idx]
        if not group['indices']:
            return
        agent_idx = group['indices'].pop()
        self._arena.remove_robot_item(agent_idx)
        self._sim_ctrl.remove_agent(agent_idx)
        group['n'] -= 1
        self._decrement_indices_above(agent_idx)

    def _on_arena_click(self, x, y, btn):
        """Select the nearest agent when the user left-clicks the arena."""
        if btn != 1 or len(self._sim_ctrl._agents) <= 1:
            return
        r_thresh = self._sim_ctrl.sim_cfg.body_radius * 2.5
        best_idx, best_dist = -1, float('inf')
        for i, agent in enumerate(self._sim_ctrl._agents):
            dx = agent.bot_pos[0] - x
            dy = agent.bot_pos[1] - y
            d = (dx * dx + dy * dy) ** 0.5
            if d < r_thresh and d < best_dist:
                best_idx, best_dist = i, d
        if best_idx >= 0:
            self._select_agent(best_idx)

    def _refresh_brain_list(self):
        current = self._brain_combo.currentText()
        self.brain_files = self.brain_mgr.discover_brains()
        self._brain_combo.blockSignals(True)
        self._brain_combo.clear()
        self._brain_combo.addItems(self.brain_files)
        if current in self.brain_files:
            self._brain_combo.setCurrentText(current)
        self._brain_combo.blockSignals(False)

    def _new_network_project(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, 'New Project', 'Project directory name:')
        if not ok or not name.strip():
            return
        name = name.strip()
        os.makedirs(os.path.join('networks', name), exist_ok=True)
        self.brain.network_project = name
        self.brain.network_file = ''
        self._rebuild_brain_params()

    def _new_network_from_sidebar(self):
        from PySide6.QtWidgets import QInputDialog
        from brain_serializer import save_network_file
        name, ok = QInputDialog.getText(self, 'New Network', 'Network name:')
        if not ok or not name.strip():
            return
        name = name.strip()
        if not name.endswith('.json'):
            name += '.json'
        project = getattr(self.brain, 'network_project', '')
        net_dir = os.path.join('networks', project) if project else 'networks'
        os.makedirs(net_dir, exist_ok=True)
        motor = MotorLayer(activation='linear', name='motor', n=2, layer=4)
        path  = os.path.join(net_dir, name)
        try:
            save_network_file(path, [], [motor], [], set(), set(), {})
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            return
        self.brain.network_file = name
        self._load_data_brain_network(name)
        self._rebuild_brain_params()

    def _load_data_brain_network(self, net_name: str):
        project = getattr(self.brain, 'network_project', '')
        full_name = os.path.join(project, net_name) if project else net_name
        hidden, disabled, col_labels, conn_params, freshness_issues = \
            self.brain_mgr.load_network_into_circuit(self.brain, full_name)
        if hidden is None:
            return
        self._connection_params = conn_params
        self._hidden_cols       = hidden
        self._disabled_cols     = disabled
        self._col_labels        = col_labels
        if self._net_viz:
            self._net_viz._hidden_cols   = hidden
            self._net_viz._disabled_cols = disabled
            self._net_viz._col_labels    = col_labels
            self._net_viz._weight_params = conn_params
            self._net_viz.build()
        self._osc_ctrl._osc_items = {'mL', 'mR'}
        self._rebuild_channels()
        self._arena.setup_sensors(self.circuit.sensors, self._osc_ctrl.channel_colors)
        poses = _rb_world_poses(self.bot_pos, self.circuit.bodies, self.circuit.joints)
        self._arena.update_child_bodies(poses, self.circuit.bodies, self.sim_cfg)
        if freshness_issues and not getattr(self, '_syncing_group', False):
            self._prompt_network_update(full_name, freshness_issues)
        # When any agent in a group loads a network, sync all siblings so the
        # group stays structurally identical. Only the network-selector params
        # (network_file / network_project) are propagated; per-agent sliders
        # (tau, speed, …) are left untouched so agents can still be individually tuned.
        # Guard: skip if we're already inside a group sync (prevents re-entrant loops).
        if not getattr(self, '_syncing_group', False):
            self._sync_group_network()

    def _sync_group_network(self):
        """Push network_file + network_project from the current agent to every
        other agent in its group so all siblings share the same network structure."""
        sel = self._sim_ctrl._selected
        group_idx = self._group_of_agent(sel)
        group = self._agent_groups[group_idx]
        if not group['module'] or len(group['indices']) <= 1:
            return
        src_brain = self._sim_ctrl._agents[sel].brain
        if src_brain is None:
            return
        # Only the structural params that determine which network is loaded
        net_params = {k: getattr(src_brain, k)
                      for k in ('network_file', 'network_project')
                      if hasattr(src_brain, k)}
        if not net_params.get('network_file'):
            return
        self._syncing_group = True
        try:
            for agent_idx in group['indices']:
                if agent_idx == sel:
                    continue
                self._sim_ctrl.select_agent(agent_idx)
                self.load_brain(group['module'], external_params=net_params)
        finally:
            self._syncing_group = False
        # Restore selection and refresh UI for the originating agent
        self._sim_ctrl.select_agent(sel)
        self._arena.select_robot(sel)
        self._rebuild_brain_params()
        self._rebuild_channels()

    def _prompt_network_update(self, net_name: str, issues: list):
        """Show a dialog reporting stale params and offer to resave with current defaults."""
        lines = ['This network file has components with new parameters since it was last saved.',
                 'New parameters will use their default values until the file is updated.\n']
        for item in issues:
            params_str = ', '.join(item['missing'])
            lines.append(f"  • {item['name']} ({item['type']}): {params_str}")
        lines.append('\nWould you like to resave the network now with current defaults?')
        reply = QMessageBox.question(
            self, 'Network file outdated',
            '\n'.join(lines),
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Ignore,
            QMessageBox.StandardButton.Ignore,
        )
        if reply == QMessageBox.StandardButton.Save:
            from brain_serializer import save_network_file
            path = os.path.join('networks', net_name)
            try:
                save_network_file(
                    path,
                    self.circuit.sensors,
                    self.circuit.layers,
                    self.circuit.connections,
                    self._hidden_cols,
                    self._disabled_cols,
                    self._col_labels,
                    self.circuit.bodies,
                    self.circuit.joints,
                    self._connection_params,
                )
            except Exception as e:
                QMessageBox.critical(self, 'Save failed', str(e))

    def load_brain(self, name=None, external_params=None):
        if name is None or isinstance(name, bool):
            name = self._brain_combo.currentText()
        if not name:
            return

        self._brain_combo.blockSignals(True)
        self._brain_combo.setCurrentText(name)
        self._brain_combo.blockSignals(False)

        # Track which brain module the current group has loaded
        sel = self._sim_ctrl._selected
        group_idx = self._group_of_agent(sel)
        if group_idx < len(self._agent_groups):
            self._agent_groups[group_idx]['module'] = name

        brain, loaded_json = self.brain_mgr.load_brain_logic(name)
        if not brain:
            return
        self._sim_ctrl.brain = brain

        cls = brain.__class__
        if isinstance(brain, DataBrain):
            self.circuit.sensors     = []
            self.circuit.connections = []
            self.circuit.layers      = []
            self.circuit.joints      = []
            self.circuit.bodies      = (self.circuit.bodies[:1] if self.circuit.bodies
                                        else [RigidBody('root', 'root', self.sim_cfg.body_radius)])
            net = getattr(brain, 'network_file', '')
            if net:
                self._load_data_brain_network(net)
        else:
            self.circuit.joints = []
            self.circuit.bodies = (self.circuit.bodies[:1] if self.circuit.bodies
                                   else [RigidBody('root', 'root', self.sim_cfg.body_radius)])
            self.circuit.sensors     = list(getattr(cls, 'sensors',     []))
            self.circuit.layers      = list(getattr(cls, 'layers',      []))
            raw_conns = list(getattr(cls, 'connections', []))
            self.circuit.connections = [
                c if isinstance(c, Connection) else Connection(*c)
                for c in raw_conns
            ]
            for layer in self.circuit.layers:
                setattr(brain, layer.name, layer)
                layer.reset()

        if self._net_viz:
            self._net_viz.build()
        brain.setup()

        if external_params:
            for k, v in external_params.items():
                setattr(brain, k, v)
            if isinstance(brain, DataBrain) and not self.circuit.layers:
                net = getattr(brain, 'network_file', '')
                if net:
                    self._load_data_brain_network(net)

        self._sim_ctrl.trail_xy.clear()
        self._rebuild_brain_params()
        self._rebuild_channels()

        if "multipliers" in loaded_json:
            self._osc_ctrl.apply_multipliers_from_json(loaded_json["multipliers"])

        self.brain_mgr.rebuild_joint_motor_layers()
        if brain.__dict__.get('layers') is not None:
            brain.layers = self.circuit.layers
            for layer in self.circuit.layers:
                if getattr(layer, '_is_joint_motor', False):
                    setattr(brain, layer.name, layer)
        self.brain_mgr.resolve_joint_sensor_refs()
        if self._net_viz:
            self._net_viz.build()
        self._refresh_agent_list()

        self._setup_world()
        self._arena.setup_sensors(self.circuit.sensors, self._osc_ctrl.channel_colors)
        poses = _rb_world_poses(self.bot_pos, self.circuit.bodies, self.circuit.joints)
        self._arena.update_child_bodies(poses, self.circuit.bodies, self.sim_cfg)
        self._osc_ctrl.setup_osc()

    def _rebuild_channels(self):
        if self.brain is not None:
            self.brain.layers      = self.circuit.layers
            self.brain.sensors     = self.circuit.sensors
            self.brain.connections = self.circuit.connections
            self.brain_mgr.resolve_joint_sensor_refs()
        self._osc_ctrl.rebuild_channels(self.brain, self.circuit)
        self._osc_ctrl.setup_osc()
        self._rebuild_robot_rows()
        if hasattr(self, '_arena'):
            self._arena.setup_sensors(self.circuit.sensors, self._osc_ctrl.channel_colors)

    def _toggle_osc_layer(self, name):
        self._osc_ctrl.toggle_layer(name)
        self._rebuild_channels()
        if self._net_viz:
            self._net_viz._redraw_nodes()

    def _new_brain(self):
        from PySide6.QtWidgets import QInputDialog
        raw, ok = QInputDialog.getText(self, "New Brain", "Name (e.g. MyBrain):")
        if not ok or not raw:
            return
        name       = ''.join(c for c in raw if c.isalnum() or c == '_').strip('_')
        class_name = name if name.startswith('Brain') else f"Brain{name[0].upper()}{name[1:]}"
        path = self.brain_mgr.create_brain_file(class_name)
        if path is None:
            print(f"Already exists: brains/{class_name}.py")
            return
        self._refresh_brain_list()
        if class_name in self.brain_files:
            self._brain_combo.setCurrentText(class_name)
            self.load_brain(class_name)
        print(f"Created {path}")

    def _toggle_network_viz(self):
        if self._net_viz:
            self._net_viz.close()
            self._net_viz = None
        else:
            self._net_viz = NetworkVisualizerWindow(self)
            self._net_viz._weight_params = self._connection_params
            self._net_viz._hidden_cols   = self._hidden_cols
            self._net_viz._disabled_cols = self._disabled_cols
            self._net_viz._col_labels    = self._col_labels
            self._net_viz.show()

    # ── Brain params UI ───────────────────────────────────────────────────────

    def _rebuild_brain_params(self):
        while self._brain_params_layout.count():
            item = self._brain_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not self.brain:
            return
        is_data_brain = isinstance(self.brain, DataBrain)
        for k, p_obj in self.brain.get_param_metadata().items():
            cv = getattr(self.brain, k)
            if is_data_brain and k == 'network_project':
                def on_change_proj(v, key=k):
                    setattr(self.brain, key, v)
                    self.brain.network_file = ''
                    self._rebuild_brain_params()
                proj_combo = self._make_param_row(
                    self._brain_params_layout, k, p_obj, cv, on_change_proj, desc=p_obj.desc)
                btn_new_proj = QPushButton("+")
                btn_new_proj.setFixedWidth(24)
                btn_new_proj.setToolTip("Create new project directory")
                btn_new_proj.clicked.connect(self._new_network_project)
                proj_combo.parent().layout().addWidget(btn_new_proj)
            elif is_data_brain and k == 'network_file':
                def on_change(v, key=k):
                    setattr(self.brain, key, v)
                    if v:
                        self._load_data_brain_network(v)
                project = getattr(self.brain, 'network_project', '')
                net_dir = os.path.join('networks', project) if project else 'networks'
                file_choices = sorted(f for f in os.listdir(net_dir) if f.endswith('.json')) \
                               if os.path.isdir(net_dir) else []
                combo = self._make_param_row(
                    self._brain_params_layout, k, p_obj, cv, on_change,
                    desc=p_obj.desc, choices=file_choices)
                btn_new = QPushButton("+")
                btn_new.setFixedWidth(24); btn_new.setToolTip("Create new network file")
                btn_new.clicked.connect(self._new_network_from_sidebar)
                combo.parent().layout().addWidget(btn_new)
                btn_net_viz = QPushButton("⬡")
                btn_net_viz.setFixedWidth(28); btn_net_viz.setToolTip("Open network visualizer")
                btn_net_viz.clicked.connect(self._toggle_network_viz)
                combo.parent().layout().addWidget(btn_net_viz)
            else:
                def on_change(v, key=k):
                    setattr(self.brain, key, v)
                self._make_param_row(
                    self._brain_params_layout, k, p_obj, cv, on_change, desc=p_obj.desc)
        btn_reset = self._make_btn("↺ Reset Defaults", C['border'])
        btn_reset.clicked.connect(self._reset_brain_params)
        self._brain_params_layout.addWidget(btn_reset)
        from neurons import LearningLayerBase as _LLB
        if any(isinstance(l, _LLB) for l in getattr(self.circuit, 'layers', [])):
            btn_reset_w = self._make_btn("↺ Reset Weights", C['warning'])
            btn_reset_w.setToolTip("Re-initialise learned weights from init params")
            btn_reset_w.clicked.connect(self._reset_learning_weights)
            self._brain_params_layout.addWidget(btn_reset_w)

    def _reset_brain_params(self):
        if not self.brain:
            return
        for k, p_obj in self.brain.get_param_metadata().items():
            setattr(self.brain, k, p_obj.default)
        self._rebuild_brain_params()
        self.brain.setup()

    def _reset_learning_weights(self):
        from neurons import LearningLayerBase as _LLB
        ll_names = {l.name for l in getattr(self.circuit, 'layers', [])
                    if isinstance(l, _LLB)}
        if not ll_names:
            return
        for conn in self.circuit.connections:
            if conn.tgt in ll_names:
                init_W = getattr(conn, 'init_W', None)
                if init_W is not None:
                    conn.W = np.asarray(init_W, dtype=float).copy()
                else:
                    conn.W = np.zeros_like(np.asarray(conn.W, dtype=float))
        # Fully invalidate the cache so the runner re-reads conn.W on the next tick.
        if hasattr(self.brain, '_w_cache'):
            self.brain._w_cache = {}
        for layer in self.circuit.layers:
            if isinstance(layer, _LLB):
                layer.reset()

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
            self._net_viz.build()
        poses = _rb_world_poses(self.bot_pos, self.circuit.bodies, self.circuit.joints)
        self._arena.update_child_bodies(poses, self.circuit.bodies, self.sim_cfg)
        if self._net_viz:
            self._net_viz.raise_(); self._net_viz.activateWindow()

    # ── Session save / load ───────────────────────────────────────────────────

    def _save_session(self):
        if not self.brain:
            print("SAVE FAILED: No brain loaded.")
            return
        raw_name   = self._session_name.text().strip()
        clean_name = "".join(x for x in raw_name if x.isalnum() or x in "._- ") or "autosave"
        path       = f"configs/{clean_name}.json"
        try:
            # Build serialisable group list with per-group brain params
            groups_data = []
            for g in self._agent_groups:
                b_params = {}
                if g['indices']:
                    agent = self._sim_ctrl._agents[g['indices'][0]]
                    if agent.brain:
                        b_params = {k: getattr(agent.brain, k)
                                    for k in agent.brain.get_param_metadata()}
                groups_data.append({**g, 'brain_params': b_params})
            save_session(
                path,
                module_name  = self._brain_combo.currentText(),
                brain        = self.brain,
                sim_cfg      = self.sim_cfg,
                world        = self.world,
                speed_mult   = self._sim_ctrl.speed_mult,
                trail_length = self._trail_len_spin.value(),
                arena_round  = self._arena_round_rb.isChecked(),
                multipliers  = self._osc_ctrl.get_multipliers(),
                groups       = groups_data,
            )
            print(f"SUCCESS: Session saved to {path}")
            self._refresh_session_list()
        except Exception as e:
            print(f"SAVE ERROR: {e}")

    def _load_session(self):
        selected = self._session_combo.currentText()
        if not selected:
            return
        path = f"configs/{selected}"
        if not os.path.exists(path):
            return
        try:
            d = load_session(path)

            p_meta = self.sim_cfg.get_param_metadata()
            for k, v in d.get("sim_params", {}).items():
                if hasattr(self.sim_cfg, k):
                    setattr(self.sim_cfg, k, v)
                    if k in self._phys_widgets and k in p_meta:
                        p = p_meta[k]
                        w = self._phys_widgets[k]
                        slider = w[0] if isinstance(w, tuple) else w
                        is_int = isinstance(v, int) and p.step >= 1
                        slider.setValue(int(v) if is_int else int((v - p.min) / p.step))
                        if isinstance(w, tuple):
                            w[1].setText(f"{v:.4g}")

            if "arena_round" in d:
                v = bool(d["arena_round"])
                self.world.arena_round = v
                self._arena_round_rb.setChecked(v)
                self._arena_square_rb.setChecked(not v)

            self._stim_cb.setChecked(bool(self.sim_cfg.toggle_stim))
            self.world.patches = d.get("patches", [])
            self.world.objects = d.get("objects", [])
            self.world.walls   = d.get("walls", [])
            sky = d.get("sky", {"enabled": False, "angle": 0.0})
            self.world.sky = sky
            self._sky_cb.setChecked(bool(sky.get("enabled", False)))

            saved_agents = d.get("agents")
            if saved_agents:
                self._load_session_agents(saved_agents)
            else:
                # Backward compat: single-agent session
                mod_name = d.get("module_name")
                if mod_name in self.brain_files:
                    self._brain_combo.blockSignals(True)
                    self._brain_combo.setCurrentText(mod_name)
                    self._brain_combo.blockSignals(False)
                    self.load_brain(mod_name, external_params=d.get("brain_params"))

            self._osc_ctrl.apply_multipliers_from_json(d.get("plot_multipliers", {}))

            if "speed_mult" in d:
                self._sim_ctrl.speed_mult = int(d["speed_mult"])
                self._speed_spin.setValue(self._sim_ctrl.speed_mult)

            if "trail_length" in d:
                n = max(10, int(d["trail_length"]))
                self._trail_len_spin.setValue(n)
                for agent in self._sim_ctrl._agents:
                    old = list(agent.trail_xy)[-n:]
                    agent.trail_xy = deque(old, maxlen=n)

            poses = _rb_world_poses(self.bot_pos, self.circuit.bodies, self.circuit.joints)
            self._arena.update_child_bodies(poses, self.circuit.bodies, self.sim_cfg)
            if self._net_viz:
                self._net_viz.build()
            self._session_name.setText(selected.replace(".json", ""))
            self._reset()
            print(f"Session Restored: {selected}")
        except Exception as e:
            print(f"Load Error: {e}")

    def _load_session_agents(self, saved_agents):
        """Reconstruct multiagent groups from a saved 'agents' array."""
        from arena_widget import _AGENT_COLORS
        # Remove all agents beyond agent 0
        while len(self._sim_ctrl._agents) > 1:
            last = len(self._sim_ctrl._agents) - 1
            self._arena.remove_robot_item(last)
            self._sim_ctrl.remove_agent(last)
        # Reset groups to a single placeholder pointing at agent 0
        self._agent_groups = [
            {'module': None, 'n': 1, 'color': '#4a7fcb', 'name': 'Group 1', 'indices': [0]}
        ]
        for i, sg in enumerate(saved_agents):
            mod   = sg.get('module_name') or ''
            color = sg.get('color', _AGENT_COLORS[i % len(_AGENT_COLORS)])
            name  = sg.get('name', f'Group {i + 1}')
            n     = max(1, int(sg.get('n', 1)))
            b_params = sg.get('brain_params') or {}
            if i == 0:
                # Update group 0 metadata and load brain into agent 0
                self._agent_groups[0].update({'module': mod or None, 'color': color, 'name': name})
                self._sim_ctrl._agents[0].color = color
                self._arena.update_robot_color(0, color)
                if mod and mod in self.brain_files:
                    self._sim_ctrl.select_agent(0)
                    self._arena.select_robot(0)
                    self.load_brain(mod, external_params=b_params)
                for _ in range(n - 1):
                    self._add_agent_to_group(0)
            else:
                # Add a new group
                self._add_agent(color=color, name=name)
                g_idx = len(self._agent_groups) - 1
                self._agent_groups[g_idx]['module'] = mod or None
                first_idx = self._agent_groups[g_idx]['indices'][0]
                if mod and mod in self.brain_files:
                    prev_sel = self._sim_ctrl._selected
                    self._sim_ctrl.select_agent(first_idx)
                    self._arena.select_robot(first_idx)
                    self.load_brain(mod, external_params=b_params)
                    self._sim_ctrl.select_agent(prev_sel)
                    self._arena.select_robot(prev_sel)
                for _ in range(n - 1):
                    self._add_agent_to_group(g_idx)
        self._sim_ctrl.select_agent(0)
        self._arena.select_robot(0)
        self._refresh_agent_list()

    def _refresh_session_list(self):
        files  = sorted(glob.glob("configs/*.json"), key=os.path.getmtime, reverse=True)
        recent = [os.path.basename(f) for f in files[:10]]
        self._session_combo.blockSignals(True)
        self._session_combo.clear()
        self._session_combo.addItems(recent)
        self._session_combo.blockSignals(False)

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, ev):
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
    _splash.finish(win)
    try:
        sys.exit(_app.exec())
    except KeyboardInterrupt:
        pass
