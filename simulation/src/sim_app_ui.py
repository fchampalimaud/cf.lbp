"""
sim_app_ui.py — widget-tree construction for SimulatorApp.

_UiBuilderMixin holds every method that only builds widgets (creates them,
lays them out, wires their .connect(...) signals to handlers that live on
SimulatorApp itself). It owns no state and makes no simulation decisions —
LBPSimulator.py keeps all of that (the __init__ sequencing, event handlers,
agent/group/world management). Mixed into SimulatorApp alongside _BrainMixin
and _SessionMixin, which are organized by feature rather than by layer and are
left where they are (see TODO.md item 6 discussion / the item-5 plan).
"""

from PySide6.QtWidgets import (
    QWidget, QDockWidget, QScrollArea, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox,
    QStatusBar, QLineEdit, QRadioButton, QFormLayout, QAbstractItemView,
    QButtonGroup, QListWidget,
)
from PySide6.QtCore import Qt, QTimer

from sim_constants import C, GRADIENT_COLORS, OBJECT_COLORS
from arena_widget import ArenaWidget
from sim_widgets import OscilloscopeWidget, ControlPanel
from world_serializer import discover_worlds
from texture_manager import discover_textures

try:
    from sim_engine_mujoco import MuJoCoEngine
    _MUJOCO_AVAILABLE = True
except Exception:
    _MUJOCO_AVAILABLE = False


class _UiBuilderMixin:
    def _build_ui(self):
        self._arena = ArenaWidget()
        self._arena.setFocusPolicy(Qt.NoFocus)   # keep key events on SimulatorApp
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
        tab_brain, tab_world, tab_session, tab_physics, tab_robot, tab_network = \
            self._panel.add_tab_widget(["Brain", "World", "Session", "Physics", "Robot", "Network"])
        self._build_brain_group(tab_brain)
        self._build_world_group(tab_world)
        self._build_session_group(tab_session)
        self._build_task_group(tab_session)
        self._build_logger_group(tab_session)
        self._build_physics_group(tab_physics)
        self._build_robot_group(tab_robot)
        self._build_network_group(tab_network)
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

    # _build_brain_group → sim_app_brain._BrainMixin
    # _build_session_group / _build_task_group / _build_logger_group → sim_app_session._SessionMixin

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

    def _build_network_group(self, panel_vl=None):
        """Network host/client mode controls (inside the Physics tab)."""
        gb, vl = self._panel.add_group("Network", panel_vl)

        # ── Mode row ──────────────────────────────────────────────────────────
        mode_row = QWidget()
        mode_hl  = QHBoxLayout(mode_row)
        mode_hl.setContentsMargins(0, 0, 0, 0)
        self._net_off_rb  = QRadioButton("Off")
        self._net_host_rb = QRadioButton("Host")
        self._net_cli_rb  = QRadioButton("Client")
        self._net_off_rb.setChecked(True)
        self._net_mode_group = QButtonGroup(self)
        for rb in (self._net_off_rb, self._net_host_rb, self._net_cli_rb):
            self._net_mode_group.addButton(rb)
            mode_hl.addWidget(rb)
        mode_hl.addStretch()
        vl.addWidget(mode_row)

        # ── Host sub-panel ────────────────────────────────────────────────────
        self._net_host_panel = QWidget()
        hvl = QVBoxLayout(self._net_host_panel)
        hvl.setContentsMargins(4, 2, 4, 2)
        hvl.setSpacing(4)

        port_row = QWidget()
        port_hl  = QHBoxLayout(port_row)
        port_hl.setContentsMargins(0, 0, 0, 0)
        port_hl.addWidget(QLabel("Port:"))
        self._net_host_port_spin = QSpinBox()
        self._net_host_port_spin.setRange(1024, 65535)
        self._net_host_port_spin.setValue(9001)
        port_hl.addWidget(self._net_host_port_spin)
        self._net_host_ip_combo = QComboBox()
        self._net_host_ip_combo.setStyleSheet(
            f"QComboBox {{ color:{C['dark']}; font-family:monospace; }}")
        port_hl.addWidget(self._net_host_ip_combo)
        port_hl.addStretch()
        hvl.addWidget(port_row)

        fr_row = QWidget()
        fr_hl  = QHBoxLayout(fr_row)
        fr_hl.setContentsMargins(0, 0, 0, 0)
        fr_hl.addWidget(QLabel("Frame rate:"))
        self._net_host_fr_spin = QSpinBox()
        self._net_host_fr_spin.setRange(1, 120)
        self._net_host_fr_spin.setValue(50)
        self._net_host_fr_spin.setSuffix(" Hz")
        self._net_host_fr_spin.valueChanged.connect(
            lambda v: setattr(self._sim_ctrl, '_net_frame_rate', v)
        )
        fr_hl.addWidget(self._net_host_fr_spin)
        fr_hl.addWidget(QLabel("Timeout:"))
        self._net_host_timeout_spin = QDoubleSpinBox()
        self._net_host_timeout_spin.setRange(0.5, 30.0)
        self._net_host_timeout_spin.setSingleStep(0.5)
        self._net_host_timeout_spin.setValue(2.0)
        self._net_host_timeout_spin.setSuffix(" s")
        self._net_host_timeout_spin.setToolTip(
            "Seconds a client can go quiet before its agent is dropped.\n"
            "Clients heartbeat every 2s — very low values may prune a client\n"
            "on a single missed/delayed heartbeat.")
        self._net_host_timeout_spin.valueChanged.connect(
            lambda v: setattr(self._sim_ctrl, '_net_disconnect_timeout', v)
        )
        fr_hl.addWidget(self._net_host_timeout_spin)
        fr_hl.addStretch()
        hvl.addWidget(fr_row)

        hvl.addWidget(QLabel("Connected clients:"))
        self._net_client_list = QListWidget()
        self._net_client_list.setFixedHeight(120)
        self._net_client_list.setStyleSheet(
            f"QListWidget {{ background:{C['bg']}; border:1px solid {C['border']}; "
            f"border-radius:3px; font-family:monospace; font-size:11px; }}"
        )
        self._net_client_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        hvl.addWidget(self._net_client_list)

        self._net_host_panel.setVisible(False)
        vl.addWidget(self._net_host_panel)

        # ── Client sub-panel ──────────────────────────────────────────────────
        self._net_cli_panel = QWidget()
        cf = QFormLayout(self._net_cli_panel)
        cf.setContentsMargins(4, 2, 4, 2)
        cf.setSpacing(4)
        self._net_cli_name_edit  = QLineEdit("Student")
        self._net_cli_host_edit  = QLineEdit("192.168.1.10")
        self._net_cli_port_spin  = QSpinBox()
        self._net_cli_port_spin.setRange(1024, 65535)
        self._net_cli_port_spin.setValue(9001)
        self._net_cli_lport_spin = QSpinBox()
        self._net_cli_lport_spin.setRange(1024, 65535)
        self._net_cli_lport_spin.setValue(9002)
        self._net_cli_conn_btn   = self._make_btn("Connect", C['primary'])
        self._net_cli_status_lbl = QLabel("disconnected")
        self._net_cli_status_lbl.setStyleSheet(f"color:{C['muted']};")
        cf.addRow("Name:", self._net_cli_name_edit)
        cf.addRow("Host:", self._net_cli_host_edit)
        cf.addRow("Port:", self._net_cli_port_spin)
        cf.addRow("Local port:", self._net_cli_lport_spin)
        cf.addRow("", self._net_cli_conn_btn)
        cf.addRow("Status:", self._net_cli_status_lbl)
        self._net_cli_panel.setVisible(False)
        vl.addWidget(self._net_cli_panel)

        # ── Wire signals ──────────────────────────────────────────────────────
        self._net_off_rb.toggled.connect(self._on_net_mode_off)
        self._net_host_rb.toggled.connect(self._on_net_mode_host)
        self._net_cli_rb.toggled.connect(self._on_net_mode_client)
        self._net_cli_conn_btn.clicked.connect(self._on_net_connect_clicked)

        # ── Status refresh timer (1 Hz) ───────────────────────────────────────
        self._net_status_timer = QTimer(self)
        self._net_status_timer.setInterval(1000)
        self._net_status_timer.timeout.connect(self._update_net_status)
        self._net_status_timer.start()

        # Keep everything top-aligned regardless of which sub-panel is visible
        if panel_vl is not None:
            panel_vl.addStretch()

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

    def _build_world_group(self, panel_vl=None):
        gb, vl = self._panel.add_group("World", panel_vl)

        world_row = QWidget(); wrl = QHBoxLayout(world_row); wrl.setContentsMargins(0, 0, 0, 0)
        self._world_combo = QComboBox()
        self._world_combo.addItems(discover_worlds())
        self._world_combo.currentTextChanged.connect(self._load_world)
        btn_save_world = self._make_btn("Save", C['primary'])
        btn_save_world.setFixedWidth(44)
        btn_save_world.setToolTip("Save current world (patches/objects/walls/sky/floor)")
        btn_save_world.clicked.connect(self._save_world)
        wrl.addWidget(QLabel("World:"))
        wrl.addWidget(self._world_combo)
        wrl.addWidget(btn_save_world)
        vl.addWidget(world_row)

        floor_row = QWidget(); frl = QHBoxLayout(floor_row); frl.setContentsMargins(0, 0, 0, 0)
        frl.addWidget(QLabel("Floor:"))
        self._floor_texture_combo = QComboBox()
        self._floor_texture_combo.addItems(["(default)"] + discover_textures())
        self._floor_texture_combo.currentTextChanged.connect(self._on_floor_texture_change)
        frl.addWidget(self._floor_texture_combo)
        vl.addWidget(floor_row)

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
        self._obj_texture_combo = QComboBox()
        self._obj_texture_combo.addItems(["(none)"] + discover_textures())
        self._obj_texture_combo.setToolTip("Texture applied to new objects/walls (MuJoCo view only)")
        self._obj_texture_combo.setFixedWidth(90)
        self._obj_texture_combo.currentTextChanged.connect(self._set_object_texture)
        ol.addWidget(self._obj_texture_combo)
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
