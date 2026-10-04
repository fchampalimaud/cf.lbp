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
from PySide6.QtGui import QFontMetrics

from sim_constants import C, GRADIENT_COLORS, OBJECT_COLORS
from arena_widget import ArenaWidget
from sim_widgets import OscilloscopeWidget, ControlPanel
from world_serializer import discover_worlds
from texture_manager import discover_textures
import data_paths


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
        self._build_video_group(tab_session)
        self._build_physics_group(tab_physics)
        self._build_robot_group(tab_robot)
        self._build_network_group(tab_network)
        self._panel.add_stretch()
        self._panel.compact()

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("■  STOPPED")
        self._status_label.setStyleSheet(f"color:{C['muted']};font-weight:bold;padding:0 8px;")
        self._timing_label = QLabel("")
        self._timing_label.setStyleSheet(f"color:{C['muted']};padding:0 8px;")
        self._record_label = QLabel("")
        self._record_label.setStyleSheet(f"color:{C['danger']};font-weight:bold;padding:0 8px;")
        self._status_bar.addWidget(self._status_label)
        self._status_bar.addPermanentWidget(self._record_label)
        self._status_bar.addPermanentWidget(self._timing_label)

        self._osc_hidden_height = 0

        def _set_dock_sizes():
            self.resizeDocks([self._left_dock], [430], Qt.Horizontal)
            self.resizeDocks([self._osc_dock],  [220], Qt.Vertical)
            if not self._osc_btn.isChecked():
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
                padding:4px 7px; font-weight:bold; color:{self._text_color(bg)};
            }}
            QPushButton:hover {{ background:{self._darken(bg)}; }}
            QPushButton:disabled {{ background:{C['border']}; color:{C['muted']}; }}
            QPushButton:checked {{ background:{self._darken(bg, 0.80)}; }}
        """)
        if checkable:
            btn.setCheckable(True)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # Space / Enter must not press it
        return btn

    def _show_shortcuts(self):
        """Open (or raise) the keyboard-shortcuts panel."""
        from shortcuts import ShortcutsDialog
        if getattr(self, '_shortcuts_dlg', None) is None:
            self._shortcuts_dlg = ShortcutsDialog(self)
        self._shortcuts_dlg.show()
        self._shortcuts_dlg.raise_()
        self._shortcuts_dlg.activateWindow()

    # ── Folder pickers: first the root, then its folders ──────────────────────

    def _make_root_combo(self):
        """'My files' / '🔒 Simulator' — which root a folder picker lists."""
        combo = QComboBox()
        combo.addItem("My files", "user")
        combo.addItem("🔒 Simulator", "builtin")
        combo.setToolTip("Your own files, or the read-only ones that ship with the simulator")
        return combo

    @staticmethod
    def _first_folder_key(kind, root):
        """Picker key of the first folder of *root* ('user' → the top level)."""
        if root == 'builtin':
            dirs = data_paths.builtin_subdirs(kind)
            return data_paths.BUILTIN + dirs[0] if dirs else data_paths.BUILTIN
        return ''

    def _fill_folder_picker(self, root_combo, dir_combo, kind, key):
        """Set root_combo from *key* and list that root's folders in dir_combo.
        Keys as in data_paths.folder_for: 'builtin:Tutorials', 'Mine', '' (the
        user's top level). Returns True when the built-in root is shown."""
        builtin = (key or '').startswith(data_paths.BUILTIN)
        root_combo.blockSignals(True)
        root_combo.setCurrentIndex(root_combo.findData('builtin' if builtin else 'user'))
        root_combo.blockSignals(False)
        if builtin:
            entries = [(d, data_paths.BUILTIN + d) for d in data_paths.builtin_subdirs(kind)]
        else:
            entries = [("(top level)", "")] + [(d, d) for d in data_paths.user_subdirs(kind)]
        dir_combo.blockSignals(True)
        dir_combo.clear()
        for label, data in entries:
            dir_combo.addItem(label, data)
        idx = dir_combo.findData(key)
        dir_combo.setCurrentIndex(idx if idx >= 0 else 0)
        dir_combo.blockSignals(False)
        return builtin

    def _make_toggle(self, text, tooltip=None):
        """A two-state (checkable) button in the one style every main-window
        toggle shares: white when off, primary light blue + bold when on.
        (The gradient / object / sky palette buttons keep their own colours.)"""
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # Space / Enter must not toggle it
        on = C['primary']
        btn.setStyleSheet(
            f"QPushButton {{ background:{C['surface']}; color:{C['dark']};"
            f" border:2px solid {C['border']}; border-radius:3px; padding:3px 7px; }}"
            f"QPushButton:hover {{ background:{C['bg']}; }}"
            f"QPushButton:checked {{ background:{on}; border:2px solid {self._darken(on, 0.75)};"
            f" font-weight:bold; }}"
            f"QPushButton:disabled {{ background:{C['bg']}; color:{C['muted']};"
            f" border:2px solid {C['border']}; }}")
        if tooltip:
            btn.setToolTip(tooltip)
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
            self._panel.compact_widget(combo)
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
        gb, group_vl = self._panel.add_group("Simulation")

        # Two columns: options (speed, robot and display toggles) on the left,
        # Run / Step / Reset stacked in a narrow column on the right.
        cols = QWidget(); cols_hl = QHBoxLayout(cols); cols_hl.setContentsMargins(0, 0, 0, 0)
        left = QWidget(); vl = QVBoxLayout(left)
        vl.setContentsMargins(0, 0, 0, 0); vl.setSpacing(4)
        right = QWidget(); run_vl = QVBoxLayout(right)
        run_vl.setContentsMargins(0, 0, 0, 0); run_vl.setSpacing(4)
        cols_hl.addWidget(left, 1)
        cols_hl.addWidget(right, 0)
        group_vl.addWidget(cols)

        self._btn_run_stop = self._make_btn("▶ Run",   C['success'])
        self._btn_step     = self._make_btn("⏭ Step", C['surface'])
        self._btn_reset    = self._make_btn("↺ Reset", C['surface'])
        self._btn_run_stop.setToolTip(
            "Run / Pause the simulation (Ctrl+Space).\n"
            "Pause freezes everything where it is; Run continues from there.")
        self._btn_step.setToolTip(
            "Advance exactly one physics tick (Ctrl+→) — frame-by-frame inspection while paused.")
        self._btn_reset.setToolTip(
            "Stop and put the robots back at their start position and heading (Ctrl+R).\n"
            "Brain state is cleared; the world (patches, objects) is kept.")
        for b in [self._btn_run_stop, self._btn_step, self._btn_reset]:
            b.setMinimumSize(88, 30)
            run_vl.addWidget(b)
        run_vl.addStretch()

        # "?" right after the group title: opens the keyboard & mouse panel (also F1).
        btn_help = QPushButton("?", gb)
        btn_help.setFixedSize(16, 16)
        btn_help.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_help.setToolTip("Keyboard and mouse (F1)")
        btn_help.setStyleSheet(
            f"QPushButton {{ background:{C['surface']}; color:{C['dark']}; font-weight:bold;"
            f" border:1px solid {C['border']}; border-radius:8px; padding:0; font-size:8pt; }}"
            f"QPushButton:hover {{ background:{C['primary']}; }}")
        btn_help.clicked.connect(self._show_shortcuts)
        title_w = QFontMetrics(gb.font()).horizontalAdvance(gb.title())
        btn_help.move(8 + title_w + 10, 0)   # QGroupBox title starts 8 px in (sim_widgets)
        btn_help.raise_()
        self._btn_run_stop.clicked.connect(self._on_run_stop)
        self._btn_step.clicked.connect(lambda: self._sim_ctrl.step())
        self._btn_reset.clicked.connect(self._reset)

        def row_label(text):
            # Left column of the options: what each row is about, aligned.
            lbl = QLabel(text)
            lbl.setFixedWidth(78)
            lbl.setStyleSheet(f"color:{C['dark']};font-weight:bold;")   # like "Gradients:"
            return lbl

        row2 = QWidget(); hl2 = QHBoxLayout(row2); hl2.setContentsMargins(0, 0, 0, 0)
        hl2.addWidget(row_label("Time"))
        hl2.addWidget(QLabel("Speed ×"))
        self._speed_spin = QSpinBox()
        self._speed_spin.setMinimum(1); self._speed_spin.setMaximum(500)
        self._speed_spin.setValue(200)
        self._speed_spin.valueChanged.connect(lambda v: self._sim_ctrl.set_speed_mult(v))
        self._speed_tip = (   # also restored when Real time is switched off
            "Speed ×: physics ticks run per display frame.\n"
            "Higher = faster than real time. The physics step (dt) is unchanged, so results\n"
            "are the same — the screen just updates less often per simulated second.")
        self._speed_spin.setToolTip(self._speed_tip)
        hl2.addWidget(self._speed_spin)
        self._rt_cb = QCheckBox("Real time")
        self._rt_cb.setToolTip(
            "Real time: run only as many ticks per frame as the wall clock demands,\n"
            "so one simulated second takes one real second. Overrides Speed × while on.")
        self._rt_cb.toggled.connect(self._on_rt_mode_toggled)
        hl2.addWidget(self._rt_cb)
        hl2.addStretch()
        vl.addWidget(row2)

        # Robot row: what drives / holds the robot.
        row3 = QWidget(); rl3 = QHBoxLayout(row3); rl3.setContentsMargins(0, 0, 0, 0)
        rl3.addWidget(row_label("Options"))
        self._fixate_btn = self._make_toggle(
            "📌 Fixate",
            "Fixate: hold the robot in place.\n"
            "Sensors, brain and motor commands keep running — move patches around it\n"
            "to probe how the sensors and the network respond.")
        self._fixate_btn.setChecked(bool(self.sim_cfg.fixate_robot))
        self._fixate_btn.toggled.connect(
            lambda on: setattr(self.sim_cfg, 'fixate_robot', 1.0 if on else 0.0))
        rl3.addWidget(self._fixate_btn)
        manual_btn = self._make_toggle(
            "⌨ Manual",
            "Manual: drive the robot with the keyboard — W / S forward / back, A / D turn, Space stop.\n"
            "The brain keeps running (watch it in the network window and oscilloscope);\n"
            "while on, those keys don't pick world tools.")
        manual_btn.clicked.connect(self._toggle_manual_mode)
        rl3.addWidget(manual_btn)
        self._manual_btn = manual_btn
        self._robot_btn = self._make_toggle(
            "🤖 Robot",
            "Robot: drive the real robot instead of the simulated body.\n"
            "Sensor readings come from the robot and motor commands go to it over the network (OSC/UDP).\n"
            "Each sensor's robot_address field specifies its host:port connection.\n"
            "Motor commands are sent as OSC to the motor layer's robot_address.")
        self._robot_btn.toggled.connect(self._on_robot_mode_toggle)
        rl3.addWidget(self._robot_btn)
        rl3.addStretch()
        vl.addWidget(row3)

        # Display row
        row4 = QWidget(); rl4 = QHBoxLayout(row4); rl4.setContentsMargins(0, 0, 0, 0)
        rl4.addWidget(row_label("Visualization"))
        self._hide_stim_btn = self._make_toggle(
            "Hide stim",
            "Hide stim: switch the gradient patches off.\n"
            "They disappear from the arena and gradient sensors read zero; the patches are\n"
            "kept and come back when you switch this off.")
        self._hide_stim_btn.setChecked(not self.sim_cfg.toggle_stim)
        self._hide_stim_btn.toggled.connect(lambda hide: self._on_toggle_stim(not hide))
        rl4.addWidget(self._hide_stim_btn)
        self._osc_btn = self._make_toggle(
            "Osc", "Osc: show / hide the oscilloscope dock — live traces of the brain's plots()\n"
                   "and of any node added from the network window (right-click → Add to oscilloscope).")
        self._osc_btn.toggled.connect(self._on_toggle_osc)
        rl4.addWidget(self._osc_btn)
        rl4.addStretch()
        vl.addWidget(row4)

        self._manual_hint = QLabel("W/S=fwd/back  A/D=turn  Space=stop")
        self._manual_hint.setStyleSheet(f"color:{C['muted']};font-size:8pt;")
        self._manual_hint.setVisible(False)
        vl.addWidget(self._manual_hint)

    # _build_brain_group → sim_app_brain._BrainMixin
    # _build_session_group / _build_task_group / _build_logger_group / _build_video_group → sim_app_session._SessionMixin

    def _build_physics_group(self, panel_vl=None):
        gb, vl = self._panel.add_group("Physics", panel_vl)
        self._phys_widgets = {}
        phys_params = ['dt', 'motor_gain', 'body_radius', 'arena_scale',
                       'sense_radius', 'init_x', 'init_y']
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
            lambda v: setattr(self._sim_ctrl.network, 'frame_rate', v)
        )
        fr_hl.addWidget(self._net_host_fr_spin)
        fr_hl.addStretch()
        hvl.addWidget(fr_row)
        fr_row = QWidget(); fr_hl = QHBoxLayout(fr_row); fr_hl.setContentsMargins(0, 0, 0, 0)
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
            lambda v: setattr(self._sim_ctrl.network, 'disconnect_timeout', v)
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

    # _build_robot_group / _add_robot_row → sim_app_robot._RobotMixin

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
        tl.addStretch()
        vl.addWidget(trail_row)

        ml = tl   # Show 3D / Top view share the trail row
        self._mujoco_viewer_btn = QPushButton("Show 3D")
        self._mujoco_viewer_btn.setFixedHeight(22)
        self._mujoco_viewer_btn.setEnabled(False)
        self._mujoco_viewer_btn.setToolTip("Open the MuJoCo 3D viewer window")
        self._mujoco_viewer_btn.clicked.connect(lambda: self._sim_ctrl.mujoco.show_viewer())
        ml.addWidget(self._mujoco_viewer_btn)
        self._view_3d_btn = self._make_toggle(
            "Top view",
            "Show MuJoCo overhead (top-down) render in the arena.\n"
            "Object editing still works — switch back to 2D to see gradients.")
        self._view_3d_btn.setEnabled(False)
        self._view_3d_btn.clicked.connect(self._on_view_3d_toggle)
        ml.addWidget(self._view_3d_btn)

        arena_row = QWidget(); al = QHBoxLayout(arena_row); al.setContentsMargins(0, 0, 0, 0)
        al.addWidget(QLabel("Arena:"))
        self._arena_square_rb = QRadioButton("Square")
        self._arena_round_rb  = QRadioButton("Round")
        self._arena_square_rb.setChecked(True)
        self._arena_square_rb.toggled.connect(self._on_arena_type_change)
        al.addWidget(self._arena_square_rb); al.addWidget(self._arena_round_rb)
        al.addStretch()
        self._move_btn = self._make_toggle(
            "↖ Move", "Drag gradients, objects, or a robot to a new position")
        self._move_btn.clicked.connect(self._set_move_mode)
        al.addWidget(self._move_btn)
        btn_clear = self._make_btn("✕ Clear World", C['muted'])
        btn_clear.setToolTip("Remove all gradient patches and objects (walls stay)")
        btn_clear.clicked.connect(self._clear_world)
        al.addWidget(btn_clear)
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
        gl.addStretch()
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
        ol.addStretch()
        vl.addWidget(obj_row)

        tex_row = QWidget(); txl = QHBoxLayout(tex_row); txl.setContentsMargins(0, 0, 0, 0)
        txl.addWidget(QLabel("Texture:"))
        self._obj_texture_combo = QComboBox()
        self._obj_texture_combo.addItems(["(none)"] + discover_textures())
        self._obj_texture_combo.setToolTip("Texture applied to new objects/walls (MuJoCo view only)")
        self._obj_texture_combo.currentTextChanged.connect(self._set_object_texture)
        txl.addWidget(self._obj_texture_combo, 1)
        vl.addWidget(tex_row)

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
        # Real default draw_mode ('move') lives on WorldEditor, which doesn't exist
        # yet at this point in startup — just sync the button visuals here so they
        # don't show a stale "Object" selection once the editor is constructed.
        self._clear_mode_buttons()
        self._move_btn.setChecked(True)
