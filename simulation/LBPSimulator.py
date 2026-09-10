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
sys.path.insert(0, os.path.join(_HERE, "brains"))

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

try:
    from sim_engine_mujoco import MuJoCoEngine
    _MUJOCO_AVAILABLE = True
except Exception:
    _MUJOCO_AVAILABLE = False

from rigid_body import RigidBody, Joint, world_poses as _rb_world_poses
from sim_constants import C, _CHAN_PALETTE, _TRAIL_COLOR, GRADIENT_COLORS, OBJECT_COLORS
from circuit_model import CircuitModel, Connection
from logger import SimLogger
from sim_app_brain import _BrainMixin
from sim_app_session import _SessionMixin
from sim_app_ui import _UiBuilderMixin

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
from world_editor import WorldEditor
from osc_controller import OscChannelManager
from sim_controller import SimController
from app_version import get_app_version
if _splash: _splash.set_progress(80)


# ============================================================
# MAIN WINDOW
# ============================================================
class SimulatorApp(_UiBuilderMixin, _BrainMixin, _SessionMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"LBP Simulator v{get_app_version()}")
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
        self._video_recorders = []

        # ── Brain manager (per-agent; circuit/brain_mgr are proxy properties) ─
        _brain_mgr0 = BrainManager(_circuit0, self.sim_cfg)
        self.brain_files = _brain_mgr0.discover_brains()
        # Agent-table groups now live on SimController (self._sim_ctrl.groups_ordered()
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
        self._sim_ctrl.sig_client_ready.connect(self._update_net_status)
        self._sim_ctrl.sig_client_disconnect.connect(self._update_net_status)
        self._sim_ctrl.sig_client_registered.connect(
            self._on_client_registered, type=Qt.ConnectionType.QueuedConnection)
        self._sim_ctrl.sig_agent_removed.connect(
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
        self._refresh_session_dirs()
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
        return self._sim_ctrl.circuit

    @property
    def brain_mgr(self):
        return self._sim_ctrl.brain_mgr

    # ── UI Construction ───────────────────────────────────────────────────────
    # _build_ui / _make_btn / _darken / _text_color / _make_param_row /
    # _build_sim_group / _build_physics_group / _build_network_group /
    # _build_robot_group / _add_robot_row / _build_world_group → sim_app_ui._UiBuilderMixin
    # _build_brain_group → sim_app_brain._BrainMixin
    # _build_session_group / _build_task_group / _build_logger_group → sim_app_session._SessionMixin

    def _on_net_mode_off(self, checked):
        if not checked:
            return
        self._net_host_panel.setVisible(False)
        self._net_cli_panel.setVisible(False)
        self._sim_ctrl.disable_network_host()
        self._sim_ctrl.disconnect_from_host()

    def _on_net_mode_host(self, checked):
        if not checked:
            return
        self._net_cli_panel.setVisible(False)
        self._net_host_panel.setVisible(True)
        # Clear all existing agents from the UI — host starts with no local brain.
        # enable_network_host() below clears SimController's own agents/groups;
        # refresh the table only after that so it reflects the cleared state.
        for agent_idx in range(len(self._sim_ctrl._agents) - 1, -1, -1):
            self._arena.remove_robot_item(agent_idx)
        port = self._net_host_port_spin.value()
        try:
            self._sim_ctrl.enable_network_host(port)
        except OSError as exc:
            from PySide6.QtWidgets import QMessageBox
            self._net_off_rb.setChecked(True)
            QMessageBox.warning(self, "Network Error",
                                f"Could not open host port {port}:\n{exc}")
            self._refresh_agent_list()
            return
        self._refresh_agent_list()
        import socket as _socket
        try:
            _s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            _s.connect(('8.8.8.8', 80))
            preferred_ip = _s.getsockname()[0]
            _s.close()
        except OSError:
            preferred_ip = '127.0.0.1'
        try:
            infos = _socket.getaddrinfo(_socket.gethostname(), None, _socket.AF_INET)
            all_ips = sorted({info[4][0] for info in infos
                              if not info[4][0].startswith('127.')})
        except OSError:
            all_ips = []
        if not all_ips:
            all_ips = [preferred_ip]
        elif preferred_ip not in all_ips:
            all_ips.insert(0, preferred_ip)
        self._net_host_ip_combo.clear()
        for ip in all_ips:
            self._net_host_ip_combo.addItem(ip)
        idx = self._net_host_ip_combo.findText(preferred_ip)
        if idx >= 0:
            self._net_host_ip_combo.setCurrentIndex(idx)

    def _on_net_mode_client(self, checked):
        if not checked:
            return
        self._net_host_panel.setVisible(False)
        self._net_cli_panel.setVisible(True)

    def _on_net_connect_clicked(self):
        if self._sim_ctrl._net_client is not None:
            self._sim_ctrl.disconnect_from_host()
            self._net_cli_conn_btn.setText("Connect")
        else:
            local_port = self._net_cli_lport_spin.value()
            from brain_serializer import serialize_network_json as _ser_net
            c = self._sim_ctrl.circuit
            _circuit_json = _ser_net(
                c.sensors, c.layers, c.connections,
                set(), set(), {},
                bodies=c.bodies, joints=c.joints,
            )
            try:
                self._sim_ctrl.connect_to_host(
                    host         = self._net_cli_host_edit.text().strip(),
                    host_port    = self._net_cli_port_spin.value(),
                    local_port   = local_port,
                    name         = self._net_cli_name_edit.text().strip(),
                    circuit_json = _circuit_json,
                )
            except OSError as exc:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Network Error",
                                    f"Could not bind local port {local_port}:\n{exc}")
                return
            self._net_cli_conn_btn.setText("Disconnect")

    def _on_client_registered(self, info):
        """Main-thread handler: create a physics agent from the client's circuit JSON."""
        from brain_serializer import load_network_json
        from circuit_model import CircuitModel
        from rigid_body import RigidBody
        from brain_manager import BrainManager
        from arena_widget import _AGENT_COLORS

        slot_idx     = info['slot_idx']
        name         = info.get('name', '') or f'Remote {slot_idx}'
        circuit_json = info.get('circuit_json')

        new_circuit = CircuitModel()
        if circuit_json is not None:
            try:
                result = load_network_json(circuit_json)
                sensors, bodies, joints = result[0], result[6], result[7]
                new_circuit.sensors = sensors
                new_circuit.bodies  = bodies if bodies else \
                    [RigidBody('root', 'root', self.sim_cfg.body_radius)]
                new_circuit.joints  = joints
            except Exception as exc:
                print(f'[SimNet] Failed to parse circuit from slot {slot_idx}: {exc}')
                new_circuit.bodies = [RigidBody('root', 'root', self.sim_cfg.body_radius)]
        else:
            new_circuit.bodies = [RigidBody('root', 'root', self.sim_cfg.body_radius)]

        new_brain_mgr = BrainManager(new_circuit, self.sim_cfg)
        color = _AGENT_COLORS[len(self._sim_ctrl.groups_ordered()) % len(_AGENT_COLORS)]
        pos = len(self._sim_ctrl._agents)
        agent_id = self._sim_ctrl.add_agent(new_circuit, new_brain_mgr,
                                            name=name, color=color)

        # Minimal brain for the host side: just a sensor-value container.
        # loop() is never called because motor_override is always set from the client.
        class _RemoteBrain:
            def setup(self): pass
            def loop(self, _dt): return 0.0, 0.0
        brain = _RemoteBrain()
        brain.setup()
        self._sim_ctrl._agents[pos].brain  = brain
        self._sim_ctrl._agents[pos].remote = True

        # Register slot↔agent-id mapping so _tick() routes motors/sensors correctly.
        self._sim_ctrl._slot_to_agent_id[slot_idx] = agent_id
        self._sim_ctrl._agent_id_to_slot[agent_id] = slot_idx

        bp = self._sim_ctrl._agents[pos].bot_pos
        self._arena.add_robot_item(pos, color=color, x=bp[0], y=bp[1],
                                    r=self.sim_cfg.body_radius, theta=bp[2])
        self._sim_ctrl.create_group(module=None, color=color, name=name,
                                     first_agent_id=agent_id)
        self._refresh_agent_list()
        print(f'[SimNet] slot {slot_idx} → agent {agent_id} created ({name})')

    def _on_remote_agent_removed(self, agent_id):
        """Main-thread handler: remove the arena item and group for a disconnected client.
        agent_id is the disconnected agent's stable id (sig_agent_removed's payload)."""
        pos = self._sim_ctrl.index_of_agent(agent_id)
        if pos is None:
            return
        group_id = self._sim_ctrl.group_of_agent(agent_id)
        self._arena.remove_robot_item(pos)
        self._sim_ctrl.remove_agent(agent_id)   # also detaches it from its group
        if group_id is not None:
            g = self._sim_ctrl.get_group(group_id)
            if g is not None and not g.member_ids:
                self._sim_ctrl.remove_group(group_id)   # remote groups are always singleton
        # Re-sync the arena highlight in case removal changed the selected agent.
        new_pos = self._sim_ctrl.index_of_agent(self._sim_ctrl._selected_id)
        if new_pos is not None:
            self._arena.select_robot(new_pos)
        self._refresh_agent_list()
        print(f'[SimNet] agent {agent_id} removed')

    def _update_net_status(self):
        """Refresh network status labels (called at 1 Hz)."""
        host = self._sim_ctrl._net_host
        if host is not None:
            # User-configurable via the Timeout spinbox next to Frame rate
            # (Network tab); default matches the client's 2s heartbeat interval.
            host.prune_stale(timeout_s=self._sim_ctrl._net_disconnect_timeout)
            slots = host.connected_slots()
            self._net_client_list.clear()
            agent_list_dirty = False
            if slots:
                for idx in sorted(slots):
                    s = host.get_slot(idx)
                    if s is None:
                        continue
                    idle = s.idle_seconds
                    idle_txt = f"{idle:.1f}s" if idle < 10 else f"{idle:.0f}s"
                    # Sync the slot name into the agent and agent_groups if it changed.
                    # Use slot→agent-id mapping (slot_idx ≠ agent position in general),
                    # then translate the id to its current list position.
                    agent_id = self._sim_ctrl._slot_to_agent_id.get(idx)
                    pos = self._sim_ctrl.index_of_agent(agent_id) if agent_id is not None else None
                    if s.name and pos is not None:
                        agent = self._sim_ctrl._agents[pos]
                        if agent.name != s.name:
                            agent.name = s.name
                            group_id = self._sim_ctrl.group_of_agent(agent_id)
                            if group_id is not None:
                                self._sim_ctrl.get_group(group_id).name = s.name
                            agent_list_dirty = True
                    display_name = s.name or (
                        self._sim_ctrl._agents[pos].name
                        if pos is not None
                        else f"slot {idx}"
                    )
                    state_tag = " ✓ready" if s.ready else ""
                    heartbeat_ok = idle < 4.0   # 2× the 2 s heartbeat interval
                    hb_dot = "●"
                    hb_color = C['success'] if heartbeat_ok else C['warning']
                    self._net_client_list.addItem(
                        f"{hb_dot} slot {idx}  {display_name}  —"
                        f"  {s.client_addr}:{s.client_port}"
                        f"  ↓{s.send_hz:.0f}Hz ↑{s.recv_hz:.0f}Hz"
                        f"  idle {idle_txt}{state_tag}"
                    )
                    item = self._net_client_list.item(self._net_client_list.count() - 1)
                    item.setForeground(QColor(hb_color))
            else:
                self._net_client_list.addItem("(no clients connected)")
                item = self._net_client_list.item(0)
                item.setForeground(QColor(C['muted']))
            if agent_list_dirty:
                self._refresh_agent_list()

        client = self._sim_ctrl._net_client
        if client is not None:
            status = client.status
            colors = {
                'connecting': C['warning'],
                'connected':  C['success'],
                'ready':      C['warning'],
                'running':    C['success'],
                'lost':       C['danger'],
            }
            color = colors.get(status, C['muted'])
            self._net_cli_status_lbl.setText(status)
            self._net_cli_status_lbl.setStyleSheet(f"color:{color};")
            if status == 'connected':
                self._net_cli_conn_btn.setText("Disconnect")

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
        if not self._sim_ctrl._robot_mode:
            return
        now = time.monotonic()
        sensor_rates = self._sim_ctrl._robot_driver.get_rates()
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
        mt = self._sim_ctrl._motor_thread
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
            self._editor.set_object_mode(color, getattr(self._editor, 'object_texture', None))
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
                self._editor.set_object_mode(self._editor.object_color, self._editor.object_texture)

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

        if self._video_auto_cb.isChecked():
            if text == "●  RUNNING" and not self._video_recorders:
                self._video_start()
            elif text == "■  STOPPED" and self._video_recorders:
                self._video_stop()

    def _on_run_stop(self):
        if self._sim_ctrl.running:
            self._sim_ctrl.stop()
        else:
            self._reset()
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

    # ── World setup ──────────────────────────────────────────────────────────

    def _setup_world(self, rebuild=True):
        self._arena.setup(self.sim_cfg, self.world)
        self._sim_ctrl.sync_robot_markers()
        if rebuild:
            self._sim_ctrl.rebuild_mujoco(self.world, self.sim_cfg)
            if self._sim_ctrl._view_3d and not self._sim_ctrl.running:
                self._sim_ctrl.render_mujoco_overhead()
        elif self._sim_ctrl._view_3d and not self._sim_ctrl.running:
            # Called mid-drag (e.g. dragging the robot) — a full rebuild is too
            # expensive per mouse-move, but the overhead render must still track
            # the live position, or the MuJoCo image shows a stale robot until drop.
            self._sim_ctrl.reposition_mujoco_robots()

    def _clear_world(self):
        self.world.patches = []
        self.world.objects = []
        self._setup_world()

    # ── World save/load (mirrors _BrainMixin's brain combo pattern) ────────────

    def _save_world(self):
        from PySide6.QtWidgets import QInputDialog
        current = self._world_combo.currentText()
        default = current[:-5] if current.endswith('.json') else current
        raw, ok = QInputDialog.getText(self, "Save World", "Name:", text=default)
        if not ok or not raw:
            return
        name = "".join(c for c in raw if c.isalnum() or c in "._- ") or "world"
        if not name.endswith('.json'):
            name += '.json'
        save_world_file(self.world, os.path.join('worlds', name))
        self._refresh_world_list()
        self._world_combo.setCurrentText(name)
        print(f"World saved to worlds/{name}")

    def _refresh_world_list(self):
        current = self._world_combo.currentText()
        self._world_combo.blockSignals(True)
        self._world_combo.clear()
        self._world_combo.addItems(discover_worlds())
        if current in discover_worlds():
            self._world_combo.setCurrentText(current)
        self._world_combo.blockSignals(False)

    def _load_world(self, name):
        if not name:
            return
        path = os.path.join('worlds', name)
        if not os.path.exists(path):
            return
        load_world_file(path, self.world)
        self._arena_round_rb.setChecked(self.world.arena_round)
        self._arena_square_rb.setChecked(not self.world.arena_round)
        self._sky_cb.setChecked(bool(self.world.sky.get("enabled", False)))
        floor = self.world.floor_texture or "(default)"
        self._floor_texture_combo.blockSignals(True)
        self._floor_texture_combo.setCurrentText(floor)
        self._floor_texture_combo.blockSignals(False)
        self._setup_world(rebuild=True)
        print(f"World loaded from {path}")

    def _on_floor_texture_change(self, name):
        self.world.floor_texture = None if name == "(default)" else name
        self._setup_world(rebuild=True)

    def _set_object_texture(self, name):
        texture = None if name == "(none)" else name
        if hasattr(self, '_editor'):
            self._editor.object_texture = texture

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
        """Rebuild the agent table from SimController's groups (one row per group)."""
        groups = self._sim_ctrl.groups_ordered()
        self._agent_table.blockSignals(True)
        self._agent_table.setRowCount(0)
        for i, group in enumerate(groups):
            self._agent_table.insertRow(i)

            # Col 0: color swatch button (centered in cell)
            btn = QPushButton()
            btn.setFixedSize(18, 18)
            btn.setStyleSheet(
                f"background-color: {group.color}; border: none; border-radius: 3px;"
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
            name_item = QTableWidgetItem(group.name)
            self._agent_table.setItem(i, 1, name_item)

            # Col 2: brain module (read-only)
            brain_item = QTableWidgetItem(group.module or "")
            brain_item.setFlags(brain_item.flags() & ~Qt.ItemIsEditable)
            self._agent_table.setItem(i, 2, brain_item)

            # Col 3: N spinbox (setValue before connecting to avoid spurious signal)
            sb = QSpinBox()
            sb.setRange(1, 20)
            sb.setValue(len(group.member_ids))
            sb.valueChanged.connect(lambda v, g=i: self._on_agent_n_changed(g, v))
            self._agent_table.setCellWidget(i, 3, sb)

        sel_group_id = self._sim_ctrl.group_of_agent(self._sim_ctrl._selected_id)
        group_row = next((i for i, g in enumerate(groups) if g.id == sel_group_id), -1)
        self._agent_table.setCurrentCell(group_row, 1)
        self._agent_table.blockSignals(False)

    def _add_agent(self, color=None, name=None):
        """Add a new agent group (n=1) with an optional brain load."""
        from arena_widget import _AGENT_COLORS
        new_circuit = CircuitModel()
        new_circuit.bodies = [RigidBody('root', 'root', self.sim_cfg.body_radius)]
        new_brain_mgr = BrainManager(new_circuit, self.sim_cfg)
        n_groups = len(self._sim_ctrl.groups_ordered())
        if color is None:
            color = _AGENT_COLORS[n_groups % len(_AGENT_COLORS)]
        if name is None:
            name = f'Group {n_groups + 1}'
        # add_agent() returns the new agent's stable id, not its list position;
        # capture the position beforehand (it always appends at the end).
        pos = len(self._sim_ctrl._agents)
        agent_id = self._sim_ctrl.add_agent(new_circuit, new_brain_mgr,
                                            name=f"Agent {pos + 1}", color=color)
        bp = self._sim_ctrl._agents[pos].bot_pos
        self._arena.add_robot_item(pos, color=color, x=bp[0], y=bp[1],
                                    r=self.sim_cfg.body_radius, theta=bp[2])
        self._sim_ctrl.create_group(module=None, color=color, name=name,
                                     first_agent_id=agent_id)
        self._refresh_agent_list()
        # Select the new group and load the current brain into it
        self._select_agent(agent_id)
        current_brain = self._brain_combo.currentText()
        if current_brain:
            self.load_brain(current_brain)

    def _remove_agent(self):
        """Remove the currently selected group and all its agents (min 1 agent total)."""
        row = self._agent_table.currentRow()
        groups = self._sim_ctrl.groups_ordered()
        if row < 0 or row >= len(groups):
            return
        group = groups[row]
        # Refuse if this would leave zero agents
        other_total = len(self._sim_ctrl._agents) - len(group.member_ids)
        if other_total < 1:
            return
        # Stable ids never invalidate each other, so removal order doesn't matter.
        for agent_id in list(group.member_ids):
            pos = self._sim_ctrl.index_of_agent(agent_id)
            if pos is not None:
                self._arena.remove_robot_item(pos)
            self._sim_ctrl.remove_agent(agent_id)   # also detaches it from the group
        self._sim_ctrl.remove_group(group.id)
        self._refresh_agent_list()

    def _pick_agent_color(self, group_idx):
        """Open a color dialog and apply the chosen color to all agents in the group."""
        groups = self._sim_ctrl.groups_ordered()
        if group_idx >= len(groups):
            return
        group = groups[group_idx]
        color = QColorDialog.getColor(QColor(group.color), self)
        if color.isValid():
            group.color = color.name()
            for agent_id in group.member_ids:
                pos = self._sim_ctrl.index_of_agent(agent_id)
                if pos is not None:
                    self._sim_ctrl._agents[pos].color = color.name()
                    self._arena.update_robot_color(pos, color.name())
            self._refresh_agent_list()

    def _on_agent_name_changed(self, item):
        """Persist inline name edits back to the group data model."""
        if item.column() == 1:
            row = item.row()
            groups = self._sim_ctrl.groups_ordered()
            if 0 <= row < len(groups):
                groups[row].name = item.text()

    def _on_agent_selected(self, row):
        """Select the first agent in the clicked group row."""
        groups = self._sim_ctrl.groups_ordered()
        if row < 0 or row >= len(groups):
            return
        group = groups[row]
        if not group.member_ids:
            return
        self._select_agent(group.member_ids[0])

    # ── Group / agent helpers ─────────────────────────────────────────────────

    def _select_agent(self, agent_id):
        """Select a specific agent for oscilloscope / network viz, sync group table row."""
        if self._sim_ctrl._agent_by_id(agent_id) is None:
            return
        self._sim_ctrl.select_agent(agent_id)
        pos = self._sim_ctrl.index_of_agent(agent_id)
        self._arena.select_robot(pos)
        if hasattr(self, '_editor'):
            self._editor._bot_pos = self._sim_ctrl.bot_pos
        group_id = self._sim_ctrl.group_of_agent(agent_id)
        group = self._sim_ctrl.get_group(group_id) if group_id is not None else None
        groups = self._sim_ctrl.groups_ordered()
        group_row = next((i for i, g in enumerate(groups) if g.id == group_id), -1)
        if group is not None:
            mod = group.module
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
        self._agent_table.setCurrentCell(group_row, 1)
        self._agent_table.blockSignals(False)

    def _on_agent_n_changed(self, group_idx, new_n):
        """Spinbox value changed: add or remove agents for the group."""
        groups = self._sim_ctrl.groups_ordered()
        if group_idx >= len(groups):
            return
        group = groups[group_idx]
        old_n = len(group.member_ids)
        delta = new_n - old_n
        prev_sel_id = self._sim_ctrl._selected_id
        if delta > 0:
            for _ in range(delta):
                self._add_agent_to_group(group_idx)
        elif delta < 0:
            for _ in range(-delta):
                self._remove_agent_from_group(group_idx)
        # Restore the agent that was selected before the resize, if it still
        # exists; otherwise fall back to the last remaining member of this
        # group (there's no positional index left to clamp, unlike before).
        if self._sim_ctrl._agent_by_id(prev_sel_id) is not None:
            self._select_agent(prev_sel_id)
        elif group.member_ids:
            self._select_agent(group.member_ids[-1])
        elif self._sim_ctrl._agents:
            self._select_agent(self._sim_ctrl._agents[0].id)

    def _add_agent_to_group(self, group_idx):
        """Spawn one more agent for the given group and load its brain."""
        group = self._sim_ctrl.groups_ordered()[group_idx]
        new_circuit = CircuitModel()
        new_circuit.bodies = [RigidBody('root', 'root', self.sim_cfg.body_radius)]
        new_brain_mgr = BrainManager(new_circuit, self.sim_cfg)
        # add_agent() returns the new agent's stable id, not its list position;
        # capture the position beforehand (it always appends at the end).
        pos = len(self._sim_ctrl._agents)
        agent_id = self._sim_ctrl.add_agent(new_circuit, new_brain_mgr,
                                            name=f"Agent {pos + 1}",
                                            color=group.color)
        bp = self._sim_ctrl._agents[pos].bot_pos
        self._arena.add_robot_item(pos, color=group.color, x=bp[0], y=bp[1],
                                    r=self.sim_cfg.body_radius, theta=bp[2])
        self._sim_ctrl.add_agent_to_group(group.id, agent_id)
        if group.module:
            # Copy brain params from the group's first agent (member_ids[0] is
            # always the first agent added to this group). This carries over
            # network_file/network_project for DataBrain so load_brain can find and
            # load the same JSON network for the new agent.
            original_brain = self._sim_ctrl._agent_by_id(group.member_ids[0]).brain
            params_copy = (
                {k: getattr(original_brain, k) for k in original_brain.get_param_metadata()}
                if original_brain is not None else None
            )
            # Temporarily select the new agent so load_brain targets it correctly
            self._sim_ctrl.select_agent(agent_id)
            self.load_brain(group.module, external_params=params_copy)
            # Selection is restored by _on_agent_n_changed after all agents are added

    def _remove_agent_from_group(self, group_idx):
        """Remove the last agent from the group (refuses if it would leave zero total)."""
        if len(self._sim_ctrl._agents) <= 1:
            return
        group = self._sim_ctrl.groups_ordered()[group_idx]
        if not group.member_ids:
            return
        agent_id = group.member_ids[-1]
        pos = self._sim_ctrl.index_of_agent(agent_id)
        if pos is not None:
            self._arena.remove_robot_item(pos)
        self._sim_ctrl.remove_agent(agent_id)   # also detaches it from the group

    def _remove_selected_agent(self):
        """Remove the specific agent currently selected (Delete key), regardless
        of its position within its group; removes the group too if left empty."""
        agent_id = self._sim_ctrl._selected_id
        if agent_id is None or len(self._sim_ctrl._agents) <= 1:
            return
        group_id = self._sim_ctrl.group_of_agent(agent_id)
        pos = self._sim_ctrl.index_of_agent(agent_id)
        if pos is not None:
            self._arena.remove_robot_item(pos)
        self._sim_ctrl.remove_agent(agent_id)   # also detaches it from the group
        if group_id is not None:
            group = self._sim_ctrl.get_group(group_id)
            if group is not None and not group.member_ids:
                self._sim_ctrl.remove_group(group_id)
        self._refresh_agent_list()

    def _on_arena_click(self, x, y, btn):
        """Select the nearest agent when the user left-clicks the arena."""
        # WorldEditor already consumes left-clicks in these modes (placing a polygon
        # vertex / painting a wall); a click there must not also reselect the agent.
        if self._editor.draw_mode in ('object', 'wall_paint'):
            return
        if btn != 1 or len(self._sim_ctrl._agents) <= 1:
            return
        r_thresh = self._sim_ctrl.sim_cfg.body_radius * 2.5
        best_agent, best_dist = None, float('inf')
        for agent in self._sim_ctrl._agents:
            dx = agent.bot_pos[0] - x
            dy = agent.bot_pos[1] - y
            d = (dx * dx + dy * dy) ** 0.5
            if d < r_thresh and d < best_dist:
                best_agent, best_dist = agent, d
        if best_agent is not None:
            self._select_agent(best_agent.id)

    # Brain loading / network viz / brain params UI → sim_app_brain._BrainMixin

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

    # _save_session / _load_session / _load_session_agents / _refresh_session_list → sim_app_session._SessionMixin

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Delete:
            self._remove_selected_agent()
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
