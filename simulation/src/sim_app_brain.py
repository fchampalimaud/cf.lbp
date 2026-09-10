import os
import time
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QMessageBox,
)
from PySide6.QtCore import Qt

from sim_constants import C
from circuit_model import CircuitModel, Connection
from rigid_body import RigidBody, world_poses as _rb_world_poses
from brain_base import DataBrain
from neurons import MotorLayer
from network_viz import NetworkVisualizerWindow
from network_viz_context import NetworkVizContext


class _BrainMixin:

    # ── Panel builder ─────────────────────────────────────────────────────────

    def _build_brain_group(self, panel_vl=None):
        gb, vl = self._panel.add_group("Brain", panel_vl)

        agent_hdr = QWidget(); ahl = QHBoxLayout(agent_hdr)
        ahl.setContentsMargins(0, 0, 0, 0)
        ahl.addWidget(QLabel("Agents"))
        ahl.addStretch()
        btn_add_agent = self._make_btn("+", C['primary'])
        btn_add_agent.setFixedWidth(24)
        btn_add_agent.setToolTip("Add agent")
        # Wrapped in a lambda: clicked(checked: bool) would otherwise land in
        # _add_agent's `color` parameter and get passed straight through to
        # pg.mkPen(), crashing the new agent's trail construction.
        btn_add_agent.clicked.connect(lambda: self._add_agent())
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

    # ── Brain list ────────────────────────────────────────────────────────────

    def _refresh_brain_list(self):
        current = self._brain_combo.currentText()
        self.brain_files = self.brain_mgr.discover_brains()
        self._brain_combo.blockSignals(True)
        self._brain_combo.clear()
        self._brain_combo.addItems(self.brain_files)
        if current in self.brain_files:
            self._brain_combo.setCurrentText(current)
        self._brain_combo.blockSignals(False)

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

    # ── Network file helpers ──────────────────────────────────────────────────

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
        hidden, disabled, container_labels, container_notes, conn_params, freshness_issues = \
            self.brain_mgr.load_network_into_circuit(self.brain, full_name)
        if hidden is None:
            return
        self._connection_params = conn_params
        self._hidden_containers       = hidden
        self._disabled_containers     = disabled
        self._container_labels        = container_labels
        self._container_notes         = container_notes
        if self._net_viz:
            self._net_viz._hidden_containers   = hidden
            self._net_viz._disabled_containers = disabled
            self._net_viz._container_labels    = container_labels
            self._net_viz._container_notes     = container_notes
            self._net_viz._weight_params = conn_params
            self._net_viz.build()
        self._osc_ctrl._osc_items = {'mL', 'mR'}
        self._rebuild_channels()
        self._arena.setup_sensors(self.circuit.sensors, self._osc_ctrl.channel_colors)
        if self._sim_ctrl.robot.enabled:
            # Real-robot mode was already on when this (different) network loaded —
            # RobotDriver was started against the *previous* circuit's sensors and
            # won't pick up this network's robot_address fields on its own, since
            # nothing re-triggers a connect except the checkbox itself. Reconnect
            # now against the freshly loaded circuit.sensors.
            self._sim_ctrl.enable_robot_mode(True)
            self._robot_last_seen = {}
            self._robot_connect_t = time.monotonic()
        poses = _rb_world_poses(self.bot_pos, self.circuit.bodies, self.circuit.joints)
        self._arena.update_child_bodies(poses, self.circuit.bodies, self.sim_cfg)
        if freshness_issues and not getattr(self, '_syncing_group', False):
            self._prompt_network_update(full_name, freshness_issues)
        if not getattr(self, '_syncing_group', False):
            self._sync_group_network()

    def _sync_group_network(self):
        """Push network_file + network_project from the current agent to every
        other agent in its group so all siblings share the same network structure."""
        sel_id = self._sim_ctrl._selected_id
        group_id = self._sim_ctrl.group_of_agent(sel_id)
        if group_id is None:
            return
        group = self._sim_ctrl.get_group(group_id)
        if not group.module or len(group.member_ids) <= 1:
            return
        src_brain = self._sim_ctrl._agent_by_id(sel_id).brain
        if src_brain is None:
            return
        net_params = {k: getattr(src_brain, k)
                      for k in ('network_file', 'network_project')
                      if hasattr(src_brain, k)}
        if not net_params.get('network_file'):
            return
        self._syncing_group = True
        try:
            for agent_id in group.member_ids:
                if agent_id == sel_id:
                    continue
                self._sim_ctrl.select_agent(agent_id)
                self.load_brain(group.module, external_params=net_params)
        finally:
            self._syncing_group = False
        self._sim_ctrl.select_agent(sel_id)
        self._arena.select_robot(self._sim_ctrl.index_of_agent(sel_id))
        self._rebuild_brain_params()
        self._rebuild_channels()
        if self._net_viz:
            self._net_viz.build()

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
                    self._hidden_containers,
                    self._disabled_containers,
                    self._container_labels,
                    self.circuit.bodies,
                    self.circuit.joints,
                    self._connection_params,
                    container_notes=self._container_notes,
                )
            except Exception as e:
                QMessageBox.critical(self, 'Save failed', str(e))

    # ── Brain loading ─────────────────────────────────────────────────────────

    def load_brain(self, name=None, external_params=None):
        if name is None or isinstance(name, bool):
            name = self._brain_combo.currentText()
        if not name:
            return

        self._brain_combo.blockSignals(True)
        self._brain_combo.setCurrentText(name)
        self._brain_combo.blockSignals(False)

        group_id = self._sim_ctrl.group_of_agent(self._sim_ctrl._selected_id)
        if group_id is not None:
            self._sim_ctrl.get_group(group_id).module = name

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

    # ── Channel / oscilloscope ────────────────────────────────────────────────

    def _rebuild_channels(self):
        if self.brain is not None:
            self.brain.layers      = self.circuit.layers
            self.brain.sensors     = self.circuit.sensors
            self.brain.connections = self.circuit.connections
            self.brain_mgr.resolve_joint_sensor_refs()
        self._osc_ctrl.rebuild_channels(self.brain, self.circuit)
        self._osc_ctrl.setup_osc()
        self._rebuild_robot_rows()
        self._sync_mujoco_collision_geometry()

    def _mujoco_collision_sensor_signature(self):
        """Snapshot of everything that affects the per-sector MuJoCo geometry
        MuJoCoEngine builds for CollisionSensors (see sim_engine_mujoco.py) —
        compared cheaply on every _rebuild_channels() call so a full MuJoCo
        model rebuild only happens when this actually changes, not on every
        unrelated circuit edit (which would otherwise make interactive
        editing/dragging noticeably laggy)."""
        from sensors import CollisionSensor
        sig = []
        for agent in self._sim_ctrl.registry.agents:
            for s in getattr(agent.circuit, 'sensors', []):
                if isinstance(s, CollisionSensor):
                    sig.append((agent.id, s.name, s.n, round(s.angle_spread, 6),
                                round(s.arc_angle, 6), round(s.radius, 6),
                                tuple(getattr(s, 'body_ids', None) or ())))
        return tuple(sig)

    def _sync_mujoco_collision_geometry(self):
        """Rebuild the MuJoCo model when a CollisionSensor's geometry-relevant
        params changed since the last check — covers creation, deletion, and
        property edits uniformly without needing a rebuild call at every
        individual editing site. No-op when MuJoCo is off."""
        if not self._sim_ctrl.mujoco.enabled:
            return
        sig = self._mujoco_collision_sensor_signature()
        if sig != getattr(self, '_mujoco_collision_sig', None):
            self._mujoco_collision_sig = sig
            self._sim_ctrl.rebuild_mujoco(self.world, self.sim_cfg)
        if hasattr(self, '_arena'):
            self._arena.setup_sensors(self.circuit.sensors, self._osc_ctrl.channel_colors)

    def _toggle_osc_layer(self, name):
        self._osc_ctrl.toggle_layer(name)
        self._rebuild_channels()
        if self._net_viz:
            self._net_viz._redraw_nodes()

    # ── Network visualizer ────────────────────────────────────────────────────

    def _toggle_network_viz(self):
        if self._net_viz:
            self._net_viz.close()
            self._net_viz = None
        else:
            self._net_viz = NetworkVisualizerWindow(NetworkVizContext(self))
            self._net_viz._weight_params = self._connection_params
            self._net_viz._hidden_containers   = self._hidden_containers
            self._net_viz._disabled_containers = self._disabled_containers
            self._net_viz._container_labels    = self._container_labels
            self._net_viz._container_notes     = self._container_notes
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
        if hasattr(self.brain, '_w_cache'):
            self.brain._w_cache = {}
        for layer in self.circuit.layers:
            if isinstance(layer, _LLB):
                layer.reset()
