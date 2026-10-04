import os
import time
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QMessageBox, QButtonGroup,
)
from PySide6.QtCore import Qt, QTimer

from sim_constants import C
from rigid_body import world_poses as _rb_world_poses
from brain_base import DataBrain
from neurons import MotorLayer
from neurons_base import fast_taus, fast_tau_warning
import data_paths
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

        # Mode: the selected group runs either a code brain (a brains/*.py
        # class) or a network (a networks/*.json circuit, run by the
        # network brain module). Derived from the group's module, so old
        # sessions need nothing special.
        mode_row = QWidget(); mhl = QHBoxLayout(mode_row); mhl.setContentsMargins(0, 0, 0, 0)
        self._mode_code_btn = self._make_toggle(
            "Code brain", "Run a brain written in Python (brains/*.py)")
        self._mode_net_btn = self._make_toggle(
            "Network", "Run a circuit designed in the network visualizer (networks/*.json)")
        self._brain_mode_group = QButtonGroup(self)
        self._brain_mode_group.setExclusive(True)
        for b in (self._mode_net_btn, self._mode_code_btn):
            self._brain_mode_group.addButton(b)
            mhl.addWidget(b)
        self._mode_code_btn.clicked.connect(lambda: self._set_brain_mode('code'))
        self._mode_net_btn.clicked.connect(lambda: self._set_brain_mode('network'))
        vl.addWidget(mode_row)

        # Code brain: class + reload + scaffold
        self._code_row = QWidget(); hl1 = QHBoxLayout(self._code_row)
        hl1.setContentsMargins(0, 0, 0, 0)
        self._brain_combo = QComboBox()
        self._brain_combo.addItems(self._code_brain_files())
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
        vl.addWidget(self._code_row)

        # Network: project + file + open visualizer
        self._net_panel = QWidget(); nvl = QVBoxLayout(self._net_panel)
        nvl.setContentsMargins(0, 0, 0, 0); nvl.setSpacing(4)
        proj_row = QWidget(); phl = QHBoxLayout(proj_row); phl.setContentsMargins(0, 0, 0, 0)
        phl.addWidget(QLabel("Project"))
        self._net_root_combo = self._make_root_combo()
        self._net_root_combo.currentIndexChanged.connect(
            lambda _i: self._on_net_project_chosen(
                self._first_folder_key('networks', self._net_root_combo.currentData())))
        phl.addWidget(self._net_root_combo)
        self._net_project_combo = QComboBox()
        self._net_project_combo.currentIndexChanged.connect(
            lambda _i: self._on_net_project_chosen(self._net_project_combo.currentData() or ''))
        phl.addWidget(self._net_project_combo, 1)
        self._btn_new_proj = self._make_btn("+", C['primary'])
        self._btn_new_proj.setFixedWidth(28)
        self._btn_new_proj.setToolTip("Create a new project folder in your files")
        self._btn_new_proj.clicked.connect(self._new_network_project)
        phl.addWidget(self._btn_new_proj)
        nvl.addWidget(proj_row)
        file_row = QWidget(); fhl = QHBoxLayout(file_row); fhl.setContentsMargins(0, 0, 0, 0)
        fhl.addWidget(QLabel("Network"))
        self._net_file_combo = QComboBox()
        self._net_file_combo.currentTextChanged.connect(self._on_net_file_chosen)
        fhl.addWidget(self._net_file_combo, 1)
        btn_new_net = self._make_btn("+", C['primary'])
        btn_new_net.setFixedWidth(28)
        btn_new_net.setToolTip("Create a new network file (motor layer only)")
        btn_new_net.clicked.connect(self._new_network_from_sidebar)
        fhl.addWidget(btn_new_net)
        nvl.addWidget(file_row)
        btn_open_viz = self._make_btn("⬡ Open visualizer", C['primary'])
        btn_open_viz.setToolTip("Open the network visualizer for the selected group")
        btn_open_viz.clicked.connect(self._open_network_viz)
        nvl.addWidget(btn_open_viz)
        vl.addWidget(self._net_panel)

        self._brain_params_group, self._brain_params_layout = \
            self._panel.add_group("Brain Parameters", panel_vl)

    # ── Brain list ────────────────────────────────────────────────────────────

    def _refresh_brain_list(self):
        current = self._brain_combo.currentText()
        self.brain_files = self.brain_mgr.discover_brains()
        self.network_brain_modules = self.brain_mgr.network_brain_modules
        code = self._code_brain_files()
        self._brain_combo.blockSignals(True)
        self._brain_combo.clear()
        self._brain_combo.addItems(code)
        if current in code:
            self._brain_combo.setCurrentText(current)
        self._brain_combo.blockSignals(False)

    # ── Code brain / Network mode ─────────────────────────────────────────────

    def _code_brain_files(self):
        """Brain modules written in Python — the network brain module is not
        offered here; it is what Network mode runs."""
        return [m for m in self.brain_files
                if m not in getattr(self, 'network_brain_modules', ())]

    def _network_module(self):
        """The module that runs a JSON circuit (BrainGUI), or None."""
        mods = sorted(getattr(self, 'network_brain_modules', ()))
        return 'BrainGUI' if 'BrainGUI' in mods else (mods[0] if mods else None)

    def _selected_group(self):
        reg = self._sim_ctrl.registry
        gid = reg.group_of_agent(reg.selected_id)
        return reg.get_group(gid) if gid is not None else None

    def _is_network_module(self, module):
        return bool(module) and module in getattr(self, 'network_brain_modules', ())

    def _set_brain_mode(self, mode):
        """Switch the selected group between running a code brain and a network."""
        group = self._selected_group()
        current = group.module if group is not None else None
        if mode == 'network':
            if self._is_network_module(current):
                return
            net_mod = self._network_module()
            if net_mod is None:
                QMessageBox.warning(self, "Network", "No network brain module found in brains/.")
                self._refresh_brain_mode_ui()
                return
            self.load_brain(net_mod)
        else:
            if current and not self._is_network_module(current):
                return
            code = self._code_brain_files()
            target = (group.last_code_module if group is not None else None) \
                or self._brain_combo.currentText() or (code[0] if code else None)
            if target:
                self.load_brain(target)

    def _refresh_brain_mode_ui(self):
        """Show the selected group's mode: toggle state, and the code row or the
        network panel (with its project / file lists)."""
        group = self._selected_group()
        is_net = self._is_network_module(group.module if group is not None else None)
        self._mode_net_btn.setChecked(is_net)
        self._mode_code_btn.setChecked(not is_net)
        self._code_row.setVisible(not is_net)
        self._net_panel.setVisible(is_net)
        if is_net:
            self._refresh_network_panel()
        self._update_net_viz_title()

    def _refresh_network_panel(self):
        """Project / file lists for the selected group's network brain."""
        brain = self.brain
        key = self._net_project_key(getattr(brain, 'network_project', '') or '')
        current = getattr(brain, 'network_file', '') or ''
        # Root (My files / 🔒 Simulator), then that root's projects.
        builtin = self._fill_folder_picker(self._net_root_combo, self._net_project_combo, 'networks', key)
        self._btn_new_proj.setEnabled(not builtin)   # the simulator's projects are read-only
        folder = data_paths.folder_for('networks', key)
        files = sorted(f for f in os.listdir(folder) if f.endswith('.json')) \
            if folder.is_dir() else []
        combo = self._net_file_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([''] + files)
        combo.setCurrentText(current if current in files else '')
        combo.blockSignals(False)

    @staticmethod
    def _net_project_key(project):
        """Picker key for a stored network_project: older sessions store a
        built-in project by its plain name ('Tutorials'); the picker shows it
        as the built-in entry unless the user has a folder of that name."""
        if (project and not project.startswith(data_paths.BUILTIN)
                and project not in data_paths.user_subdirs('networks')
                and project in data_paths.builtin_subdirs('networks')):
            return data_paths.BUILTIN + project
        return project

    def _on_net_project_chosen(self, project):
        if self.brain is None or not hasattr(self.brain, 'network_project'):
            return
        self.brain.network_project = project
        self.brain.network_file = ''
        self._refresh_network_panel()

    def _on_net_file_chosen(self, net_name):
        if self.brain is None or not hasattr(self.brain, 'network_file'):
            return
        self.brain.network_file = net_name
        if net_name:
            self._load_data_brain_network(net_name)
        self._refresh_agent_list()
        self._update_net_viz_title()

    def _open_network_viz(self):
        if self._net_viz:
            self._net_viz.raise_()
            self._net_viz.activateWindow()
        else:
            self._toggle_network_viz()

    def _update_net_viz_title(self):
        """Visualizer title names the group and what it runs (one visualizer,
        following the selected group)."""
        if not self._net_viz:
            return
        group = self._selected_group()
        name = group.name if group is not None else ''
        if self._is_network_module(group.module if group is not None else None):
            what = getattr(self.brain, 'network_file', '') or '(no network file)'
        else:
            what = (group.module if group is not None else '') or ''
        self._net_viz.setWindowTitle(f"Network — {name}: {what}" if name else "Network")

    def _new_brain(self):
        from PySide6.QtWidgets import QInputDialog
        raw, ok = QInputDialog.getText(self, "New Brain", "Name (e.g. MyBrain):")
        if not ok or not raw:
            return
        name       = ''.join(c for c in raw if c.isalnum() or c == '_').strip('_')
        class_name = name if name.startswith('Brain') else f"Brain{name[0].upper()}{name[1:]}"
        path = self.brain_mgr.create_brain_file(class_name)
        if path is None:
            QMessageBox.information(self, 'New Brain',
                                    f"A brain named {class_name} already exists (yours or built-in).")
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
        data_paths.user_path('networks', name).mkdir(parents=True, exist_ok=True)
        self.brain.network_project = name
        self.brain.network_file = ''
        self._refresh_network_panel()

    def _new_network_from_sidebar(self):
        from PySide6.QtWidgets import QInputDialog
        from brain_serializer import save_network_file
        name, ok = QInputDialog.getText(self, 'New Network', 'Network name:')
        if not ok or not name.strip():
            return
        name = name.strip()
        if not name.endswith('.json'):
            name += '.json'
        # New networks always go to the user's folder; a built-in project is
        # mirrored there by name.
        project = getattr(self.brain, 'network_project', '') or ''
        if project.startswith(data_paths.BUILTIN):
            project = project[len(data_paths.BUILTIN):]
            self.brain.network_project = project
        net_dir = data_paths.user_path('networks', project)
        net_dir.mkdir(parents=True, exist_ok=True)
        motor = MotorLayer(activation='linear', name='motor', n=2, layer=4)
        path  = os.path.join(net_dir, name)
        try:
            save_network_file(path, [], [motor], [], set(), set(), {})
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            return
        self.brain.network_file = name
        self._load_data_brain_network(name)
        self._refresh_network_panel()
        self._refresh_agent_list()
        self._update_net_viz_title()

    def _load_data_brain_network(self, net_name: str):
        net_info = self.brain_mgr.load_data_network(self.brain, net_name)
        if net_info is not None:
            self._apply_loaded_network(net_info)

    def _apply_loaded_network(self, net_info):
        """UI half of loading a DataBrain network (the model half is
        BrainManager.load_data_network): visualizer state, channels, arena,
        robot reconnect, freshness prompt, group sync."""
        full_name, hidden, disabled, container_labels, container_notes, conn_params, freshness_issues = net_info
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
            # Ask once loading has finished (not mid-load), about this very
            # circuit — the selected group may have changed by then.
            saved = (self.circuit, self._hidden_containers, self._disabled_containers,
                     self._container_labels, self._connection_params, self._container_notes)
            QTimer.singleShot(0, lambda: self._prompt_network_update(
                full_name, freshness_issues, saved))
        if not getattr(self, '_syncing_group', False):
            self._sync_group_network()
        self._report_fast_taus()

    def _report_fast_taus(self):
        """Status-bar alert if any agent's circuit has a tau shorter than dt
        (TODO 1.1 — only reported; the integration is left as it is)."""
        dt = self.sim_cfg.dt
        hits = []
        for agent in self._sim_ctrl.registry.agents:
            c = agent.circuit
            hits += fast_taus(list(c.sensors) + list(c.layers), dt)
        msg = fast_tau_warning(sorted(set(hits)), dt)
        if msg:
            self._status_bar.showMessage(msg, 20000)

    def _sync_group_network(self):
        """Push network_file + network_project from the current agent to every
        other agent in its group so all siblings share the same network structure."""
        sel_id = self._sim_ctrl.registry.selected_id
        group_id = self._sim_ctrl.registry.group_of_agent(sel_id)
        if group_id is None:
            return
        group = self._sim_ctrl.registry.get_group(group_id)
        if not group.module or len(group.member_ids) <= 1:
            return
        src_brain = self._sim_ctrl.registry.agent_by_id(sel_id).brain
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
                self._sim_ctrl.registry.select_agent(agent_id)
                self.load_brain(group.module, external_params=net_params)
        finally:
            self._syncing_group = False
        self._sim_ctrl.registry.select_agent(sel_id)   # the arena highlight follows
        self._rebuild_brain_params()
        self._rebuild_channels()
        if self._net_viz:
            self._net_viz.build()

    def _prompt_network_update(self, net_name: str, issues: list, saved):
        """Show a dialog reporting stale params and offer to resave with current
        defaults. saved = (circuit, hidden, disabled, labels, conn_params, notes)
        captured when the network was loaded."""
        circuit, hidden, disabled, labels, conn_params, notes = saved
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
            path = data_paths.resolve('networks', net_name)
            if path is None or data_paths.is_builtin(path):
                QMessageBox.information(
                    self, 'Built-in network',
                    'This network ships with the simulator and is read-only. Use Save in '
                    'the network window to keep your own up-to-date copy.')
                return
            try:
                save_network_file(
                    path,
                    circuit.sensors,
                    circuit.layers,
                    circuit.connections,
                    hidden,
                    disabled,
                    labels,
                    circuit.bodies,
                    circuit.joints,
                    conn_params,
                    container_notes=notes,
                )
                self._status_bar.showMessage(f"✓ Network updated: {path}", 5000)
            except Exception as e:
                QMessageBox.critical(self, 'Save failed', str(e))

    # ── Brain loading ─────────────────────────────────────────────────────────

    def load_brain(self, name=None, external_params=None):
        if name is None or isinstance(name, bool):
            name = self._brain_combo.currentText()
        if not name:
            return

        if not self._is_network_module(name):
            self._brain_combo.blockSignals(True)
            self._brain_combo.setCurrentText(name)
            self._brain_combo.blockSignals(False)

        group = self._selected_group()
        if group is not None:
            group.module = name
            if not self._is_network_module(name):
                group.last_code_module = name

        # Model half (shared with headless runs): instantiate + wire the circuit.
        brain, loaded_json, net_info = self.brain_mgr.install_brain(name, external_params)
        if not brain:
            return
        self._sim_ctrl.registry.brain = brain
        if net_info is not None:
            self._apply_loaded_network(net_info)

        self._sim_ctrl.registry.trail_xy.clear()
        self._rebuild_brain_params()
        self._rebuild_channels()

        if "multipliers" in loaded_json:
            self._osc_ctrl.apply_multipliers_from_json(loaded_json["multipliers"])

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
            self._sim_ctrl.mujoco.rebuild()
        if hasattr(self, '_arena'):
            self._arena.setup_sensors(self.circuit.sensors, self._osc_ctrl.channel_colors)

    def _toggle_osc_layer(self, name):
        self._osc_ctrl.toggle_layer(name)
        self._rebuild_channels()
        if self._net_viz:
            self._net_viz.renderer.redraw_nodes()

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
            self._update_net_viz_title()

    # ── Brain params UI ───────────────────────────────────────────────────────

    def _rebuild_brain_params(self):
        while self._brain_params_layout.count():
            item = self._brain_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not self.brain:
            return
        self._refresh_brain_mode_ui()
        is_data_brain = isinstance(self.brain, DataBrain)
        for k, p_obj in self.brain.get_param_metadata().items():
            if is_data_brain and k in ('network_project', 'network_file'):
                continue   # shown in the Network panel at the top of the Brain tab
            cv = getattr(self.brain, k)
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
