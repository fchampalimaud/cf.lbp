import os
import glob
import datetime
from collections import deque
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit, QMessageBox,
    QCheckBox, QDoubleSpinBox,
)
from PySide6.QtCore import QTimer

from sim_constants import C
from session_io import save_session, load_session
from tasks import discover_tasks, load_task
from rigid_body import world_poses as _rb_world_poses

_VIDEO_CAPTURE_FPS = 20   # frames are grabbed at this real-time rate regardless of Speed


class _SessionMixin:

    # ── Panel builders ────────────────────────────────────────────────────────

    def _build_session_group(self, panel_vl=None):
        gb, vl = self._panel.add_group("Sessions", panel_vl)

        dir_row = QWidget(); dl = QHBoxLayout(dir_row); dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(QLabel("Directory:"))
        self._session_dir_combo = QComboBox()
        dl.addWidget(self._session_dir_combo, 1)
        btn_new_dir = self._make_btn("+", C['muted'])
        btn_new_dir.setFixedWidth(24)
        btn_new_dir.setToolTip("Create new subdirectory")
        btn_new_dir.clicked.connect(self._new_session_dir)
        dl.addWidget(btn_new_dir)
        vl.addWidget(dir_row)
        self._session_dir_combo.currentIndexChanged.connect(self._refresh_session_list)

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

        viz_row = QWidget(); vzl = QHBoxLayout(viz_row); vzl.setContentsMargins(0, 0, 0, 0)
        btn_viz = self._make_btn("Visualize trajectories", C['primary'])
        btn_viz.clicked.connect(self._open_trajectory_viewer)
        vzl.addWidget(btn_viz)
        vl.addWidget(viz_row)

    def _build_video_group(self, panel_vl=None):
        gb, vl = self._panel.add_group("Video", panel_vl)
        row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(QLabel("Name:"))
        self._video_name = QLineEdit("trial1")
        hl.addWidget(self._video_name)
        self._btn_vid_start = self._make_btn("● Record", C['danger'])
        self._btn_vid_stop  = self._make_btn("■ Stop",   C['muted'])
        self._btn_vid_stop.setEnabled(False)
        self._btn_vid_start.clicked.connect(self._video_start)
        self._btn_vid_stop.clicked.connect(self._video_stop)
        hl.addWidget(self._btn_vid_start)
        hl.addWidget(self._btn_vid_stop)
        vl.addWidget(row)

        opts_row = QWidget(); ol = QHBoxLayout(opts_row); ol.setContentsMargins(0, 0, 0, 0)
        self._video_timestamp_cb = QCheckBox("Add timestamp")
        self._video_timestamp_cb.setChecked(True)
        ol.addWidget(self._video_timestamp_cb)
        self._video_auto_cb = QCheckBox("Auto (Run/Stop)")
        ol.addWidget(self._video_auto_cb)
        ol.addWidget(QLabel("Speed:"))
        self._video_speed_spin = QDoubleSpinBox()
        self._video_speed_spin.setRange(0.1, 20.0)
        self._video_speed_spin.setSingleStep(0.5)
        self._video_speed_spin.setValue(1.0)
        self._video_speed_spin.setSuffix("×")
        ol.addWidget(self._video_speed_spin)
        vl.addWidget(opts_row)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _update_record_status(self):
        parts = []
        if self._logger.running:
            parts.append("CSV")
        if self._video_recorders:
            parts.append("Video")
        self._record_label.setText("●  REC: " + " + ".join(parts) if parts else "")

    # ── Task ──────────────────────────────────────────────────────────────────

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
        self._update_record_status()

    def _logger_stop(self):
        self._logger.stop()
        self._btn_log_start.setEnabled(True)
        self._btn_log_stop.setEnabled(False)
        self._update_record_status()

    def _open_trajectory_viewer(self):
        from trajectory_viz import TrajectoryWindow as TrajectoryViewer
        viewer = TrajectoryViewer.for_latest(log_dir='logs', parent=self)
        if viewer is None:
            QMessageBox.information(self, "No logs",
                                    "No trajectory files found in logs/.\n"
                                    "Record a session first with ● Record.")
            return
        viewer.show()
        self._traj_viewer = viewer

    # ── Video ─────────────────────────────────────────────────────────────────

    def _video_start(self):
        try:
            from video_recorder import VideoRecorder
        except ImportError:
            QMessageBox.critical(self, "Missing dependency",
                                  "Video capture requires 'imageio' and 'imageio-ffmpeg'.\n"
                                  "Install with: pip install imageio imageio-ffmpeg")
            return

        raw_name   = self._video_name.text().strip()
        clean_name = "".join(x for x in raw_name if x.isalnum() or x in "._- ") or "video"
        stamp = ""
        if self._video_timestamp_cb.isChecked():
            stamp = "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        out_dir = os.path.join("logs", "videos")
        os.makedirs(out_dir, exist_ok=True)

        paths = [os.path.join(out_dir, f"{clean_name}_arena{stamp}.mp4")]
        if self._net_viz is not None and self._net_viz.isVisible():
            paths.append(os.path.join(out_dir, f"{clean_name}_network{stamp}.mp4"))

        # Frames are always grabbed at _VIDEO_CAPTURE_FPS; Speed instead changes
        # the fps stored in the file, so playback runs faster/slower without
        # dropping or duplicating any captured frame.
        out_fps = _VIDEO_CAPTURE_FPS * self._video_speed_spin.value()
        try:
            self._video_recorders = [VideoRecorder(self._arena, paths[0], fps=out_fps)]
            if len(paths) > 1:
                self._video_recorders.append(VideoRecorder(self._net_viz, paths[1], fps=out_fps))
        except Exception as e:
            QMessageBox.critical(self, "Video error", str(e))
            return

        if not hasattr(self, '_video_timer'):
            self._video_timer = QTimer(self)
            self._video_timer.timeout.connect(self._video_capture_tick)
        self._video_timer.start(int(1000 / _VIDEO_CAPTURE_FPS))

        self._btn_vid_start.setEnabled(False)
        self._btn_vid_stop.setEnabled(True)
        self._update_record_status()
        print("[Video] Recording → " + ", ".join(paths))

    def _video_capture_tick(self):
        for rec in self._video_recorders:
            rec.capture_frame()

    def _video_stop(self):
        self._video_timer.stop()
        n = 0
        paths = [rec.path for rec in self._video_recorders]
        for rec in self._video_recorders:
            rec.close()
            n = max(n, rec.frame_count)
        self._video_recorders = []
        self._btn_vid_start.setEnabled(True)
        self._btn_vid_stop.setEnabled(False)
        self._update_record_status()
        print(f"[Video] Saved {n} frames → " + ", ".join(paths))

    # ── Session directory ────────────────────────────────────────────────────

    def _current_session_dir(self):
        directory = self._session_dir_combo.currentData()
        return os.path.join("configs", directory) if directory else "configs"

    def _refresh_session_dirs(self, select=None):
        os.makedirs("configs", exist_ok=True)
        subdirs = [""] + sorted(
            e.name for e in os.scandir("configs")
            if e.is_dir() and not e.name.startswith(".")
        )
        if select is None:
            select = self._session_dir_combo.currentData() if self._session_dir_combo.count() else ""
        self._session_dir_combo.blockSignals(True)
        self._session_dir_combo.clear()
        for d in subdirs:
            self._session_dir_combo.addItem(d if d else "(configs root)", d)
        idx = self._session_dir_combo.findData(select)
        self._session_dir_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._session_dir_combo.blockSignals(False)
        self._refresh_session_list()

    def _new_session_dir(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New Directory", "Directory name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        os.makedirs(os.path.join("configs", name), exist_ok=True)
        self._refresh_session_dirs(select=name)

    # ── Session save / load ───────────────────────────────────────────────────

    def _patches_for_save(self):
        """Translate world.patches' mounted_on (a live RobotAgent.id) into a
        stable positional index into self._sim_ctrl._agents. Agent ids are
        never reused (see AgentRegistry._next_agent_id) and are re-minted
        fresh for every agent but the very first on session reload (see
        _load_session_agents), so a saved raw id would silently go stale —
        position survives because group/member order is reproduced exactly."""
        agent_pos = {a.id: i for i, a in enumerate(self._sim_ctrl._agents)}
        out = []
        for p in self.world.patches:
            p = dict(p)
            mounted_on = p.get('mounted_on')
            if mounted_on is not None:
                if mounted_on in agent_pos:
                    p['mounted_on'] = agent_pos[mounted_on]
                else:
                    p.pop('mounted_on', None)
            out.append(p)
        return out

    def _resolve_mounted_patches(self):
        """Reverse of _patches_for_save: once agents are (re)created for a
        loaded session, translate each patch's mounted_on positional index
        back into that agent's real (freshly-minted) id."""
        agents = self._sim_ctrl._agents
        for p in self.world.patches:
            idx = p.get('mounted_on')
            if idx is None:
                continue
            if isinstance(idx, int) and 0 <= idx < len(agents):
                p['mounted_on'] = agents[idx].id
            else:
                p.pop('mounted_on', None)

    def _save_session(self):
        if not self.brain:
            print("SAVE FAILED: No brain loaded.")
            return
        raw_name    = self._session_name.text().strip()
        clean_name  = "".join(x for x in raw_name if x.isalnum() or x in "._- ") or "autosave"
        session_dir = self._current_session_dir()
        os.makedirs(session_dir, exist_ok=True)
        path        = os.path.join(session_dir, f"{clean_name}.json")
        try:
            groups_data = []
            for g in self._sim_ctrl.groups_ordered():
                b_params = {}
                if g.member_ids:
                    agent = self._sim_ctrl._agent_by_id(g.member_ids[0])
                    if agent is not None and agent.brain:
                        b_params = {k: getattr(agent.brain, k)
                                    for k in agent.brain.get_param_metadata()}
                groups_data.append({
                    'module': g.module, 'name': g.name, 'color': g.color,
                    'n': len(g.member_ids), 'brain_params': b_params,
                })
            net_cfg = None
            if self._sim_ctrl._net_host is not None:
                net_cfg = {'mode': 'host', 'host_port': self._sim_ctrl._net_host.port}
            elif self._sim_ctrl._net_client is not None:
                net_cfg = {
                    'mode':        'client',
                    'client_host': self._net_cli_host_edit.text().strip(),
                    'client_port': self._net_cli_port_spin.value(),
                    'local_port':  self._net_cli_lport_spin.value(),
                }
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
                net_cfg      = net_cfg,
                patches      = self._patches_for_save(),
            )
            print(f"SUCCESS: Session saved to {path}")
            self._refresh_session_list()
        except Exception as e:
            print(f"SAVE ERROR: {e}")

    def _load_session(self):
        selected = self._session_combo.currentText()
        if not selected:
            return
        path = os.path.join(self._current_session_dir(), selected)
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
            self.world.patches       = d.get("patches", [])
            self.world.objects       = d.get("objects", [])
            self.world.walls         = d.get("walls", [])
            self.world.floor_texture = d.get("floor_texture")
            if hasattr(self, '_floor_texture_combo'):
                self._floor_texture_combo.blockSignals(True)
                self._floor_texture_combo.setCurrentText(self.world.floor_texture or "(default)")
                self._floor_texture_combo.blockSignals(False)
            sky = d.get("sky", {"enabled": False, "angle": 0.0})
            self.world.sky = sky
            self._sky_cb.setChecked(bool(sky.get("enabled", False)))

            saved_agents = d.get("agents")
            if saved_agents:
                self._load_session_agents(saved_agents)
            else:
                mod_name = d.get("module_name")
                if mod_name in self.brain_files:
                    self._brain_combo.blockSignals(True)
                    self._brain_combo.setCurrentText(mod_name)
                    self._brain_combo.blockSignals(False)
                    self.load_brain(mod_name, external_params=d.get("brain_params"))
            self._resolve_mounted_patches()

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

            net = d.get('net')
            if net:
                mode = net.get('mode')
                if mode == 'host':
                    self._net_host_port_spin.setValue(net.get('host_port', 9001))
                    self._net_host_rb.setChecked(True)
                    self._sim_ctrl.enable_network_host(net.get('host_port', 9001))
                elif mode == 'client':
                    self._net_cli_host_edit.setText(net.get('client_host', ''))
                    self._net_cli_port_spin.setValue(net.get('client_port', 9001))
                    self._net_cli_lport_spin.setValue(net.get('local_port', 9002))
                    self._net_cli_rb.setChecked(True)
                else:
                    self._net_off_rb.setChecked(True)

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
        while len(self._sim_ctrl._agents) > 1:
            last = len(self._sim_ctrl._agents) - 1
            last_id = self._sim_ctrl._agents[last].id
            self._arena.remove_robot_item(last)
            self._sim_ctrl.remove_agent(last_id)
        # Only agent 0 (and its group) survives the teardown above; discard any
        # other lingering groups and start fresh from a single "Group 1".
        agent0_id = self._sim_ctrl._agents[0].id
        self._sim_ctrl._groups.clear()
        group0_id = self._sim_ctrl.create_group(module=None, color='#4a7fcb',
                                                 name='Group 1', first_agent_id=agent0_id)
        for i, sg in enumerate(saved_agents):
            mod   = sg.get('module_name') or ''
            color = sg.get('color', _AGENT_COLORS[i % len(_AGENT_COLORS)])
            name  = sg.get('name', f'Group {i + 1}')
            n     = max(1, int(sg.get('n', 1)))
            b_params = sg.get('brain_params') or {}
            if i == 0:
                group0 = self._sim_ctrl.get_group(group0_id)
                group0.module, group0.color, group0.name = (mod or None), color, name
                self._sim_ctrl._agents[0].color = color
                self._arena.update_robot_color(0, color)
                if mod and mod in self.brain_files:
                    self._sim_ctrl.select_agent(agent0_id)
                    self._arena.select_robot(0)
                    self.load_brain(mod, external_params=b_params)
                for _ in range(n - 1):
                    self._add_agent_to_group(0)
            else:
                self._add_agent(color=color, name=name)
                groups = self._sim_ctrl.groups_ordered()
                g_idx = len(groups) - 1
                group = groups[g_idx]
                group.module = mod or None
                first_id = group.member_ids[0]
                if mod and mod in self.brain_files:
                    prev_sel_id = self._sim_ctrl._selected_id
                    self._sim_ctrl.select_agent(first_id)
                    self._arena.select_robot(self._sim_ctrl.index_of_agent(first_id))
                    self.load_brain(mod, external_params=b_params)
                    self._sim_ctrl.select_agent(prev_sel_id)
                    self._arena.select_robot(self._sim_ctrl.index_of_agent(prev_sel_id))
                for _ in range(n - 1):
                    self._add_agent_to_group(g_idx)
        self._sim_ctrl.select_agent(agent0_id)
        self._arena.select_robot(0)
        self._refresh_agent_list()

    def _refresh_session_list(self):
        session_dir = self._current_session_dir()
        files   = sorted(glob.glob(os.path.join(session_dir, "*.json")), key=os.path.getmtime, reverse=True)
        recent  = [os.path.basename(f) for f in files]
        self._session_combo.blockSignals(True)
        self._session_combo.clear()
        self._session_combo.addItems(recent)
        self._session_combo.blockSignals(False)
