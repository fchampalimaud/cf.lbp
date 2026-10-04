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
from session_io import save_session, load_session, brain_from_file
from session_loader import apply_session_world, resolve_mounted_patches
from tasks import discover_tasks, load_task
from rigid_body import world_poses as _rb_world_poses
import data_paths

_VIDEO_CAPTURE_FPS = 20   # frames are grabbed at this real-time rate regardless of Speed


class _SessionMixin:

    # ── Panel builders ────────────────────────────────────────────────────────

    def _build_session_group(self, panel_vl=None):
        gb, vl = self._panel.add_group("Sessions", panel_vl)

        # Where the user's own files live (sessions, networks, worlds, …);
        # 🔒 entries in the pickers ship with the app and are read-only.
        files_row = QWidget(); fl = QHBoxLayout(files_row); fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(QLabel("My files:"))
        from sim_widgets import ElidedLabel
        self._user_dir_label = ElidedLabel()      # never widens the panel
        self._user_dir_label.setStyleSheet(f"color:{C['muted']};")
        fl.addWidget(self._user_dir_label, 1)
        for text, tip, slot in (("📂", "Open your folder in the file explorer", self._open_user_folder),
                                ("…", "Use another folder for your files", self._change_user_folder),
                                ("↺", "Back to the default folder", self._reset_user_folder)):
            b = self._make_btn(text, C['muted'])
            b.setFixedWidth(24)                   # same as the '+' buttons
            b.setToolTip(tip)
            b.clicked.connect(slot)
            fl.addWidget(b)
        vl.addWidget(files_row)
        self._show_user_dir()
        self._build_update_row(vl)

        dir_row = QWidget(); dl = QHBoxLayout(dir_row); dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(QLabel("Directory:"))
        self._session_root_combo = self._make_root_combo()
        dl.addWidget(self._session_root_combo)
        self._session_dir_combo = QComboBox()
        dl.addWidget(self._session_dir_combo, 1)
        self._btn_new_session_dir = self._make_btn("+", C['muted'])
        self._btn_new_session_dir.setFixedWidth(24)
        self._btn_new_session_dir.setToolTip("Create a new folder in your files")
        self._btn_new_session_dir.clicked.connect(self._new_session_dir)
        dl.addWidget(self._btn_new_session_dir)
        vl.addWidget(dir_row)
        self._session_root_combo.currentIndexChanged.connect(
            lambda _i: self._refresh_session_dirs(
                select=self._first_folder_key('configs', self._session_root_combo.currentData())))
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
        ol.addStretch()
        vl.addWidget(opts_row)
        opts_row = QWidget(); ol = QHBoxLayout(opts_row); ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(QLabel("Speed:"))
        self._video_speed_spin = QDoubleSpinBox()
        self._video_speed_spin.setRange(0.1, 20.0)
        self._video_speed_spin.setSingleStep(0.5)
        self._video_speed_spin.setValue(1.0)
        self._video_speed_spin.setSuffix("×")
        ol.addWidget(self._video_speed_spin)
        ol.addStretch()
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
            self._sim_ctrl.sim.task = None
            return
        try:
            task = load_task(name)
            task.setup(self.world, self.sim_cfg)
            self._sim_ctrl.sim.task = task
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
        log_dir = str(data_paths.user_path('logs'))
        viewer = TrajectoryViewer.for_latest(log_dir=log_dir, parent=self)
        if viewer is None:
            QMessageBox.information(self, "No logs",
                                    f"No trajectory files found in {log_dir}.\n"
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

        out_dir = str(data_paths.user_path('logs', 'videos'))
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

    # ── Updates (public copies only: release.json, see updater.py) ──────────

    def _build_update_row(self, vl):
        import updater
        from app_version import get_app_version
        if updater.release_info() is None:
            return               # development copy: updates come from git
        row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(QLabel(f"Simulator v{get_app_version()}"))
        hl.addStretch()
        self._update_btn = self._make_btn("Check for updates", C['muted'])
        self._update_btn.setToolTip("Update from the public repository — your files are never touched")
        self._update_btn.clicked.connect(self._on_update_clicked)
        hl.addWidget(self._update_btn)
        vl.addWidget(row)
        # Quiet check at start-up, off the UI thread; silent when offline.
        import threading
        self._latest_version = None
        self._update_check_done = False
        def _check():
            self._latest_version = updater.latest_version()
            self._update_check_done = True
        threading.Thread(target=_check, daemon=True).start()
        self._update_poll = QTimer(self)
        self._update_poll.timeout.connect(self._on_update_check_done)
        self._update_poll.start(500)

    def _on_update_check_done(self):
        if not self._update_check_done:
            return
        self._update_poll.stop()
        import updater
        from app_version import get_app_version
        latest = self._latest_version
        if latest and updater.is_newer(latest, get_app_version()):
            self._update_btn.setText(f"Update to v{latest}")
            self._status_bar.showMessage(f"Simulator v{latest} is available (Session tab)", 10000)

    def _on_update_clicked(self):
        import sys
        import updater
        from app_version import get_app_version
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt, QProcess
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            latest = updater.latest_version()
        finally:
            QApplication.restoreOverrideCursor()
        if not latest:
            QMessageBox.information(self, "Update", "Could not reach the public repository.")
            return
        if not updater.is_newer(latest, get_app_version()):
            QMessageBox.information(self, "Update", f"You have the latest version (v{get_app_version()}).")
            return
        if QMessageBox.question(
                self, "Update",
                f"Update from v{get_app_version()} to v{latest}?\n\n"
                f"Built-in sessions and networks are replaced; your files in\n"
                f"{data_paths.user_dir()}\nare not touched.") != QMessageBox.StandardButton.Yes:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            new_version = updater.apply_update()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Update failed", f"{e}\n\nNothing was changed.")
            return
        QApplication.restoreOverrideCursor()
        if QMessageBox.question(self, "Update",
                                f"Updated to v{new_version}. Restart now?") == QMessageBox.StandardButton.Yes:
            QProcess.startDetached(sys.executable, sys.argv)
            QApplication.quit()

    # ── The user's folder ────────────────────────────────────────────────────

    def _show_user_dir(self):
        path = str(data_paths.user_dir())
        self._user_dir_label.setText(path)
        self._user_dir_label.setToolTip(
            f"{path}\nYour sessions, networks, worlds, motifs, brains and logs live here; "
            "updates never touch it.\n🔒 entries ship with the simulator and are read-only.")

    def _open_user_folder(self):
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(data_paths.user_dir())))

    def _change_user_folder(self):
        from PySide6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Folder for your files", str(data_paths.user_dir()))
        if path:
            self._use_user_folder(path)

    def _reset_user_folder(self):
        self._use_user_folder(None)

    def _use_user_folder(self, path):
        data_paths.set_user_dir(path)
        self._show_user_dir()
        self._refresh_session_dirs(select="")
        self._status_bar.showMessage(
            f"Your files: {data_paths.user_dir()} — brains in it are picked up after a restart", 8000)

    # ── Session directory ────────────────────────────────────────────────────

    def _current_session_dir(self):
        """Folder of the selected entry: 'builtin:Tutorials' → the app's
        configs/Tutorials (read-only), 'X' → the user's configs/X, '' → the
        user's configs/ (data_paths)."""
        return str(data_paths.folder_for('configs', self._session_dir_combo.currentData() or ""))

    def _refresh_session_dirs(self, select=None):
        """Root (My files / 🔒 Simulator) and its folders; *select* is a folder
        key ('builtin:Tutorials', 'Mine', '' = your top level)."""
        if select is None:
            select = self._session_dir_combo.currentData() if self._session_dir_combo.count() else ""
        builtin = self._fill_folder_picker(self._session_root_combo, self._session_dir_combo,
                                           'configs', select or "")
        self._btn_new_session_dir.setEnabled(not builtin)   # the simulator's folders are read-only
        self._refresh_session_list()

    def _new_session_dir(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New Directory", "Directory name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        data_paths.user_path('configs', name).mkdir(parents=True, exist_ok=True)
        self._refresh_session_dirs(select=name)

    # ── Session save / load ───────────────────────────────────────────────────

    def _patches_for_save(self):
        """Translate world.patches' mounted_on (a live RobotAgent.id) into a
        stable positional index into self._sim_ctrl.registry.agents. Agent ids are
        never reused (see AgentRegistry._next_agent_id) and are re-minted
        fresh for every agent but the very first on session reload (see
        _load_session_agents), so a saved raw id would silently go stale —
        position survives because group/member order is reproduced exactly."""
        agent_pos = {a.id: i for i, a in enumerate(self._sim_ctrl.registry.agents)}
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
        resolve_mounted_patches(self.world, self._sim_ctrl.registry.agents)

    def _save_session(self):
        if not self.brain:
            print("SAVE FAILED: No brain loaded.")
            return
        raw_name    = self._session_name.text().strip()
        clean_name  = "".join(x for x in raw_name if x.isalnum() or x in "._- ") or "autosave"
        key         = self._session_dir_combo.currentData() or ""
        redirected  = key.startswith(data_paths.BUILTIN)
        if redirected:   # the simulator's folders are read-only: save to My files / <same name>
            key = key[len(data_paths.BUILTIN):]
        session_dir = data_paths.folder_for('configs', key)
        session_dir.mkdir(parents=True, exist_ok=True)
        path        = os.path.join(str(session_dir), f"{clean_name}.json")
        if self._write_session(path):
            print(f"SUCCESS: Session saved to {path}")
            note = " (the simulator's sessions are read-only — saved to My files)" if redirected else ""
            self._status_bar.showMessage(f"✓ Session saved: {path}{note}", 8000)
            if redirected:
                self._refresh_session_dirs(select=key)
            self._refresh_session_list()
            self._session_combo.setCurrentText(f"{clean_name}.json")
        else:
            self._status_bar.showMessage(f"✕ Session could not be saved to {path} (see console)", 8000)

    # Saved on close, loaded at start-up: the app reopens as it was left.
    # Lives in the user's configs/ folder (data_paths).
    LATEST_SESSION_NAME = "latest_session.json"

    def _save_latest_session(self):
        if self.brain:
            self._write_session(str(data_paths.user_path('configs', self.LATEST_SESSION_NAME)))

    def _load_latest_session(self):
        """Load configs/latest_session.json; False if there is none."""
        if not data_paths.user_path('configs', self.LATEST_SESSION_NAME).exists():
            return False
        self._refresh_session_dirs(select="")
        self._session_combo.setCurrentText(self.LATEST_SESSION_NAME)
        self._load_session()
        return True

    def _write_session(self, path):
        """Write the current state as a session file. True on success."""
        try:
            groups_data = []
            for g in self._sim_ctrl.registry.groups_ordered():
                b_params = {}
                if g.member_ids:
                    agent = self._sim_ctrl.registry.agent_by_id(g.member_ids[0])
                    if agent is not None and agent.brain:
                        b_params = {k: getattr(agent.brain, k)
                                    for k in agent.brain.get_param_metadata()}
                groups_data.append({
                    'module': g.module, 'name': g.name, 'color': g.color,
                    'n': len(g.member_ids), 'brain_params': b_params,
                })
            net_cfg = None
            if self._sim_ctrl.network.host is not None:
                net_cfg = {'mode': 'host', 'host_port': self._sim_ctrl.network.host.port}
            elif self._sim_ctrl.network.client is not None:
                net_cfg = {
                    'mode':        'client',
                    'client_host': self._net_cli_host_edit.text().strip(),
                    'client_port': self._net_cli_port_spin.value(),
                    'local_port':  self._net_cli_lport_spin.value(),
                }
            save_session(
                path,
                module_name  = (self._selected_group().module if self._selected_group() else None)
                               or self._brain_combo.currentText(),
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
            return True
        except Exception as e:
            print(f"SAVE ERROR: {e}")
            return False

    def _load_session(self):
        selected = self._session_combo.currentText()
        if not selected:
            return
        path = os.path.join(self._current_session_dir(), selected)
        if not os.path.exists(path):
            return
        try:
            d = load_session(path)
            # Model half (shared with headless runs), then sync the widgets to it.
            apply_session_world(d, self.world, self.sim_cfg)

            p_meta = self.sim_cfg.get_param_metadata()
            for k in d.get("sim_params", {}):
                if k in self._phys_widgets and k in p_meta:
                    v = getattr(self.sim_cfg, k)
                    p = p_meta[k]
                    w = self._phys_widgets[k]
                    slider = w[0] if isinstance(w, tuple) else w
                    is_int = isinstance(v, int) and p.step >= 1
                    slider.setValue(int(v) if is_int else int((v - p.min) / p.step))
                    if isinstance(w, tuple):
                        w[1].setText(f"{v:.4g}")

            if "arena_round" in d:
                self._arena_round_rb.setChecked(self.world.arena_round)
                self._arena_square_rb.setChecked(not self.world.arena_round)

            self._hide_stim_btn.setChecked(not self.sim_cfg.toggle_stim)
            self._fixate_btn.setChecked(bool(self.sim_cfg.fixate_robot))
            if hasattr(self, '_floor_texture_combo'):
                self._floor_texture_combo.blockSignals(True)
                self._floor_texture_combo.setCurrentText(self.world.floor_texture or "(default)")
                self._floor_texture_combo.blockSignals(False)
            self._sky_cb.setChecked(bool(self.world.sky.get("enabled", False)))

            saved_agents = d.get("agents")
            if saved_agents:
                self._load_session_agents(saved_agents)
            else:
                mod_name, b_params = brain_from_file(d)
                if mod_name in self.brain_files:
                    self.load_brain(mod_name, external_params=b_params)
            self._resolve_mounted_patches()

            self._osc_ctrl.apply_multipliers_from_json(d.get("plot_multipliers", {}))

            if "speed_mult" in d:
                self._sim_ctrl.speed_mult = int(d["speed_mult"])
                self._speed_spin.setValue(self._sim_ctrl.speed_mult)

            if "trail_length" in d:
                n = max(10, int(d["trail_length"]))
                self._trail_len_spin.setValue(n)
                for agent in self._sim_ctrl.registry.agents:
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
        while len(self._sim_ctrl.registry.agents) > 1:
            last = len(self._sim_ctrl.registry.agents) - 1
            last_id = self._sim_ctrl.registry.agents[last].id
            self._sim_ctrl.remove_agent(last_id)
        # Only agent 0 (and its group) survives the teardown above; discard any
        # other lingering groups and start fresh from a single "Group 1".
        agent0_id = self._sim_ctrl.registry.agents[0].id
        self._sim_ctrl.registry.clear_groups()
        group0_id = self._sim_ctrl.registry.create_group(module=None, color='#4a7fcb',
                                                 name='Group 1', first_agent_id=agent0_id)
        for i, sg in enumerate(saved_agents):
            mod, b_params = brain_from_file(sg)   # code brain or network
            color = sg.get('color', _AGENT_COLORS[i % len(_AGENT_COLORS)])
            name  = sg.get('name', f'Group {i + 1}')
            n     = max(1, int(sg.get('n', 1)))
            if i == 0:
                group0 = self._sim_ctrl.registry.get_group(group0_id)
                group0.module, group0.color, group0.name = (mod or None), color, name
                self._sim_ctrl.registry.agents[0].color = color
                self._sim_ctrl.sync_robot_items()
                if mod and mod in self.brain_files:
                    self._sim_ctrl.registry.select_agent(agent0_id)
                    self.load_brain(mod, external_params=b_params)
                for _ in range(n - 1):
                    self._add_agent_to_group(0)
            else:
                self._add_agent(color=color, name=name)
                groups = self._sim_ctrl.registry.groups_ordered()
                g_idx = len(groups) - 1
                group = groups[g_idx]
                group.module = mod or None
                first_id = group.member_ids[0]
                if mod and mod in self.brain_files:
                    prev_sel_id = self._sim_ctrl.registry.selected_id
                    self._sim_ctrl.registry.select_agent(first_id)
                    self.load_brain(mod, external_params=b_params)
                    self._sim_ctrl.registry.select_agent(prev_sel_id)
                for _ in range(n - 1):
                    self._add_agent_to_group(g_idx)
        self._sim_ctrl.registry.select_agent(agent0_id)
        self._refresh_agent_list()

    def _refresh_session_list(self):
        session_dir = self._current_session_dir()
        files   = sorted(glob.glob(os.path.join(session_dir, "*.json")), key=os.path.getmtime, reverse=True)
        recent  = [os.path.basename(f) for f in files]
        self._session_combo.blockSignals(True)
        self._session_combo.clear()
        self._session_combo.addItems(recent)
        self._session_combo.blockSignals(False)
