"""
sim_app_world.py — World tab: draw modes (gradients, objects, walls, sky, move), world
setup / clear, world save / load and textures.
"""

import os
import glob

from PySide6.QtWidgets import QColorDialog, QMessageBox, QLabel
from PySide6.QtGui import QColor

from sim_constants import C, GRADIENT_COLORS, OBJECT_COLORS
from world_serializer import save_world_file, load_world_file
import data_paths


class _WorldMixin:

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

    # ── World setup ──────────────────────────────────────────────────────────

    def _setup_world(self, rebuild=True):
        self._arena.setup(self.sim_cfg, self.world)
        self._sim_ctrl.sync_robot_markers()
        if rebuild:
            self._sim_ctrl.mujoco.rebuild()
            if self._sim_ctrl.mujoco.view_3d and not self._sim_ctrl.running:
                self._sim_ctrl.mujoco.render_overhead()
        elif self._sim_ctrl.mujoco.view_3d and not self._sim_ctrl.running:
            # Called mid-drag (e.g. dragging the robot) — a full rebuild is too
            # expensive per mouse-move, but the overhead render must still track
            # the live position, or the MuJoCo image shows a stale robot until drop.
            self._sim_ctrl.mujoco.reposition_robots(self._sim_ctrl.registry.agents)

    def _clear_world(self):
        n_p, n_o = len(self.world.patches), len(self.world.objects)
        if not (n_p or n_o):
            return
        reply = QMessageBox.question(
            self, "Clear World",
            f"Remove all {n_p} gradient patch(es) and {n_o} object(s) from the arena?\n"
            "(Walls stay.) This can't be undone — save the world first if you need it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.world.patches = []
        self.world.objects = []
        self._setup_world()

    # ── World directory (mirrors _SessionMixin's session directory pattern) ────

    def _current_world_dir(self):
        """Folder of the selected entry: 'builtin:Tutorials' → the app's
        worlds/Tutorials, 'X' → the user's worlds/X, '' → the user's worlds/
        (data_paths)."""
        return str(data_paths.folder_for('worlds', self._world_dir_combo.currentData() or ""))

    def _refresh_world_dirs(self, select=None):
        """Root (My files / 🔒 Simulator) and its folders; *select* is a folder
        key ('builtin:Tutorials', 'Mine', '' = your top level)."""
        if select is None:
            select = self._world_dir_combo.currentData() if self._world_dir_combo.count() else ""
        builtin = self._fill_folder_picker(self._world_root_combo, self._world_dir_combo,
                                           'worlds', select or "")
        self._btn_new_world_dir.setEnabled(not builtin or data_paths.builtin_writable())
        self._refresh_world_list()

    def _new_world_dir(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New Directory", "Directory name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        key = data_paths.BUILTIN + name if self._world_root_combo.currentData() == 'builtin' else name
        data_paths.folder_for('worlds', key).mkdir(parents=True, exist_ok=True)
        self._refresh_world_dirs(select=key)

    # ── World save/load ──────────────────────────────────────────────────────

    def _save_world(self):
        """Ask for a directory (as for saving a session) and a name, then write.
        Folders as data_paths.save_folders: the user's, plus the simulator's
        own in the development copy."""
        from PySide6.QtWidgets import (
            QDialog, QFormLayout, QComboBox, QLineEdit, QDialogButtonBox, QPushButton,
            QHBoxLayout, QInputDialog,
        )

        entries, current_idx = data_paths.save_folders(
            'worlds', self._world_dir_combo.currentData() or "", '(My Worlds)')

        dlg = QDialog(self)
        dlg.setWindowTitle('Save World')
        form = QFormLayout(dlg)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        note = QLabel()
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{C['muted']}; font-style:italic;")
        form.addRow(note)

        dir_row = QHBoxLayout()
        dir_combo = QComboBox()
        for label, key in entries:
            dir_combo.addItem(label, key)
        dir_combo.setCurrentIndex(current_idx)
        btn_new_dir = QPushButton('+')
        btn_new_dir.setFixedWidth(24)
        btn_new_dir.setToolTip('Create a new folder in your files')
        def _new_dir():
            name_d, ok = QInputDialog.getText(dlg, 'New directory', 'Directory name:')
            if not ok or not name_d.strip():
                return
            name_d = name_d.strip()
            data_paths.user_path('worlds', name_d).mkdir(parents=True, exist_ok=True)
            if dir_combo.findData(name_d) < 0:
                dir_combo.addItem(name_d, name_d)
            dir_combo.setCurrentIndex(dir_combo.findData(name_d))
        btn_new_dir.clicked.connect(_new_dir)
        dir_row.addWidget(dir_combo, 1)
        dir_row.addWidget(btn_new_dir)
        form.addRow('Directory:', dir_row)

        current_name = self._world_combo.currentText()
        default_name = current_name[:-5] if current_name.endswith('.json') else current_name
        name_edit = QLineEdit(default_name)
        form.addRow('Name:', name_edit)

        preview = QLabel()
        preview.setWordWrap(True)
        preview.setStyleSheet(f"color:{C['muted']};")
        form.addRow('Saves to:', preview)

        def _update_preview():
            chosen = dir_combo.currentData() or ""
            raw    = name_edit.text().strip()
            clean  = "".join(x for x in raw if x.isalnum() or x in "._- ") or "world"
            preview.setText(str(data_paths.folder_for('worlds', chosen) / f"{clean}.json"))
            note.setText(data_paths.save_note(chosen))
        dir_combo.currentIndexChanged.connect(_update_preview)
        name_edit.textChanged.connect(_update_preview)
        _update_preview()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        chosen_dir = dir_combo.currentData() or ""
        raw_name   = name_edit.text().strip()
        clean_name = "".join(x for x in raw_name if x.isalnum() or x in "._- ") or "world"
        world_dir  = data_paths.folder_for('worlds', chosen_dir)
        world_dir.mkdir(parents=True, exist_ok=True)
        path = os.path.join(str(world_dir), f"{clean_name}.json")
        save_world_file(self.world, path)
        self._refresh_world_dirs(select=chosen_dir)
        self._world_combo.setCurrentText(f"{clean_name}.json")
        print(f"World saved to {path}")

    def _refresh_world_list(self):
        world_dir = self._current_world_dir()
        files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(world_dir, "*.json")))
        current = self._world_combo.currentText()
        self._world_combo.blockSignals(True)
        self._world_combo.clear()
        self._world_combo.addItems(files)
        if current in files:
            self._world_combo.setCurrentText(current)
        self._world_combo.blockSignals(False)

    def _load_world(self, name):
        if not name:
            return
        path = os.path.join(self._current_world_dir(), name)
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
