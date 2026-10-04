"""
sim_app_world.py — World tab: draw modes (gradients, objects, walls, sky, move), world
setup / clear, world save / load and textures.
"""

import os

from PySide6.QtWidgets import QColorDialog, QMessageBox
from PySide6.QtGui import QColor

from sim_constants import C, GRADIENT_COLORS, OBJECT_COLORS
from world_serializer import discover_worlds, save_world_file, load_world_file
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
        path = data_paths.user_path('worlds', name)   # always the user's folder
        save_world_file(self.world, str(path))
        self._refresh_world_list()
        self._world_combo.setCurrentText(name)
        print(f"World saved to {path}")

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
        path = data_paths.resolve('worlds', name)
        if path is None:
            return
        load_world_file(str(path), self.world)
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
