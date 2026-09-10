"""
network_viz_editing.py — interaction/editing for the network visualizer
(everything except the modal dialogs, which live in network_viz_dialogs.py).

_EditingMixin holds event handlers, edit-mode/panel toggles, and every
operation that mutates the circuit model (self.gui.circuit) in response to a
user action — adding/removing layers/sensors/connections, dragging nodes
between columns, motif insert/paste, etc. It calls into network_viz_layout.py
for pure depth/position math and network_viz_render.py for the handful of
pyqtgraph item restyles that follow a click (highlight/selection), but never
creates graphics items itself.
"""

import os
import json

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QMenu, QDialog, QMessageBox, QApplication
from PySide6.QtCore import Qt, QTimer, QEvent

from sim_constants import C
from neurons import SumLayer
from side_view import SideViewWindow
from brain_serializer import _layer_from_dict

from network_viz_layout import _sensor_is_lateralized, _mirror_name
from network_viz_render import WeightEntryWidget, ActivationEntryWidget
from network_viz_dialogs import PaletteChip, WeightMatrixDialog, FilterStackDialog
from network_viz_serialization import MOTIFS_DIR


# ============================================================
# CUSTOM VIEWBOX
# ============================================================
class NetworkViewBox(pg.ViewBox):
    """Custom ViewBox that intercepts horizontal drags in edit mode to reorder nodes."""

    def __init__(self, nw):
        super().__init__()
        self._nw       = nw
        self._dragging = False

    def mouseDragEvent(self, ev, axis=None):
        nw = self._nw
        # Only the left button drags notes/nodes/connections — a right-button
        # drag (e.g. an incidental mouse move during what's meant to be a
        # right-click for a context menu) must not create a connection or
        # reposition anything.
        if nw._edit_mode and ev.button() == Qt.LeftButton:
            ev.accept()
            if ev.isStart():
                self._dragging = True
                start_pt = self.mapSceneToView(ev.buttonDownScenePos())
                # Notes draw on top of everything and are never column-snapped —
                # check them first, before any node hit-testing.
                note_hit = nw._note_at(start_pt)
                if note_hit is not None:
                    note, zone = note_hit
                    if zone == 'toggle':
                        nw._push_undo()
                        note.collapsed = not note.collapsed
                        nw._redraw_note(note)
                        self._dragging = False
                        return
                    # 'icon' (collapsed) and 'body' (expanded) both drag freely.
                    nw._push_undo()
                    nw._dragging_note = note
                    nw._selected_note = note
                    nw._selected      = None
                    nw._clear_edge_selection()
                    return
                nw._dragging_note = None
                hit = nw._node_at(start_pt)
                # Alt+drag repositions the node (between columns); a plain
                # drag always starts a connection, independent of whatever
                # is currently selected — the two are disambiguated by the
                # Alt key alone, not by selection state.
                if hit is not None and bool(ev.modifiers() & Qt.AltModifier):
                    nw._selected      = hit
                    nw._selected_note = None
                    nw._spot_pen_override.clear()
                    nw._spot_pen_override[hit] = 'selected'
                    nw._redraw_nodes()
                    nw._conn_from = None
                else:
                    nw._conn_from = hit   # connect from this node to another (None if drag started on empty canvas)
            if nw._dragging_note is not None:
                if self._dragging:
                    mouse_pt = self.mapSceneToView(ev.scenePos())
                    nw._dragging_note.x = mouse_pt.x()
                    nw._dragging_note.y = mouse_pt.y()
                    nw._redraw_note(nw._dragging_note)
                if ev.isFinish():
                    self._dragging   = False
                    nw._dragging_note = None
                return
            if self._dragging:
                mouse_pt = self.mapSceneToView(ev.scenePos())
                if nw._conn_from is not None:
                    nw._update_conn_preview(mouse_pt)
                elif nw._selected is not None:
                    snap_x = nw._get_snap_x(mouse_pt.x())
                    layer_name = nw._selected.rsplit('_', 1)[0]
                    layer = next((l for l in nw.gui.circuit.layers if l.name == layer_name), None)
                    nw._show_drag_indicator(snap_x, mouse_pt.y(), layer)
            if ev.isFinish():
                self._dragging = False
                mouse_pt = self.mapSceneToView(ev.scenePos())
                if nw._conn_from is not None:
                    nw._hide_conn_preview()
                    hit = nw._node_at(mouse_pt)
                    if hit is not None:
                        src_layer = nw._conn_from.rsplit('_', 1)[0]
                        tgt_layer = hit.rsplit('_', 1)[0]
                        if tgt_layer != src_layer:
                            nw._finish_connection(nw._conn_from, hit)
                    nw._conn_from = None
                elif nw._selected is not None:
                    snap_x = nw._get_snap_x(mouse_pt.x())
                    nw._hide_drag_indicator()
                    nw._on_node_drag_drop(snap_x, mouse_pt.y())
        else:
            super().mouseDragEvent(ev, axis)

class _EditingMixin:
    def keyPressEvent(self, ev):
        ctrl = ev.modifiers() & Qt.ControlModifier
        if ev.key() == Qt.Key_Delete:
            if self._selected_note is not None:
                self._remove_selected_note()
            elif self._selected_edge is not None:
                self._remove_selected_connection()
            elif self._selected is not None:
                lname = self._selected.rsplit('_', 1)[0]
                if any(l.name == lname for l in self.gui.circuit.layers):
                    self._remove_selected_layer()
                elif any(s.name == lname for s in self.gui.circuit.sensors):
                    self._remove_selected_sensor()
        elif ctrl and ev.key() == Qt.Key_C:
            self._copy_selection()
        elif ctrl and ev.key() == Qt.Key_V:
            self._paste_selection()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self._refresh_timer.stop()
        self.gui.notify_closed()
        super().closeEvent(ev)

    def _on_spots_clicked(self, _plot, spots, ev):
        if len(spots):
            self._on_node_clicked(spots[0].data(), ev)

    # ── Edit-mode interactions ────────────────────────────────────────────────

    def _toggle_edit(self):
        self._edit_mode = not self._edit_mode
        if self._edit_mode:
            self._btn_edit.setStyleSheet(f"background:{C['primary']};color:white;")
            self._btn_save.setEnabled(True)
            self._btn_cols.setEnabled(False)
            self._palette_bar.setVisible(True)
            if self._col_panel.isVisible():
                self._toggle_col_win()
            self._clear_edge_highlight()
            self._clear_multi_selection()
        else:
            self._btn_edit.setStyleSheet("")
            self._btn_cols.setEnabled(True)
            self._palette_bar.setVisible(False)
            self._selected = None
            self._clear_edge_selection()
            self._clear_selection_highlight()

    def _on_compact_toggled(self, state):
        self._compact_mode = bool(state)
        self.build()

    def _toggle_col_win(self):
        panel_w = self._col_panel.sizeHint().width()
        if self._col_panel.isVisible():
            self._col_panel.setVisible(False)
            self.resize(self.width() - panel_w, self.height())
        else:
            self._col_panel.setVisible(True)
            self.resize(self.width() + panel_w, self.height())

    def _toggle_notes_visible(self, hidden):
        """Master show/hide-all for notes — independent of each note's own collapsed state."""
        self._notes_visible = not hidden
        self._btn_notes.setText("Show Notes" if hidden else "Hide Notes")
        for item in self._note_items:
            item.setVisible(self._notes_visible)

    def _toggle_weight_panel(self):
        panel_w = self._weight_panel.sizeHint().width()
        if self._weight_panel.isVisible():
            self._weight_panel.setVisible(False)
            self.resize(self.width() - panel_w, self.height())
        else:
            self._weight_panel.setVisible(True)
            self.resize(self.width() + panel_w, self.height())

    def _toggle_activation_panel(self):
        panel_w = self._activation_panel.sizeHint().width()
        if self._activation_panel.isVisible():
            self._activation_panel.setVisible(False)
            self.resize(self.width() - panel_w, self.height())
        else:
            self._activation_panel.setVisible(True)
            self.resize(self.width() + panel_w, self.height())

    def _on_z_slider_changed(self, val):
        self._z_slider_label.setText(f"Depth: {val}")
        if not hasattr(self, '_z_build_timer'):
            self._z_build_timer = QTimer()
            self._z_build_timer.setSingleShot(True)
            self._z_build_timer.timeout.connect(self.build)
        self._z_build_timer.start(120)

    def _update_z_slider(self):
        c = self.gui.circuit
        all_objs = list(c.sensors) + [l for l in c.layers if l.n is not None]
        all_z    = [(getattr(o, 'z', 0) or 0) for o in all_objs]
        max_z    = max(all_z) if all_z else 0
        has_depths = max_z > 0
        self._z_slider.setVisible(has_depths)
        self._z_slider_label.setVisible(has_depths)
        if has_depths:
            self._z_slider.blockSignals(True)
            old_max = self._z_slider.maximum()
            old_val = self._z_slider.value()
            self._z_slider.setMaximum(max_z)
            # When the slider was at "show all" (old_val == old_max, including
            # first activation where both are 0), advance it to the new max so
            # all elements remain visible.
            if old_val >= old_max or old_val > max_z:
                self._z_slider.setValue(max_z)
            self._z_slider.blockSignals(False)
            self._z_cut = self._z_slider.value()
            self._z_slider_label.setText(f"Depth: {self._z_cut}")
        else:
            self._z_cut = None

    def _toggle_side_view(self):
        try:
            visible = self._side_view is not None and self._side_view.isVisible()
        except RuntimeError:
            self._side_view = None
            visible = False
        if visible:
            self._side_view.close()
        else:
            self._side_view = SideViewWindow(self)
            self._side_view.destroyed.connect(lambda: setattr(self, '_side_view', None))
            self._side_view.show()
            self._side_view.refresh()

    def _toggle_net3d(self):
        try:
            visible = self._net3d_view is not None and self._net3d_view.isVisible()
        except RuntimeError:
            self._net3d_view = None
            visible = False
        if visible:
            self._net3d_view.close()
        else:
            from net_view_3d import NetView3DWindow
            self._net3d_view = NetView3DWindow(self)
            self._net3d_view.destroyed.connect(
                lambda: setattr(self, '_net3d_view', None))
            self._net3d_view.show()
            self._net3d_view.refresh()

    def _toggle_weight_entry(self, src, tgt):
        key = (src, tgt)
        if key in self._weight_pinned:
            widget = self._weight_pinned.pop(key)
            self._weight_entries_layout.removeWidget(widget)
            widget.deleteLater()
        else:
            widget = WeightEntryWidget(src, tgt)
            idx = self._weight_entries_layout.count() - 1   # before trailing stretch
            self._weight_entries_layout.insertWidget(idx, widget)
            self._weight_pinned[key] = widget
            if not self._weight_panel.isVisible():
                self._toggle_weight_panel()
        self._update_weight_panel()

    def _toggle_activation_entry(self, name):
        if name in self._activation_pinned:
            widget = self._activation_pinned.pop(name)
            self._activation_entries_layout.removeWidget(widget)
            widget.deleteLater()
        else:
            widget = ActivationEntryWidget(name)
            idx = self._activation_entries_layout.count() - 1   # before trailing stretch
            self._activation_entries_layout.insertWidget(idx, widget)
            self._activation_pinned[name] = widget
            if not self._activation_panel.isVisible():
                self._toggle_activation_panel()
        self._update_activation_panel()

    def _on_container_hide(self, container, hidden):
        if container in self._disabled_containers:
            return  # disabled columns are always hidden
        if hidden:
            self._hidden_containers.add(container)
        else:
            self._hidden_containers.discard(container)
        if self._compact_mode:
            self.build()
        else:
            self._rebuild_group_buttons()
            self._apply_group_visibility()

    def _on_container_disable(self, container, disabled):
        all_layers = {l.name: l for l in self.gui.circuit.layers}
        container_layers = [l for nk, container in self._node_container_map.items()
                      if container == container
                      for l in [all_layers.get(nk.rsplit('_', 1)[0])]
                      if l is not None]
        container_layers_unique = list({id(l): l for l in container_layers}.values())
        if disabled:
            self._disabled_containers.add(container)
            self._hidden_containers.add(container)
            for l in container_layers_unique:
                l.muted = True
        else:
            self._disabled_containers.discard(container)
            self._hidden_containers.discard(container)
            for l in container_layers_unique:
                l.muted = False
        if self._compact_mode:
            self.build()
        else:
            self._rebuild_group_buttons()
            self._apply_group_visibility()

    def _on_show_all(self):
        all_layers = {l.name: l for l in self.gui.circuit.layers}
        for container in list(self._disabled_containers):
            container_layers = [l for nk, container in self._node_container_map.items()
                          if container == container
                          for l in [all_layers.get(nk.rsplit('_', 1)[0])]
                          if l is not None]
            for l in {id(lyr): lyr for lyr in container_layers}.values():
                l.muted = False
        self._hidden_containers.clear()
        self._disabled_containers.clear()
        self.build()

    def _on_enable_all(self):
        all_layers = {l.name: l for l in self.gui.circuit.layers}
        for container in list(self._disabled_containers):
            container_layers = [l for nk, container in self._node_container_map.items()
                          if container == container
                          for l in [all_layers.get(nk.rsplit('_', 1)[0])]
                          if l is not None]
            for l in {id(lyr): lyr for lyr in container_layers}.values():
                l.muted = False
        self._hidden_containers -= self._disabled_containers
        self._disabled_containers.clear()
        self.build()

    def _show_container_label_menu(self, container, screen_pos):
        from PySide6.QtWidgets import QInputDialog
        key = self._container_key(container)
        current = self._container_labels.get(key, '')
        current_note = self._container_notes.get(key, '')
        menu = QMenu(self)
        act_set   = menu.addAction("Set label…")
        act_clear = menu.addAction("Clear label")
        act_clear.setEnabled(bool(current))
        menu.addSeparator()
        act_note      = menu.addAction("Edit note…" if current_note else "Set note…")
        act_clear_note = menu.addAction("Clear note")
        act_clear_note.setEnabled(bool(current_note))
        chosen = menu.exec(screen_pos)
        if chosen == act_set:
            text, ok = QInputDialog.getText(
                self, "Column label", "Label:", text=current)
            if ok:
                if text.strip():
                    self._container_labels[key] = text.strip()
                else:
                    self._container_labels.pop(key, None)
                self._refresh_container_label(container)
        elif chosen == act_clear:
            self._container_labels.pop(key, None)
            self._refresh_container_label(container)
        elif chosen == act_note:
            if self._container_note_dialog(container):
                self._refresh_container_note(container)
        elif chosen == act_clear_note:
            self._push_undo()
            self._container_notes.pop(key, None)
            self._refresh_container_note(container)

    def _apply_edge_highlight(self):
        if self._selected_edge:
            self._highlight_selected_edge(*self._selected_edge)

    def _remove_selected_connection(self):
        if not self._selected_edge:
            return
        self._push_undo()
        self._pin_implicit_containers()
        src, tgt = self._selected_edge
        mirror_src = _mirror_name(src)
        mirror_tgt = _mirror_name(tgt)
        def _is_deleted(c):
            if c.src == src and c.tgt == tgt:
                return True
            if mirror_src and mirror_tgt and c.src == mirror_src and c.tgt == mirror_tgt:
                return True
            return False
        self.gui.circuit.connections = [
            c for c in self.gui.circuit.connections if not _is_deleted(c)
        ]
        self._sync_brain_to_circuit()
        self._selected_edge = None
        self.build()

    def _show_edge_edit(self, src, tgt):
        from sensors import CameraSensor as _CamSensor
        W_raw = None
        for c in self.gui.circuit.connections:
            if c.src == src and c.tgt == tgt:
                W_raw = np.asarray(c.W, dtype=float)
                break
        if W_raw is None:
            return

        # Resolve target layer and source sensor
        tgt_layer  = next((l for l in self.gui.circuit.layers  if l.name == tgt), None)
        src_sensor = next((s for s in self.gui.circuit.sensors if s.name == src), None)
        if src_sensor is None:
            src_sensor = next((s for s in self.gui.circuit.sensors
                               if s.name == src.rsplit('_', 1)[0]), None)

        is_conv2d = (W_raw.ndim == 4
                     or (tgt_layer is not None
                         and hasattr(tgt_layer, 'n_filters')
                         and hasattr(tgt_layer, 'kernel_size')))

        _src_is_lat_half = (src_sensor is not None
                            and _sensor_is_lateralized(src_sensor, self.gui.circuit)
                            and (src.endswith('_L') or src.endswith('_R')))

        if is_conv2d and tgt_layer is not None:
            kernel_size = tgt_layer.kernel_size
            if isinstance(src_sensor, _CamSensor):
                in_ch = getattr(src_sensor, 'in_ch', 1)
            else:
                in_ch = max(1, getattr(src_sensor, 'n', 1) if src_sensor else 1)
            is_lat_half = _src_is_lat_half
            W_init = W_raw if W_raw.ndim == 4 else None
            dlg = FilterStackDialog(self, src, tgt,
                                    in_ch, kernel_size, kernel_size, W_init,
                                    lateralized_half=is_lat_half)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            W, ok = dlg.get_result()
            if not ok:
                return
        else:
            W_cur = np.atleast_2d(W_raw)
            n_map  = self._n_map()
            ns     = n_map.get(src, W_cur.shape[1])
            nt     = n_map.get(tgt, W_cur.shape[0])
            # Combined joint-pair sensor connection: ns stored in W is n_L + n_R.
            if _src_is_lat_half and W_cur.shape[1] == ns * 2:
                ns = W_cur.shape[1]
            W, accepted, params = self._open_weight_dialog(
                src, tgt, ns, nt, W_cur, self._weight_params.get((src, tgt)))
            if not accepted:
                return
            if params is not None:
                self._weight_params[(src, tgt)] = params

        self._push_undo()
        from dataclasses import replace as _dc_replace
        new_conns = [_dc_replace(c, W=W) if c.src == src and c.tgt == tgt else c
                     for c in self.gui.circuit.connections]
        self.gui.circuit.connections = new_conns
        if hasattr(self.gui.brain, 'connections'):
            self.gui.brain.connections = new_conns
        if hasattr(self.gui.brain, '_w_cache'):
            self.gui.brain._w_cache = {}
        self.build()

    def _on_node_drag_drop(self, snap_x, mouse_y=None):
        if not self._selected:
            return
        self._push_undo()
        item_name = self._selected.rsplit('_', 1)[0]
        layer  = next((l for l in self.gui.circuit.layers  if l.name == item_name), None)
        sensor = next((s for s in self.gui.circuit.sensors if s.name == item_name), None)
        obj    = layer or sensor
        if obj is None:
            # Lateralized camera halves: 'camera_L_0' → item_name='camera_L' → parent 'camera'
            parent_name = item_name.rsplit('_', 1)[0]
            sensor = next((s for s in self.gui.circuit.sensors if s.name == parent_name), None)
            obj = sensor
        if obj is None:
            return

        current_container = getattr(obj, 'layer', None)

        if self._snap_is_existing(snap_x):
            target_container = next(
                (d for d, x in self._container_x_map.items() if abs(x - snap_x) < 1e-9), None
            )
        else:
            target_container = self._insert_midpoint_container(snap_x, exclude=item_name)

        # Same column and y available → vertical reorder within column.
        if target_container == current_container and mouse_y is not None and layer is not None:
            self._reorder_layer_by_y(layer, mouse_y)
            return

        partner_name = getattr(obj, 'lateral_pair', None)
        if target_container is not None:
            obj.layer = target_container
            # Lateralized layer pairs must always share the same container.
            if partner_name:
                partner = next((l for l in self.gui.circuit.layers
                                if l.name == partner_name), None)
                if partner is not None:
                    partner.layer = target_container
            # _container_labels is keyed by container identity (the occupant's
            # own name set), not position, so it needs no cleanup here — moving
            # this layer to a new slot doesn't touch anyone else's label.
        self._compact_containers()
        self._build_without_selection_filter()

    def _build_without_selection_filter(self):
        """Call build() with _selected cleared so _draw_edges draws all connections,
        then restore the selection highlight for visual feedback.

        _draw_edges uses _selected to filter edges to only the selected neuron's
        index, which is useful for interactive inspection but must not suppress
        unrelated connections after a drag operation.
        """
        sel = self._selected
        self._selected = None
        self.build()
        self._selected = sel
        if sel and sel in self._positions:
            self._spot_pen_override[sel] = 'selected'
            self._redraw_nodes()

    def _reorder_layer_by_y(self, layer, mouse_y):
        """Insert `layer` at the vertical slot nearest to `mouse_y` within its column."""
        # Note: _push_undo() already called by _on_node_drag_drop before delegation here.
        _, insert_idx = self._get_container_snap_y(layer, mouse_y)

        others = self._container_mates(layer)

        def top_y(l):
            ys = [self._positions[f'{l.name}_{j}'][1]
                  for j in range(l.n or 0)
                  if f'{l.name}_{j}' in self._positions]
            return max(ys) if ys else 0.5

        others.sort(key=top_y, reverse=True)  # top-first (highest y = top)

        ordered = others[:insert_idx] + [layer] + others[insert_idx:]
        for i, l in enumerate(ordered):
            l.viz_row = i

        self._build_without_selection_filter()

    # ── Motif palette ─────────────────────────────────────────────────────────

    def _insert_motif(self, name, snap_x):
        from brain_serializer import _sensor_from_dict
        path = os.path.join(MOTIFS_DIR, f'{name}.json')
        try:
            with open(path) as fh:
                payload = json.load(fh)
            if 'layers' not in payload:
                return
        except Exception:
            return

        existing_layers  = {l.name: l for l in self.gui.circuit.layers}
        existing_sensors = {s.name: s for s in self.gui.circuit.sensors}

        # Validate before touching anything: size mismatch on a name collision is fatal.
        for ld in payload['layers']:
            lname   = ld['name']
            motif_n = ld.get('n', 1)
            existing = existing_layers.get(lname) or existing_sensors.get(lname)
            if existing is not None:
                circuit_n = getattr(existing, 'n', 1)
                if motif_n != circuit_n:
                    QMessageBox.warning(
                        self, "Motif import failed",
                        f"'{lname}' already exists with size {circuit_n} "
                        f"but the motif requires size {motif_n}."
                    )
                    return

        self._push_undo()

        # Add sensors that are not already in the circuit.
        brain = self.gui.brain if self.gui else None
        for sd in payload.get('sensors', []):
            sname = sd.get('name')
            if sname and sname not in existing_sensors:
                sensor = _sensor_from_dict(sd)
                sensor.reset()
                self.gui.circuit.sensors.append(sensor)
                existing_sensors[sname] = sensor
                if brain is not None:
                    setattr(brain, sname, np.zeros(sensor.n or 1))

        # Only add layers that are not already in the circuit.
        layers_to_add = [ld for ld in payload['layers']
                         if ld['name'] not in existing_layers
                         and ld['name'] not in existing_sensors]

        new_layers = []
        if layers_to_add:
            if self._snap_is_existing(snap_x):
                target_container = next(
                    (d for d, x in self._container_x_map.items() if abs(x - snap_x) < 1e-9), 1
                )
            else:
                target_container = self._insert_midpoint_container(snap_x, exclude=None)

            pasted_containers = [ld.get('layer') for ld in layers_to_add
                             if ld.get('layer') is not None]
            container_offset  = target_container - (min(pasted_containers) if pasted_containers else target_container)

            for ld in layers_to_add:
                ld = dict(ld)
                if ld.get('layer') is not None:
                    ld['layer'] = ld['layer'] + container_offset
                layer = _layer_from_dict(ld)
                layer.reset()
                self.gui.circuit.layers.append(layer)
                new_layers.append(layer)

        # Add connections, skipping any pair that already exists.
        existing_pairs = {(c.src, c.tgt) for c in self.gui.circuit.connections}
        for cd in payload.get('connections', []):
            src, tgt = cd['src'], cd['tgt']
            if src and tgt and (src, tgt) not in existing_pairs:
                from circuit_model import Connection as _Conn
                self.gui.circuit.connections.append(
                    _Conn(src, tgt, np.array(cd['W'], dtype=float),
                          learning=cd.get('learning'), lr=cd.get('lr', 0.01))
                )
                if cd.get('params'):
                    self._weight_params[(src, tgt)] = cd['params']
                existing_pairs.add((src, tgt))

        # Force a fresh connections list identity so network_runner's
        # conn_id-gated caches (_w_cache, _conn_by_tgt, size-reconciliation,
        # shape-metadata propagation) pick up connections appended above —
        # .append() alone doesn't change id(), so those caches would
        # otherwise silently miss the pasted connections until some
        # unrelated edit replaces the list.
        self.gui.circuit.connections = list(self.gui.circuit.connections)

        # Sync new layers onto the brain so step_network can find them by name
        # and _redraw_nodes can read their output for activity colouring.
        if brain is not None:
            for layer in new_layers:
                setattr(brain, layer.name, layer)
            brain.layers      = self.gui.circuit.layers
            brain.connections = self.gui.circuit.connections
            brain.sensors     = self.gui.circuit.sensors

        self.build()

    def _reload_motifs_palette(self):
        lay = self._motifs_palette_layout
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        os.makedirs(MOTIFS_DIR, exist_ok=True)
        names = sorted(f[:-5] for f in os.listdir(MOTIFS_DIR) if f.endswith('.json'))
        if not names:
            self._motifs_palette_widget.setVisible(False)
            return
        lbl = QLabel("Motifs:")
        lbl.setStyleSheet(f"color:{C['muted']};font-size:9px;min-width:40px;")
        lay.addWidget(lbl)
        for name in names:
            chip = PaletteChip(name, f'motif:{name}')
            chip.setStyleSheet(
                "QPushButton{padding:0 8px;font-size:9px;"
                "border:1px solid #C0A0C8;border-radius:3px;background:#EEE4F4;}"
                "QPushButton:hover{background:#E0D0F0;}"
            )
            lay.addWidget(chip)
        self._motifs_palette_widget.setVisible(True)

    # ── Palette drop (add layer via drag-and-drop) ────────────────────────────

    def eventFilter(self, obj, ev):
        if obj is not self._gw:
            return False
        t = ev.type()
        if t == QEvent.Type.DragEnter:
            if self._edit_mode and ev.mimeData().hasText():
                ev.acceptProposedAction()
                return True
        elif t == QEvent.Type.DragMove:
            if self._edit_mode and ev.mimeData().hasText():
                vb_pt = self._vb.mapSceneToView(
                    self._gw.mapToScene(ev.position().toPoint())
                )
                # Notes are free-positioned — no column-snap preview, just the
                # raw drop point (no indicator to show).
                if ev.mimeData().text() != 'note':
                    self._show_drag_indicator(self._get_snap_x(vb_pt.x()))
                ev.acceptProposedAction()
                return True
        elif t == QEvent.Type.DragLeave:
            self._hide_drag_indicator()
            return True
        elif t == QEvent.Type.Drop:
            self._hide_drag_indicator()
            if self._edit_mode and ev.mimeData().hasText():
                vb_pt = self._vb.mapSceneToView(
                    self._gw.mapToScene(ev.position().toPoint())
                )
                if ev.mimeData().text() == 'note':
                    self._on_note_drop(vb_pt)
                else:
                    snap_x = self._get_snap_x(vb_pt.x())
                    self._on_palette_drop(ev.mimeData().text(), snap_x)
                ev.acceptProposedAction()
                return True
        return False

    def _on_note_drop(self, vb_pt):
        note = self._note_dialog(pos=vb_pt)
        if note is not None:
            self._redraw_note(note)

    def _on_palette_drop(self, mime_text, snap_x):
        if mime_text.startswith('sensor:'):
            if self._snap_is_existing(snap_x):
                sensor_target_container = next(
                    (d for d, x in self._container_x_map.items() if abs(x - snap_x) < 1e-9), 0
                )
            else:
                sensor_target_container = self._insert_midpoint_container(snap_x, exclude=None)
            self._sensor_dialog(mime_text[len('sensor:'):], target_container=sensor_target_container)
            return
        if mime_text.startswith('motif:'):
            self._insert_motif(mime_text[len('motif:'):], snap_x)
            return
        if mime_text == 'joint':
            if self.gui is not None:
                self.gui.add_joint()
            return

        ltype = mime_text
        if self._snap_is_existing(snap_x):
            target_container = next(
                (d for d, x in self._container_x_map.items() if abs(x - snap_x) < 1e-9), 1
            )
        else:
            target_container = self._insert_midpoint_container(snap_x, exclude=None)

        self._layer_dialog(ltype, target_container=target_container)

    def _on_node_clicked(self, name, ev=None):
        self._node_just_clicked = True
        if ev is not None and getattr(ev, 'double', lambda: False)():
            self._selected = name
            self._show_selected_props()
            return
        shift = (ev is not None and ev.modifiers() & Qt.ShiftModifier)
        if self._edit_mode:
            # Edit mode: select only, no edge highlighting
            self._clear_edge_selection()
            self._selected = name
            self._spot_pen_override.clear()
            self._spot_pen_override[name] = 'selected'
            self._redraw_nodes()
        elif shift:
            # View mode + shift: toggle node in multi-selection
            if name in self._multi_selected:
                self._multi_selected.discard(name)
                self._spot_pen_override.pop(name, None)
            else:
                self._multi_selected.add(name)
                self._spot_pen_override[name] = 'multi'
            self._redraw_nodes()
        else:
            # View mode plain click: clear multi-selection, highlight edges
            self._clear_multi_selection()
            self._highlight_edges(name)

    def _clear_multi_selection(self):
        for n in self._multi_selected:
            self._spot_pen_override.pop(n, None)
        self._multi_selected.clear()
        self._redraw_nodes()

    # ── Copy / paste circuit subgraph ─────────────────────────────────────────

    def _paste_selection(self):
        text = QApplication.clipboard().text().strip()
        if not text:
            return
        try:
            payload = json.loads(text)
            if 'layers' not in payload:
                return
        except Exception:
            return
        self._push_undo()

        existing_names = {l.name for l in self.gui.circuit.layers} | \
                         {s.name for s in self.gui.circuit.sensors}

        name_map = {}  # old name → new name (for connection remapping)
        for ld in payload['layers']:
            base = ld['name']
            new_name = base
            suffix = 2
            while new_name in existing_names:
                new_name = f'{base}_{suffix}'
                suffix += 1
            name_map[base] = new_name
            existing_names.add(new_name)

        # Offset pasted containers so they appear after the rightmost existing one.
        existing_containers = [getattr(o, 'layer', 0)
                           for o in list(self.gui.circuit.layers) + list(self.gui.circuit.sensors)
                           if getattr(o, 'layer', None) is not None]
        base = (max(existing_containers) + 1) if existing_containers else 1
        pasted_containers = [ld.get('layer') for ld in payload['layers'] if ld.get('layer') is not None]
        container_offset = base - (min(pasted_containers) if pasted_containers else base)

        for ld in payload['layers']:
            ld = dict(ld)
            ld['name'] = name_map[ld['name']]
            if ld.get('layer') is not None:
                ld['layer'] = ld['layer'] + container_offset
            layer = _layer_from_dict(ld)
            self.gui.circuit.layers.append(layer)

        for cd in payload.get('connections', []):
            src = name_map.get(cd['src'])
            tgt = name_map.get(cd['tgt'])
            if src and tgt:
                from circuit_model import Connection as _Conn
                self.gui.circuit.connections.append(
                    _Conn(src, tgt, np.array(cd['W'], dtype=float),
                          learning=cd.get('learning'), lr=cd.get('lr', 0.01))
                )
                if cd.get('params'):
                    self._weight_params[(src, tgt)] = cd['params']

        # Force a fresh connections list identity — see the matching comment
        # in _insert_motif — so network_runner's conn_id-gated caches don't
        # silently miss the connections just appended above.
        self.gui.circuit.connections = list(self.gui.circuit.connections)

        self.build()

    def _show_selected_props(self):
        if not self._selected:
            return
        obj_name = self._selected.rsplit('_', 1)[0]
        layer = next((l for l in self.gui.circuit.layers  if l.name == obj_name), None)
        if layer is not None:
            if getattr(layer, '_is_joint_motor', False):
                self._body_dialog(layer)
            else:
                self._layer_dialog(type(layer).__name__, layer=layer)
            return
        sensor = next((s for s in self.gui.circuit.sensors if s.name == obj_name), None)
        if sensor is None:
            # Handle split camera half names (e.g. camera_L → camera)
            parent = obj_name.rsplit('_', 1)[0]
            sensor = next((s for s in self.gui.circuit.sensors if s.name == parent), None)
        if sensor is not None:
            self._sensor_dialog(type(sensor).__name__, sensor=sensor)

    def _remove_selected_body(self):
        if not self._selected:
            return
        self._push_undo()
        lname = self._selected.rsplit('_', 1)[0]
        linked_joints = [j for j in self.gui.circuit.joints if j.motor_layer_name == lname]
        child_ids = {j.child_id for j in linked_joints}
        self.gui.circuit.bodies      = [b for b in self.gui.circuit.bodies
                                         if b.id not in child_ids]
        self.gui.circuit.joints      = [j for j in self.gui.circuit.joints
                                         if j.child_id not in child_ids]
        self.gui.circuit.layers      = [l for l in self.gui.circuit.layers
                                         if l.name != lname]
        self.gui.circuit.connections = [c for c in self.gui.circuit.connections
                                         if c.src != lname and c.tgt != lname]
        self._selected = None
        from rigid_body import world_poses
        poses = world_poses(self.gui.bot_pos, self.gui.circuit.bodies, self.gui.circuit.joints)
        self.gui.sync_body_after_removal(poses, self.gui.circuit.bodies)
        self._compact_containers()
        self.build()

    def _on_scene_click(self, event):
        if event.button() == Qt.LeftButton:
            if self._node_just_clicked:
                self._node_just_clicked = False
                return   # click was on a node — don't clear the highlight it just set
            # Notes and container notes are annotation, not circuit structure,
            # so opening/toggling them works regardless of edit mode — unlike
            # everything else in this handler, which is edit-mode-only circuit
            # editing/selection. Checked first since notes draw on top of
            # everything and aren't part of _positions. This is the path that
            # actually fires for a plain click (no movement); NetworkViewBox.
            # mouseDragEvent only runs for genuine drags, so it can't be
            # relied on for selection.
            view_pt = self._vb.mapSceneToView(event.scenePos())
            note_hit = self._note_at(view_pt)
            if note_hit is not None:
                note, zone = note_hit
                if zone == 'toggle':
                    self._push_undo()
                    note.collapsed = not note.collapsed
                    self._redraw_note(note)
                elif self._edit_mode and not event.double():
                    # Edit mode: a plain click selects (for the context menu /
                    # Delete key); only a double click opens the editor. Outside
                    # edit mode there's no selection concept, so any click opens.
                    self._clear_edge_selection()
                    self._selected      = None
                    self._selected_note = note
                else:
                    if self._note_dialog(note=note) is not None:
                        self._redraw_note(note)
                return
            note_container = self._container_note_at(view_pt)
            if note_container is not None:
                if self._container_note_dialog(note_container):
                    self._refresh_container_note(note_container)
                return
            if self._edit_mode:
                edge = self._edge_at(view_pt)
                if edge is not None:
                    self._clear_edge_selection()
                    self._selected_edge = edge
                    self._selected      = None
                    self._selected_note = None
                    self._spot_pen_override.clear()
                    self._redraw_nodes()
                    self._highlight_selected_edge(*edge)
                else:
                    self._clear_edge_selection()
                    # Image-type nodes (Leaky2dLayer, camera halves) have a small
                    # scatter dot hidden under the image circle, so sigClicked rarely
                    # fires on a plain click.  Fall back to _node_at so any click on
                    # the image still selects the node.
                    hit = self._node_at(view_pt)
                    if hit is not None:
                        self._selected      = hit
                        self._selected_note = None
                        self._spot_pen_override.clear()
                        self._spot_pen_override[hit] = 'selected'
                        self._redraw_nodes()
                    else:
                        # Empty canvas click — deselect. Without this, a node
                        # selected earlier (e.g. before adding a new layer)
                        # stays "selected" forever, since nothing else clears
                        # it — which silently blocks the container-label
                        # right-click menu below (guarded on `not self._selected`)
                        # even when the click has moved on to an unrelated
                        # container.
                        if self._selected is not None or self._selected_note is not None:
                            self._selected      = None
                            self._selected_note = None
                            self._spot_pen_override.clear()
                            self._redraw_nodes()
            else:
                if self._highlighted_node:
                    self._clear_edge_highlight()
                if not (event.modifiers() & Qt.ShiftModifier):
                    self._clear_multi_selection()
        if event.button() == Qt.RightButton:
            # Multi-selection context menu (shift-clicked nodes)
            if self._multi_selected:
                event.accept()
                menu = QMenu(self)
                motif_act = menu.addAction("Save as motif…")
                copy_act  = menu.addAction("Copy selection")
                menu.addSeparator()
                clear_act = menu.addAction("Clear selection")
                chosen = menu.exec(event.screenPos().toPoint())
                if chosen == motif_act:
                    self._save_as_motif()
                elif chosen == copy_act:
                    self._copy_selection()
                elif chosen == clear_act:
                    self._clear_multi_selection()
                return
            # View mode: right-click on a node → oscilloscope toggle
            if not self._edit_mode and not self._multi_selected and self._all_scatter is not None:
                local_pt = self._all_scatter.mapFromScene(event.scenePos())
                pts = self._all_scatter.pointsAt(local_pt)
                if pts:
                    lname = pts[0].data().rsplit('_', 1)[0]
                    is_layer  = any(l.name == lname for l in self.gui.circuit.layers)
                    is_sensor = any(s.name == lname for s in self.gui.circuit.sensors)
                    if not is_sensor and not is_layer:
                        parent = lname.rsplit('_', 1)[0]
                        if any(s.name == parent for s in self.gui.circuit.sensors):
                            is_sensor = True
                            lname = parent
                    if is_layer or is_sensor:
                        event.accept()
                        osc_items = self.gui.tracked_osc_items()
                        node_key  = pts[0].data()
                        layer_obj = next((l for l in self.gui.circuit.layers
                                          if l.name == lname), None)
                        n_neurons = getattr(layer_obj, 'n', 1) or 1
                        # Dense layers: track the individual neuron clicked; sparse: the whole layer
                        osc_key = node_key if n_neurons > self._DENSE_THRESHOLD else lname
                        in_osc  = osc_key in osc_items
                        is_muted = getattr(layer_obj, 'muted', False)
                        in_activation_panel = lname in self._activation_pinned
                        menu = QMenu(self)
                        props_act = menu.addAction("Properties…")
                        menu.addSeparator()
                        osc_act = menu.addAction(
                            "Remove from oscilloscope" if in_osc else "Add to oscilloscope")
                        mute_act = None
                        activation_act = None
                        if layer_obj is not None:
                            menu.addSeparator()
                            mute_act = menu.addAction(
                                "Unmute layer" if is_muted else "Mute layer")
                            activation_act = menu.addAction(
                                "Remove from activation panel" if in_activation_panel
                                else "Add to activation panel")
                        chosen = menu.exec(event.screenPos().toPoint())
                        if chosen == props_act:
                            self._selected = node_key
                            self._show_selected_props()
                        elif chosen == osc_act:
                            self.gui.toggle_osc_layer(osc_key)
                        elif mute_act is not None and chosen == mute_act:
                            layer_obj.muted = not is_muted
                            self._redraw_nodes()
                        elif activation_act is not None and chosen == activation_act:
                            self._toggle_activation_entry(lname)
                        return
            # Right-click near a connection arc → weight panel toggle
            if not self._edit_mode and not self._multi_selected:
                view_pt = self._vb.mapSceneToView(event.scenePos())
                edge = self._edge_at(view_pt)
                if edge is not None:
                    src, tgt = edge
                    event.accept()
                    pinned = (src, tgt) in self._weight_pinned
                    menu = QMenu(self)
                    act = menu.addAction(
                        "Remove from weight panel" if pinned else "Add to weight panel")
                    if menu.exec(event.screenPos().toPoint()) == act:
                        self._toggle_weight_entry(src, tgt)
                    return
            # Edit mode: edit/delete for a note — hit-tested at the click
            # position directly (like every edit-mode menu below), not gated
            # on whatever happened to be selected by an earlier click.
            note_hit = None
            if self._edit_mode:
                view_pt = self._vb.mapSceneToView(event.scenePos())
                note_hit = self._note_at(view_pt)
            if note_hit is not None:
                note, _zone = note_hit
                self._selected_note = note
                self._selected      = None
                self._clear_edge_selection()
                event.accept()
                menu = QMenu(self)
                edit_act = menu.addAction("Edit note…")
                collapse_act = menu.addAction("Expand" if note.collapsed else "Collapse")
                menu.addSeparator()
                rm_act = menu.addAction("Delete note")
                chosen = menu.exec(event.screenPos().toPoint())
                if chosen == edit_act:
                    if self._note_dialog(note=note) is not None:
                        self._redraw_note(note)
                elif chosen == collapse_act:
                    self._push_undo()
                    note.collapsed = not note.collapsed
                    self._redraw_note(note)
                elif chosen == rm_act:
                    self._remove_selected_note()
                return
            # Edit mode: edit/remove for a connection — hit-tested at the
            # click position directly, same reasoning as the note block above.
            edge_hit = None
            if self._edit_mode:
                view_pt = self._vb.mapSceneToView(event.scenePos())
                edge_hit = self._edge_at(view_pt)
            if edge_hit is not None:
                self._clear_edge_selection()
                self._selected_edge = edge_hit
                self._selected       = None
                self._selected_note  = None
                self._spot_pen_override.clear()
                self._redraw_nodes()
                self._highlight_selected_edge(*edge_hit)
                event.accept()
                src, tgt = self._selected_edge
                menu = QMenu(self)
                edit_act = menu.addAction(f"Edit weight '{src} → {tgt}'…")
                edit_act.triggered.connect(lambda: self._show_edge_edit(src, tgt))
                menu.addSeparator()
                rm_act = menu.addAction(f"Remove '{src} → {tgt}'")
                rm_act.triggered.connect(self._remove_selected_connection)
                menu.exec(event.screenPos().toPoint())
                return
            # Edit mode: properties + remove for a node — hit-tested at the
            # click position directly, same reasoning as note/edge above.
            node_hit = None
            if self._edit_mode:
                view_pt = self._vb.mapSceneToView(event.scenePos())
                node_hit = self._node_at(view_pt)
            if node_hit is not None:
                self._selected      = node_hit
                self._selected_note = None
                self._clear_edge_selection()
                self._spot_pen_override.clear()
                self._spot_pen_override[node_hit] = 'selected'
                self._redraw_nodes()
                event.accept()
                # Resolve node key → actual name.  Try exact match first so that
                # layer names like 'layer1_L' are not wrongly stripped to 'layer1'.
                # Fall back to stripping suffix for lateralized sensor halves
                # ('sensor0_L' → 'sensor0' → parent sensor name).
                lname    = self._selected
                is_layer  = any(l.name == lname for l in self.gui.circuit.layers)
                is_sensor = any(s.name == lname for s in self.gui.circuit.sensors)
                if not is_layer and not is_sensor:
                    lname    = self._selected.rsplit('_', 1)[0]
                    is_layer  = any(l.name == lname for l in self.gui.circuit.layers)
                    is_sensor = any(s.name == lname for s in self.gui.circuit.sensors)
                    if not is_sensor and not is_layer:
                        parent = lname.rsplit('_', 1)[0]
                        if any(s.name == parent for s in self.gui.circuit.sensors):
                            is_sensor = True
                            lname = parent
                if is_layer or is_sensor:
                    layer_obj = next((l for l in self.gui.circuit.layers
                                      if l.name == lname), None)
                    is_muted  = getattr(layer_obj, 'muted', False)
                    is_body   = getattr(layer_obj, '_is_joint_motor', False)
                    menu = QMenu(self)
                    if is_body:
                        props_act = menu.addAction("Edit body…")
                    else:
                        props_act = menu.addAction("Properties…")
                    props_act.triggered.connect(self._show_selected_props)
                    if layer_obj is not None and not is_body:
                        mute_act = menu.addAction(
                            "Unmute layer" if is_muted else "Mute layer")
                        mute_act.triggered.connect(
                            lambda _checked, lo=layer_obj, m=is_muted:
                                (setattr(lo, 'muted', not m), self._redraw_nodes()))
                    menu.addSeparator()
                    if is_layer:
                        if is_body:
                            rm_act = menu.addAction(f"Remove body '{lname}'")
                            rm_act.triggered.connect(self._remove_selected_body)
                        else:
                            rm_act = menu.addAction(f"Remove layer '{lname}'")
                            if lname == 'motor':
                                rm_act.setEnabled(False)
                                rm_act.setToolTip("Motor layer cannot be removed")
                            else:
                                rm_act.triggered.connect(self._remove_selected_layer)
                    else:
                        rm_act = menu.addAction(f"Remove sensor '{lname}'")
                        rm_act.triggered.connect(self._remove_selected_sensor)
                    menu.addSeparator()
                    node_key   = self._selected
                    osc_items  = self.gui.tracked_osc_items()
                    n_neurons  = getattr(layer_obj, 'n', 1) or 1
                    # Dense layers: track the individual neuron clicked; sparse: the whole layer
                    osc_key    = node_key if n_neurons > self._DENSE_THRESHOLD else lname
                    in_osc     = osc_key in osc_items
                    osc_act = menu.addAction(
                        "Remove from oscilloscope" if in_osc else "Add to oscilloscope")
                    osc_act.triggered.connect(
                        lambda _checked, k=osc_key: self.gui.toggle_osc_layer(k))
                    if layer_obj is not None:
                        in_activation_panel = lname in self._activation_pinned
                        activation_act = menu.addAction(
                            "Remove from activation panel" if in_activation_panel
                            else "Add to activation panel")
                        activation_act.triggered.connect(
                            lambda _checked, n=lname: self._toggle_activation_entry(n))
                    menu.exec(event.screenPos().toPoint())
                    return
            # Edit mode: right-click on empty container-panel space — no
            # note/edge/node was hit at this position above — → label/note
            # annotation menu. Purely position-based like everything above,
            # so it no longer depends on there being nothing selected.
            if self._edit_mode:
                view_pt = self._vb.mapSceneToView(event.scenePos())
                hit_container = self._panel_at(view_pt)
                if hit_container is not None:
                    event.accept()
                    self._show_container_label_menu(hit_container, event.screenPos().toPoint())
                    return

    def _sync_brain_to_circuit(self):
        """Re-point brain.layers/connections/sensors to the current circuit lists."""
        brain = getattr(self.gui, 'brain', None)
        if brain is None:
            return
        if hasattr(brain, 'layers'):
            brain.layers      = self.gui.circuit.layers
        if hasattr(brain, 'connections'):
            brain.connections = self.gui.circuit.connections
        if hasattr(brain, 'sensors'):
            brain.sensors     = self.gui.circuit.sensors

    def _remove_selected_layer(self):
        if not self._selected:
            return
        # Node key IS the layer name — do not strip suffix here.
        lname = self._selected
        if not any(l.name == lname for l in self.gui.circuit.layers):
            lname = self._selected.rsplit('_', 1)[0]
        if lname == 'motor':
            return
        self._push_undo()
        self._pin_implicit_containers()
        # Collect all names to delete: layer + its lateral pair (if any).
        lyr_obj = next((l for l in self.gui.circuit.layers if l.name == lname), None)
        names_to_delete = {lname}
        if lyr_obj is not None:
            pair = getattr(lyr_obj, 'lateral_pair', None)
            if pair:
                names_to_delete.add(pair)
        self.gui.circuit.layers      = [l for l in self.gui.circuit.layers
                                         if l.name not in names_to_delete]
        self.gui.circuit.connections = [c for c in self.gui.circuit.connections
                                         if c.src not in names_to_delete
                                         and c.tgt not in names_to_delete]
        self._sync_brain_to_circuit()
        self._selected = None
        self._compact_containers()
        self.build()

    def _remove_selected_sensor(self):
        if not self._selected:
            return
        self._push_undo()
        self._pin_implicit_containers()
        sname = self._selected.rsplit('_', 1)[0]
        # If sname is a split-camera half (camera_L / camera_R), resolve to parent
        if not any(s.name == sname for s in self.gui.circuit.sensors):
            sname = sname.rsplit('_', 1)[0]
        self.gui.circuit.sensors     = [s for s in self.gui.circuit.sensors
                                         if s.name != sname]
        # Also remove connections to/from lateralized halves (e.g. sensor0_L, sensor0_R).
        pfx = sname + '_'
        self.gui.circuit.connections = [c for c in self.gui.circuit.connections
                                         if c.src != sname and c.tgt != sname
                                         and not c.src.startswith(pfx)
                                         and not c.tgt.startswith(pfx)]
        self._sync_brain_to_circuit()
        self._selected = None
        self.build()

    def _remove_selected_note(self):
        note = self._selected_note
        self._selected_note = None
        if note is None or note not in self.gui.circuit.notes:
            return  # stale reference (e.g. circuit reloaded via undo) — nothing to do
        self._push_undo()
        self._remove_note_items(note)
        self.gui.circuit.notes.remove(note)

    # ── Shared weight-matrix dialog ───────────────────────────────────────────

    def _open_weight_dialog(self, src_name, tgt_name, ns, nt, W_init,
                            saved_params=None, conv_params=None):
        """Weight-matrix editor. Returns (W_final, True, params) or (None, False, None)."""
        dlg = WeightMatrixDialog(self, src_name, tgt_name, ns, nt, W_init,
                                 self.gui.circuit, saved_params,
                                 conv_params=conv_params)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None, False, None
        W_final, params_out = dlg.get_result()
        return W_final, True, params_out

    def _finish_connection(self, src_key, tgt_key):
        from sensors import CameraSensor as _CamSensor
        src_name = src_key.rsplit('_', 1)[0]
        tgt_name = tgt_key.rsplit('_', 1)[0]

        tgt_layer  = next((l for l in self.gui.circuit.layers  if l.name == tgt_name), None)
        src_sensor = next((s for s in self.gui.circuit.sensors if s.name == src_name), None)
        if src_sensor is None:
            # Lateralized sensor half: 'sensor_L' → resolve to parent 'sensor'
            parent = src_name.rsplit('_', 1)[0]
            src_sensor = next((s for s in self.gui.circuit.sensors if s.name == parent), None)

        # True when source is one half of a lateralized sensor (camera or joint-pair)
        _src_is_lat_half = (src_sensor is not None
                            and _sensor_is_lateralized(src_sensor, self.gui.circuit)
                            and (src_name.endswith('_L') or src_name.endswith('_R')))

        is_conv2d = (tgt_layer is not None
                     and hasattr(tgt_layer, 'n_filters')
                     and hasattr(tgt_layer, 'kernel_size'))

        from neurons import Leaky2dLayer as _L2d, Reichardt2dLayer as _R2d
        is_leaky2d = (tgt_layer is not None and isinstance(tgt_layer, (_L2d, _R2d)))

        # ── Image-layer input validation ──────────────────────────────────────
        if is_leaky2d or is_conv2d:
            _src_lyr_v     = next((l for l in self.gui.circuit.layers if l.name == src_name), None)
            _src_is_camera = isinstance(src_sensor, _CamSensor)
            _src_is_img    = isinstance(_src_lyr_v, (_L2d, _R2d))

            if is_leaky2d and not _src_is_camera:
                _src_type = (type(_src_lyr_v).__name__ if _src_lyr_v
                             else type(src_sensor).__name__ if src_sensor else "unknown")
                QMessageBox.warning(
                    self, "Invalid connection",
                    f"'{tgt_name}' ({type(tgt_layer).__name__}) requires a camera sensor as input.\n\n"
                    f"'{src_name}' is a {_src_type}, which does not produce an image.\n\n"
                    f"Valid sources: GrayCameraSensor, RGBCameraSensor."
                )
                return

            if is_conv2d and not (_src_is_camera or _src_is_img):
                _src_type = (type(_src_lyr_v).__name__ if _src_lyr_v
                             else type(src_sensor).__name__ if src_sensor else "unknown")
                QMessageBox.warning(
                    self, "Invalid connection",
                    f"'{tgt_name}' (Conv2dLayer) requires a camera sensor or an image layer as input.\n\n"
                    f"'{src_name}' is a {_src_type}, which does not produce an image.\n\n"
                    f"Valid sources: GrayCameraSensor, RGBCameraSensor, Leaky2dLayer, Reichardt2dLayer."
                )
                return

        _pair_name = None   # lateral pair partner of the source layer (if any)
        _pair_lyr  = None   # partner layer object

        if is_leaky2d:
            # Leaky2dLayer: auto-wire a 1-D ones passthrough weight (no dialog).
            # Derive pixel count from source camera; fall back to n_map.
            # NOTE: check _L/_R suffix FIRST — src_sensor is resolved to the parent
            # camera above, so isinstance(src_sensor, _CamSensor) would be True even
            # for a half, causing full-camera dimensions to be used incorrectly.
            if src_name.endswith(('_L', '_R')):
                # Lateralized camera half — src_sensor was resolved to parent above.
                parent_sensor = src_sensor
                if isinstance(parent_sensor, _CamSensor) and getattr(parent_sensor, 'lateralized', False):
                    mid     = parent_sensor.width // 2
                    overlap = getattr(parent_sensor, 'overlap', 0)
                    if src_name.endswith('_L'):
                        half_w = int(np.clip(mid + overlap, 0, parent_sensor.width))
                    else:
                        r_start = int(np.clip(mid - overlap, 0, parent_sensor.width))
                        half_w  = parent_sensor.width - r_start
                    ns = half_w * parent_sensor.height * parent_sensor.in_ch
                    tgt_layer.in_ch   = parent_sensor.in_ch
                    tgt_layer.frame_h = parent_sensor.height
                    tgt_layer.frame_w = half_w
                else:
                    ns = self._n_map().get(src_name, 1)
            elif isinstance(src_sensor, _CamSensor):
                ns = src_sensor.width * src_sensor.height * src_sensor.in_ch
                tgt_layer.in_ch   = src_sensor.in_ch
                tgt_layer.frame_h = src_sensor.height
                tgt_layer.frame_w = src_sensor.width
            else:
                ns = self._n_map().get(src_name, 1)

            def _ensure_image_n(lyr, n):
                # Leaky2dLayer always outputs the full image (n == pixel count).
                # A pooled Reichardt2dLayer (pool != 'none') outputs only
                # n_directions scalars — its own .n must stay at n_directions,
                # not the (much larger) incoming pixel count; frame_h/frame_w/
                # in_ch (set above) are all step() needs to reshape the input.
                if isinstance(lyr, _R2d) and lyr.pool != 'none':
                    return
                lyr._ensure_n(n)

            _ensure_image_n(tgt_layer, ns)
            W = np.ones(ns, dtype=np.float32)
            self._push_undo()
            from circuit_model import Connection as _Conn
            new_conns = [c for c in self.gui.circuit.connections
                         if not (c.src == src_name and c.tgt == tgt_name)]
            new_conns.append(_Conn(src_name, tgt_name, W))
            # Auto-wire mirror: sensor_L→leaky_L triggers sensor_R→leaky_R
            _l2d_pair = getattr(tgt_layer, 'lateral_pair', None)
            if _l2d_pair and src_name.endswith(('_L', '_R')):
                mirror_src = _mirror_name(src_name)
                if mirror_src and mirror_src != src_name:
                    mirror_tgt = _l2d_pair
                    mirror_lyr = next(
                        (l for l in self.gui.circuit.layers if l.name == mirror_tgt), None)
                    if mirror_lyr is not None:
                        _ensure_image_n(mirror_lyr, ns)
                        mirror_lyr.in_ch   = tgt_layer.in_ch
                        mirror_lyr.frame_h = tgt_layer.frame_h
                        mirror_lyr.frame_w = tgt_layer.frame_w
                    W_mirror = np.ones(ns, dtype=np.float32)
                    new_conns = [c for c in new_conns
                                 if not (c.src == mirror_src and c.tgt == mirror_tgt)]
                    new_conns.append(_Conn(mirror_src, mirror_tgt, W_mirror))
            self.gui.circuit.connections = new_conns
            self.build()
            return

        if is_conv2d:
            # Conv2dLayer path — open FilterStackDialog
            kernel_size = tgt_layer.kernel_size
            _src_lyr_c  = next((l for l in self.gui.circuit.layers if l.name == src_name), None)
            if isinstance(src_sensor, _CamSensor):
                in_ch = getattr(src_sensor, 'in_ch', 1)
            elif isinstance(_src_lyr_c, (_L2d, _R2d)):
                in_ch = max(1, _src_lyr_c.in_ch)
            else:
                in_ch = max(1, getattr(src_sensor, 'n', 1) if src_sensor else 1)
            is_lat_half = _src_is_lat_half or (
                isinstance(_src_lyr_c, (_L2d, _R2d))
                and getattr(_src_lyr_c, 'lateral_pair', None) is not None
                and (src_name.endswith('_L') or src_name.endswith('_R'))
            )
            W_init = None
            for c in self.gui.circuit.connections:
                if c.src == src_name and c.tgt == tgt_name:
                    arr = np.asarray(c.W, dtype=float)
                    if arr.ndim == 4:
                        W_init = arr.copy()
                    break
            dlg = FilterStackDialog(self, src_name, tgt_name,
                                    in_ch, kernel_size, kernel_size, W_init,
                                    lateralized_half=is_lat_half)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            W, ok = dlg.get_result()
            if not ok:
                return
        else:
            # Standard linear path
            n_map  = self._n_map()
            ns     = n_map.get(src_name, 1)
            nt     = n_map.get(tgt_name, 1)
            # Lateralized conv pair source: combine both halves into one weight column block.
            # e.g. conv1_L (n=1 filter) + conv1_R (n=1 filter) → ns=2, W shape (nt, 2).
            _src_lyr    = next((l for l in self.gui.circuit.layers if l.name == src_name), None)
            _pair_name  = getattr(_src_lyr, 'lateral_pair', None) if _src_lyr else None
            _pair_lyr   = None
            if _pair_name:
                _pair_lyr = next((l for l in self.gui.circuit.layers if l.name == _pair_name), None)
                if _pair_lyr:
                    ns = (_src_lyr.n or 0) + (_pair_lyr.n or 0)
            # Lateralized joint-pair sensor half → non-lat target: combined matrix (n_L + n_R).
            # When target is itself lateralized (has _lateral_pair), auto-mirror is used instead.
            _tgt_is_lat = (tgt_layer is not None
                           and getattr(tgt_layer, 'lateral_pair', None) is not None)
            if not _pair_name and _src_is_lat_half and not _tgt_is_lat:
                ns = ns * 2
            W_init = np.zeros((nt, ns))
            for c in self.gui.circuit.connections:
                if c.src == src_name and c.tgt == tgt_name:
                    ew = np.atleast_2d(np.asarray(c.W, dtype=float))
                    if ew.shape == (nt, ns):
                        W_init = ew.copy()
                    break
            W, accepted, params = self._open_weight_dialog(
                src_name, tgt_name, ns, nt, W_init,
                self._weight_params.get((src_name, tgt_name)))
            if not accepted:
                return
            if params is not None:
                self._weight_params[(src_name, tgt_name)] = params

        self._push_undo()
        from circuit_model import Connection as _Conn
        new_conns = [c for c in self.gui.circuit.connections
                     if not (c.src == src_name and c.tgt == tgt_name)]
        # When using a combined pair weight, also remove the partner's separate connection.
        if not is_conv2d and _pair_lyr is not None:
            new_conns = [c for c in new_conns
                         if not (c.src == _pair_name and c.tgt == tgt_name)]
        W_snap = np.asarray(W, dtype=float).copy() if W is not None else None
        new_conns.append(_Conn(src_name, tgt_name, W, init_W=W_snap))

        # Auto-wire the mirror connection.
        # For conv2d: sensor_L→conv_L auto-wires sensor_R→conv_R (lat→lat);
        #             sensor_L→conv auto-wires sensor_R→conv (lat→non-lat camera case).
        # For linear: lat sensor half → lat layer auto-wires the R half to the partner.
        pair_name = getattr(tgt_layer, 'lateral_pair', None)
        if is_conv2d:
            mirror_src = _mirror_name(src_name)
            if mirror_src and mirror_src != src_name:
                if pair_name:
                    mirror_tgt = pair_name
                elif is_lat_half:
                    mirror_tgt = tgt_name
                else:
                    mirror_tgt = None
                if mirror_tgt:
                    new_conns = [c for c in new_conns
                                 if not (c.src == mirror_src and c.tgt == mirror_tgt)]
                    new_conns.append(_Conn(mirror_src, mirror_tgt, W, init_W=W_snap))
        elif _src_is_lat_half and pair_name:
            # Linear lat sensor half → lat target: auto-wire mirror side.
            mirror_src = _mirror_name(src_name)
            if mirror_src and mirror_src != src_name:
                new_conns = [c for c in new_conns
                             if not (c.src == mirror_src and c.tgt == pair_name)]
                new_conns.append(_Conn(mirror_src, pair_name, W, init_W=W_snap))

        self.gui.circuit.connections = new_conns
        if hasattr(self.gui.brain, 'connections'):
            self.gui.brain.connections = new_conns
        if hasattr(self.gui.brain, '_w_cache'):
            self.gui.brain._w_cache = {}
        self._build_without_selection_filter()

    def _new_network(self):
        """Create a blank circuit pre-populated with a motor output layer."""
        motor = SumLayer(activation='linear', name='motor', n=2, layer=4)
        self.gui.circuit.sensors     = []
        self.gui.circuit.layers      = [motor]
        self.gui.circuit.connections = []
        self.gui.brain.layers        = [motor]
        self.gui.brain.connections   = []
        setattr(self.gui.brain, 'motor', motor)
        self.gui.brain.network_file  = ''
        self._hidden_containers   = set()
        self._disabled_containers = set()
        self._container_labels    = {}
        self._container_notes     = {}
        self._selected = None
        self.build()

