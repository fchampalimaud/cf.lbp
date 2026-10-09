"""
network_viz_editing.py — interaction/editing for the network visualizer
(everything except the modal dialogs, which live in network_viz_dialogs.py).

NetworkEditing (the window's `editing`) handles clicks and context menus,
owns the Selection and drag state, hit-tests what is under the mouse, and does
every operation that mutates the circuit model (self.gui.circuit) in response
to a user action — adding/removing layers/sensors/connections, dragging nodes
between columns, motif insert/paste, etc. It asks the LayoutEngine for
column math and the renderer for the restyles that follow a click
(highlight/selection), but never creates graphics items itself.
NetworkViewBox routes drags on the plot to it.
"""

import os
import json

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QMenu, QDialog, QMessageBox, QApplication
from PySide6.QtCore import Qt

from brain_serializer import _layer_from_dict

from network_viz_layout import _sensor_is_lateralized, _mirror_name, _dist_to_segment
from network_viz_dialogs import WeightMatrixDialog, FilterStackDialog
import data_paths
from circuit_editor import edit_transaction, find_mirror
from circuit_model import Connection
from lateral import side_of, parent_sensor, partner_layer, is_lateral_half, half_names
from dataclasses import dataclass, field


@dataclass
class Selection:
    """What the user has selected or highlighted in the network editor, kept
    as self.editing.sel. Owned by NetworkEditing; render reads it to draw rings and
    fade unrelated edges."""
    node: str = None          # selected node key, or None
    edge: tuple = None        # selected connection (src_name, tgt_name), or None
    note: object = None       # selected Note (for the Delete key), or None
    multi: set = field(default_factory=set)            # node keys shift+clicked
    highlighted: str = None   # node whose connections are highlighted
    pen_override: dict = field(default_factory=dict)   # node key → 'selected' | 'multi' ring

def _without(conns, src, tgt):
    """conns minus any src → tgt connection."""
    return [c for c in conns if not (c.src == src and c.tgt == tgt)]


# ============================================================
# CUSTOM VIEWBOX
# ============================================================

class NetworkViewBox(pg.ViewBox):
    """Custom ViewBox: left-drags connect (node onto node), move (Alt) or drag
    notes; empty-space drags and every drag while locked pan."""

    def __init__(self, nw):
        super().__init__()
        self._nw       = nw
        self._dragging = False
        self._panning  = False   # this drag pans / zooms the view (decided at its start)

    def mouseDragEvent(self, ev, axis=None):
        nw = self._nw
        # Only the left button drags notes/nodes/connections — a right-button
        # drag (e.g. an incidental mouse move during what's meant to be a
        # right-click for a context menu) must not create a connection or
        # reposition anything. A drag that starts on empty space pans, and
        # so does every drag while locked.
        if ev.isStart():
            start_pt = self.mapSceneToView(ev.buttonDownScenePos())
            self._panning = (nw._locked or ev.button() != Qt.LeftButton
                             or (nw.editing.note_at(start_pt) is None
                                 and nw.editing.node_at(start_pt) is None))
        if not self._panning:
            ev.accept()
            if ev.isStart():
                self._dragging = True
                start_pt = self.mapSceneToView(ev.buttonDownScenePos())
                # Notes draw on top of everything and are never column-snapped —
                # check them first, before any node hit-testing.
                note_hit = nw.editing.note_at(start_pt)
                if note_hit is not None:
                    note, zone = note_hit
                    if zone == 'toggle':
                        with nw._editor.edit():
                            note.collapsed = not note.collapsed
                        nw._on_edit_committed()
                        nw.renderer.redraw_note(note)
                        self._dragging = False
                        return
                    # 'icon' (collapsed) and 'body' (expanded) both drag freely.
                    # One undo step for the whole drag: begun here, committed on finish.
                    nw._editor.begin()
                    nw.editing.dragging_note = note
                    nw.editing.sel.note = note
                    nw.editing.sel.node      = None
                    nw.renderer.clear_edge_selection()
                    return
                nw.editing.dragging_note = None
                hit = nw.editing.node_at(start_pt)
                # Alt+drag repositions the node (between columns); a plain
                # drag always starts a connection, independent of whatever
                # is currently selected — the two are disambiguated by the
                # Alt key alone, not by selection state.
                if hit is not None and bool(ev.modifiers() & Qt.AltModifier):
                    nw.editing.sel.node      = hit
                    nw.editing.sel.note = None
                    nw.editing.sel.pen_override.clear()
                    nw.editing.sel.pen_override[hit] = 'selected'
                    nw.renderer.redraw_nodes()
                    nw.editing.conn_from = None
                else:
                    nw.editing.conn_from = hit   # connect from this node to another
            if nw.editing.dragging_note is not None:
                if self._dragging:
                    mouse_pt = self.mapSceneToView(ev.scenePos())
                    nw.editing.dragging_note.x = mouse_pt.x()
                    nw.editing.dragging_note.y = mouse_pt.y()
                    nw.renderer.redraw_note(nw.editing.dragging_note)
                if ev.isFinish():
                    self._dragging   = False
                    nw.editing.dragging_note = None
                    nw._editor.commit()
                    nw._on_edit_committed()
                return
            if self._dragging:
                mouse_pt = self.mapSceneToView(ev.scenePos())
                if nw.editing.conn_from is not None:
                    nw.renderer.update_conn_preview(mouse_pt)
                elif nw.editing.sel.node is not None:
                    snap_x = nw.layout_engine.get_snap_x(mouse_pt.x())
                    layer_name = nw.editing.sel.node.rsplit('_', 1)[0]
                    layer = next((l for l in nw.gui.circuit.layers if l.name == layer_name), None)
                    nw.renderer.show_drag_indicator(snap_x, mouse_pt.y(), layer)
            if ev.isFinish():
                self._dragging = False
                mouse_pt = self.mapSceneToView(ev.scenePos())
                if nw.editing.conn_from is not None:
                    nw.renderer.hide_conn_preview()
                    hit = nw.editing.node_at(mouse_pt)
                    if hit is not None:
                        src_layer = nw.editing.conn_from.rsplit('_', 1)[0]
                        tgt_layer = hit.rsplit('_', 1)[0]
                        if tgt_layer != src_layer:
                            nw.editing.finish_connection(nw.editing.conn_from, hit)
                    nw.editing.conn_from = None
                elif nw.editing.sel.node is not None:
                    snap_x = nw.layout_engine.get_snap_x(mouse_pt.x())
                    nw.renderer.hide_drag_indicator()
                    nw.editing.on_node_drag_drop(snap_x, mouse_pt.y())
        else:
            super().mouseDragEvent(ev, axis)

class NetworkEditing:
    """Everything the user does to the circuit in the network window: clicks
    and context menus, selection, connecting, removing, pasting, inserting
    motifs, dropping palette chips, and hit-testing what is under the mouse.
    Circuit changes run as edit transactions on the window's editor."""


    def __init__(self, win):
        self.win = win
        self.sel               = Selection()
        self.conn_from         = None    # node key being dragged for connection
        self.node_just_clicked = False   # prevents scene click from clearing a fresh highlight
        self.dragging_note     = None    # Note being drag-moved, or None

    # edit_transaction runs against the window's circuit editor.
    @property
    def _editor(self):
        return self.win._editor

    def _on_edit_committed(self):
        self.win._on_edit_committed()

    def on_spots_clicked(self, _plot, spots, ev):
        if len(spots):
            self._on_node_clicked(spots[0].data(), ev)

    @edit_transaction
    def _show_container_label_menu(self, container, screen_pos):
        from PySide6.QtWidgets import QInputDialog
        key = self.win.layout_engine.container_key(container)
        current = self.win._container_labels.get(key, '')
        current_note = self.win._container_notes.get(key, '')
        menu = QMenu(self.win)
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
                self.win, "Column label", "Label:", text=current)
            if ok:
                if text.strip():
                    self.win._container_labels[key] = text.strip()
                else:
                    self.win._container_labels.pop(key, None)
                self.win.renderer.refresh_container_label(container)
        elif chosen == act_clear:
            self.win._container_labels.pop(key, None)
            self.win.renderer.refresh_container_label(container)
        elif chosen == act_note:
            if self.win.dialogs.container_note_dialog(container):
                self.win.renderer.refresh_container_note(container)
        elif chosen == act_clear_note:
            self.win._container_notes.pop(key, None)
            self.win.renderer.refresh_container_note(container)

    def apply_edge_highlight(self):
        if self.sel.edge:
            self.win.renderer.highlight_selected_edge(*self.sel.edge)

    @edit_transaction
    def remove_selected_connection(self):
        if not self.sel.edge:
            return
        self.win.layout_engine.pin_implicit_containers()
        src, tgt = self.sel.edge
        # The auto-wired mirror goes too (also lateral camera halves → one shared target).
        deleted = {(src, tgt)}
        mirror = find_mirror(self.win.gui.circuit, src, tgt)
        if mirror:
            deleted.add(mirror)
        self.win.gui.circuit.connections = [
            c for c in self.win.gui.circuit.connections if (c.src, c.tgt) not in deleted
        ]
        for key in deleted:
            self.win._weight_params.pop(key, None)
        self._sync_brain_to_circuit()
        self.sel.edge = None
        self.win.build()

    @edit_transaction
    def _show_edge_edit(self, src, tgt):
        W_raw = None
        for c in self.win.gui.circuit.connections:
            if c.src == src and c.tgt == tgt:
                W_raw = np.asarray(c.W, dtype=float)
                break
        if W_raw is None:
            return

        # Resolve target layer and source sensor (a half's parent camera/sensor)
        tgt_layer  = next((l for l in self.win.gui.circuit.layers  if l.name == tgt), None)
        src_sensor = next((s for s in self.win.gui.circuit.sensors if s.name == src), None)
        if src_sensor is None:
            src_sensor = parent_sensor(self.win.gui.circuit.sensors, src)

        is_conv2d = W_raw.ndim == 4 or (tgt_layer is not None and tgt_layer.kernel_weights)

        _src_is_lat_half = (src_sensor is not None and side_of(src) is not None
                            and _sensor_is_lateralized(src_sensor, self.win.gui.circuit))

        if is_conv2d and tgt_layer is not None:
            kernel_size = tgt_layer.kernel_size
            if getattr(src_sensor, 'is_camera', False):
                in_ch = src_sensor.in_ch
            else:
                in_ch = max(1, getattr(src_sensor, 'n', 1) if src_sensor else 1)
            is_lat_half = _src_is_lat_half
            W_init = W_raw if W_raw.ndim == 4 else None
            dlg = FilterStackDialog(self.win, src, tgt,
                                    in_ch, kernel_size, kernel_size, W_init,
                                    lateralized_half=is_lat_half)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            W, ok = dlg.get_result()
            if not ok:
                return
        else:
            W_cur = np.atleast_2d(W_raw)
            n_map  = self.win.layout_engine.n_map()
            ns     = n_map.get(src, W_cur.shape[1])
            nt     = n_map.get(tgt, W_cur.shape[0])
            # Combined joint-pair sensor connection: ns stored in W is n_L + n_R.
            if _src_is_lat_half and W_cur.shape[1] == ns * 2:
                ns = W_cur.shape[1]
            W, accepted, params = self._open_weight_dialog(
                src, tgt, ns, nt, W_cur, self.win._weight_params.get((src, tgt)))
            if not accepted:
                return
            if params is not None:
                self.win._weight_params[(src, tgt)] = params

        # The mirror connection gets the same weights (rules/network_viz.md).
        edited = {(src, tgt)}
        mirror = find_mirror(self.win.gui.circuit, src, tgt)
        if mirror:
            m_conn = next(c for c in self.win.gui.circuit.connections if (c.src, c.tgt) == mirror)
            if np.shape(m_conn.W) == np.shape(W):
                edited.add(mirror)
                if (src, tgt) in self.win._weight_params:
                    self.win._weight_params[mirror] = dict(self.win._weight_params[(src, tgt)])
        from dataclasses import replace as _dc_replace
        self.win.gui.circuit.connections = [
            _dc_replace(c, W=np.array(W, copy=True)) if (c.src, c.tgt) in edited else c
            for c in self.win.gui.circuit.connections]
        self.win.build()

    @edit_transaction
    def on_node_drag_drop(self, snap_x, mouse_y=None):
        if not self.sel.node:
            return
        item_name = self.sel.node.rsplit('_', 1)[0]
        layer  = next((l for l in self.win.gui.circuit.layers  if l.name == item_name), None)
        sensor = next((s for s in self.win.gui.circuit.sensors if s.name == item_name), None)
        obj    = layer or sensor
        if obj is None:
            # Lateralized camera halves: 'camera_L_0' → item_name='camera_L' → parent 'camera'
            parent_name = item_name.rsplit('_', 1)[0]
            sensor = next((s for s in self.win.gui.circuit.sensors if s.name == parent_name), None)
            obj = sensor
        if obj is None:
            return

        current_container = getattr(obj, 'layer', None)

        if self.win.layout_engine.snap_is_existing(snap_x):
            target_container = next(
                (d for d, x in self.win._lay.container_x_map.items() if abs(x - snap_x) < 1e-9), None
            )
        else:
            target_container = self.win.layout_engine.insert_midpoint_container(snap_x, exclude=item_name)

        # Same column and y available → vertical reorder within column.
        if target_container == current_container and mouse_y is not None and layer is not None:
            self._reorder_layer_by_y(layer, mouse_y)
            return

        partner_name = getattr(obj, 'lateral_pair', None)
        if target_container is not None:
            obj.layer = target_container
            # Lateralized layer pairs must always share the same container.
            if partner_name:
                partner = next((l for l in self.win.gui.circuit.layers
                                if l.name == partner_name), None)
                if partner is not None:
                    partner.layer = target_container
            # _container_labels is keyed by container identity (the occupant's
            # own name set), not position, so it needs no cleanup here — moving
            # this layer to a new slot doesn't touch anyone else's label.
        self.win.layout_engine.compact_containers()
        self.build_without_selection_filter()

    def build_without_selection_filter(self):
        """Call build() with _selected cleared so _draw_edges draws all connections,
        then restore the selection highlight for visual feedback.

        _draw_edges uses _selected to filter edges to only the selected neuron's
        index, which is useful for interactive inspection but must not suppress
        unrelated connections after a drag operation.
        """
        sel = self.sel.node
        self.sel.node = None
        self.win.build()
        self.sel.node = sel
        if sel and sel in self.win._lay.positions:
            self.sel.pen_override[sel] = 'selected'
            self.win.renderer.redraw_nodes()

    def _reorder_layer_by_y(self, layer, mouse_y):
        """Insert `layer` at the vertical slot nearest to `mouse_y` within its column."""
        # Runs inside _on_node_drag_drop's edit transaction (one undo step).
        _, insert_idx = self.win.layout_engine.get_container_snap_y(layer, mouse_y)

        others = self.win.layout_engine.container_mates(layer)

        def top_y(l):
            ys = [self.win._lay.positions[f'{l.name}_{j}'][1]
                  for j in range(l.n or 0)
                  if f'{l.name}_{j}' in self.win._lay.positions]
            return max(ys) if ys else 0.5

        others.sort(key=top_y, reverse=True)  # top-first (highest y = top)

        ordered = others[:insert_idx] + [layer] + others[insert_idx:]
        for i, l in enumerate(ordered):
            l.viz_row = i

        self.build_without_selection_filter()

    @edit_transaction
    def _insert_motif(self, name, snap_x):
        from brain_serializer import _sensor_from_dict
        path = data_paths.resolve('motifs', f'{name}.json')
        try:
            with open(path) as fh:
                payload = json.load(fh)
            if 'layers' not in payload:
                return
        except Exception:
            return

        existing_layers  = {l.name: l for l in self.win.gui.circuit.layers}
        existing_sensors = {s.name: s for s in self.win.gui.circuit.sensors}

        # Validate before touching anything: size mismatch on a name collision is fatal.
        for ld in payload['layers']:
            lname   = ld['name']
            motif_n = ld.get('n', 1)
            existing = existing_layers.get(lname) or existing_sensors.get(lname)
            if existing is not None:
                circuit_n = getattr(existing, 'n', 1)
                if motif_n != circuit_n:
                    QMessageBox.warning(
                        self.win, "Motif import failed",
                        f"'{lname}' already exists with size {circuit_n} "
                        f"but the motif requires size {motif_n}."
                    )
                    return


        # Add sensors that are not already in the circuit.
        brain = self.win.gui.brain if self.win.gui else None
        for sd in payload.get('sensors', []):
            sname = sd.get('name')
            if sname and sname not in existing_sensors:
                sensor = _sensor_from_dict(sd)
                sensor.reset()
                self.win.gui.circuit.sensors.append(sensor)
                existing_sensors[sname] = sensor
                if brain is not None:
                    setattr(brain, sname, np.zeros(sensor.n or 1))

        # Only add layers that are not already in the circuit.
        layers_to_add = [ld for ld in payload['layers']
                         if ld['name'] not in existing_layers
                         and ld['name'] not in existing_sensors]

        new_layers = []
        if layers_to_add:
            if self.win.layout_engine.snap_is_existing(snap_x):
                target_container = next(
                    (d for d, x in self.win._lay.container_x_map.items() if abs(x - snap_x) < 1e-9), 1
                )
            else:
                target_container = self.win.layout_engine.insert_midpoint_container(snap_x, exclude=None)

            pasted_containers = [ld.get('layer') for ld in layers_to_add
                             if ld.get('layer') is not None]
            container_offset  = target_container - (min(pasted_containers) if pasted_containers else target_container)

            for ld in layers_to_add:
                ld = dict(ld)
                if ld.get('layer') is not None:
                    ld['layer'] = ld['layer'] + container_offset
                layer = _layer_from_dict(ld)
                layer.reset()
                self.win.gui.circuit.layers.append(layer)
                new_layers.append(layer)

        # Add connections, skipping any pair that already exists.
        existing_pairs = {(c.src, c.tgt) for c in self.win.gui.circuit.connections}
        for cd in payload.get('connections', []):
            src, tgt = cd['src'], cd['tgt']
            if src and tgt and (src, tgt) not in existing_pairs:
                from circuit_model import Connection as _Conn
                self.win.gui.circuit.connections.append(
                    _Conn(src, tgt, np.array(cd['W'], dtype=float),
                          learning=cd.get('learning'), lr=cd.get('lr', 0.01))
                )
                if cd.get('params'):
                    self.win._weight_params[(src, tgt)] = cd['params']
                existing_pairs.add((src, tgt))

        # Force a fresh connections list identity so network_runner's
        # conn_id-gated caches (_w_cache, _conn_by_tgt, size-reconciliation,
        # shape-metadata propagation) pick up connections appended above —
        # .append() alone doesn't change id(), so those caches would
        # otherwise silently miss the pasted connections until some
        # unrelated edit replaces the list.
        self.win.gui.circuit.connections = list(self.win.gui.circuit.connections)

        # Sync new layers onto the brain so step_network can find them by name
        # and _redraw_nodes can read their output for activity colouring.
        if brain is not None:
            for layer in new_layers:
                setattr(brain, layer.name, layer)
            brain.layers      = self.win.gui.circuit.layers
            brain.connections = self.win.gui.circuit.connections
            brain.sensors     = self.win.gui.circuit.sensors

        self.win.build()

    def on_note_drop(self, vb_pt):
        note = self.win.dialogs.note_dialog(pos=vb_pt)
        if note is not None:
            self.win.renderer.redraw_note(note)

    @edit_transaction
    def on_palette_drop(self, mime_text, snap_x):
        if mime_text.startswith('sensor:'):
            if self.win.layout_engine.snap_is_existing(snap_x):
                sensor_target_container = next(
                    (d for d, x in self.win._lay.container_x_map.items() if abs(x - snap_x) < 1e-9), 0
                )
            else:
                sensor_target_container = self.win.layout_engine.insert_midpoint_container(snap_x, exclude=None)
            self.win.dialogs.sensor_dialog(mime_text[len('sensor:'):], target_container=sensor_target_container)
            return
        if mime_text.startswith('motif:'):
            self._insert_motif(mime_text[len('motif:'):], snap_x)
            return
        if mime_text == 'joint':
            if self.win.gui is not None:
                self.win.gui.add_joint()
            return

        ltype = mime_text
        if self.win.layout_engine.snap_is_existing(snap_x):
            target_container = next(
                (d for d, x in self.win._lay.container_x_map.items() if abs(x - snap_x) < 1e-9), 1
            )
        else:
            target_container = self.win.layout_engine.insert_midpoint_container(snap_x, exclude=None)

        self.win.dialogs.layer_dialog(ltype, target_container=target_container)

    def _on_node_clicked(self, name, ev=None):
        self.node_just_clicked = True
        if ev is not None and getattr(ev, 'double', lambda: False)():
            self.sel.node = name
            self._show_selected_props()
            return
        self._click_node(name, bool(ev is not None and ev.modifiers() & Qt.ShiftModifier))

    def _click_node(self, name, shift):
        """Shift+click toggles the node in the multi-selection; a plain click
        selects it (Delete / context menu) and highlights its connections."""
        if shift:
            if name in self.sel.multi:
                self.sel.multi.discard(name)
                self.sel.pen_override.pop(name, None)
            else:
                self.sel.multi.add(name)
                self.sel.pen_override[name] = 'multi'
            self.win.renderer.redraw_nodes()
            return
        self.win.renderer.clear_edge_selection()
        self.sel.multi.clear()
        self.sel.note = None
        self.sel.node = name
        self.sel.pen_override.clear()
        self.sel.pen_override[name] = 'selected'
        self.win.renderer.highlight_edges(name)

    def clear_multi_selection(self):
        for n in self.sel.multi:
            self.sel.pen_override.pop(n, None)
        self.sel.multi.clear()
        self.win.renderer.redraw_nodes()

    @edit_transaction
    def paste_selection(self):
        text = QApplication.clipboard().text().strip()
        if not text:
            return
        try:
            payload = json.loads(text)
            if 'layers' not in payload:
                return
        except Exception:
            return

        existing_names = {l.name for l in self.win.gui.circuit.layers} | \
                         {s.name for s in self.win.gui.circuit.sensors}

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
                           for o in list(self.win.gui.circuit.layers) + list(self.win.gui.circuit.sensors)
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
            self.win.gui.circuit.layers.append(layer)

        for cd in payload.get('connections', []):
            src = name_map.get(cd['src'])
            tgt = name_map.get(cd['tgt'])
            if src and tgt:
                from circuit_model import Connection as _Conn
                self.win.gui.circuit.connections.append(
                    _Conn(src, tgt, np.array(cd['W'], dtype=float),
                          learning=cd.get('learning'), lr=cd.get('lr', 0.01))
                )
                if cd.get('params'):
                    self.win._weight_params[(src, tgt)] = cd['params']

        # Force a fresh connections list identity — see the matching comment
        # in _insert_motif — so network_runner's conn_id-gated caches don't
        # silently miss the connections just appended above.
        self.win.gui.circuit.connections = list(self.win.gui.circuit.connections)

        self.win.build()

    def _show_selected_props(self):
        if not self.sel.node:
            return
        obj_name = self.sel.node.rsplit('_', 1)[0]
        layer = next((l for l in self.win.gui.circuit.layers  if l.name == obj_name), None)
        if layer is not None:
            if getattr(layer, '_is_joint_motor', False):
                self.win.dialogs.body_dialog(layer)
            else:
                self.win.dialogs.layer_dialog(type(layer).__name__, layer=layer)
            return
        sensor = next((s for s in self.win.gui.circuit.sensors if s.name == obj_name), None)
        if sensor is None:
            # Handle split camera half names (e.g. camera_L → camera)
            parent = obj_name.rsplit('_', 1)[0]
            sensor = next((s for s in self.win.gui.circuit.sensors if s.name == parent), None)
        if sensor is not None:
            self.win.dialogs.sensor_dialog(type(sensor).__name__, sensor=sensor)

    @edit_transaction
    def _remove_selected_body(self):
        if not self.sel.node:
            return
        lname = self.sel.node.rsplit('_', 1)[0]
        linked_joints = [j for j in self.win.gui.circuit.joints if j.motor_layer_name == lname]
        child_ids = {j.child_id for j in linked_joints}
        self.win.gui.circuit.bodies      = [b for b in self.win.gui.circuit.bodies
                                         if b.id not in child_ids]
        self.win.gui.circuit.joints      = [j for j in self.win.gui.circuit.joints
                                         if j.child_id not in child_ids]
        self.win.gui.circuit.layers      = [l for l in self.win.gui.circuit.layers
                                         if l.name != lname]
        self.win.gui.circuit.connections = [c for c in self.win.gui.circuit.connections
                                         if c.src != lname and c.tgt != lname]
        self.sel.node = None
        from rigid_body import world_poses
        poses = world_poses(self.win.gui.bot_pos, self.win.gui.circuit.bodies, self.win.gui.circuit.joints)
        self.win.gui.sync_body_after_removal(poses, self.win.gui.circuit.bodies)
        self.win.layout_engine.compact_containers()
        self.win.build()

    def on_scene_click(self, event):
        if event.button() == Qt.LeftButton:
            self._on_left_click(event)
        if event.button() == Qt.RightButton:
            self._on_right_click(event)

    def _on_left_click(self, event):
        if self.node_just_clicked:
            self.node_just_clicked = False
            return   # click was on a node — don't clear the highlight it just set
        # Notes draw on top of everything and aren't part of _positions, so
        # they're checked first. This is the path that actually fires for a
        # plain click (no movement); NetworkViewBox.mouseDragEvent only runs
        # for genuine drags, so it can't be relied on for selection.
        view_pt = self.win._vb.mapSceneToView(event.scenePos())
        note_hit = self.note_at(view_pt)
        if note_hit is not None:
            note, zone = note_hit
            if zone == 'toggle':
                self._toggle_note_collapsed(note)
            elif not self.win._locked and not event.double():
                # A plain click selects (for the context menu / Delete key);
                # a double click opens the editor. Locked: nothing to delete,
                # so any click opens.
                self.win.renderer.clear_edge_selection()
                self.sel.node      = None
                self.sel.note = note
            else:
                if self.win.dialogs.note_dialog(note=note) is not None:
                    self.win.renderer.redraw_note(note)
            return
        note_container = self._container_note_at(view_pt)
        if note_container is not None:
            if self.win.dialogs.container_note_dialog(note_container):
                self.win.renderer.refresh_container_note(note_container)
            return
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        # Image-type nodes (Leaky2dLayer, camera halves) have a small
        # scatter dot hidden under the image circle, so sigClicked rarely
        # fires on a plain click.  Fall back to node_at so any click on
        # the image still selects the node.
        hit = self.node_at(view_pt)
        if hit is not None:
            if event.double():
                self.sel.node = hit
                self._show_selected_props()
            else:
                self._click_node(hit, shift)
            return
        if self.sel.highlighted:
            self.win.renderer.clear_edge_highlight()
        edge = self._edge_at(view_pt)
        if edge is not None:
            self._select_edge(edge)
            return
        # Empty canvas click — deselect everything (Shift keeps the
        # multi-selection, so a missed Shift+click doesn't lose it).
        self.win.renderer.clear_edge_selection()
        if self.sel.node is not None:
            self.sel.pen_override.pop(self.sel.node, None)
        self.sel.node = None
        self.sel.note = None
        if not shift:
            self.sel.multi.clear()
            self.sel.pen_override.clear()
        self.win.renderer.redraw_nodes()

    def _select_edge(self, edge):
        self.win.renderer.clear_edge_selection()
        self.sel.edge = edge
        self.sel.node      = None
        self.sel.note = None
        self.sel.multi.clear()
        self.sel.pen_override.clear()
        self.win.renderer.redraw_nodes()
        self.win.renderer.highlight_selected_edge(*edge)

    def _select_node(self, node_key):
        self.sel.node      = node_key
        self.sel.note = None
        self.sel.pen_override.clear()
        self.sel.pen_override[node_key] = 'selected'
        self.win.renderer.redraw_nodes()

    def _toggle_note_collapsed(self, note):
        with self.win._editor.edit():
            note.collapsed = not note.collapsed
        self.win._on_edit_committed()
        self.win.renderer.redraw_note(note)

    def _on_right_click(self, event):
        """Context menus, tried in order; the first one that applies wins.
        Hit-tested at the click position directly, not gated on whatever an
        earlier click selected. Locked: only the view entries (no edit /
        remove)."""
        if self.sel.multi:
            self._multi_selection_menu(event)
            return
        view_pt = self.win._vb.mapSceneToView(event.scenePos())
        if (self._note_menu(event, view_pt) or self._node_menu(event, view_pt)
                or self._edge_menu(event, view_pt)):
            return
        # Right-click on empty container-panel space → label/note menu.
        hit_container = self._panel_at(view_pt)
        if hit_container is not None:
            event.accept()
            self._show_container_label_menu(hit_container, event.screenPos().toPoint())

    def _multi_selection_menu(self, event):
        """Shift-clicked nodes: save as motif / copy / clear."""
        event.accept()
        menu = QMenu(self.win)
        motif_act = menu.addAction("Save as motif…")
        copy_act  = menu.addAction("Copy selection")
        menu.addSeparator()
        clear_act = menu.addAction("Clear selection")
        chosen = menu.exec(event.screenPos().toPoint())
        if chosen == motif_act:
            self.win.persistence.save_as_motif()
        elif chosen == copy_act:
            self.win.persistence.copy_selection()
        elif chosen == clear_act:
            self.clear_multi_selection()

    def _resolve_node_name(self, name):
        """(name, is_layer, is_sensor) for a node key or object name; a
        lateralized sensor half ('sensor0_L') resolves to its parent sensor."""
        is_layer  = any(l.name == name for l in self.win.gui.circuit.layers)
        is_sensor = any(s.name == name for s in self.win.gui.circuit.sensors)
        if not is_sensor and not is_layer:
            parent = name.rsplit('_', 1)[0]
            if any(s.name == parent for s in self.win.gui.circuit.sensors):
                return parent, False, True
        return name, is_layer, is_sensor

    def _osc_key(self, node_key, lname, layer_obj):
        # Dense layers: track the individual neuron clicked; sparse: the whole layer
        n_neurons = getattr(layer_obj, 'n', 1) or 1
        return node_key if n_neurons > self.win._DENSE_THRESHOLD else lname

    def _toggle_mute(self, layer_obj):
        """Mute / unmute a layer; a lateral pair is one layer split spatially,
        so both halves follow."""
        muted = not getattr(layer_obj, 'muted', False)
        partner = next((l for l in self.win.gui.circuit.layers
                        if l.name == getattr(layer_obj, 'lateral_pair', None)), None)
        for lyr in (layer_obj, partner):
            if lyr is not None:
                lyr.muted = muted
        self.win.renderer.redraw_nodes()

    def _note_menu(self, event, view_pt):
        """Right-click a note: edit / collapse / delete (Delete only when unlocked)."""
        note_hit = self.note_at(view_pt)
        if note_hit is None:
            return False
        note, _zone = note_hit
        self.sel.note = note
        self.sel.node      = None
        self.win.renderer.clear_edge_selection()
        event.accept()
        menu = QMenu(self.win)
        edit_act = menu.addAction("Edit note…")
        collapse_act = menu.addAction("Expand" if note.collapsed else "Collapse")
        rm_act = None
        if not self.win._locked:
            menu.addSeparator()
            rm_act = menu.addAction("Delete note")
        chosen = menu.exec(event.screenPos().toPoint())
        if chosen == edit_act:
            if self.win.dialogs.note_dialog(note=note) is not None:
                self.win.renderer.redraw_note(note)
        elif chosen == collapse_act:
            self._toggle_note_collapsed(note)
        elif rm_act is not None and chosen == rm_act:
            self.remove_selected_note()
        return True

    def _edge_menu(self, event, view_pt):
        """Right-click a connection: edit weights / remove (unlocked) and the
        weight panel toggle."""
        edge_hit = self._edge_at(view_pt)
        if edge_hit is None:
            return False
        self._select_edge(edge_hit)
        event.accept()
        src, tgt = edge_hit
        menu = QMenu(self.win)
        if not self.win._locked:
            edit_act = menu.addAction(f"Edit weight '{src} → {tgt}'…")
            edit_act.triggered.connect(lambda: self._show_edge_edit(src, tgt))
            rm_act = menu.addAction(f"Remove '{src} → {tgt}'")
            rm_act.triggered.connect(self.remove_selected_connection)
            menu.addSeparator()
        pinned = (src, tgt) in self.win._weight_pinned
        pin_act = menu.addAction("Remove from weight panel" if pinned else "Add to weight panel")
        pin_act.triggered.connect(lambda: self.win._toggle_weight_entry(src, tgt))
        menu.exec(event.screenPos().toPoint())
        return True

    def _node_menu(self, event, view_pt):
        """Right-click a node: properties / mute / remove (unlocked) /
        oscilloscope / activation panel. Returns False (no menu) for an
        unresolvable node."""
        node_hit = self.node_at(view_pt)
        if node_hit is None:
            return False
        # Resolve node key → actual name.  Try exact match first so that
        # layer names like 'layer1_L' are not wrongly stripped to 'layer1'.
        circuit = self.win.gui.circuit
        if any(o.name == node_hit for o in list(circuit.layers) + list(circuit.sensors)):
            lname, is_layer, is_sensor = self._resolve_node_name(node_hit)
        else:
            lname, is_layer, is_sensor = self._resolve_node_name(node_hit.rsplit('_', 1)[0])
        if not (is_layer or is_sensor):
            return False
        self.win.renderer.clear_edge_selection()
        self._select_node(node_hit)
        event.accept()
        layer_obj  = next((l for l in circuit.layers if l.name == lname), None)
        sensor_obj = next((s for s in circuit.sensors if s.name == lname), None)
        is_muted  = getattr(layer_obj, 'muted', False)
        is_body   = getattr(layer_obj, '_is_joint_motor', False)
        menu = QMenu(self.win)
        props_act = menu.addAction("Edit body…" if is_body else "Properties…")
        props_act.triggered.connect(self._show_selected_props)
        if layer_obj is not None and not is_body:
            mute_act = menu.addAction("Unmute layer" if is_muted else "Mute layer")
            mute_act.triggered.connect(lambda _checked, lo=layer_obj: self._toggle_mute(lo))
        if not self.win._locked:
            menu.addSeparator()
            if is_layer and is_body:
                rm_act = menu.addAction(f"Remove body '{lname}'")
                rm_act.triggered.connect(self._remove_selected_body)
            elif is_layer:
                rm_act = menu.addAction(f"Remove layer '{lname}'")
                if lname == 'motor':
                    rm_act.setEnabled(False)
                    rm_act.setToolTip("Motor layer cannot be removed")
                else:
                    rm_act.triggered.connect(self.remove_selected_layer)
            else:
                rm_act = menu.addAction(f"Remove sensor '{lname}'")
                rm_act.triggered.connect(self.remove_selected_sensor)
        menu.addSeparator()
        osc_key = self._osc_key(node_hit, lname, layer_obj)
        in_osc  = osc_key in self.win.gui.tracked_osc_items()
        osc_act = menu.addAction("Remove from oscilloscope" if in_osc else "Add to oscilloscope")
        osc_act.triggered.connect(lambda _checked, k=osc_key: self.win.gui.toggle_osc_layer(k))
        activation_target = layer_obj if layer_obj is not None else sensor_obj
        if activation_target is not None and not getattr(activation_target, 'is_image_node', False):
            activation_act = menu.addAction(
                "Remove from activation panel" if lname in self.win._activation_pinned
                else "Add to activation panel")
            activation_act.triggered.connect(
                lambda _checked, n=lname: self.win._toggle_activation_entry(n))
        menu.exec(event.screenPos().toPoint())
        return True

    def _sync_brain_to_circuit(self):
        """Re-point brain.layers/connections/sensors to the current circuit lists."""
        brain = getattr(self.win.gui, 'brain', None)
        if brain is None:
            return
        if hasattr(brain, 'layers'):
            brain.layers      = self.win.gui.circuit.layers
        if hasattr(brain, 'connections'):
            brain.connections = self.win.gui.circuit.connections
        if hasattr(brain, 'sensors'):
            brain.sensors     = self.win.gui.circuit.sensors

    @edit_transaction
    def remove_selected_layer(self):
        if not self.sel.node:
            return
        # Node key IS the layer name — do not strip suffix here.
        lname = self.sel.node
        if not any(l.name == lname for l in self.win.gui.circuit.layers):
            lname = self.sel.node.rsplit('_', 1)[0]
        if lname == 'motor':
            return
        self.win.layout_engine.pin_implicit_containers()
        # Collect all names to delete: layer + its lateral pair (if any).
        lyr_obj = next((l for l in self.win.gui.circuit.layers if l.name == lname), None)
        names_to_delete = {lname}
        if lyr_obj is not None:
            pair = getattr(lyr_obj, 'lateral_pair', None)
            if pair:
                names_to_delete.add(pair)
        self.win.gui.circuit.layers      = [l for l in self.win.gui.circuit.layers
                                         if l.name not in names_to_delete]
        self.win.gui.circuit.connections = [c for c in self.win.gui.circuit.connections
                                         if c.src not in names_to_delete
                                         and c.tgt not in names_to_delete]
        self._sync_brain_to_circuit()
        self.sel.node = None
        self.win.layout_engine.compact_containers()
        self.win.build()

    @edit_transaction
    def remove_selected_sensor(self):
        if not self.sel.node:
            return
        self.win.layout_engine.pin_implicit_containers()
        sname = self.sel.node.rsplit('_', 1)[0]
        # If sname is a split-camera half (camera_L / camera_R), resolve to parent
        if not any(s.name == sname for s in self.win.gui.circuit.sensors):
            sname = sname.rsplit('_', 1)[0]
        self.win.gui.circuit.sensors     = [s for s in self.win.gui.circuit.sensors
                                         if s.name != sname]
        # Also remove connections to/from its lateralized halves (sensor0_L,
        # sensor0_R) — exact names, so e.g. 'sensor0_extra' is left alone.
        gone = {sname, *half_names(sname)}
        self.win.gui.circuit.connections = [c for c in self.win.gui.circuit.connections
                                         if c.src not in gone and c.tgt not in gone]
        for key in [k for k in self.win._weight_params if k[0] in gone or k[1] in gone]:
            del self.win._weight_params[key]
        self._sync_brain_to_circuit()
        self.sel.node = None
        self.win.build()

    @edit_transaction
    def remove_selected_note(self):
        note = self.sel.note
        self.sel.note = None
        if note is None or note not in self.win.gui.circuit.notes:
            return  # stale reference (e.g. circuit reloaded via undo) — nothing to do
        self.win.renderer.remove_note_items(note)
        self.win.gui.circuit.notes.remove(note)

    def _open_weight_dialog(self, src_name, tgt_name, ns, nt, W_init,
                            saved_params=None, conv_params=None):
        """Weight-matrix editor. Returns (W_final, True, params) or (None, False, None)."""
        dlg = WeightMatrixDialog(self.win, src_name, tgt_name, ns, nt, W_init,
                                 self.win.gui.circuit, saved_params,
                                 conv_params=conv_params)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None, False, None
        W_final, params_out = dlg.get_result()
        return W_final, True, params_out

    @edit_transaction
    def finish_connection(self, src_key, tgt_key):
        # Node keys are '<name>_<neuron index>'.
        src_name = src_key.rsplit('_', 1)[0]
        tgt_name = tgt_key.rsplit('_', 1)[0]

        tgt_layer  = next((l for l in self.win.gui.circuit.layers  if l.name == tgt_name), None)
        src_sensor = next((s for s in self.win.gui.circuit.sensors if s.name == src_name), None)
        if src_sensor is None:
            # Lateralized sensor half: 'sensor_L' → its parent 'sensor'
            src_sensor = parent_sensor(self.win.gui.circuit.sensors, src_name)

        # True when source is one half of a lateralized sensor (camera or joint-pair)
        src_is_lat_half = (src_sensor is not None and side_of(src_name) is not None
                           and _sensor_is_lateralized(src_sensor, self.win.gui.circuit))

        if tgt_layer is not None and tgt_layer.accepts_image:
            if not self._check_image_source(src_name, src_sensor, tgt_name, tgt_layer):
                return
        # Capabilities declared by the target class (see LayerBase).
        if tgt_layer is not None and tgt_layer.passthrough_input:
            self._wire_passthrough(src_name, src_sensor, tgt_name, tgt_layer)
            return

        is_conv2d = tgt_layer is not None and tgt_layer.kernel_weights
        if is_conv2d:
            is_lat_half = src_is_lat_half or is_lateral_half(self.win.gui.circuit, src_name)
            W = self._ask_filter_stack(src_name, src_sensor, tgt_name, tgt_layer, is_lat_half)
            if W is None:
                return
            pair_name = None
        else:
            result = self._ask_linear_weight(src_name, tgt_name, tgt_layer, src_is_lat_half)
            if result is None:
                return
            W, pair_name = result

        new_conns = _without(self.win.gui.circuit.connections, src_name, tgt_name)
        # When using a combined pair weight, also remove the partner's separate connection.
        if pair_name is not None:
            new_conns = _without(new_conns, pair_name, tgt_name)
        W_snap = np.asarray(W, dtype=float).copy() if W is not None else None
        new_conns.append(Connection(src_name, tgt_name, W, init_W=W_snap))

        # Auto-wire the mirror connection.
        # For conv2d: sensor_L→conv_L auto-wires sensor_R→conv_R (lat→lat);
        #             sensor_L→conv auto-wires sensor_R→conv (lat→non-lat camera case).
        # For linear: lat sensor half → lat layer auto-wires the R half to the partner.
        tgt_pair = getattr(tgt_layer, 'lateral_pair', None)
        mirror_src = _mirror_name(src_name)
        mirror_tgt = None
        if mirror_src and mirror_src != src_name:
            if is_conv2d:
                mirror_tgt = tgt_pair or (tgt_name if is_lat_half else None)
            elif src_is_lat_half:
                mirror_tgt = tgt_pair
        if mirror_tgt:
            new_conns = _without(new_conns, mirror_src, mirror_tgt)
            # Own copies: learning on one side must not change the other.
            new_conns.append(Connection(mirror_src, mirror_tgt, np.array(W, copy=True),
                                        init_W=None if W_snap is None else W_snap.copy()))

        self.win.gui.circuit.connections = new_conns
        mirror = find_mirror(self.win.gui.circuit, src_name, tgt_name)
        if mirror and (src_name, tgt_name) in self.win._weight_params:
            self.win._weight_params[mirror] = dict(self.win._weight_params[(src_name, tgt_name)])
        self.build_without_selection_filter()

    def _check_image_source(self, src_name, src_sensor, tgt_name, tgt_layer):
        """An image layer needs a camera (or, unless it needs a camera, an image
        layer) as input. Warns and returns False otherwise."""
        src_lyr       = next((l for l in self.win.gui.circuit.layers if l.name == src_name), None)
        src_is_camera = getattr(src_sensor, 'is_camera', False)
        src_is_img    = getattr(src_lyr, 'is_image_node', False)
        src_type = (type(src_lyr).__name__ if src_lyr
                    else type(src_sensor).__name__ if src_sensor else "unknown")

        if tgt_layer.needs_camera_input and not src_is_camera:
            QMessageBox.warning(
                self.win, "Invalid connection",
                f"'{tgt_name}' ({type(tgt_layer).__name__}) requires a camera sensor as input.\n\n"
                f"'{src_name}' is a {src_type}, which is not a camera.\n\n"
                f"Valid sources: GrayCameraSensor, RGBCameraSensor."
            )
            return False

        if not (src_is_camera or src_is_img):
            QMessageBox.warning(
                self.win, "Invalid connection",
                f"'{tgt_name}' ({type(tgt_layer).__name__}) requires a camera sensor or an image layer as input.\n\n"
                f"'{src_name}' is a {src_type}, which does not produce an image.\n\n"
                f"Valid sources: cameras, and layers that output an image "
                f"(Leaky2dLayer; Reichardt2dLayer / Conv2dLayer with pool='none')."
            )
            return False
        return True

    def _wire_passthrough(self, src_name, src_sensor, tgt_name, tgt_layer):
        """Leaky2dLayer: auto-wire a 1-D ones passthrough weight (no dialog).
        Pixel count comes from the source camera; falls back to n_map."""
        # NOTE: check for a half FIRST — src_sensor is resolved to the parent
        # camera, so it is a camera even for a half, which would use
        # full-camera dimensions incorrectly.
        if side_of(src_name):
            cam = src_sensor
            if getattr(cam, 'is_camera', False) and getattr(cam, 'lateralized', False):
                half_w = cam.half_width(side_of(src_name))
                ns = half_w * cam.height * cam.in_ch
                tgt_layer.in_ch   = cam.in_ch
                tgt_layer.frame_h = cam.height
                tgt_layer.frame_w = half_w
            else:
                ns = self.win.layout_engine.n_map().get(src_name, 1)
        elif getattr(src_sensor, 'is_camera', False):
            ns = src_sensor.width * src_sensor.height * src_sensor.in_ch
            tgt_layer.in_ch   = src_sensor.in_ch
            tgt_layer.frame_h = src_sensor.height
            tgt_layer.frame_w = src_sensor.width
        else:
            ns = self.win.layout_engine.n_map().get(src_name, 1)

        def _ensure_image_n(lyr, n):
            # An unpooled image layer outputs the full image (n == pixel
            # count). A pooled one (e.g. Reichardt2dLayer, pool != 'none')
            # outputs only a few scalars — its own .n must stay put, not the
            # (much larger) incoming pixel count; frame_h/frame_w/in_ch (set
            # above) are all step() needs to reshape the input.
            if not lyr.n_follows_input:
                return
            lyr._ensure_n(n)

        _ensure_image_n(tgt_layer, ns)
        new_conns = _without(self.win.gui.circuit.connections, src_name, tgt_name)
        new_conns.append(Connection(src_name, tgt_name, np.ones(ns, dtype=np.float32)))
        # Auto-wire mirror: sensor_L→leaky_L triggers sensor_R→leaky_R
        mirror_tgt = getattr(tgt_layer, 'lateral_pair', None)
        mirror_src = _mirror_name(src_name) if mirror_tgt and side_of(src_name) else None
        if mirror_src and mirror_src != src_name:
            mirror_lyr = partner_layer(self.win.gui.circuit.layers, tgt_layer)
            if mirror_lyr is not None:
                _ensure_image_n(mirror_lyr, ns)
                mirror_lyr.in_ch   = tgt_layer.in_ch
                mirror_lyr.frame_h = tgt_layer.frame_h
                mirror_lyr.frame_w = tgt_layer.frame_w
            new_conns = _without(new_conns, mirror_src, mirror_tgt)
            new_conns.append(Connection(mirror_src, mirror_tgt, np.ones(ns, dtype=np.float32)))
        self.win.gui.circuit.connections = new_conns
        self.win.build()

    def _ask_filter_stack(self, src_name, src_sensor, tgt_name, tgt_layer, is_lat_half):
        """Conv2dLayer target: FilterStackDialog. Returns the 4-D kernel, or None."""
        kernel_size = tgt_layer.kernel_size
        src_lyr     = next((l for l in self.win.gui.circuit.layers if l.name == src_name), None)
        if getattr(src_sensor, 'is_camera', False):
            in_ch = src_sensor.in_ch
        elif getattr(src_lyr, 'is_image_node', False):
            in_ch = max(1, getattr(src_lyr, 'in_ch', 1) or 1)
        else:
            in_ch = max(1, getattr(src_sensor, 'n', 1) if src_sensor else 1)
        W_init = None
        for c in self.win.gui.circuit.connections:
            if c.src == src_name and c.tgt == tgt_name:
                arr = np.asarray(c.W, dtype=float)
                if arr.ndim == 4:
                    W_init = arr.copy()
                break
        dlg = FilterStackDialog(self.win, src_name, tgt_name,
                                in_ch, kernel_size, kernel_size, W_init,
                                lateralized_half=is_lat_half)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        W, ok = dlg.get_result()
        return W if ok else None

    def _ask_linear_weight(self, src_name, tgt_name, tgt_layer, src_is_lat_half):
        """Standard target: WeightMatrixDialog. Returns (W, pair_name) — pair_name
        is the source's lateral partner when W covers both halves — or None."""
        n_map = self.win.layout_engine.n_map()
        ns    = n_map.get(src_name, 1)
        nt    = n_map.get(tgt_name, 1)
        # Lateralized conv pair source: combine both halves into one weight column block.
        # e.g. conv1_L (n=1 filter) + conv1_R (n=1 filter) → ns=2, W shape (nt, 2).
        src_lyr   = next((l for l in self.win.gui.circuit.layers if l.name == src_name), None)
        pair_name = getattr(src_lyr, 'lateral_pair', None) if src_lyr else None
        pair_lyr  = None
        if pair_name:
            pair_lyr = next((l for l in self.win.gui.circuit.layers if l.name == pair_name), None)
            if pair_lyr:
                ns = (src_lyr.n or 0) + (pair_lyr.n or 0)
        # Lateralized joint-pair sensor half → non-lat target: combined matrix (n_L + n_R).
        # When target is itself lateralized (has lateral_pair), auto-mirror is used instead.
        tgt_is_lat = (tgt_layer is not None
                      and getattr(tgt_layer, 'lateral_pair', None) is not None)
        if not pair_name and src_is_lat_half and not tgt_is_lat:
            ns = ns * 2
        W_init = np.zeros((nt, ns))
        for c in self.win.gui.circuit.connections:
            if c.src == src_name and c.tgt == tgt_name:
                ew = np.atleast_2d(np.asarray(c.W, dtype=float))
                if ew.shape == (nt, ns):
                    W_init = ew.copy()
                break
        W, accepted, params = self._open_weight_dialog(
            src_name, tgt_name, ns, nt, W_init,
            self.win._weight_params.get((src_name, tgt_name)))
        if not accepted:
            return None
        if params is not None:
            self.win._weight_params[(src_name, tgt_name)] = params
        return W, (pair_name if pair_lyr is not None else None)

    def _panel_at(self, view_pt):
        """Return container of the column panel the view-space point falls inside, or None.

        Bounds must match _draw_panels' rendered rect exactly:
        - `he` half-extent widening for containers that span multiple columns
          (e.g. a wide winner subsuming narrower same-column-at-other-depth
          containers) — without it, most of a spanning container's visible
          box was a dead zone that never hit-tested to anything.
        - vertical extent from the same column-spanning "envelope" used to
          size the rendered rect (_span_envelope_ys), not just this
          container's own nodes — a wide panel can be taller than its own
          occupants if a narrower column it spans over is taller.
        """
        r, px = self.win._NODE_R, self.win._PAD_X
        py = 0.04
        x, y = view_pt.x(), view_pt.y()
        container_data = {}
        for node_key, container in self.win._lay.node_container_map.items():
            if node_key not in self.win._lay.positions:
                continue
            nx, ny = self.win._lay.positions[node_key]
            if container not in container_data:
                container_data[container] = {'xs': [], 'ys': []}
            container_data[container]['xs'].append(nx)
            container_data[container]['ys'].append(ny)
        span_map = self.win._lay.container_span_map
        x_unit   = self.win._lay.x_unit
        for container, data in container_data.items():
            x_col = sum(data['xs']) / len(data['xs'])
            span  = span_map.get(container, 1)
            ys    = self.win.renderer.span_envelope_ys(container_data, container, span) or data['ys']
            y_min = min(ys) - r - py
            y_max = max(ys) + r + py
            he    = (span - 1) / 2.0 * x_unit
            x_min = x_col - r - px - he
            x_max = x_col + r + px + he
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return container
        return None

    def _edge_at(self, view_pt):
        """Return (src_name, tgt_name) of the edge nearest to view_pt, or None."""
        px, py = view_pt.x(), view_pt.y()
        best = self.win._EDGE_CLICK_DIST
        result = None
        for item, sn, tn, _, is_curve, *_ in self.win.renderer.drawn.edge_items_tagged:
            if not is_curve:
                continue
            try:
                xs, ys = item.getData()
            except Exception:
                continue
            if xs is None or len(xs) < 2:
                continue
            for i in range(len(xs) - 1):
                d = _dist_to_segment(px, py, xs[i], ys[i], xs[i+1], ys[i+1])
                if d < best:
                    best = d
                    result = (sn.rsplit('_', 1)[0], tn.rsplit('_', 1)[0])
        return result

    def node_at(self, pt):
        """Return the node key nearest to view-space point *pt*, or None."""
        r2 = (self.win._NODE_R * 1.3) ** 2
        best, best_d2 = None, float('inf')
        for name, (x, y) in self.win._lay.positions.items():
            d2 = (pt.x() - x) ** 2 + (pt.y() - y) ** 2
            if d2 <= r2 and d2 < best_d2:
                best, best_d2 = name, d2
        return best

    def note_at(self, pt):
        """Return (note, zone) for the topmost note hit at view-space point
        *pt* — zone is 'icon' (collapsed note; drag/select/double-click target,
        same as 'body'), 'toggle' (collapse glyph on an expanded note — the
        only zone with its own click behavior), or 'body' (expanded note).
        None if no hit. Checked before _node_at since notes draw on top of
        everything else.
        """
        if not getattr(self.win, '_notes_visible', True):
            return None
        dx, dy = self.win._vb.viewPixelSize()
        # Generous click targets — bigger than the tiny rendered glyphs themselves.
        icon_r2   = ((self.win._NOTE_ICON_PX + 5) * dx) ** 2 + ((self.win._NOTE_ICON_PX + 5) * dy) ** 2
        toggle_dx = (self.win._NOTE_TOGGLE_PX + 4) * dx
        toggle_dy = (self.win._NOTE_TOGGLE_PX + 4) * dy
        for note in reversed(self.win.gui.circuit.notes):
            if note.collapsed:
                d2 = (pt.x() - note.x) ** 2 + (pt.y() - note.y) ** 2
                if d2 <= icon_r2:
                    return note, 'icon'
                continue
            w, h = self.win.renderer.note_size(note)
            x0, x1 = note.x, note.x + w
            y0, y1 = note.y - h, note.y
            if not (x0 <= pt.x() <= x1 and y0 <= pt.y() <= y1):
                continue
            if pt.x() >= x1 - toggle_dx and pt.y() >= y1 - toggle_dy:
                return note, 'toggle'
            return note, 'body'
        return None

    def _container_note_at(self, pt):
        """Return the container whose note glyph is hit at view-space point
        *pt*, or None. Mirrors _note_at's collapsed-icon hit-test, generous
        click target included."""
        dx, dy = self.win._vb.viewPixelSize()
        r2 = ((self.win._CONTAINER_NOTE_ICON_PX + 5) * dx) ** 2 + \
             ((self.win._CONTAINER_NOTE_ICON_PX + 5) * dy) ** 2
        for container, (ix, iy) in self.win.renderer.drawn.container_note_icon_pos.items():
            d2 = (pt.x() - ix) ** 2 + (pt.y() - iy) ** 2
            if d2 <= r2:
                return container
        return None
