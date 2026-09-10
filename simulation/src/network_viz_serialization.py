"""
network_viz_serialization.py - serialization-trigger for the network visualizer.

_SerializationMixin holds every operation that reads/writes the circuit as
JSON or Python source: undo snapshots, motif save, clipboard copy, and the
Save/Copy-Bonsai/Copy-SVG toolbar actions. It never mutates self.gui.circuit
directly except to restore it wholesale from a snapshot (_undo) - everyday
circuit edits live in network_viz_editing.py.
"""

import os
import json
import inspect

from PySide6.QtWidgets import QMessageBox, QApplication
from PySide6.QtCore import QTimer, QMimeData

from brain_base import DataBrain
from brain_serializer import serialize_brain, _layer_to_dict, _connection_to_dict

MOTIFS_DIR = os.path.join(os.path.dirname(__file__), '..', 'motifs')

class _SerializationMixin:
    # ── Undo ──────────────────────────────────────────────────────────────────

    def _push_undo(self):
        from brain_serializer import serialize_network_json
        snapshot = serialize_network_json(
            self.gui.circuit.sensors,
            self.gui.circuit.layers,
            self.gui.circuit.connections,
            self._hidden_containers,
            self._disabled_containers,
            self._container_labels,
            bodies=self.gui.circuit.bodies,
            joints=self.gui.circuit.joints,
            connection_params=self._weight_params,
            notes=self.gui.circuit.notes,
            container_notes=self._container_notes,
        )
        self._undo_stack.append(snapshot)
        self._btn_undo.setEnabled(True)

    def _undo(self):
        if not self._undo_stack:
            return
        from brain_serializer import load_network_json
        snapshot = self._undo_stack.pop()
        sensors, layers, connections, hidden, disabled, container_labels, _bodies, _joints, conn_params, notes, container_notes = load_network_json(snapshot)
        self._weight_params = conn_params
        c = self.gui.circuit
        c.sensors     = sensors
        c.layers      = layers
        c.connections = connections
        c.notes       = notes
        self._hidden_containers   = hidden
        self._disabled_containers = disabled
        self._container_labels    = container_labels
        self._container_notes     = container_notes
        if hasattr(self.gui, 'brain') and self.gui.brain:
            brain = self.gui.brain
            for l in getattr(brain, 'layers', []):
                try:
                    delattr(brain, l.name)
                except AttributeError:
                    pass
            brain.layers      = layers
            brain.connections = connections
            for l in layers:
                setattr(brain, l.name, l)
        self._btn_undo.setEnabled(bool(self._undo_stack))
        self.build()


    def _save_as_motif(self):
        from PySide6.QtWidgets import QInputDialog
        from brain_serializer import _sensor_to_dict
        layer_names = {n.rsplit('_', 1)[0] for n in self._multi_selected}
        if not layer_names:
            return
        layers = [l for l in self.gui.circuit.layers if l.name in layer_names]
        # Include connections where at least one endpoint is a selected layer;
        # this captures sensor→layer edges as well as layer→layer edges.
        all_names = layer_names | {s.name for s in self.gui.circuit.sensors}
        conns = [c for c in self.gui.circuit.connections
                 if c.tgt in layer_names and c.src in (layer_names | all_names)]
        # Save sensors whose output is consumed by the selected layers.
        src_names = {c.src for c in conns}
        sensors = [s for s in self.gui.circuit.sensors if s.name in src_names]
        from app_version import get_app_version
        payload = {
            'saved_with_app_version': get_app_version(),
            'sensors':     [_sensor_to_dict(s) for s in sensors],
            'layers':      [_layer_to_dict(l) for l in layers],
            'connections': [_connection_to_dict(c, self._weight_params.get((c.src, c.tgt)))
                            for c in conns],
        }
        name, ok = QInputDialog.getText(self, "Save motif", "Motif name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        os.makedirs(MOTIFS_DIR, exist_ok=True)
        with open(os.path.join(MOTIFS_DIR, f'{name}.json'), 'w') as fh:
            json.dump(payload, fh, indent=2)
        self._reload_motifs_palette()


    def _copy_selection(self):
        layer_names = {n.rsplit('_', 1)[0] for n in self._multi_selected}
        if not layer_names:
            return
        layers = [l for l in self.gui.circuit.layers if l.name in layer_names]
        conns  = [c for c in self.gui.circuit.connections
                  if c.src in layer_names and c.tgt in layer_names]
        payload = {
            'layers':      [_layer_to_dict(l) for l in layers],
            'connections': [_connection_to_dict(c, self._weight_params.get((c.src, c.tgt)))
                            for c in conns],
        }
        QApplication.clipboard().setText(json.dumps(payload, indent=2))


    def _save_circuit(self):
        if isinstance(self.gui.brain, DataBrain):
            self._save_network_json()
        else:
            self._save_brain_python()

    def _save_brain_python(self):
        brain     = self.gui.brain
        brain_cls = brain.__class__
        try:
            src_file = inspect.getfile(brain_cls)
        except Exception as e:
            QMessageBox.critical(self, 'Save', f'Cannot determine brain source file:\n{e}')
            return

        try:
            with open(src_file, 'r', encoding='utf-8') as f:
                content = f.read()
            content = serialize_brain(
                content,
                self.gui.circuit.layers,
                self.gui.circuit.connections,
            )
            with open(src_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception:
            import traceback
            QMessageBox.critical(self, 'Save error', traceback.format_exc())
            return

        print(f'[NetworkViz] Saved circuit to {src_file}')
        self.gui.load_brain()
        self.build()
        QMessageBox.information(self, 'Save', f'Saved to\n{os.path.basename(src_file)}')

    def _save_network_json(self):
        from brain_serializer import save_network_file
        from PySide6.QtWidgets import QDialog, QFormLayout, QComboBox, QLineEdit, QDialogButtonBox, QPushButton, QHBoxLayout, QInputDialog
        current = getattr(self.gui.brain, 'network_file', '') or ''
        project = getattr(self.gui.brain, 'network_project', '')

        # ── Build list of subdirectories under networks/ ──────────────────────
        nets_root = 'networks'
        os.makedirs(nets_root, exist_ok=True)
        subdirs = [''] + sorted(
            e.name for e in os.scandir(nets_root)
            if e.is_dir() and not e.name.startswith('.')
        )

        dlg = QDialog(self)
        dlg.setWindowTitle('Save Network')
        form = QFormLayout(dlg)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        dir_row = QHBoxLayout()
        dir_combo = QComboBox()
        dir_combo.addItems([d if d else '(networks root)' for d in subdirs])
        current_idx = subdirs.index(project) if project in subdirs else 0
        dir_combo.setCurrentIndex(current_idx)
        btn_new_dir = QPushButton('+')
        btn_new_dir.setFixedWidth(24)
        btn_new_dir.setToolTip('Create new subdirectory')
        def _new_dir():
            name_d, ok = QInputDialog.getText(dlg, 'New directory', 'Directory name:')
            if not ok or not name_d.strip():
                return
            name_d = name_d.strip()
            os.makedirs(os.path.join(nets_root, name_d), exist_ok=True)
            if name_d not in subdirs:
                subdirs.append(name_d)
                subdirs.sort()
                subdirs_display = [d if d else '(networks root)' for d in subdirs]
                dir_combo.clear()
                dir_combo.addItems(subdirs_display)
            dir_combo.setCurrentIndex(subdirs.index(name_d))
        btn_new_dir.clicked.connect(_new_dir)
        dir_row.addWidget(dir_combo, 1)
        dir_row.addWidget(btn_new_dir)
        form.addRow('Directory:', dir_row)

        suggested_name = (current.replace('.json', '') if current else 'my_network')
        name_edit = QLineEdit(suggested_name)
        form.addRow('Name:', name_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        chosen_project = subdirs[dir_combo.currentIndex()]
        name = name_edit.text().strip()
        if not name:
            return
        if not name.endswith('.json'):
            name += '.json'

        net_dir = os.path.join(nets_root, chosen_project) if chosen_project else nets_root
        os.makedirs(net_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(net_dir, name))
        try:
            save_network_file(
                path,
                self.gui.circuit.sensors,
                self.gui.circuit.layers,
                self.gui.circuit.connections,
                self._hidden_containers,
                self._disabled_containers,
                self._container_labels,
                bodies=self.gui.circuit.bodies,
                joints=self.gui.circuit.joints,
                connection_params=self._weight_params,
                notes=self.gui.circuit.notes,
                container_notes=self._container_notes,
            )
        except Exception:
            import traceback
            QMessageBox.critical(self, 'Save error', traceback.format_exc())
            return
        self.gui.brain.network_file    = name
        self.gui.brain.network_project = chosen_project
        self.gui.connection_params = self._weight_params
        # Sync brain attribute references to the live circuit layers (joints may
        # have been added/removed since the last full load).
        self.gui.brain.layers      = self.gui.circuit.layers
        self.gui.brain.sensors     = self.gui.circuit.sensors
        self.gui.brain.connections = self.gui.circuit.connections
        self.gui.brain_mgr.resolve_joint_sensor_refs()
        # Update the sidebar network-file combo to reflect the (possibly new) name.
        self.gui.rebuild_brain_params()
        print(f'[NetworkViz] Saved network to {path}')
        QMessageBox.information(self, 'Save', f'Network saved to\n{path}')

    # ── Bonsai export ─────────────────────────────────────────────────────────

    def _copy_bonsai(self):
        circuit = self.gui.circuit
        try:
            xml = self._generate_bonsai_xml(circuit)
        except Exception:
            import traceback
            QMessageBox.critical(self, 'Bonsai Export',
                                 f'Error generating XML:\n{traceback.format_exc()}')
            return
        QApplication.clipboard().setText(xml)
        old_text = self._btn_bonsai.text()
        self._btn_bonsai.setText('Copied!')
        QTimer.singleShot(1500, lambda: self._btn_bonsai.setText(old_text))

    def _generate_bonsai_xml(self, circuit):
        from bonsai_exporter import generate_bonsai_xml
        return generate_bonsai_xml(circuit)

    def _copy_svg(self):
        try:
            from pyqtgraph.exporters import SVGExporter
            exporter = SVGExporter(self._plot)
            svg_bytes = exporter.export(toBytes=True)
        except Exception:
            import traceback
            QMessageBox.critical(self, 'SVG Export',
                                 f'Error generating SVG:\n{traceback.format_exc()}')
            return
        if svg_bytes:
            mime = QMimeData()
            mime.setData('image/svg+xml', svg_bytes)
            mime.setText(svg_bytes.decode('utf-8'))
            QApplication.clipboard().setMimeData(mime)
            old_text = self._btn_svg.text()
            self._btn_svg.setText('Copied!')
            QTimer.singleShot(1500, lambda: self._btn_svg.setText(old_text))

