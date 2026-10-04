"""
network_viz_serialization.py - serialization-trigger for the network visualizer.

NetworkPersistence (the window's `persistence`) writes the circuit out:
network JSON / brain Python (Save), motifs, copy selection, and the
Copy-Bonsai / Copy-SVG toolbar actions. The edit-transaction / undo hooks
(circuit_editor.py) are on NetworkVisualizerWindow.
"""

import os
import json
import inspect

from PySide6.QtWidgets import QMessageBox, QApplication
from PySide6.QtCore import QTimer, QMimeData

from brain_base import DataBrain
from brain_serializer import serialize_brain, _layer_to_dict, _connection_to_dict

import data_paths

class NetworkPersistence:
    """Saving and exporting the network: network JSON / brain Python, motifs,
    copy selection, Bonsai XML and SVG to the clipboard."""


    def __init__(self, win):
        self.win = win

    def save_as_motif(self):
        from PySide6.QtWidgets import QInputDialog
        from brain_serializer import _sensor_to_dict
        layer_names = {n.rsplit('_', 1)[0] for n in self.win.editing.sel.multi}
        if not layer_names:
            return
        layers = [l for l in self.win.gui.circuit.layers if l.name in layer_names]
        # Include connections where at least one endpoint is a selected layer;
        # this captures sensor→layer edges as well as layer→layer edges.
        all_names = layer_names | {s.name for s in self.win.gui.circuit.sensors}
        conns = [c for c in self.win.gui.circuit.connections
                 if c.tgt in layer_names and c.src in (layer_names | all_names)]
        # Save sensors whose output is consumed by the selected layers.
        src_names = {c.src for c in conns}
        sensors = [s for s in self.win.gui.circuit.sensors if s.name in src_names]
        from app_version import get_app_version
        payload = {
            'saved_with_app_version': get_app_version(),
            'sensors':     [_sensor_to_dict(s) for s in sensors],
            'layers':      [_layer_to_dict(l) for l in layers],
            'connections': [_connection_to_dict(c, self.win._weight_params.get((c.src, c.tgt)))
                            for c in conns],
        }
        name, ok = QInputDialog.getText(self.win, "Save motif", "Motif name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        with open(data_paths.user_path('motifs', f'{name}.json'), 'w') as fh:
            json.dump(payload, fh, indent=2)
        self.win._reload_motifs_palette()


    def copy_selection(self):
        layer_names = {n.rsplit('_', 1)[0] for n in self.win.editing.sel.multi}
        if not layer_names:
            return
        layers = [l for l in self.win.gui.circuit.layers if l.name in layer_names]
        conns  = [c for c in self.win.gui.circuit.connections
                  if c.src in layer_names and c.tgt in layer_names]
        payload = {
            'layers':      [_layer_to_dict(l) for l in layers],
            'connections': [_connection_to_dict(c, self.win._weight_params.get((c.src, c.tgt)))
                            for c in conns],
        }
        QApplication.clipboard().setText(json.dumps(payload, indent=2))


    def save_circuit(self):
        if isinstance(self.win.gui.brain, DataBrain):
            self._save_network_json()
        else:
            self._save_brain_python()

    def _save_brain_python(self):
        brain     = self.win.gui.brain
        brain_cls = brain.__class__
        try:
            src_file = inspect.getfile(brain_cls)
        except Exception as e:
            QMessageBox.critical(self.win, 'Save', f'Cannot determine brain source file:\n{e}')
            return
        if data_paths.is_builtin(src_file):
            QMessageBox.information(
                self.win, 'Save',
                f'{os.path.basename(src_file)} ships with the simulator and is read-only.\n'
                'Create your own brain (+ next to the brain list) to edit its circuit.')
            return

        try:
            with open(src_file, 'r', encoding='utf-8') as f:
                content = f.read()
            content = serialize_brain(
                content,
                self.win.gui.circuit.layers,
                self.win.gui.circuit.connections,
            )
            with open(src_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception:
            import traceback
            QMessageBox.critical(self.win, 'Save error', traceback.format_exc())
            return

        print(f'[NetworkViz] Saved circuit to {src_file}')
        self.win.gui.load_brain()
        self.win.build()
        QMessageBox.information(self.win, 'Save', f'Saved to\n{os.path.basename(src_file)}')

    def _save_network_json(self):
        from brain_serializer import save_network_file
        from PySide6.QtWidgets import QDialog, QFormLayout, QComboBox, QLineEdit, QDialogButtonBox, QPushButton, QHBoxLayout, QInputDialog
        current = getattr(self.win.gui.brain, 'network_file', '') or ''
        project = getattr(self.win.gui.brain, 'network_project', '')

        # ── Saving always goes to the user's networks/ — built-in ones are
        # read-only. A built-in network's folder is offered there by name.
        nets_root = data_paths.user_path('networks')
        subdirs = [''] + data_paths.user_subdirs('networks')
        project = (project or '')
        if project.startswith(data_paths.BUILTIN):
            project = project[len(data_paths.BUILTIN):]
        if project and project not in subdirs:
            subdirs = sorted(subdirs + [project])

        dlg = QDialog(self.win)
        dlg.setWindowTitle('Save Network')
        form = QFormLayout(dlg)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        dir_row = QHBoxLayout()
        dir_combo = QComboBox()
        dir_combo.addItems([d if d else '(My Networks)' for d in subdirs])
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
                subdirs_display = [d if d else '(My Networks)' for d in subdirs]
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
                self.win.gui.circuit.sensors,
                self.win.gui.circuit.layers,
                self.win.gui.circuit.connections,
                self.win._hidden_containers,
                self.win._disabled_containers,
                self.win._container_labels,
                bodies=self.win.gui.circuit.bodies,
                joints=self.win.gui.circuit.joints,
                connection_params=self.win._weight_params,
                notes=self.win.gui.circuit.notes,
                container_notes=self.win._container_notes,
            )
        except Exception:
            import traceback
            QMessageBox.critical(self.win, 'Save error', traceback.format_exc())
            return
        self.win.gui.brain.network_file    = name
        self.win.gui.brain.network_project = chosen_project
        self.win.gui.connection_params = self.win._weight_params
        # Sync brain attribute references to the live circuit layers (joints may
        # have been added/removed since the last full load).
        self.win.gui.brain.layers      = self.win.gui.circuit.layers
        self.win.gui.brain.sensors     = self.win.gui.circuit.sensors
        self.win.gui.brain.connections = self.win.gui.circuit.connections
        self.win.gui.brain_mgr.resolve_joint_sensor_refs()
        # Update the sidebar network-file combo to reflect the (possibly new) name.
        self.win.gui.rebuild_brain_params()
        print(f'[NetworkViz] Saved network to {path}')
        QMessageBox.information(self.win, 'Save', f'Network saved to\n{path}')

    def copy_bonsai(self):
        circuit = self.win.gui.circuit
        try:
            xml = self._generate_bonsai_xml(circuit)
        except Exception:
            import traceback
            QMessageBox.critical(self.win, 'Bonsai Export',
                                 f'Error generating XML:\n{traceback.format_exc()}')
            return
        QApplication.clipboard().setText(xml)
        old_text = self.win._btn_bonsai.text()
        self.win._btn_bonsai.setText('Copied!')
        QTimer.singleShot(1500, lambda: self.win._btn_bonsai.setText(old_text))

    def _generate_bonsai_xml(self, circuit):
        from bonsai_exporter import generate_bonsai_xml
        return generate_bonsai_xml(circuit)

    def copy_svg(self):
        try:
            from pyqtgraph.exporters import SVGExporter
            exporter = SVGExporter(self.win._plot)
            svg_bytes = exporter.export(toBytes=True)
        except Exception:
            import traceback
            QMessageBox.critical(self.win, 'SVG Export',
                                 f'Error generating SVG:\n{traceback.format_exc()}')
            return
        if svg_bytes:
            mime = QMimeData()
            mime.setData('image/svg+xml', svg_bytes)
            mime.setText(svg_bytes.decode('utf-8'))
            QApplication.clipboard().setMimeData(mime)
            old_text = self.win._btn_svg.text()
            self.win._btn_svg.setText('Copied!')
            QTimer.singleShot(1500, lambda: self.win._btn_svg.setText(old_text))
