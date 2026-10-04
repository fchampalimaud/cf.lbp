import os

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QCheckBox, QScrollArea, QFrame, QMessageBox, QGridLayout,
)
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QColor, QShortcut, QKeySequence

from sim_constants import C
from circuit_editor import CircuitEditor, edit_transaction
from side_view import SideViewWindow

from network_viz_layout import LayoutEngine, LayoutResult
from network_viz_render import NetworkRenderer, WeightEntryWidget, ActivationEntryWidget
from network_viz_dialogs import NetworkDialogs, PaletteChip
from network_viz_editing import NetworkEditing, NetworkViewBox
from network_viz_serialization import NetworkPersistence
import data_paths


def _fill_grid(grid, chips, first_row=0, cols=2):
    """Lay palette chips out in a grid, `cols` per row."""
    for i, chip in enumerate(chips):
        grid.addWidget(chip, first_row + i // cols, i % cols)

class NetworkVisualizerWindow(QWidget):
    _ACTIVE_RGB   = (0.95, 0.45, 0.25)
    _EDGE_POS     = '#3A6FA8'
    _HL_EDGE      = '#E07828'
    _TD_EDGE      = (200, 130, 50)   # amber RGB for learnable (LearningLayerBase) connections
    _TEACH_EDGE   = (60, 170, 70)    # green RGB for SnapshotLayer's teach connection (fixed structural readout)
    _NODE_R       = 0.1125
    _MARKER_R     = 0.066
    _CROSS_BOW    = 0.35
    _INTERNAL_BOW = 0.22
    _PAD_X        = 0.10
    _RING_SCALE        = 0.82   # ring node size and ring radius relative to _NODE_R
    _DENSE_NODE_SCALE  = 0.68   # node size scale when layer has more than 4 neurons
    _MIDLINE_GAP       = 0.05   # extra gap inserted at y=0.5 for even-n layers
    _DENSE_THRESHOLD = 4   # max(ns, nt) > this → sampled edges + badge
    _DENSE_SAMPLE    = 3   # neurons sampled per side for dense connections
    _PAD_Y        = 0.10

    # Connection kind tokens — see rules/network_viz.md "Connection classifier".
    _CK_TEACH  = 'teach'   # SnapshotLayer's teach connection → solid green, fixed style
    _CK_TD     = 'td'      # target is LearningLayerBase → amber style
    _CK_CONV4D = 'conv4d'  # 4-D conv kernel → one arc per filter
    _CK_DENSE  = 'dense'   # max(ns, nt) > _DENSE_THRESHOLD → sampled thin arcs
    _CK_THIN   = 'thin'    # lw=0.6, no marker
    _CK_THICK  = 'thick'   # lw=3.0, with marker

    _PANEL_CFG    = {
        'sensor': ('#D8EED8', '#A8CCA8'),
        'layer':  ('#D4E4F4', '#A0C0DC'),
    }
    _LABEL_TOP_MARGIN = 0.008   # gap above a container's own top edge, front and ghost labels alike
    _EDGE_CLICK_DIST = 0.04  # view-space distance threshold for edge selection
    _NOTE_W          = 0.34
    _NOTE_PAD        = 0.02
    _NOTE_ICON_PX    = 12    # collapsed-icon visual radius, in screen pixels
    _NOTE_TOGGLE_PX  = 10    # collapse-toggle glyph click-zone radius, in screen pixels
    _NOTE_FONT_FAMILY = 'Segoe UI'
    _NOTE_FONT_SIZE   = 7
    _CONTAINER_NOTE_ICON_PX = 8      # visual radius, in screen pixels (vs. 12 for loose notes)
    _CONTAINER_NOTE_PAD     = 0.006  # inset from the container rect's corner, in data coords

    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        # The window's parts; each holds a reference back to the window.
        self.layout_engine = LayoutEngine(self)
        self.dialogs       = NetworkDialogs(self)
        self.renderer      = NetworkRenderer(self)
        self.persistence   = NetworkPersistence(self)
        self.editing       = NetworkEditing(self)
        self.setWindowTitle("Network")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.StrongFocus)
        self.resize(980, 600)

        self._z_cut         = None
        self._side_view     = None
        self._net3d_view    = None
        self._lay              = LayoutResult()   # replaced by every build()
        self._building         = False
        self._locked           = False   # 🔒 Lock: no structural edits (move, connect, add, delete)
        self._weight_params    = {}   # (src, tgt) → last-used pattern params dict
        self._palette_y        = 1.22
        self._container_labels  = {}   # container identity (occupant name set, '|'-joined) → str
        self._container_notes   = {}   # container identity → note text (same keying as labels)
        self._hidden_containers   = set()   # containers hidden from viz but still computed
        self._disabled_containers = set()   # containers excluded from computation (and hidden)
        self._notes_visible     = True    # master show/hide-all toggle, independent of per-note collapse
        self._compact_mode = True
        self._weight_pinned = {}       # (src, tgt) -> WeightEntryWidget
        self._activation_pinned = {}   # layer name -> ActivationEntryWidget

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._build_toolbar())

        # Content area: palette + column panel + weight panel + activation panel + graph side by side
        _content = QWidget()
        _content_lay = QHBoxLayout(_content)
        _content_lay.setContentsMargins(0, 0, 0, 0)
        _content_lay.setSpacing(0)
        _content_lay.addWidget(self._build_palette())
        _content_lay.addWidget(self._build_column_panel())
        # Live weight / activation visualization panels (hidden by default,
        # toggled by the Weights / Activations buttons).
        self._weight_panel, self._weight_entries_layout = self._entry_panel()
        self._activation_panel, self._activation_entries_layout = self._entry_panel()
        _content_lay.addWidget(self._weight_panel)
        _content_lay.addWidget(self._activation_panel)
        _content_lay.addWidget(self._build_plot(), 1)
        layout.addWidget(_content, 1)

        self._connect_toolbar()
        QTimer.singleShot(50, self.build)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(100)
        self._refresh_timer.timeout.connect(self.renderer.on_refresh_timer)
        self._refresh_timer.start()

    def _build_toolbar(self):
        """Row 1: palette / lock / save / export / panel toggles, depth slider, help."""
        toolbar = QWidget()
        tb_lay  = QHBoxLayout(toolbar)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        self._btn_palette = QPushButton("Palette")
        self._btn_palette.setCheckable(True)
        self._btn_palette.setChecked(True)
        self._btn_palette.setToolTip("Show / hide the palette of sensors, layers and motifs")
        self._btn_lock   = QPushButton("🔒 Lock")
        self._btn_lock.setCheckable(True)
        self._btn_lock.setToolTip("Lock the circuit: no moving, connecting, adding or removing")
        self._btn_save   = QPushButton("Save")
        self._btn_bonsai = QPushButton("Copy Bonsai")
        self._btn_svg    = QPushButton("Copy SVG")
        self._btn_undo   = QPushButton("Undo")
        self._btn_save.setEnabled(True)
        self._btn_undo.setEnabled(False)
        tb_lay.addWidget(self._btn_palette)
        tb_lay.addWidget(self._btn_lock)
        tb_lay.addWidget(self._btn_save)
        tb_lay.addWidget(self._btn_bonsai)
        tb_lay.addWidget(self._btn_svg)
        self._btn_cols        = QPushButton("Columns")
        self._btn_weights     = QPushButton("Weights")
        self._btn_activations = QPushButton("Activations")
        self._btn_top         = QPushButton("Side View")
        self._btn_net3d       = QPushButton("3D")
        self._btn_notes       = QPushButton("Hide Notes")
        self._btn_notes.setCheckable(True)
        tb_lay.addWidget(self._btn_undo)
        tb_lay.addWidget(self._btn_cols)
        tb_lay.addWidget(self._btn_weights)
        tb_lay.addWidget(self._btn_activations)
        tb_lay.addWidget(self._btn_top)
        tb_lay.addWidget(self._btn_net3d)
        tb_lay.addWidget(self._btn_notes)
        tb_lay.addStretch()
        self._z_slider_label = QLabel("Depth: 0")
        self._z_slider       = QSlider(Qt.Horizontal)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(0)
        self._z_slider.setFixedWidth(100)
        self._z_slider.setVisible(False)
        self._z_slider_label.setVisible(False)
        tb_lay.addWidget(self._z_slider_label)
        tb_lay.addWidget(self._z_slider)
        self._btn_help = QPushButton("?")
        self._btn_help.setFixedWidth(24)
        self._btn_help.setToolTip("Mouse controls")
        self._btn_help.clicked.connect(self._show_mouse_help)
        tb_lay.addWidget(self._btn_help)
        return toolbar

    def _build_palette(self):
        """Left sidebar: draggable palette chips, one section per kind.
        Collapsed by the Palette button; hidden while the circuit is locked."""
        self._palette_bar = QFrame()
        self._palette_bar.setFixedWidth(176)
        self._palette_bar.setObjectName("palette")
        self._palette_bar.setStyleSheet("QFrame#palette { border: 1px solid #B0C0D0; }")
        outer = QVBoxLayout(self._palette_bar)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        pb_vlay = QVBoxLayout(inner)
        pb_vlay.setContentsMargins(4, 4, 4, 4)
        pb_vlay.setSpacing(2)

        def _section(label, chips):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{C['muted']};font-size:9px;font-weight:bold;margin-top:4px;")
            pb_vlay.addWidget(lbl)
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(3)
            _fill_grid(grid, chips)
            pb_vlay.addLayout(grid)

        _section("Sensors", [
            PaletteChip(s.replace('Sensor', ''), f'sensor:{s}', sensor=True)
            for s in ['GradientSensor', 'ColorSensor',
                      'CollisionSensor', 'DistanceSensor', 'InteroceptiveSensor',
                      'ProprioceptiveSensor', 'WhiskerSensor', 'SkyCompassSensor',
                      'ManualBumpSensor', 'ManualCueSensor', 'GrayCameraSensor', 'RGBCameraSensor']
        ])
        _section("Layers", [
            PaletteChip(t.replace('Layer', ''), t)
            for t in ['LeakyLayer', 'ProductLayer', 'Leaky2dLayer', 'SumLayer', 'ConstantLayer', 'SineLayer',
                      'MatsuokaLayer', 'AdaptiveLayer', 'PulseLayer',
                      'RingAttractorLayer', 'Conv2dLayer', 'Reichardt2dLayer',
                      'AccumulatorLayer']
        ])
        _section("Learning", [PaletteChip(t.replace('Layer', ''), t)
                              for t in ['TDLayer', 'DeltaLayer', 'ThreeFactorLayer', 'SnapshotLayer']])
        _section("Body / Note", [PaletteChip('Body', 'joint'), PaletteChip('Note', 'note')])

        self._motifs_palette_widget = QWidget()
        self._motifs_palette_layout = QGridLayout(self._motifs_palette_widget)
        self._motifs_palette_layout.setContentsMargins(0, 0, 0, 0)
        self._motifs_palette_layout.setSpacing(3)
        pb_vlay.addWidget(self._motifs_palette_widget)
        pb_vlay.addStretch()
        self._reload_motifs_palette()

        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return self._palette_bar

    def _build_column_panel(self):
        """Column visibility panel (left sidebar, toggled by "Columns" button)."""
        self._col_panel = QFrame()
        self._col_panel.setFixedWidth(140)
        self._col_panel.setFrameShape(QFrame.Shape.Box)
        self._col_panel.setFrameShadow(QFrame.Shadow.Plain)
        self._col_panel.setLineWidth(1)
        self._col_panel.setStyleSheet("QFrame { border: 1px solid #B0C0D0; }")
        self._col_panel.setVisible(False)
        _gb_outer = QVBoxLayout(self._col_panel)
        _gb_outer.setContentsMargins(4, 4, 4, 4)
        _gb_outer.setSpacing(4)

        self._compact_cb = QCheckBox("Compact")
        self._compact_cb.setChecked(True)
        self._compact_cb.setStyleSheet("font-size:9px;")
        self._compact_cb.stateChanged.connect(self._on_compact_toggled)
        _gb_outer.addWidget(self._compact_cb)

        self._group_scroll = QScrollArea()
        self._group_scroll.setWidgetResizable(True)
        self._group_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._group_scroll.setFrameShape(self._group_scroll.Shape.NoFrame)
        self._group_cols_widget = QWidget()
        self._group_bar_lay = QVBoxLayout(self._group_cols_widget)
        self._group_bar_lay.setContentsMargins(0, 0, 0, 0)
        self._group_bar_lay.setSpacing(2)
        self._group_bar_lay.addStretch()
        self._group_scroll.setWidget(self._group_cols_widget)
        _gb_outer.addWidget(self._group_scroll, 1)

        _action_ss = (
            "QPushButton{padding:2px 4px;font-size:9px;"
            "border:1px solid #A0B0C0;border-radius:3px;}"
        )
        self._btn_show_all   = QPushButton("Show All")
        self._btn_enable_all = QPushButton("Enable All")
        for b in (self._btn_show_all, self._btn_enable_all):
            b.setFixedHeight(20)
            b.setStyleSheet(_action_ss)
        self._btn_show_all.clicked.connect(self._on_show_all)
        self._btn_enable_all.clicked.connect(self._on_enable_all)
        _btn_row = QHBoxLayout()
        _btn_row.setSpacing(4)
        _btn_row.addWidget(self._btn_show_all)
        _btn_row.addWidget(self._btn_enable_all)
        _gb_outer.addLayout(_btn_row)
        return self._col_panel

    @staticmethod
    def _entry_panel():
        """A hidden right-side panel holding a scrollable column of entries
        (the Weights and Activations panels). Returns (frame, entries_layout)."""
        panel = QFrame()
        panel.setFixedWidth(230)
        panel.setFrameShape(QFrame.Shape.Box)
        panel.setFrameShadow(QFrame.Shadow.Plain)
        panel.setLineWidth(1)
        panel.setStyleSheet("QFrame { border: 1px solid #B0C0D0; }")
        panel.setVisible(False)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        entries = QVBoxLayout(inner)
        entries.setContentsMargins(0, 0, 0, 0)
        entries.setSpacing(6)
        entries.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return panel, entries

    def _build_plot(self):
        """The graph: pyqtgraph widget, view box and plot, plus overlay items."""
        self._gw   = pg.GraphicsLayoutWidget()
        self._gw.setBackground(QColor(C['bg']))
        self._gw.setFocusPolicy(Qt.NoFocus)   # keep key events on NetworkVisualizerWindow
        self._vb   = NetworkViewBox(self)
        self._plot = self._gw.addPlot(viewBox=self._vb)
        self._plot.setAspectLocked(True)
        self._plot.hideAxis('bottom')
        self._plot.hideAxis('left')
        self._plot.setMenuEnabled(False)
        self._vb.setMenuEnabled(False)
        self._vb.disableAutoRange()
        self._gw.scene().sigMouseClicked.connect(
            self.editing.on_scene_click, Qt.ConnectionType.QueuedConnection
        )
        self.renderer.create_overlays()
        self._gw.setAcceptDrops(True)
        self._gw.installEventFilter(self)
        return self._gw

    def _connect_toolbar(self):
        self._btn_cols.clicked.connect(self._toggle_col_win)
        self._btn_weights.clicked.connect(self._toggle_weight_panel)
        self._btn_activations.clicked.connect(self._toggle_activation_panel)
        self._btn_top.clicked.connect(self._toggle_side_view)
        self._btn_net3d.clicked.connect(self._toggle_net3d)
        self._btn_notes.toggled.connect(self._toggle_notes_visible)
        self._btn_palette.toggled.connect(self._toggle_palette)
        self._btn_lock.toggled.connect(self._toggle_lock)
        self._btn_save.clicked.connect(self.persistence.save_circuit)
        self._btn_bonsai.clicked.connect(self.persistence.copy_bonsai)
        self._btn_svg.clicked.connect(self.persistence.copy_svg)
        self._btn_undo.clicked.connect(self._undo)
        self._z_slider.valueChanged.connect(self._on_z_slider_changed)
        QShortcut(QKeySequence.StandardKey.Undo, self).activated.connect(self._undo)

    def renumber_columns(self, new_number):
        """Renumber the hidden / disabled column sets after columns moved:
        new_number(old) → new column, or None to drop it. In place — the app
        keeps references to these same set objects (and saves them)."""
        for columns in (self._hidden_containers, self._disabled_containers):
            moved = {new_number(d) for d in columns}
            moved.discard(None)
            columns.clear()
            columns.update(moved)

    def _show_mouse_help(self):
        QMessageBox.information(self, "Mouse controls", """\
&bull; Click a node — select it and highlight its connections<br>
&bull; Shift+click nodes — multi-select (right-click for a group menu)<br>
&bull; Double-click a node — open its properties dialog<br>
&bull; Drag a node onto another — connect them<br>
&bull; Alt+drag a node — move it to a different column<br>
&bull; Drag empty space / wheel — pan / zoom<br>
&bull; Right-click a node / connection / note / column — context menu<br>
&bull; Delete — remove the selected node / connection / note<br>
&bull; Ctrl+C / Ctrl+V — copy / paste the selected subgraph<br>
&bull; Drag a chip from the palette — add a new sensor / layer / motif<br>
<br>
🔒 Lock — no moving, connecting, adding or removing; properties and
view options (oscilloscope, panels, mute) still work.<br>
F1 lists every shortcut of the app.""")

    # ── Building the drawing ────────────────────────────────────────────────────

    def build(self):
        self.layout_engine.compact_containers()
        self._update_z_slider()
        self._building = True
        self._gw.setUpdatesEnabled(False)
        try:
            self.renderer.draw()
        finally:
            self._building = False
            self._gw.setUpdatesEnabled(True)
            self._gw.update()
        # Redraw edges once the widget has settled to its final pixel size.
        QTimer.singleShot(0, self.renderer.rebuild_edges)
        # Reapply camera image rects after the event loop settles (in case
        # Qt reorders transforms during addItem or scene attachment).
        QTimer.singleShot(0, self.renderer.reapply_camera_rects)
        # Notify the app so arena overlays and oscilloscope channels stay in sync.
        self.gui.rebuild_channels()
        if self._side_view and self._side_view.isVisible():
            self._side_view.refresh()
        # The circuit may have changed under us (agent switch, brain reload).
        self._btn_undo.setEnabled(self._editor.can_undo())

    def update(self, brain):
        if self._building or self.renderer.drawn.all_scatter is None:
            return
        try:
            self.renderer.redraw_nodes(brain)
        except Exception:
            pass

    # ── Edit transactions and undo ──────────────────────────────────────────────

    @property
    def _editor(self):
        """Editor for the current agent's circuit. Its undo history lives on the
        circuit itself, so it survives closing this window and switching agents."""
        return CircuitEditor(self.gui.circuit, getattr(self.gui, 'brain', None),
                             meta=self, brain_mgr=getattr(self.gui, 'brain_mgr', None))

    def _on_edit_committed(self):
        """After every committed edit (and undo): refresh what build() doesn't —
        the Undo button, the 3-D network view and the arena's body drawing."""
        self._btn_undo.setEnabled(self._editor.can_undo())
        view = getattr(self, '_net3d_view', None)
        try:
            if view is not None and view.isVisible():
                view.refresh()
        except RuntimeError:          # window already destroyed
            self._net3d_view = None
        sync_bodies = getattr(self.gui, 'sync_bodies', None)
        if sync_bodies is not None:
            sync_bodies()

    def _undo(self):
        if not self._editor.undo():
            return
        self.editing.sel.node = None
        self.editing.sel.note = None
        self.editing.sel.edge = None
        self._on_edit_committed()
        self.build()

    # ── Window chrome: lock, palette, side panels, z slider, palette, Qt event hooks ────

    def keyPressEvent(self, ev):
        ctrl = ev.modifiers() & Qt.ControlModifier
        if self._locked and (ev.key() == Qt.Key_Delete or (ctrl and ev.key() == Qt.Key_V)):
            return
        if ev.key() == Qt.Key_Delete:
            if self.editing.sel.note is not None:
                self.editing.remove_selected_note()
            elif self.editing.sel.edge is not None:
                self.editing.remove_selected_connection()
            elif self.editing.sel.node is not None:
                lname = self.editing.sel.node.rsplit('_', 1)[0]
                if any(l.name == lname for l in self.gui.circuit.layers):
                    self.editing.remove_selected_layer()
                elif any(s.name == lname for s in self.gui.circuit.sensors):
                    self.editing.remove_selected_sensor()
        elif ctrl and ev.key() == Qt.Key_C:
            self.persistence.copy_selection()
        elif ctrl and ev.key() == Qt.Key_V:
            self.editing.paste_selection()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self._refresh_timer.stop()
        self.gui.notify_closed()
        super().closeEvent(ev)

    def _toggle_palette(self, shown):
        self._palette_bar.setVisible(shown and not self._locked)

    def _toggle_lock(self, locked):
        """🔒 Lock: the circuit can be inspected and its parameters tuned, but
        nothing is moved, connected, added or removed. Hides the palette."""
        self._locked = bool(locked)
        self._btn_lock.setStyleSheet(f"background:{C['primary']};color:white;" if locked else "")
        self._btn_palette.setEnabled(not locked)
        self._palette_bar.setVisible(self._btn_palette.isChecked() and not locked)
        self.editing.conn_from = None
        self.renderer.hide_conn_preview()
        self.renderer.hide_drag_indicator()

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
        for item in self.renderer.drawn.note_items:
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
        self.renderer.update_weight_panel()

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
        self.renderer.update_activation_panel()

    @edit_transaction
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
            self.renderer.rebuild_group_buttons()
            self.renderer.apply_group_visibility()

    def _layers_in_container(self, container):
        """The layers drawn in *container*, each once."""
        all_layers = {l.name: l for l in self.gui.circuit.layers}
        found = {}
        for node_key, c in self._lay.node_container_map.items():
            lyr = all_layers.get(node_key.rsplit('_', 1)[0])
            if c == container and lyr is not None:
                found[id(lyr)] = lyr
        return list(found.values())

    @edit_transaction
    def _on_container_disable(self, container, disabled):
        if disabled:
            self._disabled_containers.add(container)
            self._hidden_containers.add(container)
        else:
            self._disabled_containers.discard(container)
            self._hidden_containers.discard(container)
        for l in self._layers_in_container(container):
            l.muted = disabled
        if self._compact_mode:
            self.build()
        else:
            self.renderer.rebuild_group_buttons()
            self.renderer.apply_group_visibility()

    @edit_transaction
    def _on_show_all(self):
        for container in list(self._disabled_containers):
            for l in self._layers_in_container(container):
                l.muted = False
        self._hidden_containers.clear()
        self._disabled_containers.clear()
        self.build()

    @edit_transaction
    def _on_enable_all(self):
        for container in list(self._disabled_containers):
            for l in self._layers_in_container(container):
                l.muted = False
        self._hidden_containers -= self._disabled_containers
        self._disabled_containers.clear()
        self.build()

    def _reload_motifs_palette(self):
        lay = self._motifs_palette_layout
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        names = sorted(name[:-5] for name, _p in data_paths.files('motifs', '*.json'))
        if not names:
            self._motifs_palette_widget.setVisible(False)
            return
        lbl = QLabel("Motifs")
        lbl.setStyleSheet(f"color:{C['muted']};font-size:9px;font-weight:bold;margin-top:4px;")
        lay.addWidget(lbl, 0, 0, 1, 2)
        chips = []
        for name in names:
            chip = PaletteChip(name, f'motif:{name}')
            chip.setStyleSheet(
                "QPushButton{padding:0 8px;font-size:9px;"
                "border:1px solid #C0A0C8;border-radius:3px;background:#EEE4F4;}"
                "QPushButton:hover{background:#E0D0F0;}"
            )
            chips.append(chip)
        _fill_grid(lay, chips, first_row=1)
        self._motifs_palette_widget.setVisible(True)

    def eventFilter(self, obj, ev):
        if obj is not self._gw:
            return False
        t = ev.type()
        if t == QEvent.Type.DragEnter:
            if not self._locked and ev.mimeData().hasText():
                ev.acceptProposedAction()
                return True
        elif t == QEvent.Type.DragMove:
            if not self._locked and ev.mimeData().hasText():
                vb_pt = self._vb.mapSceneToView(
                    self._gw.mapToScene(ev.position().toPoint())
                )
                # Notes are free-positioned — no column-snap preview, just the
                # raw drop point (no indicator to show).
                if ev.mimeData().text() != 'note':
                    self.renderer.show_drag_indicator(self.layout_engine.get_snap_x(vb_pt.x()))
                ev.acceptProposedAction()
                return True
        elif t == QEvent.Type.DragLeave:
            self.renderer.hide_drag_indicator()
            return True
        elif t == QEvent.Type.Drop:
            self.renderer.hide_drag_indicator()
            if not self._locked and ev.mimeData().hasText():
                vb_pt = self._vb.mapSceneToView(
                    self._gw.mapToScene(ev.position().toPoint())
                )
                if ev.mimeData().text() == 'note':
                    self.editing.on_note_drop(vb_pt)
                else:
                    snap_x = self.layout_engine.get_snap_x(vb_pt.x())
                    self.editing.on_palette_drop(ev.mimeData().text(), snap_x)
                ev.acceptProposedAction()
                return True
        return False

