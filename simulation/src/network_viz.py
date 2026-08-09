import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QCheckBox, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor

from sim_constants import C

from network_viz_layout import _LayoutMixin
from network_viz_render import _RenderMixin
from network_viz_dialogs import _DialogsMixin, PaletteChip
from network_viz_editing import _EditingMixin, NetworkViewBox
from network_viz_serialization import _SerializationMixin

class NetworkVisualizerWindow(_LayoutMixin, _RenderMixin, _DialogsMixin, _EditingMixin,
                               _SerializationMixin, QWidget):
    _ACTIVE_RGB   = (0.95, 0.45, 0.25)
    _EDGE_POS     = '#3A6FA8'
    _HL_EDGE      = '#E07828'
    _TD_EDGE      = (200, 130, 50)   # amber RGB for learnable (LearningLayerBase) connections
    _NODE_R       = 0.1125
    _MARKER_R     = 0.066
    _CROSS_BOW    = 0.35
    _INTERNAL_BOW = 0.22
    _PAD_X        = 0.10
    _RING_SCALE        = 0.82   # ring node size and ring radius relative to _NODE_R
    _DENSE_NODE_SCALE  = 0.68   # node size scale when layer has more than 4 neurons
    _MIDLINE_GAP       = 0.05   # extra gap inserted at y=0.5 for even-n layers
    _CAM_H_DATA        = 0.1125 # camera thumbnail height in data coords (must match _build_inner)
    _DENSE_THRESHOLD = 4   # max(ns, nt) > this → sampled edges + badge
    _DENSE_SAMPLE    = 3   # neurons sampled per side for dense connections
    _PAD_Y        = 0.10
    _CAM_WEIGHT   = 3.0    # slot weight for image nodes (camera thumbnails need extra vertical space)

    # Connection kind tokens — see rules/network_viz.md "Connection classifier".
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
    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        self.setWindowTitle("Network")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.StrongFocus)
        self.resize(800, 600)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Row 1: edit controls
        toolbar = QWidget()
        tb_lay  = QHBoxLayout(toolbar)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        self._btn_edit   = QPushButton("Edit")
        self._btn_save   = QPushButton("Save")
        self._btn_bonsai = QPushButton("Copy Bonsai")
        self._btn_svg    = QPushButton("Copy SVG")
        self._btn_undo   = QPushButton("Undo")
        self._btn_save.setEnabled(True)
        self._btn_undo.setEnabled(False)
        tb_lay.addWidget(self._btn_edit)
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
        self._z_cut         = None
        self._active_names  = None
        self._side_view      = None
        self._net3d_view    = None
        layout.addWidget(toolbar)

        # Rows 2+: draggable palette chips (shown only in edit mode)
        self._palette_bar = QWidget()
        pb_vlay = QVBoxLayout(self._palette_bar)
        pb_vlay.setContentsMargins(2, 1, 2, 1)
        pb_vlay.setSpacing(2)

        def _palette_row(label, chips):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{C['muted']};font-size:9px;min-width:40px;")
            row.addWidget(lbl)
            for chip in chips:
                row.addWidget(chip)
            row.addStretch()
            return row

        pb_vlay.addLayout(_palette_row("Sensors:", [
            PaletteChip(s.replace('Sensor', ''), f'sensor:{s}', sensor=True)
            for s in ['GradientSensor', 'ColorSensor',
                      'CollisionSensor', 'DistanceSensor', 'InteroceptiveSensor',
                      'ProprioceptiveSensor', 'WhiskerSensor', 'SkyCompassSensor',
                      'ManualBumpSensor', 'GrayCameraSensor', 'RGBCameraSensor']
        ]))
        pb_vlay.addLayout(_palette_row("Layers:", [
            PaletteChip(t.replace('Layer', ''), t)
            for t in ['LeakyLayer', 'ProductLayer', 'Leaky2dLayer', 'SumLayer', 'ConstantLayer', 'SineLayer',
                      'MatsuokaLayer', 'AdaptiveLayer', 'PulseLayer',
                      'RingAttractorLayer', 'Conv2dLayer', 'Reichardt2dLayer',
                      'AccumulatorLayer']
        ]))

        # Combined row: Body | Learning layers | Motifs (dynamic)
        combo_row = QHBoxLayout()
        combo_row.setContentsMargins(0, 0, 0, 0)
        combo_row.setSpacing(4)
        lbl_body = QLabel("Body:")
        lbl_body.setStyleSheet(f"color:{C['muted']};font-size:9px;min-width:40px;")
        combo_row.addWidget(lbl_body)
        combo_row.addWidget(PaletteChip('Body', 'joint'))
        lbl_note = QLabel("Note:")
        lbl_note.setStyleSheet(f"color:{C['muted']};font-size:9px;margin-left:6px;")
        combo_row.addWidget(lbl_note)
        combo_row.addWidget(PaletteChip('Note', 'note'))
        lbl_learn = QLabel("Learning:")
        lbl_learn.setStyleSheet(f"color:{C['muted']};font-size:9px;margin-left:6px;")
        combo_row.addWidget(lbl_learn)
        for t in ['TDLayer', 'DeltaLayer', 'ThreeFactorLayer']:
            combo_row.addWidget(PaletteChip(t.replace('Layer', ''), t))
        self._motifs_palette_widget = QWidget()
        self._motifs_palette_layout = QHBoxLayout(self._motifs_palette_widget)
        self._motifs_palette_layout.setContentsMargins(0, 0, 0, 0)
        self._motifs_palette_layout.setSpacing(4)
        combo_row.addWidget(self._motifs_palette_widget)
        combo_row.addStretch()
        pb_vlay.addLayout(combo_row)
        self._reload_motifs_palette()

        self._palette_bar.setVisible(False)
        layout.addWidget(self._palette_bar)

        # Column visibility panel (left sidebar, toggled by "Columns" button)
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

        self._btn_cols.clicked.connect(self._toggle_col_win)
        self._btn_weights.clicked.connect(self._toggle_weight_panel)
        self._btn_activations.clicked.connect(self._toggle_activation_panel)
        self._btn_top.clicked.connect(self._toggle_side_view)
        self._btn_net3d.clicked.connect(self._toggle_net3d)
        self._btn_notes.toggled.connect(self._toggle_notes_visible)

        # Live weight visualization panel (hidden by default, toggled by Weights button)
        self._weight_panel = QFrame()
        self._weight_panel.setFixedWidth(230)
        self._weight_panel.setFrameShape(QFrame.Shape.Box)
        self._weight_panel.setFrameShadow(QFrame.Shadow.Plain)
        self._weight_panel.setLineWidth(1)
        self._weight_panel.setStyleSheet("QFrame { border: 1px solid #B0C0D0; }")
        self._weight_panel.setVisible(False)
        _wp_outer = QVBoxLayout(self._weight_panel)
        _wp_outer.setContentsMargins(4, 4, 4, 4)
        _wp_outer.setSpacing(4)
        _wp_scroll = QScrollArea()
        _wp_scroll.setWidgetResizable(True)
        _wp_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _wp_scroll.setFrameShape(QFrame.Shape.NoFrame)
        _wp_inner = QWidget()
        self._weight_entries_layout = QVBoxLayout(_wp_inner)
        self._weight_entries_layout.setContentsMargins(0, 0, 0, 0)
        self._weight_entries_layout.setSpacing(6)
        self._weight_entries_layout.addStretch()
        _wp_scroll.setWidget(_wp_inner)
        _wp_outer.addWidget(_wp_scroll)
        self._weight_pinned = {}   # (src, tgt) -> WeightEntryWidget

        # Live activation visualization panel (hidden by default, toggled by
        # Activations button) — bar chart of a layer's current per-neuron output.
        self._activation_panel = QFrame()
        self._activation_panel.setFixedWidth(230)
        self._activation_panel.setFrameShape(QFrame.Shape.Box)
        self._activation_panel.setFrameShadow(QFrame.Shadow.Plain)
        self._activation_panel.setLineWidth(1)
        self._activation_panel.setStyleSheet("QFrame { border: 1px solid #B0C0D0; }")
        self._activation_panel.setVisible(False)
        _ap_outer = QVBoxLayout(self._activation_panel)
        _ap_outer.setContentsMargins(4, 4, 4, 4)
        _ap_outer.setSpacing(4)
        _ap_scroll = QScrollArea()
        _ap_scroll.setWidgetResizable(True)
        _ap_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _ap_scroll.setFrameShape(QFrame.Shape.NoFrame)
        _ap_inner = QWidget()
        self._activation_entries_layout = QVBoxLayout(_ap_inner)
        self._activation_entries_layout.setContentsMargins(0, 0, 0, 0)
        self._activation_entries_layout.setSpacing(6)
        self._activation_entries_layout.addStretch()
        _ap_scroll.setWidget(_ap_inner)
        _ap_outer.addWidget(_ap_scroll)
        self._activation_pinned = {}   # layer name -> ActivationEntryWidget

        # Content area: column panel + weight panel + activation panel + graph side by side
        _content = QWidget()
        _content_lay = QHBoxLayout(_content)
        _content_lay.setContentsMargins(0, 0, 0, 0)
        _content_lay.setSpacing(0)
        _content_lay.addWidget(self._col_panel)
        _content_lay.addWidget(self._weight_panel)
        _content_lay.addWidget(self._activation_panel)

        self._gw   = pg.GraphicsLayoutWidget()
        self._gw.setBackground(QColor(C['bg']))
        self._gw.setFocusPolicy(Qt.NoFocus)   # keep key events on NetworkVisualizerWindow
        self._vb   = NetworkViewBox(self)
        self._plot = self._gw.addPlot(viewBox=self._vb)
        # Store slot reference once so connect/disconnect always use the same object
        self._rebuild_edges_slot = self._rebuild_edges
        self._plot.setAspectLocked(True)
        self._plot.hideAxis('bottom')
        self._plot.hideAxis('left')
        self._plot.setMenuEnabled(False)
        self._vb.setMenuEnabled(False)
        self._vb.disableAutoRange()
        _content_lay.addWidget(self._gw, 1)
        layout.addWidget(_content, 1)
        self._gw.scene().sigMouseClicked.connect(
            self._on_scene_click, Qt.ConnectionType.QueuedConnection
        )

        self._drag_indicator = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen('#6AAAD4', width=2, style=Qt.DashLine),
        )
        self._drag_indicator.setVisible(False)
        self._plot.addItem(self._drag_indicator)

        self._h_drag_indicator = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen('#6AAAD4', width=2, style=Qt.DashLine),
        )
        self._h_drag_indicator.setVisible(False)
        self._plot.addItem(self._h_drag_indicator)

        self._conn_preview = pg.PlotDataItem(
            pen=pg.mkPen('#E07828', width=2, style=Qt.DashLine)
        )
        self._conn_preview.setVisible(False)
        self._conn_preview.setZValue(20)
        self._plot.addItem(self._conn_preview)

        self._pen_default  = pg.mkPen(C['dark'], width=1.5)
        self._pen_selected = pg.mkPen(C['primary'], width=6)
        self._pen_multi    = pg.mkPen('#E07828', width=6)
        self._pen_osc      = pg.mkPen('#00AAAA', width=3)   # teal = tracked in oscilloscope
        self._pen_muted    = pg.mkPen('#909090', width=1.5, style=Qt.DashLine)

        self._redrawing        = False   # reentrancy guard for _redraw_nodes
        self._all_scatter      = None  # single ScatterPlotItem for all nodes
        self._ring_scatter     = None  # hollow rings: receiver rings + source waves
        self._deriv_scatter    = None  # small dot overlay for derivative=True nodes
        self._wave_phase       = {}    # {nt_name: float 0→1}  advances when active
        self._src_nodes        = {}    # {node_key: nt_name}
        self._rcv_nodes        = {}    # {node_key: [nt_name, ...]}
        self._mod_colors       = {}    # {nt_name: (r,g,b)} built at build time
        self._mod_pens         = {}    # {nt_name: QPen} cached modulator border pens
        self._spot_names       = []   # ordered node keys
        self._spot_base_rgb    = {}   # name → (r,g,b) floats [0,1]
        self._spot_alpha       = {}   # name → float [0,1]  (edge highlight fade)
        self._spot_visible     = {}   # name → bool  (group hide)
        self._spot_pen_override = {}  # name → 'selected' | 'multi'
        self._positions        = {}
        self._selected         = None
        self._selected_edge    = None  # (src_name, tgt_name) or None
        self._conn_from        = None  # node key being dragged for connection
        self._range_signal_connected = False
        self._rebuilding_edges = False
        self._building         = False
        self._edit_mode        = False
        self._weight_params    = {}   # (src, tgt) → last-used pattern params dict
        self._x_unit           = 1.0
        self._container_ids    = []
        self._container_x_map  = {}
        self._palette_x        = 0.5
        self._palette_y        = 1.22
        self._edge_items        = []
        self._edge_items_tagged = []  # 6-tuples: (item, sn, tn, excitatory, is_curve, original_pen)
        self._edge_params       = []  # raw draw params, replayed on zoom to fix rim coords
        self._panel_items       = []
        self._panel_rect_map    = {}   # container → PlotDataItem (panel background rect)
        self._container_label_items = {}   # container → TextItem (annotation above rect)
        self._container_labels  = {}   # container identity (occupant name set, '|'-joined) → str
        self._text_items        = []
        self._text_map          = {}   # node_key → TextItem
        self._node_container_map = {}   # node_key → container (int)
        self._hidden_containers   = set()   # containers hidden from viz but still computed
        self._disabled_containers = set()   # containers excluded from computation (and hidden)
        self._sensor_nodes      = set()   # node keys that are sensor outputs (hollow nodes)
        self._sensor_pens       = {}      # node_key → QPen (palette border)
        self._sensor_active_rgb = {}      # node_key → (r,g,b) palette colour for activity
        self._camera_items      = {}      # sensor.name → pg.ImageItem (CameraSensor only)
        self._camera_rects      = {}      # same keys → (x, y, w, h) for re-anchoring after setImage
        self._image_node_items  = {}      # node_key → list of plot items (circle, img, label, …)
        self._highlighted_node  = None
        self._node_just_clicked = False   # prevents scene click from clearing a fresh highlight
        self._multi_selected    = set()   # node keys selected via shift+click

        self._note_items        = []      # flat list of all note graphics items (removed each rebuild)
        self._note_item_map     = {}      # id(note) → {'items': [...]} for incremental drag updates
        self._selected_note     = None    # Note currently selected (for Delete key)
        self._dragging_note     = None    # Note being drag-moved, or None
        self._notes_visible     = True    # master show/hide-all toggle, independent of per-note collapse

        self._compact_mode = True
        self._undo_stack = []
        self._btn_edit.clicked.connect(self._toggle_edit)
        self._btn_save.clicked.connect(self._save_circuit)
        self._btn_bonsai.clicked.connect(self._copy_bonsai)
        self._btn_svg.clicked.connect(self._copy_svg)
        self._btn_undo.clicked.connect(self._undo)
        self._z_slider.valueChanged.connect(self._on_z_slider_changed)
        from PySide6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence.StandardKey.Undo, self).activated.connect(self._undo)

        self._gw.setAcceptDrops(True)
        self._gw.installEventFilter(self)

        QTimer.singleShot(50, self.build)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(100)
        self._refresh_timer.timeout.connect(self._on_refresh_timer)
        self._refresh_timer.start()

