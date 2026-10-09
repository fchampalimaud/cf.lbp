"""
network_viz_render.py — rendering for the network visualizer.

NetworkRenderer (the window's `renderer`) holds every method that creates or
mutates pyqtgraph/Qt graphics items, and owns them (SceneItems). It draws the LayoutResult (self._lay) that LayoutEngine.compute()
returns — this file never computes column/container/position layout itself.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolTip
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter

from sim_constants import C, _CHAN_PALETTE

from network_viz_layout import (_sensor_is_lateralized, _mirror_name, _tile_conv_filters,
                                _reversed_idx, _ctrl_pt, _bezier_pts, _hemisphere, _signed_bow)
from lateral import SIDES, side_of, base_name, half_names, parent_sensor
from network_viz_dialogs import _small_bold_font

# Image nodes (cameras, image layers): thumbnail size in data coordinates,
# independent of the frame size, and the thumbnail height in pixels.
_IMAGE_W, _IMAGE_H, _IMAGE_DISP_H = 0.15, 0.1125, 32

def _self_loop_pts(p0, r):
    """Self-loop: small arc from 12 o'clock to 2 o'clock on the neuron rim,
    bowing outward via a quadratic bezier with control point at 1 o'clock."""
    p_start = (p0[0] + r * np.cos(np.radians(90)),
               p0[1] + r * np.sin(np.radians(90)))
    p_end   = (p0[0] + r * np.cos(np.radians(30)),
               p0[1] + r * np.sin(np.radians(30)))
    ctrl    = (p0[0] + (r + r * 1.5) * np.cos(np.radians(60)),
               p0[1] + (r + r * 1.5) * np.sin(np.radians(60)))
    t  = np.linspace(0, 1, 30)
    xs = ((1-t)**2*p_start[0] + 2*(1-t)*t*ctrl[0] + t**2*p_end[0]).tolist()
    ys = ((1-t)**2*p_start[1] + 2*(1-t)*t*ctrl[1] + t**2*p_end[1]).tolist()
    return xs, ys


def _rim_point(p, ctrl, other, r):
    """Point on the circle of radius r around p, aimed at ctrl (or at the
    other endpoint when ctrl coincides with p)."""
    d = np.hypot(ctrl[0] - p[0], ctrl[1] - p[1])
    if d > 1e-6:
        return (p[0] + r * (ctrl[0] - p[0]) / d,
                p[1] + r * (ctrl[1] - p[1]) / d)
    nx, ny = other[0] - p[0], other[1] - p[1]
    nd = np.hypot(nx, ny) + 1e-9
    return (p[0] + r * nx / nd, p[1] + r * ny / nd)


@dataclass
class SceneItems:
    """Everything one build() draws, plus the per-node caches the refresh
    timer reads. Owned by NetworkRenderer, kept as renderer.drawn; a rebuild
    removes the old items from the plot and starts a fresh SceneItems."""
    all_scatter: object = None                                # single ScatterPlotItem for all nodes
    ring_scatter: object = None                               # hollow rings: receiver rings + source waves
    deriv_scatter: object = None                              # small dot overlay for derivative=True nodes
    spot_names: list = field(default_factory=list)            # ordered node keys
    spot_base_rgb: dict = field(default_factory=dict)         # name → (r,g,b) floats [0,1]
    spot_alpha: dict = field(default_factory=dict)            # name → float [0,1]  (edge highlight fade)
    spot_visible: dict = field(default_factory=dict)          # name → bool  (group hide)
    text_items: list = field(default_factory=list)
    text_map: dict = field(default_factory=dict)              # node_key → TextItem
    sensor_pens: dict = field(default_factory=dict)           # node_key → QPen (palette border)
    sensor_active_rgb: dict = field(default_factory=dict)     # node_key → (r,g,b) palette colour for activity
    edge_items: list = field(default_factory=list)
    edge_items_tagged: list = field(default_factory=list)     # 6-tuples: (item, sn, tn, excitatory, is_curve, original_pen)
    edge_params: list = field(default_factory=list)           # raw draw params, replayed on zoom to fix rim coords
    panel_items: list = field(default_factory=list)
    panel_rect_map: dict = field(default_factory=dict)        # container → PlotDataItem (panel background rect)
    container_label_items: dict = field(default_factory=dict)  # container → TextItem (annotation above rect)
    container_note_items: dict = field(default_factory=dict)  # container → [ScatterPlotItem, TextItem] (bottom-right glyph)
    container_note_icon_pos: dict = field(default_factory=dict)  # container → (x, y) of the glyph, for hit-testing
    camera_items: dict = field(default_factory=dict)          # sensor.name → pg.ImageItem (CameraSensor only)
    camera_rects: dict = field(default_factory=dict)          # same keys → (x, y, w, h) for re-anchoring after setImage
    image_node_items: dict = field(default_factory=dict)      # node_key → list of plot items (circle, img, label, …)
    note_items: list = field(default_factory=list)            # flat list of all note graphics items (removed each rebuild)
    note_item_map: dict = field(default_factory=dict)         # id(note) → {'items': [...]} for incremental drag updates
    mod_colors: dict = field(default_factory=dict)            # {nt_name: (r,g,b)} built at build time
    mod_pens: dict = field(default_factory=dict)              # {nt_name: QPen} cached modulator border pens
    src_nodes: dict = field(default_factory=dict)             # {node_key: nt_name}
    rcv_nodes: dict = field(default_factory=dict)             # {node_key: [nt_name, ...]}
    wave_phase: dict = field(default_factory=dict)            # {nt_name: float 0→1}  advances when active
    deriv_node_positions: list = field(default_factory=list)  # node positions that get a derivative dot
    img_node_keys: set = field(default_factory=set)           # node keys of image nodes (edges end at their ring)


class NetworkRenderer:
    """Draws the network: nodes, edges, panels, notes, image thumbnails, and
    the live refresh (activity colours, panels) — from the window's
    LayoutResult. Owns the drawn items (`drawn`, a SceneItems)."""

    _NOTE_FILL   = '#F5E08A'
    _NOTE_BORDER = '#C8A030'

    _FADE_OPACITY = 0.12


    def __init__(self, win):
        self.win = win
        self.drawn = SceneItems()          # replaced by every draw()
        self.redrawing        = False      # reentrancy guard for redraw_nodes
        self.rebuilding_edges = False
        self.range_signal_connected = False
        # Stored once so connect/disconnect always use the same slot object.
        self.rebuild_edges_slot = self.rebuild_edges
        self.pen_default  = pg.mkPen(C['dark'], width=1.5)
        self.pen_selected = pg.mkPen(C['primary'], width=6)
        self.pen_multi    = pg.mkPen('#E07828', width=6)
        self.pen_osc      = pg.mkPen('#00AAAA', width=3)   # teal = tracked in oscilloscope
        self.pen_muted    = pg.mkPen('#909090', width=1.5, style=Qt.DashLine)

    def create_overlays(self):
        """Drag indicators (column snap) and the connection-drag preview line,
        added to the window's plot once it exists."""
        plot = self.win._plot
        self.drag_indicator = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen('#6AAAD4', width=2, style=Qt.DashLine),
        )
        self.drag_indicator.setVisible(False)
        plot.addItem(self.drag_indicator)

        self.h_drag_indicator = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen('#6AAAD4', width=2, style=Qt.DashLine),
        )
        self.h_drag_indicator.setVisible(False)
        plot.addItem(self.h_drag_indicator)

        self.conn_preview = pg.PlotDataItem(
            pen=pg.mkPen('#E07828', width=2, style=Qt.DashLine)
        )
        self.conn_preview.setVisible(False)
        self.conn_preview.setZValue(20)
        plot.addItem(self.conn_preview)

    def on_refresh_timer(self):
        if not self.win._building and self.drawn.all_scatter is not None:
            brain = getattr(self.win.gui, 'brain', None)
            if brain is not None:
                try:
                    self.redraw_nodes(brain)
                except Exception:
                    pass
                try:
                    self._update_anim(brain)
                except Exception:
                    pass
                try:
                    self._update_camera_nodes()
                except Exception:
                    pass
                try:
                    self.update_weight_panel()
                except Exception:
                    pass
                try:
                    self.update_activation_panel()
                except Exception:
                    pass

    def _update_camera_nodes(self):
        for obj in list(self.win.gui.circuit.layers) + list(self.win.gui.circuit.sensors):
            if not obj.is_image_node:
                continue
            for key, data in obj.thumbnail_frames(32):
                item = self.drawn.camera_items.get(key)
                if item is None:
                    continue
                item.setImage(data, axisOrder='row-major')
                rect = self.drawn.camera_rects.get(key)
                if rect is not None:
                    item.setRect(*rect)

    def _update_anim(self, brain):
        if self.drawn.ring_scatter is None:
            return

        # Activation per neuromodulator substance
        nt_activation = {}
        if brain is not None:
            for layer in getattr(self.win.gui.circuit, 'layers', []):
                nt = getattr(layer, 'neuromodulator_transmitter', None)
                if not nt or nt not in self.drawn.mod_colors:
                    continue
                attr = getattr(brain, layer.name, None)
                if attr is None:
                    continue
                arr = np.atleast_1d(attr.output if hasattr(attr, 'output') else attr)
                act = float(np.clip(np.mean(arr), 0, 1)) if arr.size > 0 else 0.0
                nt_activation[nt] = max(nt_activation.get(nt, 0.0), act)
            for sensor in getattr(self.win.gui.circuit, 'sensors', []):
                nt = getattr(sensor, 'neuromodulator_transmitter', None)
                if not nt or nt not in self.drawn.mod_colors:
                    continue
                val = getattr(brain, sensor.name, None)
                if val is None:
                    continue
                arr = np.atleast_1d(val)
                act = float(np.clip(np.mean(arr), 0, 1)) if arr.size > 0 else 0.0
                nt_activation[nt] = max(nt_activation.get(nt, 0.0), act)

        # Advance wave phase only when the source is active
        for nt, act in nt_activation.items():
            if act >= 0.05:
                self.drawn.wave_phase[nt] = (self.drawn.wave_phase.get(nt, 0.0) + 0.07) % 1.0

        r_node    = self.win._NODE_R
        ring_spots = []

        # Receiver satellites — small filled dot whose edge touches the node edge
        SAT_SIZE = 9               # dot diameter in pixels
        SAT_BASE = np.pi / 4       # starting angle (upper-right)
        dx       = self.win._vb.viewPixelSize()[0]          # data units per pixel
        SAT_R    = (r_node * 120 + SAT_SIZE / 2) * dx  # pixel-node-radius + sat px-radius → scene units
        for node_key, nt_list in self.drawn.rcv_nodes.items():
            if not self.drawn.spot_visible.get(node_key, True):
                continue
            pos = self.win._lay.positions.get(node_key)
            if pos is None:
                continue
            x, y = pos
            n = len(nt_list)
            for idx, mod_name in enumerate(nt_list):
                angle = SAT_BASE + idx * (2 * np.pi / n)
                sx    = x + SAT_R * np.cos(angle)
                sy    = y + SAT_R * np.sin(angle)
                rgb   = self.drawn.mod_colors[mod_name]
                color = QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                ring_spots.append({
                    'pos':   (sx, sy),
                    'size':  SAT_SIZE,
                    'brush': pg.mkBrush(color),
                    'pen':   pg.mkPen(None),
                })

        # Source waves — 3 concentric expanding rings, activation-gated
        N_WAVES   = 3
        MAX_EXTRA = 60   # px the wave expands beyond the node edge
        for node_key, nt in self.drawn.src_nodes.items():
            if not self.drawn.spot_visible.get(node_key, True):
                continue
            pos = self.win._lay.positions.get(node_key)
            if pos is None:
                continue
            act = nt_activation.get(nt, 0.0)
            if act < 0.05:
                continue
            x, y  = pos
            rgb   = self.drawn.mod_colors[nt]
            base_color = QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            phase = self.drawn.wave_phase.get(nt, 0.0)
            for i in range(N_WAVES):
                p     = (phase + i / N_WAVES) % 1.0
                alpha = int(220 * (1 - p) * act)
                size  = r_node * 240 + (4 + p * MAX_EXTRA)
                c     = QColor(base_color)
                c.setAlpha(alpha)
                ring_spots.append({
                    'pos': (x, y),
                    'size': size,
                    'brush': pg.mkBrush(None),
                    'pen': pg.mkPen(c, width=2),
                })

        self.drawn.ring_scatter.setData(spots=ring_spots)

    @staticmethod
    def hex_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def _image_node_keys(self):
        """Position keys of image-display nodes (cameras / camera halves,
        Leaky2dLayer, Conv2dLayer pool='none', viz_n == 1 layers)."""
        keys = set()
        for s in self.win.gui.circuit.sensors:
            if s.is_image_node:
                if _sensor_is_lateralized(s, self.win.gui.circuit):
                    keys.update(f'{half}_0' for half in half_names(s.name))
                else:
                    keys.add(f'{s.name}_0')
        for l in self.win.gui.circuit.layers:
            if l.is_image_node or getattr(l, 'viz_n', None) == 1:
                keys.add(f'{l.name}_0')
        return keys

    def _draw_edges(self, positions, n_map):
        # _draw_edge uses this to offset arc endpoints to the image circle rim
        # rather than the neuron-dot rim.
        self.drawn.img_node_keys = self._image_node_keys()

        # When a specific neuron is selected, filter edges to only show connections
        # involving that neuron's index (rows/cols in the weight matrix).
        sel_layer = sel_idx = None
        if self.win.editing.sel.node:
            parts = self.win.editing.sel.node.rsplit('_', 1)
            if len(parts) == 2 and parts[1].lstrip('-').isdigit():
                sel_layer, sel_idx = parts[0], int(parts[1])

        by_target  = defaultdict(list)
        tgt_hem_map = {}

        _active   = self.win._lay.active_names
        _subsumed = self.win._lay.subsumed_by
        for conn in self.win.gui.circuit.connections:
            src, tgt, W = conn.src, conn.tgt, conn.W
            src_draw     = _subsumed.get(src, src)
            tgt_draw     = _subsumed.get(tgt, tgt)
            _is_remapped = (src_draw != src) or (tgt_draw != tgt)
            if _active is not None:
                if src_draw not in _active or tgt_draw not in _active:
                    continue
                if _is_remapped and src_draw == tgt_draw:
                    continue   # both collapsed to same subsumer — skip self.win-loop

            # Connections involving hidden (subsumed) layers are not drawn —
            # the z-cut view shows only the active layers and their direct connections.
            if _is_remapped:
                continue

            W  = np.asarray(W, dtype=float)

            ns = n_map.get(src, 1)
            nt = n_map.get(tgt, 1)

            # Resolve endpoint objects for the classifier (src may be layer or sensor).
            src_obj = (next((l for l in self.win.gui.circuit.layers  if l.name == src), None)
                       or next((s for s in self.win.gui.circuit.sensors if s.name == src), None))
            tgt_obj = next((l for l in self.win.gui.circuit.layers if l.name == tgt), None)

            kind = self.win.layout_engine.connection_kind(src_obj, tgt_obj, W, ns, nt)

            if kind == self.win._CK_TEACH:
                self._draw_teach_connection(src, tgt, W, ns, nt, positions,
                                            sel_layer=sel_layer, sel_idx=sel_idx)
            elif kind == self.win._CK_TD:
                self._draw_td_connection(src, tgt, W, ns, nt, positions,
                                         sel_layer=sel_layer, sel_idx=sel_idx)
            elif kind == self.win._CK_CONV4D:
                self._draw_conv4d_connection(src, tgt, W, positions, n_map,
                                             sel_layer, sel_idx)
            elif kind == self.win._CK_DENSE:
                self._draw_dense_connection(src, tgt, W, ns, nt, positions,
                                            sel_layer=sel_layer, sel_idx=sel_idx)
            else:
                self._collect_arcs(src, tgt, W, ns, nt, kind == self.win._CK_THIN, positions,
                                   sel_layer, sel_idx, by_target, tgt_hem_map)

        self._draw_collected_arcs(by_target, tgt_hem_map, positions)
        self._draw_internal_edges(positions, sel_layer, sel_idx)

    def _collect_arcs(self, src, tgt, W, ns, nt, is_thin, positions, sel_layer, sel_idx,
                      by_target, tgt_hem_map):
        """THIN or THICK connection: one arc per non-zero weight, accumulated per
        target node in by_target (drawn together so arcs into a node fan out).

        A combined lateralized source is detected first. Two cases:
          A) Conv2dLayer pair: src has _lateral_pair → partner nodes from partner layer.
          B) Joint-pair sensor half: src ends _L/_R from a pair sensor → partner nodes
             from the mirror sensor half (e.g. sensor0_R for src=sensor0_L)."""
        _src_lyr_d   = next((l for l in self.win.gui.circuit.layers if l.name == src), None)
        _pair_nm_d   = getattr(_src_lyr_d, 'lateral_pair', None) if _src_lyr_d else None
        _pair_lyr_d  = None
        _pair_sensor_half_d = None
        _n_L_d       = ns
        if _pair_nm_d and W.ndim == 2:
            _p = next((l for l in self.win.gui.circuit.layers if l.name == _pair_nm_d), None)
            if _p and W.shape[1] == (_src_lyr_d.n or 0) + (_p.n or 0):
                _pair_lyr_d = _p
                _n_L_d = _src_lyr_d.n or 0
                ns = W.shape[1]
        elif W.ndim == 2 and side_of(src):
            _src_snsr_d = parent_sensor(self.win.gui.circuit.sensors, src)
            if (_src_snsr_d is not None
                    and _sensor_is_lateralized(_src_snsr_d, self.win.gui.circuit)
                    and W.shape[1] == ns * 2):
                _pair_sensor_half_d = _mirror_name(src)
                _n_L_d = ns
                ns = W.shape[1]
        if W.ndim == 2 and (W.shape[0] < nt or W.shape[1] < ns):
            return  # stale W matrix; skip rather than IndexError
        for i in range(nt):
            if sel_idx is not None and tgt == sel_layer and i != sel_idx:
                continue
            tn      = f'{tgt}_{i}'
            _p_tn   = positions.get(tn)
            tgt_hem = ((_p_tn[1] >= 0.5) if nt > 1 else None) if _p_tn is not None \
                      else _hemisphere(i, nt)
            tgt_hem_map[tn] = tgt_hem
            for j in range(ns):
                if sel_idx is not None and src == sel_layer and j != sel_idx:
                    continue
                w = float(W[i, j]) if W.ndim == 2 else float(W.flat[0])
                if abs(w) < 1e-10:
                    continue
                # For combined pair connections, j≥n_L_d → nodes from the R half.
                # R neurons appear in visual top-to-bottom order, which is reverse
                # index order: column n_L → R_(n_R-1), column n_L+1 → R_(n_R-2), etc.
                if _pair_lyr_d and j >= _n_L_d:
                    _n_R_d = _pair_lyr_d.n or 1
                    sn = f'{_pair_nm_d}_{_reversed_idx(j - _n_L_d, _n_R_d, "R")}'
                elif _pair_sensor_half_d and j >= _n_L_d:
                    _n_R_d = _n_L_d   # each side has the same n
                    sn = f'{_pair_sensor_half_d}_{_reversed_idx(j - _n_L_d, _n_R_d, "R")}'
                else:
                    sn = f'{src}_{j}'
                if sn not in positions or tn not in positions:
                    continue
                _p_sn   = positions.get(sn)
                src_hem = ((_p_sn[1] >= 0.5) if ns > 1 else None) if _p_sn is not None \
                          else _hemisphere(j, ns)
                by_target[tn].append((sn, w, src_hem, is_thin))

    def _draw_collected_arcs(self, by_target, tgt_hem_map, positions):
        """Draw the arcs collected by _collect_arcs, fanned out per target node."""
        for tn, incoming in by_target.items():
            p1      = positions[tn]
            k       = len(incoming)
            tgt_hem = tgt_hem_map[tn]
            incoming_sorted = sorted(incoming, key=lambda t: positions[t[0]][1], reverse=True)
            for idx, (sn, w, src_hem, is_thin) in enumerate(incoming_sorted):
                p0  = positions[sn]
                t   = (idx / (k - 1) - 0.5) if k > 1 else 0.0
                sb  = _signed_bow(src_hem, tgt_hem, self.win._CROSS_BOW,
                                       (p0[1] + p1[1]) / 2)
                _lw = 0.6 if is_thin else 3.0
                self._draw_edge(p0, p1, w, sb, ctrl_perp=0.18 * t, sn=sn, tn=tn,
                                lw=_lw, mark=not is_thin)

    def _draw_internal_edges(self, positions, sel_layer, sel_idx):
        """Within-layer edges declared by a layer's internal_edges()."""
        for layer in self.win.gui.circuit.layers:
            if not hasattr(layer, 'internal_edges'):
                continue
            n      = layer.n or 2
            is_ring = getattr(layer, 'viz_layout', None) == 'ring'
            for fi, ti, w in layer.internal_edges():
                if sel_idx is not None and layer.name == sel_layer:
                    if fi != sel_idx and ti != sel_idx:
                        continue
                sn = f'{layer.name}_{fi}'
                tn = f'{layer.name}_{ti}'
                if sn not in positions or tn not in positions:
                    continue
                # Bow sign from the *unordered* pair: a reciprocal edge (e.g. a
                # Matsuoka pair's mutual inhibition, drawn as (0,1,-w) AND (1,0,-w))
                # must get the same sign both times. Keying it off (fi, ti) in
                # draw order instead flips sign on the reverse edge, and with the
                # endpoints swapped too that lands on the identical curve — the
                # second edge silently draws right on top of the first.
                lo, hi = (fi, ti) if fi < ti else (ti, fi)
                ln, hn = f'{layer.name}_{lo}', f'{layer.name}_{hi}'
                p_lo, p_hi = positions[ln], positions[hn]
                if is_ring:
                    # Bow each edge outward from the ring centre.
                    # _ctrl_pt with bow>0 shifts the midpoint by (cdy, -cdx),
                    # i.e. 90° CW from the chord. Choose the sign so that shift
                    # aligns with the outward direction (centre → chord midpoint).
                    cx  = getattr(layer, '_ring_cx', 0.5)
                    cy  = getattr(layer, '_ring_cy', 0.5)
                    cdx, cdy = p_hi[0] - p_lo[0], p_hi[1] - p_lo[1]
                    odx = (p_lo[0] + p_hi[0]) / 2 - cx   # chord-midpoint – centre
                    ody = (p_lo[1] + p_hi[1]) / 2 - cy
                    # dot( (cdy,-cdx), (odx,ody) ) > 0 → bow>0 is already outward
                    sb = self.win._INTERNAL_BOW if (cdy * odx - cdx * ody) >= 0 \
                         else -self.win._INTERNAL_BOW
                else:
                    sb = _signed_bow(_hemisphere(lo, n), _hemisphere(hi, n),
                                          self.win._INTERNAL_BOW,
                                          (p_lo[1] + p_hi[1]) / 2)
                self._draw_edge(positions[sn], positions[tn], w, sb, sn=sn, tn=tn,
                               lw=0.6, mark=False)

    def _draw_conv4d_connection(self, src, tgt, W, positions, n_map, sel_layer, sel_idx):
        """Draw a Conv2d 4-D kernel connection: one arc per filter from {src}_0.

        lw=0.6 when n_filters > 4 (THIN rule), lw=3.0 otherwise (THICK rule).
        Lateralized R-side source reverses the target neuron ordering to mirror the L layout.
        """
        n_filters, _, kH, kW = W.shape
        p_s = positions.get(f'{src}_0')
        if p_s is None:
            return
        src_sensor = next((s for s in self.win.gui.circuit.sensors if s.name == src), None)
        if src_sensor is None:
            src_sensor = parent_sensor(self.win.gui.circuit.sensors, src)
        is_lat_R = (side_of(src) == 'R'
                    and src_sensor is not None
                    and _sensor_is_lateralized(src_sensor, self.win.gui.circuit))
        tgt_n = n_map.get(tgt, n_filters)
        lw    = 0.6 if n_filters > 4 else 3.0
        mark  = n_filters <= 4
        for i in range(n_filters):
            tgt_idx = _reversed_idx(i, tgt_n, 'R' if is_lat_R else 'L')
            if sel_idx is not None and tgt == sel_layer and sel_idx != tgt_idx:
                continue
            tn = f'{tgt}_{tgt_idx}'
            p_t = positions.get(tn)
            if p_t is None:
                continue
            cw = abs(float(W[i, :, kH // 2, kW // 2].mean()))
            if cw < 1e-10:
                cw = float(np.abs(W[i]).max())
            mid_y = (p_s[1] + p_t[1]) / 2
            sb = self.win._CROSS_BOW if mid_y >= 0.5 else -self.win._CROSS_BOW
            self._draw_edge(p_s, p_t, cw, sb, sn=f'{src}_0', tn=tn, lw=lw, mark=mark)

    def _draw_td_connection(self, src, tgt, W, ns, nt, positions,
                            sel_layer=None, sel_idx=None):
        """Draw all connections into a TDLayer — amber, always visible, ghost lines for zero weights."""
        amber_active = self.win._TD_EDGE + (210,)   # solid learned weight
        amber_zero   = self.win._TD_EDGE + (100,)   # ghost: unlearned slot (slightly transparent)
        for i in range(nt):
            if sel_idx is not None and tgt == sel_layer and i != sel_idx:
                continue
            tn_key = f'{tgt}_{i}'
            if tn_key not in positions:
                continue
            p_t = positions[tn_key]
            for j in range(ns):
                if sel_idx is not None and src == sel_layer and j != sel_idx:
                    continue
                sn_key = f'{src}_{j}'
                if sn_key not in positions:
                    continue
                w = float(W[i, j]) if W.ndim == 2 else float(W.flat[0])
                p_s = positions[sn_key]
                mid_y = (p_s[1] + p_t[1]) / 2
                sb = self.win._CROSS_BOW if mid_y >= 0.5 else -self.win._CROSS_BOW
                if abs(w) < 1e-10:
                    self._draw_edge(p_s, p_t, 1.0, sb, sn=sn_key, tn=tn_key,
                                    lw=1.0, color=amber_zero, style=Qt.DashLine,
                                    mark=False, tgt_gap=0.4)
                else:
                    st = Qt.SolidLine if w >= 0 else Qt.DashLine
                    self._draw_edge(p_s, p_t, w, sb, sn=sn_key, tn=tn_key,
                                    lw=3.0, color=amber_active, style=st,
                                    mark=True, marker_style='tick', tgt_gap=0.4)

    def _draw_teach_connection(self, src, tgt, W, ns, nt, positions,
                              sel_layer=None, sel_idx=None):
        """Draw a SnapshotLayer's teach connection — solid green, fixed style.

        Unlike _draw_td_connection, there is no ghost-for-zero-weight and no
        sign-based solid/dashed distinction: this connection isn't a trained
        Hebbian weight and isn't a signed synapse in the functional sense —
        it's a fixed structural readout (see SnapshotLayer.help_text). Every
        edge gets the same appearance regardless of W's actual value.
        """
        teach_color = self.win._TEACH_EDGE + (210,)
        for i in range(nt):
            if sel_idx is not None and tgt == sel_layer and i != sel_idx:
                continue
            tn_key = f'{tgt}_{i}'
            if tn_key not in positions:
                continue
            p_t = positions[tn_key]
            for j in range(ns):
                if sel_idx is not None and src == sel_layer and j != sel_idx:
                    continue
                sn_key = f'{src}_{j}'
                if sn_key not in positions:
                    continue
                w = float(W[i, j]) if W.ndim == 2 else float(W.flat[0])
                p_s = positions[sn_key]
                mid_y = (p_s[1] + p_t[1]) / 2
                sb = self.win._CROSS_BOW if mid_y >= 0.5 else -self.win._CROSS_BOW
                self._draw_edge(p_s, p_t, w, sb, sn=sn_key, tn=tn_key,
                                lw=3.0, color=teach_color, style=Qt.SolidLine,
                                mark=False, tgt_gap=0.4)

    def _draw_dense_connection(self, src, tgt, W, ns, nt, positions,
                               sel_layer=None, sel_idx=None):
        """Draw all edges thin (solid excitatory, dashed inhibitory) for dense connections.

        Bow direction is determined purely by whether the edge midpoint is above or
        below y=0.5, giving visually symmetric arcs across the midline regardless of
        how neurons are indexed (important for ring layouts).
        When sel_layer/sel_idx are set, only draw edges involving that specific neuron.
        """
        for i in range(nt):
            if sel_idx is not None and tgt == sel_layer:
                if src == sel_layer:
                    pass  # self.win-connection: decide per (i,j) below
                elif i != sel_idx:
                    continue
            tn_key = f'{tgt}_{i}'
            if tn_key not in positions:
                continue
            p_t = positions[tn_key]
            for j in range(ns):
                if sel_idx is not None:
                    if src == sel_layer and tgt == sel_layer:
                        # self.win-connection: show only edges touching the selected neuron
                        if i != sel_idx and j != sel_idx:
                            continue
                    elif src == sel_layer and j != sel_idx:
                        continue
                sn_key = f'{src}_{j}'
                if sn_key not in positions:
                    continue
                w = float(W[i, j]) if W.ndim == 2 else float(W.flat[0])
                if abs(w) < 1e-10:
                    continue
                p_s = positions[sn_key]
                mid_y = (p_s[1] + p_t[1]) / 2
                sb = self.win._CROSS_BOW if mid_y >= 0.5 else -self.win._CROSS_BOW
                self._draw_edge(p_s, p_t, w, sb, sn=sn_key, tn=tn_key, lw=0.6)

    def rebuild_edges(self):
        """Redraw all edges at the current viewPixelSize — called on every zoom/pan."""
        if self.win._building or self.rebuilding_edges:
            return
        self.rebuilding_edges = True
        try:
            saved = list(self.drawn.edge_params)
            for item in self.drawn.edge_items:
                try:
                    self.win._plot.removeItem(item)
                except Exception:
                    pass
            self.drawn.edge_items = []
            self.drawn.edge_items_tagged = []
            self.drawn.edge_params = []          # _draw_edge will repopulate
            for p in saved:
                self._draw_edge(*p)
            self.win.editing.apply_edge_highlight()    # restore any active selection highlight
            if self.win._hidden_containers:
                self.apply_group_visibility()
        finally:
            self.rebuilding_edges = False

    def _draw_edge(self, p0, p1, weight, signed_bow, ctrl_perp=0.0, sn=None, tn=None, lw=None, color=None, style=None, mark=None, marker_style='circle', tgt_gap=0.0):
        self.drawn.edge_params.append((p0, p1, weight, signed_bow, ctrl_perp, sn, tn, lw, color, style, mark, marker_style, tgt_gap))
        excitatory = weight >= 0
        color      = color if color is not None else self.win._EDGE_POS
        lw         = lw if lw is not None else 3.0
        style      = style if style is not None else (Qt.SolidLine if excitatory else Qt.DashLine)

        # r_vis: actual scene-unit node radius at current zoom (nodes are pxMode=True,
        # size=NODE_R*240px, so pixel radius = NODE_R*120 → scene units = NODE_R*120*dy).
        try:
            _, _dy = self.win._vb.viewPixelSize()
        except Exception:
            _dy = 1.0 / 120.0
        r = self.win._NODE_R * 120 * _dy

        # Image-display nodes (cameras, Leaky2dLayer, Conv2dLayer pool='none') have a
        # much larger circular border drawn in data coords.  Use that radius for the
        # endpoint offset so arcs terminate at the circle rim, not at the node centre.
        _IMAGE_R = 0.12   # data units — just outside the image node's ring (see _add_image_node)
        _img = self.drawn.img_node_keys
        r_src = _IMAGE_R if sn in _img else r
        r_tgt_base = _IMAGE_R if tn in _img else r

        d_nodes = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        edge_pen = pg.mkPen(color, width=lw, style=style)
        if d_nodes < 1e-9:
            xs, ys = _self_loop_pts(p0, r)
            self._add_edge_item(pg.PlotDataItem(xs, ys, pen=edge_pen), 3,
                                sn, tn, excitatory, True, edge_pen)
            return

        # Enforce a minimum bow so the arc clears the node circles.  The geometric
        # minimum ensures the bezier excursion > r; the 0.4 visual margin is added
        # only for same-column nodes (d < 4r) where a tiny bow would be invisible.
        r_bow = max(r_src, r_tgt_base)
        geo_min = np.sqrt(max(0.0, (2 * r_bow / d_nodes) ** 2 - 1.0))
        visual_margin = 0.4 if d_nodes < 4 * r_bow else 0.0
        min_bow = geo_min + visual_margin
        if abs(signed_bow) < min_bow:
            if signed_bow != 0:
                signed_bow = np.copysign(min_bow, signed_bow)
            else:
                # Safety for any remaining zero-bow edge: preserve up/down
                # symmetry by signing with source-relative-to-target y.
                signed_bow = min_bow if p0[1] >= p1[1] else -min_bow

        ctrl = _ctrl_pt(p0, p1, signed_bow)
        if ctrl_perp:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            d = np.hypot(dx, dy) + 1e-9
            ctrl = (ctrl[0] + ctrl_perp * (-dy / d),
                    ctrl[1] + ctrl_perp * (dx  / d))

        # Rim points: start/end aimed toward the control point.
        # tgt_gap > 0 stops the arc tgt_gap*r before the target rim (gap in scene units).
        p0r = _rim_point(p0, ctrl, p1, r_src)
        p1r = _rim_point(p1, ctrl, p0, r_tgt_base * (1.0 + tgt_gap))

        thin = lw < 1.5
        pts = _bezier_pts(p0r, ctrl, p1r, n=25 if thin else 80)
        if len(pts) < 2:
            return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self._add_edge_item(pg.PlotDataItem(xs, ys, pen=edge_pen), 3,
                            sn, tn, excitatory, True, edge_pen)

        do_mark = (not thin) if mark is None else mark
        if do_mark:
            self._draw_edge_marker(pts[-1], p1, color, thin, excitatory, marker_style,
                                   _dy, sn, tn)

    def _add_edge_item(self, item, z, sn, tn, excitatory, is_curve, pen):
        item.setZValue(z)
        self.win._plot.addItem(item)
        self.drawn.edge_items.append(item)
        self.drawn.edge_items_tagged.append((item, sn, tn, excitatory, is_curve, pen))

    def _draw_edge_marker(self, end, p1, color, thin, excitatory, marker_style, _dy, sn, tn):
        """Synapse marker at the arc's end: a filled (excitatory) / hollow
        (inhibitory) dot tangent to the target node, or a perpendicular tick."""
        ex, ey = end
        ux, uy = ex - p1[0], ey - p1[1]
        ud = np.hypot(ux, uy)
        if ud > 1e-9:
            ux /= ud; uy /= ud
        m_pen_w = 3.0 if not thin else 2.0
        marker_pen = pg.mkPen(color, width=m_pen_w)

        if marker_style == 'tick':
            # Perpendicular tick at the target rim — PlotDataItem renders reliably
            perp_x, perp_y = -uy, ux
            tick_half = self.win._NODE_R * 55 * _dy
            tick_item = pg.PlotDataItem(
                [ex - tick_half * perp_x, ex + tick_half * perp_x],
                [ey - tick_half * perp_y, ey + tick_half * perp_y],
                pen=marker_pen,
            )
            self._add_edge_item(tick_item, 6, sn, tn, excitatory, True, marker_pen)
        else:
            # Circle marker: filled dot tangent to the target node surface
            half_m = self.win._MARKER_R * 60 * _dy
            scatter = pg.ScatterPlotItem(
                [ex + half_m * ux], [ey + half_m * uy],
                size=self.win._MARKER_R * 120,
                brush=pg.mkBrush(color) if excitatory else pg.mkBrush(C['bg']),
                pen=marker_pen,
            )
            self._add_edge_item(scatter, 6, sn, tn, excitatory, False, marker_pen)

    def reapply_camera_rects(self):
        for key, item in self.drawn.camera_items.items():
            rect = self.drawn.camera_rects.get(key)
            if rect is not None:
                item.setRect(*rect)

    def draw(self):
        """Rebuild the whole network drawing from the circuit, in steps."""
        self.win.layout_engine.infer_and_set_n()
        old_image_items = self._clear_scene()

        self.win._lay = self.win.layout_engine.compute()
        positions, sensor_nodes, groups = self.win._lay.positions, self.win._lay.sensor_nodes, self.win._lay.groups
        n_map = self.win.layout_engine.n_map()

        self.rebuild_group_buttons()
        self._draw_panels(positions, groups)
        self._draw_edges(positions, n_map)
        self._connect_range_signal()

        image_node_keys = self._draw_node_spots(positions, sensor_nodes)
        self._draw_camera_nodes(positions, old_image_items)
        self._draw_image_layer_nodes(positions, old_image_items)
        self._remove_stale_image_items(old_image_items)
        self._prepare_derivative_markers(positions, image_node_keys)
        self._prepare_neuromod_rings()
        self._fit_view(positions)

        self.win._vb.disableAutoRange()
        self.apply_group_visibility()
        self._draw_notes()

    def _clear_scene(self):
        """Remove every drawn item and start a fresh SceneItems. Image items
        are returned (not destroyed) so the rebuild can reuse them — this avoids
        the transform/scene detach race that caused stale images to appear at
        wrong positions on rebuild; leftovers are removed afterwards."""
        old = self.drawn
        for item in (old.edge_items + old.panel_items + old.text_items
                     + list(old.container_label_items.values())
                     + [it for items in old.container_note_items.values() for it in items]
                     + old.note_items):
            try:
                self.win._plot.removeItem(item)
            except Exception:
                pass
            try:
                item.setParentItem(None)
            except Exception:
                pass
            try:
                if item.scene() is not None:
                    item.scene().removeItem(item)
            except Exception:
                pass
        for item in (old.all_scatter, old.ring_scatter, old.deriv_scatter):
            if item is not None:
                try:
                    self.win._plot.removeItem(item)
                except Exception:
                    pass
        self.drawn = SceneItems()
        self.win.editing.sel.pen_override = {}
        self.win.editing.sel.highlighted  = None
        self.win.editing.sel.edge     = None
        return dict(old.camera_items)

    def _connect_range_signal(self):
        """Redraw edges whenever zoom/pan changes viewPixelSize."""
        if self.range_signal_connected:
            try:
                self.win._vb.sigRangeChanged.disconnect(self.rebuild_edges_slot)
            except Exception:
                pass
        self.win._vb.sigRangeChanged.connect(self.rebuild_edges_slot)
        self.range_signal_connected = True

    def _node_objects(self):
        """Layer / sensor (and lateral sensor half) name → object."""
        objs = {l.name: l for l in self.win.gui.circuit.layers}
        objs.update({s.name: s for s in self.win.gui.circuit.sensors})
        for s in self.win.gui.circuit.sensors:
            if _sensor_is_lateralized(s, self.win.gui.circuit):
                for half in half_names(s.name):
                    objs[half] = s
        return objs

    def _draw_node_spots(self, positions, sensor_nodes):
        """One scatter spot (+ centre label) per neuron. Sensors are hollow
        circles; image nodes get an invisible zero-size spot (their thumbnail is
        drawn separately) so hide/show works the same for every node.
        Returns the set of image-node keys."""
        layer_color = {layer.name: layer.color for layer in self.win.gui.circuit.layers
                       if getattr(layer, 'color', None) is not None}

        # Sensors: hollow circles; palette colour used for border + activity fill.
        self.drawn.sensor_pens       = {}
        self.drawn.sensor_active_rgb = {}
        bg_rgb = self.hex_rgb(C['bg'])
        for i, sensor in enumerate(self.win.gui.circuit.sensors):
            col = getattr(sensor, '_viz_color', None) or _CHAN_PALETTE[i % len(_CHAN_PALETTE)]
            if _sensor_is_lateralized(sensor, self.win.gui.circuit):
                keys = [f'{sensor.name}_{side}_{j}' for side in SIDES
                        for j in range(sensor.n_per_side())]
            else:
                keys = [f'{sensor.name}_{j}' for j in range(sensor.n_total or 0)]
            for key in keys:
                self.drawn.sensor_pens[key]       = pg.mkPen(col, width=2.5)
                self.drawn.sensor_active_rgb[key] = self.hex_rgb(col)

        objs = self._node_objects()
        # Image nodes — suppress both centre spot and centre text so they don't
        # overlay the thumbnail.
        image_node_keys = self._image_node_keys()

        # Lateralized sensor halves are labelled continuously: sensor_L_j → sensor_j,
        # sensor_R_j → sensor_{n_L + j}.
        lat_sensor_n_L = {s.name: s.n_per_side() for s in self.win.gui.circuit.sensors
                          if _sensor_is_lateralized(s, self.win.gui.circuit)}

        r = self.win._NODE_R
        spots = []
        for name, (x, y) in positions.items():
            layer_name = name.rsplit('_', 1)[0]
            if name in sensor_nodes:
                rgb   = bg_rgb
                brush = pg.mkBrush(C['bg'])
                pen_s = self.drawn.sensor_pens.get(name, pg.mkPen(C['primary'], width=2.5))
            else:
                fc    = layer_color.get(layer_name, C['primary'])
                rgb   = self.hex_rgb(fc)
                brush = pg.mkBrush(fc)
                pen_s = pg.mkPen(C['dark'], width=1.5)
            self.drawn.spot_names.append(name)
            self.drawn.spot_base_rgb[name]  = rgb
            self.drawn.spot_alpha[name]     = 1.0
            self.drawn.spot_visible[name]   = True

            if name in image_node_keys:
                spots.append({'pos': (x, y), 'size': 0,
                              'brush': pg.mkBrush(0, 0, 0, 0), 'pen': pg.mkPen(None),
                              'data': name})
                continue  # no centre text for image nodes

            obj     = objs.get(layer_name)
            n_obj   = getattr(obj, 'n', 1) or 1
            is_ring = getattr(obj, 'viz_layout', None) == 'ring'
            scale   = self.win._RING_SCALE if is_ring else (self.win._DENSE_NODE_SCALE if n_obj > 4 else 1.0)
            spots.append({'pos': (x, y), 'size': r * scale * 240,
                          'brush': brush, 'pen': pen_s, 'data': name})
            parts = name.rsplit('_', 2)
            if name in sensor_nodes and len(parts) == 3 and parts[1] in SIDES:
                sbase, side, sidx = parts
                j = int(sidx) + (lat_sensor_n_L.get(sbase, 1) if side == 'R' else 0)
                display_name = f'{sbase}_{j}'
            else:
                display_name = name
            txt = pg.TextItem(display_name, color=C['dark'], anchor=(0.5, 0.5))
            txt.setPos(x, y)
            txt.setFont(QFont('Segoe UI', 6))
            txt.setZValue(8)
            self.win._plot.addItem(txt)
            self.drawn.text_items.append(txt)
            self.drawn.text_map[name] = txt

        self.drawn.all_scatter = pg.ScatterPlotItem()
        self.drawn.all_scatter.setData(spots=spots)
        self.drawn.all_scatter.setZValue(5)
        self.drawn.all_scatter.sigClicked.connect(self.win.editing.on_spots_clicked)
        self.win._plot.addItem(self.drawn.all_scatter)
        return image_node_keys

    def _add_image_node(self, cx, cy, ring_color, item_key, w_px, label, old_image_items, fill=0):
        """Draw one image node — a ring, a thumbnail ImageItem (reused from the
        previous build when possible) and a label. Returns the drawn items."""
        theta = np.linspace(0, 2 * np.pi, 65)
        r     = np.hypot(_IMAGE_W / 2, _IMAGE_H / 2)
        circ  = pg.PlotDataItem(
            cx + r * np.cos(theta), cy + r * np.sin(theta),
            pen=pg.mkPen(ring_color, width=2.0),
            fillLevel=cy - r, brush=pg.mkBrush(C['bg']),
        )
        circ.setZValue(5.5)
        self.win._plot.addItem(circ)
        self.drawn.panel_items.append(circ)
        img_item = old_image_items.pop(item_key, None)
        if img_item is None:
            img_item = pg.ImageItem()
            img_item.setZValue(6)
            img_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.win._plot.addItem(img_item, ignoreBounds=True)
        img_item.setImage(np.full((_IMAGE_DISP_H, max(w_px, 1), 3), fill, dtype=np.uint8),
                          axisOrder='row-major')
        rect = (cx - _IMAGE_W / 2, cy - _IMAGE_H / 2, _IMAGE_W, _IMAGE_H)
        img_item.setRect(*rect)
        self.drawn.camera_items[item_key] = img_item
        self.drawn.camera_rects[item_key] = rect
        lbl = pg.TextItem(label, color=C['dark'], anchor=(0.5, 1.0))
        lbl.setPos(cx, cy + r + 0.01)
        lbl.setFont(QFont('Segoe UI', 6))
        lbl.setZValue(8)
        self.win._plot.addItem(lbl)
        self.drawn.text_items.append(lbl)
        return [circ, img_item, lbl]

    def _draw_camera_nodes(self, positions, old_image_items):
        """One image node per camera — two for a lateralized camera (one per half)."""
        for sensor in self.win.gui.circuit.sensors:
            if not sensor.is_camera:
                continue
            col = getattr(sensor, '_viz_color', None) or '#888888'
            if getattr(sensor, 'lateralized', False):
                for idx, side in enumerate(SIDES):
                    node_key = f'{sensor.name}_{side}_0'
                    pos = positions.get(node_key)
                    if pos is None:
                        continue
                    self.drawn.image_node_items[node_key] = self._add_image_node(
                        *pos, col, f'{sensor.name}_{side}', sensor.half_width(side),
                        f'{sensor.name}_{idx}', old_image_items)
            else:
                pos = positions.get(f'{sensor.name}_0')
                if pos is None:
                    continue
                self.drawn.image_node_items[f'{sensor.name}_0'] = self._add_image_node(
                    *pos, col, sensor.name, sensor.width, sensor.name, old_image_items)

    def _draw_image_layer_nodes(self, positions, old_image_items):
        """Image layers (is_image_node: Leaky2dLayer; Conv2dLayer / Reichardt2dLayer
        with pool='none') — displayed like cameras, a single thumbnail each."""
        for lyr in self.win.gui.circuit.layers:
            if not lyr.is_image_node:
                continue
            pos = positions.get(f'{lyr.name}_0')
            if pos is None:
                continue
            cx, cy = pos
            # Placeholder width until the first frame arrives.
            in_ch = max(getattr(lyr, 'in_ch', 1) or 1, 1)
            w_px  = getattr(lyr, 'frame_w', None) or max(1, int(((lyr.n or 1) // in_ch) ** 0.5))
            side  = side_of(lyr.name)
            label = base_name(lyr.name) + ('_0' if side == 'L' else '_1') if side else lyr.name
            # Signed outputs (e.g. Reichardt correlation maps) start at mid-grey.
            node_items = self._add_image_node(
                cx, cy, getattr(lyr, 'color', None) or '#888888', lyr.name, w_px, label,
                old_image_items, fill=127 if lyr.signed_image else 0)
            if getattr(lyr, 'output_mode', 'none') == 'derivative':
                r = np.hypot(_IMAGE_W / 2, _IMAGE_H / 2)
                dot = pg.ScatterPlotItem(x=[cx], y=[cy + (_IMAGE_H / 2 + r) / 2], size=6,
                                         symbol='o', brush=pg.mkBrush(C['dark']), pen=pg.mkPen(None))
                dot.setZValue(7)
                self.win._plot.addItem(dot)
                self.drawn.panel_items.append(dot)
                node_items.append(dot)
            self.drawn.image_node_items[f'{lyr.name}_0'] = node_items

    def _remove_stale_image_items(self, old_image_items):
        """Remove ImageItems for sensors/layers that are no longer in the circuit."""
        for item in old_image_items.values():
            try:
                self.win._vb.removeItem(item)
            except Exception:
                pass
            try:
                if item.scene() is not None:
                    item.scene().removeItem(item)
            except Exception:
                pass

    def _prepare_derivative_markers(self, positions, image_node_keys):
        """Derivative node markers: positions stored here; spots are updated
        dynamically in _redraw_nodes using viewPixelSize() so the dot always
        sits at the north of the node circle regardless of zoom level."""
        layer_map = {l.name: l for l in self.win.gui.circuit.layers}
        self.drawn.deriv_node_positions = [
            (x, y)
            for name, (x, y) in positions.items()
            if getattr(layer_map.get(name.rsplit('_', 1)[0]), 'output_mode', 'none') == 'derivative'
            and name not in image_node_keys
        ]
        self.drawn.deriv_scatter = pg.ScatterPlotItem()
        self.drawn.deriv_scatter.setZValue(7)
        self.win._plot.addItem(self.drawn.deriv_scatter)

    def _prepare_neuromod_rings(self):
        """Neuromodulator colours, and which nodes send / receive each signal
        (drawn as rings by the refresh timer)."""
        self.drawn.mod_colors = {}
        for obj in list(self.win.gui.circuit.layers) + list(self.win.gui.circuit.sensors):
            nt = getattr(obj, 'neuromodulator_transmitter', None)
            nc = getattr(obj, 'neuromodulator_color', None)
            if nt and nc:
                self.drawn.mod_colors[nt] = self.hex_rgb(nc)
        self.drawn.mod_pens = {
            nt: pg.mkPen(QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)), width=4)
            for nt, rgb in self.drawn.mod_colors.items()
        }

        objs = self._node_objects()
        self.drawn.src_nodes = {}
        self.drawn.rcv_nodes = {}
        for node_key in self.drawn.spot_names:
            obj = objs.get(node_key.rsplit('_', 1)[0])
            if obj is None:
                continue
            nt = getattr(obj, 'neuromodulator_transmitter', None)
            if nt and nt in self.drawn.mod_colors:
                self.drawn.src_nodes[node_key] = nt
                if nt not in self.drawn.wave_phase:
                    self.drawn.wave_phase[nt] = 0.0
            rcv = [row[0] for row in getattr(obj, 'modulators', []) if row[0] in self.drawn.mod_colors]
            if rcv:
                self.drawn.rcv_nodes[node_key] = rcv

        self.drawn.ring_scatter = pg.ScatterPlotItem()
        self.drawn.ring_scatter.setZValue(4.5)
        self.win._plot.addItem(self.drawn.ring_scatter)

    def _fit_view(self, positions):
        """Fit the view to the network. Sourced from _full_positions (all z-levels
        combined), not the just-drawn `positions`, so scrubbing the z-cut slider —
        which only changes which subset is shown — doesn't rescale the view.
        Nodes not currently z-cut-visible fall through the hidden-column check
        (`.get` → None) and still count toward the fit; explicitly hidden columns
        (Columns panel) still shrink the view."""
        ref_positions = self.win._lay.full_positions or positions
        if not ref_positions:
            self.win._plot.setRange(xRange=(-0.25, 1.25), yRange=(-0.30, 1.40), padding=0)
            return
        vis_pos = [p for nk, p in ref_positions.items()
                   if self.win._lay.node_container_map.get(nk) not in self.win._hidden_containers]
        use = vis_pos if vis_pos else list(ref_positions.values())
        xs = [p[0] for p in use]
        ys = [p[1] for p in use]
        self.win._plot.setRange(
            xRange=(min(xs) - 0.35, max(xs) + 0.25),
            yRange=(min(ys) - 0.30, max(ys) + 0.40),
            padding=0,
        )

    def _panel_label_text(self, container):
        """Text for a column panel's title: the manual "Set label..." nickname
        for whoever currently fronts this container, if one has been set
        (container_key() already resolves to the current z-winner's identity
        for a column shared across z-levels, so this stays correct as the
        z-cut slider changes who's in front — no staleness risk); else an
        auto-title of the current winner's name if this column is shared
        across z-levels; else the name of whichever occupant was added to
        the circuit first, so a fresh container starts out captioned instead
        of blank."""
        label = self.win._container_labels.get(self.win.layout_engine.container_key(container))
        if label:
            return label
        ghost_count = self.win._lay.ghost_count
        container_winner_names = self.win._lay.container_winner_names
        if ghost_count.get(container, 0) > 0 and container_winner_names.get(container):
            return ', '.join(container_winner_names[container])
        return self.win.layout_engine.first_occupant_name(container)

    def _ghost_display_name(self, name):
        """A ghost's own container label if it has one, else its raw name."""
        return self.win._container_labels.get(name, name)

    def _isolated_height_extent(self, n):
        """Y-extent (min, max) an object with n neurons would have if it were the
        sole occupant of its column (the "outside-in" layout algorithm, centred
        on y=0.5) — used to estimate a hidden container's height when it isn't
        currently active, so its real neuron positions aren't available."""
        n = max(1, n)
        spacing = min(2 * self.win._NODE_R, 1.0 / n)
        half = (n - 1) * spacing / 2.0
        return 0.5 - half, 0.5 + half

    def span_envelope_ys(self, container_data, start_col, span):
        """All y-values spanned by native columns [start_col, start_col+span-1]:
        the real positions of whichever object currently wins each column, or an
        isolated-height estimate (from its raw neuron count) for one that's
        currently hidden — so a container spanning several columns gets a height
        that reflects everything it covers, not just whichever single native
        column it happens to be rendered/anchored from."""
        native_col_n = self.win._lay.native_col_n
        ys = []
        for off in range(span):
            col = start_col + off
            cd = container_data.get(col)
            if cd and cd['ys']:
                ys.extend(cd['ys'])
            else:
                n = native_col_n.get(col)
                if n:
                    ys.extend(self._isolated_height_extent(n))
        return ys

    def _draw_panels(self, positions, groups):
        r, px = self.win._NODE_R, self.win._PAD_X
        py    = 0.04
        container_data = defaultdict(lambda: {'ys': [], 'types': set(), 'x': 0.0})
        for container_type, _, container_nodes, x_col, container, *_ in groups:
            ys = [positions[n][1] for n in container_nodes if n in positions]
            container_data[container]['ys'].extend(ys)
            container_data[container]['types'].add(container_type)
            container_data[container]['x'] = self.win._lay.container_x_map.get(container, x_col)

        ghosts_by_container = self.win._lay.ghosts_by_container

        _GHOST_DX = 0.022   # rightward shift per ghost step (data units) — kept tight
        _GHOST_DY = 0.070   # upward shift per ghost step  — room for its own label

        for container, data in sorted(container_data.items()):
            if not data['ys']:
                continue
            x_col    = data['x']
            container_type = 'sensor' if 'sensor' in data['types'] else 'layer'
            fc_s, ec_s = self.win._PANEL_CFG[container_type]
            _span     = self.win._lay.container_span_map.get(container, 1)
            _x_unit   = self.win._lay.x_unit
            _he       = (_span - 1) / 2.0 * _x_unit   # extra half-width each side
            # Envelope over every column this panel spans, not just its own —
            # a wide winner should be at least as tall as whatever it's hiding.
            ys = self.span_envelope_ys(container_data, container, _span) or data['ys']
            rect_bot = min(ys) - r - py
            rect_top = max(ys) + r + py

            # Ghost rects: one per hidden container behind this panel, each drawn
            # at ITS OWN native column + span (not this panel's), so a wide
            # dormant container still reads as wide, and several narrower
            # containers hidden by one wide winner each get their own labeled
            # rect instead of being merged into one shadow.
            ghosts = ghosts_by_container.get(container, [])
            if ghosts:
                # x_col is this panel's OWN midpoint, already shifted right by its
                # own _he when it spans multiple columns (see "Spanning containers
                # — x-position rule": x = (depth + (span-1)/2) * x_unit) — so it
                # must first be un-shifted back to the panel's native starting
                # column before walking to a neighbouring ghost's column, and each
                # ghost's own centre must then be re-shifted by ITS OWN half-span
                # the same way, or a wide ghost ends up centred on its starting
                # column instead of spanning outward from it.
                _container_start_x = x_col - _he
                _stack_idx = defaultdict(int)   # native_container -> next stack step
                for entry in ghosts:
                    nc = entry['native_container']
                    _stack_idx[nc] += 1
                    g  = _stack_idx[nc]
                    gx = _GHOST_DX * g
                    gy = _GHOST_DY * g
                    g_he    = (entry['span'] - 1) / 2.0 * _x_unit
                    g_x_col = _container_start_x + (nc - container) * _x_unit + g_he + gx
                    # A wide ghost (span>1) gets its own height envelope across
                    # everything IT covers, same as a wide real winner does above.
                    if entry['span'] > 1:
                        g_ys = self.span_envelope_ys(container_data, nc, entry['span'])
                        g_rect_bot = min(g_ys) - r - py if g_ys else rect_bot
                        g_rect_top = max(g_ys) + r + py if g_ys else rect_top
                    else:
                        g_rect_bot, g_rect_top = rect_bot, rect_top
                    ghost = pg.PlotDataItem(
                        [g_x_col - r - px - g_he, g_x_col + r + px + g_he,
                         g_x_col + r + px + g_he, g_x_col - r - px - g_he,
                         g_x_col - r - px - g_he],
                        [g_rect_bot + gy, g_rect_bot + gy,
                         g_rect_top + gy, g_rect_top + gy, g_rect_bot + gy],
                        pen=pg.mkPen(ec_s, width=0.8),
                        brush=pg.mkBrush(QColor(fc_s).lighter(110)),
                        fillLevel=g_rect_bot + gy,
                    )
                    ghost.setZValue(0)
                    self.win._plot.addItem(ghost)
                    self.drawn.panel_items.append(ghost)

                    ghost_lbl = pg.TextItem(self._ghost_display_name(entry['name']),
                                            anchor=(0.0, 0.0), color=QColor(ec_s))
                    ghost_lbl.setFont(_small_bold_font())
                    ghost_lbl.setPos(g_x_col - r - px - g_he + 0.003,
                                     g_rect_top + gy + self.win._LABEL_TOP_MARGIN)
                    ghost_lbl.setZValue(0.5)
                    self.win._plot.addItem(ghost_lbl)
                    self.drawn.panel_items.append(ghost_lbl)

            rect_item = pg.PlotDataItem(
                [x_col - r - px - _he, x_col + r + px + _he,
                 x_col + r + px + _he, x_col - r - px - _he,
                 x_col - r - px - _he],
                [rect_bot, rect_bot, rect_top, rect_top, rect_bot],
                pen=pg.mkPen(ec_s, width=0.8),
                brush=pg.mkBrush(QColor(fc_s).lighter(110)),
                fillLevel=rect_bot,
            )
            rect_item.setZValue(1)
            self.win._plot.addItem(rect_item)
            self.drawn.panel_items.append(rect_item)
            self.drawn.panel_rect_map[container] = rect_item

            # Columns shared across z-levels auto-title from whoever currently
            # wins — a fixed column-position nickname would go stale the moment
            # the z-cut slider changes who's in front. Single-occupant columns
            # keep the manual "Set label..." nickname as before.
            label = self._panel_label_text(container)
            if label:
                lbl_item = pg.TextItem(label, anchor=(0.0, 0.0),
                                       color=QColor(ec_s))
                lbl_item.setFont(_small_bold_font())
                lbl_item.setPos(x_col - r - px - _he + 0.003, rect_top + self.win._LABEL_TOP_MARGIN)
                lbl_item.setZValue(10)
                self.win._plot.addItem(lbl_item)
                self.drawn.container_label_items[container] = lbl_item

            note_text = self.win._container_notes.get(self.win.layout_engine.container_key(container), '')
            if note_text:
                icon_x = x_col + r + px + _he - self.win._CONTAINER_NOTE_PAD
                icon_y = rect_bot + self.win._CONTAINER_NOTE_PAD
                note_items = self._make_container_note_items(icon_x, icon_y)
                for item in note_items:
                    self.win._plot.addItem(item)
                self.drawn.container_note_items[container]    = note_items
                self.drawn.container_note_icon_pos[container] = (icon_x, icon_y)

    def _make_container_note_items(self, x, y):
        """Build the small note-glyph items for a container — same colours as
        a collapsed sticky note, just smaller and anchored to a fixed point
        (the container's bottom-right corner) instead of being draggable."""
        icon = pg.ScatterPlotItem(
            x=[x], y=[y], size=self.win._CONTAINER_NOTE_ICON_PX * 2, symbol='o',
            pen=pg.mkPen(self._NOTE_BORDER, width=1.2),
            brush=pg.mkBrush(self._NOTE_FILL),
        )
        icon.setZValue(20)
        label_font = QFont(self.win._NOTE_FONT_FAMILY, self.win._NOTE_FONT_SIZE - 1)
        label_font.setBold(True)
        label = pg.TextItem('N', anchor=(0.5, 0.5), color=self._NOTE_BORDER)
        label.setFont(label_font)
        label.setPos(x, y)
        label.setZValue(21)
        return [icon, label]

    def refresh_container_note(self, container):
        """Redraw a single container's note glyph in place after an edit —
        mirrors _refresh_container_label."""
        old = self.drawn.container_note_items.pop(container, None)
        self.drawn.container_note_icon_pos.pop(container, None)
        if old is not None:
            for item in old:
                try:
                    self.win._plot.removeItem(item)
                except Exception:
                    pass
        note_text = self.win._container_notes.get(self.win.layout_engine.container_key(container), '')
        if not note_text:
            return
        r, px = self.win._NODE_R, self.win._PAD_X
        py = 0.04
        container_nodes = [nk for nk, c in self.win._lay.node_container_map.items() if c == container]
        if not container_nodes:
            return
        x_col = self.win._lay.container_x_map.get(container, 0.5)
        col_ys = [self.win._lay.positions[nk][1] for nk in container_nodes if nk in self.win._lay.positions]
        if not col_ys:
            return
        rect_bot = min(col_ys) - r - py
        _span    = self.win._lay.container_span_map.get(container, 1)
        _x_unit  = self.win._lay.x_unit
        _he      = (_span - 1) / 2.0 * _x_unit
        icon_x = x_col + r + px + _he - self.win._CONTAINER_NOTE_PAD
        icon_y = rect_bot + self.win._CONTAINER_NOTE_PAD
        note_items = self._make_container_note_items(icon_x, icon_y)
        for item in note_items:
            self.win._plot.addItem(item)
        self.drawn.container_note_items[container]    = note_items
        self.drawn.container_note_icon_pos[container] = (icon_x, icon_y)

    def _make_note_items(self, note):
        """Build the graphics items for one note (collapsed icon, or box+text+toggle)."""
        note_font = QFont(self.win._NOTE_FONT_FAMILY, self.win._NOTE_FONT_SIZE)
        if note.collapsed:
            icon = pg.ScatterPlotItem(
                x=[note.x], y=[note.y], size=self.win._NOTE_ICON_PX * 2, symbol='o',
                pen=pg.mkPen(self._NOTE_BORDER, width=1.5),
                brush=pg.mkBrush(self._NOTE_FILL),
            )
            icon.setZValue(20)
            label_font = QFont(self.win._NOTE_FONT_FAMILY, self.win._NOTE_FONT_SIZE)
            label_font.setBold(True)
            label = pg.TextItem('N', anchor=(0.5, 0.5), color=self._NOTE_BORDER)
            label.setFont(label_font)
            label.setPos(note.x, note.y)
            label.setZValue(21)
            items = [icon, label]
        else:
            w, h = self.note_size(note)
            x0, x1 = note.x, note.x + w
            y0, y1 = note.y - h, note.y
            rect = pg.PlotDataItem(
                [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                pen=pg.mkPen(self._NOTE_BORDER, width=1.0),
                brush=pg.mkBrush(self._NOTE_FILL),
                fillLevel=y0,
            )
            rect.setZValue(20)

            txt = pg.TextItem('\n'.join(self.note_lines(note)),
                              anchor=(0.0, 0.0), color=C['dark'])
            txt.setFont(note_font)
            txt.setPos(x0 + self.win._NOTE_PAD, y1 - self.win._NOTE_PAD)
            txt.setZValue(21)

            toggle = pg.TextItem('−', anchor=(1.0, 0.0), color=self._NOTE_BORDER)
            toggle.setFont(note_font)
            toggle.setPos(x1 - 0.004, y1 - 0.002)
            toggle.setZValue(21)

            items = [rect, txt, toggle]

        for item in items:
            item.setVisible(self.win._notes_visible)
        return items

    def _draw_notes(self):
        """Draw sticky notes — free-positioned, always on top of the network.

        A single malformed/broken note must never take down the rest of the
        rebuild (edge repositioning, channel rebuild, etc. all run *after*
        this in build() and would otherwise silently never happen again)."""
        self.drawn.note_items    = []
        self.drawn.note_item_map = {}
        for note in self.win.gui.circuit.notes:
            try:
                items = self._make_note_items(note)
            except Exception as e:
                print(f"[NetworkViz] Failed to draw note {note!r}: {e}")
                continue
            for item in items:
                self.win._plot.addItem(item)
            self.drawn.note_items.extend(items)
            self.drawn.note_item_map[id(note)] = items

    def redraw_note(self, note):
        """Redraw a single note in place — used for drag/collapse updates so
        we don't pay for a full build() on every mouse-move frame."""
        old_items = self.drawn.note_item_map.pop(id(note), [])
        for item in old_items:
            try:
                self.win._plot.removeItem(item)
            except Exception:
                pass
            try:
                self.drawn.note_items.remove(item)
            except ValueError:
                pass
        items = self._make_note_items(note)
        for item in items:
            self.win._plot.addItem(item)
        self.drawn.note_items.extend(items)
        self.drawn.note_item_map[id(note)] = items

    def remove_note_items(self, note):
        """Remove a note's graphics items from the scene without touching circuit.notes."""
        old_items = self.drawn.note_item_map.pop(id(note), [])
        for item in old_items:
            try:
                self.win._plot.removeItem(item)
            except Exception:
                pass
            try:
                self.drawn.note_items.remove(item)
            except ValueError:
                pass

    def redraw_nodes(self, brain=None):
        """Rebuild all spot data and push to the single ScatterPlotItem in one call."""
        if self.redrawing or self.drawn.all_scatter is None or not self.drawn.spot_names:
            return
        self.redrawing = True
        try:
            active = self.win._ACTIVE_RGB
            r_node = self.win._NODE_R
            spots      = []
            all_layers  = {l.name: l for l in self.win.gui.circuit.layers}
            all_sensors = {s.name: s for s in self.win.gui.circuit.sensors}
            for name in self.drawn.spot_names:
                if not self.drawn.spot_visible.get(name, True):
                    continue
                if name not in self.win._lay.positions:
                    continue
                x, y      = self.win._lay.positions[name]
                base_rgb  = self.drawn.spot_base_rgb.get(name, (0.5, 0.5, 0.5))
                alpha     = self.drawn.spot_alpha.get(name, 1.0)
                layer_name = name.rsplit('_', 1)[0]

                is_muted  = getattr(all_layers.get(layer_name), 'muted', False)
                is_sensor = name in self.win._lay.sensor_nodes

                if is_muted:
                    # Muted: flat grey, no activity colouring
                    base_rgb = (0.72, 0.72, 0.72)
                elif brain is not None:
                    try:
                        idx  = int(name.rsplit('_', 1)[1])
                        attr = getattr(brain, layer_name, None)
                        if attr is not None:
                            arr = np.atleast_1d(
                                attr.output if hasattr(attr, 'output') else attr)
                            if idx < len(arr):
                                val = float(np.clip(arr[idx], 0, 1))
                                act = active  # all nodes blend toward bright active orange
                                base_rgb = (
                                    base_rgb[0] + (act[0] - base_rgb[0]) * val,
                                    base_rgb[1] + (act[1] - base_rgb[1]) * val,
                                    base_rgb[2] + (act[2] - base_rgb[2]) * val,
                                )
                    except (ValueError, AttributeError):
                        pass

                a_int  = int(alpha * 255)
                brush  = pg.mkBrush(QColor(int(base_rgb[0] * 255),
                                           int(base_rgb[1] * 255),
                                           int(base_rgb[2] * 255), a_int))
                po = self.win.editing.sel.pen_override.get(name)
                if po == 'selected':
                    pen = self.pen_selected
                elif po == 'multi':
                    pen = self.pen_multi
                elif is_muted:
                    pen = self.pen_muted
                elif is_sensor:
                    _s_nt = getattr(all_sensors.get(layer_name),
                                    'neuromodulator_transmitter', None)
                    if _s_nt and _s_nt in self.drawn.mod_pens:
                        pen = self.drawn.mod_pens[_s_nt]
                    else:
                        pen = self.drawn.sensor_pens.get(name, self.pen_default)
                elif layer_name in self.win.gui.tracked_osc_items():
                    pen = self.pen_osc
                elif getattr(all_layers.get(layer_name), 'neuromodulator_transmitter', None) in self.drawn.mod_pens:
                    nt = all_layers[layer_name].neuromodulator_transmitter
                    pen = self.drawn.mod_pens[nt]
                else:
                    pen = self.pen_default
                obj = all_layers.get(layer_name)
                is_ring = getattr(obj, 'viz_layout', None) == 'ring'
                _n_rd   = getattr(obj, 'n', 1) or 1
                _scale  = (self.win._RING_SCALE if is_ring
                           else (self.win._DENSE_NODE_SCALE if _n_rd > 4 else 1.0))
                dot_r = r_node * _scale * 240
                spots.append({'pos': (x, y), 'size': dot_r,
                              'brush': brush, 'pen': pen, 'data': name})

            self.drawn.all_scatter.setData(spots=spots)

            # Image nodes (cameras, Leaky2dLayer, Conv2dLayer pool='none') — visibility
            # mirrors _spot_visible so hide/show works the same way as regular nodes.
            for node_key, items in self.drawn.image_node_items.items():
                visible = self.drawn.spot_visible.get(node_key, True)
                for item in items:
                    item.setVisible(visible)

            # Derivative dot: recompute y_off each frame via viewPixelSize so the
            # dot always touches the north of the node circle at any zoom level.
            if self.drawn.deriv_scatter is not None and self.drawn.deriv_node_positions:
                r_d = r_node * 0.20   # slightly smaller than before
                try:
                    _, dy = self.win._vb.viewPixelSize()
                    y_off = (r_node - r_d) * 120 * dy
                except Exception:
                    y_off = (r_node - r_d)
                d_spots = [
                    {'pos': (x, y + y_off), 'size': r_d * 240,
                     'brush': pg.mkBrush(C['dark']), 'pen': pg.mkPen(None)}
                    for (x, y) in self.drawn.deriv_node_positions
                ]
                self.drawn.deriv_scatter.setData(spots=d_spots)
        finally:
            self.redrawing = False
        # Keep camera images pinned to their scatter nodes (guards against any
        # transform reset that may occur when setImage changes the image shape).
        for key, item in self.drawn.camera_items.items():
            rect = self.drawn.camera_rects.get(key)
            if rect is not None:
                item.setRect(*rect)


    def update_weight_panel(self):
        if not self.win._weight_pinned:
            return
        conn_map = {(c.src, c.tgt): c for c in self.win.gui.circuit.connections}
        for (src, tgt), entry_widget in self.win._weight_pinned.items():
            conn = conn_map.get((src, tgt))
            if conn is None:
                continue
            W = np.asarray(conn.W, dtype=float)
            if W.ndim == 4:
                W = _tile_conv_filters(W)
            elif W.ndim != 2:
                W = W.reshape(1, -1)
            entry_widget.set_matrix(W)

    def update_activation_panel(self):
        if not self.win._activation_pinned:
            return
        layer_map = {l.name: l for l in self.win.gui.circuit.layers}
        brain = getattr(self.win.gui, 'brain', None)
        for name, entry_widget in self.win._activation_pinned.items():
            layer_obj = layer_map.get(name)
            if layer_obj is not None:
                out = layer_obj.output
            elif brain is not None:
                out = getattr(brain, name, None)   # sensor reading, set each tick by sim_engine
            else:
                out = None
            if out is None:
                continue
            if hasattr(out, 'detach'):
                out = out.detach().numpy()
            entry_widget.set_values(np.ravel(np.asarray(out, dtype=float)))

    def clear_selection_highlight(self):
        self.win.editing.sel.pen_override.clear()
        self.redraw_nodes()

    def rebuild_group_buttons(self):
        while self.win._group_bar_lay.count() > 1:
            item = self.win._group_bar_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        sorted_containers = sorted(self.win._lay.container_x_map.keys())
        ghost_count            = self.win._lay.ghost_count
        container_winner_names = self.win._lay.container_winner_names
        _btn_ss = (
            "QPushButton{padding:0 4px;font-size:9px;"
            "border:1px solid #A0B0C0;border-radius:3px;background:#E8F0F8;}"
            "QPushButton:checked{color:white;}"
        )
        for i, container in enumerate(sorted_containers):
            if ghost_count.get(container, 0) > 0 and container_winner_names.get(container):
                container_label = ', '.join(container_winner_names[container])
            else:
                container_label = self.win._container_labels.get(
                    self.win.layout_engine.container_key(container), str(i + 1))
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(2)

            name_lbl = QLabel(container_label)
            name_lbl.setStyleSheet(f"font-size:9px;color:{C['dark']};")
            name_lbl.setMinimumWidth(30)
            row.addWidget(name_lbl, 1)

            btn_vis = QPushButton("H")
            btn_vis.setCheckable(True)
            btn_vis.setChecked(container in self.win._hidden_containers)
            btn_vis.setEnabled(container not in self.win._disabled_containers)
            btn_vis.setFixedSize(20, 18)
            btn_vis.setToolTip("Hide / show this column")
            btn_vis.setStyleSheet(
                _btn_ss + f"QPushButton:checked{{background:{C['muted']};}}"
            )
            btn_vis.toggled.connect(lambda checked, container=container: self.win._on_container_hide(container, checked))
            row.addWidget(btn_vis)

            btn_dis = QPushButton("D")
            btn_dis.setCheckable(True)
            btn_dis.setChecked(container in self.win._disabled_containers)
            btn_dis.setFixedSize(20, 18)
            btn_dis.setToolTip("Disable / enable this column")
            btn_dis.setStyleSheet(
                _btn_ss + "QPushButton:checked{background:#C0392B;color:white;}"
            )
            btn_dis.toggled.connect(lambda checked, container=container: self.win._on_container_disable(container, checked))
            row.addWidget(btn_dis)

            row_w.setFixedHeight(22)
            self.win._group_bar_lay.insertWidget(i, row_w)

        has_any = bool(self.win._hidden_containers or self.win._disabled_containers)
        self.win._btn_show_all.setEnabled(bool(self.win._hidden_containers))
        self.win._btn_enable_all.setEnabled(bool(self.win._disabled_containers))

    def apply_group_visibility(self):
        for node_key in self.drawn.spot_names:
            container = self.win._lay.node_container_map.get(node_key)
            hidden = container in self.win._hidden_containers
            self.drawn.spot_visible[node_key] = not hidden
            txt = self.drawn.text_map.get(node_key)
            if txt:
                txt.setVisible(not hidden)
        self.redraw_nodes()
        for item, sn, tn, *_ in self.drawn.edge_items_tagged:
            s_dv = self.win._lay.node_container_map.get(sn)
            t_dv = self.win._lay.node_container_map.get(tn)
            hidden = (s_dv in self.win._hidden_containers) or (t_dv in self.win._hidden_containers)
            item.setVisible(not hidden)
        for container, rect_item in self.drawn.panel_rect_map.items():
            rect_item.setVisible(container not in self.win._hidden_containers)
    def refresh_container_label(self, container):
        old = self.drawn.container_label_items.pop(container, None)
        if old is not None:
            self.win._plot.removeItem(old)
        # Shared (multi-z) columns auto-title from the current winner, same as
        # in _draw_panels — a manually-set nickname only applies to
        # single-occupant columns.
        label = self._panel_label_text(container)
        if label:
            r, px = self.win._NODE_R, self.win._PAD_X
            py = 0.04
            container_type_nodes = [nk for nk, c in self.win._lay.node_container_map.items()
                              if c == container]
            if container_type_nodes:
                x_col = self.win._lay.container_x_map.get(container, 0.5)
                is_sensor = any(nk in self.win._lay.sensor_nodes for nk in container_type_nodes)
                _, ec_s = self.win._PANEL_CFG['sensor' if is_sensor else 'layer']
                col_ys = [self.win._lay.positions[nk][1]
                          for nk in container_type_nodes
                          if nk in self.win._lay.positions]
                rect_top = (max(col_ys) + r + py) if col_ys else 0.6
                _span   = self.win._lay.container_span_map.get(container, 1)
                _x_unit = self.win._lay.x_unit
                _he     = (_span - 1) / 2.0 * _x_unit
                lbl_item = pg.TextItem(label, anchor=(0.0, 0.0),
                                       color=QColor(ec_s))
                lbl_item.setFont(_small_bold_font())
                lbl_item.setPos(x_col - r - px - _he + 0.003, rect_top + self.win._LABEL_TOP_MARGIN)
                lbl_item.setZValue(10)
                self.win._plot.addItem(lbl_item)
                self.drawn.container_label_items[container] = lbl_item
        self.rebuild_group_buttons()

    def highlight_edges(self, center_node):
        self.win.editing.sel.highlighted = center_node
        center_layer = center_node.rsplit('_', 1)[0]

        # Build directed adjacency from cross-layer edges only.
        # Skipping within-layer edges (e.g. mutual inhibition) prevents them
        # from pulling in the entire contralateral circuit during BFS.
        fwd = defaultdict(set)   # source → {targets}
        bwd = defaultdict(set)   # target → {sources}
        for _, sn, tn, *_ in self.drawn.edge_items_tagged:
            if sn.rsplit('_', 1)[0] == tn.rsplit('_', 1)[0]:
                continue  # skip within-layer edges for traversal
            fwd[sn].add(tn)
            bwd[tn].add(sn)

        def _bfs(start, adj):
            visited, queue = {start}, [start]
            while queue:
                node = queue.pop()
                for nb in adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            return visited

        fwd_set = _bfs(center_node, fwd)
        bwd_set = _bfs(center_node, bwd)
        circuit = fwd_set | bwd_set

        # For dense layers (many neurons), include only the clicked neuron in the
        # same-layer set — this prevents all 64 edges of an 8×8 ring attractor from
        # lighting up when you click one neuron.  For small layers, include the whole
        # layer so mutual-inhibition edges are shown in full.
        n_map = self.win.layout_engine.n_map()
        n_center = n_map.get(center_layer, 1)
        if n_center > self.win._DENSE_THRESHOLD:
            same_layer = {center_node}
        else:
            same_layer = {n for n in self.drawn.spot_names
                          if n.rsplit('_', 1)[0] == center_layer}
        display_set = circuit | same_layer

        # Fade nodes and labels not in display_set
        for node_key in self.drawn.spot_names:
            self.drawn.spot_alpha[node_key] = (1.0 if node_key in display_set
                                          else self._FADE_OPACITY)
        for node_key, txt in self.drawn.text_map.items():
            txt.setOpacity(1.0 if node_key in display_set else self._FADE_OPACITY)
        self.redraw_nodes()

        # Highlight edges:
        # - within-layer: show if any endpoint is in display_set
        # - cross-layer:  show only if both endpoints are on the same traversal side
        #   (both in bwd_set OR both in fwd_set). This prevents "shortcut" edges
        #   (e.g. light→motor when viewing layer5) from lighting up just because
        #   their endpoints happen to be reachable via independent paths.
        for item, sn, tn, *_ in self.drawn.edge_items_tagged:
            if sn.rsplit('_', 1)[0] == tn.rsplit('_', 1)[0]:
                lit = (sn in display_set) or (tn in display_set)
            else:
                lit = ((sn in bwd_set and tn in bwd_set) or
                       (sn in fwd_set and tn in fwd_set))
            item.setOpacity(1.0 if lit else self._FADE_OPACITY)

    def highlight_selected_edge(self, src, tgt):
        for item, sn, tn, _, is_curve, *_ in self.drawn.edge_items_tagged:
            if sn.rsplit('_', 1)[0] == src and tn.rsplit('_', 1)[0] == tgt:
                if is_curve:
                    item.setPen(pg.mkPen('#E07828', width=4.5))
                else:
                    item.setBrush(pg.mkBrush('#E07828'))
                    item.setPen(pg.mkPen('#E07828', width=3.0))

    def clear_edge_selection(self):
        if not self.win.editing.sel.edge:
            return
        src, tgt = self.win.editing.sel.edge
        for item, sn, tn, excitatory, is_curve, original_pen in self.drawn.edge_items_tagged:
            if sn.rsplit('_', 1)[0] == src and tn.rsplit('_', 1)[0] == tgt:
                if is_curve:
                    item.setPen(original_pen)
                else:
                    orig_rgba = original_pen.color().getRgb()
                    item.setBrush(pg.mkBrush(orig_rgba) if excitatory else pg.mkBrush(C['bg']))
                    item.setPen(original_pen)
        self.win.editing.sel.edge = None

    def clear_edge_highlight(self):
        self.win.editing.sel.highlighted = None
        for node_key in self.drawn.spot_names:
            self.drawn.spot_alpha[node_key] = 1.0
        for _, txt in self.drawn.text_map.items():
            txt.setOpacity(1.0)
        self.redraw_nodes()
        for item, *_ in self.drawn.edge_items_tagged:
            item.setOpacity(1.0)

    def show_drag_indicator(self, snap_x, mouse_y=None, layer=None):
        # Same column and y provided → horizontal slot indicator.
        if layer is not None and mouse_y is not None:
            # Use the layer's visual x position rather than its raw .layer position
            # so that layers with layer=None (e.g. the _L half of a lateralized
            # pair) are detected as same-column when they should be.
            my_x = next(
                (self.win._lay.positions[f'{layer.name}_{j}'][0]
                 for j in range(layer.n or 0)
                 if f'{layer.name}_{j}' in self.win._lay.positions),
                None,
            )
            col_x = my_x
            if col_x is not None and abs(snap_x - col_x) < 1e-9:
                snap_y, _ = self.win.layout_engine.get_container_snap_y(layer, mouse_y)
                self.h_drag_indicator.setValue(snap_y)
                self.h_drag_indicator.setVisible(True)
                self.drag_indicator.setVisible(False)
                return
        self.drag_indicator.setValue(snap_x)
        self.drag_indicator.setVisible(True)
        self.h_drag_indicator.setVisible(False)

    def hide_drag_indicator(self):
        self.drag_indicator.setVisible(False)
        self.h_drag_indicator.setVisible(False)

    def update_conn_preview(self, mouse_pt):
        sx, sy = self.win._lay.positions[self.win.editing.conn_from]
        self.conn_preview.setData([sx, mouse_pt.x()], [sy, mouse_pt.y()])
        self.conn_preview.setVisible(True)

    def hide_conn_preview(self):
        self.conn_preview.setVisible(False)

    # ── Note text metrics ────────────────────────────────────────────────────

    def note_font_metrics(self):
        from PySide6.QtGui import QFont, QFontMetrics
        return QFontMetrics(QFont(self.win._NOTE_FONT_FAMILY, self.win._NOTE_FONT_SIZE))

    def note_lines(self, note):
        """Word-wrap a note's text to fit _NOTE_W, measured in *actual* pixel
        font metrics (not a guessed character count) so the background box
        computed by note_size always matches what's rendered — a fixed
        chars-per-line heuristic drifts badly across the wide range of zoom
        levels different-sized networks end up at.
        """
        fm = self.note_font_metrics()
        dx, _dy = self.win._vb.viewPixelSize()
        max_px = (self.win._NOTE_W - 2 * self.win._NOTE_PAD) / dx if dx > 0 else 200.0
        max_px = max(20.0, max_px)
        lines = []
        for para in (note.text or '').split('\n'):
            cur = ''
            for word in para.split(' '):
                trial = f'{cur} {word}'.strip() if cur else word
                if not cur or fm.horizontalAdvance(trial) <= max_px:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
        return lines or ['']

    def note_size(self, note):
        """Return (w, h) of an expanded note's background box, in data coords.

        Measures a throwaway TextItem's real boundingRect() rather than
        estimating from QFontMetrics — pyqtgraph's own multi-line text layout
        uses more vertical space per line than QFontMetrics.height()/
        lineSpacing() predict, so an estimate consistently undersizes the box
        and the text spills out the bottom.
        """
        import pyqtgraph as pg
        from PySide6.QtGui import QFont
        txt = pg.TextItem('\n'.join(self.note_lines(note)))
        txt.setFont(QFont(self.win._NOTE_FONT_FAMILY, self.win._NOTE_FONT_SIZE))
        _dx, dy = self.win._vb.viewPixelSize()
        h = 2 * self.win._NOTE_PAD + txt.boundingRect().height() * dy
        return self.win._NOTE_W, h


# ============================================================
# MATRIX HEATMAP WIDGET
# ============================================================
class _MatrixHeatmapWidget(QWidget):
    """Blue–white–red heatmap for a 2D weight matrix (nt rows × ns cols).

    Layout (all sizes in pixels):
      _LH  top margin  — source (column) axis labels
      _LW  left margin — target (row) axis labels
      _SW  right strip — coloured squares for row sums, labelled "Σ"
    """
    _MAX_CELL = 30
    _MIN_CELL = 5
    _LH = 16   # top label margin
    _LW = 20   # left label margin
    _SW = 12   # row-sum strip width
    _SG = 8    # gap between weight grid and sum strip

    def __init__(self, parent=None):
        super().__init__(parent)
        self._W    = np.zeros((1, 1))
        self._cell = self._MIN_CELL
        self.setMinimumSize(40, 40)
        self.setMouseTracking(True)

    def set_matrix(self, W):
        W = np.atleast_2d(np.asarray(W, dtype=float))
        self._W = W
        nt, ns = W.shape
        self._cell = max(self._MIN_CELL, min(self._MAX_CELL, 240 // max(nt, ns, 1)))
        c = self._cell
        self.setFixedSize(self._LW + ns * c + 2 + self._SG + self._SW + 2,
                          self._LH + nt * c + 2)
        self.update()

    def _cell_at(self, pt):
        """Return (row, col) for local QPoint pt, or None."""
        x = pt.x() - self._LW - 1
        y = pt.y() - self._LH - 1
        if x < 0 or y < 0:
            return None
        nt, ns = self._W.shape
        c = self._cell
        col, row = int(x // c), int(y // c)
        if 0 <= row < nt and 0 <= col < ns:
            return row, col
        return None

    def mouseMoveEvent(self, ev):
        pt = ev.position().toPoint()
        rc = self._cell_at(pt)
        if rc is not None:
            r, c = rc
            QToolTip.showText(self.mapToGlobal(pt),
                              f"[tgt {r}, src {c}] = {self._W[r, c]:.4g}", self)
        else:
            QToolTip.hideText()

    def _weight_color(self, v):
        """v in [–1, +1] → QColor (red positive, blue negative)."""
        if v >= 0:
            return QColor(255, int(255 * (1 - v)), int(255 * (1 - v)))
        return QColor(int(255 * (1 + v)), int(255 * (1 + v)), 255)

    def paintEvent(self, _ev):
        W = np.nan_to_num(self._W, nan=0.0, posinf=0.0, neginf=0.0)
        nt, ns = W.shape
        c    = self._cell
        vmax = max(float(np.abs(W).max()), 1e-9)
        lw, lh, sw = self._LW, self._LH, self._SW

        p = QPainter(self)
        p.setPen(Qt.NoPen)

        # ── weight cells ────────────────────────────────────────────────────
        for i in range(nt):
            for j in range(ns):
                p.fillRect(int(lw + 1 + j * c), int(lh + 1 + i * c),
                           max(1, c), max(1, c),
                           self._weight_color(float(W[i, j]) / vmax))

        # ── row-sum strip ───────────────────────────────────────────────────
        sums = W.sum(axis=1)
        smax = max(float(np.abs(sums).max()), 1e-9)
        sx = lw + 1 + ns * c + self._SG
        for i in range(nt):
            p.fillRect(sx, int(lh + 1 + i * c), sw, max(1, c),
                       self._weight_color(float(sums[i]) / smax))

        # ── axis labels ─────────────────────────────────────────────────────
        p.setPen(QColor(160, 160, 160))
        font = p.font()
        font.setPointSize(6)
        p.setFont(font)

        # top margin: "src" title + column indices
        p.drawText(lw + 1, 0, ns * c, lh - 1,
                   Qt.AlignHCenter | Qt.AlignBottom, "src →")
        step = max(1, ns // 8)
        for j in range(0, ns, step):
            p.drawText(int(lw + 1 + j * c), 0, c * step, lh - 1,
                       Qt.AlignHCenter | Qt.AlignBottom, str(j))

        # left margin: "tgt" title + row indices
        p.drawText(0, lh + 1, lw - 1, nt * c,
                   Qt.AlignRight | Qt.AlignVCenter, "tgt")
        step = max(1, nt // 8)
        for i in range(0, nt, step):
            p.drawText(0, int(lh + 1 + i * c), lw - 1, c * step,
                       Qt.AlignRight | Qt.AlignVCenter, str(i))

        # row-sum strip label
        p.drawText(sx, 0, sw, lh - 1, Qt.AlignHCenter | Qt.AlignBottom, "Σ")

        p.end()


# ============================================================
# WEIGHT ENTRY WIDGET  (one slot in the live weight panel)
# ============================================================
class WeightEntryWidget(QWidget):
    """Shows a label and a live-updating heatmap for one connection's weight matrix."""

    def __init__(self, src, tgt, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 4)
        lay.setSpacing(2)
        lbl = QLabel(f"{src}  →  {tgt}")
        lbl.setStyleSheet("font-size:9px;font-weight:bold;")
        lay.addWidget(lbl)
        self._heatmap = _MatrixHeatmapWidget()
        self._heatmap.setFixedHeight(100)
        lay.addWidget(self._heatmap)

    def set_matrix(self, W):
        self._heatmap.set_matrix(W)


# ============================================================
# ACTIVATION ENTRY WIDGET  (one slot in the live activation panel)
# ============================================================
class ActivationEntryWidget(QWidget):
    """Shows a label and a live-updating bar chart of one layer's or sensor's
    current per-neuron output — x-axis is neuron index, y-axis is activation value."""

    def __init__(self, name, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 4)
        lay.setSpacing(2)
        lbl = QLabel(name)
        lbl.setStyleSheet("font-size:9px;font-weight:bold;")
        lbl.setFixedHeight(16)
        lay.addWidget(lbl)
        self._plot = pg.PlotWidget()
        self._plot.setFixedHeight(100)
        self._plot.setBackground(QColor(C['surface']))
        self._plot.getPlotItem().showGrid(x=False, y=True, alpha=0.3)
        self._plot.getPlotItem().getAxis('left').setStyle(tickTextOffset=4)
        self._plot.getPlotItem().getAxis('bottom').setStyle(tickTextOffset=2)
        self._bars = pg.BarGraphItem(x=[0], height=[0], width=0.8,
                                     brush=pg.mkBrush(C['primary']),
                                     pen=pg.mkPen(None))
        self._plot.addItem(self._bars)
        self._plot.getViewBox().setMouseEnabled(x=False, y=False)
        lay.addWidget(self._plot)

    def set_values(self, values):
        values = np.atleast_1d(np.asarray(values, dtype=float))
        n = len(values)
        self._bars.setOpts(x=np.arange(n), height=values, width=0.8)
        # Range always includes 0 as a reference line; adapts to whichever
        # sign(s) are actually present instead of always being symmetric
        # (most layers are relu-activated and never go negative).
        vmin = min(0.0, float(values.min()))
        vmax = max(0.0, float(values.max()))
        pad  = max(vmax - vmin, 1e-9) * 0.15
        self._plot.getViewBox().setRange(xRange=(-0.6, n - 0.4),
                                         yRange=(vmin - pad, vmax + pad), padding=0)
