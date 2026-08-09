"""3D network visualizer — read-only perspective view.

Coordinate axes
  X  sensory → motor  (pipeline container position, normalized to [0, 1])
  Y  medio-lateral    (side-view neuron y-position recentered: midline = 0)
  Z  subsumption depth (layer.z from the Top View, scaled)

Nodes  : one sphere per neuron, positioned using the actual layout coordinates
         from the side view (_positions).  Image sensor/layer groups are shown
         as a single plane instead.  Color matches side-view container panels.
Edges  : per-neuron Bezier arcs when W dimensions match group sizes (≤500 arcs);
         centroid-to-centroid fallback for image nodes or large matrices.
         blue = excitatory (#6AAAD4), red = inhibitory (#E07878).
"""

import numpy as np
from collections import defaultdict

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QPushButton, QLabel)
from PySide6.QtCore import Qt

try:
    import pyqtgraph.opengl as gl
    _GL_OK = True
except Exception:
    _GL_OK = False

# ── Visual constants ────────────────────────────────────────────────────────────
_C_SENSOR = (168/255, 204/255, 168/255, 1.0)   # #A8CCA8  green
_C_LAYER  = (160/255, 192/255, 220/255, 1.0)   # #A0C0DC  blue

_C_EXCIT = (106/255, 170/255, 212/255)          # #6AAAD4
_C_INHIB = (224/255, 120/255, 120/255)          # #E07878

_R_NEURON = 0.018   # sphere radius for each individual neuron

_Z_SCALE  = 0.50    # world-unit spacing per z-level

_N_SEGS   = 40      # Bezier segments
_ARC_K    = 0.20    # arc lift = K × distance, in +Z direction

_CAM_HW_MIN = 0.03
_CAM_HW_MAX = 0.08
_CAM_ALPHA  = 0.85

_LOOP_R          = 0.06   # radius of the self-connection loop
_RING_RADIUS     = 0.14   # world-unit radius for ring-layout neuron circles
_MAX_CROSS_ARCS  = 500    # per-neuron arc limit; fall back to centroid above this


# ── Geometry ───────────────────────────────────────────────────────────────────

def _bezier(p0, p1, k=_ARC_K, ring_center=None):
    """Quadratic Bezier arc from p0 to p1.

    ring_center — when given, the arc curves radially outward from the ring
                  centre (used for ring-attractor per-neuron connections).
    Otherwise   — arc lifts perpendicular to diff in the XY plane.
    Self-connection (dist ≈ 0) → circular loop in the XY plane, extending in +X.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    diff   = p1 - p0
    dist   = float(np.linalg.norm(diff))

    if dist < 1e-6:
        # Self-loop: circle in XY plane centred one radius to the +X side of p0.
        r      = _LOOP_R
        angles = np.linspace(np.pi, -np.pi, _N_SEGS)
        pts    = np.empty((_N_SEGS, 3))
        pts[:, 0] = (p0[0] + r) + r * np.cos(angles)
        pts[:, 1] = p0[1]       + r * np.sin(angles)
        pts[:, 2] = p0[2]
        return pts

    mid = (p0 + p1) * 0.5

    if ring_center is not None:
        # Outward radial lift: control point pushed away from the ring centre so
        # arcs curve visibly around the outside of the neuron ring.
        rc      = np.asarray(ring_center, float)
        outward = mid - rc
        out_len = float(np.linalg.norm(outward))
        if out_len > 1e-6:
            outward /= out_len
        else:
            outward = np.array([0.0, 0.0, 1.0])
        ctrl = mid + outward * (k * dist + _RING_RADIUS * 0.30)
    else:
        # Lift upward in +Z so arcs arch gracefully above the network plane.
        # Both forward and backward arcs between the same layers lift in the
        # same direction, forming a visible bundle rather than crossing in XY.
        ctrl = mid + np.array([0.0, 0.0, k * dist])

    t = np.linspace(0.0, 1.0, _N_SEGS)[:, None]
    return (1 - t)**2 * p0 + 2*(1-t)*t * ctrl + t**2 * p1



# ── Main window ─────────────────────────────────────────────────────────────────

class NetView3DWindow(QWidget):
    """Read-only 3D graph of the current network circuit."""

    def __init__(self, nvw):
        super().__init__(None, Qt.Window)
        self._nvw = nvw
        self.setWindowTitle("Network — 3D View")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(960, 720)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(3)

        tb = QHBoxLayout()
        btn = QPushButton("↺ Refresh")
        btn.setFixedWidth(84)
        btn.clicked.connect(self.refresh)
        tb.addWidget(btn)
        tb.addStretch()
        root.addLayout(tb)

        if not _GL_OK:
            root.addWidget(QLabel(
                "PyOpenGL not found — run:  pip install PyOpenGL PyOpenGL_accelerate"))
            self._gw = None
            return

        self._gw = gl.GLViewWidget()
        # When a previous GLViewWidget is destroyed its OpenGL context is
        # invalidated, but pyqtgraph keeps a global cache of compiled shader
        # program IDs.  A new context reuses those stale IDs, causing
        # glUseProgram to raise GLError 1281 on the first paint.
        # Resetting _prog forces recompilation in the new context.
        try:
            from pyqtgraph.opengl import shaders as _pg_shaders
            for _s in _pg_shaders.Shaders:
                _s._prog = None
        except Exception:
            pass
        # Camera: look along −X so sensory (left) → motor (right) is horizontal;
        # Y (medio-lateral) goes left-right in the viewport;
        # Z (subsumption depth) goes upward.
        self._gw.setCameraPosition(distance=2.8, elevation=25, azimuth=-60)
        root.addWidget(self._gw)

        self._items      = []
        self._mesh_cache = {}

    # ── Sphere mesh cache ──────────────────────────────────────────────────────

    def _sphere(self, r):
        key = round(r, 3)
        if key not in self._mesh_cache:
            self._mesh_cache[key] = gl.MeshData.sphere(rows=14, cols=14, radius=key)
        return self._mesh_cache[key]

    # ── Main rebuild ───────────────────────────────────────────────────────────

    def refresh(self):
        if not _GL_OK or self._gw is None:
            return

        for item in self._items:
            self._gw.removeItem(item)
        self._items.clear()

        nvw = self._nvw
        try:
            c = nvw.gui.circuit
        except RuntimeError:
            return

        # ── X axis: pipeline container position ─────────────────────────────────────────
        depth = nvw._compute_container()

        # Lateralized layer pairs share the same container
        for l in c.layers:
            pair = getattr(l, 'lateral_pair', None)
            if pair:
                partner = next((pl for pl in c.layers if pl.name == pair), None)
                if partner:
                    d = max(depth.get(l.name, 1), depth.get(partner.name, 1), 1)
                    depth[l.name] = d
                    depth[partner.name] = d

        all_containers  = sorted(set(depth.values()))
        container_min   = all_containers[0]
        container_range = max(all_containers[-1] - container_min, 1)

        def _x(container):
            return (container - container_min) / container_range

        # ── Per-object info table ──────────────────────────────────────────────
        # obj_name → {container, z, is_sensor, is_image, aspect, n}
        obj_info = {}

        for s in c.sensors:
            container     = depth.get(s.name, 0)
            z      = float(getattr(s, 'z', 0) or 0) * _Z_SCALE
            is_img = getattr(s, 'is_image_node', False)
            cam_w  = getattr(s, 'width',  1) or 1
            cam_h  = getattr(s, 'height', 1) or 1
            n      = getattr(s, 'n', 1) or 1
            entry  = dict(container=container, z=z, is_sensor=True,
                          is_image=is_img, aspect=cam_h / cam_w, n=n)
            if s.is_lateralized(c):
                obj_info[s.name + '_L'] = entry
                obj_info[s.name + '_R'] = entry
            else:
                obj_info[s.name] = entry

        ring_layers = set()
        for l in c.layers:
            if l.n is None:
                continue
            container     = depth.get(l.name, 1)
            z      = float(getattr(l, 'z', 0) or 0) * _Z_SCALE
            is_img = getattr(l, 'is_image_node', False)
            cam_w  = getattr(l, 'width',  1) or 1
            cam_h  = getattr(l, 'height', 1) or 1
            obj_info[l.name] = dict(container=container, z=z, is_sensor=False,
                                    is_image=is_img, aspect=cam_h / cam_w, n=l.n)
            if getattr(l, 'viz_layout', None) == 'ring':
                ring_layers.add(l.name)

        if not obj_info:
            return

        # ── Per-neuron 3D positions ────────────────────────────────────────────
        # group_pts[obj_name] = list of np.array([x, y, z]), one entry per neuron.
        #
        # Layout is computed entirely from obj_info (pipeline depth, z, n) and is
        # independent of the 2D view's z-cut state.  Each pipeline container
        # divides its Y range equally among its layers; neurons spread uniformly
        # within that slice.  This guarantees every layer is treated identically —
        # same n in the same container always maps to the same Y positions.
        group_pts = defaultdict(list)   # obj_name → [np.array([x,y,z]), ...]

        container_slots = defaultdict(list)
        for name, info in obj_info.items():
            container_slots[info['container']].append(name)

        for container, names in container_slots.items():
            n_grp = len(names)
            for gi, name in enumerate(sorted(names)):
                info   = obj_info[name]
                g_y    = (gi + 0.5) / n_grp - 0.5   # group centre, centred at 0
                x3d    = _x(container)
                z3d    = info['z']
                n_neur = info['n']
                if n_neur <= 1 or info['is_image']:
                    group_pts[name].append(np.array([x3d, g_y, z3d]))
                else:
                    # Space neurons ±spread/2 around the group centre
                    spread = min(0.8 / max(n_grp, 1), 0.14)
                    half   = (n_neur - 1) / 2.0
                    for k in range(n_neur):
                        y3d = g_y + (k - half) / half * spread / 2.0 if half else g_y
                        group_pts[name].append(np.array([x3d, y3d, z3d]))

        # ── Refine Y from the 2D top-view layout ──────────────────────────────
        # The heuristic above gives systematic, z-cut-independent positions, but
        # it knows nothing about which neurons are ipsilateral vs contralateral or
        # where the midline falls.  _positions encodes the actual medio-lateral
        # ordering from the 2D layout.  Override Y for every visible layer so the
        # 3D view faithfully reproduces the left/right structure of the 2D view.
        _raw_pos = getattr(nvw, '_positions', {})
        if _raw_pos:
            all_yp = [yp for (_, yp) in _raw_pos.values()]
            y_mid  = (min(all_yp) + max(all_yp)) / 2.0

            layer_ys = defaultdict(dict)   # name → {neuron_idx: canvas_y}
            for nk, (_, yp) in _raw_pos.items():
                sep = nk.rfind('_')
                if sep > 0 and nk[sep + 1:].isdigit():
                    name = nk[:sep]
                    if name in obj_info:
                        layer_ys[name][int(nk[sep + 1:])] = yp

            for name, ys in layer_ys.items():
                info = obj_info[name]
                x3d  = _x(info['container'])
                z3d  = info['z']
                group_pts[name] = [
                    np.array([x3d, ys[idx] - y_mid, z3d])
                    for idx in sorted(ys)
                ]

        # ── Propagate Y-refinement to hidden (subsumed) layers ───────────────
        # Layers absent from _positions were hidden by the z-cut, so their Y
        # coordinates are still heuristic.  If a refined sibling exists in the
        # same pipeline container with the same n, copy its Y values so that
        # connections between the two layers remain ipsilateral in 3D.
        if _raw_pos:
            refined_names = set(layer_ys.keys())
            for name, info in obj_info.items():
                if name in refined_names:
                    continue
                siblings = [
                    n2 for n2, i2 in obj_info.items()
                    if n2 != name
                    and n2 in refined_names
                    and i2['container'] == info['container']
                    and i2['n'] == info['n']
                ]
                if siblings:
                    ref  = siblings[0]
                    x3d  = _x(info['container'])
                    z3d  = info['z']
                    group_pts[name] = [
                        np.array([x3d, float(p[1]), z3d])
                        for p in group_pts[ref]
                    ]

        # ── Ring-layout layers: rearrange neurons into a YZ circle ────────────
        # Neurons whose layer has viz_layout='ring' are placed on a circle in
        # the YZ plane (perpendicular to the pipeline X axis) so the ring
        # topology is visible in 3D.  Index order from the 2D layout is
        # preserved so the weight pattern maps correctly onto the ring.
        for rname in ring_layers:
            pts = group_pts.get(rname, [])
            n   = len(pts)
            if n < 3:
                continue
            cx = float(pts[0][0])
            cy = float(np.mean([p[1] for p in pts]))
            cz = float(np.mean([p[2] for p in pts]))
            group_pts[rname] = [
                np.array([cx,
                           cy + _RING_RADIUS * np.cos(2 * np.pi * k / n),
                           cz + _RING_RADIUS * np.sin(2 * np.pi * k / n)])
                for k in range(n)
            ]

        # ── Group centroids (arc endpoints) ────────────────────────────────────
        centroids = {name: np.mean(pts, axis=0)
                     for name, pts in group_pts.items()}

        # Lateralized sensor base-name → midpoint between L and R centroids
        # (needed for connections stored as conn.src = base sensor name)
        for s in c.sensors:
            if s.is_lateralized(c):
                kL, kR = s.name + '_L', s.name + '_R'
                if kL in centroids and kR in centroids:
                    centroids[s.name] = (centroids[kL] + centroids[kR]) * 0.5

        n_max = max((info['n'] for info in obj_info.values()), default=1) or 1

        def _hw(n):
            return _CAM_HW_MIN + (_CAM_HW_MAX - _CAM_HW_MIN) * (n / n_max) ** 0.5

        # ── Draw nodes ─────────────────────────────────────────────────────────
        md_neuron = self._sphere(_R_NEURON)   # shared mesh for all neurons

        for obj_name, pts in group_pts.items():
            info = obj_info[obj_name]
            col  = _C_SENSOR if info['is_sensor'] else _C_LAYER

            if info['is_image']:
                # Single plane centred on the group, facing the pipeline axis (+X)
                px, py, pz = centroids[obj_name]
                hw  = _hw(info['n'])
                hh  = hw * info['aspect']
                verts = np.array([
                    [px, py - hw, pz - hh],
                    [px, py + hw, pz - hh],
                    [px, py + hw, pz + hh],
                    [px, py - hw, pz + hh],
                ], dtype=np.float32)
                faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
                md = gl.MeshData(vertexes=verts, faces=faces)
                rgba = (*col[:3], _CAM_ALPHA)
                m = gl.GLMeshItem(meshdata=md, smooth=False, color=rgba,
                                  shader='shaded', glOptions='translucent',
                                  drawEdges=True,
                                  edgeColor=(col[0]*0.6, col[1]*0.6, col[2]*0.6, 1.0))
                self._gw.addItem(m)
                self._items.append(m)
            else:
                for pt in pts:
                    m = gl.GLMeshItem(meshdata=md_neuron, smooth=True, color=col,
                                      shader='shaded', glOptions='opaque')
                    m.translate(float(pt[0]), float(pt[1]), float(pt[2]))
                    self._gw.addItem(m)
                    self._items.append(m)

        # ── Arcs ──────────────────────────────────────────────────────────────────
        for conn in c.connections:
            W    = np.asarray(conn.W, dtype=float)
            frob = float(np.linalg.norm(W))
            if frob < 1e-9:
                continue

            # ── Same-layer (recurrent) connection — draw per-neuron arcs ──────────
            # This renders the actual weight pattern (e.g. ring attractor bump)
            # rather than a single self-loop at the group centroid.
            if (conn.src == conn.tgt
                    and conn.src in group_pts
                    and W.ndim == 2):
                pts_arr = group_pts[conn.src]
                n_pos   = len(pts_arr)
                if n_pos > 1 and W.shape[0] == n_pos and W.shape[1] == n_pos:
                    max_abs   = float(np.max(np.abs(W))) or 1.0
                    thr       = 0.1 * max_abs
                    ring_c    = (centroids[conn.src]
                                 if conn.src in ring_layers else None)
                    for j in range(n_pos):
                        for i in range(n_pos):
                            if i == j:
                                continue
                            w_ij = float(W[i, j])
                            if abs(w_ij) < thr:
                                continue
                            c_ij = _C_EXCIT if w_ij >= 0 else _C_INHIB
                            a_ij = float(np.clip(
                                0.2 + 0.8 * abs(w_ij) / max_abs, 0.15, 1.0))
                            bpts   = _bezier(pts_arr[j], pts_arr[i],
                                             ring_center=ring_c)
                            colors = np.tile(
                                (*c_ij, a_ij), (_N_SEGS, 1)).astype(np.float32)
                            line = gl.GLLinePlotItem(
                                pos=bpts.astype(np.float32),
                                color=colors, width=1.0,
                                antialias=True, mode='line_strip')
                            self._gw.addItem(line)
                            self._items.append(line)
                    continue
                # Fall through to centroid self-loop if dimensions don't match

            # ── Cross-layer connection — per-neuron arcs when W matches ──────────
            # Resolve source positions; lateralized sensors store as name+'_L'/'_R'.
            src_pts = group_pts.get(conn.src, [])
            if not src_pts:
                pts_L = group_pts.get(conn.src + '_L', [])
                pts_R = group_pts.get(conn.src + '_R', [])
                src_pts = pts_L + pts_R

            tgt_pts = group_pts.get(conn.tgt, [])
            if not src_pts or not tgt_pts:
                continue

            n_src, n_tgt = len(src_pts), len(tgt_pts)

            src_info = obj_info.get(conn.src) or obj_info.get(conn.src + '_L', {})
            tgt_info = obj_info.get(conn.tgt, {})

            drew_per_neuron = False
            if (W.ndim == 2 and W.shape == (n_tgt, n_src)
                    and not src_info.get('is_image')
                    and not tgt_info.get('is_image')
                    and (n_src > 1 or n_tgt > 1)):
                max_abs = float(np.max(np.abs(W))) or 1.0
                thr     = 0.1 * max_abs
                pairs   = [(i, j) for j in range(n_src) for i in range(n_tgt)
                           if abs(float(W[i, j])) >= thr]
                if len(pairs) <= _MAX_CROSS_ARCS:
                    for i, j in pairs:
                        w_ij = float(W[i, j])
                        c_ij = _C_EXCIT if w_ij >= 0 else _C_INHIB
                        a_ij = float(np.clip(
                            0.2 + 0.8 * abs(w_ij) / max_abs, 0.15, 1.0))
                        bpts   = _bezier(src_pts[j], tgt_pts[i])
                        colors = np.tile(
                            (*c_ij, a_ij), (_N_SEGS, 1)).astype(np.float32)
                        line   = gl.GLLinePlotItem(
                            pos=bpts.astype(np.float32),
                            color=colors, width=1.0,
                            antialias=True, mode='line_strip')
                        self._gw.addItem(line)
                        self._items.append(line)
                    drew_per_neuron = True

            if not drew_per_neuron:
                # Centroid fallback (image nodes, large matrices, shape mismatch)
                if conn.src in centroids:
                    src_cents = [centroids[conn.src]]
                elif conn.src + '_L' in centroids and conn.src + '_R' in centroids:
                    src_cents = [centroids[conn.src + '_L'],
                                 centroids[conn.src + '_R']]
                else:
                    continue
                if conn.tgt not in centroids:
                    continue

                base  = _C_EXCIT if float(W.mean()) >= 0 else _C_INHIB
                alpha = float(np.clip(0.25 + 0.75 * np.tanh(frob / 3.0), 0.15, 1.0))
                width = float(1.0 + 2.5 * np.tanh(frob / 5.0))

                for sc in src_cents:
                    bpts   = _bezier(sc, centroids[conn.tgt])
                    colors = np.tile((*base, alpha), (_N_SEGS, 1)).astype(np.float32)
                    line   = gl.GLLinePlotItem(pos=bpts.astype(np.float32),
                                               color=colors, width=width,
                                               antialias=True, mode='line_strip')
                    self._gw.addItem(line)
                    self._items.append(line)

        # ── Reference grid (XY plane at Z = 0) ────────────────────────────────
        grid = gl.GLGridItem()
        grid.setSize(x=1.4, y=1.0)
        grid.setSpacing(x=0.2, y=0.1)
        grid.translate(0.5, 0.0, -0.05)
        grid.setColor((160, 170, 190, 50))
        self._gw.addItem(grid)
        self._items.append(grid)

        # ── Axis indicator: thick colored lines + labels ──────────────────────
        # Placed at the upper-left of the scene (matches pyqtgraph convention:
        # X=yellow, Y=green, Z=blue).
        _ax_o = np.array([-0.12, 0.58, 0.0])   # origin of the axis widget
        _ax_s = 0.16                             # arm length
        _ax_arms = [
            ('X', _ax_o + np.array([_ax_s, 0,     0    ]), (1.0, 0.85, 0.0, 1.0)),
            ('Y', _ax_o + np.array([0,     _ax_s, 0    ]), (0.2, 0.9,  0.2, 1.0)),
            ('Z', _ax_o + np.array([0,     0,     _ax_s]), (0.3, 0.55, 1.0, 1.0)),
        ]
        for lbl, tip, rgba in _ax_arms:
            seg  = np.array([_ax_o, tip], dtype=np.float32)
            cols = np.tile(rgba, (2, 1)).astype(np.float32)
            line = gl.GLLinePlotItem(pos=seg, color=cols, width=3.5,
                                     antialias=True, mode='line_strip')
            self._gw.addItem(line)
            self._items.append(line)
            try:
                txt = gl.GLTextItem(pos=tip.astype(np.float32),
                                    text=lbl,
                                    color=(int(rgba[0]*255), int(rgba[1]*255),
                                           int(rgba[2]*255), 255))
                self._gw.addItem(txt)
                self._items.append(txt)
            except Exception:
                pass
