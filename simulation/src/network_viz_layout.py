"""
network_viz_layout.py — pure data/layout model for the network visualizer.

LayoutEngine (the window's `layout_engine`) computes where every node and
column goes — compute() returns a LayoutResult, kept by the window as self._lay —
and holds the column bookkeeping that edits use (snapping, insert / compact /
pin containers). Those only touch plain circuit-model fields (obj.layer /
.viz_row), never Qt items. Pure geometry helpers (bezier, bow, hemisphere) are
module functions.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

from lateral import SIDES, mirror_name, half_names, side_of


@dataclass
class LayoutResult:
    """Where everything goes: the output of LayoutEngine.compute(), kept as
    self._lay. Only compute() writes it; render, editing and the side/3D views
    read it."""
    positions: dict = field(default_factory=dict)           # node_key → (x, y)
    sensor_nodes: set = field(default_factory=set)          # node keys drawn as hollow sensor nodes
    groups: list = field(default_factory=list)              # (kind, name, node_keys, x, container, top, bottom, color)
    container_x_map: dict = field(default_factory=dict)     # container → x
    node_container_map: dict = field(default_factory=dict)  # node_key → container
    container_span_map: dict = field(default_factory=dict)  # container → span (only spans > 1)
    container_ids: list = field(default_factory=list)       # occupied containers, sorted
    x_unit: float = 1.0                                     # data-x per container step
    palette_x: float = 0.5                                  # x centre of the network
    # Positions of the whole network ignoring the z-cut — the view fit uses
    # these so scrubbing the z-cut slider doesn't rescale the picture.
    full_positions: dict = field(default_factory=dict)
    # z-cut state (see _apply_z_cut); active_names is None when no z-cut is set.
    active_names: set = None
    subsumed_by: dict = field(default_factory=dict)
    ghost_count: dict = field(default_factory=dict)
    ghosts_by_container: dict = field(default_factory=dict)
    native_col_n: dict = field(default_factory=dict)
    container_winner_names: dict = field(default_factory=dict)


def _eff_n(obj):
    """Effective neuron count for layout. Multi-body sensors return n_total;
    image-display layers (viz_n set) return that override; others return n."""
    vn = getattr(obj, 'viz_n', None)
    if vn is not None:
        return vn
    nt = getattr(obj, 'n_total', None)
    return (nt if nt is not None else (obj.n or 0)) or 0


class _SplitHalf:
    """Stand-in for one half of a lateralized sensor (camera or joint-pair) in the network layout."""
    viz_layout = None

    def __init__(self, sensor, side):
        self.name         = f'{sensor.name}_{side}'
        self._viz_color   = getattr(sensor, '_viz_color', '#888888')
        self.viz_color    = self._viz_color
        self._sensor      = sensor
        self._side        = side
        self.is_image_node = sensor.is_image_node
        self.lateral_pair = mirror_name(self.name)
        if sensor.is_camera:
            # Camera sensor: pixel-split half
            l_end, r_start = sensor._half_bounds()
            self._px_start, self._px_end = (0, l_end) if side == 'L' else (r_start, sensor.width)
            self.n       = 1
            self.n_total = 1
        else:
            # Non-camera lateralized sensor (joint-pair): each half has the sensor's n outputs
            self.n       = sensor.n or 1
            self.n_total = sensor.n or 1


def _sensor_is_lateralized(sensor, circuit=None):
    return sensor.is_lateralized(circuit)


_mirror_name = mirror_name   # kept for callers in the visualizer modules


def _make_conv2d_filter(name, in_ch, kH, kW):
    """Generate a (in_ch, kH, kW) kernel array for a named preset."""
    W  = np.zeros((in_ch, kH, kW), dtype=float)
    ch = kH // 2
    cw = kW // 2  # centre pixel

    if name == 'Grayscale':
        # True channel average: output = (R+G+B)/3 per pixel. No zero-sum so
        # absolute luminance is preserved — useful with pool='none' to convert
        # an RGB camera to a grayscale spatial map.
        W[:, ch, cw] = 1.0 / max(in_ch, 1)
        return W   # skip zero-sum enforcement

    elif name == 'Luminance':
        W[:, ch, cw] = 1.0 / max(in_ch, 1)

    elif name == 'R−G' and in_ch >= 2:
        W[0, ch, cw] =  1.0
        W[1, ch, cw] = -1.0

    elif name == 'G−R' and in_ch >= 2:
        W[0, ch, cw] = -1.0
        W[1, ch, cw] =  1.0

    elif name == 'R−B' and in_ch >= 3:
        W[0, ch, cw] =  1.0
        W[2, ch, cw] = -1.0

    elif name == 'B−R' and in_ch >= 3:
        W[0, ch, cw] = -1.0
        W[2, ch, cw] =  1.0

    elif name in ('On-centre', 'Off-centre'):
        kx = np.arange(kW, dtype=float) - cw
        ky = np.arange(kH, dtype=float) - ch
        se = max(min(kH, kW) / 4.0, 0.5)
        si = max(min(kH, kW) / 2.0, 1.0)
        xx, yy = np.meshgrid(kx, ky)
        d2 = xx**2 + yy**2
        sp = 2.0 * np.exp(-d2 / (2 * se**2)) - np.exp(-d2 / (2 * si**2))
        mx = np.abs(sp).max()
        sp = sp / mx if mx > 0 else sp
        if name == 'Off-centre':
            sp = -sp
        for c in range(in_ch):
            W[c] = sp / max(in_ch, 1)

    elif name in ('Edge →', 'Edge ←', 'Edge ↑', 'Edge ↓'):
        if name in ('Edge →', 'Edge ←'):
            row = np.zeros(kW)
            if kW >= 3:
                row[cw - 1] = -1.0 / max(in_ch, 1)
                row[cw + 1] =  1.0 / max(in_ch, 1)
            if name == 'Edge ←':
                row = -row
            for c in range(in_ch):
                W[c] = row[np.newaxis, :]
        else:
            col = np.zeros(kH)
            if kH >= 3:
                col[ch - 1] = -1.0 / max(in_ch, 1)
                col[ch + 1] =  1.0 / max(in_ch, 1)
            if name == 'Edge ↓':
                col = -col
            for c in range(in_ch):
                W[c] = col[:, np.newaxis]

    elif name.startswith('Ch ') and in_ch > 1:
        idx = int(name.split()[1])
        if idx < in_ch:
            W[idx, ch, cw] = 1.0

    # Enforce zero-sum: subtract global mean so uniform input gives zero output.
    # No-op for filters that already sum to zero (chromatic, edge).
    W -= W.mean()
    return W


def _tile_conv_filters(W):
    """(n_filters, in_ch, kH, kW) → 2-D tiled array for heatmap display."""
    n_filters, _in_ch, kH, kW = W.shape
    kernels = W.mean(axis=1)                     # (n_filters, kH, kW)
    cols = max(1, int(np.ceil(np.sqrt(n_filters))))
    rows = int(np.ceil(n_filters / cols))
    tile = np.zeros((rows * kH, cols * kW))
    for i, k in enumerate(kernels):
        r, c = divmod(i, cols)
        tile[r * kH:(r + 1) * kH, c * kW:(c + 1) * kW] = k
    return tile


def _z_of(obj):
    return getattr(obj, 'z', 0) or 0


def _z_cut_winners(by_container, z_cut, aliases):
    """Per-column winners: the element(s) with the highest z ≤ z_cut.

    Returns (active_names, subsumed_by, container_winners):
    active_names      — names (incl. aliases) of every winner;
    subsumed_by       — same-column loser name → first winner's name, so
                        connection drawing can remap e.g. S→A1 to S→B when B
                        wins column 1 over A1;
    container_winners — container → (winners, winner_z)."""
    active_names = set()
    subsumed_by  = {}   # element_name → subsumer_name
    container_winners  = {}   # container → (winners_list, winner_z_val)
    for container, container_objs in by_container.items():
        eligible = [o for o in container_objs if _z_of(o) <= z_cut]
        if not eligible:
            continue
        # Keep ALL elements at the peak z level — max() would drop
        # siblings sharing the same column and z (e.g. two sensors
        # both at z=0 depth=0), which makes their connections vanish.
        max_z = max(_z_of(o) for o in eligible)
        winners = [o for o in eligible if _z_of(o) == max_z]
        losers  = [o for o in eligible if _z_of(o) < max_z]
        for o in winners:
            active_names.update(aliases(o))
        # Map each same-column loser to the first winner so connections
        # through that loser are drawn via the subsumer instead.
        if losers and winners:
            for loser in losers:
                for name in aliases(loser):
                    subsumed_by[name] = winners[0].name
        container_winners[container] = (winners, max_z)
    return active_names, subsumed_by, container_winners


def _span_covered(by_container, container_winners):
    """(winner, container, nbr_container, nbr_obj) for every lower-z element
    natively in a column that a spanning winner's span reaches into."""
    for container, (winners, winner_z) in container_winners.items():
        for winner in winners:
            span = max(1, getattr(winner, 'span', 1) or 1)
            for j in range(1, span):
                nbr_container = container + j
                for nbr_obj in by_container.get(nbr_container, ()):
                    if _z_of(nbr_obj) < winner_z:
                        yield winner, container, nbr_container, nbr_obj


def _ghosts(by_container, container_winners):
    """ghosts_by_container[container] = every object hidden "behind" the
    panel currently active at `container`, each carrying its OWN native column
    + span (not the winner's) so it renders as its own correctly sized/
    positioned rect + label rather than a merged shadow. Two ways an object
    ends up behind a winner:
      (a) same native column, different z (classic same-column stack)
      (b) the winner's own span reaches into this object's native column
          (e.g. a two-column A->B pipeline sitting entirely behind a wider,
          higher-z container spanning both columns)
    Symmetric by construction: whichever of two overlapping objects currently
    loses is a ghost of whichever currently wins, and a spanning object that
    ISN'T currently winning anything still shows up (case a, at its own
    native column) as a ghost sized to its own full span — e.g. a dormant wide
    container reappears as one wide ghost band behind the several narrower
    winners that currently occupy the columns it would otherwise cover.
    Lateralized halves are skipped — the base name covers both."""
    def ghost(obj, native_container):
        return {'name': obj.name, 'native_container': native_container,
                'span': max(1, getattr(obj, 'span', 1) or 1), 'z': _z_of(obj)}

    ghosts = defaultdict(list)
    for container, container_objs in by_container.items():
        winners, _ = container_winners.get(container, ([], None))
        if not winners:
            continue   # nothing active here to anchor a ghost to
        winner_names = {o.name for o in winners}
        seen = set()
        for o in container_objs:
            if o.name in winner_names or o.name in seen or side_of(o.name):
                continue
            seen.add(o.name)
            ghosts[container].append(ghost(o, container))
    for _winner, container, nbr_container, nbr_obj in _span_covered(by_container, container_winners):
        if not side_of(nbr_obj.name):
            ghosts[container].append(ghost(nbr_obj, nbr_container))
    return dict(ghosts)


def _slot_list(regular_items):
    """One column's (kind, obj) items as slots: lateral pairs share a slot,
    L half first — [('pair', L_ct, L, R_ct, R) | ('single', ct, obj)]."""
    slot_list  = []
    seen_pairs = set()
    for ct, obj in regular_items:
        pair_name = getattr(obj, 'lateral_pair', None)
        if pair_name is None:
            slot_list.append(('single', ct, obj))
            continue
        key = tuple(sorted([obj.name, pair_name]))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        partner_item = next(
            ((pct, pobj) for pct, pobj in regular_items if pobj.name == pair_name), None)
        if partner_item is None:
            slot_list.append(('single', ct, obj))   # orphaned half
        else:
            pct, pobj = partner_item
            # The L half always comes first in a pair slot.
            if side_of(obj.name) == 'R':
                slot_list.append(('pair', pct, pobj, ct, obj))
            else:
                slot_list.append(('pair', ct, obj, pct, pobj))
    return slot_list


def _reversed_idx(i: int, n: int, side: str) -> int:
    """Map neuron/filter index i to visual position.
    R-side reverses ordering so R[0] is at visual bottom, R[n-1] near midline."""
    return n - 1 - i if side == 'R' else i


def _ctrl_pt(p0, p1, bow):
    mx, my = (p0[0]+p1[0])/2, (p0[1]+p1[1])/2
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    chord = (dx*dx + dy*dy) ** 0.5
    if chord > 1e-9 and bow != 0:
        # bow*chord gives perpendicular displacement; cap it so long-span arcs stay on-canvas
        scale = min(1.0, 0.35 / (abs(bow) * chord))
        return (mx + bow * dy * scale, my - bow * dx * scale)
    return (mx + bow * dy, my - bow * dx)


def _bezier_pts(p0, ctrl, p1, n=80):
    ts = np.linspace(0, 1, n)
    xs = (1-ts)**2 * p0[0] + 2*(1-ts)*ts * ctrl[0] + ts**2 * p1[0]
    ys = (1-ts)**2 * p0[1] + 2*(1-ts)*ts * ctrl[1] + ts**2 * p1[1]
    return list(zip(xs.tolist(), ys.tolist()))


def _clip_pts(pts, src, tgt, r):
    start = 0
    for i, (x, y) in enumerate(pts):
        if np.hypot(x - src[0], y - src[1]) > r:
            start = i
            break
    else:
        return []
    end = len(pts) - 1
    for i in range(start, len(pts)):
        if np.hypot(pts[i][0] - tgt[0], pts[i][1] - tgt[1]) < r:
            end = max(start, i - 1)
            break
    return pts[start:end + 1]


def _hemisphere(idx, n):
    """True = left/top half, False = right/bottom half, None = single neuron (midline)."""
    if n <= 1:
        return None
    return idx < n // 2


def _signed_bow(src_hem, tgt_hem, base_bow, fallback_y):
    """Compute signed bow from hemisphere assignments.

    Same hemisphere  → bow away from midline (±base_bow).
    Cross hemisphere → small symmetric bow: src-upper positive, src-lower negative.
    n=1 midline neuron → fall back to y-average.
    """
    if src_hem is None or tgt_hem is None:
        return base_bow if fallback_y >= 0.5 else -base_bow
    if src_hem == tgt_hem:
        return base_bow if src_hem else -base_bow
    # Cross: half-bow so it's visually distinct from ipsilateral, but never 0
    # (a zero bow would be clamped to +min_bow for all cross connections,
    # breaking the up/down symmetry).
    return (base_bow * 0.4) if src_hem else -(base_bow * 0.4)


def _dist_to_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    denom = dx*dx + dy*dy
    if denom == 0:
        return np.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1)*dx + (py - y1)*dy) / denom))
    return np.hypot(px - (x1 + t*dx), py - (y1 + t*dy))


class LayoutEngine:
    """The layout model of the network editor: where every node and column
    goes (compute() → LayoutResult), plus the column bookkeeping edits use
    (snapping, inserting / compacting containers). Reads the circuit and the
    view settings from the window; no Qt."""

    def __init__(self, win):
        self.win = win

    def connection_kind(self, src_obj, tgt_obj, W, ns, nt):
        """Classify a connection for drawing dispatch (see rules/network_viz.md).

        Priority: TD → CONV4D → DENSE → THIN → THICK.
        src_obj / tgt_obj are the resolved layer or sensor objects (tgt is always a layer).
        """
        from neurons import LearningLayerBase as _LLB, SnapshotLayer as _SnapL
        if (isinstance(tgt_obj, _SnapL) and src_obj is not None
                and getattr(src_obj, 'name', None) == getattr(tgt_obj, 'teach_source', None)):
            return self.win._CK_TEACH
        if isinstance(tgt_obj, _LLB):
            return self.win._CK_TD
        if W.ndim == 4:
            return self.win._CK_CONV4D
        if max(ns, nt) > self.win._DENSE_THRESHOLD:
            return self.win._CK_DENSE
        if (ns * nt > 4
                or (getattr(src_obj, 'is_image_node', False)
                    and getattr(tgt_obj, 'is_image_node', False))):
            return self.win._CK_THIN
        return self.win._CK_THICK

    def n_map(self):
        circuit = self.win.gui.circuit
        m = {}
        for s in circuit.sensors:
            if _sensor_is_lateralized(s, circuit):
                if s.is_camera:
                    for side, half in zip(SIDES, half_names(s.name)):
                        m[half] = s.half_width(side) * s.in_ch
                else:
                    # Joint-pair sensor: each half has sensor.n outputs
                    m[f'{s.name}_L'] = s.n or 1
                    m[f'{s.name}_R'] = s.n or 1
            else:
                m[s.name] = s.n
        m.update({l.name: (getattr(l, 'viz_n', None) or l.n)
                  for l in circuit.layers if l.n is not None})
        return m

    def compute_container(self):
        c     = self.win.gui.circuit
        depth = {s.name: getattr(s, 'layer', 0) for s in c.sensors}
        for s in c.sensors:
            if _sensor_is_lateralized(s, c):
                d = getattr(s, 'layer', 0)
                depth[f'{s.name}_L'] = d
                depth[f'{s.name}_R'] = d
        for lyr in c.layers:
            if getattr(lyr, 'layer', None) is not None:
                depth[lyr.name] = lyr.layer
        for lyr in c.layers:
            if getattr(lyr, 'layer', None) is not None:
                continue
            src_depths = [depth[conn.src] for conn in c.connections
                          if conn.tgt == lyr.name and conn.src in depth]
            depth[lyr.name] = (max(src_depths) + 1) if src_depths else 1
        return depth

    def compute(self):
        """Compute the layout of the current circuit as a LayoutResult."""
        lay = LayoutResult()
        circuit  = self.win.gui.circuit
        layers   = [l for l in circuit.layers if l.n is not None]
        actual_sensors = list(circuit.sensors)
        # Unfiltered snapshots, taken before the z-cut below may reassign
        # `layers`/`actual_sensors` to a subsumed-down subset — used to compute
        # a stable view-fit extent that doesn't rescale as the z-cut slider moves.
        full_layers, full_sensors = layers, actual_sensors

        if self.win._z_cut is not None:
            layers, actual_sensors = self._apply_z_cut(layers, actual_sensors, self.win._z_cut, lay)

        brain_cls   = self.win.gui.brain.__class__ if self.win.gui.brain else None
        viz_columns = getattr(brain_cls, 'viz_columns', None)

        (lay.positions, lay.sensor_nodes, lay.groups, lay.container_span_map,
         lay.x_unit, lay.container_ids) = self._compute_positions(actual_sensors, layers, viz_columns)

        # The view's fit range is sourced from full_positions rather than
        # `positions` so that scrubbing the z-cut slider — which only changes which subset
        # of `full_layers`/`full_sensors` survives filtering above, never the underlying
        # circuit — doesn't rescale the whole picture. Skipped when z-cut is inactive
        # (active_names is None): filtered == full already, no need to redo the work.
        if lay.active_names is not None:
            lay.full_positions, *_ = self._compute_positions(
                full_sensors, full_layers, viz_columns, record_side_effects=False)
        else:
            lay.full_positions = lay.positions

        self._assign_columns(lay)
        if lay.positions:
            xs = [p[0] for p in lay.positions.values()]
            lay.palette_x = (min(xs) + max(xs)) / 2
        return lay

    def _assign_columns(self, lay):
        """Record each column's x and each node's column; in compact mode,
        redistribute the visible columns evenly over the same x range."""
        positions = lay.positions
        for _, _, container_nodes, x_col, container, *_ in lay.groups:
            lay.container_x_map[container] = x_col
            for node_key in container_nodes:
                lay.node_container_map[node_key] = container

        # Compact layout: redistribute visible columns evenly, preserving the
        # coordinate space (x_unit) used by _compute_positions so spacing is consistent.
        if not (self.win._compact_mode and self.win._hidden_containers):
            return
        all_containers = sorted(lay.container_x_map.keys())
        vis_containers = [d for d in all_containers if d not in self.win._hidden_containers]
        n_vis = len(vis_containers)
        if not 0 < n_vis < len(all_containers):
            return
        all_xs = [lay.container_x_map[d] for d in all_containers]
        x_min, x_max = min(all_xs), max(all_xs)
        if n_vis == 1:
            new_xs = {vis_containers[0]: (x_min + x_max) / 2}
        else:
            step = (x_max - x_min) / (n_vis - 1)
            new_xs = {container: x_min + i * step for i, container in enumerate(vis_containers)}
        for node_key, container in lay.node_container_map.items():
            if container in new_xs and node_key in positions:
                old_x, y = positions[node_key]
                dx = new_xs[container] - lay.container_x_map[container]
                positions[node_key] = (old_x + dx, y)
        for container, nx in new_xs.items():
            lay.container_x_map[container] = nx

    def _paired_container_map(self, layers):
        """_compute_container(), with lateral pairs forced into the same column
        (≥1, never the sensor column 0)."""
        depth = self.compute_container()
        for l in layers:
            pair_name = getattr(l, 'lateral_pair', None)
            if pair_name:
                partner = next((pl for pl in layers if pl.name == pair_name), None)
                if partner:
                    d = max(depth.get(l.name, 1), depth.get(partner.name, 1), 1)
                    depth[l.name] = d
                    depth[partner.name] = d
        return depth

    def _apply_z_cut(self, layers, actual_sensors, z_cut, lay):
        """Z-depth filter: for each column keep only the element(s) with the
        highest z ≤ z_cut. Records the z-cut state in `lay` (active_names,
        subsumed_by, ghost_count, ghosts_by_container, native_col_n,
        container_winner_names) and returns the surviving (layers, sensors)."""
        circuit = self.win.gui.circuit
        zcol_container = self._paired_container_map(layers)
        by_container = defaultdict(list)
        for obj in actual_sensors + layers:
            by_container[zcol_container.get(obj.name, 0)].append(obj)
        # native_col_n[container] = tallest neuron count among objects that
        # natively live there — used to estimate a ghost's height when it
        # isn't currently active (so its real position isn't available).
        lay.native_col_n = {
            col: max((getattr(o, 'viz_n', None) or getattr(o, 'n', 1) or 1) for o in objs)
            for col, objs in by_container.items()
        }

        def aliases(o):
            # The names an element is known by in connections: its own, its
            # lateral partner's, and — for a lateralized sensor — its _L/_R
            # halves (the sensor's own name never appears as a Connection
            # endpoint, only its halves do).
            names = [o.name]
            if getattr(o, 'lateral_pair', None):
                names.append(o.lateral_pair)
            if o in actual_sensors and _sensor_is_lateralized(o, circuit):
                names += [f'{o.name}_L', f'{o.name}_R']
            return names

        active_names, subsumed_by, container_winners = _z_cut_winners(by_container, z_cut, aliases)

        # Span-based subsumption — a spanning winner at z>0 also removes
        # lower-z elements in the columns its span covers. E.g. a two-column
        # pipeline A (col 0) -> B (col 1) at z=0 running in parallel with a
        # wider container C (col 0, span=2, z=1): C wins both columns, and A/B
        # are recorded (in ghosts_by_container) so C's panel can show each of
        # them as its own ghost.
        for winner, _container, _nbr_container, nbr_obj in _span_covered(by_container, container_winners):
            for name in aliases(nbr_obj):
                active_names.discard(name)
                subsumed_by[name] = winner.name

        lay.active_names = active_names
        lay.subsumed_by  = subsumed_by
        # ghost_count[container] = (distinct z-levels in column container) - 1.
        # Computed from all elements regardless of z_cut direction, so the
        # count is the same whether you're looking up or down the z stack.
        # Used only for _container_key / auto-titling a same-column-shared
        # panel from its current winner — the ghost *rendering* itself is
        # driven by ghosts_by_container below.
        all_container_z = defaultdict(set)
        for obj in list(circuit.sensors) + [l for l in circuit.layers if l.n is not None]:
            dv = zcol_container.get(obj.name)
            if dv is not None:
                all_container_z[dv].add(getattr(obj, 'z', 0) or 0)
        lay.ghost_count = {dv: len(zs) - 1 for dv, zs in all_container_z.items() if len(zs) > 1}
        lay.ghosts_by_container = _ghosts(by_container, container_winners)
        # winner_names[container] = name(s) of whoever currently wins that
        # container — for containers shared across z-levels, the panel
        # title auto-follows this instead of a manual label, so it never
        # goes stale as the z-cut slider changes who's in front.
        lay.container_winner_names = {
            container: [o.name for o in winners]
            for container, (winners, _) in container_winners.items() if winners
        }
        return ([l for l in layers if l.name in active_names],
                [s for s in actual_sensors if s.name in active_names])

    def _compute_positions(self, actual_sensors, layers, viz_columns, record_side_effects=True):
        """Column/slot layout for `actual_sensors`/`layers` (already z-cut-filtered by the
        caller, or not, for the caller's own purposes — this method doesn't care).

        record_side_effects gates the two persistent-object mutations this method makes
        (`obj.viz_row`, ring layout's `obj._ring_cx/_ring_cy/_ring_r`) — set False for a
        "shadow" call whose only purpose is measuring an extent, so it doesn't clobber
        state a real/filtered call already set (or will set) on the same objects.
        """
        circuit = self.win.gui.circuit

        # Replace lateralized sensors with L/R SplitHalf stand-ins for layout purposes
        sensors = []
        for s in actual_sensors:
            if _sensor_is_lateralized(s, circuit):
                sensors.append(_SplitHalf(s, 'L'))
                sensors.append(_SplitHalf(s, 'R'))
            else:
                sensors.append(s)

        container_groups, container_ids = self._container_groups(sensors, layers, viz_columns)

        positions    = {}
        sensor_nodes = set()
        groups       = []

        panel_hw = self.win._NODE_R + self.win._PAD_X
        container_gap  = 0.06
        # x comes from the column number, not the rank among occupied columns:
        # spans and ghost panels (z-cut) are measured in column numbers too, so
        # rank-based x made them overlap their neighbours. The closest pair of
        # occupied columns sits one panel width + gap apart; no minimum total
        # width (that spread small networks far apart).
        if len(container_ids) > 1:
            min_ds  = min(b - a for a, b in zip(container_ids, container_ids[1:]))
            x_unit  = (2 * panel_hw + container_gap) / min_ds
        else:
            x_unit  = 1.0

        container_span_map = {}   # container → span  (only entries with span > 1)

        for col, container in zip(container_groups, container_ids):
            # Spanning containers sit at the midpoint of their covered columns.
            _max_span = (max((getattr(obj, 'span', 1) or 1) for _, obj in col)
                         if col else 1)
            if _max_span > 1:
                container_span_map[container] = _max_span
            x = (container + (_max_span - 1) / 2.0) * x_unit

            regular_items = [(ct, obj) for ct, obj in col
                             if getattr(obj, 'viz_layout', None) != 'ring']
            self._place_slots(_slot_list(regular_items), x, container,
                              positions, sensor_nodes, groups, record_side_effects)
            for _, obj in col:
                if getattr(obj, 'viz_layout', None) == 'ring':
                    self._place_ring(obj, x, container, positions, groups, record_side_effects)

        return positions, sensor_nodes, groups, container_span_map, x_unit, container_ids

    def _container_groups(self, sensors, layers, viz_columns):
        """The ordered (kind, obj) members of each column, and the column ids —
        from the brain's viz_columns if it declares them, else from depth."""
        name_map = {s.name: ('sensor', s) for s in sensors}
        name_map.update({l.name: ('layer', l) for l in layers})

        if viz_columns:
            container_groups = [
                [name_map[n] for n in grp if n in name_map]
                for grp in viz_columns
            ]
            container_groups = [col for col in container_groups if col]
            return container_groups, list(range(len(container_groups)))

        depth = self._paired_container_map(layers)
        by_container = defaultdict(list)
        for s in sensors:
            by_container[depth.get(s.name, 0)].append(('sensor', s))
        for l in layers:
            by_container[depth.get(l.name, 1)].append(('layer', l))
        sorted_depths = sorted(by_container)
        container_groups = []
        for d in sorted_depths:
            items = by_container[d]
            # Sort by viz_row if set; preserve natural insertion order otherwise.
            indexed = list(enumerate(items))
            indexed.sort(key=lambda p: getattr(p[1][1], 'viz_row', 1000 + p[0]))
            container_groups.append([item for _, item in indexed])
        return container_groups, sorted_depths

    def _place_slots(self, slot_list, x, container, positions, sensor_nodes, groups,
                     record_side_effects):
        """Outside-in index assignment for one column's slots.

        The first entry claims the outermost positions (top + bottom), the
        second entry claims the next-innermost, and so on until all entries
        converge to the centre.  Pairs: L→top, R→bottom.
        Odd-n entry: ceil(n/2) top, floor(n/2) bottom."""
        total_N = 0
        for entry in slot_list:
            if entry[0] == 'pair':
                _, L_ct, L_obj, R_ct, R_obj = entry
                total_N += max(_eff_n(L_obj), 1) + max(_eff_n(R_obj), 1)
            else:
                total_N += max(_eff_n(entry[2]), 1)
        total_N = total_N or 1
        d       = 2.0 * self.win._NODE_R
        spacing = min(d, 1.0 / total_N)

        def _iy(k):
            return 0.5 + (total_N - 1 - 2 * k) * spacing / 2

        top_ptr = 0
        bot_ptr = total_N - 1

        for slot_idx, entry in enumerate(slot_list):
            if entry[0] == 'pair':
                _, L_ct, L_obj, R_ct, R_obj = entry
                n_L = max(_eff_n(L_obj), 1)
                n_R = max(_eff_n(R_obj), 1)
                t0, b0 = top_ptr, bot_ptr
                for i in range(n_L):
                    positions[f'{L_obj.name}_{i}'] = (x, _iy(top_ptr + i))
                top_ptr += n_L
                r_start = bot_ptr - n_R + 1
                for i in range(n_R):
                    positions[f'{R_obj.name}_{i}'] = (x, _iy(r_start + i))
                bot_ptr -= n_R
                slot_top = _iy(t0) + spacing / 2
                slot_bot = _iy(b0) - spacing / 2
                L_nodes = [f'{L_obj.name}_{j}' for j in range(n_L)]
                R_nodes = [f'{R_obj.name}_{j}' for j in range(n_R)]
                if L_ct == 'sensor': sensor_nodes.update(L_nodes)
                if R_ct == 'sensor': sensor_nodes.update(R_nodes)
                groups.append((L_ct, L_obj.name, L_nodes, x, container,
                               slot_top, slot_bot, L_obj.viz_color))
                groups.append((R_ct, R_obj.name, R_nodes, x, container,
                               slot_top, slot_bot, R_obj.viz_color))
                if record_side_effects:
                    if L_ct == 'layer': L_obj.viz_row = slot_idx
                    if R_ct == 'layer': R_obj.viz_row = slot_idx

            else:   # 'single'
                _, ct, obj = entry
                en      = max(_eff_n(obj), 1)
                n_top   = (en + 1) // 2
                n_bot   = en // 2
                y_order = list(getattr(obj, 'y_order', None) or range(en))
                t0, b0  = top_ptr, bot_ptr
                for i in range(n_top):
                    positions[f'{obj.name}_{y_order[i]}'] = (x, _iy(top_ptr + i))
                top_ptr += n_top
                b_start = bot_ptr - n_bot + 1
                for i in range(n_bot):
                    positions[f'{obj.name}_{y_order[n_top + i]}'] = (x, _iy(b_start + i))
                bot_ptr -= n_bot
                slot_top = _iy(t0) + spacing / 2
                slot_bot = _iy(b0) - spacing / 2
                container_nodes = [f'{obj.name}_{j}' for j in range(en)]
                if ct == 'sensor': sensor_nodes.update(container_nodes)
                groups.append((ct, obj.name, container_nodes, x, container,
                               slot_top, slot_bot, obj.viz_color))
                if record_side_effects and ct == 'layer':
                    obj.viz_row = slot_idx

    def _place_ring(self, obj, x, container, positions, groups, record_side_effects):
        """Ring layout: neurons arranged in a circle, neuron 0 at the top."""
        n_ring  = obj.n
        ring_r  = max(0.15, min(0.38,
                      self.win._RING_SCALE * self.win._NODE_R / (2 * np.sin(np.pi / n_ring) + 1e-9)))
        cx, cy  = x, 0.5
        if record_side_effects:
            obj._ring_cx = cx
            obj._ring_cy = cy
            obj._ring_r  = ring_r
        for i in range(n_ring):
            angle = np.pi / 2 - 2 * np.pi * i / n_ring  # CCW, node 0 at top
            positions[f'{obj.name}_{i}'] = (
                cx + ring_r * np.cos(angle),
                cy + ring_r * np.sin(angle),
            )
        container_nodes = [f'{obj.name}_{j}' for j in range(n_ring)]
        groups.append(('layer', obj.name, container_nodes, x, container,
                       cy + ring_r + self.win._NODE_R,
                       cy - ring_r - self.win._NODE_R, getattr(obj, 'color', None)))

    def infer_and_set_n(self):
        for conn in self.win.gui.circuit.connections:
            src, tgt, W = conn.src, conn.tgt, conn.W
            W = np.asarray(W, dtype=float)
            if W.ndim != 2:
                continue
            for layer in self.win.gui.circuit.layers:
                if layer.n is not None:
                    continue
                if layer.name == src:
                    layer._ensure_n(W.shape[1])
                elif layer.name == tgt:
                    layer._ensure_n(W.shape[0])

    def container_key(self, container):
        """Stable identity string for whichever occupant currently fronts this
        container slot — used to key manual labels so they survive drags and
        renumbering instead of a raw position. Usually a single name; more
        than one only when the container is shared across z-levels, in which
        case it's whoever currently wins that slot."""
        names = None
        if self.win._lay.ghost_count.get(container, 0) > 0:
            names = self.win._lay.container_winner_names.get(container)
        if not names:
            names = sorted({nk.rsplit('_', 1)[0]
                             for nk, c in self.win._lay.node_container_map.items() if c == container})
        return '|'.join(sorted(names))

    def first_occupant_name(self, container):
        """Name of whichever occupant of *container* was added to the circuit
        earliest (sensors/layers lists are append-only, so list order is
        creation order) — used as the default display caption for a
        container that has no explicit "Set label..." value."""
        names = {nk.rsplit('_', 1)[0]
                 for nk, c in self.win._lay.node_container_map.items() if c == container}
        if not names:
            return None
        for obj in list(self.win.gui.circuit.sensors) + list(self.win.gui.circuit.layers):
            if obj.name in names:
                return obj.name
        return None

    def get_snap_x(self, mouse_x):
        # Exclude hidden columns so drops never silently land in an invisible column.
        vis_xs = sorted(cx for d, cx in self.win._lay.container_x_map.items()
                        if d not in self.win._hidden_containers)
        col_xs = vis_xs if vis_xs else sorted(self.win._lay.container_x_map.values())
        if not col_xs:
            return mouse_x
        nearest = min(col_xs, key=lambda cx: abs(cx - mouse_x))
        if abs(nearest - mouse_x) < self.win._lay.x_unit * 0.4:
            return nearest
        for a, b in zip(col_xs, col_xs[1:]):
            if a < mouse_x < b:
                return (a + b) / 2
        # Beyond the edge columns — allow creating a new column outside the existing range.
        if mouse_x < col_xs[0]:
            return col_xs[0] - self.win._lay.x_unit
        if mouse_x > col_xs[-1]:
            return col_xs[-1] + self.win._lay.x_unit
        return nearest

    def snap_is_existing(self, snap_x):
        return any(abs(snap_x - cx) < 1e-9
                   for d, cx in self.win._lay.container_x_map.items()
                   if d not in self.win._hidden_containers)

    def container_mates(self, layer):
        """Return all layers that share the same visual column as *layer*.

        Uses the current ``_positions`` dict rather than the raw ``.layer``
        attribute so that layers with ``layer=None`` (e.g. the _L half of a
        lateralized pair, whose column is decided by enforcement at layout time)
        are found correctly.
        """
        my_x = None
        for j in range(layer.n or 0):
            pos = self.win._lay.positions.get(f'{layer.name}_{j}')
            if pos is not None:
                my_x = pos[0]
                break
        if my_x is None:
            return []
        return [
            l for l in self.win.gui.circuit.layers
            if l is not layer and (l.n or 0) > 0
            and any(
                abs((self.win._lay.positions.get(f'{l.name}_{k}') or (float('inf'),))[0] - my_x) < 1e-9
                for k in range(l.n or 0)
            )
        ]

    def get_container_snap_y(self, layer, mouse_y):
        """Snap mouse_y to the nearest insertion boundary within layer's column.
        Returns (snap_y, insert_idx) where insert_idx=0 means top slot."""
        others = [
            l for l in self.container_mates(layer)
            if (l.n or 0) > 0
        ]

        def top_y(l):
            # Bilateral layers are symmetric around 0.5, so avg_y is always 0.5.
            # Use the top (max y) neuron as the layer's representative position.
            ys = [self.win._lay.positions[f'{l.name}_{j}'][1]
                  for j in range(l.n or 0)
                  if f'{l.name}_{j}' in self.win._lay.positions]
            return max(ys) if ys else 0.5

        centers = sorted([top_y(l) for l in others], reverse=True)  # top-first

        if not centers:
            return (mouse_y, 0)

        spacing = (centers[0] - centers[-1]) / len(centers) if len(centers) > 1 else 0.2
        half = spacing / 2

        boundaries = [centers[0] + half] + \
                     [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)] + \
                     [centers[-1] - half]

        insert_idx = min(range(len(boundaries)), key=lambda i: abs(boundaries[i] - mouse_y))
        return (boundaries[insert_idx], insert_idx)

    def insert_midpoint_container(self, snap_x, exclude=None):
        """Return integer depth for a new slot between the two visible columns bracketing snap_x.
        If the natural insert depth is hidden or consecutive with the right bound, shifts all
        explicit layer/sensor depths (and hidden_containers) at/above that point by +1 to free the slot."""
        # Use only visible columns for bracketing so hidden columns are transparent to the user.
        vis_pairs = sorted(
            ((d, x) for d, x in self.win._lay.container_x_map.items() if d not in self.win._hidden_containers),
            key=lambda dx: dx[1],
        )
        if not vis_pairs:
            return 1
        col_xs         = [x for _, x in vis_pairs]
        existing_depths = [d for d, _ in vis_pairs]

        # New column to the left of the leftmost visible column.
        if snap_x < col_xs[0]:
            new_d = existing_depths[0] - 1
            if new_d <= 0:
                for obj in list(self.win.gui.circuit.layers) + list(self.win.gui.circuit.sensors):
                    if obj.name == exclude:
                        continue
                    if getattr(obj, 'layer', None) is not None:
                        obj.layer += 1
                self.win.renumber_columns(lambda d: d + 1)
                return existing_depths[0]
            return new_d
        # New column to the right of the rightmost visible column.
        if snap_x > col_xs[-1]:
            return existing_depths[-1] + 1

        for i, (a, b) in enumerate(zip(col_xs, col_xs[1:])):
            if a <= snap_x <= b:
                left_d  = existing_depths[i]
                right_d = existing_depths[i + 1]
                mid_d   = int(left_d) + 1
                if mid_d not in self.win._hidden_containers and right_d - left_d > 1:
                    # A free, visible slot already exists — use it directly.
                    return mid_d
                # Either the slot is occupied by a hidden column, or left/right are
                # consecutive.  Shift everything from mid_d upward to free the slot.
                for obj in list(self.win.gui.circuit.layers) + list(self.win.gui.circuit.sensors):
                    if obj.name == exclude:
                        continue
                    d = getattr(obj, 'layer', None)
                    if d is not None and d >= mid_d:
                        obj.layer = d + 1
                self.win.renumber_columns(lambda d: d + 1 if d >= mid_d else d)
                return mid_d
        return None

    def compact_containers(self):
        """Close gaps in explicit depths while preserving the minimum depth value.

        Also accounts for connectivity-computed depths (layers without an explicit
        'layer' attribute) so compaction never maps an explicit depth onto a slot
        already occupied by a computed layer.
        """
        expl_objs = (
            [l for l in self.win.gui.circuit.layers  if getattr(l, 'layer', None) is not None] +
            [s for s in self.win.gui.circuit.sensors if getattr(s, 'layer', None) is not None]
        )
        if not expl_objs:
            return
        # Include connectivity-computed depths so we don't close gaps they occupy.
        computed = self.compute_container()
        implicit_depths = {
            computed[l.name]
            for l in self.win.gui.circuit.layers
            if getattr(l, 'layer', None) is None and l.name in computed
        }
        all_unique = sorted({o.layer for o in expl_objs} | implicit_depths)
        min_d = all_unique[0]
        depth_map = {d: min_d + i for i, d in enumerate(all_unique)}
        for o in expl_objs:
            o.layer = depth_map[o.layer]
        # Keep hidden/disabled container sets consistent with remapped positions.
        # Drop entries whose position no longer exists (container became empty after compaction).
        # _container_labels is keyed by container identity (object name set), not
        # position, so it needs no remapping here — it's unaffected by renumbering.
        self.win.renumber_columns(depth_map.get)

    def pin_implicit_containers(self):
        """Freeze connectivity-inferred column positions before a deletion mutates the graph.

        Layers that have never been manually dragged (lyr.layer is None) get their
        depth assigned explicitly so that removing a connection or sensor doesn't
        cause _compute_container to fall back to column 1 for nodes that have lost their
        upstream dependency.  Called after _push_undo so that undo restores the
        un-pinned state correctly.
        """
        computed = self.compute_container()
        for lyr in self.win.gui.circuit.layers:
            if getattr(lyr, 'layer', None) is None and lyr.name in computed:
                lyr.layer = computed[lyr.name]
