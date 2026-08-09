"""
network_viz_layout.py — pure data/layout model for the network visualizer.

_LayoutMixin holds every method that computes column/depth/position layout or
does pure hit-test geometry against that layout — zero Qt/pyqtgraph state
mutation anywhere in this file. network_viz_render.py reads this mixin's
outputs (self._positions, self._active_names, self._container_x_map, ...) to draw;
network_viz_editing.py calls the depth-mutating helpers here (_compact_containers,
_pin_implicit_containers, _insert_midpoint_container) in response to user actions, but
those helpers only ever touch plain circuit-model fields (obj.layer/.viz_row),
never Qt/pyqtgraph items — that's what keeps them "layout" and not "editing".
"""

import numpy as np
from collections import defaultdict


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
        self.lateral_pair = f'{sensor.name}_{"R" if side == "L" else "L"}'
        if hasattr(sensor, 'width'):
            # Camera sensor: pixel-split half
            half = sensor.width // 2
            ovl  = getattr(sensor, 'overlap', 0)
            if side == 'L':
                self._px_start = 0
                self._px_end   = int(np.clip(half + ovl, 0, sensor.width))
            else:
                self._px_start = int(np.clip(half - ovl, 0, sensor.width))
                self._px_end   = sensor.width
            self.n       = 1
            self.n_total = 1
        else:
            # Non-camera lateralized sensor (joint-pair): each half has the sensor's n outputs
            self.n       = sensor.n or 1
            self.n_total = sensor.n or 1


def _sensor_is_lateralized(sensor, circuit=None):
    return sensor.is_lateralized(circuit)


def _mirror_name(name):
    """Swap _L ↔ _R suffix. Returns None if name has neither suffix."""
    if name.endswith('_L'):
        return name[:-2] + '_R'
    if name.endswith('_R'):
        return name[:-2] + '_L'
    return None


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


class _LayoutMixin:
    _EDGE_CLICK_DIST = 0.04  # view-space distance threshold for edge selection

    @staticmethod
    def _reversed_idx(i: int, n: int, side: str) -> int:
        """Map neuron/filter index i to visual position.
        R-side reverses ordering so R[0] is at visual bottom, R[n-1] near midline."""
        return n - 1 - i if side == 'R' else i

    def _connection_kind(self, src_obj, tgt_obj, W, ns, nt):
        """Classify a connection for drawing dispatch (see rules/network_viz.md).

        Priority: TD → CONV4D → DENSE → THIN → THICK.
        src_obj / tgt_obj are the resolved layer or sensor objects (tgt is always a layer).
        """
        from neurons import LearningLayerBase as _LLB
        if isinstance(tgt_obj, _LLB):
            return self._CK_TD
        if W.ndim == 4:
            return self._CK_CONV4D
        if max(ns, nt) > self._DENSE_THRESHOLD:
            return self._CK_DENSE
        if (ns * nt > 4
                or (getattr(src_obj, 'is_image_node', False)
                    and getattr(tgt_obj, 'is_image_node', False))):
            return self._CK_THIN
        return self._CK_THICK

    def _n_map(self):
        from sensors import CameraSensor as _CamSensor
        circuit = self.gui.circuit
        m = {}
        for s in circuit.sensors:
            if _sensor_is_lateralized(s, circuit):
                if isinstance(s, _CamSensor):
                    half    = s.width // 2
                    ovl     = getattr(s, 'overlap', 0)
                    l_end   = int(np.clip(half + ovl, 0, s.width))
                    r_start = int(np.clip(half - ovl, 0, s.width))
                    m[f'{s.name}_L'] = l_end * getattr(s, 'in_ch', 3)
                    m[f'{s.name}_R'] = (s.width - r_start) * getattr(s, 'in_ch', 3)
                else:
                    # Joint-pair sensor: each half has sensor.n outputs
                    m[f'{s.name}_L'] = s.n or 1
                    m[f'{s.name}_R'] = s.n or 1
            else:
                m[s.name] = s.n
        m.update({l.name: (getattr(l, 'viz_n', None) or l.n)
                  for l in circuit.layers if l.n is not None})
        return m

    def _compute_container(self):
        c     = self.gui.circuit
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

    @staticmethod
    def _slot_weight(entry) -> float:
        """Proportional vertical weight for one slot-list entry (unused by _layout,
        kept for reference). Image nodes get _CAM_WEIGHT; others get neuron count."""
        _CAM_W = NetworkVisualizerWindow._CAM_WEIGHT
        if entry[0] == 'pair':
            _, L_ct, L_obj, R_ct, R_obj = entry
            if (getattr(L_obj, 'is_image_node', False) or getattr(L_obj, 'viz_n', None) == 1 or
                    getattr(R_obj, 'is_image_node', False) or getattr(R_obj, 'viz_n', None) == 1):
                return _CAM_W
            return max(_eff_n(L_obj), _eff_n(R_obj), 1)
        else:
            _, ct, obj = entry
            if getattr(obj, 'is_image_node', False) or getattr(obj, 'viz_n', None) == 1:
                return _CAM_W
            return max(_eff_n(obj), 1)

    def _layout(self):
        circuit  = self.gui.circuit
        layers   = [l for l in circuit.layers if l.n is not None]
        actual_sensors = list(circuit.sensors)
        # Unfiltered snapshots, taken before the z-cut block below may reassign
        # `layers`/`actual_sensors` to a subsumed-down subset — used to compute
        # a stable view-fit extent that doesn't rescale as the z-cut slider moves.
        full_layers, full_sensors = layers, actual_sensors

        # Z-depth filter: for each column keep only the layer with max z ≤ slider value.
        z_cut = self._z_cut
        if z_cut is not None:
            _zcol_container = self._compute_container()
            # Enforce lateral pairs into the same column for the z-cut grouping.
            for _l in layers:
                _pn = getattr(_l, 'lateral_pair', None)
                if _pn:
                    _pp = next((pl for pl in layers if pl.name == _pn), None)
                    if _pp:
                        _d = max(_zcol_container.get(_l.name, 1),
                                 _zcol_container.get(_pp.name, 1), 1)
                        _zcol_container[_l.name] = _d
                        _zcol_container[_pp.name] = _d
            by_container = defaultdict(list)
            for obj in actual_sensors + layers:
                col = _zcol_container.get(obj.name, 0)
                by_container[col].append(obj)
            # native_col_n[container] = tallest neuron count among objects that
            # natively live there — used to estimate a ghost's height when it
            # isn't currently active (so its real position isn't available).
            self._native_col_n = {
                _col: max((getattr(_o, 'viz_n', None) or getattr(_o, 'n', 1) or 1)
                          for _o in _objs)
                for _col, _objs in by_container.items()
            }

            # Pass 1: per-column winners (highest z ≤ z_cut).
            # subsumed_by is also populated here for same-column losers (lower-z
            # elements displaced by the winner), so connection drawing can remap
            # e.g. S→A1 to S→B when B wins column 1 over A1.
            active_names = set()
            subsumed_by  = {}   # element_name → subsumer_name
            container_winners  = {}   # container → (winners_list, winner_z_val)
            for container, container_objs in by_container.items():
                eligible = [o for o in container_objs if (getattr(o, 'z', 0) or 0) <= z_cut]
                if eligible:
                    # Keep ALL elements at the peak z level — max() would drop
                    # siblings sharing the same column and z (e.g. two sensors
                    # both at z=0 depth=0), which makes their connections vanish.
                    _max_z_val = max((getattr(o, 'z', 0) or 0) for o in eligible)
                    winners = [o for o in eligible
                               if (getattr(o, 'z', 0) or 0) == _max_z_val]
                    losers  = [o for o in eligible
                               if (getattr(o, 'z', 0) or 0) < _max_z_val]
                    for _o in winners:
                        active_names.add(_o.name)
                        _pair = getattr(_o, 'lateral_pair', None)
                        if _pair:
                            active_names.add(_pair)
                        # Lateralized sensors are split into _L/_R node names for
                        # display/connections — the sensor's own name never appears
                        # as a Connection endpoint, only its halves do, so both must
                        # be marked active or every connection touching them is
                        # silently dropped below.
                        if _o in actual_sensors and _sensor_is_lateralized(_o, circuit):
                            active_names.add(f'{_o.name}_L')
                            active_names.add(f'{_o.name}_R')
                    # Map each same-column loser to the first winner so connections
                    # through that loser are drawn via the subsumer instead.
                    if losers and winners:
                        _rep = winners[0].name
                        for _loser in losers:
                            subsumed_by[_loser.name] = _rep
                            _lp = getattr(_loser, 'lateral_pair', None)
                            if _lp:
                                subsumed_by[_lp] = _rep
                            if _loser in actual_sensors and _sensor_is_lateralized(_loser, circuit):
                                subsumed_by[f'{_loser.name}_L'] = _rep
                                subsumed_by[f'{_loser.name}_R'] = _rep
                    container_winners[container] = (winners, _max_z_val)

            # Pass 2: span-based subsumption — a spanning winner at z>0 also
            # removes lower-z elements in the columns its span covers. E.g. a
            # two-column pipeline A (col 0) -> B (col 1) at z=0 running in
            # parallel with a wider container C (col 0, span=2, z=1): C wins
            # both columns, and A/B are recorded (below, in _ghosts_by_container)
            # so C's panel can show each of them as its own ghost.
            for container, (winners, winner_z) in container_winners.items():
                for winner in winners:
                    span = max(1, getattr(winner, 'span', 1) or 1)
                    if span <= 1:
                        continue
                    for j in range(1, span):
                        nbr_container = container + j
                        if nbr_container not in by_container:
                            continue
                        for nbr_obj in by_container[nbr_container]:
                            if (getattr(nbr_obj, 'z', 0) or 0) < winner_z:
                                active_names.discard(nbr_obj.name)
                                subsumed_by[nbr_obj.name] = winner.name
                                _nbr_pair = getattr(nbr_obj, 'lateral_pair', None)
                                if _nbr_pair:
                                    active_names.discard(_nbr_pair)
                                    subsumed_by[_nbr_pair] = winner.name
                                if nbr_obj in actual_sensors and _sensor_is_lateralized(nbr_obj, circuit):
                                    active_names.discard(f'{nbr_obj.name}_L')
                                    active_names.discard(f'{nbr_obj.name}_R')
                                    subsumed_by[f'{nbr_obj.name}_L'] = winner.name
                                    subsumed_by[f'{nbr_obj.name}_R'] = winner.name

            layers         = [l for l in layers         if l.name in active_names]
            actual_sensors = [s for s in actual_sensors if s.name in active_names]
            self._active_names = active_names
            self._subsumed_by  = subsumed_by
            # ghost_count[container] = (distinct z-levels in column container) - 1.
            # Computed from all elements regardless of z_cut direction, so the
            # count is the same whether you're looking up or down the z stack.
            # Used only for _container_key / auto-titling a same-column-shared
            # panel from its current winner — the ghost *rendering* itself is
            # driven by _ghosts_by_container below.
            _all_container_z = defaultdict(set)
            for _obj in list(circuit.sensors) + [_l for _l in circuit.layers
                                                  if _l.n is not None]:
                _dv = _zcol_container.get(_obj.name)
                if _dv is not None:
                    _all_container_z[_dv].add(getattr(_obj, 'z', 0) or 0)
            self._ghost_count = {_dv: len(_zs) - 1
                                  for _dv, _zs in _all_container_z.items()
                                  if len(_zs) > 1}
            # ghosts_by_container[container] = every object hidden "behind" the
            # panel currently active at `container`, each carrying its OWN
            # native column + span (not the winner's) so it renders as its own
            # correctly sized/positioned rect + label rather than a merged
            # shadow. Two ways an object ends up behind a winner:
            #   (a) same native column, different z (classic same-column stack)
            #   (b) the winner's own span reaches into this object's native
            #       column (e.g. a two-column A->B pipeline sitting entirely
            #       behind a wider, higher-z container spanning both columns)
            # Symmetric by construction: whichever of two overlapping objects
            # currently loses is a ghost of whichever currently wins, and a
            # spanning object that ISN'T currently winning anything still
            # shows up (case a, at its own native column) as a ghost sized to
            # its own full span — e.g. a dormant wide container reappears as
            # one wide ghost band behind the several narrower winners that
            # currently occupy the columns it would otherwise cover.
            _ghosts_by_container = defaultdict(list)
            for _container, _container_objs in by_container.items():
                _winners, _ = container_winners.get(_container, ([], None))
                if not _winners:
                    continue   # nothing active here to anchor a ghost to
                _winner_names = {_o.name for _o in _winners}
                _seen = set()
                for _o in _container_objs:
                    if _o.name in _winner_names or _o.name in _seen:
                        continue
                    if _o.name.endswith(('_L', '_R')):
                        continue   # skip lateralized halves — the base name covers both
                    _seen.add(_o.name)
                    _ghosts_by_container[_container].append({
                        'name': _o.name,
                        'native_container': _container,
                        'span': max(1, getattr(_o, 'span', 1) or 1),
                        'z': getattr(_o, 'z', 0) or 0,
                    })
            for container, (winners, winner_z) in container_winners.items():
                for winner in winners:
                    span = max(1, getattr(winner, 'span', 1) or 1)
                    if span <= 1:
                        continue
                    for j in range(1, span):
                        nbr_container = container + j
                        if nbr_container not in by_container:
                            continue
                        for nbr_obj in by_container[nbr_container]:
                            if (getattr(nbr_obj, 'z', 0) or 0) < winner_z \
                                    and not nbr_obj.name.endswith(('_L', '_R')):
                                _ghosts_by_container[container].append({
                                    'name': nbr_obj.name,
                                    'native_container': nbr_container,
                                    'span': max(1, getattr(nbr_obj, 'span', 1) or 1),
                                    'z': getattr(nbr_obj, 'z', 0) or 0,
                                })
            self._ghosts_by_container = dict(_ghosts_by_container)
            # winner_names[container] = name(s) of whoever currently wins that
            # container — for containers shared across z-levels, the panel
            # title auto-follows this instead of a manual label, so it never
            # goes stale as the z-cut slider changes who's in front.
            self._container_winner_names = {
                _container: [_o.name for _o in _winners]
                for _container, (_winners, _) in container_winners.items() if _winners
            }
        else:
            self._active_names = None
            self._subsumed_by  = {}
            self._ghost_count  = {}
            self._ghosts_by_container = {}
            self._native_col_n = {}
            self._container_winner_names = {}

        brain_cls   = self.gui.brain.__class__ if self.gui.brain else None
        viz_columns = getattr(brain_cls, 'viz_columns', None)

        positions, sensor_nodes, groups, self._container_span_map, self._x_unit, self._container_ids = \
            self._compute_positions(actual_sensors, layers, viz_columns)

        # The view's fit range (see build()) is sourced from _full_positions rather than
        # `positions` so that scrubbing the z-cut slider — which only changes which subset
        # of `full_layers`/`full_sensors` survives filtering above, never the underlying
        # circuit — doesn't rescale the whole picture. Skipped when z-cut is inactive
        # (self._active_names is None): filtered == full already, no need to redo the work.
        if self._active_names is not None:
            full_positions, *_ = self._compute_positions(
                full_sensors, full_layers, viz_columns, record_side_effects=False)
            self._full_positions = full_positions
        else:
            self._full_positions = positions

        return positions, sensor_nodes, groups

    def _compute_positions(self, actual_sensors, layers, viz_columns, record_side_effects=True):
        """Column/slot layout for `actual_sensors`/`layers` (already z-cut-filtered by the
        caller, or not, for the caller's own purposes — this method doesn't care).

        record_side_effects gates the two persistent-object mutations this method makes
        (`obj.viz_row`, ring layout's `obj._ring_cx/_ring_cy/_ring_r`) — set False for a
        "shadow" call whose only purpose is measuring an extent, so it doesn't clobber
        state a real/filtered call already set (or will set) on the same objects.
        """
        circuit = self.gui.circuit

        # Replace lateralized sensors with L/R SplitHalf stand-ins for layout purposes
        sensors = []
        for s in actual_sensors:
            if _sensor_is_lateralized(s, circuit):
                sensors.append(_SplitHalf(s, 'L'))
                sensors.append(_SplitHalf(s, 'R'))
            else:
                sensors.append(s)

        name_map = {s.name: ('sensor', s) for s in sensors}
        name_map.update({l.name: ('layer', l) for l in layers})

        if viz_columns:
            container_groups = [
                [name_map[n] for n in grp if n in name_map]
                for grp in viz_columns
            ]
            container_groups = [col for col in container_groups if col]
            container_ids = list(range(len(container_groups)))
        else:
            depth = self._compute_container()
            # Enforce lateralized pairs into the same column (≥1, never the sensor column 0)
            for l in layers:
                pair_name = getattr(l, 'lateral_pair', None)
                if pair_name:
                    partner = next((pl for pl in layers if pl.name == pair_name), None)
                    if partner:
                        d = max(depth.get(l.name, 1), depth.get(partner.name, 1), 1)
                        depth[l.name] = d
                        depth[partner.name] = d
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
            container_ids    = sorted_depths

        positions    = {}
        sensor_nodes = set()
        groups       = []

        max_container_id = max(container_ids) if container_ids else 1
        panel_hw = self._NODE_R + self._PAD_X
        container_gap  = 0.06
        if len(container_ids) > 1:
            min_ds  = min(b - a for a, b in zip(container_ids, container_ids[1:]))
            x_unit  = max((2 * panel_hw + container_gap) / min_ds,
                          1.0 / max(max_container_id, 1))
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

            ring_items    = [(ct, obj) for ct, obj in col
                             if getattr(obj, 'viz_layout', None) == 'ring']
            regular_items = [(ct, obj) for ct, obj in col
                             if getattr(obj, 'viz_layout', None) != 'ring']

            from neurons import Conv2dLayer as _Conv2dLayer

            # ── Build slot list: pair up lateralized items, keep singles ─────
            slot_list  = []    # [('pair', L_ct, L, R_ct, R) | ('single', ct, obj)]
            seen_pairs = set()
            for ct, obj in regular_items:
                pair_name = getattr(obj, 'lateral_pair', None)
                if pair_name is not None:
                    key = tuple(sorted([obj.name, pair_name]))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    partner_item = next(
                        ((pct, pobj) for pct, pobj in regular_items
                         if pobj.name == pair_name), None)
                    if partner_item is None:
                        slot_list.append(('single', ct, obj))   # orphaned half
                    else:
                        pct, pobj = partner_item
                        if obj.name.endswith('_L'):
                            slot_list.append(('pair', ct, obj, pct, pobj))
                        elif obj.name.endswith('_R'):
                            slot_list.append(('pair', pct, pobj, ct, obj))
                        else:
                            slot_list.append(('pair', ct, obj, pct, pobj))
                else:
                    slot_list.append(('single', ct, obj))

            # ── Outside-in index assignment ────────────────────────────────────
            # The first entry claims the outermost positions (top + bottom),
            # the second entry claims the next-innermost, and so on until all
            # entries converge to the centre.  Pairs: L→top, R→bottom.
            # Odd-n entry: ceil(n/2) top, floor(n/2) bottom.
            total_N = 0
            for entry in slot_list:
                if entry[0] == 'pair':
                    _, L_ct, L_obj, R_ct, R_obj = entry
                    total_N += max(_eff_n(L_obj), 1) + max(_eff_n(R_obj), 1)
                else:
                    total_N += max(_eff_n(entry[2]), 1)
            total_N = total_N or 1
            d       = 2.0 * self._NODE_R
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
                    L_color = L_obj.viz_color
                    R_color = R_obj.viz_color
                    groups.append((L_ct, L_obj.name, L_nodes, x, container,
                                   slot_top, slot_bot, L_color))
                    groups.append((R_ct, R_obj.name, R_nodes, x, container,
                                   slot_top, slot_bot, R_color))
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
                    obj_color = obj.viz_color
                    groups.append((ct, obj.name, container_nodes, x, container,
                                   slot_top, slot_bot, obj_color))
                    if record_side_effects and ct == 'layer':
                        obj.viz_row = slot_idx

            # Ring layout: neurons arranged in a circle, neuron 0 at the top.
            for _, obj in ring_items:
                n_ring  = obj.n
                ring_r  = max(0.15, min(0.38,
                              self._RING_SCALE * self._NODE_R / (2 * np.sin(np.pi / n_ring) + 1e-9)))
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
                obj_color = getattr(obj, 'color', None)
                groups.append(('layer', obj.name, container_nodes, x, container,
                               cy + ring_r + self._NODE_R,
                               cy - ring_r - self._NODE_R, obj_color))

        return positions, sensor_nodes, groups, container_span_map, x_unit, container_ids

    @staticmethod
    def _ctrl_pt(p0, p1, bow):
        mx, my = (p0[0]+p1[0])/2, (p0[1]+p1[1])/2
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
        chord = (dx*dx + dy*dy) ** 0.5
        if chord > 1e-9 and bow != 0:
            # bow*chord gives perpendicular displacement; cap it so long-span arcs stay on-canvas
            scale = min(1.0, 0.35 / (abs(bow) * chord))
            return (mx + bow * dy * scale, my - bow * dx * scale)
        return (mx + bow * dy, my - bow * dx)

    @staticmethod
    def _bezier_pts(p0, ctrl, p1, n=80):
        ts = np.linspace(0, 1, n)
        xs = (1-ts)**2 * p0[0] + 2*(1-ts)*ts * ctrl[0] + ts**2 * p1[0]
        ys = (1-ts)**2 * p0[1] + 2*(1-ts)*ts * ctrl[1] + ts**2 * p1[1]
        return list(zip(xs.tolist(), ys.tolist()))

    @staticmethod
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

    @staticmethod
    def _hemisphere(idx, n):
        """True = left/top half, False = right/bottom half, None = single neuron (midline)."""
        if n <= 1:
            return None
        return idx < n // 2

    @staticmethod
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

    def _infer_and_set_n(self):
        for conn in self.gui.circuit.connections:
            src, tgt, W = conn.src, conn.tgt, conn.W
            W = np.asarray(W, dtype=float)
            if W.ndim != 2:
                continue
            for layer in self.gui.circuit.layers:
                if layer.n is not None:
                    continue
                if layer.name == src:
                    layer._ensure_n(W.shape[1])
                elif layer.name == tgt:
                    layer._ensure_n(W.shape[0])

    def _panel_at(self, view_pt):
        """Return container of the column panel the view-space point falls inside, or None."""
        r, px = self._NODE_R, self._PAD_X
        py = 0.04
        x, y = view_pt.x(), view_pt.y()
        container_data = {}
        for node_key, container in self._node_container_map.items():
            if node_key not in self._positions:
                continue
            nx, ny = self._positions[node_key]
            if container not in container_data:
                container_data[container] = {'xs': [], 'ys': []}
            container_data[container]['xs'].append(nx)
            container_data[container]['ys'].append(ny)
        for container, data in container_data.items():
            x_col = sum(data['xs']) / len(data['xs'])
            y_min = min(data['ys']) - r - py
            y_max = max(data['ys']) + r + py
            x_min = x_col - r - px
            x_max = x_col + r + px
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return container
        return None

    def _container_key(self, container):
        """Stable identity string for whichever occupant currently fronts this
        container slot — used to key manual labels so they survive drags and
        renumbering instead of a raw position. Usually a single name; more
        than one only when the container is shared across z-levels, in which
        case it's whoever currently wins that slot."""
        names = None
        if getattr(self, '_ghost_count', {}).get(container, 0) > 0:
            names = getattr(self, '_container_winner_names', {}).get(container)
        if not names:
            names = sorted({nk.rsplit('_', 1)[0]
                             for nk, c in self._node_container_map.items() if c == container})
        return '|'.join(sorted(names))

    def _edge_at(self, view_pt):
        """Return (src_name, tgt_name) of the edge nearest to view_pt, or None."""
        px, py = view_pt.x(), view_pt.y()
        best = self._EDGE_CLICK_DIST
        result = None
        for item, sn, tn, _, is_curve, *_ in self._edge_items_tagged:
            if not is_curve:
                continue
            try:
                xs, ys = item.getData()
            except Exception:
                continue
            if xs is None or len(xs) < 2:
                continue
            for i in range(len(xs) - 1):
                d = self._dist_to_segment(px, py, xs[i], ys[i], xs[i+1], ys[i+1])
                if d < best:
                    best = d
                    result = (sn.rsplit('_', 1)[0], tn.rsplit('_', 1)[0])
        return result

    @staticmethod
    def _dist_to_segment(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        denom = dx*dx + dy*dy
        if denom == 0:
            return np.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1)*dx + (py - y1)*dy) / denom))
        return np.hypot(px - (x1 + t*dx), py - (y1 + t*dy))

    def _get_snap_x(self, mouse_x):
        # Exclude hidden columns so drops never silently land in an invisible column.
        vis_xs = sorted(cx for d, cx in self._container_x_map.items()
                        if d not in self._hidden_containers)
        col_xs = vis_xs if vis_xs else sorted(self._container_x_map.values())
        if not col_xs:
            return mouse_x
        nearest = min(col_xs, key=lambda cx: abs(cx - mouse_x))
        if abs(nearest - mouse_x) < self._x_unit * 0.4:
            return nearest
        for a, b in zip(col_xs, col_xs[1:]):
            if a < mouse_x < b:
                return (a + b) / 2
        # Beyond the edge columns — allow creating a new column outside the existing range.
        if mouse_x < col_xs[0]:
            return col_xs[0] - self._x_unit
        if mouse_x > col_xs[-1]:
            return col_xs[-1] + self._x_unit
        return nearest

    def _snap_is_existing(self, snap_x):
        return any(abs(snap_x - cx) < 1e-9
                   for d, cx in self._container_x_map.items()
                   if d not in self._hidden_containers)

    def _container_mates(self, layer):
        """Return all layers that share the same visual column as *layer*.

        Uses the current ``_positions`` dict rather than the raw ``.layer``
        attribute so that layers with ``layer=None`` (e.g. the _L half of a
        lateralized pair, whose column is decided by enforcement at layout time)
        are found correctly.
        """
        my_x = None
        for j in range(layer.n or 0):
            pos = self._positions.get(f'{layer.name}_{j}')
            if pos is not None:
                my_x = pos[0]
                break
        if my_x is None:
            return []
        return [
            l for l in self.gui.circuit.layers
            if l is not layer and (l.n or 0) > 0
            and any(
                abs((self._positions.get(f'{l.name}_{k}') or (float('inf'),))[0] - my_x) < 1e-9
                for k in range(l.n or 0)
            )
        ]

    def _get_container_snap_y(self, layer, mouse_y):
        """Snap mouse_y to the nearest insertion boundary within layer's column.
        Returns (snap_y, insert_idx) where insert_idx=0 means top slot."""
        others = [
            l for l in self._container_mates(layer)
            if (l.n or 0) > 0
        ]

        def top_y(l):
            # Bilateral layers are symmetric around 0.5, so avg_y is always 0.5.
            # Use the top (max y) neuron as the layer's representative position.
            ys = [self._positions[f'{l.name}_{j}'][1]
                  for j in range(l.n or 0)
                  if f'{l.name}_{j}' in self._positions]
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

    def _insert_midpoint_container(self, snap_x, exclude=None):
        """Return integer depth for a new slot between the two visible columns bracketing snap_x.
        If the natural insert depth is hidden or consecutive with the right bound, shifts all
        explicit layer/sensor depths (and hidden_containers) at/above that point by +1 to free the slot."""
        # Use only visible columns for bracketing so hidden columns are transparent to the user.
        vis_pairs = sorted(
            ((d, x) for d, x in self._container_x_map.items() if d not in self._hidden_containers),
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
                for obj in list(self.gui.circuit.layers) + list(self.gui.circuit.sensors):
                    if obj.name == exclude:
                        continue
                    if getattr(obj, 'layer', None) is not None:
                        obj.layer += 1
                self._hidden_containers   = {d + 1 for d in self._hidden_containers}
                self._disabled_containers = {d + 1 for d in self._disabled_containers}
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
                if mid_d not in self._hidden_containers and right_d - left_d > 1:
                    # A free, visible slot already exists — use it directly.
                    return mid_d
                # Either the slot is occupied by a hidden column, or left/right are
                # consecutive.  Shift everything from mid_d upward to free the slot.
                for obj in list(self.gui.circuit.layers) + list(self.gui.circuit.sensors):
                    if obj.name == exclude:
                        continue
                    d = getattr(obj, 'layer', None)
                    if d is not None and d >= mid_d:
                        obj.layer = d + 1
                self._hidden_containers   = {d + 1 if d >= mid_d else d for d in self._hidden_containers}
                self._disabled_containers = {d + 1 if d >= mid_d else d for d in self._disabled_containers}
                return mid_d
        return None

    def _compact_containers(self):
        """Close gaps in explicit depths while preserving the minimum depth value.

        Also accounts for connectivity-computed depths (layers without an explicit
        'layer' attribute) so compaction never maps an explicit depth onto a slot
        already occupied by a computed layer.
        """
        expl_objs = (
            [l for l in self.gui.circuit.layers  if getattr(l, 'layer', None) is not None] +
            [s for s in self.gui.circuit.sensors if getattr(s, 'layer', None) is not None]
        )
        if not expl_objs:
            return
        # Include connectivity-computed depths so we don't close gaps they occupy.
        computed = self._compute_container()
        implicit_depths = {
            computed[l.name]
            for l in self.gui.circuit.layers
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
        self._hidden_containers   = {depth_map[d] for d in self._hidden_containers   if d in depth_map}
        self._disabled_containers = {depth_map[d] for d in self._disabled_containers if d in depth_map}

    def _pin_implicit_containers(self):
        """Freeze connectivity-inferred column positions before a deletion mutates the graph.

        Layers that have never been manually dragged (lyr.layer is None) get their
        depth assigned explicitly so that removing a connection or sensor doesn't
        cause _compute_container to fall back to column 1 for nodes that have lost their
        upstream dependency.  Called after _push_undo so that undo restores the
        un-pinned state correctly.
        """
        computed = self._compute_container()
        for lyr in self.gui.circuit.layers:
            if getattr(lyr, 'layer', None) is None and lyr.name in computed:
                lyr.layer = computed[lyr.name]

    def _node_at(self, pt):
        """Return the node key nearest to view-space point *pt*, or None."""
        r2 = (self._NODE_R * 1.3) ** 2
        best, best_d2 = None, float('inf')
        for name, (x, y) in self._positions.items():
            d2 = (pt.x() - x) ** 2 + (pt.y() - y) ** 2
            if d2 <= r2 and d2 < best_d2:
                best, best_d2 = name, d2
        return best

    # ── Notes — free-positioned, no column/container role ──────────────────────
    # (x, y) is the box's top-left corner; the box extends right by _NOTE_W and
    # down by the wrapped-text height. The collapsed icon is centred on the
    # same (x, y) anchor so collapsing/expanding never appears to move the note.

    _NOTE_W          = 0.34
    _NOTE_PAD        = 0.02
    _NOTE_ICON_PX    = 12    # collapsed-icon visual radius, in screen pixels
    _NOTE_TOGGLE_PX  = 10    # collapse-toggle glyph click-zone radius, in screen pixels
    _NOTE_FONT_FAMILY = 'Segoe UI'
    _NOTE_FONT_SIZE   = 7

    def _note_font_metrics(self):
        from PySide6.QtGui import QFont, QFontMetrics
        return QFontMetrics(QFont(self._NOTE_FONT_FAMILY, self._NOTE_FONT_SIZE))

    def _note_lines(self, note):
        """Word-wrap a note's text to fit _NOTE_W, measured in *actual* pixel
        font metrics (not a guessed character count) so the background box
        computed by _note_size always matches what's rendered — a fixed
        chars-per-line heuristic drifts badly across the wide range of zoom
        levels different-sized networks end up at.
        """
        fm = self._note_font_metrics()
        dx, _dy = self._vb.viewPixelSize()
        max_px = (self._NOTE_W - 2 * self._NOTE_PAD) / dx if dx > 0 else 200.0
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

    def _note_size(self, note):
        """Return (w, h) of an expanded note's background box, in data coords.

        Measures a throwaway TextItem's real boundingRect() rather than
        estimating from QFontMetrics — pyqtgraph's own multi-line text layout
        uses more vertical space per line than QFontMetrics.height()/
        lineSpacing() predict, so an estimate consistently undersizes the box
        and the text spills out the bottom.
        """
        import pyqtgraph as pg
        from PySide6.QtGui import QFont
        txt = pg.TextItem('\n'.join(self._note_lines(note)))
        txt.setFont(QFont(self._NOTE_FONT_FAMILY, self._NOTE_FONT_SIZE))
        _dx, dy = self._vb.viewPixelSize()
        h = 2 * self._NOTE_PAD + txt.boundingRect().height() * dy
        return self._NOTE_W, h

    def _note_at(self, pt):
        """Return (note, zone) for the topmost note hit at view-space point
        *pt* — zone is 'icon' (collapsed note; drag/select/double-click target,
        same as 'body'), 'toggle' (collapse glyph on an expanded note — the
        only zone with its own click behavior), or 'body' (expanded note).
        None if no hit. Checked before _node_at since notes draw on top of
        everything else.
        """
        if not getattr(self, '_notes_visible', True):
            return None
        dx, dy = self._vb.viewPixelSize()
        # Generous click targets — bigger than the tiny rendered glyphs themselves.
        icon_r2   = ((self._NOTE_ICON_PX + 5) * dx) ** 2 + ((self._NOTE_ICON_PX + 5) * dy) ** 2
        toggle_dx = (self._NOTE_TOGGLE_PX + 4) * dx
        toggle_dy = (self._NOTE_TOGGLE_PX + 4) * dy
        for note in reversed(self.gui.circuit.notes):
            if note.collapsed:
                d2 = (pt.x() - note.x) ** 2 + (pt.y() - note.y) ** 2
                if d2 <= icon_r2:
                    return note, 'icon'
                continue
            w, h = self._note_size(note)
            x0, x1 = note.x, note.x + w
            y0, y1 = note.y - h, note.y
            if not (x0 <= pt.x() <= x1 and y0 <= pt.y() <= y1):
                continue
            if pt.x() >= x1 - toggle_dx and pt.y() >= y1 - toggle_dy:
                return note, 'toggle'
            return note, 'body'
        return None
