"""
network_runner.py — standalone neural forward pass.

Extracted from BaseBrain.step_network so it can be tested independently and
imported without subclassing BaseBrain.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from neurons import _activate
from lateral import side_of, mirror_name, parent_sensor, is_camera_half, is_body_pair_half

_DEBUG_LAYER = os.environ.get('LBP_DEBUG_LAYER')   # layer name to trace input/output each tick


def _unpack_mod_row(row):
    """Unpack a `.modulators` row, tolerating every historical row length.

    Legacy rows are 3-tuples `(name, scale, site)`. Rows may now also carry a
    4th `mode` element (`'absolute'` / `'derivative'` / `'integral'`), and —
    on learning layers only — 5th/6th `drives_plasticity`/`threshold`
    elements. Missing trailing elements default to today's exact old
    behavior: absolute value, does not drive plasticity.
    """
    name, scale, site = row[0], row[1], row[2]
    mode              = row[3] if len(row) > 3 else 'absolute'
    drives_plasticity = row[4] if len(row) > 4 else False
    threshold         = row[5] if len(row) > 5 else 0.0
    return name, scale, site, mode, drives_plasticity, threshold


def _image_layer_fan_in(layer):
    """Input-buffer size for Leaky2dLayer / Reichardt2dLayer: the incoming
    flat image size (in_ch * H * W). For Leaky2dLayer, or a Reichardt2dLayer
    with pool='none', this equals layer.n. For a pooled Reichardt2dLayer
    (pool='global_avg'/'global_max'), layer.n is n_directions — much smaller
    than the image it still needs as raw input — so the two must not be
    conflated when sizing the connection-accumulation buffer."""
    if layer.frame_h and layer.frame_w:
        return layer.in_ch * layer.frame_h * layer.frame_w
    return layer.n


def _conv_forward(src_val, w_tensor, layer, src_sensor=None):
    """Apply F.conv2d + pooling for a Conv2dLayer connection.

    w_tensor  shape: (n_filters, in_ch, kH, kW)
    src_val   shape: flat (in_ch * H * W,)
    src_sensor: sensor object with _last_frame (H, W, in_ch) or (H, W) for shape hint.
    Returns   shape: (n_filters,) for global_avg/max, or (n_filters*H_out*W_out,) for none.
    """
    n_filters, in_ch, kH, kW = w_tensor.shape
    n_pixels = src_val.shape[0] // max(in_ch, 1)

    # Resolve spatial dimensions from sensor's last frame when available.
    # For lateralized halves, the parent sensor's frame gives H; W comes from
    # the actual data size (the half has fewer columns than the full frame).
    if src_sensor is not None and getattr(src_sensor, '_last_frame', None) is not None:
        H = src_sensor._last_frame.shape[0]
        W = n_pixels // max(H, 1)
        if W == 0:
            H, W = src_sensor._last_frame.shape[:2]
    elif n_pixels == int(n_pixels ** 0.5) ** 2:
        H = W = int(n_pixels ** 0.5)
    else:
        H, W = 1, n_pixels   # fallback: single-row image

    src_4d = src_val.reshape(1, in_ch, H, W)                     # (1, in_ch, H, W)
    pad_h  = kH // 2 if layer.padding == 'same' else 0
    pad_w  = kW // 2 if layer.padding == 'same' else 0
    out    = F.conv2d(src_4d, w_tensor,
                      padding=(pad_h, pad_w), stride=layer.stride) # (1, n_filters, H_out, W_out)
    out    = out.squeeze(0)                                         # (n_filters, H_out, W_out)

    # Activation on feature maps — before pooling so spatial statistics are preserved
    out = _activate(out, getattr(layer, 'activation', 'relu'), alpha=getattr(layer, 'alpha', 1.0))

    if layer.pool == 'global_avg':
        return out.mean(dim=(-2, -1))                              # (n_filters,)
    elif layer.pool == 'global_max':
        return out.flatten(1).max(dim=-1).values                   # (n_filters,)
    else:
        _, H_out, W_out = out.shape
        layer.frame_h = H_out
        layer.frame_w = W_out
        return out.reshape(-1)                                     # (n_filters * H_out * W_out,)


def _build_conn_meta(conn, layer, w_cached, sensors, layer_map):
    """Classify a connection's routing (conv4d / 1-D passthrough / lateralized-
    pair concatenation) — computed once per topology change and cached on
    brain._conn_meta[i] (see step_network). Everything here depends only on
    shapes and object identity that are stable between topology changes, never
    on a tick's data, so re-deriving it every tick (isinstance/endswith/linear
    scans over sensors) was pure waste."""
    src  = conn.src
    meta = {
        'is_conv4d': False, 'src_sensor': None, 'is_lat_cam_half': False,
        'is_passthrough1d': False,
        'pair_obj': None, 'is_lat_sensor_half': False, 'sensor_partner': None,
    }
    if w_cached.ndim == 4 and layer.kernel_weights:
        # Shape source: the sensor itself, the camera a half belongs to, or an
        # image layer (e.g. Leaky2dLayer) as shape proxy.
        src_sensor = next((s for s in sensors if s.name == src), None)
        if src_sensor is None:
            src_sensor = parent_sensor(sensors, src)
        if src_sensor is None:
            src_sensor = layer_map.get(src)
        meta['is_conv4d']       = True
        meta['src_sensor']      = src_sensor
        meta['is_lat_cam_half'] = is_camera_half(sensors, src)
    elif w_cached.ndim == 1 and layer.passthrough_input:
        meta['is_passthrough1d'] = True
    elif w_cached.ndim == 2:
        src_obj  = layer_map.get(src)
        lat_pair = getattr(src_obj, 'lateral_pair', None) if src_obj else None
        if lat_pair:
            # Conv2dLayer pair: partner output from layer_map.
            cand = layer_map.get(lat_pair)
            if cand is not None and w_cached.shape[1] == (src_obj.n or 0) + (cand.n or 0):
                meta['pair_obj'] = cand
        elif is_body_pair_half(sensors, src):
            # Joint-pair sensor half: partner output from brain attributes.
            meta['is_lat_sensor_half'] = True
            meta['sensor_partner'] = mirror_name(src)
    return meta


def _initial_input(layer, is_product, is_2d):
    """Empty fan-in: ones for a product layer, zeros otherwise (image size
    for a passthrough image layer)."""
    if is_product:
        return torch.ones(layer.n)
    return torch.zeros(_image_layer_fan_in(layer) if is_2d else layer.n)


@torch.no_grad()
def _modulator_value(layer, cache, mod_name, mode, value, dt):
    """A modulator row's transformed value (absolute / derivative / integral),
    computed once per layer per tick: rows sharing (modulator, mode) — e.g. a
    'pre' and a 'post' row — share state, so a second transform in the same
    tick would read a zero derivative or add the integral twice."""
    key = (mod_name, mode)
    if key not in cache:
        cache[key] = layer._transform_modulator_value(key, mode, value, dt)
    return cache[key]


def step_network(brain, dt):
    """
    Run one forward pass of the neural circuit stored on *brain*.

    Reads layers/connections/sensors from brain instance dict first,
    falling back to the class attributes — this mirrors how DataBrain
    sets per-instance attributes after a network file is loaded while
    Python-coded brains keep them as class-level lists.

    Mutates layer.output in-place. Weight tensors are cached on brain
    as _w_cache / _w_cache_conn_id and invalidated when the circuit changes:
    the connections or layers list object is replaced, or brain's
    _topology_version is bumped (circuit_editor does this on every committed
    edit, including property edits that change layer/sensor shapes). The same
    gate also guards the size-reconciliation and shape-metadata passes below.
    """
    layers      = brain.__dict__.get('layers')      or getattr(brain.__class__, 'layers',      [])
    connections = brain.__dict__.get('connections') or getattr(brain.__class__, 'connections', [])
    sensors     = brain.__dict__.get('sensors')     or getattr(brain.__class__, 'sensors',     [])
    active_layers = layers

    if _DEBUG_LAYER:
        names = tuple(l.name for l in active_layers)
        if names != getattr(step_network, '_debug_last_names', None):
            step_network._debug_last_names = names
            present = _DEBUG_LAYER in names
            print(f"[DEBUG-LAYER] active_layers changed ({len(names)} layers, "
                  f"{_DEBUG_LAYER!r} present={present}): {names}")

    for _layer in active_layers:
        if getattr(_layer, 'muted', False) and _layer.output is not None:
            # A fresh zero array, never zeroed in place: output can share memory
            # with the layer's state (Accumulator's _x, a linear Leaky's _x), and
            # that state must survive muting so the layer resumes on unmute.
            out = _layer.output
            _layer.output = torch.zeros_like(out) if isinstance(out, torch.Tensor) else np.zeros_like(out)
            _layer._was_muted = True
        elif getattr(_layer, '_was_muted', False):
            _layer._was_muted = False
            _layer.on_unmute()

    # conn_id gate: the connections list object is only replaced when topology
    # changes (add/remove/rename) or when a dialog edit deliberately bumps it
    # (see docstring above) — not on every tick. Everything below that only
    # depends on connection shapes / source-sensor shapes is gated by it,
    # including layer_map/layer_meta/conn_meta — precomputed classification
    # (isinstance/endswith/linear-scan work) that used to be re-derived on
    # every single tick regardless of whether topology had actually changed.
    # _topology_version is bumped by circuit_editor on every committed edit, so
    # in-place edits (e.g. a layer appended to the same list) are seen too.
    conn_id = (id(connections), id(active_layers), getattr(brain, '_topology_version', 0))
    topology_changed = getattr(brain, '_w_cache_conn_id', None) != conn_id

    if topology_changed:
        brain._layer_map = {l.name: l for l in active_layers}
    layer_map = brain._layer_map

    if topology_changed:
        # Size-reconciliation pass: ensure layer sizes match weight matrices.
        for conn in connections:
            src, tgt, W = conn.src, conn.tgt, conn.W
            W_arr = np.asarray(W, dtype=float)
            if W_arr.ndim == 4:
                # Conv2d kernel: (n_filters, in_ch, kH, kW) — tgt n = n_filters (global pool).
                # For a lateralized camera half (src ends _L/_R) feeding a NON-paired conv layer,
                # n = 2 * n_filters because the filter bank is mirrored on both sides
                # (L0…Ln R[n]…R0 ordering).  When the target IS part of a lateralized pair
                # (layer1_L / layer1_R), each half owns its own n_filters neurons — no doubling.
                nt = W_arr.shape[0]
                if is_camera_half(sensors, src):
                    _tgt_lyr = layer_map.get(tgt)
                    if getattr(_tgt_lyr, 'lateral_pair', None) is None:
                        nt = nt * 2
                if tgt in layer_map:
                    layer_map[tgt]._ensure_n(nt)
            elif W_arr.ndim == 3:
                # Legacy conv1d kernel: (n_filters, in_ch, kernel_size)
                nt = W_arr.shape[0]
                if tgt in layer_map:
                    layer_map[tgt]._ensure_n(nt)
            else:
                nt = W_arr.shape[0] if W_arr.ndim == 2 else W_arr.size
                ns = W_arr.shape[1] if W_arr.ndim == 2 else W_arr.size
                if tgt in layer_map:
                    _tgt_obj = layer_map[tgt]
                    # A passthrough layer whose output is pooled (e.g. Reichardt2dLayer
                    # with pool != 'none') receives its image as a 1-D ones passthrough
                    # sized to the pixel count — that's the connection's weight width
                    # (nt here), not the layer's own output width. Resizing .n to nt
                    # would blow it up to thousands of "neurons", freezing the visualizer.
                    _skip = (W_arr.ndim == 1 and _tgt_obj.passthrough_input
                             and not _tgt_obj.n_follows_input)
                    if not _skip:
                        _tgt_obj._ensure_n(nt)
                if src in layer_map and hasattr(layer_map[src], '_ensure_n'):
                    _src_obj  = layer_map[src]
                    _lat_pair = getattr(_src_obj, 'lateral_pair', None)
                    _pair_obj = layer_map.get(_lat_pair) if _lat_pair else None
                    # Skip _ensure_n when this is a combined lateralized-pair connection:
                    # W columns = n_L + n_R, which is larger than the src layer's own n.
                    _is_combined = (_pair_obj is not None and
                                    ns == (_src_obj.n or 0) + (_pair_obj.n or 0))
                    if not _is_combined:
                        try:
                            layer_map[src]._ensure_n(ns)
                        except ValueError:
                            pass

        # Propagate shape metadata (in_ch, frame_h, frame_w) from camera sources to
        # passthrough image layers so _update_last_frame reshapes correctly even
        # after a save/load cycle where those fields were not persisted.
        for conn in connections:
            tgt_obj = layer_map.get(conn.tgt)
            if tgt_obj is None or not tgt_obj.passthrough_input:
                continue
            src_sensor = next((s for s in sensors if s.name == conn.src), None)
            if src_sensor is None and side_of(conn.src):
                # Lateralized camera half — its parent camera gives the shape.
                cam = parent_sensor(sensors, conn.src)
                if getattr(cam, 'is_camera', False) and getattr(cam, 'lateralized', False):
                    tgt_obj.in_ch   = cam.in_ch
                    tgt_obj.frame_h = cam.height
                    tgt_obj.frame_w = cam.half_width(side_of(conn.src))
                continue
            if getattr(src_sensor, 'is_camera', False):
                tgt_obj.in_ch   = src_sensor.in_ch
                tgt_obj.frame_h = src_sensor.height
                tgt_obj.frame_w = src_sensor.width

    # Which layers / sensors transmit or receive neuromodulators — topology
    # (transmitter and receptor edits are edit transactions, which bump it),
    # so it's found once instead of scanning every element every step.
    if topology_changed or not hasattr(brain, '_mod_transmitters'):
        brain._mod_transmitters = (
            [l for l in active_layers if getattr(l, 'neuromodulator_transmitter', None)],
            [s for s in sensors if getattr(s, 'neuromodulator_transmitter', None)])
        brain._mod_sensors = [s for s in sensors if getattr(s, 'modulators', [])]
    tx_layers, tx_sensors = brain._mod_transmitters

    # Build neuromodulator map from transmitter layers and sensors.
    mod_map = {}
    for layer in tx_layers:
        nt = layer.neuromodulator_transmitter
        if layer.output is not None:
            out = layer.output
            mod_map[nt] = float(out.mean() if isinstance(out, torch.Tensor)
                                else np.mean(np.atleast_1d(out)))
    for sensor in tx_sensors:
        nt = sensor.neuromodulator_transmitter
        val = getattr(brain, sensor.name, None)
        if val is not None:
            mod_map[nt] = float(np.mean(np.atleast_1d(val)))

    # Apply neuromodulation to sensor outputs.
    for sensor in brain._mod_sensors:
        mods = getattr(sensor, 'modulators', [])
        if not mods:
            continue
        val = getattr(brain, sensor.name, None)
        if val is None:
            continue
        pre_gain  = 1.0
        post_gain = 1.0
        for row in mods:
            mod_name, scale, site, mode, _dp, _th = _unpack_mod_row(row)
            if mod_name in mod_map:
                v = sensor._transform_modulator_value((mod_name, mode), mode, mod_map[mod_name], dt)
                if site == 'pre':
                    pre_gain  += scale * v
                else:
                    post_gain += scale * v
        # Pre-site: re-apply activation on gain-scaled pre-activation value so the
        # activation function (e.g. relu) acts after the gain, not before.
        # output_mode runs upstream of _pre_activation_output (before the leaky
        # filter), so any derivative/integral transform is already baked into
        # it here — no special-casing needed regardless of output_mode.
        # Must also reapply sensor.scale, since BaseSensor._process() now applies
        # it after activation (matching LeakyLayer) — recomputing activation alone
        # would silently drop the sensor's scale factor.
        pre_act = getattr(sensor, '_pre_activation_output', None)
        if pre_gain != 1.0 and pre_act is not None:
            from neurons import _activate as _act
            activation = getattr(sensor, 'activation', 'linear')
            sensor_scale = getattr(sensor, 'scale', 1.0)
            val = np.asarray(_act(pre_act * pre_gain, activation) * sensor_scale, dtype=np.float32)
            setattr(brain, sensor.name, val)
        elif pre_gain != 1.0:
            post_gain *= pre_gain   # fallback: treat as post
        if post_gain != 1.0:
            val = np.atleast_1d(getattr(brain, sensor.name, val))
            setattr(brain, sensor.name, val * post_gain)

    # Weight tensor cache — invalidated when the connections list object is replaced.
    # The target->connections index below is invalidated on the same event (same
    # conn_id gate computed above) since both only need rebuilding when the
    # connections list itself is swapped out, not on every tick.
    if topology_changed:
        brain._w_cache = {}
        brain._conn_meta = {}
        brain._w_cache_conn_id = conn_id
        conn_by_tgt = {}
        conn_by_src = {}
        for i, conn in enumerate(connections):
            conn_by_tgt.setdefault(conn.tgt, []).append((i, conn))
            conn_by_src.setdefault(conn.src, []).append((i, conn))
        brain._conn_by_tgt = conn_by_tgt
        brain._conn_by_src = conn_by_src
        # Per-layer classification — declared by each layer class, never per-tick data.
        brain._layer_meta = {
            l.name: {
                'is_learning': l.is_learning,
                'is_product':  l.combines_by_product,
                'is_2d':       l.passthrough_input,
                'has_outgoing_plasticity': l.has_outgoing_plasticity,
            }
            for l in active_layers
        }
    conn_by_tgt = brain._conn_by_tgt
    conn_by_src = brain._conn_by_src
    layer_meta  = brain._layer_meta

    # Main forward pass: accumulate weighted inputs, step each layer.
    for layer in active_layers:
        if layer.n is None:
            if layer.name == _DEBUG_LAYER:
                print(f"[DEBUG-LAYER] {layer.name}: skipped, layer.n is None")
            continue
        if getattr(layer, 'muted', False):
            if layer.name == _DEBUG_LAYER:
                print(f"[DEBUG-LAYER] {layer.name}: skipped, layer.muted=True")
            continue

        lmeta = layer_meta[layer.name]

        # TDLayer: pass raw (src_val, w_cached, conn_index, conn) so it can
        # compute its own weighted sum and update the connection weights in place.
        if lmeta['is_learning']:
            src_inputs = []
            for i, conn in conn_by_tgt.get(layer.name, []):
                src_val = getattr(brain, conn.src, None)
                if src_val is None:
                    continue
                if hasattr(src_val, 'output'):
                    src_val = src_val.output
                if src_val is None:
                    continue
                if not isinstance(src_val, torch.Tensor):
                    src_val = torch.as_tensor(np.atleast_1d(src_val), dtype=torch.float32)
                if i not in brain._w_cache:
                    arr = np.asarray(conn.W, dtype=np.float32)
                    n_tgt = layer.n or 1
                    n_src = int(src_val.shape[0])
                    if arr.ndim < 2 or arr.shape != (n_tgt, n_src) or np.any(~np.isfinite(arr)):
                        arr = np.zeros((n_tgt, n_src), dtype=np.float32)
                        conn.W = arr.copy()
                    brain._w_cache[i] = torch.from_numpy(arr.copy())
                src_inputs.append((src_val, brain._w_cache[i], i, conn))
            mod_cache = {}   # (modulator, mode) → value, advanced once per tick
            reward_total = 0.0
            for row in getattr(layer, 'modulators', []):
                mod_name, scale, site, mode, drives_plasticity, threshold = _unpack_mod_row(row)
                if not drives_plasticity or mod_name not in mod_map:
                    continue
                v = _modulator_value(layer, mod_cache, mod_name, mode, mod_map[mod_name], dt)
                if v >= threshold:
                    reward_total += scale * v
            rm = getattr(layer, 'reward_modulator', None)   # legacy field, still honored additively
            if rm:
                reward_total += mod_map.get(rm, 0.0)
            layer._reward = reward_total

            # SnapshotLayer-only: gather the layer's OUTGOING connections (it's
            # the src, not the tgt) so step_td can overwrite their weights
            # directly — the one LearningLayerBase subclass whose plasticity
            # lives on outgoing edges rather than incoming ones. Reuses the
            # same brain._w_cache dict incoming connections already use
            # (keyed by connection index, agnostic to which endpoint is
            # treating it as incoming vs. outgoing).
            outgoing = None
            if lmeta.get('has_outgoing_plasticity'):
                outgoing = []
                for i, conn in conn_by_src.get(layer.name, []):
                    tgt_obj = layer_map.get(conn.tgt)
                    n_tgt = (tgt_obj.n or 1) if tgt_obj is not None else None
                    if n_tgt is None:
                        continue
                    if i not in brain._w_cache:
                        arr   = np.asarray(conn.W, dtype=np.float32)
                        n_src = layer.n or 1
                        if arr.ndim < 2 or arr.shape != (n_tgt, n_src) or np.any(~np.isfinite(arr)):
                            arr = np.zeros((n_tgt, n_src), dtype=np.float32)
                            conn.W = arr.copy()
                        brain._w_cache[i] = torch.from_numpy(arr.copy())
                    outgoing.append((brain._w_cache[i], i, conn))

            layer.step_td(src_inputs, dt, outgoing=outgoing)
            post = 1.0
            for row in getattr(layer, 'modulators', []):
                mod_name, scale, site, mode, _dp, _th = _unpack_mod_row(row)
                if site == 'post' and mod_name in mod_map:
                    v = _modulator_value(layer, mod_cache, mod_name, mode, mod_map[mod_name], dt)
                    post += scale * v
            if post != 1.0 and layer.output is not None:
                layer.output = layer.output * post
            continue

        _is_product = lmeta['is_product']
        _is_2d      = lmeta['is_2d']
        # inp starts as None: a plain sum takes its first contribution as is
        # (0 + c == c exactly), saving a zeros tensor and an add per layer per step.
        # Paths that write into inp (conv halves, passthrough) allocate it first.
        inp = None
        for i, conn in conn_by_tgt.get(layer.name, []):
            src, tgt, W = conn.src, conn.tgt, conn.W
            src_val = getattr(brain, src, None)
            if src_val is None:
                if layer.passthrough_input:
                    print(f"[L2D] src_val=None for brain.{src!r} — attr missing!")
                if tgt == _DEBUG_LAYER:
                    print(f"[DEBUG-LAYER] {tgt}: conn from {src!r} — brain.{src} is None, skipped")
                continue
            if hasattr(src_val, 'output'):
                src_val = src_val.output
            if src_val is None:
                if tgt == _DEBUG_LAYER:
                    print(f"[DEBUG-LAYER] {tgt}: conn from {src!r} — .output is None, skipped")
                continue
            if not isinstance(src_val, torch.Tensor):
                src_val = torch.as_tensor(np.atleast_1d(src_val), dtype=torch.float32)
            if tgt == _DEBUG_LAYER:
                print(f"[DEBUG-LAYER] {tgt}: conn from {src!r} value={src_val}")
            if i not in brain._w_cache:
                arr = np.asarray(W, dtype=np.float32)
                if arr.ndim == 0:
                    arr = arr.reshape(1, 1)
                elif arr.ndim == 1 and not _is_2d:
                    # Leaky2dLayer / Reichardt2dLayer keep 1-D weights as-is for element-wise passthrough.
                    arr = np.diag(arr)
                brain._w_cache[i] = torch.from_numpy(arr.copy())
            w_cached = brain._w_cache[i]

            # Connection routing (conv4d / 1-D passthrough / lateralized-pair
            # concatenation) is static between topology changes — classified
            # once per connection and cached, instead of re-deriving it
            # (isinstance + endswith + linear scans over sensors) every tick.
            meta = brain._conn_meta.get(i)
            if meta is None:
                meta = _build_conn_meta(conn, layer, w_cached, sensors, layer_map)
                brain._conn_meta[i] = meta

            if inp is None and (_is_product or _is_2d or meta['is_conv4d'] or meta['is_passthrough1d']):
                inp = _initial_input(layer, _is_product, _is_2d)
            if meta['is_conv4d']:
                result = _conv_forward(src_val, w_cached, layer, meta['src_sensor'])
                n_half = result.shape[0]
                if meta['is_lat_cam_half'] and layer.n == 2 * n_half:
                    # Lateralized camera → single conv: mirrored layout L0…L[n/2-1] R[n/2-1]…R0.
                    # L side fills the top half sequentially; R side fills the bottom half reversed.
                    if side_of(src) == 'L':
                        inp = inp.clone()
                        inp[:n_half] = inp[:n_half] + result
                    else:
                        inp = inp.clone()
                        inp[n_half:] = inp[n_half:] + result.flip(0)
                else:
                    inp = inp + result
            elif meta['is_passthrough1d']:
                # Leaky2dLayer / Reichardt2dLayer passthrough: element-wise multiply (avoids n×n diag matrix).
                # Skip if sizes mismatch (stale connection from a size-change).
                if src_val.shape[0] == w_cached.shape[0] == _image_layer_fan_in(layer):
                    inp = inp + src_val * w_cached
            else:
                # Combined lateralized source: W has n_L+n_R columns.
                # Concatenate both halves' outputs before the linear transform.
                # R half is reversed (flip(0)) so columns follow visual top-to-bottom order.
                if meta['pair_obj'] is not None:
                    _pair_out = meta['pair_obj'].output
                    if _pair_out is not None:
                        if not isinstance(_pair_out, torch.Tensor):
                            _pair_out = torch.as_tensor(
                                np.atleast_1d(_pair_out), dtype=torch.float32)
                        src_val = torch.cat([src_val, _pair_out.flip(0)])
                elif meta['is_lat_sensor_half']:
                    n_half = src_val.shape[0]
                    if w_cached.shape[1] == n_half * 2:
                        _pair_val = getattr(brain, meta['sensor_partner'], None)
                        if _pair_val is not None:
                            if not isinstance(_pair_val, torch.Tensor):
                                _pair_val = torch.as_tensor(
                                    np.atleast_1d(_pair_val), dtype=torch.float32)
                            if side_of(src) == 'L':
                                src_val = torch.cat([src_val, _pair_val.flip(0)])
                            else:
                                src_val = torch.cat([_pair_val.flip(0), src_val])
                contrib = F.linear(src_val, w_cached)
                if inp is None:
                    inp = contrib
                else:
                    inp = inp * contrib if _is_product else inp + contrib

        if inp is None:
            inp = _initial_input(layer, _is_product, _is_2d)

        mod_cache = {}   # (modulator, mode) → value, advanced once per tick
        pre = 1.0
        for row in getattr(layer, 'modulators', []):
            mod_name, scale, site, mode, _dp, _th = _unpack_mod_row(row)
            if site == 'pre' and mod_name in mod_map:
                v = _modulator_value(layer, mod_cache, mod_name, mode, mod_map[mod_name], dt)
                pre += scale * v
        if pre != 1.0:
            inp = inp * pre

        rm = getattr(layer, 'reward_modulator', None)
        if rm and hasattr(layer, '_reward'):
            layer._reward = mod_map.get(rm, 0.0)

        if layer.name == _DEBUG_LAYER:
            print(f"[DEBUG-LAYER] {layer.name}: pre-step inp={inp}")

        layer.step(inp, dt)

        if layer.name == _DEBUG_LAYER:
            print(f"[DEBUG-LAYER] {layer.name}: post-step output={layer.output}")

        post = 1.0
        for row in getattr(layer, 'modulators', []):
            mod_name, scale, site, mode, _dp, _th = _unpack_mod_row(row)
            if site == 'post' and mod_name in mod_map:
                v = _modulator_value(layer, mod_cache, mod_name, mode, mod_map[mod_name], dt)
                post += scale * v
        if post != 1.0 and layer.output is not None:
            layer.output = layer.output * post
