"""
network_runner.py — standalone neural forward pass.

Extracted from BaseBrain.step_network so it can be tested independently and
imported without subclassing BaseBrain.
"""

import numpy as np
import torch
import torch.nn.functional as F
from neurons import Conv2dLayer as _Conv2dLayer, LearningLayerBase as _LLB, _activate, Leaky2dLayer as _L2d, Reichardt2dLayer as _R2d


def _is_lat_cam_half(src_name, sensors):
    """Return True when src_name is a lateralized camera half (ends _L/_R, parent is lateralized)."""
    if not (src_name.endswith('_L') or src_name.endswith('_R')):
        return False
    parent_name = src_name.rsplit('_', 1)[0]
    return any(s.name == parent_name and getattr(s, 'lateralized', False) for s in sensors)


def _is_lat_sensor_half(src_name, sensors):
    """Return True when src_name is a joint-pair sensor half (ends _L/_R, parent has 2 body_ids).

    Unlike camera halves (explicit lateralized=True), joint-pair halves are detected by
    len(body_ids)==2 — this is the runtime check; mirror_group validation is done at
    edit time in the visualizer where circuit.bodies is available.
    """
    if not (src_name.endswith('_L') or src_name.endswith('_R')):
        return False
    parent_name = src_name.rsplit('_', 1)[0]
    for s in sensors:
        if s.name == parent_name:
            if getattr(s, 'lateralized', False):
                return False   # camera half — handled by _is_lat_cam_half
            return len(getattr(s, 'body_ids', [])) == 2
    return False


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
    out = _activate(out, getattr(layer, 'activation', 'relu'))

    if layer.pool == 'global_avg':
        return out.mean(dim=(-2, -1))                              # (n_filters,)
    elif layer.pool == 'global_max':
        return out.flatten(1).max(dim=-1).values                   # (n_filters,)
    else:
        _, H_out, W_out = out.shape
        layer.frame_h = H_out
        layer.frame_w = W_out
        return out.reshape(-1)                                     # (n_filters * H_out * W_out,)


def step_network(brain, dt):
    """
    Run one forward pass of the neural circuit stored on *brain*.

    Reads layers/connections/sensors from brain instance dict first,
    falling back to the class attributes — this mirrors how DataBrain
    sets per-instance attributes after a network file is loaded while
    Python-coded brains keep them as class-level lists.

    Mutates layer.output in-place. Weight tensors are cached on brain
    as _w_cache / _w_cache_conn_id and invalidated when the connections
    list object is replaced.
    """
    layers      = brain.__dict__.get('layers')      or getattr(brain.__class__, 'layers',      [])
    connections = brain.__dict__.get('connections') or getattr(brain.__class__, 'connections', [])
    sensors     = brain.__dict__.get('sensors')     or getattr(brain.__class__, 'sensors',     [])
    active_layers = layers

    for _layer in active_layers:
        if getattr(_layer, 'muted', False) and _layer.output is not None:
            _layer.output[:] = 0.0

    # Size-reconciliation pass: ensure layer sizes match weight matrices.
    layer_map = {l.name: l for l in active_layers}
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
            if _is_lat_cam_half(src, sensors):
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
                layer_map[tgt]._ensure_n(nt)
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
    # Leaky2dLayer targets so _update_last_frame reshapes correctly even after a
    # save/load cycle where those fields were not persisted.
    from sensors import GrayCameraSensor as _GrayCam, RGBCameraSensor as _RGBCam
    for conn in connections:
        tgt_obj = layer_map.get(conn.tgt)
        if not isinstance(tgt_obj, (_L2d, _R2d)):
            continue
        src_sensor = next((s for s in sensors if s.name == conn.src), None)
        if src_sensor is None and conn.src.endswith(('_L', '_R')):
            # Lateralized camera half — find parent and derive half width.
            parent_name   = conn.src.rsplit('_', 1)[0]
            parent_sensor = next((s for s in sensors if s.name == parent_name), None)
            if isinstance(parent_sensor, (_GrayCam, _RGBCam)) and getattr(parent_sensor, 'lateralized', False):
                mid     = parent_sensor.width // 2
                overlap = getattr(parent_sensor, 'overlap', 0)
                if conn.src.endswith('_L'):
                    half_w = int(np.clip(mid + overlap, 0, parent_sensor.width))
                else:
                    r_start = int(np.clip(mid - overlap, 0, parent_sensor.width))
                    half_w  = parent_sensor.width - r_start
                tgt_obj.in_ch   = 3 if isinstance(parent_sensor, _RGBCam) else 1
                tgt_obj.frame_h = parent_sensor.height
                tgt_obj.frame_w = half_w
            continue
        if isinstance(src_sensor, (_GrayCam, _RGBCam)):
            tgt_obj.in_ch   = 3 if isinstance(src_sensor, _RGBCam) else 1
            tgt_obj.frame_h = getattr(src_sensor, 'height', None)
            tgt_obj.frame_w = getattr(src_sensor, 'width', None)

    # Build neuromodulator map from transmitter layers and sensors.
    mod_map = {}
    for layer in active_layers:
        nt = getattr(layer, 'neuromodulator_transmitter', None)
        if nt and layer.output is not None:
            out = layer.output
            mod_map[nt] = float(out.mean() if isinstance(out, torch.Tensor)
                                else np.mean(np.atleast_1d(out)))
    for sensor in sensors:
        nt = getattr(sensor, 'neuromodulator_transmitter', None)
        if nt:
            val = getattr(brain, sensor.name, None)
            if val is not None:
                mod_map[nt] = float(np.mean(np.atleast_1d(val)))

    # Apply neuromodulation to sensor outputs.
    for sensor in sensors:
        mods = getattr(sensor, 'modulators', [])
        if not mods:
            continue
        val = getattr(brain, sensor.name, None)
        if val is None:
            continue
        pre_gain  = 1.0
        post_gain = 1.0
        for mod_name, scale, site in mods:
            if mod_name in mod_map:
                if site == 'pre':
                    pre_gain  += scale * mod_map[mod_name]
                else:
                    post_gain += scale * mod_map[mod_name]
        # Pre-site: re-apply activation on gain-scaled pre-activation value so the
        # activation function (e.g. relu) acts after the gain, not before.
        # Not applicable to differential sensors (history already consumed).
        pre_act = getattr(sensor, '_pre_activation_output', None)
        if pre_gain != 1.0 and pre_act is not None and not getattr(sensor, 'differential', False):
            from neurons import _activate as _act
            activation = getattr(sensor, 'activation', 'linear')
            val = np.asarray(_act(pre_act * pre_gain, activation), dtype=np.float32)
            setattr(brain, sensor.name, val)
        elif pre_gain != 1.0:
            post_gain *= pre_gain   # fallback: treat as post
        if post_gain != 1.0:
            val = np.atleast_1d(getattr(brain, sensor.name, val))
            setattr(brain, sensor.name, val * post_gain)

    # Weight tensor cache — invalidated when the connections list object is replaced.
    conn_id = id(connections)
    if getattr(brain, '_w_cache_conn_id', None) != conn_id:
        brain._w_cache = {}
        brain._w_cache_conn_id = conn_id

    # Main forward pass: accumulate weighted inputs, step each layer.
    for layer in active_layers:
        if layer.n is None:
            continue
        if getattr(layer, 'muted', False):
            continue

        # TDLayer: pass raw (src_val, w_cached, conn_index, conn) so it can
        # compute its own weighted sum and update the connection weights in place.
        if isinstance(layer, _LLB):
            src_inputs = []
            for i, conn in enumerate(connections):
                if conn.tgt != layer.name:
                    continue
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
            rm = getattr(layer, 'reward_modulator', None)
            if rm:
                layer._reward = mod_map.get(rm, 0.0)
            layer.step_td(src_inputs, dt)
            post = 1.0
            for mod_name, scale, site in getattr(layer, 'modulators', []):
                if site == 'post' and mod_name in mod_map:
                    post += scale * mod_map[mod_name]
            if post != 1.0 and layer.output is not None:
                layer.output = layer.output * post
            continue

        inp = torch.zeros(layer.n)
        for i, conn in enumerate(connections):
            src, tgt, W = conn.src, conn.tgt, conn.W
            if tgt != layer.name:
                continue
            src_val = getattr(brain, src, None)
            if src_val is None:
                if isinstance(layer, _L2d):
                    print(f"[L2D] src_val=None for brain.{src!r} — attr missing!")
                continue
            if hasattr(src_val, 'output'):
                src_val = src_val.output
            if src_val is None:
                continue
            if not isinstance(src_val, torch.Tensor):
                src_val = torch.as_tensor(np.atleast_1d(src_val), dtype=torch.float32)
            if i not in brain._w_cache:
                arr = np.asarray(W, dtype=np.float32)
                if arr.ndim == 0:
                    arr = arr.reshape(1, 1)
                elif arr.ndim == 1 and not isinstance(layer, (_L2d, _R2d)):
                    # Leaky2dLayer / Reichardt2dLayer keep 1-D weights as-is for element-wise passthrough.
                    arr = np.diag(arr)
                brain._w_cache[i] = torch.from_numpy(arr.copy())
            w_cached = brain._w_cache[i]
            if w_cached.ndim == 4 and isinstance(layer, _Conv2dLayer):
                src_sensor = next((s for s in sensors if s.name == src), None)
                if src_sensor is None:
                    # Lateralized half: 'sensor0_L' → resolve to parent 'sensor0'
                    parent = src.rsplit('_', 1)[0]
                    src_sensor = next((s for s in sensors if s.name == parent), None)
                if src_sensor is None:
                    # Source may be a Leaky2dLayer — use it as shape proxy.
                    src_sensor = layer_map.get(src)
                result = _conv_forward(src_val, w_cached, layer, src_sensor)
                n_half = result.shape[0]
                if _is_lat_cam_half(src, sensors) and layer.n == 2 * n_half:
                    # Lateralized camera → single conv: mirrored layout L0…L[n/2-1] R[n/2-1]…R0.
                    # L side fills the top half sequentially; R side fills the bottom half reversed.
                    if src.endswith('_L'):
                        inp = inp.clone()
                        inp[:n_half] = inp[:n_half] + result
                    else:
                        inp = inp.clone()
                        inp[n_half:] = inp[n_half:] + result.flip(0)
                else:
                    inp = inp + result
            elif w_cached.ndim == 1 and isinstance(layer, (_L2d, _R2d)):
                # Leaky2dLayer / Reichardt2dLayer passthrough: element-wise multiply (avoids n×n diag matrix).
                # Skip if sizes mismatch (stale connection from a size-change).
                if src_val.shape[0] == w_cached.shape[0] == layer.n:
                    inp = inp + src_val * w_cached
            else:
                # Combined lateralized source: W has n_L+n_R columns.
                # Concatenate both halves' outputs before the linear transform.
                # R half is reversed (flip(0)) so columns follow visual top-to-bottom order.
                if w_cached.ndim == 2:
                    _src_obj  = layer_map.get(src)
                    _lat_pair = getattr(_src_obj, 'lateral_pair', None) if _src_obj else None
                    if _lat_pair:
                        # Conv2dLayer pair: partner output from layer_map.
                        _pair_obj = layer_map.get(_lat_pair)
                        if (_pair_obj is not None and
                                w_cached.shape[1] == (_src_obj.n or 0) + (_pair_obj.n or 0)):
                            _pair_out = _pair_obj.output
                            if _pair_out is not None:
                                if not isinstance(_pair_out, torch.Tensor):
                                    _pair_out = torch.as_tensor(
                                        np.atleast_1d(_pair_out), dtype=torch.float32)
                                src_val = torch.cat([src_val, _pair_out.flip(0)])
                    elif _is_lat_sensor_half(src, sensors):
                        # Joint-pair sensor half: partner output from brain attributes.
                        n_half = src_val.shape[0]
                        if w_cached.shape[1] == n_half * 2:
                            partner = src[:-2] + '_R' if src.endswith('_L') else src[:-2] + '_L'
                            _pair_val = getattr(brain, partner, None)
                            if _pair_val is not None:
                                if not isinstance(_pair_val, torch.Tensor):
                                    _pair_val = torch.as_tensor(
                                        np.atleast_1d(_pair_val), dtype=torch.float32)
                                if src.endswith('_L'):
                                    src_val = torch.cat([src_val, _pair_val.flip(0)])
                                else:
                                    src_val = torch.cat([_pair_val.flip(0), src_val])
                inp = inp + F.linear(src_val, w_cached)

        pre = 1.0
        for mod_name, scale, site in getattr(layer, 'modulators', []):
            if site == 'pre' and mod_name in mod_map:
                pre += scale * mod_map[mod_name]
        if pre != 1.0:
            inp = inp * pre

        rm = getattr(layer, 'reward_modulator', None)
        if rm and hasattr(layer, '_reward'):
            layer._reward = mod_map.get(rm, 0.0)

        layer.step(inp, dt)

        post = 1.0
        for mod_name, scale, site in getattr(layer, 'modulators', []):
            if site == 'post' and mod_name in mod_map:
                post += scale * mod_map[mod_name]
        if post != 1.0 and layer.output is not None:
            layer.output = layer.output * post
