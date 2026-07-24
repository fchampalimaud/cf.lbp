"""
Brain serializer: code-generation and file-patching utilities.

All functions are pure (no Qt, no GUI state). They can be called from
network_viz, tests, or any future tool that needs to write brain .py files.
"""

import json
import os
import re
import numpy as np
from neurons import (LeakyLayer, MatsuokaLayer, ConstantLayer,
                     AdaptiveLayer, SumLayer, PulseLayer, SineLayer,
                     RingAttractorLayer, Conv2dLayer, Leaky2dLayer, LAYER_REGISTRY)
from circuit_model import Connection

_ALL_NEURON_TYPES = sorted(LAYER_REGISTRY.keys())


# ── Code generation helpers ────────────────────────────────────────────────────

def _append_mod_parts(lyr, parts):
    """Append modulator / neuromodulator kwargs to a parts list."""
    if getattr(lyr, 'modulators', None):
        parts.append(f'modulators={lyr.modulators!r}')
    if getattr(lyr, 'neuromodulator_transmitter', None):
        parts.append(f'neuromodulator_transmitter={lyr.neuromodulator_transmitter!r}')
    if getattr(lyr, 'neuromodulator_color', None):
        parts.append(f'neuromodulator_color={lyr.neuromodulator_color!r}')


def _append_dynamic_parts(lyr, parts):
    """Append optional filter attrs shared by LeakyLayer and AdaptiveLayer."""
    if lyr.bias != 0.0:
        parts.append(f'bias={lyr.bias}')
    if lyr.activation != 'relu':
        parts.append(f"activation='{lyr.activation}'")
    if getattr(lyr, 'noise_std', 0.0):
        parts.append(f'noise_std={lyr.noise_std}')
    if getattr(lyr, 'noise_tau', 0.0):
        parts.append(f'noise_tau={lyr.noise_tau}')
    if getattr(lyr, 'scale', 1.0) != 1.0:
        parts.append(f'scale={lyr.scale}')




# ── Code generation ────────────────────────────────────────────────────────────

def generate_layer_code(lyr) -> str:
    """Return a constructor expression string for *lyr* (one line, no trailing comma)."""
    t     = type(lyr).__name__
    parts = [f"name='{lyr.name}'"]
    if getattr(lyr, 'color', None) is not None:
        parts.append(f"color='{lyr.color}'")
    if getattr(lyr, 'layer', None) is not None:
        parts.append(f'layer={lyr.layer}')

    if isinstance(lyr, LeakyLayer):
        parts.insert(0, f'tau_rise={lyr.tau_rise}')
        parts.insert(1, f'tau_decay={lyr.tau_decay}')
        if lyr.n is not None:
            parts.append(f'n={lyr.n}')
        if getattr(lyr, 'derivative', False):
            parts.append('derivative=True')
        _append_dynamic_parts(lyr, parts)
        _append_mod_parts(lyr, parts)
        return f'LeakyLayer({", ".join(parts)})'

    if isinstance(lyr, MatsuokaLayer):
        parts = [f'tau_rise={lyr.tau_rise}', f'tau_a={lyr.tau_a}',
                 f'beta={lyr.beta}', f'w={lyr.w}'] + parts
        if lyr.bias != 0.0:
            parts.append(f'bias={lyr.bias}')
        _append_mod_parts(lyr, parts)
        return f'MatsuokaLayer({", ".join(parts)})'

    if isinstance(lyr, ConstantLayer):
        v    = lyr._value
        vstr = repr(float(v[0])) if v.size == 1 else repr(v.tolist())
        parts = [f'value={vstr}', f'n={lyr.n}'] + parts
        if getattr(lyr, 'noise_std', 0.0):
            parts.append(f'noise_std={lyr.noise_std}')
        _append_mod_parts(lyr, parts)
        return f'ConstantLayer({", ".join(parts)})'

    if isinstance(lyr, AdaptiveLayer):
        parts.insert(0, f'tau_rise={lyr.tau_rise}')
        parts.insert(1, f'tau_decay={lyr.tau_decay}')
        parts.append(f'tau_a={lyr.tau_a}')
        parts.append(f'beta={lyr.beta}')
        if lyr.w != 0.0:
            parts.append(f'w={lyr.w}')
        if lyr.n is not None:
            parts.append(f'n={lyr.n}')
        _append_dynamic_parts(lyr, parts)
        _append_mod_parts(lyr, parts)
        return f'AdaptiveLayer({", ".join(parts)})'

    if isinstance(lyr, SumLayer):
        if lyr.activation != 'relu':
            parts.insert(0, f"activation='{lyr.activation}'")
        if getattr(lyr, 'scale', 1.0) != 1.0:
            parts.append(f'scale={lyr.scale}')
        if lyr.n is not None:
            parts.append(f'n={lyr.n}')
        _append_mod_parts(lyr, parts)
        return f'SumLayer({", ".join(parts)})'

    if isinstance(lyr, PulseLayer):
        parts.insert(0, f'tau_rise={lyr.tau_rise}')
        parts.insert(1, f'tau_decay={lyr.tau_decay}')
        parts.insert(2, f'tau_hold={lyr.tau_hold}')
        if lyr.theta != 0.0:
            parts.append(f'theta={lyr.theta}')
        if lyr.w_s != 1.0:
            parts.append(f'w_s={lyr.w_s}')
        if lyr.drain != 1.0:
            parts.append(f'drain={lyr.drain}')
        if lyr.n is not None:
            parts.append(f'n={lyr.n}')
        if lyr.bias != 0.0:
            parts.append(f'bias={lyr.bias}')
        if lyr.activation != 'relu':
            parts.append(f"activation='{lyr.activation}'")
        if getattr(lyr, 'scale', 1.0) != 1.0:
            parts.append(f'scale={lyr.scale}')
        _append_mod_parts(lyr, parts)
        return f'PulseLayer({", ".join(parts)})'

    if isinstance(lyr, SineLayer):
        parts.insert(0, f'amplitude={lyr.amplitude}')
        parts.insert(1, f'frequency={lyr.frequency}')
        if lyr.phase != 0.0:
            parts.append(f'phase={lyr.phase}')
        if getattr(lyr, 'n', 1) != 1:
            parts.append(f'n={lyr.n}')
        _append_mod_parts(lyr, parts)
        return f'SineLayer({", ".join(parts)})'

    if isinstance(lyr, RingAttractorLayer):
        parts.insert(0, f'n={lyr.n}')
        parts.insert(1, f'tau_rise={lyr.tau_rise}')
        if lyr.tau_decay != lyr.tau_rise:
            parts.insert(2, f'tau_decay={lyr.tau_decay}')
        if lyr.activation != 'relu':
            parts.append(f"activation='{lyr.activation}'")
        if lyr.bias != 0.0:
            parts.append(f'bias={lyr.bias}')
        if lyr.noise_std != 0.0:
            parts.append(f'noise_std={lyr.noise_std}')
        if lyr.noise_tau != 0.0:
            parts.append(f'noise_tau={lyr.noise_tau}')
        _append_mod_parts(lyr, parts)
        return f'RingAttractorLayer({", ".join(parts)})'

    if isinstance(lyr, Conv2dLayer):
        parts.insert(0, f'n_filters={lyr.n_filters}')
        parts.insert(1, f'kernel_size={lyr.kernel_size}')
        if lyr.stride != 1:
            parts.append(f'stride={lyr.stride}')
        if lyr.padding != 'same':
            parts.append(f"padding='{lyr.padding}'")
        if lyr.pool != 'global_avg':
            parts.append(f"pool='{lyr.pool}'")
        if lyr.activation != 'relu':
            parts.append(f"activation='{lyr.activation}'")
        if lyr.tau_rise != 0.0:
            parts.append(f'tau_rise={lyr.tau_rise}')
        if lyr.tau_decay != lyr.tau_rise:
            parts.append(f'tau_decay={lyr.tau_decay}')
        if lyr.bias != 0.0:
            parts.append(f'bias={lyr.bias}')
        if getattr(lyr, 'scale', 1.0) != 1.0:
            parts.append(f'scale={lyr.scale}')
        if getattr(lyr, 'lateralized', False):
            parts.append('lateralized=True')
        _append_mod_parts(lyr, parts)
        return f'Conv2dLayer({", ".join(parts)})'

    if isinstance(lyr, Leaky2dLayer):
        parts.insert(0, f'tau_rise={lyr.tau_rise}')
        parts.insert(1, f'tau_decay={lyr.tau_decay}')
        if lyr.activation != 'linear':
            parts.append(f"activation='{lyr.activation}'")
        if lyr.n is not None:
            parts.append(f'n={lyr.n}')
        if getattr(lyr, 'derivative', False):
            parts.append('derivative=True')
        if lyr.in_ch != 1:
            parts.append(f'in_ch={lyr.in_ch}')
        if lyr.frame_h is not None:
            parts.append(f'frame_h={lyr.frame_h}')
        if lyr.frame_w is not None:
            parts.append(f'frame_w={lyr.frame_w}')
        if lyr.bias != 0.0:
            parts.append(f'bias={lyr.bias}')
        if getattr(lyr, 'noise_std', 0.0):
            parts.append(f'noise_std={lyr.noise_std}')
        if getattr(lyr, 'noise_tau', 0.0):
            parts.append(f'noise_tau={lyr.noise_tau}')
        if getattr(lyr, 'scale', 1.0) != 1.0:
            parts.append(f'scale={lyr.scale}')
        _append_mod_parts(lyr, parts)
        return f'Leaky2dLayer({", ".join(parts)})'

    return f'{t}(name="{lyr.name}")'


def generate_weight_code(W, ns: int, nt: int) -> str:
    """Return a numpy expression string that reconstructs weight matrix *W*."""
    W = np.asarray(W, dtype=float)
    if W.ndim in (3, 4):
        # Conv kernel: 3D=(n_filters,in_ch,ksize) legacy, 4D=(n_filters,in_ch,kH,kW) conv2d
        # Write as a compact nested array literal.
        def _fmt_nd(arr):
            if arr.ndim == 1:
                return '[' + ', '.join(f'{v:.6g}' for v in arr) + ']'
            return '[' + ', '.join(_fmt_nd(sub) for sub in arr) + ']'
        return 'np.array([' + ', '.join(_fmt_nd(filt) for filt in W) + '])'
    n   = max(ns, nt)
    tol = 1e-9
    if W.shape == (n, n):
        eye  = np.eye(n)
        flip = np.fliplr(eye)
        ones = np.ones((n, n))
        for M, label in [(eye, f'np.eye({n})'), (flip, f'np.fliplr(np.eye({n}))')]:
            for sign in (1.0, -1.0):
                if np.allclose(W, sign * M, atol=tol):
                    return label if abs(sign - 1.0) < tol else f'-{label}'
            ratio      = W / (M + 1e-30)
            ratio_flat = ratio[np.abs(M) > tol]
            if ratio_flat.size > 0 and np.allclose(ratio_flat, ratio_flat[0], atol=tol):
                v = ratio_flat[0]
                if v != 0:
                    return f'{v} * {label}'
        for sign in (1.0, -1.0):
            if np.allclose(W, sign * ones, atol=tol):
                return f'np.ones(({n}, {n}))' if sign > 0 else f'-np.ones(({n}, {n}))'
        asym = eye - flip
        for sign in (1.0, -1.0):
            if np.allclose(W, sign * asym, atol=tol):
                s = 'np.eye(n) - np.fliplr(np.eye(n))'
                return s if sign > 0 else f'-({s})'
    rows = ['[' + ', '.join(f'{v:.4g}' for v in row) + ']' for row in W]
    return 'np.array([' + ', '.join(rows) + '])'


# ── Block builders ─────────────────────────────────────────────────────────────

def gen_layers_block(layers) -> str:
    lines = ['    layers = [\n']
    for lyr in layers:
        lines.append(f'        {generate_layer_code(lyr)},\n')
    lines.append('    ]')
    return ''.join(lines)


def gen_connections_block(connections) -> str:
    lines = ['    connections = [\n']
    for conn in connections:
        src, tgt, W = conn.src, conn.tgt, conn.W
        W  = np.asarray(W, dtype=float)
        if W.ndim == 4:
            # Conv2d kernel: (n_filters, in_ch, kH, kW)
            wc = generate_weight_code(W, 0, 0)
        elif W.ndim == 3:
            n_filters, in_ch, kernel_size = W.shape
            wc = generate_weight_code(W, in_ch * kernel_size, n_filters)
        else:
            ns = W.shape[1] if W.ndim == 2 else 1
            nt = W.shape[0] if W.ndim == 2 else 1
            wc = generate_weight_code(W, ns, nt)
        lines.append(f"        ('{src}', '{tgt}', {wc}),\n")
    lines.append('    ]')
    return ''.join(lines)


# ── Content patching ───────────────────────────────────────────────────────────

def replace_block(content: str, varname: str, new_block: str) -> str:
    """Replace the `    varname = [...]` class-level block with *new_block*."""
    marker = f'    {varname} = ['
    start  = content.find(marker)
    if start == -1:
        insert_at = content.find('\n    def ')
        if insert_at == -1:
            raise ValueError(
                f"Cannot save: no '{marker}' block and no class method found.")
        return content[:insert_at + 1] + new_block + '\n\n' + content[insert_at + 1:]
    depth, i = 0, start
    while i < len(content):
        if content[i] == '[':
            depth += 1
        elif content[i] == ']':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    else:
        raise ValueError(f"Unmatched '[' for '{varname}' block")
    return content[:start] + new_block + content[end:]


def update_imports(content: str, layers) -> str:
    """Ensure the `from neurons import …` line lists every used layer type."""
    used = sorted({type(l).__name__ for l in layers} & set(_ALL_NEURON_TYPES))
    if not used:
        return content
    import_line = f'from neurons import {", ".join(used)}'
    new_content, n = re.subn(r'from neurons import [^\n]+', import_line, content)
    if n > 0:
        return new_content
    lines = content.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_at = i + 1
    lines.insert(insert_at, import_line + '\n')
    return ''.join(lines)


def serialize_brain(content: str, layers, connections) -> str:
    """Apply all modifications to *content* and return the updated source."""
    content = replace_block(content, 'layers',      gen_layers_block(layers))
    content = replace_block(content, 'connections', gen_connections_block(connections))
    content = update_imports(content, layers)
    return content


# ── Network JSON (data-driven brain format) ────────────────────────────────────

# Attributes serialized per sensor type (angle values stored in degrees for readability).
_SENSOR_ATTRS = {
    'GradientSensor':  ['n', 'angle_spread_deg', 'center_angle_deg', 'dist',
                        'color_channel', 'gradient', 'scale', 'tau_rise', 'tau_decay', 'activation'],
    'ColorSensor':     ['n', 'angle_spread_deg', 'center_angle_deg', 'dist',
                        'color_channel', 'scale', 'tau_rise', 'tau_decay', 'activation'],
    'CollisionSensor': ['n', 'angle_spread_deg', 'arc_angle_deg',
                        'radius', 'scale', 'bias', 'noise_std', 'tau_rise', 'tau_decay', 'activation'],
    'DistanceSensor':      ['n', 'angle_spread_deg', 'max_range', 'scale', 'tau_rise', 'tau_decay', 'activation'],
    'InteroceptiveSensor': ['gradient', 'scale', 'max_val', 'start_val', 'bias', 'tau_rise', 'tau_decay'],
    'WhiskerSensor':          ['length', 'mount_dist', 'mount_angle', 'n',
                              'scale', 'tau_rise', 'tau_decay', 'activation'],
    'ProprioceptiveSensor':   ['joint_id', 'use_velocity', 'scale', 'tau_rise', 'tau_decay', 'activation'],
    'SkyCompassSensor':       ['n', 'scale', 'phase', 'tau_rise', 'tau_decay', 'activation',
                              'noise_std', 'noise_tau', 'derivative'],
    'CameraSensor':           ['width', 'height', 'fov_deg', 'center_angle_deg',
                               'max_range', 'mode', 'lateralized', 'overlap',
                               'tau_rise', 'tau_decay', 'activation', 'differential', 'noise_std'],
    'GrayCameraSensor':       ['width', 'height', 'fov_deg', 'center_angle_deg',
                               'max_range', 'lateralized', 'overlap',
                               'tau_rise', 'tau_decay', 'activation', 'differential', 'noise_std'],
    'RGBCameraSensor':        ['width', 'height', 'fov_deg', 'center_angle_deg',
                               'max_range', 'lateralized', 'overlap',
                               'tau_rise', 'tau_decay', 'activation', 'differential', 'noise_std'],
}


def _sensor_to_dict(sensor) -> dict:
    t = type(sensor).__name__
    d = {'type': t, 'name': sensor.name}
    if getattr(sensor, 'robot_address', ''):
        d['robot_address'] = sensor.robot_address
    for attr in _SENSOR_ATTRS.get(t, []):
        if attr == 'angle_spread_deg':
            d[attr] = round(float(np.degrees(sensor.angle_spread)), 4)
        elif attr == 'center_angle_deg':
            d[attr] = round(float(np.degrees(sensor.center_angle)), 4)
        elif attr == 'arc_angle_deg':
            d[attr] = round(float(np.degrees(sensor.arc_angle)), 4)
        elif attr == 'fov_deg':
            d[attr] = round(float(np.degrees(sensor.fov)), 4)
        else:
            val = getattr(sensor, attr, None)
            # Always write dynamics attrs even when None (null in JSON) so the
            # freshness check finds the key and doesn't flag them as missing.
            _always_write = {'tau_rise', 'tau_decay', 'activation', 'differential',
                             'noise_std', 'noise_tau'}
            if val is not None or attr in _always_write:
                d[attr] = val
    if getattr(sensor, 'modulators', None):
        d['modulators'] = sensor.modulators
    for attr in ('neuromodulator_transmitter', 'neuromodulator_color'):
        val = getattr(sensor, attr, None)
        if val:
            d[attr] = val
    body_ids = getattr(sensor, 'body_ids', None) or ['root']
    if body_ids and body_ids != ['root']:
        d['body_ids'] = body_ids
    viz_layer = getattr(sensor, 'layer', None)
    if viz_layer is not None and viz_layer != 0:
        d['viz_layer'] = viz_layer
    viz_z = getattr(sensor, 'z', None)
    if viz_z:
        d['viz_z'] = viz_z
    return d


def _sensor_from_dict(d: dict):
    from sensors import SENSOR_REGISTRY
    t    = d['type']
    # Backward compat: old JSON used generic 'CameraSensor' with a 'mode' field.
    # Redirect to the appropriate typed subclass so in_ch is always correct.
    if t == 'CameraSensor':
        t = 'RGBCameraSensor' if d.get('mode') == 'rgb' else 'GrayCameraSensor'
    cls  = SENSOR_REGISTRY[t]
    kw   = {k: v for k, v in d.items() if k not in ('type', 'mode')}
    # Backward compat: old JSON used 'bonsai_subject' → now 'robot_address'
    if 'bonsai_subject' in kw:
        kw.setdefault('robot_address', kw.pop('bonsai_subject'))
    else:
        kw.pop('bonsai_subject', None)
    # Convert degree fields back to radians for the constructor
    for deg_key, rad_key in [('angle_spread_deg', 'angle_spread'),
                              ('center_angle_deg', 'center_angle'),
                              ('arc_angle_deg',    'arc_angle'),
                              ('fov_deg',          'fov')]:
        if deg_key in kw:
            kw[rad_key] = float(kw.pop(deg_key))
    # Backward compat: CollisionSensor 'noise' → 'noise_std'
    if t == 'CollisionSensor' and 'noise' in kw:
        kw.setdefault('noise_std', kw.pop('noise'))
    # Backward compat: old JSON used 'tau' for symmetric filtering
    if 'tau' in kw:
        v = kw.pop('tau')
        kw.setdefault('tau_rise', v)
        kw.setdefault('tau_decay', v)
    kw.pop('group', None)          # removed field — silently drop from old JSON
    kw.pop('osc_path', None)       # removed field — path is now embedded in robot_address
    viz_layer = kw.pop('viz_layer', None)
    viz_z     = kw.pop('viz_z', None)
    body_ids = kw.pop('body_ids', None)
    body_id  = kw.pop('body_id', 'root')   # backward compat with old JSON
    neuromod_transmitter = kw.pop('neuromodulator_transmitter', None)
    neuromod_color       = kw.pop('neuromodulator_color', None)
    sensor = cls(**kw)
    if body_ids is not None:
        sensor.body_ids = body_ids
    elif body_id != 'root':
        sensor.body_ids = [body_id]
    if viz_layer is not None:
        sensor.layer = viz_layer
    if viz_z is not None:
        sensor.z = viz_z
    if neuromod_transmitter:
        sensor.neuromodulator_transmitter = neuromod_transmitter
    if neuromod_color:
        sensor.neuromodulator_color = neuromod_color
    return sensor


def _layer_to_dict(layer) -> dict:
    t = type(layer).__name__
    d = {'type': t, 'name': layer.name}
    for attr in ('color', 'layer', 'group'):
        val = getattr(layer, attr, None)
        if val is not None:
            d[attr] = val
    for name, *_ in type(layer).param_defs():
        val = getattr(layer, name, None)
        if isinstance(val, np.ndarray):
            val = val.tolist()
        d[name] = val
    for attr in ('modulators', 'neuromodulator_transmitter', 'neuromodulator_color'):
        val = getattr(layer, attr, None)
        if val:
            d[attr] = val
    # Leaky2dLayer: persist shape metadata not covered by param_defs().
    from neurons import Leaky2dLayer as _L2d
    if isinstance(layer, _L2d):
        for attr in ('n', 'in_ch', 'frame_h', 'frame_w'):
            val = getattr(layer, attr, None)
            if val is not None:
                d[attr] = val
    if getattr(layer, 'muted', False):
        d['muted'] = True
    y_order = getattr(layer, 'y_order', None)
    if y_order and y_order != list(range(len(y_order))):
        d['y_order'] = y_order
    viz_row = getattr(layer, 'viz_row', None)
    if viz_row is not None:
        d['viz_row'] = viz_row
    viz_z = getattr(layer, 'z', None)
    if viz_z:
        d['viz_z'] = viz_z
    lateral_pair = getattr(layer, 'lateral_pair', None)
    if lateral_pair is not None:
        d['lateral_pair'] = lateral_pair
    return d


def _layer_from_dict(d: dict):
    t   = d['type']
    cls = LAYER_REGISTRY[t]
    kw  = {k: v for k, v in d.items() if k != 'type'}
    # Backward compat: MatsuokaLayer old JSON used tauM/tauA → tau_rise/tau_a
    if t == 'MatsuokaLayer':
        if 'tauM' in kw:
            kw.setdefault('tau_rise', kw.pop('tauM'))
        if 'tauA' in kw:
            kw.setdefault('tau_a', kw.pop('tauA'))
    # Backward compat: old single 'tau' field → asymmetric tau_rise/tau_decay
    if t in ('LeakyLayer', 'AdaptiveLayer', 'RingAttractorLayer') and 'tau' in kw:
        v = kw.pop('tau')
        kw.setdefault('tau_rise', v)
        kw.setdefault('tau_decay', v)
    # Backward compat: ConstantLayer 'noise' → 'noise_std'
    if t == 'ConstantLayer' and 'noise' in kw:
        kw.setdefault('noise_std', kw.pop('noise'))
    kw.pop('group', None)          # removed field — silently drop from old JSON
    y_order      = kw.pop('y_order', None)
    viz_row      = kw.pop('viz_row', None)
    viz_z        = kw.pop('viz_z', None)
    muted        = kw.pop('muted', False)
    lateral_pair = kw.pop('lateral_pair', None)
    layer   = cls(**kw)
    if y_order:
        layer.y_order = y_order
    if viz_row is not None:
        layer.viz_row = viz_row
    if muted:
        layer.muted = True
    if viz_z is not None:
        layer.z = viz_z
    if lateral_pair is not None:
        layer.lateral_pair = lateral_pair
    return layer


def _connection_to_dict(conn: Connection, params=None) -> dict:
    W = np.asarray(conn.W, dtype=float)
    d = {'src': conn.src, 'tgt': conn.tgt, 'W': W.tolist()}
    init_W = getattr(conn, 'init_W', None)
    if init_W is not None:
        d['init_W'] = np.asarray(init_W, dtype=float).tolist()
    if conn.learning is not None:
        d['learning'] = conn.learning
        d['lr'] = conn.lr
    if params is not None:
        d['params'] = params
    return d


def _w_from_params(params: dict, n_tgt: int, n_src: int, saved_W=None):
    """Reconstruct the initial weight matrix from saved WeightMatrixDialog params.

    Handles old numbering (pre rand-uniform/rand-normal) and new numbering
    transparently. Returns None if the pattern cannot be reconstructed
    (expression, or unknown index).
    """
    if not params:
        return None
    idx = params.get('type', -1)
    has_new_keys = 'rand_uniform' in params or 'rand_normal' in params
    if has_new_keys:
        # 0=Uniform 1=Cosine 2=Gaussian 3=MexHat 4=OneToOne
        # 5=RandUnif 6=RandNorm 7=Expression 8=Manual
        names = ['uniform', 'cosine', 'gaussian', 'mexican_hat', 'one_to_one',
                 'rand_uniform', 'rand_normal', 'expression', 'manual']
    else:
        # 0=Uniform 1=Cosine 2=Gaussian 3=MexHat 4=OneToOne 5=Expression 6=Manual
        names = ['uniform', 'cosine', 'gaussian', 'mexican_hat', 'one_to_one',
                 'expression', 'manual']
    pattern = names[idx] if 0 <= idx < len(names) else 'manual'

    nt, ns = n_tgt, n_src
    if pattern == 'uniform':
        amp = params.get('uniform', {}).get('amp', 1.0)
        return amp * np.ones((nt, ns))
    if pattern == 'cosine':
        p = params.get('cosine', {})
        th_s = 2 * np.pi * np.arange(ns) / max(ns, 1)
        ph_t = np.deg2rad(p.get('ph0', 0.0) + np.arange(nt) * p.get('step', 180.0))
        return p.get('amp', 1.0) * np.cos(th_s[None, :] + ph_t[:, None])
    if pattern == 'gaussian':
        p = params.get('gaussian', {})
        js   = np.arange(ns) / max(ns, 1)
        is_  = np.arange(nt) / max(nt, 1) + p.get('off', 0.0)
        dist = np.abs(is_[:, None] - js[None, :])
        dist = np.minimum(dist, 1 - dist)
        sig  = max(p.get('sig', 0.2), 1e-9)
        return p.get('amp', 1.0) * np.exp(-dist**2 / (2 * sig**2)) + p.get('base', 0.0)
    if pattern == 'mexican_hat':
        p = params.get('mexican_hat', {})
        js   = np.arange(ns) / max(ns, 1)
        is_  = np.arange(nt) / max(nt, 1)
        dist = np.abs(is_[:, None] - js[None, :])
        dist = np.minimum(dist, 1 - dist)
        se   = max(p.get('sige', 0.35), 1e-9)
        si   = max(p.get('sigi', 0.75), 1e-9)
        W = (p.get('exc', 2.0) * np.exp(-dist**2 / (2 * se**2))
             - p.get('inh', 1.0) * np.exp(-dist**2 / (2 * si**2)))
        if nt == ns:
            np.fill_diagonal(W, 0.0)
        return W
    if pattern == 'one_to_one':
        p = params.get('one_to_one', {})
        W = np.zeros((nt, ns))
        off = p.get('off', 0)
        for ii in range(nt):
            jj = int(round(ii * ns / max(nt, 1) + off)) % max(ns, 1)
            W[ii, jj] = p.get('amp', 1.0)
        return W
    if pattern == 'rand_uniform':
        amp = params.get('rand_uniform', {}).get('amp', 1.0)
        return amp * np.random.uniform(-1, 1, (nt, ns))
    if pattern == 'rand_normal':
        std = params.get('rand_normal', {}).get('std', 0.1)
        return std * np.random.randn(nt, ns)
    if pattern == 'manual':
        # Use the saved W as the best available reference for the original manual values.
        return np.asarray(saved_W, dtype=float).copy() if saved_W is not None else None
    return None  # expression or unknown


def _connection_from_dict(d: dict) -> Connection:
    raw_W    = d['W']
    raw_init = d.get('init_W')
    if raw_init is None:
        # Try to reconstruct init_W from the dialog params that were saved alongside
        # the connection.  For random patterns this generates a fresh sample; for
        # deterministic patterns it reproduces the exact original matrix.
        W_arr = np.array(raw_W, dtype=float)
        if W_arr.ndim == 2:
            raw_init_arr = _w_from_params(
                d.get('params'), W_arr.shape[0], W_arr.shape[1], saved_W=raw_W)
        else:
            # Conv (4-D) or 1-D: params type is always Manual for these; use saved W.
            raw_init_arr = W_arr.copy()
    else:
        raw_init_arr = np.array(raw_init, dtype=float)
    return Connection(
        src=d['src'],
        tgt=d['tgt'],
        W=np.array(raw_W, dtype=float),
        learning=d.get('learning'),
        lr=d.get('lr', 0.01),
        init_W=raw_init_arr,
    )


def serialize_network_json(sensors, layers, connections,
                           hidden_cols: set, disabled_cols: set,
                           col_labels: dict,
                           bodies=None, joints=None,
                           connection_params=None) -> dict:
    """Return a JSON-serialisable dict describing the complete circuit."""
    cp = connection_params or {}
    d = {
        'version':        1,
        'motor_layer':    'motor',
        'hidden_cols':    sorted(hidden_cols),
        'disabled_cols':  sorted(disabled_cols),
        'col_labels':     {str(k): v for k, v in col_labels.items()},
        'sensors':        [_sensor_to_dict(s) for s in sensors],
        'layers':         [_layer_to_dict(l) for l in layers
                           if not getattr(l, '_is_joint_motor', False)],
        'connections':    [_connection_to_dict(c, cp.get((c.src, c.tgt)))
                           for c in connections],
    }
    if bodies and len(bodies) > 1:
        d['bodies'] = [b.to_dict() for b in bodies]
    if joints:
        d['joints'] = [j.to_dict() for j in joints]
    return d


def load_network_json(data: dict):
    """Reconstruct circuit components from a serialised dict.

    Returns (sensors, layers, connections, hidden_cols, disabled_cols,
             col_labels, bodies, joints, connection_params).
    connection_params is a dict keyed by (src, tgt) containing the weight
    generation params saved by WeightMatrixDialog (pattern type + all options).
    """
    from rigid_body import RigidBody, Joint
    sensors     = [_sensor_from_dict(d) for d in data.get('sensors', [])]
    layers      = [_layer_from_dict(d)  for d in data.get('layers', [])]
    _seen_names = {}
    for obj in sensors + layers:
        n = getattr(obj, 'name', None)
        if n in _seen_names:
            import warnings
            warnings.warn(
                f"Network JSON contains duplicate name '{n}' "
                f"(first: {type(_seen_names[n]).__name__}, "
                f"second: {type(obj).__name__}). "
                "The second entry will shadow the first in connections and layout.",
                stacklevel=3,
            )
        else:
            _seen_names[n] = obj
    connections = [_connection_from_dict(d) for d in data.get('connections', [])]
    hidden      = set(data.get('hidden_cols', []))
    disabled    = set(data.get('disabled_cols', []))
    col_labels  = {int(k): v for k, v in data.get('col_labels', {}).items()}
    bodies      = [RigidBody.from_dict(b) for b in data.get('bodies', [])]
    joints      = [Joint.from_dict(j)     for j in data.get('joints', [])]
    connection_params = {
        (d['src'], d['tgt']): d['params']
        for d in data.get('connections', [])
        if 'params' in d
    }
    # Migrate old ring attractors: if they carried a legacy kernel, add it as a
    # self-connection (same layer as src and tgt) when none already exists.
    existing_self = {c.src for c in connections if c.src == c.tgt}
    for layer in layers:
        if isinstance(layer, RingAttractorLayer) and layer._legacy_W is not None:
            if layer.name not in existing_self:
                connections.append(Connection(layer.name, layer.name,
                                              layer._legacy_W.copy()))
            layer._legacy_W = None
    # Re-establish lateral_pair cross-links for lateralized Conv2dLayer / Leaky2dLayer pairs.
    # New JSONs already have lateral_pair set via _layer_from_dict(); guard with is None check.
    from neurons import Leaky2dLayer as _L2dR
    _lat_candidates = {l.name: l for l in layers
                       if isinstance(l, (Conv2dLayer, _L2dR)) and getattr(l, 'lateralized', False)}
    for name, layer in _lat_candidates.items():
        if layer.lateral_pair is not None:
            continue  # already restored from JSON
        if name.endswith('_L'):
            partner = name[:-2] + '_R'
        elif name.endswith('_R'):
            partner = name[:-2] + '_L'
        else:
            continue
        if partner in _lat_candidates:
            layer.lateral_pair = partner
    lat_conv = {n: l for n, l in _lat_candidates.items() if isinstance(l, Conv2dLayer)}
    # Sync operational params from _L to _R so both sides are always in step.
    _LATERAL_SYNC = ('n_filters', 'kernel_size', 'stride', 'padding', 'pool',
                     'activation', 'tau_rise', 'tau_decay', 'bias', 'scale')
    for name, layer in lat_conv.items():
        if name.endswith('_L'):
            partner_name = layer.lateral_pair
            if partner_name and partner_name in lat_conv:
                partner = lat_conv[partner_name]
                for attr in _LATERAL_SYNC:
                    if hasattr(layer, attr):
                        setattr(partner, attr, getattr(layer, attr))
    # Auto-create the mirror camera→conv connection for lateralized pairs that only
    # have the _L side wired (e.g. JSONs saved before the auto-wiring feature was added).
    import copy as _copy
    conn_set = {(c.src, c.tgt) for c in connections}
    for name, layer in lat_conv.items():
        if not name.endswith('_L'):
            continue
        partner_name = layer.lateral_pair
        if not partner_name:
            continue
        for conn in list(connections):
            if conn.tgt != name:
                continue
            mirror_src = (conn.src[:-2] + '_R') if conn.src.endswith('_L') else \
                         (conn.src[:-2] + '_L') if conn.src.endswith('_R') else None
            if mirror_src and (mirror_src, partner_name) not in conn_set:
                connections.append(Connection(
                    mirror_src, partner_name,
                    _copy.deepcopy(conn.W),
                    init_W=_copy.deepcopy(conn.init_W),
                ))
                conn_set.add((mirror_src, partner_name))
    return sensors, layers, connections, hidden, disabled, col_labels, bodies, joints, connection_params


def save_network_file(path: str, sensors, layers, connections,
                      hidden_cols: set, disabled_cols: set,
                      col_labels: dict = None,
                      bodies=None, joints=None, connection_params=None):
    """Write the circuit to a JSON file at *path*."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    data = serialize_network_json(sensors, layers, connections,
                                  hidden_cols, disabled_cols, col_labels or {},
                                  bodies, joints, connection_params)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def check_network_freshness(data: dict, sensors, layers) -> list:
    """Compare loaded sensors and layers against the current simulator param_defs.

    Returns a list of dicts, one per component that has new params not in the JSON:
        {'kind': 'layer'|'sensor', 'name': str, 'type': str, 'missing': [str]}

    'missing' = params present in the current param_defs that are absent from the
    saved JSON — i.e. new params added to the simulator since the file was last saved.
    An empty list means everything is up-to-date.
    """
    from neurons import DynamicsBase

    # Keys that are structural/display or saved only when non-empty —
    # never report these as "missing" regardless of which side they appear on.
    _SKIP = {
        'type', 'name', 'color', 'layer', 'group', 'modulators',
        'neuromodulator_transmitter', 'neuromodulator_color',
        'body_ids', 'robot_address', 'viz_layer', 'viz_z', 'muted',
        'y_order', 'viz_row', 'mode', 'lateral_pair',
    }

    layer_data  = {d['name']: d for d in data.get('layers',  [])}
    sensor_data = {d['name']: d for d in data.get('sensors', [])}

    issues = []

    for layer in layers:
        if getattr(layer, '_is_joint_motor', False):
            continue
        t   = type(layer).__name__
        cls = LAYER_REGISTRY.get(t)
        if cls is None or not hasattr(cls, 'param_defs'):
            continue
        expected = list(cls.param_defs())
        if issubclass(cls, DynamicsBase):
            existing = {p[0] for p in expected}
            expected += [p for p in DynamicsBase._dynamics_param_defs()
                         if p[0] not in existing]
        expected_names = {p[0] for p in expected} - _SKIP
        saved_keys     = set(layer_data.get(layer.name, {}).keys()) - _SKIP
        missing        = sorted(expected_names - saved_keys)
        if missing:
            issues.append({'kind': 'layer', 'name': layer.name,
                           'type': t, 'missing': missing})

    for sensor in sensors:
        t = type(sensor).__name__
        raw_keys = set(sensor_data.get(sensor.name, {}).keys())
        # Normalise _deg variants: angle_spread_deg counts as angle_spread too
        saved_keys = raw_keys - _SKIP
        for k in list(raw_keys):
            if k.endswith('_deg'):
                saved_keys.add(k[:-4])

        from sensors import BaseSensor, SENSOR_REGISTRY
        cls = SENSOR_REGISTRY.get(t)
        expected_names = set()
        if cls is not None and hasattr(cls, 'param_defs'):
            expected_names.update(p[0] for p in cls.param_defs())
        expected_names.update(p[0] for p in BaseSensor._sensor_base_param_defs())
        expected_names -= _SKIP

        missing = sorted(expected_names - saved_keys)
        if missing:
            issues.append({'kind': 'sensor', 'name': sensor.name,
                           'type': t, 'missing': missing})

    return issues


def load_network_file(path: str):
    """Read a JSON file and return (sensors, layers, connections, hidden, disabled,
    bodies, joints, connection_params)."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return load_network_json(data)
