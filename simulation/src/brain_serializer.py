"""
Brain serializer: code-generation and file-patching utilities.

All functions are pure (no Qt, no GUI state). They can be called from
network_viz, tests, or any future tool that needs to write brain .py files.
"""

import json
import os
import re
import numpy as np
from neurons import RingAttractorLayer, LAYER_REGISTRY
from lateral import mirror_name, side_of
from circuit_model import Connection, Note
from app_version import get_app_version

_ALL_NEURON_TYPES = sorted(LAYER_REGISTRY.keys())


# ── Code generation ────────────────────────────────────────────────────────────

def generate_layer_code(lyr) -> str:
    """Return a constructor expression string for *lyr* (one line, no trailing comma)."""
    return f'{type(lyr).__name__}({", ".join(lyr.init_code_parts())})'


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

def _sensor_to_dict(sensor) -> dict:
    from sensors import SENSOR_REGISTRY, BaseSensor
    t = type(sensor).__name__
    d = {'type': t, 'name': sensor.name}
    if getattr(sensor, 'robot_address', ''):
        d['robot_address'] = sensor.robot_address
    cls = SENSOR_REGISTRY.get(t)
    all_params = list(cls.param_defs()) if cls and hasattr(cls, 'param_defs') else []
    # Append shared base params (noise_std, tau_rise, activation, etc.) not already
    # covered by the sensor-specific param_defs — mirrors what the UI dialogs do.
    existing = {p[0] for p in all_params}
    all_params += [p for p in BaseSensor._sensor_base_param_defs()
                   if p[0] not in existing]
    _always_write = {'tau_rise', 'tau_decay', 'activation', 'output_mode',
                     'noise_std', 'noise_tau'}
    for param_name, *_ in all_params:
        if param_name in BaseSensor._ANGLE_ATTRS:
            raw = getattr(sensor, param_name, None)
            if raw is not None:
                d[f'{param_name}_deg'] = round(float(np.degrees(raw)), 4)
        else:
            val = getattr(sensor, param_name, None)
            if val is not None or param_name in _always_write:
                d[param_name] = val
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
    for deg_key, rad_key in [('angle_spread_deg',    'angle_spread'),
                              ('center_angle_deg',   'center_angle'),
                              ('arc_angle_deg',      'arc_angle'),
                              ('fov_deg',            'fov'),
                              ('vertical_angle_deg', 'vertical_angle')]:
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
    # Backward compat: old JSON used a boolean 'differential' field (rate-of-change
    # only); output_mode generalizes it to a none/derivative/integral choice.
    if 'differential' in kw:
        kw.setdefault('output_mode', 'derivative' if kw.pop('differential') else 'none')
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
    # Everything the edit dialog shows (param_defs() plus the shared dynamics
    # params, e.g. output_mode) is a live, settable attribute, so it must be
    # persisted — otherwise it silently reverts on reload.
    for name, *_ in type(layer).all_param_defs():
        val = getattr(layer, name, None)
        if isinstance(val, np.ndarray):
            val = val.tolist()
        d[name] = val
    # reward_modulator is no longer in LearningLayerBase-family param_defs()
    # (superseded by a 'drives_plasticity' row in 'modulators'), but the
    # constructor/attribute still exists for back-compat — persist it
    # explicitly so a layer with a custom (non-default) value doesn't
    # silently lose it on re-save just because it's no longer GUI-editable.
    for attr in ('modulators', 'neuromodulator_transmitter', 'neuromodulator_color', 'reward_modulator'):
        val = getattr(layer, attr, None)
        if val:
            d[attr] = val
    # Runtime state a class declares worth saving beyond param_defs() (e.g. image
    # shape metadata) — {json key: attribute}; passed back to the constructor on load.
    for key, attr in type(layer).saved_state.items():
        val = getattr(layer, attr, None)
        if val is not None:
            d[key] = val
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
    viz_span = getattr(layer, 'span', 1)
    if viz_span and viz_span != 1:
        d['viz_span'] = viz_span
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
    # Backward compat: Reichardt2dLayer old JSON used a fixed n_dirs enum
    # (1/2/4/8) plus a same-named 'n' for the live (possibly pool='none'-
    # expanded) buffer length. n_dirs -> n_directions; the old 'n' becomes
    # 'flat_n' since 'n' is now the n_directions hyperparameter itself.
    if t == 'Reichardt2dLayer' and 'n_dirs' in kw:
        old_flat_n = kw.pop('n', None)
        kw['n_directions'] = kw.pop('n_dirs')
        if old_flat_n is not None:
            kw['flat_n'] = old_flat_n
    # Backward compat: LeakyLayer/ProductLayer/Leaky2dLayer used to expose their
    # own boolean 'derivative' (x-vs-u novelty trick); output_mode now covers
    # the same rate-of-change idea generically (plus 'integral') for every
    # DynamicsBase layer. Other layer types never exposed 'differential' before
    # (see _dynamics_param_defs), so no compat shim is needed for them.
    if t in ('LeakyLayer', 'ProductLayer', 'Leaky2dLayer') and 'derivative' in kw:
        kw.setdefault('output_mode', 'derivative' if kw.pop('derivative') else 'none')
    kw.pop('group', None)          # removed field — silently drop from old JSON
    y_order      = kw.pop('y_order', None)
    viz_row      = kw.pop('viz_row', None)
    viz_z        = kw.pop('viz_z', None)
    viz_span     = kw.pop('viz_span', None)
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
    if viz_span is not None:
        layer.span = viz_span
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
    from sim_constants import WEIGHT_PATTERNS
    idx = params.get('type', -1)
    has_new_keys = 'rand_uniform' in params or 'rand_normal' in params
    if has_new_keys:
        names = WEIGHT_PATTERNS
    else:
        # Old JSON (pre rand_uniform/rand_normal): 0=Uniform…4=OneToOne 5=Expression 6=Manual
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
                           hidden_containers: set, disabled_containers: set,
                           container_labels: dict,
                           bodies=None, joints=None,
                           connection_params=None, notes=None,
                           container_notes: dict = None) -> dict:
    """Return a JSON-serialisable dict describing the complete circuit."""
    cp = connection_params or {}
    d = {
        'version':                 1,
        'saved_with_app_version':  get_app_version(),
        'motor_layer':      'motor',
        'hidden_cols':      sorted(hidden_containers),
        'disabled_cols':    sorted(disabled_containers),
        # Keyed by container identity ('|'-joined occupant name set), not
        # position — see load_network_json's migration for the old,
        # position-keyed format.
        'container_labels': dict(container_labels),
        'container_notes':  dict(container_notes or {}),
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
    if notes:
        d['notes'] = [n.to_dict() for n in notes]
    return d


def load_network_json(data: dict):
    """Reconstruct circuit components from a serialised dict.

    Returns (sensors, layers, connections, hidden_cols, disabled_cols,
             container_labels, bodies, joints, connection_params, notes,
             container_notes).
    connection_params is a dict keyed by (src, tgt) containing the weight
    generation params saved by WeightMatrixDialog (pattern type + all options).
    container_labels and container_notes are keyed by container identity
    (the sorted, '|'-joined set of occupant names), not position — see the
    migration below for files saved before this change, which used position
    (depth_val) keys.
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

    if 'container_labels' in data:
        container_labels = dict(data['container_labels'])
    else:
        # Old format: keys were raw depth_val positions. Migrate by finding
        # which objects currently sit at that position, now that sensors/
        # layers above are already reconstructed with their .layer values.
        raw_labels = data.get('col_labels', {})
        container_labels = {}
        for k, v in raw_labels.items():
            try:
                old_pos = int(k)
            except ValueError:
                continue
            names = sorted(o.name for o in (sensors + layers)
                            if getattr(o, 'layer', None) == old_pos)
            if names:
                container_labels['|'.join(names)] = v

    container_notes = dict(data.get('container_notes', {}))

    bodies      = [RigidBody.from_dict(b) for b in data.get('bodies', [])]
    joints      = [Joint.from_dict(j)     for j in data.get('joints', [])]
    notes       = [Note.from_dict(n)      for n in data.get('notes', [])]
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
    # Re-establish lateral_pair cross-links for lateralized pairs (layer classes
    # with supports_lateral). New JSONs already have lateral_pair set via
    # _layer_from_dict(); guard with is None check.
    _lat_candidates = {l.name: l for l in layers
                       if l.supports_lateral and getattr(l, 'lateralized', False)}
    for name, layer in _lat_candidates.items():
        if layer.lateral_pair is not None:
            continue  # already restored from JSON
        partner = mirror_name(name)
        if partner in _lat_candidates:
            layer.lateral_pair = partner
    # Sync operational params from _L to _R so both sides are always in step —
    # the same params the edit dialog keeps in sync (lateral_sync_params).
    for name, layer in _lat_candidates.items():
        partner = _lat_candidates.get(layer.lateral_pair)
        if side_of(name) == 'L' and partner is not None and type(partner) is type(layer):
            for attr in type(layer).lateral_sync_params():
                if hasattr(layer, attr):
                    setattr(partner, attr, getattr(layer, attr))
    # Auto-create the mirror camera→conv connection for lateralized pairs that only
    # have the _L side wired (e.g. JSONs saved before the auto-wiring feature was added).
    import copy as _copy
    conn_set = {(c.src, c.tgt) for c in connections}
    for name, layer in _lat_candidates.items():
        if side_of(name) != 'L':
            continue
        partner_name = layer.lateral_pair
        if not partner_name:
            continue
        for conn in list(connections):
            if conn.tgt != name:
                continue
            mirror_src = mirror_name(conn.src)
            if mirror_src and (mirror_src, partner_name) not in conn_set:
                connections.append(Connection(
                    mirror_src, partner_name,
                    _copy.deepcopy(conn.W),
                    init_W=_copy.deepcopy(conn.init_W),
                ))
                conn_set.add((mirror_src, partner_name))
    return (sensors, layers, connections, hidden, disabled, container_labels,
            bodies, joints, connection_params, notes, container_notes)


def save_network_file(path: str, sensors, layers, connections,
                      hidden_cols: set, disabled_cols: set,
                      container_labels: dict = None,
                      bodies=None, joints=None, connection_params=None, notes=None,
                      container_notes: dict = None):
    """Write the circuit to a JSON file at *path*."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    data = serialize_network_json(sensors, layers, connections,
                                  hidden_cols, disabled_cols, container_labels or {},
                                  bodies, joints, connection_params, notes,
                                  container_notes=container_notes or {})
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
        expected_names = {p[0] for p in expected} - _SKIP
        saved_keys     = set(layer_data.get(layer.name, {}).keys()) - _SKIP
        missing        = sorted(expected_names - saved_keys)
        if missing:
            issues.append({'kind': 'layer', 'name': layer.name,
                           'type': t, 'missing': missing})

    def _normalised(keys):
        # angle_spread_deg counts as angle_spread
        return {k[:-4] if k.endswith('_deg') else k for k in keys} - _SKIP

    for sensor in sensors:
        t = type(sensor).__name__
        saved_keys = _normalised(sensor_data.get(sensor.name, {}).keys())
        # Expect exactly what saving this sensor now would write. Params whose
        # value is None (e.g. GradientSensor.gradient = "all labels") are left
        # out of the file by design, so they must not count as missing —
        # otherwise the prompt can never be satisfied by re-saving.
        expected_names = _normalised(_sensor_to_dict(sensor).keys())

        missing = sorted(expected_names - saved_keys)
        if missing:
            issues.append({'kind': 'sensor', 'name': sensor.name,
                           'type': t, 'missing': missing})

    return issues


def load_network_file(path: str):
    """Read a JSON file and return (sensors, layers, connections, hidden, disabled,
    container_labels, bodies, joints, connection_params, notes, container_notes)."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return load_network_json(data)
