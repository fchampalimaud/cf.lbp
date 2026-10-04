"""
lateral.py — lateral (L/R) pairs in one place. No Qt.

A lateral pair is one layer or sensor split spatially into a left and a right
half. Two kinds exist:

  layer pairs   two layer objects named <base>_L / <base>_R, linked both ways
                through their `lateral_pair` attribute (e.g. a Conv2dLayer
                pair fed by the two halves of a camera)
  sensor halves one sensor object <base> whose output is split into <base>_L
                and <base>_R: a camera with lateralized=True, or a sensor
                mounted on a mirrored body pair (two body_ids)

Everything that needs to know about pairs — find a partner, tell whether a
name is a half, find a half's parent sensor, build or strip half names —
asks here instead of parsing `_L` / `_R` itself.
"""

SIDES = ('L', 'R')


def side_of(name):
    """'L' or 'R' if *name* ends in _L / _R, else None."""
    if name.endswith('_L'):
        return 'L'
    if name.endswith('_R'):
        return 'R'
    return None


def base_name(name):
    """*name* without its _L / _R suffix (unchanged if it has none)."""
    return name[:-2] if side_of(name) else name


def mirror_name(name):
    """Swap the _L / _R suffix; None if the name has neither."""
    side = side_of(name)
    if side is None:
        return None
    return name[:-2] + ('_R' if side == 'L' else '_L')


def half_names(base):
    """(<base>_L, <base>_R)."""
    return f'{base}_L', f'{base}_R'


def partner_layer(layers, layer):
    """The other half of a layer pair (via lateral_pair), or None."""
    pair = getattr(layer, 'lateral_pair', None)
    if not pair:
        return None
    return next((l for l in layers if l.name == pair), None)


def parent_sensor(sensors, name):
    """The sensor that *name* is a half of (<base>_L / <base>_R → sensor <base>),
    or None. Does not check that the sensor is actually lateralized."""
    if side_of(name) is None:
        return None
    base = name[:-2]
    return next((s for s in sensors if s.name == base), None)


def is_camera_half(sensors, name):
    """True for <base>_L / <base>_R of a camera with lateralized=True."""
    parent = parent_sensor(sensors, name)
    return parent is not None and bool(getattr(parent, 'lateralized', False))


def is_body_pair_half(sensors, name):
    """True for <base>_L / <base>_R of a sensor mounted on two bodies (a mirrored
    joint pair). This is the runtime check; mirror_group validation needs the
    circuit's bodies — see is_lateral_half."""
    parent = parent_sensor(sensors, name)
    return (parent is not None and not getattr(parent, 'lateralized', False)
            and len(getattr(parent, 'body_ids', [])) == 2)


def is_lateral_half(circuit, name):
    """True if *name* is one half of a lateral pair: a layer whose lateral_pair
    is its mirror name, or the _L/_R half of a lateralized sensor (checked
    against the circuit's bodies for mirrored joint pairs)."""
    other = mirror_name(name)
    if other is None:
        return False
    lyr = next((l for l in circuit.layers if l.name == name), None)
    if lyr is not None:
        return getattr(lyr, 'lateral_pair', None) == other
    sensor = parent_sensor(circuit.sensors, name)
    return sensor is not None and sensor.is_lateralized(circuit)
