"""
circuit_editor.py — one transaction for every circuit edit. No Qt.

Every user edit of a circuit (network editor, side view, joint dialog, …)
runs inside a transaction:

    with CircuitEditor(circuit, brain, meta, brain_mgr).edit():
        ...mutate circuit...

or, for visualizer methods, the @edit_transaction decorator. A transaction:

  - takes the undo snapshot BEFORE anything changes (nested transactions
    join the outermost one), and records it only if something actually
    changed — a cancelled dialog leaves no undo step;
  - afterwards re-syncs the brain with the circuit (lists, one attribute
    per layer, stale attributes removed, placeholder sensor readings) and
    bumps brain._topology_version so step_network rebuilds its caches.

The undo history lives on the CircuitModel (circuit.history), so it survives
closing the editor window, can't leak into another agent's circuit, and is
cleared whenever a brain or network file replaces the circuit
(clear_history).

`meta` is the editor-side state that is saved with a network but isn't part
of CircuitModel: an object with _hidden_containers, _disabled_containers,
_container_labels, _container_notes and _weight_params (the visualizer
window). Undo restores it in place, so references shared with the app stay
valid.
"""

import functools
import json
from contextlib import contextmanager

import numpy as np

from lateral import mirror_name, is_lateral_half, partner_layer, half_names, side_of

MAX_UNDO = 50

_META_ATTRS = ('_hidden_containers', '_disabled_containers', '_container_labels',
               '_container_notes', '_weight_params')


class EditHistory:
    """Per-circuit undo stack plus the state of the transaction in progress."""
    __slots__ = ('undo_stack', 'depth', 'pending', 'layer_names')

    def __init__(self):
        self.undo_stack  = []     # JSON snapshot strings, oldest first
        self.depth       = 0      # nesting level of the open transaction
        self.pending     = None   # snapshot taken when the outermost transaction began
        self.layer_names = set()  # layer names when it began (for stale brain attributes)


def clear_history(circuit):
    """Forget the undo history — call when a brain/network replaces the circuit."""
    if circuit is not None:
        circuit.history = None


class CircuitEditor:
    def __init__(self, circuit, brain=None, meta=None, brain_mgr=None):
        self.circuit   = circuit
        self.brain     = brain
        self.meta      = meta
        self.brain_mgr = brain_mgr

    @property
    def history(self):
        if self.circuit.history is None:
            self.circuit.history = EditHistory()
        return self.circuit.history

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def _meta(self, attr, default):
        value = getattr(self.meta, attr, None) if self.meta is not None else None
        return value if value is not None else default

    def snapshot(self) -> str:
        """The whole circuit (incl. bodies, joints, notes and editor metadata) as JSON."""
        from brain_serializer import serialize_network_json
        from sim_constants import _NumpyEncoder
        c = self.circuit
        data = serialize_network_json(
            c.sensors, c.layers, c.connections,
            self._meta('_hidden_containers', set()),
            self._meta('_disabled_containers', set()),
            self._meta('_container_labels', {}),
            bodies=c.bodies, joints=c.joints,
            connection_params=self._meta('_weight_params', {}),
            notes=c.notes,
            container_notes=self._meta('_container_notes', {}),
        )
        data.pop('saved_with_app_version', None)
        return json.dumps(data, cls=_NumpyEncoder, sort_keys=True)

    # ── Transactions ──────────────────────────────────────────────────────────

    def begin(self):
        h = self.history
        if h.depth == 0:
            h.pending = self.snapshot()
            h.layer_names = {l.name for l in self.circuit.layers}
        h.depth += 1

    def commit(self) -> bool:
        """Close the transaction. Returns True if an undo step was recorded."""
        h = self.history
        if h.depth == 0:
            return False
        h.depth -= 1
        if h.depth > 0:
            return False
        before, h.pending = h.pending, None
        changed = before is not None and before != self.snapshot()
        if changed:
            h.undo_stack.append(before)
            del h.undo_stack[:-MAX_UNDO]
        self.sync_brain(h.layer_names)
        return changed

    @contextmanager
    def edit(self):
        self.begin()
        try:
            yield self
        finally:
            self.commit()

    # ── Undo ──────────────────────────────────────────────────────────────────

    def can_undo(self) -> bool:
        h = self.circuit.history
        return bool(h and h.undo_stack)

    def undo(self) -> bool:
        """Restore the circuit to before the last recorded edit."""
        if not self.can_undo():
            return False
        from brain_serializer import load_network_json
        h = self.history
        old_names = {l.name for l in self.circuit.layers}
        data = json.loads(h.undo_stack.pop())
        (sensors, layers, connections, hidden, disabled, labels,
         bodies, joints, conn_params, notes, container_notes) = load_network_json(data)
        c = self.circuit
        c.sensors, c.layers, c.connections, c.notes = sensors, layers, connections, notes
        # Snapshots only store bodies beyond the root one; empty means "root only".
        c.bodies = bodies if bodies else c.bodies[:1]
        c.joints = joints or []
        if self.meta is not None:
            for attr, value in zip(_META_ATTRS, (hidden, disabled, labels, container_notes, conn_params)):
                _replace_contents(self.meta, attr, value)
        if self.brain_mgr is not None and self.brain_mgr.circuit is c:
            # Joint actuator layers aren't saved; recreate them for the restored joints.
            self.brain_mgr.rebuild_joint_motor_layers()
            self.brain_mgr.resolve_joint_sensor_refs()
        self.sync_brain(old_names)
        return True

    # ── Brain sync ────────────────────────────────────────────────────────────

    def sync_brain(self, previous_layer_names=()):
        """Point the brain at the circuit again after an edit."""
        brain = self.brain
        if brain is None:
            return
        c = self.circuit
        brain.layers      = c.layers
        brain.sensors     = c.sensors
        brain.connections = c.connections
        current = {l.name for l in c.layers}
        for name in set(previous_layer_names) - current:
            if hasattr(brain.__dict__.get(name), 'step'):   # only layer objects we set
                delattr(brain, name)
        for layer in c.layers:
            setattr(brain, layer.name, layer)
        for sensor in c.sensors:
            if sensor.name not in brain.__dict__:
                setattr(brain, sensor.name, np.zeros(max(1, sensor.n_total), dtype=np.float32))
        brain._topology_version = getattr(brain, '_topology_version', 0) + 1


# ── Circuit rules: mirror connections and renames (rules/network_viz.md) ──
# (Lateral-pair basics — partners, halves, names — live in lateral.py.)

def _layer(circuit, name):
    return next((l for l in circuit.layers if l.name == name), None)


def find_mirror(circuit, src, tgt):
    """The (src, tgt) of the existing mirror of connection src → tgt, or None.
    The mirror runs from the other half of a lateral source to the target's
    lateral partner — or to the same target when it has none (e.g. both halves
    of a lateralized camera feeding one Conv2dLayer)."""
    if not is_lateral_half(circuit, src):
        return None
    m_src = mirror_name(src)
    tgt_layer = _layer(circuit, tgt)
    m_tgt = getattr(tgt_layer, 'lateral_pair', None) or tgt
    if any(c.src == m_src and c.tgt == m_tgt for c in circuit.connections):
        return m_src, m_tgt
    return None


def apply_renames(circuit, meta, renames):
    """Rename connection endpoints and their saved weight settings
    (meta._weight_params keys) according to {old_name: new_name}."""
    from dataclasses import replace
    if not renames:
        return
    circuit.connections = [
        replace(c, src=renames.get(c.src, c.src), tgt=renames.get(c.tgt, c.tgt))
        if (c.src in renames or c.tgt in renames) else c
        for c in circuit.connections
    ]
    params = getattr(meta, '_weight_params', None) if meta is not None else None
    if params:
        rekeyed = {(renames.get(s, s), renames.get(t, t)): v for (s, t), v in params.items()}
        params.clear()
        params.update(rekeyed)


def rename_layer(circuit, layer, new_name, meta=None):
    """Rename *layer*; a lateral pair is renamed as a unit (foo_L / foo_R) with
    both lateral_pair links, connections and weight settings updated.
    Returns {old_name: new_name} for everything renamed."""
    old_name = layer.name
    if not new_name or new_name == old_name:
        return {}
    renames = {}
    partner = partner_layer(circuit.layers, layer)
    side = side_of(old_name)
    if partner is not None and side:
        base = new_name[:-2] if side_of(new_name) == side else new_name
        new_name, new_partner = half_names(base) if side == 'L' else half_names(base)[::-1]
        renames[partner.name] = new_partner
        partner.name = new_partner
        partner.lateral_pair = new_name
        layer.lateral_pair = new_partner
    renames[old_name] = new_name
    layer.name = new_name
    apply_renames(circuit, meta, renames)
    return renames


def unpair_layer(circuit, layer, meta=None):
    """Turn a lateral pair back into one layer (lateralized switched off):
    *layer* is kept and takes the base name (foo_L → foo, unless that name is
    taken); its partner is removed with its connections and weight settings.
    Returns (removed partner name or None, {old_name: new_name} renames)."""
    partner = partner_layer(circuit.layers, layer)
    layer.lateral_pair = None
    removed = None
    if partner is not None:
        removed = partner.name
        partner.lateral_pair = None
        circuit.layers.remove(partner)   # in place: brain re-synced by the transaction
        circuit.connections = [c for c in circuit.connections
                               if removed not in (c.src, c.tgt)]
        params = getattr(meta, '_weight_params', None) if meta is not None else None
        if params:
            for key in [k for k in params if removed in k]:
                del params[key]
    renames = {}
    if side_of(layer.name):
        base = layer.name[:-2]
        if not any(l.name == base for l in circuit.layers) and \
                not any(s.name == base for s in circuit.sensors):
            renames[layer.name] = base
            layer.name = base
            apply_renames(circuit, meta, renames)
    return removed, renames


def _replace_contents(owner, attr, value):
    """Update a set/dict attribute in place (keeps shared references valid)."""
    current = getattr(owner, attr, None)
    if isinstance(current, (set, dict)) and type(current) is type(value):
        current.clear()
        current.update(value)
    else:
        setattr(owner, attr, value)


def edit_transaction(method):
    """Run a visualizer method as one circuit edit (see module docstring).
    The owner must provide self._editor -> CircuitEditor."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        editor = self._editor
        editor.begin()
        try:
            return method(self, *args, **kwargs)
        finally:
            editor.commit()
            on_commit = getattr(self, '_on_edit_committed', None)
            if on_commit is not None:
                on_commit()
    return wrapper
