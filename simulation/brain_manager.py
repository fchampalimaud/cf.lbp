"""
brain_manager.py — discovers, loads, and wires brain plugin classes.

Owns: brain discovery/import, circuit topology management (joint-motor layer
synthesis and proprioceptive sensor wiring), network-file loading into the
circuit, and new-brain file scaffolding.  No Qt, no display logic.
"""

import glob
import importlib
import importlib.util
import json
import math
import os
import sys
import traceback
import uuid

from neurons import SumLayer, MotorLayer  # SumLayer kept for isinstance checks in migration
from rigid_body import RigidBody, Joint
from sensors import ProprioceptiveSensor


# ── Brain file scaffold ───────────────────────────────────────────────────────

_BRAIN_TEMPLATE = '''\
from brain_base import BaseBrain, Param
from sensors import GradientSensor, ColorSensor
from neurons import LeakyLayer, MatsuokaLayer, ConstantLayer, AdaptiveLayer, SumLayer
import numpy as np


class {class_name}(BaseBrain):
    sensors = []

    layers = []

    connections = []

    speed = Param(50.0, 0, 100, step=1.0, desc="Base speed")

    def setup(self):
        pass

    def loop(self, dt):
        mL = self.speed
        mR = self.speed
        return mL, mR

    def plots(self):
        return []
'''


class BrainManager:
    """
    Owns brain discovery, instantiation, and circuit topology management.

    Parameters
    ----------
    circuit  : CircuitModel  — shared circuit state (mutated in-place)
    sim_cfg  : SimConfig     — read for body_radius default
    """

    def __init__(self, circuit, sim_cfg):
        self.circuit = circuit
        self.sim_cfg = sim_cfg

    # ── Discovery ─────────────────────────────────────────────────────────────

    def discover_brains(self):
        valid_brains = []
        for f in glob.glob("brains/*.py"):
            module_name = os.path.splitext(os.path.basename(f))[0]
            try:
                spec   = importlib.util.spec_from_file_location(module_name, f)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and attr.__module__ == module_name:
                        if all(hasattr(attr, m) for m in ['setup', 'loop']):
                            valid_brains.append(module_name)
                            break
            except Exception:
                print(f"[discover] Skipping {module_name}:")
                traceback.print_exc()
        return sorted(valid_brains)

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_brain_logic(self, name):
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        module = importlib.import_module(name)

        brain_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr.__module__ == name:
                brain_class = attr
                break

        if not brain_class:
            return None, {}

        brain     = brain_class()
        safe_name = brain.__class__.__name__
        blueprint = brain.get_param_metadata()

        fname       = f"configs/brain_{safe_name}.json"
        loaded_json = {}
        if os.path.exists(fname):
            try:
                with open(fname, "r") as f:
                    loaded_json = json.load(f)
            except Exception:
                pass

        saved_params = loaded_json.get("params", loaded_json)
        for k, p_obj in blueprint.items():
            setattr(brain, k, saved_params.get(k, p_obj.default))

        return brain, loaded_json

    # ── Circuit topology ──────────────────────────────────────────────────────

    def add_joint(self, parent_id, layer_name, radius, dist,
                  attach_angle_deg, amin, amax, mirrored):
        """Create body/joint/motor-layer objects and append them to the circuit.

        Returns the new SumLayer (flagged _is_joint_motor=True).
        """
        next_depth = (max(
            (getattr(l, 'layer', None) or 0)
            for l in self.circuit.layers if l.n is not None
        ) + 1) if self.circuit.layers else 1

        if mirrored:
            id_L = str(uuid.uuid4())[:8]
            id_R = str(uuid.uuid4())[:8]
            body_L = RigidBody(id_L, f'{layer_name}_L', radius, mirror_group=layer_name)
            body_R = RigidBody(id_R, f'{layer_name}_R', radius, mirror_group=layer_name)
            joint_L = Joint(parent_id=parent_id, child_id=id_L,
                            attach_dist=dist,
                            attach_angle=math.radians(attach_angle_deg),
                            angle_min=amin, angle_max=amax,
                            motor_layer_name=layer_name, motor_output_idx=0)
            joint_R = Joint(parent_id=parent_id, child_id=id_R,
                            attach_dist=dist,
                            attach_angle=math.radians(-attach_angle_deg),
                            angle_min=amin, angle_max=amax,
                            motor_layer_name=layer_name, motor_output_idx=1)
            self.circuit.bodies += [body_L, body_R]
            self.circuit.joints += [joint_L, joint_R]
            lyr = MotorLayer(activation='linear', n=2, name=layer_name, layer=next_depth)
        else:
            cid = str(uuid.uuid4())[:8]
            body = RigidBody(cid, layer_name, radius)
            joint = Joint(parent_id=parent_id, child_id=cid,
                          attach_dist=dist,
                          attach_angle=math.radians(attach_angle_deg),
                          angle_min=amin, angle_max=amax,
                          motor_layer_name=layer_name, motor_output_idx=0)
            self.circuit.bodies.append(body)
            self.circuit.joints.append(joint)
            lyr = MotorLayer(activation='linear', n=1, name=layer_name, layer=next_depth)

        lyr._is_joint_motor = True
        self.circuit.layers.append(lyr)
        return lyr

    def rebuild_joint_motor_layers(self):
        """Remove stale _is_joint_motor layers and re-add fresh ones where needed.

        If a joint's motor_layer_name refers to an existing network layer
        (e.g. the 'motor' SumLayer for the drive wheels), that layer is left
        alone — the wheels are always present and are not deletable.  Only
        additional joints whose motor_layer_name has no matching network layer
        get a synthesised _is_joint_motor stub.
        """
        self.circuit.layers = [l for l in self.circuit.layers
                               if not getattr(l, '_is_joint_motor', False)]
        existing_names = {l.name for l in self.circuit.layers}
        base_depth = (max(
            (getattr(l, 'layer', None) or 0)
            for l in self.circuit.layers if l.n is not None
        ) + 1) if self.circuit.layers else 1
        seen = {}  # layer_name → max output_idx
        for joint in self.circuit.joints:
            lname = joint.motor_layer_name
            seen[lname] = max(seen.get(lname, 0), joint.motor_output_idx)
        i = 0
        for lname, max_idx in seen.items():
            if lname in existing_names:
                continue  # existing network layer (e.g. 'motor') handles this joint
            lyr = MotorLayer(activation='linear', n=max_idx + 1,
                            name=lname, layer=base_depth + i)
            lyr._is_joint_motor = True
            self.circuit.layers.append(lyr)
            i += 1

    def create_brain_file(self, class_name):
        """Write a new brain file from the template. Returns the path, or None if it already exists."""
        file_path = os.path.join('brains', f'{class_name}.py')
        if os.path.exists(file_path):
            return None
        with open(file_path, 'w') as f:
            f.write(_BRAIN_TEMPLATE.format(class_name=class_name))
        return file_path

    def load_network_into_circuit(self, brain, net_name):
        """
        Load a network JSON, wire sensors/layers/connections into the circuit,
        and sync all references onto the brain instance.

        Returns (hidden, disabled, col_labels, conn_params, freshness_issues) for the
        caller to update its UI, or (None, None, {}, {}, []) when the file does not exist.
        freshness_issues is a list of dicts (see brain_serializer.check_network_freshness).
        """
        import json as _json
        from brain_serializer import load_network_json, check_network_freshness
        path = os.path.join('networks', net_name)
        if not os.path.exists(path):
            return None, None, {}, {}, []
        try:
            with open(path, 'r', encoding='utf-8') as _f:
                _data = _json.load(_f)
            sensors, layers, connections, hidden, disabled, col_labels, net_bodies, net_joints, conn_params = \
                load_network_json(_data)
        except Exception as e:
            print(f'[BrainManager] Failed to load {path}: {e}')
            return None, None, {}, {}, []

        freshness_issues = check_network_freshness(_data, sensors, layers)

        self.circuit.sensors     = sensors
        self.circuit.layers      = layers
        self.circuit.connections = connections

        if net_bodies:
            self.circuit.bodies = net_bodies
        elif not self.circuit.bodies:
            self.circuit.bodies = [RigidBody('root', 'root', self.sim_cfg.body_radius)]
        if net_joints is not None:
            self.circuit.joints = net_joints
            if not net_joints and len(self.circuit.bodies) > 1:
                self.circuit.bodies = self.circuit.bodies[:1]

        brain.sensors     = sensors
        brain.layers      = layers
        brain.connections = connections
        for layer in layers:
            setattr(brain, layer.name, layer)
            layer.reset()
        for sensor in sensors:
            sensor.reset()

        self.rebuild_joint_motor_layers()
        self._upgrade_motor_layers()
        # Re-sync brain.layers after rebuild/upgrade so all layer objects are current.
        brain.layers = self.circuit.layers
        for layer in self.circuit.layers:
            setattr(brain, layer.name, layer)
        self.resolve_joint_sensor_refs()

        return hidden, disabled, col_labels, conn_params, freshness_issues

    def _upgrade_motor_layers(self):
        """Upgrade any SumLayer that acts as a motor output to MotorLayer.

        Called after loading a network from JSON so old files gain robot_address
        without requiring a manual migration step.  MotorLayer is a SumLayer
        subclass, so sim behaviour is identical.
        """
        motor_names = {j.motor_layer_name for j in self.circuit.joints}
        motor_names.add('motor')  # default wheel motor name convention
        for i, lyr in enumerate(self.circuit.layers):
            if isinstance(lyr, SumLayer) and not isinstance(lyr, MotorLayer) \
                    and lyr.name in motor_names:
                new_lyr = MotorLayer(
                    activation=getattr(lyr, 'activation', 'linear'),
                    scale=getattr(lyr, 'scale', 1.0),
                    name=lyr.name,
                    n=lyr.n,
                    color=getattr(lyr, 'color', None),
                    layer=getattr(lyr, 'layer', None),
                )
                if hasattr(lyr, '_is_joint_motor'):
                    new_lyr._is_joint_motor = lyr._is_joint_motor
                self.circuit.layers[i] = new_lyr

    def resolve_joint_sensor_refs(self):
        """Wire _joint_refs or _layer_ref for ProprioceptiveSensor.

        Prefers physical joints (grouped by motor_layer_name == joint_id).
        Falls back to reading the named layer directly when no joints match —
        this lets ProprioceptiveSensor read motor output without physical bodies.
        """
        layer_map = {l.name: l for l in self.circuit.layers}
        for sensor in self.circuit.sensors:
            if isinstance(sensor, ProprioceptiveSensor) and sensor.joint_id:
                group = sorted(
                    [jt for jt in self.circuit.joints
                     if jt.motor_layer_name == sensor.joint_id],
                    key=lambda j: j.motor_output_idx
                )
                sensor._joint_refs = group
                sensor._layer_ref  = None
                if group:
                    sensor.n = len(group)
                else:
                    lyr = layer_map.get(sensor.joint_id)
                    sensor._layer_ref = lyr
                    if lyr is not None:
                        sensor.n = lyr.n or 1
