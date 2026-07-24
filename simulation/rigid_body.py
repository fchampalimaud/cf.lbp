import math
from collections import deque


class RigidBody:
    def __init__(self, id: str, name: str, radius: float = 0.12, mirror_group: str = ''):
        self.id = id
        self.name = name
        self.radius = radius
        self.mirror_group = mirror_group  # non-empty = part of a mirrored pair (shared motor layer name)

    def to_dict(self):
        d = {'id': self.id, 'name': self.name, 'radius': self.radius}
        if self.mirror_group:
            d['mirror_group'] = self.mirror_group
        return d

    @staticmethod
    def from_dict(d):
        return RigidBody(d['id'], d['name'], d.get('radius', 0.12),
                         mirror_group=d.get('mirror_group', ''))


class Joint:
    def __init__(self, parent_id: str, child_id: str,
                 attach_dist: float = 0.2, attach_angle: float = 0.0,
                 angle: float = 0.0,
                 angle_min: float = -1.5708, angle_max: float = 1.5708,
                 motor_layer_name: str = '', motor_output_idx: int = 0):
        self.parent_id = parent_id
        self.child_id = child_id
        self.attach_dist = attach_dist
        self.attach_angle = attach_angle   # radians, local frame from parent heading
        self.angle = angle                 # current joint angle, radians
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.vel = 0.0                     # angular velocity (written each tick by brain)
        self.motor_layer_name = motor_layer_name   # which SumLayer drives this joint
        self.motor_output_idx = motor_output_idx   # which output index of that layer

    def to_dict(self):
        return {
            'parent_id': self.parent_id,
            'child_id': self.child_id,
            'attach_dist': self.attach_dist,
            'attach_angle': self.attach_angle,
            'angle': self.angle,
            'angle_min': self.angle_min,
            'angle_max': self.angle_max,
            'motor_layer_name': self.motor_layer_name,
            'motor_output_idx': self.motor_output_idx,
        }

    @staticmethod
    def from_dict(d):
        return Joint(
            parent_id=d['parent_id'],
            child_id=d['child_id'],
            attach_dist=d.get('attach_dist', 0.2),
            attach_angle=d.get('attach_angle', 0.0),
            angle=d.get('angle', 0.0),
            angle_min=d.get('angle_min', -1.5708),
            angle_max=d.get('angle_max', 1.5708),
            motor_layer_name=d.get('motor_layer_name', ''),
            motor_output_idx=d.get('motor_output_idx', 0),
        )


def world_poses(root_pose, bodies: list, joints: list) -> dict:
    """Return {body_id: (x, y, theta)} for every body via BFS from root."""
    if not bodies:
        return {}

    x0, y0, th0 = root_pose[0], root_pose[1], root_pose[2]
    poses = {bodies[0].id: (x0, y0, th0)}

    # Build parent→children map
    children = {}
    joint_map = {}
    for j in joints:
        children.setdefault(j.parent_id, []).append(j.child_id)
        joint_map[j.child_id] = j

    queue = deque([bodies[0].id])
    while queue:
        pid = queue.popleft()
        px, py, pth = poses[pid]
        for cid in children.get(pid, []):
            jt = joint_map[cid]
            attach_dir = pth + jt.attach_angle
            cx = px + jt.attach_dist * math.cos(attach_dir)
            cy = py + jt.attach_dist * math.sin(attach_dir)
            cth = pth + jt.attach_angle + jt.angle
            poses[cid] = (cx, cy, cth)
            queue.append(cid)

    return poses


def integrate_joints(joints: list, dt: float):
    """Advance each joint angle by vel*dt and clamp to limits."""
    for j in joints:
        j.angle = max(j.angle_min, min(j.angle_max, j.angle + j.vel * dt))
