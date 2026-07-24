"""
sim_engine.py — pure physics step for the 2-D Braitenberg simulator.

No Qt, no display logic. All functions take explicit arguments and are
testable in isolation.
"""

import inspect
import numpy as np
import torch
from sensors import SENSOR_DIST

WHEEL_DIAMETER = 0.06
MAX_SPEED_MS   = (196 * np.pi * WHEEL_DIAMETER) / 60.0


def _sample_sensors(brain, sensors, x, y, theta, world, sim_cfg, poses=None):
    """Sample all sensors into brain attributes. Returns (mL, mR).

    If poses is provided ({body_id: (x, y, theta)}), sensors mounted on a
    child body are sampled from that body's world pose instead of the root.
    Sensors with multiple body_ids (mirrored pairs) are sampled once per body
    and the outputs are concatenated.
    """
    root_pose = (x, y, theta)
    for sensor in sensors:
        body_ids = getattr(sensor, 'body_ids', None) or ['root']
        if len(body_ids) == 1:
            sx, sy, sth = poses.get(body_ids[0], root_pose) if poses and body_ids[0] != 'root' else root_pose
            out = sensor.sample(sx, sy, sth, world, sim_cfg)
        else:
            parts = []
            sensor._contact_dist_per_body = {}
            for i, bid in enumerate(body_ids):
                sx, sy, sth = poses.get(bid, root_pose) if poses and bid != 'root' else root_pose
                if i > 0 and hasattr(sensor, 'mount_angle'):
                    orig = sensor.mount_angle
                    sensor.mount_angle = -orig
                    try:
                        parts.append(sensor.sample(sx, sy, sth, world, sim_cfg))
                    finally:
                        sensor.mount_angle = orig
                else:
                    parts.append(sensor.sample(sx, sy, sth, world, sim_cfg))
                if hasattr(sensor, '_contact_dist'):
                    sensor._contact_dist_per_body[bid] = sensor._contact_dist
            out = np.concatenate(parts)
            # Store per-body halves so network_runner and visualizer can access them.
            sensor._left_output  = parts[0]
            sensor._right_output = parts[-1]
        setattr(brain, sensor.name, out)
        # Write _L / _R brain attributes for both camera sensors (explicit lateralized=True)
        # and joint-pair sensors (implicit: len(body_ids) == 2 with _left/_right_output set above).
        if getattr(sensor, 'lateralized', False):
            setattr(brain, sensor.name + '_L', sensor._left_output)
            setattr(brain, sensor.name + '_R', sensor._right_output)
        elif (len(body_ids) == 2
              and getattr(sensor, '_left_output', None) is not None):
            setattr(brain, sensor.name + '_L', sensor._left_output)
            setattr(brain, sensor.name + '_R', sensor._right_output)
    return brain.loop(sim_cfg.dt)


def _collect_layer_input(tgt_name, n_tgt, connections, layer_outputs, sensor_outputs):
    """Build input vector for a target layer from circuit connections.

    connections: list of (src_name, tgt_name, W) where W.shape = (n_tgt, n_src).
    """
    inp = np.zeros(n_tgt)
    for conn in connections:
        src, tgt, W = conn.src, conn.tgt, conn.W
        if tgt != tgt_name:
            continue
        W_arr = np.asarray(W, dtype=float)
        if W_arr.ndim != 2 or W_arr.shape[0] != n_tgt:
            continue
        src_vec = None
        if src in layer_outputs:
            src_vec = layer_outputs[src]
        elif src in sensor_outputs:
            src_vec = sensor_outputs[src]
        if src_vec is None or len(src_vec) != W_arr.shape[1]:
            continue
        inp += W_arr @ src_vec
    return inp


def _run_joint_motors(brain, circuit, dt):
    """Forward pass for joint motor layers → write joint.vel."""
    sensor_outputs = {s.name: np.atleast_1d(getattr(brain, s.name, np.zeros(s.n)))
                      for s in circuit.sensors}
    layer_outputs  = {l.name: np.atleast_1d(l.output)
                      for l in circuit.layers if l.output is not None}
    layer_map = {l.name: l for l in circuit.layers}

    # Step each unique motor layer once, then distribute outputs to joints.
    # _is_joint_motor layers (whisker/limb actuators) are stepped here.
    # Existing network layers (e.g. 'motor') were already stepped by step_network
    # — just read their output to avoid double-stepping.
    stepped = {}  # layer_name → output array
    for joint in circuit.joints:
        name = joint.motor_layer_name
        if not name or name in stepped:
            continue
        lyr = layer_map.get(name)
        if lyr is None:
            continue
        if getattr(lyr, '_is_joint_motor', False):
            n   = lyr.n or 1
            inp = _collect_layer_input(name, n, circuit.connections, layer_outputs, sensor_outputs)
            stepped[name] = np.atleast_1d(lyr.step(inp, dt))
        else:
            # Already stepped by step_network; read current output.
            stepped[name] = np.atleast_1d(lyr.output) if lyr.output is not None else np.zeros(lyr.n or 1)

    body_map = {b.id: b for b in circuit.bodies}
    for joint in circuit.joints:
        out = stepped.get(joint.motor_layer_name)
        if out is None:
            continue
        idx = joint.motor_output_idx
        vel = float(out[min(idx, len(out) - 1)])
        if idx > 0:
            body = body_map.get(joint.child_id)
            if body and getattr(body, 'mirror_group', ''):
                vel = -vel
        joint.vel = vel


def _clamp_whisker_joint_vel(circuit, sensors, poses, world, sim_cfg, bot_pos):
    """Zero the vel of any joint whose whisker would sweep through an obstacle next tick."""
    if not world.objects:
        return
    from rigid_body import world_poses

    body_to_joint = {j.child_id: j for j in circuit.joints}

    for sensor in sensors:
        if getattr(sensor, 'viz_type', None) != 'whisker':
            continue
        body_ids = getattr(sensor, 'body_ids', ['root'])
        for i, bid in enumerate(body_ids):
            joint = body_to_joint.get(bid)
            if joint is None or joint.vel == 0:
                continue

            per_body = getattr(sensor, '_contact_dist_per_body', None)
            cd = per_body.get(bid) if per_body else sensor._contact_dist
            if cd is None:
                continue  # not in contact

            # Tentatively advance the joint angle by one tick
            test_angle = max(joint.angle_min,
                             min(joint.angle_max, joint.angle + joint.vel * sim_cfg.dt))
            if abs(test_angle - joint.angle) < 1e-9:
                continue  # already at angular limit

            orig_angle = joint.angle
            joint.angle = test_angle
            test_poses = world_poses(bot_pos, circuit.bodies, circuit.joints)
            joint.angle = orig_angle

            if bid not in test_poses:
                continue

            tx, ty, tth = test_poses[bid]
            mount_angle = -sensor.mount_angle if i > 0 else sensor.mount_angle
            ray_th = tth + mount_angle
            ox = tx + sensor.mount_dist * np.cos(ray_th)
            oy = ty + sensor.mount_dist * np.sin(ray_th)

            new_cd = sensor._ray_cast(ox, oy, ray_th, world, sim_cfg)
            # Clamp only when the whisker is significantly in contact (d < 70% of
            # length) and would completely lose it in one step — that is a sweep-through.
            # When barely touching (d > 70%), allow natural contact loss during retraction.
            if new_cd is None and cd < sensor.length * 0.7:
                joint.vel = 0.0


def _sample_legacy(brain, x, y, theta, world, sim_cfg):
    """Legacy two-sensor path (no explicit sensor objects). Returns (mL, mR, sL, sR)."""
    sl_t = theta + sim_cfg.sensor_angle
    sr_t = theta - sim_cfg.sensor_angle
    sL   = world.get_signal(x + SENSOR_DIST * np.cos(sl_t),
                            y + SENSOR_DIST * np.sin(sl_t), sl_t)
    sR   = world.get_signal(x + SENSOR_DIST * np.cos(sr_t),
                            y + SENSOR_DIST * np.sin(sr_t), sr_t)
    mL, mR = brain.loop(sL, sR, sim_cfg.dt)
    return mL, mR, sL, sR


def _integrate_movement(bot_pos, mL, mR, sim_cfg):
    """Differential-drive integration. Mutates bot_pos in-place."""
    x, y, theta = bot_pos
    conv  = (MAX_SPEED_MS / 100.0) * sim_cfg.motor_gain
    v     = ((mL + mR) * conv) / 2.0
    omega = ((mR - mL) * conv) / (sim_cfg.body_radius * 2)
    bot_pos[0] = x + v * np.cos(theta) * sim_cfg.dt
    bot_pos[1] = y + v * np.sin(theta) * sim_cfg.dt
    bot_pos[2] = (theta + omega * sim_cfg.dt) % (2 * np.pi)


def _resolve_object_collisions(bot_pos, world, sim_cfg):
    """Push robot out of any overlapping circular objects or polygon walls. Mutates bot_pos in-place."""
    br = sim_cfg.body_radius
    for obj in world.objects:
        dx   = bot_pos[0] - obj['x']
        dy   = bot_pos[1] - obj['y']
        dist = np.hypot(dx, dy)
        if obj.get('external', True):
            min_d = br + obj['r']
            if dist < min_d and dist > 1e-9:
                bot_pos[0] = obj['x'] + (dx / dist) * min_d
                bot_pos[1] = obj['y'] + (dy / dist) * min_d
        else:
            max_d = obj['r'] - br
            if max_d > 0 and dist > max_d and dist > 1e-9:
                bot_pos[0] = obj['x'] + (dx / dist) * max_d
                bot_pos[1] = obj['y'] + (dy / dist) * max_d
    for wall in world.walls:
        pts = wall['points']
        for i in range(len(pts)):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % len(pts)]
            dx, dy = bx - ax, by - ay
            len2 = dx * dx + dy * dy
            if len2 < 1e-12:
                continue
            t = np.clip(((bot_pos[0] - ax) * dx + (bot_pos[1] - ay) * dy) / len2, 0.0, 1.0)
            cx = ax + t * dx
            cy = ay + t * dy
            ddx = bot_pos[0] - cx
            ddy = bot_pos[1] - cy
            dist = np.hypot(ddx, ddy)
            if dist < br and dist > 1e-9:
                bot_pos[0] = cx + ddx / dist * br
                bot_pos[1] = cy + ddy / dist * br


def _clamp_to_arena(bot_pos, world, sim_cfg):
    """Clamp robot position to arena boundary. Mutates bot_pos in-place."""
    limit = sim_cfg.arena_scale
    br    = sim_cfg.body_radius
    if world.arena_round:
        d = np.hypot(bot_pos[0], bot_pos[1])
        max_d = limit - br
        if d > max_d and d > 1e-9:
            bot_pos[0] = bot_pos[0] / d * max_d
            bot_pos[1] = bot_pos[1] / d * max_d
    else:
        bot_pos[0] = max(-limit + br, min(limit - br, bot_pos[0]))
        bot_pos[1] = max(-limit + br, min(limit - br, bot_pos[1]))


def tick_physics(bot_pos, brain, sensors, world, sim_cfg, circuit=None,
                 motor_override=None) -> dict:
    """
    Run one physics step. Mutates bot_pos, brain state, and joint angles in-place.

    Parameters
    ----------
    bot_pos  : list[float]    [x, y, theta] — modified in-place
    brain    : BaseBrain
    sensors  : list           explicit sensor objects, or empty list for legacy mode
    world    : World
    sim_cfg  : SimConfig
    circuit  : CircuitModel   optional; required for hierarchical body FK + joint motors

    Returns
    -------
    dict  raw signal values keyed as 'mL', 'mR', and per-sensor names.
          In legacy mode also contains 'sL', 'sR'.
    """
    x, y, theta = bot_pos
    raw = {}

    # Integrate joint angles from previous tick's commanded velocities, then FK
    poses = None
    if circuit and circuit.joints:
        from rigid_body import integrate_joints, world_poses
        integrate_joints(circuit.joints, sim_cfg.dt)
        poses = world_poses(bot_pos, circuit.bodies, circuit.joints)
    elif circuit and circuit.bodies:
        from rigid_body import world_poses
        poses = world_poses(bot_pos, circuit.bodies, circuit.joints)

    if sensors:
        # Tick order (one-tick delay is intentional):
        #  1. integrate_joints  — advance angles from last tick's joint.vel
        #  2. _sample_sensors   — read sensors; calls brain.loop → step_network
        #                         ProprioceptiveSensor reads joint.vel / layer.output
        #                         from the PREVIOUS tick, giving a one-tick delay
        #  3. _run_joint_motors — read freshly computed motor output → write joint.vel
        #                         for step 1 of the NEXT tick
        mL, mR = _sample_sensors(brain, sensors, x, y, theta, world, sim_cfg, poses)
        raw['mL'] = mL
        raw['mR'] = mR
        for sensor in sensors:
            vals = np.atleast_1d(getattr(brain, sensor.name, np.zeros(sensor.n)))
            for i, v in enumerate(vals):
                raw[f'{sensor.name}_{i}'] = float(v)
        if circuit and circuit.joints:
            _run_joint_motors(brain, circuit, sim_cfg.dt)
            for _jnt in circuit.joints:
                _jnt._vel_pre_clamp = _jnt.vel
            _clamp_whisker_joint_vel(circuit, sensors, poses, world, sim_cfg, bot_pos)
    else:
        # Determine call convention from loop's signature.
        # Legacy brains: loop(sL, sR, bumpers)  → 3 bound params
        # New-style brains: loop(dt)             → 1 bound param
        try:
            n_loop_params = brain._loop_n_params
        except AttributeError:
            try:
                n_loop_params = len(inspect.signature(brain.loop).parameters)
            except (ValueError, TypeError):
                n_loop_params = 1
            brain._loop_n_params = n_loop_params
        if n_loop_params >= 2:
            mL, mR, sL, sR = _sample_legacy(brain, x, y, theta, world, sim_cfg)
            raw.update({'sL': sL, 'sR': sR})
        else:
            mL, mR = brain.loop(sim_cfg.dt)
        raw.update({'mL': mL, 'mR': mR})

    # If root-level joints exist (wheels), read their motor layer output for movement.
    # This makes the wheels joint the canonical motor output, so ProprioceptiveSensor
    # and _integrate_movement both use the same signal. Falls back to brain.loop()
    # mL/mR when no root joints are present (backward compatible).
    # NOTE: skip synthesized _is_joint_motor layers (limb/whisker actuators) — only
    # user-placed motor layers (e.g. 'motor') should drive wheel velocity.
    if circuit and circuit.bodies and circuit.joints:
        root_id     = circuit.bodies[0].id
        root_joints = [j for j in circuit.joints if j.parent_id == root_id]
        motor_name  = None
        for rj in root_joints:
            lyr = next((l for l in circuit.layers if l.name == rj.motor_layer_name), None)
            if lyr is not None and not getattr(lyr, '_is_joint_motor', False):
                motor_name = rj.motor_layer_name
                break
        if motor_name:
            motor_lyr = next((l for l in circuit.layers if l.name == motor_name), None)
            if motor_lyr is not None and motor_lyr.output is not None:
                out = np.atleast_1d(motor_lyr.output)
                mL  = float(out[0]) if len(out) > 0 else mL
                mR  = float(out[1]) if len(out) > 1 else mR
                raw['mL'] = mL
                raw['mR'] = mR

    if motor_override is not None:
        mL, mR = motor_override
        raw['mL'] = mL
        raw['mR'] = mR
        # Sync motor feedback so ProprioceptiveSensor reads the override.
        # Path 1: wheel joints present — write into joint.vel (integrates next tick)
        _synced = False
        if circuit and circuit.bodies and circuit.joints:
            _ov_root = circuit.bodies[0].id
            _ov_bmap = {b.id: b for b in circuit.bodies}
            for _jt in circuit.joints:
                if _jt.parent_id != _ov_root:
                    continue
                _idx = _jt.motor_output_idx
                _vel = mL if _idx == 0 else mR
                _body = _ov_bmap.get(_jt.child_id)
                if _idx > 0 and _body and getattr(_body, 'mirror_group', ''):
                    _vel = -_vel
                _jt.vel = _vel
                _synced = True
        # Path 2: no wheel joints — push override into _layer_ref output on any
        # sensor that uses a layer for motor feedback (e.g. ProprioceptiveSensor).
        if not _synced and circuit and circuit.sensors:
            _ov_t = torch.tensor([mL, mR], dtype=torch.float32)
            for _s in circuit.sensors:
                _lref = getattr(_s, '_layer_ref', None)
                if _lref is None or _lref.output is None:
                    continue
                _n = int(_lref.output.numel()) if hasattr(_lref.output, 'numel') else len(np.atleast_1d(_lref.output))
                _lref.output = _ov_t[:_n]

    mL = max(-100.0, min(100.0, mL))
    mR = max(-100.0, min(100.0, mR))
    raw['mL'] = mL
    raw['mR'] = mR

    if sim_cfg.fixate_robot < 0.5:
        _integrate_movement(bot_pos, mL, mR, sim_cfg)
        _resolve_object_collisions(bot_pos, world, sim_cfg)
        _clamp_to_arena(bot_pos, world, sim_cfg)

    return raw
