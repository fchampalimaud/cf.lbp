"""
sim_engine.py — one simulation step for the 2-D LBP simulator.

No Qt, no display logic. All functions take explicit arguments and are
testable in isolation.

Responsibilities are split between two subsystems that drive different
aspects of the same simulation:

  2-D fields (this module + sensors.py)
      patches, gradients, sky polarization, interoception, rays — anything
      MuJoCo has no notion of.
  MuJoCo (sim_engine_mujoco.MuJoCoEngine, passed in as `engine`)
      body movement, contacts (eligible CollisionSensors) and cameras.

step_agents() runs every agent through the same phases:

  1. SENSE   all sensor readings describe the world at time t
  2. THINK   brain.loop(dt), once per agent
  3. MOTORS  one motor source per agent (keyboard / network / brain —
             see rules/motor_commands.md)
  4. ACT     MuJoCo moves all agents together
"""

from typing import Protocol, runtime_checkable

import numpy as np
import torch

WHEEL_DIAMETER = 0.06
MAX_SPEED_MS   = (196 * np.pi * WHEEL_DIAMETER) / 60.0


@runtime_checkable
class Engine(Protocol):
    """What step_agents needs from the physics side (MuJoCoEngine, or a
    stand-in in tests): which sensors it samples, sensing, and moving."""

    def owns_sensor(self, sensor) -> bool:
        """True for sensors the engine samples (the rest are 2-D fields)."""

    def sample_collision_sensors(self, brain, sensors, sim_cfg, agent_idx: int = 0) -> None:
        """SENSE: write contact sensor readings at the current positions onto brain."""

    def render_cameras(self, brain, cam_sensors, agent_idx: int, t: float, sim_dt: float) -> None:
        """SENSE: render the given cameras and write their outputs onto brain."""

    def move_agents(self, bot_positions, commands, sim_cfg) -> None:
        """ACT: drive each bot_pos (mutated in place) from its (mL, mR) command."""


# ── SENSE ─────────────────────────────────────────────────────────────────────

def advance_joints(circuit, bot_pos, dt):
    """Integrate joint angles from the previous tick's commanded velocities and
    return world poses {body_id: (x, y, theta)}, or None without child bodies."""
    if not (circuit and circuit.bodies):
        return None
    from rigid_body import integrate_joints, world_poses
    if circuit.joints:
        integrate_joints(circuit.joints, dt)
    return world_poses(bot_pos, circuit.bodies, circuit.joints)


def sample_field_sensors(brain, sensors, bot_pos, world, sim_cfg, poses=None, other_agents=None):
    """Sample the 2-D-owned sensors into brain attributes.

    If poses is provided ({body_id: (x, y, theta)}), sensors mounted on a
    child body are sampled from that body's world pose instead of the root.
    Sensors with multiple body_ids (mirrored pairs) are sampled once per body
    and the outputs are concatenated.

    other_agents, if given, is a list of {'x','y','r'} circles for every
    other agent in the session — DistanceSensor and CollisionSensor treat
    them as additional obstacles.
    """
    root_pose = tuple(bot_pos)
    for sensor in sensors:
        kwargs = ({'other_agents': other_agents}
                  if other_agents and sensor.needs_other_agents else {})
        body_ids = getattr(sensor, 'body_ids', None) or ['root']
        if len(body_ids) == 1:
            sx, sy, sth = poses.get(body_ids[0], root_pose) if poses and body_ids[0] != 'root' else root_pose
            out = sensor.sample(sx, sy, sth, world, sim_cfg, **kwargs)
        else:
            parts = []
            sensor._contact_dist_per_body = {}
            for i, bid in enumerate(body_ids):
                sx, sy, sth = poses.get(bid, root_pose) if poses and bid != 'root' else root_pose
                if i > 0 and hasattr(sensor, 'mount_angle'):
                    orig = sensor.mount_angle
                    sensor.mount_angle = -orig
                    try:
                        parts.append(sensor.sample(sx, sy, sth, world, sim_cfg, **kwargs))
                    finally:
                        sensor.mount_angle = orig
                else:
                    parts.append(sensor.sample(sx, sy, sth, world, sim_cfg, **kwargs))
                if hasattr(sensor, '_contact_dist'):
                    sensor._contact_dist_per_body[bid] = sensor._contact_dist
            out = np.concatenate(parts)
            # Store per-body halves so network_runner and visualizer can access them.
            sensor._left_output  = parts[0]
            sensor._right_output = parts[-1]
        setattr(brain, sensor.name, out)
        sensor.publish_halves(brain)   # joint-pair halves (set above)


def due_cameras(sensors, t):
    """Cameras with a fixed frame rate (fps > 0) whose next frame is due at sim time t.
    fps == 0 cameras are free-running and rendered by the display loop instead."""
    return [s for s in sensors
            if s.is_camera and s.fps > 0 and s.frame_due(t)]


def free_running_cameras(sensors, t):
    """fps == 0 cameras that have not yet rendered at sim time t."""
    return [s for s in sensors
            if s.is_camera and s.fps <= 0 and s.frame_due(t)]


# ── THINK / MOTORS ────────────────────────────────────────────────────────────

def run_joint_motors(brain, circuit, dt):
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


def wheel_command_from_layers(circuit, mL, mR):
    """If root-level joints (wheels) exist, their motor layer is the canonical
    wheel command, so ProprioceptiveSensor and movement use the same signal.
    Falls back to brain.loop()'s (mL, mR) when no root joints are present.
    Synthesized _is_joint_motor layers (limb/whisker actuators) never drive
    the wheels — only user-placed motor layers (e.g. 'motor') do."""
    if not (circuit and circuit.bodies and circuit.joints):
        return mL, mR
    root_id = circuit.bodies[0].id
    layer_map = {l.name: l for l in circuit.layers}
    for rj in circuit.joints:
        if rj.parent_id != root_id:
            continue
        lyr = layer_map.get(rj.motor_layer_name)
        if lyr is None or getattr(lyr, '_is_joint_motor', False):
            continue
        if lyr.output is not None:
            out = np.atleast_1d(lyr.output)
            mL = float(out[0]) if len(out) > 0 else mL
            mR = float(out[1]) if len(out) > 1 else mR
        break
    return mL, mR


def write_motor_layers(circuit, cmd):
    """Write a non-brain motor command (keyboard / network) into every user
    motor layer's output, so the visualizer and oscilloscope show what is
    actually driving the wheels. Synthesized joint actuators are left alone."""
    for layer in circuit.layers:
        if not getattr(layer, 'drives_wheels', False):
            continue
        if layer.output is None:
            continue
        n = int(layer.output.numel()) if hasattr(layer.output, 'numel') \
            else len(np.atleast_1d(layer.output))
        vals = [float(cmd[j]) if j < len(cmd) else 0.0 for j in range(n)]
        layer.output = torch.tensor(vals, dtype=torch.float32)


def apply_motor_command(circuit, cmd):
    """Make a non-brain motor command (keyboard / network) the wheel command:
    feed it back to wheel joints / proprioception and to the motor layers.
    The brain itself has already run this tick — this only changes where the
    final wheel values come from (rules/motor_commands.md)."""
    mL, mR = cmd
    # Path 1: wheel joints present — write into joint.vel (integrates next tick).
    synced = False
    if circuit.bodies and circuit.joints:
        root_id = circuit.bodies[0].id
        body_map = {b.id: b for b in circuit.bodies}
        for jt in circuit.joints:
            if jt.parent_id != root_id:
                continue
            idx = jt.motor_output_idx
            vel = mL if idx == 0 else mR
            body = body_map.get(jt.child_id)
            if idx > 0 and body and getattr(body, 'mirror_group', ''):
                vel = -vel
            jt.vel = vel
            synced = True
    # Path 2: no wheel joints — push the command into _layer_ref output on any
    # sensor that uses a layer for motor feedback (e.g. ProprioceptiveSensor).
    if not synced:
        cmd_t = torch.tensor([mL, mR], dtype=torch.float32)
        for s in circuit.sensors:
            lref = getattr(s, '_layer_ref', None)
            if lref is None or lref.output is None:
                continue
            n = int(lref.output.numel()) if hasattr(lref.output, 'numel') else len(np.atleast_1d(lref.output))
            lref.output = cmd_t[:n]
    write_motor_layers(circuit, cmd)


def run_brain(brain, dt):
    """THINK: run the brain once. Returns the (mL, mR) wheel command loop() returns.
    Shared by every mode — simulator, real robot and network client."""
    mL, mR = brain.loop(dt)
    return float(mL), float(mR)


def choose_motor_command(circuit, brain_cmd, external_cmd=None):
    """MOTORS: exactly one wheel command source per agent (rules/motor_commands.md).

    brain_cmd    : (mL, mR) from run_brain, or None if the agent has no brain
    external_cmd : keyboard / network (mL, mR), or None — replaces the brain's
                   command and is written back into the motor layer
    Returns the final (mL, mR), clamped to the motor duty range [-100, 100].
    Shared by every mode — simulator, real robot and network client."""
    mL = mR = 0.0
    if brain_cmd is not None:
        mL, mR = wheel_command_from_layers(circuit, *brain_cmd)
    if external_cmd is not None:
        mL, mR = external_cmd
        apply_motor_command(circuit, external_cmd)
    return max(-100.0, min(100.0, mL)), max(-100.0, min(100.0, mR))


def raw_sensor_values(brain, sensors):
    """'<sensor>_<i>' → value for every non-camera sensor reading on the brain."""
    raw = {}
    for sensor in sensors:
        if sensor.is_camera:
            continue
        vals = np.atleast_1d(getattr(brain, sensor.name, np.zeros(sensor.n)))
        for j, v in enumerate(vals):
            raw[f'{sensor.name}_{j}'] = float(v)
    return raw


# ── Step ──────────────────────────────────────────────────────────────────────

def step_agents(agents, world, sim_cfg, engine, t, motor_for=None):
    """Advance every agent by one step of sim_cfg.dt.

    agents    : objects with .bot_pos [x, y, theta] (mutated in place), .brain, .circuit
    engine    : Engine (MuJoCoEngine) — owns movement, contacts and cameras
    t         : simulation time at the start of this step (seconds)
    motor_for : callable(agent) -> (mL, mR) | None; a non-None result is a
                keyboard / network motor command that replaces the brain's
                wheel command for that agent

    Returns a list of raw signal dicts, one per agent ('mL', 'mR', and
    '<sensor>_<i>' for every non-camera sensor).
    """
    dt = sim_cfg.dt
    # Snapshot every agent's pre-step position once, so sensing doesn't depend
    # on iteration order and other agents are seen as obstacles.
    circles = [{'x': a.bot_pos[0], 'y': a.bot_pos[1], 'r': sim_cfg.body_radius}
               for a in agents]

    # 1. SENSE — every reading describes the world at time t, before any brain runs.
    poses_list = []
    for i, agent in enumerate(agents):
        circuit, brain = agent.circuit, agent.brain
        poses = advance_joints(circuit, agent.bot_pos, dt)
        poses_list.append(poses)
        if brain is None:
            continue
        sensors = circuit.sensors
        field_sensors = [s for s in sensors if not engine.owns_sensor(s)]
        sample_field_sensors(brain, field_sensors, agent.bot_pos, world, sim_cfg,
                             poses, circles[:i] + circles[i + 1:])
        engine.sample_collision_sensors(brain, sensors, sim_cfg, agent_idx=i)
        engine.render_cameras(brain, due_cameras(sensors, t), i, t, dt)

    # 2. THINK, 3. MOTORS
    raws, commands = [], []
    for agent, poses in zip(agents, poses_list):
        circuit, brain = agent.circuit, agent.brain
        raw = {}
        brain_cmd = None
        if brain is not None:
            brain_cmd = run_brain(brain, dt)
            raw.update(raw_sensor_values(brain, circuit.sensors))
            if circuit.joints:
                run_joint_motors(brain, circuit, dt)
                for jnt in circuit.joints:
                    jnt._vel_pre_clamp = jnt.vel
                _clamp_whisker_joint_vel(circuit, circuit.sensors, poses, world, sim_cfg, agent.bot_pos)

        external = motor_for(agent) if motor_for is not None else None
        mL, mR = choose_motor_command(circuit, brain_cmd, external)
        raw['mL'] = mL
        raw['mR'] = mR
        raws.append(raw)
        commands.append((mL, mR))

    # 4. ACT — MuJoCo moves all agents together.
    engine.move_agents([a.bot_pos for a in agents], commands, sim_cfg)
    return raws
