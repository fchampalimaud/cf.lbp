"""
sim_engine_mujoco.py — MuJoCo-backed simulation engine.

MuJoCo runs the physics (mj_step, contacts, collision with objects/walls).
The 2D canvas reads robot position FROM MuJoCo each tick.
Gradient/patch sensors are computed analytically as before (not by MuJoCo).
CameraSensor renders a real 3D perspective image via mujoco.Renderer.

Architecture
------------
  tick_physics():
    1. Sample gradient/touch/etc. sensors at current bot_pos (analytical).
    2. brain.loop() → mL, mR.
    3. Set robot velocity in MuJoCo from mL, mR (kinematic drive).
    4. mj_step() — MuJoCo resolves contacts (objects, walls, arena).
    5. Read bot_pos back from MuJoCo qpos.
    6. Render camera sensors from MuJoCo scene.

  rebuild(world, sim_cfg):
    Regenerate MuJoCo model XML from current world state.
    Call after any structural world edit (add/remove object or wall).
"""

import time
import numpy as np
import threading

try:
    import mujoco
    import mujoco.viewer
    _MUJOCO_OK = True
except ImportError:
    _MUJOCO_OK = False

from sim_engine import tick_physics as _tick_2d, MAX_SPEED_MS
from sensors import CameraSensor

# Geometry half-sizes (MuJoCo convention)
_ROBOT_H = 0.04   # robot cylinder half-height
_OBJ_H   = 0.18   # solid-object cylinder half-height
_WALL_H  = 0.15   # arena/polygon wall box half-height
_WALL_T  = 0.04   # arena/polygon wall box half-thickness

# Overhead camera: placed at arena_scale × _OVERHEAD_H_FACTOR above the centre.
# FOV is chosen so the image covers exactly [-arena_scale, +arena_scale] on both axes
# when rendered square (W == H).  tan(fovy/2) = 1 / _OVERHEAD_H_FACTOR.
_OVERHEAD_H_FACTOR = 3.0

# Viewer sync interval (seconds) — decoupled from physics rate to reduce GPU load
_VIEWER_SYNC_DT = 0.033   # ~30 fps

# Per-agent body colors (matches arena_widget._AGENT_COLORS, stored as RGB floats)
_ROBOT_COLORS = [
    (0.29, 0.50, 0.80),   # blue   — agent 0
    (0.88, 0.36, 0.36),   # red    — agent 1
    (0.36, 0.72, 0.36),   # green  — agent 2
    (0.94, 0.65, 0.00),   # gold   — agent 3
    (0.61, 0.36, 0.72),   # purple — agent 4
    (0.09, 0.64, 0.72),   # teal   — agent 5
]


class MuJoCoEngine:
    """
    MuJoCo-backed simulation engine.

    MuJoCo is the physics authority.  The 2D canvas reads robot position
    from this engine each tick.  Gradients are computed analytically.
    """

    def __init__(self, world, sim_cfg, n_agents=1):
        if not _MUJOCO_OK:
            raise RuntimeError("mujoco package is not installed")
        self._lock              = threading.Lock()
        self._renderer          = None   # front-camera renderer (per CameraSensor size)
        self._overhead_renderer = None   # overhead view renderer (fixed square)
        self._viewer        = None
        self._viewer_thread = None
        self._stop_viewer   = threading.Event()
        self.model          = None
        self.data           = None
        self._n_agents      = max(1, n_agents)
        self._robot_body_ids = []        # list[int], one per agent
        self._robot_body_id  = -1       # backward-compat alias for agent 0
        self.rebuild(world, sim_cfg, n_agents=n_agents)

    # ── XML ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_xml(world, sim_cfg, n_agents=1) -> str:
        s  = sim_cfg.arena_scale
        br = sim_cfg.body_radius
        dt = sim_cfg.dt
        wh = _WALL_H
        wt = _WALL_T

        lines = [
            f'<mujoco model="braitenberg3d">',
            f'  <option gravity="0 0 0" integrator="Euler" timestep="{dt:.4f}"/>',
            f'  <visual>',
            f'    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0 0 0"/>',
            f'    <map znear="0.001"/>',
            f'    <global offwidth="512" offheight="512"/>',
            f'  </visual>',
            f'  <asset>',
            f'    <texture name="grid" type="2d" builtin="checker"',
            f'             rgb1=".52 .52 .52" rgb2=".94 .94 .94" width="512" height="512"/>',
            f'    <material name="floor_mat" texture="grid" texrepeat="4 4" reflectance="0.0"/>',
            f'  </asset>',
            f'  <worldbody>',
            f'    <light pos="0 0 {s*3:.2f}" dir="0 0 -1"',
            f'           diffuse="0.8 0.8 0.8" specular="0 0 0" directional="true"/>',
            f'    <geom name="floor" type="plane" size="{s+1:.2f} {s+1:.2f} 0.1"',
            f'          material="floor_mat" contype="0" conaffinity="0"/>',
            # Overhead camera: fixed above the arena centre, looking straight down.
            # xyaxes: cam-X = world +X (east), cam-Y = world +Y (north) → cam-Z = +Z,
            # so the camera looks along -Z (downward).  FOV chosen so a square render
            # covers exactly ±arena_scale in both directions.
            f'    <camera name="overhead_cam"',
            f'            pos="0 0 {s * _OVERHEAD_H_FACTOR:.3f}"',
            f'            xyaxes="1 0 0 0 1 0"',
            f'            fovy="{float(np.degrees(2.0 * np.arctan(1.0 / _OVERHEAD_H_FACTOR))):.2f}"/>',
        ]

        # Arena boundary (solid — robot contacts these)
        wall_rgba = 'rgba="0.4 0.4 0.4 1"'
        if getattr(world, 'arena_round', False):
            # Enough segments that the ring looks smooth (~1 per 8 cm of arc).
            N = max(64, round(2 * np.pi * s / 0.08))
            for i in range(N):
                a      = 2 * np.pi * (i + 0.5) / N
                half_c = np.pi * s / N          # half chord length
                cx, cy = s * np.cos(a), s * np.sin(a)
                ang    = np.degrees(a) + 90.0   # box local-X tangential
                lines.append(
                    f'    <geom type="box" pos="{cx:.4f} {cy:.4f} {wh:.4f}" '
                    f'size="{half_c:.4f} {wt:.4f} {wh:.4f}" euler="0 0 {ang:.2f}" '
                    f'{wall_rgba} contype="1" conaffinity="1"/>'
                )
        else:
            for px, py, sx, sy in [
                ( s,  0, wt,  s + wt),
                (-s,  0, wt,  s + wt),
                ( 0,  s, s + wt, wt),
                ( 0, -s, s + wt, wt),
            ]:
                lines.append(
                    f'    <geom type="box" pos="{px:.4f} {py:.4f} {wh:.4f}" '
                    f'size="{sx:.4f} {sy:.4f} {wh:.4f}" '
                    f'{wall_rgba} contype="1" conaffinity="1"/>'
                )

        # Objects → solid cylinder (external=True) or ring of wall segments (external=False / room).
        for obj in world.objects:
            c       = obj.get('color', [0.75, 0.75, 0.75])
            r       = float(obj['r'])
            rgba    = f'rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1"'
            is_room = not obj.get('external', True)
            if is_room:
                N = max(16, round(2 * np.pi * r / 0.12))
                for i in range(N):
                    a      = 2 * np.pi * (i + 0.5) / N
                    half_c = np.pi * r / N
                    cx_    = obj['x'] + r * np.cos(a)
                    cy_    = obj['y'] + r * np.sin(a)
                    ang    = np.degrees(a) + 90.0
                    lines.append(
                        f'    <geom type="box"'
                        f' pos="{cx_:.4f} {cy_:.4f} {_WALL_H:.4f}"'
                        f' size="{half_c:.4f} {_WALL_T:.4f} {_WALL_H:.4f}"'
                        f' euler="0 0 {ang:.2f}" {rgba}'
                        f' contype="1" conaffinity="1"/>'
                    )
            else:
                lines.append(
                    f'    <geom type="cylinder"'
                    f' pos="{obj["x"]:.4f} {obj["y"]:.4f} {_OBJ_H:.4f}"'
                    f' size="{r:.4f} {_OBJ_H:.4f}" {rgba}'
                    f' contype="1" conaffinity="1"/>'
                )

        # Polygon walls → one box per edge (solid)
        for wall in getattr(world, 'walls', []):
            pts = wall['points']
            c   = wall.get('color', [0.5, 0.5, 0.5])
            for i in range(len(pts)):
                ax, ay = pts[i]
                bx, by = pts[(i + 1) % len(pts)]
                cx_ = (ax + bx) / 2
                cy_ = (ay + by) / 2
                seg = np.hypot(bx - ax, by - ay)
                ang = np.degrees(np.arctan2(by - ay, bx - ax))
                lines.append(
                    f'    <geom type="box" pos="{cx_:.4f} {cy_:.4f} {wh:.4f}" '
                    f'size="{seg/2:.4f} {wt:.4f} {wh:.4f}" euler="0 0 {ang:.2f}" '
                    f'rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1" '
                    f'contype="1" conaffinity="1"/>'
                )

        # Robot bodies — one per agent, dynamic (freejoint), contacts enabled.
        # Each gets its own freejoint (root_i), collision cylinder, directional marker,
        # and front camera (front_cam_i).  Initial x-offset spreads robots apart at spawn.
        for i in range(n_agents):
            r, g, b = _ROBOT_COLORS[i % len(_ROBOT_COLORS)]
            lines += [
                f'    <body name="robot_{i}" pos="{i*0.5:.3f} 0 {_ROBOT_H:.4f}">',
                f'      <freejoint name="root_{i}"/>',
                f'      <geom type="cylinder" size="{br:.4f} {_ROBOT_H:.4f}"',
                f'            density="500" rgba="{r:.2f} {g:.2f} {b:.2f} 1"',
                f'            contype="1" conaffinity="1" friction="0.5 0.1 0.1"/>',
                f'      <geom type="box" pos="{br*0.8:.4f} 0 {_ROBOT_H:.4f}"',
                f'            size="{br*0.35:.4f} {br*0.12:.4f} {_ROBOT_H*0.6:.4f}"',
                f'            rgba="1.0 0.3 0.3 1" contype="0" conaffinity="0"/>',
                f'      <camera name="front_cam_{i}" pos="{br*0.7:.4f} 0 {_ROBOT_H:.4f}"',
                f'              xyaxes="0 -1 0 0 0 1" fovy="90"/>',
                f'    </body>',
            ]
        lines += [
            f'  </worldbody>',
            f'</mujoco>',
        ]
        return '\n'.join(lines)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def rebuild(self, world, sim_cfg, bot_pos=None, n_agents=None):
        """Regenerate model from world state. Restarts viewer if it was open."""
        if n_agents is not None:
            self._n_agents = max(1, n_agents)

        viewer_was_open = self._viewer is not None and self._viewer.is_running()
        if viewer_was_open:
            self._stop_passive_viewer()

        xml = self._build_xml(world, sim_cfg, self._n_agents)
        with self._lock:
            new_model = mujoco.MjModel.from_xml_string(xml)
            new_data  = mujoco.MjData(new_model)
            for attr in ('_renderer', '_overhead_renderer'):
                r = getattr(self, attr)
                if r is not None:
                    r.close()
                    setattr(self, attr, None)
            self.model = new_model
            self.data  = new_data
            self._robot_body_ids = [
                mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_BODY, f'robot_{i}')
                for i in range(self._n_agents)
            ]
            self._robot_body_id = self._robot_body_ids[0] if self._robot_body_ids else -1
            if bot_pos is not None:
                if bot_pos and isinstance(bot_pos[0], (list, tuple)):
                    for i, pos in enumerate(bot_pos):
                        if i < self._n_agents:
                            self._sync_robot(pos, i)
                else:
                    self._sync_robot(bot_pos, 0)
                mujoco.mj_forward(self.model, self.data)

        if viewer_was_open:
            self.launch_viewer()

    def reset(self, bot_pos_or_list):
        """Teleport robots to given positions and zero velocities. Call after simulator reset.
        Accepts a flat [x, y, theta] (single agent) or [[x,y,t], ...] (multi-agent)."""
        with self._lock:
            if bot_pos_or_list and isinstance(bot_pos_or_list[0], (list, tuple)):
                for i, pos in enumerate(bot_pos_or_list):
                    if i < self._n_agents:
                        self._sync_robot(pos, i)
            else:
                self._sync_robot(bot_pos_or_list, 0)
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def launch_viewer(self):
        """Open the passive MuJoCo 3D viewer. No-op if already open."""
        if self._viewer is not None and self._viewer.is_running():
            return
        self._stop_viewer.clear()
        self._viewer_thread = threading.Thread(target=self._run_viewer, daemon=True)
        self._viewer_thread.start()

    def viewer_is_open(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def _run_viewer(self):
        with self._lock:
            model = self.model
            data  = self.data
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance  = model.stat.extent * 2.5
            viewer.cam.elevation = -45
            viewer.cam.azimuth   = 135
            viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
            self._viewer = viewer
            _last_sync = 0.0
            while viewer.is_running() and not self._stop_viewer.is_set():
                now = time.perf_counter()
                if now - _last_sync >= _VIEWER_SYNC_DT:
                    viewer.sync()   # no lock — reads only; visual tears acceptable
                    _last_sync = now
                time.sleep(0.005)
        self._viewer = None

    def _stop_passive_viewer(self):
        self._stop_viewer.set()
        if self._viewer_thread is not None:
            self._viewer_thread.join(timeout=2.0)
        self._viewer = None

    def close(self):
        """Release all GPU/GL resources."""
        self._stop_passive_viewer()
        with self._lock:
            for attr in ('_renderer', '_overhead_renderer'):
                r = getattr(self, attr)
                if r is not None:
                    r.close()
                    setattr(self, attr, None)

    # ── Drive helpers ─────────────────────────────────────────────────────────

    def _sync_robot(self, bot_pos, agent_idx=0):
        """Write bot_pos into MuJoCo qpos for the given agent (no mj_forward — caller's responsibility)."""
        x, y, theta = bot_pos
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'root_{agent_idx}')
        offset = int(self.model.jnt_qposadr[jnt_id]) if jnt_id >= 0 else agent_idx * 7
        self.data.qpos[offset]   = x
        self.data.qpos[offset+1] = y
        self.data.qpos[offset+2] = _ROBOT_H
        self.data.qpos[offset+3] = np.cos(theta / 2)
        self.data.qpos[offset+4] = 0.0
        self.data.qpos[offset+5] = 0.0
        self.data.qpos[offset+6] = np.sin(theta / 2)

    def _integrate_drive(self, bot_pos, mL, mR, sim_cfg):
        """Differential drive kinematics. Mutates bot_pos in-place."""
        conv  = (MAX_SPEED_MS / 100.0) * sim_cfg.motor_gain
        v     = ((mL + mR) * conv) / 2.0
        omega = ((mR - mL) * conv) / (sim_cfg.body_radius * 2.0)
        theta = bot_pos[2]
        bot_pos[0] += v * np.cos(theta) * sim_cfg.dt
        bot_pos[1] += v * np.sin(theta) * sim_cfg.dt
        bot_pos[2]  = (theta + omega * sim_cfg.dt) % (2 * np.pi)

    def _resolve_contacts(self, bot_pos, agent_idx=0):
        """Push the given agent's robot out of any penetrating contacts.
        Caller must call _sync_robot + mj_forward before this."""
        rid = (self._robot_body_ids[agent_idx]
               if agent_idx < len(self._robot_body_ids) else self._robot_body_id)
        for _ in range(5):
            if self.data.ncon == 0:
                break
            moved = False
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if c.dist >= 0.0:
                    continue
                b1 = self.model.geom_bodyid[c.geom1]
                b2 = self.model.geom_bodyid[c.geom2]
                if b1 != rid and b2 != rid:
                    continue
                # Push robot away from contact point (contact.pos is in world frame)
                rx = bot_pos[0] - c.pos[0]
                ry = bot_pos[1] - c.pos[1]
                d  = np.hypot(rx, ry)
                if d > 1e-9:
                    pen = -c.dist
                    bot_pos[0] += (rx / d) * pen
                    bot_pos[1] += (ry / d) * pen
                    moved = True
            if not moved:
                break
            self._sync_robot(bot_pos, agent_idx)
            mujoco.mj_forward(self.model, self.data)

    # ── Camera ────────────────────────────────────────────────────────────────

    def _get_renderer(self, width: int, height: int) -> 'mujoco.Renderer':
        r = self._renderer
        if r is None or r.width != width or r.height != height:
            if r is not None:
                r.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderer

    def _get_overhead_renderer(self, width: int, height: int) -> 'mujoco.Renderer':
        r = self._overhead_renderer
        if r is None or r.width != width or r.height != height:
            if r is not None:
                r.close()
            self._overhead_renderer = mujoco.Renderer(self.model, height=height, width=width)
        return self._overhead_renderer

    def render_overhead(self, width: int = 512, height: int = 512) -> np.ndarray:
        """Render a top-down view from overhead_cam. Returns (height, width, 3) uint8.

        The image covers exactly [-arena_scale, +arena_scale] on both axes when
        width == height (square render).  Row 0 = world +Y (north); row H-1 = -Y (south).
        """
        with self._lock:
            renderer = self._get_overhead_renderer(width, height)
            renderer.update_scene(self.data, camera='overhead_cam')
            return renderer.render().copy()

    def _render_camera(self, sensor: CameraSensor, agent_idx: int = 0) -> np.ndarray:
        """Render a robot's front camera. Returns (H, W, 3) float32 [0, 1]."""
        W = sensor.width
        H = max(sensor.height, 32)
        renderer = self._get_renderer(W, H)

        cam_name = f'front_cam_{agent_idx}'
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id >= 0:
            self.model.cam_fovy[cam_id] = float(np.degrees(sensor.fov))
            # Apply center_angle by rotating camera axes around robot Z.
            # Baseline: cam_X=(0,-1,0), cam_Y=(0,0,1), cam_Z=(-1,0,0) → looks +X_body.
            ca   = float(sensor.center_angle)
            c, s = np.cos(ca), np.sin(ca)
            self.model.cam_mat0[cam_id] = np.array([
                [ s, -c,  0],
                [ 0,  0,  1],
                [-c, -s,  0],
            ], dtype=np.float64).flatten()

        renderer.update_scene(self.data, camera=cam_name)
        return renderer.render()[::-1].copy().astype(np.float32) / 255.0

    # ── Public API ────────────────────────────────────────────────────────────

    def tick_physics_batch(self, agent_list, world, sim_cfg, overrides=None):
        """
        Physics step for all agents simultaneously.

        agent_list : [(bot_pos, brain, sensors, circuit), ...]
        overrides  : [motor_override_or_None, ...]  — one per agent

        All positions are written to qpos before the single mj_forward call so
        MuJoCo resolves inter-agent contacts correctly in one shot.

        Returns: list of raw dicts (one per agent, same format as tick_physics).
        """
        n = len(agent_list)
        if overrides is None:
            overrides = [None] * n
        orig_fixate = sim_cfg.fixate_robot

        # Phase 1 — sensor sampling + brain.loop per agent (analytical).
        # fixate_robot=1 suppresses 2D kinematic integration inside _tick_2d.
        raws = []
        for i, (bot_pos, brain, sensors, circuit) in enumerate(agent_list):
            other_sensors = [s for s in sensors if not isinstance(s, CameraSensor)]
            sim_cfg.fixate_robot = 1.0
            try:
                raw = _tick_2d(bot_pos, brain, other_sensors, world, sim_cfg,
                               circuit=circuit, motor_override=overrides[i])
            finally:
                sim_cfg.fixate_robot = orig_fixate
            raws.append(raw)

        with self._lock:
            self.model.opt.timestep = sim_cfg.dt
            if orig_fixate < 0.5:
                # Phase 2 — integrate all drives; sync all positions simultaneously.
                for i, (bot_pos, brain, sensors, circuit) in enumerate(agent_list):
                    self._integrate_drive(bot_pos, raws[i]['mL'], raws[i]['mR'], sim_cfg)
                    self._sync_robot(bot_pos, i)
                # One mj_forward for all bodies — all inter-agent contacts resolved together.
                mujoco.mj_forward(self.model, self.data)
                # Phase 3 — resolve penetrations per agent (may each call mj_forward again).
                for i, (bot_pos, brain, sensors, circuit) in enumerate(agent_list):
                    self._resolve_contacts(bot_pos, i)
            else:
                for i, (bot_pos, brain, sensors, circuit) in enumerate(agent_list):
                    self._sync_robot(bot_pos, i)
                mujoco.mj_forward(self.model, self.data)

        return raws

    def tick_physics(self, bot_pos, brain, sensors, world, sim_cfg,
                     circuit=None, motor_override=None) -> dict:
        """
        MuJoCo-authoritative physics step.

        Sensor sampling and brain.loop are handled analytically (gradients, touch, etc.)
        by calling _tick_2d with fixate_robot=1 so no movement occurs there.
        Kinematic integration and collision resolution are then done against MuJoCo
        geometry via mj_forward + contact data, so the robot bounces off walls and
        objects that exist only in the MuJoCo model (no 2D scene math needed).
        """
        other_sensors = [s for s in sensors if not isinstance(s, CameraSensor)]

        # Sensor sampling + brain.loop only (fixate_robot=1 suppresses 2D movement).
        orig_fixate = sim_cfg.fixate_robot
        sim_cfg.fixate_robot = 1.0
        try:
            raw = _tick_2d(bot_pos, brain, other_sensors, world, sim_cfg,
                           circuit=circuit, motor_override=motor_override)
        finally:
            sim_cfg.fixate_robot = orig_fixate

        with self._lock:
            self.model.opt.timestep = sim_cfg.dt
            if orig_fixate < 0.5:
                self._integrate_drive(bot_pos, raw['mL'], raw['mR'], sim_cfg)
                self._sync_robot(bot_pos, 0)
                mujoco.mj_forward(self.model, self.data)
                self._resolve_contacts(bot_pos, 0)
            else:
                self._sync_robot(bot_pos, 0)
                mujoco.mj_forward(self.model, self.data)

        return raw

    def render_cameras(self, brain, sensors, agent_idx: int = 0) -> None:
        """Render all CameraSensors for the given agent and inject results into brain attributes.
        Call once per display frame, not once per physics tick."""
        import numpy as _np
        cam_sensors = [s for s in sensors if isinstance(s, CameraSensor)]
        if not cam_sensors:
            return
        with self._lock:
            for sensor in cam_sensors:
                frame = self._render_camera(sensor, agent_idx)  # (H, W, 3) HWC float32 [0,1]
                is_rgb = getattr(sensor, 'in_ch', 1) == 3
                # Store _last_frame in the sensor's native format so the visualiser
                # can display it correctly (grayscale thumbnail for GrayCameraSensor).
                sensor._last_frame = frame if is_rgb else _np.mean(frame, axis=-1)
                row = frame[0]
                out = (row.reshape(-1) if is_rgb
                       else _np.mean(row, axis=-1)).astype(_np.float32)
                setattr(brain, sensor.name, out)
                if getattr(sensor, 'lateralized', False):
                    mid     = sensor.width // 2
                    ovl     = getattr(sensor, 'overlap', 0)
                    l_end   = int(_np.clip(mid + ovl, 0, sensor.width))
                    r_start = int(_np.clip(mid - ovl, 0, sensor.width))
                    if is_rgb:
                        # CHW so _conv_forward gets planar channels, not HWC-interleaved.
                        sensor._left_output  = frame[:, :l_end,   :].transpose(2, 0, 1).reshape(-1).astype(_np.float32)
                        sensor._right_output = frame[:, r_start:, :].transpose(2, 0, 1).reshape(-1).astype(_np.float32)
                    else:
                        luma = _np.mean(frame, axis=-1)
                        sensor._left_output  = luma[:, :l_end  ].reshape(-1).astype(_np.float32)
                        sensor._right_output = luma[:, r_start:].reshape(-1).astype(_np.float32)
                    setattr(brain, sensor.name + '_L', sensor._left_output)
                    setattr(brain, sensor.name + '_R', sensor._right_output)
