"""
sim_engine_mujoco.py — MuJoCo-backed simulation engine.

MuJoCo runs the physics (mj_step, contacts, collision with objects/walls).
The 2D canvas reads robot position FROM MuJoCo each tick.
Gradient/patch/sky sensors are computed analytically as before (not by
MuJoCo — it has no notion of a gradient field or sky polarization).
CameraSensor renders a real 3D perspective image via mujoco.Renderer.

CollisionSensor (when root-mounted — see _mujoco_collision_eligible) reads
MuJoCo's own contact array directly instead of the analytic geometry check
in sensors.py: each sector of each such sensor gets its own small, purely
detection-only geom built into the robot body (see _build_xml), positioned
and sized to match that sector's real angular coverage and probe radius —
so "is sector i hit" is just "does data.contact contain this specific geom
ID", with no bearing math or angular-footprint approximation needed. This
is both much cheaper than the analytic check (MuJoCo already computes
contacts every tick for physics regardless — see TODO.md Performance) and
behaviorally exact, including multiple sectors firing at once from one
touching object, since MuJoCo's own collision engine (not an approximation
of it) determines whether each sector's geometry actually overlaps
something. These geoms carry gap >> margin so they never contribute
physical force — see _SENSOR_GEOM_GAP. Whenever a CollisionSensor's
geometry-relevant params change, the model must be rebuilt (see
SimulatorApp._rebuild_channels' signature-diffing gate) for the new
geometry to take effect — sensor readings otherwise keep using whatever
geometry was baked in at the last rebuild.

Architecture
------------
  The step itself lives in sim_engine.step_agents(); this engine provides
  the MuJoCo-owned parts of it:
    owns_sensor()              which sensors MuJoCo samples (cameras, eligible
                               CollisionSensors) — everything else is 2-D
    sample_collision_sensors() SENSE: contacts at the current positions
    render_cameras()           SENSE: camera frames (on each camera's fps
                               schedule, or free-running from the display loop)
    move_agents()              ACT: differential drive, then MuJoCo resolves
                               contacts (objects, walls, arena, other agents)

  rebuild(world, sim_cfg, agent_sensors=...):
    Regenerate MuJoCo model XML from current world state (+ CollisionSensor
    geometry, from agent_sensors). Call after any structural world edit
    (add/remove object or wall) or CollisionSensor geometry change.
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

from sim_engine import MAX_SPEED_MS
from sensors import CameraSensor
from texture_manager import texture_exists, texture_bytes

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

# Per-sector CollisionSensor geoms carry margin=0 (detect only genuine overlap
# — no lookahead trick needed here, since a lookahead sensor's geom is simply
# built at the lookahead radius directly) and gap >> margin, which (per
# MuJoCo's contact semantics: active iff dist < margin - gap) keeps them
# permanently non-physical regardless of penetration depth — verified
# empirically (constraint force is exactly zero at gap=5.0 for a body
# penetrating 0.2m; see _scratch/ during development). 100.0 is generously
# larger than any realistic penetration depth at this sim's scale.
_SENSOR_GEOM_GAP = 100.0

# Radial half-thickness of each sector geom. Kept small and specifically
# separate from _WALL_T (which is sized for visible wall thickness, not
# detection precision) — the geom is positioned so its outer face sits
# exactly at the sensor's true probe radius (see the robot-body loop in
# _build_xml), so this value only sets how far its *inner* face reaches
# back toward the robot center. Not load-bearing for accuracy, just needs
# to be small relative to typical body_radius scales (~0.1-0.3m).
_SENSOR_GEOM_THICKNESS = 0.003


def _collision_sensor_geom_name(agent_idx: int, sensor_name: str, sector_idx: int, sub_idx: int) -> str:
    return f"colsens_{agent_idx}_{sensor_name}_{sector_idx}_{sub_idx}"


def _collision_sector_n_pts(arc_angle_deg: float) -> int:
    """Same discretization CollisionSensor._detect_hits uses for its own probe
    count — matching it here means the MuJoCo geometry approximates each
    sector's true arc at the same resolution the analytic method does,
    instead of a single flat chord (which visibly cuts the corner for wide
    arc_angle / large radius, i.e. deviates most exactly where it matters)."""
    return max(5, int(arc_angle_deg / 5))


def _mujoco_collision_eligible(sensor) -> bool:
    """True for sensors that get dedicated per-sector MuJoCo geometry built into
    the robot body (see _build_xml) instead of the analytic geometry check in
    sensors.py — declared by the sensor (CollisionSensor.uses_mujoco_contacts:
    mounted on the root body). Any radius works — the geometry is built at
    exactly the sensor's own probe radius."""
    return bool(sensor.uses_mujoco_contacts)


def log_collision_sensor_routing(agents) -> None:
    """Print, per agent, which CollisionSensors are using MuJoCo's per-sector
    contact geometry vs the analytic path, and why. Called whenever the
    MuJoCo engine is (re)built — the routing decision has no other visible
    effect on the running app, so this is the way to actually confirm it
    from the console rather than take it on faith."""
    from sensors import CollisionSensor
    for i, agent in enumerate(agents):
        for sensor in getattr(agent.circuit, 'sensors', []):
            if not isinstance(sensor, CollisionSensor):
                continue
            body_ids = getattr(sensor, 'body_ids', None)
            if not _mujoco_collision_eligible(sensor):
                print(f"[MuJoCo] agent {i}: CollisionSensor '{sensor.name}' "
                      f"(body_ids={body_ids}) -> analytic path (not mounted on root body)")
            else:
                print(f"[MuJoCo] agent {i}: CollisionSensor '{sensor.name}' "
                      f"(n={sensor.n}, radius={sensor.radius}) -> MuJoCo contact path "
                      f"(per-sector geometry)")


class MuJoCoEngine:
    """
    MuJoCo-backed simulation engine.

    MuJoCo is the physics authority.  The 2D canvas reads robot position
    from this engine each tick.  Gradients are computed analytically.
    """

    def __init__(self, world, sim_cfg, n_agents=1, agent_sensors=None):
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
        self._collision_sector_geom_ids = {}   # (agent_idx, sensor_name) -> [geom_id, ...]; set by rebuild()
        self.rebuild(world, sim_cfg, n_agents=n_agents, agent_sensors=agent_sensors)

    # ── XML ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_xml(world, sim_cfg, n_agents=1, agent_sensors=None):
        """Return (xml_string, assets_dict). assets_dict maps the virtual filenames
        referenced by <texture file="..."/> to raw file bytes, as required by
        mujoco.MjModel.from_xml_string(xml, assets) — there is no XML file on disk
        for relative/texturedir paths to resolve against.

        agent_sensors: optional list of length n_agents; agent_sensors[i] is that
        agent's list of MuJoCo-eligible CollisionSensors (see
        _mujoco_collision_eligible), each of which gets one small,
        non-physical, detection-only geom per sector built into that agent's
        robot body (see _collision_sensor_geom_name). None/omitted means no
        agents have MuJoCo-routed CollisionSensors — e.g. tests that only
        care about world geometry.
        """
        s  = sim_cfg.arena_scale
        br = sim_cfg.body_radius
        dt = sim_cfg.dt
        wh = _WALL_H
        wt = _WALL_T

        # Build a texture-filename -> material-name map, deduped so two objects/
        # walls sharing one file reuse one <texture>/<material> pair (MuJoCo errors
        # on a repeated <texture name=...>). Missing files are silently skipped —
        # callers fall back to rgba (see object/wall loops below).
        #
        # Object/wall geoms use type="cube" (same file repeated for all 6 faces) —
        # MuJoCo's type="2d" mapping on a cylinder/box only wraps the texture's
        # horizontal axis around the side (vertical variation is lost, producing
        # plain vertical stripes instead of the actual pattern); type="cube" maps
        # each face with the full image, which renders correctly on the side.
        # The floor plane keeps type="2d" (its planar tiling via texrepeat is
        # correct and unaffected by this issue).
        tex_names = []
        for obj in world.objects:
            t = obj.get('texture')
            if t and texture_exists(t) and t not in tex_names:
                tex_names.append(t)
        for wall in getattr(world, 'walls', []):
            t = wall.get('texture')
            if t and texture_exists(t) and t not in tex_names:
                tex_names.append(t)

        tex_material = {name: f'mat{i}' for i, name in enumerate(tex_names)}
        assets       = {name: texture_bytes(name) for name in tex_names}

        asset_lines = []
        for name, mat in tex_material.items():
            idx = list(tex_material).index(name)
            asset_lines.append(
                f'    <texture name="tex{idx}" type="cube"'
                f' fileright="{name}" fileleft="{name}" fileup="{name}" filedown="{name}"'
                f' filefront="{name}" fileback="{name}"/>')
            asset_lines.append(
                f'    <material name="{mat}" texture="tex{idx}" texuniform="true" reflectance="0.0"/>')

        floor_tex = getattr(world, 'floor_texture', None)
        if floor_tex and texture_exists(floor_tex):
            assets[floor_tex] = texture_bytes(floor_tex)
            asset_lines.append(f'    <texture name="floor_tex" type="2d" file="{floor_tex}"/>')
            asset_lines.append(
                f'    <material name="floor_mat" texture="floor_tex" texrepeat="4 4" reflectance="0.0"/>')
            floor_material = 'floor_mat'
        else:
            # Default checker floor — unchanged from before when no floor texture is set.
            asset_lines.insert(0,
                f'    <material name="floor_mat" texture="grid" texrepeat="4 4" reflectance="0.0"/>')
            asset_lines.insert(0,
                f'    <texture name="grid" type="2d" builtin="checker"'
                f' rgb1=".52 .52 .52" rgb2=".94 .94 .94" width="512" height="512"/>')
            floor_material = 'floor_mat'

        lines = [
            f'<mujoco model="braitenberg3d">',
            f'  <option gravity="0 0 0" integrator="Euler" timestep="{dt:.4f}"/>',
            f'  <visual>',
            f'    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0 0 0"/>',
            f'    <map znear="0.001"/>',
            f'    <global offwidth="512" offheight="512"/>',
            f'  </visual>',
            f'  <asset>',
            *asset_lines,
            f'  </asset>',
            f'  <worldbody>',
            f'    <light pos="0 0 {s*3:.2f}" dir="0 0 -1"',
            f'           diffuse="0.8 0.8 0.8" specular="0 0 0" directional="true"/>',
            f'    <geom name="floor" type="plane" size="{s+1:.2f} {s+1:.2f} 0.1"',
            f'          material="{floor_material}" contype="0" conaffinity="0"/>',
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
            mat     = tex_material.get(obj.get('texture'))
            style   = f'material="{mat}"' if mat else f'rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1"'
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
                        f' euler="0 0 {ang:.2f}" {style}'
                        f' contype="1" conaffinity="1"/>'
                    )
            else:
                lines.append(
                    f'    <geom type="cylinder"'
                    f' pos="{obj["x"]:.4f} {obj["y"]:.4f} {_OBJ_H:.4f}"'
                    f' size="{r:.4f} {_OBJ_H:.4f}" {style}'
                    f' contype="1" conaffinity="1"/>'
                )

        # Polygon walls → one box per edge (solid)
        for wall in getattr(world, 'walls', []):
            pts = wall['points']
            c   = wall.get('color', [0.5, 0.5, 0.5])
            mat = tex_material.get(wall.get('texture'))
            style = f'material="{mat}"' if mat else f'rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1"'
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
                    f'{style} '
                    f'contype="1" conaffinity="1"/>'
                )

        # Robot bodies — one per agent, dynamic (freejoint), contacts enabled.
        # Each gets its own freejoint (root_i), collision cylinder, directional marker,
        # front camera (front_cam_i), and — when it has eligible CollisionSensors — a
        # nested "robot_{i}_sensors" body holding one small non-physical geom per
        # sector (see _collision_sensor_geom_name / _mujoco_collision_eligible).
        # That sector geometry MUST live on its own body, not directly on robot_i:
        # sitting right at the robot's own surface, it would otherwise overlap the
        # main collision cylinder and register a same-body contact (MuJoCo does not
        # exclude those automatically) — every sector would falsely read "hit" all
        # the time. The nested body + <exclude> below solves that: MuJoCo still
        # detects genuinely external contacts (a different body) normally, since
        # exclude only suppresses this one specific body pair.
        exclude_lines = []
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
            ]
            sensors_i = agent_sensors[i] if agent_sensors and i < len(agent_sensors) else []
            if sensors_i:
                lines.append(f'      <body name="robot_{i}_sensors">')
                for sensor in sensors_i:
                    r_sensor = br * sensor.radius
                    # Box center sits r_sensor - thickness from the robot's own center, so
                    # the box's OUTER face lands exactly at r_sensor (the sensor's true
                    # probe radius) instead of overshooting it by the box's own thickness —
                    # that overshoot previously caused literal-touch (radius=1.0) sensors to
                    # fire before real contact.
                    thickness = _SENSOR_GEOM_THICKNESS
                    radial_c  = max(thickness, r_sensor - thickness)
                    n_pts = _collision_sector_n_pts(np.degrees(sensor.arc_angle))
                    sub_width  = sensor.arc_angle / n_pts
                    sub_half   = max(1e-4, r_sensor * np.sin(sub_width / 2.0))
                    for j, center_a in enumerate(sensor._sensor_centers()):
                        # n_pts small chord segments approximating this sector's true arc —
                        # same discretization CollisionSensor._detect_hits uses for its own
                        # probes (see _collision_sector_n_pts), so the MuJoCo geometry
                        # doesn't "cut the corner" the way a single flat chord across the
                        # whole arc_angle would for wide arcs / large radius.
                        for k in range(n_pts):
                            sub_a = center_a - sensor.arc_angle / 2.0 + (k + 0.5) * sub_width
                            cx, cy = radial_c * np.cos(sub_a), radial_c * np.sin(sub_a)
                            ang_deg = np.degrees(sub_a) + 90.0   # tangential, matching wall convention
                            gname = _collision_sensor_geom_name(i, sensor.name, j, k)
                            lines.append(
                                f'        <geom name="{gname}" type="box"'
                                f' pos="{cx:.4f} {cy:.4f} {_ROBOT_H:.4f}"'
                                f' size="{sub_half:.4f} {thickness:.4f} {_ROBOT_H:.4f}"'
                                f' euler="0 0 {ang_deg:.2f}" rgba="1 1 0 0.3"'
                                f' margin="0" gap="{_SENSOR_GEOM_GAP}"'
                                f' contype="1" conaffinity="1"/>'
                            )
                lines.append(f'      </body>')
                exclude_lines.append(f'    <exclude body1="robot_{i}" body2="robot_{i}_sensors"/>')
            lines.append(f'    </body>')
        lines.append(f'  </worldbody>')
        if exclude_lines:
            lines += [f'  <contact>', *exclude_lines, f'  </contact>']
        lines.append(f'</mujoco>')
        return '\n'.join(lines), assets

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def rebuild(self, world, sim_cfg, bot_pos=None, n_agents=None, agent_sensors=None):
        """Regenerate model from world state (+ CollisionSensor sector geometry
        from agent_sensors, if given — see _build_xml). Restarts viewer if it
        was open. Call after any structural world edit, agent-count change,
        or CollisionSensor geometry-relevant param change (n/angle_spread/
        arc_angle/radius/body_ids) — the new geometry only takes effect from
        the next rebuild onward."""
        if n_agents is not None:
            self._n_agents = max(1, n_agents)

        viewer_was_open = self._viewer is not None and self._viewer.is_running()
        if viewer_was_open:
            self._stop_passive_viewer()

        xml, assets = self._build_xml(world, sim_cfg, self._n_agents, agent_sensors=agent_sensors)
        with self._lock:
            new_model = mujoco.MjModel.from_xml_string(xml, assets)
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
            self._collision_sector_geom_ids = {}
            for i, sensors_i in enumerate(agent_sensors or []):
                for sensor in sensors_i:
                    n_pts = _collision_sector_n_pts(np.degrees(sensor.arc_angle))
                    # One sub-list of sub-geom ids per sector — a sector is "hit" if ANY
                    # of its sub-arc segments has a contact (see sample_collision_sensors).
                    sector_ids = [
                        [mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_GEOM,
                                            _collision_sensor_geom_name(i, sensor.name, j, k))
                         for k in range(n_pts)]
                        for j in range(sensor.n)
                    ]
                    self._collision_sector_geom_ids[(i, sensor.name)] = sector_ids
            if bot_pos:
                if isinstance(bot_pos[0], (list, tuple)):
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
        Accepts a flat [x, y, theta] (single agent), [[x,y,t], ...] (multi-agent), or an
        empty list (no local agents, e.g. a bare network host) — in which case the
        placeholder robot body MuJoCo always keeps (_n_agents is clamped to >= 1) is left
        at its default spawn pose."""
        with self._lock:
            if bot_pos_or_list:
                if isinstance(bot_pos_or_list[0], (list, tuple)):
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
        """Render a robot's front camera. Returns (sensor.height, W, 3) float32 [0, 1]."""
        W = sensor.width
        H = max(sensor.height, 32)   # rendered at >= 32 rows, then row-sampled below
        renderer = self._get_renderer(W, H)

        cam_name = f'front_cam_{agent_idx}'
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id >= 0:
            self.model.cam_fovy[cam_id] = float(np.degrees(sensor.fov))
            # Build camera orientation in body frame.
            # Rows = [cam_right, cam_up, cam_backward] expressed in the robot body frame.
            # Baseline (ca=0, va=0): right=-Y_body, up=+Z_body, back=-X_body → looks +X_body.
            # Yaw by ca around body-Z, then pitch-down by va around the yawed right axis.
            ca   = float(sensor.center_angle)
            va   = float(getattr(sensor, 'vertical_angle', 0.0))
            c,  s  = np.cos(ca), np.sin(ca)
            cv, sv = np.cos(va), np.sin(va)
            cam_body_mat = np.array([
                [ s,        -c,         0 ],   # right  (yaw only)
                [ c * sv,    s * sv,    cv],   # up     (tilts back as camera noses down)
                [-c * cv,   -s * cv,    sv],   # back   (at va=90° points +Z_body = straight down)
            ], dtype=np.float64)
            # MuJoCo's cam_mat0 is read-only after model load — mj_forward ignores runtime changes.
            # Instead, write cam_xmat directly: world_cam = body_xmat @ cam_body_mat.T
            # (transpose because MuJoCo stores rotation matrices column-major: columns = axes).
            body_id   = int(self.model.cam_bodyid[cam_id])
            body_xmat = self.data.xmat[body_id].reshape(3, 3)
            self.data.cam_xmat[cam_id] = (body_xmat @ cam_body_mat.T).flatten()

        renderer.update_scene(self.data, camera=cam_name)
        frame = renderer.render()[::-1].astype(np.float32) / 255.0
        if H != sensor.height:
            # Pick each output row from the centre of its band, so a 1-row
            # camera sees the horizon row, not the top of the image.
            rows = ((np.arange(sensor.height) + 0.5) * H / sensor.height).astype(int)
            frame = frame[rows]
        return frame

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def owns_sensor(sensor) -> bool:
        """True for sensors MuJoCo samples (cameras, sensors reading MuJoCo
        contacts); every other sensor is sampled by the 2-D field code in sim_engine."""
        return sensor.is_camera or sensor.uses_mujoco_contacts

    def move_agents(self, bot_positions, commands, sim_cfg) -> None:
        """ACT phase: drive every agent from its (mL, mR) command, then let MuJoCo
        resolve contacts. Mutates each bot_pos in place.

        All positions are written to qpos before the single mj_forward call so
        MuJoCo resolves inter-agent contacts correctly in one shot. With
        sim_cfg.fixate_robot set, robots stay where they are (debug option)."""
        with self._lock:
            self.model.opt.timestep = sim_cfg.dt
            if sim_cfg.fixate_robot < 0.5:
                for i, (bot_pos, (mL, mR)) in enumerate(zip(bot_positions, commands)):
                    self._integrate_drive(bot_pos, mL, mR, sim_cfg)
                    self._sync_robot(bot_pos, i)
                # One mj_forward for all bodies — all inter-agent contacts resolved together.
                mujoco.mj_forward(self.model, self.data)
                # Resolve penetrations per agent (may each call mj_forward again).
                for i, bot_pos in enumerate(bot_positions):
                    self._resolve_contacts(bot_pos, i)
            else:
                for i, bot_pos in enumerate(bot_positions):
                    self._sync_robot(bot_pos, i)
                mujoco.mj_forward(self.model, self.data)

    def render_cameras(self, brain, cam_sensors, agent_idx: int, t: float, sim_dt: float) -> None:
        """SENSE phase for cameras: render each given CameraSensor for one agent at
        sim time t and store its processed output as brain.<name> (plus raw
        brain.<name>_L / _R halves when lateralized). Which cameras to render
        is the caller's decision — see sim_engine.due_cameras /
        free_running_cameras."""
        if not cam_sensors:
            return
        with self._lock:
            frames = [self._render_camera(sensor, agent_idx) for sensor in cam_sensors]
        for sensor, frame in zip(cam_sensors, frames):
            setattr(brain, sensor.name, sensor.process_frame(frame, t, sim_dt))
            sensor.publish_halves(brain)

    def sample_collision_sensors(self, brain, sensors, sim_cfg, agent_idx: int = 0) -> None:
        """Populate eligible CollisionSensors' outputs from MuJoCo's own contact
        array instead of the analytic 2-D geometry check in sensors.py.

        Each eligible sensor's sectors already have dedicated, correctly
        positioned/sized, non-physical geoms baked into the robot body (see
        _build_xml / _mujoco_collision_eligible) — so "is sector i hit" is
        just "does data.contact contain that specific geom id", with no
        bearing math or angular-footprint approximation needed. MuJoCo's own
        collision engine (not an approximation of it) determines whether each
        sector's real geometry overlaps a wall/object/other-agent, which is
        also why multiple sectors correctly fire at once for one touching
        object, the same way the analytic multi-probe check does.

        This is also much cheaper than the analytic check — MuJoCo already
        computes every contact each tick as an unavoidable part of physics,
        whether or not any sensor reads it (see TODO.md Performance,
        _scratch/measure_collision_backends.py).

        Contacts read here reflect the mj_forward call from the END of the
        previous step's move_agents() — exactly the robots' current positions,
        since nothing has moved yet in this step's SENSE phase — so no extra
        mj_forward is needed, and the brain sees contacts in the same step as
        every other sensor.

        Only sensors _mujoco_collision_eligible() accepts, AND that already
        have geometry built for them (i.e. survived since the last rebuild —
        see the (agent_idx, sensor.name) lookup below), are handled here;
        callers must still run everything else through the analytic path.
        """
        col_sensors = [s for s in sensors if s.uses_mujoco_contacts]
        if not col_sensors:
            return
        with self._lock:
            touched_geoms = set()
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                touched_geoms.add(c.geom1)
                touched_geoms.add(c.geom2)
        for sensor in col_sensors:
            sector_ids = self._collision_sector_geom_ids.get((agent_idx, sensor.name))
            if sector_ids is None:
                continue   # geometry not built yet (rebuild pending) — leave brain.<name> as-is
            # Each sector is a list of sub-arc-segment geom ids (see _build_xml /
            # _collision_sector_n_pts) — hit iff ANY of its own segments touched something.
            hit_arr = np.array([
                1.0 if any(gid in touched_geoms for gid in sub_ids) else 0.0
                for sub_ids in sector_ids
            ])
            setattr(brain, sensor.name, sensor._finalize_hits(hit_arr, sim_cfg))
