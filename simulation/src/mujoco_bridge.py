import time
import threading
import numpy as np
import mujoco
import mujoco.viewer


class MuJocoBridge:
    """
    Passive MuJoCo window that mirrors the 2D LBP simulator state.
    The 2D sim remains source of truth; this class is a read-only visualiser.

    Thread model:
      - Viewer runs in a daemon thread (GLFW has its own event loop).
      - update() is called from the Qt main thread each display frame.
      - A lock serialises data writes against viewer.sync() reads.
    """

    def __init__(self, arena_scale: float):
        self._scale = arena_scale
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._viewer = None
        self._ready = threading.Event()

        xml = self._build_xml(arena_scale)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self._thread = threading.Thread(target=self._run_viewer, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=8.0)

    # ── XML ───────────────────────────────────────────────────────────────────

    def _build_xml(self, s: float) -> str:
        return f"""
<mujoco model="braitenberg">
  <option gravity="0 0 0" integrator="RK4"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.35 0.35 0.35" specular="0 0 0"/>
  </visual>
  <worldbody>
    <light pos="0 0 {s * 3:.1f}" dir="0 0 -1"
           diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1" directional="true"/>
    <geom name="floor" type="plane" size="{s + 1:.1f} {s + 1:.1f} 0.1"
          rgba="0.13 0.13 0.13 1" contype="0" conaffinity="0"/>
    <geom type="box" pos=" {s:.2f} 0 0.1" size="0.05 {s:.2f} 0.1"
          rgba="0.55 0.55 0.55 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="-{s:.2f} 0 0.1" size="0.05 {s:.2f} 0.1"
          rgba="0.55 0.55 0.55 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="0  {s:.2f} 0.1" size="{s:.2f} 0.05 0.1"
          rgba="0.55 0.55 0.55 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="0 -{s:.2f} 0.1" size="{s:.2f} 0.05 0.1"
          rgba="0.55 0.55 0.55 1" contype="0" conaffinity="0"/>
    <body name="robot" pos="0 0 0.05">
      <freejoint name="root"/>
      <geom type="cylinder" size="0.2 0.04"
            rgba="0.2 0.55 1.0 1" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.16 0 0.04" size="0.07 0.025 0.025"
            rgba="1.0 0.3 0.3 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

    # ── Viewer thread ─────────────────────────────────────────────────────────

    def _run_viewer(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = self._scale * 2.5
            viewer.cam.elevation = -90
            viewer.cam.azimuth = 0
            viewer.cam.lookat[:] = [0.0, 0.0, 0.0]

            self._viewer = viewer
            self._ready.set()

            while viewer.is_running() and not self._stop_event.is_set():
                with self._lock:
                    viewer.sync()
                time.sleep(0.02)

        self._viewer = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, bot_pos: list, world) -> None:
        """Push 2D sim state into the MuJoCo scene. Called from the Qt thread."""
        viewer = self._viewer
        if viewer is None or not viewer.is_running():
            return

        x, y, theta = bot_pos

        with self._lock:
            # Teleport robot (freejoint qpos: x y z | w qx qy qz)
            self.data.qpos[0] = x
            self.data.qpos[1] = y
            self.data.qpos[2] = 0.05
            self.data.qpos[3] = np.cos(theta / 2)   # w
            self.data.qpos[4] = 0.0                  # qx
            self.data.qpos[5] = 0.0                  # qy
            self.data.qpos[6] = np.sin(theta / 2)    # qz (rotation around Z)
            mujoco.mj_forward(self.model, self.data)

            # Rebuild world geoms from patches and objects
            scn = viewer.user_scn
            scn.ngeom = 0
            identity = np.eye(3).flatten()

            for patch in world.patches:
                if scn.ngeom >= scn.maxgeom:
                    break
                if patch.get('type') == 'wall':
                    continue  # wall patches have no single position
                color = patch.get('color', [1.0, 0.0, 0.0])
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=np.array([patch['r'] * 0.7, patch['r'] * 0.7, 0.001]),
                    pos=np.array([patch['x'], patch['y'], 0.0005]),
                    mat=identity,
                    rgba=np.array([color[0], color[1], color[2], 0.18],
                                  dtype=np.float32),
                )
                scn.ngeom += 1

            for obj in world.objects:
                if scn.ngeom >= scn.maxgeom:
                    break
                color = obj.get('color', [0.75, 0.75, 0.75])
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=np.array([obj['r'] * 1.15, obj['r'] * 1.15, 0.18]),
                    pos=np.array([obj['x'], obj['y'], 0.18]),
                    mat=identity,
                    rgba=np.array([color[0], color[1], color[2], 1.0],
                                  dtype=np.float32),
                )
                scn.ngeom += 1

    def close(self) -> None:
        """Signal the viewer thread to exit."""
        self._stop_event.set()
