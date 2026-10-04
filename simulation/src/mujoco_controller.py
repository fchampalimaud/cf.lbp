"""
mujoco_controller.py — MuJoCo display concerns for the 2D simulator.

The engine itself belongs to Simulation (simulation.py); this class only adds
what the app's display needs on top of it: the "Top view" overhead image in
the arena, the interactive 3-D viewer, and cheap repositioning while editing.
"""


class MuJoCoController:
    def __init__(self, arena, sim_cfg, sim):
        self._arena = arena
        self.sim_cfg = sim_cfg
        self._sim = sim
        self.view_3d = False

    @property
    def engine(self):
        return self._sim.engine

    @property
    def enabled(self):
        return self._sim.engine is not None

    def start(self):
        """Create the MuJoCo engine. Returns (ok, error_str_or_None)."""
        return self._sim.start_engine()

    def rebuild(self):
        """Rebuild the MuJoCo model after world/agent-count changes. No-op
        before the engine has started. Catches its own errors (print +
        continue) so callers — e.g. AgentRegistry's on_agents_changed
        callback — don't need MuJoCo-specific error handling."""
        try:
            self._sim.rebuild_engine()
        except Exception as e:
            print(f"[MuJoCo] rebuild error: {e}")

    def render_overhead(self):
        """Push an overhead render to the arena when in 3D view mode."""
        if self.engine is not None and self.view_3d:
            rgb = self.engine.render_overhead(512, 512)
            self._arena.set_overhead_frame(rgb, self.sim_cfg.arena_scale)

    def reposition_robots(self, agents):
        """Cheap MuJoCo sync while editing (e.g. dragging a robot): teleport
        robots to their current bot_pos and re-render the overhead frame,
        without rebuilding the model. Keeps the overhead camera image aligned
        with the live-dragged robot marker instead of only updating on drop."""
        if self.engine is None:
            return
        try:
            self.engine.reset([a.bot_pos for a in agents])
            self.render_overhead()
        except Exception as e:
            print(f"[MuJoCo] reposition error: {e}")

    def show_viewer(self):
        if self.engine is not None:
            self.engine.launch_viewer()

    def set_view_3d(self, enabled):
        self.view_3d = enabled
        self._arena.set_3d_mode(enabled)
        if enabled and self.engine is not None:
            rgb = self.engine.render_overhead(512, 512)
            self._arena.set_overhead_frame(rgb, self.sim_cfg.arena_scale)

    def close(self):
        self._sim.close()
