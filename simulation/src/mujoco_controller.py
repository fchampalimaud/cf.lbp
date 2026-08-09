"""
mujoco_controller.py — MuJoCo engine lifecycle and rendering for the 2D simulator.

Takes agents/world as method arguments rather than storing them, since the
live agent list is owned by AgentRegistry and the world by SimController —
this class only needs them transiently per call.
"""


def _agent_collision_sensors(agents):
    """Per agent, the list of CollisionSensors MuJoCoEngine should build
    dedicated sector geometry for (see _mujoco_collision_eligible) — passed
    to MuJoCoEngine(...)/​.rebuild(...) so the model's sensor geometry stays
    in sync with what's actually configured on each agent's circuit."""
    from sensors import CollisionSensor
    from sim_engine_mujoco import _mujoco_collision_eligible
    return [
        [s for s in getattr(a.circuit, 'sensors', [])
         if isinstance(s, CollisionSensor) and _mujoco_collision_eligible(s)]
        for a in agents
    ]


class MuJoCoController:
    def __init__(self, arena, sim_cfg):
        self._arena = arena
        self.sim_cfg = sim_cfg
        self.engine = None
        self.view_3d = False

    @property
    def enabled(self):
        return self.engine is not None

    def enable(self, state, world, agents):
        """Enable or disable the MuJoCo engine. Returns (ok, error_str_or_None)."""
        if state:
            try:
                from sim_engine_mujoco import MuJoCoEngine, log_collision_sensor_routing
                self.engine = MuJoCoEngine(world, self.sim_cfg, n_agents=len(agents),
                                            agent_sensors=_agent_collision_sensors(agents))
                self.engine.reset([a.bot_pos for a in agents])
                log_collision_sensor_routing(agents)
                return True, None
            except Exception as e:
                self.engine = None
                return False, str(e)
        else:
            if self.engine is not None:
                self.engine.close()
                self.engine = None
            if self.view_3d:
                self.view_3d = False
                self._arena.set_3d_mode(False)
            return True, None

    def rebuild(self, world, agents):
        """Rebuild the MuJoCo model after world/agent-count changes. No-op
        when the engine is off. Catches its own errors (print + continue) so
        callers — e.g. AgentRegistry's on_agents_changed callback — don't need
        MuJoCo-specific error handling."""
        if self.engine is None:
            return
        try:
            from sim_engine_mujoco import log_collision_sensor_routing
            all_pos = [a.bot_pos for a in agents]
            self.engine.rebuild(world, self.sim_cfg, bot_pos=all_pos, n_agents=len(agents),
                                 agent_sensors=_agent_collision_sensors(agents))
            log_collision_sensor_routing(agents)
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
        if self.engine is not None:
            self.engine.close()
            self.engine = None
