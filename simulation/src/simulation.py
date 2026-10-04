"""
simulation.py — the simulated world and its agents, advanced one step at a time.

No Qt. Everything that touches a screen, a wall clock, a file or a network
(timer pacing, signals, arena/oscilloscope drawing, logging, network host /
client, real-robot I/O) is the caller's job — in the app that is
SimController; headless runs use it directly (see headless.py).
"""

from sim_engine import step_agents, free_running_cameras


class Simulation:
    """
    world    : World       — arena contents (mutated by tasks and mounted patches)
    sim_cfg  : SimConfig
    registry : AgentRegistry — the agents (bot_pos, brain, circuit) and their groups
    """

    def __init__(self, world, sim_cfg, registry):
        self.world    = world
        self.sim_cfg  = sim_cfg
        self.registry = registry
        self.engine   = None    # MuJoCoEngine — created by start_engine()
        self.task     = None    # optional BaseTask, ticked once per step
        self.time_index = 0
        self.sim_time   = 0.0   # simulated seconds since reset (camera fps schedule)

    @property
    def agents(self):
        return self.registry.agents

    # ── MuJoCo engine lifecycle ───────────────────────────────────────────────

    def start_engine(self):
        """Create the MuJoCo engine (required — it owns movement, contacts and
        cameras). Returns (ok, error_str_or_None)."""
        try:
            from sim_engine_mujoco import MuJoCoEngine, log_collision_sensor_routing
            self.engine = MuJoCoEngine(self.world, self.sim_cfg, n_agents=len(self.agents),
                                       agent_sensors=_agent_collision_sensors(self.agents))
            self.engine.reset([a.bot_pos for a in self.agents])
            log_collision_sensor_routing(self.agents)
            return True, None
        except Exception as e:
            self.engine = None
            return False, str(e)

    def rebuild_engine(self):
        """Rebuild the MuJoCo model after world / agent-count / CollisionSensor
        geometry changes. No-op before the engine has started."""
        if self.engine is None:
            return
        from sim_engine_mujoco import log_collision_sensor_routing
        self.engine.rebuild(self.world, self.sim_cfg, bot_pos=[a.bot_pos for a in self.agents],
                            n_agents=len(self.agents),
                            agent_sensors=_agent_collision_sensors(self.agents))
        log_collision_sensor_routing(self.agents)

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    # ── Stepping ──────────────────────────────────────────────────────────────

    def step(self, motor_for=None):
        """Advance every agent by one step of sim_cfg.dt (sense → think →
        motors → act), then tick the task. motor_for(agent) may return a
        keyboard / network (mL, mR) command for that agent, or None to let its
        brain drive. Returns one raw-signal dict per agent (see step_agents)."""
        self.time_index += 1
        self.sync_mounted_patches()
        if self.engine is None or not self.agents:
            return []
        raws = step_agents(self.agents, self.world, self.sim_cfg, self.engine,
                           self.sim_time, motor_for)
        self.sim_time += self.sim_cfg.dt
        if self.task is not None:
            self.task.tick(self.world, [a.bot_pos for a in self.agents],
                           self.sim_cfg, self.sim_cfg.dt)
        return raws

    def render_free_running_cameras(self):
        """Render every agent's fps == 0 cameras that haven't rendered at the
        current sim time. Called by the caller between batches of steps (once
        per display cycle in the app) — as fast as the caller runs, without
        holding up the per-step sensors. fps > 0 cameras render inside step()."""
        if self.engine is None:
            return
        for i, agent in enumerate(self.agents):
            if agent.brain is None:
                continue
            self.engine.render_cameras(agent.brain,
                                       free_running_cameras(agent.circuit.sensors, self.sim_time),
                                       i, self.sim_time, self.sim_cfg.dt)

    def sync_mounted_patches(self):
        """Sync every mounted gradient patch's position to its carrying robot.
        mounted_on stores a stable agent id (not a list position), so a mount
        can never silently drift onto the wrong robot after some other agent
        is added/removed. Called both per step (before sensing) and right
        after reset() repositions agents, so a mounted patch never renders at
        a stale position."""
        for patch in self.world.patches:
            agent_id = patch.get('mounted_on')
            if agent_id is not None:
                agent = self.registry.agent_by_id(agent_id)
                if agent is not None:
                    patch['x'] = agent.bot_pos[0]
                    patch['y'] = agent.bot_pos[1]

    def reset(self):
        """Back to t = 0: agents to their start positions, joints, brains,
        layers and sensors reset, MuJoCo and the task reset."""
        self.time_index = 0
        self.sim_time   = 0.0
        for i, agent in enumerate(self.agents):
            offset_x = i * 0.5 if i > 0 else 0.0
            agent.bot_pos[:] = [self.sim_cfg.init_x + offset_x, self.sim_cfg.init_y, 0.0]
            agent.trail_xy.clear()
            for joint in agent.circuit.joints:
                joint.angle = 0.0
                joint.vel   = 0.0
            if agent.brain:
                agent.brain.setup()
                for layer in agent.circuit.layers:
                    layer.reset()
                for sensor in agent.circuit.sensors:
                    sensor.reset()
        self.sync_mounted_patches()
        if self.engine is not None:
            self.engine.reset([a.bot_pos for a in self.agents])
        if self.task is not None:
            self.task.reset(self.world, self.sim_cfg)


def _agent_collision_sensors(agents):
    """Per agent, the CollisionSensors MuJoCoEngine builds dedicated sector
    geometry for (see _mujoco_collision_eligible) — passed to
    MuJoCoEngine(...)/.rebuild(...) so the model's sensor geometry stays in
    sync with what's actually configured on each agent's circuit."""
    return [[s for s in getattr(a.circuit, 'sensors', []) if s.uses_mujoco_contacts]
            for a in agents]
