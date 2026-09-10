"""
sim_controller.py — the core simulation loop, composed from four focused
collaborators: AgentRegistry (agent_registry.py), NetworkController
(network_controller.py), RobotModeController (robot_mode_controller.py), and
MuJoCoController (mujoco_controller.py).

SimController itself owns only genuinely core-loop concerns: the QTimer,
running/timing state, and the per-frame/per-tick orchestration that calls into
the four collaborators. Every method/property the collaborators used to expose
directly on SimController is kept as a thin delegating wrapper, so existing
external code (LBPSimulator.py, sim_app_brain.py, sim_app_session.py,
network_viz_actions.py, world_editor.py) that reaches into e.g.
self._sim_ctrl._agents / ._net_host / ._mujoco_engine / .add_agent(...) keeps
working unchanged. Migrating those callers to address self._sim_ctrl.registry/
.network/.robot/.mujoco directly is a deliberate, separate follow-up — see
TODO.md.
"""

import time
import numpy as np

from PySide6.QtCore import QObject, QTimer, Qt, Signal

from sim_constants import C
from sim_engine import tick_physics
from agent_registry import AgentRegistry
from network_controller import NetworkController
from robot_mode_controller import RobotModeController
from mujoco_controller import MuJoCoController


class SimController(QObject):
    """
    Drives the simulation: owns the QTimer and per-frame orchestration across
    the agent registry, networking, real-robot mode, and MuJoCo collaborators.

    Parameters
    ----------
    circuit             : CircuitModel       — initial agent's circuit
    sim_cfg             : SimConfig          — simulation parameters (shared reference)
    world               : World             — arena contents (shared reference)
    arena               : ArenaWidget       — for per-frame display updates
    osc_ctrl            : OscChannelManager — channel list, trace data, multiplier cache
    logger              : SimLogger         — data recording
    get_trail_visible   : callable → bool   — whether to show/accumulate the robot trail
    get_motor_override  : callable → (mL, mR) | None  — manual control hook; None = brain drives
    brain_mgr           : BrainManager      — initial agent's brain manager
    """

    sig_status_changed = Signal(str, str)   # (label_text, css_color)
    sig_timing_updated = Signal(str)        # timing label text
    sig_frame_ready     = Signal(object, object, object, object, bool, object)
        # (agents, selected_agent_id, sim_cfg, world, trail_visible, overhead_rgb)
        # once per rendered frame — NOT once per physics substep (see sig_tick_values,
        # used for oscilloscope trace-appending, which needs full physics-rate resolution)
    sig_tick_values     = Signal(object)     # dict[str, float] — once per physics substep

    def __init__(self, circuit, sim_cfg, world, arena, osc_ctrl, logger,
                 get_trail_visible, get_motor_override, parent=None,
                 brain_mgr=None):
        super().__init__(parent)

        self.sim_cfg   = sim_cfg
        self.world     = world
        self._arena    = arena
        self._osc_ctrl = osc_ctrl
        self._logger   = logger
        self._trail_visible  = get_trail_visible
        self._motor_override = get_motor_override

        self.registry = AgentRegistry(sim_cfg, circuit, brain_mgr)
        self.network  = NetworkController(parent=self)
        self.robot    = RobotModeController(sim_cfg, get_motor_override)
        self.mujoco   = MuJoCoController(arena, sim_cfg)

        # Resolve the bidirectional writes the old monolithic class had
        # between the registry and MuJoCo/networking via explicit callbacks,
        # instead of the registry reaching directly into either.
        self.registry.on_agents_changed = lambda: self.mujoco.rebuild(self.world, self.registry.agents)
        self.registry.on_agent_removed  = self.network.forget_agent
        self.network.get_running        = lambda: self.running

        self._active_task = None

        self.running    = False
        self.time_index = 0
        self.speed_mult = 1
        self._rt_mode   = False
        self._time_debt = 0.0
        self._phys_ms_acc = 0.0
        self._last_loop_t = None
        self._frame_count = 0

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._loop)

    # ── Backward-compatible delegates: agent registry ───────────────────────

    @property
    def _agents(self):
        return self.registry.agents

    @property
    def _selected_id(self):
        return self.registry.selected_id

    @property
    def _agent(self):
        return self.registry.agent

    @property
    def _groups(self):
        return self.registry._groups

    @property
    def circuit(self):
        return self.registry.circuit

    @property
    def brain_mgr(self):
        return self.registry.brain_mgr

    @property
    def brain(self):
        return self.registry.brain

    @brain.setter
    def brain(self, value):
        self.registry.brain = value

    @property
    def bot_pos(self):
        return self.registry.bot_pos

    @property
    def trail_xy(self):
        return self.registry.trail_xy

    def _agent_by_id(self, agent_id):
        return self.registry._agent_by_id(agent_id)

    def index_of_agent(self, agent_id):
        return self.registry.index_of_agent(agent_id)

    def add_agent(self, circuit, brain_mgr, name=None, color=None) -> int:
        return self.registry.add_agent(circuit, brain_mgr, name=name, color=color)

    def remove_agent(self, agent_id: int):
        self.registry.remove_agent(agent_id, allow_empty=self.network.is_hosting)

    def select_agent(self, agent_id: int):
        self.registry.select_agent(agent_id)

    def create_group(self, module, color, name, first_agent_id) -> int:
        return self.registry.create_group(module, color, name, first_agent_id)

    def add_agent_to_group(self, group_id: int, agent_id: int):
        self.registry.add_agent_to_group(group_id, agent_id)

    def group_of_agent(self, agent_id):
        return self.registry.group_of_agent(agent_id)

    def get_group(self, group_id):
        return self.registry.get_group(group_id)

    def groups_ordered(self):
        return self.registry.groups_ordered()

    def remove_group(self, group_id):
        self.registry.remove_group(group_id)

    # ── Backward-compatible delegates: networking ───────────────────────────

    @property
    def sig_client_ready(self):
        return self.network.sig_client_ready

    @property
    def sig_client_disconnect(self):
        return self.network.sig_client_disconnect

    @property
    def sig_go_received(self):
        return self.network.sig_go_received

    @property
    def sig_client_registered(self):
        return self.network.sig_client_registered

    @property
    def sig_agent_removed(self):
        return self.network.sig_agent_removed

    @property
    def _net_host(self):
        return self.network.host

    @property
    def _net_client(self):
        return self.network.client

    @property
    def _slot_to_agent_id(self):
        return self.network._slot_to_agent_id

    @property
    def _agent_id_to_slot(self):
        return self.network._agent_id_to_slot

    @property
    def _net_frame_rate(self):
        return self.network._net_frame_rate

    @_net_frame_rate.setter
    def _net_frame_rate(self, value):
        self.network._net_frame_rate = value

    @property
    def _net_disconnect_timeout(self):
        return self.network._net_disconnect_timeout

    @_net_disconnect_timeout.setter
    def _net_disconnect_timeout(self, value):
        self.network._net_disconnect_timeout = value

    def enable_network_host(self, port=None):
        """Start a SimNetHost. Host mode is a pure physics server — the
        registry is emptied first (no local agents; clients arrive dynamically)."""
        if self.network.is_hosting:
            return
        self.registry.clear()
        self.network.enable_host(port)

    def disable_network_host(self):
        """Stop the SimNetHost and clear the remote flag on all agents."""
        self.network.disable_host()
        for agent in self.registry.agents:
            agent.remote = False

    def connect_to_host(self, *args, **kwargs):
        """Connect to a remote SimNetHost as a brain terminal."""
        # sig_go_received is a one-shot connection that self-disconnects once
        # fired (see _on_go_received) — reconnected on every call, matching
        # the original per-connection lifecycle.
        self.network.sig_go_received.connect(self._on_go_received, type=Qt.ConnectionType.QueuedConnection)
        self.network.connect_to_host(*args, **kwargs)

    def disconnect_from_host(self):
        self.network.disconnect_from_host()
        try:
            self.network.sig_go_received.disconnect(self._on_go_received)
        except RuntimeError:
            pass

    def _on_go_received(self):
        """Called on the Qt main thread when the host sends 'go'. Starts the
        tick loop — this mutates core run-state, so it stays on SimController
        rather than NetworkController even though it's networking-triggered."""
        self.running = True
        self.sig_status_changed.emit("●  RUNNING", C['success'])
        self._timer.start()
        try:
            self.network.sig_go_received.disconnect(self._on_go_received)
        except RuntimeError:
            pass

    # ── Backward-compatible delegates: real-robot mode ──────────────────────

    @property
    def _robot_mode(self):
        return self.robot.enabled

    @property
    def _robot_driver(self):
        return self.robot.driver

    @property
    def _motor_thread(self):
        return self.robot.motor_thread

    def enable_robot_mode(self, state: bool):
        """
        Toggle real-robot mode.  When enabled the sim physics are bypassed:
        sensor values come from RobotDriver threads and motor commands go to
        the robot over OSC.  dt is the actual wall-clock inter-step interval.
        """
        was_running = self.running
        if was_running:
            self.stop()

        self.robot.enable(state, self.circuit)
        if state:
            self._osc_ctrl.add_channel_names({'mL_sent', 'mR_sent'})
        else:
            self._osc_ctrl.remove_channel_names({'mL_sent', 'mR_sent'})

        if was_running:
            self.start()

    def _get_motor_commands(self):
        return self.robot.get_motor_commands(self.circuit, self.brain)

    def _send_motor_stop(self):
        self.robot.send_motor_stop(self.circuit)

    # ── Backward-compatible delegates: MuJoCo ───────────────────────────────

    @property
    def _mujoco_engine(self):
        return self.mujoco.engine

    @property
    def _view_3d(self):
        return self.mujoco.view_3d

    def enable_mujoco(self, state, world, sim_cfg):
        """Enable or disable the MuJoCo engine. Returns (ok, error_str_or_None)."""
        return self.mujoco.enable(state, world, self.registry.agents)

    def rebuild_mujoco(self, world, sim_cfg):
        self.mujoco.rebuild(world, self.registry.agents)

    def render_mujoco_overhead(self):
        self.mujoco.render_overhead()

    def sync_robot_markers(self):
        """Push each agent's current bot_pos into its arena vector marker,
        without touching physics/brain/trail state. The marker is only ever
        moved from reset()/_loop() otherwise, so manual edits made while the
        sim is paused (e.g. dragging the robot in the World editor) would
        leave it stuck at its pre-edit position without this. Not a MuJoCo
        concern despite living alongside reposition_mujoco_robots in
        LBPSimulator._setup_world — it only ever touches the arena."""
        self._arena.sync_agents(self.registry.agents, self.registry.selected_id,
                                 self.sim_cfg, self._trail_visible(), include_trail=False)

    def reposition_mujoco_robots(self):
        self.mujoco.reposition_robots(self.registry.agents)

    def show_mujoco_viewer(self):
        self.mujoco.show_viewer()

    def set_view_3d(self, enabled):
        self.mujoco.set_view_3d(enabled)

    # ── Speed / real-time mode ────────────────────────────────────────────

    def set_speed_mult(self, v):
        self.speed_mult = v

    def set_rt_mode(self, checked):
        self._rt_mode  = checked
        self._time_debt = 0.0

    # ── Run control ──────────────────────────────────────────────────────────

    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self):
        # Host mode needs no local brain — agents arrive dynamically as clients connect.
        if not self.network.is_hosting and not self.brain:
            return
        # Client mode: arm (send 'ready') and wait for host 'go' before ticking.
        if self.network.is_client:
            self.network.client.arm()
            self.sig_status_changed.emit("●  READY", C['warning'])
            return
        self.running = True
        self.sig_status_changed.emit("●  RUNNING", C['success'])
        if self.robot.enabled:
            self.robot.start_motor_thread(self._get_motor_commands)
        # Host mode: broadcast 'go' to all connected clients before starting own loop.
        if self.network.is_hosting:
            self.network.host.broadcast_go()
        self._timer.start()

    def stop(self):
        self.running = False
        self._timer.stop()
        self.robot.stop_motor_thread()
        self.sig_status_changed.emit("■  STOPPED", C['muted'])
        if self.robot.enabled:
            self._send_motor_stop()

    def step(self):
        if self.brain and not self.running:
            self._tick()
            overhead_rgb = None
            if self.mujoco.view_3d and self.mujoco.engine is not None:
                overhead_rgb = self.mujoco.engine.render_overhead(512, 512)
            # Also refreshes arena position/trail/sensor visuals, which
            # single-stepping never did before this refactor — an intentional
            # fix (see TODO.md), not a side effect to be wary of.
            self._emit_frame_ready(overhead_rgb)

    def _sync_mounted_patches(self):
        """Sync every mounted gradient patch's position to its carrying robot.
        mounted_on stores a stable agent id (not a list position), so a mount
        can never silently drift onto the wrong robot after some other agent
        is added/removed. Called both per-tick (before sensor sampling) and
        right after reset() repositions agents, so a mounted patch never
        renders at a stale position."""
        for patch in self.world.patches:
            agent_id = patch.get('mounted_on')
            if agent_id is not None:
                agent = self.registry._agent_by_id(agent_id)
                if agent is not None:
                    patch['x'] = agent.bot_pos[0]
                    patch['y'] = agent.bot_pos[1]

    def reset(self):
        self.stop()
        self.time_index = 0

        for i, agent in enumerate(self.registry.agents):
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

        # Snap mounted gradients to their (now-reset) robots immediately, rather
        # than leaving them at their pre-reset position until the first tick —
        # _setup_world()'s gradient render (called by the caller right after
        # this) would otherwise draw a mounted patch at a stale, possibly
        # far-away spot for a frame, looking like the gradient vanished.
        self._sync_mounted_patches()

        if self.mujoco.engine is not None:
            self.mujoco.engine.reset([a.bot_pos for a in self.registry.agents])

        self._osc_ctrl.reset_trace()
        self._arena.setup_sensors(self.circuit.sensors, self._osc_ctrl.channel_colors)
        if self._active_task is not None:
            self._active_task.reset(self.world, self.sim_cfg)

        sel = self.registry.agent
        self._arena.sync_agents(self.registry.agents, sel.id if sel is not None else None,
                                 self.sim_cfg, self._trail_visible())

        self._osc_ctrl.update_osc()

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _emit_frame_ready(self, overhead_rgb=None):
        """Emit sig_frame_ready with the current selected-agent id resolved the
        same way self._agent is (falls back to the last agent if the stored
        selection no longer matches a live one)."""
        sel = self.registry.agent
        self.sig_frame_ready.emit(self.registry.agents, sel.id if sel is not None else None,
                                   self.sim_cfg, self.world, self._trail_visible(),
                                   overhead_rgb)

    def _loop(self):
        if not self.running:
            return
        self._osc_ctrl.refresh_mult_cache()
        self._phys_ms_acc = 0.0
        _t0 = time.perf_counter()
        _cycle_ms = (_t0 - self._last_loop_t) * 1000 if self._last_loop_t else 0.0
        self._last_loop_t = _t0

        steps_done = 0
        if self.robot.enabled:
            # Same inner loop as simulation — run network as many times as possible
            # within the deadline. dt is real wall-clock time between steps.
            # Motor commands are sent by MotorThread independently at ~60 Hz.
            t_deadline = _t0 + 0.050
            while True:
                self._tick_robot()
                steps_done += 1
                if time.perf_counter() > t_deadline:
                    break
        elif self._rt_mode:
            self._time_debt += min(_cycle_ms / 1000.0, 0.200)
            t_deadline = _t0 + 0.050
            while self._time_debt >= self.sim_cfg.dt:
                self._tick()
                steps_done += 1
                self._time_debt -= self.sim_cfg.dt
                if time.perf_counter() > t_deadline:
                    self._time_debt = 0.0
                    break
        else:
            t_deadline = _t0 + 0.050
            for _ in range(self.speed_mult):
                self._tick()
                steps_done += 1
                if time.perf_counter() > t_deadline:
                    break

        overhead_rgb = None
        if self.mujoco.engine is not None:
            # render_cameras() is gated ONLY on the engine existing (the "3D
            # (MuJoCo)" checkbox), NOT on view_3d ("Top view"/"Show 3D"). So
            # every CameraSensor's _last_frame gets a real, textured MuJoCo
            # render as soon as the checkbox is on, even if the arena canvas
            # still displays the plain 2-D view below. Don't infer what a
            # camera sensor sees from what the canvas looks like — see
            # docs/simulator/architecture.md "Views on the same world".
            if self.brain is not None:
                self.mujoco.engine.render_cameras(
                    self.brain, self.circuit.sensors,
                    agent_idx=self.registry.index_of_agent(self.registry.selected_id) or 0)
            if self.mujoco.view_3d:
                overhead_rgb = self.mujoco.engine.render_overhead(512, 512)

        step_ms = self._phys_ms_acc / max(steps_done, 1)
        _t1 = time.perf_counter()

        # All arena/oscilloscope visual updates happen once, here, via the
        # signal — see ArenaWidget.on_frame_ready / OscChannelManager.on_frame_ready.
        self._emit_frame_ready(overhead_rgb)
        _t4 = time.perf_counter()

        total_ms         = (_t4 - _t0) * 1000
        render_ms        = (_t4 - _t1) * 1000
        phys_ms          = (_t1 - _t0) * 1000
        sim_ms_per_cycle = steps_done * self.sim_cfg.dt * 1000
        speedup          = sim_ms_per_cycle / _cycle_ms if _cycle_ms > 0 else 0.0
        if total_ms > 80:
            print(f"[SLOW] total={total_ms:.0f}ms  phys={phys_ms:.0f}ms  "
                  f"render={render_ms:.0f}ms  steps={steps_done}")
        self._frame_count += 1
        if self._frame_count % 100 == 0:
            print(f"[TIMING] cycle={_cycle_ms:.0f}ms  phys={phys_ms:.0f}ms  "
                  f"render={render_ms:.0f}ms  steps={steps_done}  "
                  f"step={step_ms:.2f}ms  dt={self.sim_cfg.dt*1000:.1f}ms  "
                  f"sim/real={speedup:.1f}x")
        if self.robot.enabled:
            tick_hz = steps_done * 1000.0 / _cycle_ms if _cycle_ms > 0 else 0.0
            self.sig_timing_updated.emit(f"step: {step_ms:.2f} ms  robot  {tick_hz:.0f} Hz")
        else:
            mode_tag = "RT" if self._rt_mode else f"×{self.speed_mult}"
            self.sig_timing_updated.emit(f"step: {step_ms:.2f} ms  {mode_tag}  {speedup:.1f}× real")

        if self.running:
            self._timer.start()

    def _tick_robot(self):
        """One physics-substep of real-robot mode, driven by RobotModeController."""
        values = self.robot.tick(self.circuit, self.brain, self._osc_ctrl)
        self.sig_tick_values.emit(values)
        self.time_index += 1

    def _tick(self):
        self.time_index += 1
        _t0 = time.perf_counter()

        # ── CLIENT MODE ───────────────────────────────────────────────────────
        # When connected to a remote host, receive sensor data, run brain, and
        # send motors. Skip local physics entirely (world runs on the host).
        if self.network.is_client:
            agent = self.registry.agent
            if agent.brain is not None:
                data, dt = self.network.client.pop_sensors()
                if data is not None:
                    for name, values in data.items():
                        setattr(agent.brain, name,
                                np.array(values, dtype=np.float32))
                    try:
                        result = agent.brain.loop(dt or self.sim_cfg.dt)
                        if isinstance(result, (tuple, list)) and len(result) >= 2:
                            mL, mR = float(result[0]), float(result[1])
                        else:
                            mL = mR = 0.0
                    except Exception as exc:
                        print(f"[net_client] brain.loop error: {exc}")
                        mL = mR = 0.0
                    self.network.client.send_motors(mL, mR)
            # Update oscilloscope from brain attributes (set just above)
            self.sig_tick_values.emit({
                k: (getattr(agent.brain, k, 0) if agent.brain else 0)
                for k in self._osc_ctrl.channels
            })
            self._phys_ms_acc += (time.perf_counter() - _t0) * 1000
            return
        # ─────────────────────────────────────────────────────────────────────

        self._sync_mounted_patches()

        override = self._motor_override()
        _engine  = self.mujoco.engine
        agents   = self.registry.agents
        selected_id = self.registry.selected_id

        def _motor_for(agent):
            """Return motor override for agent, or None for brain-driven."""
            if agent.id == selected_id and override is not None:
                return override
            return self.network.resolve_remote_motors(agent.id)

        # Run physics for every agent; accumulate selected agent's raw output.
        # MuJoCo path: batch all agents into one mj_forward so inter-agent contacts
        # are resolved correctly in a single physics step.
        selected_raw = {}
        if _engine is not None:
            agent_list = [(a.bot_pos, a.brain, a.circuit.sensors, a.circuit)
                          for a in agents]
            overrides_list = [_motor_for(a) for a in agents]
            raws = _engine.tick_physics_batch(agent_list, self.world, self.sim_cfg, overrides_list)
            if self._trail_visible():
                for agent in agents:
                    agent.trail_xy.append((agent.bot_pos[0], agent.bot_pos[1]))
            if agents:
                sel_pos = self.registry.index_of_agent(selected_id)
                selected_raw = raws[sel_pos] if sel_pos is not None else raws[-1]
            # HOST: send sensor data to any connected remote clients — unified
            # with the non-MuJoCo path below (previously this batch path sent
            # after the loop while the per-agent path sent inline inside it).
            if self.network.is_hosting:
                for a in agents:
                    slot_idx = self.network.slot_for_agent(a.id)
                    if slot_idx is not None and a.brain is not None:
                        self.network.host_maybe_send(slot_idx, a.brain, a.circuit.sensors, self.sim_cfg.dt)
        else:
            # Snapshot every agent's pre-tick position as a circle, so DistanceSensor/
            # CollisionSensor can see other agents as obstacles — snapshotted once up
            # front (not re-read per agent) so sensing doesn't depend on iteration
            # order as agents move one after another below.
            all_circles = [{'x': a.bot_pos[0], 'y': a.bot_pos[1], 'r': self.sim_cfg.body_radius}
                           for a in agents]
            for i, agent in enumerate(agents):
                mo = _motor_for(agent)
                other_agents = all_circles[:i] + all_circles[i + 1:]
                raw = tick_physics(
                    agent.bot_pos, agent.brain, agent.circuit.sensors,
                    self.world, self.sim_cfg, circuit=agent.circuit, motor_override=mo,
                    other_agents=other_agents)
                if self._trail_visible():
                    agent.trail_xy.append((agent.bot_pos[0], agent.bot_pos[1]))
                if agent.id == selected_id:
                    selected_raw = raw
            # HOST: send sensor data to remote clients — moved out of the
            # per-agent physics loop to match the MuJoCo path above.
            if self.network.is_hosting:
                for agent in agents:
                    slot_idx = self.network.slot_for_agent(agent.id)
                    if slot_idx is not None and agent.brain is not None:
                        self.network.host_maybe_send(slot_idx, agent.brain, agent.circuit.sensors, self.sim_cfg.dt)

        # Selected agent post-processing: override write-back, oscilloscope, logger.
        if not agents:
            self._phys_ms_acc += (time.perf_counter() - _t0) * 1000
            return
        agent = self.registry.agent
        raw   = selected_raw

        if override is not None:
            from neurons import MotorLayer as _MotorLayer
            import torch as _torch
            for _layer in agent.circuit.layers:
                if isinstance(_layer, _MotorLayer):
                    _lobj = getattr(agent.brain, _layer.name, _layer)
                    if hasattr(_lobj, 'output') and _lobj.output is not None:
                        _n = int(_lobj.output.numel()) if hasattr(_lobj.output, 'numel') \
                             else len(np.atleast_1d(_lobj.output))
                        _vals = [float(override[_j]) if _j < len(override) else 0.0
                                 for _j in range(_n)]
                        _lobj.output = _torch.tensor(_vals, dtype=_torch.float32)

        layer_names = {l.name for l in agent.circuit.layers}
        for lname in self._osc_ctrl._osc_items - {'mL', 'mR', 'sL', 'sR'}:
            if lname in layer_names:
                layer = getattr(agent.brain, lname, None)
                if layer is not None and hasattr(layer, 'output') and layer.output is not None:
                    for _j, v in enumerate(np.atleast_1d(layer.output)):
                        raw[f'{lname}_{_j}'] = float(v)
            else:
                parts = lname.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit() and parts[0] in layer_names:
                    layer = getattr(agent.brain, parts[0], None)
                    if (layer is not None and hasattr(layer, 'output')
                            and layer.output is not None):
                        out = np.atleast_1d(layer.output)
                        jdx = int(parts[1])
                        if jdx < len(out):
                            raw[lname] = float(out[jdx])

        if self._active_task is not None:
            self._active_task.tick(self.world, agent.bot_pos, self.sim_cfg, self.sim_cfg.dt)

        self._logger.log(self.time_index, agent.bot_pos, raw, self.world)
        self._phys_ms_acc += (time.perf_counter() - _t0) * 1000

        self.sig_tick_values.emit({k: raw.get(k, getattr(agent.brain, k, 0))
                                    for k in self._osc_ctrl.channels})

    def close(self):
        self.robot.driver.stop()
        self.network.close()
        self.mujoco.close()
