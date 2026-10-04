"""
sim_controller.py — the app's simulation loop around a Qt-free Simulation
(simulation.py: world, agents, MuJoCo engine, sim time, task, step/reset),
plus four collaborators: AgentRegistry (agent_registry.py, owned by the
Simulation), NetworkController (network_controller.py), RobotModeController
(robot_mode_controller.py), and MuJoCoController (mujoco_controller.py,
display side only).

SimController itself owns only what needs Qt or the outside world: the
QTimer, running/timing state, signals, arena/oscilloscope/logger feeds, and
network / real-robot I/O. Callers use the collaborators directly
(sim_ctrl.registry / .sim / .network / .robot / .mujoco); SimController only
keeps methods that coordinate more than one of them.
"""

import time
import numpy as np

from PySide6.QtCore import QObject, QTimer, Qt, Signal

from sim_constants import C
from simulation import Simulation
from sim_engine import run_brain, choose_motor_command
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
        self.sim      = Simulation(world, sim_cfg, self.registry)
        self.network  = NetworkController(parent=self)
        self.robot    = RobotModeController(sim_cfg, get_motor_override)
        self.mujoco   = MuJoCoController(arena, sim_cfg, self.sim)

        # Resolve the bidirectional writes the old monolithic class had
        # between the registry and MuJoCo/networking via explicit callbacks,
        # instead of the registry reaching directly into either.
        self.registry.on_agents_changed    = self._on_agents_changed
        self.registry.on_agent_removed     = self.network.forget_agent
        self.registry.on_selection_changed = self.sync_robot_items
        self.network.get_running           = lambda: self.running

        self.running    = False
        self.speed_mult = 200
        self._rt_mode   = False
        self._time_debt = 0.0
        self._phys_ms_acc = 0.0
        self._last_loop_t = None
        self._frame_count = 0

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._loop)

        self.sync_robot_items()   # one arena disk for the initial agent

    # ── Agents ─────────────────────────────────────────────────────────────

    def remove_agent(self, agent_id: int):
        """Remove an agent; while hosting the registry may become empty
        (clients arrive and leave dynamically)."""
        self.registry.remove_agent(agent_id, allow_empty=self.network.is_hosting)

    # ── Networking ─────────────────────────────────────────────────────────

    def enable_network_host(self, port=None):
        """Start a SimNetHost. Host mode is a pure physics server — the
        registry is emptied first (no local agents; clients arrive dynamically)."""
        if self.network.is_hosting:
            return
        self.network.enable_host(port)   # raises OSError if the port can't be opened —
        self.registry.clear()            # then the local agents are kept

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

    # ── Real-robot mode ────────────────────────────────────────────────────

    def enable_robot_mode(self, state: bool):
        """
        Toggle real-robot mode.  When enabled the sim physics are bypassed:
        sensor values come from RobotDriver threads and motor commands go to
        the robot over OSC.  dt is the actual wall-clock inter-step interval.
        """
        was_running = self.running
        if was_running:
            self.stop()

        self.robot.enable(state, self.registry.circuit)
        if state:
            self._osc_ctrl.add_channel_names({'mL_sent', 'mR_sent'})
        else:
            self._osc_ctrl.remove_channel_names({'mL_sent', 'mR_sent'})

        if was_running:
            self.start()

    def _get_motor_commands(self):
        return self.robot.get_motor_commands(self.registry.circuit, self.registry.brain)

    def _send_motor_stop(self):
        self.robot.send_motor_stop(self.registry.circuit)

    # ── Arena ──────────────────────────────────────────────────────────────

    def _on_agents_changed(self):
        """Agents were added or removed: rebuild MuJoCo, update the arena disks."""
        self.mujoco.rebuild()
        self.sync_robot_items()

    def sync_robot_items(self):
        """Arena disks follow the agent list (and colors / selection). Called on
        every registry change; call it after changing an agent's color."""
        self._arena.sync_robot_items(self.registry.agents, self.registry.selected_id)

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
        if not self.network.is_hosting and not self.registry.brain:
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
        # reset() calls stop() unconditionally (even when not running) to force
        # timer/motor state clean; only announce a real running->stopped
        # transition, so a Run click's reset() doesn't emit a spurious STOPPED
        # that makes Auto-record (sim_app_session._SessionMixin) finalize an
        # in-progress manual recording right before immediately starting a new one.
        was_running = self.running
        self.running = False
        self._timer.stop()
        self.robot.stop_motor_thread()
        if was_running:
            # Pause: everything stays where it is; Run continues from here.
            self.sig_status_changed.emit("⏸  PAUSED", C['warning'])
        if self.robot.enabled:
            self._send_motor_stop()

    def _render_free_running_cameras(self):
        """Render every agent's fps == 0 cameras once per display cycle, between
        batches of simulation steps — as fast as the app runs, without holding
        up the other sensors' per-step sampling. Cameras with fps > 0 render
        inside step_agents on their own sim-time schedule instead. Independent
        of "Top view"/"Show 3D": what a camera sees is always a MuJoCo render,
        whatever the arena canvas shows (docs/simulator/architecture.md
        "Views on the same world")."""
        # Real robot and network-client modes get camera data from elsewhere.
        if self.robot.enabled or self.network.is_client:
            return
        self.sim.render_free_running_cameras()

    def step(self):
        if self.registry.brain and not self.running:
            self._tick()
            self._render_free_running_cameras()
            overhead_rgb = None
            if self.mujoco.view_3d and self.mujoco.engine is not None:
                overhead_rgb = self.mujoco.engine.render_overhead(512, 512)
            # Also refreshes arena position/trail/sensor visuals, which
            # single-stepping never did before this refactor — an intentional
            # fix (see TODO.md), not a side effect to be wary of.
            self._emit_frame_ready(overhead_rgb)

    def reset(self):
        self.stop()
        # Also snaps mounted gradients to their (now-reset) robots immediately,
        # rather than leaving them at their pre-reset position until the first
        # tick — _setup_world()'s gradient render (called by the caller right
        # after this) would otherwise draw a mounted patch at a stale spot.
        self.sim.reset()

        self._osc_ctrl.reset_trace()
        self._arena.setup_sensors(self.registry.circuit.sensors, self._osc_ctrl.channel_colors)

        sel = self.registry.agent
        self._arena.sync_agents(self.registry.agents, sel.id if sel is not None else None,
                                 self.sim_cfg, self._trail_visible())

        self._osc_ctrl.update_osc()
        # Back at the start: Run begins a fresh run.
        self.sig_status_changed.emit("■  STOPPED", C['muted'])

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _emit_frame_ready(self, overhead_rgb=None):
        """Emit sig_frame_ready with the current selected-agent id resolved the
        same way registry.agent is (falls back to the last agent if the stored
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
        self._render_free_running_cameras()
        if self.mujoco.engine is not None and self.mujoco.view_3d:
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
        values = self.robot.tick(self.registry.circuit, self.registry.brain, self._osc_ctrl)
        self.sig_tick_values.emit(values)
        self.sim.time_index += 1

    def _tick(self):
        _t0 = time.perf_counter()

        # ── CLIENT MODE ───────────────────────────────────────────────────────
        # The world runs on the remote host: sensing = the host's sensor packet,
        # acting = send the motors back. Think / motors are the same code as
        # the simulator, including the keyboard taking priority.
        if self.network.is_client:
            self.sim.time_index += 1
            agent = self.registry.agent
            if agent.brain is not None:
                data, dt = self.network.client.pop_sensors()
                if data is not None:
                    for name, values in data.items():
                        setattr(agent.brain, name,
                                np.array(values, dtype=np.float32))
                    try:
                        brain_cmd = run_brain(agent.brain, dt or self.sim_cfg.dt)
                    except Exception as exc:
                        print(f"[net_client] brain.loop error: {exc}")
                        brain_cmd = (0.0, 0.0)
                    mL, mR = choose_motor_command(agent.circuit, brain_cmd, self._motor_override())
                    self.network.client.send_motors(mL, mR)
            # Update oscilloscope from brain attributes (set just above)
            self.sig_tick_values.emit({
                k: (getattr(agent.brain, k, 0) if agent.brain else 0)
                for k in self._osc_ctrl.channels
            })
            self._phys_ms_acc += (time.perf_counter() - _t0) * 1000
            return
        # ─────────────────────────────────────────────────────────────────────

        keyboard = self._motor_override()
        agents   = self.registry.agents
        selected_id = self.registry.selected_id

        def _motor_for(agent):
            """Motor source per rules/motor_commands.md: keyboard (selected agent)
            > network client > None (the agent's own brain drives)."""
            if agent.id == selected_id and keyboard is not None:
                return keyboard
            return self.network.resolve_remote_motors(agent.id)

        raws = self.sim.step(_motor_for)
        if not raws:
            self._phys_ms_acc += (time.perf_counter() - _t0) * 1000
            return

        if self._trail_visible():
            for agent in agents:
                agent.trail_xy.append((agent.bot_pos[0], agent.bot_pos[1]))
        # HOST: send sensor data to any connected remote clients.
        if self.network.is_hosting:
            for a in agents:
                slot_idx = self.network.slot_for_agent(a.id)
                if slot_idx is not None and a.brain is not None:
                    self.network.host_maybe_send(slot_idx, a.brain, a.circuit.sensors, self.sim_cfg.dt)

        # Selected agent post-processing: oscilloscope, logger.
        sel_pos = self.registry.index_of_agent(selected_id)
        raw   = raws[sel_pos] if sel_pos is not None else raws[-1]
        agent = self.registry.agent

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

        self._logger.log(self.sim.time_index, agent.bot_pos, raw, self.world)
        self._phys_ms_acc += (time.perf_counter() - _t0) * 1000

        self.sig_tick_values.emit({k: raw.get(k, getattr(agent.brain, k, 0))
                                    for k in self._osc_ctrl.channels})

    def close(self):
        self.robot.driver.stop()
        self.network.close()
        self.mujoco.close()
