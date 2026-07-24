"""
sim_controller.py — simulation loop and mutable physics state.

Owns the QTimer, bot_pos, running flag, speed/RT settings, MuJoCo engine
lifecycle, manual-control motor override, active task, and logger. Display
updates are dispatched through the arena/osc_ctrl references passed at
construction so this class is independent of panel layout code.
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, QTimer, Signal


from sim_constants import C
from sim_engine import tick_physics
from rigid_body import world_poses as _rb_world_poses
from robot_driver import RobotDriver, MotorThread


@dataclass
class RobotAgent:
    """Per-agent state bundle."""
    circuit:   object                        # CircuitModel
    brain:     object                        # BaseBrain instance or None
    bot_pos:   list                          # [x, y, theta]
    trail_xy:  deque = field(default_factory=lambda: deque(maxlen=500))
    brain_mgr: object = None                 # BrainManager
    name:      str    = ""
    color:     str    = ""


class SimController(QObject):
    """
    Drives the simulation: owns the QTimer, physics state, and per-frame I/O.

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

        self._agents = [  # list[RobotAgent]
            RobotAgent(
                circuit   = circuit,
                brain     = None,
                bot_pos   = [0.0, 0.0, 0.0],
                trail_xy  = deque(maxlen=500),
                brain_mgr = brain_mgr,
                name      = "Agent 1",
                color     = '#4a7fcb',
            )
        ]
        self._selected: int = 0

        self._active_task    = None
        self._mujoco_engine  = None
        self._view_3d        = False

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

        # Real-robot mode
        self._robot_mode        = False
        self._robot_driver      = RobotDriver()
        self._last_robot_tick_t = None
        self._motor_thread = None  # type: MotorThread | None

    # ── Agent access ─────────────────────────────────────────────────────────

    @property
    def _agent(self) -> RobotAgent:
        return self._agents[self._selected]

    @property
    def circuit(self):
        return self._agent.circuit

    @property
    def brain(self):
        return self._agent.brain

    @brain.setter
    def brain(self, value):
        self._agent.brain = value

    @property
    def bot_pos(self):
        return self._agent.bot_pos

    @property
    def trail_xy(self):
        return self._agent.trail_xy

    # ── Multi-agent management ────────────────────────────────────────────────

    def add_agent(self, circuit, brain_mgr, name=None, color=None) -> int:
        """Append a new agent. Returns its index."""
        idx = len(self._agents)
        offset_x = idx * 0.5
        agent = RobotAgent(
            circuit   = circuit,
            brain     = None,
            bot_pos   = [self.sim_cfg.init_x + offset_x, self.sim_cfg.init_y, 0.0],
            trail_xy  = deque(maxlen=500),
            brain_mgr = brain_mgr,
            name      = name  or f"Agent {idx + 1}",
            color     = color or '#4a7fcb',
        )
        self._agents.append(agent)
        if self._mujoco_engine is not None:
            try:
                all_pos = [a.bot_pos for a in self._agents]
                self._mujoco_engine.rebuild(self.world, self.sim_cfg,
                                             bot_pos=all_pos, n_agents=len(self._agents))
            except Exception as e:
                print(f"[MuJoCo] add_agent rebuild error: {e}")
        return len(self._agents) - 1

    def remove_agent(self, idx: int):
        if len(self._agents) <= 1 or idx >= len(self._agents):
            return
        self._agents.pop(idx)
        self._selected = min(self._selected, len(self._agents) - 1)
        if self._mujoco_engine is not None:
            try:
                all_pos = [a.bot_pos for a in self._agents]
                self._mujoco_engine.rebuild(self.world, self.sim_cfg,
                                             bot_pos=all_pos, n_agents=len(self._agents))
            except Exception as e:
                print(f"[MuJoCo] remove_agent rebuild error: {e}")

    def select_agent(self, idx: int):
        if 0 <= idx < len(self._agents):
            self._selected = idx

    # ── Run control ──────────────────────────────────────────────────────────

    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self):
        if not self.brain:
            return
        self.running = True
        self.sig_status_changed.emit("●  RUNNING", C['success'])
        if self._robot_mode:
            self._motor_thread = MotorThread(self._get_motor_commands, self._robot_driver)
            self._motor_thread.start()
        self._timer.start()

    def stop(self):
        self.running = False
        self._timer.stop()
        if self._motor_thread is not None:
            self._motor_thread.stop()
            self._motor_thread.join(timeout=1.0)
            self._motor_thread = None
        self.sig_status_changed.emit("■  STOPPED", C['muted'])
        if self._robot_mode:
            self._send_motor_stop()

    def step(self):
        if self.brain and not self.running:
            self._osc_ctrl.refresh_mult_cache()
            self._tick()
            if self._view_3d and self._mujoco_engine is not None:
                rgb = self._mujoco_engine.render_overhead(512, 512)
                self._arena.set_overhead_frame(rgb, self.sim_cfg.arena_scale)
            self._osc_ctrl.update_osc()

    def reset(self):
        self.stop()
        self.time_index = 0

        for i, agent in enumerate(self._agents):
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

        if self._mujoco_engine is not None:
            self._mujoco_engine.reset([a.bot_pos for a in self._agents])

        self._osc_ctrl.reset_trace()
        self._arena.setup_sensors(self.circuit.sensors, self._osc_ctrl.channel_colors)
        if self._active_task is not None:
            self._active_task.reset(self.world, self.sim_cfg)

        x, y, theta = self.bot_pos
        poses = None
        if len(self.circuit.bodies) > 1:
            poses = _rb_world_poses(self.bot_pos, self.circuit.bodies, self.circuit.joints)
            self._arena.update_child_bodies(poses, self.circuit.bodies, self.sim_cfg)
        self._arena.update_robot(x, y, self.sim_cfg.body_radius, theta,
                                 self.sim_cfg, poses, self.circuit)

        for i, agent in enumerate(self._agents):
            self._arena.update_robot_pos(i, agent.bot_pos[0], agent.bot_pos[1],
                                         self.sim_cfg.body_radius, agent.bot_pos[2])
            self._arena.update_trail(agent.trail_xy, self._trail_visible(), agent_idx=i)

        self._osc_ctrl.update_osc()

    # ── Real-robot mode ───────────────────────────────────────────────────────

    def enable_robot_mode(self, state: bool):
        """
        Toggle real-robot mode.  When enabled the sim physics are bypassed:
        sensor values come from RobotDriver threads and motor commands go to
        the robot over OSC.  dt is the actual wall-clock inter-step interval.

        Each sensor's robot_address field ('host:port') determines its connection.
        Motor /wheels commands are sent to the host:port of every non-camera sensor.
        """
        was_running = self.running
        if was_running:
            self.stop()

        self._robot_mode = state

        if state:
            self._last_robot_tick_t = None
            self._robot_driver.start(self.circuit.sensors)
            self._osc_ctrl._osc_items.update({'mL_sent', 'mR_sent'})
        else:
            self._robot_driver.stop()
            self._robot_driver.clear_robot_values(self.circuit.sensors)
            self._osc_ctrl._osc_items.discard('mL_sent')
            self._osc_ctrl._osc_items.discard('mR_sent')

        if was_running:
            self.start()

    # ── Speed / real-time mode ────────────────────────────────────────────────

    def set_speed_mult(self, v):
        self.speed_mult = v

    def set_rt_mode(self, checked):
        self._rt_mode  = checked
        self._time_debt = 0.0

    # ── MuJoCo lifecycle ─────────────────────────────────────────────────────

    def enable_mujoco(self, state, world, sim_cfg):
        """Enable or disable the MuJoCo engine. Returns (ok, error_str_or_None)."""
        if state:
            try:
                from sim_engine_mujoco import MuJoCoEngine
                self._mujoco_engine = MuJoCoEngine(world, sim_cfg, n_agents=len(self._agents))
                self._mujoco_engine.reset([a.bot_pos for a in self._agents])
                return True, None
            except Exception as e:
                self._mujoco_engine = None
                return False, str(e)
        else:
            if self._mujoco_engine is not None:
                self._mujoco_engine.close()
                self._mujoco_engine = None
            if self._view_3d:
                self._view_3d = False
                self._arena.set_3d_mode(False)
            return True, None

    def rebuild_mujoco(self, world, sim_cfg):
        """Rebuild the MuJoCo model after world changes. No-op when engine is off."""
        if self._mujoco_engine is None:
            return
        try:
            all_pos = [a.bot_pos for a in self._agents]
            self._mujoco_engine.rebuild(world, sim_cfg, bot_pos=all_pos,
                                         n_agents=len(self._agents))
        except Exception as e:
            print(f"[MuJoCo] rebuild error: {e}")

    def render_mujoco_overhead(self):
        """Push an overhead render to the arena when in 3D view mode."""
        if self._mujoco_engine is not None and self._view_3d:
            rgb = self._mujoco_engine.render_overhead(512, 512)
            self._arena.set_overhead_frame(rgb, self.sim_cfg.arena_scale)

    def show_mujoco_viewer(self):
        if self._mujoco_engine is not None:
            self._mujoco_engine.launch_viewer()

    def set_view_3d(self, enabled):
        self._view_3d = enabled
        self._arena.set_3d_mode(enabled)
        if enabled and self._mujoco_engine is not None:
            rgb = self._mujoco_engine.render_overhead(512, 512)
            self._arena.set_overhead_frame(rgb, self.sim_cfg.arena_scale)

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        if not self.running:
            return
        self._osc_ctrl.refresh_mult_cache()
        self._phys_ms_acc = 0.0
        _t0 = time.perf_counter()
        _cycle_ms = (_t0 - self._last_loop_t) * 1000 if self._last_loop_t else 0.0
        self._last_loop_t = _t0

        steps_done = 0
        if self._robot_mode:
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

        if self._mujoco_engine is not None:
            if self.brain is not None:
                self._mujoco_engine.render_cameras(self.brain, self.circuit.sensors,
                                                    agent_idx=self._selected)
            if self._view_3d:
                rgb = self._mujoco_engine.render_overhead(512, 512)
                self._arena.set_overhead_frame(rgb, self.sim_cfg.arena_scale)

        step_ms = self._phys_ms_acc / max(steps_done, 1)
        _t1 = time.perf_counter()

        # Lightweight position update for every agent's disk
        for i, agent in enumerate(self._agents):
            self._arena.update_robot_pos(i, agent.bot_pos[0], agent.bot_pos[1],
                                         self.sim_cfg.body_radius, agent.bot_pos[2])

        # Full render (wheels, sensors, child bodies) for selected agent only
        x, y, theta = self.bot_pos
        poses = None
        if len(self.circuit.bodies) > 1:
            poses = _rb_world_poses(self.bot_pos, self.circuit.bodies, self.circuit.joints)
            self._arena.update_child_bodies(poses, self.circuit.bodies, self.sim_cfg)
        _t2 = time.perf_counter()
        self._arena.update_objects(self.world)
        self._arena.update_walls(self.world)
        # Refresh gradient overlay each frame when any patch is robot-mounted
        if any(p.get('mounted_on') is not None for p in self.world.patches):
            self._arena._rebuild_gradient()
        self._arena.update_robot(x, y, self.sim_cfg.body_radius, theta,
                                 self.sim_cfg, poses, self.circuit)
        _t3 = time.perf_counter()
        for i, agent in enumerate(self._agents):
            self._arena.update_trail(agent.trail_xy, self._trail_visible(), agent_idx=i)
        self._osc_ctrl.update_osc()
        _t4 = time.perf_counter()

        total_ms         = (_t4 - _t0) * 1000
        render_ms        = (_t4 - _t1) * 1000
        phys_ms          = (_t1 - _t0) * 1000
        sim_ms_per_cycle = steps_done * self.sim_cfg.dt * 1000
        speedup          = sim_ms_per_cycle / _cycle_ms if _cycle_ms > 0 else 0.0
        if total_ms > 80:
            print(f"[SLOW] total={total_ms:.0f}ms  phys={phys_ms:.0f}ms  "
                  f"bodies={(_t2-_t1)*1000:.0f}ms  robot={(_t3-_t2)*1000:.0f}ms  "
                  f"osc={(_t4-_t3)*1000:.0f}ms  steps={steps_done}")
        self._frame_count += 1
        if self._frame_count % 100 == 0:
            print(f"[TIMING] cycle={_cycle_ms:.0f}ms  phys={phys_ms:.0f}ms  "
                  f"render={render_ms:.0f}ms  steps={steps_done}  "
                  f"step={step_ms:.2f}ms  dt={self.sim_cfg.dt*1000:.1f}ms  "
                  f"sim/real={speedup:.1f}x")
        if self._robot_mode:
            tick_hz = steps_done * 1000.0 / _cycle_ms if _cycle_ms > 0 else 0.0
            self.sig_timing_updated.emit(f"step: {step_ms:.2f} ms  robot  {tick_hz:.0f} Hz")
        else:
            mode_tag = "RT" if self._rt_mode else f"×{self.speed_mult}"
            self.sig_timing_updated.emit(f"step: {step_ms:.2f} ms  {mode_tag}  {speedup:.1f}× real")

        if self.running:
            self._timer.start()

    def _get_motor_commands(self):
        """Return (host, port, osc_path, vL, vR) for every active MotorLayer.

        Called by MotorThread at ~60 Hz; reads the latest network output.
        Manual override takes priority and is written back into layer.output
        so the oscilloscope reflects what the robot is actually doing.
        """
        from neurons import MotorLayer as _MotorLayer
        from robot_driver import RobotDriver as _RD
        import torch as _torch

        cmds = []
        override = self._motor_override()
        for layer in self.circuit.layers:
            if not isinstance(layer, _MotorLayer):
                continue
            layer_obj = getattr(self.brain, layer.name, layer)
            if override is not None:
                mL = float(override[0])
                mR = float(override[1]) if len(override) > 1 else mL
                if hasattr(layer_obj, 'output') and layer_obj.output is not None:
                    n    = layer_obj.n or 2
                    vals = [float(override[i]) if i < len(override) else 0.0
                            for i in range(n)]
                    layer_obj.output = _torch.tensor(vals, dtype=_torch.float32)
            else:
                if hasattr(layer_obj, 'output') and layer_obj.output is not None:
                    out = np.atleast_1d(layer_obj.output)
                    mL  = float(out[0]) if len(out) > 0 else 0.0
                    mR  = float(out[1]) if len(out) > 1 else mL
                else:
                    mL = mR = 0.0
            motor_addr = getattr(layer_obj, 'robot_address', '').strip()
            if motor_addr:
                host, port, osc_path, _, _ = _RD._parse_address(motor_addr)
                if host and port and osc_path:
                    cmds.append((host, port, osc_path, mL, mR))
        return cmds

    def _send_motor_stop(self):
        """Send zero motor commands to every MotorLayer's robot address."""
        from neurons import MotorLayer as _MotorLayer
        from robot_driver import RobotDriver as _RD
        for layer in self.circuit.layers:
            if not isinstance(layer, _MotorLayer):
                continue
            motor_addr = getattr(layer, 'robot_address', '').strip()
            if motor_addr:
                host, port, osc_path, _, _ = _RD._parse_address(motor_addr)
                if host and port and osc_path:
                    self._robot_driver.send_motor(host, port, osc_path, 0.0, 0.0)

    def _tick_robot(self):
        """One network step driven by real robot sensor data."""
        from network_runner import step_network

        now = time.perf_counter()
        if self._last_robot_tick_t is None:
            dt = self.sim_cfg.dt   # first tick: fall back to configured dt
        else:
            dt = now - self._last_robot_tick_t
        self._last_robot_tick_t = now

        brain   = self.brain
        sensors = self.circuit.sensors

        # Simple object so sensor._process() can read .dt without a real SimConfig.
        class _Cfg:
            pass
        _cfg = _Cfg()
        _cfg.dt = dt

        # Push latest sensor values onto the brain object, applying the same
        # scale/bias + dynamics (tau, activation) pipeline that sample() uses.
        for sensor in sensors:
            raw = sensor._robot_value
            if raw is None:
                n = getattr(sensor, 'n_total', None) or getattr(sensor, 'n', 1) or 1
                val = np.zeros(n, dtype=np.float32)
            else:
                val = sensor.process_robot_value(raw, _cfg)
            setattr(brain, sensor.name, val)
            # Lateralized camera halves (raw passthrough — cameras handle own processing)
            if getattr(sensor, 'lateralized', False):
                lv = getattr(sensor, '_left_output',  None)
                rv = getattr(sensor, '_right_output', None)
                if lv is not None:
                    setattr(brain, sensor.name + '_L', lv)
                if rv is not None:
                    setattr(brain, sensor.name + '_R', rv)

        step_network(brain, dt)

        # Write manual override into motor layer outputs so the network visualizer
        # and oscilloscope motor-neuron channels reflect what actually drives the robot.
        _override = self._motor_override()
        if _override is not None:
            import torch as _torch
            from neurons import MotorLayer as _MotorLayerR
            for _layer in self.circuit.layers:
                if isinstance(_layer, _MotorLayerR):
                    _lobj = getattr(brain, _layer.name, _layer)
                    if hasattr(_lobj, 'output') and _lobj.output is not None:
                        _n = int(_lobj.output.numel()) if hasattr(_lobj.output, 'numel') \
                             else len(np.atleast_1d(_lobj.output))
                        _vals = [float(_override[_j]) if _j < len(_override) else 0.0
                                 for _j in range(_n)]
                        _lobj.output = _torch.tensor(_vals, dtype=_torch.float32)

        # Build raw dict for the oscilloscope — mirrors what tick_physics returns
        # in sim mode: motor values, indexed sensor values, and indexed layer outputs.
        from neurons import MotorLayer as _MotorLayer
        mL = mR = 0.0
        for layer in self.circuit.layers:
            if isinstance(layer, _MotorLayer):
                layer_obj = getattr(brain, layer.name, layer)
                if hasattr(layer_obj, 'output') and layer_obj.output is not None:
                    out = np.atleast_1d(layer_obj.output)
                    mL  = float(out[0]) if len(out) > 0 else 0.0
                    mR  = float(out[1]) if len(out) > 1 else mL
                break

        raw = {'mL': mL, 'mR': mR}

        # Actual integer values sent to robot (staircase at ~60 Hz)
        if self._motor_thread is not None:
            raw['mL_sent'] = self._motor_thread.last_vL
            raw['mR_sent'] = self._motor_thread.last_vR

        # Indexed sensor values: brain.collision → collision_0, collision_1, …
        for sensor in self.circuit.sensors:
            val = getattr(brain, sensor.name, None)
            if val is not None:
                for i, v in enumerate(np.atleast_1d(val)):
                    raw[f'{sensor.name}_{i}'] = float(v)

        # Indexed layer outputs (non-motor layers tracked by the oscilloscope)
        layer_names = {l.name for l in self.circuit.layers}
        for lname in self._osc_ctrl._osc_items - {'mL', 'mR', 'sL', 'sR'}:
            if lname in layer_names:
                layer_obj = getattr(brain, lname, None)
                if layer_obj is not None and hasattr(layer_obj, 'output') \
                        and layer_obj.output is not None:
                    for i, v in enumerate(np.atleast_1d(layer_obj.output)):
                        raw[f'{lname}_{i}'] = float(v)

        for k in self._osc_ctrl.channels:
            val = raw.get(k, 0.0)
            self._osc_ctrl.append_trace(k, val)

        self.time_index += 1

    def _tick(self):
        self.time_index += 1
        _t0 = time.perf_counter()

        # Sync mounted gradient positions to their robots before sensor sampling
        for patch in self.world.patches:
            agent_idx = patch.get('mounted_on')
            if agent_idx is not None and 0 <= agent_idx < len(self._agents):
                bp = self._agents[agent_idx].bot_pos
                patch['x'] = bp[0]
                patch['y'] = bp[1]

        override = self._motor_override()
        _engine  = self._mujoco_engine

        # Run physics for every agent; accumulate selected agent's raw output.
        # MuJoCo path: batch all agents into one mj_forward so inter-agent contacts
        # are resolved correctly in a single physics step.
        selected_raw = {}
        if _engine is not None:
            agent_list = [(a.bot_pos, a.brain, a.circuit.sensors, a.circuit)
                          for a in self._agents]
            overrides_list = [override if i == self._selected else None
                              for i in range(len(self._agents))]
            raws = _engine.tick_physics_batch(agent_list, self.world, self.sim_cfg, overrides_list)
            if self._trail_visible():
                for agent in self._agents:
                    agent.trail_xy.append((agent.bot_pos[0], agent.bot_pos[1]))
            selected_raw = raws[self._selected]
        else:
            for i, agent in enumerate(self._agents):
                mo = override if i == self._selected else None
                raw = tick_physics(
                    agent.bot_pos, agent.brain, agent.circuit.sensors,
                    self.world, self.sim_cfg, circuit=agent.circuit, motor_override=mo)
                if self._trail_visible():
                    agent.trail_xy.append((agent.bot_pos[0], agent.bot_pos[1]))
                if i == self._selected:
                    selected_raw = raw

        # Selected agent post-processing: override write-back, oscilloscope, logger.
        # If batching is desired in the future, check that all agents share the same
        # brain class and implement a batch_forward path here.
        agent = self._agent
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

        for k in self._osc_ctrl.channels:
            val = raw.get(k, getattr(agent.brain, k, 0))
            self._osc_ctrl.append_trace(k, val)

    def close(self):
        self._robot_driver.stop()
        if self._mujoco_engine is not None:
            self._mujoco_engine.close()
            self._mujoco_engine = None
