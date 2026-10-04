"""
robot_mode_controller.py — real-robot mode for the 2D simulator.

When enabled, the real robot takes the place of the simulated world: sensor
values come from RobotDriver threads reading the real robot over OSC, and
motor commands are sent back to it. The brain runs through the same
think / motors code as the simulator. Takes circuit/brain/osc_ctrl as method arguments rather than
storing cross-references, since it only needs them transiently per call — this
keeps it from needing a back-reference into the agent registry.
"""

import time

import numpy as np

from robot_driver import RobotDriver, MotorThread
from sim_engine import run_brain, choose_motor_command, raw_sensor_values


class RobotModeController:
    def __init__(self, sim_cfg, get_motor_override):
        self.sim_cfg = sim_cfg
        self._motor_override = get_motor_override
        self.enabled = False
        self.driver = RobotDriver()
        self._last_tick_t = None
        self.motor_thread = None   # type: MotorThread | None
        self.wheel_cmd = (0.0, 0.0)   # latest (mL, mR) chosen by tick(); read by MotorThread

    def enable(self, state: bool, circuit):
        """Toggle real-robot mode. Each sensor's robot_address field
        ('host:port') determines its connection; motor /wheels commands are
        sent to every motor layer's robot_address."""
        self.enabled = state
        self.wheel_cmd = (0.0, 0.0)
        if state:
            self._last_tick_t = None
            self.driver.start(circuit.sensors)
        else:
            self.driver.stop()
            self.driver.clear_robot_values(circuit.sensors)

    def start_motor_thread(self, get_motor_commands):
        self.wheel_cmd = (0.0, 0.0)   # never resend a stale command from a previous run
        self.motor_thread = MotorThread(get_motor_commands, self.driver)
        self.motor_thread.start()

    def stop_motor_thread(self):
        if self.motor_thread is not None:
            self.motor_thread.stop()
            self.motor_thread.join(timeout=1.0)
            self.motor_thread = None

    def get_motor_commands(self, circuit, brain):
        """Return (host, port, osc_path, vL, vR) for every user motor layer with
        a robot_address.

        Called by MotorThread at ~60 Hz. Every brain follows the same rule:
        the latest wheel command chosen by tick() — the keyboard, or what the
        brain's loop() returned — is sent to every motor layer's robot_address.
        A code-only brain therefore just needs a motor layer with an address.
        """
        from robot_driver import RobotDriver as _RD

        mL, mR = self.wheel_cmd
        cmds = []
        for layer in circuit.layers:
            if not getattr(layer, 'drives_wheels', False):
                continue
            motor_addr = getattr(layer, 'robot_address', '').strip()
            if motor_addr:
                host, port, osc_path, *_ = _RD._parse_address(motor_addr)
                if host and port and osc_path:
                    cmds.append((host, port, osc_path, mL, mR))
        return cmds

    def send_motor_stop(self, circuit):
        """Send zero motor commands to every wheel motor layer's robot address
        (the same layers get_motor_commands drives)."""
        from robot_driver import RobotDriver as _RD
        for layer in circuit.layers:
            if not getattr(layer, 'drives_wheels', False):
                continue
            motor_addr = getattr(layer, 'robot_address', '').strip()
            if motor_addr:
                host, port, osc_path, *_ = _RD._parse_address(motor_addr)
                if host and port and osc_path:
                    self.driver.send_motor(host, port, osc_path, 0.0, 0.0)

    def tick(self, circuit, brain, osc_ctrl) -> dict:
        """One step driven by real robot sensor data — the same think / motors
        code as the simulator (sim_engine.run_brain / choose_motor_command);
        only sensing (latest UDP reading per sensor) and acting (MotorThread
        sends self.wheel_cmd) differ. dt is the real elapsed time. Returns the
        resolved oscilloscope channel values — the caller (SimController)
        emits sig_tick_values and advances time_index, since both are core
        loop state this class doesn't own."""
        now = time.perf_counter()
        if self._last_tick_t is None:
            dt = self.sim_cfg.dt   # first tick: fall back to configured dt
        else:
            dt = now - self._last_tick_t
        self._last_tick_t = now

        sensors = circuit.sensors

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
            # Lateralized camera halves (raw passthrough — cameras handle own
            # processing). Joint-pair halves only exist in simulation.
            if sensor.is_camera:
                sensor.publish_halves(brain)

        # THINK + MOTORS — keyboard commands are also written into the motor
        # layer, so the visualizer and oscilloscope show what drives the robot.
        brain_cmd = run_brain(brain, dt)
        mL, mR = choose_motor_command(circuit, brain_cmd, self._motor_override())
        # ACT — MotorThread picks this up at its own rate.
        self.wheel_cmd = (mL, mR)

        # Build raw dict for the oscilloscope — mirrors what step_agents returns
        # in sim mode: motor values, indexed sensor values, and indexed layer outputs.
        raw = {'mL': mL, 'mR': mR}

        # Actual integer values sent to robot (staircase at ~60 Hz)
        if self.motor_thread is not None:
            raw['mL_sent'] = self.motor_thread.last_vL
            raw['mR_sent'] = self.motor_thread.last_vR

        # Indexed sensor values (brain.collision → collision_0, …), cameras
        # excluded — the same values the simulator's step returns.
        raw.update(raw_sensor_values(brain, circuit.sensors))

        # Indexed layer outputs (non-motor layers tracked by the oscilloscope)
        layer_names = {l.name for l in circuit.layers}
        for lname in osc_ctrl._osc_items - {'mL', 'mR', 'sL', 'sR'}:
            if lname in layer_names:
                layer_obj = getattr(brain, lname, None)
                if layer_obj is not None and hasattr(layer_obj, 'output') \
                        and layer_obj.output is not None:
                    for i, v in enumerate(np.atleast_1d(layer_obj.output)):
                        raw[f'{lname}_{i}'] = float(v)

        return {k: raw.get(k, 0.0) for k in osc_ctrl.channels}
