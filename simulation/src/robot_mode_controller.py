"""
robot_mode_controller.py — real-robot mode for the 2D simulator.

When enabled, simulated physics is bypassed entirely: sensor values come from
RobotDriver threads reading the real robot over OSC, and motor commands are
sent back to it. Takes circuit/brain/osc_ctrl as method arguments rather than
storing cross-references, since it only needs them transiently per call — this
keeps it from needing a back-reference into the agent registry.
"""

import time

import numpy as np

from robot_driver import RobotDriver, MotorThread


class RobotModeController:
    def __init__(self, sim_cfg, get_motor_override):
        self.sim_cfg = sim_cfg
        self._motor_override = get_motor_override
        self.enabled = False
        self.driver = RobotDriver()
        self._last_tick_t = None
        self.motor_thread = None   # type: MotorThread | None

    def enable(self, state: bool, circuit):
        """Toggle real-robot mode. Each sensor's robot_address field
        ('host:port') determines its connection; motor /wheels commands are
        sent to the host:port of every non-camera sensor."""
        self.enabled = state
        if state:
            self._last_tick_t = None
            self.driver.start(circuit.sensors)
        else:
            self.driver.stop()
            self.driver.clear_robot_values(circuit.sensors)

    def start_motor_thread(self, get_motor_commands):
        self.motor_thread = MotorThread(get_motor_commands, self.driver)
        self.motor_thread.start()

    def stop_motor_thread(self):
        if self.motor_thread is not None:
            self.motor_thread.stop()
            self.motor_thread.join(timeout=1.0)
            self.motor_thread = None

    def get_motor_commands(self, circuit, brain):
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
        for layer in circuit.layers:
            if not isinstance(layer, _MotorLayer):
                continue
            layer_obj = getattr(brain, layer.name, layer)
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

    def send_motor_stop(self, circuit):
        """Send zero motor commands to every MotorLayer's robot address."""
        from neurons import MotorLayer as _MotorLayer
        from robot_driver import RobotDriver as _RD
        for layer in circuit.layers:
            if not isinstance(layer, _MotorLayer):
                continue
            motor_addr = getattr(layer, 'robot_address', '').strip()
            if motor_addr:
                host, port, osc_path, _, _ = _RD._parse_address(motor_addr)
                if host and port and osc_path:
                    self.driver.send_motor(host, port, osc_path, 0.0, 0.0)

    def tick(self, circuit, brain, osc_ctrl) -> dict:
        """One physics-substep driven by real robot sensor data. Returns the
        resolved oscilloscope channel values — the caller (SimController)
        emits sig_tick_values and advances time_index, since both are core
        loop state this class doesn't own."""
        from network_runner import step_network

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
            for _layer in circuit.layers:
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
        for layer in circuit.layers:
            if isinstance(layer, _MotorLayer):
                layer_obj = getattr(brain, layer.name, layer)
                if hasattr(layer_obj, 'output') and layer_obj.output is not None:
                    out = np.atleast_1d(layer_obj.output)
                    mL  = float(out[0]) if len(out) > 0 else 0.0
                    mR  = float(out[1]) if len(out) > 1 else mL
                break

        raw = {'mL': mL, 'mR': mR}

        # Actual integer values sent to robot (staircase at ~60 Hz)
        if self.motor_thread is not None:
            raw['mL_sent'] = self.motor_thread.last_vL
            raw['mR_sent'] = self.motor_thread.last_vR

        # Indexed sensor values: brain.collision → collision_0, collision_1, …
        for sensor in circuit.sensors:
            val = getattr(brain, sensor.name, None)
            if val is not None:
                for i, v in enumerate(np.atleast_1d(val)):
                    raw[f'{sensor.name}_{i}'] = float(v)

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
