from brain_base import Param, BaseConfig


class SimConfig(BaseConfig):
    dt           = Param(0.01, 0.001, 0.5,  step=0.001, desc="Simulation time step (s)")
    arena_scale  = Param(5.0,  1.0,   20.0, step=0.1,   desc="Arena half-width (m)")
    motor_gain   = Param(1.0,  0.0,   10.0, step=0.1,   desc="Motor speed multiplier")
    sense_radius = Param(1.0,  0.1,   10.0, step=0.05,  desc="Sensor ray length (m)")
    init_x       = Param(0.0, -10.0,  10.0, step=0.1,   desc="Robot start X position")
    init_y       = Param(0.0, -10.0,  10.0, step=0.1,   desc="Robot start Y position")
    stim_radius  = Param(0.5,  0.1,   5.0,  step=0.05,  desc="Radius of new stimulus patches")
    body_radius  = Param(0.2,  0.05,  1.0,  step=0.01,  desc="Robot body radius (m)")
    sensor_angle = Param(0.2,  0.0,   3.14, step=0.01,  desc="Sensor angle offset (legacy brains)")
    toggle_stim  = Param(1,    0,     1,    step=1,      desc="Show / hide stimulus patches")
    fixate_robot = Param(0.0,  0,     1,    step=1,      desc="Freeze robot position (debug)")
