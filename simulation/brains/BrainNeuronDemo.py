import numpy as np
from brain_base import BaseBrain, Param
from sensors import GradientSensor
from neurons import LeakyLayer, MatsuokaLayer


class BrainNeuronDemo(BaseBrain):
    """
    Sensor input is smoothed by leaky integrators, then drives
    a Matsuoka half-centre oscillator that generates rhythmic motor output.

    Circuit:
        light[0,1]  ──W1──>  smoother[0,1]  ──W2──>  osc[0,1]  ──>  motors
    """

    # ── Sensors ───────────────────────────────────────────────────────────────
    sensors = [GradientSensor(n=2, angle_spread=0.5, name='light')]

    # ── Neural layers ─────────────────────────────────────────────────────────
    layers = [
        LeakyLayer(tau_rise=0.05, tau_decay=0.05, name='smoother'),
        MatsuokaLayer(tauM=0.1, tauA=0.4, beta=2.5, w=2.5, bias=2.0, name='osc'),
        LeakyLayer(tau_rise=0.02, tau_decay=0.02, name='motor'),
    ]

    # ── Connectivity (source, target, weight_matrix) ──────────────────────────
    connections = [
        ('light',    'smoother', np.array([[1.0, 0.0],
                                           [0.0, 1.0]])),
        ('smoother', 'osc',      np.array([[1.0, 0.0],
                                           [0.0, 1.0]])),
        ('osc', 'motor',         np.array([[1.0, 0.0],
                                           [0.0, 1.0]])),
    ]

    # ── Tunable parameters ────────────────────────────────────────────────────
    speed = Param(60.0, 0.0, 100.0, step=1.0, desc="Motor gain")

    # ── Brain logic ───────────────────────────────────────────────────────────
    def setup(self):
        self.osc_L = self.osc_R = 0.0

    def loop(self, dt):
        self.step_network(dt)
        self.osc_L = float(self.motor.output[0])
        self.osc_R = float(self.motor.output[1])
        return self.osc_L * self.speed, self.osc_R * self.speed

    def plots(self):
        return ['osc_L', 'osc_R']
