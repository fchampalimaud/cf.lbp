import numpy as np
from brain_base import BaseBrain, Param
from sensors import GradientSensor, CollisionSensor
from neurons import ConstantLayer, LeakyLayer, SumLayer


class BrainNetworkGUI(BaseBrain):
    """
    Circuit:
        sugar -> forward -+
        temp  -> forward  +-> motor
        drive -> forward -+

        bumper -> asym (eye - fliplr)  -> back_long/back_short -> backward -> -motor
        bumper -> bilat (ones, slow)   -> wta (Matsuoka)        -> backward
    """

    # ── Sensors ───────────────────────────────────────────────────────────────
    sensors = [
        GradientSensor(n=2, angle_spread=30, scale=40, tau_rise=0.05, tau_decay=0.05, gradient='A', name='sugar'),
        GradientSensor(n=2, angle_spread=30, scale=40, tau_rise=0.05, tau_decay=0.05, gradient='B', name='temp'),
        CollisionSensor(n=2, angle_spread=90, scale=150, radius=1.3, tau_rise=0.05, tau_decay=0.05, arc_angle=60, name='bumper'),
    ]

    # ── Neural layers ─────────────────────────────────────────────────────────
    _W_ASYM = np.eye(2) - np.fliplr(np.eye(2))   # fires when one side > other

    layers = [
        SumLayer(activation='linear', name='motor', layer=4, n=2),
        ConstantLayer(value=1.0, n=2, name='Drive', layer=1),
        LeakyLayer(tau_rise=0.1, tau_decay=0.1, name='fw', layer=3, n=2),
        LeakyLayer(tau_rise=0.1, tau_decay=1.0, name='bw', layer=3, n=2),
        LeakyLayer(tau_rise=0.03, tau_decay=0.7, name='bp_long', layer=2, n=2),
        LeakyLayer(tau_rise=0.03, tau_decay=0.4, name='bp_short', layer=2, n=2),
    ]

    # ── Connectivity (source, target, weight_matrix) ──────────────────────────
    connections = [
        ('Drive', 'fw', 60.0 * np.eye(2)),
        ('sugar', 'fw', -np.eye(2)),
        ('fw', 'motor', np.eye(2)),
        ('bw', 'motor', -np.eye(2)),
        ('bumper', 'bp_long', np.fliplr(np.eye(2))),
        ('bp_long', 'bw', np.eye(2)),
        ('bumper', 'bp_short', np.eye(2)),
        ('bp_short', 'bw', np.eye(2)),
        ('temp', 'fw', np.eye(2)),
    ]


    # ── Brain logic ───────────────────────────────────────────────────────────
    def setup(self):
        self.mL = self.mR = 0.0

    def loop(self, dt):
        self.step_network(dt)
        motor = getattr(self, 'motor', None)
        if motor is not None and motor.output is not None:
            self.mL = float(motor.output[0])
            self.mR = float(motor.output[1])
        return self.mL, self.mR

    def plots(self):
        return []
