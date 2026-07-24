import numpy as np
from brain_base import BaseBrain, Param
from sensors import GradientSensor, CollisionSensor
from neurons import LeakyLayer, SumLayer, ConstantLayer, MatsuokaLayer


class BrainNeuron2GradientsAndBumpers(BaseBrain):
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
        ConstantLayer(value=60.0, n=2, name='drive'),
        LeakyLayer(tau_rise=0.05, tau_decay=0.05, name='forward', layer=4),

        # Asymmetry path: active on single-sided collision
        LeakyLayer(tau_rise=0.05, tau_decay=0.05, name='asym', color='#A8C8E8', layer=2),
        LeakyLayer(tau_rise=0.05, tau_decay=1.0, name='back_long',  color='#EEC898', layer=3),
        LeakyLayer(tau_rise=0.05, tau_decay=0.5, name='back_short', color='#EEC898', layer=3),

        # Bilateral path: active on symmetric collision
        LeakyLayer(tau_rise=0.05, tau_decay=0.8, name='bilat', color='#F0A0BC', layer=2),
        MatsuokaLayer(tauM=0.15, tauA=0.6, beta=2.5, w=2.5,   name='wta',   color='#C0A0D0', layer=3),

        SumLayer(activation='relu',   name='backward', layer=4),
        SumLayer(activation='linear', name='motor'),
    ]

    # ── Connectivity (source, target, weight_matrix) ──────────────────────────
    connections = [
        # Forward drive
        ('sugar',      'forward',    -np.eye(2)),
        ('temp',       'forward',     np.eye(2)),
        ('drive',      'forward',     np.eye(2)),

        # Asymmetry detector: fires when one bumper > other
        ('bumper',     'asym',        _W_ASYM),
        ('asym',       'back_long',   np.eye(2)),
        ('asym',       'back_short',  np.eye(2)),
        ('back_long',  'backward',    np.fliplr(np.eye(2))),
        ('back_short', 'backward',    np.eye(2)),

        # Bilateral gate: sums both bumpers, drives Matsuoka
        ('bumper',     'bilat',       np.ones((2, 2))),
        ('bilat',      'wta',         np.eye(2)),
        ('wta',        'backward',    np.eye(2)),

        # Motor output
        ('forward',    'motor',       np.eye(2)),
        ('backward',   'motor',      -np.eye(2)),
    ]


    # ── Brain logic ───────────────────────────────────────────────────────────
    def setup(self):
        self.mL = self.mR = 0.0

    def loop(self, dt):
        self.step_network(dt)
        self.mL = float(self.motor.output[0])
        self.mR = float(self.motor.output[1])
        return self.mL, self.mR
