from brain_base import BaseBrain, Param
from sensors import GradientSensor, ColorSensor
from neurons import LeakyLayer, MatsuokaLayer, ConstantLayer, AdaptiveLayer, SumLayer
import numpy as np


class BrainMyBrain(BaseBrain):
    # GradientSensor detects soft gradient patches
    # ColorSensor detects solid colored objects
    sensors = [
        GradientSensor(n=2, angle_spread=0.4, name='light'),
    ]

    layers = []

    connections = []

    speed = Param(50.0, 0, 100, step=1.0, desc="Base speed")

    def setup(self):
        pass

    def loop(self, dt):
        sL, sR = self.light
        mL = self.speed
        mR = self.speed
        return mL, mR

    def plots(self):
        return []
