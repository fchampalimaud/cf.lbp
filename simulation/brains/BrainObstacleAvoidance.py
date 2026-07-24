from brain_base import BaseBrain, Param
from sensors import GradientSensor, DistanceSensor

class BrainObstacleAvoidance(BaseBrain):
    gradient_sensors = [GradientSensor(n=2, angle_spread=0.6, name='light')]
    distance_sensors = [DistanceSensor(n=2, angle_spread=0.6, max_range=1.0, name='distance')]

    speed_base   = Param(50.0,  0, 100,    step=1.0, desc="Base forward speed")
    gain_ipsi    = Param(-50.0, -100, 100, step=1.0, desc="Weight for same-side sensor")
    gain_contra  = Param(0.0,   -100, 100, step=1.0, desc="Weight for opposite-side sensor")
    use_gradient = Param(1.0,   0, 1,      step=1.0, desc="1 = react to gradient patches, 0 = react to distance (walls+objects)")

    def setup(self):
        self.counter = 0

    def loop(self, dt):
        if self.use_gradient >= 0.5:
            sL, sR = self.light
        else:
            sL, sR = self.distance

        mL = self.speed_base + (sL * self.gain_ipsi) + (sR * self.gain_contra)
        mR = self.speed_base + (sR * self.gain_ipsi) + (sL * self.gain_contra)

        self.counter += 1
        return mL, mR

    def plots(self):
        return []
