from brain_base import BaseBrain, Param
from sensors import CollisionSensor, GradientSensor, DistanceSensor

class BrainObstacleAvoidance(BaseBrain):
    sensors = [
        GradientSensor(n=2, angle_spread=34.0, name='light'),
        DistanceSensor(n=2, angle_spread=34.0, max_range=2.0, name='distance'),
        CollisionSensor(n=2, angle_spread=90.0, arc_angle=90.0, radius=1.3, name='bumpers'),
    ]

    speed_base   = Param(50.0,  0, 100,    step=1.0, desc="Base forward speed")
    gain_ipsi    = Param(-50.0, -100, 100, step=1.0, desc="Weight for same-side sensor")
    gain_contra  = Param(0.0,   -100, 100, step=1.0, desc="Weight for opposite-side sensor")
    use_gradient = Param(1.0,   0, 1,      step=1.0, desc="1 = react to gradient patches, 0 = react to distance (walls+objects)")
    bump_gain    = Param(-80.0, -200, 0,   step=5.0, desc="Motor correction when bumper active")

    def setup(self):
        self.counter = 0

    def loop(self, dt):
        if self.use_gradient >= 0.5:
            sL, sR = self.light
        else:
            sL, sR = self.distance

        bL, bR = self.bumpers

        mL = self.speed_base + (sL * self.gain_ipsi) + (sR * self.gain_contra) + (bL * self.bump_gain)
        mR = self.speed_base + (sR * self.gain_ipsi) + (sL * self.gain_contra) + (bR * self.bump_gain)

        self.counter += 1
        return mL, mR

    def plots(self):
        return ['bumpers_0', 'bumpers_1']
