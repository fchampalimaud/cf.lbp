from brain_base import BaseBrain, Param
from sensors import GradientSensor

class BrainBraitenberg(BaseBrain):
    sensors = [GradientSensor(n=2, angle_spread=0.4, name='light')]

    speed_base  = Param(50.0,  0, 100,    step=1.0,  desc="Base forward speed")
    gain_ipsi   = Param(-50.0, -100, 100, step=1.0,  desc="Weight for same-side sensor")
    gain_contra = Param(0.0,   -100, 100, step=1.0,  desc="Weight for opposite-side sensor")

    def setup(self):
        self.counter = 0

    def loop(self, dt):
        sL, sR = self.light

        mL = self.speed_base + (sL * self.gain_ipsi) + (sR * self.gain_contra)
        mR = self.speed_base + (sR * self.gain_ipsi) + (sL * self.gain_contra)

        self.counter += 1
        return mL, mR

    def plots(self):
        return []