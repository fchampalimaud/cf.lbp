from brain_base import BaseBrain, Param
from sensors import GradientSensor


class BrainMyFirst(BaseBrain):

    sensors = [GradientSensor(n=2, angle_spread=0.4, name='light')]

    speed = Param(60.0, 0, 100, step=1.0, desc='Maximum motor speed')

    def setup(self):
        pass

    def loop(self, dt):
        sL, sR = self.light

        # Crossed wiring: left sensor drives RIGHT motor, right sensor drives LEFT motor
        mL = self.speed * sR
        mR = self.speed * sL

        return mL, mR
