from brain_base import BaseBrain, Param
from sensors import GradientSensor

class BrainOscillatorBraitenberg(BaseBrain):
    sensors = [GradientSensor(n=2, angle_spread=0.4, name='light')]

    speed_base = Param(40.0, 0, 100, step=1.0, desc="Base speed")
    osc_gain = Param(40.0, 0, 100, step=1.0, desc="Oscillation gain")
    osc_drive    = Param(3, 0, 10, step=0.1, desc="External stimulus (I_ext)")
    m_tau      = Param(0.05, 0.001, 1.0, step=0.001, desc="Membrane time constant")
    f_tau      = Param(0.1, 0.001, 5.0, step=0.01, desc="Fatigue recovery time")
    f_gain     = Param(5, 0, 10, step=0.1, desc="Fatigue suppression (beta)")
    w_ihn      = Param(2.0, 0, 10, step=0.1, desc="Mutual inhibition strength")

    def setup(self):
        self.u1, self.u2 = 0.1, 0.2
        self.a1, self.a2 = 0.0, 0.0
        self.f1, self.f2 = 0.0, 0.0

    def loop(self, dt):

        self.f1, self.f2 = self.get_oscillator_neurons(dt)

        mL = self.f1 * self.osc_gain + self.speed_base
        mR = self.f2 * self.osc_gain + self.speed_base

        return mL, mR

    def plots(self):
        return ["u1", "u2", "f1", "f2"]


    def get_oscillator_neurons(self, dt):
                # Differential equations for the oscillator
        self.du1 = (-self.u1 - (self.w_ihn * self.f2) - (self.f_gain * self.a1) + self.osc_drive) / self.m_tau
        self.du2 = (-self.u2 - (self.w_ihn * self.f1) - (self.f_gain * self.a2) + self.osc_drive) / self.m_tau

        self.da1 = (-self.a1 + self.f1) / self.f_tau
        self.da2 = (-self.a2 + self.f2) / self.f_tau

        # Euler integration
        self.u1 += self.du1 * dt
        self.u2 += self.du2 * dt
        self.a1 += self.da1 * dt
        self.a2 += self.da2 * dt

        # Firing rates (ReLU)
        self.f1 = max(0, self.u1)
        self.f2 = max(0, self.u2)
        return self.f1, self.f2
