from brain_base import BaseBrain, Param
from sensors import GradientSensor

class BrainTemporalBraitenberg(BaseBrain):
    sensors = [GradientSensor(n=2, angle_spread=0, name='light')]

    speed_base = Param(30.0, 0, 100, step=1.0, desc="Base speed")
    osc_gain = Param(60.0, 0, 100, step=1.0, desc="Oscillation gain")
    osc_drive    = Param(4, 0, 10, step=0.1, desc="External stimulus (I_ext)")
    m_tau      = Param(0.05, 0.001, 1.0, step=0.001, desc="Membrane time constant")
    f_tau      = Param(0.1, 0.001, 5.0, step=0.01, desc="Fatigue recovery time")
    f_gain     = Param(5, 0, 10, step=0.1, desc="Fatigue suppression (beta)")
    w_ihn      = Param(2.0, 0, 10, step=0.1, desc="Mutual inhibition strength")
    ds_gain     = Param(70.0, 0, 1000, step=1.0, desc="Sensitivity to sensory change")

    def setup(self):
        self.ds = 0.0
        self.s, self.prev_s = 0.0, 0.0
        self.u1, self.u2 = 0.1, 0.2
        self.a1, self.a2 = 0.0, 0.0
        self.f1, self.f2 = 0.0, 0.0
        self.oscL, self.oscR = 0.0, 0.0
        self.biasL, self.biasR = 0.0, 0.0

    def loop(self, dt):
        sL, sR = self.light

        self.f1, self.f2 = self.get_oscillator_neurons(dt)

        self.s = 0.4 * (sL + sR) / 2 + 0.6 * self.s  # Low-pass filter for sensor input
        self.ds = self.s - self.prev_s  # Simple high-pass filter to detect changes
        
        self.oscL = self.f1 * self.osc_gain
        self.oscR = self.f2 * self.osc_gain
        
        self.biasL = self.ds_gain * self.ds * self.oscL
        self.biasR = self.ds_gain * self.ds * self.oscR
        
        mL = self.speed_base + self.oscL + self.biasL
        mR = self.speed_base + self.oscR + self.biasR

        mL = max(0, min(100, mL))
        mR = max(0, min(100, mR))
        
        self.prev_s = self.s
        return mL, mR

    def plots(self):
        return ["oscL", "oscR", "biasL", "biasR", "ds"]
    
    
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