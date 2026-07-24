import numpy as np
from brain_base import BaseBrain, Param
from sensors import GradientSensor, CollisionSensor

class BrainARS(BaseBrain):
    sensors = [
        GradientSensor(n=2, angle_spread=0.4, name='light'),
        CollisionSensor(n=1, angle_spread=0, arc_angle=360.0, name='bump'),
    ]

    sensory_gain       = Param(50.0, 0, 100, step=1.0)
    base_drive         = Param(22.159, 0, 50, step=0.1)
    gate_decay         = Param(0.992, 0.8, 0.999, step=0.001)
    gate_rate          = Param(0.075, 0.001, 0.5, step=0.001)
    gate_gain          = Param(30.0, 1, 100, step=1.0)
    memory_decay       = Param(0.9, 0.5, 0.999, step=0.001)
    interneuron_decay  = Param(0.3, 0.1, 0.95, step=0.01)
    lateral_inhibition = Param(1.25, 0, 10.0, step=0.05)
    fatigue_rate       = Param(0.009, 0.0, 0.2, step=0.001)
    fatigue_gain       = Param(1.761, 0, 5.0, step=0.01)
    turn_boost         = Param(111.364, 0, 200, step=1.0)
    noise_drive        = Param(0.05, 0, 0.5, step=0.01)
    noise_fatigue      = Param(0.466, 0, 1.0, step=0.01)

    def setup(self):
        self.iL, self.iR = 0.0, 0.0
        self.fL, self.fR = 0.0, 0.0
        self.v_gate = 0.0
        self.gate = 0.0
        self.s_memory = 0.0

    def loop(self, dt):
        sL, sR = self.light
        s_total = sL + sR

        # 1. Memory & Gating
        if s_total > self.s_memory:
            self.s_memory = s_total
        else:
            self.s_memory *= self.memory_decay

        disappointment = max(0, self.s_memory - s_total)
        self.v_gate = (self.v_gate * self.gate_decay) + (disappointment * self.gate_rate)
        self.gate = np.tanh(self.v_gate * self.gate_gain)

        # 2. Interneurons logic
        drive_jitter = np.random.normal(0, self.noise_drive) if self.gate > 0.01 else 0
        drive = self.gate + drive_jitter

        self.fL = (self.fL * (1 - self.fatigue_rate)) + (self.iL * self.fatigue_rate)
        self.fR = (self.fR * (1 - self.fatigue_rate)) + (self.iR * self.fatigue_rate)

        f_noise = self.noise_fatigue
        f_sens_L = self.fatigue_gain * (1 + np.random.uniform(-f_noise, f_noise))
        f_sens_R = self.fatigue_gain * (1 + np.random.uniform(-f_noise, f_noise))

        next_iL = (self.iL * self.interneuron_decay + (drive * (1.0 + sL))
                   - (self.iR * self.lateral_inhibition) - (self.fL * f_sens_L))
        next_iR = (self.iR * self.interneuron_decay + (drive * (1.0 + sR))
                   - (self.iL * self.lateral_inhibition) - (self.fR * f_sens_R))

        self.iL, self.iR = max(0, next_iL), max(0, next_iR)

        # 3. Motors
        mL = self.base_drive + (sL * self.sensory_gain) + (self.iR * self.turn_boost)
        mR = self.base_drive + (sR * self.sensory_gain) + (self.iL * self.turn_boost)

        if self.bump[0] > 0.5:
            mL, mR = -10.0, 10.0

        return mL, mR

    def plots(self):
        return ['s_memory', 'gate', 'iL', 'iR', 'fL', 'fR']
