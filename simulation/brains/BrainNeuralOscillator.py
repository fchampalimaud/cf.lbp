import numpy as np
from brain_base import BaseBrain, Param
from sensors import GradientSensor
from neurons import LeakyLayer, MatsuokaLayer


class BrainNeuralOscillator(BaseBrain):
    """
    Half-centre oscillator with sensory change detection.

    A LeakyLayer low-pass filters the averaged sensor input; the derivative
    of that signal modulates the gain of a MatsuokaLayer half-centre oscillator
    to couple locomotion rhythm to light-field changes.
    """
    sensors = [GradientSensor(n=2, angle_spread=0, name='light')]

    speed_base = Param(30.0, 0,     100,  step=1.0,   desc="Base speed")
    osc_gain   = Param(60.0, 0,     100,  step=1.0,   desc="Oscillation gain")
    osc_drive  = Param(4.0,  0,     10,   step=0.1,   desc="Constant drive to oscillator")
    tau_m      = Param(0.05, 0.001, 1.0,  step=0.001, desc="Membrane time constant")
    tau_a      = Param(0.1,  0.001, 5.0,  step=0.01,  desc="Adaptation time constant")
    beta       = Param(5.0,  0,     10,   step=0.1,   desc="Adaptation gain (beta)")
    w_ihn      = Param(2.0,  0,     10,   step=0.1,   desc="Mutual inhibition weight")
    tau_s      = Param(0.25, 0.01,  2.0,  step=0.01,  desc="Sensor low-pass time constant")
    ds_gain    = Param(70.0, 0,     1000, step=1.0,   desc="Sensory change gain")

    def setup(self):
        self._smoother = LeakyLayer(tau_rise=self.tau_s, tau_decay=self.tau_s, activation='linear', n=1, name='smoother')
        self._osc      = MatsuokaLayer(tauM=self.tau_m, tauA=self.tau_a,
                                       beta=self.beta, w=self.w_ihn)
        self._smoother.reset()
        self._osc.reset()
        self.prev_s = 0.0
        self.oscL = self.oscR = 0.0
        self.biasL = self.biasR = 0.0
        self.ds = 0.0

    def loop(self, dt):
        sL, sR = self.light

        # Sync slider values each tick so live changes take effect
        self._osc.tauM  = self.tau_m
        self._osc.tauA  = self.tau_a
        self._osc.beta  = self.beta
        self._osc.w     = self.w_ihn
        self._smoother.tau_rise = self._smoother.tau_decay = self.tau_s

        # Sensor smoothing
        self._smoother.step(np.array([(sL + sR) / 2.0]), dt)
        s = float(self._smoother.output[0])

        # Half-centre oscillator
        self._osc.step(np.array([self.osc_drive, self.osc_drive]), dt)

        # Sensory change detection
        self.ds     = s - self.prev_s
        self.prev_s = s

        # Motor outputs
        self.oscL  = float(self._osc.output[0]) * self.osc_gain
        self.oscR  = float(self._osc.output[1]) * self.osc_gain
        self.biasL = self.ds_gain * self.ds * self.oscL
        self.biasR = self.ds_gain * self.ds * self.oscR

        mL = max(0, min(100, self.speed_base + self.oscL + self.biasL))
        mR = max(0, min(100, self.speed_base + self.oscR + self.biasR))
        return mL, mR

    def plots(self):
        return ['oscL', 'oscR', 'biasL', 'biasR', 'ds']
