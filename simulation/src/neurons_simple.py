import numpy as np
import torch
from neurons_base import ACTIVATIONS, _activate, LayerBase

class ConstantLayer(LayerBase):
    """
    Outputs a fixed constant vector every step, ignoring any input.

    Used as a tonic (always-on) drive source. Connect it to other layers to
    provide baseline excitation — e.g. a ConstantLayer → motor connection sets
    the robot's cruising speed; a ConstantLayer → AdaptiveLayer drives the
    oscillator independently of sensory input.

    The value can be a scalar (broadcast to all n neurons) or a list of length n
    for per-neuron values.

    Parameters
    ----------
    value : float or list   Constant output. Scalar is broadcast to all neurons.
    n     : int             Number of neurons (inferred from value length if omitted).
    """

    help_text = """\
## ConstantLayer — fixed tonic drive

Outputs a fixed value every step, ignoring incoming connections.

**Parameters:**
- `value` (v) — constant output (scalar or list of length n); default 1.0
- `n` — number of neurons; default 2
- `noise_std` (σ) — Gaussian noise std added each step; default 0.0
**Output:**

$$\\text{output}_i = v + \\varepsilon_i, \\quad \\varepsilon_i \\sim \\mathcal{N}(0,\\,\\sigma)$$

`noise_std = 0` — pure constant. `value` can be a per-neuron list.

**Typical use:** tonic drive for downstream layers.
Keep `value` small (0.1–1.0) relative to downstream activation thresholds.
For ring attractors: use a **one-to-one** connection so each ring neuron gets
exactly `value`, not `value × n`.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published each tick.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute / derivative / integral). Only `site="post"` has effect (multiplies output by `1 + scale × signal`); `site="pre"` does nothing because this layer ignores incoming connections.
"""

    def __init__(self, value=1.0, n=None, noise_std=0.0, noise=None,
                 name='const', color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        super().__init__(name=name, color=color, layer=layer,
                         modulators=modulators,
                         neuromodulator_transmitter=neuromodulator_transmitter,
                         neuromodulator_color=neuromodulator_color)
        self.noise_std = float(noise if noise is not None else noise_std)
        self._value = np.asarray(value, dtype=float).ravel()
        if n is not None:
            self.n      = n
            self.output = self._make_output(n)
        elif self._value.size > 1:
            self.n      = self._value.size
            self.output = self._value.copy()
        else:
            self.n      = None
            self.output = None

    @classmethod
    def param_defs(cls):
        return [
            ('value',     float, '1.0', 'constant output value'),
            ('n',         int,   '2',   'number of neurons'),
            ('noise_std', float, '0.0', 'std dev of Gaussian noise added each step'),
        ]

    @property
    def value(self):
        return float(self._value[0]) if self._value.size == 1 else self._value.tolist()

    @value.setter
    def value(self, v):
        self._value = np.asarray(v, dtype=float).ravel()
        if self.n is not None:
            self.output = self._make_output(self.n)

    def _make_output(self, n):
        if self._value.size == 1:
            return np.full(n, float(self._value[0]))
        return np.asarray(self._value[:n], dtype=float)

    def _ensure_n(self, n):
        if self.n is None:
            self.n      = n
            self.output = self._make_output(n)
        elif self.n != n:
            raise ValueError(
                f"ConstantLayer '{self.name}': declared n={self.n} but connection implies n={n}")

    def reset(self):
        if self.n is not None:
            self.output = self._make_output(self.n)

    def step(self, _input_vec, _dt):
        if self.noise_std > 0.0:
            self.output = self._make_output(self.n) + np.random.normal(0.0, self.noise_std, self.output.shape)
        return self.output

    def on_unmute(self):
        # step() keeps the output as is, so the zeros of a mute would stay.
        if self.n is not None:
            self.output = self._make_output(self.n)

    def init_code_parts(self):
        v    = self._value
        vstr = repr(float(v[0])) if v.size == 1 else repr(v.tolist())
        parts = [f'value={vstr}', f'n={self.n}']
        parts += self._base_code_parts()
        if getattr(self, 'noise_std', 0.0):
            parts.append(f'noise_std={self.noise_std}')
        parts += self._mod_code_parts()
        return parts


class SumLayer(LayerBase):
    """
    Instantaneous linear combinator — no dynamics, no memory.

    Outputs the weighted sum of all incoming connections in the same step,
    optionally passed through a nonlinearity. Because there is no time constant,
    the output tracks its inputs without lag.

    Useful as an intermediate mixing/summing node before a recurrent layer,
    or as a direct motor output in sim-only networks (use MotorLayer instead
    when robot actuation is required).

    Contrast with LeakyLayer (filtered, with optional derivative mode) and
    AdaptiveLayer (filtered + adaptation + oscillation).

    Parameters
    ----------
    activation : str   Output nonlinearity: relu / sigmoid / tanh / linear.
    n          : int   Number of neurons (inferred from connections if None).
    """

    help_text = """\
## SumLayer — instantaneous linear combinator

No dynamics, no memory. Output tracks input in the **same step**.

**Parameters:**
- `activation` (f) — nonlinearity: `relu`, `sigmoid`, `tanh`, `linear`; default `relu`
- `scale` (s) — output multiplier; default 1.0
- `n` — number of neurons; default 2
**Output:**

$$\\text{output} = f\\!\\left(\\sum_k W_k \\cdot \\text{input}_k\\right) \\times s$$

- `activation='linear'` — pass-through (standard for motor output layer).
- `activation='relu'` — clips negative sums to zero.
- `scale = -1` — invert output without changing the weight matrix.

Also useful as a mixing / summing node before a recurrent layer.
Use **MotorLayer** (a SumLayer subclass) when you need robot actuation.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published to the bus each tick and can modulate any layer that lists it in `modulators`.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute / derivative / integral):
  - `site="pre"`: multiplies the input sum by `1 + scale × signal` before integration.
  - `site="post"`: multiplies the output by `1 + scale × signal` after integration.
  - `site="none"`: declares the neuromodulator for learning/visualization only — no signal amplification.
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
  - Feedback paths use the previous tick's value (one-step delay).
"""

    def __init__(self, activation='relu', scale=1.0, n=None, alpha=1.0, name='sum', color=None, layer=None,
                 modulators=None, neuromodulator_transmitter=None, neuromodulator_color=None):
        super().__init__(name=name, color=color, layer=layer,
                         modulators=modulators,
                         neuromodulator_transmitter=neuromodulator_transmitter,
                         neuromodulator_color=neuromodulator_color)
        self.activation = activation
        self.scale      = scale
        self.alpha      = float(alpha)
        self.n          = n
        self.output     = torch.zeros(n) if n is not None else None

    @classmethod
    def param_defs(cls):
        return [
            ('activation', str,   'relu', 'output nonlinearity', ACTIVATIONS),
            ('scale',      float, '1.0',  'output multiplier applied after activation'),
            ('n',          int,   '2',    'number of neurons'),
            ('alpha',      float, '1.0',  'ELU negative-side saturation level (only used when activation=elu)'),
        ]

    def _ensure_n(self, n):
        if self.n is None:
            self.n      = n
            self.output = torch.zeros(n)
        elif self.n != n:
            raise ValueError(f"SumLayer '{self.name}': input size mismatch, expected {self.n} got {n}")

    def reset(self):
        if self.n is not None:
            self.output = torch.zeros(self.n)

    def step(self, input_vec, _dt):
        inp = torch.as_tensor(input_vec, dtype=torch.float32)
        out = _activate(inp, self.activation, alpha=self.alpha) * self.scale
        self.output = out.detach()
        return self.output

    def init_code_parts(self):
        parts = []
        if self.activation != 'relu':
            parts.append(f"activation='{self.activation}'")
        parts += self._base_code_parts()
        if getattr(self, 'scale', 1.0) != 1.0:
            parts.append(f'scale={self.scale}')
        if getattr(self, 'alpha', 1.0) != 1.0:
            parts.append(f'alpha={self.alpha}')
        if self.n is not None:
            parts.append(f'n={self.n}')
        parts += self._mod_code_parts()
        return parts


class MotorLayer(SumLayer):
    """
    Motor output layer — SumLayer computation + physical actuation.

    In sim mode: drives wheel velocity or joint angle exactly like a SumLayer.
    In robot mode: the output is sent as an OSC message to *robot_address*.

    Use this instead of SumLayer whenever a layer directly commands a motor
    (wheels, joints, or any other actuator on real hardware).

    Parameters
    ----------
    activation   : str   Output nonlinearity (default 'linear' for motors).
    n            : int   Number of motor outputs (default 2 for left/right wheels).
    scale        : float Output multiplier.
    robot_address: str   Full target address: ip:port/osc_path
                         e.g. 192.168.0.1:2390/wheels
                         Leave empty to suppress sending in robot mode.
    """

    # Synthesized limb/whisker actuators (BrainManager.add_joint /
    # rebuild_joint_motor_layers) set this True on the instance.
    _is_joint_motor = False

    @property
    def drives_wheels(self):
        """True for a user motor layer (wheels); False for a joint actuator."""
        return not self._is_joint_motor

    help_text = """\
## MotorLayer — motor output with robot actuation

SumLayer computation (instantaneous weighted sum) with a **robot_address** that
routes output to physical hardware in real-robot mode.

**Parameters:**
- `activation` — nonlinearity; default `linear` (pass-through)
- `n` — number of motor outputs; default 2 (left wheel, right wheel)
- `scale` — output multiplier; default 1.0
- `robot_address` — full OSC target: `ip:port/osc_path`
**Output:**

$$\\text{output} = f\\!\\left(\\sum_k W_k \\cdot \\text{input}_k\\right) \\times s$$

In **sim mode** the output drives wheel velocity or joint angle via the circuit,
identical to a SumLayer.  In **robot mode** the output values are packed into an
OSC message and sent to `robot_address` each tick.

**Note:** manual-control override also writes through this layer so the circuit
sees what the wheels are actually doing.
"""

    def __init__(self, activation='linear', n=None, scale=1.0, alpha=1.0, robot_address='',
                 name='motor', color=None,
                 layer=None, modulators=None, neuromodulator_transmitter=None,
                 neuromodulator_color=None):
        super().__init__(activation=activation, scale=scale, alpha=alpha, name=name, n=n,
                         color=color, layer=layer,
                         modulators=modulators,
                         neuromodulator_transmitter=neuromodulator_transmitter,
                         neuromodulator_color=neuromodulator_color)
        self.robot_address = robot_address

    @classmethod
    def param_defs(cls):
        return [
            ('activation',    str,   'linear', 'output nonlinearity', ACTIVATIONS),
            ('n',             int,   '2',      'number of motor outputs'),
            ('scale',         float, '1.0',    'output multiplier'),
            ('alpha',         float, '1.0',    'ELU negative-side saturation level (only used when activation=elu)'),
            ('robot_address', str,   '',
             'ip:port/osc_path for robot mode (e.g. 192.168.0.1:2390/wheels)'),
        ]

    def init_code_parts(self):
        parts = super().init_code_parts()
        if self.robot_address:
            parts.append(f'robot_address={self.robot_address!r}')
        return parts


