import os
import numpy as np
from neurons import _activate, DynamicsBase, ACTIVATIONS, OUTPUT_MODES

SENSOR_DIST = 0.12

_GRADIENT_LABEL_COLOR = {
    'A': '#FFAAAA', 'B': '#AAFFAA', 'C': '#AAAAFF',
    'D': '#FFFFA0', 'E': '#FFAAFF', 'F': '#AAFFFF',
}


class BaseSensor(DynamicsBase):
    """
    Base class for all sensors.
    The simulator calls sample() each tick and stores the result as brain.<name>.

    Shared parameters:
        tau_rise     : rise time constant. None = no dynamics (pass-through).
        tau_decay    : decay time constant. Defaults to tau_rise when only one is given.
        activation   : 'linear' (default), 'relu', 'sigmoid', or 'tanh'.
        output_mode  : 'none' (default) / 'derivative' / 'integral' — applied to the
                       raw (post-noise) reading BEFORE the leaky filter/activation
                       above run. 'derivative' replaces it with the rate-of-change
                       (dRaw/dt), zero on the first tick; 'integral' replaces it
                       with the running accumulation (∫Raw dt).
        group        : circuit group name for network view show/hide.
        body_id      : id of the RigidBody this sensor is mounted on ('root' = drive disk).

    Visualization:
        viz_type   : string key used by ArenaWidget to choose the rendering item type.
                     None = sensor has no arena overlay.
        _viz_dashed: for viz_type='ray', whether to draw as a dashed line.
        _viz_lw    : for viz_type='ray', line width in pixels.
    """
    name                       = 'sensor'
    group                      = None
    neuromodulator_transmitter = None   # name of the signal this sensor emits to the mod bus
    neuromodulator_color       = None   # hex color for this transmitter in the visualizer
    robot_address  = ''   # remote host:port this sensor reads from in real-robot mode
    body_ids       = ['root']  # list of rigid body IDs to sample from; one output per body
    _robot_value   = None  # latest value written by RobotDriver; None = no data yet
    z              = 0     # subsumption depth plane (top-view y-axis)
    span           = 1     # number of column slots this container spans (top-view)

    # Visualization — subclasses override these instead of requiring isinstance checks.
    viz_type   = None   # 'ray' | 'arc' | 'touch' | 'mouth' | 'whisker' | 'sky' | 'camera_fov' | None
    _viz_dashed = True  # for viz_type == 'ray': dashed vs solid
    _viz_lw     = 1.0   # for viz_type == 'ray': line width

    @property
    def body_id(self):
        return self.body_ids[0] if self.body_ids else 'root'

    @body_id.setter
    def body_id(self, value):
        self.body_ids = [value] if value else ['root']

    @property
    def n_total(self):
        """Total number of outputs: n outputs per body × number of bodies."""
        return (self.n or 0) * max(1, len(self.body_ids))

    output_mode  = 'none'   # class-level default; overridden per instance
    _prev_output = None     # cleared on reset(); used by output_mode='derivative'
    _integral    = None     # cleared on reset(); used by output_mode='integral'
    _noise_buf   = None     # cleared on reset(); OU noise state, used when noise_tau > 0
    _mod_row_state = None   # dict, cleared/initialized on reset(); per-(name,mode) modulator-row state

    # Set True by subclasses that already add their own noise_std-scaled noise
    # in sample() (e.g. gated by a hit/event) before calling _process() — prevents
    # _process()'s generic noise stage from applying the same noise_std a second time.
    _noise_applied_in_sample = False

    # Shared channel index — subclasses may override _CHANNEL_COLOR
    _CHANNEL_IDX = {'R': 0, 'G': 1, 'B': 2}

    def _apply_output_mode(self, out, dt):
        """Transform *out* per self.output_mode: 'none' passthrough,
        'derivative' (rate-of-change, zeros on the first tick), or
        'integral' (running ∫out dt, forward-Euler accumulation)."""
        if self.output_mode == 'derivative':
            prev = self._prev_output
            if prev is None or prev.shape != out.shape:
                self._prev_output = out.copy()
                return np.zeros_like(out)
            d = (out - prev) / max(dt, 1e-9)
            self._prev_output = out.copy()
            return d
        if self.output_mode == 'integral':
            if self._integral is None or self._integral.shape != out.shape:
                self._integral = np.zeros_like(out)
            self._integral = self._integral + out * dt
            return self._integral.copy()
        return out

    def _transform_modulator_value(self, key, mode, value, dt):
        """Apply a modulator response mode to a raw scalar reading from the mod bus.

        See `LayerBase._transform_modulator_value` (neurons_base.py) for the
        full contract — this is the sensor-side twin, kept independent of
        this sensor's own `_prev_output`/`_integral` (its own output_mode
        state) and of every other modulator row.
        """
        if self._mod_row_state is None:
            self._mod_row_state = {}
        if mode == 'derivative':
            state = self._mod_row_state.setdefault(key, {'prev': value})
            prev = state['prev']
            state['prev'] = value
            return (value - prev) / max(float(dt), 1e-9)
        if mode == 'integral':
            state = self._mod_row_state.setdefault(key, {'integral': 0.0})
            state['integral'] += value * dt
            return state['integral']
        return value

    def _apply_noise(self, raw, dt):
        """Add noise to raw. White noise (fresh independent sample each tick)
        if noise_tau == 0; otherwise an Ornstein-Uhlenbeck process with that
        correlation time — numpy equivalent of DynamicsBase._apply_noise
        (neurons_base.py), which sensors can't use directly since it's torch-
        based and depends on _init_dynamics_buffers, a layer-only call."""
        noise_std = getattr(self, 'noise_std', 0.0)
        if not noise_std:
            return raw
        noise_tau = getattr(self, 'noise_tau', 0.0)
        if noise_tau and noise_tau > 0:
            if self._noise_buf is None or self._noise_buf.shape != np.shape(raw):
                self._noise_buf = np.zeros_like(raw, dtype=float)
            self._noise_buf = self._noise_buf + (
                -self._noise_buf / noise_tau + noise_std * np.random.randn(*np.shape(raw))
            ) * dt
            return raw + self._noise_buf
        return raw + np.random.randn(*np.shape(raw)) * noise_std

    def _ray_angles(self):
        """Fan of n ray angles centred at self.center_angle ± self.angle_spread/2."""
        if self.n == 1:
            return np.array([getattr(self, 'center_angle', 0.0)])
        spread = getattr(self, 'angle_spread', 0.0)
        centre = getattr(self, 'center_angle', 0.0)
        return np.linspace(centre + spread / 2, centre - spread / 2, self.n)

    def _process(self, raw, sim_cfg):
        """Apply bias, optional noise, output_mode, asymmetric leaky dynamics,
        activation, and finally scale — same pipeline order as DynamicsBase-driven
        layers (see LeakyLayer.step): bias joins the raw input first, scale is the
        only thing applied after activation.

        output_mode transforms the raw (post-noise) reading BEFORE the leaky
        filter/activation below run, not the final output — otherwise an
        integral would keep growing during the leaky filter's own decay tail
        even after the true reading returned to zero, since it never touched
        the raw signal at all.
        """
        raw = raw + getattr(self, 'bias', 0.0)
        if not self._noise_applied_in_sample:
            raw = self._apply_noise(raw, sim_cfg.dt)
        raw = self._apply_output_mode(raw, sim_cfg.dt)
        tr = getattr(self, 'tau_rise', None)
        td = getattr(self, 'tau_decay', None)
        if tr is not None or td is not None:
            if tr is None: tr = td
            if td is None: td = tr
            if self._x is None:
                self._x = np.zeros_like(raw, dtype=float)
            tau = np.where(raw > self._x, tr, td)
            self._x += (raw - self._x) / tau * sim_cfg.dt
            out = self._x.copy()
        else:
            out = raw
        self._pre_activation_output = out.copy()
        out = _activate(out, getattr(self, 'activation', 'linear'))
        out = out * getattr(self, 'scale', 1.0)
        return out

    @classmethod
    def _sensor_base_param_defs(cls):
        """Canonical base param entries appended to every sensor's dialog."""
        return [
            ('noise_std',     float, 0.0,      'Gaussian noise σ added each tick (0 = off)'),
            ('noise_tau',     float, 0.0,      'OU correlation time (0 = white noise)'),
            ('tau_rise',      float, '',        'rise τ in seconds (empty = passthrough)'),
            ('tau_decay',     float, '',        'decay τ in seconds (empty = passthrough)'),
            ('activation',    str,   'linear',  'nonlinearity applied after dynamics',
             ACTIVATIONS),
            ('output_mode',   str,   'none',    'output transform: none / derivative (dV/dt) / integral (∫V dt)',
             OUTPUT_MODES),
            ('robot_address', str,   '',        'host:port/osc_path[types](indices) for real-robot mode'),
        ]

    @classmethod
    def _sensor_neuromod_param_defs(cls):
        """Neuromodulation transmitter entries appended to every sensor's dialog."""
        return [
            ('neuromodulator_transmitter', str, '', 'name of signal this sensor emits (e.g. dopamine)'),
            ('neuromodulator_color',       str, '', 'hex color for this transmitter (e.g. #FF6600)'),
        ]

    def reset(self):
        self._x = None
        self._prev_output = None
        self._integral = None
        self._noise_buf = None
        self._mod_row_state = {}

    def is_lateralized(self, circuit=None) -> bool:
        """True when this sensor has a lateralized L/R structure (camera or mirrored joint pair)."""
        if getattr(self, 'lateralized', False):
            return True
        body_ids = getattr(self, 'body_ids', ['root'])
        if len(body_ids) != 2:
            return False
        if circuit is None:
            return True
        body_map = {b.id: b for b in getattr(circuit, 'bodies', [])}
        b0 = body_map.get(body_ids[0])
        b1 = body_map.get(body_ids[1])
        if b0 is None or b1 is None:
            return False
        mg0 = getattr(b0, 'mirror_group', None)
        mg1 = getattr(b1, 'mirror_group', None)
        return bool(mg0 and mg0 == mg1)

    # ── Visualization / serialization protocol ─────────────────────────────────

    is_image_node = False  # overridden to True by CameraSensor

    _ANGLE_ATTRS = frozenset({'angle_spread', 'center_angle', 'arc_angle', 'fov', 'vertical_angle'})

    def thumbnail_frames(self, disp_h=32):
        """Yield (key, uint8_data) tuples for image thumbnail display. Default: nothing."""
        yield from ()

    @property
    def viz_color(self):
        """Color used for the node in the network visualizer."""
        return getattr(self, '_viz_color', None)

    def n_per_side(self):
        """Number of outputs per lateralized half (1 for cameras, self.n for other sensors)."""
        return self.n or 1

    def process_robot_value(self, raw, sim_cfg) -> np.ndarray:
        """Apply the full pipeline (bias, noise, tau, activation, scale) to a raw robot input.

        Subclasses override this to add sensor-specific noise.
        """
        raw = np.asarray(raw, dtype=float)
        return np.asarray(self._process(raw, sim_cfg), dtype=np.float32)

    def sample(self, x, y, theta, world, sim_cfg) -> np.ndarray:
        raise NotImplementedError


class GradientSensor(BaseSensor):
    """
    Detects soft gradient patches. Casts n rays in a fan, each returning the
    stimulus intensity for the selected gradient label ('A', 'B', …).

    Example:
        GradientSensor(n=2, angle_spread=30, gradient='A', name='sugar')
        GradientSensor(n=2, angle_spread=30, name='light')   # responds to all
    """

    help_text = """\
## GradientSensor — soft gradient detector

Casts `n` rays in a fan and reads gradient intensity at each tip.

$$\\alpha_i = \\text{center\\_angle} + \\text{fan}(i, n, \\text{angle\\_spread})$$

$$p_i = \\bigl(x + d\\cos(\\theta+\\alpha_i),\\; y + d\\sin(\\theta+\\alpha_i)\\bigr)$$

$$r_i = \\text{gradient}(p_i,\\; \\text{label})$$

`gradient = ''` — responds to all labels. `color_channel = 'R'/'G'/'B'` — single channel.

**Output pipeline** (all sensors):

$$\\tau = \\begin{cases}\\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x\\end{cases}, \\quad \\frac{dx}{dt} = \\frac{u - x}{\\tau}$$

$$\\text{output} = f(x) \\times \\text{scale} \\quad (\\text{or}\\ f(u) \\times \\text{scale}\\ \\text{if no dynamics})$$

**Order of operations** (per tick):
1. `r_i = gradient(p_i, label)` — raw ray reading above
2. `u = r_i + bias`
3. add noise to `u` (if `noise_std > 0`)
4. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
5. `x = leaky(u)` — asymmetric τ_rise/τ_decay integration (passthrough if no τ set)
6. `output = activation(x) × scale`

---

**Neuromodulation:**

- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute / derivative / integral). Multiplies the sensor output each tick by `1 + Σ (scale × signal)`. The `site` field is accepted but not used — all modulators contribute a single gain.
  - `scale > 0` → excitatory (boosts sensitivity); `scale < 0` → inhibitory.
"""

    _CHANNEL_COLOR = {'R': '#FF8888', 'G': '#88CC88', 'B': '#88AAFF', None: None}
    viz_type    = 'ray'
    _viz_dashed = True
    _viz_lw     = 1.2

    def __init__(self, n=2, angle_spread=30.0, center_angle=0.0, dist=SENSOR_DIST,
                 color_channel='', gradient=None, scale=1.0, bias=0.0, tau_rise=None,
                 tau_decay=None, activation='linear',
                 output_mode='none', noise_std=0.0, noise_tau=0.0, name='light', group=None,
                 modulators=None, robot_address=''):
        self.n              = n
        self.angle_spread   = np.radians(angle_spread)
        self.center_angle   = np.radians(center_angle)
        self.dist           = dist
        self.color_channel  = color_channel
        self.gradient       = gradient
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay,
                            activation=activation, scale=scale, bias=bias,
                            noise_std=noise_std, noise_tau=noise_tau, output_mode=output_mode)
        self.name           = name
        self.group          = group
        self.modulators     = modulators or []
        self.robot_address = robot_address
        self._x            = None
        if gradient is not None:
            self._viz_color = _GRADIENT_LABEL_COLOR.get(gradient)
        else:
            self._viz_color = self._CHANNEL_COLOR.get(color_channel)

    @classmethod
    def param_defs(cls):
        return [
            ('n',             int,   2,        'number of rays'),
            ('angle_spread',  float, np.radians(30.0), 'fan width in degrees'),
            ('center_angle',  float, 0.0,             'center angle in degrees'),
            ('dist',          float, 0.12,             'placement distance'),
            ('color_channel', str,   '',               'R, G, B or empty for all'),
            ('gradient',      str,   '',               'label A–F or empty for all'),
            ('scale',         float, 1.0,      'output scale'),
            ('bias',          float, 0.0,      'constant offset added after scale'),
            ('tau_rise',      float, '',       'rise τ (empty = passthrough)'),
            ('tau_decay',     float, '',       'decay τ (empty = passthrough)'),
            ('activation',    str,   'linear', 'output activation', ACTIVATIONS),
            ('output_mode',   str,   'none',   'none / derivative / integral', OUTPUT_MODES),
        ]

    def sample(self, x, y, theta, world, sim_cfg) -> np.ndarray:
        ch   = self._CHANNEL_IDX.get(self.color_channel)
        vals = []
        for a in self._ray_angles():
            sa = theta + a
            sx = x + self.dist * np.cos(sa)
            sy = y + self.dist * np.sin(sa)
            vals.append(world.get_signal(sx, sy, sa, ch, label=self.gradient))
        return self._process(np.array(vals), sim_cfg)


class ColorSensor(BaseSensor):
    """
    Detects solid colored objects via ray-circle intersection.

    Example:
        ColorSensor(n=2, angle_spread=30, name='objects')
        ColorSensor(n=3, angle_spread=60, color_channel='R', name='red_obj')
    """

    help_text = """\
## ColorSensor — solid object detector

Casts `n` rays in a fan and detects solid coloured circular objects.

**Geometry:** same fan as `GradientSensor` — rays span `angle_spread` degrees around `center_angle`.

**Raw signal** (per ray `i`):

$$r_i = \\text{object\\_color}(\\text{ray}_i,\\; \\text{channel})$$

Returns 1.0 on a hit (0.0 otherwise) if no colour filter. `color_channel = 'R'/'G'/'B'` reads only that channel.

**Output pipeline** (all sensors):

$$\\tau = \\begin{cases}\\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x\\end{cases}, \\quad \\frac{dx}{dt} = \\frac{u - x}{\\tau}$$

$$\\text{output} = f(x) \\times \\text{scale} \\quad (\\text{or}\\ f(u) \\times \\text{scale}\\ \\text{if no dynamics})$$

**Order of operations** (per tick):
1. `r_i = object_color(ray_i, channel)` — raw ray reading above
2. `u = r_i + bias`
3. add noise to `u` (if `noise_std > 0`)
4. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
5. `x = leaky(u)` — asymmetric τ_rise/τ_decay integration (passthrough if no τ set)
6. `output = activation(x) × scale`

---

**Neuromodulation:**

- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute / derivative / integral). Multiplies the sensor output each tick by `1 + Σ (scale × signal)`. The `site` field is accepted but not used — all modulators contribute a single gain.
  - `scale > 0` → excitatory (boosts sensitivity); `scale < 0` → inhibitory.
"""

    _CHANNEL_COLOR = {'R': '#FF5555', 'G': '#55BB55', 'B': '#5555FF', None: None}
    viz_type    = 'ray'
    _viz_dashed = False
    _viz_lw     = 1.5

    def __init__(self, n=2, angle_spread=30.0, center_angle=0.0, dist=SENSOR_DIST,
                 color_channel='', scale=1.0, bias=0.0, tau_rise=None,
                 tau_decay=None, activation='linear',
                 output_mode='none', noise_std=0.0, noise_tau=0.0, name='objects', group=None,
                 modulators=None, robot_address=''):
        self.n              = n
        self.angle_spread   = np.radians(angle_spread)
        self.center_angle   = np.radians(center_angle)
        self.dist           = dist
        self.color_channel  = color_channel
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay,
                            activation=activation, scale=scale, bias=bias,
                            noise_std=noise_std, noise_tau=noise_tau, output_mode=output_mode)
        self.name           = name
        self.group          = group
        self.modulators     = modulators or []
        self.robot_address = robot_address
        self._x            = None
        self._viz_color    = self._CHANNEL_COLOR.get(color_channel)

    @classmethod
    def param_defs(cls):
        return [
            ('n',             int,   2,        'number of rays'),
            ('angle_spread',  float, np.radians(30.0), 'fan width in degrees'),
            ('center_angle',  float, 0.0,             'center angle in degrees'),
            ('dist',          float, 0.12,             'placement distance'),
            ('color_channel', str,   '',               'R, G, B or empty for all'),
            ('scale',         float, 1.0,              'output scale'),
            ('bias',          float, 0.0,      'constant offset added after scale'),
            ('tau_rise',      float, '',       'rise τ (empty = passthrough)'),
            ('tau_decay',     float, '',       'decay τ (empty = passthrough)'),
            ('activation',    str,   'linear', 'output activation', ACTIVATIONS),
            ('output_mode',   str,   'none',   'none / derivative / integral', OUTPUT_MODES),
        ]

    def sample(self, x, y, theta, world, sim_cfg) -> np.ndarray:
        ch   = self._CHANNEL_IDX.get(self.color_channel)
        vals = []
        for a in self._ray_angles():
            sa = theta + a
            sx = x + self.dist * np.cos(sa)
            sy = y + self.dist * np.sin(sa)
            vals.append(world.get_object_signal(sx, sy, sa, ch))
        return self._process(np.array(vals), sim_cfg)



class CollisionSensor(BaseSensor):
    """
    Detects contact within n configurable arc sectors around the robot perimeter.
    Returns 1.0 for each sector touching a wall or object, 0.0 otherwise.

    Example:
        CollisionSensor(n=2, angle_spread=90, arc_angle=90, name='touch')
    """

    help_text = """\
## CollisionSensor — contact detector

Checks `n` arc sectors around the robot perimeter for wall or object contact.

**Sector layout:**

Sector centres are evenly distributed across `angle_spread`, symmetric around the robot heading:

$$c_i = \\theta + \\frac{\\text{angle\\_spread}}{2} - i \\cdot \\frac{\\text{angle\\_spread}}{n-1}, \\quad i = 0 \\ldots n-1$$

(For `n=1` the single sector is centred on the heading.)

**Probe geometry** (per sector `i`):

Each sector samples `n_pts = max(5, \\lfloor arc\\_angle_{deg} / 5 \\rfloor)` probe points along the arc at radius $r = r_{body} \\times \\text{radius}$:

$$p_k = \\bigl(x + r\\cos(a_k),\\; y + r\\sin(a_k)\\bigr), \\quad a_k \\in \\left[c_i - \\tfrac{\\text{arc\\_angle}}{2},\\; c_i + \\tfrac{\\text{arc\\_angle}}{2}\\right]$$

A hit is registered if any probe falls outside the arena boundary or overlaps an object or wall segment, using:

$$\\text{wall\\_threshold} = \\max\\!\\left(0.01,\\; r_{body} \\times (\\text{radius} - 0.9)\\right)$$

**Raw detection** (per sector `i`):

$$h_i = \\begin{cases}1 & \\text{any probe in sector } i \\text{ touches an obstacle} \\\\ 0 & \\text{otherwise}\\end{cases}$$

$$o_i = h_i + h_i \\cdot \\text{noise\\_std} \\cdot \\varepsilon_i, \\quad \\varepsilon_i \\sim \\mathcal{N}(0,1)$$

(Noise is only added when there is a hit, and is applied here — directly on the
raw 0/1 signal — rather than in the generic noise stage below, so `noise_std`
is in raw-signal units, not output-scale units.)

**Output pipeline** (all sensors):

$$\\tau = \\begin{cases}\\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x\\end{cases}, \\quad \\frac{dx}{dt} = \\frac{u - x}{\\tau}$$

$$\\text{output} = f(x) \\times \\text{scale} \\quad (\\text{or}\\ f(u) \\times \\text{scale}\\ \\text{if no dynamics})$$

**Order of operations** (per tick):
1. `h_i = 1` if any probe in sector `i` touches an obstacle, else `0`
2. `o_i = h_i + h_i × noise_std × ε_i` — noise gated on a hit, added directly to the raw signal (no separate noise stage below)
3. `u = o_i + bias`
4. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
5. `x = leaky(u)` — asymmetric τ_rise/τ_decay integration (passthrough if no τ set)
6. `output = activation(x) × scale`

- `n` — number of arc sectors (outputs).
- `angle_spread` — total angular span covered by all sectors (degrees).
- `arc_angle` — angular width of each individual sector (degrees); controls detection resolution within a sector.
- `radius` — probe radius as a multiplier on body radius (`1.0` = robot surface, `> 1.0` = lookahead).

In multi-agent sessions, sectors also fire on contact with other agents' bodies (treated as circles of `body_radius`), not just walls/objects.

---

**Neuromodulation:**

- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute / derivative / integral). Multiplies the sensor output each tick by `1 + Σ (scale × signal)`. The `site` field is accepted but not used — all modulators contribute a single gain.
  - `scale > 0` → excitatory (boosts sensitivity); `scale < 0` → inhibitory.
"""

    _ACTIVE_COLOR = '#FF6633'  # kept for backward compatibility
    viz_type      = 'arc'
    _noise_applied_in_sample = True  # noise is hit-gated in sample(); don't double-apply in _process()

    def __init__(self, n=4, angle_spread=90.0, arc_angle=45.0,
                 radius=1.2, scale=1.0, bias=0.0, noise_std=0.0, noise_tau=0.0, tau_rise=None,
                 tau_decay=None, activation='linear',
                 output_mode='none', noise=None, name='collision', group=None,
                 modulators=None, robot_address=''):
        self.n              = n
        self.angle_spread   = np.radians(angle_spread)
        self.arc_angle      = np.radians(arc_angle)
        self.radius         = radius
        _noise_std = float(noise if noise is not None else noise_std)
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay,
                            activation=activation, scale=scale, bias=bias,
                            noise_std=_noise_std, noise_tau=noise_tau, output_mode=output_mode)
        self.name           = name
        self.group          = group
        self.modulators     = modulators or []
        self.robot_address  = robot_address
        self._x           = None
        self._values      = np.zeros(n)
        self._active_color = '#FF6633'

    def _sensor_centers(self):
        if self.n == 1:
            return np.array([0.0])
        return np.linspace(self.angle_spread / 2, -self.angle_spread / 2, self.n)

    @classmethod
    def param_defs(cls):
        return [
            ('n',             int,   4,        'number of arc sectors'),
            ('angle_spread',  float, np.radians(90.0),  'total spread in degrees'),
            ('arc_angle',     float, np.radians(45.0),  'arc width per sector in degrees'),
            ('radius',        float, 1.2,      'radius multiplier'),
            ('scale',         float, 1.0,      'output scale'),
            ('bias',          float, 0.0,      'constant offset added after scale'),
            ('noise_std',      float, 0.0,      'Gaussian noise std added on collision (zero otherwise)'),
            ('tau_rise',      float, '',       'rise τ (empty = passthrough)'),
            ('tau_decay',     float, '',       'decay τ (empty = passthrough)'),
            ('activation',    str,   'linear', 'output activation', ACTIVATIONS),
            ('output_mode',   str,   'none',   'none / derivative / integral', OUTPUT_MODES),
        ]

    def _get_radius(self, sim_cfg):
        return sim_cfg.body_radius * self.radius

    def sample(self, x, y, theta, world, sim_cfg, other_agents=None) -> np.ndarray:
        hit_arr = self._detect_hits(x, y, theta, world, sim_cfg, other_agents)
        return self._finalize_hits(hit_arr, sim_cfg)

    def _detect_hits(self, x, y, theta, world, sim_cfg, other_agents=None) -> np.ndarray:
        """Vectorized analytic 2-D geometry check: per sector, is any probe point
        along its arc within threshold of the arena boundary, an object, a wall,
        or another agent's body (other_agents, a list of {'x','y','r'} circles —
        see sim_controller._tick's per-agent snapshot)?

        Numpy-vectorized equivalent of the original nested-Python-loop version
        (same math, verified bit-identical) — see TODO.md Performance review.
        Used whenever MuJoCo isn't computing this sensor's hits instead (see
        MuJoCoEngine.sample_collision_sensors, which reads other agents from
        MuJoCo's own contact array instead of this list).
        """
        r      = self._get_radius(sim_cfg)
        limit  = sim_cfg.arena_scale
        n_pts  = max(5, int(np.degrees(self.arc_angle) / 5))
        # Threshold for polygon wall hits: probe extends body_radius*(radius-1) past the
        # robot surface, so the wall-to-probe distance when touching is that amount.
        wall_threshold = max(0.01, sim_cfg.body_radius * (self.radius - 0.9))

        centers = self._sensor_centers()
        half    = self.arc_angle / 2
        # (n_sectors, n_pts) grid of probe angles/points.
        angles  = (theta + centers)[:, None] + np.linspace(-half, half, n_pts)[None, :]
        px      = x + r * np.cos(angles)
        py      = y + r * np.sin(angles)

        if world.arena_round:
            hit = np.hypot(px, py) > (limit - wall_threshold)
        else:
            hit = (np.abs(px) > limit - wall_threshold) | (np.abs(py) > limit - wall_threshold)

        objs = world.objects + other_agents if other_agents else world.objects
        if objs:
            ox  = np.array([o['x'] for o in objs])
            oy  = np.array([o['y'] for o in objs])
            orr = np.array([o['r'] for o in objs])
            d   = np.hypot(px[..., None] - ox, py[..., None] - oy)
            hit |= np.any(d <= (orr + 0.005), axis=-1)

        walls = getattr(world, 'walls', [])
        if walls:
            # Each wall polygon's edges, flattened into one (n_edges, 2) a/b array.
            a_pts, b_pts = [], []
            for wall in walls:
                pts = wall['points']
                for i in range(len(pts)):
                    a_pts.append(pts[i])
                    b_pts.append(pts[(i + 1) % len(pts)])
            a = np.array(a_pts)   # (n_edges, 2)
            b = np.array(b_pts)
            ddx, ddy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
            len2 = ddx * ddx + ddy * ddy
            len2 = np.where(len2 < 1e-12, 1.0, len2)
            pxf, pyf = px[..., None], py[..., None]           # (n_sectors, n_pts, 1)
            t   = np.clip(((pxf - a[:, 0]) * ddx + (pyf - a[:, 1]) * ddy) / len2, 0.0, 1.0)
            cx_, cy_ = a[:, 0] + t * ddx, a[:, 1] + t * ddy
            dseg = np.hypot(pxf - cx_, pyf - cy_)
            hit |= np.any(dseg < wall_threshold, axis=-1)

        return np.any(hit, axis=-1).astype(float)   # (n_sectors,)

    def _finalize_hits(self, hit_arr: np.ndarray, sim_cfg) -> np.ndarray:
        """Shared post-processing for any raw per-sector hit array (0/1), regardless
        of whether it came from _detect_hits (analytic) or MuJoCo's contact data.

        Noise is hit-gated here (proportional to hit_arr, zero when no contact)
        and applied on the raw 0/1 signal, before _process's bias/scale — so
        noise_std is in raw-signal units, not output-scale units. Uses the
        shared _apply_noise (white or OU per noise_tau, same as every other
        sensor) on a zero baseline so the noise term itself is hit-gated
        without gating away the OU state's own continuity between contacts."""
        out = hit_arr
        if self.noise_std > 0.0:
            noise = self._apply_noise(np.zeros_like(hit_arr, dtype=float), sim_cfg.dt)
            out = out + hit_arr * noise
        self._values = self._process(out, sim_cfg)
        return self._values

    def process_robot_value(self, raw, sim_cfg) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        out = raw
        if self.noise_std > 0.0:
            noise = self._apply_noise(np.zeros_like(raw, dtype=float), sim_cfg.dt)
            out = out + raw * noise
        return np.asarray(self._process(out, sim_cfg), dtype=np.float32)


class DistanceSensor(BaseSensor):
    """
    Casts n rays outward, returning normalised distance to the nearest obstacle.
    1.0 = touching, 0.0 = at maximum range.

    Example:
        DistanceSensor(n=5, angle_spread=90, name='dist')
    """

    help_text = """\
## DistanceSensor — proximity detector

Casts `n` rays outward in a fan. Output is normalised proximity: 1.0 = touching, 0.0 = at `max_range`.

**Raw signal** (per ray `i`):

$$r_i = \\max\\!\\left(0,\\;1 - \\frac{d_i}{\\text{max\\_range}}\\right)$$

where $d_i$ is the distance to the nearest obstacle along ray $i$.

**Output pipeline** (all sensors):

$$\\tau = \\begin{cases}\\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x\\end{cases}, \\quad \\frac{dx}{dt} = \\frac{u - x}{\\tau}$$

$$\\text{output} = f(x) \\times \\text{scale} \\quad (\\text{or}\\ f(u) \\times \\text{scale}\\ \\text{if no dynamics})$$

**Order of operations** (per tick):
1. `r_i = max(0, 1 − d_i / max_range)` — raw ray reading above
2. `u = r_i + bias`
3. add noise to `u` (if `noise_std > 0`)
4. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
5. `x = leaky(u)` — asymmetric τ_rise/τ_decay integration (passthrough if no τ set)
6. `output = activation(x) × scale`

- `angle_spread` — total angular fan width in degrees.
- `max_range` — rays beyond this distance read 0.

In multi-agent sessions, rays also hit other agents' bodies (treated as circles of `body_radius`), not just walls/objects.

---

**Neuromodulation:**

- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute / derivative / integral). Multiplies the sensor output each tick by `1 + Σ (scale × signal)`. The `site` field is accepted but not used — all modulators contribute a single gain.
  - `scale > 0` → excitatory (boosts sensitivity); `scale < 0` → inhibitory.
"""

    viz_type    = 'ray'
    _viz_dashed = True
    _viz_lw     = 1.2

    def __init__(self, n=5, angle_spread=90.0, max_range=1.0, scale=1.0, bias=0.0,
                 tau_rise=None, tau_decay=None, activation='linear',
                 output_mode='none', noise_std=0.0, noise_tau=0.0, name='dist', group=None,
                 modulators=None, robot_address=''):
        self.n              = n
        self.angle_spread   = np.radians(angle_spread)
        self.center_angle   = 0.0
        self.max_range      = max_range
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay,
                            activation=activation, scale=scale, bias=bias,
                            noise_std=noise_std, noise_tau=noise_tau, output_mode=output_mode)
        self.name           = name
        self.group          = group
        self.modulators     = modulators or []
        self.robot_address = robot_address
        self._x           = None

    @classmethod
    def param_defs(cls):
        return [
            ('n',             int,   5,        'number of rays'),
            ('angle_spread',  float, np.radians(90.0), 'fan width in degrees'),
            ('max_range',     float, 1.0,      'maximum detection range'),
            ('scale',         float, 1.0,      'output scale'),
            ('bias',          float, 0.0,      'constant offset added after scale'),
            ('tau_rise',      float, '',       'rise τ (empty = passthrough)'),
            ('tau_decay',     float, '',       'decay τ (empty = passthrough)'),
            ('activation',    str,   'linear', 'output activation', ACTIVATIONS),
            ('output_mode',   str,   'none',   'none / derivative / integral', OUTPUT_MODES),
        ]

    @staticmethod
    def _round_wall_dist(x, y, heading, R):
        dx, dy = np.cos(heading), np.sin(heading)
        b      = x*dx + y*dy
        c      = x*x + y*y - R*R
        disc   = b*b - c
        if disc < 0:
            return R * 2
        t = -b + np.sqrt(disc)
        return t if t > 0 else R * 2

    @staticmethod
    def _wall_dist(x, y, heading, limit):
        dx, dy = np.cos(heading), np.sin(heading)
        dists  = []
        for wall in [-limit, limit]:
            if abs(dx) > 1e-9:
                t = (wall - x) / dx
                if t > 0 and abs(y + t * dy) <= limit:
                    dists.append(t)
            if abs(dy) > 1e-9:
                t = (wall - y) / dy
                if t > 0 and abs(x + t * dx) <= limit:
                    dists.append(t)
        return min(dists) if dists else limit * 2

    @staticmethod
    def _object_dist(x, y, heading, objects):
        rdx, rdy = np.cos(heading), np.sin(heading)
        min_d = np.inf
        for obj in objects:
            ox, oy = obj['x'] - x, obj['y'] - y
            proj   = ox * rdx + oy * rdy
            if proj <= 0:
                continue
            perp2 = ox*ox + oy*oy - proj*proj
            r2    = obj['r'] ** 2
            if perp2 >= r2:
                continue
            hit_d = proj - np.sqrt(r2 - perp2)
            if hit_d < min_d:
                min_d = max(0.0, hit_d)
        return min_d

    @staticmethod
    def _walls_ray_dist(x, y, heading, walls):
        rdx, rdy = np.cos(heading), np.sin(heading)
        min_t = np.inf
        for wall in walls:
            pts = wall['points']
            for i in range(len(pts)):
                ax, ay = pts[i]
                bx, by = pts[(i + 1) % len(pts)]
                sx, sy = bx - ax, by - ay
                denom = rdx * sy - rdy * sx
                if abs(denom) < 1e-9:
                    continue
                wx, wy = ax - x, ay - y
                t = (wx * sy - wy * sx) / denom
                s = (wx * rdy - wy * rdx) / denom
                if t > 1e-6 and 0.0 <= s <= 1.0:
                    min_t = min(min_t, t)
        return min_t

    def sample(self, x, y, theta, world, sim_cfg, other_agents=None) -> np.ndarray:
        limit = sim_cfg.arena_scale
        objs  = world.objects + other_agents if other_agents else world.objects
        vals  = []
        for a in self._ray_angles():
            sa     = theta + a
            if getattr(world, 'arena_round', False):
                d_wall = self._round_wall_dist(x, y, sa, limit)
            else:
                d_wall = self._wall_dist(x, y, sa, limit)
            d_obj   = self._object_dist(x, y, sa, objs)
            d_walls = self._walls_ray_dist(x, y, sa, getattr(world, 'walls', []))
            d       = min(d_wall, d_obj, d_walls)
            vals.append(0.0 if d >= self.max_range else 1.0 - d / self.max_range)
        return self._process(np.array(vals), sim_cfg)


class InteroceptiveSensor(BaseSensor):
    """
    Interoceptive (gut) sensor: tracks cumulative exposure to a specific gradient.

    Samples the gradient at a single point on the robot's circumference at
    heading angle zero (the "mouth"). Integrates over time with asymmetric
    time constants — rises quickly when food is present, decays slowly when
    absent. Output is a scalar clamped to [0, max_val].

    Parameters
    ----------
    gradient  : gradient label to track ('A'–'F')
    scale     : multiplier applied to the raw gradient sample before integration
    max_val   : saturation ceiling for the internal state
    tau_rise  : time constant when input exceeds current state (getting fed)
    tau_decay : time constant when input is below current state (getting hungry)

    Example:
        InteroceptiveSensor(gradient='A', tau_rise=1.0, tau_decay=30.0, name='gut')
    """

    help_text = """\
## InteroceptiveSensor — gut / internal state

Samples gradient intensity at the **mouth** (heading angle 0) and integrates it over time as an internal energy/satiation state.

**Order of operations** (one tick):

1. **Sample** — $r \\in [0,1]$: maximum gradient intensity within `body_radius` of the mouth point.
2. **Scale & clip** — $s_{\\text{target}} = \\min(r \\times \\text{scale},\\; 1) \\times \\text{max\\_val}$
   - `scale` controls how sensitive the mouth is to the gradient. At `scale=1` full-intensity gradient drives the state all the way to `max_val`. At `scale=0.5` even a full-intensity patch only drives it to `max_val / 2`.
   - `max_val` is the ceiling of the internal state — the highest value the sensor can reach.
3. apply the `output_mode` transform to $s_{\\text{target}}$ — derivative/integral (if not `none`) — **before** the leaky integration below runs, so a derivative/integral tracks the true stimulus, not this sensor's own filter lag.
4. **Integrate** — asymmetric leaky dynamics toward the (possibly transformed) target:

$$\\tau = \\begin{cases}\\tau_{rise} & s_{\\text{target}} > s \\\\ \\tau_{decay} & s_{\\text{target}} \\leq s\\end{cases}, \\qquad \\frac{ds}{dt} = \\frac{s_{\\text{target}} - s}{\\tau}$$

   - `tau_rise` — seconds to reach satiation when food is at the mouth.
   - `tau_decay` — seconds to return to zero when food is absent (should be >> `tau_rise` for realistic hunger).

5. **Clip & output** — $\\text{output} = \\text{clip}(s,\\; 0,\\; \\text{max\\_val}) + \\text{bias}$
   - `start_val` — initial value of $s$ at reset (0 = empty stomach, `max_val` = full).
   - `bias` — constant offset added to the output every tick. Use a negative bias to shift the resting output below zero (e.g. to encode a hunger drive rather than a satiation level).

**Example:** `scale=50, max_val=100` — a patch at half intensity (`r=0.5`) drives `s_target = clip(0.5×50, 0, 1) × 100 = 100` (full satiation), so `scale` here acts as a sensitivity amplifier and anything above `1/scale = 0.02` patch intensity triggers full satiation.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — if set, publishes the sensor's output to the neuromodulator bus under this name (e.g. `"satiety"`). Other layers/sensors can then receive it via their `modulators` list.
- `neuromodulator_color` — hex color (`#RRGGBB`) used to draw the transmitter signal in the network visualizer. Only meaningful when `neuromodulator_transmitter` is set.
- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute / derivative / integral). Multiplies the sensor output by `1 + Σ (scale × signal)`. The `site` field is accepted but unused (no activation function to gate).
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
"""

    viz_type = 'mouth'

    def __init__(self, gradient='A', scale=1.0, max_val=1.0, start_val=0.0,
                 bias=0.0, tau_rise=1.0, tau_decay=30.0, output_mode='none',
                 activation='linear', noise_std=0.0, noise_tau=0.0,
                 name='gut', group=None, modulators=None, robot_address=''):
        self.n              = 1
        self.gradient       = gradient
        self.max_val        = max_val
        self.start_val      = start_val
        self.bias           = bias
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay,
                            activation=activation, scale=scale, noise_std=noise_std,
                            noise_tau=noise_tau, output_mode=output_mode)
        self.name           = name
        self.group          = group
        self.modulators     = modulators or []
        self.robot_address = robot_address
        self._state    = float(start_val)
        self._viz_color = _GRADIENT_LABEL_COLOR.get(gradient, '#FFDD88')

    @classmethod
    def param_defs(cls):
        return [
            ('gradient',  str,   'A',    'gradient label to track (A–F)'),
            ('scale',     float, 1.0,    'multiplier on raw gradient sample'),
            ('max_val',   float, 1.0,    'saturation ceiling'),
            ('start_val', float, 0.0,    'initial state (0 = hungry, max_val = fed)'),
            ('bias',      float, 0.0,    'constant offset added to the output'),
            ('tau_rise',  float, 1.0,    'rise time constant (seconds)'),
            ('tau_decay',      float, 30.0, 'decay time constant (seconds)'),
            ('output_mode',    str,   'none', 'none / derivative / integral', OUTPUT_MODES),
        ]

    def reset(self):
        self._state = float(self.start_val)
        self._prev_output = None
        self._integral = None
        self._mod_row_state = {}

    def sample(self, x, y, theta, world, sim_cfg) -> np.ndarray:
        mx  = x + sim_cfg.body_radius * np.cos(theta)
        my  = y + sim_cfg.body_radius * np.sin(theta)
        # Sample gradient intensity directly at the mouth point (not a forward ray)
        raw = 0.0
        if getattr(world.cfg, 'toggle_stim', True):
            for patch in world.patches:
                if patch.get('label') != self.gradient:
                    continue
                if patch.get('type') == 'wall':
                    continue
                dist = np.hypot(mx - patch['x'], my - patch['y'])
                sig  = max(0.0, 1.0 - dist / max(patch['r'], 1e-9))
                if sig > raw:
                    raw = sig
        # scale controls sensitivity; target maps [0,1] signal → [0, max_val]
        target = min(raw * self.scale, 1.0) * self.max_val
        # output_mode transforms the raw target BEFORE the leaky hunger/satiety
        # filter below runs — not the filtered state — so derivative/integral
        # track the true stimulus, not this sensor's own filter lag.
        target = float(self._apply_output_mode(np.array([target]), sim_cfg.dt)[0])
        tau = self.tau_rise if target > self._state else self.tau_decay
        self._state += (target - self._state) / tau * sim_cfg.dt
        self._state  = float(np.clip(self._state, 0.0, self.max_val))
        return np.array([self._state + self.bias])


class ProprioceptiveSensor(BaseSensor):
    """
    Reads joint angle or angular velocity with leaky neural dynamics.

    Identified by motor_layer_name (the joint_id field), which groups all joints
    sharing the same motor layer — so a mirrored wheel pair produces n=2 outputs
    and a single head joint produces n=1. n is set automatically by the simulator
    when the circuit is assembled.

    Parameters
    ----------
    joint_id     : motor_layer_name of the joint group to read
    use_velocity : read angular velocity instead of angle
    scale        : output multiplier
    tau_rise     : rise time constant (s). Empty = passthrough.
    tau_decay    : decay time constant (s). Defaults to tau_rise.
    activation   : output nonlinearity
    """

    help_text = """\
## ProprioceptiveSensor — joint angle / velocity

Reads joint angle or angular velocity and optionally applies leaky dynamics.

**Raw signal** (per joint `i`):

$$r_i = \\text{joint}_i.\\text{angle} \\quad (\\text{or}\\ \\text{vel}\\ \\text{if use\\_velocity=True})$$

`n` is set automatically from the number of joints in the named group.

**Output pipeline** (all sensors):

$$\\tau = \\begin{cases}\\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x\\end{cases}, \\quad \\frac{dx}{dt} = \\frac{u - x}{\\tau}$$

$$\\text{output} = f(x) \\times \\text{scale} \\quad (\\text{or}\\ f(u) \\times \\text{scale}\\ \\text{if no dynamics})$$

**Order of operations** (per tick):
1. `r_i = joint_i.angle` (or `.vel` if `use_velocity=True`) — raw reading above
2. `u = r_i + bias`
3. add noise to `u` (if `noise_std > 0`)
4. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
5. `x = leaky(u)` — asymmetric τ_rise/τ_decay integration (passthrough if no τ set)
6. `output = activation(x) × scale`

- `joint_id` — `motor_layer_name` of the joint group. A mirrored wheel pair → `n=2`; a single head joint → `n=1`.
- `use_velocity = True` — reads angular velocity instead of angle.

---

**Neuromodulation:**

- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute / derivative / integral). Multiplies the sensor output each tick by `1 + Σ (scale × signal)`. The `site` field is accepted but not used — all modulators contribute a single gain.
  - `scale > 0` → excitatory (boosts sensitivity); `scale < 0` → inhibitory.
"""

    _viz_color = '#AADDFF'

    def __init__(self, joint_id='root', use_velocity=False, scale=1.0, bias=0.0,
                 tau_rise=None, tau_decay=None, activation='linear', output_mode='none',
                 noise_std=0.0, noise_tau=0.0, name='proprio', group=None, modulators=None,
                 robot_address=''):
        self.n              = 1
        self.joint_id       = joint_id
        self.use_velocity   = use_velocity
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay,
                            activation=activation, scale=scale, bias=bias,
                            noise_std=noise_std, noise_tau=noise_tau, output_mode=output_mode)
        self.name           = name
        self.group          = group
        self.modulators     = modulators or []
        self.robot_address = robot_address
        self._x             = None
        self._joint_refs    = []   # set by resolve_joint_sensor_refs: joints sorted by motor_output_idx
        self._layer_ref     = None  # fallback when joint_id names a layer directly (no physical joints)

    @classmethod
    def param_defs(cls):
        return [
            ('joint_id',       str,   'root',   'motor_layer_name of the joint group to read'),
            ('use_velocity',   bool,  False,    'read angular velocity instead of angle'),
            ('scale',          float, 1.0,      'output multiplier'),
            ('bias',           float, 0.0,      'constant offset added after scale'),
            ('tau_rise',       float, '',       'rise τ (empty = passthrough)'),
            ('tau_decay',      float, '',       'decay τ (empty = passthrough)'),
            ('activation',     str,   'linear', 'output activation', ACTIVATIONS),
            ('output_mode',    str,   'none',   'none / derivative / integral', OUTPUT_MODES),
        ]

    def reset(self):
        self._x = None
        self._prev_output = None
        self._integral = None
        self._noise_buf = None
        self._mod_row_state = {}

    def sample(self, x, y, theta, world, sim_cfg) -> np.ndarray:
        if self._joint_refs:
            raw = np.array([
                (jt.vel if self.use_velocity else jt.angle)
                for jt in self._joint_refs
            ])
        elif self._layer_ref is not None:
            out = self._layer_ref.output
            if out is None:
                return np.zeros(max(1, self.n))
            raw = np.atleast_1d(np.asarray(out, dtype=float))
        else:
            return np.zeros(max(1, self.n))
        return self._process(raw, sim_cfg)



class WhiskerSensor(BaseSensor):
    """
    Casts a ray along the body's heading. Output is proportional to bending:
    0 = no contact, 1 = contact at the base.
    Caches _contact_dist (None or float) for the arena renderer.
    """

    help_text = """\
## WhiskerSensor — tactile whisker

Casts a ray from a mount point along the body heading. Output is bending proportion: 0 = no contact, 1 = contact at the base.

**Mount pose:**

$$\\text{origin} = \\bigl(x + d_m\\cos(\\theta + \\alpha_m),\\; y + d_m\\sin(\\theta + \\alpha_m)\\bigr)$$

where $d_m$ = `mount_dist`, $\\alpha_m$ = `mount_angle`.

**Raw signal:**

$$r = \\text{clip}\\!\\left(\\frac{\\text{length} - d}{\\text{length}},\\; 0,\\; 1\\right)$$

where $d$ = distance to first obstacle. If no contact: $r = 0$.

**Output pipeline** (all sensors):

$$\\tau = \\begin{cases}\\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x\\end{cases}, \\quad \\frac{dx}{dt} = \\frac{u - x}{\\tau}$$

$$\\text{output} = f(x) \\times \\text{scale} \\quad (\\text{or}\\ f(u) \\times \\text{scale}\\ \\text{if no dynamics})$$

**Order of operations** (per tick):
1. `r = clip((length − d) / length, 0, 1)` if contact, else `0` — raw reading above
2. `u = r + bias`
3. add noise to `u` (if `noise_std > 0`)
4. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
5. `x = leaky(u)` — asymmetric τ_rise/τ_decay integration (passthrough if no τ set)
6. `output = activation(x) × scale`
"""

    _viz_color = '#DDCC55'
    viz_type   = 'whisker'

    def __init__(self, length=0.15, mount_dist=0.0, mount_angle=0.0,
                 n=1, scale=1.0, bias=0.0, tau_rise=None, tau_decay=None, activation='linear',
                 output_mode='none', noise_std=0.0, noise_tau=0.0, name='whisker', group=None,
                 modulators=None, robot_address=''):
        self.length         = length
        self.mount_dist     = mount_dist
        self.mount_angle    = mount_angle
        self.n              = n
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay,
                            activation=activation, scale=scale, bias=bias,
                            noise_std=noise_std, noise_tau=noise_tau, output_mode=output_mode)
        self.name           = name
        self.group          = group
        self.modulators     = modulators or []
        self.robot_address = robot_address
        self._contact_dist  = None
        self._x             = None

    @classmethod
    def param_defs(cls):
        return [
            ('length',       float, 0.15,     'whisker length in world units'),
            ('mount_dist',   float, 0.0,      'distance from body centre to whisker base'),
            ('mount_angle',  float, 0.0,      'mount angle offset from body heading (radians)'),
            ('n',            int,   1,         'number of neurons'),
            ('scale',        float, 1.0,       'output scale'),
            ('bias',         float, 0.0,       'constant offset added after scale'),
            ('tau_rise',     float, '',        'rise τ (empty = passthrough)'),
            ('tau_decay',    float, '',        'decay τ (empty = passthrough)'),
            ('activation',   str,   'linear',  'output activation', ACTIVATIONS),
            ('output_mode',  str,   'none',    'none / derivative / integral', OUTPUT_MODES),
        ]

    def _mount_pose(self, x, y, theta):
        """Return (ox, oy, ray_theta) for the whisker base in world frame."""
        ray_th = theta + self.mount_angle
        ox = x + self.mount_dist * np.cos(ray_th)
        oy = y + self.mount_dist * np.sin(ray_th)
        return ox, oy, ray_th

    def reset(self):
        self._contact_dist = None
        self._x = None
        self._prev_output = None
        self._integral = None
        self._noise_buf = None
        self._mod_row_state = {}

    def sample(self, x, y, theta, world, sim_cfg) -> np.ndarray:
        ox, oy, ray_th = self._mount_pose(x, y, theta)
        d = self._ray_cast(ox, oy, ray_th, world, sim_cfg)
        if d is None:
            self._contact_dist = None
            raw = np.zeros(self.n or 1)
        else:
            self._contact_dist = d
            signal = (self.length - d) / self.length   # 0 at tip, 1 at base
            raw = np.full(self.n or 1, float(np.clip(signal, 0.0, 1.0)))
        return self._process(raw, sim_cfg)

    def _ray_cast(self, x, y, theta, world, sim_cfg):
        """Return distance to first intersection within whisker length, or None."""
        cx, cy  = np.cos(theta), np.sin(theta)
        min_t   = self.length

        # World objects (circles)
        for obj in world.objects:
            dx, dy  = obj['x'] - x, obj['y'] - y
            t_mid   = dx * cx + dy * cy
            if t_mid <= 0:
                continue
            perp_sq = dx**2 + dy**2 - t_mid**2
            r       = obj['r']
            if perp_sq >= r**2:
                continue
            t_hit = t_mid - np.sqrt(r**2 - perp_sq)
            if 0 < t_hit < min_t:
                min_t = t_hit

        # Arena boundary
        limit = sim_cfg.arena_scale
        if world.arena_round:
            dx, dy  = -x, -y
            t_mid   = dx * cx + dy * cy
            perp_sq = x**2 + y**2 - t_mid**2
            disc_sq = limit**2 - perp_sq
            if disc_sq >= 0:
                t_exit = t_mid + np.sqrt(disc_sq)
                if 0 < t_exit < min_t:
                    min_t = t_exit
        else:
            for wall_x in (limit, -limit):
                if abs(cx) > 1e-9:
                    t = (wall_x - x) / cx
                    if 0 < t < min_t and abs(y + t * cy) <= limit + 1e-6:
                        min_t = t
            for wall_y in (limit, -limit):
                if abs(cy) > 1e-9:
                    t = (wall_y - y) / cy
                    if 0 < t < min_t and abs(x + t * cx) <= limit + 1e-6:
                        min_t = t

        return min_t if min_t < self.length else None


class SkyCompassSensor(BaseSensor):
    """
    Dorsal rim area (DRA) sky polarization compass sensor.

    Outputs an n-neuron cosine bump encoding the robot's heading relative to
    the sun direction. Always active — world.sky["enabled"] only controls
    whether the polarization field is drawn in the arena, not sensing.

    sun_dir = world.sky["angle"] + π/2   (perpendicular to the e-vector bars)
    output[k] = activation( cos(theta - sun_dir - k * 2π/n) * scale )
    """

    help_text = """\
## SkyCompassSensor — polarisation sky compass (DRA)

Encodes the robot's heading relative to the sun via a cosine tuning curve, mimicking insect dorsal rim area (DRA) photoreceptors.

Always senses, whether or not the polarization field is shown in the arena — the "Sky: Polarization field" checkbox and drag control only set/display the sun direction, they don't gate sensing.

**Tuning curve** (per neuron `k`):

$$r_k = \\cos\\!\\left(\\theta - \\phi_{\\text{sun}} - \\phi_0 - k \\cdot \\frac{2\\pi}{n}\\right)$$

where $\\phi_{\\text{sun}} = \\text{sky.angle} + \\pi/2$ (perpendicular to e-vector bars) and $\\phi_0$ = `phase`.

**Output pipeline** (all sensors):

$$\\tau = \\begin{cases}\\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x\\end{cases}, \\quad \\frac{dx}{dt} = \\frac{u - x}{\\tau}$$

$$\\text{output} = f(x) \\times \\text{scale} \\quad (\\text{or}\\ f(u) \\times \\text{scale}\\ \\text{if no dynamics})$$

**Order of operations** (per tick):
1. `r_k = cos(θ − φ_sun − φ₀ − k·2π/n)` — raw tuning-curve reading above
2. `u = r_k + bias`
3. add noise to `u` (if `noise_std > 0`)
4. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
5. `x = leaky(u)` — asymmetric τ_rise/τ_decay integration (passthrough if no τ set)
6. `output = activation(x) × scale`

- `n` — number of DRA neurons (heading directions sampled).
- `phase` — rotates neuron 0 to align with a reference direction.
- `output_mode = 'derivative'` — reports rate of change (heading change detection); `'integral'` — running accumulation.
- `noise_std / noise_tau` — additive Ornstein–Uhlenbeck noise on raw signal.

---

**Neuromodulation:**

- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute / derivative / integral). Multiplies the sensor output each tick by `1 + Σ (scale × signal)`. The `site` field is accepted but not used — all modulators contribute a single gain.
  - `scale > 0` → excitatory (boosts sensitivity); `scale < 0` → inhibitory.
"""

    _viz_color = '#FFDD88'
    viz_type   = 'sky'

    def __init__(self, n=8, scale=1.0, bias=0.0, phase=0.0,
                 tau_rise=None, tau_decay=None,
                 activation='relu', noise_std=0.0,
                 noise_tau=0.0, output_mode='none',
                 name='sky', group=None, modulators=None, robot_address=''):
        self.n              = n
        self.phase          = phase
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay,
                            activation=activation, scale=scale, bias=bias,
                            noise_std=noise_std, noise_tau=noise_tau,
                            output_mode=output_mode)
        self.name           = name
        self.group          = group
        self.modulators     = modulators or []
        self.robot_address = robot_address
        self._x             = None
        self._last_output   = np.zeros(n)

    @classmethod
    def param_defs(cls):
        return [
            ('n',              int,   8,        'number of DRA neurons'),
            ('scale',          float, 1.0,      'output multiplier'),
            ('bias',           float, 0.0,      'constant offset added after scale'),
            ('phase',          float, 0.0,      'phase offset (rad) — aligns neuron 0 to field direction'),
            ('tau_rise',       float, '',       'rise τ (empty = passthrough)'),
            ('tau_decay',      float, '',       'decay τ (empty = passthrough)'),
            ('activation',     str,   'relu',   'output activation', ACTIVATIONS),
            ('noise_std',      float, 0.0,      'noise amplitude'),
            ('noise_tau',      float, 0.0,      'OU correlation time (0 = white noise)'),
            ('output_mode',    str,   'none',   'none / derivative / integral', OUTPUT_MODES),
        ]

    def reset(self):
        self._x           = None
        self._noise_buf   = None
        self._last_output = np.zeros(self.n)
        self._prev_output = None
        self._integral = None
        self._mod_row_state = {}

    def sample(self, x, y, theta, world, sim_cfg) -> np.ndarray:
        sun_dir = world.get_sky_sun_dir()
        k   = np.arange(self.n)
        raw = np.cos(theta - sun_dir - self.phase - k * 2 * np.pi / self.n)
        result = self._process(raw, sim_cfg)
        self._last_output = result
        if os.environ.get('LBP_DEBUG_COMPASS'):
            self._debug_tick = getattr(self, '_debug_tick', 0) + 1
            if self._debug_tick % 25 == 0:
                print(f"[DEBUG-COMPASS] {self.name}: theta={theta:.3f} sun_dir={sun_dir:.3f} "
                      f"raw={np.round(raw, 3)} result={np.round(result, 3)}")
        return result


class ManualBumpSensor(BaseSensor):
    """
    Fake sensor for testing ring-attractor-style circuits: outputs an n-neuron
    cosine bump whose position is driven entirely by keyboard input (Up/Down
    arrow keys step the bump by one neuron), not by robot pose or world state.

    Example:
        ManualBumpSensor(n=8, name='bump')
    """

    help_text = """\
## ManualBumpSensor — keyboard-steered bump

Encodes a manually-controlled cosine bump over `n` neurons, for testing circuits (e.g. `RingAttractorLayer`) without needing a real heading- or world-driven input.

Pressing the **Up** arrow key advances the bump by one neuron step (`+2π/n`); **Down** retreats it by one step. This is global to the 2D simulator window — it works regardless of Manual-drive mode or which motor command source is active.

**Tuning curve** (per neuron `k`):

$$r_k = \\cos\\!\\left(k \\cdot \\frac{2\\pi}{n} - \\phi\\right)$$

where $\\phi$ is the internal phase, stepped by `±2π/n` per key press (starts at 0, wraps at `2π`).

**Output pipeline** (all sensors):

$$\\tau = \\begin{cases}\\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x\\end{cases}, \\quad \\frac{dx}{dt} = \\frac{u - x}{\\tau}$$

$$\\text{output} = f(x) \\times \\text{scale} \\quad (\\text{or}\\ f(u) \\times \\text{scale}\\ \\text{if no dynamics})$$

**Order of operations** (per tick):
1. `r_k = cos(k·2π/n − φ)` — raw tuning-curve reading above
2. `u = r_k + bias`
3. add noise to `u` (if `noise_std > 0`)
4. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
5. `x = leaky(u)` — asymmetric τ_rise/τ_decay integration (passthrough if no τ set)
6. `output = activation(x) × scale`

- `n` — number of neurons around the bump.
- `noise_std` / `noise_tau` — additive Ornstein-Uhlenbeck noise on the raw signal.

---

**Neuromodulation:**

- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute / derivative / integral). Multiplies the sensor output each tick by `1 + Σ (scale × signal)`. The `site` field is accepted but not used — all modulators contribute a single gain.
  - `scale > 0` → excitatory (boosts sensitivity); `scale < 0` → inhibitory.
"""

    _viz_color = '#C8A0FF'
    viz_type   = None

    def __init__(self, n=8, scale=1.0, bias=0.0, tau_rise=None, tau_decay=None,
                 activation='relu', noise_std=0.0, noise_tau=0.0,
                 output_mode='none', name='bump', group=None,
                 modulators=None, robot_address=''):
        self.n              = n
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay,
                            activation=activation, scale=scale, bias=bias,
                            noise_std=noise_std, noise_tau=noise_tau,
                            output_mode=output_mode)
        self.name           = name
        self.group          = group
        self.modulators     = modulators or []
        self.robot_address = robot_address
        self._x             = None
        self._phase         = 0.0

    @classmethod
    def param_defs(cls):
        return [
            ('n',              int,   8,        'number of neurons around the bump'),
            ('tau_rise',       float, '',       'rise τ (empty = passthrough)'),
            ('tau_decay',      float, '',       'decay τ (empty = passthrough)'),
            ('activation',     str,   'relu',   'output activation', ACTIVATIONS),
            ('scale',          float, 1.0,      'output multiplier'),
            ('bias',           float, 0.0,      'constant offset added after scale'),
            ('noise_std',      float, 0.0,      'noise amplitude'),
            ('noise_tau',      float, 0.0,      'OU correlation time (0 = white noise)'),
            ('output_mode',    str,   'none',   'none / derivative / integral', OUTPUT_MODES),
        ]

    def reset(self):
        self._x     = None
        self._phase = 0.0
        self._prev_output = None
        self._integral = None
        self._noise_buf = None
        self._mod_row_state = {}

    def step(self, direction):
        """Advance (direction>0) or retreat (direction<0) the bump by one neuron step."""
        self._phase = (self._phase + direction * 2 * np.pi / self.n) % (2 * np.pi)

    def sample(self, x, y, theta, world, sim_cfg) -> np.ndarray:
        k   = np.arange(self.n)
        raw = np.cos(k * 2 * np.pi / self.n - self._phase)
        return self._process(raw, sim_cfg)


class ManualCueSensor(BaseSensor):
    """
    Fake sensor for testing reward/context-driven circuits: outputs 1.0 while
    a configured keyboard key is held down, 0.0 otherwise. A generic stand-in
    for any binary event a real sensor would eventually detect (reward
    arrival, odor context, ...) — not driven by robot pose or world state.

    Example:
        ManualCueSensor(key='R', name='reward')
    """

    help_text = """\
## ManualCueSensor — keyboard-held binary cue

Outputs 1.0 while a configured key is held down, 0.0 otherwise — a stand-in
for any binary event sensor (reward arrival, odor context, ...) you don't
have a real detector for yet. This is global to the 2D simulator window — it
works regardless of Manual-drive mode or which motor command source is
active, and regardless of which top-level window has keyboard focus.

**Raw signal:**

$$r = \\begin{cases}1 & \\text{key held} \\\\ 0 & \\text{otherwise}\\end{cases}$$

Run this through `tau_decay` to shape a quick tap into a brief phasic pulse
(e.g. a dopamine-like reward signal), or hold the key down for a sustained
context cue — same sensor, different use depending on how long you hold the
key and how the dynamics are configured.

**Output pipeline** (all sensors):

$$\\tau = \\begin{cases}\\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x\\end{cases}, \\quad \\frac{dx}{dt} = \\frac{u - x}{\\tau}$$

$$\\text{output} = f(x) \\times \\text{scale} \\quad (\\text{or}\\ f(u) \\times \\text{scale}\\ \\text{if no dynamics})$$

**Order of operations** (per tick):
1. `r = 1` if the key is held, else `0` — raw reading above
2. `u = r + bias`
3. add noise to `u` (if `noise_std > 0`)
4. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
5. `x = leaky(u)` — asymmetric τ_rise/τ_decay integration (passthrough if no τ set)
6. `output = activation(x) × scale`

- `key` — single keyboard character that drives this sensor (case-insensitive).
  Avoid keys already bound elsewhere: WASD/Space (manual drive), Up/Down
  (`ManualBumpSensor`), and world-edit letters (gradients A–F, objects
  Z/Y/X/W/V/U).
- `noise_std` / `noise_tau` — additive Ornstein-Uhlenbeck noise on the raw signal.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — if set, publishes this sensor's output to the
  neuromodulator bus under this name (e.g. `"dopamine"`) — the natural way to
  wire a faked reward key straight into a `ThreeFactorLayer`'s reward gate.
- `neuromodulator_color` — hex color for this transmitter in the visualizer.
- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute /
  derivative / integral). Multiplies the sensor output each tick by
  `1 + Σ (scale × signal)`. The `site` field is accepted but not used.
  - `scale > 0` → excitatory (boosts sensitivity); `scale < 0` → inhibitory.
"""

    _viz_color = '#FFD866'
    viz_type   = None

    def __init__(self, key='R', scale=1.0, bias=0.0, tau_rise=None, tau_decay=None,
                 activation='linear', noise_std=0.0, noise_tau=0.0,
                 output_mode='none', name='cue', group=None,
                 modulators=None, robot_address=''):
        self.n              = 1
        self.key            = (key or 'R')[:1].upper()
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay,
                            activation=activation, scale=scale, bias=bias,
                            noise_std=noise_std, noise_tau=noise_tau,
                            output_mode=output_mode)
        self.name           = name
        self.group          = group
        self.modulators     = modulators or []
        self.robot_address = robot_address
        self._x             = None
        self._pressed       = False

    @classmethod
    def param_defs(cls):
        return [
            ('key',            str,   'R',      'keyboard key that drives this sensor (held = 1)'),
            ('tau_rise',       float, '',       'rise τ (empty = passthrough)'),
            ('tau_decay',      float, '',       'decay τ (empty = passthrough)'),
            ('activation',     str,   'linear', 'output activation', ACTIVATIONS),
            ('scale',          float, 1.0,      'output multiplier'),
            ('bias',           float, 0.0,      'constant offset added after scale'),
            ('noise_std',      float, 0.0,      'noise amplitude'),
            ('noise_tau',      float, 0.0,      'OU correlation time (0 = white noise)'),
            ('output_mode',    str,   'none',   'none / derivative / integral', OUTPUT_MODES),
        ]

    def reset(self):
        self._x     = None
        self._prev_output = None
        self._integral = None
        self._noise_buf = None
        self._mod_row_state = {}

    def set_pressed(self, pressed):
        """Called externally by the app-wide cue-key filter when this sensor's bound key changes state."""
        self._pressed = bool(pressed)

    def sample(self, x, y, theta, world, sim_cfg) -> np.ndarray:
        raw = np.array([1.0 if self._pressed else 0.0])
        return self._process(raw, sim_cfg)


class CameraSensor(BaseSensor):
    """
    Base class for simulated cameras. Casts `width` rays in a configurable FOV
    and returns the colour of the nearest hit per ray.

    Subclasses fix the output format:
        GrayCameraSensor  — luminance, in_ch=1
        RGBCameraSensor   — colour (CHW), in_ch=3

    The full 2-D frame is cached in `_last_frame` as (H, W) [gray] or
    (H, W, 3) [RGB] for the network visualiser to read.
    """
    viz_type      = 'camera_fov'
    in_ch         = 1       # overridden by subclasses; used by FilterStackDialog
    mode          = 'gray'  # overridden by subclasses; kept for backward-compat checks
    is_image_node = True

    def n_per_side(self):
        return 1

    def thumbnail_frames(self, disp_h=32):
        frame = getattr(self, '_last_frame', None)
        if frame is None:
            return
        if getattr(self, 'in_ch', 1) == 3:
            data = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        else:
            g    = frame if frame.ndim == 2 else np.mean(frame, axis=-1)
            g    = (np.clip(g, 0, 1) * 255).astype(np.uint8)
            data = np.stack([g, g, g], axis=-1)
        reps = max(1, disp_h // data.shape[0])
        data = np.repeat(data, reps, axis=0)[:disp_h]
        if getattr(self, 'lateralized', False):
            half    = self.width // 2
            ovl     = getattr(self, 'overlap', 0)
            l_end   = int(np.clip(half + ovl, 0, self.width))
            r_start = int(np.clip(half - ovl, 0, self.width))
            yield f'{self.name}_L', data[:, :l_end, :]
            yield f'{self.name}_R', data[:, r_start:, :]
        else:
            yield self.name, data

    def __init__(self, width=64, height=48, fov=90.0, center_angle=0.0,
                 vertical_angle=0.0, max_range=10.0, lateralized=False, overlap=0,
                 output_mode='none', noise_std=0.0, noise_tau=0.0, tau_rise=None, tau_decay=None,
                 name='camera', group=None, body_id='root', robot_address='',
                 **_ignored):
        self.width          = width
        self.height         = max(1, height)
        self.fov            = np.radians(fov)
        self.center_angle   = np.radians(center_angle)
        self.vertical_angle = np.radians(vertical_angle)
        self.max_range      = max_range
        self.lateralized    = lateralized
        self.overlap        = overlap
        self._init_dynamics(tau_rise=tau_rise, tau_decay=tau_decay,
                            activation='linear', noise_std=noise_std, noise_tau=noise_tau,
                            output_mode=output_mode)
        self.name           = name
        self.group          = group
        self.body_ids       = [body_id]
        self.robot_address = robot_address
        self.n              = 1
        self._last_frame    = None
        self._left_output   = None
        self._right_output  = None
        self._x             = None
        self._viz_color     = None

    @classmethod
    def param_defs(cls):
        return [
            ('width',          int,   64,    'number of rays (horizontal resolution)'),
            ('height',         int,   48,    'tile rows (1 = single strip)'),
            ('fov',            float, np.radians(90.0), 'field of view in degrees'),
            ('center_angle',   float, 0.0,   'center offset from heading (degrees)'),
            ('vertical_angle', float, 0.0,   'vertical tilt in degrees (positive = tilt down)'),
            ('max_range',      float, 10.0,  'max ray length (world units)'),
            ('lateralized',    bool,  False, 'split output into left/right halves ({name}_L, {name}_R)'),
            ('overlap',        int,   0,     'pixels past midline included in each half (negative = gap)'),
            ('noise_std',      float, 0.0,   'Gaussian noise σ added to pixel values each tick (0 = off)'),
            ('tau_rise',       float, '',    'rise τ in seconds per pixel (empty = passthrough)'),
            ('tau_decay',      float, '',    'decay τ in seconds per pixel (empty = passthrough)'),
            ('output_mode',    str,   'none', 'none / derivative / integral', OUTPUT_MODES),
        ]

    # ------------------------------------------------------------------
    # Vectorised raycasting — all width rays cast in parallel via numpy.
    # ------------------------------------------------------------------

    @staticmethod
    def _cast_objects_v(x, y, angles, objects):
        """(width,) distances and (width,3) colors for nearest circular object."""
        n   = len(angles)
        rdx = np.cos(angles)
        rdy = np.sin(angles)
        min_d   = np.full(n, np.inf, dtype=np.float32)
        min_col = np.zeros((n, 3), dtype=np.float32)
        for obj in objects:
            ox    = obj['x'] - x
            oy    = obj['y'] - y
            proj  = ox * rdx + oy * rdy
            perp2 = ox*ox + oy*oy - proj*proj
            r2    = obj['r'] ** 2
            valid = (proj > 0) & (perp2 < r2)
            hit_d = np.where(valid, proj - np.sqrt(np.maximum(0.0, r2 - perp2)), np.inf)
            closer = valid & (hit_d >= 0.0) & (hit_d < min_d)
            min_d   = np.where(closer, hit_d, min_d)
            col     = np.array(obj.get('color', [1.0, 0.0, 0.0])[:3], dtype=np.float32)
            min_col = np.where(closer[:, np.newaxis], col, min_col)
        return min_d, min_col

    @staticmethod
    def _cast_walls_v(x, y, angles, walls):
        """(width,) distances and (width,3) colors for nearest polygon wall."""
        n   = len(angles)
        rdx = np.cos(angles)
        rdy = np.sin(angles)
        min_t   = np.full(n, np.inf, dtype=np.float32)
        min_col = np.zeros((n, 3), dtype=np.float32)
        for wall in walls:
            pts = wall['points']
            col = np.array(wall.get('color', [0.5, 0.5, 0.5])[:3], dtype=np.float32)
            for i in range(len(pts)):
                ax, ay = pts[i]
                bx, by = pts[(i + 1) % len(pts)]
                sx_ = bx - ax
                sy_ = by - ay
                wx  = ax - x
                wy  = ay - y
                denom = rdx * sy_ - rdy * sx_
                safe  = np.abs(denom) > 1e-9
                inv   = np.where(safe, 1.0 / np.where(safe, denom, 1.0), 0.0)
                t = (wx * sy_ - wy * sx_) * inv
                s = (wx * rdy - wy * rdx) * inv
                hit = safe & (t > 1e-6) & (s >= 0.0) & (s <= 1.0) & (t < min_t)
                min_t   = np.where(hit, t, min_t)
                min_col = np.where(hit[:, np.newaxis], col, min_col)
        return min_t, min_col

    @staticmethod
    def _cast_arena_sq_v(x, y, angles, limit):
        """(width,) distances to the square-arena boundary."""
        dx    = np.cos(angles)
        dy    = np.sin(angles)
        dists = np.full(len(angles), limit * 2, dtype=np.float32)
        for w in (-limit, limit):
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(np.abs(dx) > 1e-9, (w - x) / dx, np.inf)
            valid = (t > 0) & (np.abs(y + t * dy) <= limit)
            dists = np.where(valid & (t < dists), t, dists)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(np.abs(dy) > 1e-9, (w - y) / dy, np.inf)
            valid = (t > 0) & (np.abs(x + t * dx) <= limit)
            dists = np.where(valid & (t < dists), t, dists)
        return dists

    @staticmethod
    def _cast_arena_round_v(x, y, angles, R):
        """(width,) distances to the circular-arena boundary."""
        dx   = np.cos(angles)
        dy   = np.sin(angles)
        b    = x * dx + y * dy
        c    = x*x + y*y - R*R
        disc = b*b - c
        t    = np.where(disc >= 0, -b + np.sqrt(np.maximum(0.0, disc)), R * 2)
        return np.where(t > 0, t, np.float32(R * 2))

    # ------------------------------------------------------------------

    def _raycast(self, x, y, theta, world, sim_cfg):
        """Run all rays; return ((width,3) RGB pixels, (width,) distances)."""
        limit  = sim_cfg.arena_scale
        angles = np.linspace(
            theta + self.center_angle + self.fov / 2,
            theta + self.center_angle - self.fov / 2,
            self.width,
        )
        d_obj,  c_obj  = self._cast_objects_v(x, y, angles, world.objects)
        d_wall, c_wall = self._cast_walls_v(x, y, angles, getattr(world, 'walls', []))
        if getattr(world, 'arena_round', False):
            d_arena = self._cast_arena_round_v(x, y, angles, limit)
        else:
            d_arena = self._cast_arena_sq_v(x, y, angles, limit)

        best_d   = np.minimum(np.minimum(d_obj, d_wall), d_arena)
        in_range = best_d < self.max_range
        use_obj   = in_range & (d_obj <= d_wall) & (d_obj <= d_arena)
        use_wall  = in_range & ~use_obj & (d_wall <= d_arena)
        use_arena = in_range & ~use_obj & ~use_wall

        pixels = np.zeros((self.width, 3), dtype=np.float32)
        pixels = np.where(use_obj[:, np.newaxis],   np.clip(c_obj,  0.0, 1.0), pixels)
        pixels = np.where(use_wall[:, np.newaxis],  np.clip(c_wall, 0.0, 1.0), pixels)
        pixels = np.where(use_arena[:, np.newaxis], 1.0,                       pixels)
        return pixels, best_d

    def _build_frame(self, pixels, best_d):
        """Tile to (H, W, 3), clipping each row's range by vertical_angle.

        Effective max range per row = max_range * cos(row_vert_angle):
          - top row    → less steep → farther (larger range)
          - bottom row → more steep → closer  (smaller range)
        At vertical_angle=0 all rows are identical (no perspective effect).
        """
        vert = self.vertical_angle
        if vert <= 0.0:
            return np.tile(pixels[np.newaxis, :, :], (self.height, 1, 1))
        # Per-row vertical angle: assume square pixels so vertical FOV ∝ aspect ratio.
        aspect   = self.height / max(self.width, 1)
        vfov     = self.fov * aspect                     # vertical FOV (radians)
        row_t    = np.linspace(-0.5, 0.5, self.height)  # top=-0.5 (far), bottom=+0.5 (near)
        row_vert = vert + row_t * vfov                   # per-row tilt below horizontal
        # Rows above horizontal → full range; rows below → cos-scaled range
        row_max  = np.where(
            row_vert <= 0,
            self.max_range,
            self.max_range * np.cos(np.clip(row_vert, 0.0, np.pi / 2))
        )  # (H,)
        # Mask out pixels beyond each row's effective range
        out_mask = best_d[np.newaxis, :] >= row_max[:, np.newaxis]  # (H, W)
        frame    = np.tile(pixels[np.newaxis, :, :], (self.height, 1, 1)).astype(np.float32)
        frame[out_mask] = 0.0
        return frame

    def sample(self, x, y, theta, world, sim_cfg) -> np.ndarray:
        raise NotImplementedError


class GrayCameraSensor(CameraSensor):
    """Camera that outputs luminance (1 channel). in_ch=1."""

    help_text = """\
## GrayCameraSensor — grayscale camera

Raycasts `width × height` pixels across the field of view and returns luminance.

**Per-pixel luminance** (for ray `i`):

$$\\text{pixel}_i = \\frac{R_i + G_i + B_i}{3}$$

**Output shape:**

**Lateralized** (`lateralized=True`): the frame is split at the horizontal midline (with `overlap` pixels), and each half carries the **full** captured image — use this to feed a `Conv2dLayer` pair with real 2-D spatial structure:

$$\\text{sensor\\_L} \\in \\mathbb{R}^{H \\times (W/2 + \\text{overlap})}, \\quad \\text{sensor\\_R} \\in \\mathbb{R}^{H \\times (W/2 + \\text{overlap})}$$

**Not lateralized** (`lateralized=False`, default): the plain `output` read by other layers is only the frame's **centre row** — a `(W,)` vector, *not* the full `(H, W)` image. The complete `(H, W)` frame is still computed every tick and cached in `_last_frame` for the visualizer thumbnail, but only its centre row reaches the network — set `lateralized=True` if a downstream layer needs the full image.

Each half connects to its own `Conv2dLayer` (`_L` / `_R` pair).

**Order of operations** (per tick):
1. raycast `width` rays across the FOV; each ray returns the colour of the nearest hit (object / wall / arena boundary), or black if nothing within `max_range`
2. tile the ray colours into `height` rows (`vertical_angle` masks rows beyond a row-dependent, cosine-scaled range to black)
3. `pixel = (R + G + B) / 3` per pixel → grayscale `(H, W)` frame `r`, cached as `_last_frame`
4. if `lateralized`: split `r` at the midline (± `overlap`) into `sensor_L`/`sensor_R` — these are the **raw pixel values**, bypassing steps 5-7 below entirely (no noise or dynamics applied)
5. otherwise: take `r`'s **centre row only** (shape `(W,)`) as `u`
6. add noise to `u` (if `noise_std > 0`)
7. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
8. per-pixel `output = leaky(u)` (if `tau_rise`/`tau_decay` set; else `output = u`) — this sensor's `bias`/`scale`/`activation` aren't exposed as parameters and are fixed at `0` / `1` / `linear`, so they have no effect

- `vertical_angle` — camera tilt in degrees. Positive = tilted down toward ground, negative = tilted up. Each image row sees a different ground distance: bottom rows see closer, top rows see farther. At 90° the camera looks straight down and the image goes black.
"""

    in_ch = 1
    mode  = 'gray'

    def sample(self, x, y, theta, world, sim_cfg) -> np.ndarray:
        pixels, best_d = self._raycast(x, y, theta, world, sim_cfg)
        rgb_frame      = self._build_frame(pixels, best_d)         # (H, W, 3)
        frame          = np.mean(rgb_frame, axis=-1)               # (H, W) luminance
        self._last_frame = frame
        if self.lateralized:
            mid     = self.width // 2
            l_end   = int(np.clip(mid + self.overlap, 0, self.width))
            r_start = int(np.clip(mid - self.overlap, 0, self.width))
            self._left_output  = frame[:, :l_end  ].reshape(-1).astype(np.float32)
            self._right_output = frame[:, r_start:].reshape(-1).astype(np.float32)
        out = frame[self.height // 2].astype(np.float32)           # (W,) centre row
        return self._process(out, sim_cfg)


class RGBCameraSensor(CameraSensor):
    """Camera that outputs colour in CHW planar format (3 channels). in_ch=3."""

    help_text = """\
## RGBCameraSensor — colour camera

Raycasts `width × height` pixels across the field of view and returns RGB colour (same raycasting as `GrayCameraSensor`, all 3 channels retained).

**Output shape (channels-first / CHW):**

$$\\text{output} \\in \\mathbb{R}^{3 \\times H \\times W} \\quad \\text{(flat: channel, row, col)}$$

**Lateralized** (`lateralized=True`):

$$\\text{sensor\\_L} \\in \\mathbb{R}^{3 \\times H \\times (W/2 + \\text{overlap})}, \\quad \\text{sensor\\_R} \\in \\mathbb{R}^{3 \\times H \\times (W/2 + \\text{overlap})}$$

Connect to a `Conv2dLayer` with `in_ch=3` (set automatically from camera mode).

Unlike `GrayCameraSensor`, the plain (`lateralized=False`) `output` here is already the **full** frame — see step 4 below — not just a centre row.

**Order of operations** (per tick):
1. raycast `width` rays across the FOV (same geometry as `GrayCameraSensor`); each ray returns the RGB colour of the nearest hit, or black if nothing within `max_range`
2. tile into `height` rows (`vertical_angle` masks far rows to black), giving frame `r` (H × W × 3), cached as `_last_frame`
3. if `lateralized`: split `r` at the midline (± `overlap`) into `sensor_L`/`sensor_R`, transposed to CHW — these are the **raw pixel values**, bypassing steps 5-7 below entirely (no noise or dynamics applied)
4. otherwise: flatten the **full** frame to CHW as `u` (all rows and channels — not a centre-row subset)
5. add noise to `u` (if `noise_std > 0`)
6. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
7. per-pixel `output = leaky(u)` (if `tau_rise`/`tau_decay` set; else `output = u`) — this sensor's `bias`/`scale`/`activation` aren't exposed as parameters and are fixed at `0` / `1` / `linear`, so they have no effect

- `vertical_angle` — camera tilt in degrees. Positive = tilted down toward ground, negative = tilted up. Each image row sees a different ground distance: bottom rows see closer, top rows see farther. At 90° the camera looks straight down and the image goes black.
"""

    in_ch = 3
    mode  = 'rgb'

    def sample(self, x, y, theta, world, sim_cfg) -> np.ndarray:
        pixels, best_d = self._raycast(x, y, theta, world, sim_cfg)
        frame  = self._build_frame(pixels, best_d)                        # (H, W, 3)
        self._last_frame = frame
        if self.lateralized:
            mid     = self.width // 2
            l_end   = int(np.clip(mid + self.overlap, 0, self.width))
            r_start = int(np.clip(mid - self.overlap, 0, self.width))
            # CHW so _conv_forward receives planar channels, not HWC-interleaved.
            self._left_output  = frame[:, :l_end,   :].transpose(2, 0, 1).reshape(-1).astype(np.float32)
            self._right_output = frame[:, r_start:, :].transpose(2, 0, 1).reshape(-1).astype(np.float32)
        out = frame.transpose(2, 0, 1).reshape(-1).astype(np.float32)    # CHW (3*H*W,)
        return self._process(out, sim_cfg)


# Registry — single source of truth for all sensor types.
SENSOR_REGISTRY = {
    'GradientSensor':        GradientSensor,
    'ColorSensor':           ColorSensor,
    'CollisionSensor':       CollisionSensor,
    'DistanceSensor':        DistanceSensor,
    'InteroceptiveSensor':   InteroceptiveSensor,
    'ProprioceptiveSensor':  ProprioceptiveSensor,
    'WhiskerSensor':         WhiskerSensor,
    'SkyCompassSensor':      SkyCompassSensor,
    'ManualBumpSensor':      ManualBumpSensor,
    'ManualCueSensor':       ManualCueSensor,
    'CameraSensor':          GrayCameraSensor,   # backward-compat alias (mode='gray' default)
    'GrayCameraSensor':      GrayCameraSensor,
    'RGBCameraSensor':       RGBCameraSensor,
}
