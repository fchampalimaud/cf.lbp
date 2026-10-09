import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neurons_base import ACTIVATIONS, _activate, DynamicsBase, LayerBase

class Conv2dLayer(DynamicsBase, LayerBase):
    """
    2-D convolution layer for camera image input (H × W, 1 or 3 channels).

    The connection weight is a 4-D array (n_filters, in_ch, kH, kW).
    in_ch is 1 (grayscale) or 3 (RGB), determined by the connected camera's mode.
    Filters are defined in the Filter Stack dialog — n_filters equals the number
    of filters added to the stack.

    With pool='global_avg' or 'global_max' the output is (n_filters,) scalars,
    one per filter — n is known once the first connection is made.
    With pool='none' the full spatial feature map is output as a flat vector.

    Parameters
    ----------
    kernel_size : int   Square kernel size (kH = kW). Default 3.
    stride      : int   Convolution stride. Default 1.
    padding     : str   'same' — zero-pad to preserve H×W. 'valid' — no padding.
    pool        : str   'global_avg', 'global_max', or 'none'.
    activation  : str   Nonlinearity applied to feature maps BEFORE pooling (relu/sigmoid/tanh/linear).
    tau_rise    : float Rise τ for optional leaky dynamics on pooled output (0 = instant rise).
    tau_decay   : float Decay τ. None = instant decay.
    bias        : float Constant added to pooled output.
    """

    help_text = """\
## Conv2dLayer — 2-D convolution over camera input

Connect a `CameraSensor` to `Conv2dLayer`, then open the **Filter Stack** dialog.
Each filter is a `(in_ch, kH, kW)` kernel; `in_ch` is inferred from camera mode.

**Parameters:**
- `n_filters` — number of filters (= output neurons with global pool); default 1
- `kernel_size` — square kernel side kH = kW; default 3
- `stride` — convolution stride; default 1
- `padding` — `same` (preserve H×W) or `valid` (no padding); default `same`
- `pool` — `global_avg`, `global_max`, or `none`; default `global_avg`
- `activation` (f) — nonlinearity applied to feature maps before pooling; default `relu`
- `tau_rise` (τ_rise) — leaky dynamics rise τ on pooled output (0 = off); default 0.0
- `tau_decay` (τ_decay) — leaky dynamics decay τ; blank/None = instant decay
- `tau_a` (τ_a) — adaptation time constant (0 = off); default 0.0
- `beta` (β) — adaptation strength; 0 = no adaptation; default 0.0
- `bias` (b) — constant added to each pooled output; default 0.0
- `scale` — output multiplier; default 1.0
- `noise_std` — Gaussian noise σ injected each tick (0 = off)
- `noise_tau` — noise correlation time (0 = white noise)
- `lateralized` — create mirrored _L / _R pair for split-camera input; default False
**Forward pass** (I: in_ch × H × W, W: n_filters × in_ch × kH × kW):

$$M = f(\\text{conv2d}(I,\\, W)), \\quad \\text{pooled} = \\text{pool}(M) + b$$

**Optional leaky dynamics** (when `tau_rise > 0`):

$$\\frac{dx}{dt} = \\frac{\\text{pooled} - x}{\\tau}, \\quad \\text{output} = x \\times \\text{scale}$$

If both `tau_rise = 0` and `tau_decay = 0`: output = pooled × scale directly (no filtering).

**Optional adaptation** (when `beta > 0` and `tau_a > 0`):

A slow variable *a* tracks the layer's own output and subtracts from the effective input:

$$u_{eff} = \\text{pooled} - \\beta \\, a, \\quad \\frac{da}{dt} = \\frac{\\text{output} - a}{\\tau_a}$$

Larger β → stronger suppression of sustained responses (burst-then-adapt).
Smaller τ_a → faster adaptation, more transient responses.

**Order of operations** (per tick, picking up from `pooled` above) — note:
`noise_std`/`noise_tau` are exposed as parameters but are **not** currently
applied anywhere in this layer's per-tick update:
1. `u = pooled + bias`
2. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
3. subtract adaptation: `u -= β × a` (if `tau_a > 0` and `beta > 0`)
4. `x = leaky(u)` — τ_rise/τ_decay (instant on whichever side is 0; passthrough only if both are 0)
5. update adaptation from the pre-scale `x`: `a += (x − a) / τ_a × dt` (if `tau_a > 0` and `beta > 0`)
6. `output = x × scale`

- `pool='global_avg'` / `'global_max'` — output shape: `(n_filters,)`
- `pool='none'` — output shape: `(n_filters, H_out, W_out)`
- Use `padding='valid'` with zero-sum kernels to avoid edge artifacts.

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

    def __init__(self, n_filters=1, kernel_size=3, stride=1, padding='same',
                 pool='global_avg', lateralized=False,
                 name='conv2d', n=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.kernel_size = int(kernel_size)
        self.stride      = max(1, int(stride))
        self.padding     = padding
        self.pool        = pool
        self.n_filters   = max(1, int(n_filters))
        self.lateralized = bool(lateralized)
        if self.pool == 'none' and self.n_filters > 1:
            raise ValueError(
                f"Conv2dLayer '{name}': pool='none' requires n_filters=1 "
                f"(got n_filters={self.n_filters}). Set n_filters=1 to use spatial output mode."
            )
        if self.pool == 'none':
            self.viz_n       = 1
            self._last_frame = None
            self.frame_h     = None
            self.frame_w     = None
        n      = n if n is not None else self.n_filters
        self.n = n
        self._init_dynamics_buffers(n)
        self.output = torch.zeros(n)

    @classmethod
    def param_defs(cls):
        return [
            ('n_filters',   int,   '1',          'number of filters (= output neurons with global pool)'),
            ('kernel_size', int,   '3',          'square kernel size (kH = kW)'),
            ('stride',      int,   '1',           'convolution stride'),
            ('padding',     str,   'same',        'same: preserve H×W   valid: no padding',
             ['same', 'valid']),
            ('pool',        str,   'global_avg',  'how to collapse spatial dims to a scalar per filter',
             ['global_avg', 'global_max', 'none']),
            ('activation',  str,   'relu',        'nonlinearity applied to feature maps before pooling',
             ACTIVATIONS),
            ('tau_rise',    float, '0.0',         'leaky rise τ on pooled output (0 = instant rise)'),
            ('tau_decay',   float, '0.0',         'leaky decay τ (blank/None = instant decay)'),
            ('tau_a',       float, '0.0',         'adaptation time constant (0 = off)'),
            ('beta',        float, '0.0',         'adaptation strength (0 = off)'),
            ('bias',        float, '0.0',         'constant added to pooled output'),
            ('scale',       float, '1.0',         'output scale factor applied after dynamics'),
            ('noise_std',   float, '0.0',         'Gaussian noise σ injected each tick (0 = off)'),
            ('noise_tau',   float, '0.0',         'noise correlation time (0 = white noise)'),
            ('lateralized', bool,  False,         'create mirrored _L / _R pair for split-camera input'),
        ]

    def _ensure_n(self, n):
        if self.n != n:
            self.n = n
            # n_filters is set by the weight tensor shape; for lateralized camera n = 2 * n_filters.
            self._init_dynamics_buffers(n)
            self.output = torch.zeros(n)

    def step(self, input_vec, dt):
        # input_vec has activation applied in _conv_forward before pooling — no activation here.
        x = self._filter(self._input(input_vec, dt), dt)
        # _a tracks pre-scale output so beta operates in the same units as the input.
        self._update_adaptation(x, dt)
        out = self._emit(x, activate=False)
        self.output = out.detach()
        if self.pool == 'none':
            self._update_last_frame(self.output.numpy())
        return self.output

    def _update_last_frame(self, arr):
        """Reshape flat spatial output to (H, W) for display (pool='none' only)."""
        H = getattr(self, 'frame_h', None)
        W = getattr(self, 'frame_w', None)
        if H is None or W is None:
            n = arr.shape[0]
            sq = int(n ** 0.5)
            H = sq
            W = max(1, n // max(sq, 1))
        try:
            self._last_frame = arr.reshape(H, W)
        except ValueError:
            pass

    # Capabilities (see LayerBase): image in (camera or image layer), 4-D kernels.
    accepts_image    = True
    kernel_weights   = True
    supports_lateral = True

    @property
    def is_image_node(self):
        return self.pool == 'none'

    def thumbnail_frames(self, disp_h=32):
        if self.pool != 'none':
            return
        frame = getattr(self, '_last_frame', None)
        if frame is None:
            return
        g    = frame if frame.ndim == 2 else np.mean(frame, axis=-1)
        data = np.clip(g * 255, 0, 255).astype(np.uint8)
        data = np.stack([data, data, data], axis=-1)
        reps = max(1, disp_h // max(data.shape[0], 1))
        data = np.repeat(data, reps, axis=0)[:disp_h]
        yield self.name, data

    def init_code_parts(self):
        parts = [f'n_filters={self.n_filters}', f'kernel_size={self.kernel_size}']
        parts += self._base_code_parts()
        if self.stride != 1:
            parts.append(f'stride={self.stride}')
        if self.padding != 'same':
            parts.append(f"padding='{self.padding}'")
        if self.pool != 'global_avg':
            parts.append(f"pool='{self.pool}'")
        if self.activation != 'relu':
            parts.append(f"activation='{self.activation}'")
        if self.tau_rise != 0.0:
            parts.append(f'tau_rise={self.tau_rise}')
        if self.tau_decay != self.tau_rise:
            parts.append(f'tau_decay={self.tau_decay}')
        if self.bias != 0.0:
            parts.append(f'bias={self.bias}')
        if getattr(self, 'scale', 1.0) != 1.0:
            parts.append(f'scale={self.scale}')
        if getattr(self, 'output_mode', 'none') != 'none':
            parts.append(f"output_mode='{self.output_mode}'")
        if getattr(self, 'lateralized', False):
            parts.append('lateralized=True')
        parts += self._mod_code_parts()
        return parts


class Leaky2dLayer(DynamicsBase, LayerBase):
    """
    Pixel-wise leaky integrator that preserves the full spatial image structure.

    Applies first-order temporal low-pass filtering independently to each pixel
    of a camera sensor's output (or another Leaky2dLayer's output).  Unlike
    Conv2dLayer — which pools spatial responses down to a feature vector —
    Leaky2dLayer keeps the image dimensions intact so the result can be fed
    directly into a Conv2dLayer or inspected as a filtered image.

    Typical uses
    ────────────
    • Retinal temporal adaptation: slow low-pass filter per pixel.
    • Motion / optic-flow detection: set output_mode='derivative' → each
      pixel's raw input becomes its own rate of change (finite difference
      between consecutive ticks) before the leaky filter/activation run;
      use scale=-1 to flip which direction reads positive.

    Connection weight
    ─────────────────
    The connection from a CameraSensor to a Leaky2dLayer uses a 1-D ones
    vector (shape n_pixels) stored as an element-wise passthrough.  No matrix
    multiply is performed; the weight just gates each pixel individually.

    Frame metadata
    ──────────────
    After each step() the layer updates _last_frame (H×W or H×W×3 numpy array)
    so a downstream Conv2dLayer can infer the correct spatial dimensions.
    frame_h / frame_w are set automatically when a camera is connected and are
    also serialized so the shape survives save / load.
    """

    help_text = """\
## Leaky2dLayer — pixel-wise temporal filter (image → image)

Applies the same first-order leaky integration as `LeakyLayer` but independently
to **each pixel** of the input image, producing an output image of the same spatial size.
Downstream `Conv2dLayer` nodes can use this as their source.

**Parameters:**
- `lateralized` — create mirrored _L/_R pair for split-camera input
- `tau_rise` (τ_rise) — rise time constant (s); default 0.1
- `tau_decay` (τ_decay) — decay τ (s); blank/None = instant decay
- `activation` (f) — nonlinearity per pixel: `linear`, `relu`, `sigmoid`, `tanh`; default `linear`
- `output_mode` — `none` / `derivative` / `integral` (per pixel); default `none`
- `bias` (b) — constant added to each pixel before integration; default 0.0
- `scale` (s) — output multiplier; default 1.0
- `noise_std` / `noise_tau` — per-pixel noise (same as `LeakyLayer`)
- `noise_tau` — noise correlation τ (0 = white noise)
**Dynamics** (u = pixel value + b, transformed by `output_mode` before filtering):

$$\\frac{dx}{dt} = \\frac{u - x}{\\tau}, \\quad \\tau = \\begin{cases} \\tau_{rise} & u > x \\\\ \\tau_{decay} & u \\leq x \\end{cases}$$

$$\\text{output pixel} = f(x) \\times s$$

**Motion mode** (`output_mode='derivative'`): each pixel's raw input becomes
its own rate of change between ticks *before* the leaky filter/activation
run. Use `scale = -1` to flip which direction (brightening vs. darkening)
reads positive.

**Order of operations** (per tick, per pixel):
1. `u = pixel + bias`
2. add noise to `u` (if `noise_std > 0`)
3. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
4. `x = leaky(u)` — asymmetric τ_rise/τ_decay integration
5. `output pixel = activation(x) × scale`

**Optic flow recipe:**
1. Connect `GrayCameraSensor` → `Leaky2dLayer(tau_rise=0.2, output_mode='derivative', scale=-1, activation='relu')`
2. Connect `Leaky2dLayer` → `Conv2dLayer` to extract spatial motion features.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits.
- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute / derivative / integral) (pre / post / none).
"""

    viz_n        = 1     # show as a single image node in the network visualizer (like a camera)

    # Capabilities (see LayerBase): camera image in, image out, n = pixel count.
    is_image_node      = True
    accepts_image      = True
    needs_camera_input = True
    passthrough_input  = True
    supports_lateral   = True
    n_follows_input    = True
    saved_state        = {'n': 'n', 'in_ch': 'in_ch', 'frame_h': 'frame_h', 'frame_w': 'frame_w'}

    def __init__(self, lateralized=False,
                 in_ch=1, frame_h=None, frame_w=None,
                 name='leaky2d', n=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.lateralized = bool(lateralized)
        self.in_ch      = int(in_ch)
        self.frame_h    = int(frame_h) if frame_h is not None else None
        self.frame_w    = int(frame_w) if frame_w is not None else None
        self._last_frame = None
        n = n if n is not None else 1  # placeholder so the node is visible before connecting
        self.n = n
        self._init_dynamics_buffers(n)
        self.output = torch.zeros(n)

    @classmethod
    def param_defs(cls):
        return [
            ('lateralized', bool,  False,    'create mirrored _L/_R pair for split-camera input'),
            ('tau_rise',   float, '0.1',    'rise τ (s)'),
            ('tau_decay',  float, '0.1',    'decay τ (s; blank/None = instant decay)'),
            ('activation', str,   'linear', 'per-pixel nonlinearity', ACTIVATIONS),
            ('bias',       float, '0.0',    'constant added to each pixel input'),
            ('scale',      float, '1.0',    'output multiplier'),
            ('noise_std',  float, '0.0',    'per-pixel noise amplitude (0 = off)'),
            ('noise_tau',  float, '0.0',    'noise correlation τ (0 = white noise)'),
        ]

    def _ensure_n(self, n):
        if self.n != n:
            self.n = n
            self._init_dynamics_buffers(n)
            self.output = torch.zeros(n)

    def step(self, input_vec, dt):
        out = self._emit(self._filter(self._input(input_vec, dt), dt))
        self.output = out.detach()
        self._update_last_frame(self.output.numpy())
        return self.output

    def _update_last_frame(self, arr):
        """Reshape flat output → (H, W) or (H, W, C) for downstream Conv2d."""
        n = arr.shape[0]
        n_pixels = n // max(self.in_ch, 1)
        H = self.frame_h if self.frame_h else int(n_pixels ** 0.5)
        W = self.frame_w if self.frame_w else max(1, n_pixels // max(H, 1))
        # Guard against non-exact integer divisions (e.g. in_ch wrong after reload).
        if H * W != n_pixels:
            W = n_pixels  # fall back to a 1×n_pixels strip
            H = 1
        try:
            if self.in_ch > 1:
                self._last_frame = arr.reshape(self.in_ch, H, W).transpose(1, 2, 0)
            else:
                self._last_frame = arr.reshape(H, W)
        except ValueError:
            pass  # dimensions still inconsistent; keep previous frame

    def thumbnail_frames(self, disp_h=32):
        frame = getattr(self, '_last_frame', None)
        if frame is None:
            return
        if self.in_ch == 3:
            rgb = frame
        else:
            g   = frame if frame.ndim == 2 else np.mean(frame, axis=-1)
            rgb = np.stack([g, g, g], axis=-1)
        if getattr(self, 'output_mode', 'none') != 'none':
            # derivative/integral output can be negative or exceed [0,1] — use a
            # wider display range centered at mid-grey instead of clipping to [0,1].
            data = np.clip((rgb + 1.0) * 127.5, 0, 255).astype(np.uint8)
        else:
            data = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        reps = max(1, disp_h // max(data.shape[0], 1))
        data = np.repeat(data, reps, axis=0)[:disp_h]
        yield self.name, data

    def init_code_parts(self):
        parts = [f'tau_rise={self.tau_rise}', f'tau_decay={self.tau_decay}']
        parts += self._base_code_parts()
        if self.activation != 'linear':
            parts.append(f"activation='{self.activation}'")
        if self.n is not None:
            parts.append(f'n={self.n}')
        if self.in_ch != 1:
            parts.append(f'in_ch={self.in_ch}')
        if self.frame_h is not None:
            parts.append(f'frame_h={self.frame_h}')
        if self.frame_w is not None:
            parts.append(f'frame_w={self.frame_w}')
        if self.bias != 0.0:
            parts.append(f'bias={self.bias}')
        if getattr(self, 'noise_std', 0.0):
            parts.append(f'noise_std={self.noise_std}')
        if getattr(self, 'noise_tau', 0.0):
            parts.append(f'noise_tau={self.noise_tau}')
        if getattr(self, 'scale', 1.0) != 1.0:
            parts.append(f'scale={self.scale}')
        if getattr(self, 'output_mode', 'none') != 'none':
            parts.append(f"output_mode='{self.output_mode}'")
        parts += self._mod_code_parts()
        return parts


class Reichardt2dLayer(DynamicsBase, LayerBase):
    """
    Elementary motion detector (Reichardt correlator) over camera image input.

    For each direction (dy, dx) and pixel (i, j), the balanced Reichardt response is:

        R(i, j) = I_del(i, j) · I_cur(i+dy, j+dx) − I_del(i+dy, j+dx) · I_cur(i, j)

    where I_del is an exponential low-pass of the input image with time constant
    tau_delay. Positive output means motion in the preferred direction; negative
    means motion in the opposite direction.

    n_directions evenly-spaced preferred directions are sampled starting at
    init_direction:

        direction_k = init_direction + k * 360 / n_directions   (k = 0 .. n_directions-1)

    Angles are degrees, 0° = rightward, increasing counter-clockwise (matching
    the simulator's heading convention elsewhere). Each direction's pixel
    offset is sampled with bilinear interpolation, so directions need not
    land on the integer pixel grid — n_directions and init_direction can be
    any values, not just multiples of 90°/45°.

    With pool='global_avg' or 'global_max' the output is (n_directions,) scalars, one
    per direction. With pool='none' all direction maps are stacked into a
    (n_directions × H × W) motion-field image, displayed as RGB with user-selected
    direction-to-channel mapping (disp_r, disp_g, disp_b). The flat output vector has
    size n_directions × H × W.

    pool='global_avg' averages the raw signed R map over the whole image FIRST,
    then applies activation to that single scalar — not the other way round.
    R is a signed, spatially-balanced quantity (positive pixels vote for this
    direction, negative pixels vote for the opposite one) that is meant to
    average out to a clean net direction signal. Clipping every pixel with
    activation *before* averaging (the right order for a feature detector
    like Conv2dLayer) would throw away exactly that cancellation, leaving two
    opposite-direction channels responding almost identically on any textured
    image. pool='global_max' is unaffected by this ordering (every activation
    here is monotonic, so max(activate(x)) == activate(max(x))).

    Parameters
    ----------
    n_directions   : int   Number of evenly-spaced preferred directions. Default 4.
    init_direction : float Angle (degrees) of direction 0; 0 = rightward, CCW. Default 0.0.
    offset     : int   Spatial pixel offset Δ for the correlation. Default 1.
    tau_delay  : float Internal Reichardt delay time constant (s). Default 0.1.
    pool       : str   'global_avg', 'global_max', or 'none'. Default 'global_avg'.
    activation : str   Nonlinearity applied after pooling (global_avg/global_max)
                       or per-pixel (pool='none').
    tau_rise   : float Rise τ for optional leaky dynamics on pooled output (0 = instant rise).
    tau_decay  : float Decay τ. None = instant decay.
    tau_a      : float Adaptation τ (0 = off). Default 0.0.
    beta       : float Adaptation strength. Default 0.0.
    bias       : float Constant added to pooled output. Default 0.0.
    scale      : float Output multiplier. Default 1.0.
    lateralized: bool  Create mirrored _L / _R pair for split-camera input.
    disp_r     : int   Direction index for Red channel in image display (-1 = black).
    disp_g     : int   Direction index for Green channel (-1 = black).
    disp_b     : int   Direction index for Blue channel (-1 = black).
    """

    help_text = """\
## Reichardt2dLayer — elementary motion detector over camera input

Implements the balanced Reichardt (EM detector) correlator. For each direction and
pixel `(i, j)`:

$$R(i,j) = I_{del}(i,j) \\cdot I_{cur}(i{+}\\Delta y,\\,j{+}\\Delta x)
          - I_{del}(i{+}\\Delta y,\\,j{+}\\Delta x) \\cdot I_{cur}(i,j)$$

where $I_{del}$ is an exponential low-pass of the input with time constant `tau_delay`.
Positive output → motion in the preferred direction. Negative → opposite direction.

**Parameters:**
- `n_directions` — number of evenly-spaced preferred directions; default 4
- `init_direction` (φ₀) — angle of direction 0, degrees (0 = right, CCW); others follow
  `init_direction + k * 360 / n_directions`; any combination works (non-axis-aligned angles
  fall back to bilinear interpolation), but `n_directions` ∈ {1, 2, 4} with `init_direction`
  a multiple of 90° keeps every shift on exact integer pixels — no interpolation, noticeably
  cheaper per tick; default 0.0
- `offset` (Δ) — spatial pixel offset for the correlation; default 1
- `tau_delay` (τ_d) — internal Reichardt delay time constant (s); default 0.1
- `pool` — `global_avg`, `global_max`, or `none`; default `global_avg`
- `activation` (f) — nonlinearity applied **after** pooling for `global_avg`/`global_max`
  (see note below), or per-pixel for `pool='none'`; default `relu`
- `tau_rise` (τ_rise) — leaky rise τ on pooled output (0 = off); default 0.0
- `tau_decay` (τ_decay) — leaky decay τ; blank/None = instant decay; default 0.0
- `tau_a` (τ_a) — adaptation time constant (0 = off); default 0.0
- `beta` (β) — adaptation strength (0 = off); default 0.0
- `bias` (b) — constant added to each pooled output; default 0.0
- `scale` — output multiplier; default 1.0
- `lateralized` — create mirrored _L / _R pair for split-camera input; default False
- `noise_std` (σ) — Gaussian noise std on pooled output (0 = off); default 0.0
- `noise_tau` — Ornstein-Uhlenbeck time constant for noise (0 = white noise); default 0.0
**Forward pass:**

1. Low-pass filter each pixel with `tau_delay` → $I_{del}$
2. For each direction: compute balanced Reichardt map $R$ (see equation above)
3. `global_avg`/`global_max`: pool the raw signed $R$ map first, **then** apply
   activation to that single scalar. `pool='none'`: activation is applied per pixel
   (there's no pooling to protect).
4. Add bias → optional noise → optional `output_mode` transform → optional
   adaptation subtraction → leaky dynamics → update adaptation → × scale
   (full per-tick breakdown below)

**Order of operations** (per tick, in full):
1. reshape the flat input vector to `(in_ch, H, W)`
2. update the delay buffer: `I_del += (I_cur − I_del) × min(1, dt / τ_delay)` — exponential low-pass of the image
3. for each of the `n_directions` preferred directions `(dy, dx)`:
   - shift `I_cur` and `I_del` by `(dy, dx)`
   - `R = mean_channels(I_del × I_cur_shifted − I_del_shifted × I_cur)`
   - `global_avg`: `pooled_k = activation(mean(R))`; `global_max`: `pooled_k = activation(max(R))`; `none`: `pooled_k = activation(R)` per pixel
4. stack the `n_directions` responses into `pooled`
5. `u = pooled + bias`
6. add noise to `u` (if `noise_std > 0`)
7. apply `output_mode` transform to `u` — derivative/integral (if not `none`)
8. subtract adaptation: `u -= β × a` (if `tau_a > 0` and `beta > 0`)
9. `x = leaky(u)` — asymmetric τ_rise/τ_decay integration
10. update adaptation from the pre-scale `x`: `a += (x − a) / τ_a × dt` (if `tau_a > 0` and `beta > 0`)
11. `output = x × scale`

**Why pool before activation (global_avg/global_max):**
$R$ is signed and spatially balanced — positive pixels vote for this direction,
negative pixels vote for the opposite one, and they're meant to cancel out when
averaged, leaving a clean net direction signal. Clipping every pixel with
`activation` *before* averaging (the order Conv2dLayer uses, correct for a
feature detector) throws away that cancellation: any isolated positive blip in
the "wrong" direction's map survives the clip and gets counted, so two exactly
opposite directions end up responding almost identically on a textured image
instead of one being clearly suppressed. `pool='global_max'` doesn't have this
problem either way — every built-in activation is monotonic, so
`max(activation(x)) == activation(max(x))`.

**Output shape:**
- `pool = 'global_avg'` / `'global_max'` → `(n_directions,)` scalars
- `pool = 'none'` → `(n_directions × H × W,)` flat motion-field vector; displayed as RGB image
  with `disp_r`, `disp_g`, `disp_b` selecting which direction maps to each channel (-1 = black)

**Tuning tips:**
- Set `tau_delay ≈ 1 / fps` for one-frame delay (optimal for frame-rate motion).
- `offset = 1` detects smallest shifts; increase for slower / large-scale motions.
- Use `activation='linear'` to preserve sign — negative means opposite-direction motion.
- For push-pull motor drive: `n_directions=2`, wire outputs [0] and [1] with opposite signs.

---

**Neuromodulation:**

- `neuromodulator_transmitter` — name of the signal this layer emits; its mean output is published to the bus each tick.
- `neuromodulator_color` — display color for this neuromodulator in the visualizer.
- `modulators` — list of `(name, scale, site, mode)` rows (`mode`: absolute / derivative / integral):
  - `site="pre"`: multiplies the pooled output by `1 + scale × signal` before dynamics.
  - `site="post"`: multiplies the output by `1 + scale × signal` after dynamics.
  - `site="none"`: declares the neuromodulator for learning/visualization only.
  - `scale > 0` → excitatory; `scale < 0` → inhibitory.
"""

    def __init__(self, n_directions=4, init_direction=0.0, offset=1, tau_delay=0.1,
                 pool='global_avg', lateralized=False,
                 in_ch=1, frame_h=None, frame_w=None,
                 disp_r=0, disp_g=1, disp_b=2,
                 name='reichardt', flat_n=None, **kwargs):
        super().__init__(name=name, **kwargs)
        if int(n_directions) < 1:
            raise ValueError(
                f"Reichardt2dLayer '{name}': n_directions must be >= 1 (got {n_directions})")

        self.n_directions   = int(n_directions)
        self.init_direction = float(init_direction)
        self.offset      = max(1, int(offset))
        self.tau_delay   = float(tau_delay)
        self.pool        = pool
        self.lateralized = bool(lateralized)
        self.in_ch       = int(in_ch)
        self.frame_h     = int(frame_h) if frame_h is not None else None
        self.frame_w     = int(frame_w) if frame_w is not None else None
        self.disp_r      = int(disp_r)
        self.disp_g      = int(disp_g)
        self.disp_b      = int(disp_b)
        self._last_frame = None
        self._shift_grid_cache = {}   # (dy, dx, H, W) -> cached grid_sample grid

        if pool == 'none':
            self.viz_n = 1

        # `flat_n` restores an already-expanded (pool='none') buffer length at
        # load time, before the first step() has a chance to call _ensure_n —
        # kept separate from n_directions (the hyperparameter) since `.n` is
        # also read generically elsewhere (connections, visualizer) as "this
        # layer's current output length", which differs from n_directions
        # whenever pool='none' expands it to n_directions * H * W.
        n_out = flat_n if flat_n is not None else self.n_directions
        self.n = n_out
        self._init_dynamics_buffers(n_out)
        self.output = torch.zeros(n_out)
        self.register_buffer('_I_del', None)

    def _direction_vectors(self):
        """(dy, dx) pixel-offset vectors (scaled by offset) for each of the
        n_directions evenly-spaced angles starting at init_direction.
        Degrees, 0°=right, increasing counter-clockwise."""
        vectors = []
        for k in range(self.n_directions):
            theta = math.radians(self.init_direction + k * 360.0 / self.n_directions)
            dx = math.cos(theta) * self.offset
            dy = -math.sin(theta) * self.offset
            vectors.append((dy, dx))
        return vectors

    @classmethod
    def param_defs(cls):
        return [
            ('n_directions',   int,   '4',   'number of evenly-spaced preferred directions; '
                                              '1/2/4 with init_direction a multiple of 90° keeps '
                                              'every direction axis-aligned (fast integer-pixel '
                                              'shift, no interpolation) — other values/angles fall '
                                              'back to a slower bilinear-interpolated shift'),
            ('init_direction', float, '0.0', 'angle of direction 0, degrees (0=right, CCW); '
                                              'others follow init_direction + k*360/n_directions; '
                                              'a multiple of 90° keeps shifts axis-aligned (fast path)'),
            ('offset',      int,   '1',          'spatial pixel offset Δ for the correlation'),
            ('tau_delay',   float, '0.1',        'internal Reichardt delay τ (s)'),
            ('pool',        str,   'global_avg', 'global_avg/global_max → (n_directions,) scalars; none → (n_directions×H×W) image',
             ['global_avg', 'global_max', 'none']),
            ('activation',  str,   'relu',       'nonlinearity applied to correlation maps before pooling',
             ACTIVATIONS),
            ('tau_rise',    float, '0.0',        'leaky rise τ on pooled output (0 = instant rise)'),
            ('tau_decay',   float, '0.0',        'leaky decay τ (blank/None = instant decay)'),
            ('tau_a',       float, '0.0',        'adaptation time constant (0 = off)'),
            ('beta',        float, '0.0',        'adaptation strength (0 = off)'),
            ('bias',        float, '0.0',        'constant added to pooled output'),
            ('scale',       float, '1.0',        'output scale factor applied after dynamics'),
            ('lateralized', bool,  False,        'create mirrored _L / _R pair for split-camera input'),
            ('noise_std',   float, '0.0',        'noise amplitude on pooled output (0 = off)'),
            ('noise_tau',   float, '0.0',        'noise correlation τ (0 = white noise)'),
            ('disp_r',      int,   '0',          'R: direction index → Red channel (pool=none only; -1 = black)',   None, 'display RGB'),
            ('disp_g',      int,   '1',          'G: direction index → Green channel (-1 = black)',                  None, 'display RGB'),
            ('disp_b',      int,   '2',          'B: direction index → Blue channel (-1 = black)',                   None, 'display RGB'),
        ]

    def _ensure_n(self, n):
        if self.n != n:
            self.n = n
            self._init_dynamics_buffers(n)
            self.output = torch.zeros(n)

    def reset(self):
        self._reset_dynamics()
        if self._I_del is not None:
            self._I_del.zero_()
        self.output = torch.zeros(self.n)

    def _shift(self, img, dy, dx):
        """Zero-pad-shift img (in_ch × H × W) by (dy, dx) pixels so that
        result[i,j] = img sampled at (i+dy, j+dx), zero outside bounds —
        matching R(i,j) = I_del(i,j) * I_cur(i+dy,j+dx) - ... in the class
        docstring, so the (dy,dx)=(0,1) 'rightward' filter actually responds
        to rightward motion.

        Exact-integer (dy, dx) — always the case for axis-aligned direction
        sets (e.g. n_directions=1/2/4 with init_direction a multiple of 90°)
        — take a zero-padded slice: no interpolation, no per-tick tensor
        allocation. Fractional (dy, dx) (diagonal/odd-angle direction sets)
        fall back to bilinear grid_sample, with the sampling grid cached
        per (dy, dx, H, W) instead of rebuilt every tick.
        """
        H, W = img.shape[-2], img.shape[-1]
        idy, idx = round(dy), round(dx)
        if abs(dy - idy) < 1e-6 and abs(dx - idx) < 1e-6:
            return self._shift_int(img, idy, idx)

        key = (dy, dx, H, W)
        grid = self._shift_grid_cache.get(key)
        if grid is None:
            ys = torch.arange(H, dtype=torch.float32, device=img.device) + dy
            xs = torch.arange(W, dtype=torch.float32, device=img.device) + dx
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            norm_x = grid_x / max(W - 1, 1) * 2 - 1
            norm_y = grid_y / max(H - 1, 1) * 2 - 1
            grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
            self._shift_grid_cache[key] = grid
        sampled = F.grid_sample(img.unsqueeze(0), grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=True)
        return sampled.squeeze(0)

    @staticmethod
    def _shift_int(img, dy, dx):
        """Exact-integer version of _shift: zero-padded slice, no interpolation."""
        if dy == 0 and dx == 0:
            return img
        _, H, W = img.shape
        out = torch.zeros_like(img)
        y0, y1 = max(0, -dy), min(H, H - dy)
        x0, x1 = max(0, -dx), min(W, W - dx)
        if y1 > y0 and x1 > x0:
            out[:, y0:y1, x0:x1] = img[:, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        return out

    def step(self, input_vec, dt):
        inp = torch.as_tensor(input_vec, dtype=torch.float32)

        # --- reshape flat input to (in_ch, H, W) ---
        n_pixels = inp.numel() // max(self.in_ch, 1)
        H = self.frame_h if self.frame_h else int(n_pixels ** 0.5)
        W = self.frame_w if self.frame_w else max(1, n_pixels // max(H, 1))
        I_cur = inp.reshape(self.in_ch, H, W)

        # --- update internal Reichardt delay buffer (exponential low-pass) ---
        if self._I_del is None or self._I_del.shape != I_cur.shape:
            self.register_buffer('_I_del', I_cur.detach().clone())
        # In-place update (.copy_ rather than `self._I_del = ...`) — avoids
        # nn.Module.__setattr__'s expensive re-registration of an already-
        # registered buffer on every tick; see _apply_leaky's docstring.
        I_del_prev = self._I_del.detach()
        alpha = min(1.0, dt / max(self.tau_delay, 1e-9))
        self._I_del.copy_(I_del_prev + (I_cur - I_del_prev) * alpha)

        I_del = self._I_del

        # --- Reichardt correlation for each direction ---
        if self.pool == 'none':
            self._ensure_n(self.n_directions * H * W)

        dir_responses = []
        for (dy, dx) in self._direction_vectors():
            I_cur_sh = self._shift(I_cur, dy, dx)
            I_del_sh = self._shift(I_del, dy, dx)
            R_map = I_del * I_cur_sh - I_del_sh * I_cur    # (in_ch, H, W)
            R_map = R_map.mean(dim=0, keepdim=True)          # (1, H, W) — collapse channels
            if self.pool == 'global_avg':
                # Pool the raw signed correlator FIRST, then activate the scalar.
                # R_map is a signed, spatially-balanced quantity (positive pixels
                # vote for this direction, negative pixels vote for the opposite
                # one) that is supposed to average out to a clean net signal.
                # Activating per-pixel before pooling (the Conv2dLayer order,
                # right for a feature detector) would clip away exactly that
                # cancellation, leaving both opposite-direction channels
                # correlated on any textured image. See help_text.
                dir_responses.append(_activate(R_map.mean(), self.activation, alpha=self.alpha))
            elif self.pool == 'global_max':
                # Order doesn't matter here: every ACTIVATIONS entry is
                # monotonic non-decreasing, so activate(max(x)) == max(activate(x)).
                dir_responses.append(_activate(R_map.amax(), self.activation, alpha=self.alpha))
            else:  # pool='none' — keep full spatial map per direction
                dir_responses.append(_activate(R_map, self.activation, alpha=self.alpha).squeeze(0))    # (H, W)

        if self.pool == 'none':
            pooled = torch.stack(dir_responses).reshape(-1)  # (n_directions * H * W,)
        else:
            pooled = torch.stack(dir_responses)              # (n_directions,)

        # --- output dynamics: bias, noise, output_mode, adaptation, leaky, scale ---
        x = self._filter(self._input(pooled, dt), dt)   # activation already ran per direction
        self._update_adaptation(x, dt)
        out = self._emit(x, activate=False)
        self.output = out.detach()

        if self.pool == 'none':
            self._last_frame = self.output.numpy().reshape(self.n_directions, H, W)

        return self.output

    # Capabilities (see LayerBase): camera image in through a ones passthrough;
    # with pool='none' the output is a signed correlation image.
    signed_image       = True
    accepts_image      = True
    needs_camera_input = True
    passthrough_input  = True
    supports_lateral   = True
    # 'n' is the n_directions hyperparameter (param_defs) — 'flat_n' is the
    # possibly pool='none'-expanded live buffer length (n_directions * H * W),
    # restored eagerly on load so connections/visualizer see the right size
    # before the first step().
    saved_state        = {'flat_n': 'n', 'in_ch': 'in_ch', 'frame_h': 'frame_h', 'frame_w': 'frame_w'}

    @property
    def is_image_node(self):
        return self.pool == 'none'

    @property
    def n_follows_input(self):
        return self.pool == 'none'

    def thumbnail_frames(self, disp_h=32):
        if self.pool != 'none':
            return
        frame = getattr(self, '_last_frame', None)   # (n_directions, H, W) numpy
        if frame is None:
            return
        n, H, W = frame.shape

        def _ch(idx):
            if idx < 0 or idx >= n:
                return np.zeros((H, W), dtype=np.uint8)
            # Correlation is signed; centre at 127.5 grey.
            return np.clip((frame[idx] + 1.0) * 127.5, 0, 255).astype(np.uint8)

        data = np.stack([_ch(self.disp_r), _ch(self.disp_g), _ch(self.disp_b)], axis=-1)
        scale = max(1, disp_h // max(H, 1))
        data = np.repeat(np.repeat(data, scale, axis=0), scale, axis=1)[:disp_h]
        yield self.name, data

    def init_code_parts(self):
        parts = [f'n_directions={self.n_directions}', f'offset={self.offset}',
                 f'tau_delay={self.tau_delay}']
        if self.init_direction != 0.0:
            parts.append(f'init_direction={self.init_direction}')
        parts += self._base_code_parts()
        if self.pool != 'global_avg':
            parts.append(f"pool='{self.pool}'")
        if self.activation != 'relu':
            parts.append(f"activation='{self.activation}'")
        if self.tau_rise != 0.0:
            parts.append(f'tau_rise={self.tau_rise}')
        if self.tau_decay != self.tau_rise:
            parts.append(f'tau_decay={self.tau_decay}')
        if self.bias != 0.0:
            parts.append(f'bias={self.bias}')
        if getattr(self, 'scale', 1.0) != 1.0:
            parts.append(f'scale={self.scale}')
        if getattr(self, 'lateralized', False):
            parts.append('lateralized=True')
        if self.in_ch != 1:
            parts.append(f'in_ch={self.in_ch}')
        if self.frame_h is not None:
            parts.append(f'frame_h={self.frame_h}')
        if self.frame_w is not None:
            parts.append(f'frame_w={self.frame_w}')
        if getattr(self, 'noise_std', 0.0):
            parts.append(f'noise_std={self.noise_std}')
        if getattr(self, 'noise_tau', 0.0):
            parts.append(f'noise_tau={self.noise_tau}')
        if getattr(self, 'output_mode', 'none') != 'none':
            parts.append(f"output_mode='{self.output_mode}'")
        if self.disp_r != 0:
            parts.append(f'disp_r={self.disp_r}')
        if self.disp_g != 1:
            parts.append(f'disp_g={self.disp_g}')
        if self.disp_b != 2:
            parts.append(f'disp_b={self.disp_b}')
        parts += self._mod_code_parts()
        return parts


