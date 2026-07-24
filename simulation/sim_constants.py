import json
import numpy as np

# ── Theme ──────────────────────────────────────────────────────────────────────
C = {
    'bg':      '#F0F2F5',
    'surface': '#FFFFFF',
    'border':  '#D0D8E0',
    'primary': '#A8C8E8',
    'success': '#A8D4B4',
    'danger':  '#EEB0B0',
    'warning': '#EEC898',
    'dark':    '#4A5A6A',
    'muted':   '#8A9AAA',
}

_CHAN_PALETTE = ['#6CC87A', '#6AAAD4', '#E07878', '#D4A060',
                 '#C0A0D0', '#7ECECE', '#EDD080', '#F0A0BC', '#F5B87A']

_TRAIL_COLOR = '#282C34'


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# UI palette for gradient patch draw buttons.
# Each entry: (letter, display_name, rgb_float_list, button_bg_hex)
GRADIENT_COLORS = [
    ('A', 'Red',     [1.0, 0.0, 0.0], '#FFAAAA'),
    ('B', 'Green',   [0.0, 1.0, 0.0], '#AAFFAA'),
    ('C', 'Blue',    [0.0, 0.0, 1.0], '#AAAAFF'),
    ('D', 'Yellow',  [1.0, 1.0, 0.0], '#FFFFA0'),
    ('E', 'Magenta', [1.0, 0.0, 1.0], '#FFAAFF'),
    ('F', 'Cyan',    [0.0, 1.0, 1.0], '#AAFFFF'),
]

# UI palette for solid-object draw buttons.
OBJECT_COLORS = [
    ('Z', 'Red',    [1.0, 0.0, 0.0], '#FF4444'),
    ('Y', 'Green',  [0.0, 0.7, 0.0], '#22BB22'),
    ('X', 'Blue',   [0.0, 0.0, 1.0], '#4444FF'),
    ('W', 'Yellow', [1.0, 0.8, 0.0], '#FFCC00'),
    ('V', 'Orange', [1.0, 0.5, 0.0], '#FF8800'),
    ('U', 'Purple', [0.6, 0.0, 0.8], '#9900CC'),
]
