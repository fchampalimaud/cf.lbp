"""
osc_controller.py — oscilloscope channel management for the LBP simulator.

Owns the set of tracked channels, their colours, the per-channel multiplier
spinboxes, trace ring-buffers, and the logic that discovers which channels the
current brain/circuit expose. The oscilloscope plot widget and its side-panel
layout are passed in so this class can rebuild them without touching
SimulatorApp.
"""

from collections import deque

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel

from sim_constants import C, _CHAN_PALETTE
from sim_widgets import MonetarySpinBox


class OscChannelManager:
    """
    Manages oscilloscope channels for the running simulation.

    Parameters
    ----------
    osc_widget   : OscilloscopeWidget  — the plot to call setup/update on
    mult_layout  : QVBoxLayout         — side-panel layout for per-channel spinboxes
    buf_len      : int                 — trace ring-buffer length
    """

    def __init__(self, osc_widget, mult_layout, buf_len=1000):
        self._osc         = osc_widget
        self._mult_layout = mult_layout
        self.buf_len      = buf_len

        self.channels        = []
        self.channel_colors  = {}
        self.trace_data      = {}
        self._osc_items      = {'mL', 'mR'}
        self._mult_spinboxes = {}
        self._mult_cache     = {}

    # ── Channel discovery ────────────────────────────────────────────────────

    def rebuild_channels(self, brain, circuit):
        """Rediscover channels from brain and circuit; reset trace buffers."""
        channels = []
        for ch in ('mL', 'mR', 'mL_sent', 'mR_sent'):
            if ch in self._osc_items:
                channels.append(ch)
        for sensor in circuit.sensors:
            if sensor.name in self._osc_items:
                for i in range(sensor.n_total):
                    channels.append(f'{sensor.name}_{i}')
        for layer in circuit.layers:
            if layer.n is None:
                continue
            if layer.name in self._osc_items:
                for i in range(layer.n):
                    channels.append(f'{layer.name}_{i}')
            else:
                for i in range(layer.n):
                    key = f'{layer.name}_{i}'
                    if key in self._osc_items:
                        channels.append(key)
        if not circuit.sensors:
            for ch in ('sL', 'sR'):
                if ch in self._osc_items:
                    channels.append(ch)
        channels += list(brain.plots() or [])
        self.channels = channels

        self.channel_colors = {
            ch: _CHAN_PALETTE[i % len(_CHAN_PALETTE)]
            for i, ch in enumerate(self.channels)
        }
        self.trace_data = {k: deque([0.0], maxlen=self.buf_len) for k in self.channels}
        self._rebuild_osc_controls()
        self._mult_cache = {k: sp.value() for k, sp in self._mult_spinboxes.items()}

    # ── Widget management ────────────────────────────────────────────────────

    def _rebuild_osc_controls(self):
        while self._mult_layout.count():
            item = self._mult_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._mult_spinboxes = {}

        for ch in self.channels:
            color = self.channel_colors.get(ch, '#888888')
            row   = QWidget()
            hl    = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            swatch = QLabel("■")
            swatch.setStyleSheet(f"color:{color};")
            hl.addWidget(swatch)
            lbl = QLabel(ch)
            lbl.setStyleSheet("font-size:8pt;")
            hl.addWidget(lbl)
            hl.addStretch()
            spin = MonetarySpinBox()
            spin.setValue(1.0)
            spin.setFixedWidth(62)
            hl.addWidget(spin)
            self._mult_spinboxes[ch] = spin
            self._mult_layout.addWidget(row)
        self._mult_layout.addStretch()

    def setup_osc(self):
        """Push current channels to the oscilloscope plot (call after rebuild_channels)."""
        self._osc.setup_channels(self.channels, self.channel_colors, self.buf_len)

    # ── Per-frame operations ─────────────────────────────────────────────────

    def refresh_mult_cache(self):
        """Read current spinbox values into the cache. Returns the cache dict."""
        self._mult_cache = {k: sp.value() for k, sp in self._mult_spinboxes.items()}
        return self._mult_cache

    def update_osc(self):
        """Push current trace data to the oscilloscope plot."""
        self._osc.update_channels(self.trace_data, self._mult_cache, self.channels)

    def append_trace(self, k, val):
        if k not in self.trace_data:
            self.trace_data[k] = deque([0.0], maxlen=self.buf_len)
        self.trace_data[k].append(val)

    def reset_trace(self):
        """Clear all trace ring-buffers."""
        for buf in self.trace_data.values():
            buf.clear()

    # ── Layer toggle (called from NetworkViz right-click) ────────────────────

    def toggle_layer(self, name):
        """Add/remove a layer name from the tracked set."""
        if name in self._osc_items:
            self._osc_items.discard(name)
        else:
            self._osc_items.add(name)

    # ── Session I/O helpers ──────────────────────────────────────────────────

    def apply_multipliers_from_json(self, multipliers_dict):
        """Set spinbox values from a loaded session or brain JSON."""
        for chan, val in multipliers_dict.items():
            if chan in self._mult_spinboxes:
                self._mult_spinboxes[chan].setValue(float(val))

    def get_multipliers(self):
        """Return current spinbox values as a dict (for session save)."""
        return {k: sp.value() for k, sp in self._mult_spinboxes.items()}
