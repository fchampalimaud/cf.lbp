import os
import json
import inspect
import numpy as np
from collections import defaultdict
from brain_serializer import (serialize_brain, _layer_to_dict,
                               _layer_from_dict, _connection_to_dict)

import pyqtgraph as pg

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QLineEdit, QCheckBox,
    QMenu, QDialog, QFormLayout, QDialogButtonBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout, QApplication,
    QComboBox, QDoubleSpinBox, QStackedWidget, QTextEdit, QGroupBox, QToolTip,
    QListWidget, QAbstractItemView, QSizePolicy, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, QTimer, QPointF, QEvent, QMimeData, QObject
from PySide6.QtGui import QColor, QFont, QDrag, QPainter

from sim_constants import C, _CHAN_PALETTE
from neurons import SumLayer, Conv2dLayer
from brain_base import DataBrain

MOTIFS_DIR = os.path.join(os.path.dirname(__file__), 'motifs')


def _make_help_html(markdown_text):
    """Generate a standalone HTML help page rendered with marked.js + KaTeX.

    Math is extracted before markdown processing so that `_` inside $...$ is
    not treated as italic, then restored via KaTeX renderToString.
    """
    import json
    raw = json.dumps(markdown_text)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Help</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px; line-height: 1.6; color: #222;
    max-width: 680px; margin: 0 auto; padding: 24px 28px 48px;
  }}
  h2 {{ margin-top: 20px; color: #1a1a2e; font-size: 1.22em; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
  h3 {{ margin-top: 14px; color: #333; font-size: 1.05em; }}
  code {{
    background: #f0f0f0; padding: 2px 5px; border-radius: 3px;
    font-family: "Cascadia Code", "Fira Code", "Consolas", monospace; font-size: 0.88em;
  }}
  pre {{ background: #f5f5f5; padding: 10px 14px; border-radius: 5px; overflow-x: auto; font-size: 0.88em; }}
  pre code {{ background: none; padding: 0; }}
  .katex-display {{ margin: 16px 0; overflow-x: auto; }}
  ul, ol {{ padding-left: 1.4em; }}
  li {{ margin: 3px 0; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 16px 0; }}
</style>
</head>
<body>
<div id="content"></div>
<script>
const raw = {raw};
function render(text) {{
  const blocks = [];
  // protect display math $$...$$ before inline
  text = text.replace(/\\$\\$([\\s\\S]+?)\\$\\$/g, function(_, m) {{
    blocks.push({{display: true, math: m}});
    return 'LBPM4TH' + (blocks.length - 1) + 'LBPM4TH';
  }});
  // protect inline math $...$
  text = text.replace(/\\$([^$\\n]+?)\\$/g, function(_, m) {{
    blocks.push({{display: false, math: m}});
    return 'LBPM4TH' + (blocks.length - 1) + 'LBPM4TH';
  }});
  let html = marked.parse(text);
  html = html.replace(/LBPM4TH(\\d+)LBPM4TH/g, function(_, i) {{
    const blk = blocks[parseInt(i)];
    try {{
      return katex.renderToString(blk.math, {{displayMode: blk.display, throwOnError: false}});
    }} catch(e) {{
      return '<code style="color:red">' + blk.math + '</code>';
    }}
  }});
  return html;
}}
document.getElementById('content').innerHTML = render(raw);
</script>
</body>
</html>"""


def _eff_n(obj):
    """Effective neuron count for layout. Multi-body sensors return n_total;
    image-display layers (viz_n set) return that override; others return n."""
    vn = getattr(obj, 'viz_n', None)
    if vn is not None:
        return vn
    nt = getattr(obj, 'n_total', None)
    return (nt if nt is not None else (obj.n or 0)) or 0


def _small_bold_font():
    f = QFont()
    f.setPointSize(7)
    f.setBold(True)
    return f


class _SplitHalf:
    """Stand-in for one half of a lateralized sensor (camera or joint-pair) in the network layout."""
    viz_layout = None

    def __init__(self, sensor, side):
        self.name       = f'{sensor.name}_{side}'
        self._viz_color = getattr(sensor, '_viz_color', '#888888')
        self._sensor      = sensor
        self._side        = side
        self.is_image     = hasattr(sensor, 'width')
        self.lateral_pair = f'{sensor.name}_{"R" if side == "L" else "L"}'
        if hasattr(sensor, 'width'):
            # Camera sensor: pixel-split half
            half = sensor.width // 2
            ovl  = getattr(sensor, 'overlap', 0)
            if side == 'L':
                self._px_start = 0
                self._px_end   = int(np.clip(half + ovl, 0, sensor.width))
            else:
                self._px_start = int(np.clip(half - ovl, 0, sensor.width))
                self._px_end   = sensor.width
            self.n       = 1
            self.n_total = 1
        else:
            # Non-camera lateralized sensor (joint-pair): each half has the sensor's n outputs
            self.n       = sensor.n or 1
            self.n_total = sensor.n or 1


def _sensor_is_lateralized(sensor, circuit=None):
    """Return True when sensor has a lateralized L/R structure.

    Covers two cases:
      1. CameraSensor with lateralized=True (explicit attribute)
      2. Any sensor with exactly 2 body_ids from the same mirror_group (joint-pair sensor)
    """
    if getattr(sensor, 'lateralized', False):
        return True
    body_ids = getattr(sensor, 'body_ids', ['root'])
    if len(body_ids) != 2:
        return False
    if circuit is None:
        return True   # no circuit info — assume pair
    body_map = {b.id: b for b in getattr(circuit, 'bodies', [])}
    b0 = body_map.get(body_ids[0])
    b1 = body_map.get(body_ids[1])
    if b0 is None or b1 is None:
        return False
    mg0 = getattr(b0, 'mirror_group', None)
    mg1 = getattr(b1, 'mirror_group', None)
    return bool(mg0 and mg0 == mg1)


# ============================================================
# HOVER STATUS FILTER
# ============================================================
class _HoverStatus(QObject):
    """Event filter that shows a description in a status QLabel on mouse enter/leave."""
    def __init__(self, label, desc, parent=None):
        super().__init__(parent)
        self._label = label
        self._desc  = desc

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Enter:
            self._label.setText(self._desc)
        elif event.type() == QEvent.Type.Leave:
            self._label.clear()
        return False


# ============================================================
# MATRIX HEATMAP WIDGET
# ============================================================
class _MatrixHeatmapWidget(QWidget):
    """Blue–white–red heatmap for a 2D weight matrix (nt rows × ns cols).

    Layout (all sizes in pixels):
      _LH  top margin  — source (column) axis labels
      _LW  left margin — target (row) axis labels
      _SW  right strip — coloured squares for row sums, labelled "Σ"
    """
    _MAX_CELL = 30
    _MIN_CELL = 5
    _LH = 16   # top label margin
    _LW = 20   # left label margin
    _SW = 12   # row-sum strip width
    _SG = 8    # gap between weight grid and sum strip

    def __init__(self, parent=None):
        super().__init__(parent)
        self._W    = np.zeros((1, 1))
        self._cell = self._MIN_CELL
        self.setMinimumSize(40, 40)
        self.setMouseTracking(True)

    def set_matrix(self, W):
        W = np.atleast_2d(np.asarray(W, dtype=float))
        self._W = W
        nt, ns = W.shape
        self._cell = max(self._MIN_CELL, min(self._MAX_CELL, 240 // max(nt, ns, 1)))
        c = self._cell
        self.setFixedSize(self._LW + ns * c + 2 + self._SG + self._SW + 2,
                          self._LH + nt * c + 2)
        self.update()

    def _cell_at(self, pt):
        """Return (row, col) for local QPoint pt, or None."""
        x = pt.x() - self._LW - 1
        y = pt.y() - self._LH - 1
        if x < 0 or y < 0:
            return None
        nt, ns = self._W.shape
        c = self._cell
        col, row = int(x // c), int(y // c)
        if 0 <= row < nt and 0 <= col < ns:
            return row, col
        return None

    def mouseMoveEvent(self, ev):
        pt = ev.position().toPoint()
        rc = self._cell_at(pt)
        if rc is not None:
            r, c = rc
            QToolTip.showText(self.mapToGlobal(pt),
                              f"[tgt {r}, src {c}] = {self._W[r, c]:.4g}", self)
        else:
            QToolTip.hideText()

    def _weight_color(self, v):
        """v in [–1, +1] → QColor (red positive, blue negative)."""
        if v >= 0:
            return QColor(255, int(255 * (1 - v)), int(255 * (1 - v)))
        return QColor(int(255 * (1 + v)), int(255 * (1 + v)), 255)

    def paintEvent(self, _ev):
        W = np.nan_to_num(self._W, nan=0.0, posinf=0.0, neginf=0.0)
        nt, ns = W.shape
        c    = self._cell
        vmax = max(float(np.abs(W).max()), 1e-9)
        lw, lh, sw = self._LW, self._LH, self._SW

        p = QPainter(self)
        p.setPen(Qt.NoPen)

        # ── weight cells ────────────────────────────────────────────────────
        for i in range(nt):
            for j in range(ns):
                p.fillRect(int(lw + 1 + j * c), int(lh + 1 + i * c),
                           max(1, c), max(1, c),
                           self._weight_color(float(W[i, j]) / vmax))

        # ── row-sum strip ───────────────────────────────────────────────────
        sums = W.sum(axis=1)
        smax = max(float(np.abs(sums).max()), 1e-9)
        sx = lw + 1 + ns * c + self._SG
        for i in range(nt):
            p.fillRect(sx, int(lh + 1 + i * c), sw, max(1, c),
                       self._weight_color(float(sums[i]) / smax))

        # ── axis labels ─────────────────────────────────────────────────────
        p.setPen(QColor(160, 160, 160))
        font = p.font()
        font.setPointSize(6)
        p.setFont(font)

        # top margin: "src" title + column indices
        p.drawText(lw + 1, 0, ns * c, lh - 1,
                   Qt.AlignHCenter | Qt.AlignBottom, "src →")
        step = max(1, ns // 8)
        for j in range(0, ns, step):
            p.drawText(int(lw + 1 + j * c), 0, c * step, lh - 1,
                       Qt.AlignHCenter | Qt.AlignBottom, str(j))

        # left margin: "tgt" title + row indices
        p.drawText(0, lh + 1, lw - 1, nt * c,
                   Qt.AlignRight | Qt.AlignVCenter, "tgt")
        step = max(1, nt // 8)
        for i in range(0, nt, step):
            p.drawText(0, int(lh + 1 + i * c), lw - 1, c * step,
                       Qt.AlignRight | Qt.AlignVCenter, str(i))

        # row-sum strip label
        p.drawText(sx, 0, sw, lh - 1, Qt.AlignHCenter | Qt.AlignBottom, "Σ")

        p.end()


# ============================================================
# WEIGHT ENTRY WIDGET  (one slot in the live weight panel)
# ============================================================
class WeightEntryWidget(QWidget):
    """Shows a label and a live-updating heatmap for one connection's weight matrix."""

    def __init__(self, src, tgt, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 4)
        lay.setSpacing(2)
        lbl = QLabel(f"{src}  →  {tgt}")
        lbl.setStyleSheet("font-size:9px;font-weight:bold;")
        lay.addWidget(lbl)
        self._heatmap = _MatrixHeatmapWidget()
        self._heatmap.setFixedHeight(100)
        lay.addWidget(self._heatmap)

    def set_matrix(self, W):
        self._heatmap.set_matrix(W)


def _tile_conv_filters(W):
    """(n_filters, in_ch, kH, kW) → 2-D tiled array for heatmap display."""
    n_filters, _in_ch, kH, kW = W.shape
    kernels = W.mean(axis=1)                     # (n_filters, kH, kW)
    cols = max(1, int(np.ceil(np.sqrt(n_filters))))
    rows = int(np.ceil(n_filters / cols))
    tile = np.zeros((rows * kH, cols * kW))
    for i, k in enumerate(kernels):
        r, c = divmod(i, cols)
        tile[r * kH:(r + 1) * kH, c * kW:(c + 1) * kW] = k
    return tile


# ============================================================
# WEIGHT MATRIX DIALOG
# ============================================================
class WeightMatrixDialog(QDialog):
    """Weight-matrix editor with pattern presets and live heatmap preview.

    Call exec(); on acceptance call get_result() for (W_final, params_out).
    W_final has shape (nt, ns). params_out is a dict of the last-used settings.
    """

    def __init__(self, parent, src_name, tgt_name, ns, nt, W_init,
                 circuit=None, saved_params=None, conv_params=None):
        super().__init__(parent)
        self.setWindowTitle(f"{src_name}  →  {tgt_name}  ({nt} × {ns})")
        self._ns = ns
        self._nt = nt

        W_init = np.atleast_2d(np.asarray(W_init, dtype=float))
        if W_init.shape != (nt, ns):
            W_init = np.zeros((nt, ns))

        main = QVBoxLayout(self)
        main.setSpacing(8)

        self._filters = []
        status_lbl = QLabel("")
        status_lbl.setFixedHeight(18)
        status_lbl.setStyleSheet("color:#888;font-style:italic;padding:0 4px;")

        def _tip(w, desc):
            f = _HoverStatus(status_lbl, desc, self)
            w.installEventFilter(f)
            self._filters.append(f)

        tgt_layer = next((l for l in circuit.layers if l.name == tgt_name), None) if circuit else None
        tgt_ht = getattr(type(tgt_layer), 'help_text', None) if tgt_layer else None

        info_row = QHBoxLayout()
        info_row.addWidget(QLabel(
            f"<span style='font-size:9px;color:{C['muted']}'>"
            f"Rows&nbsp;=&nbsp;<b>{tgt_name}</b>&nbsp;neurons&nbsp;(target)&emsp;"
            f"Cols&nbsp;=&nbsp;<b>{src_name}</b>&nbsp;neurons&nbsp;(source)&emsp;"
            f"+&nbsp;excitatory&emsp;−&nbsp;inhibitory</span>"
        ))
        if tgt_ht:
            ltype_name = type(tgt_layer).__name__
            help_btn = QPushButton("?")
            help_btn.setFixedSize(22, 22)
            help_btn.setToolTip(f"About {ltype_name}")
            help_btn.clicked.connect(lambda: QMessageBox.information(self, ltype_name, tgt_ht))
            info_row.addWidget(help_btn)
        main.addLayout(info_row)

        # ── Conv filter presets (only for ConvLayer connections) ──────────────
        if conv_params:
            n_flt = conv_params['n_filters']
            in_ch = conv_params['in_ch']
            ksz   = conv_params['kernel_size']

            def _make_preset_btn(label, fn, nf=n_flt, ic=in_ch, k=ksz):
                btn = QPushButton(label)
                btn.setFixedHeight(22)
                btn.setStyleSheet(
                    "QPushButton{padding:0 6px;font-size:8px;"
                    "border:1px solid #C8A830;border-radius:3px;background:#FFFAE0;}"
                    "QPushButton:hover{background:#FFF0A0;}"
                )
                btn.clicked.connect(lambda: self._apply_conv_preset(fn(), nf, ic, k))
                return btn

            def _grayscale(nf=n_flt, ic=in_ch, k=ksz):
                W = np.zeros((nf, ic, k)); ck = k // 2
                for f in range(nf):
                    for c in range(ic):
                        W[f, c, ck] = 1.0 / max(ic, 1)
                return W

            def _lum(nf=n_flt, ic=in_ch, k=ksz):
                W = np.zeros((nf, ic, k)); ck = k // 2
                for f in range(nf):
                    for c in range(ic):
                        W[f, c, ck] = 1.0 / max(ic, 1)
                return W

            def _rg(nf=n_flt, ic=in_ch, k=ksz):
                W = np.zeros((nf, ic, k)); ck = k // 2
                for f in range(nf):
                    W[f, 0, ck] = 1.0
                    if ic >= 2: W[f, 1, ck] = -1.0
                return W

            def _gr(nf=n_flt, ic=in_ch, k=ksz):
                W = np.zeros((nf, ic, k)); ck = k // 2
                for f in range(nf):
                    if ic >= 2: W[f, 1, ck] = 1.0
                    W[f, 0, ck] = -1.0
                return W

            def _rb(nf=n_flt, ic=in_ch, k=ksz):
                W = np.zeros((nf, ic, k)); ck = k // 2
                for f in range(nf):
                    W[f, 0, ck] = 1.0
                    if ic >= 3: W[f, 2, ck] = -1.0
                return W

            def _on_centre(nf=n_flt, ic=in_ch, k=ksz):
                kx = np.arange(k, dtype=float) - k // 2
                se = max(k / 6.0, 0.5); si = max(k / 3.0, 1.0)
                sp = 2.0 * np.exp(-kx**2 / (2 * se**2)) - np.exp(-kx**2 / (2 * si**2))
                mx = np.abs(sp).max()
                sp = sp / mx if mx > 0 else sp
                W = np.zeros((nf, ic, k))
                for f in range(nf):
                    for c in range(ic):
                        W[f, c, :] = sp / max(ic, 1)
                return W

            def _off_centre(nf=n_flt, ic=in_ch, k=ksz):
                return -_on_centre(nf, ic, k)

            def _edge_lr(nf=n_flt, ic=in_ch, k=ksz):
                W = np.zeros((nf, ic, k)); ck = k // 2
                for f in range(nf):
                    for c in range(ic):
                        if k >= 3:
                            W[f, c, ck - 1] = -1.0 / max(ic, 1)
                            W[f, c, ck + 1] =  1.0 / max(ic, 1)
                        elif k >= 2:
                            W[f, c, 0] = -1.0 / max(ic, 1)
                            W[f, c, 1] =  1.0 / max(ic, 1)
                return W

            def _edge_rl(nf=n_flt, ic=in_ch, k=ksz):
                return -_edge_lr(nf, ic, k)

            def _rgb(nf=n_flt, ic=in_ch, k=ksz):
                W = np.zeros((nf, ic, k)); ck = k // 2
                for f in range(min(nf, ic)):
                    W[f, f, ck] = 1.0
                return W

            preset_gb = QGroupBox("Filter presets")
            preset_vl = QVBoxLayout(preset_gb)
            preset_vl.setSpacing(4)
            preset_vl.setContentsMargins(6, 4, 6, 4)
            row1 = QHBoxLayout(); row1.setSpacing(4)
            row2 = QHBoxLayout(); row2.setSpacing(4)
            row3 = QHBoxLayout(); row3.setSpacing(4)
            row1.addWidget(_make_preset_btn("Grayscale", _grayscale))
            row1.addWidget(_make_preset_btn("Luminance",   _lum))
            row1.addWidget(_make_preset_btn("R−G",    _rg))
            row1.addWidget(_make_preset_btn("G−R",    _gr))
            row1.addWidget(_make_preset_btn("R−B",    _rb))
            row2.addWidget(_make_preset_btn("On-centre",   _on_centre))
            row2.addWidget(_make_preset_btn("Off-centre",  _off_centre))
            row2.addWidget(_make_preset_btn("Edge →", _edge_lr))
            row2.addWidget(_make_preset_btn("Edge ←", _edge_rl))
            row3.addWidget(_make_preset_btn("RGB channels", _rgb))
            row3.addStretch()
            preset_vl.addLayout(row1)
            preset_vl.addLayout(row2)
            preset_vl.addLayout(row3)
            preset_vl.addWidget(QLabel(
                f"<span style='font-size:8px;color:{C['muted']}'>"
                f"{n_flt} filter{'s' if n_flt != 1 else ''}"
                f" × {in_ch} ch × kernel {ksz}"
                f"</span>"))
            main.addWidget(preset_gb)

        # Pattern controls (left) + heatmap (right)
        mid = QHBoxLayout()
        mid.setSpacing(14)

        ctrl_gb = QGroupBox("Pattern")
        ctrl_vl = QVBoxLayout(ctrl_gb)
        ctrl_vl.setSpacing(6)

        self._pattern_cb = QComboBox()
        self._pattern_cb.addItems(['Uniform', 'Cosine', 'Gaussian', 'Mexican hat',
                                   'One-to-one', 'Rand uniform', 'Rand normal',
                                   'Expression', 'Manual'])
        ctrl_vl.addWidget(self._pattern_cb)

        stack = QStackedWidget()

        # Uniform
        unif_w = QWidget(); unif_f = QFormLayout(unif_w); unif_f.setContentsMargins(0, 4, 0, 0)
        self._unif_amp = QDoubleSpinBox(); self._unif_amp.setRange(-100, 100); self._unif_amp.setSingleStep(0.1); self._unif_amp.setValue(1.0)
        unif_f.addRow("Amplitude", self._unif_amp)
        _tip(self._unif_amp, "Uniform weight applied to all connections")
        stack.addWidget(unif_w)

        # Cosine
        cos_w = QWidget(); cos_f = QFormLayout(cos_w); cos_f.setContentsMargins(0, 4, 0, 0)
        self._cos_amp  = QDoubleSpinBox(); self._cos_amp.setRange(-100, 100); self._cos_amp.setSingleStep(0.1); self._cos_amp.setValue(1.0)
        self._cos_ph0  = QDoubleSpinBox(); self._cos_ph0.setRange(-360, 360); self._cos_ph0.setSingleStep(5); self._cos_ph0.setValue(0); self._cos_ph0.setSuffix(" °")
        self._cos_step = QDoubleSpinBox(); self._cos_step.setRange(-360, 360); self._cos_step.setSingleStep(5)
        self._cos_step.setValue(round(360.0 / nt, 1) if nt > 1 else 180.0); self._cos_step.setSuffix(" °")
        cos_f.addRow("Amplitude",  self._cos_amp)
        _tip(self._cos_amp, "Peak amplitude of the cosine wave")
        cos_f.addRow("Phase₀",    self._cos_ph0)
        _tip(self._cos_ph0, "Phase offset for target neuron 0 (degrees)")
        cos_f.addRow("Phase step", self._cos_step)
        _tip(self._cos_step, "Phase increment per target neuron (degrees)")
        cos_f.addRow(QLabel(f"<span style='font-size:8px;color:{C['muted']}'>"
                            f"W[i,j] = amp·cos(2π·j/{ns} + phase₀ + i·step)</span>"))
        stack.addWidget(cos_w)

        # Gaussian
        gau_w = QWidget(); gau_f = QFormLayout(gau_w); gau_f.setContentsMargins(0, 4, 0, 0)
        self._gau_amp  = QDoubleSpinBox(); self._gau_amp.setRange(-100, 100); self._gau_amp.setSingleStep(0.1);  self._gau_amp.setValue(1.0)
        self._gau_sig  = QDoubleSpinBox(); self._gau_sig.setRange(0.01, 2.0); self._gau_sig.setSingleStep(0.05); self._gau_sig.setValue(0.2); self._gau_sig.setDecimals(3)
        self._gau_off  = QDoubleSpinBox(); self._gau_off.setRange(-1.0, 1.0); self._gau_off.setSingleStep(0.05); self._gau_off.setValue(0.0); self._gau_off.setDecimals(3)
        self._gau_base = QDoubleSpinBox(); self._gau_base.setRange(-100, 100); self._gau_base.setSingleStep(0.1); self._gau_base.setValue(0.0)
        gau_f.addRow("Amplitude", self._gau_amp)
        _tip(self._gau_amp, "Peak amplitude of the Gaussian")
        gau_f.addRow("Sigma", self._gau_sig)
        _tip(self._gau_sig, "Width of the Gaussian — larger = broader spread")
        gau_f.addRow("Peak shift", self._gau_off)
        _tip(self._gau_off, "Offset the Gaussian peak along the source axis (wraps around)")
        gau_f.addRow("Baseline", self._gau_base)
        _tip(self._gau_base, "Constant added to all weights — negative creates lateral inhibition")
        gau_f.addRow(QLabel(f"<span style='font-size:8px;color:{C['muted']}'>"
                            f"W[i,j] = amp·exp(−dist²/2σ²) + baseline</span>"))
        stack.addWidget(gau_w)

        # Mexican hat
        _nn_spacing = round(1.0 / max(ns, nt), 3) if max(ns, nt) > 0 else 0.125
        mxh_w = QWidget(); mxh_f = QFormLayout(mxh_w); mxh_f.setContentsMargins(0, 4, 0, 0)
        self._mxh_exc  = QDoubleSpinBox(); self._mxh_exc.setRange(0, 100);  self._mxh_exc.setSingleStep(0.1); self._mxh_exc.setValue(2.0)
        self._mxh_sige = QDoubleSpinBox(); self._mxh_sige.setRange(0.01, 1.0); self._mxh_sige.setSingleStep(0.01); self._mxh_sige.setValue(round(_nn_spacing * 1.4, 3)); self._mxh_sige.setDecimals(3)
        self._mxh_inh  = QDoubleSpinBox(); self._mxh_inh.setRange(0, 100);  self._mxh_inh.setSingleStep(0.1); self._mxh_inh.setValue(1.0)
        self._mxh_sigi = QDoubleSpinBox(); self._mxh_sigi.setRange(0.01, 2.0); self._mxh_sigi.setSingleStep(0.01); self._mxh_sigi.setValue(round(_nn_spacing * 3.0, 3)); self._mxh_sigi.setDecimals(3)
        mxh_f.addRow("Exc. amplitude",    self._mxh_exc)
        _tip(self._mxh_exc, "Excitatory peak amplitude — drives near-neighbour excitation")
        mxh_f.addRow("σ_exc (ring frac.)", self._mxh_sige)
        _tip(self._mxh_sige,
             f"Excitatory width in ring-fraction units [0,1]. "
             f"Must exceed the nearest-neighbour spacing ({_nn_spacing}) "
             f"to produce genuine local excitation. "
             f"For a single bump, σ_exc ≥ 1.0 – 1.5 × spacing.")
        mxh_f.addRow("Inh. amplitude",    self._mxh_inh)
        _tip(self._mxh_inh, "Inhibitory peak amplitude — suppresses distant neurons")
        mxh_f.addRow("σ_inh (ring frac.)", self._mxh_sigi)
        _tip(self._mxh_sigi,
             "Inhibitory width — must be larger than σ_exc to create the Mexican-hat profile. "
             "If σ_inh ≤ σ_exc the kernel is purely excitatory and no bump forms.")
        mxh_f.addRow(QLabel(f"<span style='font-size:8px;color:{C['muted']}'>"
                            f"W[i,j] = w_e·exp(−d²/2σ_e²) − w_i·exp(−d²/2σ_i²)  ·  diag=0</span>"))
        mxh_f.addRow(QLabel(f"<span style='font-size:8px;color:{C['muted']}'>"
                            f"Nearest-neighbour spacing: {_nn_spacing}  "
                            f"(= 1/{max(ns, nt)} ring fractions)</span>"))
        stack.addWidget(mxh_w)

        # One-to-one
        oto_w = QWidget(); oto_f = QFormLayout(oto_w); oto_f.setContentsMargins(0, 4, 0, 0)
        self._oto_amp = QDoubleSpinBox(); self._oto_amp.setRange(-100, 100); self._oto_amp.setSingleStep(0.1); self._oto_amp.setValue(1.0)
        self._oto_off = QSpinBox(); self._oto_off.setRange(-max(ns, nt), max(ns, nt)); self._oto_off.setValue(0)
        oto_f.addRow("Amplitude",   self._oto_amp)
        _tip(self._oto_amp, "Weight of each one-to-one pairing")
        oto_f.addRow("Offset (src)", self._oto_off)
        _tip(self._oto_off, "Shift source pairing index by this many neurons")
        oto_f.addRow(QLabel(f"<span style='font-size:8px;color:{C['muted']}'>"
                            f"W[i,j] = amp if j==round(i·{ns}/{nt}+off)%{ns}</span>"))
        stack.addWidget(oto_w)

        # Rand uniform
        rnd_unif_w = QWidget(); rnd_unif_f = QFormLayout(rnd_unif_w); rnd_unif_f.setContentsMargins(0, 4, 0, 0)
        self._rnd_unif_amp = QDoubleSpinBox(); self._rnd_unif_amp.setRange(0.001, 100); self._rnd_unif_amp.setSingleStep(0.1); self._rnd_unif_amp.setValue(1.0)
        rnd_unif_f.addRow("Amplitude", self._rnd_unif_amp)
        _tip(self._rnd_unif_amp, "Half-range of uniform distribution: W[i,j] ~ U(−amp, +amp)")
        rnd_unif_f.addRow(QLabel(f"<span style='font-size:8px;color:{C['muted']}'>"
                                 f"W[i,j] ~ U(−amp, +amp)</span>"))
        stack.addWidget(rnd_unif_w)

        # Rand normal
        rnd_norm_w = QWidget(); rnd_norm_f = QFormLayout(rnd_norm_w); rnd_norm_f.setContentsMargins(0, 4, 0, 0)
        self._rnd_norm_std = QDoubleSpinBox(); self._rnd_norm_std.setRange(0.001, 100); self._rnd_norm_std.setSingleStep(0.1); self._rnd_norm_std.setValue(0.1)
        rnd_norm_f.addRow("Std dev", self._rnd_norm_std)
        _tip(self._rnd_norm_std, "Standard deviation of normal distribution: W[i,j] ~ N(0, std)")
        rnd_norm_f.addRow(QLabel(f"<span style='font-size:8px;color:{C['muted']}'>"
                                 f"W[i,j] ~ N(0, std)</span>"))
        stack.addWidget(rnd_norm_w)

        # Expression
        expr_w = QWidget(); expr_vl = QVBoxLayout(expr_w); expr_vl.setContentsMargins(0, 4, 0, 0)
        self._expr_edit = QTextEdit()
        self._expr_edit.setFont(QFont('Courier New', 9))
        self._expr_edit.setFixedHeight(100)
        self._expr_edit.setPlaceholderText(
            f"# Assign result to W  —  shape must be ({nt}, {ns})\n"
            f"# Variables: nt={nt}  ns={ns}  np\n"
            f"# i = np.arange(nt)[:,None]   j = np.arange(ns)[None,:]\n"
            f"W = np.cos(2*np.pi*j/ns + np.pi*i/nt)")
        _tip(self._expr_edit, "Python expression — assign result to W; shape must be (nt, ns)")
        expr_vl.addWidget(self._expr_edit)
        expr_vl.addWidget(QLabel(f"<span style='font-size:8px;color:{C['muted']}'>"
                                 f"i = arange(nt)[:,None] &nbsp; j = arange(ns)[None,:]</span>"))
        stack.addWidget(expr_w)

        # Manual
        manual_w = QWidget(); manual_vl = QVBoxLayout(manual_w); manual_vl.setContentsMargins(0, 4, 0, 0)
        manual_lbl = QLabel(
            "<span style='font-size:9px;color:#888'>"
            "Current weights are preserved as-is.<br>"
            "Click <b>OK</b> to keep them, or choose another<br>"
            "pattern and click <b>Apply ▶</b> to regenerate.</span>")
        manual_lbl.setWordWrap(True)
        manual_vl.addWidget(manual_lbl)
        manual_vl.addStretch()
        stack.addWidget(manual_w)

        ctrl_vl.addWidget(stack)

        apply_btn = QPushButton("Apply ▶")
        apply_btn.setDefault(False)
        apply_btn.setAutoDefault(False)
        ctrl_vl.addWidget(apply_btn)
        ctrl_vl.addStretch()
        mid.addWidget(ctrl_gb, stretch=0)

        # Heatmap
        hmap_vl = QVBoxLayout()
        hmap_vl.setSpacing(2)
        hmap_vl.addWidget(QLabel("<span style='font-size:8px;color:gray'>Preview</span>"),
                          alignment=Qt.AlignHCenter)
        self._hmap = _MatrixHeatmapWidget()
        self._hmap.set_matrix(W_init)
        hmap_vl.addWidget(self._hmap, alignment=Qt.AlignHCenter)
        hmap_vl.addStretch()
        mid.addLayout(hmap_vl, stretch=1)
        main.addLayout(mid)

        # Raw table (collapsible)
        table_gb = QGroupBox("Raw matrix")
        table_gb.setCheckable(True)
        table_gb.setChecked(max(nt, ns) <= 6)
        table_vl = QVBoxLayout(table_gb)
        table_vl.setContentsMargins(4, 4, 4, 4)

        self._table = QTableWidget(nt, ns)
        self._table.setHorizontalHeaderLabels([f'{src_name}_{j}' for j in range(ns)])
        self._table.setVerticalHeaderLabels([f'{tgt_name}_{i}' for i in range(nt)])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i in range(nt):
            for j in range(ns):
                it = QTableWidgetItem(f'{float(W_init[i, j]):g}')
                it.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(i, j, it)
        self._table.setFixedHeight(min(280, max(80, nt * 30 + 30)))
        table_vl.addWidget(self._table)
        self._table.setVisible(table_gb.isChecked())
        table_gb.toggled.connect(self._table.setVisible)
        main.addWidget(table_gb)

        main.addWidget(status_lbl)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        main.addWidget(bb)

        apply_btn.clicked.connect(self._apply)
        self._pattern_cb.currentIndexChanged.connect(stack.setCurrentIndex)

        # Pre-populate from saved_params
        if saved_params:
            p = saved_params
            u = p.get('uniform', {})
            if 'amp' in u: self._unif_amp.setValue(u['amp'])
            c_ = p.get('cosine', {})
            if 'amp'  in c_: self._cos_amp.setValue(c_['amp'])
            if 'ph0'  in c_: self._cos_ph0.setValue(c_['ph0'])
            if 'step' in c_: self._cos_step.setValue(c_['step'])
            g = p.get('gaussian', {})
            if 'amp'  in g: self._gau_amp.setValue(g['amp'])
            if 'sig'  in g: self._gau_sig.setValue(g['sig'])
            if 'off'  in g: self._gau_off.setValue(g['off'])
            if 'base' in g: self._gau_base.setValue(g['base'])
            m = p.get('mexican_hat', {})
            if 'exc'  in m: self._mxh_exc.setValue(m['exc'])
            if 'sige' in m: self._mxh_sige.setValue(m['sige'])
            if 'inh'  in m: self._mxh_inh.setValue(m['inh'])
            if 'sigi' in m: self._mxh_sigi.setValue(m['sigi'])
            o = p.get('one_to_one', {})
            if 'amp' in o: self._oto_amp.setValue(o['amp'])
            if 'off' in o: self._oto_off.setValue(o['off'])
            ru = p.get('rand_uniform', {})
            if 'amp' in ru: self._rnd_unif_amp.setValue(ru['amp'])
            rn = p.get('rand_normal', {})
            if 'std' in rn: self._rnd_norm_std.setValue(rn['std'])
            e = p.get('expression', {})
            if 'code' in e: self._expr_edit.setPlainText(e['code'])
            self._pattern_cb.setCurrentIndex(p.get('type', 8))
        else:
            self._pattern_cb.setCurrentIndex(8)

        self.resize(max(520, ns * 35 + 320), 520)

    def _compute_W(self):
        ns, nt = self._ns, self._nt
        idx = self._pattern_cb.currentIndex()
        if idx == 8:
            return None
        try:
            if idx == 0:
                W = self._unif_amp.value() * np.ones((nt, ns))
            elif idx == 1:
                th_s = 2 * np.pi * np.arange(ns) / ns
                ph_t = np.deg2rad(self._cos_ph0.value() + np.arange(nt) * self._cos_step.value())
                W = self._cos_amp.value() * np.cos(th_s[None, :] + ph_t[:, None])
            elif idx == 2:
                js   = np.arange(ns) / ns
                is_  = np.arange(nt) / nt + self._gau_off.value()
                dist = np.abs(is_[:, None] - js[None, :])
                dist = np.minimum(dist, 1 - dist)
                sig  = max(self._gau_sig.value(), 1e-9)
                W = self._gau_amp.value() * np.exp(-dist**2 / (2 * sig**2)) + self._gau_base.value()
            elif idx == 3:
                js   = np.arange(ns) / ns
                is_  = np.arange(nt) / nt
                dist = np.abs(is_[:, None] - js[None, :])
                dist = np.minimum(dist, 1 - dist)
                se   = max(self._mxh_sige.value(), 1e-9)
                si   = max(self._mxh_sigi.value(), 1e-9)
                W = (self._mxh_exc.value() * np.exp(-dist**2 / (2 * se**2))
                     - self._mxh_inh.value() * np.exp(-dist**2 / (2 * si**2)))
                if nt == ns:
                    np.fill_diagonal(W, 0.0)
            elif idx == 4:
                W = np.zeros((nt, ns))
                off = self._oto_off.value()
                for ii in range(nt):
                    jj = int(round(ii * ns / nt + off)) % ns
                    W[ii, jj] = self._oto_amp.value()
            elif idx == 5:
                W = self._rnd_unif_amp.value() * np.random.uniform(-1, 1, (nt, ns))
            elif idx == 6:
                W = self._rnd_norm_std.value() * np.random.randn(nt, ns)
            else:
                code = self._expr_edit.toPlainText().strip()
                env  = {'np': np, 'nt': nt, 'ns': ns,
                        'i': np.arange(nt)[:, None],
                        'j': np.arange(ns)[None, :]}
                exec(code, env)  # noqa: S102
                W = np.asarray(env.get('W', 0), dtype=float)
                if W.shape != (nt, ns):
                    raise ValueError(f"Result shape {W.shape} ≠ ({nt}, {ns})")
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc))
            return None
        return W

    def _apply(self):
        W = self._compute_W()
        if W is None:
            return
        self._hmap.set_matrix(W)
        ns, nt = self._ns, self._nt
        for ii in range(nt):
            for jj in range(ns):
                it = self._table.item(ii, jj)
                if it is None:
                    it = QTableWidgetItem()
                    self._table.setItem(ii, jj, it)
                it.setText(f'{W[ii, jj]:g}')

    def accept(self):
        """Auto-apply the selected pattern before accepting, so OK without Apply works."""
        if self._pattern_cb.currentIndex() != 8:  # 8 = Manual (preserve table as-is)
            self._apply()
        super().accept()

    def get_result(self):
        """Return (W_final, params_out) after a successful exec()."""
        ns, nt = self._ns, self._nt
        W_final = np.zeros((nt, ns))
        for ii in range(nt):
            for jj in range(ns):
                it = self._table.item(ii, jj)
                try:
                    W_final[ii, jj] = float(it.text()) if it else 0.0
                except ValueError:
                    pass
        params_out = {
            'type':        self._pattern_cb.currentIndex(),
            'uniform':     {'amp': self._unif_amp.value()},
            'cosine':      {'amp': self._cos_amp.value(), 'ph0': self._cos_ph0.value(),
                            'step': self._cos_step.value()},
            'gaussian':    {'amp': self._gau_amp.value(), 'sig': self._gau_sig.value(),
                            'off': self._gau_off.value(), 'base': self._gau_base.value()},
            'mexican_hat': {'exc': self._mxh_exc.value(), 'sige': self._mxh_sige.value(),
                            'inh': self._mxh_inh.value(), 'sigi': self._mxh_sigi.value()},
            'one_to_one':  {'amp': self._oto_amp.value(), 'off': self._oto_off.value()},
            'rand_uniform': {'amp': self._rnd_unif_amp.value()},
            'rand_normal':  {'std': self._rnd_norm_std.value()},
            'expression':  {'code': self._expr_edit.toPlainText()},
        }
        return W_final, params_out

    def _apply_conv_preset(self, W_3d, n_flt, in_ch, ksz):
        W_flat = np.asarray(W_3d, dtype=float).reshape(n_flt, in_ch * ksz)
        self._hmap.set_matrix(W_flat)
        for ii in range(n_flt):
            for jj in range(in_ch * ksz):
                it = self._table.item(ii, jj)
                if it is None:
                    it = QTableWidgetItem()
                    self._table.setItem(ii, jj, it)
                it.setText(f'{W_flat[ii, jj]:g}')
        self._pattern_cb.setCurrentIndex(8)  # switch to Manual to preserve preset


# ============================================================
# FILTER STACK DIALOG  (Conv2dLayer connections)
# ============================================================

def _mirror_name(name):
    """Swap _L ↔ _R suffix. Returns None if name has neither suffix."""
    if name.endswith('_L'):
        return name[:-2] + '_R'
    if name.endswith('_R'):
        return name[:-2] + '_L'
    return None


def _make_conv2d_filter(name, in_ch, kH, kW):
    """Generate a (in_ch, kH, kW) kernel array for a named preset."""
    W  = np.zeros((in_ch, kH, kW), dtype=float)
    ch = kH // 2
    cw = kW // 2  # centre pixel

    if name == 'Grayscale':
        # True channel average: output = (R+G+B)/3 per pixel. No zero-sum so
        # absolute luminance is preserved — useful with pool='none' to convert
        # an RGB camera to a grayscale spatial map.
        W[:, ch, cw] = 1.0 / max(in_ch, 1)
        return W   # skip zero-sum enforcement

    elif name == 'Luminance':
        W[:, ch, cw] = 1.0 / max(in_ch, 1)

    elif name == 'R−G' and in_ch >= 2:
        W[0, ch, cw] =  1.0
        W[1, ch, cw] = -1.0

    elif name == 'G−R' and in_ch >= 2:
        W[0, ch, cw] = -1.0
        W[1, ch, cw] =  1.0

    elif name == 'R−B' and in_ch >= 3:
        W[0, ch, cw] =  1.0
        W[2, ch, cw] = -1.0

    elif name == 'B−R' and in_ch >= 3:
        W[0, ch, cw] = -1.0
        W[2, ch, cw] =  1.0

    elif name in ('On-centre', 'Off-centre'):
        kx = np.arange(kW, dtype=float) - cw
        ky = np.arange(kH, dtype=float) - ch
        se = max(min(kH, kW) / 4.0, 0.5)
        si = max(min(kH, kW) / 2.0, 1.0)
        xx, yy = np.meshgrid(kx, ky)
        d2 = xx**2 + yy**2
        sp = 2.0 * np.exp(-d2 / (2 * se**2)) - np.exp(-d2 / (2 * si**2))
        mx = np.abs(sp).max()
        sp = sp / mx if mx > 0 else sp
        if name == 'Off-centre':
            sp = -sp
        for c in range(in_ch):
            W[c] = sp / max(in_ch, 1)

    elif name in ('Edge →', 'Edge ←', 'Edge ↑', 'Edge ↓'):
        if name in ('Edge →', 'Edge ←'):
            row = np.zeros(kW)
            if kW >= 3:
                row[cw - 1] = -1.0 / max(in_ch, 1)
                row[cw + 1] =  1.0 / max(in_ch, 1)
            if name == 'Edge ←':
                row = -row
            for c in range(in_ch):
                W[c] = row[np.newaxis, :]
        else:
            col = np.zeros(kH)
            if kH >= 3:
                col[ch - 1] = -1.0 / max(in_ch, 1)
                col[ch + 1] =  1.0 / max(in_ch, 1)
            if name == 'Edge ↓':
                col = -col
            for c in range(in_ch):
                W[c] = col[:, np.newaxis]

    elif name.startswith('Ch ') and in_ch > 1:
        idx = int(name.split()[1])
        if idx < in_ch:
            W[idx, ch, cw] = 1.0

    # Enforce zero-sum: subtract global mean so uniform input gives zero output.
    # No-op for filters that already sum to zero (chromatic, edge).
    W -= W.mean()
    return W


class FilterStackDialog(QDialog):
    """Filter stack editor for Conv2dLayer connections.

    Left panel: clickable preset chips.
    Right panel: ordered filter list (drag to reorder, click × to remove).
    Bottom: heatmap preview of the selected filter.

    Returns weight tensor (n_filters, in_ch, kH, kW) on acceptance.
    """

    def __init__(self, parent, src_name, tgt_name, in_ch, kH, kW,
                 W_init=None, lateralized_half=False):
        super().__init__(parent)
        self.setWindowTitle(f"{src_name}  →  {tgt_name}  |  Filter Stack")
        self._in_ch = in_ch
        self._kH    = kH
        self._kW    = kW

        # Filter stack: list of {'name': str, 'kernel': (in_ch, kH, kW)}
        self._stack = []
        if W_init is not None and W_init.ndim == 4:
            for i, k in enumerate(W_init):
                self._stack.append({'name': f'Filter {i}', 'kernel': k.copy()})

        main = QVBoxLayout(self)
        main.setSpacing(8)

        # Info row
        ch_label = 'RGB' if in_ch == 3 else ('Gray' if in_ch == 1 else f'{in_ch}ch')
        main.addWidget(QLabel(
            f"<span style='font-size:9px;color:{C['muted']}'>"
            f"Camera: {ch_label} &nbsp;|&nbsp; Kernel: {kH}×{kW} &nbsp;|&nbsp; "
            f"Each filter outputs one scalar (global avg pool)</span>"))

        if lateralized_half:
            main.addWidget(QLabel(
                f"<span style='font-size:9px;color:#A06010'>"
                f"Lateralized camera — define n/2 filters only. "
                f"The same filters are mirrored on the opposite half "
                f"(layout: L₀ L₁ … R₁ R₀).</span>"))

        body = QHBoxLayout()
        body.setSpacing(12)

        # ── Left: preset chips ────────────────────────────────────────────
        left_gb = QGroupBox("Presets")
        left_vl = QVBoxLayout(left_gb)
        left_vl.setSpacing(4)

        presets = ['Grayscale', 'Luminance', 'On-centre', 'Off-centre',
                   'Edge →', 'Edge ←', 'Edge ↑', 'Edge ↓']
        if in_ch >= 2:
            presets += ['R−G', 'G−R']
        if in_ch >= 3:
            presets += ['R−B', 'B−R']
        for i in range(in_ch):
            presets.append(f'Ch {i}')
        presets.append('── Custom ──')

        for label in presets:
            if label.startswith('──'):
                sep = QLabel(label)
                sep.setStyleSheet(f"color:{C['muted']};font-size:8px;")
                left_vl.addWidget(sep)
                continue
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            btn.setStyleSheet(
                "QPushButton{padding:0 6px;font-size:8px;"
                "border:1px solid #C8A830;border-radius:3px;background:#FFFAE0;}"
                "QPushButton:hover{background:#FFF0A0;}"
            )
            btn.clicked.connect(lambda checked, n=label: self._add_preset(n))
            left_vl.addWidget(btn)

        code_btn = QPushButton("{ } Code filter…")
        code_btn.setFixedHeight(22)
        code_btn.setStyleSheet(
            "QPushButton{padding:0 6px;font-size:8px;"
            "border:1px solid #8888C8;border-radius:3px;background:#F0F0FF;}"
            "QPushButton:hover{background:#E0E0FF;}"
        )
        code_btn.clicked.connect(self._add_code_filter)
        left_vl.addWidget(code_btn)
        left_vl.addStretch()
        body.addWidget(left_gb, stretch=0)

        # ── Right: filter stack ───────────────────────────────────────────
        right_gb  = QGroupBox("Filter stack (drag to reorder)")
        right_vl  = QVBoxLayout(right_gb)
        right_vl.setSpacing(4)

        self._list = QListWidget()
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setMinimumWidth(180)
        self._list.currentRowChanged.connect(self._on_selection)
        right_vl.addWidget(self._list)

        rm_btn = QPushButton("Remove selected")
        rm_btn.setFixedHeight(22)
        rm_btn.clicked.connect(self._remove_selected)
        right_vl.addWidget(rm_btn)
        body.addWidget(right_gb, stretch=1)

        # ── Preview heatmap ───────────────────────────────────────────────
        prev_vl = QVBoxLayout()
        prev_vl.addWidget(QLabel("<span style='font-size:8px;color:gray'>Preview (ch 0)</span>"),
                          alignment=Qt.AlignHCenter)
        self._hmap = _MatrixHeatmapWidget()
        self._hmap.set_matrix(np.zeros((kH, kW)))
        prev_vl.addWidget(self._hmap, alignment=Qt.AlignHCenter)
        prev_vl.addStretch()
        body.addLayout(prev_vl)

        main.addLayout(body)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        main.addWidget(bb)

        self._rebuild_list()
        self.resize(620, 420)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _rebuild_list(self):
        self._list.clear()
        for entry in self._stack:
            self._list.addItem(entry['name'])

    def _on_selection(self, row):
        if 0 <= row < len(self._stack):
            k = self._stack[row]['kernel']
            self._hmap.set_matrix(k[0])   # preview channel 0

    def _add_preset(self, name):
        kernel = _make_conv2d_filter(name, self._in_ch, self._kH, self._kW)
        self._stack.append({'name': name, 'kernel': kernel})
        self._list.addItem(name)
        self._list.setCurrentRow(len(self._stack) - 1)

    def _add_code_filter(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Code filter")
        vl  = QVBoxLayout(dlg)
        in_ch, kH, kW = self._in_ch, self._kH, self._kW
        vl.addWidget(QLabel(
            f"<span style='font-size:9px'>Assign a ({in_ch}, {kH}, {kW}) array to <b>W</b>.<br>"
            f"Variables: <tt>np, in_ch={in_ch}, kH={kH}, kW={kW}, "
            f"ch={kH//2}, cw={kW//2}</tt></span>"))
        editor = QTextEdit()
        editor.setFont(QFont('Courier New', 9))
        editor.setFixedHeight(120)
        editor.setPlainText(
            f"# example: Gabor-like filter\n"
            f"kx = np.arange(kW) - cw\n"
            f"ky = np.arange(kH) - ch\n"
            f"xx, yy = np.meshgrid(kx, ky)\n"
            f"W = np.cos(xx) * np.exp(-(xx**2 + yy**2) / 4)\n"
            f"W = np.stack([W / in_ch] * in_ch)"
        )
        vl.addWidget(editor)
        name_edit = QLineEdit("Custom")
        vl.addWidget(QLabel("Filter name:"))
        vl.addWidget(name_edit)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        vl.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        code = editor.toPlainText().strip()
        env  = {'np': np, 'in_ch': in_ch, 'kH': kH, 'kW': kW,
                'ch': kH // 2, 'cw': kW // 2}
        try:
            exec(code, env)  # noqa: S102
            kernel = np.asarray(env.get('W'), dtype=float)
            if kernel.shape != (in_ch, kH, kW):
                raise ValueError(f"W shape must be ({in_ch}, {kH}, {kW}), got {kernel.shape}")
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc))
            return
        name = name_edit.text().strip() or 'Custom'
        self._stack.append({'name': name, 'kernel': kernel})
        self._list.addItem(name)
        self._list.setCurrentRow(len(self._stack) - 1)

    def _remove_selected(self):
        row = self._list.currentRow()
        if 0 <= row < len(self._stack):
            self._stack.pop(row)
            self._list.takeItem(row)

    def get_result(self):
        """Return (W, True) where W is (n_filters, in_ch, kH, kW), or (None, False)."""
        # Sync list order → stack order (drag-drop may have reordered list)
        new_order = [self._list.item(i).text() for i in range(self._list.count())]
        name_to_entry = {e['name']: e for e in self._stack}
        self._stack = [name_to_entry.get(n, self._stack[i])
                       for i, n in enumerate(new_order)]
        if not self._stack:
            QMessageBox.warning(self, "Empty stack", "Add at least one filter.")
            return None, False
        W = np.stack([e['kernel'] for e in self._stack])   # (n_filters, in_ch, kH, kW)
        return W, True

    def accept(self):
        _, ok = self.get_result()
        if ok:
            super().accept()


# ============================================================
# PALETTE CHIP
# ============================================================
class PaletteChip(QPushButton):
    """Draggable chip; label is the button text, mime_data is the drop payload."""
    def __init__(self, label, mime_data=None, sensor=False):
        super().__init__(label)
        self._mime_data   = mime_data if mime_data is not None else label
        self._drag_origin = QPointF(0, 0)
        self.setFixedHeight(24)
        bg = '#E8F4E8' if sensor else '#E8F0F8'
        hv = '#D0ECD0' if sensor else '#D0E0F0'
        self.setStyleSheet(
            f"QPushButton{{padding:0 8px;font-size:9px;"
            f"border:1px solid #A0B0C0;border-radius:3px;background:{bg};}}"
            f"QPushButton:hover{{background:{hv};}}"
        )

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag_origin = ev.position()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if (ev.buttons() & Qt.LeftButton) and \
                (ev.position() - self._drag_origin).manhattanLength() > 8:
            drag = QDrag(self)
            md   = QMimeData()
            md.setText(self._mime_data)
            drag.setMimeData(md)
            drag.exec(Qt.CopyAction)


# ============================================================
# CUSTOM VIEWBOX
# ============================================================
class NetworkViewBox(pg.ViewBox):
    """Custom ViewBox that intercepts horizontal drags in edit mode to reorder nodes."""

    def __init__(self, nw):
        super().__init__()
        self._nw       = nw
        self._dragging = False

    def mouseDragEvent(self, ev, axis=None):
        nw = self._nw
        if nw._edit_mode:
            ev.accept()
            if ev.isStart():
                self._dragging = True
                start_pt = self.mapSceneToView(ev.buttonDownScenePos())
                hit = nw._node_at(start_pt)
                # Dragging an already-selected node → reposition; dragging any other node → connect
                if hit is not None and nw._selected is not None and \
                        hit.rsplit('_', 1)[0] == nw._selected.rsplit('_', 1)[0]:
                    nw._conn_from = None   # reposition mode (node already selected)
                elif hit is not None and nw._selected is None:
                    # Nothing selected: auto-select the hit node and reposition it
                    nw._selected = hit
                    nw._spot_pen_override.clear()
                    nw._spot_pen_override[hit] = 'selected'
                    nw._redraw_nodes()
                    nw._conn_from = None
                else:
                    nw._conn_from = hit   # connect from this unselected node to another
            if self._dragging:
                mouse_pt = self.mapSceneToView(ev.scenePos())
                if nw._conn_from is not None:
                    nw._update_conn_preview(mouse_pt)
                elif nw._selected is not None:
                    snap_x = nw._get_snap_x(mouse_pt.x())
                    layer_name = nw._selected.rsplit('_', 1)[0]
                    layer = next((l for l in nw.gui.circuit.layers if l.name == layer_name), None)
                    nw._show_drag_indicator(snap_x, mouse_pt.y(), layer)
            if ev.isFinish():
                self._dragging = False
                mouse_pt = self.mapSceneToView(ev.scenePos())
                if nw._conn_from is not None:
                    nw._hide_conn_preview()
                    hit = nw._node_at(mouse_pt)
                    if hit is not None:
                        src_layer = nw._conn_from.rsplit('_', 1)[0]
                        tgt_layer = hit.rsplit('_', 1)[0]
                        if tgt_layer != src_layer:
                            nw._finish_connection(nw._conn_from, hit)
                    nw._conn_from = None
                elif nw._selected is not None:
                    snap_x = nw._get_snap_x(mouse_pt.x())
                    nw._hide_drag_indicator()
                    nw._on_node_drag_drop(snap_x, mouse_pt.y())
        else:
            super().mouseDragEvent(ev, axis)


# ============================================================
# NETWORK VISUALIZER WINDOW
# ============================================================
class NetworkVisualizerWindow(QWidget):
    _ACTIVE_RGB   = (0.95, 0.45, 0.25)
    _EDGE_POS     = '#3A6FA8'
    _HL_EDGE      = '#E07828'
    _TD_EDGE      = (200, 130, 50)   # amber RGB for learnable (LearningLayerBase) connections
    _NODE_R       = 0.1125
    _MARKER_R     = 0.066
    _CROSS_BOW    = 0.35
    _INTERNAL_BOW = 0.22
    _PAD_X        = 0.10
    _RING_SCALE        = 0.82   # ring node size and ring radius relative to _NODE_R
    _DENSE_NODE_SCALE  = 0.68   # node size scale when layer has more than 4 neurons
    _MIDLINE_GAP       = 0.05   # extra gap inserted at y=0.5 for even-n layers
    _CAM_H_DATA        = 0.135  # camera thumbnail height in data coords (must match _build_inner)
    _DENSE_THRESHOLD = 4   # max(ns, nt) > this → sampled edges + badge
    _DENSE_SAMPLE    = 3   # neurons sampled per side for dense connections
    _PAD_Y        = 0.10
    _CAM_WEIGHT   = 3.0    # slot weight for image nodes (camera thumbnails need extra vertical space)

    @staticmethod
    def _reversed_idx(i: int, n: int, side: str) -> int:
        """Map neuron/filter index i to visual position.
        R-side reverses ordering so R[0] is at visual bottom, R[n-1] near midline."""
        return n - 1 - i if side == 'R' else i

    _PANEL_CFG    = {
        'sensor': ('#D8EED8', '#A8CCA8'),
        'layer':  ('#D4E4F4', '#A0C0DC'),
    }
    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        self.setWindowTitle("Network")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.StrongFocus)
        self.resize(800, 600)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Row 1: edit controls
        toolbar = QWidget()
        tb_lay  = QHBoxLayout(toolbar)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        self._btn_edit   = QPushButton("Edit")
        self._btn_save   = QPushButton("Save")
        self._btn_bonsai = QPushButton("Copy Bonsai")
        self._btn_svg    = QPushButton("Copy SVG")
        self._btn_3d     = QPushButton("3D")
        self._btn_undo   = QPushButton("Undo")
        self._btn_save.setEnabled(True)
        self._btn_undo.setEnabled(False)
        tb_lay.addWidget(self._btn_edit)
        tb_lay.addWidget(self._btn_save)
        tb_lay.addWidget(self._btn_bonsai)
        tb_lay.addWidget(self._btn_svg)
        tb_lay.addWidget(self._btn_3d)
        self._btn_cols     = QPushButton("Columns")
        self._btn_weights  = QPushButton("Weights")
        tb_lay.addWidget(self._btn_undo)
        tb_lay.addWidget(self._btn_cols)
        tb_lay.addWidget(self._btn_weights)
        tb_lay.addStretch()
        layout.addWidget(toolbar)

        # Rows 2+: draggable palette chips (shown only in edit mode)
        self._palette_bar = QWidget()
        pb_vlay = QVBoxLayout(self._palette_bar)
        pb_vlay.setContentsMargins(2, 1, 2, 1)
        pb_vlay.setSpacing(2)

        def _palette_row(label, chips):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{C['muted']};font-size:9px;min-width:40px;")
            row.addWidget(lbl)
            for chip in chips:
                row.addWidget(chip)
            row.addStretch()
            return row

        pb_vlay.addLayout(_palette_row("Sensors:", [
            PaletteChip(s.replace('Sensor', ''), f'sensor:{s}', sensor=True)
            for s in ['GradientSensor', 'ColorSensor',
                      'CollisionSensor', 'DistanceSensor', 'InteroceptiveSensor',
                      'ProprioceptiveSensor', 'WhiskerSensor', 'SkyCompassSensor',
                      'GrayCameraSensor', 'RGBCameraSensor']
        ]))
        pb_vlay.addLayout(_palette_row("Layers:", [
            PaletteChip(t.replace('Layer', ''), t)
            for t in ['LeakyLayer', 'Leaky2dLayer', 'SumLayer', 'ConstantLayer', 'SineLayer',
                      'MatsuokaLayer', 'AdaptiveLayer', 'PulseLayer',
                      'RingAttractorLayer', 'Conv2dLayer']
        ]))

        # Combined row: Body | Learning layers | Motifs (dynamic)
        combo_row = QHBoxLayout()
        combo_row.setContentsMargins(0, 0, 0, 0)
        combo_row.setSpacing(4)
        lbl_body = QLabel("Body:")
        lbl_body.setStyleSheet(f"color:{C['muted']};font-size:9px;min-width:40px;")
        combo_row.addWidget(lbl_body)
        combo_row.addWidget(PaletteChip('Body', 'joint'))
        lbl_learn = QLabel("Learning:")
        lbl_learn.setStyleSheet(f"color:{C['muted']};font-size:9px;margin-left:6px;")
        combo_row.addWidget(lbl_learn)
        for t in ['TDLayer', 'DeltaLayer', 'ThreeFactorLayer']:
            combo_row.addWidget(PaletteChip(t.replace('Layer', ''), t))
        self._motifs_palette_widget = QWidget()
        self._motifs_palette_layout = QHBoxLayout(self._motifs_palette_widget)
        self._motifs_palette_layout.setContentsMargins(0, 0, 0, 0)
        self._motifs_palette_layout.setSpacing(4)
        combo_row.addWidget(self._motifs_palette_widget)
        combo_row.addStretch()
        pb_vlay.addLayout(combo_row)
        self._reload_motifs_palette()

        self._palette_bar.setVisible(False)
        layout.addWidget(self._palette_bar)

        # Column visibility panel (left sidebar, toggled by "Columns" button)
        self._col_panel = QFrame()
        self._col_panel.setFixedWidth(140)
        self._col_panel.setFrameShape(QFrame.Shape.Box)
        self._col_panel.setFrameShadow(QFrame.Shadow.Plain)
        self._col_panel.setLineWidth(1)
        self._col_panel.setStyleSheet("QFrame { border: 1px solid #B0C0D0; }")
        self._col_panel.setVisible(False)
        _gb_outer = QVBoxLayout(self._col_panel)
        _gb_outer.setContentsMargins(4, 4, 4, 4)
        _gb_outer.setSpacing(4)

        self._compact_cb = QCheckBox("Compact")
        self._compact_cb.setChecked(True)
        self._compact_cb.setStyleSheet("font-size:9px;")
        self._compact_cb.stateChanged.connect(self._on_compact_toggled)
        _gb_outer.addWidget(self._compact_cb)

        self._group_scroll = QScrollArea()
        self._group_scroll.setWidgetResizable(True)
        self._group_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._group_scroll.setFrameShape(self._group_scroll.Shape.NoFrame)
        self._group_cols_widget = QWidget()
        self._group_bar_lay = QVBoxLayout(self._group_cols_widget)
        self._group_bar_lay.setContentsMargins(0, 0, 0, 0)
        self._group_bar_lay.setSpacing(2)
        self._group_bar_lay.addStretch()
        self._group_scroll.setWidget(self._group_cols_widget)
        _gb_outer.addWidget(self._group_scroll, 1)

        _action_ss = (
            "QPushButton{padding:2px 4px;font-size:9px;"
            "border:1px solid #A0B0C0;border-radius:3px;}"
        )
        self._btn_show_all   = QPushButton("Show All")
        self._btn_enable_all = QPushButton("Enable All")
        for b in (self._btn_show_all, self._btn_enable_all):
            b.setFixedHeight(20)
            b.setStyleSheet(_action_ss)
        self._btn_show_all.clicked.connect(self._on_show_all)
        self._btn_enable_all.clicked.connect(self._on_enable_all)
        _btn_row = QHBoxLayout()
        _btn_row.setSpacing(4)
        _btn_row.addWidget(self._btn_show_all)
        _btn_row.addWidget(self._btn_enable_all)
        _gb_outer.addLayout(_btn_row)

        self._btn_cols.clicked.connect(self._toggle_col_win)
        self._btn_weights.clicked.connect(self._toggle_weight_panel)

        # Live weight visualization panel (hidden by default, toggled by Weights button)
        self._weight_panel = QFrame()
        self._weight_panel.setFixedWidth(230)
        self._weight_panel.setFrameShape(QFrame.Shape.Box)
        self._weight_panel.setFrameShadow(QFrame.Shadow.Plain)
        self._weight_panel.setLineWidth(1)
        self._weight_panel.setStyleSheet("QFrame { border: 1px solid #B0C0D0; }")
        self._weight_panel.setVisible(False)
        _wp_outer = QVBoxLayout(self._weight_panel)
        _wp_outer.setContentsMargins(4, 4, 4, 4)
        _wp_outer.setSpacing(4)
        _wp_scroll = QScrollArea()
        _wp_scroll.setWidgetResizable(True)
        _wp_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _wp_scroll.setFrameShape(QFrame.Shape.NoFrame)
        _wp_inner = QWidget()
        self._weight_entries_layout = QVBoxLayout(_wp_inner)
        self._weight_entries_layout.setContentsMargins(0, 0, 0, 0)
        self._weight_entries_layout.setSpacing(6)
        self._weight_entries_layout.addStretch()
        _wp_scroll.setWidget(_wp_inner)
        _wp_outer.addWidget(_wp_scroll)
        self._weight_pinned = {}   # (src, tgt) -> WeightEntryWidget

        # Content area: column panel + weight panel + graph side by side
        _content = QWidget()
        _content_lay = QHBoxLayout(_content)
        _content_lay.setContentsMargins(0, 0, 0, 0)
        _content_lay.setSpacing(0)
        _content_lay.addWidget(self._col_panel)
        _content_lay.addWidget(self._weight_panel)

        self._gw   = pg.GraphicsLayoutWidget()
        self._gw.setBackground(QColor(C['bg']))
        self._gw.setFocusPolicy(Qt.NoFocus)   # keep key events on NetworkVisualizerWindow
        self._vb   = NetworkViewBox(self)
        self._plot = self._gw.addPlot(viewBox=self._vb)
        # Store slot reference once so connect/disconnect always use the same object
        self._rebuild_edges_slot = self._rebuild_edges
        self._plot.setAspectLocked(True)
        self._plot.hideAxis('bottom')
        self._plot.hideAxis('left')
        self._plot.setMenuEnabled(False)
        self._vb.setMenuEnabled(False)
        self._vb.disableAutoRange()
        _content_lay.addWidget(self._gw, 1)
        layout.addWidget(_content, 1)
        self._gw.scene().sigMouseClicked.connect(
            self._on_scene_click, Qt.ConnectionType.QueuedConnection
        )

        self._drag_indicator = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen('#6AAAD4', width=2, style=Qt.DashLine),
        )
        self._drag_indicator.setVisible(False)
        self._plot.addItem(self._drag_indicator)

        self._h_drag_indicator = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen('#6AAAD4', width=2, style=Qt.DashLine),
        )
        self._h_drag_indicator.setVisible(False)
        self._plot.addItem(self._h_drag_indicator)

        self._conn_preview = pg.PlotDataItem(
            pen=pg.mkPen('#E07828', width=2, style=Qt.DashLine)
        )
        self._conn_preview.setVisible(False)
        self._conn_preview.setZValue(20)
        self._plot.addItem(self._conn_preview)

        self._pen_default  = pg.mkPen(C['dark'], width=1.5)
        self._pen_selected = pg.mkPen(C['primary'], width=6)
        self._pen_multi    = pg.mkPen('#E07828', width=6)
        self._pen_osc      = pg.mkPen('#00AAAA', width=3)   # teal = tracked in oscilloscope
        self._pen_muted    = pg.mkPen('#909090', width=1.5, style=Qt.DashLine)

        self._redrawing        = False   # reentrancy guard for _redraw_nodes
        self._all_scatter      = None  # single ScatterPlotItem for all nodes
        self._ring_scatter     = None  # hollow rings: receiver rings + source waves
        self._deriv_scatter    = None  # small dot overlay for derivative=True nodes
        self._wave_phase       = {}    # {nt_name: float 0→1}  advances when active
        self._src_nodes        = {}    # {node_key: nt_name}
        self._rcv_nodes        = {}    # {node_key: [nt_name, ...]}
        self._mod_colors       = {}    # {nt_name: (r,g,b)} built at build time
        self._mod_pens         = {}    # {nt_name: QPen} cached modulator border pens
        self._spot_names       = []   # ordered node keys
        self._spot_base_rgb    = {}   # name → (r,g,b) floats [0,1]
        self._spot_alpha       = {}   # name → float [0,1]  (edge highlight fade)
        self._spot_visible     = {}   # name → bool  (group hide)
        self._spot_pen_override = {}  # name → 'selected' | 'multi'
        self._positions        = {}
        self._selected         = None
        self._selected_edge    = None  # (src_name, tgt_name) or None
        self._conn_from        = None  # node key being dragged for connection
        self._range_signal_connected = False
        self._rebuilding_edges = False
        self._building         = False
        self._edit_mode        = False
        self._weight_params    = {}   # (src, tgt) → last-used pattern params dict
        self._x_unit           = 1.0
        self._depth_vals       = []
        self._col_x_map        = {}
        self._palette_x        = 0.5
        self._palette_y        = 1.22
        self._edge_items        = []
        self._edge_items_tagged = []  # 6-tuples: (item, sn, tn, excitatory, is_curve, original_pen)
        self._edge_params       = []  # raw draw params, replayed on zoom to fix rim coords
        self._panel_items       = []
        self._panel_rect_map    = {}   # depth_val → PlotDataItem (column background rect)
        self._col_label_items   = {}   # depth_val → TextItem (annotation above rect)
        self._col_labels        = {}   # depth_val → str  (user annotation, optional)
        self._text_items        = []
        self._text_map          = {}   # node_key → TextItem
        self._node_col_map      = {}   # node_key → depth_val (int)
        self._hidden_cols       = set()   # depth_vals hidden from viz but still computed
        self._disabled_cols     = set()   # depth_vals excluded from computation (and hidden)
        self._sensor_nodes      = set()   # node keys that are sensor outputs (hollow nodes)
        self._sensor_pens       = {}      # node_key → QPen (palette border)
        self._sensor_active_rgb = {}      # node_key → (r,g,b) palette colour for activity
        self._camera_items      = {}      # sensor.name → pg.ImageItem (CameraSensor only)
        self._camera_rects      = {}      # same keys → (x, y, w, h) for re-anchoring after setImage
        self._highlighted_node  = None
        self._node_just_clicked = False   # prevents scene click from clearing a fresh highlight
        self._multi_selected    = set()   # node keys selected via shift+click

        self._3d_mode = False
        self._compact_mode = True
        self._undo_stack = []
        self._btn_edit.clicked.connect(self._toggle_edit)
        self._btn_save.clicked.connect(self._save_circuit)
        self._btn_bonsai.clicked.connect(self._copy_bonsai)
        self._btn_svg.clicked.connect(self._copy_svg)
        self._btn_3d.clicked.connect(self._toggle_3d)
        self._btn_undo.clicked.connect(self._undo)
        from PySide6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence.StandardKey.Undo, self).activated.connect(self._undo)

        self._gw.setAcceptDrops(True)
        self._gw.installEventFilter(self)

        QTimer.singleShot(50, self.build)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(100)
        self._refresh_timer.timeout.connect(self._on_refresh_timer)
        self._refresh_timer.start()

    def _on_refresh_timer(self):
        if not self._building and self._all_scatter is not None:
            brain = getattr(self.gui, 'brain', None)
            if brain is not None:
                try:
                    self._redraw_nodes(brain)
                except Exception:
                    pass
                try:
                    self._update_anim(brain)
                except Exception:
                    pass
                try:
                    self._update_camera_nodes()
                except Exception:
                    pass
                try:
                    self._update_weight_panel()
                except Exception:
                    pass

    def _update_camera_nodes(self):
        from sensors import CameraSensor as _CamSensor
        from neurons import Leaky2dLayer as _L2d, Conv2dLayer as _Conv2d
        DISP_H = 32

        # Leaky2dLayer image thumbnails.
        for lyr in self.gui.circuit.layers:
            if not isinstance(lyr, _L2d):
                continue
            frame = getattr(lyr, '_last_frame', None)
            if frame is None:
                continue
            item = self._camera_items.get(lyr.name)
            if item is None:
                continue
            # Convert to (H, W, 3) uint8.
            # Derivative mode: fixed [-1, 1] → [0, 255] so zero → mid-gray
            # and a static scene correctly appears as uniform gray.
            # Normal mode: fixed [0, 1] → [0, 255].
            if lyr.in_ch == 3:
                rgb = frame  # already (H, W, 3)
            else:
                g = frame if frame.ndim == 2 else np.mean(frame, axis=-1)
                rgb = np.stack([g, g, g], axis=-1)
            if getattr(lyr, 'derivative', False):
                data = np.clip((rgb + 1.0) * 127.5, 0, 255).astype(np.uint8)
            else:
                data = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            reps = max(1, DISP_H // max(data.shape[0], 1))
            data = np.repeat(data, reps, axis=0)[:DISP_H]
            item.setImage(data, axisOrder='row-major')
            rect = self._camera_rects.get(lyr.name)
            if rect is not None:
                item.setRect(*rect)

        # Conv2dLayer pool='none' image thumbnails.
        for lyr in self.gui.circuit.layers:
            if not isinstance(lyr, _Conv2d) or getattr(lyr, 'pool', '') != 'none':
                continue
            frame = getattr(lyr, '_last_frame', None)
            if frame is None:
                continue
            item = self._camera_items.get(lyr.name)
            if item is None:
                continue
            g = frame if frame.ndim == 2 else np.mean(frame, axis=-1)
            data = np.clip(g * 255, 0, 255).astype(np.uint8)
            data = np.stack([data, data, data], axis=-1)
            reps = max(1, DISP_H // max(data.shape[0], 1))
            data = np.repeat(data, reps, axis=0)[:DISP_H]
            item.setImage(data, axisOrder='row-major')
            rect = self._camera_rects.get(lyr.name)
            if rect is not None:
                item.setRect(*rect)

        for sensor in self.gui.circuit.sensors:
            if not isinstance(sensor, _CamSensor):
                continue
            frame = getattr(sensor, '_last_frame', None)
            if frame is None:
                continue
            # Use in_ch (sensor property) rather than frame ndim: MuJoCo always
            # stores (H, W, 3) in _last_frame even for GrayCameraSensor.
            if getattr(sensor, 'in_ch', 1) == 3:      # RGB
                data = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:                                      # gray — compute luma if needed
                g = frame if frame.ndim == 2 else np.mean(frame, axis=-1)
                g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
                data = np.stack([g, g, g], axis=-1)
            # Tile rows to a fixed display height.
            reps = max(1, DISP_H // data.shape[0])
            data = np.repeat(data, reps, axis=0)[:DISP_H]

            if getattr(sensor, 'lateralized', False):
                half    = sensor.width // 2
                ovl     = getattr(sensor, 'overlap', 0)
                l_end   = int(np.clip(half + ovl, 0, sensor.width))
                r_start = int(np.clip(half - ovl, 0, sensor.width))
                key_L = f'{sensor.name}_L'
                item_L = self._camera_items.get(key_L)
                if item_L is not None:
                    item_L.setImage(data[:, :l_end, :], axisOrder='row-major')
                    rect_L = self._camera_rects.get(key_L)
                    if rect_L is not None:
                        item_L.setRect(*rect_L)
                key_R = f'{sensor.name}_R'
                item_R = self._camera_items.get(key_R)
                if item_R is not None:
                    item_R.setImage(data[:, r_start:, :], axisOrder='row-major')
                    rect_R = self._camera_rects.get(key_R)
                    if rect_R is not None:
                        item_R.setRect(*rect_R)
            else:
                item = self._camera_items.get(sensor.name)
                if item is not None:
                    item.setImage(data, axisOrder='row-major')
                    rect = self._camera_rects.get(sensor.name)
                    if rect is not None:
                        item.setRect(*rect)

    def _update_anim(self, brain):
        if self._ring_scatter is None:
            return

        # Activation per neuromodulator substance
        nt_activation = {}
        if brain is not None:
            for layer in getattr(self.gui.circuit, 'layers', []):
                nt = getattr(layer, 'neuromodulator_transmitter', None)
                if not nt or nt not in self._mod_colors:
                    continue
                attr = getattr(brain, layer.name, None)
                if attr is None:
                    continue
                arr = np.atleast_1d(attr.output if hasattr(attr, 'output') else attr)
                act = float(np.clip(np.mean(arr), 0, 1)) if arr.size > 0 else 0.0
                nt_activation[nt] = max(nt_activation.get(nt, 0.0), act)
            for sensor in getattr(self.gui.circuit, 'sensors', []):
                nt = getattr(sensor, 'neuromodulator_transmitter', None)
                if not nt or nt not in self._mod_colors:
                    continue
                val = getattr(brain, sensor.name, None)
                if val is None:
                    continue
                arr = np.atleast_1d(val)
                act = float(np.clip(np.mean(arr), 0, 1)) if arr.size > 0 else 0.0
                nt_activation[nt] = max(nt_activation.get(nt, 0.0), act)

        # Advance wave phase only when the source is active
        for nt, act in nt_activation.items():
            if act >= 0.05:
                self._wave_phase[nt] = (self._wave_phase.get(nt, 0.0) + 0.07) % 1.0

        r_node    = self._NODE_R
        ring_spots = []

        # Receiver satellites — small filled dot whose edge touches the node edge
        SAT_SIZE = 9               # dot diameter in pixels
        SAT_BASE = np.pi / 4       # starting angle (upper-right)
        dx       = self._vb.viewPixelSize()[0]          # data units per pixel
        SAT_R    = (r_node * 120 + SAT_SIZE / 2) * dx  # pixel-node-radius + sat px-radius → scene units
        for node_key, nt_list in self._rcv_nodes.items():
            if not self._spot_visible.get(node_key, True):
                continue
            pos = self._positions.get(node_key)
            if pos is None:
                continue
            x, y = pos
            n = len(nt_list)
            for idx, mod_name in enumerate(nt_list):
                angle = SAT_BASE + idx * (2 * np.pi / n)
                sx    = x + SAT_R * np.cos(angle)
                sy    = y + SAT_R * np.sin(angle)
                rgb   = self._mod_colors[mod_name]
                color = QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                ring_spots.append({
                    'pos':   (sx, sy),
                    'size':  SAT_SIZE,
                    'brush': pg.mkBrush(color),
                    'pen':   pg.mkPen(None),
                })

        # Source waves — 3 concentric expanding rings, activation-gated
        N_WAVES   = 3
        MAX_EXTRA = 60   # px the wave expands beyond the node edge
        for node_key, nt in self._src_nodes.items():
            if not self._spot_visible.get(node_key, True):
                continue
            pos = self._positions.get(node_key)
            if pos is None:
                continue
            act = nt_activation.get(nt, 0.0)
            if act < 0.05:
                continue
            x, y  = pos
            rgb   = self._mod_colors[nt]
            base_color = QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            phase = self._wave_phase.get(nt, 0.0)
            for i in range(N_WAVES):
                p     = (phase + i / N_WAVES) % 1.0
                alpha = int(220 * (1 - p) * act)
                size  = r_node * 240 + (4 + p * MAX_EXTRA)
                c     = QColor(base_color)
                c.setAlpha(alpha)
                ring_spots.append({
                    'pos': (x, y),
                    'size': size,
                    'brush': pg.mkBrush(None),
                    'pen': pg.mkPen(c, width=2),
                })

        self._ring_scatter.setData(spots=ring_spots)

    def keyPressEvent(self, ev):
        ctrl = ev.modifiers() & Qt.ControlModifier
        if ev.key() == Qt.Key_Delete:
            if self._selected_edge is not None:
                self._remove_selected_connection()
            elif self._selected is not None:
                lname = self._selected.rsplit('_', 1)[0]
                if any(l.name == lname for l in self.gui.circuit.layers):
                    self._remove_selected_layer()
                elif any(s.name == lname for s in self.gui.circuit.sensors):
                    self._remove_selected_sensor()
        elif ctrl and ev.key() == Qt.Key_C:
            self._copy_selection()
        elif ctrl and ev.key() == Qt.Key_V:
            self._paste_selection()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self._refresh_timer.stop()
        self.gui._net_viz = None
        super().closeEvent(ev)

    @staticmethod
    def _hex_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def _n_map(self):
        from sensors import CameraSensor as _CamSensor
        circuit = self.gui.circuit
        m = {}
        for s in circuit.sensors:
            if _sensor_is_lateralized(s, circuit):
                if isinstance(s, _CamSensor):
                    half    = s.width // 2
                    ovl     = getattr(s, 'overlap', 0)
                    l_end   = int(np.clip(half + ovl, 0, s.width))
                    r_start = int(np.clip(half - ovl, 0, s.width))
                    m[f'{s.name}_L'] = l_end * getattr(s, 'in_ch', 3)
                    m[f'{s.name}_R'] = (s.width - r_start) * getattr(s, 'in_ch', 3)
                else:
                    # Joint-pair sensor: each half has sensor.n outputs
                    m[f'{s.name}_L'] = s.n or 1
                    m[f'{s.name}_R'] = s.n or 1
            else:
                m[s.name] = s.n
        m.update({l.name: l.n for l in circuit.layers if l.n is not None})
        return m

    def _compute_depth(self):
        c     = self.gui.circuit
        depth = {s.name: getattr(s, 'layer', 0) for s in c.sensors}
        for s in c.sensors:
            if _sensor_is_lateralized(s, c):
                d = getattr(s, 'layer', 0)
                depth[f'{s.name}_L'] = d
                depth[f'{s.name}_R'] = d
        for lyr in c.layers:
            if getattr(lyr, 'layer', None) is not None:
                depth[lyr.name] = lyr.layer
        for lyr in c.layers:
            if getattr(lyr, 'layer', None) is not None:
                continue
            src_depths = [depth[conn.src] for conn in c.connections
                          if conn.tgt == lyr.name and conn.src in depth]
            depth[lyr.name] = (max(src_depths) + 1) if src_depths else 1
        return depth

    def _compute_depth_3d(self):
        """Feedforward depth from connectivity; sensors and ConstantLayers are depth 0."""
        c = self.gui.circuit
        depth = {s.name: 0 for s in c.sensors}
        for lyr in c.layers:
            if type(lyr).__name__ == 'ConstantLayer':
                depth[lyr.name] = 0
        # Iterative relaxation: longest path from any source.
        changed = True
        while changed:
            changed = False
            for conn in c.connections:
                if conn.src in depth:
                    new_d = depth[conn.src] + 1
                    if depth.get(conn.tgt, -1) < new_d:
                        depth[conn.tgt] = new_d
                        changed = True
        for lyr in c.layers:
            if lyr.name not in depth:
                depth[lyr.name] = 1
        return depth

    def _compute_z_planes(self):
        """Auto-assign Z planes from topology.
        Each source (sensor or ConstantLayer) gets Z = index+1.
        Nodes exclusively downstream of one source share that source's Z.
        Nodes reachable from multiple sources get Z=0 (shared/front plane).
        """
        c = self.gui.circuit
        source_names = [s.name for s in c.sensors]
        source_names += [l.name for l in c.layers
                         if type(l).__name__ == 'ConstantLayer']

        # BFS forward reachability from each source
        reachable = {}
        for src in source_names:
            visited = {src}
            queue = [src]
            while queue:
                node = queue.pop()
                for conn in c.connections:
                    if conn.src == node and conn.tgt not in visited:
                        visited.add(conn.tgt)
                        queue.append(conn.tgt)
            reachable[src] = visited

        source_z = {src: i + 1 for i, src in enumerate(source_names)}
        z_map = dict(source_z)

        for lyr in c.layers:
            name = lyr.name
            if name in z_map:
                continue
            owners = [src for src in source_names if name in reachable[src]]
            z_map[name] = source_z[owners[0]] if len(owners) == 1 else 0

        return z_map

    def _layout_3d(self):
        """Oblique 3D layout: topology-based Z planes, feedforward depth X, bilateral Y.
        Negative Z_DX pushes back-plane sources to the upper-left so all edges flow
        consistently left→right toward the shared convergence layers at Z=0.
        """
        Z_DX, Z_DY = -0.20, 0.15

        c = self.gui.circuit
        sensors = c.sensors
        layers  = [l for l in c.layers if l.n is not None]

        depth = self._compute_depth_3d()
        z_map = self._compute_z_planes()
        self._z_map = z_map
        # Build reverse label map: z_index → source name for plane tabs
        source_names = [s.name for s in c.sensors]
        source_names += [l.name for l in c.layers
                         if type(l).__name__ == 'ConstantLayer']
        self._z_labels = {0: 'shared'}
        self._z_labels.update({i + 1: name for i, name in enumerate(source_names)})

        # Group nodes by (z_plane, feedforward_depth)
        by_z_depth = defaultdict(lambda: defaultdict(list))
        for s in sensors:
            z = z_map.get(s.name, 0)
            d = depth.get(s.name, 0)
            by_z_depth[z][d].append(('sensor', s))
        for l in layers:
            z = z_map.get(l.name, 0)
            d = depth.get(l.name, 0)
            by_z_depth[z][d].append(('layer', l))

        all_depths = ([depth.get(s.name, 0) for s in sensors] +
                      [depth.get(l.name, 0) for l in layers])
        max_depth = max(all_depths) if all_depths else 0
        x_unit = 1.0 / max(max_depth, 1)

        positions    = {}
        sensor_nodes = set()
        groups       = []

        for z_plane in sorted(by_z_depth):
            y_off = z_plane * Z_DY
            x_off = z_plane * Z_DX

            for d in sorted(by_z_depth[z_plane]):
                col = by_z_depth[z_plane][d]
                indexed = list(enumerate(col))
                indexed.sort(key=lambda p: getattr(p[1][1], 'viz_row', 1000 + p[0]))
                col = [item for _, item in indexed]

                x_base = d * x_unit + x_off

                split_groups = [(ct, obj) for ct, obj in col if _eff_n(obj) >= 2]
                mid_groups   = [(ct, obj) for ct, obj in col if _eff_n(obj) == 1]

                total_left = sum(_eff_n(obj) // 2 for _, obj in split_groups)
                if total_left > 0:
                    slot_h   = 0.5 / total_left
                    left_idx = 0
                    for ct, obj in split_groups:
                        en          = _eff_n(obj)
                        half        = en // 2
                        y_order     = list(getattr(obj, 'y_order', None) or range(en))
                        midline_adj = (self._MIDLINE_GAP / 2) if en % 2 == 0 else 0.0
                        for jl in range(half):
                            y_left  = 1.0 - slot_h / 2 - left_idx * slot_h + midline_adj
                            y_right = 1.0 - y_left
                            positions[f'{obj.name}_{y_order[jl]}']            = (x_base, y_left  + y_off)
                            positions[f'{obj.name}_{y_order[en - 1 - jl]}']   = (x_base, y_right + y_off)
                            left_idx += 1

                # Lateralized n=1 pairs: place L in upper half, R in lower half.
                _seen_mid_lat_3d = set()
                lateral_mid_3d   = []
                regular_mid_3d   = []
                for ct, obj in mid_groups:
                    pair_name = getattr(obj, 'lateral_pair', None)
                    if pair_name:
                        key = tuple(sorted([obj.name, pair_name]))
                        if key not in _seen_mid_lat_3d:
                            _seen_mid_lat_3d.add(key)
                            partner = next((o for _, o in mid_groups if o.name == pair_name), None)
                            if partner is not None:
                                L = obj if obj.name.endswith('_L') else partner
                                R = partner if L is obj else obj
                                lateral_mid_3d.append((L, R))
                            else:
                                regular_mid_3d.append((ct, obj))
                        continue
                    regular_mid_3d.append((ct, obj))

                for L_obj, R_obj in lateral_mid_3d:
                    positions[f'{L_obj.name}_0'] = (x_base, 0.75 + y_off)
                    positions[f'{R_obj.name}_0'] = (x_base, 0.25 + y_off)

                for ct, obj in regular_mid_3d:
                    en      = _eff_n(obj)
                    y_order = list(getattr(obj, 'y_order', None) or range(en or 1))
                    positions[f'{obj.name}_{y_order[0]}'] = (x_base, 0.5 + y_off)

                for ct, obj in col:
                    col_nodes = [f'{obj.name}_{j}' for j in range(_eff_n(obj))]
                    if ct == 'sensor':
                        sensor_nodes.update(col_nodes)
                    groups.append((ct, obj.name, col_nodes, x_base, d,
                                   1.0 + y_off, 0.0 + y_off, getattr(obj, 'color', None)))

        self._x_unit     = x_unit
        self._depth_vals = sorted({d for z in by_z_depth for d in by_z_depth[z]})
        return positions, sensor_nodes, groups

    @staticmethod
    def _slot_weight(entry) -> float:
        """Proportional vertical weight for one slot-list entry.
        Image nodes (camera halves, viz_n=1 layers) get _CAM_WEIGHT so thumbnails
        have adequate spacing; all other items get their effective neuron count."""
        _CAM_W = NetworkVisualizerWindow._CAM_WEIGHT
        if entry[0] == 'pair':
            _, L_ct, L_obj, R_ct, R_obj = entry
            if (getattr(L_obj, 'is_image', False) or getattr(L_obj, 'viz_n', None) == 1 or
                    getattr(R_obj, 'is_image', False) or getattr(R_obj, 'viz_n', None) == 1):
                return _CAM_W
            return max(_eff_n(L_obj), _eff_n(R_obj), 1)
        else:
            _, ct, obj = entry
            if getattr(obj, 'is_image', False) or getattr(obj, 'viz_n', None) == 1:
                return _CAM_W
            return max(_eff_n(obj), 1)

    def _layout(self):
        from sensors import CameraSensor as _CamSensor
        circuit  = self.gui.circuit
        layers   = [l for l in circuit.layers if l.n is not None]

        # Replace lateralized sensors with L/R SplitHalf stand-ins for layout purposes
        sensors = []
        for s in circuit.sensors:
            if _sensor_is_lateralized(s, circuit):
                sensors.append(_SplitHalf(s, 'L'))
                sensors.append(_SplitHalf(s, 'R'))
            else:
                sensors.append(s)

        brain_cls   = self.gui.brain.__class__ if self.gui.brain else None
        viz_columns = getattr(brain_cls, 'viz_columns', None)

        name_map = {s.name: ('sensor', s) for s in sensors}
        name_map.update({l.name: ('layer', l) for l in layers})

        if viz_columns:
            depth_cols = [
                [name_map[n] for n in grp if n in name_map]
                for grp in viz_columns
            ]
            depth_cols = [col for col in depth_cols if col]
            depth_vals = list(range(len(depth_cols)))
        else:
            depth = self._compute_depth()
            # Enforce lateralized pairs into the same column (≥1, never the sensor column 0)
            for l in layers:
                pair_name = getattr(l, 'lateral_pair', None)
                if pair_name:
                    partner = next((pl for pl in layers if pl.name == pair_name), None)
                    if partner:
                        d = max(depth.get(l.name, 1), depth.get(partner.name, 1), 1)
                        depth[l.name] = d
                        depth[partner.name] = d
            by_depth = defaultdict(list)
            for s in sensors:
                by_depth[depth.get(s.name, 0)].append(('sensor', s))
            for l in layers:
                by_depth[depth.get(l.name, 1)].append(('layer', l))
            sorted_depths = sorted(by_depth)
            depth_cols = []
            for d in sorted_depths:
                items = by_depth[d]
                # Sort by viz_row if set; preserve natural insertion order otherwise.
                indexed = list(enumerate(items))
                indexed.sort(key=lambda p: getattr(p[1][1], 'viz_row', 1000 + p[0]))
                depth_cols.append([item for _, item in indexed])
            depth_vals    = sorted_depths

        positions    = {}
        sensor_nodes = set()
        groups       = []

        max_depth_val = max(depth_vals) if depth_vals else 1
        panel_hw = self._NODE_R + self._PAD_X
        col_gap  = 0.06
        if len(depth_vals) > 1:
            min_ds  = min(b - a for a, b in zip(depth_vals, depth_vals[1:]))
            x_unit  = max((2 * panel_hw + col_gap) / min_ds,
                          1.0 / max(max_depth_val, 1))
        else:
            x_unit  = 1.0

        for col, depth_val in zip(depth_cols, depth_vals):
            x = depth_val * x_unit

            ring_items    = [(ct, obj) for ct, obj in col
                             if getattr(obj, 'viz_layout', None) == 'ring']
            regular_items = [(ct, obj) for ct, obj in col
                             if getattr(obj, 'viz_layout', None) != 'ring']

            from neurons import Conv2dLayer as _Conv2dLayer

            # ── Build slot list: pair up lateralized items, keep singles ─────
            slot_list  = []    # [('pair', L_ct, L, R_ct, R) | ('single', ct, obj)]
            seen_pairs = set()
            for ct, obj in regular_items:
                pair_name = getattr(obj, 'lateral_pair', None)
                if pair_name is not None:
                    key = tuple(sorted([obj.name, pair_name]))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    partner_item = next(
                        ((pct, pobj) for pct, pobj in regular_items
                         if pobj.name == pair_name), None)
                    if partner_item is None:
                        slot_list.append(('single', ct, obj))   # orphaned half
                    else:
                        pct, pobj = partner_item
                        if obj.name.endswith('_L'):
                            slot_list.append(('pair', ct, obj, pct, pobj))
                        elif obj.name.endswith('_R'):
                            slot_list.append(('pair', pct, pobj, ct, obj))
                        else:
                            slot_list.append(('pair', ct, obj, pct, pobj))
                else:
                    slot_list.append(('single', ct, obj))

            # ── Compute total weight and place each slot ──────────────────────
            total_weight = sum(self._slot_weight(e) for e in slot_list) or 1.0
            cumulative   = 0.0

            for slot_idx, entry in enumerate(slot_list):
                w        = self._slot_weight(entry)
                slot_top = 1.0 - cumulative / total_weight
                slot_bot = 1.0 - (cumulative + w) / total_weight
                slot_ctr = (slot_top + slot_bot) / 2
                slot_h   = slot_top - slot_bot

                if entry[0] == 'pair':
                    _, L_ct, L_obj, R_ct, R_obj = entry
                    n_L   = max(_eff_n(L_obj), 1)
                    n_R   = max(_eff_n(R_obj), 1)
                    L_img = getattr(L_obj, 'is_image', False) or getattr(L_obj, 'viz_n', None) == 1
                    R_img = getattr(R_obj, 'is_image', False) or getattr(R_obj, 'viz_n', None) == 1
                    half_h  = slot_h / 2
                    _h_half = self._CAM_H_DATA / 2

                    # L in upper half [slot_ctr, slot_top]
                    if L_img:
                        positions[f'{L_obj.name}_0'] = (
                            x, min(slot_top - _h_half, slot_ctr + half_h / 2))
                    else:
                        for i in range(n_L):
                            positions[f'{L_obj.name}_{i}'] = (
                                x, slot_top - (i + 0.5) * half_h / n_L)

                    # R in lower half [slot_bot, slot_ctr]
                    if R_img:
                        positions[f'{R_obj.name}_0'] = (
                            x, max(slot_bot + _h_half, slot_ctr - half_h / 2))
                    else:
                        for i in range(n_R):
                            positions[f'{R_obj.name}_{i}'] = (
                                x, slot_bot + (i + 0.5) * half_h / n_R)

                    # Groups and sensor_nodes
                    L_nodes = [f'{L_obj.name}_{j}' for j in range(n_L)]
                    R_nodes = [f'{R_obj.name}_{j}' for j in range(n_R)]
                    if L_ct == 'sensor': sensor_nodes.update(L_nodes)
                    if R_ct == 'sensor': sensor_nodes.update(R_nodes)
                    L_color = (getattr(L_obj, '_viz_color', None) if L_ct == 'sensor'
                               else getattr(L_obj, 'color', None))
                    R_color = (getattr(R_obj, '_viz_color', None) if R_ct == 'sensor'
                               else getattr(R_obj, 'color', None))
                    groups.append((L_ct, L_obj.name, L_nodes, x, depth_val,
                                   slot_top, slot_bot, L_color))
                    groups.append((R_ct, R_obj.name, R_nodes, x, depth_val,
                                   slot_top, slot_bot, R_color))
                    if L_ct == 'layer': L_obj.viz_row = slot_idx
                    if R_ct == 'layer': R_obj.viz_row = slot_idx

                else:   # 'single'
                    _, ct, obj = entry
                    en     = max(_eff_n(obj), 1)
                    is_img = (getattr(obj, 'is_image', False) or
                              getattr(obj, 'viz_n', None) == 1)

                    if is_img or en == 1:
                        # Image node or single neuron: centre of slot
                        positions[f'{obj.name}_0'] = (x, slot_ctr)

                    elif isinstance(obj, _Conv2dLayer) and not getattr(obj, 'lateralized', False):
                        # Non-lateralized Conv2dLayer: filters stacked top-to-bottom in slot
                        for j in range(en):
                            positions[f'{obj.name}_{j}'] = (
                                x, slot_top - (j + 0.5) * slot_h / en)

                    else:
                        # Bilateral: n//2 neurons above slot_ctr mirrored below
                        half        = en // 2
                        midline_adj = self._MIDLINE_GAP / 2 if en % 2 == 0 else 0.0
                        usable_h    = slot_h / 2 - midline_adj
                        y_order     = list(getattr(obj, 'y_order', None) or range(en))
                        for jl in range(half):
                            y_left  = slot_top - (jl + 0.5) * usable_h / max(half, 1)
                            y_right = 2.0 * slot_ctr - y_left   # mirror around slot centre
                            positions[f'{obj.name}_{y_order[jl]}']          = (x, y_left)
                            positions[f'{obj.name}_{y_order[en - 1 - jl]}'] = (x, y_right)
                        if en % 2 == 1:
                            positions[f'{obj.name}_{y_order[half]}'] = (x, slot_ctr)

                    # Groups and sensor_nodes
                    col_nodes = [f'{obj.name}_{j}' for j in range(en)]
                    if ct == 'sensor': sensor_nodes.update(col_nodes)
                    obj_color = (getattr(obj, '_viz_color', None) if ct == 'sensor'
                                 else getattr(obj, 'color', None))
                    groups.append((ct, obj.name, col_nodes, x, depth_val,
                                   slot_top, slot_bot, obj_color))
                    if ct == 'layer': obj.viz_row = slot_idx

                cumulative += w

            # Ring layout: neurons arranged in a circle, neuron 0 at the top.
            for _, obj in ring_items:
                n_ring  = obj.n
                ring_r  = max(0.15, min(0.38,
                              self._RING_SCALE * self._NODE_R / (2 * np.sin(np.pi / n_ring) + 1e-9)))
                cx, cy  = x, 0.5
                obj._ring_cx = cx
                obj._ring_cy = cy
                obj._ring_r  = ring_r
                for i in range(n_ring):
                    angle = np.pi / 2 - 2 * np.pi * i / n_ring  # CCW, node 0 at top
                    positions[f'{obj.name}_{i}'] = (
                        cx + ring_r * np.cos(angle),
                        cy + ring_r * np.sin(angle),
                    )
                col_nodes = [f'{obj.name}_{j}' for j in range(n_ring)]
                obj_color = getattr(obj, 'color', None)
                groups.append(('layer', obj.name, col_nodes, x, depth_val,
                               cy + ring_r + self._NODE_R,
                               cy - ring_r - self._NODE_R, obj_color))

        self._x_unit     = x_unit
        self._depth_vals = depth_vals
        return positions, sensor_nodes, groups

    @staticmethod
    def _ctrl_pt(p0, p1, bow):
        mx, my = (p0[0]+p1[0])/2, (p0[1]+p1[1])/2
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
        return (mx + bow * dy, my - bow * dx)

    @staticmethod
    def _bezier_pts(p0, ctrl, p1, n=80):
        ts = np.linspace(0, 1, n)
        xs = (1-ts)**2 * p0[0] + 2*(1-ts)*ts * ctrl[0] + ts**2 * p1[0]
        ys = (1-ts)**2 * p0[1] + 2*(1-ts)*ts * ctrl[1] + ts**2 * p1[1]
        return list(zip(xs.tolist(), ys.tolist()))

    @staticmethod
    def _clip_pts(pts, src, tgt, r):
        start = 0
        for i, (x, y) in enumerate(pts):
            if np.hypot(x - src[0], y - src[1]) > r:
                start = i
                break
        else:
            return []
        end = len(pts) - 1
        for i in range(start, len(pts)):
            if np.hypot(pts[i][0] - tgt[0], pts[i][1] - tgt[1]) < r:
                end = max(start, i - 1)
                break
        return pts[start:end + 1]

    @staticmethod
    def _hemisphere(idx, n):
        """True = left/top half, False = right/bottom half, None = single neuron (midline)."""
        if n <= 1:
            return None
        return idx < n // 2

    @staticmethod
    def _signed_bow(src_hem, tgt_hem, base_bow, fallback_y):
        """Compute signed bow from hemisphere assignments.

        Same hemisphere  → bow away from midline (±base_bow).
        Cross hemisphere → small symmetric bow: src-upper positive, src-lower negative.
        n=1 midline neuron → fall back to y-average.
        """
        if src_hem is None or tgt_hem is None:
            return base_bow if fallback_y >= 0.5 else -base_bow
        if src_hem == tgt_hem:
            return base_bow if src_hem else -base_bow
        # Cross: half-bow so it's visually distinct from ipsilateral, but never 0
        # (a zero bow would be clamped to +min_bow for all cross connections,
        # breaking the up/down symmetry).
        return (base_bow * 0.4) if src_hem else -(base_bow * 0.4)

    def _draw_edges(self, positions, n_map):
        # Build set of position keys that belong to image-display nodes (cameras,
        # Leaky2dLayer, Conv2dLayer pool='none').  _draw_edge uses this to offset
        # arc endpoints to the image circle rim rather than the neuron-dot rim.
        from sensors import CameraSensor as _CamSensorDE
        _img_keys: set = set()
        for _s in self.gui.circuit.sensors:
            if isinstance(_s, _CamSensorDE):
                if getattr(_s, 'lateralized', False):
                    _img_keys.add(f'{_s.name}_L_0')
                    _img_keys.add(f'{_s.name}_R_0')
                else:
                    _img_keys.add(f'{_s.name}_0')
        for _l in self.gui.circuit.layers:
            if getattr(_l, 'viz_n', None) == 1:
                _img_keys.add(f'{_l.name}_0')
        self._img_node_keys = _img_keys

        # When a specific neuron is selected, filter edges to only show connections
        # involving that neuron's index (rows/cols in the weight matrix).
        sel_layer = sel_idx = None
        if self._selected:
            parts = self._selected.rsplit('_', 1)
            if len(parts) == 2 and parts[1].lstrip('-').isdigit():
                sel_layer, sel_idx = parts[0], int(parts[1])

        by_target  = defaultdict(list)
        tgt_hem_map = {}

        for conn in self.gui.circuit.connections:
            src, tgt, W = conn.src, conn.tgt, conn.W
            W  = np.asarray(W, dtype=float)

            # Conv2d 4-D kernel (n_filters, in_ch, kH, kW).
            # Camera sensor halves are laid out as a single node (n=1), so skip the
            # per-pixel loop.  Draw one arc per filter from {src}_0; use the
            # centre-pixel mean to classify ON (>0 → excitatory) vs OFF (→ inhibitory).
            if W.ndim == 4:
                from sensors import CameraSensor as _CS4
                n_filters, in_ch, kH, kW = W.shape
                p_s = positions.get(f'{src}_0')
                if p_s is None:
                    continue
                # Lateralized camera half → single conv: R side writes to reversed
                # target neurons so the layout mirrors L (L0…Ln R[n]…R0 ordering).
                src_sensor_4 = next((s for s in self.gui.circuit.sensors if s.name == src), None)
                if src_sensor_4 is None:
                    parent_4 = src.rsplit('_', 1)[0]
                    src_sensor_4 = next((s for s in self.gui.circuit.sensors if s.name == parent_4), None)
                is_lat_R = (src.endswith('_R')
                            and src_sensor_4 is not None
                            and _sensor_is_lateralized(src_sensor_4, self.gui.circuit))
                tgt_n = n_map.get(tgt, n_filters)
                for i in range(n_filters):
                    tgt_idx = self._reversed_idx(i, tgt_n, 'R' if is_lat_R else 'L')
                    if sel_idx is not None and tgt == sel_layer and sel_idx != tgt_idx:
                        continue
                    tn = f'{tgt}_{tgt_idx}'
                    p_t = positions.get(tn)
                    if p_t is None:
                        continue
                    cw = abs(float(W[i, :, kH // 2, kW // 2].mean()))
                    if cw < 1e-10:
                        cw = float(np.abs(W[i]).max())
                    mid_y = (p_s[1] + p_t[1]) / 2
                    sb = self._CROSS_BOW if mid_y >= 0.5 else -self._CROSS_BOW
                    self._draw_edge(p_s, p_t, cw, sb, sn=f'{src}_0', tn=tn)
                continue

            ns = n_map.get(src, 1)
            nt = n_map.get(tgt, 1)
            # Detect combined lateralized source: W columns = n_L + n_R.
            # Two cases:
            #   A) Conv2dLayer pair: src has _lateral_pair → partner nodes from partner layer.
            #   B) Joint-pair sensor half: src ends _L/_R from a pair sensor → partner nodes
            #      from the mirror sensor half (e.g. sensor0_R for src=sensor0_L).
            _src_lyr_d   = next((l for l in self.gui.circuit.layers if l.name == src), None)
            _pair_nm_d   = getattr(_src_lyr_d, 'lateral_pair', None) if _src_lyr_d else None
            _pair_lyr_d  = None
            _pair_sensor_half_d = None   # partner sensor half name (for case B)
            _n_L_d       = ns   # neurons from the primary (L) half
            if _pair_nm_d and W.ndim == 2:
                _p = next((l for l in self.gui.circuit.layers if l.name == _pair_nm_d), None)
                if _p and W.shape[1] == (_src_lyr_d.n or 0) + (_p.n or 0):
                    _pair_lyr_d = _p
                    _n_L_d = _src_lyr_d.n or 0
                    ns = W.shape[1]
            elif (W.ndim == 2 and (src.endswith('_L') or src.endswith('_R'))):
                _src_snsr_d = next((s for s in self.gui.circuit.sensors
                                    if s.name == src.rsplit('_', 1)[0]), None)
                if (_src_snsr_d is not None
                        and _sensor_is_lateralized(_src_snsr_d, self.gui.circuit)
                        and W.shape[1] == ns * 2):
                    _pair_sensor_half_d = _mirror_name(src)
                    _n_L_d = ns
                    ns = W.shape[1]
            if W.ndim == 2 and (W.shape[0] < nt or W.shape[1] < ns):
                continue  # stale W matrix; skip rather than IndexError
            from neurons import LearningLayerBase as _LLB
            tgt_layer_obj = next((l for l in self.gui.circuit.layers if l.name == tgt), None)
            if isinstance(tgt_layer_obj, _LLB):
                self._draw_td_connection(src, tgt, W, ns, nt, positions,
                                         sel_layer=sel_layer, sel_idx=sel_idx)
                continue
            if max(ns, nt) > self._DENSE_THRESHOLD:
                self._draw_dense_connection(src, tgt, W, ns, nt, positions,
                                            sel_layer=sel_layer, sel_idx=sel_idx)
                continue
            for i in range(nt):
                if sel_idx is not None and tgt == sel_layer and i != sel_idx:
                    continue
                tn      = f'{tgt}_{i}'
                _p_tn   = positions.get(tn)
                tgt_hem = ((_p_tn[1] >= 0.5) if nt > 1 else None) if _p_tn is not None \
                          else self._hemisphere(i, nt)
                tgt_hem_map[tn] = tgt_hem
                for j in range(ns):
                    if sel_idx is not None and src == sel_layer and j != sel_idx:
                        continue
                    w = float(W[i, j]) if W.ndim == 2 else float(W.flat[0])
                    if abs(w) < 1e-10:
                        continue
                    # For combined pair connections, j≥n_L_d → nodes from the R half.
                    # R neurons appear in visual top-to-bottom order, which is reverse
                    # index order: column n_L → R_(n_R-1), column n_L+1 → R_(n_R-2), etc.
                    if _pair_lyr_d and j >= _n_L_d:
                        _n_R_d = _pair_lyr_d.n or 1
                        sn = f'{_pair_nm_d}_{self._reversed_idx(j - _n_L_d, _n_R_d, "R")}'
                    elif _pair_sensor_half_d and j >= _n_L_d:
                        _n_R_d = _n_L_d   # each side has the same n
                        sn = f'{_pair_sensor_half_d}_{self._reversed_idx(j - _n_L_d, _n_R_d, "R")}'
                    else:
                        sn = f'{src}_{j}'
                    if sn not in positions or tn not in positions:
                        continue
                    _p_sn   = positions.get(sn)
                    src_hem = ((_p_sn[1] >= 0.5) if ns > 1 else None) if _p_sn is not None \
                              else self._hemisphere(j, ns)
                    by_target[tn].append((sn, w, src_hem))

        for tn, incoming in by_target.items():
            p1      = positions[tn]
            k       = len(incoming)
            tgt_hem = tgt_hem_map[tn]
            incoming_sorted = sorted(incoming, key=lambda t: positions[t[0]][1], reverse=True)
            for idx, (sn, w, src_hem) in enumerate(incoming_sorted):
                p0  = positions[sn]
                t   = (idx / (k - 1) - 0.5) if k > 1 else 0.0
                sb  = self._signed_bow(src_hem, tgt_hem, self._CROSS_BOW,
                                       (p0[1] + p1[1]) / 2)
                self._draw_edge(p0, p1, w, sb, ctrl_perp=0.18 * t, sn=sn, tn=tn)

        for layer in self.gui.circuit.layers:
            if not hasattr(layer, 'internal_edges'):
                continue
            n      = layer.n or 2
            is_ring = getattr(layer, 'viz_layout', None) == 'ring'
            for fi, ti, w in layer.internal_edges():
                if sel_idx is not None and layer.name == sel_layer:
                    if fi != sel_idx and ti != sel_idx:
                        continue
                sn = f'{layer.name}_{fi}'
                tn = f'{layer.name}_{ti}'
                if sn not in positions or tn not in positions:
                    continue
                if is_ring:
                    # Bow each edge outward from the ring centre.
                    # _ctrl_pt with bow>0 shifts the midpoint by (cdy, -cdx),
                    # i.e. 90° CW from the chord. Choose the sign so that shift
                    # aligns with the outward direction (centre → chord midpoint).
                    p0  = positions[sn]
                    p1  = positions[tn]
                    cx  = getattr(layer, '_ring_cx', 0.5)
                    cy  = getattr(layer, '_ring_cy', 0.5)
                    cdx, cdy = p1[0] - p0[0], p1[1] - p0[1]
                    odx = (p0[0] + p1[0]) / 2 - cx   # chord-midpoint – centre
                    ody = (p0[1] + p1[1]) / 2 - cy
                    # dot( (cdy,-cdx), (odx,ody) ) > 0 → bow>0 is already outward
                    sb = self._INTERNAL_BOW if (cdy * odx - cdx * ody) >= 0 \
                         else -self._INTERNAL_BOW
                else:
                    sb = self._signed_bow(self._hemisphere(fi, n), self._hemisphere(ti, n),
                                          self._INTERNAL_BOW,
                                          (positions[sn][1] + positions[tn][1]) / 2)
                self._draw_edge(positions[sn], positions[tn], w, sb, sn=sn, tn=tn)

    def _draw_td_connection(self, src, tgt, W, ns, nt, positions,
                            sel_layer=None, sel_idx=None):
        """Draw all connections into a TDLayer — amber, always visible, ghost lines for zero weights."""
        amber_active = self._TD_EDGE + (210,)   # solid learned weight
        amber_zero   = self._TD_EDGE + (100,)   # ghost: unlearned slot (slightly transparent)
        for i in range(nt):
            if sel_idx is not None and tgt == sel_layer and i != sel_idx:
                continue
            tn_key = f'{tgt}_{i}'
            if tn_key not in positions:
                continue
            p_t = positions[tn_key]
            for j in range(ns):
                if sel_idx is not None and src == sel_layer and j != sel_idx:
                    continue
                sn_key = f'{src}_{j}'
                if sn_key not in positions:
                    continue
                w = float(W[i, j]) if W.ndim == 2 else float(W.flat[0])
                p_s = positions[sn_key]
                mid_y = (p_s[1] + p_t[1]) / 2
                sb = self._CROSS_BOW if mid_y >= 0.5 else -self._CROSS_BOW
                if abs(w) < 1e-10:
                    self._draw_edge(p_s, p_t, 1.0, sb, sn=sn_key, tn=tn_key,
                                    lw=1.0, color=amber_zero, style=Qt.DashLine,
                                    mark=False, tgt_gap=0.4)
                else:
                    st = Qt.SolidLine if w >= 0 else Qt.DashLine
                    self._draw_edge(p_s, p_t, w, sb, sn=sn_key, tn=tn_key,
                                    lw=3.0, color=amber_active, style=st,
                                    mark=True, marker_style='tick', tgt_gap=0.4)

    def _draw_dense_connection(self, src, tgt, W, ns, nt, positions,
                               sel_layer=None, sel_idx=None):
        """Draw all edges thin (solid excitatory, dashed inhibitory) for dense connections.

        Bow direction is determined purely by whether the edge midpoint is above or
        below y=0.5, giving visually symmetric arcs across the midline regardless of
        how neurons are indexed (important for ring layouts).
        When sel_layer/sel_idx are set, only draw edges involving that specific neuron.
        """
        for i in range(nt):
            if sel_idx is not None and tgt == sel_layer:
                if src == sel_layer:
                    pass  # self-connection: decide per (i,j) below
                elif i != sel_idx:
                    continue
            tn_key = f'{tgt}_{i}'
            if tn_key not in positions:
                continue
            p_t = positions[tn_key]
            for j in range(ns):
                if sel_idx is not None:
                    if src == sel_layer and tgt == sel_layer:
                        # self-connection: show only edges touching the selected neuron
                        if i != sel_idx and j != sel_idx:
                            continue
                    elif src == sel_layer and j != sel_idx:
                        continue
                sn_key = f'{src}_{j}'
                if sn_key not in positions:
                    continue
                w = float(W[i, j]) if W.ndim == 2 else float(W.flat[0])
                if abs(w) < 1e-10:
                    continue
                p_s = positions[sn_key]
                mid_y = (p_s[1] + p_t[1]) / 2
                sb = self._CROSS_BOW if mid_y >= 0.5 else -self._CROSS_BOW
                self._draw_edge(p_s, p_t, w, sb, sn=sn_key, tn=tn_key, lw=0.6)

    def _rebuild_edges(self):
        """Redraw all edges at the current viewPixelSize — called on every zoom/pan."""
        if self._building or self._rebuilding_edges:
            return
        self._rebuilding_edges = True
        try:
            saved = list(self._edge_params)
            for item in self._edge_items:
                try:
                    self._plot.removeItem(item)
                except Exception:
                    pass
            self._edge_items = []
            self._edge_items_tagged = []
            self._edge_params = []          # _draw_edge will repopulate
            for p in saved:
                self._draw_edge(*p)
            self._apply_edge_highlight()    # restore any active selection highlight
            if self._hidden_cols:
                self._apply_group_visibility()
        finally:
            self._rebuilding_edges = False

    def _draw_edge(self, p0, p1, weight, signed_bow, ctrl_perp=0.0, sn=None, tn=None, lw=None, color=None, style=None, mark=None, marker_style='circle', tgt_gap=0.0):
        self._edge_params.append((p0, p1, weight, signed_bow, ctrl_perp, sn, tn, lw, color, style, mark, marker_style, tgt_gap))
        excitatory = weight >= 0
        color      = color if color is not None else self._EDGE_POS
        lw         = lw if lw is not None else 3.0
        style      = style if style is not None else (Qt.SolidLine if excitatory else Qt.DashLine)

        # r_vis: actual scene-unit node radius at current zoom (nodes are pxMode=True,
        # size=NODE_R*240px, so pixel radius = NODE_R*120 → scene units = NODE_R*120*dy).
        try:
            _, _dy = self._vb.viewPixelSize()
        except Exception:
            _dy = 1.0 / 120.0
        r = self._NODE_R * 120 * _dy

        # Image-display nodes (cameras, Leaky2dLayer, Conv2dLayer pool='none') have a
        # much larger circular border drawn in data coords.  Use that radius for the
        # endpoint offset so arcs terminate at the circle rim, not at the node centre.
        _IMAGE_R = 0.12   # W_DATA/2 + 0.02 in data units — must match build block
        _img = getattr(self, '_img_node_keys', set())
        r_src = _IMAGE_R if sn in _img else r
        r_tgt_base = _IMAGE_R if tn in _img else r

        # Enforce a minimum bow so the arc clears the node circles.  The geometric
        # minimum ensures the bezier excursion > r; the 0.4 visual margin is added
        # only for same-column nodes (d < 4r) where a tiny bow would be invisible.
        r_bow = max(r_src, r_tgt_base)
        d_nodes = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if d_nodes > 0:
            geo_min = np.sqrt(max(0.0, (2 * r_bow / d_nodes) ** 2 - 1.0))
            visual_margin = 0.4 if d_nodes < 4 * r_bow else 0.0
            min_bow = geo_min + visual_margin
            if abs(signed_bow) < min_bow:
                if signed_bow != 0:
                    signed_bow = np.copysign(min_bow, signed_bow)
                else:
                    # Safety for any remaining zero-bow edge: preserve up/down
                    # symmetry by signing with source-relative-to-target y.
                    signed_bow = min_bow if p0[1] >= p1[1] else -min_bow

        ctrl = self._ctrl_pt(p0, p1, signed_bow)
        if ctrl_perp:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            d = np.hypot(dx, dy) + 1e-9
            ctrl = (ctrl[0] + ctrl_perp * (-dy / d),
                    ctrl[1] + ctrl_perp * (dx  / d))

        # Rim points: start/end aimed toward the control point.
        # tgt_gap > 0 stops the arc tgt_gap*r before the target rim (gap in scene units).
        d0 = np.hypot(ctrl[0] - p0[0], ctrl[1] - p0[1])
        d1 = np.hypot(ctrl[0] - p1[0], ctrl[1] - p1[1])
        r_tgt = r_tgt_base * (1.0 + tgt_gap)
        if d0 > 1e-6:
            p0r = (p0[0] + r_src * (ctrl[0] - p0[0]) / d0,
                   p0[1] + r_src * (ctrl[1] - p0[1]) / d0)
        else:
            nx, ny = p1[0] - p0[0], p1[1] - p0[1]
            nd = np.hypot(nx, ny) + 1e-9
            p0r = (p0[0] + r_src * nx / nd, p0[1] + r_src * ny / nd)
        if d1 > 1e-6:
            p1r = (p1[0] + r_tgt * (ctrl[0] - p1[0]) / d1,
                   p1[1] + r_tgt * (ctrl[1] - p1[1]) / d1)
        else:
            nx, ny = p0[0] - p1[0], p0[1] - p1[1]
            nd = np.hypot(nx, ny) + 1e-9
            p1r = (p1[0] + r_tgt * nx / nd, p1[1] + r_tgt * ny / nd)

        thin = lw < 1.5
        pts = self._bezier_pts(p0r, ctrl, p1r, n=25 if thin else 80)
        if len(pts) < 2:
            return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        edge_pen = pg.mkPen(color, width=lw, style=style)
        item = pg.PlotDataItem(xs, ys, pen=edge_pen)
        item.setZValue(3)
        self._plot.addItem(item)
        self._edge_items.append(item)
        self._edge_items_tagged.append((item, sn, tn, excitatory, True, edge_pen))

        do_mark = (not thin) if mark is None else mark
        if not do_mark:
            return

        ex, ey = pts[-1]
        ux, uy = ex - p1[0], ey - p1[1]
        ud = np.hypot(ux, uy)
        if ud > 1e-9:
            ux /= ud; uy /= ud
        m_pen_w = 3.0 if not thin else 2.0
        marker_pen = pg.mkPen(color, width=m_pen_w)

        if marker_style == 'tick':
            # Perpendicular tick at the target rim — PlotDataItem renders reliably
            perp_x, perp_y = -uy, ux
            tick_half = self._NODE_R * 55 * _dy
            tick_item = pg.PlotDataItem(
                [ex - tick_half * perp_x, ex + tick_half * perp_x],
                [ey - tick_half * perp_y, ey + tick_half * perp_y],
                pen=marker_pen,
            )
            tick_item.setZValue(6)
            self._plot.addItem(tick_item)
            self._edge_items.append(tick_item)
            self._edge_items_tagged.append((tick_item, sn, tn, excitatory, True, marker_pen))
        else:
            # Circle marker: filled dot tangent to the target node surface
            half_m = self._MARKER_R * 60 * _dy
            scatter = pg.ScatterPlotItem(
                [ex + half_m * ux], [ey + half_m * uy],
                size=self._MARKER_R * 120,
                brush=pg.mkBrush(color) if excitatory else pg.mkBrush(C['bg']),
                pen=marker_pen,
            )
            scatter.setZValue(6)
            self._plot.addItem(scatter)
            self._edge_items.append(scatter)
            self._edge_items_tagged.append((scatter, sn, tn, excitatory, False, marker_pen))

    def _infer_and_set_n(self):
        for conn in self.gui.circuit.connections:
            src, tgt, W = conn.src, conn.tgt, conn.W
            W = np.asarray(W, dtype=float)
            if W.ndim != 2:
                continue
            for layer in self.gui.circuit.layers:
                if layer.n is not None:
                    continue
                if layer.name == src:
                    layer._ensure_n(W.shape[1])
                elif layer.name == tgt:
                    layer._ensure_n(W.shape[0])

    def build(self):
        self._building = True
        self._gw.setUpdatesEnabled(False)
        try:
            self._build_inner()
        finally:
            self._building = False
            self._gw.setUpdatesEnabled(True)
            self._gw.update()
        # Redraw edges once the widget has settled to its final pixel size.
        QTimer.singleShot(0, self._rebuild_edges)
        # Reapply camera image rects after the event loop settles (in case
        # Qt reorders transforms during addItem or scene attachment).
        QTimer.singleShot(0, self._reapply_camera_rects)
        # Notify the app so arena overlays and oscilloscope channels stay in sync.
        if hasattr(self.gui, '_rebuild_channels'):
            self.gui._rebuild_channels()

    def _reapply_camera_rects(self):
        for key, item in self._camera_items.items():
            rect = self._camera_rects.get(key)
            if rect is not None:
                item.setRect(*rect)

    def _build_inner(self):
        self._infer_and_set_n()

        for item in self._edge_items + self._panel_items + self._text_items:
            try:
                self._plot.removeItem(item)
            except Exception:
                pass
            try:
                item.setParentItem(None)
            except Exception:
                pass
            try:
                if item.scene() is not None:
                    item.scene().removeItem(item)
            except Exception:
                pass
        # Keep existing camera ImageItems alive in the scene — reuse them in
        # the creation pass below rather than destroying and re-adding them.
        # Items for sensors that are no longer present are removed after the
        # creation loop.  This avoids the transform/scene detach race that
        # caused stale images to appear at wrong positions on rebuild.
        _old_cam_items = dict(self._camera_items)
        self._camera_items = {}
        self._camera_rects = {}
        if self._all_scatter is not None:
            try:
                self._plot.removeItem(self._all_scatter)
            except Exception:
                pass
            self._all_scatter = None
        if self._ring_scatter is not None:
            try:
                self._plot.removeItem(self._ring_scatter)
            except Exception:
                pass
            self._ring_scatter = None
        if self._deriv_scatter is not None:
            try:
                self._plot.removeItem(self._deriv_scatter)
            except Exception:
                pass
            self._deriv_scatter = None
        self._wave_phase = {}
        self._src_nodes  = {}
        self._rcv_nodes  = {}
        self._spot_names        = []
        self._spot_base_rgb     = {}
        self._spot_alpha        = {}
        self._spot_visible      = {}
        self._spot_pen_override = {}
        self._edge_items        = []
        self._edge_items_tagged = []
        self._edge_params       = []
        self._panel_items       = []
        self._panel_rect_map    = {}
        self._col_label_items   = {}
        self._text_items        = []
        self._text_map          = {}
        self._node_col_map      = {}
        self._highlighted_node  = None
        self._selected_edge     = None

        positions, sensor_nodes, groups = (
            self._layout_3d() if self._3d_mode else self._layout()
        )
        self._positions = positions
        n_map = self._n_map()

        self._col_x_map = {}
        for _, _, col_nodes, x_col, depth_val, *_ in groups:
            self._col_x_map[depth_val] = x_col
            for node_key in col_nodes:
                self._node_col_map[node_key] = depth_val

        # Compact layout: redistribute visible columns evenly, preserving the
        # coordinate space (x_unit) used by _layout() so spacing is consistent.
        if self._compact_mode and self._hidden_cols and not self._3d_mode:
            all_depths = sorted(self._col_x_map.keys())
            vis_depths = [d for d in all_depths if d not in self._hidden_cols]
            n_vis = len(vis_depths)
            if 0 < n_vis < len(all_depths):
                all_xs = [self._col_x_map[d] for d in all_depths]
                x_min, x_max = min(all_xs), max(all_xs)
                if n_vis == 1:
                    new_xs = {vis_depths[0]: (x_min + x_max) / 2}
                else:
                    step = (x_max - x_min) / (n_vis - 1)
                    new_xs = {dv: x_min + i * step
                              for i, dv in enumerate(vis_depths)}
                for node_key, dv in self._node_col_map.items():
                    if dv in new_xs and node_key in positions:
                        old_x, y = positions[node_key]
                        dx = new_xs[dv] - self._col_x_map[dv]
                        positions[node_key] = (old_x + dx, y)
                for dv, nx in new_xs.items():
                    self._col_x_map[dv] = nx

        if positions:
            xs = [p[0] for p in positions.values()]
            self._palette_x = (min(xs) + max(xs)) / 2
        else:
            self._palette_x = 0.5

        from sensors import CameraSensor as _CamSensor

        # Show "New Network" button only for DataBrain
        is_data_brain = isinstance(self.gui.brain, DataBrain) if self.gui.brain else False

        self._rebuild_group_buttons()
        if self._3d_mode:
            self._draw_plane_tabs(positions, groups)
        else:
            self._draw_panels(positions, groups)
        self._draw_edges(positions, n_map)

        # Reconnect zoom/pan handler so edges are redrawn whenever viewPixelSize changes.
        if self._range_signal_connected:
            try:
                self._vb.sigRangeChanged.disconnect(self._rebuild_edges_slot)
            except Exception:
                pass
        self._vb.sigRangeChanged.connect(self._rebuild_edges_slot)
        self._range_signal_connected = True

        layer_color = {
            layer.name: layer.color
            for layer in self.gui.circuit.layers
            if getattr(layer, 'color', None) is not None
        }

        # Sensors: hollow circles; palette colour used for border + activity fill.
        self._sensor_nodes      = sensor_nodes
        self._sensor_pens       = {}
        self._sensor_active_rgb = {}
        bg_rgb = self._hex_rgb(C['bg'])
        for i, sensor in enumerate(self.gui.circuit.sensors):
            col = getattr(sensor, '_viz_color', None) or _CHAN_PALETTE[i % len(_CHAN_PALETTE)]
            if _sensor_is_lateralized(sensor, self.gui.circuit):
                n_half = 1 if isinstance(sensor, _CamSensor) else (sensor.n or 1)
                for side in ('L', 'R'):
                    for j in range(n_half):
                        key = f'{sensor.name}_{side}_{j}'
                        self._sensor_pens[key]       = pg.mkPen(col, width=2.5)
                        self._sensor_active_rgb[key] = self._hex_rgb(col)
            else:
                for j in range(sensor.n_total or 0):
                    key = f'{sensor.name}_{j}'
                    self._sensor_pens[key]       = pg.mkPen(col, width=2.5)
                    self._sensor_active_rgb[key] = self._hex_rgb(col)

        camera_names = {s.name for s in self.gui.circuit.sensors
                        if isinstance(s, _CamSensor)}

        r = self._NODE_R
        _all_objs_init = {l.name: l for l in self.gui.circuit.layers}
        _all_objs_init.update({s.name: s for s in self.gui.circuit.sensors})
        for s in self.gui.circuit.sensors:
            if _sensor_is_lateralized(s, self.gui.circuit):
                _all_objs_init[f'{s.name}_L'] = s
                _all_objs_init[f'{s.name}_R'] = s
        spots = []
        for name, (x, y) in positions.items():
            layer_name = name.rsplit('_', 1)[0]
            if name in sensor_nodes:
                rgb   = bg_rgb
                brush = pg.mkBrush(C['bg'])
                pen_s = self._sensor_pens.get(name, pg.mkPen(C['primary'], width=2.5))
            else:
                fc    = layer_color.get(layer_name, C['primary'])
                rgb   = self._hex_rgb(fc)
                brush = pg.mkBrush(fc)
                pen_s = pg.mkPen(C['dark'], width=1.5)
            self._spot_names.append(name)
            self._spot_base_rgb[name]  = rgb
            self._spot_alpha[name]     = 1.0
            self._spot_visible[name]   = True
            _obj_init  = _all_objs_init.get(layer_name)
            _n_init    = getattr(_obj_init, 'n', 1) or 1
            _is_ring_i = getattr(_obj_init, 'viz_layout', None) == 'ring'
            _scale_i   = (self._RING_SCALE if _is_ring_i
                          else (self._DENSE_NODE_SCALE if _n_init > 4 else 1.0))
            spots.append({'pos': (x, y), 'size': r * _scale_i * 240,
                          'brush': brush, 'pen': pen_s,
                          'data': name})

            txt = pg.TextItem(name, color=C['dark'], anchor=(0.5, 0.5))
            txt.setPos(x, y)
            txt.setFont(QFont('Segoe UI', 6))
            txt.setZValue(8)
            self._plot.addItem(txt)
            self._text_items.append(txt)
            self._text_map[name] = txt

        self._all_scatter = pg.ScatterPlotItem()
        self._all_scatter.setData(spots=spots)
        self._all_scatter.setZValue(5)
        self._all_scatter.sigClicked.connect(self._on_spots_clicked)
        self._plot.addItem(self._all_scatter)

        # Camera image nodes — one pg.ImageItem per CameraSensor (or two for split).
        # W_DATA / H_DATA are display sizes in data coords, independent of sensor dims.
        # DISP_H must match the constant used in _update_camera_nodes.
        W_DATA, H_DATA, DISP_H = 0.18, 0.135, 32
        for sensor in self.gui.circuit.sensors:
            if not isinstance(sensor, _CamSensor):
                continue
            if getattr(sensor, 'lateralized', False):
                half = sensor.width // 2
                ovl  = getattr(sensor, 'overlap', 0)
                slices = {
                    'L': (0,          int(np.clip(half + ovl, 0, sensor.width))),
                    'R': (int(np.clip(half - ovl, 0, sensor.width)), sensor.width),
                }
                for side, (px0, px1) in slices.items():
                    pos = positions.get(f'{sensor.name}_{side}_0')
                    if pos is None:
                        continue
                    cx, cy = pos
                    _theta = np.linspace(0, 2 * np.pi, 65)
                    _r    = np.hypot(W_DATA / 2, H_DATA / 2)
                    _col  = getattr(sensor, '_viz_color', None) or '#888888'
                    _circ = pg.PlotDataItem(
                        cx + _r * np.cos(_theta), cy + _r * np.sin(_theta),
                        pen=pg.mkPen(_col, width=2.0),
                        fillLevel=cy - _r, brush=pg.mkBrush(C['bg']),
                    )
                    _circ.setZValue(5.5)
                    self._plot.addItem(_circ)
                    self._panel_items.append(_circ)
                    w_px = max(px1 - px0, 1)
                    key = f'{sensor.name}_{side}'
                    img_item = _old_cam_items.pop(key, None)
                    if img_item is None:
                        img_item = pg.ImageItem()
                        img_item.setZValue(6)
                        img_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                        self._plot.addItem(img_item, ignoreBounds=True)
                    blank = np.zeros((DISP_H, w_px, 3), dtype=np.uint8)
                    img_item.setImage(blank, axisOrder='row-major')
                    cam_rect = (cx - W_DATA / 2, cy - H_DATA / 2, W_DATA, H_DATA)
                    img_item.setRect(*cam_rect)
                    self._camera_items[key] = img_item
                    self._camera_rects[key] = cam_rect
                    lbl = pg.TextItem(key, color=C['dark'], anchor=(0.5, 1.0))
                    lbl.setPos(cx, cy + H_DATA / 2 + 0.02)
                    lbl.setFont(QFont('Segoe UI', 6))
                    lbl.setZValue(8)
                    self._plot.addItem(lbl)
                    self._text_items.append(lbl)
            else:
                pos = positions.get(f'{sensor.name}_0')
                if pos is None:
                    continue
                cx, cy = pos
                _theta = np.linspace(0, 2 * np.pi, 65)
                _r    = np.hypot(W_DATA / 2, H_DATA / 2)
                _col  = getattr(sensor, '_viz_color', None) or '#888888'
                _circ = pg.PlotDataItem(
                    cx + _r * np.cos(_theta), cy + _r * np.sin(_theta),
                    pen=pg.mkPen(_col, width=2.0),
                    fillLevel=cy - _r, brush=pg.mkBrush(C['bg']),
                )
                _circ.setZValue(5.5)
                self._plot.addItem(_circ)
                self._panel_items.append(_circ)
                key = sensor.name
                img_item = _old_cam_items.pop(key, None)
                if img_item is None:
                    img_item = pg.ImageItem()
                    img_item.setZValue(6)
                    img_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    self._plot.addItem(img_item, ignoreBounds=True)
                blank = np.zeros((DISP_H, sensor.width, 3), dtype=np.uint8)
                img_item.setImage(blank, axisOrder='row-major')
                cam_rect = (cx - W_DATA / 2, cy - H_DATA / 2, W_DATA, H_DATA)
                img_item.setRect(*cam_rect)
                self._camera_items[key] = img_item
                self._camera_rects[key] = cam_rect
                lbl = pg.TextItem(sensor.name, color=C['dark'], anchor=(0.5, 1.0))
                lbl.setPos(cx, cy + H_DATA / 2 + 0.02)
                lbl.setFont(QFont('Segoe UI', 6))
                lbl.setZValue(8)
                self._plot.addItem(lbl)
                self._text_items.append(lbl)

        # Leaky2dLayer image nodes — displayed like cameras (single image thumbnail).
        from neurons import Leaky2dLayer as _L2d
        for lyr in self.gui.circuit.layers:
            if not isinstance(lyr, _L2d):
                continue
            pos = positions.get(f'{lyr.name}_0')
            if pos is None:
                continue
            cx, cy = pos
            _theta = np.linspace(0, 2 * np.pi, 65)
            _r     = np.hypot(W_DATA / 2, H_DATA / 2)
            _col   = getattr(lyr, 'color', None) or '#888888'
            _circ  = pg.PlotDataItem(
                cx + _r * np.cos(_theta), cy + _r * np.sin(_theta),
                pen=pg.mkPen(_col, width=2.0),
                fillLevel=cy - _r, brush=pg.mkBrush(C['bg']),
            )
            _circ.setZValue(5.5)
            self._plot.addItem(_circ)
            self._panel_items.append(_circ)
            w_px = lyr.frame_w or max(1, int((lyr.n // max(lyr.in_ch, 1)) ** 0.5))
            key  = lyr.name
            img_item = _old_cam_items.pop(key, None)
            if img_item is None:
                img_item = pg.ImageItem()
                img_item.setZValue(6)
                img_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self._plot.addItem(img_item, ignoreBounds=True)
            blank = np.zeros((DISP_H, w_px, 3), dtype=np.uint8)
            img_item.setImage(blank, axisOrder='row-major')
            cam_rect = (cx - W_DATA / 2, cy - H_DATA / 2, W_DATA, H_DATA)
            img_item.setRect(*cam_rect)
            self._camera_items[key] = img_item
            self._camera_rects[key] = cam_rect
            lbl = pg.TextItem(lyr.name, color=C['dark'], anchor=(0.5, 1.0))
            lbl.setPos(cx, cy + H_DATA / 2 + 0.02)
            lbl.setFont(QFont('Segoe UI', 6))
            lbl.setZValue(8)
            self._plot.addItem(lbl)
            self._text_items.append(lbl)

        # Conv2dLayer pool='none' image nodes — displayed like cameras (single thumbnail).
        from neurons import Conv2dLayer as _Conv2d
        for lyr in self.gui.circuit.layers:
            if not isinstance(lyr, _Conv2d) or getattr(lyr, 'pool', '') != 'none':
                continue
            pos = positions.get(f'{lyr.name}_0')
            if pos is None:
                continue
            cx, cy = pos
            _theta = np.linspace(0, 2 * np.pi, 65)
            _r     = np.hypot(W_DATA / 2, H_DATA / 2)
            _col   = getattr(lyr, 'color', None) or '#888888'
            _circ  = pg.PlotDataItem(
                cx + _r * np.cos(_theta), cy + _r * np.sin(_theta),
                pen=pg.mkPen(_col, width=2.0),
                fillLevel=cy - _r, brush=pg.mkBrush(C['bg']),
            )
            _circ.setZValue(5.5)
            self._plot.addItem(_circ)
            self._panel_items.append(_circ)
            w_px = getattr(lyr, 'frame_w', None) or 1
            key  = lyr.name
            img_item = _old_cam_items.pop(key, None)
            if img_item is None:
                img_item = pg.ImageItem()
                img_item.setZValue(6)
                img_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self._plot.addItem(img_item, ignoreBounds=True)
            blank = np.zeros((DISP_H, max(w_px, 1), 3), dtype=np.uint8)
            img_item.setImage(blank, axisOrder='row-major')
            cam_rect = (cx - W_DATA / 2, cy - H_DATA / 2, W_DATA, H_DATA)
            img_item.setRect(*cam_rect)
            self._camera_items[key] = img_item
            self._camera_rects[key] = cam_rect
            lbl = pg.TextItem(lyr.name, color=C['dark'], anchor=(0.5, 1.0))
            lbl.setPos(cx, cy + H_DATA / 2 + 0.02)
            lbl.setFont(QFont('Segoe UI', 6))
            lbl.setZValue(8)
            self._plot.addItem(lbl)
            self._text_items.append(lbl)

        # Remove ImageItems for sensors/layers that are no longer in the circuit.
        for item in _old_cam_items.values():
            try:
                self._vb.removeItem(item)
            except Exception:
                pass
            try:
                if item.scene() is not None:
                    item.scene().removeItem(item)
            except Exception:
                pass

        # Derivative node markers: positions stored here; spots are updated
        # dynamically in _redraw_nodes using viewPixelSize() so the dot always
        # sits at the north of the node circle regardless of zoom level.
        layer_map = {l.name: l for l in self.gui.circuit.layers}
        self._deriv_node_positions = [
            (x, y)
            for name, (x, y) in positions.items()
            if getattr(layer_map.get(name.rsplit('_', 1)[0]), 'derivative', False)
        ]
        self._deriv_scatter = pg.ScatterPlotItem()
        self._deriv_scatter.setZValue(7)
        self._plot.addItem(self._deriv_scatter)

        self._mod_colors = {}
        for obj in list(self.gui.circuit.layers) + list(self.gui.circuit.sensors):
            nt = getattr(obj, 'neuromodulator_transmitter', None)
            nc = getattr(obj, 'neuromodulator_color', None)
            if nt and nc:
                self._mod_colors[nt] = self._hex_rgb(nc)
        self._mod_pens = {
            nt: pg.mkPen(QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)), width=4)
            for nt, rgb in self._mod_colors.items()
        }

        # Build source-node and receiver-node lookup tables; ring scatter draws both
        all_objs = {l.name: l for l in self.gui.circuit.layers}
        all_objs.update({s.name: s for s in self.gui.circuit.sensors})
        for s in self.gui.circuit.sensors:
            if _sensor_is_lateralized(s, self.gui.circuit):
                all_objs[f'{s.name}_L'] = s
                all_objs[f'{s.name}_R'] = s

        self._src_nodes = {}
        self._rcv_nodes = {}
        for node_key in self._spot_names:
            lname = node_key.rsplit('_', 1)[0]
            obj   = all_objs.get(lname)
            if obj is None:
                continue
            nt = getattr(obj, 'neuromodulator_transmitter', None)
            if nt and nt in self._mod_colors:
                self._src_nodes[node_key] = nt
                if nt not in self._wave_phase:
                    self._wave_phase[nt] = 0.0
            rcv = [mn for mn, _sc, _si in getattr(obj, 'modulators', [])
                   if mn in self._mod_colors]
            if rcv:
                self._rcv_nodes[node_key] = rcv

        self._ring_scatter = pg.ScatterPlotItem()
        self._ring_scatter.setZValue(4.5)
        self._plot.addItem(self._ring_scatter)

        if positions:
            vis_pos = [p for nk, p in positions.items()
                       if self._node_col_map.get(nk) not in self._hidden_cols]
            use = vis_pos if vis_pos else list(positions.values())
            xs = [p[0] for p in use]
            ys = [p[1] for p in use]
            self._plot.setRange(
                xRange=(min(xs) - 0.35, max(xs) + 0.25),
                yRange=(min(ys) - 0.30, max(ys) + 0.40),
                padding=0,
            )
        else:
            self._plot.setRange(xRange=(-0.25, 1.25), yRange=(-0.30, 1.40), padding=0)

        self._vb.disableAutoRange()
        self._apply_group_visibility()

    _Z_DX, _Z_DY = -0.20, 0.15   # must match _layout_3d constants

    def _draw_plane_tabs(self, _positions, groups):
        """Draw a labelled tab at the left edge of each Z plane in 3D mode."""
        z_map = getattr(self, '_z_map', {})
        z_planes = {}   # z → min_x in that plane
        for _, obj_name, _, x_col, *_ in groups:
            z = z_map.get(obj_name, 0)
            z_planes.setdefault(z, x_col)
            z_planes[z] = min(z_planes[z], x_col)

        palette = _CHAN_PALETTE
        self._plane_tab_z = {}   # will store (item, z) for click detection

        for z in sorted(z_planes):
            min_x = z_planes[z]
            tab_x = min_x - 0.12   # left of leftmost node in this Z plane
            tab_y = 0.50 + z * self._Z_DY
            col   = palette[z % len(palette)]
            z_labels = getattr(self, '_z_labels', {})
            label = z_labels.get(z, f'Z{z}')

            # Background rect behind label.
            pad = 0.035
            rect = pg.PlotDataItem(
                [tab_x - pad, tab_x + pad, tab_x + pad, tab_x - pad, tab_x - pad],
                [tab_y - pad, tab_y - pad, tab_y + pad, tab_y + pad, tab_y - pad],
                pen=pg.mkPen(col, width=1.5),
                brush=pg.mkBrush(QColor(col).lighter(160)),
                fillLevel=tab_y - pad,
            )
            rect.setZValue(2)
            self._plot.addItem(rect)
            self._panel_items.append(rect)

            txt = pg.TextItem(label, color=col, anchor=(0.5, 0.5))
            txt.setPos(tab_x, tab_y)
            txt.setFont(QFont('Segoe UI', 7, QFont.Bold))
            txt.setZValue(9)
            self._plot.addItem(txt)
            self._text_items.append(txt)

    def _draw_panels(self, positions, groups):
        r, px = self._NODE_R, self._PAD_X
        py    = 0.04
        col_data = defaultdict(lambda: {'ys': [], 'types': set(), 'x': 0.0})
        for col_type, _, col_nodes, x_col, depth_val, *_ in groups:
            ys = [positions[n][1] for n in col_nodes if n in positions]
            col_data[depth_val]['ys'].extend(ys)
            col_data[depth_val]['types'].add(col_type)
            col_data[depth_val]['x'] = self._col_x_map.get(depth_val, x_col)

        for depth_val, data in sorted(col_data.items()):
            ys = data['ys']
            if not ys:
                continue
            x_col    = data['x']
            col_type = 'sensor' if 'sensor' in data['types'] else 'layer'
            fc_s, ec_s = self._PANEL_CFG[col_type]
            rect_bot = min(ys) - r - py
            rect_top = max(ys) + r + py
            rect_item = pg.PlotDataItem(
                [x_col - r - px, x_col + r + px, x_col + r + px, x_col - r - px, x_col - r - px],
                [rect_bot, rect_bot, rect_top, rect_top, rect_bot],
                pen=pg.mkPen(ec_s, width=0.8),
                brush=pg.mkBrush(QColor(fc_s).lighter(110)),
                fillLevel=rect_bot,
            )
            rect_item.setZValue(1)
            self._plot.addItem(rect_item)
            self._panel_items.append(rect_item)
            self._panel_rect_map[depth_val] = rect_item

            label = self._col_labels.get(depth_val)
            if label:
                lbl_item = pg.TextItem(label, anchor=(0.5, 1.0),
                                       color=QColor(ec_s))
                lbl_item.setFont(_small_bold_font())
                lbl_item.setPos(x_col, rect_top + 0.01)
                lbl_item.setZValue(10)
                self._plot.addItem(lbl_item)
                self._col_label_items[depth_val] = lbl_item

    def update(self, brain):
        if self._building or self._all_scatter is None:
            return
        try:
            self._redraw_nodes(brain)
        except Exception:
            pass

    def _redraw_nodes(self, brain=None):
        """Rebuild all spot data and push to the single ScatterPlotItem in one call."""
        if self._redrawing or self._all_scatter is None or not self._spot_names:
            return
        self._redrawing = True
        try:
            active = self._ACTIVE_RGB
            r_node = self._NODE_R
            spots      = []
            all_layers  = {l.name: l for l in self.gui.circuit.layers}
            all_sensors = {s.name: s for s in self.gui.circuit.sensors}
            for name in self._spot_names:
                if not self._spot_visible.get(name, True):
                    continue
                if name not in self._positions:
                    continue
                x, y      = self._positions[name]
                base_rgb  = self._spot_base_rgb.get(name, (0.5, 0.5, 0.5))
                alpha     = self._spot_alpha.get(name, 1.0)
                layer_name = name.rsplit('_', 1)[0]

                is_muted  = getattr(all_layers.get(layer_name), 'muted', False)
                is_sensor = name in self._sensor_nodes

                if is_muted:
                    # Muted: flat grey, no activity colouring
                    base_rgb = (0.72, 0.72, 0.72)
                elif brain is not None:
                    try:
                        idx  = int(name.rsplit('_', 1)[1])
                        attr = getattr(brain, layer_name, None)
                        if attr is not None:
                            arr = np.atleast_1d(
                                attr.output if hasattr(attr, 'output') else attr)
                            if idx < len(arr):
                                val = float(np.clip(arr[idx], 0, 1))
                                act = active  # all nodes blend toward bright active orange
                                base_rgb = (
                                    base_rgb[0] + (act[0] - base_rgb[0]) * val,
                                    base_rgb[1] + (act[1] - base_rgb[1]) * val,
                                    base_rgb[2] + (act[2] - base_rgb[2]) * val,
                                )
                    except (ValueError, AttributeError):
                        pass

                a_int  = int(alpha * 255)
                brush  = pg.mkBrush(QColor(int(base_rgb[0] * 255),
                                           int(base_rgb[1] * 255),
                                           int(base_rgb[2] * 255), a_int))
                po = self._spot_pen_override.get(name)
                if po == 'selected':
                    pen = self._pen_selected
                elif po == 'multi':
                    pen = self._pen_multi
                elif is_muted:
                    pen = self._pen_muted
                elif is_sensor:
                    _s_nt = getattr(all_sensors.get(layer_name),
                                    'neuromodulator_transmitter', None)
                    if _s_nt and _s_nt in self._mod_pens:
                        pen = self._mod_pens[_s_nt]
                    else:
                        pen = self._sensor_pens.get(name, self._pen_default)
                elif layer_name in getattr(getattr(self.gui, '_osc_ctrl', None), '_osc_items', set()):
                    pen = self._pen_osc
                elif getattr(all_layers.get(layer_name), 'neuromodulator_transmitter', None) in self._mod_pens:
                    nt = all_layers[layer_name].neuromodulator_transmitter
                    pen = self._mod_pens[nt]
                else:
                    pen = self._pen_default
                obj = all_layers.get(layer_name)
                is_ring = getattr(obj, 'viz_layout', None) == 'ring'
                _n_rd   = getattr(obj, 'n', 1) or 1
                _scale  = (self._RING_SCALE if is_ring
                           else (self._DENSE_NODE_SCALE if _n_rd > 4 else 1.0))
                dot_r = r_node * _scale * 240
                spots.append({'pos': (x, y), 'size': dot_r,
                              'brush': brush, 'pen': pen, 'data': name})

            self._all_scatter.setData(spots=spots)

            # Derivative dot: recompute y_off each frame via viewPixelSize so the
            # dot always touches the north of the node circle at any zoom level.
            if self._deriv_scatter is not None and self._deriv_node_positions:
                r_d = r_node * 0.20   # slightly smaller than before
                try:
                    _, dy = self._vb.viewPixelSize()
                    y_off = (r_node - r_d) * 120 * dy
                except Exception:
                    y_off = (r_node - r_d)
                d_spots = [
                    {'pos': (x, y + y_off), 'size': r_d * 240,
                     'brush': pg.mkBrush(C['dark']), 'pen': pg.mkPen(None)}
                    for (x, y) in self._deriv_node_positions
                ]
                self._deriv_scatter.setData(spots=d_spots)
        finally:
            self._redrawing = False
        # Keep camera images pinned to their scatter nodes (guards against any
        # transform reset that may occur when setImage changes the image shape).
        for key, item in self._camera_items.items():
            rect = self._camera_rects.get(key)
            if rect is not None:
                item.setRect(*rect)

    def _on_spots_clicked(self, _plot, spots, ev):
        if len(spots):
            self._on_node_clicked(spots[0].data(), ev)

    # ── Undo ──────────────────────────────────────────────────────────────────

    def _push_undo(self):
        from brain_serializer import serialize_network_json
        snapshot = serialize_network_json(
            self.gui.circuit.sensors,
            self.gui.circuit.layers,
            self.gui.circuit.connections,
            self._hidden_cols,
            self._disabled_cols,
            self._col_labels,
            bodies=self.gui.circuit.bodies,
            joints=self.gui.circuit.joints,
            connection_params=self._weight_params,
        )
        self._undo_stack.append(snapshot)
        self._btn_undo.setEnabled(True)

    def _undo(self):
        if not self._undo_stack:
            return
        from brain_serializer import load_network_json
        snapshot = self._undo_stack.pop()
        sensors, layers, connections, hidden, disabled, col_labels, _bodies, _joints, conn_params = load_network_json(snapshot)
        self._weight_params = conn_params
        c = self.gui.circuit
        c.sensors     = sensors
        c.layers      = layers
        c.connections = connections
        self._hidden_cols   = hidden
        self._disabled_cols = disabled
        self._col_labels    = col_labels
        if hasattr(self.gui, 'brain') and self.gui.brain:
            brain = self.gui.brain
            for l in getattr(brain, 'layers', []):
                try:
                    delattr(brain, l.name)
                except AttributeError:
                    pass
            brain.layers      = layers
            brain.connections = connections
            for l in layers:
                setattr(brain, l.name, l)
        self._btn_undo.setEnabled(bool(self._undo_stack))
        self.build()

    # ── Edit-mode interactions ────────────────────────────────────────────────

    def _toggle_edit(self):
        self._edit_mode = not self._edit_mode
        if self._edit_mode:
            self._btn_edit.setStyleSheet(f"background:{C['primary']};color:white;")
            self._btn_save.setEnabled(True)
            self._btn_cols.setEnabled(False)
            self._btn_weights.setEnabled(False)
            self._palette_bar.setVisible(True)
            if self._col_panel.isVisible():
                self._toggle_col_win()
            if self._weight_panel.isVisible():
                self._toggle_weight_panel()
            self._clear_edge_highlight()
            self._clear_multi_selection()
        else:
            self._btn_edit.setStyleSheet("")
            self._btn_cols.setEnabled(True)
            self._btn_weights.setEnabled(True)
            self._palette_bar.setVisible(False)
            self._selected = None
            self._clear_edge_selection()
            self._clear_selection_highlight()

    def _on_compact_toggled(self, state):
        self._compact_mode = bool(state)
        self.build()

    def _toggle_col_win(self):
        panel_w = self._col_panel.sizeHint().width()
        if self._col_panel.isVisible():
            self._col_panel.setVisible(False)
            self.resize(self.width() - panel_w, self.height())
        else:
            self._col_panel.setVisible(True)
            self.resize(self.width() + panel_w, self.height())

    def _toggle_weight_panel(self):
        panel_w = self._weight_panel.sizeHint().width()
        if self._weight_panel.isVisible():
            self._weight_panel.setVisible(False)
            self.resize(self.width() - panel_w, self.height())
        else:
            self._weight_panel.setVisible(True)
            self.resize(self.width() + panel_w, self.height())

    def _toggle_weight_entry(self, src, tgt):
        key = (src, tgt)
        if key in self._weight_pinned:
            widget = self._weight_pinned.pop(key)
            self._weight_entries_layout.removeWidget(widget)
            widget.deleteLater()
        else:
            widget = WeightEntryWidget(src, tgt)
            idx = self._weight_entries_layout.count() - 1   # before trailing stretch
            self._weight_entries_layout.insertWidget(idx, widget)
            self._weight_pinned[key] = widget
            if not self._weight_panel.isVisible():
                self._toggle_weight_panel()
        self._update_weight_panel()

    def _update_weight_panel(self):
        if not self._weight_pinned:
            return
        conn_map = {(c.src, c.tgt): c for c in self.gui.circuit.connections}
        for (src, tgt), entry_widget in self._weight_pinned.items():
            conn = conn_map.get((src, tgt))
            if conn is None:
                continue
            W = np.asarray(conn.W, dtype=float)
            if W.ndim == 4:
                W = _tile_conv_filters(W)
            elif W.ndim != 2:
                W = W.reshape(1, -1)
            entry_widget.set_matrix(W)

    def _toggle_3d(self):
        self._3d_mode = not self._3d_mode
        if self._3d_mode:
            self._btn_3d.setStyleSheet(f"background:{C['primary']};color:white;")
            if self._edit_mode:
                self._toggle_edit()   # leave edit mode when entering 3D
        else:
            self._btn_3d.setStyleSheet("")
        self.build()

    def _clear_selection_highlight(self):
        self._spot_pen_override.clear()
        self._redraw_nodes()

    def _rebuild_group_buttons(self):
        while self._group_bar_lay.count() > 1:
            item = self._group_bar_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        sorted_depths = sorted(self._col_x_map.keys())
        _btn_ss = (
            "QPushButton{padding:0 4px;font-size:9px;"
            "border:1px solid #A0B0C0;border-radius:3px;background:#E8F0F8;}"
            "QPushButton:checked{color:white;}"
        )
        for i, depth_val in enumerate(sorted_depths):
            col_label = self._col_labels.get(depth_val, str(i + 1))
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(2)

            name_lbl = QLabel(col_label)
            name_lbl.setStyleSheet(f"font-size:9px;color:{C['dark']};")
            name_lbl.setMinimumWidth(30)
            row.addWidget(name_lbl, 1)

            btn_vis = QPushButton("H")
            btn_vis.setCheckable(True)
            btn_vis.setChecked(depth_val in self._hidden_cols)
            btn_vis.setEnabled(depth_val not in self._disabled_cols)
            btn_vis.setFixedSize(20, 18)
            btn_vis.setToolTip("Hide / show this column")
            btn_vis.setStyleSheet(
                _btn_ss + f"QPushButton:checked{{background:{C['muted']};}}"
            )
            btn_vis.toggled.connect(lambda checked, dv=depth_val: self._on_col_hide(dv, checked))
            row.addWidget(btn_vis)

            btn_dis = QPushButton("D")
            btn_dis.setCheckable(True)
            btn_dis.setChecked(depth_val in self._disabled_cols)
            btn_dis.setFixedSize(20, 18)
            btn_dis.setToolTip("Disable / enable this column")
            btn_dis.setStyleSheet(
                _btn_ss + "QPushButton:checked{background:#C0392B;color:white;}"
            )
            btn_dis.toggled.connect(lambda checked, dv=depth_val: self._on_col_disable(dv, checked))
            row.addWidget(btn_dis)

            row_w.setFixedHeight(22)
            self._group_bar_lay.insertWidget(i, row_w)

        has_any = bool(self._hidden_cols or self._disabled_cols)
        self._btn_show_all.setEnabled(bool(self._hidden_cols))
        self._btn_enable_all.setEnabled(bool(self._disabled_cols))

    def _on_col_hide(self, depth_val, hidden):
        if depth_val in self._disabled_cols:
            return  # disabled columns are always hidden
        if hidden:
            self._hidden_cols.add(depth_val)
        else:
            self._hidden_cols.discard(depth_val)
        if self._compact_mode:
            self.build()
        else:
            self._rebuild_group_buttons()
            self._apply_group_visibility()

    def _on_col_disable(self, depth_val, disabled):
        all_layers = {l.name: l for l in self.gui.circuit.layers}
        col_layers = [l for nk, dv in self._node_col_map.items()
                      if dv == depth_val
                      for l in [all_layers.get(nk.rsplit('_', 1)[0])]
                      if l is not None]
        col_layers_unique = list({id(l): l for l in col_layers}.values())
        if disabled:
            self._disabled_cols.add(depth_val)
            self._hidden_cols.add(depth_val)
            for l in col_layers_unique:
                l.muted = True
        else:
            self._disabled_cols.discard(depth_val)
            self._hidden_cols.discard(depth_val)
            for l in col_layers_unique:
                l.muted = False
        if self._compact_mode:
            self.build()
        else:
            self._rebuild_group_buttons()
            self._apply_group_visibility()

    def _on_show_all(self):
        all_layers = {l.name: l for l in self.gui.circuit.layers}
        for depth_val in list(self._disabled_cols):
            col_layers = [l for nk, dv in self._node_col_map.items()
                          if dv == depth_val
                          for l in [all_layers.get(nk.rsplit('_', 1)[0])]
                          if l is not None]
            for l in {id(lyr): lyr for lyr in col_layers}.values():
                l.muted = False
        self._hidden_cols.clear()
        self._disabled_cols.clear()
        self.build()

    def _on_enable_all(self):
        all_layers = {l.name: l for l in self.gui.circuit.layers}
        for depth_val in list(self._disabled_cols):
            col_layers = [l for nk, dv in self._node_col_map.items()
                          if dv == depth_val
                          for l in [all_layers.get(nk.rsplit('_', 1)[0])]
                          if l is not None]
            for l in {id(lyr): lyr for lyr in col_layers}.values():
                l.muted = False
        self._hidden_cols -= self._disabled_cols
        self._disabled_cols.clear()
        self.build()

    def _apply_group_visibility(self):
        for node_key in self._spot_names:
            dv = self._node_col_map.get(node_key)
            hidden = dv in self._hidden_cols
            self._spot_visible[node_key] = not hidden
            txt = self._text_map.get(node_key)
            if txt:
                txt.setVisible(not hidden)
        self._redraw_nodes()
        for item, sn, tn, *_ in self._edge_items_tagged:
            s_dv = self._node_col_map.get(sn)
            t_dv = self._node_col_map.get(tn)
            hidden = (s_dv in self._hidden_cols) or (t_dv in self._hidden_cols)
            item.setVisible(not hidden)
        for dv, rect_item in self._panel_rect_map.items():
            rect_item.setVisible(dv not in self._hidden_cols)
        for dv, lbl_item in self._col_label_items.items():
            lbl_item.setVisible(dv not in self._hidden_cols)

    def _panel_at(self, view_pt):
        """Return depth_val of the column panel the view-space point falls inside, or None."""
        r, px = self._NODE_R, self._PAD_X
        py = 0.04
        x, y = view_pt.x(), view_pt.y()
        col_data = {}
        for node_key, dv in self._node_col_map.items():
            if node_key not in self._positions:
                continue
            nx, ny = self._positions[node_key]
            if dv not in col_data:
                col_data[dv] = {'xs': [], 'ys': []}
            col_data[dv]['xs'].append(nx)
            col_data[dv]['ys'].append(ny)
        for dv, data in col_data.items():
            x_col = sum(data['xs']) / len(data['xs'])
            y_min = min(data['ys']) - r - py
            y_max = max(data['ys']) + r + py
            x_min = x_col - r - px
            x_max = x_col + r + px
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return dv
        return None

    def _show_col_label_menu(self, depth_val, screen_pos):
        from PySide6.QtWidgets import QInputDialog
        current = self._col_labels.get(depth_val, '')
        menu = QMenu(self)
        act_set   = menu.addAction("Set label…")
        act_clear = menu.addAction("Clear label")
        act_clear.setEnabled(bool(current))
        chosen = menu.exec(screen_pos)
        if chosen == act_set:
            text, ok = QInputDialog.getText(
                self, "Column label", "Label:", text=current)
            if ok:
                if text.strip():
                    self._col_labels[depth_val] = text.strip()
                else:
                    self._col_labels.pop(depth_val, None)
                self._refresh_col_label(depth_val)
        elif chosen == act_clear:
            self._col_labels.pop(depth_val, None)
            self._refresh_col_label(depth_val)

    def _refresh_col_label(self, depth_val):
        old = self._col_label_items.pop(depth_val, None)
        if old is not None:
            self._plot.removeItem(old)
        label = self._col_labels.get(depth_val)
        if label:
            r, px = self._NODE_R, self._PAD_X
            py = 0.04
            col_data_ys = [self._positions[nk][1]
                           for nk, dv in self._node_col_map.items()
                           if dv == depth_val and nk in self._positions]
            if col_data_ys:
                x_col = self._col_x_map.get(depth_val, 0.5)
                col_type_nodes = [nk for nk, dv in self._node_col_map.items()
                                  if dv == depth_val]
                is_sensor = any(nk in self._sensor_nodes for nk in col_type_nodes)
                _, ec_s = self._PANEL_CFG['sensor' if is_sensor else 'layer']
                rect_top = max(col_data_ys) + r + py
                lbl_item = pg.TextItem(label, anchor=(0.5, 1.0),
                                       color=QColor(ec_s))
                lbl_item.setFont(_small_bold_font())
                lbl_item.setPos(x_col, rect_top + 0.01)
                lbl_item.setZValue(10)
                self._plot.addItem(lbl_item)
                self._col_label_items[depth_val] = lbl_item
        self._rebuild_group_buttons()

    # ── Circuit highlighting (view mode only) ─────────────────────────────────

    _FADE_OPACITY = 0.12

    def _highlight_edges(self, center_node):
        self._highlighted_node = center_node
        center_layer = center_node.rsplit('_', 1)[0]

        # Build directed adjacency from cross-layer edges only.
        # Skipping within-layer edges (e.g. mutual inhibition) prevents them
        # from pulling in the entire contralateral circuit during BFS.
        fwd = defaultdict(set)   # source → {targets}
        bwd = defaultdict(set)   # target → {sources}
        for _, sn, tn, *_ in self._edge_items_tagged:
            if sn.rsplit('_', 1)[0] == tn.rsplit('_', 1)[0]:
                continue  # skip within-layer edges for traversal
            fwd[sn].add(tn)
            bwd[tn].add(sn)

        def _bfs(start, adj):
            visited, queue = {start}, [start]
            while queue:
                node = queue.pop()
                for nb in adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            return visited

        fwd_set = _bfs(center_node, fwd)
        bwd_set = _bfs(center_node, bwd)
        circuit = fwd_set | bwd_set

        # For dense layers (many neurons), include only the clicked neuron in the
        # same-layer set — this prevents all 64 edges of an 8×8 ring attractor from
        # lighting up when you click one neuron.  For small layers, include the whole
        # layer so mutual-inhibition edges are shown in full.
        n_map = self._n_map()
        n_center = n_map.get(center_layer, 1)
        if n_center > self._DENSE_THRESHOLD:
            same_layer = {center_node}
        else:
            same_layer = {n for n in self._spot_names
                          if n.rsplit('_', 1)[0] == center_layer}
        display_set = circuit | same_layer

        # Fade nodes and labels not in display_set
        for node_key in self._spot_names:
            self._spot_alpha[node_key] = (1.0 if node_key in display_set
                                          else self._FADE_OPACITY)
        for node_key, txt in self._text_map.items():
            txt.setOpacity(1.0 if node_key in display_set else self._FADE_OPACITY)
        self._redraw_nodes()

        # Highlight edges:
        # - within-layer: show if any endpoint is in display_set
        # - cross-layer:  show only if both endpoints are on the same traversal side
        #   (both in bwd_set OR both in fwd_set). This prevents "shortcut" edges
        #   (e.g. light→motor when viewing layer5) from lighting up just because
        #   their endpoints happen to be reachable via independent paths.
        for item, sn, tn, *_ in self._edge_items_tagged:
            if sn.rsplit('_', 1)[0] == tn.rsplit('_', 1)[0]:
                lit = (sn in display_set) or (tn in display_set)
            else:
                lit = ((sn in bwd_set and tn in bwd_set) or
                       (sn in fwd_set and tn in fwd_set))
            item.setOpacity(1.0 if lit else self._FADE_OPACITY)

    def _apply_edge_highlight(self):
        if self._selected_edge:
            self._highlight_selected_edge(*self._selected_edge)

    def _highlight_selected_edge(self, src, tgt):
        for item, sn, tn, _, is_curve, *_ in self._edge_items_tagged:
            if sn.rsplit('_', 1)[0] == src and tn.rsplit('_', 1)[0] == tgt:
                if is_curve:
                    item.setPen(pg.mkPen('#E07828', width=4.5))
                else:
                    item.setBrush(pg.mkBrush('#E07828'))
                    item.setPen(pg.mkPen('#E07828', width=3.0))

    def _clear_edge_selection(self):
        if not self._selected_edge:
            return
        src, tgt = self._selected_edge
        for item, sn, tn, excitatory, is_curve, original_pen in self._edge_items_tagged:
            if sn.rsplit('_', 1)[0] == src and tn.rsplit('_', 1)[0] == tgt:
                if is_curve:
                    item.setPen(original_pen)
                else:
                    orig_rgba = original_pen.color().getRgb()
                    item.setBrush(pg.mkBrush(orig_rgba) if excitatory else pg.mkBrush(C['bg']))
                    item.setPen(original_pen)
        self._selected_edge = None

    _EDGE_CLICK_DIST = 0.04  # view-space distance threshold for edge selection

    def _edge_at(self, view_pt):
        """Return (src_name, tgt_name) of the edge nearest to view_pt, or None."""
        px, py = view_pt.x(), view_pt.y()
        best = self._EDGE_CLICK_DIST
        result = None
        for item, sn, tn, _, is_curve, *_ in self._edge_items_tagged:
            if not is_curve:
                continue
            try:
                xs, ys = item.getData()
            except Exception:
                continue
            if xs is None or len(xs) < 2:
                continue
            for i in range(len(xs) - 1):
                d = self._dist_to_segment(px, py, xs[i], ys[i], xs[i+1], ys[i+1])
                if d < best:
                    best = d
                    result = (sn.rsplit('_', 1)[0], tn.rsplit('_', 1)[0])
        return result

    @staticmethod
    def _dist_to_segment(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        denom = dx*dx + dy*dy
        if denom == 0:
            return np.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1)*dx + (py - y1)*dy) / denom))
        return np.hypot(px - (x1 + t*dx), py - (y1 + t*dy))

    def _remove_selected_connection(self):
        if not self._selected_edge:
            return
        self._push_undo()
        self._pin_implicit_depths()
        src, tgt = self._selected_edge
        self.gui.circuit.connections = [
            c for c in self.gui.circuit.connections
            if not (c.src == src and c.tgt == tgt)
        ]
        self._sync_brain_to_circuit()
        self._selected_edge = None
        self.build()

    def _show_edge_edit(self, src, tgt):
        from sensors import CameraSensor as _CamSensor
        W_raw = None
        for c in self.gui.circuit.connections:
            if c.src == src and c.tgt == tgt:
                W_raw = np.asarray(c.W, dtype=float)
                break
        if W_raw is None:
            return

        # Resolve target layer and source sensor
        tgt_layer  = next((l for l in self.gui.circuit.layers  if l.name == tgt), None)
        src_sensor = next((s for s in self.gui.circuit.sensors if s.name == src), None)
        if src_sensor is None:
            src_sensor = next((s for s in self.gui.circuit.sensors
                               if s.name == src.rsplit('_', 1)[0]), None)

        is_conv2d = (W_raw.ndim == 4
                     or (tgt_layer is not None
                         and hasattr(tgt_layer, 'n_filters')
                         and hasattr(tgt_layer, 'kernel_size')))

        _src_is_lat_half = (src_sensor is not None
                            and _sensor_is_lateralized(src_sensor, self.gui.circuit)
                            and (src.endswith('_L') or src.endswith('_R')))

        if is_conv2d and tgt_layer is not None:
            kernel_size = tgt_layer.kernel_size
            if isinstance(src_sensor, _CamSensor):
                in_ch = getattr(src_sensor, 'in_ch', 1)
            else:
                in_ch = max(1, getattr(src_sensor, 'n', 1) if src_sensor else 1)
            is_lat_half = _src_is_lat_half
            W_init = W_raw if W_raw.ndim == 4 else None
            dlg = FilterStackDialog(self, src, tgt,
                                    in_ch, kernel_size, kernel_size, W_init,
                                    lateralized_half=is_lat_half)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            W, ok = dlg.get_result()
            if not ok:
                return
        else:
            W_cur = np.atleast_2d(W_raw)
            n_map  = self._n_map()
            ns     = n_map.get(src, W_cur.shape[1])
            nt     = n_map.get(tgt, W_cur.shape[0])
            # Combined joint-pair sensor connection: ns stored in W is n_L + n_R.
            if _src_is_lat_half and W_cur.shape[1] == ns * 2:
                ns = W_cur.shape[1]
            W, accepted, params = self._open_weight_dialog(
                src, tgt, ns, nt, W_cur, self._weight_params.get((src, tgt)))
            if not accepted:
                return
            if params is not None:
                self._weight_params[(src, tgt)] = params

        self._push_undo()
        from dataclasses import replace as _dc_replace
        new_conns = [_dc_replace(c, W=W) if c.src == src and c.tgt == tgt else c
                     for c in self.gui.circuit.connections]
        self.gui.circuit.connections = new_conns
        if hasattr(self.gui.brain, 'connections'):
            self.gui.brain.connections = new_conns
        self.build()

    def _clear_edge_highlight(self):
        self._highlighted_node = None
        for node_key in self._spot_names:
            self._spot_alpha[node_key] = 1.0
        for _, txt in self._text_map.items():
            txt.setOpacity(1.0)
        self._redraw_nodes()
        for item, *_ in self._edge_items_tagged:
            item.setOpacity(1.0)

    # ── Node drag (reorder depth) ─────────────────────────────────────────────

    def _get_snap_x(self, mouse_x):
        # Exclude hidden columns so drops never silently land in an invisible column.
        vis_xs = sorted(cx for d, cx in self._col_x_map.items()
                        if d not in self._hidden_cols)
        col_xs = vis_xs if vis_xs else sorted(self._col_x_map.values())
        if not col_xs:
            return mouse_x
        nearest = min(col_xs, key=lambda cx: abs(cx - mouse_x))
        if abs(nearest - mouse_x) < self._x_unit * 0.4:
            return nearest
        for a, b in zip(col_xs, col_xs[1:]):
            if a < mouse_x < b:
                return (a + b) / 2
        # Beyond the edge columns — allow creating a new column outside the existing range.
        if mouse_x < col_xs[0]:
            return col_xs[0] - self._x_unit
        if mouse_x > col_xs[-1]:
            return col_xs[-1] + self._x_unit
        return nearest

    def _snap_is_existing(self, snap_x):
        return any(abs(snap_x - cx) < 1e-9
                   for d, cx in self._col_x_map.items()
                   if d not in self._hidden_cols)

    def _show_drag_indicator(self, snap_x, mouse_y=None, layer=None):
        # Same column and y provided → horizontal slot indicator.
        if layer is not None and mouse_y is not None:
            # Use the layer's visual x position rather than its raw .layer depth
            # so that layers with layer=None (e.g. the _L half of a lateralized
            # pair) are detected as same-column when they should be.
            my_x = next(
                (self._positions[f'{layer.name}_{j}'][0]
                 for j in range(layer.n or 0)
                 if f'{layer.name}_{j}' in self._positions),
                None,
            )
            col_x = my_x
            if col_x is not None and abs(snap_x - col_x) < 1e-9:
                snap_y, _ = self._get_col_snap_y(layer, mouse_y)
                self._h_drag_indicator.setValue(snap_y)
                self._h_drag_indicator.setVisible(True)
                self._drag_indicator.setVisible(False)
                return
        self._drag_indicator.setValue(snap_x)
        self._drag_indicator.setVisible(True)
        self._h_drag_indicator.setVisible(False)

    def _hide_drag_indicator(self):
        self._drag_indicator.setVisible(False)
        self._h_drag_indicator.setVisible(False)

    def _col_mates(self, layer):
        """Return all layers that share the same visual column as *layer*.

        Uses the current ``_positions`` dict rather than the raw ``.layer``
        attribute so that layers with ``layer=None`` (e.g. the _L half of a
        lateralized pair, whose column is decided by enforcement at layout time)
        are found correctly.
        """
        my_x = None
        for j in range(layer.n or 0):
            pos = self._positions.get(f'{layer.name}_{j}')
            if pos is not None:
                my_x = pos[0]
                break
        if my_x is None:
            return []
        return [
            l for l in self.gui.circuit.layers
            if l is not layer and (l.n or 0) > 0
            and any(
                abs((self._positions.get(f'{l.name}_{k}') or (float('inf'),))[0] - my_x) < 1e-9
                for k in range(l.n or 0)
            )
        ]

    def _get_col_snap_y(self, layer, mouse_y):
        """Snap mouse_y to the nearest insertion boundary within layer's column.
        Returns (snap_y, insert_idx) where insert_idx=0 means top slot."""
        others = [
            l for l in self._col_mates(layer)
            if (l.n or 0) > 0
        ]

        def top_y(l):
            # Bilateral layers are symmetric around 0.5, so avg_y is always 0.5.
            # Use the top (max y) neuron as the layer's representative position.
            ys = [self._positions[f'{l.name}_{j}'][1]
                  for j in range(l.n or 0)
                  if f'{l.name}_{j}' in self._positions]
            return max(ys) if ys else 0.5

        centers = sorted([top_y(l) for l in others], reverse=True)  # top-first

        if not centers:
            return (mouse_y, 0)

        spacing = (centers[0] - centers[-1]) / len(centers) if len(centers) > 1 else 0.2
        half = spacing / 2

        boundaries = [centers[0] + half] + \
                     [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)] + \
                     [centers[-1] - half]

        insert_idx = min(range(len(boundaries)), key=lambda i: abs(boundaries[i] - mouse_y))
        return (boundaries[insert_idx], insert_idx)

    def _insert_midpoint_depth(self, snap_x, exclude=None):
        """Return integer depth for a new slot between the two visible columns bracketing snap_x.
        If the natural insert depth is hidden or consecutive with the right bound, shifts all
        explicit layer/sensor depths (and hidden_cols) at/above that point by +1 to free the slot."""
        # Use only visible columns for bracketing so hidden columns are transparent to the user.
        vis_pairs = sorted(
            ((d, x) for d, x in self._col_x_map.items() if d not in self._hidden_cols),
            key=lambda dx: dx[1],
        )
        if not vis_pairs:
            return 1
        col_xs         = [x for _, x in vis_pairs]
        existing_depths = [d for d, _ in vis_pairs]

        # New column to the left of the leftmost visible column.
        if snap_x < col_xs[0]:
            new_d = existing_depths[0] - 1
            if new_d <= 0:
                for obj in list(self.gui.circuit.layers) + list(self.gui.circuit.sensors):
                    if obj.name == exclude:
                        continue
                    if getattr(obj, 'layer', None) is not None:
                        obj.layer += 1
                self._hidden_cols = {d + 1 for d in self._hidden_cols}
                return existing_depths[0]
            return new_d
        # New column to the right of the rightmost visible column.
        if snap_x > col_xs[-1]:
            return existing_depths[-1] + 1

        for i, (a, b) in enumerate(zip(col_xs, col_xs[1:])):
            if a <= snap_x <= b:
                left_d  = existing_depths[i]
                right_d = existing_depths[i + 1]
                mid_d   = int(left_d) + 1
                if mid_d not in self._hidden_cols and right_d - left_d > 1:
                    # A free, visible slot already exists — use it directly.
                    return mid_d
                # Either the slot is occupied by a hidden column, or left/right are
                # consecutive.  Shift everything from mid_d upward to free the slot.
                for obj in list(self.gui.circuit.layers) + list(self.gui.circuit.sensors):
                    if obj.name == exclude:
                        continue
                    d = getattr(obj, 'layer', None)
                    if d is not None and d >= mid_d:
                        obj.layer = d + 1
                self._hidden_cols = {d + 1 if d >= mid_d else d
                                     for d in self._hidden_cols}
                return mid_d
        return None

    def _on_node_drag_drop(self, snap_x, mouse_y=None):
        if not self._selected:
            return
        self._push_undo()
        item_name = self._selected.rsplit('_', 1)[0]
        layer  = next((l for l in self.gui.circuit.layers  if l.name == item_name), None)
        sensor = next((s for s in self.gui.circuit.sensors if s.name == item_name), None)
        obj    = layer or sensor
        if obj is None:
            # Lateralized camera halves: 'camera_L_0' → item_name='camera_L' → parent 'camera'
            parent_name = item_name.rsplit('_', 1)[0]
            sensor = next((s for s in self.gui.circuit.sensors if s.name == parent_name), None)
            obj = sensor
        if obj is None:
            return

        current_depth = getattr(obj, 'layer', None)

        if self._snap_is_existing(snap_x):
            target_depth = next(
                (d for d, x in self._col_x_map.items() if abs(x - snap_x) < 1e-9), None
            )
        else:
            target_depth = self._insert_midpoint_depth(snap_x, exclude=item_name)

        # Same column and y available → vertical reorder within column.
        if target_depth == current_depth and mouse_y is not None and layer is not None:
            self._reorder_layer_by_y(layer, mouse_y)
            return

        if target_depth is not None:
            obj.layer = target_depth
        self._compact_depths()
        self._build_without_selection_filter()

    def _build_without_selection_filter(self):
        """Call build() with _selected cleared so _draw_edges draws all connections,
        then restore the selection highlight for visual feedback.

        _draw_edges uses _selected to filter edges to only the selected neuron's
        index, which is useful for interactive inspection but must not suppress
        unrelated connections after a drag operation.
        """
        sel = self._selected
        self._selected = None
        self.build()
        self._selected = sel
        if sel and sel in self._positions:
            self._spot_pen_override[sel] = 'selected'
            self._redraw_nodes()

    def _reorder_layer_by_y(self, layer, mouse_y):
        """Insert `layer` at the vertical slot nearest to `mouse_y` within its column."""
        # Note: _push_undo() already called by _on_node_drag_drop before delegation here.
        _, insert_idx = self._get_col_snap_y(layer, mouse_y)

        others = self._col_mates(layer)

        def top_y(l):
            ys = [self._positions[f'{l.name}_{j}'][1]
                  for j in range(l.n or 0)
                  if f'{l.name}_{j}' in self._positions]
            return max(ys) if ys else 0.5

        others.sort(key=top_y, reverse=True)  # top-first (highest y = top)

        ordered = others[:insert_idx] + [layer] + others[insert_idx:]
        for i, l in enumerate(ordered):
            l.viz_row = i

        self._build_without_selection_filter()

    # ── Motif palette ─────────────────────────────────────────────────────────

    def _save_as_motif(self):
        from PySide6.QtWidgets import QInputDialog
        from brain_serializer import _sensor_to_dict
        layer_names = {n.rsplit('_', 1)[0] for n in self._multi_selected}
        if not layer_names:
            return
        layers = [l for l in self.gui.circuit.layers if l.name in layer_names]
        # Include connections where at least one endpoint is a selected layer;
        # this captures sensor→layer edges as well as layer→layer edges.
        all_names = layer_names | {s.name for s in self.gui.circuit.sensors}
        conns = [c for c in self.gui.circuit.connections
                 if c.tgt in layer_names and c.src in (layer_names | all_names)]
        # Save sensors whose output is consumed by the selected layers.
        src_names = {c.src for c in conns}
        sensors = [s for s in self.gui.circuit.sensors if s.name in src_names]
        payload = {
            'sensors':     [_sensor_to_dict(s) for s in sensors],
            'layers':      [_layer_to_dict(l) for l in layers],
            'connections': [_connection_to_dict(c, self._weight_params.get((c.src, c.tgt)))
                            for c in conns],
        }
        name, ok = QInputDialog.getText(self, "Save motif", "Motif name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        os.makedirs(MOTIFS_DIR, exist_ok=True)
        with open(os.path.join(MOTIFS_DIR, f'{name}.json'), 'w') as fh:
            json.dump(payload, fh, indent=2)
        self._reload_motifs_palette()

    def _insert_motif(self, name, snap_x):
        from brain_serializer import _sensor_from_dict
        path = os.path.join(MOTIFS_DIR, f'{name}.json')
        try:
            with open(path) as fh:
                payload = json.load(fh)
            if 'layers' not in payload:
                return
        except Exception:
            return

        existing_layers  = {l.name: l for l in self.gui.circuit.layers}
        existing_sensors = {s.name: s for s in self.gui.circuit.sensors}

        # Validate before touching anything: size mismatch on a name collision is fatal.
        for ld in payload['layers']:
            lname   = ld['name']
            motif_n = ld.get('n', 1)
            existing = existing_layers.get(lname) or existing_sensors.get(lname)
            if existing is not None:
                circuit_n = getattr(existing, 'n', 1)
                if motif_n != circuit_n:
                    QMessageBox.warning(
                        self, "Motif import failed",
                        f"'{lname}' already exists with size {circuit_n} "
                        f"but the motif requires size {motif_n}."
                    )
                    return

        self._push_undo()

        # Add sensors that are not already in the circuit.
        brain = self.gui.brain if self.gui else None
        for sd in payload.get('sensors', []):
            sname = sd.get('name')
            if sname and sname not in existing_sensors:
                sensor = _sensor_from_dict(sd)
                sensor.reset()
                self.gui.circuit.sensors.append(sensor)
                existing_sensors[sname] = sensor
                if brain is not None:
                    setattr(brain, sname, np.zeros(sensor.n or 1))

        # Only add layers that are not already in the circuit.
        layers_to_add = [ld for ld in payload['layers']
                         if ld['name'] not in existing_layers
                         and ld['name'] not in existing_sensors]

        new_layers = []
        if layers_to_add:
            if self._snap_is_existing(snap_x):
                target_depth = next(
                    (d for d, x in self._col_x_map.items() if abs(x - snap_x) < 1e-9), 1
                )
            else:
                target_depth = self._insert_midpoint_depth(snap_x, exclude=None)

            pasted_depths = [ld.get('layer') for ld in layers_to_add
                             if ld.get('layer') is not None]
            depth_offset  = target_depth - (min(pasted_depths) if pasted_depths else target_depth)

            for ld in layers_to_add:
                ld = dict(ld)
                if ld.get('layer') is not None:
                    ld['layer'] = ld['layer'] + depth_offset
                layer = _layer_from_dict(ld)
                layer.reset()
                self.gui.circuit.layers.append(layer)
                new_layers.append(layer)

        # Add connections, skipping any pair that already exists.
        existing_pairs = {(c.src, c.tgt) for c in self.gui.circuit.connections}
        for cd in payload.get('connections', []):
            src, tgt = cd['src'], cd['tgt']
            if src and tgt and (src, tgt) not in existing_pairs:
                from circuit_model import Connection as _Conn
                self.gui.circuit.connections.append(
                    _Conn(src, tgt, np.array(cd['W'], dtype=float),
                          learning=cd.get('learning'), lr=cd.get('lr', 0.01))
                )
                if cd.get('params'):
                    self._weight_params[(src, tgt)] = cd['params']
                existing_pairs.add((src, tgt))

        # Sync new layers onto the brain so step_network can find them by name
        # and _redraw_nodes can read their output for activity colouring.
        if brain is not None:
            for layer in new_layers:
                setattr(brain, layer.name, layer)
            brain.layers      = self.gui.circuit.layers
            brain.connections = self.gui.circuit.connections
            brain.sensors     = self.gui.circuit.sensors

        self.build()

    def _reload_motifs_palette(self):
        lay = self._motifs_palette_layout
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        os.makedirs(MOTIFS_DIR, exist_ok=True)
        names = sorted(f[:-5] for f in os.listdir(MOTIFS_DIR) if f.endswith('.json'))
        if not names:
            self._motifs_palette_widget.setVisible(False)
            return
        lbl = QLabel("Motifs:")
        lbl.setStyleSheet(f"color:{C['muted']};font-size:9px;min-width:40px;")
        lay.addWidget(lbl)
        for name in names:
            chip = PaletteChip(name, f'motif:{name}')
            chip.setStyleSheet(
                "QPushButton{padding:0 8px;font-size:9px;"
                "border:1px solid #C0A0C8;border-radius:3px;background:#EEE4F4;}"
                "QPushButton:hover{background:#E0D0F0;}"
            )
            lay.addWidget(chip)
        self._motifs_palette_widget.setVisible(True)

    # ── Palette drop (add layer via drag-and-drop) ────────────────────────────

    def eventFilter(self, obj, ev):
        if obj is not self._gw:
            return False
        t = ev.type()
        if t == QEvent.Type.DragEnter:
            if self._edit_mode and ev.mimeData().hasText():
                ev.acceptProposedAction()
                return True
        elif t == QEvent.Type.DragMove:
            if self._edit_mode and ev.mimeData().hasText():
                vb_pt = self._vb.mapSceneToView(
                    self._gw.mapToScene(ev.position().toPoint())
                )
                self._show_drag_indicator(self._get_snap_x(vb_pt.x()))
                ev.acceptProposedAction()
                return True
        elif t == QEvent.Type.DragLeave:
            self._hide_drag_indicator()
            return True
        elif t == QEvent.Type.Drop:
            self._hide_drag_indicator()
            if self._edit_mode and ev.mimeData().hasText():
                vb_pt = self._vb.mapSceneToView(
                    self._gw.mapToScene(ev.position().toPoint())
                )
                snap_x = self._get_snap_x(vb_pt.x())
                self._on_palette_drop(ev.mimeData().text(), snap_x)
                ev.acceptProposedAction()
                return True
        return False

    def _on_palette_drop(self, mime_text, snap_x):
        if mime_text.startswith('sensor:'):
            if self._snap_is_existing(snap_x):
                sensor_target_depth = next(
                    (d for d, x in self._col_x_map.items() if abs(x - snap_x) < 1e-9), 0
                )
            else:
                sensor_target_depth = self._insert_midpoint_depth(snap_x, exclude=None)
            self._add_sensor_dialog(mime_text[len('sensor:'):], target_depth=sensor_target_depth)
            return
        if mime_text.startswith('motif:'):
            self._insert_motif(mime_text[len('motif:'):], snap_x)
            return
        if mime_text == 'joint':
            if self.gui is not None:
                self.gui._add_joint()
            return

        ltype = mime_text
        if self._snap_is_existing(snap_x):
            target_depth = next(
                (d for d, x in self._col_x_map.items() if abs(x - snap_x) < 1e-9), 1
            )
        else:
            target_depth = self._insert_midpoint_depth(snap_x, exclude=None)

        prev_count = len(self.gui.circuit.layers)
        self._add_layer_dialog(ltype)
        if len(self.gui.circuit.layers) > prev_count:
            self.gui.circuit.layers[-1].layer = target_depth
            self.build()

    def _make_body_combo(self, bodies, current_body_ids=None):
        """Build a QComboBox for sensor body selection.

        Shows 'root', individual singleton bodies, and one combined entry per
        mirror group (e.g. 'body1  (pair)').  currentData() returns a list of
        body IDs so callers can assign sensor.body_ids directly.
        """
        combo = QComboBox()
        combo.addItem("root", ['root'])
        seen_groups = set()
        for b in bodies:
            if b.id == 'root':
                continue
            mg = getattr(b, 'mirror_group', '')
            if mg:
                if mg not in seen_groups:
                    seen_groups.add(mg)
                    group_ids = [x.id for x in bodies
                                 if getattr(x, 'mirror_group', '') == mg]
                    combo.addItem(f"{mg}  (pair)", group_ids)
            else:
                combo.addItem(b.name, [b.id])
        if current_body_ids:
            for i in range(combo.count()):
                if combo.itemData(i) == current_body_ids:
                    combo.setCurrentIndex(i)
                    break
        return combo

    # ── Dialog shell helpers ──────────────────────────────────────────────────

    @staticmethod
    def _show_help_window(parent, title, markdown_text):
        """Open an embedded Chromium dialog with help rendered via marked.js + KaTeX.

        Falls back to the system browser if PySide6-WebEngine is not installed.
        """
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            from PySide6.QtCore import QUrl
        except ImportError:
            import tempfile, webbrowser
            html = _make_help_html(markdown_text)
            with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False,
                                            encoding='utf-8') as fh:
                fh.write(html)
                path = fh.name
            webbrowser.open('file:///' + path.replace('\\', '/'))
            return

        html = _make_help_html(markdown_text)
        d = QDialog(parent)
        d.setWindowTitle(title + " — Help")
        d.resize(720, 600)
        lay = QVBoxLayout(d)
        lay.setContentsMargins(0, 0, 0, 8)
        view = QWebEngineView()
        view.setHtml(html, QUrl('https://cdn.jsdelivr.net'))
        lay.addWidget(view)
        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn.rejected.connect(d.reject)
        lay.addWidget(btn)
        d.exec()

    def _make_param_dialog(self, title, help_text=None):
        """Build the standard dialog shell used by all param-editing dialogs.

        Returns (dlg, form, status_lbl).  form is already added to the dialog
        layout; the caller just adds rows to it.
        """
        dlg   = QDialog(self)
        dlg.setWindowTitle(title)
        outer = QVBoxLayout(dlg)
        outer.setSpacing(2)
        if help_text:
            help_row = QHBoxLayout()
            help_row.addStretch()
            help_btn = QPushButton("?")
            help_btn.setFixedSize(22, 22)
            help_btn.setToolTip("Help")
            help_btn.clicked.connect(
                lambda: self._show_help_window(dlg, title, help_text))
            help_row.addWidget(help_btn)
            outer.addLayout(help_row)
        form = QFormLayout()
        outer.addLayout(form)
        status_lbl = QLabel("")
        status_lbl.setFixedHeight(18)
        status_lbl.setStyleSheet("color:#888;font-style:italic;padding:0 4px;")
        outer.addWidget(status_lbl)
        dlg._filters = []
        return dlg, form, status_lbl

    def _receptor_table_widget(self, existing_mods):
        """Build the modulator receptors table and its add/remove buttons.

        Returns (table, btns_widget) ready to pass to form.addRow().
        """
        _SITES = ['pre', 'post', 'none']

        def _make_site_combo(current='post'):
            cb = QComboBox()
            cb.addItems(_SITES)
            cb.setCurrentText(current if current in _SITES else 'post')
            return cb

        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Modulator", "Scale", "Site"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setFixedHeight(140)
        for mod_name, scale, site in (existing_mods or []):
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(mod_name))
            table.setItem(row, 1, QTableWidgetItem(str(scale)))
            table.setCellWidget(row, 2, _make_site_combo(site))

        btns = QWidget()
        lay  = QHBoxLayout(btns)
        lay.setContentsMargins(0, 0, 0, 0)
        btn_add = QPushButton("Add Receptor")
        btn_rem = QPushButton("Remove Selected")

        def _add():
            r = table.rowCount()
            table.insertRow(r)
            table.setItem(r, 0, QTableWidgetItem(''))
            table.setItem(r, 1, QTableWidgetItem('1.0'))
            table.setCellWidget(r, 2, _make_site_combo('post'))
            table.editItem(table.item(r, 0))

        def _remove():
            for r in sorted({i.row() for i in table.selectedIndexes()}, reverse=True):
                table.removeRow(r)

        btn_add.clicked.connect(_add)
        btn_rem.clicked.connect(_remove)
        lay.addWidget(btn_add)
        lay.addWidget(btn_rem)
        lay.addStretch()
        return table, btns

    def _build_sensor_param_editors(self, form, dlg, status_lbl, params, joints, bodies, is_proprio, cur_values):
        """Build param editor widgets from param_defs and add them to form.

        cur_values: {pname: current_value} — use param default when key is absent.
        Returns editors dict: {pname: (kind_or_ptype, widget, is_angle)}.
        """
        editors = {}
        for param in params:
            pname, ptype, default, desc = param[:4]
            choices = param[4] if len(param) > 4 else None
            cur = cur_values.get(pname, default)

            if pname == 'joint_id' and is_proprio:
                w = QComboBox()
                seen = {}
                for jt in joints:
                    if jt.motor_layer_name in seen:
                        continue
                    seen[jt.motor_layer_name] = True
                    group = [j for j in joints if j.motor_layer_name == jt.motor_layer_name]
                    label = (jt.motor_layer_name if len(group) > 1
                             else next((b.name for b in bodies if b.id == jt.child_id),
                                       jt.motor_layer_name))
                    w.addItem(label, jt.motor_layer_name)
                idx = w.findData(cur or '')
                if idx >= 0:
                    w.setCurrentIndex(idx)
                form.addRow("joint", w)
                editors[pname] = ('joint_combo', w, False)
                continue

            if choices is not None:
                w = QComboBox()
                for ch in choices:
                    w.addItem(ch)
                idx = w.findText(str(cur))
                if idx >= 0:
                    w.setCurrentIndex(idx)
                form.addRow(pname, w)
                f = _HoverStatus(status_lbl, desc, dlg)
                w.installEventFilter(f)
                dlg._filters.append(f)
                editors[pname] = ('choice_combo', w, False)
                continue

            is_angle = pname in self._ANGLE_PARAMS
            if is_angle and isinstance(cur, float):
                cur = round(np.degrees(cur), 4)

            if ptype == bool:
                w = QCheckBox()
                w.setChecked(bool(cur))
            elif ptype == int:
                w = QSpinBox()
                w.setMinimum(0); w.setMaximum(1000)
                try:    w.setValue(int(cur))
                except Exception: pass
            else:
                w = QLineEdit(str(cur) if cur != '' else str(default))

            form.addRow(pname, w)
            f = _HoverStatus(status_lbl, desc, dlg)
            w.installEventFilter(f)
            dlg._filters.append(f)
            editors[pname] = (ptype, w, is_angle)

        return editors

    def _read_sensor_param_editors(self, editors):
        """Read current values from editors dict. Returns {pname: value}."""
        result = {}
        for pname, (kind, w, is_angle) in editors.items():
            if kind == 'joint_combo':
                result[pname] = w.currentData() or ''
            elif kind == 'choice_combo':
                result[pname] = w.currentText()
            elif isinstance(w, QCheckBox):
                result[pname] = w.isChecked()
            else:
                ptype = kind
                raw = w.value() if hasattr(w, 'value') else w.text()
                try:
                    if str(raw).strip() != '':
                        val = ptype(raw)
                        if is_angle:
                            val = np.radians(val)
                        result[pname] = val
                except (ValueError, TypeError):
                    pass
        return result

    def _add_sensor_dialog(self, stype, target_depth=None):
        from sensors import SENSOR_REGISTRY, ProprioceptiveSensor, BaseSensor
        cls = SENSOR_REGISTRY.get(stype)
        if cls is None:
            return
        params = list(cls.param_defs() if hasattr(cls, 'param_defs') else [])
        existing = {p[0] for p in params}
        params += [p for p in BaseSensor._sensor_base_param_defs() if p[0] not in existing]
        dlg, form, status_lbl = self._make_param_dialog(
            f"Add {stype}", getattr(cls, 'help_text', None))
        name_edit = QLineEdit(f"sensor{len(self.gui.circuit.sensors)}")
        form.addRow("name", name_edit)
        joints = self.gui.circuit.joints if hasattr(self.gui.circuit, 'joints') else []
        bodies = self.gui.circuit.bodies if hasattr(self.gui.circuit, 'bodies') else []
        cur_values = {p[0]: p[2] for p in params}
        editors = self._build_sensor_param_editors(
            form, dlg, status_lbl, params, joints, bodies,
            issubclass(cls, ProprioceptiveSensor), cur_values)

        body_combo = None
        if not issubclass(cls, ProprioceptiveSensor) and bodies:
            body_combo = self._make_body_combo(bodies)
            form.addRow("mounted on body", body_combo)

        nt_edit = QLineEdit('')
        nt_edit.setPlaceholderText("e.g. dopamine  (leave empty if not a transmitter)")
        form.addRow("neuromodulator transmitter", nt_edit)

        mod_color_edit = QLineEdit('')
        mod_color_edit.setPlaceholderText("#FF6600  (hex color for this transmitter)")
        form.addRow("transmitter color", mod_color_edit)

        receptor_table, receptor_btns = self._receptor_table_widget([])
        form.addRow("Modulator receptors", receptor_table)
        form.addRow(receptor_btns)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._push_undo()
        kwargs = {'name': name_edit.text().strip() or f'sensor{len(self.gui.circuit.sensors)}'}
        kwargs.update(self._read_sensor_param_editors(editors))
        angle_vals = {k: kwargs.pop(k) for k in list(kwargs) if k in self._ANGLE_PARAMS}
        new_sensor = cls(**kwargs)
        for pname, val in angle_vals.items():
            setattr(new_sensor, pname, val)
        if body_combo is not None:
            new_sensor.body_ids = body_combo.currentData() or ['root']
        nt = nt_edit.text().strip()
        new_sensor.neuromodulator_transmitter = nt if nt else None
        mc = mod_color_edit.text().strip()
        new_sensor.neuromodulator_color = mc if mc else None
        new_mods = []
        for row in range(receptor_table.rowCount()):
            name_item  = receptor_table.item(row, 0)
            scale_item = receptor_table.item(row, 1)
            site_combo = receptor_table.cellWidget(row, 2)
            if not (name_item and scale_item and site_combo):
                continue
            n = name_item.text().strip()
            if not n:
                continue
            try:
                new_mods.append((n, float(scale_item.text()), site_combo.currentText()))
            except ValueError:
                pass
        if new_mods:
            new_sensor.modulators = new_mods
        if isinstance(new_sensor, ProprioceptiveSensor) and new_sensor.joint_id:
            group = sorted(
                [jt for jt in joints if jt.motor_layer_name == new_sensor.joint_id],
                key=lambda j: j.motor_output_idx
            )
            new_sensor._joint_refs = group
            new_sensor.n = len(group) if group else 1
        self.gui.circuit.sensors.append(new_sensor)
        if target_depth is not None:
            new_sensor.layer = target_depth
        self.build()

    def _on_node_clicked(self, name, ev=None):
        self._node_just_clicked = True
        if ev is not None and getattr(ev, 'double', lambda: False)():
            self._selected = name
            self._show_selected_props()
            return
        shift = (ev is not None and ev.modifiers() & Qt.ShiftModifier)
        if self._edit_mode:
            # Edit mode: select only, no edge highlighting
            self._clear_edge_selection()
            self._selected = name
            self._spot_pen_override.clear()
            self._spot_pen_override[name] = 'selected'
            self._redraw_nodes()
        elif shift:
            # View mode + shift: toggle node in multi-selection
            if name in self._multi_selected:
                self._multi_selected.discard(name)
                self._spot_pen_override.pop(name, None)
            else:
                self._multi_selected.add(name)
                self._spot_pen_override[name] = 'multi'
            self._redraw_nodes()
        else:
            # View mode plain click: clear multi-selection, highlight edges
            self._clear_multi_selection()
            self._highlight_edges(name)

    def _clear_multi_selection(self):
        for n in self._multi_selected:
            self._spot_pen_override.pop(n, None)
        self._multi_selected.clear()
        self._redraw_nodes()

    # ── Copy / paste circuit subgraph ─────────────────────────────────────────

    def _copy_selection(self):
        layer_names = {n.rsplit('_', 1)[0] for n in self._multi_selected}
        if not layer_names:
            return
        layers = [l for l in self.gui.circuit.layers if l.name in layer_names]
        conns  = [c for c in self.gui.circuit.connections
                  if c.src in layer_names and c.tgt in layer_names]
        payload = {
            'layers':      [_layer_to_dict(l) for l in layers],
            'connections': [_connection_to_dict(c, self._weight_params.get((c.src, c.tgt)))
                            for c in conns],
        }
        QApplication.clipboard().setText(json.dumps(payload, indent=2))

    def _paste_selection(self):
        text = QApplication.clipboard().text().strip()
        if not text:
            return
        try:
            payload = json.loads(text)
            if 'layers' not in payload:
                return
        except Exception:
            return
        self._push_undo()

        existing_names = {l.name for l in self.gui.circuit.layers} | \
                         {s.name for s in self.gui.circuit.sensors}

        name_map = {}  # old name → new name (for connection remapping)
        for ld in payload['layers']:
            base = ld['name']
            new_name = base
            suffix = 2
            while new_name in existing_names:
                new_name = f'{base}_{suffix}'
                suffix += 1
            name_map[base] = new_name
            existing_names.add(new_name)

        # Offset pasted depths so they appear after the rightmost existing column.
        existing_depths = [getattr(o, 'layer', 0)
                           for o in list(self.gui.circuit.layers) + list(self.gui.circuit.sensors)
                           if getattr(o, 'layer', None) is not None]
        base = (max(existing_depths) + 1) if existing_depths else 1
        pasted_depths = [ld.get('layer') for ld in payload['layers'] if ld.get('layer') is not None]
        depth_offset = base - (min(pasted_depths) if pasted_depths else base)

        for ld in payload['layers']:
            ld = dict(ld)
            ld['name'] = name_map[ld['name']]
            if ld.get('layer') is not None:
                ld['layer'] = ld['layer'] + depth_offset
            layer = _layer_from_dict(ld)
            self.gui.circuit.layers.append(layer)

        for cd in payload.get('connections', []):
            src = name_map.get(cd['src'])
            tgt = name_map.get(cd['tgt'])
            if src and tgt:
                from circuit_model import Connection as _Conn
                self.gui.circuit.connections.append(
                    _Conn(src, tgt, np.array(cd['W'], dtype=float),
                          learning=cd.get('learning'), lr=cd.get('lr', 0.01))
                )
                if cd.get('params'):
                    self._weight_params[(src, tgt)] = cd['params']

        self.build()

    def _show_selected_props(self):
        if not self._selected:
            return
        obj_name = self._selected.rsplit('_', 1)[0]
        layer = next((l for l in self.gui.circuit.layers  if l.name == obj_name), None)
        if layer is not None:
            if getattr(layer, '_is_joint_motor', False):
                self._show_body_edit(layer)
            else:
                self._show_node_edit(layer)
            return
        sensor = next((s for s in self.gui.circuit.sensors if s.name == obj_name), None)
        if sensor is None:
            # Handle split camera half names (e.g. camera_L → camera)
            parent = obj_name.rsplit('_', 1)[0]
            sensor = next((s for s in self.gui.circuit.sensors if s.name == parent), None)
        if sensor is not None:
            self._show_sensor_edit(sensor)

    def _show_body_edit(self, layer):
        import math
        lname   = layer.name
        circuit = self.gui.circuit

        linked_joints = [j for j in circuit.joints if j.motor_layer_name == lname]
        if not linked_joints:
            return
        mirrored  = len(linked_joints) == 2
        ref_joint = next((j for j in linked_joints if j.motor_output_idx == 0), linked_joints[0])
        ref_body  = next((b for b in circuit.bodies if b.id == ref_joint.child_id), None)
        if ref_body is None:
            return
        other_joint = None
        other_body  = None
        if mirrored:
            other_joint = next((j for j in linked_joints if j.motor_output_idx == 1), None)
            if other_joint:
                other_body = next((b for b in circuit.bodies if b.id == other_joint.child_id), None)

        # Derive display base name
        if mirrored and ref_body.name.endswith('_L'):
            base_name = ref_body.name[:-2]
        else:
            base_name = ref_body.name

        dlg  = QDialog(self)
        dlg.setWindowTitle("Edit Body")
        form = QFormLayout(dlg)

        name_edit = QLineEdit(base_name)
        form.addRow("Name", name_edit)

        radius_spin = QDoubleSpinBox()
        radius_spin.setRange(0.01, 2.0); radius_spin.setSingleStep(0.01)
        radius_spin.setValue(ref_body.radius)
        form.addRow("Radius", radius_spin)

        attach_dist_spin = QDoubleSpinBox()
        attach_dist_spin.setRange(0.0, 5.0); attach_dist_spin.setSingleStep(0.01)
        attach_dist_spin.setValue(ref_joint.attach_dist)
        form.addRow("Attach distance", attach_dist_spin)

        attach_angle_spin = QDoubleSpinBox()
        attach_angle_spin.setRange(-180.0, 180.0); attach_angle_spin.setSingleStep(1.0)
        attach_angle_spin.setValue(round(math.degrees(ref_joint.attach_angle), 2))
        attach_angle_spin.setSuffix("°")
        form.addRow("Attach angle (each side)" if mirrored else "Attach angle", attach_angle_spin)

        angle_min_spin = QDoubleSpinBox()
        angle_min_spin.setRange(-180.0, 0.0); angle_min_spin.setSingleStep(5.0)
        angle_min_spin.setValue(round(math.degrees(ref_joint.angle_min), 2))
        angle_min_spin.setSuffix("°")
        form.addRow("Angle min", angle_min_spin)

        angle_max_spin = QDoubleSpinBox()
        angle_max_spin.setRange(0.0, 180.0); angle_max_spin.setSingleStep(5.0)
        angle_max_spin.setValue(round(math.degrees(ref_joint.angle_max), 2))
        angle_max_spin.setSuffix("°")
        form.addRow("Angle max", angle_max_spin)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._push_undo()
        new_name   = name_edit.text().strip() or base_name
        new_radius = radius_spin.value()
        new_dist   = attach_dist_spin.value()
        new_angle  = math.radians(attach_angle_spin.value())
        new_amin   = math.radians(angle_min_spin.value())
        new_amax   = math.radians(angle_max_spin.value())

        # Rename layer + connections + joint refs if name changed
        if new_name != lname:
            layer.name = new_name
            from dataclasses import replace as _dc_replace
            circuit.connections = [
                _dc_replace(c,
                            src=new_name if c.src == lname else c.src,
                            tgt=new_name if c.tgt == lname else c.tgt)
                for c in circuit.connections
            ]
            for j in linked_joints:
                j.motor_layer_name = new_name

        # Update body and joint params
        ref_joint.attach_dist = new_dist
        ref_joint.attach_angle = new_angle
        ref_joint.angle_min = new_amin
        ref_joint.angle_max = new_amax
        ref_body.radius = new_radius
        if mirrored and other_joint and other_body:
            other_joint.attach_dist  = new_dist
            other_joint.attach_angle = -new_angle
            other_joint.angle_min    = new_amin
            other_joint.angle_max    = new_amax
            other_body.radius = new_radius
            if new_name != lname:
                ref_body.name   = f'{new_name}_L'
                other_body.name = f'{new_name}_R'
        else:
            if new_name != lname:
                ref_body.name = new_name

        self._build_without_selection_filter()

    def _remove_selected_body(self):
        if not self._selected:
            return
        self._push_undo()
        lname = self._selected.rsplit('_', 1)[0]
        linked_joints = [j for j in self.gui.circuit.joints if j.motor_layer_name == lname]
        child_ids = {j.child_id for j in linked_joints}
        self.gui.circuit.bodies      = [b for b in self.gui.circuit.bodies
                                         if b.id not in child_ids]
        self.gui.circuit.joints      = [j for j in self.gui.circuit.joints
                                         if j.child_id not in child_ids]
        self.gui.circuit.layers      = [l for l in self.gui.circuit.layers
                                         if l.name != lname]
        self.gui.circuit.connections = [c for c in self.gui.circuit.connections
                                         if c.src != lname and c.tgt != lname]
        self._selected = None
        from rigid_body import world_poses
        poses = world_poses(self.gui.bot_pos, self.gui.circuit.bodies, self.gui.circuit.joints)
        self.gui._arena.update_child_bodies(poses, self.gui.circuit.bodies, self.gui.sim_cfg)
        self._compact_depths()
        self.build()

    def _show_node_edit(self, layer):
        ltype  = type(layer).__name__
        from neurons import LAYER_REGISTRY, DynamicsBase
        cls    = LAYER_REGISTRY.get(ltype)
        params = list(cls.param_defs() if cls is not None else [])
        if not params:
            return
        if cls is not None and issubclass(cls, DynamicsBase):
            existing = {p[0] for p in params}
            params += [p for p in DynamicsBase._dynamics_param_defs() if p[0] not in existing]
        dlg, form, status_lbl = self._make_param_dialog(
            f"Edit {ltype}: {layer.name}", getattr(cls, 'help_text', None))
        pair_name = getattr(layer, 'lateral_pair', None)
        if pair_name:
            note = QLabel(
                f"<span style='font-size:8px;color:#4888CC'>"
                f"Lateralized pair — edits also apply to <b>{pair_name}</b></span>")
            form.addRow(note)
        name_edit = QLineEdit(layer.name)
        form.addRow("name", name_edit)
        editors = {}
        for param in params:
            pname, ptype, default, desc = param[:4]
            choices = param[4] if len(param) > 4 else None
            cur = getattr(layer, pname, None)
            cur = cur if cur is not None else default
            if choices is not None:
                w = QComboBox()
                for ch in choices:
                    w.addItem(ch)
                idx = w.findText(str(cur))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif ptype == bool:
                w = QCheckBox()
                w.setChecked(bool(cur))
            elif ptype == int:
                w = QSpinBox()
                w.setMinimum(0); w.setMaximum(1000)
                try:    w.setValue(int(cur))
                except Exception: pass
            else:
                w = QLineEdit(str(cur) if cur != '' else default)
            form.addRow(pname, w)
            f = _HoverStatus(status_lbl, desc, dlg)
            w.installEventFilter(f)
            dlg._filters.append(f)
            editors[pname] = (ptype, w)
        z_spin = QSpinBox()
        z_spin.setMinimum(0); z_spin.setMaximum(20)
        z_spin.setValue(getattr(layer, 'z', 0) or 0)
        z_spin.setToolTip("3D view: which Z plane this layer belongs to")
        form.addRow("Z plane (3D view)", z_spin)

        cur_nt = getattr(layer, 'neuromodulator_transmitter', None) or ''
        nt_edit = QLineEdit(cur_nt)
        nt_edit.setPlaceholderText("e.g. dopamine  (leave empty if not a transmitter)")
        form.addRow("neuromodulator transmitter", nt_edit)

        cur_mod_color = getattr(layer, 'neuromodulator_color', None) or ''
        mod_color_edit = QLineEdit(cur_mod_color)
        mod_color_edit.setPlaceholderText("#FF6600  (hex color for this transmitter)")
        form.addRow("transmitter color", mod_color_edit)

        # Modulator receptors table
        receptor_table, receptor_btns = self._receptor_table_widget(
            getattr(layer, 'modulators', []) or [])
        form.addRow("Modulator receptors", receptor_table)
        form.addRow(receptor_btns)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # ── Warn if n changes and connections become incompatible ──────────────
        old_n = layer.n
        new_n = old_n
        n_entry = editors.get('n')
        if n_entry is not None:
            ptype_n, w_n = n_entry
            raw_n = w_n.value() if hasattr(w_n, 'value') else w_n.text()
            try:
                new_n = ptype_n(raw_n)
            except (ValueError, TypeError):
                pass

        if new_n != old_n:
            check_names = {layer.name}
            if pair_name:
                check_names.add(pair_name)
            incompatible = []
            for conn in self.gui.circuit.connections:
                W = np.asarray(conn.W, dtype=float)
                mismatch = False
                if conn.tgt in check_names:
                    if W.ndim in (2, 4) and W.shape[0] != new_n:
                        mismatch = True
                if not mismatch and conn.src in check_names:
                    if W.ndim == 2 and W.shape[1] != new_n:
                        mismatch = True
                if mismatch:
                    incompatible.append(conn)
            if incompatible:
                lines = '\n'.join(
                    f"  {c.src} → {c.tgt}  (W {np.asarray(c.W).shape})"
                    for c in incompatible
                )
                reply = QMessageBox.question(
                    self, "Incompatible connections",
                    f"Changing n from {old_n} to {new_n} makes "
                    f"{len(incompatible)} connection(s) incompatible:\n\n{lines}"
                    f"\n\nDelete these connections and continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
                _bad = {id(c) for c in incompatible}
                self.gui.circuit.connections = [
                    c for c in self.gui.circuit.connections if id(c) not in _bad
                ]
                if hasattr(self.gui.brain, 'connections'):
                    self.gui.brain.connections = self.gui.circuit.connections

        self._push_undo()
        # Rename layer and update all connection references
        new_name = name_edit.text().strip()
        if new_name and new_name != layer.name:
            old_name = layer.name
            layer.name = new_name
            from dataclasses import replace as _dc_replace
            self.gui.circuit.connections = [
                _dc_replace(c,
                            src=new_name if c.src == old_name else c.src,
                            tgt=new_name if c.tgt == old_name else c.tgt)
                for c in self.gui.circuit.connections
            ]
        # Apply tau last so it overrides tau_rise/tau_decay when explicitly set
        tau_entry = editors.pop('tau', None)
        for pname, (ptype, w) in editors.items():
            if isinstance(w, QCheckBox):
                setattr(layer, pname, w.isChecked())
            elif isinstance(w, QComboBox):
                try:
                    setattr(layer, pname, ptype(w.currentText()))
                except (ValueError, TypeError):
                    setattr(layer, pname, w.currentText())
            else:
                raw = w.value() if hasattr(w, 'value') else w.text()
                try:
                    if str(raw).strip() != '':
                        setattr(layer, pname, ptype(raw))
                except (ValueError, TypeError):
                    pass
        if tau_entry is not None:
            editors['tau'] = tau_entry
            ptype, w = tau_entry
            raw = w.text().strip()
            if raw:
                try:
                    setattr(layer, 'tau', ptype(raw))
                except (ValueError, TypeError):
                    pass
        layer.z = z_spin.value() or None

        nt = nt_edit.text().strip()
        layer.neuromodulator_transmitter = nt if nt else None

        mc = mod_color_edit.text().strip()
        layer.neuromodulator_color = mc if mc else None

        new_mods = []
        for row in range(receptor_table.rowCount()):
            name_item  = receptor_table.item(row, 0)
            scale_item = receptor_table.item(row, 1)
            site_combo = receptor_table.cellWidget(row, 2)
            if not (name_item and scale_item and site_combo):
                continue
            n = name_item.text().strip()
            if not n:
                continue
            try:
                new_mods.append((n, float(scale_item.text()), site_combo.currentText()))
            except ValueError:
                pass
        layer.modulators = new_mods

        # For Conv2dLayer, validate pool='none' + n_filters and sync n / viz attrs.
        import torch as _torch
        from neurons import Conv2dLayer as _C2d
        if isinstance(layer, _C2d) and layer.pool == 'none' and layer.n_filters > 1:
            QMessageBox.warning(self, "Conv2dLayer",
                                "pool='none' requires n_filters=1.\n\n"
                                "n_filters has been corrected to 1.")
            layer.n_filters = 1
        def _sync_conv_n(lyr):
            if isinstance(lyr, _C2d):
                if lyr.pool == 'none':
                    lyr.viz_n = 1
                    if not hasattr(lyr, '_last_frame'):
                        lyr._last_frame = None
                    if not hasattr(lyr, 'frame_h'):
                        lyr.frame_h = None
                    if not hasattr(lyr, 'frame_w'):
                        lyr.frame_w = None
                else:
                    try:
                        del lyr.viz_n
                    except AttributeError:
                        pass
                new_n = lyr.n_filters
                if lyr.n != new_n:
                    lyr.n = new_n
                    lyr.register_buffer('_x', _torch.zeros(new_n))
                    lyr.output = _torch.zeros(new_n)
        _sync_conv_n(layer)

        # Propagate all edited params to the lateralized partner (if any).
        if pair_name:
            partner = next((l for l in self.gui.circuit.layers
                            if l.name == pair_name), None)
            if partner is not None:
                for pname, *_ in params:
                    if pname != 'lateralized' and hasattr(layer, pname):
                        setattr(partner, pname, getattr(layer, pname))
                for attr in ('group', 'z', 'neuromodulator_transmitter',
                             'neuromodulator_color', 'modulators'):
                    if hasattr(layer, attr):
                        setattr(partner, attr, getattr(layer, attr))
                _sync_conv_n(partner)

        self._build_without_selection_filter()

    _ANGLE_PARAMS = {'angle_spread', 'center_angle', 'arc_angle', 'mount_angle', 'fov'}

    def _show_sensor_edit(self, sensor):
        from sensors import SENSOR_REGISTRY, ProprioceptiveSensor as _ProprioSensor, BaseSensor
        stype  = type(sensor).__name__
        cls    = SENSOR_REGISTRY.get(stype)
        params = list(cls.param_defs() if cls is not None else [])
        existing = {p[0] for p in params}
        params += [p for p in BaseSensor._sensor_base_param_defs() if p[0] not in existing]
        dlg, form, status_lbl = self._make_param_dialog(
            f"Edit {stype}: {sensor.name}", getattr(cls, 'help_text', None))
        name_edit = QLineEdit(sensor.name)
        form.addRow("name", name_edit)
        joints = self.gui.circuit.joints if hasattr(self.gui.circuit, 'joints') else []
        bodies = self.gui.circuit.bodies
        cur_values = {p[0]: getattr(sensor, p[0], p[2]) for p in params}
        editors = self._build_sensor_param_editors(
            form, dlg, status_lbl, params, joints, bodies,
            isinstance(sensor, _ProprioSensor), cur_values)

        body_combo = None
        if not isinstance(sensor, _ProprioSensor) and bodies:
            cur_body_ids = getattr(sensor, 'body_ids', None) or ['root']
            body_combo = self._make_body_combo(bodies, cur_body_ids)
            form.addRow("mounted on body", body_combo)

        cur_nt = getattr(sensor, 'neuromodulator_transmitter', None) or ''
        nt_edit = QLineEdit(cur_nt)
        nt_edit.setPlaceholderText("e.g. dopamine  (leave empty if not a transmitter)")
        form.addRow("neuromodulator transmitter", nt_edit)

        cur_mod_color = getattr(sensor, 'neuromodulator_color', None) or ''
        mod_color_edit = QLineEdit(cur_mod_color)
        mod_color_edit.setPlaceholderText("#FF6600  (hex color for this transmitter)")
        form.addRow("transmitter color", mod_color_edit)

        receptor_table, receptor_btns = self._receptor_table_widget(
            getattr(sensor, 'modulators', []) or [])
        form.addRow("Modulator receptors", receptor_table)
        form.addRow(receptor_btns)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._push_undo()
        new_name = name_edit.text().strip()
        if new_name and new_name != sensor.name:
            old_name = sensor.name
            sensor.name = new_name
            from dataclasses import replace as _dc_replace
            self.gui.circuit.connections = [
                _dc_replace(c, src=new_name) if c.src == old_name else
                _dc_replace(c, src=new_name + c.src[len(old_name):])
                if c.src.startswith(old_name + '_') else c
                for c in self.gui.circuit.connections
            ]
        for pname, val in self._read_sensor_param_editors(editors).items():
            setattr(sensor, pname, val)
        if body_combo is not None:
            sensor.body_ids = body_combo.currentData() or ['root']

        nt = nt_edit.text().strip()
        sensor.neuromodulator_transmitter = nt if nt else None
        mc = mod_color_edit.text().strip()
        sensor.neuromodulator_color = mc if mc else None

        new_mods = []
        for row in range(receptor_table.rowCount()):
            name_item  = receptor_table.item(row, 0)
            scale_item = receptor_table.item(row, 1)
            site_combo = receptor_table.cellWidget(row, 2)
            if not (name_item and scale_item and site_combo):
                continue
            n = name_item.text().strip()
            if not n:
                continue
            try:
                new_mods.append((n, float(scale_item.text()), site_combo.currentText()))
            except ValueError:
                pass
        sensor.modulators = new_mods

        self._build_without_selection_filter()

    def _on_scene_click(self, event):
        if event.button() == Qt.LeftButton:
            if self._node_just_clicked:
                self._node_just_clicked = False
                return   # click was on a node — don't clear the highlight it just set
            if self._edit_mode:
                view_pt = self._vb.mapSceneToView(event.scenePos())
                edge = self._edge_at(view_pt)
                if edge is not None:
                    self._clear_edge_selection()
                    self._selected_edge = edge
                    self._selected = None
                    self._spot_pen_override.clear()
                    self._redraw_nodes()
                    self._highlight_selected_edge(*edge)
                else:
                    self._clear_edge_selection()
                    # Image-type nodes (Leaky2dLayer, camera halves) have a small
                    # scatter dot hidden under the image circle, so sigClicked rarely
                    # fires on a plain click.  Fall back to _node_at so any click on
                    # the image still selects the node.
                    hit = self._node_at(view_pt)
                    if hit is not None:
                        self._selected = hit
                        self._spot_pen_override.clear()
                        self._spot_pen_override[hit] = 'selected'
                        self._redraw_nodes()
            else:
                if self._highlighted_node:
                    self._clear_edge_highlight()
                if not (event.modifiers() & Qt.ShiftModifier):
                    self._clear_multi_selection()
        if event.button() == Qt.RightButton:
            # Multi-selection context menu (shift-clicked nodes)
            if self._multi_selected:
                event.accept()
                menu = QMenu(self)
                motif_act = menu.addAction("Save as motif…")
                copy_act  = menu.addAction("Copy selection")
                menu.addSeparator()
                clear_act = menu.addAction("Clear selection")
                chosen = menu.exec(event.screenPos().toPoint())
                if chosen == motif_act:
                    self._save_as_motif()
                elif chosen == copy_act:
                    self._copy_selection()
                elif chosen == clear_act:
                    self._clear_multi_selection()
                return
            # View mode: right-click on a node → oscilloscope toggle
            if not self._edit_mode and not self._multi_selected and self._all_scatter is not None:
                local_pt = self._all_scatter.mapFromScene(event.scenePos())
                pts = self._all_scatter.pointsAt(local_pt)
                if pts:
                    lname = pts[0].data().rsplit('_', 1)[0]
                    is_layer  = any(l.name == lname for l in self.gui.circuit.layers)
                    is_sensor = any(s.name == lname for s in self.gui.circuit.sensors)
                    if not is_sensor and not is_layer:
                        parent = lname.rsplit('_', 1)[0]
                        if any(s.name == parent for s in self.gui.circuit.sensors):
                            is_sensor = True
                            lname = parent
                    if is_layer or is_sensor:
                        event.accept()
                        osc_items = getattr(getattr(self.gui, '_osc_ctrl', None), '_osc_items', set())
                        node_key  = pts[0].data()
                        layer_obj = next((l for l in self.gui.circuit.layers
                                          if l.name == lname), None)
                        n_neurons = getattr(layer_obj, 'n', 1) or 1
                        # Dense layers: track the individual neuron clicked; sparse: the whole layer
                        osc_key = node_key if n_neurons > self._DENSE_THRESHOLD else lname
                        in_osc  = osc_key in osc_items
                        is_muted = getattr(layer_obj, 'muted', False)
                        menu = QMenu(self)
                        props_act = menu.addAction("Properties…")
                        menu.addSeparator()
                        osc_act = menu.addAction(
                            "Remove from oscilloscope" if in_osc else "Add to oscilloscope")
                        mute_act = None
                        if layer_obj is not None:
                            menu.addSeparator()
                            mute_act = menu.addAction(
                                "Unmute layer" if is_muted else "Mute layer")
                        chosen = menu.exec(event.screenPos().toPoint())
                        if chosen == props_act:
                            self._selected = node_key
                            self._show_selected_props()
                        elif chosen == osc_act:
                            self.gui._toggle_osc_layer(osc_key)
                        elif mute_act is not None and chosen == mute_act:
                            layer_obj.muted = not is_muted
                            self._redraw_nodes()
                        return
            # Right-click near a connection arc → weight panel toggle
            if not self._edit_mode and not self._multi_selected:
                view_pt = self._vb.mapSceneToView(event.scenePos())
                edge = self._edge_at(view_pt)
                if edge is not None:
                    src, tgt = edge
                    event.accept()
                    pinned = (src, tgt) in self._weight_pinned
                    menu = QMenu(self)
                    act = menu.addAction(
                        "Remove from weight panel" if pinned else "Add to weight panel")
                    if menu.exec(event.screenPos().toPoint()) == act:
                        self._toggle_weight_entry(src, tgt)
                    return
            # Right-click on a column panel rect → label annotation
            if not self._edit_mode and not self._multi_selected:
                view_pt = self._vb.mapSceneToView(event.scenePos())
                hit_dv = self._panel_at(view_pt)
                if hit_dv is not None:
                    event.accept()
                    self._show_col_label_menu(hit_dv, event.screenPos().toPoint())
                    return
            # Edit mode: remove for selected connection
            if self._edit_mode and self._selected_edge:
                event.accept()
                src, tgt = self._selected_edge
                menu = QMenu(self)
                edit_act = menu.addAction(f"Edit weight '{src} → {tgt}'…")
                edit_act.triggered.connect(lambda: self._show_edge_edit(src, tgt))
                menu.addSeparator()
                rm_act = menu.addAction(f"Remove '{src} → {tgt}'")
                rm_act.triggered.connect(self._remove_selected_connection)
                menu.exec(event.screenPos().toPoint())
                return
            # Edit mode: properties + remove for selected node
            if self._edit_mode and self._selected:
                event.accept()
                # Resolve node key → actual name.  Try exact match first so that
                # layer names like 'layer1_L' are not wrongly stripped to 'layer1'.
                # Fall back to stripping suffix for lateralized sensor halves
                # ('sensor0_L' → 'sensor0' → parent sensor name).
                lname    = self._selected
                is_layer  = any(l.name == lname for l in self.gui.circuit.layers)
                is_sensor = any(s.name == lname for s in self.gui.circuit.sensors)
                if not is_layer and not is_sensor:
                    lname    = self._selected.rsplit('_', 1)[0]
                    is_layer  = any(l.name == lname for l in self.gui.circuit.layers)
                    is_sensor = any(s.name == lname for s in self.gui.circuit.sensors)
                    if not is_sensor and not is_layer:
                        parent = lname.rsplit('_', 1)[0]
                        if any(s.name == parent for s in self.gui.circuit.sensors):
                            is_sensor = True
                            lname = parent
                if is_layer or is_sensor:
                    layer_obj = next((l for l in self.gui.circuit.layers
                                      if l.name == lname), None)
                    is_muted  = getattr(layer_obj, 'muted', False)
                    is_body   = getattr(layer_obj, '_is_joint_motor', False)
                    menu = QMenu(self)
                    if is_body:
                        props_act = menu.addAction("Edit body…")
                    else:
                        props_act = menu.addAction("Properties…")
                    props_act.triggered.connect(self._show_selected_props)
                    if layer_obj is not None and not is_body:
                        mute_act = menu.addAction(
                            "Unmute layer" if is_muted else "Mute layer")
                        mute_act.triggered.connect(
                            lambda _checked, lo=layer_obj, m=is_muted:
                                (setattr(lo, 'muted', not m), self._redraw_nodes()))
                    menu.addSeparator()
                    if is_layer:
                        if is_body:
                            rm_act = menu.addAction(f"Remove body '{lname}'")
                            rm_act.triggered.connect(self._remove_selected_body)
                        else:
                            rm_act = menu.addAction(f"Remove layer '{lname}'")
                            if lname == 'motor':
                                rm_act.setEnabled(False)
                                rm_act.setToolTip("Motor layer cannot be removed")
                            else:
                                rm_act.triggered.connect(self._remove_selected_layer)
                    else:
                        rm_act = menu.addAction(f"Remove sensor '{lname}'")
                        rm_act.triggered.connect(self._remove_selected_sensor)
                    menu.exec(event.screenPos().toPoint())

    def _add_layer_dialog(self, ltype):
        from neurons import LAYER_REGISTRY, DynamicsBase
        cls    = LAYER_REGISTRY.get(ltype)
        params = list(cls.param_defs() if cls is not None else [])
        if cls is not None and issubclass(cls, DynamicsBase):
            existing = {p[0] for p in params}
            params += [p for p in DynamicsBase._dynamics_param_defs() if p[0] not in existing]
        dlg, form, status_lbl = self._make_param_dialog(
            f"Add {ltype}", getattr(cls, 'help_text', None))

        name_edit = QLineEdit(f"layer{len(self.gui.circuit.layers)}")
        form.addRow("name", name_edit)
        editors = {}
        for param in params:
            pname, ptype, default, desc = param[:4]
            choices = param[4] if len(param) > 4 else None
            if choices is not None:
                w = QComboBox()
                for ch in choices:
                    w.addItem(ch)
                idx = w.findText(str(default))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif ptype == bool:
                w = QCheckBox()
                w.setChecked(bool(default))
            elif ptype == int:
                w = QSpinBox()
                w.setMinimum(0); w.setMaximum(1000)
                try:    w.setValue(int(default))
                except Exception: pass
            else:
                w = QLineEdit(str(default))
            form.addRow(pname, w)
            f = _HoverStatus(status_lbl, desc, dlg)
            w.installEventFilter(f)
            dlg._filters.append(f)
            editors[pname] = (ptype, w)
        nt_edit = QLineEdit('')
        nt_edit.setPlaceholderText("e.g. dopamine  (leave empty if not a transmitter)")
        form.addRow("neuromodulator transmitter", nt_edit)

        mod_color_edit = QLineEdit('')
        mod_color_edit.setPlaceholderText("#FF6600  (hex color for this transmitter)")
        form.addRow("transmitter color", mod_color_edit)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        kwargs = {'name': name_edit.text().strip() or f'layer{len(self.gui.circuit.layers)}'}
        for pname, (ptype, w) in editors.items():
            if isinstance(w, QCheckBox):
                kwargs[pname] = w.isChecked()
            elif isinstance(w, QComboBox):
                try:
                    kwargs[pname] = ptype(w.currentText())
                except (ValueError, TypeError):
                    kwargs[pname] = w.currentText()
            else:
                raw = w.value() if hasattr(w, 'value') else w.text()
                try:
                    if str(raw).strip() != '':
                        kwargs[pname] = ptype(raw)
                except (ValueError, TypeError):
                    pass
        nt = nt_edit.text().strip()
        if nt:
            kwargs['neuromodulator_transmitter'] = nt
        mc = mod_color_edit.text().strip()
        if mc:
            kwargs['neuromodulator_color'] = mc
        self._push_undo()
        from neurons import RingAttractorLayer as _RAL, Conv2dLayer as _C2d, Leaky2dLayer as _L2dCls
        if issubclass(cls, _C2d) and kwargs.get('pool') == 'none' and int(kwargs.get('n_filters', 1)) > 1:
            QMessageBox.warning(self, "Conv2dLayer",
                                "pool='none' requires n_filters=1.\n\n"
                                "n_filters has been corrected to 1.")
            kwargs['n_filters'] = 1
        if kwargs.get('lateralized') and issubclass(cls, (_C2d, _L2dCls)):
            base_name = kwargs.pop('name')
            new_layers = []
            for side in ('_L', '_R'):
                kw = dict(kwargs, name=base_name + side, lateralized=True)
                lyr = cls(**kw)
                self.gui.circuit.layers.append(lyr)
                setattr(self.gui.brain, lyr.name, lyr)
                new_layers.append(lyr)
            # Cross-link so _finish_connection can auto-wire the mirror side
            new_layers[0].lateral_pair = new_layers[1].name
            new_layers[1].lateral_pair = new_layers[0].name
        else:
            new_layer = cls(**kwargs)
            self.gui.circuit.layers.append(new_layer)
            setattr(self.gui.brain, new_layer.name, new_layer)
            if isinstance(new_layer, _RAL):
                W = _RAL.default_kernel(new_layer.n)
                from circuit_model import Connection as _Conn
                self.gui.circuit.connections.append(
                    _Conn(new_layer.name, new_layer.name, W))
        self.build()

    def _compact_depths(self):
        """Close gaps in explicit depths while preserving the minimum depth value.

        Also accounts for connectivity-computed depths (layers without an explicit
        'layer' attribute) so compaction never maps an explicit depth onto a slot
        already occupied by a computed layer.
        """
        expl_objs = (
            [l for l in self.gui.circuit.layers  if getattr(l, 'layer', None) is not None] +
            [s for s in self.gui.circuit.sensors if getattr(s, 'layer', None) is not None]
        )
        if not expl_objs:
            return
        # Include connectivity-computed depths so we don't close gaps they occupy.
        computed = self._compute_depth()
        implicit_depths = {
            computed[l.name]
            for l in self.gui.circuit.layers
            if getattr(l, 'layer', None) is None and l.name in computed
        }
        all_unique = sorted({o.layer for o in expl_objs} | implicit_depths)
        min_d = all_unique[0]
        depth_map = {d: min_d + i for i, d in enumerate(all_unique)}
        for o in expl_objs:
            o.layer = depth_map[o.layer]

    def _pin_implicit_depths(self):
        """Freeze connectivity-inferred column positions before a deletion mutates the graph.

        Layers that have never been manually dragged (lyr.layer is None) get their
        depth assigned explicitly so that removing a connection or sensor doesn't
        cause _compute_depth to fall back to column 1 for nodes that have lost their
        upstream dependency.  Called after _push_undo so that undo restores the
        un-pinned state correctly.
        """
        computed = self._compute_depth()
        for lyr in self.gui.circuit.layers:
            if getattr(lyr, 'layer', None) is None and lyr.name in computed:
                lyr.layer = computed[lyr.name]

    def _sync_brain_to_circuit(self):
        """Re-point brain.layers/connections/sensors to the current circuit lists."""
        brain = getattr(self.gui, 'brain', None)
        if brain is None:
            return
        if hasattr(brain, 'layers'):
            brain.layers      = self.gui.circuit.layers
        if hasattr(brain, 'connections'):
            brain.connections = self.gui.circuit.connections
        if hasattr(brain, 'sensors'):
            brain.sensors     = self.gui.circuit.sensors

    def _remove_selected_layer(self):
        if not self._selected:
            return
        # Node key IS the layer name — do not strip suffix here.
        lname = self._selected
        if not any(l.name == lname for l in self.gui.circuit.layers):
            lname = self._selected.rsplit('_', 1)[0]
        if lname == 'motor':
            return
        self._push_undo()
        self._pin_implicit_depths()
        # Collect all names to delete: layer + its lateral pair (if any).
        lyr_obj = next((l for l in self.gui.circuit.layers if l.name == lname), None)
        names_to_delete = {lname}
        if lyr_obj is not None:
            pair = getattr(lyr_obj, 'lateral_pair', None)
            if pair:
                names_to_delete.add(pair)
        self.gui.circuit.layers      = [l for l in self.gui.circuit.layers
                                         if l.name not in names_to_delete]
        self.gui.circuit.connections = [c for c in self.gui.circuit.connections
                                         if c.src not in names_to_delete
                                         and c.tgt not in names_to_delete]
        self._sync_brain_to_circuit()
        self._selected = None
        self._compact_depths()
        self.build()

    def _remove_selected_sensor(self):
        if not self._selected:
            return
        self._push_undo()
        self._pin_implicit_depths()
        sname = self._selected.rsplit('_', 1)[0]
        # If sname is a split-camera half (camera_L / camera_R), resolve to parent
        if not any(s.name == sname for s in self.gui.circuit.sensors):
            sname = sname.rsplit('_', 1)[0]
        self.gui.circuit.sensors     = [s for s in self.gui.circuit.sensors
                                         if s.name != sname]
        # Also remove connections to/from lateralized halves (e.g. sensor0_L, sensor0_R).
        pfx = sname + '_'
        self.gui.circuit.connections = [c for c in self.gui.circuit.connections
                                         if c.src != sname and c.tgt != sname
                                         and not c.src.startswith(pfx)
                                         and not c.tgt.startswith(pfx)]
        self._sync_brain_to_circuit()
        self._selected = None
        self.build()

    # ── Connection drag helpers ────────────────────────────────────────────────

    def _node_at(self, pt):
        """Return the node key nearest to view-space point *pt*, or None."""
        r2 = (self._NODE_R * 1.3) ** 2
        best, best_d2 = None, float('inf')
        for name, (x, y) in self._positions.items():
            d2 = (pt.x() - x) ** 2 + (pt.y() - y) ** 2
            if d2 <= r2 and d2 < best_d2:
                best, best_d2 = name, d2
        return best

    def _update_conn_preview(self, mouse_pt):
        sx, sy = self._positions[self._conn_from]
        self._conn_preview.setData([sx, mouse_pt.x()], [sy, mouse_pt.y()])
        self._conn_preview.setVisible(True)

    def _hide_conn_preview(self):
        self._conn_preview.setVisible(False)

    # ── Shared weight-matrix dialog ───────────────────────────────────────────

    def _open_weight_dialog(self, src_name, tgt_name, ns, nt, W_init,
                            saved_params=None, conv_params=None):
        """Weight-matrix editor. Returns (W_final, True, params) or (None, False, None)."""
        dlg = WeightMatrixDialog(self, src_name, tgt_name, ns, nt, W_init,
                                 self.gui.circuit, saved_params,
                                 conv_params=conv_params)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None, False, None
        W_final, params_out = dlg.get_result()
        return W_final, True, params_out

    def _finish_connection(self, src_key, tgt_key):
        from sensors import CameraSensor as _CamSensor
        src_name = src_key.rsplit('_', 1)[0]
        tgt_name = tgt_key.rsplit('_', 1)[0]

        tgt_layer  = next((l for l in self.gui.circuit.layers  if l.name == tgt_name), None)
        src_sensor = next((s for s in self.gui.circuit.sensors if s.name == src_name), None)
        if src_sensor is None:
            # Lateralized sensor half: 'sensor_L' → resolve to parent 'sensor'
            parent = src_name.rsplit('_', 1)[0]
            src_sensor = next((s for s in self.gui.circuit.sensors if s.name == parent), None)

        # True when source is one half of a lateralized sensor (camera or joint-pair)
        _src_is_lat_half = (src_sensor is not None
                            and _sensor_is_lateralized(src_sensor, self.gui.circuit)
                            and (src_name.endswith('_L') or src_name.endswith('_R')))

        is_conv2d = (tgt_layer is not None
                     and hasattr(tgt_layer, 'n_filters')
                     and hasattr(tgt_layer, 'kernel_size'))

        from neurons import Leaky2dLayer as _L2d
        is_leaky2d = (tgt_layer is not None and isinstance(tgt_layer, _L2d))

        _pair_name = None   # lateral pair partner of the source layer (if any)
        _pair_lyr  = None   # partner layer object

        if is_leaky2d:
            # Leaky2dLayer: auto-wire a 1-D ones passthrough weight (no dialog).
            # Derive pixel count from source camera; fall back to n_map.
            # NOTE: check _L/_R suffix FIRST — src_sensor is resolved to the parent
            # camera above, so isinstance(src_sensor, _CamSensor) would be True even
            # for a half, causing full-camera dimensions to be used incorrectly.
            if src_name.endswith(('_L', '_R')):
                # Lateralized camera half — src_sensor was resolved to parent above.
                parent_sensor = src_sensor
                if isinstance(parent_sensor, _CamSensor) and getattr(parent_sensor, 'lateralized', False):
                    mid     = parent_sensor.width // 2
                    overlap = getattr(parent_sensor, 'overlap', 0)
                    if src_name.endswith('_L'):
                        half_w = int(np.clip(mid + overlap, 0, parent_sensor.width))
                    else:
                        r_start = int(np.clip(mid - overlap, 0, parent_sensor.width))
                        half_w  = parent_sensor.width - r_start
                    ns = half_w * parent_sensor.height * parent_sensor.in_ch
                    tgt_layer.in_ch   = parent_sensor.in_ch
                    tgt_layer.frame_h = parent_sensor.height
                    tgt_layer.frame_w = half_w
                else:
                    ns = self._n_map().get(src_name, 1)
            elif isinstance(src_sensor, _CamSensor):
                ns = src_sensor.width * src_sensor.height * src_sensor.in_ch
                tgt_layer.in_ch   = src_sensor.in_ch
                tgt_layer.frame_h = src_sensor.height
                tgt_layer.frame_w = src_sensor.width
            else:
                ns = self._n_map().get(src_name, 1)
            tgt_layer._ensure_n(ns)
            W = np.ones(ns, dtype=np.float32)
            self._push_undo()
            from circuit_model import Connection as _Conn
            new_conns = [c for c in self.gui.circuit.connections
                         if not (c.src == src_name and c.tgt == tgt_name)]
            new_conns.append(_Conn(src_name, tgt_name, W))
            # Auto-wire mirror: sensor_L→leaky_L triggers sensor_R→leaky_R
            _l2d_pair = getattr(tgt_layer, 'lateral_pair', None)
            if _l2d_pair and src_name.endswith(('_L', '_R')):
                mirror_src = _mirror_name(src_name)
                if mirror_src and mirror_src != src_name:
                    mirror_tgt = _l2d_pair
                    mirror_lyr = next(
                        (l for l in self.gui.circuit.layers if l.name == mirror_tgt), None)
                    if mirror_lyr is not None:
                        mirror_lyr._ensure_n(ns)
                        mirror_lyr.in_ch   = tgt_layer.in_ch
                        mirror_lyr.frame_h = tgt_layer.frame_h
                        mirror_lyr.frame_w = tgt_layer.frame_w
                    W_mirror = np.ones(ns, dtype=np.float32)
                    new_conns = [c for c in new_conns
                                 if not (c.src == mirror_src and c.tgt == mirror_tgt)]
                    new_conns.append(_Conn(mirror_src, mirror_tgt, W_mirror))
            self.gui.circuit.connections = new_conns
            self.build()
            return

        if is_conv2d:
            # Conv2dLayer path — open FilterStackDialog
            kernel_size = tgt_layer.kernel_size
            _src_lyr_c  = next((l for l in self.gui.circuit.layers if l.name == src_name), None)
            if isinstance(src_sensor, _CamSensor):
                in_ch = getattr(src_sensor, 'in_ch', 1)
            elif isinstance(_src_lyr_c, _L2d):
                in_ch = max(1, _src_lyr_c.in_ch)
            else:
                in_ch = max(1, getattr(src_sensor, 'n', 1) if src_sensor else 1)
            is_lat_half = _src_is_lat_half or (
                isinstance(_src_lyr_c, _L2d)
                and getattr(_src_lyr_c, 'lateral_pair', None) is not None
                and (src_name.endswith('_L') or src_name.endswith('_R'))
            )
            W_init = None
            for c in self.gui.circuit.connections:
                if c.src == src_name and c.tgt == tgt_name:
                    arr = np.asarray(c.W, dtype=float)
                    if arr.ndim == 4:
                        W_init = arr.copy()
                    break
            dlg = FilterStackDialog(self, src_name, tgt_name,
                                    in_ch, kernel_size, kernel_size, W_init,
                                    lateralized_half=is_lat_half)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            W, ok = dlg.get_result()
            if not ok:
                return
        else:
            # Standard linear path
            n_map  = self._n_map()
            ns     = n_map.get(src_name, 1)
            nt     = n_map.get(tgt_name, 1)
            # Lateralized conv pair source: combine both halves into one weight column block.
            # e.g. conv1_L (n=1 filter) + conv1_R (n=1 filter) → ns=2, W shape (nt, 2).
            _src_lyr    = next((l for l in self.gui.circuit.layers if l.name == src_name), None)
            _pair_name  = getattr(_src_lyr, 'lateral_pair', None) if _src_lyr else None
            _pair_lyr   = None
            if _pair_name:
                _pair_lyr = next((l for l in self.gui.circuit.layers if l.name == _pair_name), None)
                if _pair_lyr:
                    ns = (_src_lyr.n or 0) + (_pair_lyr.n or 0)
            # Lateralized joint-pair sensor half → non-lat target: combined matrix (n_L + n_R).
            # When target is itself lateralized (has _lateral_pair), auto-mirror is used instead.
            _tgt_is_lat = (tgt_layer is not None
                           and getattr(tgt_layer, 'lateral_pair', None) is not None)
            if not _pair_name and _src_is_lat_half and not _tgt_is_lat:
                ns = ns * 2
            W_init = np.zeros((nt, ns))
            for c in self.gui.circuit.connections:
                if c.src == src_name and c.tgt == tgt_name:
                    ew = np.atleast_2d(np.asarray(c.W, dtype=float))
                    if ew.shape == (nt, ns):
                        W_init = ew.copy()
                    break
            W, accepted, params = self._open_weight_dialog(
                src_name, tgt_name, ns, nt, W_init,
                self._weight_params.get((src_name, tgt_name)))
            if not accepted:
                return
            if params is not None:
                self._weight_params[(src_name, tgt_name)] = params

        self._push_undo()
        from circuit_model import Connection as _Conn
        new_conns = [c for c in self.gui.circuit.connections
                     if not (c.src == src_name and c.tgt == tgt_name)]
        # When using a combined pair weight, also remove the partner's separate connection.
        if not is_conv2d and _pair_lyr is not None:
            new_conns = [c for c in new_conns
                         if not (c.src == _pair_name and c.tgt == tgt_name)]
        W_snap = np.asarray(W, dtype=float).copy() if W is not None else None
        new_conns.append(_Conn(src_name, tgt_name, W, init_W=W_snap))

        # Auto-wire the mirror connection.
        # For conv2d: sensor_L→conv_L auto-wires sensor_R→conv_R (lat→lat);
        #             sensor_L→conv auto-wires sensor_R→conv (lat→non-lat camera case).
        # For linear: lat sensor half → lat layer auto-wires the R half to the partner.
        pair_name = getattr(tgt_layer, 'lateral_pair', None)
        if is_conv2d:
            mirror_src = _mirror_name(src_name)
            if mirror_src and mirror_src != src_name:
                if pair_name:
                    mirror_tgt = pair_name
                elif is_lat_half:
                    mirror_tgt = tgt_name
                else:
                    mirror_tgt = None
                if mirror_tgt:
                    new_conns = [c for c in new_conns
                                 if not (c.src == mirror_src and c.tgt == mirror_tgt)]
                    new_conns.append(_Conn(mirror_src, mirror_tgt, W, init_W=W_snap))
        elif _src_is_lat_half and pair_name:
            # Linear lat sensor half → lat target: auto-wire mirror side.
            mirror_src = _mirror_name(src_name)
            if mirror_src and mirror_src != src_name:
                new_conns = [c for c in new_conns
                             if not (c.src == mirror_src and c.tgt == pair_name)]
                new_conns.append(_Conn(mirror_src, pair_name, W, init_W=W_snap))

        self.gui.circuit.connections = new_conns
        self._build_without_selection_filter()

    def _new_network(self):
        """Create a blank circuit pre-populated with a motor output layer."""
        motor = SumLayer(activation='linear', name='motor', n=2, layer=4)
        self.gui.circuit.sensors     = []
        self.gui.circuit.layers      = [motor]
        self.gui.circuit.connections = []
        self.gui.brain.layers        = [motor]
        self.gui.brain.connections   = []
        setattr(self.gui.brain, 'motor', motor)
        self.gui.brain.network_file  = ''
        self._hidden_cols   = set()
        self._disabled_cols = set()
        self._col_labels    = {}
        self._selected = None
        self.build()

    def _save_circuit(self):
        if isinstance(self.gui.brain, DataBrain):
            self._save_network_json()
        else:
            self._save_brain_python()

    def _save_brain_python(self):
        brain     = self.gui.brain
        brain_cls = brain.__class__
        try:
            src_file = inspect.getfile(brain_cls)
        except Exception as e:
            QMessageBox.critical(self, 'Save', f'Cannot determine brain source file:\n{e}')
            return

        try:
            with open(src_file, 'r', encoding='utf-8') as f:
                content = f.read()
            content = serialize_brain(
                content,
                self.gui.circuit.layers,
                self.gui.circuit.connections,
            )
            with open(src_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception:
            import traceback
            QMessageBox.critical(self, 'Save error', traceback.format_exc())
            return

        print(f'[NetworkViz] Saved circuit to {src_file}')
        self.gui.load_brain()
        self.build()
        QMessageBox.information(self, 'Save', f'Saved to\n{os.path.basename(src_file)}')

    def _save_network_json(self):
        from brain_serializer import save_network_file
        from PySide6.QtWidgets import QInputDialog
        current = getattr(self.gui.brain, 'network_file', '') or ''
        suggested = current.replace('.json', '') if current else 'my_network'
        name, ok = QInputDialog.getText(
            self, 'Save Network', 'Network name:', text=suggested)
        if not ok or not name.strip():
            return
        name = name.strip()
        if not name.endswith('.json'):
            name += '.json'
        project = getattr(self.gui.brain, 'network_project', '')
        net_dir = os.path.join('networks', project) if project else 'networks'
        os.makedirs(net_dir, exist_ok=True)
        path = os.path.join(net_dir, name)
        try:
            save_network_file(
                path,
                self.gui.circuit.sensors,
                self.gui.circuit.layers,
                self.gui.circuit.connections,
                self._hidden_cols,
                self._disabled_cols,
                self._col_labels,
                bodies=self.gui.circuit.bodies,
                joints=self.gui.circuit.joints,
                connection_params=self._weight_params,
            )
        except Exception:
            import traceback
            QMessageBox.critical(self, 'Save error', traceback.format_exc())
            return
        self.gui.brain.network_file = name
        if hasattr(self.gui, '_connection_params'):
            self.gui._connection_params = self._weight_params
        # Sync brain attribute references to the live circuit layers (joints may
        # have been added/removed since the last full load).
        self.gui.brain.layers      = self.gui.circuit.layers
        self.gui.brain.sensors     = self.gui.circuit.sensors
        self.gui.brain.connections = self.gui.circuit.connections
        self.gui.brain_mgr.resolve_joint_sensor_refs()
        # Update the sidebar network-file combo to reflect the (possibly new) name.
        self.gui._rebuild_brain_params()
        print(f'[NetworkViz] Saved network to {path}')
        QMessageBox.information(self, 'Save', f'Network saved to\n{name}')

    # ── Bonsai export ─────────────────────────────────────────────────────────

    def _copy_bonsai(self):
        circuit = self.gui.circuit
        try:
            xml = self._generate_bonsai_xml(circuit)
        except Exception:
            import traceback
            QMessageBox.critical(self, 'Bonsai Export',
                                 f'Error generating XML:\n{traceback.format_exc()}')
            return
        QApplication.clipboard().setText(xml)
        old_text = self._btn_bonsai.text()
        self._btn_bonsai.setText('Copied!')
        QTimer.singleShot(1500, lambda: self._btn_bonsai.setText(old_text))

    def _generate_bonsai_xml(self, circuit):
        from bonsai_exporter import generate_bonsai_xml
        return generate_bonsai_xml(circuit)

    def _copy_svg(self):
        try:
            from pyqtgraph.exporters import SVGExporter
            exporter = SVGExporter(self._plot)
            svg_bytes = exporter.export(toBytes=True)
        except Exception:
            import traceback
            QMessageBox.critical(self, 'SVG Export',
                                 f'Error generating SVG:\n{traceback.format_exc()}')
            return
        if svg_bytes:
            mime = QMimeData()
            mime.setData('image/svg+xml', svg_bytes)
            mime.setText(svg_bytes.decode('utf-8'))
            QApplication.clipboard().setMimeData(mime)
            old_text = self._btn_svg.text()
            self._btn_svg.setText('Copied!')
            QTimer.singleShot(1500, lambda: self._btn_svg.setText(old_text))

