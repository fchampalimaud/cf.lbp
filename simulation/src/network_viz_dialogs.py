import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QLineEdit, QCheckBox, QDialog, QFormLayout, QDialogButtonBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QDoubleSpinBox,
    QStackedWidget, QTextEdit, QGroupBox, QListWidget, QAbstractItemView,
)
from PySide6.QtCore import Qt, QPointF, QEvent, QMimeData, QObject
from PySide6.QtGui import QFont, QDrag

from sim_constants import C, _IDX_MANUAL

from network_viz_layout import _make_conv2d_filter

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


def _small_bold_font():
    f = QFont()
    f.setPointSize(9)
    f.setBold(True)
    return f


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
        self._unif_amp = QDoubleSpinBox(); self._unif_amp.setRange(-100, 100); self._unif_amp.setSingleStep(0.001); self._unif_amp.setDecimals(6); self._unif_amp.setValue(1.0)
        unif_f.addRow("Amplitude", self._unif_amp)
        _tip(self._unif_amp, "Uniform weight applied to all connections")
        stack.addWidget(unif_w)

        # Cosine
        cos_w = QWidget(); cos_f = QFormLayout(cos_w); cos_f.setContentsMargins(0, 4, 0, 0)
        self._cos_amp  = QDoubleSpinBox(); self._cos_amp.setRange(-100, 100); self._cos_amp.setSingleStep(0.001); self._cos_amp.setDecimals(6); self._cos_amp.setValue(1.0)
        self._cos_ph0  = QDoubleSpinBox(); self._cos_ph0.setRange(-360, 360); self._cos_ph0.setSingleStep(5); self._cos_ph0.setValue(0); self._cos_ph0.setSuffix(" °")
        self._cos_step = QDoubleSpinBox(); self._cos_step.setRange(-360, 360); self._cos_step.setSingleStep(5)
        self._cos_step.setValue(round(360.0 / nt, 1) if nt > 1 else 180.0); self._cos_step.setSuffix(" °")
        self._cos_bias = QDoubleSpinBox(); self._cos_bias.setRange(-100, 100); self._cos_bias.setSingleStep(0.001); self._cos_bias.setDecimals(6); self._cos_bias.setValue(0.0)
        cos_f.addRow("Amplitude",  self._cos_amp)
        _tip(self._cos_amp, "Peak amplitude of the cosine wave — scales the whole pattern")
        cos_f.addRow("Phase₀",    self._cos_ph0)
        _tip(self._cos_ph0, "Phase offset for target neuron 0 (degrees)")
        cos_f.addRow("Phase step", self._cos_step)
        _tip(self._cos_step, "Phase increment per target neuron (degrees)")
        cos_f.addRow("Bias", self._cos_bias)
        _tip(self._cos_bias, "Constant added to every weight — shifts the wave up or down")
        cos_f.addRow(QLabel(f"<span style='font-size:8px;color:{C['muted']}'>"
                            f"W[i,j] = amp·cos(2π·j/{ns} − phase₀ − i·step) + bias</span>"))
        stack.addWidget(cos_w)

        # Gaussian
        gau_w = QWidget(); gau_f = QFormLayout(gau_w); gau_f.setContentsMargins(0, 4, 0, 0)
        self._gau_amp  = QDoubleSpinBox(); self._gau_amp.setRange(-100, 100); self._gau_amp.setSingleStep(0.001); self._gau_amp.setDecimals(6); self._gau_amp.setValue(1.0)
        self._gau_sig  = QDoubleSpinBox(); self._gau_sig.setRange(0.01, 2.0); self._gau_sig.setSingleStep(0.05); self._gau_sig.setValue(0.2); self._gau_sig.setDecimals(3)
        self._gau_off  = QDoubleSpinBox(); self._gau_off.setRange(-1.0, 1.0); self._gau_off.setSingleStep(0.05); self._gau_off.setValue(0.0); self._gau_off.setDecimals(3)
        self._gau_base = QDoubleSpinBox(); self._gau_base.setRange(-100, 100); self._gau_base.setSingleStep(0.001); self._gau_base.setDecimals(6); self._gau_base.setValue(0.0)
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
        self._mxh_exc  = QDoubleSpinBox(); self._mxh_exc.setRange(0, 100);  self._mxh_exc.setSingleStep(0.001); self._mxh_exc.setDecimals(6); self._mxh_exc.setValue(2.0)
        self._mxh_sige = QDoubleSpinBox(); self._mxh_sige.setRange(0.01, 1.0); self._mxh_sige.setSingleStep(0.01); self._mxh_sige.setValue(round(_nn_spacing * 1.4, 3)); self._mxh_sige.setDecimals(3)
        self._mxh_inh  = QDoubleSpinBox(); self._mxh_inh.setRange(0, 100);  self._mxh_inh.setSingleStep(0.001); self._mxh_inh.setDecimals(6); self._mxh_inh.setValue(1.0)
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
        self._mxh_zero_diag = QCheckBox("Zero diagonal (no self-connections)"); self._mxh_zero_diag.setChecked(True)
        mxh_f.addRow(self._mxh_zero_diag)
        mxh_f.addRow(QLabel(f"<span style='font-size:8px;color:{C['muted']}'>"
                            f"W[i,j] = w_e·exp(−d²/2σ_e²) − w_i·exp(−d²/2σ_i²)</span>"))
        mxh_f.addRow(QLabel(f"<span style='font-size:8px;color:{C['muted']}'>"
                            f"Nearest-neighbour spacing: {_nn_spacing}  "
                            f"(= 1/{max(ns, nt)} ring fractions)</span>"))
        stack.addWidget(mxh_w)

        # One-to-one
        oto_w = QWidget(); oto_f = QFormLayout(oto_w); oto_f.setContentsMargins(0, 4, 0, 0)
        self._oto_amp = QDoubleSpinBox(); self._oto_amp.setRange(-100, 100); self._oto_amp.setSingleStep(0.001); self._oto_amp.setDecimals(6); self._oto_amp.setValue(1.0)
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
        self._rnd_unif_amp = QDoubleSpinBox(); self._rnd_unif_amp.setRange(0.000001, 100); self._rnd_unif_amp.setSingleStep(0.001); self._rnd_unif_amp.setDecimals(6); self._rnd_unif_amp.setValue(1.0)
        rnd_unif_f.addRow("Amplitude", self._rnd_unif_amp)
        _tip(self._rnd_unif_amp, "Half-range of uniform distribution: W[i,j] ~ U(−amp, +amp)")
        rnd_unif_f.addRow(QLabel(f"<span style='font-size:8px;color:{C['muted']}'>"
                                 f"W[i,j] ~ U(−amp, +amp)</span>"))
        stack.addWidget(rnd_unif_w)

        # Rand normal
        rnd_norm_w = QWidget(); rnd_norm_f = QFormLayout(rnd_norm_w); rnd_norm_f.setContentsMargins(0, 4, 0, 0)
        self._rnd_norm_std = QDoubleSpinBox(); self._rnd_norm_std.setRange(0.000001, 100); self._rnd_norm_std.setSingleStep(0.001); self._rnd_norm_std.setDecimals(6); self._rnd_norm_std.setValue(0.1)
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
        manual_w = QWidget(); manual_f = QFormLayout(manual_w); manual_f.setContentsMargins(0, 4, 0, 0)
        manual_lbl = QLabel(
            "<span style='font-size:9px;color:#888'>"
            "Current weights are preserved as-is.<br>"
            "Click <b>OK</b> to keep them, or choose another<br>"
            "pattern and click <b>Apply ▶</b> to regenerate.</span>")
        manual_lbl.setWordWrap(True)
        manual_f.addRow(manual_lbl)
        self._manual_scale = QDoubleSpinBox(); self._manual_scale.setRange(-1000, 1000)
        self._manual_scale.setSingleStep(0.1); self._manual_scale.setDecimals(6)
        self._manual_scale.setValue(1.0)
        manual_f.addRow("Scale", self._manual_scale)
        _tip(self._manual_scale, "Multiplies every hand-entered weight — scale the whole "
                                  "matrix up or down without retyping each cell")
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
        from network_viz_render import _MatrixHeatmapWidget
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
            if 'bias' in c_: self._cos_bias.setValue(c_['bias'])
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
            ma = p.get('manual', {})
            if 'scale' in ma: self._manual_scale.setValue(ma['scale'])
            self._pattern_cb.setCurrentIndex(p.get('type', 8))
        else:
            self._pattern_cb.setCurrentIndex(_IDX_MANUAL)

        self.resize(max(520, ns * 35 + 320), 520)

    def _compute_W(self):
        ns, nt = self._ns, self._nt
        idx = self._pattern_cb.currentIndex()
        if idx == _IDX_MANUAL:
            return None
        try:
            if idx == 0:
                W = self._unif_amp.value() * np.ones((nt, ns))
            elif idx == 1:
                # Difference of source angle and target phase (not a sum) so this
                # is a proper circulant/ring kernel — constant along i-j=const
                # diagonals, matching the Gaussian/Mexican-hat patterns below,
                # which also key off |i-j| rather than i+j. A sum here would band
                # along anti-diagonals instead, which looks visually "rotated
                # perpendicular" relative to every other pattern in this dialog.
                th_s = 2 * np.pi * np.arange(ns) / ns
                ph_t = np.deg2rad(self._cos_ph0.value() + np.arange(nt) * self._cos_step.value())
                W = (self._cos_amp.value() * np.cos(th_s[None, :] - ph_t[:, None])
                     + self._cos_bias.value())
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
                if nt == ns and self._mxh_zero_diag.isChecked():
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
            # Manual mode: preview current cell values (scaled) without overwriting them
            ns, nt = self._ns, self._nt
            W_cur = np.zeros((nt, ns))
            for ii in range(nt):
                for jj in range(ns):
                    it = self._table.item(ii, jj)
                    try:
                        W_cur[ii, jj] = float(it.text()) if it else 0.0
                    except ValueError:
                        pass
            self._hmap.set_matrix(W_cur * self._manual_scale.value())
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
        if self._pattern_cb.currentIndex() != _IDX_MANUAL:
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
        if self._pattern_cb.currentIndex() == _IDX_MANUAL:
            W_final = W_final * self._manual_scale.value()
        params_out = {
            'type':        self._pattern_cb.currentIndex(),
            'uniform':     {'amp': self._unif_amp.value()},
            'cosine':      {'amp': self._cos_amp.value(), 'ph0': self._cos_ph0.value(),
                            'step': self._cos_step.value(), 'bias': self._cos_bias.value()},
            'gaussian':    {'amp': self._gau_amp.value(), 'sig': self._gau_sig.value(),
                            'off': self._gau_off.value(), 'base': self._gau_base.value()},
            'mexican_hat': {'exc': self._mxh_exc.value(), 'sige': self._mxh_sige.value(),
                            'inh': self._mxh_inh.value(), 'sigi': self._mxh_sigi.value()},
            'one_to_one':  {'amp': self._oto_amp.value(), 'off': self._oto_off.value()},
            'rand_uniform': {'amp': self._rnd_unif_amp.value()},
            'rand_normal':  {'std': self._rnd_norm_std.value()},
            'expression':  {'code': self._expr_edit.toPlainText()},
            'manual':      {'scale': self._manual_scale.value()},
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
        self._pattern_cb.setCurrentIndex(_IDX_MANUAL)  # switch to Manual to preserve preset


# ============================================================
# FILTER STACK DIALOG  (Conv2dLayer connections)
# ============================================================

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
        from network_viz_render import _MatrixHeatmapWidget
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


class _DialogsMixin:
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

    def _receptor_table_widget(self, existing_mods, learning=False):
        """Build the modulator receptors table and its add/remove buttons.

        Every row always carries a response `Mode` (absolute / derivative /
        integral) alongside the modulator name/scale/site — absolute
        reproduces today's only behavior (raw current value), derivative
        reacts to a rise (onset-like), integral accumulates over time.

        learning=True (LearningLayerBase-family layers only) adds two more
        columns: `Drives Plasticity` (checkbox) and `Threshold` — the row's
        mode-transformed value must cross Threshold for that row to
        contribute to the layer's reward that tick. This supersedes the old
        standalone `reward_modulator` field (still honored for back-compat,
        additively, if a saved layer still has it set).

        Returns (table, btns_widget) ready to pass to form.addRow().
        """
        _SITES = ['pre', 'post', 'none']
        _MODES = ['absolute', 'derivative', 'integral']
        n_cols = 6 if learning else 4

        def _make_site_combo(current='post'):
            cb = QComboBox()
            cb.addItems(_SITES)
            cb.setCurrentText(current if current in _SITES else 'post')
            return cb

        def _make_mode_combo(current='absolute'):
            cb = QComboBox()
            cb.addItems(_MODES)
            cb.setCurrentText(current if current in _MODES else 'absolute')
            return cb

        headers = ["Modulator", "Scale", "Site", "Mode"]
        if learning:
            headers += ["Drives Plasticity", "Threshold"]

        table = QTableWidget(0, n_cols)
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, n_cols):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setFixedHeight(140)

        def _add_row(mod_name='', scale='1.0', site='post', mode='absolute',
                     drives_plasticity=False, threshold='0.0'):
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(mod_name))
            table.setItem(row, 1, QTableWidgetItem(str(scale)))
            table.setCellWidget(row, 2, _make_site_combo(site))
            table.setCellWidget(row, 3, _make_mode_combo(mode))
            if learning:
                chk = QCheckBox()
                chk.setChecked(bool(drives_plasticity))
                table.setCellWidget(row, 4, chk)
                table.setItem(row, 5, QTableWidgetItem(str(threshold)))
            return row

        for row_data in (existing_mods or []):
            name, scale, site = row_data[0], row_data[1], row_data[2]
            mode              = row_data[3] if len(row_data) > 3 else 'absolute'
            drives_plasticity = row_data[4] if len(row_data) > 4 else False
            threshold         = row_data[5] if len(row_data) > 5 else 0.0
            _add_row(name, scale, site, mode, drives_plasticity, threshold)

        btns = QWidget()
        lay  = QHBoxLayout(btns)
        lay.setContentsMargins(0, 0, 0, 0)
        btn_add = QPushButton("Add Receptor")
        btn_rem = QPushButton("Remove Selected")

        def _add():
            r = _add_row()
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

    def _build_param_editors(self, form, dlg, status_lbl, params, cur_values,
                             joints=None, bodies=None, is_proprio=False, layers=None):
        """Build param editor widgets from param_defs and add them to form.

        cur_values: {pname: current_value} — use param default when key is absent.
        joints/bodies/is_proprio/layers are sensor-only; omit (or pass None/False) for layers.
        Returns editors dict: {pname: (kind_or_ptype, widget, is_angle)}.
        """
        if joints is None:
            joints = []
        if bodies is None:
            bodies = []
        if layers is None:
            layers = []
        editors = {}
        _rendered = set()
        for param in params:
            pname, ptype, default, desc = param[:4]
            if pname in _rendered:
                continue
            choices   = param[4] if len(param) > 4 else None
            group_tag = param[5] if len(param) > 5 else None
            cur = cur_values.get(pname, default)

            # --- grouped row: render several params side-by-side ---
            if group_tag is not None:
                grp = [p for p in params if len(p) > 5 and p[5] == group_tag]
                row_w = QWidget()
                hl = QHBoxLayout(row_w)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.setSpacing(4)
                for gp in grp:
                    gpname, gptype, gpdefault, gpdesc = gp[:4]
                    lbl = gpname.rsplit('_', 1)[-1].upper()
                    hl.addWidget(QLabel(lbl + ':'))
                    sw = QSpinBox()
                    sw.setMinimum(-1); sw.setMaximum(100)
                    try:    sw.setValue(int(cur_values.get(gpname, gpdefault)))
                    except Exception: pass
                    hl.addWidget(sw)
                    f = _HoverStatus(status_lbl, gpdesc, dlg)
                    sw.installEventFilter(f)
                    dlg._filters.append(f)
                    editors[gpname] = (gptype, sw, False)
                    _rendered.add(gpname)
                hl.addStretch()
                form.addRow(group_tag, row_w)
                continue

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
                from neurons_simple import MotorLayer
                for lyr in layers:
                    if lyr.name not in seen and isinstance(lyr, MotorLayer) and lyr.n is not None:
                        w.addItem(lyr.name, lyr.name)
                        seen[lyr.name] = True
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
                w = QLineEdit('' if cur is None else (str(cur) if cur != '' else str(default)))

            form.addRow(pname, w)
            f = _HoverStatus(status_lbl, desc, dlg)
            w.installEventFilter(f)
            dlg._filters.append(f)
            editors[pname] = (ptype, w, is_angle)

        return editors

    def _read_param_editors(self, editors):
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

    def _sensor_dialog(self, stype, sensor=None, target_container=None):
        """Unified create/edit dialog for sensors.

        sensor=None: creation mode (title "Add …", fields show defaults).
        sensor=<obj>: edit mode (title "Edit …: name", fields show live values).
        """
        from sensors import SENSOR_REGISTRY, ProprioceptiveSensor, BaseSensor
        cls = SENSOR_REGISTRY.get(stype)
        if cls is None:
            return
        params = list(cls.param_defs() if hasattr(cls, 'param_defs') else [])
        existing = {p[0] for p in params}
        params += [p for p in BaseSensor._sensor_base_param_defs() if p[0] not in existing]

        is_edit = sensor is not None
        is_proprio = (isinstance(sensor, ProprioceptiveSensor) if is_edit
                      else issubclass(cls, ProprioceptiveSensor))
        title = f"Edit {stype}: {sensor.name}" if is_edit else f"Add {stype}"
        dlg, form, status_lbl = self._make_param_dialog(title, getattr(cls, 'help_text', None))

        default_name = sensor.name if is_edit else f"sensor{len(self.gui.circuit.sensors)}"
        name_edit = QLineEdit(default_name)
        form.addRow("name", name_edit)

        joints = self.gui.circuit.joints if hasattr(self.gui.circuit, 'joints') else []
        bodies = self.gui.circuit.bodies if hasattr(self.gui.circuit, 'bodies') else []
        cur_values = ({p[0]: getattr(sensor, p[0], p[2]) for p in params}
                      if is_edit else {p[0]: p[2] for p in params})
        editors = self._build_param_editors(
            form, dlg, status_lbl, params, cur_values,
            joints=joints, bodies=bodies, is_proprio=is_proprio,
            layers=self.gui.circuit.layers)

        body_combo = None
        if not is_proprio and bodies:
            cur_body_ids = (getattr(sensor, 'body_ids', None) or ['root']) if is_edit else None
            body_combo = self._make_body_combo(bodies, cur_body_ids)
            form.addRow("mounted on body", body_combo)

        cur_nt = (getattr(sensor, 'neuromodulator_transmitter', None) or '') if is_edit else ''
        nt_edit = QLineEdit(cur_nt)
        nt_edit.setPlaceholderText("e.g. dopamine  (leave empty if not a transmitter)")
        form.addRow("neuromodulator transmitter", nt_edit)

        cur_mod_color = (getattr(sensor, 'neuromodulator_color', None) or '') if is_edit else ''
        mod_color_edit = QLineEdit(cur_mod_color)
        mod_color_edit.setPlaceholderText("#FF6600  (hex color for this transmitter)")
        form.addRow("transmitter color", mod_color_edit)

        existing_mods = (getattr(sensor, 'modulators', []) or []) if is_edit else []
        receptor_table, receptor_btns = self._receptor_table_widget(existing_mods)
        form.addRow("Modulator receptors", receptor_table)
        form.addRow(receptor_btns)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_mods = []
        for row in range(receptor_table.rowCount()):
            name_item  = receptor_table.item(row, 0)
            scale_item = receptor_table.item(row, 1)
            site_combo = receptor_table.cellWidget(row, 2)
            mode_combo = receptor_table.cellWidget(row, 3)
            if not (name_item and scale_item and site_combo and mode_combo):
                continue
            n = name_item.text().strip()
            if not n:
                continue
            try:
                new_mods.append((n, float(scale_item.text()), site_combo.currentText(), mode_combo.currentText()))
            except ValueError:
                pass

        if is_edit:
            # ── Edit mode ─────────────────────────────────────────────────────
            self._push_undo()
            new_name = name_edit.text().strip()
            if new_name and new_name != sensor.name:
                old_name = sensor.name
                sensor.name = new_name
                self._rekey_container_metadata(old_name, new_name)
                from dataclasses import replace as _dc_replace
                self.gui.circuit.connections = [
                    _dc_replace(c, src=new_name) if c.src == old_name else
                    _dc_replace(c, src=new_name + c.src[len(old_name):])
                    if c.src.startswith(old_name + '_') else c
                    for c in self.gui.circuit.connections
                ]
            for pname, val in self._read_param_editors(editors).items():
                setattr(sensor, pname, val)
            # Force a fresh connections list identity so network_runner's
            # conn_id-gated shape caches (size-reconciliation, camera shape
            # propagation) re-run and pick up any edited shape param (e.g.
            # camera width/height) instead of staying frozen at pre-edit values.
            self.gui.circuit.connections = list(self.gui.circuit.connections)
            if body_combo is not None:
                sensor.body_ids = body_combo.currentData() or ['root']
            nt = nt_edit.text().strip()
            sensor.neuromodulator_transmitter = nt if nt else None
            mc = mod_color_edit.text().strip()
            sensor.neuromodulator_color = mc if mc else None
            sensor.modulators = new_mods
            if isinstance(sensor, ProprioceptiveSensor) and sensor.joint_id:
                grp = sorted(
                    [jt for jt in joints if jt.motor_layer_name == sensor.joint_id],
                    key=lambda j: j.motor_output_idx
                )
                sensor._joint_refs = grp
                sensor._layer_ref  = None
                if grp:
                    sensor.n = len(grp)
                else:
                    lyr = next((l for l in self.gui.circuit.layers
                                if l.name == sensor.joint_id), None)
                    sensor._layer_ref = lyr
                    sensor.n = (lyr.n or 1) if lyr is not None else 1
            self._build_without_selection_filter()

        else:
            # ── Create mode ───────────────────────────────────────────────────
            self._push_undo()
            kwargs = {'name': name_edit.text().strip() or f'sensor{len(self.gui.circuit.sensors)}'}
            kwargs.update(self._read_param_editors(editors))
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
            if new_mods:
                new_sensor.modulators = new_mods
            if isinstance(new_sensor, ProprioceptiveSensor) and new_sensor.joint_id:
                group = sorted(
                    [jt for jt in joints if jt.motor_layer_name == new_sensor.joint_id],
                    key=lambda j: j.motor_output_idx
                )
                new_sensor._joint_refs = group
                new_sensor._layer_ref  = None
                if group:
                    new_sensor.n = len(group)
                else:
                    lyr = next((l for l in self.gui.circuit.layers
                                if l.name == new_sensor.joint_id), None)
                    new_sensor._layer_ref = lyr
                    new_sensor.n = (lyr.n or 1) if lyr is not None else 1
            self.gui.circuit.sensors.append(new_sensor)
            if target_container is not None:
                new_sensor.layer = target_container
            self.build()

    def _body_dialog(self, layer):
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

        if mirrored and ref_body.name.endswith('_L'):
            base_name = ref_body.name[:-2]
        else:
            base_name = ref_body.name

        dlg, form, _ = self._make_param_dialog("Edit Body")

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
            self._rekey_container_metadata(lname, new_name)
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

    def _rekey_container_metadata(self, old_name, new_name):
        """Migrate any container_labels/container_notes entry referencing
        *old_name* to use *new_name* instead, so an explicit label/note set on
        a container survives renaming one of its occupants — container
        identity is derived from occupant names (see _container_key), so
        without this the entry would otherwise be silently orphaned under a
        now-stale key."""
        if old_name == new_name:
            return
        for store in (self._container_labels, self._container_notes):
            for old_key in list(store.keys()):
                parts = old_key.split('|')
                if old_name not in parts:
                    continue
                new_key = '|'.join(sorted(new_name if p == old_name else p for p in parts))
                if new_key != old_key:
                    store[new_key] = store.pop(old_key)

    def _container_note_dialog(self, container):
        """Add/edit/clear a container's note. Returns True if the note text
        changed (caller should call _refresh_container_note), else False."""
        key     = self._container_key(container)
        current = self._container_notes.get(key, '')
        dlg = QDialog(self)
        dlg.setWindowTitle('Container note')
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel(
            "<span style='font-size:9px'>Free text — not part of the circuit.</span>"))
        editor = QTextEdit()
        editor.setPlainText(current)
        editor.setMinimumSize(220, 120)
        vl.addWidget(editor)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        vl.addWidget(bb)
        editor.setFocus()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        text = editor.toPlainText()
        if text == current:
            return False
        self._push_undo()
        if text.strip():
            self._container_notes[key] = text
        else:
            self._container_notes.pop(key, None)
        return True

    def _note_dialog(self, note=None, pos=None):
        """Add or edit a sticky note. Returns the Note, or None if cancelled."""
        from circuit_model import Note
        is_edit = note is not None
        dlg = QDialog(self)
        dlg.setWindowTitle('Edit note' if is_edit else 'Add note')
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel(
            "<span style='font-size:9px'>Free text — not part of the circuit.</span>"))
        editor = QTextEdit()
        editor.setPlainText(note.text if is_edit else '')
        editor.setMinimumSize(220, 120)
        vl.addWidget(editor)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        vl.addWidget(bb)
        editor.setFocus()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        text = editor.toPlainText()
        self._push_undo()
        if is_edit:
            note.text = text
            return note
        new_note = Note(x=pos.x(), y=pos.y(), text=text)
        self.gui.circuit.notes.append(new_note)
        return new_note

    def _layer_dialog(self, ltype, layer=None, target_container=None):
        """Unified create/edit dialog for layers.

        layer=None: creation mode (title "Add …", fields show defaults).
        layer=<obj>: edit mode (title "Edit …: name", fields show live values).
        """
        from neurons import LAYER_REGISTRY, DynamicsBase, LearningLayerBase
        cls    = LAYER_REGISTRY.get(ltype)
        is_learning = cls is not None and issubclass(cls, LearningLayerBase)
        params = list(cls.param_defs() if cls is not None else [])
        if not params:
            return
        if cls is not None and issubclass(cls, DynamicsBase):
            existing = {p[0] for p in params}
            params += [p for p in DynamicsBase._dynamics_param_defs() if p[0] not in existing]

        is_edit = layer is not None
        title   = f"Edit {ltype}: {layer.name}" if is_edit else f"Add {ltype}"
        dlg, form, status_lbl = self._make_param_dialog(title, getattr(cls, 'help_text', None))

        pair_name = getattr(layer, 'lateral_pair', None) if is_edit else None
        if pair_name:
            note = QLabel(
                f"<span style='font-size:8px;color:#4888CC'>"
                f"Lateralized pair — edits also apply to <b>{pair_name}</b></span>")
            form.addRow(note)

        default_name = layer.name if is_edit else f"layer{len(self.gui.circuit.layers)}"
        name_edit = QLineEdit(default_name)
        form.addRow("name", name_edit)

        cur_values = ({p[0]: getattr(layer, p[0], p[2]) for p in params}
                      if is_edit else {p[0]: p[2] for p in params})
        editors = self._build_param_editors(form, dlg, status_lbl, params, cur_values)

        z_val = (getattr(layer, 'z', 0) or 0) if is_edit else 0
        z_spin = QSpinBox()
        z_spin.setMinimum(0); z_spin.setMaximum(20)
        z_spin.setValue(z_val)
        z_spin.setToolTip("Subsumption depth — controls Side View row and 3D visualizer depth axis")
        form.addRow("Subsumption depth", z_spin)

        cur_nt = (getattr(layer, 'neuromodulator_transmitter', None) or '') if is_edit else ''
        nt_edit = QLineEdit(cur_nt)
        nt_edit.setPlaceholderText("e.g. dopamine  (leave empty if not a transmitter)")
        form.addRow("neuromodulator transmitter", nt_edit)

        cur_mod_color = (getattr(layer, 'neuromodulator_color', None) or '') if is_edit else ''
        mod_color_edit = QLineEdit(cur_mod_color)
        mod_color_edit.setPlaceholderText("#FF6600  (hex color for this transmitter)")
        form.addRow("transmitter color", mod_color_edit)

        existing_mods = (getattr(layer, 'modulators', []) or []) if is_edit else []
        receptor_table, receptor_btns = self._receptor_table_widget(existing_mods, learning=is_learning)
        form.addRow("Modulator receptors", receptor_table)
        form.addRow(receptor_btns)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_mods = []
        for row in range(receptor_table.rowCount()):
            name_item  = receptor_table.item(row, 0)
            scale_item = receptor_table.item(row, 1)
            site_combo = receptor_table.cellWidget(row, 2)
            mode_combo = receptor_table.cellWidget(row, 3)
            if not (name_item and scale_item and site_combo and mode_combo):
                continue
            n = name_item.text().strip()
            if not n:
                continue
            try:
                mod_row = [n, float(scale_item.text()), site_combo.currentText(), mode_combo.currentText()]
                if is_learning:
                    chk = receptor_table.cellWidget(row, 4)
                    th_item = receptor_table.item(row, 5)
                    mod_row.append(bool(chk.isChecked()) if chk is not None else False)
                    mod_row.append(float(th_item.text()) if th_item is not None else 0.0)
                new_mods.append(tuple(mod_row))
            except ValueError:
                pass

        if is_edit:
            # ── Edit mode ─────────────────────────────────────────────────────
            all_vals = self._read_param_editors(editors)
            old_n = layer.n
            new_n = all_vals.get('n', old_n)
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
            new_name = name_edit.text().strip()
            if new_name and new_name != layer.name:
                old_name = layer.name
                layer.name = new_name
                self._rekey_container_metadata(old_name, new_name)
                from dataclasses import replace as _dc_replace
                self.gui.circuit.connections = [
                    _dc_replace(c,
                                src=new_name if c.src == old_name else c.src,
                                tgt=new_name if c.tgt == old_name else c.tgt)
                    for c in self.gui.circuit.connections
                ]
            # Apply params — tau last so it overrides tau_rise/tau_decay
            tau_val = all_vals.pop('tau', None)
            for pname, val in all_vals.items():
                setattr(layer, pname, val)
            if tau_val is not None:
                setattr(layer, 'tau', tau_val)
            layer.z = z_spin.value()
            nt = nt_edit.text().strip()
            layer.neuromodulator_transmitter = nt if nt else None
            mc = mod_color_edit.text().strip()
            layer.neuromodulator_color = mc if mc else None
            layer.modulators = new_mods

            import torch as _torch
            from neurons import Conv2dLayer as _C2d, Reichardt2dLayer as _R2d, Leaky2dLayer as _L2d
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

            def _sync_reichardt(lyr):
                if isinstance(lyr, _R2d):
                    if lyr.pool == 'none':
                        lyr.viz_n = 1
                        if not hasattr(lyr, '_last_frame'):
                            lyr._last_frame = None
                    else:
                        try:
                            del lyr.viz_n
                        except AttributeError:
                            pass
                        lyr.n = lyr.n_directions

            _sync_conv_n(layer)
            _sync_reichardt(layer)

            # If lateralized was just turned on (had no pair before), create the _R partner now.
            if getattr(layer, 'lateralized', False) and not pair_name and isinstance(layer, (_C2d, _L2d, _R2d)):
                base_name = layer.name
                if base_name.endswith('_L') or base_name.endswith('_R'):
                    base_name = base_name[:-2]
                if not layer.name.endswith('_L'):
                    old_layer_name = layer.name
                    layer.name = base_name + '_L'
                    if hasattr(self.gui.brain, old_layer_name):
                        delattr(self.gui.brain, old_layer_name)
                    setattr(self.gui.brain, layer.name, layer)
                    from dataclasses import replace as _dc_repl
                    self.gui.circuit.connections = [
                        _dc_repl(c,
                                 src=layer.name if c.src == old_layer_name else c.src,
                                 tgt=layer.name if c.tgt == old_layer_name else c.tgt)
                        for c in self.gui.circuit.connections
                    ]
                r_kwargs = {pname: getattr(layer, pname)
                            for pname, *_ in params if hasattr(layer, pname)}
                r_kwargs['name'] = base_name + '_R'
                r_kwargs['lateralized'] = True
                partner = type(layer)(**r_kwargs)
                for attr in ('z', 'layer', 'group', 'neuromodulator_transmitter',
                             'neuromodulator_color', 'modulators'):
                    if hasattr(layer, attr):
                        setattr(partner, attr, getattr(layer, attr))
                self.gui.circuit.layers.append(partner)
                setattr(self.gui.brain, partner.name, partner)
                _sync_conv_n(partner)
                _sync_reichardt(partner)
                layer.lateral_pair = partner.name
                partner.lateral_pair = layer.name
                pair_name = partner.name

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
                    _sync_reichardt(partner)

            # Force a fresh connections list identity so network_runner's
            # conn_id-gated shape caches (size-reconciliation, camera shape
            # propagation) re-run and pick up any edited shape param (e.g.
            # n_filters/kernel_size) instead of staying frozen at pre-edit values.
            self.gui.circuit.connections = list(self.gui.circuit.connections)
            self._build_without_selection_filter()

        else:
            # ── Create mode ───────────────────────────────────────────────────
            kwargs = {'name': name_edit.text().strip() or f'layer{len(self.gui.circuit.layers)}'}
            kwargs.update(self._read_param_editors(editors))
            nt = nt_edit.text().strip()
            if nt:
                kwargs['neuromodulator_transmitter'] = nt
            mc = mod_color_edit.text().strip()
            if mc:
                kwargs['neuromodulator_color'] = mc
            if new_mods:
                kwargs['modulators'] = new_mods

            self._push_undo()
            from neurons import RingAttractorLayer as _RAL, Conv2dLayer as _C2d, Leaky2dLayer as _L2dCls, Reichardt2dLayer as _R2dCls
            if issubclass(cls, _C2d) and kwargs.get('pool') == 'none' and int(kwargs.get('n_filters', 1)) > 1:
                QMessageBox.warning(self, "Conv2dLayer",
                                    "pool='none' requires n_filters=1.\n\n"
                                    "n_filters has been corrected to 1.")
                kwargs['n_filters'] = 1

            if kwargs.get('lateralized') and issubclass(cls, (_C2d, _L2dCls, _R2dCls)):
                base_name = kwargs.pop('name')
                new_layers = []
                for side in ('_L', '_R'):
                    kw = dict(kwargs, name=base_name + side, lateralized=True)
                    lyr = cls(**kw)
                    self.gui.circuit.layers.append(lyr)
                    setattr(self.gui.brain, lyr.name, lyr)
                    new_layers.append(lyr)
                new_layers[0].lateral_pair = new_layers[1].name
                new_layers[1].lateral_pair = new_layers[0].name
                for lyr in new_layers:
                    lyr.z = z_spin.value() or None
                    if target_container is not None:
                        lyr.layer = target_container
            else:
                new_layer = cls(**kwargs)
                self.gui.circuit.layers.append(new_layer)
                setattr(self.gui.brain, new_layer.name, new_layer)
                new_layer.z = z_spin.value()
                if target_container is not None:
                    new_layer.layer = target_container
                if isinstance(new_layer, _RAL):
                    W = _RAL.default_kernel(new_layer.n)
                    from circuit_model import Connection as _Conn
                    self.gui.circuit.connections.append(
                        _Conn(new_layer.name, new_layer.name, W))
                    # Force a fresh connections list identity — see the matching
                    # comment in network_viz_editing.py's _insert_motif — so
                    # network_runner's conn_id-gated caches don't silently miss
                    # this default self-recurrent connection.
                    self.gui.circuit.connections = list(self.gui.circuit.connections)
            self.build()

    _ANGLE_PARAMS = {'angle_spread', 'center_angle', 'arc_angle', 'mount_angle', 'fov', 'vertical_angle'}

