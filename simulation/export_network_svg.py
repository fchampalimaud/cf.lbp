"""
Headless SVG export for a network JSON file.

Usage (from simulation/2d/):
    python export_network_svg.py networks/FeedingBrain.json docs/assets/figures/feedingbrain_network.svg

Use --stroke-scale to shrink stroke widths so lines match the visualiser's
apparent thickness when the SVG is displayed at page width (default 0.5).
"""
import os, sys, re

# Must be set before any Qt import
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import argparse

from PySide6.QtWidgets import QApplication
from PySide6.QtCore    import QTimer

from circuit_model   import CircuitModel
from brain_serializer import load_network_file
from network_viz     import NetworkVisualizerWindow


class _FakeGui:
    """Minimal stand-in for SimulatorApp — only what NetworkVisualizerWindow reads."""
    def __init__(self, circuit):
        self.circuit  = circuit
        self.brain    = None
        self._net_viz = None          # window writes back here on close

    # These are called by toolbar buttons we never click; stubs prevent errors.
    def load_brain(self):   pass
    def _load_data_brain_network(self, *_): pass


def _fix_strokes(svg_bytes: bytes, scale: float) -> bytes:
    """Make strokes scale with the SVG geometry instead of staying fixed in screen pixels.

    pyqtgraph emits vector-effect="non-scaling-stroke" on its polylines, which
    keeps stroke widths constant in screen pixels regardless of how large the
    SVG is displayed.  That makes lines look proportionally thicker when the SVG
    is shown smaller than its export resolution.  Removing the attribute (or
    changing it to "none") restores natural scaling so the diagram matches what
    you see in the running visualiser.  An optional scale factor rescales all
    stroke-width values in addition.
    """
    # Drop non-scaling-stroke so strokes scale with the geometry
    svg_bytes = svg_bytes.replace(
        b'vector-effect="non-scaling-stroke"',
        b'vector-effect="none"',
    )
    # Also handle the stylesheet form if present
    svg_bytes = svg_bytes.replace(
        b'vector-effect:non-scaling-stroke',
        b'vector-effect:none',
    )
    if scale != 1.0:
        def _replace(m):
            val = float(m.group(1))
            return f'stroke-width="{val * scale:.4g}"'.encode()
        svg_bytes = re.sub(
            rb'stroke-width="([0-9]*\.?[0-9]+)"',
            _replace,
            svg_bytes,
        )
    return svg_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('network_json',
                        help='Path to the network JSON file (relative to simulation/2d/)')
    parser.add_argument('out_svg',
                        help='Destination SVG path')
    parser.add_argument('--width',        type=int,   default=900)
    parser.add_argument('--height',       type=int,   default=600)
    parser.add_argument('--stroke-scale', type=float, default=0.5,
                        help='Scale factor applied to all SVG stroke widths (default 0.5)')
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)

    sensors, layers, connections, hidden, disabled, _col_labels, bodies, joints, _ = \
        load_network_file(args.network_json)

    circuit = CircuitModel(
        sensors=sensors,
        layers=layers,
        connections=connections,
        bodies=bodies,
        joints=joints,
    )

    # Replicate what LBPSimulator._resolve_joint_sensor_refs() does:
    # ProprioceptiveSensor.n defaults to 1 and is updated at runtime based on
    # how many joints share its joint_id.  Do the same here so the SVG shows
    # the correct number of sensor nodes.
    from sensors import ProprioceptiveSensor
    for sensor in circuit.sensors:
        if isinstance(sensor, ProprioceptiveSensor) and sensor.joint_id:
            group = sorted(
                [jt for jt in circuit.joints
                 if jt.motor_layer_name == sensor.joint_id],
                key=lambda j: j.motor_output_idx,
            )
            sensor.n = len(group) if group else 1

    gui = _FakeGui(circuit)
    win = NetworkVisualizerWindow(gui)
    win._hidden_groups   = set(hidden)
    win._disabled_groups = set(disabled)
    win.resize(args.width, args.height)
    win.show()

    out_path = os.path.abspath(args.out_svg)

    def _export():
        try:
            from pyqtgraph.exporters import SVGExporter
            # Re-fit the view to scene content in data-coordinate space so the
            # SVG viewBox and item positions are consistent regardless of the
            # physical/logical DPI ratio of the offscreen platform.
            win._plot.autoRange(padding=0.08)
            exporter = SVGExporter(win._plot)
            svg_bytes = exporter.export(toBytes=True)
            svg_bytes = _fix_strokes(svg_bytes, args.stroke_scale)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'wb') as f:
                f.write(svg_bytes)
            print(f'Saved: {out_path}  (stroke-scale={args.stroke_scale})')
        except Exception as e:
            print(f'Export error: {e}', file=sys.stderr)
            import traceback; traceback.print_exc()
        finally:
            app.quit()

    # Give the visualizer two ticks to finish building before exporting
    QTimer.singleShot(300, _export)
    app.exec()


if __name__ == '__main__':
    main()
