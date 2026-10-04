"""
shortcuts.py — every keyboard shortcut and mouse interaction of the simulator,
listed per scope (where it works), the app-wide simulation shortcuts, and the
help panel that shows the list.

Keyboard shortcuts must be unique across the whole app. When adding a key or
a mouse gesture anywhere (main window, network visualizer, side view, ...),
add it to INTERACTIONS so the help panel shows it and clashes are visible in
one place.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QLabel, QTabWidget,
)

# scope → [(input, what it does)] — one tab per scope in the help panel.
INTERACTIONS = {
    'Everywhere': [
        ('F1',           'This panel (also the ? next to "Simulation")'),
        ('Ctrl+Space',   'Run / Pause the simulation (Pause keeps everything where it is)'),
        ('Ctrl+→',       'Step one physics tick'),
        ('Ctrl+R',       'Reset to the start'),
        ('W / S',        'Drive forward / back — while ⌨ Manual is on'),
        ('A / D',        'Turn left / right — while ⌨ Manual is on'),
        ('Space',        'Stop the motors — while ⌨ Manual is on'),
        ('↑ / ↓',        'Step every ManualBumpSensor up / down'),
        ('Cue letter',   'Hold to fire a ManualCueSensor (its own key, default R)'),
    ],
    'Arena': [
        ('A – F',              'Pick gradient tool A–F (not while ⌨ Manual is on)'),
        ('Z Y X W V',          'Pick object tool (not while ⌨ Manual is on)'),
        ('Drag (gradient tool)', 'Place a gradient patch: press at the centre, drag out the radius'),
        ('Drag from a robot (gradient tool)', 'Place a patch mounted on that robot (moves with it)'),
        ('Drag (Wall gradient)', 'Place a wall gradient: drag out its width from the arena edge'),
        ('Click (object tool)', 'Add a polygon vertex; click the first vertex to close the shape'),
        ('Shift+click (object tool)', 'Add a vertex snapped to 45°'),
        ('Right-click (object tool)', 'Cancel the polygon being drawn'),
        ('Click / drag (Wall paint)', 'Paint the nearest object / wall in the chosen colour'),
        ('Drag (↕ Sky tool)',  'Set the polarization e-vector direction'),
        ('Drag (↖ Move)',      'Move a gradient patch, an object or a robot'),
        ('Click a robot',      'Select that agent (Move / gradient / Wall / Sky tools)'),
        ('Right-click',        'Delete the patch / object / wall under the cursor'),
        ('Right-click a robot', 'Remove the gradients mounted on it'),
        ('Delete',             'Remove the selected agent'),
        ('Mouse wheel / right-drag', 'Zoom'),
    ],
    'Network': [
        ('Click a node',       'Select it and highlight its connections (click empty space to clear)'),
        ('Shift+click nodes',  'Multi-select (right-click for the group menu: save motif, copy)'),
        ('Double-click a node', 'Open its properties dialog'),
        ('Drag a node onto another', 'Connect them (opens the weight / filter dialog)'),
        ('Alt+drag a node',    'Move it to another column / position'),
        ('Drag a palette chip', 'Add a sensor / layer / body / note / motif where you drop it'),
        ('Right-click a node', 'Properties, mute, remove, oscilloscope, activation panel'),
        ('Right-click a connection', 'Edit weights, remove, add to / remove from the weight panel'),
        ('Right-click a note', 'Edit / collapse / delete'),
        ('Right-click a column', 'Set label / note'),
        ('Click a note',       'Select it (double-click opens it); click its ▸ glyph to collapse / expand'),
        ('Drag a note',        'Move it'),
        ('Delete',             'Remove the selected node / connection / note'),
        ('Ctrl+Z',             'Undo the last circuit edit'),
        ('Ctrl+C / Ctrl+V',    'Copy / paste the selected sub-circuit'),
        ('Drag empty space / wheel', 'Pan / zoom'),
        ('🔒 Lock',            'No moving, connecting, adding or removing — properties and view options still work'),
    ],
    'Side view': [
        ('Drag a container',   'Move it to another column / depth'),
        ('Drag into a gap',    'Insert it as a new column'),
        ('Drag the right edge', 'Span more columns; drag back to shrink the span'),
        ('Drag the left edge', 'Span leftwards, keeping the right edge fixed'),
    ],
}


def install_simulation_shortcuts(window, run_stop, step, reset, show_help):
    """Run / Step / Reset on Ctrl+Space / Ctrl+→ / Ctrl+R and this help on F1,
    active in every window of the application (main window, network
    visualizer, docks)."""
    keys = [('Ctrl+Space', run_stop), ('Ctrl+Right', step), ('Ctrl+R', reset),
            ('F1', show_help)]
    shortcuts = []
    for seq, slot in keys:
        sc = QShortcut(QKeySequence(seq), window)
        sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc.activated.connect(slot)
        shortcuts.append(sc)
    return shortcuts


def _table(rows):
    table = QTableWidget(len(rows), 2)
    table.setHorizontalHeaderLabels(["Input", "What it does"])
    for r, (inp, what) in enumerate(rows):
        a = QTableWidgetItem(inp)
        f = a.font(); f.setBold(True); a.setFont(f)
        table.setItem(r, 0, a)
        table.setItem(r, 1, QTableWidgetItem(what))
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setWordWrap(True)
    hdr = table.horizontalHeader()
    hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    return table


class ShortcutsDialog(QDialog):
    """Non-modal panel: one tab per scope, listing INTERACTIONS."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard and mouse")
        vl = QVBoxLayout(self)
        tabs = QTabWidget()
        for scope, rows in INTERACTIONS.items():
            tabs.addTab(_table(rows), scope)
        vl.addWidget(tabs)
        vl.addWidget(QLabel("Keys are unique across the app."))
        self.resize(640, 520)
