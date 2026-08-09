from collections import defaultdict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsLineItem,
    QGraphicsSimpleTextItem,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPen, QBrush, QPainter

# ── Grid geometry ───────────────────────────────────────────────────────────────
_CELL_W   = 130   # width of one column slot (px)
_CELL_H   = 72    # cell height
_GAP_X    = 12    # horizontal gap between slots (shows as background = "table border")
_GAP_Y    = 12    # vertical gap
_PAD_TOP  = 8     # top margin (column labels are now drawn inside their container)
_PAD_LEFT = 40    # room for z-row labels

_SLOT_W = _CELL_W + _GAP_X
_SLOT_H = _CELL_H + _GAP_Y

_RESIZE_W = 12    # px from right edge that triggers resize cursor

# ── Colors — copied from side-view _PANEL_CFG ──────────────────────────────────
_FILL_LAYER    = QColor('#D4E4F4')
_BORDER_LAYER  = QColor('#90B4D0')
_TEXT_LAYER    = QColor('#2A4460')

_FILL_SENSOR   = QColor('#D8EED8')
_BORDER_SENSOR = QColor('#90B890')
_TEXT_SENSOR   = QColor('#1E4A1E')

_BORDER_EMPTY  = QColor('#C8D4DC')
_SCENE_BG      = QColor('#F0F2F5')

# Ghost / drop-indicator colors
_GHOST_PEN   = QColor('#7090B0')
_GHOST_FILL  = QColor(100, 140, 180, 30)

# Dimmed colors for containers that are subsumed by a higher-z element
_FILL_LAYER_DIM    = QColor('#EEF2F8')
_BORDER_LAYER_DIM  = QColor('#B8C8D8')
_FILL_SENSOR_DIM   = QColor('#EEF6EE')
_BORDER_SENSOR_DIM = QColor('#B0C8B0')
_TEXT_DIM          = QColor('#9AABBB')


def _container_width(span: int) -> int:
    """Pixel width of a container spanning `span` column slots."""
    return span * _CELL_W + (span - 1) * _GAP_X


# ── Container cell ─────────────────────────────────────────────────────────────

class _ContainerNode(QGraphicsRectItem):
    """One container in the (container_idx, z) grid.

    Occupied cells
      • Drag body → shows dashed ghost at target cell; on release moves container.
      • Drag right edge (resize zone) → spans discretely across columns.

    Empty cells are transparent drop targets shown with a faint dashed border.
    """

    def __init__(self, names, hidden_names, container, z_val,
                 container_idx, z_idx, is_empty, container_type, span, top_win,
                 is_subsumed=False):
        w = _container_width(span) if not is_empty else _CELL_W
        super().__init__(0, 0, w, _CELL_H)

        self._names        = sorted(names)
        self._hidden_names = sorted(hidden_names)   # names subsumed within this span
        self._container    = container
        self._z_val        = z_val
        self._container_idx      = container_idx
        self._z_idx        = z_idx
        self._is_empty     = is_empty
        self._container_type     = container_type   # 'sensor' | 'layer' | 'mixed'
        self._span         = span
        self._top_win      = top_win
        self._is_subsumed  = is_subsumed

        self._resizing          = False
        self._resize_from_left  = False
        self._resize_start_x    = 0.0
        self._resize_orig_span  = span
        self._resize_orig_x     = 0.0
        self._preview_span      = span

        self._ghost             = None   # dashed rect or line shown during drag
        self._insert_after_container  = None   # set when hovering in a gap zone

        self.setBrush(QBrush(Qt.NoBrush))
        self.setPen(QPen(Qt.NoPen))

        if not is_empty:
            self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
            self.setAcceptHoverEvents(True)
            self.setCursor(Qt.SizeAllCursor)

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _colors(self):
        if self._is_subsumed:
            if self._container_type == 'sensor':
                return _FILL_SENSOR_DIM, _BORDER_SENSOR_DIM, _TEXT_DIM
            return _FILL_LAYER_DIM, _BORDER_LAYER_DIM, _TEXT_DIM
        if self._container_type == 'sensor':
            return _FILL_SENSOR, _BORDER_SENSOR, _TEXT_SENSOR
        return _FILL_LAYER, _BORDER_LAYER, _TEXT_LAYER

    def paint(self, painter, _option, _widget=None):
        span = self._preview_span

        if self._is_empty:
            pen = QPen(_BORDER_EMPTY, 1.0, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(1, 1, _CELL_W - 2, _CELL_H - 2)
            return

        fill, border, text_col = self._colors()
        w = _container_width(span)

        # Solid background across full span
        painter.fillRect(0, 0, w, _CELL_H, fill)
        painter.setPen(QPen(border, 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(0, 0, w - 1, _CELL_H - 1)

        # Resize-handle hints: double line on both the left and right edges —
        # containers can grow/shrink from either side.
        painter.setPen(QPen(border.darker(130), 1))
        rx = w - 1
        painter.drawLine(rx - 3, 6, rx - 3, _CELL_H - 6)
        painter.drawLine(rx - 1, 6, rx - 1, _CELL_H - 6)
        painter.drawLine(1, 6, 1, _CELL_H - 6)
        painter.drawLine(3, 6, 3, _CELL_H - 6)

        # Text: primary names + hidden names (in italics / lighter)
        f = QFont(); f.setPointSize(8)
        painter.setFont(f)
        fm     = painter.fontMetrics()
        line_h = fm.height() + 2
        all_labels = self._names + ([f'({n})' for n in self._hidden_names])
        total_h = len(all_labels) * line_h
        y0 = max(4, (_CELL_H - total_h) // 2)

        for i, label in enumerate(all_labels):
            y = y0 + i * line_h
            if y + line_h > _CELL_H - 2:
                break
            # Hidden names shown in a lighter colour to distinguish them
            col = text_col if i < len(self._names) else text_col.lighter(160)
            painter.setPen(col)
            painter.drawText(6, y, _CELL_W - 12, line_h,
                             Qt.AlignLeft | Qt.AlignVCenter, label)

    # ── Hover ─────────────────────────────────────────────────────────────────

    def _in_right_resize_zone(self, local_x: float) -> bool:
        right_x = (self._preview_span - 1) * _SLOT_W + _CELL_W
        return local_x >= right_x - _RESIZE_W

    def _in_left_resize_zone(self, local_x: float) -> bool:
        return local_x <= _RESIZE_W

    def hoverMoveEvent(self, event):
        if self._is_empty:
            return
        x = event.pos().x()
        resizing = self._in_right_resize_zone(x) or self._in_left_resize_zone(x)
        self.setCursor(Qt.SizeHorCursor if resizing else Qt.SizeAllCursor)

    def hoverLeaveEvent(self, _event):
        if not self._is_empty:
            self.setCursor(Qt.SizeAllCursor)

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self._is_empty:
            return
        x = event.pos().x()
        if event.button() == Qt.LeftButton and (self._in_right_resize_zone(x)
                                                 or self._in_left_resize_zone(x)):
            self._resizing         = True
            self._resize_from_left = self._in_left_resize_zone(x) and not self._in_right_resize_zone(x)
            self._resize_start_x   = event.scenePos().x()
            self._resize_orig_span = self._span
            self._resize_orig_x    = self.x()
            self._preview_span     = self._span
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_empty:
            return
        if self._resizing:
            delta = event.scenePos().x() - self._resize_start_x
            if self._resize_from_left:
                delta = -delta   # dragging left grows the container
            raw_w    = _container_width(self._resize_orig_span) + delta
            new_span = max(1, round(raw_w / _SLOT_W))
            if self._resize_from_left:
                # Can't push the start column past 0.
                new_span = min(new_span, self._container + self._resize_orig_span)
            if new_span != self._preview_span:
                self._preview_span = new_span
                new_w = _container_width(new_span)
                self.setRect(0, 0, new_w, _CELL_H)
                if self._resize_from_left:
                    # Keep the right edge fixed: shift the item left as it grows.
                    orig_w = _container_width(self._resize_orig_span)
                    self.setPos(self._resize_orig_x + orig_w - new_w, self.y())
                self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)
            self.setPos(max(0, self.x()), max(0, self.y()))
            self._update_ghost()

    def mouseReleaseEvent(self, event):
        if self._is_empty:
            return
        ins_container = self._insert_after_container
        self._remove_ghost()
        if self._resizing:
            self._resizing = False
            new_span = self._preview_span
            container, z = self._container, self._z_val
            if new_span != self._span:
                new_container = container
                if self._resize_from_left:
                    delta_span = new_span - self._resize_orig_span
                    new_container = max(0, container - delta_span)
                QTimer.singleShot(0, lambda: self._top_win._set_container_span(
                    container, z, new_span, new_container=new_container))
            else:
                QTimer.singleShot(0, self._top_win.refresh)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
            tw = self._top_win
            old_container, old_z  = self._container, self._z_val
            old_ci, old_zi = self._container_idx,   self._z_idx
            new_z_idx = max(0, min(round((self.y() - _PAD_TOP) / _SLOT_H),
                                   len(tw._z_levels)))
            if ins_container is not None:
                # Gap-zone drop → insert a new column
                QTimer.singleShot(0, lambda: tw._on_insert_container(
                    old_container, old_z, ins_container, new_z_idx))
            else:
                new_container_idx = max(0, min(round((self.x() - _PAD_LEFT) / _SLOT_W),
                                         len(tw._ordered_containers)))
                QTimer.singleShot(0, lambda: tw._on_container_dropped(
                    old_container, old_z, old_ci, old_zi, new_container_idx, new_z_idx))

    # ── Ghost indicator ────────────────────────────────────────────────────────

    def _update_ghost(self):
        """Draw a dashed rect (snap to column) or a vertical line (insert gap)."""
        scene = self.scene()
        if scene is None:
            return
        tw = self._top_win

        raw_container = (self.x() - _PAD_LEFT) / _SLOT_W
        in_gap  = raw_container >= 0 and (raw_container - int(raw_container)) > 0.6

        if in_gap:
            # ── Insertion ghost: vertical dashed line between two columns ──────
            container_int = max(0, min(int(raw_container), len(tw._ordered_containers) - 1))
            self._insert_after_container = container_int
            ins_x   = float(_PAD_LEFT + (container_int + 1) * _SLOT_W - _GAP_X // 2)
            n_rows  = len(tw._z_levels) + 1
            y_top   = float(_PAD_TOP)
            y_bot   = float(_PAD_TOP + n_rows * _SLOT_H)

            # If ghost is the wrong type, discard and recreate.
            if self._ghost is not None and not isinstance(self._ghost, QGraphicsLineItem):
                scene.removeItem(self._ghost)
                self._ghost = None
            if self._ghost is None:
                self._ghost = QGraphicsLineItem(ins_x, y_top, ins_x, y_bot)
                self._ghost.setPen(QPen(_GHOST_PEN, 2.0, Qt.DashLine))
                self._ghost.setZValue(100)
                scene.addItem(self._ghost)
            else:
                self._ghost.setLine(ins_x, y_top, ins_x, y_bot)
        else:
            # ── Column-snap ghost: dashed rect ────────────────────────────────
            self._insert_after_container = None
            new_container_idx = max(0, min(round(raw_container), len(tw._ordered_containers)))
            new_z_idx   = max(0, min(round((self.y() - _PAD_TOP) / _SLOT_H),
                                     len(tw._z_levels)))
            gx = _PAD_LEFT + new_container_idx * _SLOT_W
            gy = _PAD_TOP  + new_z_idx   * _SLOT_H
            gw = _container_width(self._span)

            if self._ghost is not None and not isinstance(self._ghost, QGraphicsRectItem):
                scene.removeItem(self._ghost)
                self._ghost = None
            if self._ghost is None:
                self._ghost = QGraphicsRectItem(gx, gy, gw, _CELL_H)
                self._ghost.setPen(QPen(_GHOST_PEN, 1.5, Qt.DashLine))
                self._ghost.setBrush(QBrush(_GHOST_FILL))
                self._ghost.setZValue(100)
                scene.addItem(self._ghost)
            else:
                self._ghost.setRect(gx, gy, gw, _CELL_H)

    def _remove_ghost(self):
        if self._ghost is not None:
            scene = self._ghost.scene()
            if scene is not None:
                scene.removeItem(self._ghost)
            self._ghost = None


# ── Side-view window ───────────────────────────────────────────────────────────

class SideViewWindow(QWidget):
    """Side-view network layout: containers on X (processing position), subsumption level on Y.

    Drag container body → moves with dashed-ghost indicator; release to place.
    Drag right edge → spans across columns (Word-table-selector snap).
    Reducing span (drag right edge left) makes previously-hidden containers visible again.
    """

    def __init__(self, nvw):
        super().__init__(None, Qt.Window)
        self._nvw          = nvw
        self._ordered_containers = []
        self._z_levels     = [0]

        self.setWindowTitle("Side View  —  col × depth")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(860, 280)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)

        info = QLabel(
            "Drag body to move  ·  drag right edge to span  ·  "
            "drag right edge left to shrink span")
        info.setStyleSheet("color: #8A9AAA; font-size: 9px;")
        lay.addWidget(info)

        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QBrush(_SCENE_BG))

        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.Antialiasing)
        self._view.setDragMode(QGraphicsView.NoDrag)
        self._view.setStyleSheet(f"background:{_SCENE_BG.name()}; border:none;")
        lay.addWidget(self._view)

    # ── Scene rebuild ──────────────────────────────────────────────────────────

    def refresh(self):
        self._scene.clear()
        nvw = self._nvw

        try:
            c = nvw.gui.circuit
        except RuntimeError:
            return

        depth = nvw._compute_container()

        for l in c.layers:
            pair_name = getattr(l, 'lateral_pair', None)
            if pair_name:
                partner = next((pl for pl in c.layers if pl.name == pair_name), None)
                if partner:
                    d = max(depth.get(l.name, 1), depth.get(partner.name, 1), 1)
                    depth[l.name] = d
                    depth[partner.name] = d

        container_labels = getattr(nvw, '_container_labels', {})

        # Build containers: (container, z) → {display names}
        containers           = defaultdict(set)
        container_has_layer  = {}
        container_has_sensor = {}

        for s in c.sensors:
            container = depth.get(s.name, 0)
            z  = getattr(s, 'z', 0) or 0
            if s.is_lateralized(c):
                containers[(container, z)].add(s.name + '_L')
                containers[(container, z)].add(s.name + '_R')
            else:
                containers[(container, z)].add(s.name)
            container_has_sensor[(container, z)] = True

        for l in c.layers:
            if l.n is None:
                continue
            container = depth.get(l.name, 1)
            z  = getattr(l, 'z', 0) or 0
            containers[(container, z)].add(l.name)
            container_has_layer[(container, z)] = True

        def _container_type(key):
            has_s = container_has_sensor.get(key, False)
            has_l = container_has_layer.get(key, False)
            if has_s and not has_l:
                return 'sensor'
            return 'layer'

        # Span per container
        container_spans = {}
        for s in c.sensors:
            container  = depth.get(s.name, 0)
            z   = getattr(s, 'z', 0) or 0
            key = (container, z)
            container_spans[key] = max(container_spans.get(key, 1),
                                       getattr(s, 'span', 1))
        for l in c.layers:
            if l.n is None:
                continue
            container  = depth.get(l.name, 1)
            z   = getattr(l, 'z', 0) or 0
            key = (container, z)
            container_spans[key] = max(container_spans.get(key, 1),
                                       getattr(l, 'span', 1))

        _occupied_containers = sorted({container for container, _ in containers})
        # Fill every integer in the range so the user sees empty slots between
        # non-adjacent occupied columns and can drag containers into them.
        if _occupied_containers:
            all_containers = list(range(0, max(_occupied_containers) + 2))
        else:
            all_containers = [0, 1]
        z_levels = sorted({z  for _, z  in containers}) or [0]
        self._ordered_containers = all_containers
        self._z_levels     = z_levels

        n_rows = len(z_levels) + 1   # +1 for the drop-zone row

        # ── Compute subsumed cells ─────────────────────────────────────────────
        # A cell (container_idx, z_val) is subsumed when a higher-z container exists
        # in that column (directly or via span).  Subsumed containers are shown
        # dimmed to indicate they are overridden by the higher-z element.
        max_z_per_container = {}   # container_idx → highest z_val that "claims" that column
        for z_val in z_levels:
            ci = 0
            while ci < len(all_containers):
                container  = all_containers[ci]
                key = (container, z_val)
                if containers.get(key):
                    sp = max(1, min(container_spans.get(key, 1),
                                    len(all_containers) - ci))
                    for j in range(sp):
                        if z_val > max_z_per_container.get(ci + j, -1):
                            max_z_per_container[ci + j] = z_val
                    ci += sp
                else:
                    ci += 1

        subsumed_cells = set()   # (container_idx, z_val) pairs
        for container_i, mz in max_z_per_container.items():
            for z_val in z_levels:
                if z_val < mz:
                    subsumed_cells.add((container_i, z_val))

        # ── Compute span layout for the drop-zone row ─────────────────────────
        # dzrow_spans[container_idx] = max span of any container that starts at container_idx
        # across all data z-rows.  Used so the drop-zone row mirrors the data-
        # row column layout: a wide container in z=0 becomes a wide drop target.
        # secondary_containers = col_idxes that are consumed (interior of a span) in
        # at least one data row — skipped entirely in the drop-zone row.
        dzrow_spans    = {}   # container_idx → span
        secondary_containers = set()
        for z_val in z_levels:
            ci = 0
            while ci < len(all_containers):
                container  = all_containers[ci]
                key = (container, z_val)
                if containers.get(key):
                    sp = max(1, min(container_spans.get(key, 1),
                                    len(all_containers) - ci))
                    if sp > dzrow_spans.get(ci, 1):
                        dzrow_spans[ci] = sp
                    for j in range(1, sp):
                        secondary_containers.add(ci + j)
                    ci += sp
                else:
                    ci += 1

        # container → z_idx of the topmost occupied row in that column, so column
        # labels can be drawn inside that container instead of in a separate
        # header strip above the grid.
        container_top_z_idx = {}

        # ── Render data rows ───────────────────────────────────────────────────
        for z_idx in range(n_rows):
            if z_idx < len(z_levels):
                z_val    = z_levels[z_idx]
                is_dzrow = False
            else:
                z_val    = (max(z_levels) + 1) if z_levels else 1
                is_dzrow = True

            container_idx = 0
            while container_idx <= len(all_containers):
                if container_idx < len(all_containers):
                    container    = all_containers[container_idx]
                    key   = (container, z_val)
                    names = containers.get(key, set())

                    # In the drop-zone row skip secondary (spanned-over) cols
                    if is_dzrow and container_idx in secondary_containers:
                        container_idx += 1
                        continue

                    if names:
                        if container not in container_top_z_idx:
                            container_top_z_idx[container] = z_idx
                        span  = max(1, min(container_spans.get(key, 1),
                                           len(all_containers) - container_idx))
                        ctype = _container_type(key)
                        # Collect names from spanned-over containers (hidden)
                        hidden = set()
                        for j in range(1, span):
                            if container_idx + j < len(all_containers):
                                nbr_key = (all_containers[container_idx + j], z_val)
                                hidden.update(containers.get(nbr_key, set()))
                    else:
                        # Drop-zone row: mirror the span of the corresponding data row
                        span   = dzrow_spans.get(container_idx, 1) if is_dzrow else 1
                        ctype  = 'layer'
                        hidden = set()
                else:
                    container = -1; names = set(); span = 1; ctype = 'layer'; hidden = set()

                is_empty = not names
                is_sub   = (not is_empty
                             and (container_idx, z_val) in subsumed_cells)
                node = _ContainerNode(
                    names, hidden, container, z_val, container_idx, z_idx,
                    is_empty, ctype, span, self, is_subsumed=is_sub)
                node.setPos(_PAD_LEFT + container_idx * _SLOT_W,
                            _PAD_TOP  + z_idx   * _SLOT_H)
                self._scene.addItem(node)

                # In the drop-zone row advance by span even for empty targets,
                # so wide drop-targets mirror the data-row column layout.
                container_idx += span if (not is_empty or is_dzrow) else 1

        # ── Axis labels ────────────────────────────────────────────────────────
        f_lbl = QFont(); f_lbl.setPointSize(8); f_lbl.setBold(True)

        for z_idx, z_val in enumerate(z_levels):
            lbl = QGraphicsSimpleTextItem(f"z={z_val}")
            lbl.setFont(f_lbl)
            lbl.setBrush(QBrush(QColor('#6A8AAA')))
            cy = _PAD_TOP + z_idx * _SLOT_H + _CELL_H / 2
            lbl.setPos(2, cy - lbl.boundingRect().height() / 2)
            self._scene.addItem(lbl)

        for container_idx, container in enumerate(all_containers):
            z_idx = container_top_z_idx.get(container)
            # Key by the topmost container's own occupant name(s) — identity,
            # not position — so the label stays attached to whoever it was
            # actually set for even as containers get dragged/reordered.
            top_names = (containers.get((container, z_levels[z_idx]))
                         if z_idx is not None and z_idx < len(z_levels) else None)
            label = container_labels.get('|'.join(sorted(top_names))) if top_names else None
            if not label:
                continue
            lbl = QGraphicsSimpleTextItem(label)
            lbl.setFont(f_lbl)
            lbl.setBrush(QBrush(QColor('#3A5A7A')))
            x = _PAD_LEFT + container_idx * _SLOT_W + 4
            y = (_PAD_TOP + z_idx * _SLOT_H + 2) if z_idx is not None else 4
            lbl.setPos(x, y)
            lbl.setZValue(5)
            self._scene.addItem(lbl)

        n_containers_drawn = len(all_containers) + 1
        w = _PAD_LEFT + n_containers_drawn * _SLOT_W + 8
        h = _PAD_TOP  + n_rows       * _SLOT_H + 8
        self._scene.setSceneRect(0, 0, w, h)

    # ── Span setter ────────────────────────────────────────────────────────────

    def _set_container_span(self, container, z_val, new_span, new_container=None):
        """Resize a container. new_container, when given, also moves its start column
        (used when growing/shrinking from the left edge — the span still covers
        [new_container, new_container + new_span - 1], just anchored differently)."""
        nvw   = self._nvw
        c     = nvw.gui.circuit
        depth = nvw._compute_container()
        target_container = container if new_container is None else max(0, new_container)
        for s in c.sensors:
            if (depth.get(s.name) == container
                    and (getattr(s, 'z', 0) or 0) == z_val):
                s.span = new_span
                s.layer = target_container
        for l in c.layers:
            if (depth.get(l.name) == container
                    and (getattr(l, 'z', 0) or 0) == z_val):
                l.span = new_span
                l.layer = target_container
        self.refresh()

    # ── Insert handler ────────────────────────────────────────────────────────

    def _on_insert_container(self, old_container, old_z, insert_after_container_idx, new_z_idx):
        """Create a new column slot by shifting existing columns and placing the
        dragged container at the gap position."""
        nvw = self._nvw
        c   = nvw.gui.circuit

        ordered_containers = self._ordered_containers
        z_levels     = self._z_levels

        # Determine the new container for the inserted slot.
        container_idx_clamped = min(insert_after_container_idx, len(ordered_containers) - 1)
        insert_at_container    = (ordered_containers[container_idx_clamped] + 1
                           if ordered_containers else 1)

        new_z = (z_levels[new_z_idx] if new_z_idx < len(z_levels)
                 else (max(z_levels, default=0) + 1))

        depth = nvw._compute_container()
        # Align lateral-pair containers (mirrors _on_container_dropped).
        for _l in c.layers:
            _pn = getattr(_l, 'lateral_pair', None)
            if _pn:
                _pp = next((pl for pl in c.layers if pl.name == _pn), None)
                if _pp:
                    _d = max(depth.get(_l.name, 1), depth.get(_pp.name, 1), 1)
                    depth[_l.name] = _d
                    depth[_pp.name] = _d

        moved_layers  = [l for l in c.layers
                         if l.n is not None
                         and depth.get(l.name) == old_container
                         and (getattr(l, 'z', 0) or 0) == old_z]
        moved_sensors = [s for s in c.sensors
                         if depth.get(s.name, 0) == old_container
                         and (getattr(s, 'z', 0) or 0) == old_z]
        moved_ids_l   = {id(l) for l in moved_layers}
        moved_ids_s   = {id(s) for s in moved_sensors}

        # Shift all non-moved elements at container >= insert_at_container up by 1.
        for l in c.layers:
            if l.n is None or id(l) in moved_ids_l:
                continue
            container = depth.get(l.name)
            if container is not None:
                l.layer = container + 1 if container >= insert_at_container else container
        for s in c.sensors:
            if id(s) in moved_ids_s:
                continue
            container = depth.get(s.name, 0) or 0
            s.layer = container + 1 if container >= insert_at_container else container

        # _container_labels is keyed by container identity (occupant name
        # set), not position, so shifting other containers' positions here
        # needs no label bookkeeping — nobody's label is affected by this.

        # Place the moved container at the new slot.
        for l in moved_layers:
            l.layer = insert_at_container
            l.z     = new_z
        for s in moved_sensors:
            s.layer = insert_at_container
            s.z     = new_z

        self._advance_z_cut(nvw, new_z)
        self.refresh()

    # ── Drop handler ──────────────────────────────────────────────────────────

    def _on_container_dropped(self, old_container, old_z, _old_container_idx, _old_z_idx,
                               new_container_idx, new_z_idx):
        nvw = self._nvw
        c   = nvw.gui.circuit

        ordered_containers = self._ordered_containers
        z_levels     = self._z_levels

        if new_container_idx < len(ordered_containers):
            new_container = ordered_containers[new_container_idx]
        else:
            new_container = (max(ordered_containers) + 1) if ordered_containers else 1

        new_z = (z_levels[new_z_idx] if new_z_idx < len(z_levels)
                 else (max(z_levels, default=0) + 1))

        if new_container == old_container and new_z == old_z:
            self.refresh()
            return

        depth = nvw._compute_container()
        for l in c.layers:
            pair_name = getattr(l, 'lateral_pair', None)
            if pair_name:
                partner = next((pl for pl in c.layers if pl.name == pair_name), None)
                if partner:
                    d = max(depth.get(l.name, 1), depth.get(partner.name, 1), 1)
                    depth[l.name] = d
                    depth[partner.name] = d

        if new_container == old_container:
            # Same column, different z — also set layer.layer so z_cut works
            for layer in c.layers:
                if depth.get(layer.name) is not None:
                    layer.layer = depth[layer.name]
            for sensor in c.sensors:
                sensor.layer = depth.get(sensor.name, 0)
            for layer in c.layers:
                if (depth.get(layer.name) == old_container
                        and (getattr(layer, 'z', 0) or 0) == old_z):
                    layer.z = new_z
            for sensor in c.sensors:
                if (depth.get(sensor.name, 0) == old_container
                        and (getattr(sensor, 'z', 0) or 0) == old_z):
                    sensor.z = new_z
            self._advance_z_cut(nvw, new_z)
            self.refresh()
            return

        moved_layers  = [l for l in c.layers
                         if depth.get(l.name) == old_container
                         and (getattr(l, 'z', 0) or 0) == old_z]
        moved_sensors = [s for s in c.sensors
                         if depth.get(s.name, 0) == old_container
                         and (getattr(s, 'z', 0) or 0) == old_z]
        moved_ids_l   = {id(l) for l in moved_layers}
        moved_ids_s   = {id(s) for s in moved_sensors}

        # Block drop if target cell (new_container, new_z) is already occupied —
        # either directly or because it falls inside another container's span.
        # Check is done BEFORE any mutations so an aborted drop leaves state clean.
        def _target_occupied():
            for _l in c.layers:
                if _l.n is None or id(_l) in moved_ids_l:
                    continue
                _l_container = depth.get(_l.name)
                _lz  = (getattr(_l, 'z', 0) or 0)
                _lsp = max(1, getattr(_l, 'span', 1) or 1)
                if _lz != new_z or _l_container is None:
                    continue
                if _l_container == new_container or _l_container < new_container <= _l_container + _lsp - 1:
                    return True
            for _s in c.sensors:
                if id(_s) in moved_ids_s:
                    continue
                _s_container = depth.get(_s.name, 0) or 0
                _sz  = (getattr(_s, 'z', 0) or 0)
                _ssp = max(1, getattr(_s, 'span', 1) or 1)
                if _sz != new_z:
                    continue
                if _s_container == new_container or _s_container < new_container <= _s_container + _ssp - 1:
                    return True
            return False

        if _target_occupied():
            self.refresh()
            return

        # Stamp explicit layer values on all non-moved elements so container position is stable.
        for layer in c.layers:
            if id(layer) not in moved_ids_l:
                container = depth.get(layer.name)
                if container is not None:
                    layer.layer = container
        for sensor in c.sensors:
            if id(sensor) not in moved_ids_s:
                sensor.layer = depth.get(sensor.name, 0)

        for layer in moved_layers:
            layer.layer = new_container
            layer.z     = new_z
        for sensor in moved_sensors:
            sensor.layer = new_container
            sensor.z     = new_z

        # _container_labels is keyed by container identity (occupant name
        # set), not position — moving this container to (new_container, new_z)
        # doesn't change who's in it, so its label needs no bookkeeping here.

        self._advance_z_cut(nvw, new_z)
        self.refresh()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Rebuild the top view once when the side view is closed."""
        super().closeEvent(event)
        try:
            self._nvw.build()
        except RuntimeError:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _advance_z_cut(nvw, new_z):
        if new_z <= 0:
            return
        slider = nvw._z_slider
        if slider.maximum() < new_z:
            slider.setMaximum(new_z)
        if slider.value() < new_z:
            slider.blockSignals(True)
            slider.setValue(new_z)
            slider.blockSignals(False)
            nvw._z_cut = new_z
            nvw._z_slider_label.setText(f"Depth: {new_z}")
            nvw._z_slider.setVisible(True)
            nvw._z_slider_label.setVisible(True)
