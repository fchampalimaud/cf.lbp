import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QPainterPath, QPainterPathStroker, QPen, QBrush, QColor

from sim_constants import C, _CHAN_PALETTE, _TRAIL_COLOR

# Distinct fill colors for each agent's robot disk (agent 0 uses the existing primary blue)
_AGENT_COLORS = ['#FF4444', '#22BB22', '#4444FF', '#FFCC00', '#FF8800', '#9900CC']


# ============================================================
# POLYGON WALL GRAPHICS ITEM
# ============================================================
_ROOM_BAND = 0.2  # exterior band width (world units) for room objects


class PolyWallItem(pg.GraphicsObject):
    def __init__(self, points, color, external=True):
        super().__init__()
        self._points   = points
        self._color    = QColor(*[int(c * 255) for c in color])
        self._external = external

    def boundingRect(self):
        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]
        x0, y0 = min(xs), min(ys)
        w = max(xs) - x0 or 0.01
        h = max(ys) - y0 or 0.01
        pad = _ROOM_BAND if not self._external else 0.0
        return QRectF(x0 - pad, y0 - pad, w + 2 * pad, h + 2 * pad)

    def paint(self, p, option, widget):  # noqa: ARG002
        path = QPainterPath()
        path.moveTo(self._points[0][0], self._points[0][1])
        for pt in self._points[1:]:
            path.lineTo(pt[0], pt[1])
        path.closeSubpath()
        p.setRenderHint(QPainter.Antialiasing)
        if self._external:
            # Solid fill with full color
            p.setBrush(QBrush(self._color))
            p.setPen(Qt.NoPen)
            p.drawPath(path)
        else:
            # Colored band OUTSIDE the polygon boundary, interior empty
            stroker = QPainterPathStroker()
            stroker.setWidth(_ROOM_BAND * 2)  # half falls outside, half inside; subtract interior
            exterior = stroker.createStroke(path).subtracted(path)
            p.setBrush(QBrush(self._color))
            p.setPen(Qt.NoPen)
            p.drawPath(exterior)
        # Solid black cosmetic outline on the polygon boundary
        outline = QPen(QColor('#111111'), 2)
        outline.setCosmetic(True)
        p.setPen(outline)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)


# ============================================================
# ARENA VIEWBOX — captures mouse events
# ============================================================
class ArenaViewBox(pg.ViewBox):
    sigArenaClick = Signal(float, float, int)
    sigArenaDrag  = Signal(float, float, bool, bool)
    sigArenaHover = Signal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drag_active = False
        self.setAcceptHoverEvents(True)

    def hoverEvent(self, ev):
        if not ev.isExit():
            pos = self.mapSceneToView(ev.scenePos())
            self.sigArenaHover.emit(pos.x(), pos.y())

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            pos = self.mapSceneToView(ev.scenePos())
            self.sigArenaClick.emit(pos.x(), pos.y(), 2)
            ev.accept()
        elif ev.button() == Qt.LeftButton:
            pos = self.mapSceneToView(ev.scenePos())
            self.sigArenaClick.emit(pos.x(), pos.y(), 1)
            ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        pos = self.mapSceneToView(ev.scenePos())
        x, y = pos.x(), pos.y()
        if ev.button() == Qt.LeftButton:
            is_start  = ev.isStart()
            is_finish = ev.isFinish()
            self._drag_active = not is_finish
            self.sigArenaDrag.emit(x, y, is_start, is_finish)
            ev.accept()
        else:
            super().mouseDragEvent(ev, axis)


# ============================================================
# ROBOT GRAPHICS ITEM
# ============================================================
class RobotItem(pg.GraphicsObject):
    def __init__(self, color=None):
        super().__init__()
        self._x      = 0.0
        self._y      = 0.0
        self._r      = 0.2
        self._theta  = 0.0
        self._color  = QColor(color) if color else QColor(C['primary'])
        self._border = QColor(C['dark'])
        self.setZValue(10)

    def setColor(self, color):
        self._color = QColor(color)
        self.update()

    def setSelected(self, selected: bool):
        self._border = QColor('white') if selected else QColor(C['dark'])
        self.update()

    def setRobot(self, x, y, r, theta):
        if r != self._r:
            self.prepareGeometryChange()
            self._r = r
        self._theta = theta
        self.setPos(x, y)
        self.update()

    def boundingRect(self):
        r = self._r
        return QRectF(-r, -r, 2*r, 2*r)

    def paint(self, p, option, widget):  # noqa: ARG002
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self._color))
        p.setPen(QPen(self._border, self._r * 0.08))
        p.drawEllipse(QPointF(0, 0), self._r, self._r)
        ex = self._r * 0.9 * np.cos(self._theta)
        ey = self._r * 0.9 * np.sin(self._theta)
        p.setPen(QPen(self._border, self._r * 0.12))
        p.drawLine(QPointF(0, 0), QPointF(ex, ey))


# ============================================================
# CHILD BODY GRAPHICS ITEM (for hierarchical body disks)
# ============================================================
class ChildBodyItem(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self._x     = 0.0
        self._y     = 0.0
        self._r     = 0.1
        self._theta = 0.0
        self._color  = QColor('#7a9fcb')
        self._border = QColor(C['dark'])
        self.setZValue(9)

    def setBody(self, x, y, r, theta):
        if r != self._r:
            self.prepareGeometryChange()
            self._r = r
        self._theta = theta
        self.setPos(x, y)
        self.update()

    def boundingRect(self):
        r = self._r
        return QRectF(-r, -r, 2*r, 2*r)

    def paint(self, p, option, widget):  # noqa: ARG002
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self._color))
        p.setPen(QPen(self._border, self._r * 0.08))
        p.drawEllipse(QPointF(0, 0), self._r, self._r)
        ex = self._r * 0.85 * np.cos(self._theta)
        ey = self._r * 0.85 * np.sin(self._theta)
        p.setPen(QPen(self._border, self._r * 0.12))
        p.drawLine(QPointF(0, 0), QPointF(ex, ey))


# ============================================================
# CIRCLE GRAPHICS ITEM (for arena objects)
# ============================================================
class CircleItem(pg.GraphicsObject):
    def __init__(self, x, y, r, fill_hex, border='#111111', external=True, alpha=255):
        super().__init__()
        self._x        = x
        self._y        = y
        self._r        = r
        self._fill     = QColor(fill_hex)
        self._fill.setAlpha(alpha)
        self._border   = QColor(border)
        self._external = external

    def boundingRect(self):
        r = self._r + (_ROOM_BAND if not self._external else 0.0)
        return QRectF(self._x - r, self._y - r, 2 * r, 2 * r)

    def paint(self, p, option, widget):  # noqa: ARG002
        p.setRenderHint(QPainter.Antialiasing)
        if self._external:
            p.setBrush(QBrush(self._fill))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(self._x, self._y), self._r, self._r)
        else:
            # Colored annulus OUTSIDE the circle boundary, interior empty
            outer = QPainterPath()
            outer.addEllipse(QPointF(self._x, self._y), self._r + _ROOM_BAND, self._r + _ROOM_BAND)
            inner = QPainterPath()
            inner.addEllipse(QPointF(self._x, self._y), self._r, self._r)
            ring = outer.subtracted(inner)
            p.setBrush(QBrush(self._fill))
            p.setPen(Qt.NoPen)
            p.drawPath(ring)
        # Solid black cosmetic outline on the boundary circle
        outline = QPen(QColor('#111111'), 2)
        outline.setCosmetic(True)
        p.setPen(outline)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(self._x, self._y), self._r, self._r)

    def move(self, x, y):
        self.prepareGeometryChange()
        self._x = x
        self._y = y
        self.update()


# ============================================================
# ARENA WIDGET
# ============================================================
class ArenaWidget(pg.GraphicsLayoutWidget):
    sigClick = Signal(float, float, int)
    sigDrag  = Signal(float, float, bool, bool)
    sigHover = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(QColor(C['bg']))

        self._vb = ArenaViewBox(lockAspect=True)
        self._vb.sigArenaClick.connect(self.sigClick)
        self._vb.sigArenaDrag.connect(self.sigDrag)
        self._vb.sigArenaHover.connect(self.sigHover)

        self._plot = pg.PlotItem(viewBox=self._vb)
        self._plot.setAspectLocked(True)
        self._plot.hideAxis('bottom')
        self._plot.hideAxis('left')
        self.addItem(self._plot)

        self._img_item    = pg.ImageItem()     # gradient / stimulus overlay (2D)
        self._plot.addItem(self._img_item)

        self._overhead_item = pg.ImageItem()  # MuJoCo top-down render (3D view)
        self._overhead_item.setZValue(0)      # below gradient overlay (ZValue=1), above background
        self._overhead_item.setVisible(False)
        self._overhead_item.setAcceptedMouseButtons(Qt.NoButton)  # let events fall through to ViewBox
        self._plot.addItem(self._overhead_item)

        self._boundary    = pg.PlotDataItem(pen=pg.mkPen(C['dark'], width=2))
        self._plot.addItem(self._boundary)

        self._object_items = []
        self._wall_items   = []
        self._poly_preview = None

        self._robot_items        = []   # list[RobotItem]
        self._trail_items        = []   # list[pg.PlotDataItem]
        self._selected_agent_idx = 0
        self._add_robot_item_internal(0)   # creates first RobotItem + trail
        self._robot_items[0].setSelected(True)

        self._wheel_L = pg.PlotDataItem(pen=pg.mkPen(C['dark'], width=6))
        self._wheel_L.setZValue(9)
        self._wheel_R = pg.PlotDataItem(pen=pg.mkPen(C['dark'], width=6))
        self._wheel_R.setZValue(9)
        self._plot.addItem(self._wheel_L)
        self._plot.addItem(self._wheel_R)

        self._sky_item = pg.PlotDataItem(
            pen=pg.mkPen(QColor(0xFF, 0xDD, 0x88, 210), width=1.5),
        )
        self._sky_item.setZValue(2)
        self._plot.addItem(self._sky_item)

        self._sensor_items     = []
        self._ghost_item       = None
        self._child_body_items = []
        self._whisker_bend_cache = {}

        self._sim_cfg      = None
        self._world        = None
        self._lim          = 5.0

    # ── Multi-agent robot/trail helpers ──────────────────────────────────────

    @property
    def _robot(self):
        """Backward-compat: returns the selected agent's RobotItem."""
        return self._robot_items[self._selected_agent_idx]

    def _add_robot_item_internal(self, agent_idx, color=None):
        if color is None:
            color = _AGENT_COLORS[agent_idx % len(_AGENT_COLORS)]
        robot = RobotItem(color)
        self._plot.addItem(robot)
        self._robot_items.append(robot)

        trail = pg.PlotDataItem(pen=pg.mkPen(color, width=3, alpha=140))
        trail.setZValue(4)
        self._plot.addItem(trail)
        self._trail_items.append(trail)

    def add_robot_item(self, agent_idx, color=None):
        """Add a new robot disk + trail for the given agent index."""
        self._add_robot_item_internal(agent_idx, color=color)

    def update_robot_color(self, idx, color):
        """Update the color of an existing robot disk and its trail."""
        if idx < len(self._robot_items):
            self._robot_items[idx].setColor(color)
        if idx < len(self._trail_items):
            self._trail_items[idx].setPen(pg.mkPen(color, width=3, alpha=140))

    def remove_robot_item(self, agent_idx):
        """Remove the robot disk and trail for the given agent index."""
        if agent_idx < len(self._robot_items):
            self._robot_items[agent_idx].setSelected(False)
            self._plot.removeItem(self._robot_items.pop(agent_idx))
        if agent_idx < len(self._trail_items):
            self._plot.removeItem(self._trail_items.pop(agent_idx))
        self._selected_agent_idx = min(self._selected_agent_idx,
                                       max(0, len(self._robot_items) - 1))

    def select_robot(self, idx):
        """Switch which agent's disk is used for the full update_robot() call."""
        if 0 <= self._selected_agent_idx < len(self._robot_items):
            self._robot_items[self._selected_agent_idx].setSelected(False)
        self._selected_agent_idx = idx
        if 0 <= idx < len(self._robot_items):
            self._robot_items[idx].setSelected(True)

    def update_robot_pos(self, agent_idx, x, y, r, theta):
        """Lightweight position-only update for non-selected agent disks."""
        if agent_idx < len(self._robot_items):
            self._robot_items[agent_idx].setRobot(x, y, r, theta)

    # ── Setup / static rebuild ────────────────────────────────────────────────

    def setup(self, sim_cfg, world):
        self._sim_cfg = sim_cfg
        self._world   = world
        self._lim     = sim_cfg.arena_scale
        self._rebuild_static()

    def _rebuild_static(self):
        lim = self._lim = self._sim_cfg.arena_scale

        for it in self._object_items:
            self._plot.removeItem(it)
        self._object_items = []

        for it in self._wall_items:
            self._plot.removeItem(it)
        self._wall_items = []

        if self._world.arena_round:
            angles = np.linspace(0, 2*np.pi, 200)
            self._boundary.setData(lim * np.cos(angles), lim * np.sin(angles))
        else:
            xs = [-lim, lim, lim, -lim, -lim]
            ys = [-lim, -lim, lim, lim, -lim]
            self._boundary.setData(xs, ys)

        for obj in self._world.objects:
            col      = obj.get('color', [1.0, 0.0, 0.0])
            hex_c    = f"#{int(col[0]*255):02x}{int(col[1]*255):02x}{int(col[2]*255):02x}"
            external = obj.get('external', True)
            item     = CircleItem(obj['x'], obj['y'], obj['r'], hex_c, external=external)
            item.setZValue(3)
            self._plot.addItem(item)
            self._object_items.append(item)

        for wall in self._world.walls:
            col      = wall.get('color', [0.5, 0.5, 0.5])
            external = wall.get('external', True)
            item     = PolyWallItem(wall['points'], col, external)
            item.setZValue(3)
            self._plot.addItem(item)
            self._wall_items.append(item)

        self._rebuild_gradient()
        self._rebuild_sky()

        self._vb.setRange(QRectF(-lim*1.05, -lim*1.05, lim*2.1, lim*2.1), padding=0)
        self._vb.disableAutoRange()

    def _rebuild_gradient(self):
        sim_cfg = self._sim_cfg
        world   = self._world
        if not (sim_cfg.toggle_stim and world.patches):
            self._img_item.clear()
            return
        lim = sim_cfg.arena_scale
        res = 80
        xv  = np.linspace(-lim, lim, res)
        X, Y = np.meshgrid(xv, xv)
        rgba = np.zeros((res, res, 4), dtype=np.float32)
        for p in world.patches:
            col = p.get("color", [1.0, 1.0, 1.0])
            if p.get('type') == 'wall':
                width = p.get('width', 0.5)
                if world.arena_round:
                    d_wall = lim - np.sqrt(X**2 + Y**2)
                else:
                    d_wall = lim - np.maximum(np.abs(X), np.abs(Y))
                sig = np.maximum(0.0, 1.0 - np.maximum(0.0, d_wall) / max(width, 1e-9))
            elif p.get('continuous'):
                dist = np.sqrt((X - p["x"])**2 + (Y - p["y"])**2)
                sig  = np.where(dist < p["r"], 1.0, 0.0).astype(np.float32)
            else:
                dist = np.sqrt((X - p["x"])**2 + (Y - p["y"])**2)
                sig  = np.maximum(0.0, 1.0 - dist / p["r"])
            for c in range(3):
                rgba[..., c] = np.maximum(rgba[..., c], sig * col[c])
            rgba[..., 3] = np.maximum(rgba[..., 3], sig * 0.75)
        if world.arena_round:
            rgba[np.sqrt(X**2 + Y**2) > lim, 3] = 0.0
        rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        self._img_item.setImage(rgba_uint8)
        self._img_item.setRect(QRectF(-lim, -lim, 2*lim, 2*lim))
        self._img_item.setZValue(1)

    def _rebuild_sky(self):
        if self._world is None or not self._world.sky.get("enabled"):
            self._sky_item.setData([], [])
            self._sky_item.setVisible(False)
            return
        self._sky_item.setVisible(True)
        angle   = self._world.sky["angle"]
        cos_a   = np.cos(angle)
        sin_a   = np.sin(angle)
        lim     = self._lim
        half    = lim * 0.18
        wing    = half * 0.35
        for_ang = angle + np.pi
        wing_r  = for_ang + np.radians(30)
        wing_l  = for_ang - np.radians(30)
        wr_cos, wr_sin = np.cos(wing_r), np.sin(wing_r)
        wl_cos, wl_sin = np.cos(wing_l), np.sin(wing_l)
        pts    = 7
        coords = np.linspace(-lim * 1.1, lim * 1.1, pts)
        all_x  = []
        all_y  = []
        for px in coords:
            for py in coords:
                tx = px + half * cos_a
                ty = py + half * sin_a
                bx = px - half * cos_a
                by = py - half * sin_a
                all_x += [bx, tx, np.nan]
                all_y += [by, ty, np.nan]
                all_x += [tx, tx + wing * wr_cos, np.nan]
                all_y += [ty, ty + wing * wr_sin, np.nan]
                all_x += [tx, tx + wing * wl_cos, np.nan]
                all_y += [ty, ty + wing * wl_sin, np.nan]
        self._sky_item.setData(all_x, all_y)

    def setup_sensors(self, sensors, channel_colors):
        for entry in self._sensor_items:
            self._plot.removeItem(entry[3])
            if entry[0] == 'whisker':
                self._plot.removeItem(entry[4])
        self._sensor_items = []
        self._whisker_bend_cache.clear()

        for i, sensor in enumerate(sensors):
            color         = getattr(sensor, '_viz_color', None) or _CHAN_PALETTE[i % len(_CHAN_PALETTE)]
            vt            = getattr(sensor, 'viz_type', None)
            body_ids_list = getattr(sensor, 'body_ids', None) or ['root']

            if vt == 'ray':
                dashed = getattr(sensor, '_viz_dashed', True)
                lw     = getattr(sensor, '_viz_lw', 1.0)
                style  = Qt.DashLine if dashed else Qt.SolidLine
                for bid in body_ids_list:
                    for j in range(sensor.n):
                        item = pg.PlotDataItem(pen=pg.mkPen(color, width=lw, style=style))
                        item.setZValue(11)
                        self._plot.addItem(item)
                        self._sensor_items.append(('ray', sensor, j, item, bid))

            elif vt == 'arc':
                for j in range(sensor.n):
                    item = pg.PlotDataItem(pen=pg.mkPen(color, width=3.0))
                    item.setZValue(12)
                    self._plot.addItem(item)
                    self._sensor_items.append(('arc', sensor, j, item, color))

            elif vt == 'touch':
                for j in range(sensor.n):
                    item = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(color), pen=pg.mkPen(None))
                    item.setZValue(12)
                    self._plot.addItem(item)
                    self._sensor_items.append(('touch', sensor, j, item))

            elif vt == 'mouth':
                item = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(color),
                                          pen=pg.mkPen(C['bg'], width=1.5))
                item.setZValue(13)
                self._plot.addItem(item)
                self._sensor_items.append(('mouth', sensor, 0, item, color))

            elif vt == 'whisker':
                for bid in body_ids_list:
                    straight = pg.PlotDataItem(pen=pg.mkPen(color, width=1.5))
                    bent     = pg.PlotDataItem(pen=pg.mkPen(color, width=1.0, style=Qt.DashLine))
                    straight.setZValue(11)
                    bent.setZValue(11)
                    self._plot.addItem(straight)
                    self._plot.addItem(bent)
                    self._sensor_items.append(('whisker', sensor, 0, straight, bent, bid))

            elif vt == 'sky':
                item = pg.PlotDataItem(pen=pg.mkPen(color, width=2.5))
                item.setZValue(12)
                self._plot.addItem(item)
                self._sensor_items.append(('sky', sensor, 0, item))

            elif vt == 'camera_fov':
                item = pg.PlotDataItem(pen=pg.mkPen(color, width=1.2, style=Qt.DashLine))
                item.setZValue(11)
                self._plot.addItem(item)
                self._sensor_items.append(('camera_fov', sensor, 0, item))

        if not sensors:
            sL_col = channel_colors.get('sL', _CHAN_PALETTE[0])
            sR_col = channel_colors.get('sR', _CHAN_PALETTE[1])
            lineL = pg.PlotDataItem(pen=pg.mkPen(sL_col, width=1.5))
            lineR = pg.PlotDataItem(pen=pg.mkPen(sR_col, width=1.5))
            lineL.setZValue(11)
            lineR.setZValue(11)
            self._plot.addItem(lineL)
            self._plot.addItem(lineR)
            self._sensor_items.append(('legacy_L', None, 0, lineL))
            self._sensor_items.append(('legacy_R', None, 1, lineR))

    def update_child_bodies(self, poses, bodies, sim_cfg):
        """Update child body disk items. Maintains a pool to avoid per-frame add/remove."""
        n_needed = max(0, len(bodies) - 1)

        while len(self._child_body_items) > n_needed:
            self._plot.removeItem(self._child_body_items.pop())
        while len(self._child_body_items) < n_needed:
            item = ChildBodyItem()
            self._plot.addItem(item)
            self._child_body_items.append(item)

        for item, body in zip(self._child_body_items, bodies[1:]):
            bx, by, bth = poses.get(body.id, (0.0, 0.0, 0.0))
            item.setBody(bx, by, body.radius, bth)

    def update_objects(self, world):
        objs = world.objects
        if len(objs) != len(self._object_items):
            self._rebuild_static()
            return
        for item, obj in zip(self._object_items, objs):
            item.move(obj['x'], obj['y'])

    def update_walls(self, world):
        if len(world.walls) != len(self._wall_items):
            self._rebuild_static()

    def start_poly_preview(self):
        self.clear_poly_preview()
        self._poly_preview = pg.PlotDataItem(
            pen=pg.mkPen('#888888', width=2, style=Qt.DashLine)
        )
        self._poly_preview.setZValue(20)
        self._plot.addItem(self._poly_preview)

    def update_poly_preview(self, vertices, cursor_xy=None):
        if self._poly_preview is None or not vertices:
            return
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        if cursor_xy:
            xs += [np.nan, vertices[-1][0], cursor_xy[0]]
            ys += [np.nan, vertices[-1][1], cursor_xy[1]]
        self._poly_preview.setData(xs, ys)

    def clear_poly_preview(self):
        if self._poly_preview is not None:
            self._plot.removeItem(self._poly_preview)
            self._poly_preview = None

    def update_robot(self, x, y, r, theta, sim_cfg, poses=None, circuit=None):
        self._robot.setRobot(x, y, r, theta)

        c, s   = np.cos(theta), np.sin(theta)
        rot    = np.array([[c, -s], [s, c]])
        fwd    = np.array([c, s])
        pos    = np.array([x, y])

        for wheel_item, sign in [(self._wheel_L, 1), (self._wheel_R, -1)]:
            center = pos + rot @ np.array([0, sign * r])
            p0 = center - fwd * 0.6 * r
            p1 = center + fwd * 0.6 * r
            wheel_item.setData([p0[0], p1[0]], [p0[1], p1[1]])

        for entry in self._sensor_items:
            kind = entry[0]
            if kind == 'ray':
                _, sensor, j, item, bid = entry
                r_sense = sim_cfg.sense_radius
                angles  = sensor._ray_angles()
                if poses and bid != 'root' and bid in poses:
                    ox, oy, oth = poses[bid]
                else:
                    ox, oy, oth = x, y, theta
                if j < len(angles):
                    a  = oth + angles[j]
                    ex = ox + r_sense * np.cos(a)
                    ey = oy + r_sense * np.sin(a)
                    item.setData([ox, ex], [oy, ey])
            elif kind == 'arc':
                _, sensor, j, item, base_color = entry
                r_arc   = sensor._get_radius(sim_cfg) * 1.05
                half    = sensor.arc_angle / 2
                centers = sensor._sensor_centers()
                if j < len(centers):
                    abs_center = theta + centers[j]
                    n_pts      = max(12, int(np.degrees(sensor.arc_angle) / 3))
                    ang_arr    = np.linspace(abs_center - half, abs_center + half, n_pts)
                    ax_vals    = x + r_arc * np.cos(ang_arr)
                    ay_vals    = y + r_arc * np.sin(ang_arr)
                    item.setData(ax_vals, ay_vals)
                    val = sensor._values[j] if j < len(sensor._values) else 0.0
                    color = sensor._active_color if val > 0.5 else base_color
                    item.setPen(pg.mkPen(color, width=3))
            elif kind == 'touch':
                _, sensor, j, item = entry
                r_dot = sim_cfg.body_radius * 1.15
                angles_arr = sensor._angles
                if j < len(angles_arr):
                    a = angles_arr[j]
                    item.setData([x + r_dot * np.cos(a)], [y + r_dot * np.sin(a)])
            elif kind == 'mouth':
                _, sensor, _, item, base_color = entry
                mx = x + r * np.cos(theta)
                my = y + r * np.sin(theta)
                satiation = sensor._state / max(sensor.max_val, 1e-9)
                size = 4 + 5 * satiation
                item.setData([mx], [my], size=size)
            elif kind == 'whisker':
                _, sensor, _, straight, bent, bid = entry
                if poses and bid != 'root' and bid in poses:
                    bx, by, bth = poses[bid]
                else:
                    bx, by, bth = x, y, theta
                body_idx = (sensor.body_ids.index(bid)
                            if hasattr(sensor, 'body_ids') and bid in sensor.body_ids else 0)
                if body_idx > 0 and hasattr(sensor, 'mount_angle'):
                    sensor.mount_angle = -sensor.mount_angle
                wx, wy, wth = sensor._mount_pose(bx, by, bth)
                if body_idx > 0 and hasattr(sensor, 'mount_angle'):
                    sensor.mount_angle = -sensor.mount_angle
                cdir = np.cos(wth); sdir = np.sin(wth)
                per_body = getattr(sensor, '_contact_dist_per_body', None)
                d = per_body.get(bid) if per_body else sensor._contact_dist
                cache_key = (sensor.name, bid)
                if d is None:
                    self._whisker_bend_cache.pop(cache_key, None)
                    straight.setData([wx, wx + sensor.length * cdir],
                                     [wy, wy + sensor.length * sdir])
                    bent.setData([], [])
                else:
                    if cache_key not in self._whisker_bend_cache:
                        rot_x, rot_y = -sdir, cdir
                        if circuit:
                            jnt = next((j for j in circuit.joints if j.child_id == bid), None)
                            if jnt is not None:
                                vel_ref = getattr(jnt, '_vel_pre_clamp', jnt.vel)
                                if vel_ref < 0:
                                    rot_x, rot_y = sdir, -cdir
                        self._whisker_bend_cache[cache_key] = (rot_x, rot_y)
                    rot_x, rot_y = self._whisker_bend_cache[cache_key]

                    p1x = wx + d * cdir
                    p1y = wy + d * sdir
                    rem = sensor.length - d

                    bow  = d * 0.35
                    midx = (wx + p1x) * 0.5 + bow * rot_x
                    midy = (wy + p1y) * 0.5 + bow * rot_y
                    ts   = np.linspace(0, 1, 15)
                    mt   = 1 - ts
                    arc_xs = mt**2*wx + 2*mt*ts*midx + ts**2*p1x
                    arc_ys = mt**2*wy + 2*mt*ts*midy + ts**2*p1y
                    straight.setData(arc_xs.tolist(), arc_ys.tolist())

                    tip_dx = p1x - midx
                    tip_dy = p1y - midy
                    tip_len = np.hypot(tip_dx, tip_dy)
                    if tip_len > 1e-9:
                        tip_dx /= tip_len
                        tip_dy /= tip_len
                    else:
                        tip_dx, tip_dy = cdir, sdir
                    bent.setData([p1x, p1x + rem * tip_dx],
                                 [p1y, p1y + rem * tip_dy])
            elif kind == 'legacy_L':
                _, _, _, item = entry
                r_sense = sim_cfg.sense_radius
                sl_t    = theta + sim_cfg.sensor_angle
                item.setData([x, x + r_sense * np.cos(sl_t)],
                             [y, y + r_sense * np.sin(sl_t)])
            elif kind == 'legacy_R':
                _, _, _, item = entry
                r_sense = sim_cfg.sense_radius
                sr_t    = theta - sim_cfg.sensor_angle
                item.setData([x, x + r_sense * np.cos(sr_t)],
                             [y, y + r_sense * np.sin(sr_t)])
            elif kind == 'sky':
                _, sensor, _, item = entry
                out = getattr(sensor, '_last_output', None)
                if (self._world and self._world.sky.get('enabled')
                        and out is not None and len(out) > 0):
                    sun_dir  = self._world.get_sky_sun_dir()
                    k_peak   = int(np.argmax(out))
                    peak_dir = sun_dir + k_peak * 2 * np.pi / len(out)
                    line_len = sim_cfg.body_radius * 1.4
                    item.setData([x, x + line_len * np.cos(peak_dir)],
                                 [y, y + line_len * np.sin(peak_dir)])
                else:
                    item.setData([], [])
            elif kind == 'camera_fov':
                _, sensor, _, item = entry
                half_fov   = sensor.fov / 2
                centre_abs = theta + sensor.center_angle
                rng        = min(sensor.max_range, sim_cfg.arena_scale * 2)
                lx = x + rng * np.cos(centre_abs + half_fov)
                ly = y + rng * np.sin(centre_abs + half_fov)
                rx = x + rng * np.cos(centre_abs - half_fov)
                ry = y + rng * np.sin(centre_abs - half_fov)
                item.setData([lx, x, rx], [ly, y, ry])

    def update_trail(self, trail_xy, visible, agent_idx=0):
        if agent_idx >= len(self._trail_items):
            return
        item = self._trail_items[agent_idx]
        if visible and len(trail_xy) > 1:
            xs, ys = zip(*trail_xy)
            item.setData(list(xs), list(ys))
            item.setVisible(True)
        else:
            item.setVisible(False)

    def show_ghost(self, x, y, radius, color_hex, is_wall=False):
        self.clear_ghost()
        if is_wall:
            lim    = self._lim
            angles = np.linspace(0, 2 * np.pi, 200)
            r      = max(0.05, lim - radius)
            self._ghost_item = pg.PlotDataItem(
                r * np.cos(angles), r * np.sin(angles),
                pen=pg.mkPen(color_hex, width=2, style=Qt.DashLine),
            )
        else:
            angles = np.linspace(0, 2 * np.pi, 64)
            self._ghost_item = pg.PlotDataItem(
                x + radius * np.cos(angles),
                y + radius * np.sin(angles),
                pen=pg.mkPen(color_hex, width=1.5, style=Qt.DashLine),
                brush=pg.mkBrush(QColor(color_hex).lighter(170)),
                fillLevel=0,
            )
        self._ghost_item.setZValue(20)
        self._plot.addItem(self._ghost_item)

    def update_ghost(self, x, y, radius, is_wall=False):
        if self._ghost_item is None:
            return
        if is_wall:
            lim    = self._lim
            angles = np.linspace(0, 2 * np.pi, 200)
            r      = max(0.05, lim - radius)
            self._ghost_item.setData(r * np.cos(angles), r * np.sin(angles))
        else:
            angles = np.linspace(0, 2 * np.pi, 64)
            self._ghost_item.setData(
                x + radius * np.cos(angles),
                y + radius * np.sin(angles),
            )

    def clear_ghost(self):
        if self._ghost_item is not None:
            self._plot.removeItem(self._ghost_item)
            self._ghost_item = None

    # ── 3D overhead view ──────────────────────────────────────────────────────

    def set_3d_mode(self, enabled: bool):
        """Switch between 2D analytical view and 3D MuJoCo overhead render.

        In 3D mode objects/walls/boundary are hidden (MuJoCo renders them).
        The gradient image stays visible as a semi-transparent overlay so
        gradient fields can be painted and seen directly on the 3D render.
        Robot marker, sensor rays, and trail remain visible as overlays.
        """
        self._img_item.setOpacity(0.45 if enabled else 1.0)
        self._boundary.setVisible(not enabled)
        for item in self._object_items:
            item.setVisible(not enabled)
        for item in self._wall_items:
            item.setVisible(not enabled)
        self._overhead_item.setVisible(enabled)
        if not enabled:
            self._overhead_item.clear()

    def set_overhead_frame(self, rgb: np.ndarray, arena_scale: float):
        """Update the overhead image from a MuJoCo render.

        rgb: (H, W, 3) uint8, row 0 = top of rendered image = world +Y (north).
        Flipped vertically so row 0 maps to world -Y (south) as pyqtgraph expects
        (row 0 at the bottom of the rect in a y-up axes system).
        """
        self._overhead_item.setImage(rgb[::-1])
        s = arena_scale
        self._overhead_item.setRect(QRectF(-s, -s, 2 * s, 2 * s))
