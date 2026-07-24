"""
world_editor.py — stateful handler for arena draw modes.

Owns draw-mode state (current mode, active palette, in-progress polygon, drag)
and implements arena click/drag/hover handlers that mutate the World and call
back to request a display refresh. No dependency on Qt main window or the
simulation loop.
"""

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from sim_constants import GRADIENT_COLORS, OBJECT_COLORS


class WorldEditor:
    """
    Manages all world-editing interactions driven by arena mouse events.

    Parameters
    ----------
    world          : World        — mutated when patches/objects/walls are placed
    sim_cfg        : SimConfig    — read for arena_scale and body_radius
    arena          : ArenaWidget  — called for ghost/preview graphics
    bot_pos        : list[float]  — [x, y, theta], mutated during robot drag
    setup_world_cb : callable(rebuild=True)  — called after any world change
    """

    def __init__(self, world, sim_cfg, arena, bot_pos, setup_world_cb, get_agents=None):
        self._world       = world
        self._sim_cfg     = sim_cfg
        self._arena       = arena
        self._bot_pos     = bot_pos
        self._setup_world = setup_world_cb
        self._get_agents  = get_agents   # callable() → list[RobotAgent] | None

        self.draw_mode              = 'gradient'
        self.gradient_color         = GRADIENT_COLORS[0][2]
        self.gradient_active_letter = GRADIENT_COLORS[0][0]
        self.gradient_continuous    = False
        self.object_color           = OBJECT_COLORS[0][2]

        self._poly_vertices  = []
        self._poly_color     = [0.5, 0.5, 0.5]
        self._poly_external  = True
        self._drag_start     = None
        self._drag_radius    = 0.0
        self._drag_item      = (None, None)
        self._hover_pos      = (0.0, 0.0)
        self._pending_mount  = None   # agent index if current drag will mount a gradient

    # ── Public mode setters (called by main window button handlers) ───────────

    def set_gradient_mode(self, color, letter):
        self._cancel_poly_wall()
        self.gradient_color         = color
        self.gradient_active_letter = letter
        self.draw_mode = 'gradient'

    def set_object_mode(self, color, letter):
        self._cancel_poly_wall()
        self.object_color = color
        self.draw_mode = 'object'

    def set_wall_mode(self):
        self._cancel_poly_wall()
        self.draw_mode = 'wall'

    def set_sky_mode(self):
        self.draw_mode = 'sky'

    def set_move_mode(self):
        self.draw_mode = 'move'

    def toggle_poly_external(self):
        """Flip solid/room polygon flag. Returns the new value."""
        self._poly_external = not self._poly_external
        return self._poly_external

    # ── Arena event handlers (connected to ArenaWidget signals) ──────────────

    def handle_hover(self, x, y):
        self._hover_pos = (x, y)
        if self._poly_vertices:
            snap_x, snap_y = self._snap_angle(x, y)
            self._arena.update_poly_preview(self._poly_vertices, (snap_x, snap_y))

    def handle_click(self, x, y, button):
        if button == 1 and self.draw_mode == 'object':
            self._place_polygon_vertex(x, y)
        elif button == 2:
            self._right_click(x, y)

    def handle_drag(self, x, y, is_start, is_finish):
        if self.draw_mode == 'sky':
            self._drag_sky(x, y, is_start, is_finish)
        elif self.draw_mode == 'move':
            self._drag_move(x, y, is_start, is_finish)
        elif self.draw_mode == 'object' and self._poly_vertices:
            return  # polygon in progress — ignore drags
        else:
            self._drag_place(x, y, is_start, is_finish)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _snap_angle(self, x, y):
        if not self._poly_vertices:
            return x, y
        if not (QApplication.keyboardModifiers() & Qt.ShiftModifier):
            return x, y
        lx, ly = self._poly_vertices[-1]
        dx, dy = x - lx, y - ly
        dist = np.hypot(dx, dy)
        if dist < 1e-9:
            return x, y
        angle   = np.arctan2(dy, dx)
        snapped = round(angle / (np.pi / 4)) * (np.pi / 4)
        return lx + dist * np.cos(snapped), ly + dist * np.sin(snapped)

    def _cancel_poly_wall(self):
        self._poly_vertices = []
        self._arena.clear_poly_preview()

    def _find_drag_item(self, x, y):
        bot_x, bot_y = self._bot_pos[0], self._bot_pos[1]
        if np.hypot(x - bot_x, y - bot_y) < self._sim_cfg.body_radius * 1.5:
            return ('robot', None)
        for p in self._world.patches:
            if p.get('mounted_on') is not None:
                continue   # mounted patches move with their robot; not independently draggable
            if 'x' in p and np.hypot(x - p['x'], y - p['y']) < p.get('r', 0.1) * 1.3:
                return ('patch', p)
        for o in self._world.objects:
            if np.hypot(x - o['x'], y - o['y']) < o.get('r', 0.1) * 1.3:
                return ('object', o)
        return (None, None)

    def _place_polygon_vertex(self, x, y):
        snap_x, snap_y = self._snap_angle(x, y)
        if not self._poly_vertices:
            self._poly_color = list(self.object_color)
            self._arena.start_poly_preview()
        close_thresh = 0.3
        if (len(self._poly_vertices) >= 3
                and np.hypot(snap_x - self._poly_vertices[0][0],
                             snap_y - self._poly_vertices[0][1]) < close_thresh):
            self._world.walls.append({
                "points":   [list(v) for v in self._poly_vertices],
                "color":    self._poly_color,
                "external": self._poly_external,
            })
            self._poly_vertices = []
            self._arena.clear_poly_preview()
            self._setup_world()
        else:
            self._poly_vertices.append([snap_x, snap_y])
            self._arena.update_poly_preview(self._poly_vertices, self._hover_pos)

    def _right_click(self, x, y):
        if self._poly_vertices:
            self._cancel_poly_wall()
            return

        # Right-click on a robot: remove all gradients mounted on it
        if self._get_agents is not None:
            br = self._sim_cfg.body_radius
            for i, agent in enumerate(self._get_agents()):
                if np.hypot(x - agent.bot_pos[0], y - agent.bot_pos[1]) < br * 1.5:
                    before = len(self._world.patches)
                    self._world.patches = [p for p in self._world.patches
                                           if p.get('mounted_on') != i]
                    if len(self._world.patches) < before:
                        self._setup_world()
                    return

        lim = self._sim_cfg.arena_scale

        def _keep_patch(p):
            if p.get('type') == 'wall':
                width = p.get('width', 0.5)
                if self._world.arena_round:
                    d_wall = lim - np.hypot(x, y)
                else:
                    d_wall = min(lim - abs(x), lim - abs(y))
                return d_wall > width
            return np.hypot(p['x'] - x, p['y'] - y) > p['r']

        def _keep_wall(w):
            thresh = 0.5
            pts = w['points']
            for i in range(len(pts)):
                ax, ay = pts[i]
                bx, by = pts[(i + 1) % len(pts)]
                dx, dy = bx - ax, by - ay
                len2 = dx * dx + dy * dy
                if len2 < 1e-12:
                    if np.hypot(x - ax, y - ay) < thresh:
                        return False
                    continue
                t = np.clip(((x - ax) * dx + (y - ay) * dy) / len2, 0, 1)
                if np.hypot(x - (ax + t * dx), y - (ay + t * dy)) < thresh:
                    return False
            return True

        self._world.patches = [p for p in self._world.patches if _keep_patch(p)]
        self._world.objects = [o for o in self._world.objects
                               if np.hypot(o['x'] - x, o['y'] - y) > o['r']]
        self._world.walls   = [w for w in self._world.walls if _keep_wall(w)]
        self._setup_world()

    def _drag_sky(self, x, y, is_start, is_finish):
        if is_start:
            self._drag_start = (x, y)
        elif is_finish and self._drag_start is not None:
            dx, dy = x - self._drag_start[0], y - self._drag_start[1]
            if np.hypot(dx, dy) > 0.01:
                self._world.sky["angle"] = float(np.arctan2(dy, dx))
                self._setup_world()
            self._drag_start = None

    def _drag_move(self, x, y, is_start, is_finish):
        if is_start:
            self._drag_item = self._find_drag_item(x, y)
        if self._drag_item[0] is not None and not is_finish:
            kind, item = self._drag_item
            if kind == 'robot':
                self._bot_pos[0] = x
                self._bot_pos[1] = y
            elif kind in ('patch', 'object'):
                item['x'] = x
                item['y'] = y
            self._setup_world(rebuild=False)
        if is_finish:
            if self._drag_item[0] is not None:
                self._setup_world(rebuild=True)
            self._drag_item = (None, None)

    def _drag_place(self, x, y, is_start, is_finish):
        if is_start:
            self._pending_mount = None
            # Snap to robot center if gradient drag starts on a robot body
            cx, cy = x, y
            if self.draw_mode == 'gradient' and self._get_agents is not None:
                br = self._sim_cfg.body_radius
                for i, agent in enumerate(self._get_agents()):
                    bx, by = agent.bot_pos[0], agent.bot_pos[1]
                    if np.hypot(x - bx, y - by) < br:
                        self._pending_mount = i
                        cx, cy = bx, by   # gradient will be centered on robot
                        break
            self._drag_start  = (cx, cy)
            self._drag_radius = 0.0
            if self.draw_mode in ('gradient', 'wall'):
                pc = self.gradient_color
            else:
                pc = self.object_color
            hex_c = f"#{int(pc[0]*255):02x}{int(pc[1]*255):02x}{int(pc[2]*255):02x}"
            self._arena.show_ghost(cx, cy, 0.1, hex_c, is_wall=(self.draw_mode == 'wall'))

        if self._drag_start is not None and not is_finish:
            radius = max(0.05, np.hypot(x - self._drag_start[0], y - self._drag_start[1]))
            self._drag_radius = radius
            self._arena.update_ghost(self._drag_start[0], self._drag_start[1],
                                     radius, is_wall=(self.draw_mode == 'wall'))

        if is_finish and self._drag_start is not None:
            width = max(0.1, self._drag_radius)
            if self.draw_mode == 'wall':
                self._world.patches.append({
                    "type":  "wall",
                    "width": float(width),
                    "color": list(self.gradient_color),
                    "label": self.gradient_active_letter,
                })
            elif self.draw_mode == 'gradient':
                patch = {
                    "x":     self._drag_start[0],
                    "y":     self._drag_start[1],
                    "r":     width,
                    "color": list(self.gradient_color),
                    "label": self.gradient_active_letter,
                }
                if self.gradient_continuous:
                    patch["continuous"] = True
                if self._pending_mount is not None:
                    patch["mounted_on"] = self._pending_mount
                self._world.patches.append(patch)
                self._pending_mount = None
            else:
                self._world.objects.append({
                    "x":        self._drag_start[0],
                    "y":        self._drag_start[1],
                    "r":        width,
                    "color":    list(self.object_color),
                    "external": self._poly_external,
                })
            self._drag_start = None
            self._arena.clear_ghost()
            self._setup_world()
