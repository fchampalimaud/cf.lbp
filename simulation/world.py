import numpy as np

from sim_constants import GRADIENT_COLORS, OBJECT_COLORS  # noqa: F401 — re-exported for callers


def _seg_pt_dist(px, py, ax, ay, bx, by):
    """Minimum distance from point (px,py) to segment (ax,ay)→(bx,by)."""
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / len2, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))



class World:
    def __init__(self, config):
        self.cfg        = config
        self.patches    = []
        self.objects    = []
        self.walls      = []
        self.arena_round = False
        self.sky        = {"enabled": False, "angle": 0.0}

    def get_sky_sun_dir(self) -> float:
        """Return sun direction (radians) = e-vector bar angle + π/2."""
        return self.sky["angle"] + np.pi / 2

    def check_bumpers(self, bx, by):
        limit, r = self.cfg.arena_scale, self.cfg.body_radius
        if self.arena_round:
            wall_hit = np.hypot(bx, by) + r >= limit
            bumpers  = [wall_hit, wall_hit, wall_hit, wall_hit]
        else:
            bumpers = [bx + r >= limit, bx - r <= -limit, by + r >= limit, by - r <= -limit]
        obj_hit = False
        for o in self.objects:
            dist = np.hypot(bx - o['x'], by - o['y'])
            if o.get('external', True):
                if dist <= r + o['r'] + 0.005:
                    obj_hit = True; break
            else:
                if dist >= o['r'] - r - 0.005:
                    obj_hit = True; break
        if not obj_hit:
            for wall in self.walls:
                pts = wall['points']
                for i in range(len(pts)):
                    ax, ay = pts[i]
                    bx2, by2 = pts[(i + 1) % len(pts)]
                    if _seg_pt_dist(bx, by, ax, ay, bx2, by2) <= r + 0.005:
                        obj_hit = True
                        break
                if obj_hit:
                    break
        bumpers.append(obj_hit)
        return bumpers

    def get_signal(self, sx, sy, stheta, color_channel=None, label=None):
        if not self.cfg.toggle_stim or not self.patches:
            return 0.0
        r_sense = self.cfg.sense_radius
        ex, ey  = sx + r_sense * np.cos(stheta), sy + r_sense * np.sin(stheta)
        dx, dy  = ex - sx, ey - sy
        den     = dx*dx + dy*dy
        lim     = self.cfg.arena_scale
        max_sig = 0.0
        for patch in self.patches:
            if label is not None and patch.get('label') != label:
                continue
            if patch.get('type') == 'wall':
                width = patch.get('width', 0.5)
                if self.arena_round:
                    d_wall = lim - np.hypot(sx, sy)
                else:
                    d_wall = min(lim - abs(sx), lim - abs(sy))
                sig = max(0.0, 1.0 - max(0.0, d_wall) / max(width, 1e-9))
            elif patch.get('continuous'):
                px, py, p_rad = patch["x"], patch["y"], patch["r"]
                sig = 1.0 if np.hypot(sx - px, sy - py) < p_rad else 0.0
            else:
                px, py, p_rad = patch["x"], patch["y"], patch["r"]
                t    = np.clip(((px - sx)*dx + (py - sy)*dy) / den, 0, 1) if den > 0 else 0
                dist = np.sqrt((px - (sx + t*dx))**2 + (py - (sy + t*dy))**2)
                sig  = max(0.0, 1.0 - dist / p_rad)
            if color_channel is not None:
                sig *= patch.get("color", [1.0, 1.0, 1.0])[color_channel]
            if sig > max_sig:
                max_sig = sig
        return max_sig

    def get_object_signal(self, sx, sy, stheta, color_channel=None):
        if not self.objects:
            return 0.0
        r_sense = self.cfg.sense_radius
        rdx, rdy = np.cos(stheta), np.sin(stheta)
        min_hit  = r_sense
        hit_col  = None
        for obj in self.objects:
            ox, oy = obj['x'] - sx, obj['y'] - sy
            proj   = ox * rdx + oy * rdy
            if proj <= 0:
                continue
            perp2 = ox*ox + oy*oy - proj*proj
            r2    = obj['r'] ** 2
            if perp2 >= r2:
                continue
            hit_d = proj - np.sqrt(r2 - perp2)
            if hit_d < 0:
                hit_d = 0.0
            if hit_d <= min_hit:
                min_hit = hit_d
                hit_col = obj.get('color', [1.0, 1.0, 1.0])
        if hit_col is None:
            return 0.0
        sig = 1.0 - min_hit / r_sense
        if color_channel is not None:
            sig *= hit_col[color_channel]
        return max(0.0, sig)
