"""
Generate docs/assets/figures/sensor_overview.svg  — no external deps needed.
Also generates individual sensor SVGs: docs/assets/figures/sensor_<name>.svg
Usage (from simulation/2d/):
    python export_sensor_figures.py
"""
import math, os

# ── Layout ────────────────────────────────────────────────────────────────────
COLS, ROWS    = 5, 2
CELL_W        = 172
CELL_H        = 218
GAP           = 12
MARGIN        = 20
TOTAL_W       = MARGIN*2 + COLS*CELL_W + (COLS-1)*GAP
TOTAL_H       = MARGIN*2 + ROWS*CELL_H + (ROWS-1)*GAP

# ── Individual SVG display size ───────────────────────────────────────────────
INDIV_W = 300
INDIV_H = int(300 * CELL_H / CELL_W)   # ~381

# ── Palette ───────────────────────────────────────────────────────────────────
BG_PAGE   = '#f5f0ea'
R_FILL    = '#e2d8cc'
R_STK     = '#5c4a3a'
PRIMARY   = '#9a6840'
ACCENT    = '#c49a6c'
TXT_DRK   = '#3a2e28'
TXT_SUB   = '#7a6050'
WALL_COL  = '#9a8070'
MOUTH_COL = '#cc3333'   # mouth marker on robot (InteroceptiveSensor only)

# ── Helpers ───────────────────────────────────────────────────────────────────
def hv(deg):
    r = math.radians(deg)
    return math.cos(r), -math.sin(r)

def cell_origin(idx):
    c, r = idx % COLS, idx // COLS
    return MARGIN + c*(CELL_W+GAP), MARGIN + r*(CELL_H+GAP)

def robot_center(idx):
    ox, oy = cell_origin(idx)
    # Robot centred in the drawable area between title (oy+22) and subtitle (oy+CELL_H-14).
    # Middle of that area = oy + 22 + (CELL_H-36)//2 = oy + 22 + 91 = oy + 113.
    # Shift 2px lower so the robot body sits very slightly below visual centre
    # (sensor objects above, state bars below — both have similar room this way).
    return ox + CELL_W//2, oy + 115

RBOT = 27

def draw_robot(cx, cy, heading=90):
    dx, dy = hv(heading)
    tip = (cx + RBOT*dx, cy + RBOT*dy)
    a1 = math.radians(heading+145); a2 = math.radians(heading-145)
    w1 = (cx + RBOT*0.5*math.cos(a1), cy - RBOT*0.5*math.sin(a1))
    w2 = (cx + RBOT*0.5*math.cos(a2), cy - RBOT*0.5*math.sin(a2))
    pts = f"{tip[0]:.1f},{tip[1]:.1f} {w1[0]:.1f},{w1[1]:.1f} {w2[0]:.1f},{w2[1]:.1f}"
    s  = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{RBOT}" fill="{R_FILL}" stroke="{R_STK}" stroke-width="1.5"/>'
    s += f'<polygon points="{pts}" fill="{R_STK}" opacity="0.35"/>'
    return s

def title_label(ox, oy, name, sub):
    x = ox + CELL_W//2
    return (
        f'<text x="{x}" y="{oy+17}" text-anchor="middle" '
        f'font-family="Inter,Arial,sans-serif" font-size="11" font-weight="700" fill="{TXT_DRK}">{name}</text>'
        f'<text x="{x}" y="{oy+CELL_H-7}" text-anchor="middle" '
        f'font-family="Inter,Arial,sans-serif" font-size="9" fill="{TXT_SUB}" font-style="italic">{sub}</text>'
    )

def ray_line(cx, cy, angle, length, col=PRIMARY, width=1.5, dash=''):
    dx, dy = hv(angle)
    ex, ey = cx+dx*length, cy+dy*length
    da = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{col}" stroke-width="{width}"{da}/>'

def fan(cx, cy, heading, spread, n, length, col=PRIMARY, width=1.5):
    out = ''
    for i in range(n):
        a = heading + spread*(i-(n-1)/2)/max(1,n-1) if n>1 else heading
        out += ray_line(cx, cy, a, length, col, width)
    return out

def dot(cx, cy, r=3, col=ACCENT):
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{col}"/>'

def wall_h(x, y, w, col=WALL_COL, h=7):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="{col}" opacity="0.65"/>'

def arc_seg(cx, cy, r, a0, a1, col=PRIMARY, sw=2.0, fill='none', opacity=1.0):
    s0, s1 = math.radians(a0), math.radians(a1)
    sx, sy = cx+r*math.cos(s0), cy-r*math.sin(s0)
    ex, ey = cx+r*math.cos(s1), cy-r*math.sin(s1)
    lg = 1 if abs(a1-a0) > 180 else 0
    return (f'<path d="M{sx:.1f},{sy:.1f} A{r},{r} 0 {lg},0 {ex:.1f},{ey:.1f}" '
            f'fill="{fill}" stroke="{col}" stroke-width="{sw}" opacity="{opacity}"/>')

def wedge(cx, cy, r_in, r_out, a0, a1, col=PRIMARY, opacity=0.45):
    pts = []
    for a in [a0, a1]:
        for r in [r_in, r_out]:
            pts.append((cx + r*math.cos(math.radians(a)), cy - r*math.sin(math.radians(a))))
    p = f"{pts[0][0]:.1f},{pts[0][1]:.1f} {pts[2][0]:.1f},{pts[2][1]:.1f} {pts[3][0]:.1f},{pts[3][1]:.1f} {pts[1][0]:.1f},{pts[1][1]:.1f}"
    return f'<polygon points="{p}" fill="{col}" opacity="{opacity}"/>'

def halo(cx, cy, r, grad_id):
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="url(#{grad_id})"/>'

# ── Global SVG defs ───────────────────────────────────────────────────────────
DEFS = '''
  <radialGradient id="grd_light" cx="50%" cy="50%" r="50%">
    <stop offset="0%" stop-color="#f5c842" stop-opacity="0.75"/>
    <stop offset="100%" stop-color="#f5c842" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="grd_food" cx="50%" cy="50%" r="50%">
    <stop offset="0%" stop-color="#7cb849" stop-opacity="0.8"/>
    <stop offset="100%" stop-color="#7cb849" stop-opacity="0"/>
  </radialGradient>
'''

# ── Sensor panels — all element positions relative to (cx, cy) ────────────────
# Usable vertical span per panel: cy-RBOT-24 = ~64px above robot top to title.
#                                  cy+RBOT+50 = ~92px below robot centre to subtitle.

def panel_gradient(idx):
    ox, oy = cell_origin(idx); cx, cy = robot_center(idx)
    # Light source 52px above robot centre; halo r=36 → top 16px below title
    hx, hy = cx, cy - 52
    s  = halo(hx, hy, 36, 'grd_light')
    s += f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="4" fill="#e8b020" opacity="0.9"/>'
    s += fan(cx, cy, 90, 32, 2, 60)
    s += draw_robot(cx, cy)
    s += title_label(ox, oy, 'GradientSensor', 'gradient intensity at ray tip')
    return s

def panel_color(idx):
    ox, oy = cell_origin(idx); cx, cy = robot_center(idx)
    # Object in the fan direction, 65px away
    dx, dy = hv(78); obj_x, obj_y = cx+dx*65, cy+dy*65
    s  = f'<circle cx="{obj_x:.1f}" cy="{obj_y:.1f}" r="13" fill="#d45050" opacity="0.85" stroke="#b03030" stroke-width="1"/>'
    s += fan(cx, cy, 90, 32, 2, 65)
    s += draw_robot(cx, cy)
    s += title_label(ox, oy, 'ColorSensor', 'object colour on ray hit')
    return s

def panel_collision(idx):
    ox, oy = cell_origin(idx); cx, cy = robot_center(idx)
    # Wall 54px above robot centre (27px above robot top edge)
    wy = cy - 54
    s  = wall_h(ox+18, wy, CELL_W-36)
    for _, a0, a1 in [(105,98,112),(90,83,97),(75,68,82)]:
        s += wedge(cx, cy, RBOT, RBOT+20, a0, a1, PRIMARY, 0.55)
    s += draw_robot(cx, cy)
    s += title_label(ox, oy, 'CollisionSensor', 'contact with wall / object')
    return s

def panel_distance(idx):
    ox, oy = cell_origin(idx); cx, cy = robot_center(idx)
    # Wall and ray tips all 54px above robot centre
    wy = cy - 54
    s  = wall_h(ox+18, wy, CELL_W-36)
    for angle, length in [(75, 52), (90, 46), (105, 52)]:
        dx2, dy2 = hv(angle)
        ex, ey = cx+dx2*length, cy+dy2*length
        s += ray_line(cx, cy, angle, length)
        s += dot(ex, ey, 3.5, ACCENT)
    s += draw_robot(cx, cy)
    s += title_label(ox, oy, 'DistanceSensor', 'proximity (1 = near, 0 = far)')
    return s

def panel_interoceptive(idx):
    ox, oy = cell_origin(idx); cx, cy = robot_center(idx)
    # Food: green gradient halo above the robot
    fx, fy = cx, cy - RBOT - 22
    s  = halo(fx, fy, 34, 'grd_food')
    s += f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="4" fill="#5d9e1f" opacity="0.9"/>'
    # Internal state bar below robot
    bx, by, bw, bh = cx-24, cy+RBOT+12, 48, 10
    s += f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="4" fill="none" stroke="{WALL_COL}" stroke-width="1"/>'
    s += f'<rect x="{bx+1}" y="{by+1}" width="30" height="{bh-2}" rx="3" fill="#7cb849" opacity="0.7"/>'
    s += (f'<text x="{cx}" y="{by+bh+12}" text-anchor="middle" '
          f'font-family="Inter,Arial,sans-serif" font-size="8.5" fill="{TXT_SUB}">internal state</text>')
    s += draw_robot(cx, cy)
    # Mouth: small red circle on the robot front edge (heading = up)
    dx, dy = hv(90)
    mx, my = cx + RBOT*dx, cy + RBOT*dy
    s += f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="4" fill="{MOUTH_COL}" stroke="#991111" stroke-width="0.8"/>'
    s += title_label(ox, oy, 'InteroceptiveSensor', 'gut / energy level')
    return s

def panel_proprioceptive(idx):
    ox, oy = cell_origin(idx); cx, cy = robot_center(idx)
    s = ''
    for sign, wr in [(-1, cx - RBOT - 10), (1, cx + RBOT + 10)]:
        s += (f'<rect x="{wr-4}" y="{cy-15}" width="8" height="30" rx="3" '
              f'fill="{WALL_COL}" opacity="0.55" stroke="{R_STK}" stroke-width="0.8"/>')
        a0, a1 = (195,255) if sign<0 else (285,345)
        s += arc_seg(wr, cy, 21, a0, a1, PRIMARY, 2.0)
        ex, ey = wr+21*math.cos(math.radians(a1)), cy-21*math.sin(math.radians(a1))
        s += dot(ex, ey, 2.5, PRIMARY)
    s += draw_robot(cx, cy)
    s += title_label(ox, oy, 'ProprioceptiveSensor', 'joint angle / velocity')
    return s

def panel_whisker(idx):
    ox, oy = cell_origin(idx); cx, cy = robot_center(idx)
    # Mount on front-left of robot
    ma = math.radians(115); md = RBOT * 0.85
    mx_w, my_w = cx + md*math.cos(ma), cy - md*math.sin(ma)
    # Whisker: quadratic bezier. tip_y = my_w - 68 ≈ cy - 88 = oy + 27 (just inside panel)
    tip_x, tip_y = mx_w - 5, my_w - 68
    ctrl_x, ctrl_y = mx_w + 10, my_w - 35
    s  = (f'<path d="M{mx_w:.1f},{my_w:.1f} Q{ctrl_x:.1f},{ctrl_y:.1f} {tip_x:.1f},{tip_y:.1f}" '
          f'fill="none" stroke="{PRIMARY}" stroke-width="1.8"/>')
    s += wall_h(tip_x - 22, tip_y - 4, 44, WALL_COL, 6)
    s += dot(tip_x, tip_y, 2.5, ACCENT)
    s += draw_robot(cx, cy)
    s += title_label(ox, oy, 'WhiskerSensor', 'bend proportion (0 = free, 1 = base)')
    return s

def panel_skycompass(idx):
    ox, oy = cell_origin(idx); cx, cy = robot_center(idx)
    # Sun upper-right, 45px above robot centre
    sx, sy = cx + 36, cy - 45
    s  = f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="10" fill="#f5c842" opacity="0.9"/>'
    for sa in range(0, 360, 45):
        ddx, ddy = hv(sa)
        s += (f'<line x1="{sx+12*ddx:.1f}" y1="{sy+12*ddy:.1f}" '
              f'x2="{sx+18*ddx:.1f}" y2="{sy+18*ddy:.1f}" stroke="#f5c842" stroke-width="1.5"/>')
    # DRA tuning ring — max radius keeps nodes within ±48px of robot centre
    n_dra, ring_r = 8, 46
    for i in range(n_dra):
        a = 360*i/n_dra
        ddx, ddy = hv(a)
        nx, ny = cx + ring_r*ddx, cy + ring_r*ddy
        resp = max(0, math.cos(math.radians(a - 90)))
        nr = 3 + resp*6
        op = 0.25 + resp*0.65
        s += f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="{nr:.1f}" fill="{PRIMARY}" opacity="{op:.2f}"/>'
    s += draw_robot(cx, cy)
    s += title_label(ox, oy, 'SkyCompassSensor', 'heading vs. sun azimuth')
    return s

def _camera_panel(idx, name, sub, color_mode):
    ox, oy = cell_origin(idx); cx, cy = robot_center(idx)
    # FOV cone, 70px deep
    fov = 48; dist = 70
    lx = cx + dist*math.cos(math.radians(90+fov))
    ly = cy - dist*math.sin(math.radians(90+fov))
    rx = cx + dist*math.cos(math.radians(90-fov))
    ry = cy - dist*math.sin(math.radians(90-fov))
    s  = (f'<polygon points="{cx:.0f},{cy:.0f} {lx:.0f},{ly:.0f} {rx:.0f},{ry:.0f}" '
          f'fill="{PRIMARY}" opacity="0.07"/>')
    s += (f'<line x1="{cx:.0f}" y1="{cy:.0f}" x2="{lx:.0f}" y2="{ly:.0f}" '
          f'stroke="{PRIMARY}" stroke-width="1" opacity="0.45"/>')
    s += (f'<line x1="{cx:.0f}" y1="{cy:.0f}" x2="{rx:.0f}" y2="{ry:.0f}" '
          f'stroke="{PRIMARY}" stroke-width="1" opacity="0.45"/>')
    # Pixel grid, positioned 22px above robot top edge
    cols_g, rows_g, px_w = 7, 4, 8
    gw, gh = cols_g*px_w, rows_g*px_w
    gx = cx - gw//2; gy = cy - RBOT - gh - 22
    grays = [0.15,0.45,0.75,0.35,0.80,0.25,0.60,
             0.55,0.85,0.40,0.70,0.30,0.65,0.50,
             0.30,0.65,0.85,0.55,0.40,0.70,0.45,
             0.80,0.35,0.25,0.90,0.60,0.50,0.70]
    rgb_c  = ['#c05050','#50a850','#5055c0','#c09030','#45b8b8','#b850a0','#7070d0',
              '#90c050','#c07040','#4080c0','#70c080','#c04060','#6090c0','#c06040',
              '#d0a040','#4890b0','#a03890','#58c068','#b05030','#3870a0','#d08040',
              '#70d060','#d05060','#4090c8','#b07848','#589048','#b060b8','#a0d060']
    for r in range(rows_g):
        for c in range(cols_g):
            i = r*cols_g + c
            px2, py2 = gx + c*px_w, gy + r*px_w
            if color_mode == 'gray':
                v = int(grays[i]*215)
                fill = f'rgb({v},{v},{v})'
            else:
                fill = rgb_c[i]
            s += f'<rect x="{px2}" y="{py2}" width="{px_w}" height="{px_w}" fill="{fill}"/>'
    s += (f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" '
          f'fill="none" stroke="{WALL_COL}" stroke-width="0.8"/>')
    s += draw_robot(cx, cy)
    s += title_label(ox, oy, name, sub)
    return s

def panel_graycamera(idx):
    return _camera_panel(idx, 'GrayCameraSensor', 'grayscale image (H×W)', 'gray')

def panel_rgbcamera(idx):
    return _camera_panel(idx, 'RGBCameraSensor', 'colour image (3×H×W)', 'rgb')

# ── Panel registry ────────────────────────────────────────────────────────────
PANELS = [
    ('gradient',       panel_gradient),
    ('color',          panel_color),
    ('collision',      panel_collision),
    ('distance',       panel_distance),
    ('interoceptive',  panel_interoceptive),
    ('proprioceptive', panel_proprioceptive),
    ('whisker',        panel_whisker),
    ('skycompass',     panel_skycompass),
    ('graycamera',     panel_graycamera),
    ('rgbcamera',      panel_rgbcamera),
]

# ── Output directory ──────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, 'docs', 'assets', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Overview SVG ──────────────────────────────────────────────────────────────
body = ''.join(fn(i) for i, (_, fn) in enumerate(PANELS))
overview_svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{TOTAL_W}" height="{TOTAL_H}" '
    f'viewBox="0 0 {TOTAL_W} {TOTAL_H}">\n'
    f'  <defs>{DEFS}  </defs>\n'
    f'  <rect width="{TOTAL_W}" height="{TOTAL_H}" rx="12" fill="{BG_PAGE}"/>\n'
    f'  {body}\n'
    f'</svg>'
)
overview_path = os.path.join(FIG_DIR, 'sensor_overview.svg')
with open(overview_path, 'w', encoding='utf-8') as f:
    f.write(overview_svg)
print(f'Saved: sensor_overview.svg  ({TOTAL_W}x{TOTAL_H})')

# ── Individual SVGs (viewBox-cropped to cell, overflow hidden) ────────────────
for i, (name, fn) in enumerate(PANELS):
    ox, oy = cell_origin(i)
    bg = f'<rect x="{ox}" y="{oy}" width="{CELL_W}" height="{CELL_H}" fill="{BG_PAGE}"/>'
    content = fn(i)
    indiv_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{INDIV_W}" height="{INDIV_H}" '
        f'viewBox="{ox} {oy} {CELL_W} {CELL_H}" '
        f'overflow="hidden">\n'
        f'  <defs>{DEFS}  </defs>\n'
        f'  {bg}\n'
        f'  {content}\n'
        f'</svg>'
    )
    out_path = os.path.join(FIG_DIR, f'sensor_{name}.svg')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(indiv_svg)
    print(f'Saved: sensor_{name}.svg')

print(f'\nDone. {len(PANELS)+1} SVG files written to docs/assets/figures/')
