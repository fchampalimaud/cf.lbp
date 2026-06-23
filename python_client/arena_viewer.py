"""
Arena viewer + keyboard teleoperation for the LBP robot.

Shows live positions of three robots:
  /robot     — forward (x1,y1) and backward (x2,y2) markers on robot 1
               arrow points from back → front (facing direction)
  /auxrobots — red robot centroid (x1,y1) then green robot centroid (x2,y2)

Keyboard control (same as manual_control.py):
  W/A/S/D — move,  Space — stop,  Esc — quit

OSC position listener: port 9000
Robot OSC: port 2390

Adjust ARENA_W / ARENA_H to match your tracking coordinate range.

Requires:  pip install pynput
"""

from __future__ import annotations

import math
import os
import sys
import threading
import tkinter as tk

sys.path.insert(0, os.path.dirname(__file__))
from robot_client import OscListener, RobotClient

try:
    from pynput import keyboard
except ImportError:
    sys.exit("Install pynput:  pip install pynput")

# ── Config ────────────────────────────────────────────────────────────────────

CANVAS_W  = 700
CANVAS_H  = 700
ARENA_W   = 1920   # coordinate range of the tracking system (pixels)
ARENA_H   = 1080

SPEED = 50
REFRESH_MS = 17    # ~60 fps

# ── Keyboard state ────────────────────────────────────────────────────────────

COMMANDS = {
    "w": ( SPEED,  SPEED),
    "s": (-SPEED, -SPEED),
    "a": (-SPEED,  SPEED),
    "d": ( SPEED, -SPEED),
    " ": (0, 0),
}
LABELS = {"w": "forward", "s": "backward", "a": "left", "d": "right", " ": "stop"}

_held: set = set()


def _key_char(key) -> str | None:
    try:
        return key.char.lower() if key.char else None
    except AttributeError:
        return " " if key == keyboard.Key.space else None


# ── Arena GUI ─────────────────────────────────────────────────────────────────

class ArenaViewer:
    def __init__(self, root: tk.Tk, robot: RobotClient):
        self.root  = root
        self.robot = robot
        root.title("LBP Arena Viewer")
        root.configure(bg="#111")
        root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H,
                                bg="#1e1e2e", highlightthickness=0)
        self.canvas.pack(padx=8, pady=(8, 0))

        self._status = tk.Label(root, text="", font=("Courier", 10),
                                bg="#111", fg="#aaa", anchor="w")
        self._status.pack(fill="x", padx=8, pady=(4, 8))

        self._lock    = threading.Lock()
        self._r1_fwd  = None
        self._r1_back = None
        self._r2      = None
        self._r3      = None
        self._corners = [None, None, None, None]

        self._driving_label = "stop"
        self._refresh()

    # ── OSC callbacks (called from listener thread) ───────────────────────────

    def on_robot(self, args):
        with self._lock:
            self._r1_fwd  = (args[0], args[1])
            self._r1_back = (args[2], args[3])

    def on_aux(self, args):
        with self._lock:
            self._r2 = (args[0], args[1])
            self._r3 = (args[2], args[3])

    def on_corners12(self, args):
        print(f"corners12: {args}")
        with self._lock:
            self._corners[0] = (args[0], args[1])
            self._corners[1] = (args[2], args[3])

    def on_corners34(self, args):
        print(f"corners34: {args}")
        with self._lock:
            self._corners[2] = (args[0], args[1])
            self._corners[3] = (args[2], args[3])

    # ── Keyboard helpers ──────────────────────────────────────────────────────

    def drive(self, label: str, vl: int, vr: int):
        self._driving_label = label
        self.robot.set_wheels(vl, vr)

    # ── Render loop ───────────────────────────────────────────────────────────

    def _refresh(self):
        with self._lock:
            r1_fwd  = self._r1_fwd
            r1_back = self._r1_back
            r2      = self._r2
            r3      = self._r3
            corners = list(self._corners)

        c = self.canvas
        c.delete("robots")
        c.delete("corners")

        canvas_corners = [self._to_canvas(*pt) if pt else None for pt in corners]
        if all(canvas_corners):
            cx_mean = sum(x for x, y in canvas_corners) / 4
            cy_mean = sum(y for x, y in canvas_corners) / 4
            sorted_corners = sorted(canvas_corners,
                                    key=lambda p: math.atan2(p[1] - cy_mean, p[0] - cx_mean))
            pts_flat = [coord for x, y in sorted_corners for coord in (x, y)]
            c.create_polygon(pts_flat, fill="", outline="#00ffff", width=1, tags="corners")
        for i, cpt in enumerate(canvas_corners):
            if cpt:
                cx, cy = cpt
                r = 5
                c.create_rectangle(cx - r, cy - r, cx + r, cy + r,
                                   outline="#00ffff", fill="", width=2, tags="corners")
                c.create_text(cx, cy - 10, text=str(i), fill="#00ffff",
                              font=("Courier", 8), tags="corners")

        if r1_fwd and r1_back:
            fx, fy = self._to_canvas(*r1_fwd)
            bx, by = self._to_canvas(*r1_back)
            self._draw_robot1(fx, fy, bx, by)

        if r2:
            self._draw_circle(*self._to_canvas(*r2), r=10, color="red")
        if r3:
            self._draw_circle(*self._to_canvas(*r3), r=10, color="#00dd00")

        r1_txt = f"r1 fwd={r1_fwd} back={r1_back}" if r1_fwd else "r1 —"
        r2_txt = f"r2={r2}" if r2 else "r2 —"
        r3_txt = f"r3={r3}" if r3 else "r3 —"
        self._status.config(
            text=f"[{self._driving_label:8s}]  {r1_txt}  {r2_txt}  {r3_txt}"
        )

        self.root.after(REFRESH_MS, self._refresh)

    def _to_canvas(self, x, y):
        return x * CANVAS_W / ARENA_W, y * CANVAS_H / ARENA_H

    def _draw_robot1(self, fx, fy, bx, by):
        dx, dy = fx - bx, fy - by
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        hw = 10   # half-width at base

        pts = [
            fx, fy,
            bx + px * hw, by + py * hw,
            bx - px * hw, by - py * hw,
        ]
        self.canvas.create_polygon(pts, fill="white", outline="#888",
                                   width=1, tags="robots")
        # yellow dot at tip
        r = 4
        self.canvas.create_oval(fx - r, fy - r, fx + r, fy + r,
                                 fill="#ffdd00", outline="", tags="robots")

    def _draw_circle(self, x, y, r, color):
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill=color, outline="white", width=1,
                                tags="robots")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    root = tk.Tk()

    robot    = RobotClient()
    listener = OscListener(port=9000)

    viewer = ArenaViewer(root, robot)
    listener.subscribe("/robot",      viewer.on_robot)
    listener.subscribe("/auxrobots",  viewer.on_aux)
    listener.subscribe("/corners12",  viewer.on_corners12)
    listener.subscribe("/corners34",  viewer.on_corners34)

    robot.start()
    listener.start()

    stop_event = threading.Event()

    def on_press(key):
        ch = _key_char(key)
        if ch in COMMANDS:
            _held.add(ch)
            for k in ("w", "s", "a", "d", " "):
                if k in _held:
                    vl, vr = COMMANDS[k]
                    viewer.drive(LABELS[k], vl, vr)
                    return
        elif key == keyboard.Key.esc:
            stop_event.set()
            root.after(0, root.destroy)
            return False

    def on_release(key):
        ch = _key_char(key)
        if ch in _held:
            _held.discard(ch)
            for k in ("w", "s", "a", "d", " "):
                if k in _held:
                    vl, vr = COMMANDS[k]
                    viewer.drive(LABELS[k], vl, vr)
                    return
            viewer.drive("stop", 0, 0)
            robot.stop_wheels()

    kb = keyboard.Listener(on_press=on_press, on_release=on_release)
    kb.start()

    def on_close():
        robot.stop_wheels()
        listener.stop()
        robot.stop()
        kb.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    run()
