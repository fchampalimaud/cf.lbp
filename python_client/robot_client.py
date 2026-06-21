"""
RobotClient — WiFi interface to the LBP MKR1010 robot + RPi camera.

OSC over UDP:
  Robot IP:   192.168.0.224, port 2390
  Listen on:  local port 2390  (robot replies to whichever IP:port last sent to it)

Camera (RPi):
  RPi IP:     192.168.0.190, port 5002
  Protocol:   send b'hello' keepalive every 0.5 s → receive raw JPEG datagrams

Requires:  pip install python-osc opencv-python numpy
"""

import socket
import threading
import time
import struct
from typing import Callable, Optional

import numpy as np

# ── Minimal inline OSC codec (no extra deps beyond python-osc) ───────────────
# If you have python-osc installed, replace with:
#   from pythonosc.osc_message import OscMessage
#   from pythonosc.osc_message_builder import OscMessageBuilder


def _pad4(n: int) -> int:
    return (n + 3) & ~3


def _encode_osc(address: str, *args) -> bytes:
    """Encode an OSC message with int or float args."""
    def pad_str(s: str) -> bytes:
        b = s.encode() + b"\x00"
        return b.ljust(_pad4(len(b)), b"\x00")

    type_tags = ","
    encoded_args = b""
    for a in args:
        if isinstance(a, int):
            type_tags += "i"
            encoded_args += struct.pack(">i", a)
        elif isinstance(a, float):
            type_tags += "f"
            encoded_args += struct.pack(">f", a)
        else:
            raise TypeError(f"Unsupported OSC arg type: {type(a)}")

    return pad_str(address) + pad_str(type_tags) + encoded_args


def _decode_osc(data: bytes) -> tuple[str, list]:
    """Decode an OSC message; returns (address, [args...])."""
    def read_str(buf: bytes, offset: int) -> tuple[str, int]:
        end = buf.index(b"\x00", offset)
        s = buf[offset:end].decode(errors="replace")
        return s, _pad4(end + 1)

    try:
        address, pos = read_str(data, 0)
        type_tags, pos = read_str(data, pos)
        args = []
        for tag in type_tags[1:]:  # skip leading ','
            if tag == "i":
                args.append(struct.unpack_from(">i", data, pos)[0])
                pos += 4
            elif tag == "f":
                args.append(struct.unpack_from(">f", data, pos)[0])
                pos += 4
            # skip unknown tags
        return address, args
    except Exception:
        return "", []


# ─────────────────────────────────────────────────────────────────────────────


class RobotClient:
    """
    Non-blocking WiFi client for the LBP robot.

    Usage::

        robot = RobotClient()
        robot.start()                   # spawns recv threads
        robot.set_wheels(50, 50)        # drive forward
        frame = robot.frame             # latest camera frame (numpy BGR)
        robot.stop()

    Or as a context manager::

        with RobotClient() as robot:
            robot.set_wheels(30, -30)
            time.sleep(1)
    """

    ROBOT_IP   = "192.168.0.224"
    ROBOT_PORT = 2390
    LOCAL_PORT = 2390           # robot replies to whichever port we send FROM

    PI_IP   = "192.168.0.190"
    PI_PORT = 5002
    PI_KEEPALIVE_INTERVAL = 0.5

    def __init__(
        self,
        robot_ip:   str = ROBOT_IP,
        robot_port: int = ROBOT_PORT,
        local_port: int = LOCAL_PORT,
        pi_ip:      str = PI_IP,
        pi_port:    int = PI_PORT,
        enable_camera: bool = True,
    ):
        self.robot_ip   = robot_ip
        self.robot_port = robot_port
        self.local_port = local_port
        self.pi_ip      = pi_ip
        self.pi_port    = pi_port
        self.enable_camera = enable_camera

        # ── Latest sensor state ───────────────────────────────────────────────
        self.bumpers:  list = [0, 0, 0, 0]      # [pin0, pin8, pin10, pin13] — 1 = pressed
        self.analogs:  list = [0, 0, 0, 0]      # A1, A2, A5(battery), A6  raw ADC
        self.encoders: list = [0, 0]            # left, right — tick delta per loop
        self.gyro:     list = [0.0, 0.0, 0.0]  # deg/s  x y z
        self.acc:      list = [0.0, 0.0, 0.0]  # g      x y z
        self.mag:      list = [0.0, 0.0, 0.0]  # uT     x y z
        self.energy: dict = {}                  # battery_level, current3/4, mkr_voltage
        self.frame: Optional[np.ndarray] = None # latest BGR image from RPi camera

        # ── Callbacks (optional) ─────────────────────────────────────────────
        self.on_bumpers:  Optional[Callable[[list], None]] = None
        self.on_analogs:  Optional[Callable[[list], None]] = None
        self.on_encoders: Optional[Callable[[list], None]] = None
        self.on_frame:    Optional[Callable[[np.ndarray], None]] = None

        self._running  = False
        self._osc_sock: Optional[socket.socket] = None
        self._cam_sock: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Open sockets and start background receive threads."""
        if self._running:
            return
        self._running = True

        # OSC socket — bind so the robot can reply to our port
        self._osc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._osc_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._osc_sock.bind(("0.0.0.0", self.local_port))
        self._osc_sock.settimeout(1.0)

        # Register our IP:port with the robot by sending an initial wheels=0 packet
        self._transmit_osc("/wheels", 0, 0)

        self._threads = [
            threading.Thread(target=self._osc_recv_loop, daemon=True, name="lbp-osc"),
        ]

        if self.enable_camera:
            self._cam_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._cam_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._cam_sock.bind(("0.0.0.0", self.pi_port))
            self._cam_sock.settimeout(1.0)
            self._threads += [
                threading.Thread(target=self._cam_recv_loop,      daemon=True, name="lbp-cam"),
                threading.Thread(target=self._cam_keepalive_loop, daemon=True, name="lbp-cam-ka"),
            ]

        for t in self._threads:
            t.start()

    def stop(self):
        """Stop threads and close sockets."""
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        if self._osc_sock:
            self._osc_sock.close()
            self._osc_sock = None
        if self._cam_sock:
            self._cam_sock.close()
            self._cam_sock = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Motor / actuator commands ─────────────────────────────────────────────

    def set_wheels(self, left: int, right: int):
        """
        Set wheel motor duty cycles.
        left, right: integer in [-100, 100]
        """
        self._transmit_osc("/wheels", int(left), int(right))

    def stop_wheels(self):
        self._transmit_osc("/wheels", 0, 0)

    def set_motor1(self, duty: int, duration_ms: int):
        """Pulse M3 at duty for duration_ms milliseconds."""
        self._transmit_osc("/motor1", int(duty), int(duration_ms))

    def set_motor2(self, duty: int, duration_ms: int):
        """Pulse M4 at duty for duration_ms milliseconds."""
        self._transmit_osc("/motor2", int(duty), int(duration_ms))

    def set_tongue(self, angle: int):
        """Set servo1 to angle (degrees)."""
        self._transmit_osc("/tongue", int(angle))

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def battery_raw(self) -> int:
        """Raw battery ADC reading on A5 (divide by ~77 for volts, approx)."""
        return self.analogs[2]

    @property
    def any_bumper(self) -> bool:
        return any(self.bumpers)

    # ── Internal OSC send ─────────────────────────────────────────────────────

    def _transmit_osc(self, address: str, *args):
        if not self._osc_sock:
            return
        data = _encode_osc(address, *args)
        with self._send_lock:
            self._osc_sock.sendto(data, (self.robot_ip, self.robot_port))

    # ── OSC receive loop ──────────────────────────────────────────────────────

    def _osc_recv_loop(self):
        while self._running:
            try:
                data, _ = self._osc_sock.recvfrom(4096)
                self._dispatch_osc(data)
            except socket.timeout:
                continue
            except OSError:
                break

    def _dispatch_osc(self, data: bytes):
        address, args = _decode_osc(data)
        if not address:
            return

        if address == "/bumpers" and len(args) == 5:
            self.bumpers = args[1:]          # drop packet counter
            if self.on_bumpers:
                self.on_bumpers(self.bumpers)

        elif address == "/analogs" and len(args) == 5:
            self.analogs = args[1:]
            if self.on_analogs:
                self.on_analogs(self.analogs)

        elif address == "/wencoders" and len(args) == 3:
            self.encoders = args[1:]
            if self.on_encoders:
                self.on_encoders(self.encoders)

        elif address == "/gyro" and len(args) == 4:
            self.gyro = args[1:]

        elif address == "/acc" and len(args) == 4:
            self.acc = args[1:]

        elif address == "/mag" and len(args) == 4:
            self.mag = args[1:]

        elif address == "/energy" and len(args) == 5:
            _, battery_level, current3, current4, mkr_voltage = args
            self.energy = {
                "battery_level": battery_level,
                "current3":      current3,
                "current4":      current4,
                "mkr_voltage":   mkr_voltage,
            }

    # ── Camera receive loop ───────────────────────────────────────────────────

    def _cam_recv_loop(self):
        try:
            import cv2
        except ImportError:
            print("[RobotClient] opencv-python not installed — camera disabled.")
            return

        while self._running:
            try:
                data, _ = self._cam_sock.recvfrom(65535)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self.frame = img
                    if self.on_frame:
                        self.on_frame(img)
            except socket.timeout:
                continue
            except OSError:
                break

    def _cam_keepalive_loop(self):
        while self._running:
            try:
                self._cam_sock.sendto(b"hello", (self.pi_ip, self.pi_port))
            except OSError:
                break
            time.sleep(self.PI_KEEPALIVE_INTERVAL)


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    def _on_bumpers(b):
        print(f"bumpers: {b}")

    def _on_encoders(e):
        print(f"encoders: {e}")

    print("Connecting to robot… (Ctrl-C to stop)")
    with RobotClient() as robot:
        robot.on_bumpers  = _on_bumpers
        robot.on_encoders = _on_encoders
        try:
            while True:
                print(
                    f"  analogs={robot.analogs}  "
                    f"bumpers={robot.bumpers}  "
                    f"enc={robot.encoders}  "
                    f"frame={'yes' if robot.frame is not None else 'none'}"
                )
                time.sleep(0.5)
        except KeyboardInterrupt:
            robot.stop_wheels()
            sys.exit(0)
