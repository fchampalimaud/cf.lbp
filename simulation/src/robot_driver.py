"""
robot_driver.py — real-robot I/O for the 2D LBP simulator.

One thread is created per unique robot_address (host:port).  The thread type
is determined by the sensor type(s) registered at that address:

  CameraSensor family  → CameraThread   : UDP client, sends keepalives,
                                          receives JPEG frames, decodes to numpy.
  Any other sensor     → OscThread      : OSC UDP server bound to the local port
                                          extracted from robot_address; dispatches
                                          incoming messages by osc_path.

Each thread writes decoded data directly into sensor._robot_value (a numpy array).
Nothing else in the codebase needs to know about protocols or threading.

Motor output
------------
call send_wheels(host, port, vL, vR) to fire a /wheels OSC message.

Usage (from sim_controller)
---------------------------
    driver = RobotDriver()
    driver.start(circuit.sensors)          # each sensor carries its own robot_address
    ...
    host, port, osc_path = RobotDriver._parse_address(motor_layer.robot_address)
    driver.send_motor(host, port, osc_path, mL, mR)
    ...
    driver.stop()
"""

import re
import socket
import struct
import threading
import time
import numpy as np

from sensors import CameraSensor


# ── OSC helpers (no external dependency) ──────────────────────────────────────

def _osc_str(s: str) -> bytes:
    """Encode a string as a null-terminated, 4-byte-padded OSC string."""
    b = s.encode('utf-8') + b'\x00'
    pad = (4 - len(b) % 4) % 4
    return b + b'\x00' * pad


def _osc_build(address: str, *args) -> bytes:
    """Build a minimal OSC message with int or float arguments."""
    type_tag = ',' + ''.join('i' if isinstance(a, int) else 'f' for a in args)
    data = _osc_str(address) + _osc_str(type_tag)
    for a in args:
        if isinstance(a, int):
            data += struct.pack('>i', a)
        else:
            data += struct.pack('>f', float(a))
    return data


def _parse_address(addr: str):
    """Parse 'host:port[/osc_path][[type_tag]][(i,j,...)]' into components.

    Returns (host, port, osc_path, type_tag, indices) where:
      type_tag : str  — OSC type string from [...], e.g. 'idddd'; '' if absent
      indices  : list[int] | None — element indices from (...); None = use all

    Examples:
      '192.168.0.1:2390/analogs[idddd](1,2)'  → ('192.168.0.1', 2390, '/analogs', 'idddd', [1, 2])
      '192.168.0.1:2390/wheels'               → ('192.168.0.1', 2390, '/wheels', '', None)
      '192.168.0.1:2390'                      → ('192.168.0.1', 2390, '', '', None)
    """
    if not addr:
        return '', 0, '', '', None

    # Strip trailing (i,j,...) index list
    indices = None
    m = re.search(r'\(([^)]*)\)\s*$', addr)
    if m:
        idx_str = m.group(1).strip()
        if idx_str:
            try:
                indices = [int(x.strip()) for x in idx_str.split(',') if x.strip()]
            except ValueError:
                pass
        addr = addr[:m.start()].rstrip()

    # Strip trailing [type_tag]
    type_tag = ''
    m = re.search(r'\[([^\]]*)\]\s*$', addr)
    if m:
        type_tag = m.group(1)
        addr = addr[:m.start()].rstrip()

    # Split osc_path at first '/'
    osc_path = ''
    slash = addr.find('/')
    if slash != -1:
        osc_path = addr[slash:]
        addr = addr[:slash]

    try:
        h, p = addr.rsplit(':', 1)
        return h.strip(), int(p), osc_path, type_tag, indices
    except (ValueError, AttributeError):
        pass
    return addr.strip(), 0, osc_path, type_tag, indices


def _osc_parse(data: bytes):
    """Parse an OSC message; return (address, values, type_str) or (None, [], '').

    type_str is the raw OSC type string without the leading comma, e.g. 'idddd'.
    Supported type tags: i (int32), f (float32), d (float64/double).
    """
    try:
        end = data.index(b'\x00')
        address = data[:end].decode('utf-8')
        off = (end + 4) & ~3

        end2 = data.index(b'\x00', off)
        tags = data[off:end2].decode('utf-8')
        off = (end2 + 4) & ~3

        type_str = tags.lstrip(',')
        values = []
        for tag in tags:
            if tag == ',':
                continue
            elif tag == 'i':
                values.append(struct.unpack('>i', data[off:off+4])[0])
                off += 4
            elif tag == 'f':
                values.append(struct.unpack('>f', data[off:off+4])[0])
                off += 4
            elif tag == 'd':
                values.append(struct.unpack('>d', data[off:off+8])[0])
                off += 8
        return address, values, type_str
    except Exception:
        return None, [], ''


# ── Per-address threads ────────────────────────────────────────────────────────

class OscThread(threading.Thread):
    """
    Binds a UDP socket to *local_port* and receives OSC messages.
    Dispatches by osc_path to the registered sensors' _robot_value buffers.

    robot_address format: 'host:port/osc_path[type_tag](i,j,...)'
      /osc_path  — OSC address to listen for (also accepted from sensor.osc_path)
      [type_tag] — expected OSC type string, e.g. 'idddd'; used for validation
      (i,j,...)  — zero-based indices of values to extract; must match sensor.n
    """

    def __init__(self, local_port: int, sensors):
        super().__init__(daemon=True, name=f'OscThread:{local_port}')
        self._port = local_port
        self._stop = threading.Event()

        # Bind the socket immediately in __init__ so send() can use it before run() fires.
        # The robot identifies the host by the source port of the first motor command it
        # receives, so motor commands must go out from the same port we receive on.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind(('', local_port))
            print(f'[Robot] OscThread bound to port {local_port}')
        except OSError as e:
            print(f'[Robot] OscThread failed to bind port {local_port}: {e}')
        self._sock.settimeout(0.5)

        # Per-path packet counters for live Hz display in the Robot tab.
        self._recv_counts: dict = {}
        self._recv_t: float = time.perf_counter()

        # map osc_path → [(sensor, type_tag, indices), ...]
        self._dispatch = {}
        for s in sensors:
            addr = getattr(s, 'robot_address', '')
            _, _, path, type_tag, indices = _parse_address(addr)
            print(f'[Robot] registering sensor "{s.name}": path={path!r} '
                  f'type_tag={type_tag!r} indices={indices} n={getattr(s, "n", None)}')
            if not path:
                print(f'[Robot]   → skipped (no osc_path in robot_address)')
                continue
            n = getattr(s, 'n', None)
            if indices is not None and n is not None and len(indices) != n:
                print(f'[Robot]   → skipped: index count {len(indices)} != n={n}')
                continue
            self._dispatch.setdefault(path, []).append((s, type_tag, indices))
            print(f'[Robot]   → registered on path {path!r}')
        print(f'[Robot] OscThread dispatch table: {list(self._dispatch.keys())}')

    def send(self, host: str, port: int, data: bytes):
        """Send a datagram from this thread's bound socket (preserves source port)."""
        try:
            self._sock.sendto(data, (host, port))
        except OSError:
            pass

    def path_rates(self) -> dict:
        """Return {osc_path: Hz} for each received path and reset counters."""
        now = time.perf_counter()
        elapsed = max(now - self._recv_t, 1e-6)
        rates = {p: c / elapsed for p, c in self._recv_counts.items()}
        self._recv_counts = {}
        self._recv_t = now
        return rates

    def run(self):
        pkt_count = 0
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            pkt_count += 1
            address, values, _ = _osc_parse(data)
            if address is None or not values:
                continue
            targets = self._dispatch.get(address, [])
            if not targets:
                continue
            self._recv_counts[address] = self._recv_counts.get(address, 0) + 1
            for sensor, _, indices in targets:
                arr = np.array(values, dtype=np.float32)
                if indices is not None:
                    try:
                        arr = arr[indices]
                    except IndexError:
                        continue
                sensor._robot_value = arr
        self._sock.close()

    def stop(self):
        self._stop.set()


class MotorThread(threading.Thread):
    """
    Sends motor commands to the robot at a fixed Hz, independently of the network step rate.

    get_commands() must return a list of (host, port, osc_path, vL, vR) tuples — one per
    MotorLayer. It is called every tick and reads the latest network output.
    """

    def __init__(self, get_commands, driver, hz=60):
        super().__init__(daemon=True, name='MotorThread')
        self._get_cmds = get_commands
        self._driver   = driver
        self._period   = 1.0 / hz
        self._stop_evt   = threading.Event()
        self.last_vL     = 0.0   # last integer value actually sent (readable by oscilloscope)
        self.last_vR     = 0.0
        self._send_counts: dict = {}
        self._send_t: float     = time.perf_counter()

    def send_rates(self) -> dict:
        """Return {osc_path: Hz} for motor sends since the last call."""
        now     = time.perf_counter()
        elapsed = max(now - self._send_t, 1e-6)
        rates   = {p: c / elapsed for p, c in self._send_counts.items()}
        self._send_counts = {}
        self._send_t      = now
        return rates

    def run(self):
        next_t = time.perf_counter()
        while not self._stop_evt.is_set():
            for host, port, osc_path, vL, vR in self._get_cmds():
                self._driver.send_motor(host, port, osc_path, vL, vR)
                self.last_vL = float(int(round(vL)))
                self.last_vR = float(int(round(vR)))
                self._send_counts[osc_path] = self._send_counts.get(osc_path, 0) + 1
            next_t += self._period
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.perf_counter()

    def stop(self):
        self._stop_evt.set()


class CameraThread(threading.Thread):
    """
    UDP client for the RPi JPEG camera stream (send_frames.py protocol).

    Sends a keepalive packet to *host:port* every second, then reads
    incoming JPEG datagrams, decodes them, resizes to each registered
    sensor's (height, width), and writes the result into sensor._robot_value.

    robot_value shape for GrayCameraSensor : (height * width,)  float32 [0,1]
    robot_value shape for RGBCameraSensor  : (3 * height * width,) float32 CHW [0,1]
    """

    KEEPALIVE_INTERVAL = 1.0   # seconds between keepalive packets

    def __init__(self, host: str, port: int, sensors):
        super().__init__(daemon=True, name=f'CameraThread:{host}:{port}')
        self._host    = host
        self._port    = port
        self._sensors = sensors
        self._stop    = threading.Event()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        last_ka = 0.0
        buf = b''

        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_ka >= self.KEEPALIVE_INTERVAL:
                try:
                    sock.sendto(b'\x00', (self._host, self._port))
                except OSError:
                    pass
                last_ka = now

            try:
                chunk, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue

            buf += chunk
            # Extract complete JPEG frames (SOI=0xFFD8 … EOI=0xFFD9)
            while True:
                start = buf.find(b'\xff\xd8')
                end   = buf.find(b'\xff\xd9', start + 2) if start != -1 else -1
                if start == -1 or end == -1:
                    break
                frame_bytes = buf[start:end + 2]
                buf = buf[end + 2:]
                self._decode_and_push(frame_bytes)

        sock.close()

    def _decode_and_push(self, jpeg_bytes: bytes):
        try:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(jpeg_bytes))
        except Exception:
            return

        for sensor in self._sensors:
            mode = getattr(sensor, 'mode', 'gray')
            w, h = sensor.width, sensor.height
            try:
                resized = img.resize((w, h), Image.BILINEAR)
                if mode == 'rgb':
                    arr = np.array(resized.convert('RGB'), dtype=np.float32) / 255.0  # (H,W,3)
                    sensor._last_frame  = arr
                    sensor._robot_value = arr.transpose(2, 0, 1).reshape(-1)  # CHW flat
                    if getattr(sensor, 'lateralized', False):
                        mid     = w // 2
                        overlap = getattr(sensor, 'overlap', 0)
                        l_end   = int(np.clip(mid + overlap, 0, w))
                        r_start = int(np.clip(mid - overlap, 0, w))
                        sensor._left_output  = arr[:, :l_end,   :].transpose(2,0,1).reshape(-1).astype(np.float32)
                        sensor._right_output = arr[:, r_start:, :].transpose(2,0,1).reshape(-1).astype(np.float32)
                else:
                    arr = np.array(resized.convert('L'), dtype=np.float32) / 255.0  # (H,W)
                    sensor._last_frame  = arr
                    sensor._robot_value = arr.reshape(-1)  # row-major flat
                    if getattr(sensor, 'lateralized', False):
                        mid     = w // 2
                        overlap = getattr(sensor, 'overlap', 0)
                        l_end   = int(np.clip(mid + overlap, 0, w))
                        r_start = int(np.clip(mid - overlap, 0, w))
                        sensor._left_output  = arr[:, :l_end  ].reshape(-1).astype(np.float32)
                        sensor._right_output = arr[:, r_start:].reshape(-1).astype(np.float32)
            except Exception:
                pass

    def stop(self):
        self._stop.set()


# ── RobotDriver ───────────────────────────────────────────────────────────────

class RobotDriver:
    """
    Manages one thread per unique robot_address among the active sensors.

    Each sensor's robot_address ('host:port') determines its connection.
    Camera sensors use CameraThread (UDP client); all others use OscThread.
    Motor /wheels commands are sent to the host:port of every OSC address
    (i.e. every non-camera sensor address), via motor_targets().
    """

    def __init__(self):
        self._threads: dict[str, threading.Thread] = {}

    def start(self, sensors):
        """Inspect sensors, group by (host, port), start one thread per group.

        All sensors sharing the same host:port get one OscThread (or CameraThread)
        regardless of their osc_path/type_tag/indices suffixes.  The thread
        dispatches incoming OSC messages internally to each sensor by osc_path.
        """
        self.stop()   # clean up any previous run

        # Group by (host, port) so sensors on the same endpoint share one socket.
        groups: dict[tuple, list] = {}
        for s in sensors:
            addr = getattr(s, 'robot_address', '').strip()
            if not addr:
                continue
            host, port, *_ = _parse_address(addr)
            if not host or not port:
                continue
            groups.setdefault((host, port), []).append(s)

        for (host, port), group in groups.items():
            names = [s.name for s in group]
            is_camera = all(isinstance(s, CameraSensor) for s in group)
            kind = 'CameraThread' if is_camera else 'OscThread'
            print(f'[Robot] starting {kind} for {host}:{port} — sensors: {names}')
            t = CameraThread(host, port, group) if is_camera else OscThread(port, group)
            t.start()
            self._threads[f'{host}:{port}'] = t

    def stop(self):
        for t in self._threads.values():
            t.stop()
        for t in self._threads.values():
            t.join(timeout=2.0)
        self._threads.clear()

    def send_motor(self, host: str, port: int, osc_path: str, vL: float, vR: float):
        """Send a motor OSC message to host:port.

        Uses the OscThread's already-bound socket (same source port the robot
        is listening on) so the robot knows which port to send sensor data back to.
        """
        vL = max(-100.0, min(100.0, vL))
        vR = max(-100.0, min(100.0, vR))
        msg = _osc_build(osc_path, int(round(vL)), int(round(vR)))
        key = f'{host}:{port}'
        t = self._threads.get(key)
        if isinstance(t, OscThread):
            t.send(host, port, msg)
        else:
            # Fallback: no OscThread for this host:port (camera-only group, etc.)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.sendto(msg, (host, port))
                s.close()
            except OSError:
                pass

    def get_rates(self) -> dict:
        """Return {osc_path: Hz} aggregated from all active OscThreads."""
        rates = {}
        for t in self._threads.values():
            if isinstance(t, OscThread):
                rates.update(t.path_rates())
        return rates

    def clear_robot_values(self, sensors):
        """Reset all sensor robot buffers (call on stop/reset)."""
        for s in sensors:
            s._robot_value = None

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_address(addr: str):
        """Parse 'host:port[/osc_path][[type_tag]][(indices)]' → 5-tuple.

        Delegates to the module-level _parse_address function.
        Returns (host, port, osc_path, type_tag, indices).
        """
        return _parse_address(addr)
