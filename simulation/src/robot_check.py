"""
robot_check.py — pre-flight connection test for real-robot mode (Qt-free).

RobotCheck reads the circuit's robot addresses once, then run() works through a
checklist and reports each result through a callback:

  config            every sensor / motor address parses and is usable
  <host:port>:port  the local UDP port the robot talks to is free
  <host:port>:ping  the robot answers ping, and how fast
  <host:port>:reply the robot streams data back after receiving a probe
  <host:port>:s:<n> each sensor's OSC path arrives and its indices/formula work

The probe is the same packet real-robot mode sends — /wheels 0 0 (stop) to the
motor address — because the firmware streams its sensors to whoever sent it the
last packet. The sockets are closed when run() returns, so real-robot mode can
bind the port straight after.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from robot_driver import (_parse_address, _osc_build, _osc_messages, _eval_formula,
                          _BUNDLE_TAG)

PENDING, RUNNING, PASS, WARN, FAIL, SKIP = 'pending', 'running', 'pass', 'warn', 'fail', 'skip'


@dataclass
class SensorSpec:
    name: str
    path: str
    indices: list | None
    formula: str
    n: int | None
    is_camera: bool


@dataclass
class Endpoint:
    host: str
    port: int
    sensors: list = field(default_factory=list)
    motor_paths: list = field(default_factory=list)

    @property
    def label(self):
        return f'{self.host}:{self.port}'

    @property
    def is_camera(self):
        return bool(self.sensors) and all(s.is_camera for s in self.sensors)


def plan(circuit):
    """Group the circuit's robot addresses by host:port, as RobotDriver does.
    Returns (endpoints, problems) — problems are human-readable config errors."""
    endpoints: dict = {}
    problems = []

    def endpoint(host, port):
        return endpoints.setdefault((host, port), Endpoint(host, port))

    for s in circuit.sensors:
        addr = getattr(s, 'robot_address', '').strip()
        if not addr:
            continue
        host, port, path, _, indices, formula = _parse_address(addr)
        if not host or not port:
            problems.append(f"Sensor '{s.name}': address '{addr}' has no host:port.")
            continue
        is_camera = bool(getattr(s, 'is_camera', False))
        n = getattr(s, 'n', None)
        if not is_camera:
            if not path:
                problems.append(f"Sensor '{s.name}': address '{addr}' has no OSC path "
                                f"(e.g. /analogs), so no packet is routed to it.")
                continue
            if indices is not None and n is not None and len(indices) != n:
                problems.append(f"Sensor '{s.name}': {len(indices)} indices in (...) but the "
                                f"sensor has {n} neurons — they must match, or it gets no data.")
                continue
        endpoint(host, port).sensors.append(
            SensorSpec(s.name, path, indices, formula, n, is_camera))

    for layer in circuit.layers:
        if not getattr(layer, 'drives_wheels', False):
            continue
        addr = getattr(layer, 'robot_address', '').strip()
        if not addr:
            continue
        host, port, path, *_ = _parse_address(addr)
        if not host or not port or not path:
            problems.append(f"Motor layer '{layer.name}': address '{addr}' needs "
                            f"host:port/path, e.g. 192.168.0.224:2390/wheels.")
            continue
        endpoint(host, port).motor_paths.append(path)

    return list(endpoints.values()), problems


def parse_ping(output: str) -> list:
    """Round-trip times (ms) of the reply lines in ping's output. Reply lines are
    found by their TTL field, which every locale of Windows and Unix ping prints."""
    times = []
    for line in output.splitlines():
        if 'ttl' not in line.lower():
            continue
        m = re.search(r'[=<]\s*(\d+(?:[.,]\d+)?)\s*ms', line)
        if m:
            times.append(float(m.group(1).replace(',', '.')))
    return times


def ping_host(host: str, count: int, cancel: threading.Event):
    """Ping host; returns reply times in ms, or None if ping can't be run."""
    if sys.platform == 'win32':
        cmd = ['ping', '-n', str(count), '-w', '1000', host]
        flags = subprocess.CREATE_NO_WINDOW
    else:
        cmd = ['ping', '-c', str(count), '-W', '1', host]
        flags = 0
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors='replace', creationflags=flags)
    except OSError:
        return None
    deadline = time.monotonic() + count * 1.5 + 5
    while True:
        try:
            out, _ = proc.communicate(timeout=0.2)
            return parse_ping(out)
        except subprocess.TimeoutExpired:
            if cancel.is_set() or time.monotonic() > deadline:
                proc.kill()
                proc.communicate()
                return []


def windows_public_network() -> bool:
    """True if Windows classes any active network connection as Public."""
    if sys.platform != 'win32':
        return False
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command', '(Get-NetConnectionProfile).NetworkCategory'],
            capture_output=True, text=True, errors='replace', timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.strip() == 'Public' for line in out.splitlines())


def analyse_sensor(spec: SensorSpec, msgs: list, window_s: float):
    """Judge one OSC sensor from the (path, values) messages received in window_s."""
    seen = sorted({p for p, _ in msgs})
    vals = [v for p, v in msgs if p == spec.path]
    if not vals:
        return FAIL, (f"No {spec.path} packets arrived. The robot sent: "
                      f"{', '.join(seen) or 'nothing'}. Check the OSC path in the Robot address, "
                      f"or whether this firmware sends {spec.path}.")
    arr = np.asarray(vals[-1], dtype=np.float32)
    if spec.indices is not None:
        bad = [i for i in spec.indices if not -len(arr) <= i < len(arr)]
        if bad:
            return FAIL, (f"{spec.path} packets have {len(arr)} values (indices 0–{len(arr) - 1}) "
                          f"but the address asks for index {bad[0]}.")
        arr = arr[spec.indices]
    if spec.formula:
        try:
            arr = np.asarray(_eval_formula(spec.formula, arr), dtype=np.float32)
        except Exception as e:
            return FAIL, f"Formula {{{spec.formula}}} fails on the received values: {e}"
    shown = ', '.join(f'{v:g}' for v in np.atleast_1d(arr)[:8])
    detail = f"{len(vals) / window_s:.0f} Hz — latest values: {shown}"
    if spec.indices is None and spec.n is not None and arr.size != spec.n:
        return WARN, (f"{detail}. The packet has {arr.size} values but the sensor has {spec.n} "
                      f"neurons — pick values with indices, e.g. (1,2).")
    return PASS, detail


def stream_stats(msgs: list, stamps: list) -> dict:
    """Per OSC path: (sent_hz, received_hz, lost_fraction) between the first and the
    last packet of that path. The robot's packet counter (first value, an int that
    increases by one per packet) gives what was sent; sent_hz and lost_fraction are
    None when a path has no such counter, everything is None with < 2 packets."""
    by_path: dict = {}
    for (path, values), t in zip(msgs, stamps):
        by_path.setdefault(path, []).append((t, values))
    stats = {}
    for path, pkts in by_path.items():
        (t0, v0), (t1, v1) = pkts[0], pkts[-1]
        span = t1 - t0
        if len(pkts) < 2 or span <= 0:
            stats[path] = (None, None, None)
            continue
        received = (len(pkts) - 1) / span
        c0 = v0[0] if v0 else None
        c1 = v1[0] if v1 else None
        if isinstance(c0, int) and isinstance(c1, int) and c1 > c0:
            stats[path] = ((c1 - c0) / span, received,
                           max(0.0, 1.0 - (len(pkts) - 1) / (c1 - c0)))
        else:
            stats[path] = (None, received, None)
    return stats


def _drain(sock):
    """Discard datagrams already queued on sock. A robot keeps streaming to the
    last host it heard from, so packets pile up while the port is bound during the
    ping check — counting them would inflate the measured rates."""
    sock.settimeout(0)
    for _ in range(100000):
        try:
            sock.recvfrom(65535)
        except BlockingIOError:
            return
        except OSError:
            continue


def _firewall_hint(port: int) -> str:
    if sys.platform != 'win32':
        return (f"The robot is on the network but nothing reaches the simulator: a firewall "
                f"on this computer is probably blocking incoming UDP on port {port}.")
    hint = ("The robot is on the network but nothing reaches the simulator: Windows Firewall "
            "is probably blocking incoming UDP. Fix it in a terminal run as administrator:\n"
            f'netsh advfirewall firewall add rule name="LBP robot UDP {port}" dir=in '
            f'action=allow protocol=UDP localport={port}')
    if windows_public_network():
        hint += ("\nWindows has this network marked as Public, which blocks incoming traffic. "
                 "Alternatively switch it to Private: Settings → Network & internet → Wi-Fi → "
                 "(this network) → Network profile type.")
    return hint + "\nA VPN or an antivirus firewall can block it the same way."


class RobotCheck:
    LISTEN_S = 2.0
    PROBE_EVERY_S = 0.25
    PING_COUNT = 5
    SLOW_PING_MS = 100
    LOSS_WARN = 0.10

    def __init__(self, circuit, ping=ping_host):
        self.endpoints, self.problems = plan(circuit)
        self._ping = ping
        self.checks = [('config', 'Robot addresses are set up')]
        for ep in self.endpoints:
            if not ep.is_camera:
                self.checks.append((f'{ep.label}:port', f'Port {ep.port} is free on this computer'))
            self.checks.append((f'{ep.label}:ping', f'Robot {ep.host} answers ping'))
            self.checks.append((f'{ep.label}:reply', 'Robot sends data back'))
            for s in ep.sensors:
                what = 'camera images' if s.is_camera else s.path
                self.checks.append((f'{ep.label}:s:{s.name}', f"Sensor '{s.name}' receives {what}"))

    def run(self, emit, cancel: threading.Event):
        """Run every check, calling emit(key, status, detail) as each starts and ends."""
        emit('config', RUNNING, '')
        if not self.endpoints and not self.problems:
            emit('config', FAIL, "No sensor or motor layer has a Robot address. Set them in the "
                                 "Robot tab of the main window.")
            return
        problems = list(self.problems)
        if not any(ep.motor_paths for ep in self.endpoints) \
                and any(not ep.is_camera for ep in self.endpoints):
            problems.append("No motor layer has a Robot address. The robot only streams its "
                            "sensors after it receives a command, so they would stay at 0 Hz.")
        if problems:
            emit('config', FAIL, '\n'.join(problems))
        else:
            n_s = sum(len(ep.sensors) for ep in self.endpoints)
            n_m = sum(len(ep.motor_paths) for ep in self.endpoints)
            emit('config', PASS, f"{n_s} sensor(s) and {n_m} motor layer(s) at "
                                 f"{', '.join(ep.label for ep in self.endpoints)}")
        for ep in self.endpoints:
            if cancel.is_set():
                return
            self._run_endpoint(ep, emit, cancel)

    def _run_endpoint(self, ep: Endpoint, emit, cancel):
        sock = None
        try:
            if ep.is_camera:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            else:
                key = f'{ep.label}:port'
                emit(key, RUNNING, '')
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    sock.bind(('', ep.port))
                    emit(key, PASS, '')
                except OSError as e:
                    sock.close()
                    sock = None
                    emit(key, FAIL, (f"Another program is using UDP port {ep.port} ({e.strerror}). "
                                     f"Close Bonsai workflows, other simulator windows or scripts "
                                     f"that talk to the robot."))

            ping_ok = self._check_ping(ep, emit, cancel)
            if cancel.is_set():
                return

            key = f'{ep.label}:reply'
            if sock is None:
                emit(key, SKIP, f"Needs port {ep.port}.")
                self._skip_sensors(ep, emit, f"Needs port {ep.port}.")
                return
            emit(key, RUNNING, '')
            msgs, stamps, raw, n_pkts, bundles = self._listen(ep, sock, cancel)
            if cancel.is_set():
                return
            if n_pkts == 0:
                if ping_ok:
                    hint = _firewall_hint(ep.port) if not ep.is_camera else (
                        "The camera is on the network but sends no images — is its streaming "
                        "program running, and is a firewall blocking incoming UDP?")
                else:
                    hint = ("Nothing came back. Is the robot switched on, on the same WiFi as "
                            "this computer, and is its IP address right?")
                emit(key, FAIL, hint)
                self._skip_sensors(ep, emit, "Nothing received from the robot.")
                return
            if ep.is_camera:
                emit(key, PASS, f"{len(raw) / 1024:.0f} kB in {self.LISTEN_S:g} s")
            else:
                emit(key, *self._judge_streams(stream_stats(msgs, stamps), msgs, *bundles))

            for s in ep.sensors:
                key = f'{ep.label}:s:{s.name}'
                if s.is_camera:
                    ok = b'\xff\xd8' in raw
                    emit(key, PASS if ok else FAIL,
                         'JPEG frames arrive' if ok else 'Data arrives, but no JPEG frame in it.')
                else:
                    emit(key, *analyse_sensor(s, msgs, self.LISTEN_S))
        finally:
            if sock is not None:
                sock.close()

    def _judge_streams(self, stats: dict, msgs: list, n_bundles: int = 0,
                       n_bundled_msgs: int = 0):
        """Status + detail for the 'Robot sends data back' row: per path what the
        robot sent (from its packet counter) vs what arrived, warning on WiFi loss.
        Says so when the robot packs its messages into OSC bundles."""
        lines, lossy = [], []
        for path in sorted({p for p, _ in msgs}):
            sent, received, lost = stats.get(path, (None, None, None))
            if received is None:
                lines.append(f"{path}: 1 packet")
            elif sent is None:
                lines.append(f"{path}: {received:.0f} Hz")
            else:
                lines.append(f"{path}: robot sent {sent:.0f}/s, received {received:.0f}/s "
                             f"({lost:.0%} lost)")
                if lost > self.LOSS_WARN:
                    lossy.append(path)
        if n_bundles:
            lines.append(f"Sent as OSC bundles ({n_bundled_msgs / n_bundles:.1f} messages "
                         f"per packet)")
        detail = '\n'.join(lines)
        if lossy:
            return WARN, (f"{detail}\nPackets are being lost on the WiFi: check the signal "
                          f"strength, other robots and devices on the same network, and WiFi "
                          f"power saving on this computer.")
        return PASS, detail

    def _check_ping(self, ep, emit, cancel) -> bool:
        key = f'{ep.label}:ping'
        emit(key, RUNNING, '')
        times = self._ping(ep.host, self.PING_COUNT, cancel)
        if times is None:
            emit(key, SKIP, "Couldn't run ping on this computer.")
            return False
        if cancel.is_set():
            return False
        if not times:
            emit(key, FAIL, ("No answer. Is the robot switched on, on the same WiFi as this "
                             "computer, and is its IP address right? (A few networks block ping — "
                             "the next check is decisive.)"))
            return False
        detail = (f"{len(times)}/{self.PING_COUNT} replies, average {np.mean(times):.0f} ms, "
                  f"slowest {max(times):.0f} ms")
        if len(times) < self.PING_COUNT or max(times) >= self.SLOW_PING_MS:
            emit(key, WARN, (f"{detail}. Slow or missing replies make motor commands lag: check "
                             f"WiFi power saving on this computer, signal strength, or try "
                             f"restarting the computer."))
        else:
            emit(key, PASS, detail)
        return True

    def _listen(self, ep, sock, cancel):
        """Probe the robot and collect what it sends for LISTEN_S seconds.
        Returns (osc_msgs [(path, values)], arrival times parallel to osc_msgs,
        raw camera bytes, packet count, (OSC bundles among the packets, messages
        inside those bundles)).
        Messages unpacked from one bundle share its arrival time."""
        if ep.is_camera:
            probe = b'\x00'
        else:
            probe = _osc_build(ep.motor_paths[0] if ep.motor_paths else '/ping', 0, 0)
        try:
            robot_ip = socket.gethostbyname(ep.host)
        except OSError:
            robot_ip = ep.host
        _drain(sock)
        sock.settimeout(0.05)
        msgs, stamps, raw, n_pkts, n_bundles, n_bundled_msgs = [], [], b'', 0, 0, 0
        t_end = time.monotonic() + self.LISTEN_S
        next_probe = 0.0
        while time.monotonic() < t_end and not cancel.is_set():
            now = time.monotonic()
            if now >= next_probe:
                try:
                    sock.sendto(probe, (ep.host, ep.port))
                except OSError:
                    pass
                next_probe = now + self.PROBE_EVERY_S
            try:
                data, addr = sock.recvfrom(65535)
            except (socket.timeout, OSError):
                continue
            if addr[0] != robot_ip:
                continue
            n_pkts += 1
            if ep.is_camera:
                raw += data
            else:
                t = time.monotonic()
                unpacked = _osc_messages(data)
                if data.startswith(_BUNDLE_TAG):
                    n_bundles += 1
                    n_bundled_msgs += len(unpacked)
                for path, values, _ in unpacked:
                    msgs.append((path, values))
                    stamps.append(t)
        return msgs, stamps, raw, n_pkts, (n_bundles, n_bundled_msgs)

    def _skip_sensors(self, ep, emit, reason):
        for s in ep.sensors:
            emit(f'{ep.label}:s:{s.name}', SKIP, reason)
