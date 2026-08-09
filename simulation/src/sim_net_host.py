"""
sim_net_host.py — UDP host for multi-computer simulation.

One UDP socket, one receiver thread. The host sends sensor data to each
connected remote client after every physics tick, and caches the most
recent motor commands from each client for use in the next tick.

Protocol (all packets UTF-8 JSON, max ~65 KB):

  Client → Host:
    {"type": "register", "slot": <int>, "port": <int>}
    {"type": "motors",   "slot": <int>, "mL": <float>, "mR": <float>}

  Host → Client:
    {"type": "ack",     "slot": <int>}
    {"type": "sensors", "slot": <int>, "dt": <float>,
     "data": {"sensor_name": [v0, v1, ...], "sensor_name_L": [...], ...}}

The host runs at its own pace. Clients that lag just reuse their stale cached
motor command, which is biologically meaningful (a sluggish brain = sluggish
body). No blocking on remote replies.
"""

import json
import collections
import socket
import threading
import time
import zlib


_MAX_PACKET  = 65507
SIM_NET_PORT = 9001
_RATE_WINDOW = 60   # keep timestamps of last N events for Hz estimation


class _RemoteSlot:
    """State for one connected remote client."""

    def __init__(self, slot, client_addr, client_port, name=""):
        self.slot        = slot
        self.client_addr = client_addr
        self.client_port = client_port
        self.name        = name
        self.mL          = 0.0
        self.mR          = 0.0
        self.ready       = False   # True after client sends {"type":"ready"}
        self._lock       = threading.Lock()
        self.last_seen   = time.monotonic()
        self._send_times = collections.deque(maxlen=_RATE_WINDOW)
        self._recv_times = collections.deque(maxlen=_RATE_WINDOW)

    def touch(self):
        with self._lock:
            self.last_seen = time.monotonic()

    def record_send(self):
        self._send_times.append(time.monotonic())

    def set_motors(self, mL, mR):
        with self._lock:
            self.mL        = float(mL)
            self.mR        = float(mR)
            self.last_seen = time.monotonic()
        self._recv_times.append(time.monotonic())

    def get_motors(self):
        with self._lock:
            return self.mL, self.mR

    @property
    def idle_seconds(self):
        with self._lock:
            return time.monotonic() - self.last_seen

    @property
    def send_hz(self):
        return _hz(self._send_times)

    @property
    def recv_hz(self):
        return _hz(self._recv_times)


def _hz(times):
    if len(times) < 2:
        return 0.0
    elapsed = times[-1] - times[0]
    return (len(times) - 1) / elapsed if elapsed > 0 else 0.0


class SimNetHost:
    """
    UDP host: binds one port, routes motor packets from remote clients to their
    per-slot motor cache, and provides a send_sensors() call for the tick loop.

    Typical usage in sim_controller._tick() for a remote agent at index i:

        slot = self._net_host.get_slot(i)
        if slot is not None:
            mL, mR = slot.get_motors()
            raw = tick_physics(..., motor_override=(mL, mR))
            data = _build_sensor_data(agent.brain, agent.circuit.sensors)
            self._net_host.send_sensors(i, data, self.sim_cfg.dt)
    """

    def __init__(self, port=SIM_NET_PORT, on_register=None, on_ready=None,
                 on_disconnect=None):
        """
        Parameters
        ----------
        port        : int       UDP port to bind.
        on_register : callable  Optional callback(slot_idx, name) fired when a new
                                client registers. Called from the receiver thread.
        on_ready    : callable  Optional callback(slot_idx) fired when a client
                                sends a 'ready' packet. Called from the receiver thread.

        Note: on_register is called as on_register(slot_idx, name, circuit_json)
        where circuit_json is the dict the client sent (may be None for old clients).
        """
        self._port          = port
        self._on_register   = on_register
        self._on_ready      = on_ready
        self._on_disconnect = on_disconnect
        self._slots         = {}          # slot_idx → _RemoteSlot
        self._slots_lock  = threading.Lock()
        self._sock        = None
        self._thread      = None
        self._running     = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(('', self._port))
        self._sock.settimeout(0.5)
        self._running = True
        self._thread  = threading.Thread(target=self._recv_loop, daemon=True,
                                         name='SimNetHost-rx')
        self._thread.start()
        print(f'[SimNetHost] listening on port {self._port}')

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            self._sock.close()
            self._sock = None
        print('[SimNetHost] stopped')

    # ── Query interface ───────────────────────────────────────────────────────

    @property
    def port(self):
        return self._port

    def get_slot(self, slot_idx):
        """Return the _RemoteSlot for this agent index, or None if not connected."""
        with self._slots_lock:
            return self._slots.get(slot_idx)

    def connected_slots(self):
        """Return list of agent indices that have connected clients."""
        with self._slots_lock:
            return list(self._slots.keys())

    def ready_slots(self):
        """Return list of slot indices whose clients have sent a 'ready' packet."""
        with self._slots_lock:
            return [idx for idx, s in self._slots.items() if s.ready]

    def prune_stale(self, timeout_s=5.0):
        """Remove slots that have not sent any packet within *timeout_s* seconds.

        Returns the list of slot indices that were removed so the caller can
        fire any necessary cleanup (e.g. clearing the remote flag on agents).
        """
        with self._slots_lock:
            stale = [idx for idx, s in self._slots.items()
                     if s.idle_seconds > timeout_s]
            for idx in stale:
                del self._slots[idx]
        for idx in stale:
            print(f'[SimNetHost] slot {idx} pruned (idle >{timeout_s:.0f}s)')
            if self._on_disconnect:
                self._on_disconnect(idx)
        return stale

    def send_go_to_slot(self, slot_idx):
        """Send a 'go' packet to a single slot (for late-joining clients)."""
        with self._slots_lock:
            s = self._slots.get(slot_idx)
        if s is None or self._sock is None:
            return
        go_pkt = json.dumps({'type': 'go'}, separators=(',', ':')).encode('utf-8')
        if self._send(go_pkt, (s.client_addr, s.client_port)):
            print(f'[SimNetHost] go → slot {slot_idx} ({s.client_addr}:{s.client_port})')

    def broadcast_go(self):
        """Send a 'go' packet to every connected client, starting simultaneous execution."""
        go_pkt = json.dumps({'type': 'go'}, separators=(',', ':')).encode('utf-8')
        with self._slots_lock:
            targets = [(idx, s.client_addr, s.client_port)
                       for idx, s in self._slots.items()]
        if self._sock is None:
            return
        if not targets:
            print('[SimNetHost] broadcast go — no clients connected')
            return
        for idx, host_addr, port in targets:
            if self._send(go_pkt, (host_addr, port)):
                print(f'[SimNetHost] go → slot {idx} ({host_addr}:{port})')

    # ── Internal send helper ──────────────────────────────────────────────────

    def _send(self, pkt_bytes, addr):
        """Compress and send a packet. Returns True on success."""
        compressed = zlib.compress(pkt_bytes)
        try:
            self._sock.sendto(compressed, addr)
            return True
        except OSError as exc:
            print(f'[SimNetHost] send FAILED ({len(compressed)} bytes compressed '
                  f'from {len(pkt_bytes)}): {exc}')
            return False

    # ── Called from tick loop ─────────────────────────────────────────────────

    def send_sensors(self, slot_idx, data, dt):
        """
        Send sensor data to the client registered for slot_idx.

        Parameters
        ----------
        slot_idx : int    Agent/slot index.
        data     : dict   Sensor name → list of floats. Include lateralized
                          halves (name+'_L', name+'_R') when the sensor is
                          lateralized so the client can inject them directly.
        dt       : float  Simulation timestep used for this tick.
        """
        with self._slots_lock:
            s = self._slots.get(slot_idx)
        if s is None or self._sock is None:
            return
        pkt = json.dumps(
            {'type': 'sensors', 'slot': slot_idx, 'dt': float(dt), 'data': data},
            separators=(',', ':'),
        ).encode('utf-8')
        first_send = len(s._send_times) == 0
        if self._send(pkt, (s.client_addr, s.client_port)):
            s.record_send()
            if first_send:
                compressed_size = len(zlib.compress(pkt))
                print(f'[SimNetHost] first sensor packet → slot {slot_idx} '
                      f'({len(pkt)} bytes → {compressed_size} compressed)')

    # ── Receiver thread ───────────────────────────────────────────────────────

    def _recv_loop(self):
        while self._running:
            try:
                raw, addr = self._sock.recvfrom(_MAX_PACKET)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                pass  # uncompressed packet (older client or probe)
            try:
                pkt = json.loads(raw.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            self._dispatch(pkt, addr)

    def _dispatch(self, pkt, addr):
        t = pkt.get('type')
        if t == 'register':
            port         = int(pkt.get('port', addr[1]))
            name         = str(pkt.get('name', ''))
            circuit_json = pkt.get('circuit')   # dict or None
            is_new = False
            with self._slots_lock:
                # Reuse existing slot if same (addr, port) is already registered
                existing = next(
                    (s for s in self._slots.values()
                     if s.client_addr == addr[0] and s.client_port == port),
                    None,
                )
                if existing is not None:
                    assigned = existing.slot
                    if name:
                        existing.name = name
                else:
                    # Auto-assign the lowest free slot index
                    assigned = 0
                    while assigned in self._slots:
                        assigned += 1
                    self._slots[assigned] = _RemoteSlot(assigned, addr[0], port, name)
                    is_new = True
            ack = json.dumps({'type': 'ack', 'slot': assigned},
                             separators=(',', ':')).encode('utf-8')
            self._send(ack, (addr[0], port))
            if is_new:
                label = f'"{name}" ' if name else ''
                print(f'[SimNetHost] slot {assigned} registered: {label}{addr[0]}:{port}')
                if self._on_register:
                    self._on_register(assigned, name, circuit_json)
            else:
                print(f'[SimNetHost] slot {assigned} re-registered (idempotent)')
        elif t == 'heartbeat':
            slot = int(pkt.get('slot', -1))
            with self._slots_lock:
                s = self._slots.get(slot)
            if s is not None:
                s.touch()
            else:
                print(f'[SimNetHost] heartbeat from unknown slot {slot} — ignored')
        elif t == 'disconnect':
            slot = int(pkt.get('slot', -1))
            with self._slots_lock:
                removed = self._slots.pop(slot, None)
            if removed is not None:
                print(f'[SimNetHost] slot {slot} disconnected gracefully')
                if self._on_disconnect:
                    self._on_disconnect(slot)
        elif t == 'ready':
            slot = int(pkt.get('slot', 0))
            with self._slots_lock:
                s = self._slots.get(slot)
            if s is not None:
                s.ready = True
                print(f'[SimNetHost] slot {slot} ready')
                if self._on_ready:
                    self._on_ready(slot)
            else:
                print(f'[SimNetHost] ready from unknown slot {slot} — ignored')
        elif t == 'motors':
            slot = int(pkt.get('slot', 0))
            with self._slots_lock:
                s = self._slots.get(slot)
            if s is not None:
                s.set_motors(pkt.get('mL', 0.0), pkt.get('mR', 0.0))
