"""
sim_net_client.py — UDP client for remote brain-terminal mode.

The client connects to a SimNetHost, receives sensor readings per tick, injects
them into the local brain, runs the network forward pass, and sends motor
commands back. It can run embedded inside the simulator GUI (embedded mode) or
fully headless from the command line (headless mode).

Protocol: see sim_net_host.py.

--- Embedded usage (inside BraitenbergSimulator) ---

    client = SimNetClient(host='192.168.1.10', host_port=9001,
                          local_port=9002, slot=0)
    client.start()

    # Inside the main tick (called from sim_controller when in client mode):
    data, dt = client.pop_sensors()
    if data is not None:
        # Inject received sensor values into the local brain
        for name, values in data.items():
            setattr(brain, name, np.array(values, dtype=np.float32))
        mL, mR = brain.loop(dt)
        client.send_motors(mL, mR)

    # Check connection state for status bar:
    client.status   # 'disconnected' | 'connecting' | 'connected' | 'lost'

    client.stop()

--- Headless usage (python sim_net_client.py host:port slot network.json) ---

    See __main__ block at the bottom of this file.
"""

import json
import socket
import threading
import time
import zlib

import numpy as np


_MAX_PACKET    = 65507
_TIMEOUT_S     = 5.0   # seconds without a packet before status → 'lost'
_RETRY_S       = 1.0   # re-send register if still connecting
_HEARTBEAT_S   = 2.0   # interval between heartbeat packets while connected
SIM_NET_PORT   = 9001


class SimNetClient:
    """
    UDP client for remote brain-terminal operation.

    Thread model: one daemon receiver thread handles all incoming packets.
    The GUI tick reads pending sensor data via pop_sensors() and calls
    send_motors() — both are safe to call from any thread.
    """

    def __init__(self, host, host_port, local_port, name="", on_connect=None,
                 on_go=None, circuit_json=None):
        """
        Parameters
        ----------
        host        : str       Host simulator IP or hostname.
        host_port   : int       UDP port the host is listening on.
        local_port  : int       Local UDP port to bind (must be free).
        name        : str       Agent name sent to the host at registration.
        on_connect  : callable  Optional callback() fired (in recv thread) on ack.
        on_go       : callable  Optional callback() fired (in recv thread) when the
                                host sends a 'go' packet to start synchronized execution.
        """
        self._host       = host
        self._host_port  = int(host_port)
        self._local_port = int(local_port)
        self._slot       = 0    # assigned by host on ACK
        self._name       = name
        self._on_connect    = on_connect
        self._on_go         = on_go
        self._circuit_json  = circuit_json

        self._sock      = None
        self._thread    = None
        self._running   = False
        self._last_seen = 0.0
        self._last_reg  = 0.0      # monotonic time of last register attempt
        self._go_received    = False  # set when host sends 'go'
        self._last_heartbeat = 0.0

        # 'disconnected' | 'connecting' | 'connected' | 'ready' | 'running' | 'lost'
        self.status = 'disconnected'

        # Latest unread sensor packet; pop_sensors() swaps this to None
        self._pending      = None       # (data_dict, dt) or None
        self._pending_lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(('', self._local_port))
        self._sock.settimeout(0.5)
        self._running = True
        self.status   = 'connecting'
        self._thread  = threading.Thread(target=self._recv_loop, daemon=True,
                                         name='SimNetClient-rx')
        self._thread.start()
        self._send_register()

    def stop(self):
        self._running = False
        self._send_disconnect()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            self._sock.close()
            self._sock = None
        self.status = 'disconnected'

    # ── Interface for GUI tick ────────────────────────────────────────────────

    def pop_sensors(self):
        """
        Return and clear the latest sensor packet as (data_dict, dt).
        Returns (None, None) if no new packet has arrived since the last call,
        or if the host has not sent 'go' yet.
        data_dict maps sensor name (and lateralized halves) to list-of-floats.
        """
        if not self._go_received:
            return None, None
        with self._pending_lock:
            val = self._pending
            self._pending = None
        if val is None:
            return None, None
        return val

    def arm(self):
        """Signal the host that this client is ready to start.

        Sends a 'ready' packet and transitions to 'ready' status.  The tick
        loop will not process sensor data until the host sends back 'go'.
        """
        if self._sock is None or self.status not in ('connected', 'ready'):
            print(f'[SimNetClient] arm() called in invalid state: {self.status!r}')
            return
        self.status       = 'ready'
        self._go_received = False
        pkt = json.dumps(
            {'type': 'ready', 'slot': self._slot},
            separators=(',', ':'),
        ).encode('utf-8')
        if self._send(pkt):
            print(f'[SimNetClient] ready → host (slot {self._slot}), waiting for go...')

    def send_motors(self, mL, mR):
        """Send (mL, mR) motor commands to the host. Safe to call from any thread."""
        if self._sock is None:
            return
        pkt = json.dumps(
            {'type': 'motors', 'slot': self._slot, 'mL': float(mL), 'mR': float(mR)},
            separators=(',', ':'),
        ).encode('utf-8')
        self._send(pkt)

    # ── Headless blocking mode ────────────────────────────────────────────────

    def run_blocking(self, brain, circuit):
        """
        Block until stop() is called. On every received sensor packet:
          1. Inject sensor values into brain attributes.
          2. Call brain.loop(dt) to run the network forward pass.
          3. Send the returned (mL, mR) to the host.

        Parameters
        ----------
        brain   : BaseBrain   The brain object whose attributes receive sensor
                              values and whose loop() drives computation.
        circuit : object      CircuitModel (unused directly; kept for future use).
        """
        def _cb(data, dt):
            for name, values in data.items():
                setattr(brain, name, np.array(values, dtype=np.float32))
            try:
                result = brain.loop(dt)
                if isinstance(result, (tuple, list)) and len(result) >= 2:
                    mL, mR = float(result[0]), float(result[1])
                else:
                    mL = mR = 0.0
            except Exception as exc:
                print(f'[SimNetClient] brain.loop error: {exc}')
                mL = mR = 0.0
            self.send_motors(mL, mR)

        self._headless_cb = _cb
        try:
            while self._running:
                time.sleep(0.05)
        finally:
            self._headless_cb = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _send(self, pkt_bytes):
        """Compress and send a packet to the host. Returns True on success."""
        compressed = zlib.compress(pkt_bytes)
        try:
            self._sock.sendto(compressed, (self._host, self._host_port))
            return True
        except OSError as exc:
            print(f'[SimNetClient] send FAILED ({len(compressed)} bytes): {exc}')
            return False

    def _send_register(self):
        if self._sock is None:
            return
        msg = {'type': 'register', 'port': self._local_port, 'name': self._name}
        if self._circuit_json is not None:
            msg['circuit'] = self._circuit_json
        pkt = json.dumps(msg, separators=(',', ':')).encode('utf-8')
        if self._send(pkt):
            self._last_reg = time.monotonic()
            print(f'[SimNetClient] register → {self._host}:{self._host_port}'
                  f' (local port {self._local_port})')

    def _send_heartbeat(self):
        if self._sock is None:
            return
        pkt = json.dumps(
            {'type': 'heartbeat', 'slot': self._slot},
            separators=(',', ':'),
        ).encode('utf-8')
        if self._send(pkt):
            self._last_heartbeat = time.monotonic()

    def _send_disconnect(self):
        if self._sock is None:
            return
        pkt = json.dumps(
            {'type': 'disconnect', 'slot': self._slot},
            separators=(',', ':'),
        ).encode('utf-8')
        if self._send(pkt):
            print(f'[SimNetClient] disconnect sent (slot {self._slot})')

    def _recv_loop(self):
        self._headless_cb = None
        while self._running:
            try:
                raw, _addr = self._sock.recvfrom(_MAX_PACKET)
            except socket.timeout:
                now = time.monotonic()
                if self.status in ('connected', 'ready', 'running') \
                        and now - self._last_seen > _TIMEOUT_S:
                    print(f'[SimNetClient] connection lost (no packet for {_TIMEOUT_S:.0f}s)')
                    self.status = 'lost'
                # Re-send register if still connecting and retry interval elapsed
                if self.status in ('connecting', 'lost') and now - self._last_reg > _RETRY_S:
                    self._go_received = False
                    self._send_register()
                # Heartbeat while connected (motors only flow after 'go', so we need
                # a separate keep-alive so the host can detect ungraceful disconnects)
                if self.status in ('connected', 'ready', 'running') \
                        and now - self._last_heartbeat > _HEARTBEAT_S:
                    self._send_heartbeat()
                continue
            except OSError:
                break
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                pass  # uncompressed packet (older host or probe)
            try:
                pkt = json.loads(raw.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            self._dispatch(pkt)

    def _dispatch(self, pkt):
        t = pkt.get('type')
        if t == 'ack':
            self._slot      = int(pkt.get('slot', 0))
            self._last_seen = time.monotonic()
            self.status     = 'connected'
            print(f'[SimNetClient] ACK received — assigned slot {self._slot}')
            if self._on_connect:
                self._on_connect()
        elif t == 'go':
            self._go_received = True
            self.status       = 'running'
            self._last_seen   = time.monotonic()
            print(f'[SimNetClient] go received — starting (slot {self._slot})')
            if self._on_go:
                self._on_go()
        elif t == 'sensors':
            self._last_seen = time.monotonic()
            if self.status != 'connected':
                self.status = 'connected'
            data = pkt.get('data', {})
            dt   = float(pkt.get('dt', 0.02))
            cb   = getattr(self, '_headless_cb', None)
            if cb is not None:
                cb(data, dt)
            else:
                with self._pending_lock:
                    self._pending = (data, dt)


# ── Headless entry point ───────────────────────────────────────────────────────

def _headless_main():
    """
    python sim_net_client.py <host>:<port> <network.json> [local_port] [name]

    Connects to a running SimNetHost, loads the given network JSON, and runs
    the brain in headless mode (no GUI). The host auto-assigns a slot. Ctrl-C to quit.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from brain_serializer import load_network_json
    from brain_manager import BrainManager

    if len(sys.argv) < 3:
        print('Usage: sim_net_client.py host:port network.json [local_port] [name]')
        sys.exit(1)

    addr_str   = sys.argv[1]
    json_path  = sys.argv[2]
    local_port = int(sys.argv[3]) if len(sys.argv) > 3 else SIM_NET_PORT + 1
    name       = sys.argv[4] if len(sys.argv) > 4 else ''

    host, _, port_str = addr_str.rpartition(':')
    if not host:
        host = '127.0.0.1'
    host_port = int(port_str) if port_str else SIM_NET_PORT

    with open(json_path, 'r') as f:
        net_json = json.load(f)

    sensors, layers, connections, bodies, joints, motor_layer_name = \
        load_network_json(net_json)

    mgr    = BrainManager(sensors, layers, connections, motor_layer_name)
    brain  = mgr.make_brain()
    from sim_config import SimConfig
    sim_cfg = SimConfig()
    mgr.reset(brain, sim_cfg)

    class _FakeCircuit:
        pass
    circuit = _FakeCircuit()

    client = SimNetClient(host=host, host_port=host_port,
                          local_port=local_port, name=name)
    client.start()
    print(f'[SimNetClient] connecting to {host}:{host_port} '
          f'(local port {local_port}) …')

    try:
        client.run_blocking(brain, circuit)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        print(f'[SimNetClient] stopped (was slot {client._slot}).')


if __name__ == '__main__':
    _headless_main()
