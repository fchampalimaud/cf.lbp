"""
network_controller.py — network host/client UDP protocol for the 2D simulator.

Owns the SimNetHost/SimNetClient lifecycle and the slot↔agent-id mapping.
Two things it deliberately does NOT own, because they're core simulation-loop
state, not networking state: the `running` flag and the QTimer — see
SimController._on_go_received, which is networking-triggered but mutates core
run-state, and SimController.get_running wiring below, which lets
_on_remote_ready check whether the loop is running without this class storing
that flag itself.
"""

import time

import numpy as np
from PySide6.QtCore import QObject, Signal

from lateral import half_names


def _build_sensor_data(brain, sensors):
    """Build a sensor-data dict suitable for SimNetHost.send_sensors().

    Returns {sensor_name: [v0, v1, ...]} including lateralized _L/_R halves.
    """
    data = {}
    for s in sensors:
        arr = np.atleast_1d(getattr(brain, s.name, []))
        data[s.name] = arr.tolist()
        if s.is_lateralized():   # camera halves and joint-pair halves alike
            for half in half_names(s.name):
                data[half] = np.atleast_1d(getattr(brain, half, [])).tolist()
    return data


class NetworkController(QObject):
    """Manages the UDP host/client protocol. slot_to_agent_id/agent_id_to_slot
    are keyed by RobotAgent.id (not list position) so entries never need to be
    shifted when an unrelated agent is added/removed elsewhere in the list."""

    sig_client_ready      = Signal(int)        # slot_idx — a remote client sent 'ready'
    sig_client_disconnect = Signal(int)        # slot_idx — a remote client disconnected
    sig_go_received       = Signal()           # host sent 'go'; client should start ticking
    sig_client_registered = Signal(object)     # dict {slot_idx, name, circuit_json}
    sig_agent_removed     = Signal(int)        # agent_id (stable id, not position) — fired just before removal

    def __init__(self, parent=None):
        super().__init__(parent)

        self._net_host       = None   # type: SimNetHost | None
        self._net_client     = None   # type: SimNetClient | None
        self.frame_rate         = 50    # target Hz for sensor sends to clients
        self.disconnect_timeout = 2.0   # seconds idle before a stale client is pruned
        self._net_send_times: dict = {}  # slot_idx → last wall-clock send time
        self._net_send_dts:   dict = {}  # slot_idx → accumulated sim dt since last send

        # Bidirectional mapping between network slot indices and stable agent ids.
        self._slot_to_agent_id: dict = {}   # slot_idx → agent_id
        self._agent_id_to_slot: dict = {}   # agent_id → slot_idx

        # Wired by SimController after construction so _on_remote_ready can
        # check the core run-state without this class owning it.
        self.get_running = None   # callable() -> bool

    @property
    def host(self):
        return self._net_host

    @property
    def client(self):
        return self._net_client

    @property
    def is_hosting(self):
        return self._net_host is not None

    @property
    def is_client(self):
        return self._net_client is not None

    # ── Slot ↔ agent-id mapping ──────────────────────────────────────────────

    def slot_for_agent(self, agent_id):
        return self._agent_id_to_slot.get(agent_id)

    def agent_for_slot(self, slot_idx):
        return self._slot_to_agent_id.get(slot_idx)

    def assign_slot(self, slot_idx, agent_id):
        """Record that remote slot *slot_idx* drives agent *agent_id*."""
        self._slot_to_agent_id[slot_idx] = agent_id
        self._agent_id_to_slot[agent_id] = slot_idx

    def forget_agent(self, agent_id):
        """Drop agent_id's slot mapping, if any. Wired as AgentRegistry's
        on_agent_removed callback."""
        slot = self._agent_id_to_slot.pop(agent_id, None)
        if slot is not None:
            self._slot_to_agent_id.pop(slot, None)

    def resolve_remote_motors(self, agent_id):
        """Motor command from a connected remote client for agent_id, or None
        if not hosting / no client registered in that slot. Extracted from
        _tick's old _motor_for closure so the host-slot lookup isn't inlined
        into the physics loop."""
        if self._net_host is None:
            return None
        slot_idx = self._agent_id_to_slot.get(agent_id)
        if slot_idx is None:
            return None
        slot = self._net_host.get_slot(slot_idx)
        return slot.get_motors() if slot is not None else (0.0, 0.0)

    # ── Host lifecycle ───────────────────────────────────────────────────────

    def enable_host(self, port=None):
        """Start a SimNetHost on *port* (default SIM_NET_PORT). No-op if already running."""
        if self._net_host is not None:
            return
        self._slot_to_agent_id.clear()
        self._agent_id_to_slot.clear()
        from sim_net_host import SimNetHost, SIM_NET_PORT as _DEFAULT_PORT
        host = SimNetHost(
            port=port or _DEFAULT_PORT,
            on_register=self._on_remote_register,
            on_ready=self._on_remote_ready,
            on_disconnect=self._on_remote_disconnect,
        )
        try:
            host.start()
        except OSError as exc:
            print(f"[SimNetHost] Failed to bind port {port or _DEFAULT_PORT}: {exc}")
            raise
        self._net_host = host
        self._net_send_times.clear()
        self._net_send_dts.clear()

    def disable_host(self):
        """Stop the SimNetHost. Caller (SimController) resets agent.remote
        flags afterward — that needs the agent registry, which this class
        doesn't own."""
        if self._net_host is not None:
            self._net_host.stop()
            self._net_host = None
        self._net_send_times.clear()
        self._net_send_dts.clear()

    def host_maybe_send(self, slot_idx, brain, sensors, dt):
        """Send sensor packet to remote slot only if the frame-rate interval has elapsed.

        Accumulates sim dt across skipped ticks and sends the total in the next packet,
        so client dynamics integrate correctly regardless of subsampling ratio. dt is
        the caller's sim_cfg.dt — this class doesn't own sim_cfg.
        """
        now = time.monotonic()
        interval = 1.0 / max(1, self.frame_rate)
        self._net_send_dts[slot_idx] = (
            self._net_send_dts.get(slot_idx, 0.0) + dt
        )
        if now - self._net_send_times.get(slot_idx, 0.0) < interval:
            return
        dt_acc = self._net_send_dts.get(slot_idx, dt)
        self._net_host.send_sensors(slot_idx, _build_sensor_data(brain, sensors), dt_acc)
        self._net_send_times[slot_idx] = now
        self._net_send_dts[slot_idx] = 0.0

    # ── Client lifecycle ─────────────────────────────────────────────────────

    def connect_to_host(self, host, host_port, local_port, name="", circuit_json=None):
        """Connect to a remote SimNetHost as a brain terminal."""
        if self._net_client is not None:
            self._net_client.stop()
            self._net_client = None
        from sim_net_client import SimNetClient
        client = SimNetClient(
            host=host,
            host_port=host_port,
            local_port=local_port,
            name=name,
            on_go=lambda: self.sig_go_received.emit(),
            circuit_json=circuit_json,
        )
        try:
            client.start()
        except OSError as exc:
            print(f"[SimNetClient] Failed to bind local port {local_port}: {exc}")
            raise
        self._net_client = client

    def disconnect_from_host(self):
        """Disconnect from the remote host."""
        if self._net_client is not None:
            self._net_client.stop()
            self._net_client = None

    # ── Host-thread callbacks (run on the SimNetHost receiver thread) ───────

    def _on_remote_register(self, slot_idx, name="", circuit_json=None):
        """Called from the SimNetHost receiver thread when a client registers.

        Emits sig_client_registered so the main thread (LBPSimulator) can create
        the agent from the circuit JSON and update the UI.
        """
        self.sig_client_registered.emit({
            'slot_idx':     slot_idx,
            'name':         name,
            'circuit_json': circuit_json,
        })

    def _on_remote_ready(self, slot_idx):
        """Called from the SimNetHost receiver thread when a client sends 'ready'."""
        self.sig_client_ready.emit(slot_idx)
        # If the host sim is already running, send 'go' directly to this late joiner.
        if self.get_running is not None and self.get_running() and self._net_host is not None:
            self._net_host.send_go_to_slot(slot_idx)

    def _on_remote_disconnect(self, slot_idx):
        """Called from the SimNetHost receiver thread when a client disconnects.

        Emits sig_agent_removed (carrying the disconnected agent's stable id, not
        its list position) so the main thread can remove the agent from the UI
        before calling remove_agent().
        """
        agent_id = self._slot_to_agent_id.get(slot_idx)
        if agent_id is not None:
            self.sig_agent_removed.emit(agent_id)
        self.sig_client_disconnect.emit(slot_idx)

    def close(self):
        if self._net_host is not None:
            self._net_host.stop()
            self._net_host = None
        if self._net_client is not None:
            self._net_client.stop()
            self._net_client = None
