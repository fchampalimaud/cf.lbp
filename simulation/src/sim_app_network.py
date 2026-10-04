"""
sim_app_network.py — Network tab: off / host / client modes, connecting, and the agents
created and removed for remote clients.
"""

from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QColor

from sim_constants import C


class _NetworkMixin:

    def _on_net_mode_off(self, checked):
        if not checked:
            return
        self._net_host_panel.setVisible(False)
        self._net_cli_panel.setVisible(False)
        self._sim_ctrl.disable_network_host()
        self._sim_ctrl.disconnect_from_host()

    def _on_net_mode_host(self, checked):
        if not checked:
            return
        self._net_cli_panel.setVisible(False)
        self._net_host_panel.setVisible(True)
        # Host starts with no local brain: enable_network_host() below clears
        # the agents/groups (the arena follows); refresh the table afterwards.
        port = self._net_host_port_spin.value()
        try:
            self._sim_ctrl.enable_network_host(port)
        except OSError as exc:
            from PySide6.QtWidgets import QMessageBox
            self._net_off_rb.setChecked(True)
            QMessageBox.warning(self, "Network Error",
                                f"Could not open host port {port}:\n{exc}")
            self._refresh_agent_list()
            return
        self._refresh_agent_list()
        import socket as _socket
        try:
            _s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            _s.connect(('8.8.8.8', 80))
            preferred_ip = _s.getsockname()[0]
            _s.close()
        except OSError:
            preferred_ip = '127.0.0.1'
        try:
            infos = _socket.getaddrinfo(_socket.gethostname(), None, _socket.AF_INET)
            all_ips = sorted({info[4][0] for info in infos
                              if not info[4][0].startswith('127.')})
        except OSError:
            all_ips = []
        if not all_ips:
            all_ips = [preferred_ip]
        elif preferred_ip not in all_ips:
            all_ips.insert(0, preferred_ip)
        self._net_host_ip_combo.clear()
        for ip in all_ips:
            self._net_host_ip_combo.addItem(ip)
        idx = self._net_host_ip_combo.findText(preferred_ip)
        if idx >= 0:
            self._net_host_ip_combo.setCurrentIndex(idx)

    def _on_net_mode_client(self, checked):
        if not checked:
            return
        self._net_host_panel.setVisible(False)
        self._net_cli_panel.setVisible(True)

    def _on_net_connect_clicked(self):
        if self._sim_ctrl.network.client is not None:
            self._sim_ctrl.disconnect_from_host()
            self._net_cli_conn_btn.setText("Connect")
        else:
            local_port = self._net_cli_lport_spin.value()
            from brain_serializer import serialize_network_json as _ser_net
            c = self._sim_ctrl.registry.circuit
            _circuit_json = _ser_net(
                c.sensors, c.layers, c.connections,
                set(), set(), {},
                bodies=c.bodies, joints=c.joints,
            )
            try:
                self._sim_ctrl.connect_to_host(
                    host         = self._net_cli_host_edit.text().strip(),
                    host_port    = self._net_cli_port_spin.value(),
                    local_port   = local_port,
                    name         = self._net_cli_name_edit.text().strip(),
                    circuit_json = _circuit_json,
                )
            except OSError as exc:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Network Error",
                                    f"Could not bind local port {local_port}:\n{exc}")
                return
            self._net_cli_conn_btn.setText("Disconnect")

    def _on_client_registered(self, info):
        """Main-thread handler: create a physics agent from the client's circuit JSON."""
        from brain_serializer import load_network_json
        from circuit_model import CircuitModel
        from rigid_body import RigidBody
        from brain_manager import BrainManager
        from arena_widget import _AGENT_COLORS

        slot_idx     = info['slot_idx']
        name         = info.get('name', '') or f'Remote {slot_idx}'
        circuit_json = info.get('circuit_json')

        new_circuit = CircuitModel()
        if circuit_json is not None:
            try:
                result = load_network_json(circuit_json)
                sensors, bodies, joints = result[0], result[6], result[7]
                new_circuit.sensors = sensors
                new_circuit.bodies  = bodies if bodies else \
                    [RigidBody('root', 'root', self.sim_cfg.body_radius)]
                new_circuit.joints  = joints
            except Exception as exc:
                print(f'[SimNet] Failed to parse circuit from slot {slot_idx}: {exc}')
                new_circuit.bodies = [RigidBody('root', 'root', self.sim_cfg.body_radius)]
        else:
            new_circuit.bodies = [RigidBody('root', 'root', self.sim_cfg.body_radius)]

        new_brain_mgr = BrainManager(new_circuit, self.sim_cfg)
        color = _AGENT_COLORS[len(self._sim_ctrl.registry.groups_ordered()) % len(_AGENT_COLORS)]
        pos = len(self._sim_ctrl.registry.agents)
        agent_id = self._sim_ctrl.registry.add_agent(new_circuit, new_brain_mgr,
                                            name=name, color=color)

        # Minimal brain for the host side: just a sensor-value container.
        # loop() is never called because motor_override is always set from the client.
        class _RemoteBrain:
            def setup(self): pass
            def loop(self, _dt): return 0.0, 0.0
        brain = _RemoteBrain()
        brain.setup()
        self._sim_ctrl.registry.agents[pos].brain  = brain
        self._sim_ctrl.registry.agents[pos].remote = True

        # Register slot↔agent-id mapping so _tick() routes motors/sensors correctly.
        self._sim_ctrl.network.assign_slot(slot_idx, agent_id)

        self._sim_ctrl.registry.create_group(module=None, color=color, name=name,
                                     first_agent_id=agent_id)
        self._refresh_agent_list()
        print(f'[SimNet] slot {slot_idx} → agent {agent_id} created ({name})')

    def _on_remote_agent_removed(self, agent_id):
        """Main-thread handler: remove the agent and group for a disconnected client
        (the arena follows). agent_id is the disconnected agent's stable id
        (sig_agent_removed's payload)."""
        if self._sim_ctrl.registry.index_of_agent(agent_id) is None:
            return
        group_id = self._sim_ctrl.registry.group_of_agent(agent_id)
        self._sim_ctrl.remove_agent(agent_id)   # also detaches it from its group
        if group_id is not None:
            g = self._sim_ctrl.registry.get_group(group_id)
            if g is not None and not g.member_ids:
                self._sim_ctrl.registry.remove_group(group_id)   # remote groups are always singleton
        self._refresh_agent_list()
        print(f'[SimNet] agent {agent_id} removed')

    def _update_net_status(self):
        """Refresh network status labels (called at 1 Hz)."""
        host = self._sim_ctrl.network.host
        if host is not None:
            # User-configurable via the Timeout spinbox next to Frame rate
            # (Network tab); default matches the client's 2s heartbeat interval.
            host.prune_stale(timeout_s=self._sim_ctrl.network.disconnect_timeout)
            slots = host.connected_slots()
            self._net_client_list.clear()
            agent_list_dirty = False
            if slots:
                for idx in sorted(slots):
                    s = host.get_slot(idx)
                    if s is None:
                        continue
                    idle = s.idle_seconds
                    idle_txt = f"{idle:.1f}s" if idle < 10 else f"{idle:.0f}s"
                    # Sync the slot name into the agent and agent_groups if it changed.
                    # Use slot→agent-id mapping (slot_idx ≠ agent position in general),
                    # then translate the id to its current list position.
                    agent_id = self._sim_ctrl.network.agent_for_slot(idx)
                    pos = self._sim_ctrl.registry.index_of_agent(agent_id) if agent_id is not None else None
                    if s.name and pos is not None:
                        agent = self._sim_ctrl.registry.agents[pos]
                        if agent.name != s.name:
                            agent.name = s.name
                            group_id = self._sim_ctrl.registry.group_of_agent(agent_id)
                            if group_id is not None:
                                self._sim_ctrl.registry.get_group(group_id).name = s.name
                            agent_list_dirty = True
                    display_name = s.name or (
                        self._sim_ctrl.registry.agents[pos].name
                        if pos is not None
                        else f"slot {idx}"
                    )
                    state_tag = " ✓ready" if s.ready else ""
                    heartbeat_ok = idle < 4.0   # 2× the 2 s heartbeat interval
                    hb_dot = "●"
                    hb_color = C['success'] if heartbeat_ok else C['warning']
                    self._net_client_list.addItem(
                        f"{hb_dot} slot {idx}  {display_name}  —"
                        f"  {s.client_addr}:{s.client_port}"
                        f"  ↓{s.send_hz:.0f}Hz ↑{s.recv_hz:.0f}Hz"
                        f"  idle {idle_txt}{state_tag}"
                    )
                    item = self._net_client_list.item(self._net_client_list.count() - 1)
                    item.setForeground(QColor(hb_color))
            else:
                self._net_client_list.addItem("(no clients connected)")
                item = self._net_client_list.item(0)
                item.setForeground(QColor(C['muted']))
            if agent_list_dirty:
                self._refresh_agent_list()

        client = self._sim_ctrl.network.client
        if client is not None:
            status = client.status
            colors = {
                'connecting': C['warning'],
                'connected':  C['success'],
                'ready':      C['warning'],
                'running':    C['success'],
                'lost':       C['danger'],
            }
            color = colors.get(status, C['muted'])
            self._net_cli_status_lbl.setText(status)
            self._net_cli_status_lbl.setStyleSheet(f"color:{color};")
            if status == 'connected':
                self._net_cli_conn_btn.setText("Disconnect")
