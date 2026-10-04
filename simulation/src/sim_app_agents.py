"""
sim_app_agents.py — agent table and agent management for the 2D simulator.

_AgentsMixin owns the agent table (one row per group: color, name, brain, N)
and every add / remove / select / recolor action on agents. It only changes
the agent registry; the arena's robot disks and the MuJoCo model follow on
their own (SimController reacts to the registry's change callbacks), so no
code here touches the arena's robot items.
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSpinBox, QTableWidgetItem, QColorDialog,
)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt

from circuit_model import CircuitModel
from rigid_body import RigidBody
from brain_manager import BrainManager


class _AgentsMixin:

    def _new_agent_circuit(self):
        """A fresh circuit (root body only) and its BrainManager."""
        circuit = CircuitModel()
        circuit.bodies = [RigidBody('root', 'root', self.sim_cfg.body_radius)]
        return circuit, BrainManager(circuit, self.sim_cfg)

    # ── Agent table ───────────────────────────────────────────────────────────

    def _refresh_agent_list(self):
        """Rebuild the agent table from the registry's groups (one row per group)."""
        registry = self._sim_ctrl.registry
        groups = registry.groups_ordered()
        self._agent_table.blockSignals(True)
        self._agent_table.setRowCount(0)
        for i, group in enumerate(groups):
            self._agent_table.insertRow(i)

            # Col 0: color swatch button (centered in cell)
            btn = QPushButton()
            btn.setFixedSize(18, 18)
            btn.setStyleSheet(
                f"background-color: {group.color}; border: none; border-radius: 3px;"
            )
            btn.setToolTip("Click to change group color")
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda checked, idx=i: self._pick_agent_color(idx))
            swatch_container = QWidget()
            swatch_layout = QHBoxLayout(swatch_container)
            swatch_layout.addWidget(btn)
            swatch_layout.setAlignment(Qt.AlignCenter)
            swatch_layout.setContentsMargins(0, 0, 0, 0)
            self._agent_table.setCellWidget(i, 0, swatch_container)

            # Col 1: group name (editable)
            name_item = QTableWidgetItem(group.name)
            self._agent_table.setItem(i, 1, name_item)

            # Col 2: what the group runs (read-only) — the code brain's module,
            # or the circuit file for a network group.
            label = group.module or ""
            if self._is_network_module(group.module) and group.member_ids:
                first = registry.agent_by_id(group.member_ids[0])
                net = getattr(getattr(first, 'brain', None), 'network_file', '') or ''
                label = f"⬡ {net}" if net else "⬡ (no network)"
            brain_item = QTableWidgetItem(label)
            brain_item.setFlags(brain_item.flags() & ~Qt.ItemIsEditable)
            self._agent_table.setItem(i, 2, brain_item)

            # Col 3: N spinbox (setValue before connecting to avoid spurious signal)
            sb = QSpinBox()
            sb.setRange(1, 20)
            sb.setValue(len(group.member_ids))
            sb.valueChanged.connect(lambda v, g=i: self._on_agent_n_changed(g, v))
            self._agent_table.setCellWidget(i, 3, sb)

        sel_group_id = registry.group_of_agent(registry.selected_id)
        group_row = next((i for i, g in enumerate(groups) if g.id == sel_group_id), -1)
        self._agent_table.setCurrentCell(group_row, 1)
        self._agent_table.blockSignals(False)

    def _on_agent_name_changed(self, item):
        """Persist inline name edits back to the group data model."""
        if item.column() == 1:
            row = item.row()
            groups = self._sim_ctrl.registry.groups_ordered()
            if 0 <= row < len(groups):
                groups[row].name = item.text()

    def _on_agent_selected(self, row):
        """Select the first agent in the clicked group row."""
        groups = self._sim_ctrl.registry.groups_ordered()
        if row < 0 or row >= len(groups):
            return
        group = groups[row]
        if not group.member_ids:
            return
        self._select_agent(group.member_ids[0])

    # ── Add / remove ──────────────────────────────────────────────────────────

    def _add_agent(self, color=None, name=None):
        """Add a new agent group (n=1) with an optional brain load."""
        from arena_widget import _AGENT_COLORS
        registry = self._sim_ctrl.registry
        # The new group starts with what the currently selected group runs
        # (same code brain, or same network file).
        prev_group = self._selected_group()
        prev_module = prev_group.module if prev_group is not None else None
        prev_params = None
        if self._is_network_module(prev_module) and self.brain is not None:
            prev_params = {k: getattr(self.brain, k) for k in ('network_project', 'network_file')
                           if hasattr(self.brain, k)}
        circuit, brain_mgr = self._new_agent_circuit()
        n_groups = len(registry.groups_ordered())
        if color is None:
            color = _AGENT_COLORS[n_groups % len(_AGENT_COLORS)]
        if name is None:
            name = f'Group {n_groups + 1}'
        agent_id = registry.add_agent(circuit, brain_mgr,
                                      name=f"Agent {len(registry.agents) + 1}", color=color)
        registry.create_group(module=None, color=color, name=name, first_agent_id=agent_id)
        self._refresh_agent_list()
        # Select the new group and load the current brain into it
        self._select_agent(agent_id)
        module = prev_module or self._brain_combo.currentText()
        if module:
            self.load_brain(module, external_params=prev_params)

    def _remove_agent(self):
        """Remove the currently selected group and all its agents (min 1 agent total)."""
        registry = self._sim_ctrl.registry
        row = self._agent_table.currentRow()
        groups = registry.groups_ordered()
        if row < 0 or row >= len(groups):
            return
        group = groups[row]
        # Refuse if this would leave zero agents
        if len(registry.agents) - len(group.member_ids) < 1:
            return
        # Stable ids never invalidate each other, so removal order doesn't matter.
        for agent_id in list(group.member_ids):
            self._sim_ctrl.remove_agent(agent_id)   # also detaches it from the group
        registry.remove_group(group.id)
        self._refresh_agent_list()

    def _remove_selected_agent(self):
        """Remove the specific agent currently selected (Delete key), regardless
        of its position within its group; removes the group too if left empty."""
        registry = self._sim_ctrl.registry
        agent_id = registry.selected_id
        if agent_id is None or len(registry.agents) <= 1:
            return
        group_id = registry.group_of_agent(agent_id)
        self._sim_ctrl.remove_agent(agent_id)   # also detaches it from the group
        if group_id is not None:
            group = registry.get_group(group_id)
            if group is not None and not group.member_ids:
                registry.remove_group(group_id)
        self._refresh_agent_list()

    # ── Groups ────────────────────────────────────────────────────────────────

    def _pick_agent_color(self, group_idx):
        """Open a color dialog and apply the chosen color to all agents in the group."""
        registry = self._sim_ctrl.registry
        groups = registry.groups_ordered()
        if group_idx >= len(groups):
            return
        group = groups[group_idx]
        color = QColorDialog.getColor(QColor(group.color), self)
        if color.isValid():
            group.color = color.name()
            for agent_id in group.member_ids:
                agent = registry.agent_by_id(agent_id)
                if agent is not None:
                    agent.color = color.name()
            self._sim_ctrl.sync_robot_items()
            self._refresh_agent_list()

    def _on_agent_n_changed(self, group_idx, new_n):
        """Spinbox value changed: add or remove agents for the group."""
        registry = self._sim_ctrl.registry
        groups = registry.groups_ordered()
        if group_idx >= len(groups):
            return
        group = groups[group_idx]
        delta = new_n - len(group.member_ids)
        prev_sel_id = registry.selected_id
        if delta > 0:
            for _ in range(delta):
                self._add_agent_to_group(group_idx)
        elif delta < 0:
            for _ in range(-delta):
                self._remove_agent_from_group(group_idx)
        # Restore the agent that was selected before the resize, if it still
        # exists; otherwise fall back to the last remaining member of this
        # group (there's no positional index left to clamp, unlike before).
        if registry.agent_by_id(prev_sel_id) is not None:
            self._select_agent(prev_sel_id)
        elif group.member_ids:
            self._select_agent(group.member_ids[-1])
        elif registry.agents:
            self._select_agent(registry.agents[0].id)

    def _add_agent_to_group(self, group_idx):
        """Spawn one more agent for the given group and load its brain."""
        registry = self._sim_ctrl.registry
        group = registry.groups_ordered()[group_idx]
        circuit, brain_mgr = self._new_agent_circuit()
        agent_id = registry.add_agent(circuit, brain_mgr,
                                      name=f"Agent {len(registry.agents) + 1}", color=group.color)
        registry.add_agent_to_group(group.id, agent_id)
        if group.module:
            # Copy brain params from the group's first agent (member_ids[0] is
            # always the first agent added to this group). This carries over
            # network_file/network_project for DataBrain so load_brain can find and
            # load the same JSON network for the new agent.
            original_brain = registry.agent_by_id(group.member_ids[0]).brain
            params_copy = (
                {k: getattr(original_brain, k) for k in original_brain.get_param_metadata()}
                if original_brain is not None else None
            )
            # Temporarily select the new agent so load_brain targets it correctly
            registry.select_agent(agent_id)
            self.load_brain(group.module, external_params=params_copy)
            # Selection is restored by _on_agent_n_changed after all agents are added

    def _remove_agent_from_group(self, group_idx):
        """Remove the last agent from the group (refuses if it would leave zero total)."""
        registry = self._sim_ctrl.registry
        if len(registry.agents) <= 1:
            return
        group = registry.groups_ordered()[group_idx]
        if not group.member_ids:
            return
        self._sim_ctrl.remove_agent(group.member_ids[-1])   # also detaches it from the group

    # ── Selection ─────────────────────────────────────────────────────────────

    def _select_agent(self, agent_id):
        """Select a specific agent for oscilloscope / network viz, sync group table row."""
        registry = self._sim_ctrl.registry
        if registry.agent_by_id(agent_id) is None:
            return
        registry.select_agent(agent_id)          # the arena highlight follows
        if hasattr(self, '_editor'):
            self._editor._bot_pos = registry.bot_pos
        group_id = registry.group_of_agent(agent_id)
        group = registry.get_group(group_id) if group_id is not None else None
        groups = registry.groups_ordered()
        group_row = next((i for i, g in enumerate(groups) if g.id == group_id), -1)
        if group is not None:
            mod = group.module
            if mod and not self._is_network_module(mod):
                self._brain_combo.blockSignals(True)
                self._brain_combo.setCurrentText(mod)
                self._brain_combo.blockSignals(False)
        self._rebuild_brain_params()
        self._rebuild_channels()
        self._arena.setup_sensors(self.circuit.sensors, self._osc_ctrl.channel_colors)
        if self._net_viz:
            self._net_viz.build()
        # Sync table row to the owning group (without triggering _on_agent_selected)
        self._agent_table.blockSignals(True)
        self._agent_table.setCurrentCell(group_row, 1)
        self._agent_table.blockSignals(False)

    def _on_arena_click(self, x, y, btn):
        """Select the nearest agent when the user left-clicks the arena."""
        # WorldEditor already consumes left-clicks in these modes (placing a polygon
        # vertex / painting a wall); a click there must not also reselect the agent.
        if self._editor.draw_mode in ('object', 'wall_paint'):
            return
        agents = self._sim_ctrl.registry.agents
        if btn != 1 or len(agents) <= 1:
            return
        r_thresh = self.sim_cfg.body_radius * 2.5
        best_agent, best_dist = None, float('inf')
        for agent in agents:
            dx = agent.bot_pos[0] - x
            dy = agent.bot_pos[1] - y
            d = (dx * dx + dy * dy) ** 0.5
            if d < r_thresh and d < best_dist:
                best_agent, best_dist = agent, d
        if best_agent is not None:
            self._select_agent(best_agent.id)
