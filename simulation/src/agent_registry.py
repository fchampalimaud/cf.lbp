"""
agent_registry.py — the live agent/group registry for the 2D simulator.

Owns the list of agents and their table groups, keyed by stable ids that are
minted once and never reused or shifted. No knowledge of physics, networking,
or rendering — SimController wires the on_agents_changed/on_agent_removed
callbacks to keep MuJoCo and the network slot maps in sync without this class
needing to know either of those subsystems exist.
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class RobotAgent:
    """Per-agent state bundle."""
    id:        int                           # stable, permanent identity — assigned once at
                                              # creation, never reused, never shifted on removal
    circuit:   object                        # CircuitModel
    brain:     object                        # BaseBrain instance or None
    bot_pos:   list                          # [x, y, theta]
    trail_xy:  deque = field(default_factory=lambda: deque(maxlen=500))
    brain_mgr: object = None                 # BrainManager
    name:      str    = ""
    color:     str    = ""
    remote:    bool   = False                # True → motors come from a SimNetHost client


@dataclass
class AgentGroup:
    """A row in the agent table: a named/colored/brain-module bundle of agents.

    member_ids is append-only except for removal; member_ids[0] is always
    "the first agent added to this group" (used as the brain-param template
    when growing the group or saving a session)."""
    id:         int
    module:     object   # str brain module name, or None
    color:      str
    name:       str
    member_ids: list
    # Last code brain this group ran — restored when it switches back from
    # Network mode to Code brain mode.
    last_code_module: object = None


class AgentRegistry:
    """Owns _agents/_groups. Pure bookkeeping keyed by stable ids — no physics,
    networking, or rendering knowledge. Cross-subsystem effects (MuJoCo rebuild,
    network slot cleanup) happen via the on_agents_changed/on_agent_removed
    callback hooks, set by the owner (SimController) after construction."""

    def __init__(self, sim_cfg, circuit, brain_mgr):
        self.sim_cfg = sim_cfg

        # Monotonic counter minting RobotAgent.id — never reset (not even when
        # _agents is cleared for host mode), so a given id is never reused.
        self._next_agent_id = 0

        _agent0_id = self._next_agent_id
        self._agents = [  # list[RobotAgent]
            RobotAgent(
                id        = _agent0_id,
                circuit   = circuit,
                brain     = None,
                bot_pos   = [0.0, 0.0, 0.0],
                trail_xy  = deque(maxlen=500),
                brain_mgr = brain_mgr,
                name      = "Agent 1",
                color     = '#4a7fcb',
            )
        ]
        self._next_agent_id += 1
        self._selected_id = _agent0_id

        # Agent-table groups, keyed by a stable group id (never a list position).
        self._groups: dict = {}   # group_id → AgentGroup
        self._next_group_id = 0
        self.create_group(module=None, color='#4a7fcb', name='Group 1',
                           first_agent_id=_agent0_id)

        # Empty CircuitModel / BrainManager returned by self.circuit / self.brain_mgr
        # when _agents is empty (host mode)
        from circuit_model import CircuitModel as _CM
        from brain_manager import BrainManager as _BM
        self._host_circuit   = _CM()
        self._host_brain_mgr = _BM(self._host_circuit, sim_cfg)

        # Optional callbacks wired by the owner (SimController) to keep other
        # subsystems (MuJoCo, network slot maps) in sync without this class
        # needing to know they exist.
        self.on_agents_changed    = None   # callable() -> None, called after add/remove/clear
        self.on_agent_removed     = None   # callable(agent_id) -> None, called just before an agent is popped
        self.on_selection_changed = None   # callable() -> None, called when the selected agent changes

    # ── Agent access ─────────────────────────────────────────────────────────

    @property
    def agents(self):
        return self._agents

    @property
    def selected_id(self):
        return self._selected_id

    @property
    def agent(self):
        if not self._agents:
            return None
        a = self.agent_by_id(self._selected_id)
        return a if a is not None else self._agents[-1]

    @property
    def circuit(self):
        a = self.agent
        return a.circuit if a is not None else self._host_circuit

    @property
    def brain_mgr(self):
        a = self.agent
        return a.brain_mgr if a is not None else self._host_brain_mgr

    @property
    def brain(self):
        a = self.agent
        return a.brain if a is not None else None

    @brain.setter
    def brain(self, value):
        if self.agent is not None:
            self.agent.brain = value

    @property
    def bot_pos(self):
        a = self.agent
        return a.bot_pos if a is not None else [0.0, 0.0, 0.0]

    @property
    def trail_xy(self):
        a = self.agent
        return a.trail_xy if a is not None else deque(maxlen=500)

    def agent_by_id(self, agent_id):
        """Look up a RobotAgent by its stable id, or None if it no longer exists."""
        for a in self._agents:
            if a.id == agent_id:
                return a
        return None

    def index_of_agent(self, agent_id):
        """Current list position of agent_id, or None if it no longer exists.
        Only ever called at add/remove/select time — never per-agent-per-frame."""
        for i, a in enumerate(self._agents):
            if a.id == agent_id:
                return i
        return None

    # ── Multi-agent management ────────────────────────────────────────────────

    def add_agent(self, circuit, brain_mgr, name=None, color=None) -> int:
        """Append a new agent. Returns its stable id (NOT its list position —
        callers needing the position must capture len(self.agents) beforehand,
        since this always appends at the end)."""
        idx = len(self._agents)
        agent_id = self._next_agent_id
        offset_x = idx * 0.5
        agent = RobotAgent(
            id        = agent_id,
            circuit   = circuit,
            brain     = None,
            bot_pos   = [self.sim_cfg.init_x + offset_x, self.sim_cfg.init_y, 0.0],
            trail_xy  = deque(maxlen=500),
            brain_mgr = brain_mgr,
            name      = name  or f"Agent {idx + 1}",
            color     = color or '#4a7fcb',
        )
        self._next_agent_id += 1
        self._agents.append(agent)
        if self.on_agents_changed:
            self.on_agents_changed()
        return agent_id

    def remove_agent(self, agent_id: int, allow_empty: bool = False) -> bool:
        """Remove agent_id. Returns True if removed. allow_empty lets the
        caller permit emptying the registry entirely (e.g. host mode) —
        otherwise at least one agent is always kept."""
        idx = self.index_of_agent(agent_id)
        if idx is None:
            return False
        if not allow_empty and len(self._agents) <= 1:
            return False
        self._agents.pop(idx)
        # Selection is untouched unless the removed agent WAS the selected one
        # — no positional clamp, so removing a different agent can never
        # silently reselect the wrong one (see TODO.md history for the bug
        # this replaces: the old `min(self._selected, len-1)` clamp).
        if self._selected_id == agent_id:
            self._selected_id = self._agents[-1].id if self._agents else None
        if self.on_agent_removed:
            self.on_agent_removed(agent_id)
        self._detach_from_group(agent_id)
        if self.on_agents_changed:
            self.on_agents_changed()
        return True

    def select_agent(self, agent_id: int):
        if self.agent_by_id(agent_id) is not None and agent_id != self._selected_id:
            self._selected_id = agent_id
            if self.on_selection_changed:
                self.on_selection_changed()

    def clear(self):
        """Empty the registry entirely (agents, groups, selection) — used when
        transitioning into network host mode."""
        self._agents.clear()
        self._selected_id = None
        self._groups.clear()
        if self.on_agents_changed:
            self.on_agents_changed()

    # ── Agent-table group management ────────────────────────────────────────

    def create_group(self, module, color, name, first_agent_id) -> int:
        """Create a new group containing exactly first_agent_id. Returns its id."""
        group_id = self._next_group_id
        self._next_group_id += 1
        self._groups[group_id] = AgentGroup(
            id=group_id, module=module, color=color, name=name,
            member_ids=[first_agent_id],
        )
        return group_id

    def add_agent_to_group(self, group_id: int, agent_id: int):
        g = self._groups.get(group_id)
        if g is not None:
            g.member_ids.append(agent_id)

    def group_of_agent(self, agent_id):
        """group_id owning agent_id, or None if it isn't in any group."""
        for g in self._groups.values():
            if agent_id in g.member_ids:
                return g.id
        return None

    def get_group(self, group_id):
        return self._groups.get(group_id)

    def clear_groups(self):
        """Drop every group (the agents themselves stay)."""
        self._groups.clear()

    def groups_ordered(self):
        """Groups in creation order — matches the agent table's row order."""
        return list(self._groups.values())

    def remove_group(self, group_id):
        self._groups.pop(group_id, None)

    def _detach_from_group(self, agent_id):
        """Remove agent_id from whichever group currently owns it, if any."""
        gid = self.group_of_agent(agent_id)
        if gid is not None:
            self._groups[gid].member_ids.remove(agent_id)
