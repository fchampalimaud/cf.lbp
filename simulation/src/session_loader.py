"""
session_loader.py — build a Simulation from a saved session file, without the GUI.

The GUI's session loading (sim_app_session._load_session) uses the same
pieces — apply_session_world, resolve_mounted_patches and
BrainManager.install_brain — and only adds widget updates on top.

Files a session names (its networks, brain_<Name>.json parameter files) are
found through data_paths — in the user's folder first, then the built-in one.
Both brains/ folders are put on sys.path automatically so brain modules can
be imported.
"""

import os
import sys

from session_io import load_session, brain_from_file

import data_paths


def apply_session_world(d, world, sim_cfg):
    """Apply a session dict's sim params and world contents (arena shape,
    patches, objects, walls, floor texture, sky) to world / sim_cfg."""
    for k, v in d.get('sim_params', {}).items():
        if hasattr(sim_cfg, k):
            setattr(sim_cfg, k, v)
    if 'arena_round' in d:
        world.arena_round = bool(d['arena_round'])
    world.patches       = d.get('patches', [])
    world.objects       = d.get('objects', [])
    world.walls         = d.get('walls', [])
    world.floor_texture = d.get('floor_texture')
    world.sky           = d.get('sky', {'enabled': False, 'angle': 0.0})


def session_groups(d):
    """The session's agent groups as dicts (module, name, color, n, brain_params).
    Older single-agent sessions (no 'agents' list) become one group."""
    saved = d.get('agents') or [{'name': 'Group 1', 'color': None, 'n': 1, **d}]
    groups = []
    for i, sg in enumerate(saved):
        module, params = brain_from_file(sg)   # code brain or network, new or old format
        groups.append({'module': module, 'name': sg.get('name', f'Group {i + 1}'),
                       'color': sg.get('color'), 'n': max(1, int(sg.get('n', 1))),
                       'brain_params': params})
    return groups


def resolve_mounted_patches(world, agents):
    """Sessions store a mounted patch's carrier as a positional agent index;
    translate it into that agent's (freshly minted) stable id."""
    for p in world.patches:
        idx = p.get('mounted_on')
        if idx is None:
            continue
        if isinstance(idx, int) and 0 <= idx < len(agents):
            p['mounted_on'] = agents[idx].id
        else:
            p.pop('mounted_on', None)


def build_simulation(path, start_engine=True):
    """Load the session at *path* (or an already-read session dict) into a
    new, reset Simulation.

    start_engine=False skips creating the MuJoCo engine (e.g. to inspect the
    loaded agents without stepping). Raises RuntimeError if MuJoCo fails."""
    from sim_config import SimConfig
    from world import World
    from circuit_model import CircuitModel
    from rigid_body import RigidBody
    from brain_manager import BrainManager
    from agent_registry import AgentRegistry
    from simulation import Simulation

    for folder in reversed(data_paths.search_dirs('brains')):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    d = load_session(path) if isinstance(path, (str, os.PathLike)) else path
    sim_cfg = SimConfig()
    world = World(sim_cfg)
    apply_session_world(d, world, sim_cfg)

    def _new_circuit():
        c = CircuitModel()
        c.bodies = [RigidBody('root', 'root', sim_cfg.body_radius)]
        return c

    circuit0 = _new_circuit()
    registry = AgentRegistry(sim_cfg, circuit0, BrainManager(circuit0, sim_cfg))
    agent0 = registry.agents[0]
    group0 = registry.get_group(registry.group_of_agent(agent0.id))

    for gi, g in enumerate(session_groups(d)):
        first_brain = None
        for k in range(g['n']):
            if gi == 0 and k == 0:
                agent = agent0
                group0.module, group0.name = g['module'] or None, g['name']
                if g['color']:
                    group0.color = agent.color = g['color']
                group_id = group0.id
            else:
                circuit = _new_circuit()
                agent_id = registry.add_agent(circuit, BrainManager(circuit, sim_cfg),
                                              color=g['color'] or None)
                agent = registry.agent_by_id(agent_id)
                if k == 0:
                    group_id = registry.create_group(module=g['module'] or None,
                                                     color=agent.color, name=g['name'],
                                                     first_agent_id=agent_id)
                else:
                    registry.add_agent_to_group(group_id, agent_id)
            if not g['module']:
                continue
            # Later members copy the group's first brain's params, as in the app.
            params = (g['brain_params'] if first_brain is None else
                      {p: getattr(first_brain, p) for p in first_brain.get_param_metadata()})
            brain, _, _ = agent.brain_mgr.install_brain(g['module'], params)
            agent.brain = brain
            if first_brain is None:
                first_brain = brain

    resolve_mounted_patches(world, registry.agents)
    sim = Simulation(world, sim_cfg, registry)
    if start_engine:
        ok, err = sim.start_engine()
        if not ok:
            raise RuntimeError(f'MuJoCo engine could not start: {err}')
    sim.reset()
    return sim
