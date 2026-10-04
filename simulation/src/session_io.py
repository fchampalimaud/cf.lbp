"""
session_io.py — JSON serialization for simulator sessions.

Defines the session file format in one place, independent of Qt widgets.
"""

import json
import os

from sim_constants import _NumpyEncoder
from app_version import get_app_version

# A group runs either a code brain (a brains/*.py module) or a network (a
# networks/*.json circuit). In a session file that reads:
#   code:    {"mode": "code", "module_name": "BrainARS", "brain_params": {...}}
#   network: {"mode": "network", "network": "Tutorials/T06 - FeedingBrain.json"}
# Internally a network runs in the network brain module, with the project and
# file as its brain params. Older files ({"module_name": "BrainGUI",
# "brain_params": {"network_project": ..., "network_file": ...}}) still load.
NETWORK_BRAIN_MODULE = 'BrainGUI'
_NETWORK_PARAMS = ('network_project', 'network_file')


def brain_to_file(module, brain_params):
    """The session-file entry for a group running *module* with *brain_params*."""
    brain_params = dict(brain_params or {})
    if module == NETWORK_BRAIN_MODULE:
        project = brain_params.pop('network_project', '') or ''
        net     = brain_params.pop('network_file', '') or ''
        entry = {'mode': 'network', 'network': f'{project}/{net}' if project and net else net}
        if brain_params:
            entry['brain_params'] = brain_params
        return entry
    return {'mode': 'code', 'module_name': module or '', 'brain_params': brain_params}


def brain_from_file(entry):
    """(module, brain_params) for a session-file group entry (or a top-level
    session dict) — new or old format."""
    params = dict(entry.get('brain_params') or {})
    if entry.get('mode') == 'network':
        project, _, net = (entry.get('network') or '').rpartition('/')
        params['network_project'] = project
        params['network_file']    = net
        return NETWORK_BRAIN_MODULE, params
    return entry.get('module_name') or '', params


def save_session(path, module_name, brain, sim_cfg, world,
                 speed_mult, trail_length, arena_round, multipliers,
                 groups=None, net_cfg=None, patches=None):
    """
    Write a session JSON to *path*.

    Parameters
    ----------
    path         : destination file path (e.g. 'configs/experiment_1.json')
    module_name  : brain module of the selected group (written as code brain
                   or network, see brain_to_file)
    brain        : active BaseBrain instance
    sim_cfg      : SimConfig instance
    world        : World instance
    speed_mult   : current speed multiplier (int)
    trail_length : current trail length (int)
    arena_round  : True if arena is circular
    multipliers  : dict of oscilloscope channel → scale value
    groups       : optional list of agent-group dicts for multiagent sessions
    patches      : optional override for world.patches — used by the caller to
                   translate a mounted-gradient patch's 'mounted_on' agent id
                   into a stable positional index (see sim_app_session), since
                   raw agent ids are re-minted on every session reload. Defaults
                   to world.patches verbatim if not given.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    data = {'saved_with_app_version': get_app_version()}
    data.update(brain_to_file(module_name,
                              {k: getattr(brain, k) for k in brain.get_param_metadata()}))
    if data['mode'] == 'code':
        data['class_name'] = brain.__class__.__name__
    data.update({
        'sim_params':       {k: getattr(sim_cfg, k) for k in sim_cfg.get_param_metadata()},
        'plot_multipliers': multipliers,
        'patches':          patches if patches is not None else world.patches,
        'objects':          world.objects,
        'walls':            world.walls,
        'sky':              world.sky,
        'floor_texture':    world.floor_texture,
        'speed_mult':       speed_mult,
        'trail_length':     trail_length,
        'arena_round':      arena_round,
    })
    if groups is not None:
        data['agents'] = [
            {'name': g['name'], 'color': g['color'], 'n': g['n'],
             **brain_to_file(g['module'], g.get('brain_params', {}))}
            for g in groups
        ]
    if net_cfg is not None:
        data['net'] = net_cfg
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, cls=_NumpyEncoder)


def load_session(path):
    """Read and return the session dict from *path*."""
    with open(path, 'r') as f:
        return json.load(f)
