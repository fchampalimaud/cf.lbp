"""
session_io.py — JSON serialization for simulator sessions.

Defines the session file format in one place, independent of Qt widgets.
"""

import json
import os

from sim_constants import _NumpyEncoder


def save_session(path, module_name, brain, sim_cfg, world,
                 speed_mult, trail_length, arena_round, multipliers, groups=None):
    """
    Write a session JSON to *path*.

    Parameters
    ----------
    path         : destination file path (e.g. 'configs/experiment_1.json')
    module_name  : brain module name as selected in the UI combo box
    brain        : active BaseBrain instance
    sim_cfg      : SimConfig instance
    world        : World instance
    speed_mult   : current speed multiplier (int)
    trail_length : current trail length (int)
    arena_round  : True if arena is circular
    multipliers  : dict of oscilloscope channel → scale value
    groups       : optional list of agent-group dicts for multiagent sessions
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    data = {
        'module_name':      module_name,
        'class_name':       brain.__class__.__name__,
        'sim_params':       {k: getattr(sim_cfg, k) for k in sim_cfg.get_param_metadata()},
        'brain_params':     {k: getattr(brain, k)   for k in brain.get_param_metadata()},
        'plot_multipliers': multipliers,
        'patches':          world.patches,
        'objects':          world.objects,
        'walls':            world.walls,
        'sky':              world.sky,
        'speed_mult':       speed_mult,
        'trail_length':     trail_length,
        'arena_round':      arena_round,
    }
    if groups is not None:
        data['agents'] = [
            {
                'module_name': g['module'] or '',
                'name':        g['name'],
                'color':       g['color'],
                'n':           g['n'],
                'brain_params': g.get('brain_params', {}),
            }
            for g in groups
        ]
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, cls=_NumpyEncoder)


def load_session(path):
    """Read and return the session dict from *path*."""
    with open(path, 'r') as f:
        return json.load(f)
