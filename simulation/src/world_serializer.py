"""
world_serializer.py — JSON serialization for standalone world files.

Worlds live as flat JSON files in `worlds/`, discovered the same way brains are
discovered in `brains/` (brain_manager.discover_brains). A world file captures
arena state independent of any session/brain: gradient patches, objects, walls,
sky, arena shape, and floor texture.
"""

import json
import os

from sim_constants import _NumpyEncoder
from app_version import get_app_version


def discover_worlds():
    """Sorted names of the world files in both roots' worlds/ (data_paths)."""
    import data_paths
    return sorted(name for name, _p in data_paths.files('worlds', '*.json'))


def serialize_world_json(world) -> dict:
    """Return a dict capturing the persistable state of *world*."""
    return {
        'version':              1,
        'saved_with_app_version': get_app_version(),
        'patches':       world.patches,
        'objects':       world.objects,
        'walls':         world.walls,
        'sky':           world.sky,
        'arena_round':   world.arena_round,
        'floor_texture': world.floor_texture,
    }


def load_world_json(data: dict, world) -> None:
    """Populate *world* in place from a previously-serialized dict."""
    world.patches       = data.get('patches', [])
    world.objects       = data.get('objects', [])
    world.walls         = data.get('walls', [])
    world.sky           = data.get('sky', {"enabled": False, "angle": 0.0})
    world.arena_round   = bool(data.get('arena_round', False))
    world.floor_texture = data.get('floor_texture')


def save_world_file(world, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(serialize_world_json(world), f, indent=4, cls=_NumpyEncoder)


def load_world_file(path, world):
    with open(path, 'r') as f:
        data = json.load(f)
    load_world_json(data, world)
