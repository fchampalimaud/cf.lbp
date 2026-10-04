"""
texture_manager.py — discovery and loading for texture image files.

Textures live as flat PNG files in `textures/`, discovered the same way brains are
discovered in `brains/` (brain_manager.discover_brains). A texture is referenced
elsewhere purely by filename (e.g. 'checkerboard.png').
"""

import os


def discover_textures():
    """Sorted names of the texture files in both roots' textures/ (data_paths)."""
    import data_paths
    return sorted(name for name, _p in data_paths.files('textures', '*.png'))


def texture_path(name):
    """The texture file for *name*: the user's first, then the built-in one."""
    import data_paths
    p = data_paths.resolve('textures', name)
    return str(p) if p is not None else str(data_paths.app_path('textures', name))


def texture_bytes(name):
    """Read and return the raw bytes of textures/<name>."""
    with open(texture_path(name), 'rb') as f:
        return f.read()


def texture_exists(name):
    return name is not None and os.path.isfile(texture_path(name))
