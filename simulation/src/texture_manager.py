"""
texture_manager.py — discovery and loading for texture image files.

Textures live as flat PNG files in `textures/`, discovered the same way brains are
discovered in `brains/` (brain_manager.discover_brains). A texture is referenced
elsewhere purely by filename (e.g. 'checkerboard.png').
"""

import glob
import os


def discover_textures():
    """Return sorted basenames of all texture files in textures/."""
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join('textures', '*.png')))


def texture_path(name):
    return os.path.join('textures', name)


def texture_bytes(name):
    """Read and return the raw bytes of textures/<name>."""
    with open(texture_path(name), 'rb') as f:
        return f.read()


def texture_exists(name):
    return name is not None and os.path.isfile(texture_path(name))
