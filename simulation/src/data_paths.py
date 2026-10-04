"""
data_paths.py — where the simulator's files live. No Qt (used headless too).

Two roots with the same layout — configs/, networks/, worlds/, motifs/,
brains/, textures/, logs/:

  APP_DIR     simulation/2d/ — built-in content shipped with the app
              (Tutorials, Demos, Default …). Read-only in the UI; an update
              replaces it.
  user_dir()  the user's own files. Writable; updates never touch it.
              Set in the Session tab (stored in settings_file(), outside the
              folder itself); default: APP_DIR/private/ when it exists (the
              private development repo, where it is committed), otherwise
              ~/LBPSimulator. The LBP_USER_DIR environment variable overrides
              both (tests, scripts).

References between files (a session naming its network, …) are paths
relative to the kind's folder, e.g. "Tutorials/T02 - DistanceBrain.json".
"builtin:" in front forces the app root; a plain reference is looked up in
the user root first, then the app root — so older sessions keep working
whichever root their network ended up in.
"""

import json
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
KINDS   = ('configs', 'networks', 'worlds', 'motifs', 'brains', 'textures', 'logs')
BUILTIN = 'builtin:'


# ── Settings (where the user folder is) ──────────────────────────────────────

def settings_file():
    """Per-user settings file, in the platform's config location."""
    if sys.platform.startswith('win'):
        base = Path(os.environ.get('APPDATA') or Path.home() / 'AppData' / 'Roaming')
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:
        base = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config')
    return base / 'LBPSimulator' / 'settings.json'


def load_settings():
    try:
        return json.loads(settings_file().read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    f = settings_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(settings, indent=1), encoding='utf-8')


def default_user_dir():
    private = APP_DIR / 'private'
    return private if private.is_dir() else Path.home() / 'LBPSimulator'


def user_dir():
    """The user's root folder (created if missing)."""
    chosen = os.environ.get('LBP_USER_DIR') or load_settings().get('user_dir')
    root = Path(chosen).expanduser() if chosen else default_user_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def set_user_dir(path):
    """Remember *path* as the user folder; None goes back to the default."""
    settings = load_settings()
    if path:
        settings['user_dir'] = str(Path(path).expanduser())
    else:
        settings.pop('user_dir', None)
    save_settings(settings)


# ── Folders of one kind ──────────────────────────────────────────────────────

def app_path(kind, rel=''):
    return APP_DIR / kind / rel if rel else APP_DIR / kind


def user_path(kind, rel='', create=True):
    """Path in the user root; creates the kind's folder (not the file)."""
    folder = user_dir() / kind
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder / rel if rel else folder


def search_dirs(kind):
    """Both roots' folders for *kind*, user first (for discovery / imports)."""
    return [p for p in (user_path(kind), app_path(kind)) if p.is_dir()]


def _inside(path, folder):
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except ValueError:
        return False


def is_builtin(path):
    """True for a file shipped with the app (read-only in the UI). The user
    root may sit inside APP_DIR (private/), so it is checked first."""
    return _inside(path, APP_DIR) and not _inside(path, user_dir())


def resolve(kind, ref):
    """The file a reference names, or None if it exists in neither root."""
    if not ref:
        return None
    ref = str(ref).replace('\\', '/')
    if ref.startswith(BUILTIN):
        p = app_path(kind, ref[len(BUILTIN):])
        return p if p.exists() else None
    for p in (user_path(kind, ref, create=False), app_path(kind, ref)):
        if p.exists():
            return p
    return None


def builtin_subdirs(kind):
    """Subfolders of the app root for *kind* (e.g. Tutorials, Demos), sorted."""
    root = app_path(kind)
    if not root.is_dir():
        return []
    return sorted(e.name for e in os.scandir(root)
                  if e.is_dir() and not e.name.startswith(('.', '_')))


def user_subdirs(kind):
    root = user_path(kind)
    return sorted(e.name for e in os.scandir(root)
                  if e.is_dir() and not e.name.startswith(('.', '_')))


def folder_for(kind, key):
    """Folder for a picker key: 'builtin:Tutorials' → app root's Tutorials,
    'Mine' → user root's Mine, '' → the user root."""
    key = key or ''
    if key.startswith(BUILTIN):
        return app_path(kind, key[len(BUILTIN):])
    return user_path(kind, key)


def files(kind, pattern):
    """(name, path) of every *pattern* file directly in either root's folder,
    user files first; a name in both roots appears once (the user's)."""
    seen, out = set(), []
    for folder in search_dirs(kind):
        for p in sorted(folder.glob(pattern)):
            if p.name not in seen:
                seen.add(p.name)
                out.append((p.name, p))
    return out


# ── Shipped-file manifest and the one-time migration ─────────────────────────
# The public copy carries manifest.json: every data file it ships. Anything
# else found in the app's data folders was made by the user (installs from
# before the user folder existed kept everything here) and is moved to the
# user folder — before an update could replace or delete it.

MANIFEST = APP_DIR / 'manifest.json'
_DATA_KINDS = ('configs', 'networks', 'worlds', 'motifs', 'brains', 'textures')


def _data_files(root=APP_DIR):
    """Relative paths ('networks/Tutorials/x.json') of every data file under root."""
    out = []
    for kind in _DATA_KINDS:
        folder = Path(root) / kind
        if folder.is_dir():
            for p in sorted(folder.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts:
                    out.append(p.relative_to(root).as_posix())
    return out


def write_manifest(root=APP_DIR):
    """List the data files shipped in *root* (run on the public copy at sync time)."""
    (Path(root) / 'manifest.json').write_text(
        json.dumps({'files': _data_files(root)}, indent=1), encoding='utf-8')


def read_manifest(root=APP_DIR):
    try:
        return set(json.loads((Path(root) / 'manifest.json').read_text(encoding='utf-8'))['files'])
    except (OSError, ValueError, KeyError):
        return None


def migrate_user_files():
    """Move every unlisted file from the app's data folders (and logs/) to the
    user folder. Only runs where a manifest exists (the public copy); a name
    already taken in the user folder gets ' (moved)' appended. Returns the
    destination paths."""
    shipped = read_manifest()
    if shipped is None:
        return []
    moved = []
    candidates = [rel for rel in _data_files() if rel not in shipped]
    logs = APP_DIR / 'logs'
    if logs.is_dir():
        candidates += [p.relative_to(APP_DIR).as_posix() for p in sorted(logs.rglob('*')) if p.is_file()]
    for rel in candidates:
        kind, _, sub = rel.partition('/')
        dest = user_path(kind, sub)
        dest.parent.mkdir(parents=True, exist_ok=True)
        while dest.exists():
            dest = dest.with_name(f'{dest.stem} (moved){dest.suffix}')
        (APP_DIR / rel).replace(dest)
        moved.append(dest)
    return moved


if __name__ == '__main__':
    # python src/data_paths.py --write-manifest [root]   (public sync step)
    if len(sys.argv) >= 2 and sys.argv[1] == '--write-manifest':
        target = Path(sys.argv[2]) if len(sys.argv) > 2 else APP_DIR
        write_manifest(target)
        print(f'wrote {target / "manifest.json"}')
