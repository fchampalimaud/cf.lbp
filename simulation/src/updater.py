"""
updater.py — update the simulator from the public repository. No Qt.

Only active in a copy installed from the public repo: that copy carries
release.json (written by the public sync, never present in the private repo):

    {"repo": "fchampalimaud/cf.lbp", "branch": "main", "path": "simulation"}

Optional "version_url" / "archive_url" override where the version and the
zip come from (any urllib URL, file:// included — used to test the updater).

An update replaces only the app's own files: the user's folder (data_paths)
is never touched, and user files still sitting in the app folder are moved
there first (data_paths.migrate_user_files).
"""

import io
import json
import shutil
import urllib.request
import zipfile

import data_paths

RELEASE = data_paths.APP_DIR / 'release.json'
STAGING = data_paths.APP_DIR / '_update'


def release_info():
    """The release marker, or None in a development copy (no updates there)."""
    try:
        return json.loads(RELEASE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _urls(info):
    repo, branch, path = info['repo'], info.get('branch', 'main'), info.get('path', 'simulation')
    version_url = info.get('version_url') or \
        f'https://raw.githubusercontent.com/{repo}/{branch}/{path}/VERSION'
    archive_url = info.get('archive_url') or \
        f'https://github.com/{repo}/archive/refs/heads/{branch}.zip'
    return version_url, archive_url


def _version_tuple(v):
    try:
        return tuple(int(x) for x in str(v).strip().split('.'))
    except ValueError:
        return ()


def is_newer(latest, current):
    return _version_tuple(latest) > _version_tuple(current)


def latest_version(timeout=5):
    """The version published in the public repo, or None (offline, no marker…)."""
    info = release_info()
    if info is None:
        return None
    try:
        with urllib.request.urlopen(_urls(info)[0], timeout=timeout) as r:
            return r.read().decode('utf-8').strip()
    except Exception:
        return None


def apply_update(timeout=60):
    """Download the public repo and replace the app's files with its copy.
    Returns the new version. Raises on any problem before files are replaced
    (download, unreadable zip, incomplete copy) — the install is then unchanged."""
    info = release_info()
    if info is None:
        raise RuntimeError('This copy has no release.json — updates come from git here.')
    with urllib.request.urlopen(_urls(info)[1], timeout=timeout) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))

    # GitHub zips hold one top folder (<repo>-<branch>/); the simulator is in <path>/ inside it.
    top = archive.namelist()[0].split('/')[0]
    inner = f"{top}/{info.get('path', 'simulation').strip('/')}/"
    shutil.rmtree(STAGING, ignore_errors=True)
    for name in archive.namelist():
        if name.startswith(inner) and not name.endswith('/'):
            dest = STAGING / name[len(inner):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(name))
    for required in ('LBPSimulator.py', 'VERSION', 'src/data_paths.py'):
        if not (STAGING / required).exists():
            shutil.rmtree(STAGING, ignore_errors=True)
            raise RuntimeError(f'Downloaded copy is incomplete (no {required}); nothing was changed.')

    data_paths.migrate_user_files()             # the user's files leave the app folder first
    old_shipped = data_paths.read_manifest() or set()
    new_shipped = data_paths.read_manifest(STAGING) or set()
    for src in sorted(p for p in STAGING.rglob('*') if p.is_file()):
        dest = data_paths.APP_DIR / src.relative_to(STAGING)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    for rel in sorted(old_shipped - new_shipped):   # shipped files the new version dropped
        (data_paths.APP_DIR / rel).unlink(missing_ok=True)
    shutil.rmtree(STAGING, ignore_errors=True)
    return (data_paths.APP_DIR / 'VERSION').read_text(encoding='utf-8').strip()
