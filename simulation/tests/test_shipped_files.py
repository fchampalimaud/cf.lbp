"""
Backward-compatibility guard for the shipped sessions and networks (TODO 8.2).

Every session in configs/ and every network in networks/ must

  1. load and run SECONDS of simulated time without errors or NaN, and
  2. behave as recorded: after a seeded run, every robot's final pose matches
     tests/reference/shipped_reference.json.

The reference stores a hash of each entry's input files (session + network
JSON). If you edit a network or session, its entry no longer matches and is
skipped (not failed) until the reference is regenerated — so (2) only fails
when *code* changes how an unchanged file behaves.

Run from simulation/2d/:
    pytest tests/test_shipped_files.py -v
Regenerate the reference after an intended behaviour change:
    python tests/test_shipped_files.py --update
"""
import glob
import hashlib
import json
import math
import os
import sys

import pytest

_HERE  = os.path.dirname(os.path.abspath(__file__))
_SIM2D = os.path.abspath(os.path.join(_HERE, '..'))
for _p in (os.path.join(_SIM2D, 'brains'), os.path.join(_SIM2D, 'src'), _SIM2D):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Independent of the user-folder setting: the private repo's private/ folder
# (committed there) is the user root; otherwise only built-in files are run.
_PRIVATE = os.path.join(_SIM2D, 'private')
if os.path.isdir(_PRIVATE):
    os.environ['LBP_USER_DIR'] = _PRIVATE
ROOTS = [_SIM2D] + ([_PRIVATE] if os.path.isdir(_PRIVATE) else [])

REFERENCE = os.path.join(_HERE, 'reference', 'shipped_reference.json')
SECONDS   = 1.0
SEED      = 0
POSE_TOL  = 1e-6
# Per-user state, rewritten every time the app closes — not a shipped file.
EXCLUDE   = {'configs/latest_session.json', 'private/configs/latest_session.json'}


def _rel(path):
    return os.path.relpath(path, _SIM2D).replace(os.sep, '/')


def _sessions():
    out = []
    for f in sorted(f for root in ROOTS
                    for f in glob.glob(os.path.join(root, 'configs', '**', '*.json'), recursive=True)):
        rel = _rel(f)
        if rel in EXCLUDE:
            continue
        try:
            d = json.load(open(f, encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and 'sim_params' in d:   # brain_*.json etc. are not sessions
            out.append(rel)
    return out


def _networks():
    return [_rel(f) for f in sorted(f for root in ROOTS
                                    for f in glob.glob(os.path.join(root, 'networks', '**', '*.json'),
                                                       recursive=True))]


CASES = [('session', s) for s in _sessions()] + [('network', n) for n in _networks()]


def _session_dict(kind, rel):
    """The session to run: the file itself, or a bare network in an empty world."""
    if kind == 'session':
        return json.load(open(os.path.join(_SIM2D, rel), encoding='utf-8'))
    net = rel.split('networks/', 1)[1]  # path inside its root's networks/
    return {'mode': 'network', 'network': net, 'sim_params': {}, 'patches': [], 'objects': []}


def _input_hash(kind, rel, d):
    """Hash of every file this case reads: the session and the networks it names."""
    files = [rel] if kind == 'session' else []
    nets = {d.get('network')} | {a.get('network') for a in d.get('agents', []) or []}
    import data_paths
    h = hashlib.sha256()
    for f in files:
        p = os.path.join(_SIM2D, f)
        h.update(f.encode())
        h.update(open(p, 'rb').read() if os.path.exists(p) else b'<missing>')
    for n in sorted(n for n in nets if n):        # found like the app finds them
        p = data_paths.resolve('networks', n)
        h.update(('networks/' + n).encode())
        h.update(p.read_bytes() if p is not None else b'<missing>')
    return h.hexdigest()[:16]


def _run(kind, rel):
    """Seeded run → (input hash, final poses). Fails on errors or non-finite values."""
    from headless import seed_everything
    from session_loader import build_simulation
    d = _session_dict(kind, rel)
    digest = _input_hash(kind, rel, d)
    cwd = os.getcwd()
    os.chdir(_SIM2D)                     # sessions name networks/ relative to here
    try:
        seed_everything(SEED)
        sim = build_simulation(d)
        try:
            for _ in range(int(round(SECONDS / sim.sim_cfg.dt))):
                raws = sim.step()
                for raw in raws:
                    for k, v in raw.items():
                        if isinstance(v, float) and not math.isfinite(v):
                            raise AssertionError(f'{k} = {v} at step {sim.time_index}')
            poses = [[float(v) for v in a.bot_pos] for a in sim.agents]
        finally:
            sim.close()
    finally:
        os.chdir(cwd)
    assert all(math.isfinite(v) for p in poses for v in p), f'non-finite pose {poses}'
    return digest, poses


def _load_reference():
    if not os.path.exists(REFERENCE):
        return {}
    return json.load(open(REFERENCE, encoding='utf-8'))


@pytest.mark.parametrize('kind,rel', CASES, ids=[r for _k, r in CASES])
def test_shipped_file_loads_runs_and_matches_reference(kind, rel):
    digest, poses = _run(kind, rel)
    ref = _load_reference().get(rel)
    if ref is None:
        pytest.skip('no reference yet — run: python tests/test_shipped_files.py --update')
    if ref['hash'] != digest:
        pytest.skip('file edited since the reference was recorded — regenerate with --update')
    assert len(poses) == len(ref['poses']), 'number of agents changed'
    for got, want in zip(poses, ref['poses']):
        assert got == pytest.approx(want, abs=POSE_TOL), \
            f'behaviour changed: final pose {got} != reference {want}'


def update_reference():
    """Re-record every case (python tests/test_shipped_files.py --update)."""
    ref = {}
    for kind, rel in CASES:
        try:
            digest, poses = _run(kind, rel)
        except Exception as e:           # report and leave it out; the test will fail on it
            print(f'FAILED  {rel}: {type(e).__name__}: {e}')
            continue
        ref[rel] = {'hash': digest, 'poses': poses}
        print(f'ok      {rel}')
    os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
    with open(REFERENCE, 'w', encoding='utf-8') as f:
        json.dump(ref, f, indent=1, sort_keys=True)
    print(f'wrote {len(ref)} entries to {_rel(REFERENCE)}')


if __name__ == '__main__':
    if '--update' in sys.argv:
        update_reference()
    else:
        print(__doc__)
