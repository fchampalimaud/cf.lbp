"""
tests/test_logger.py — SimLogger and TrajectoryLoader unit tests.

Run from simulation/2d/:
    pytest tests/test_logger.py -v
"""
import json
import sys
import os
import types

import numpy as np
import pytest

_HERE  = os.path.dirname(__file__)
_SIM2D = os.path.abspath(os.path.join(_HERE, '..'))
_SRC   = os.path.join(_SIM2D, 'src')
for _p in (_SRC, _SIM2D):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from logger import SimLogger, _scalar_signals
from trajectory_viz import TrajectoryLoader


# ── Helpers ───────────────────────────────────────────────────────────────────

def _agent(aid, name, color, x=0.0, y=0.0, theta=0.0):
    a = types.SimpleNamespace(
        id=aid, name=name, color=color,
        bot_pos=[x, y, theta],
    )
    return a


def _world(objects=(), patches=()):
    return types.SimpleNamespace(objects=list(objects), patches=list(patches))


def _raw(mL=0.0, mR=0.0, **signals):
    return {'mL': mL, 'mR': mR, **signals}


def _read_rows(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


# ── _scalar_signals ───────────────────────────────────────────────────────────

def test_scalar_signals_excludes_mL_mR():
    raw = _raw(mL=0.5, mR=0.3, light=0.9)
    s   = _scalar_signals(raw)
    assert 'mL' not in s and 'mR' not in s
    assert s['light'] == pytest.approx(0.9, abs=1e-4)


def test_scalar_signals_skips_arrays():
    raw = _raw(cam=np.array([1, 2, 3]), val=0.7)
    s   = _scalar_signals(raw)
    assert 'cam' not in s
    assert 'val' in s


def test_scalar_signals_skips_none():
    raw = _raw(bad=None, good=1.0)
    s   = _scalar_signals(raw)
    assert 'bad' not in s
    assert 'good' in s


# ── SimLogger — basic write ───────────────────────────────────────────────────

def test_logger_writes_meta_and_tick_rows(tmp_path):
    path   = str(tmp_path / 'run.jsonl')
    agents = [_agent(0, 'A1', '#ff0000'), _agent(1, 'A2', '#00ff00')]
    raws   = [_raw(mL=0.1, mR=0.2, s=0.5), _raw(mL=0.3, mR=0.4, s=0.6)]
    world  = _world()

    lg = SimLogger()
    lg.start(path, arena_scale=5.0, arena_round=False,
             agents=[{'id': a.id, 'name': a.name, 'color': a.color} for a in agents])
    lg.log(0, agents, raws, 0, world)
    lg.stop()

    rows = _read_rows(path)
    meta = rows[0]
    tick = rows[1]

    assert meta['_meta'] is True
    assert meta['version'] == 2
    assert meta['arena_scale'] == 5.0
    assert len(meta['agents']) == 2

    assert tick['t'] == 0
    assert tick['sel'] == 0
    assert len(tick['agents']) == 2
    assert tick['agents'][0]['id'] == 0
    assert tick['agents'][1]['id'] == 1
    assert tick['agents'][0]['mL'] == pytest.approx(0.1, abs=1e-4)


def test_logger_top_level_positions_per_agent(tmp_path):
    path   = str(tmp_path / 'run.jsonl')
    a0     = _agent(0, 'A', '#aaa', x=1.0, y=2.0, theta=0.5)
    a1     = _agent(1, 'B', '#bbb', x=3.0, y=4.0, theta=1.0)
    agents = [a0, a1]
    raws   = [_raw(), _raw()]
    world  = _world()

    lg = SimLogger()
    lg.start(path, agents=[{'id': a.id, 'name': a.name, 'color': a.color} for a in agents])
    lg.log(0, agents, raws, 0, world)
    lg.stop()

    tick = _read_rows(path)[1]
    a_rows = {a['id']: a for a in tick['agents']}
    assert a_rows[0]['x'] == pytest.approx(1.0, abs=1e-6)
    assert a_rows[0]['y'] == pytest.approx(2.0, abs=1e-6)
    assert a_rows[1]['x'] == pytest.approx(3.0, abs=1e-6)


def test_logger_selected_id_recorded(tmp_path):
    path   = str(tmp_path / 'run.jsonl')
    agents = [_agent(0, 'A', '#aaa'), _agent(1, 'B', '#bbb')]
    raws   = [_raw(), _raw()]
    world  = _world()

    lg = SimLogger()
    lg.start(path, agents=[{'id': a.id, 'name': a.name, 'color': a.color} for a in agents])
    lg.log(5, agents, raws, 1, world)   # agent 1 is selected
    lg.stop()

    tick = _read_rows(path)[1]
    assert tick['sel'] == 1


def test_logger_mid_recording_agent_emits_registration_row(tmp_path):
    path  = str(tmp_path / 'run.jsonl')
    a0    = _agent(0, 'A', '#aaa')
    world = _world()

    lg = SimLogger()
    lg.start(path, agents=[{'id': a0.id, 'name': a0.name, 'color': a0.color}])
    lg.log(0, [a0], [_raw()], 0, world)

    # Agent 1 joins mid-recording
    a1 = _agent(1, 'B', '#bbb')
    lg.log(1, [a0, a1], [_raw(), _raw()], 0, world)
    lg.stop()

    rows = _read_rows(path)
    # rows: meta, tick0, _agent-registration, tick1
    assert rows[2].get('_agent') is True
    assert rows[2]['id'] == 1
    assert rows[2]['name'] == 'B'
    # registration row is not emitted again on the next tick
    lg2 = SimLogger()
    lg2.start(str(tmp_path / 'run2.jsonl'),
              agents=[{'id': a0.id, 'name': a0.name, 'color': a0.color}])
    lg2.log(0, [a0, a1], [_raw(), _raw()], 0, world)
    lg2.log(1, [a0, a1], [_raw(), _raw()], 0, world)
    lg2.stop()
    rows2 = _read_rows(str(tmp_path / 'run2.jsonl'))
    agent_rows = [r for r in rows2 if r.get('_agent')]
    assert len(agent_rows) == 1   # emitted only once


def test_logger_vector_signal_does_not_crash(tmp_path):
    path   = str(tmp_path / 'run.jsonl')
    agents = [_agent(0, 'A', '#aaa')]
    raws   = [_raw(cam=np.array([0.1, 0.2, 0.3]), scalar=0.9)]
    world  = _world()

    lg = SimLogger()
    lg.start(path, agents=[{'id': a.id, 'name': a.name, 'color': a.color} for a in agents])
    lg.log(0, agents, raws, 0, world)
    lg.stop()

    tick = _read_rows(path)[1]
    sigs = tick['agents'][0]['signals']
    assert 'cam' not in sigs
    assert 'scalar' in sigs


def test_logger_row_count(tmp_path):
    path   = str(tmp_path / 'run.jsonl')
    agents = [_agent(0, 'A', '#aaa')]
    world  = _world()

    lg = SimLogger()
    lg.start(path, agents=[{'id': a.id, 'name': a.name, 'color': a.color} for a in agents])
    for i in range(5):
        lg.log(i, agents, [_raw()], 0, world)
    lg.stop()

    assert lg.row_count == 5


# ── TrajectoryLoader — v2 ─────────────────────────────────────────────────────

def _write_v2_log(path, n_agents=2, n_ticks=3):
    agent_defs = [{'id': i, 'name': f'A{i}', 'color': '#ababab'} for i in range(n_agents)]
    agents = [_agent(a['id'], a['name'], a['color']) for a in agent_defs]
    world  = _world()
    lg = SimLogger()
    lg.start(path, arena_scale=5.0, arena_round=False, agents=agent_defs)
    for t in range(n_ticks):
        for a in agents:
            a.bot_pos[0] = float(a.id * 10 + t)
        lg.log(t, agents, [_raw()] * n_agents, agents[0].id, world)
    lg.stop()


def test_trajectory_loader_v2_tracks(tmp_path):
    path = str(tmp_path / 'v2.jsonl')
    _write_v2_log(path, n_agents=2, n_ticks=4)

    tl = TrajectoryLoader.from_file(path)
    assert len(tl.tracks) == 2
    assert 0 in tl.tracks and 1 in tl.tracks
    assert len(tl.tracks[0]['x']) == 4
    assert len(tl.tracks[1]['x']) == 4


def test_trajectory_loader_v2_agent_info(tmp_path):
    path = str(tmp_path / 'v2.jsonl')
    _write_v2_log(path, n_agents=2, n_ticks=2)

    tl = TrajectoryLoader.from_file(path)
    assert tl.agent_info[0]['name'] == 'A0'
    assert tl.agent_info[1]['name'] == 'A1'


def test_trajectory_loader_v2_compat_props(tmp_path):
    path = str(tmp_path / 'v2.jsonl')
    _write_v2_log(path, n_agents=2, n_ticks=3)

    tl = TrajectoryLoader.from_file(path)
    assert tl.n_steps == 3
    assert len(tl.xs) == 3
    assert tl.arena_scale == 5.0


def test_trajectory_loader_v1_single_agent(tmp_path):
    path = str(tmp_path / 'v1.jsonl')
    rows = [
        {'_meta': True, 'arena_scale': 4.0, 'arena_round': False},
        {'t': 0, 'x': 0.1, 'y': 0.2, 'theta': 0.3,
         'mL': 0.0, 'mR': 0.0, 'signals': {},
         'patches': [], 'objects': []},
        {'t': 1, 'x': 0.4, 'y': 0.5, 'theta': 0.6,
         'mL': 0.0, 'mR': 0.0, 'signals': {},
         'patches': [], 'objects': []},
    ]
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    tl = TrajectoryLoader.from_file(path)
    assert len(tl.tracks) == 1
    assert tl.n_steps == 2
    assert tl.xs[0] == pytest.approx(0.1)
    assert tl.arena_scale == 4.0
