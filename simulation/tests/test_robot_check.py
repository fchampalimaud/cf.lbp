"""
tests/test_robot_check.py — robot connection checklist (robot_check.py) unit tests.

Run from simulation/2d/:
    pytest tests/test_robot_check.py -v
"""
import os
import socket
import sys
import threading
import time
import types

_HERE  = os.path.dirname(__file__)
_SIM2D = os.path.abspath(os.path.join(_HERE, '..'))
_SRC   = os.path.join(_SIM2D, 'src')
for _p in (_SRC, _SIM2D):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robot_check import (RobotCheck, SensorSpec, plan, parse_ping, analyse_sensor,
                         PASS, WARN, FAIL, SKIP, RUNNING)
from robot_driver import _osc_build


def _sensor(name, addr, n=2, camera=False):
    return types.SimpleNamespace(name=name, robot_address=addr, n=n, is_camera=camera)


def _motor(name, addr):
    return types.SimpleNamespace(name=name, robot_address=addr, drives_wheels=True)


def _circuit(sensors=(), layers=()):
    return types.SimpleNamespace(sensors=list(sensors), layers=list(layers))


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run(check):
    results = {}
    check.run(lambda k, st, d: results.__setitem__(k, (st, d)) if st != RUNNING else None,
              threading.Event())
    return results


# ── plan ──────────────────────────────────────────────────────────────────────

def test_plan_groups_by_host_port():
    c = _circuit([_sensor('bump', '10.0.0.5:2390/bumpers(1,2)'),
                  _sensor('dist', '10.0.0.5:2390/analogs(1,4){x/500}')],
                 [_motor('motor', '10.0.0.5:2390/wheels')])
    eps, problems = plan(c)
    assert problems == []
    assert len(eps) == 1
    assert [s.name for s in eps[0].sensors] == ['bump', 'dist']
    assert eps[0].motor_paths == ['/wheels']


def test_plan_reports_index_count_mismatch_and_missing_path():
    c = _circuit([_sensor('a', '10.0.0.5:2390/analogs(1,2,3)', n=2),
                  _sensor('b', '10.0.0.5:2390', n=2)],
                 [_motor('m', '10.0.0.5:2390')])
    eps, problems = plan(c)
    assert len(problems) == 3
    assert "3 indices" in problems[0]
    assert "no OSC path" in problems[1]
    assert "Motor layer 'm'" in problems[2]


def test_no_addresses_fails_config():
    r = _run(RobotCheck(_circuit([_sensor('a', '')])))
    assert r['config'][0] == FAIL


def test_missing_motor_address_fails_config():
    r = _run(RobotCheck(_circuit([_sensor('a', '127.0.0.1:1/analogs(1,2)')]),
                        ping=lambda h, n, c: [1.0] * n))
    assert r['config'][0] == FAIL
    assert 'No motor layer' in r['config'][1]


# ── parse_ping ────────────────────────────────────────────────────────────────

def test_parse_ping_windows_english():
    out = ("Reply from 192.168.0.224: bytes=32 time=3ms TTL=255\n"
           "Reply from 192.168.0.224: bytes=32 time<1ms TTL=255\n"
           "Request timed out.\n"
           "    Minimum = 1ms, Maximum = 3ms, Average = 2ms\n")
    assert parse_ping(out) == [3.0, 1.0]


def test_parse_ping_windows_portuguese():
    out = ("Resposta de 192.168.0.224: bytes=32 tempo=12ms TTL=255\n"
           "Esgotado o tempo limite do pedido.\n")
    assert parse_ping(out) == [12.0]


def test_parse_ping_linux():
    out = "64 bytes from 192.168.0.224: icmp_seq=1 ttl=255 time=4.37 ms\n"
    assert parse_ping(out) == [4.37]


def test_parse_ping_unreachable_is_not_a_reply():
    out = "Reply from 192.168.0.10: Destination host unreachable.\n"
    assert parse_ping(out) == []


# ── analyse_sensor ────────────────────────────────────────────────────────────

def _spec(path='/analogs', indices=(1, 4), formula='', n=2):
    return SensorSpec('dist', path, list(indices) if indices else None, formula, n, False)


def test_analyse_sensor_pass_with_formula():
    msgs = [('/analogs', [7, 500, 0, 0, 1000])] * 10
    status, detail = analyse_sensor(_spec(formula='x/500'), msgs, 1.0)
    assert status == PASS
    assert '10 Hz' in detail and '1, 2' in detail


def test_analyse_sensor_missing_path_lists_what_arrived():
    status, detail = analyse_sensor(_spec(), [('/bumpers', [0, 0, 0, 0, 0])], 1.0)
    assert status == FAIL
    assert '/bumpers' in detail


def test_analyse_sensor_index_out_of_range():
    status, detail = analyse_sensor(_spec(indices=(1, 7)), [('/analogs', [0, 1, 2])], 1.0)
    assert status == FAIL
    assert 'index 7' in detail


def test_analyse_sensor_formula_error():
    status, _ = analyse_sensor(_spec(formula='x/'), [('/analogs', [0, 1, 2, 3, 4])], 1.0)
    assert status == FAIL


def test_analyse_sensor_no_indices_size_mismatch_warns():
    status, _ = analyse_sensor(_spec(indices=None), [('/analogs', [0, 1, 2, 3, 4])], 1.0)
    assert status == WARN


# ── end-to-end against a fake robot on localhost ─────────────────────────────

def test_port_in_use_fails_and_skips_reply():
    blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    blocker.bind(('', 0))
    port = blocker.getsockname()[1]
    try:
        c = _circuit([_sensor('dist', f'127.0.0.1:{port}/analogs(1,4)')],
                     [_motor('motor', f'127.0.0.1:{port}/wheels')])
        r = _run(RobotCheck(c, ping=lambda h, n, cancel: [1.0] * n))
    finally:
        blocker.close()
    assert r[f'127.0.0.1:{port}:port'][0] == FAIL
    assert r[f'127.0.0.1:{port}:reply'][0] == SKIP
    assert r[f'127.0.0.1:{port}:s:dist'][0] == SKIP


def test_full_run_with_fake_robot_passes_and_releases_port():
    port = _free_port()
    stop = threading.Event()

    def robot():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        i = 0
        while not stop.is_set():
            s.sendto(_osc_build('/analogs', i, 100, 200, 300, 400), ('127.0.0.1', port))
            i += 1
            time.sleep(0.01)
        s.close()

    t = threading.Thread(target=robot, daemon=True)
    t.start()
    try:
        c = _circuit([_sensor('dist', f'127.0.0.1:{port}/analogs(1,4){{x/100}}')],
                     [_motor('motor', f'127.0.0.1:{port}/wheels')])
        check = RobotCheck(c, ping=lambda h, n, cancel: [2.0] * n)
        check.LISTEN_S = 0.5
        r = _run(check)
    finally:
        stop.set()
        t.join()
    ep = f'127.0.0.1:{port}'
    assert r['config'][0] == PASS
    assert r[f'{ep}:port'][0] == PASS
    assert r[f'{ep}:ping'][0] == PASS
    assert r[f'{ep}:reply'][0] == PASS and '/analogs' in r[f'{ep}:reply'][1]
    assert r[f'{ep}:s:dist'] [0] == PASS and '1, 4' in r[f'{ep}:s:dist'][1]
    # the port is free again, so real-robot mode can bind it
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('', port))
    s.close()


def test_slow_ping_warns():
    c = _circuit([], [_motor('motor', '127.0.0.1:1/wheels')])
    check = RobotCheck(c, ping=lambda h, n, cancel: [5.0, 400.0])
    check.LISTEN_S = 0.1
    r = _run(check)
    assert r['127.0.0.1:1:ping'][0] == WARN
