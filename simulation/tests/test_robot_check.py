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
                         stream_stats, PASS, WARN, FAIL, SKIP, RUNNING)
from robot_driver import _osc_build, _osc_bundle, _osc_messages, OscThread


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


# ── stream_stats ──────────────────────────────────────────────────────────────

def test_stream_stats_sent_received_and_loss():
    # robot sends at 60/s (counter 0..60 over 1 s) but every 4th packet is lost
    kept = [c for c in range(61) if c % 4 != 3 or c == 60]
    msgs = [('/analogs', [c, 1, 2, 3, 4]) for c in kept]
    stamps = [c / 60 for c in kept]
    sent, received, lost = stream_stats(msgs, stamps)['/analogs']
    assert abs(sent - 60) < 1e-6
    assert abs(received - (len(kept) - 1)) < 1e-6
    assert abs(lost - (1 - (len(kept) - 1) / 60)) < 1e-6


def test_stream_stats_without_counter_gives_received_only():
    msgs = [('/wheels', [0, 0])] * 11
    stamps = [i / 10 for i in range(11)]
    assert stream_stats(msgs, stamps)['/wheels'] == (None, 10.0, None)


def test_stream_stats_single_packet():
    assert stream_stats([('/analogs', [5, 0])], [0.0])['/analogs'] == (None, None, None)


# ── OSC bundles ───────────────────────────────────────────────────────────────

def test_osc_messages_plain_message():
    assert _osc_messages(_osc_build('/analogs', 3, 1, 2)) == [('/analogs', [3, 1, 2], 'iii')]


def test_osc_messages_bundle_of_two_in_order():
    b = _osc_bundle(_osc_build('/analogs', 7, 10, 20, 30, 40), _osc_build('/bumpers', 7, 0, 1, 0, 0))
    assert _osc_messages(b) == [('/analogs', [7, 10, 20, 30, 40], 'iiiii'),
                                ('/bumpers', [7, 0, 1, 0, 0], 'iiiii')]


def test_osc_messages_nested_bundle():
    inner = _osc_bundle(_osc_build('/bumpers', 1, 0))
    b = _osc_bundle(_osc_build('/analogs', 1, 5), inner, _osc_build('/wencoders', 1, -2, 3))
    assert [m[0] for m in _osc_messages(b)] == ['/analogs', '/bumpers', '/wencoders']


def test_osc_messages_truncated_bundle_keeps_complete_messages():
    b = _osc_bundle(_osc_build('/analogs', 1, 5), _osc_build('/bumpers', 1, 0))
    assert [m[0] for m in _osc_messages(b[:-6])] == ['/analogs']
    assert _osc_messages(b[:10]) == []          # cut inside the header
    assert _osc_messages(b'garbage') == []


def test_osc_thread_dispatches_every_message_of_a_bundle():
    dist = _sensor('dist', '127.0.0.1:1/analogs(1,4){x/100}')
    bump = _sensor('bump', '127.0.0.1:1/bumpers(2,3)')
    t = OscThread(_free_port(), [dist, bump])
    try:
        for addr, values, _ in _osc_messages(_osc_bundle(
                _osc_build('/analogs', 0, 100, 0, 0, 300), _osc_build('/bumpers', 0, 0, 1, 0, 0))):
            t._dispatch_message(addr, values)
    finally:
        t._sock.close()
    assert list(dist._robot_value) == [1.0, 3.0]
    assert list(bump._robot_value) == [1.0, 0.0]
    assert set(t.path_rates()) == {'/analogs', '/bumpers'}


# ── end-to-end against a fake robot on localhost ─────────────────────────────

def test_full_run_with_bundling_robot_counts_each_path():
    port = _free_port()
    stop = threading.Event()

    def robot():   # one bundle per cycle, like MKR1010_NeuroethologyOscBundle
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        i = 0
        while not stop.is_set():
            s.sendto(_osc_bundle(_osc_build('/analogs', i, 100, 200, 300, 400),
                                 _osc_build('/bumpers', i, 0, 1, 0, 0)), ('127.0.0.1', port))
            i += 1
            time.sleep(0.01)
        s.close()

    t = threading.Thread(target=robot, daemon=True)
    t.start()
    try:
        c = _circuit([_sensor('dist', f'127.0.0.1:{port}/analogs(1,4)'),
                      _sensor('bump', f'127.0.0.1:{port}/bumpers(2,3)')],
                     [_motor('motor', f'127.0.0.1:{port}/wheels')])
        check = RobotCheck(c, ping=lambda h, n, cancel: [1.0] * n)
        check.LISTEN_S = 0.5
        r = _run(check)
    finally:
        stop.set()
        t.join()
    ep = f'127.0.0.1:{port}'
    status, detail = r[f'{ep}:reply']
    assert status == PASS
    assert '/analogs: robot sent' in detail and '/bumpers: robot sent' in detail
    assert 'Sent as OSC bundles (2.0 messages per packet)' in detail
    assert r[f'{ep}:s:dist'][0] == PASS and r[f'{ep}:s:bump'][0] == PASS


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


def test_packets_queued_before_listening_are_not_counted():
    port = _free_port()
    stop = threading.Event()

    def robot():   # already streaming before the check starts, ~50 Hz
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while not stop.is_set():
            s.sendto(_osc_build('/analogs', 0, 1, 2, 3, 4), ('127.0.0.1', port))
            time.sleep(0.02)
        s.close()

    def slow_ping(host, n, cancel):   # packets queue up on the bound port meanwhile
        time.sleep(1.5)
        return [1.0] * n

    t = threading.Thread(target=robot, daemon=True)
    t.start()
    try:
        c = _circuit([_sensor('dist', f'127.0.0.1:{port}/analogs(1,4)')],
                     [_motor('motor', f'127.0.0.1:{port}/wheels')])
        check = RobotCheck(c, ping=slow_ping)
        check.LISTEN_S = 0.5
        r = _run(check)
    finally:
        stop.set()
        t.join()
    rate = float(r[f'127.0.0.1:{port}:s:dist'][1].split(' Hz')[0])
    assert rate < 70   # ~50 Hz real; counting the 1.5 s backlog would give ~200 Hz


def test_packet_loss_warns_with_sent_vs_received():
    port = _free_port()
    stop = threading.Event()

    def lossy_robot():   # counts every packet but only every 3 of 4 reach the host
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        i = 0
        while not stop.is_set():
            if i % 4 != 3:
                s.sendto(_osc_build('/analogs', i, 1, 2, 3, 4), ('127.0.0.1', port))
            i += 1
            time.sleep(0.01)
        s.close()

    t = threading.Thread(target=lossy_robot, daemon=True)
    t.start()
    try:
        c = _circuit([_sensor('dist', f'127.0.0.1:{port}/analogs(1,4)')],
                     [_motor('motor', f'127.0.0.1:{port}/wheels')])
        check = RobotCheck(c, ping=lambda h, n, cancel: [1.0] * n)
        check.LISTEN_S = 0.6
        r = _run(check)
    finally:
        stop.set()
        t.join()
    status, detail = r[f'127.0.0.1:{port}:reply']
    assert status == WARN
    assert '/analogs: robot sent' in detail and 'lost on the WiFi' in detail
    lost_pct = int(detail.split('(')[1].split('%')[0])
    assert 15 <= lost_pct <= 35


def test_slow_ping_warns():
    c = _circuit([], [_motor('motor', '127.0.0.1:1/wheels')])
    check = RobotCheck(c, ping=lambda h, n, cancel: [5.0, 400.0])
    check.LISTEN_S = 0.1
    r = _run(check)
    assert r['127.0.0.1:1:ping'][0] == WARN
