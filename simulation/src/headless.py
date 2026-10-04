"""
headless.py — run a saved session without the GUI.

From simulation/2d/:

    python src/headless.py "configs/Tutorials/T03 - ColisionSession.json" --seconds 60 --out run.csv
    python src/headless.py <session> --seconds 60 --seed 1          # repeatable run

Every agent is driven by its own brain (no keyboard / network). The CSV has
one row per agent per step: step, t, agent, x, y, theta, mL, mR and every
non-camera sensor reading — the values Simulation.step() returns.

From Python (e.g. a parameter sweep):

    from headless import run
    sim = run('configs/X.json', seconds=30, seed=0,
              on_step=lambda sim, raws: ...)    # called after every step
"""

import csv
import os
import random
import sys

_SIM2D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _setup_paths():
    """Same layout the app uses: run from simulation/2d/ with src/ and brains/
    importable (session files reference brains/, networks/ and configs/)."""
    os.chdir(_SIM2D)
    for p in (os.path.join(_SIM2D, 'src'), _SIM2D):
        if p not in sys.path:
            sys.path.insert(0, p)
    import data_paths
    for p in reversed(data_paths.search_dirs('brains')):   # both roots, the user's first
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(session_path, seconds, seed=None, out=None, on_step=None):
    """Load *session_path* and simulate *seconds* of simulated time.

    seed    : seeds Python / numpy / torch before loading (weight init) and running
    out     : optional CSV path for the per-agent values of every step
    on_step : optional callable(sim, raws) after every step
    Returns the Simulation, so callers can inspect the final state.

    Cameras with fps == 0 ("as fast as possible") render after every step
    here — there is no display loop to pace them."""
    _setup_paths()
    from session_loader import build_simulation

    session_path = os.path.abspath(session_path) if not os.path.isabs(session_path) else session_path
    if seed is not None:
        seed_everything(seed)
    sim = build_simulation(session_path)
    n_steps = int(round(seconds / sim.sim_cfg.dt))

    writer, fh, fields = None, None, None
    try:
        for _ in range(n_steps):
            raws = sim.step()
            sim.render_free_running_cameras()
            if on_step is not None:
                on_step(sim, raws)
            if out is not None:
                rows = [{'step': sim.time_index, 't': round(sim.sim_time, 9), 'agent': i,
                         'x': a.bot_pos[0], 'y': a.bot_pos[1], 'theta': a.bot_pos[2], **raw}
                        for i, (a, raw) in enumerate(zip(sim.agents, raws))]
                if writer is None:
                    fields = list(dict.fromkeys(k for row in rows for k in row))
                    fh = open(out, 'w', newline='')
                    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
                    writer.writeheader()
                writer.writerows(rows)
    finally:
        if fh is not None:
            fh.close()
        sim.close()
    return sim


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Run a simulator session without the GUI.')
    ap.add_argument('session', help='session JSON (e.g. configs/Tutorials/T03 - ColisionSession.json)')
    ap.add_argument('--seconds', type=float, required=True, help='simulated seconds to run')
    ap.add_argument('--seed', type=int, default=None, help='random seed for a repeatable run')
    ap.add_argument('--out', default=None, help='CSV file for per-agent values of every step')
    args = ap.parse_args(argv)

    session = os.path.abspath(args.session)
    out = os.path.abspath(args.out) if args.out else None
    sim = run(session, args.seconds, seed=args.seed, out=out)
    print(f'{os.path.basename(session)}: {sim.time_index} steps, {sim.sim_time:.2f} s simulated, '
          f'{len(sim.agents)} agent(s)')
    for i, a in enumerate(sim.agents):
        print(f'  agent {i}: x={a.bot_pos[0]:.3f} y={a.bot_pos[1]:.3f} theta={a.bot_pos[2]:.3f}')
    if out:
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
