"""Webots (3D render, real quadcopter physics) evaluation: worlds 0..N-1 x {none, old, lplc2} x 60 s, headless.
Also writes viewer/episodes_webots.json (first 3 worlds) so the runs can be replayed in the browser viewer.
Run from the project root: python -m webots.eval [N]   -> results/webots/webots_results.json"""
import json, sys, time
from pathlib import Path
import numpy as np
from webots.run_sim import run_batch
from flydrone.world import World

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'results/webots'; OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10; MODES = ['none', 'old', 'lplc2']; T = 60
NAMES = {'none': 'no avoidance (straight)', 'old': 'fly old looming', 'lplc2': 'fly LPLC2'}


def main():
    t0 = time.time(); res = {m: [] for m in MODES}; viewer = dict(dt=None, n_eye=120, fov=330, drone_r=.3, states=['warmup', 'settling', 'fixate', 'saccade', 'saccade_start'], worlds=[])
    jobs = [(m, seed, T) for seed in range(N) for m in MODES]
    summ = run_batch(jobs)
    ents = {}
    for (m, seed, _), s in zip(jobs, summ):
        if s is None: print(seed, m, 'FAILED'); continue
        res[m].append(s); print(seed, m, s, flush=True)
        if seed < 3:
            d = json.loads((ROOT / f'webots/runs/{m}_{seed}.json').read_text()); viewer['dt'] = d['dt']
            if seed not in ents:
                w = World(seed=seed)
                ents[seed] = dict(seed=seed, size=w.size, runs={}, obstacles=[[round(float(a), 3), round(float(b), 3), round(float(r), 3), int(k), round(float(ph), 3)]
                                                                            for (a, b), r, k, ph in zip(w.c, w.r, w.k, w.ph)])
            ents[seed]['runs'][NAMES[m] + ' [Webots]'] = d['run']
    viewer['worlds'] = [ents[k] for k in sorted(ents)]
    S = {}
    rng = np.random.default_rng(0)
    for m, rs in res.items():
        pm = np.array([r['post_per_min'] for r in rs]); cl = np.array([r['cells'] for r in rs]); dm = np.array([r['dist_m'] for r in rs])
        boot = pm[rng.integers(0, len(pm), (2000, len(pm)))].mean(1)
        S[NAMES[m]] = dict(n=len(rs), post_crashes_per_min=float(pm.mean()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                           cells=float(cl.mean()), dist_m=float(dm.mean()), m_per_crash=float(dm.sum() / max(1, sum(r['post_crashes'] for r in rs))))
        print(f'{NAMES[m]:26s} post crashes/min {S[NAMES[m]]["post_crashes_per_min"]:.2f} [{S[NAMES[m]]["ci"][0]:.2f}, {S[NAMES[m]]["ci"][1]:.2f}]  '
              f'm/crash {S[NAMES[m]]["m_per_crash"]:.0f}  cells {S[NAMES[m]]["cells"]:.1f}  dist {S[NAMES[m]]["dist_m"]:.0f} m')
    print(f'wall {time.time() - t0:.0f}s')
    json.dump(dict(summary=S, runs=res), open(OUT / 'webots_results.json', 'w'), indent=1)
    json.dump(viewer, open(ROOT / 'viewer/episodes_webots.json', 'w'), separators=(',', ':'))


if __name__ == '__main__':
    main()
