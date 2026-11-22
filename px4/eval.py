"""PX4 SITL + Gazebo evaluation (host side): worlds 0..N-1 x modes x 60 s, each run on a fresh sim inside the
flybrain-px4 container (px4/run.sh). Writes results/px4/px4_results.json and viewer/episodes_px4.json (first 3 worlds).
Run from the project root: python -m px4.eval [N]"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
from px4.make_gz import world, model
from flydrone.world import World

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'results/px4'; OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
COV = '--cov' in sys.argv; T = 180 if COV else 60; TAG = '_cov' if COV else ''
MODES = ({'none-cov-v1.0': 'planner, no avoidance [PX4]', 'old-cov-v1.0': 'planner + fly old [PX4]', 'lplc2-cov-v1.0': 'planner + fly LPLC2 [PX4]'} if COV else
         {'none-v1.0': 'no avoidance (straight) [PX4]', 'old-v1.0': 'fly old looming [PX4]', 'lplc2-v1.0': 'fly LPLC2 [PX4]'})


def main():
    t0 = time.time(); model(); res = {m: [] for m in MODES}; viewer = dict(dt=1 / 30, n_eye=120, fov=330, drone_r=.35, cells=dict(margin=2.0, cell=2.0) if COV else None,
                                                                      states=['warmup', 'settling', 'fixate', 'saccade', 'saccade_start'], worlds=[])
    for seed in range(N):
        world(seed); ent = None
        for m in MODES:
            out = ROOT / f'px4/runs/{m}_{seed}.json'
            if out.exists(): out.unlink()
            for attempt in range(2):
                subprocess.run(['docker', 'exec', 'flybrain-px4', 'bash', '/work/px4/run.sh', m, str(seed), str(T)], capture_output=True, timeout=T * 4 + 300)
                if out.exists(): break
            if not out.exists(): print(seed, m, 'FAILED', flush=True); continue
            d = json.loads(out.read_text()); res[m].append(d['summary']); print(seed, m, d['summary'], f'{time.time() - t0:.0f}s', flush=True)
            if seed < 3:
                if ent is None:
                    w = World(seed=seed)
                    ent = dict(seed=seed, size=w.size, runs={}, obstacles=[[round(float(a), 3), round(float(b), 3), round(float(r), 3), int(k), round(float(ph), 3)]
                                                                         for (a, b), r, k, ph in zip(w.c, w.r, w.k, w.ph)])
                ent['runs'][MODES[m]] = d['run']
        if ent: viewer['worlds'].append(ent)
        json.dump(viewer, open(ROOT / f'viewer/episodes_px4{TAG}.json', 'w'), separators=(',', ':'))
    rng = np.random.default_rng(0); S = {}
    for m, rs in res.items():
        if not rs: continue
        pm = np.array([r['post_per_min'] for r in rs]); dm = np.array([r['dist_m'] for r in rs])
        b = pm[rng.integers(0, len(pm), (2000, len(pm)))].mean(1)
        S[MODES[m]] = dict(n=len(rs), post_per_min=float(pm.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                           dist_m=float(dm.mean()), m_per_crash=float(dm.sum() / max(1, sum(r['post_crashes'] for r in rs))),
                           crash_free=float(np.mean([r['post_crashes'] == 0 for r in rs])), geofence_per_min=float(np.mean([r.get('geofence_events', 0) for r in rs])),
                           coverage=float(np.mean([r.get('coverage', 0) for r in rs])), cells=float(np.mean([r['cells'] for r in rs])))
        print(f'{MODES[m]:34s} post crashes/min {S[MODES[m]]["post_per_min"]:.2f} [{S[MODES[m]]["ci"][0]:.2f}, {S[MODES[m]]["ci"][1]:.2f}]  '
              f'coverage {S[MODES[m]]["coverage"]:.1f}%  m/crash {S[MODES[m]]["m_per_crash"]:.0f}  crash-free {S[MODES[m]]["crash_free"] * 100:.0f}%  dist {S[MODES[m]]["dist_m"]:.0f} m')
    print(f'wall {time.time() - t0:.0f}s')
    json.dump(dict(summary=S, runs=res), open(OUT / f'px4_results{TAG}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
