"""Goal-layer settings (goal_gap 2.0, goal_thr 15 deg, hold 1 s, 2 m/s) picked on validation worlds 200-239.
Goal reaching (100 worlds x 60 s) and area coverage (40 worlds x 180 s) with a GPS-like navigation layer on top of
each low-level controller. Run from the project root: python -m experiments.nav_eval   -> results/nav/"""
import json, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from flydrone.nav_sim import run_nav

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/nav'; OUT.mkdir(parents=True, exist_ok=True)
CFG = {'beeline (no avoidance)': ('none', {}, False),
       'fly old + nav': ('fly', dict(goal_gap=2.0), False),
       'LPLC2 + nav (direct goal turns)': ('fly', dict(loom_mode='lplc2', lplc2_thr=2.0, front=35), False),
       'LPLC2 + nav (gap-aware)': ('fly', dict(loom_mode='lplc2', lplc2_thr=2.0, front=35, goal_gap=2.0), False),
       'oracle + nav': ('oracle', {}, False)}
COV_EXTRA = {'LPLC2 thr2.0 wander (no nav)': ('fly', dict(loom_mode='lplc2', lplc2_thr=2.0, front=35), True)}
GOAL_W, COV_W, T_GOAL, T_COV = list(range(100)), list(range(40)), 60.0, 180.0


def ep(a):
    name, task, s, cfg = a; base, kw, wander = cfg
    r = run_nav(base, kw, s, task=task, T=T_GOAL if task == 'goal' else T_COV, wander=wander); r.pop('world'); return name, r


def ci(x, rng, B=2000):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    t0 = time.time(); rng = np.random.default_rng(0); res = {}
    jobs = [(n, 'goal', s, c) for n, c in CFG.items() for s in GOAL_W] + [(n, 'cov', s, c) for n, c in {**CFG, **COV_EXTRA}.items() for s in COV_W]
    with Pool() as P: out = P.map(ep, jobs, chunksize=4)
    for (n, task, s, _), (_, r) in zip(jobs, out): res.setdefault(task, {}).setdefault(n, []).append(r)
    S = {'goal': {}, 'coverage': {}}
    print('GOAL REACHING (100 worlds, 60 s)')
    for n, rs in res['goal'].items():
        ok = [r['reached'] for r in rs]; tt = [r['time'] for r in rs if r['reached']]
        eff = [r['straight'] / max(r['dist'], 1e-6) for r in rs if r['reached']]
        pm = [r['posts'] * 60 / r['t'] for r in rs]
        S['goal'][n] = dict(success=float(np.mean(ok)), success_ci=ci(ok, rng), time=float(np.median(tt)) if tt else None,
                            path_eff=float(np.median(eff)) if eff else None, posts_per_min=float(np.mean(pm)), posts_ci=ci(pm, rng),
                            no_post_crash=float(np.mean([r['posts'] == 0 for r in rs])))
        g = S['goal'][n]; print(f'  {n:30s} reached {g["success"] * 100:3.0f}%  median time {g["time"] or 0:5.1f}s  path eff {g["path_eff"] or 0:.2f}  '
                               f'post crashes/min {g["posts_per_min"]:.2f} [{g["posts_ci"][0]:.2f},{g["posts_ci"][1]:.2f}]  no-post-crash {g["no_post_crash"] * 100:.0f}%', flush=True)
    print('COVERAGE (40 worlds, 180 s, 324 cells of 2x2 m inside the geofence)')
    for n, rs in res['cov'].items():
        cv = [r['coverage'] * 100 for r in rs]; pm = [r['posts'] * 60 / r['t'] for r in rs]; wm = [r['walls'] * 60 / r['t'] for r in rs]
        S['coverage'][n] = dict(coverage=float(np.mean(cv)), coverage_ci=ci(cv, rng), posts_per_min=float(np.mean(pm)), posts_ci=ci(pm, rng),
                                walls_per_min=float(np.mean(wm)), no_post_crash=float(np.mean([r['posts'] == 0 for r in rs])))
        c = S['coverage'][n]; print(f'  {n:30s} coverage {c["coverage"]:5.1f}% [{c["coverage_ci"][0]:.1f},{c["coverage_ci"][1]:.1f}]  '
                                   f'post crashes/min {c["posts_per_min"]:.2f} [{c["posts_ci"][0]:.2f},{c["posts_ci"][1]:.2f}]  walls/min {c["walls_per_min"]:.2f}  '
                                   f'no-post-crash {c["no_post_crash"] * 100:.0f}%', flush=True)
    print(f'wall time {time.time() - t0:.0f}s')
    json.dump(S, open(OUT / 'nav_results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
