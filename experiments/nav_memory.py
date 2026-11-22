"""Planner with a hazard memory (where looming fired / where it crashed) for area coverage.
Heading veto (veto_r 2/3/4 m, +- post-turn slowdown) was swept on validation 200-239 separately: veto 4 m best, slowdown hurts.
Phase 1: pick settings on validation worlds 200-239. Phase 2: test on coverage worlds 0-39 (same as nav_eval), 180 s.
Run from the project root: python -m experiments.nav_memory   -> results/nav/nav_memory.json"""
import json, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from flydrone.nav_sim import run_nav
from experiments.nav_eval import ci

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'results/nav'
L2 = dict(loom_mode='lplc2', lplc2_thr=2.0, front=35, goal_gap=2.0); OLD = dict(goal_gap=2.0)
VAL = {f'LPLC2 mem d{d} line{int(lc)}': ('fly', L2, dict(memory=True, loom_d=d, line_check=lc)) for d in (1.5, 2.5) for lc in (False, True)}
VAL.update({f'old mem d{d}': ('fly', OLD, dict(memory=True, loom_d=d)) for d in (1.5, 2.5)})
VAL.update({'LPLC2 no mem': ('fly', L2, {}), 'old no mem': ('fly', OLD, {})})


def ep(a):
    n, s, cfg = a; base, kw, nk = cfg; r = run_nav(base, kw, s, task='cov', T=180.0, **nk); return n, r['coverage'] * 100, r['posts'] * 60 / r['t'], r['hazards']


def run(pool, cfgs, worlds):
    out = pool.map(ep, [(n, s, c) for n, c in cfgs.items() for s in worlds], chunksize=2); R = {}
    for n, cv, pm, hz in out: R.setdefault(n, []).append((cv, pm, hz))
    return {n: np.array(v) for n, v in R.items()}


def main():
    t0 = time.time(); rng = np.random.default_rng(0)
    with Pool() as P:
        print('VALIDATION (worlds 200-239)')
        V = run(P, VAL, range(200, 240))
        for n, a in V.items(): print(f'  {n:24s} coverage {a[:, 0].mean():5.1f}%  post crashes/min {a[:, 1].mean():5.2f}  hazards {a[:, 2].mean():.0f}', flush=True)
        def pick(prefix):
            c = {n: a for n, a in V.items() if n.startswith(prefix) and 'mem d' in n}
            return max(c, key=lambda n: c[n][:, 0].mean() - 5 * c[n][:, 1].mean())   # same trade-off as before: 1 crash/min = 5% coverage
        best_l2, best_old = pick('LPLC2'), pick('old')
        print('picked', best_l2, '|', best_old)
        TEST = {'LPLC2 no memory': VAL['LPLC2 no mem'], 'LPLC2 + hazard memory': VAL[best_l2],
                'LPLC2 + memory + heading veto 4 m': ('fly', {**L2, 'veto_r': 4.0}, dict(memory=True, loom_d=2.5)),
                'fly old no memory': VAL['old no mem'], 'fly old + hazard memory': VAL[best_old], 'oracle + nav': ('oracle', {}, {})}
        print('TEST (worlds 0-39, 180 s)')
        Tr = run(P, TEST, range(40)); S = {}
        for n, a in Tr.items():
            S[n] = dict(coverage=float(a[:, 0].mean()), coverage_ci=ci(a[:, 0], rng), posts_per_min=float(a[:, 1].mean()), posts_ci=ci(a[:, 1], rng),
                        hazards=float(a[:, 2].mean()))
            print(f'  {n:24s} coverage {S[n]["coverage"]:5.1f}% [{S[n]["coverage_ci"][0]:.1f},{S[n]["coverage_ci"][1]:.1f}]  '
                  f'post crashes/min {S[n]["posts_per_min"]:5.2f} [{S[n]["posts_ci"][0]:.2f},{S[n]["posts_ci"][1]:.2f}]', flush=True)
    print(f'wall {time.time() - t0:.0f}s')
    json.dump(dict(validation={n: dict(coverage=float(a[:, 0].mean()), posts_per_min=float(a[:, 1].mean())) for n, a in V.items()},
                   picked=[best_l2, best_old], test=S), open(OUT / 'nav_memory.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
