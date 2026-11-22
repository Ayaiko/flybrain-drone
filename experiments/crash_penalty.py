"""Retrain with a fitness that punishes crashes. Episodes no longer end at the first crash ('bounce' mode:
the drone is turned around and keeps flying), and fitness = cells explored - LAMBDA * crashes.
Compares, on 100 unseen worlds in the same bounce mode:
  fly tuned for exploration only (the 3 runs saved by robust_eval)  vs  fly tuned with the crash penalty  vs
  linear from scratch with the crash penalty, plus the untrained baselines.
Run from the project root: python -m experiments.crash_penalty [--fast]   -> results/crash/"""
import json, sys, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from flydrone.sim import run_episode
from flydrone.controllers import Random, Oracle, FlyVision, Linear
from experiments.h6_finetune import FLY, fly_kw, to_unit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/crash'; OUT.mkdir(parents=True, exist_ok=True)
FAST = '--fast' in sys.argv
LAMBDA = 5.0
N_RUNS, POP, GENS, ELITE, N_TRAIN, T_TRAIN, T_TEST = 3, 16, 12, 4, 12, 30.0, 60.0
TEST = list(range(0, 100))
if FAST: N_RUNS, POP, GENS, N_TRAIN, TEST = 2, 6, 2, 3, TEST[:8]
CLS = {'random': Random, 'oracle': Oracle, 'fly': FlyVision, 'lin': Linear}


def episode(args):
    kind, kw, seed, T = args
    r = run_episode(CLS[kind], seed, T=T, ctrl_kw=kw, crash='bounce')
    return dict(cells=r['cells'], crashes=r['crashes'], dist=r['dist'])


def fitness(eps): return float(np.mean([e['cells'] - LAMBDA * e['crashes'] for e in eps]))


def cem(pool, kind, mu, sigma, seeds, rng, log, tag):
    best = (-1e9, mu.copy())
    for g in range(GENS):
        xs = [mu + sigma * rng.standard_normal(len(mu)) for _ in range(POP)]
        if kind == 'fly': xs = [np.clip(x, 0, 1) for x in xs]
        res = pool.map(episode, [(kind, fly_kw(x) if kind == 'fly' else {'w': x}, s, T_TRAIN) for x in xs for s in seeds])
        fit = [fitness(res[i * len(seeds):(i + 1) * len(seeds)]) for i in range(len(xs))]
        order = np.argsort(fit)[::-1]; elite = np.array([xs[i] for i in order[:ELITE]]); mu = elite.mean(0)
        sigma = np.maximum(elite.std(0), .02 if kind == 'fly' else .01)
        if fit[order[0]] > best[0]: best = (fit[order[0]], xs[order[0]].copy())
        log(f'  {tag} {kind} gen {g + 1}/{GENS}: best fitness {fit[order[0]]:.1f}')
    return best[1]


def test(pool, kind, kw): return pool.map(episode, [(kind, kw, s, T_TEST) for s in TEST])


def summary(eps, rng, B=2000):
    cells = np.array([e['cells'] for e in eps]); cr = np.array([e['crashes'] for e in eps], float); d = np.array([e['dist'] for e in eps])
    n = len(eps); idx = rng.integers(0, n, (B, n)); ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return dict(n=n, cells=float(cells.mean()), cells_ci=ci(cells[idx].mean(1)),
                crashes_per_min=float(cr.mean() * 60 / T_TEST), crashes_ci=ci(cr[idx].mean(1) * 60 / T_TEST),
                m_per_crash=float(d.sum() / max(1, cr.sum())), crash_free=float((cr == 0).mean()),
                fitness=float((cells - LAMBDA * cr).mean()))


def main():
    t0 = time.time(); lines = []; rng = np.random.default_rng(0)
    def log(m): print(m, flush=True); lines.append(m)
    raw = {}; old = json.load(open(ROOT / 'results/robust/robust_results.json'))['params']
    with Pool() as pool:
        for name, kind, kw in [('no_avoidance', 'random', {}), ('oracle_depth', 'oracle', {}),
                               ('fly_innate_saccade', 'fly', {}), ('fly_innate_continuous', 'fly', {'saccadic': False})]:
            raw[name] = [test(pool, kind, kw)]; log(f'tested {name}')
        raw['fly_tuned_explore_only'] = [test(pool, 'fly', fly_kw(np.array(p['fly_u']))) for p in old[:N_RUNS]]
        log('tested fly tuned for exploration only (previous runs)')
        raw['fly_tuned_crash_penalty'], raw['linear_crash_penalty'], params = [], [], []
        innate_u = np.array([to_unit(d, lo, hi, lg) for _, lo, hi, lg, d in FLY])
        for run in range(N_RUNS):
            seeds = list(range(1000 + 50 * run, 1000 + 50 * run + N_TRAIN))
            fu = cem(pool, 'fly', innate_u.copy(), np.full(len(FLY), .15), seeds, np.random.default_rng(300 + run), log, f'run{run}')
            lw = cem(pool, 'lin', np.zeros(2 * (2 * Linear.NB + 1)), np.full(2 * (2 * Linear.NB + 1), .3), seeds,
                     np.random.default_rng(400 + run), log, f'run{run}')
            raw['fly_tuned_crash_penalty'].append(test(pool, 'fly', fly_kw(fu)))
            raw['linear_crash_penalty'].append(test(pool, 'lin', {'w': lw}))
            params.append(dict(fly_u=[float(v) for v in fu], lin_w=[float(v) for v in lw],
                               fly_params={n: round(float(v), 4) for n, v in fly_kw(fu).items() if not isinstance(v, tuple)}))
            log(f'run {run}: fly crash-penalty {summary(raw["fly_tuned_crash_penalty"][-1], rng)["crashes_per_min"]:.2f} crashes/min, '
                f'linear {summary(raw["linear_crash_penalty"][-1], rng)["crashes_per_min"]:.2f}')
    res = {k: summary(sum(v, []), rng) for k, v in raw.items()}
    per_run = {k: [round(summary(r, rng)['crashes_per_min'], 2) for r in v] for k, v in raw.items() if len(v) > 1}
    for k, v in res.items():
        log(f'{k:26s} crashes/min {v["crashes_per_min"]:5.2f} [{v["crashes_ci"][0]:.2f}, {v["crashes_ci"][1]:.2f}]  '
            f'm/crash {v["m_per_crash"]:7.1f}  crash-free {v["crash_free"] * 100:3.0f}%  cells {v["cells"]:5.1f} '
            f'[{v["cells_ci"][0]:.1f}, {v["cells_ci"][1]:.1f}]  fitness {v["fitness"]:6.1f}')
    log(f'per-run crashes/min: {per_run}'); log(f'wall time {time.time() - t0:.0f}s')
    tag = '_fast' if FAST else ''
    json.dump(dict(summary=res, per_run_crashes_per_min=per_run, params=params, lambda_=LAMBDA,
                   budget=dict(runs=N_RUNS, pop=POP, gens=GENS, n_train=N_TRAIN, test=len(TEST), T_train=T_TRAIN, T_test=T_TEST)),
              open(OUT / f'crash_results{tag}.json', 'w'), indent=1)
    (OUT / f'crash_run{tag}.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
