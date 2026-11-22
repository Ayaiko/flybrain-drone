"""Robust version of the H6 comparison: N_RUNS independent trainings (different CEM seed AND different
training worlds), 100 held-out test worlds, bootstrap 95% CIs, several metrics, paired fly-vs-linear test.
Run from the project root: python -m experiments.robust_eval [--fast]   -> results/robust/"""
import json, sys, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from flydrone.sim import run_episode
from flydrone.controllers import Random, Oracle, FlyVision, Linear
from experiments.h6_finetune import FLY, fly_kw, to_unit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/robust'; OUT.mkdir(parents=True, exist_ok=True)
FAST = '--fast' in sys.argv
N_RUNS, POP, GENS, ELITE, N_TRAIN, T_TRAIN, T_TEST = 3, 16, 12, 4, 12, 30.0, 60.0
TEST = list(range(0, 100))
if FAST: N_RUNS, POP, GENS, N_TRAIN, TEST = 2, 6, 2, 3, TEST[:8]
CLS = {'random': Random, 'oracle': Oracle, 'fly': FlyVision, 'lin': Linear}


def episode(args):
    kind, kw, seed, T = args
    r = run_episode(CLS[kind], seed, T=T, ctrl_kw=kw)
    return dict(cells=r['cells'], collided=bool(r['collided']), dist=r['dist'], time=r['time'])


def train_score(pool, kind, xs, seeds):
    jobs = [(kind, fly_kw(x) if kind == 'fly' else {'w': x}, s, T_TRAIN) for x in xs for s in seeds]
    res = pool.map(episode, jobs)
    return [np.mean([r['cells'] for r in res[i * len(seeds):(i + 1) * len(seeds)]]) for i in range(len(xs))]


def cem(pool, kind, mu, sigma, seeds, rng, log, tag):
    best = (-1, mu.copy())
    for g in range(GENS):
        xs = [mu + sigma * rng.standard_normal(len(mu)) for _ in range(POP)]
        if kind == 'fly': xs = [np.clip(x, 0, 1) for x in xs]
        fit = train_score(pool, kind, xs, seeds); order = np.argsort(fit)[::-1]
        elite = np.array([xs[i] for i in order[:ELITE]]); mu = elite.mean(0)
        sigma = np.maximum(elite.std(0), .02 if kind == 'fly' else .01)
        if fit[order[0]] > best[0]: best = (fit[order[0]], xs[order[0]].copy())
        log(f'  {tag} {kind} gen {g + 1}/{GENS}: best {fit[order[0]]:.1f}')
    return best[1]


def test(pool, kind, kw):
    return pool.map(episode, [(kind, kw, s, T_TEST) for s in TEST])


def summary(eps, rng, B=2000):
    cells = np.array([e['cells'] for e in eps]); col = np.array([e['collided'] for e in eps], float)
    dist = np.array([e['dist'] for e in eps]); n = len(eps)
    idx = rng.integers(0, n, (B, n))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    mpc = lambda d, c: d.sum() / max(1, c.sum())
    return dict(n=n, cells=float(cells.mean()), cells_ci=ci(cells[idx].mean(1)),
                collision_rate=float(col.mean()), collision_rate_ci=ci(col[idx].mean(1)),
                m_per_collision=float(mpc(dist, col)), m_per_collision_ci=ci([mpc(dist[i], col[i]) for i in idx[:500]]),
                survival=float(np.mean([e['time'] for e in eps])))


def main():
    t0 = time.time(); lines = []; rng = np.random.default_rng(0)
    def log(m): print(m, flush=True); lines.append(m)
    raw = {}
    with Pool() as pool:
        for name, kind, kw in [('no_avoidance', 'random', {}), ('oracle_depth', 'oracle', {}),
                               ('fly_innate_saccade', 'fly', {}), ('fly_innate_continuous', 'fly', {'saccadic': False})]:
            raw[name] = [test(pool, kind, kw)]; log(f'tested {name}')
        raw['fly_tuned_saccade'], raw['linear_scratch_tuned'], params = [], [], []
        innate_u = np.array([to_unit(d, lo, hi, lg) for _, lo, hi, lg, d in FLY])
        for run in range(N_RUNS):
            seeds = list(range(1000 + 50 * run, 1000 + 50 * run + N_TRAIN))      # different training worlds per run
            fu = cem(pool, 'fly', innate_u.copy(), np.full(len(FLY), .15), seeds, np.random.default_rng(100 + run), log, f'run{run}')
            lw = cem(pool, 'lin', np.zeros(2 * (2 * Linear.NB + 1)), np.full(2 * (2 * Linear.NB + 1), .3), seeds,
                     np.random.default_rng(200 + run), log, f'run{run}')
            raw['fly_tuned_saccade'].append(test(pool, 'fly', fly_kw(fu)))
            raw['linear_scratch_tuned'].append(test(pool, 'lin', {'w': lw}))
            params.append(dict(fly_u=[float(v) for v in fu], lin_w=[float(v) for v in lw]))
            log(f'run {run}: fly tuned {np.mean([e["cells"] for e in raw["fly_tuned_saccade"][-1]]):.1f} cells, '
                f'linear {np.mean([e["cells"] for e in raw["linear_scratch_tuned"][-1]]):.1f} cells')
    res = {k: summary(sum(v, []), rng) for k, v in raw.items()}
    res_runs = {k: [summary(r, rng)['cells'] for r in v] for k, v in raw.items() if len(v) > 1}
    # paired per-world difference fly tuned - linear (averaged over runs), bootstrap over worlds
    f = np.mean([[e['cells'] for e in r] for r in raw['fly_tuned_saccade']], 0)
    l = np.mean([[e['cells'] for e in r] for r in raw['linear_scratch_tuned']], 0)
    d = f - l; idx = rng.integers(0, len(d), (4000, len(d))); bd = d[idx].mean(1)
    paired = dict(mean_diff=float(d.mean()), ci=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                  worlds_fly_better=float((d > 0).mean()), worlds_tie=float((d == 0).mean()))
    for k, v in res.items():
        log(f'{k:24s} cells {v["cells"]:5.1f} [{v["cells_ci"][0]:.1f}, {v["cells_ci"][1]:.1f}]  '
            f'collisions {v["collision_rate"] * 100:4.0f}% [{v["collision_rate_ci"][0] * 100:.0f}, {v["collision_rate_ci"][1] * 100:.0f}]  '
            f'm/collision {v["m_per_collision"]:7.1f}  survival {v["survival"]:4.1f}s')
    log(f'per-run cells: {res_runs}')
    log(f'paired fly_tuned - linear: {paired["mean_diff"]:+.1f} cells, 95% CI [{paired["ci"][0]:+.1f}, {paired["ci"][1]:+.1f}], '
        f'fly better on {paired["worlds_fly_better"] * 100:.0f}% of worlds')
    log(f'wall time {time.time() - t0:.0f}s')
    tag = '_fast' if FAST else ''
    json.dump(dict(summary=res, per_run_cells=res_runs, paired=paired, params=params,
                   budget=dict(runs=N_RUNS, pop=POP, gens=GENS, n_train=N_TRAIN, test=len(TEST), T_train=T_TRAIN, T_test=T_TEST)),
              open(OUT / f'robust_results{tag}.json', 'w'), indent=1)
    (OUT / f'robust_run{tag}.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
