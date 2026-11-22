"""Fine-tune the fly controller on the new LPLC2 looming structure with the crash-penalty fitness
(bounce mode, fitness = cells - 5 x crashes, same budget as experiments/crash_penalty.py), so it is directly comparable
to tuning the old structure. Tests on the usual 100 unseen worlds x 60 s.
Run from the project root: python -m experiments.lplc2_tune [--fast]   -> results/lplc2/"""
import json, sys, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from flydrone.sim import run_episode
from flydrone.controllers import FlyVision
from experiments.h6_finetune import FLY, to_unit, from_unit
from experiments.crash_penalty import summary, LAMBDA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/lplc2'; OUT.mkdir(parents=True, exist_ok=True)
FAST = '--fast' in sys.argv
N_RUNS, POP, GENS, ELITE, N_TRAIN, T_TRAIN, T_TEST = 3, 16, 12, 4, 12, 30.0, 60.0
TEST = list(range(100))
if FAST: N_RUNS, POP, GENS, N_TRAIN, TEST = 1, 6, 2, 3, TEST[:8]
# old fly box without w_lgmd (unused in lplc2 mode) + the two new knobs
BOX = [b for b in FLY if b[0] not in ('w_lgmd', 'loom_k')] + [('lplc2_thr', 1.2, 4.0, 0, 2.0), ('lplc2_tau', 0.5, 15.0, 1, 3.0)]


def kw(u):
    v = {n: from_unit(x, lo, hi, lg) for x, (n, lo, hi, lg, _) in zip(u, BOX)}
    return dict(loom_mode='lplc2', tau_lp=v['tau_lp'], tau_hp=v['tau_hp'], side=(v['side_lo'], v['side_hi']), front=v['front'],
                balance_thr=v['balance_thr'], sac_small=v['sac_small'], sac_big=v['sac_big'], cool=v['cool'], slow=v['slow'],
                lplc2_thr=v['lplc2_thr'], lplc2_tau=v['lplc2_tau'])


INNATE = dict(loom_mode='lplc2', lplc2_thr=2.0, front=35)
innate_u = np.array([to_unit(d, lo, hi, lg) for _, lo, hi, lg, d in BOX])


def episode(a):
    k, s, T = a; r = run_episode(FlyVision, s, T=T, ctrl_kw=k, crash='bounce')
    return dict(cells=r['cells'], crashes=r['crashes'], dist=r['dist'])


def main():
    t0 = time.time(); lines = []; rng = np.random.default_rng(0)
    def log(m): print(m, flush=True); lines.append(m)
    raw = {'lplc2_innate': [], 'lplc2_tuned': []}; params = []
    with Pool() as pool:
        raw['lplc2_innate'] = [pool.map(episode, [(INNATE, s, T_TEST) for s in TEST])]; log('tested lplc2 innate')
        for run in range(N_RUNS):
            seeds = list(range(1000 + 50 * run, 1000 + 50 * run + N_TRAIN)); r2 = np.random.default_rng(500 + run)
            mu, sigma, best = innate_u.copy(), np.full(len(BOX), .15), (-1e9, innate_u.copy())
            for g in range(GENS):
                xs = [np.clip(mu + sigma * r2.standard_normal(len(mu)), 0, 1) for _ in range(POP)]
                res = pool.map(episode, [(kw(x), s, T_TRAIN) for x in xs for s in seeds])
                fit = [np.mean([e['cells'] - LAMBDA * e['crashes'] for e in res[i * len(seeds):(i + 1) * len(seeds)]]) for i in range(POP)]
                o = np.argsort(fit)[::-1]; el = np.array([xs[i] for i in o[:ELITE]]); mu = el.mean(0); sigma = np.maximum(el.std(0), .02)
                if fit[o[0]] > best[0]: best = (fit[o[0]], xs[o[0]].copy())
                log(f'  run{run} gen {g + 1}/{GENS}: best fitness {fit[o[0]]:.1f}')
            raw['lplc2_tuned'].append(pool.map(episode, [(kw(best[1]), s, T_TEST) for s in TEST]))
            p = {n: round(float(v), 4) for n, v in kw(best[1]).items() if not isinstance(v, (tuple, str))}
            params.append(dict(u=[float(x) for x in best[1]], params=p))
            log(f'run {run}: {summary(raw["lplc2_tuned"][-1], rng)["crashes_per_min"]:.2f} crashes/min, '
                f'{summary(raw["lplc2_tuned"][-1], rng)["cells"]:.1f} cells, params {p}')
    S = {k: summary(sum(v, []), rng) for k, v in raw.items()}
    per_run = [dict(crashes_per_min=round(summary(r, rng)['crashes_per_min'], 2), cells=round(summary(r, rng)['cells'], 1)) for r in raw['lplc2_tuned']]
    for k, v in S.items():
        log(f'{k:14s} crashes/min {v["crashes_per_min"]:.2f} [{v["crashes_ci"][0]:.2f}, {v["crashes_ci"][1]:.2f}]  crash-free {v["crash_free"] * 100:.0f}%  '
            f'cells {v["cells"]:.1f} [{v["cells_ci"][0]:.1f}, {v["cells_ci"][1]:.1f}]  fitness {v["fitness"]:.1f}')
    log(f'per run: {per_run}'); log(f'wall {time.time() - t0:.0f}s')
    tag = '_fast' if FAST else ''
    json.dump(dict(summary=S, per_run=per_run, params=params, lambda_=LAMBDA, box=[b[0] for b in BOX]),
              open(OUT / f'lplc2_tune{tag}.json', 'w'), indent=1)
    (OUT / f'lplc2_tune{tag}.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
