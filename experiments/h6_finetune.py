"""H6 'innate prior + fine-tune' experiment (and H1-H4 comparisons on held-out worlds).

Both learners get the same budget: CEM, POP candidates x GENS generations x len(TRAIN) episodes.
- fly_tuned: the fly circuit structure, CEM tunes its 12 parameters starting from the innate defaults.
- linear_scratch: no structure, CEM tunes 98 weights of a linear read-out of the same EMD signals from zero.
Train worlds and test worlds never overlap. Main metric: 2x2 m cells explored before the first collision. Results -> results/h6/h6_results.json. Run: python -m experiments.h6_finetune [--fast]
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
from multiprocessing import Pool
from flydrone.sim import evaluate
from flydrone.controllers import Random, Oracle, FlyVision, Linear

TRAIN = list(range(100, 112)); TEST = list(range(0, 30))
POP, GENS, ELITE, T_TRAIN, T_TEST = 16, 12, 4, 30.0, 60.0
FAST = '--fast' in sys.argv
if FAST: POP, GENS, TRAIN, TEST = 8, 3, TRAIN[:4], TEST[:6]

# fly parameter box: (name, lo, hi, log-scale, innate default)
FLY = [('tau_lp', .005, .08, 1, .035), ('tau_hp', .05, 1.0, 1, .25), ('side_lo', 0, 40, 0, 10), ('side_hi', 60, 160, 0, 110),
       ('front', 10, 60, 0, 35), ('balance_thr', .05, .8, 0, .25), ('loom_k', 1.2, 6, 0, 2.5), ('sac_small', 10, 70, 0, 35),
       ('sac_big', 40, 150, 0, 90), ('cool', .1, 1.0, 0, .35), ('w_lgmd', 0, 3, 0, 0), ('slow', .2, 1.0, 0, .5)]


def to_unit(v, lo, hi, lg): return (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if lg else (v - lo) / (hi - lo)
def from_unit(u, lo, hi, lg):
    u = float(np.clip(u, 0, 1)); return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))) if lg else lo + u * (hi - lo)


def fly_kw(u):
    v = {n: from_unit(x, lo, hi, lg) for x, (n, lo, hi, lg, _) in zip(u, FLY)}
    return dict(tau_lp=v['tau_lp'], tau_hp=v['tau_hp'], side=(v['side_lo'], v['side_hi']), front=v['front'],
                balance_thr=v['balance_thr'], loom_k=v['loom_k'], sac_small=v['sac_small'], sac_big=v['sac_big'],
                cool=v['cool'], w_lgmd=v['w_lgmd'], slow=v['slow'])


def score(args):
    kind, x, seeds, T = args
    if kind == 'fly': return evaluate(FlyVision, seeds, T=T, ctrl_kw=fly_kw(x))['cells']
    return evaluate(Linear, seeds, T=T, ctrl_kw=dict(w=x))['cells']


def cem(kind, mu, sigma, pool, log):
    rng = np.random.default_rng(1); best = (-1, mu.copy()); curve = []
    for g in range(GENS):
        xs = [mu + sigma * rng.standard_normal(len(mu)) for _ in range(POP)]
        if kind == 'fly': xs = [np.clip(x, 0, 1) for x in xs]
        fit = pool.map(score, [(kind, x, TRAIN, T_TRAIN) for x in xs])
        order = np.argsort(fit)[::-1]; elite = np.array([xs[i] for i in order[:ELITE]])
        mu = elite.mean(0); sigma = np.maximum(elite.std(0), 0.02 if kind == 'fly' else 0.01)
        if fit[order[0]] > best[0]: best = (fit[order[0]], xs[order[0]].copy())
        curve.append(float(fit[order[0]])); log(f'{kind} gen {g + 1}/{GENS}: best {fit[order[0]]:.1f} cells, mean {np.mean(fit):.1f}')
    return best[1], curve


def main():
    t0 = time.time(); lines = []
    def log(m): print(m, flush=True); lines.append(m)
    with Pool() as pool:
        innate_u = np.array([to_unit(d, lo, hi, lg) for _, lo, hi, lg, d in FLY])
        log(f'innate fly on train: {score(("fly", innate_u, TRAIN, T_TRAIN)):.1f} cells')
        fly_u, fly_curve = cem('fly', innate_u.copy(), np.full(len(FLY), .15), pool, log)
        lin_w, lin_curve = cem('lin', np.zeros(2 * (2 * Linear.NB + 1)), np.full(2 * (2 * Linear.NB + 1), .3), pool, log)
        tests = {'no_avoidance': (Random, {}), 'oracle_depth': (Oracle, {}),
                 'fly_innate_saccade': (FlyVision, {}), 'fly_innate_continuous': (FlyVision, {'saccadic': False}),
                 'fly_tuned_saccade': (FlyVision, fly_kw(fly_u)), 'linear_scratch_tuned': (Linear, {'w': lin_w})}
        res = dict(zip(tests, pool.starmap(_eval, [(c, kw) for c, kw in tests.values()])))
    for k, v in res.items(): log(f'TEST {k:24s} cells {v["cells"]:5.1f}  survival {v["survival_time"]:5.1f}s  collisions {v["collision_rate"] * 100:3.0f}%  m/collision {v["dist_per_collision"]:6.1f}')
    tuned = {n: round(from_unit(x, lo, hi, lg), 4) for x, (n, lo, hi, lg, _) in zip(fly_u, FLY)}
    log(f'tuned fly params: {tuned}'); log(f'wall time {time.time() - t0:.0f}s')
    out = dict(test=res, fly_curve=fly_curve, lin_curve=lin_curve, tuned_params=tuned, budget=dict(pop=POP, gens=GENS, train=TRAIN, test=TEST, T_train=T_TRAIN, T_test=T_TEST))
    out['lin_w'] = [float(v) for v in lin_w]
    json.dump(out, open(ROOT / 'results/h6' / ('h6_results' + ('_fast' if FAST else '') + '.json'), 'w'), indent=1)
    np.save(ROOT / 'results/h6' / ('fly_tuned_u' + ('_fast' if FAST else '') + '.npy'), fly_u)


def _eval(cls, kw): return evaluate(cls, TEST, T=T_TEST, ctrl_kw=kw)


if __name__ == '__main__':
    main()
