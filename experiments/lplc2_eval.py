"""Structural fix from the crash analysis: 'lplc2' looming mode (left/right frontal looming units, threshold relative to a
slowly adapting baseline instead of the old max-ratchet, escape away from the expanding side).
Threshold/front width chosen on validation worlds 200-239 (sweep logged in EXPERIMENTS.md); tested here on the
usual 100 unseen worlds 0-99 x 60 s in bounce mode, with ablations to see which part matters.
Run from the project root: python -m experiments.lplc2_eval   -> results/lplc2/"""
import json, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from flydrone.sim import run_episode
from flydrone.controllers import Random, Oracle, FlyVision

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/lplc2'; OUT.mkdir(parents=True, exist_ok=True)
TEST = list(range(100)); T = 60.0
CFG = {'no_avoidance': (Random, {}),
       'fly_old_looming': (FlyVision, {}),
       'lplc2_direction_only': (FlyVision, dict(loom_mode='lplc2', lplc2_base=False)),
       'lplc2_threshold_only': (FlyVision, dict(loom_mode='lplc2', lplc2_dir=False)),
       'lplc2_full': (FlyVision, dict(loom_mode='lplc2', lplc2_thr=2.0, front=35)),
       'lplc2_full_bolder': (FlyVision, dict(loom_mode='lplc2', lplc2_thr=2.5, front=35)),
       'oracle_depth': (Oracle, {})}


def ep(a):
    name, s = a; cls, kw = CFG[name]; r = run_episode(cls, s, T=T, ctrl_kw=kw, crash='bounce')
    return dict(crashes=r['crashes'], cells=r['cells'], dist=r['dist'])


def main():
    t0 = time.time(); rng = np.random.default_rng(0); B = 2000
    with Pool() as P: raw = {n: P.map(ep, [(n, s) for s in TEST]) for n in CFG}
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    idx = rng.integers(0, len(TEST), (B, len(TEST))); S = {}
    for n, eps in raw.items():
        cr = np.array([e['crashes'] for e in eps], float) * 60 / T; ce = np.array([e['cells'] for e in eps], float)
        S[n] = dict(crashes_per_min=float(cr.mean()), crashes_ci=ci(cr[idx].mean(1)), cells=float(ce.mean()), cells_ci=ci(ce[idx].mean(1)),
                    crash_free=float((cr == 0).mean()), m_per_crash=float(sum(e['dist'] for e in eps) / max(1, cr.sum() * T / 60)))
        print(f'{n:22s} crashes/min {S[n]["crashes_per_min"]:5.2f} [{S[n]["crashes_ci"][0]:.2f}, {S[n]["crashes_ci"][1]:.2f}]  '
              f'crash-free {S[n]["crash_free"] * 100:3.0f}%  m/crash {S[n]["m_per_crash"]:7.1f}  cells {S[n]["cells"]:5.1f} '
              f'[{S[n]["cells_ci"][0]:.1f}, {S[n]["cells_ci"][1]:.1f}]', flush=True)
    d = np.array([a['crashes'] - b['crashes'] for a, b in zip(raw['lplc2_full'], raw['fly_old_looming'])], float)
    paired = dict(mean_diff=float(d.mean()), ci=ci(d[idx].mean(1)), worlds_better=float((d < 0).mean()), worlds_worse=float((d > 0).mean()))
    print('paired lplc2_full - old (crashes/min):', paired, f'wall {time.time() - t0:.0f}s')
    json.dump(dict(summary=S, paired=paired, cfg={n: kw for n, (_, kw) in CFG.items()}), open(OUT / 'lplc2_results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
