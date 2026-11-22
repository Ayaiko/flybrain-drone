"""Webots: effect of turn-in-place + gyro gate, and of slower cruise speed. 10 worlds x 60 s each.
Run from the project root: python -m webots.fix_eval   -> results/webots/fix_eval.json"""
import json, sys
from pathlib import Path
import numpy as np
from webots.run_sim import run_batch

ROOT = Path(__file__).resolve().parents[1]; N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MODES = ['none-v1.0', 'lplc2', 'lplc2-v1.0', 'lplc2-g', 'lplc2-g-v1.0', 'lplc2-g-v0.5', 'old-g-v1.0']
if __name__ == '__main__':
    jobs = [(m, s, 60) for s in range(N) for m in MODES]; summ = run_batch(jobs, per_job=120)
    rng = np.random.default_rng(0); S = {}
    for m in MODES:
        rs = [x for (mm, _, _), x in zip(jobs, summ) if mm == m and x]
        pm = np.array([r['post_per_min'] for r in rs]); dm = np.array([r['dist_m'] for r in rs])
        b = pm[rng.integers(0, len(pm), (2000, len(pm)))].mean(1)
        S[m] = dict(n=len(rs), post_per_min=float(pm.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                    dist_m=float(dm.mean()), m_per_crash=float(dm.sum() / max(1, sum(r['post_crashes'] for r in rs))),
                    crash_free=float(np.mean([r['post_crashes'] == 0 for r in rs])), flips=int(sum(r.get('flipped_at') is not None for r in rs)), cells=float(np.mean([r['cells'] for r in rs])))
        print(f'{m:14s} crashes/min {S[m]["post_per_min"]:.2f} [{S[m]["ci"][0]:.2f}, {S[m]["ci"][1]:.2f}]  m/crash {S[m]["m_per_crash"]:5.0f}  '
              f'crash-free {S[m]["crash_free"] * 100:3.0f}%  flips {S[m]["flips"]}  dist {S[m]["dist_m"]:4.0f} m  cells {S[m]["cells"]:.0f}', flush=True)
    json.dump(S, open(ROOT / 'results/webots/fix_eval.json', 'w'), indent=1)
