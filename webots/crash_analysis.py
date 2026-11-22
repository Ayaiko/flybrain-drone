"""Crash anatomy in Webots (same categories as experiments/crash_analysis.py) + signal-quality checks that only exist
in 3D: body pitch/roll, yaw-rate leaking into fixation, looming signal vs. true distance ahead.
Run from the project root: python -m webots.crash_analysis [N]   -> results/webots/crash_analysis.json"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from webots.run_sim import run_batch

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'results/webots'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10; STATES = ['recover', 'settling', 'fixate', 'saccade', 'saccade_start']


def auc(x, y):
    r = np.argsort(np.argsort(x)) + 1.0; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def analyse(mode):
    rows, F = [], dict(loom=[], lab=[], front=[], wy=[], pitch=[])
    stuck = []
    for seed in range(N):
        d = json.loads((ROOT / f'webots/runs/{mode}_{seed}.json').read_text()); r = d['run']; dt = d['dt']
        st = np.array(r['state']); lm = np.array(r['looming']); fd = np.array(r['front_d']); nb = np.degrees(np.array(r['near_b']))
        n = len(st); lab = np.zeros(n, int)
        for k, *_ in r['crashes']: lab[max(0, k - int(.5 / dt)):k + 1] = 1
        fx = st == 2
        F['loom'] += list(np.array(r['loom'])[fx]); F['lab'] += list(lab[fx]); F['front'] += list(fd[fx])
        F['wy'] += list(np.abs(np.array(r['wy']))[fx]); F['pitch'] += list(np.abs(np.array(r['pitch']))[fx])
        stuck.append(dict(seed=seed, dist=d['summary']['dist_m'], frac_saccade=float(np.isin(st, [3, 4]).mean()), frac_looming=float(lm[fx].mean()) if fx.any() else 0,
                          frac_recover=float((st == 0).mean())))
        last_c = -1e9
        for k, *_ in r['crashes']:
            prev = st[max(0, k - 1)]; b = nb[k]
            win = slice(max(0, k - int(1 / dt)), k + 1)
            sacs = np.where(st[:k] == 4)[0]; since = (k - sacs[-1]) * dt if len(sacs) else None
            if (k - last_c) * dt < 3.5: cat = 'recrash (<3.5 s, during/after recovery)'
            elif prev in (3, 4): cat = 'saccade'
            elif prev == 1: cat = 'settling'
            elif abs(b) > 45: cat = 'side'
            elif not lm[win].any(): cat = 'missed'
            else: cat = 'late'
            last_c = k
            rows.append(dict(seed=seed, cat=cat, bearing=float(b), since=since, pitch=float(np.degrees(r['pitch'][k])), vf=float(r['vf'][k])))
    lab = np.array(F['lab']); loom = np.array(F['loom']); front = np.array(F['front'])
    near = (front < 1.5).astype(int)
    out = dict(n_crashes=len(rows), categories=Counter(x['cat'] for x in rows).most_common(),
               bearing_abs_median=float(np.median(np.abs([x['bearing'] for x in rows]))) if rows else None,
               auc_loom_vs_crash_0p5s=auc(loom, lab), auc_loom_vs_obstacle_within_1p5m=auc(loom, near),
               corr_loom_with_yaw_rate=float(np.corrcoef(loom, F['wy'])[0, 1]), corr_loom_with_pitch=float(np.corrcoef(loom, F['pitch'])[0, 1]),
               corr_loom_with_nearness=float(np.corrcoef(loom, 1 / np.maximum(front, .3))[0, 1]),
               fixation_yaw_rate_deg_s_median=float(np.degrees(np.median(F['wy']))), fixation_pitch_deg_median=float(np.degrees(np.median(F['pitch']))),
               worst_flights=sorted(stuck, key=lambda s: s['dist'])[:3])
    return out


if __name__ == '__main__':
    if '--no-run' not in sys.argv:
        run_batch([(m, s, 60) for s in range(N) for m in ('old', 'lplc2')])
    R = {m: analyse(m) for m in ('old', 'lplc2')}
    print(json.dumps(R, indent=1))
    json.dump(R, open(OUT / 'crash_analysis.json', 'w'), indent=1)
