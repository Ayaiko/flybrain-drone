"""Recalibrate distance-from-flow on PX4/Gazebo data (3D render, lower contrast than the 2D sim).
Uses px4/runs/calib_<seed>.json (logged by fly_px4.py in a '-calib' mode); true distances come from ray casting the same
layout (flydrone.world.World). Writes results/flow_depth_coef_px4.json. Run from the project root: python -m px4.flow_depth_calib 4 5"""
import json, sys
from pathlib import Path
import numpy as np
from flydrone.world import World

ROOT = Path(__file__).resolve().parents[1]
rows = []
for sd in [int(a) for a in sys.argv[1:]]:
    d = json.loads((ROOT / f'px4/runs/calib_{sd}.json').read_text()); w = World(seed=sd); mid = np.array(d['mid'])
    side = (np.abs(mid) > np.radians(15)) & (np.abs(mid) < np.radians(150))
    for r in d['rows']:
        dist, _ = w.cast(np.array([r['x'], r['y']]), r['h'] + mid); out = np.array(r['out'])
        m = side & (out > 1e-4) & (dist < 15)
        for k in np.where(m)[0]: rows.append((r['v'], abs(np.sin(mid[k])), out[k], dist[k]))
v, s, o, dd = np.array(rows).T
X = np.c_[np.ones(len(v)), np.log(v * s), np.log(o)]; y = np.log(dd)
coef, *_ = np.linalg.lstsq(X, y, rcond=None); pred = X @ coef; near = dd < 3
rk = np.argsort(np.argsort(-pred)) + 1.0; n1 = near.sum(); auc = (rk[near].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(near) - n1))
print('n', len(v), 'coef', coef.round(3), 'r', round(float(np.corrcoef(pred, y)[0, 1]), 3), 'base rate d<3', round(float(near.mean()), 3), 'AUC', round(float(auc), 3))
dh = np.exp(pred); best = None
for q in (1, 2, 5, 10):
    th = np.percentile(dh, q); f = dh < th
    print(f'  lowest {q}%: dhat < {th:.2f}  precision {near[f].mean():.2f}  recall {f[near].mean():.2f}')
    if best is None and near[f].mean() >= 0.5: best = float(th)
best = best or float(np.percentile(dh, 2))
json.dump(dict(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]), near_thr=best, auc=float(auc)), open(ROOT / 'results/flow_depth_coef_px4.json', 'w'), indent=1)
print('saved, near_thr', round(best, 2))
