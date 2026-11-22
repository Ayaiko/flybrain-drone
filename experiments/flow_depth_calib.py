"""Can the fly eye tell how far things are? Translational optic flow at azimuth phi is v*sin(phi)/d, so
d ~ v*|sin phi| / flow. The EMD is not a linear speed meter, so fit log d = a + b*log(v|sin phi|) + c*log(out_ema)
against ground truth during straight flight (2D sim, validation worlds). Saves the coefficients for the mapping layer.
Run from the project root: python -m experiments.flow_depth_calib"""
import json
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from flydrone.sim import DT, OVERSAMPLE, make_eye
from flydrone.world import World
from flydrone.controllers import FlyVision

ROOT = Path(__file__).resolve().parents[1]


def run(seed, speed=1.0, T=60.0):
    w = World(seed=seed); eye = make_eye(120); sa = eye[1] - eye[0]
    fine = (eye[:, None] + sa * (np.arange(OVERSAMPLE) / OVERSAMPLE - .5 + .5 / OVERSAMPLE)).ravel()
    c = FlyVision(eye, DT, seed=seed, speed=speed, loom_mode='lplc2', lplc2_thr=2.5); rng = np.random.default_rng(seed)
    p = w.start.copy(); h = np.pi / 2; t = 0; rows = []
    while t < T:
        d, l = w.cast(p, h + fine); lum = l.reshape(-1, 4).mean(1) + rng.normal(0, .02, 120); dist = d.reshape(-1, 4).min(1)
        yaw, v = c.act(lum)
        if c.diag['state'] == 'fixate' and c.t > 0.9:
            dm = np.minimum(dist[:-1], dist[1:]); m = (np.abs(c.mid) > np.radians(15)) & (np.abs(c.mid) < np.radians(150)) & (dm < 15)
            for k in np.where(m)[0][::3]: rows.append((v, abs(np.sin(c.mid[k])), c.out_ema[k], dm[k]))
        h += np.clip(yaw, -12, 12) * DT; prev = p; p = p + v * DT * np.array([np.cos(h), np.sin(h)]); t += DT
        if w.collides(p, .25): p = prev; h += np.pi
    return rows


if __name__ == '__main__':
    with Pool() as P: rows = np.array(sum(P.map(run, range(300, 330)), []))
    v, s, o, d = rows.T; ok = o > 1e-4
    X = np.c_[np.ones(ok.sum()), np.log(v[ok] * s[ok]), np.log(o[ok])]; y = np.log(d[ok])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); pred = X @ coef
    r = np.corrcoef(pred, y)[0, 1]; err = np.exp(np.abs(pred - y))
    near = d[ok] < 3; dhat = np.exp(pred)
    print('n', ok.sum(), 'coef a,b,c', coef.round(3), 'r(log)', round(r, 3), 'median factor error', round(float(np.median(err)), 2))
    rk = np.argsort(np.argsort(-pred)) + 1.0; n1 = near.sum(); auc = (rk[near].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(near) - n1))
    print(f'  base rate d<3: {near.mean():.3f}   AUC(near vs far) {auc:.3f}')
    best = None
    for q in (1, 2, 5, 10, 20):
        th = np.percentile(dhat, q); flag = dhat < th
        print(f'  dhat < {th:.2f} (lowest {q}%): precision {np.mean(near[flag]):.2f}  recall {np.mean(flag[near]):.2f}')
        if best is None and np.mean(near[flag]) >= 0.5: best = float(th)
    coef = list(coef) + [best or float(np.percentile(dhat, 2))]
    json.dump(dict(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]), r=float(r), near_thr=float(coef[3])), open(ROOT / 'results/flow_depth_coef.json', 'w'), indent=1)
