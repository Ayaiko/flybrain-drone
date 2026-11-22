"""Check the ray-cast eye against real Gazebo frames, and fit its lighting curve.

Every logged PX4 forest run (px4/runs/forest/*.json) stores the pose and the 120-px eye vector the brain saw (img,
= camera average + N(0, 0.02) noise, uint8). The eye is rendered at the logged poses; the lighting curve (sky,
ground, albedo x f(n.L), plain grey g(n.L)) is fitted by linear least squares on the seed-0/1 worlds and tested on
the seed-2 worlds (held out).

  .venv/bin/python -m fastsim.validate_eye [--step 15] [--workers 8] [--no-save]
-> fastsim/eye_calib.json (unless --no-save), results/fastsim/eye_validation.json, results/fastsim/eye_compare_*.png"""
import argparse, base64, json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from .world import World, ROOT
from .eye import RayEye, KNOTS, CALIB_FILE, default_calib

RUNS = ROOT / 'px4/runs/forest'
# runs flown with the 2026-11-22 bridge logic (newer configs use px4/forest_pilot.py with other behaviours)
FIT_CONFIGS = ('none-v0.8', 'lplc2-bold-gate-sv0-v0.8', 'lplc2-bold-gate-sv0-gg0-v0.8')


def fit_files():
    return sorted(f for f in RUNS.glob('*_ff_*.json') if f.name.rsplit('_ff_', 1)[0] in FIT_CONFIGS)


OUT = ROOT / 'results/fastsim'
NOISE = 0.02


def load_run(f):
    d = json.loads(Path(f).read_text()); r = d['run']
    img = np.frombuffer(base64.b64decode(r['img']), np.uint8).reshape(-1, 120) / 255.0
    return d['summary'], r, img


def _feat(job):
    f, step, lags, lag = job
    s, r, img = load_run(f); w = World(s['world']); e = RayEye(w)
    n = len(r['x']); idx = np.arange(5, n - 5, step)          # pose idx <-> image idx + lag (camera latency)
    F = np.array([e.features(r['x'][i], r['y'][i], r['z'][i], r['h'][i]) for i in idx])          # (m, 120, nf)
    # next frame too -> temporal differences
    F1 = np.array([e.features(r['x'][i + 1], r['y'][i + 1], r['z'][i + 1], r['h'][i + 1]) for i in idx])
    out = dict(file=Path(f).name, world=s['world'], seed=s['seed'], idx=idx, F=F, F1=F1, Y=img[idx + lag], Y1=img[idx + 1 + lag])
    if lags:                                       # pose i vs image i + lag, on a subset, with the default shading
        sub = idx[::4]; out['lag'] = {}
        for L in range(-3, 4):
            ii = sub[(sub + L >= 0) & (sub + L < n)]
            v = np.array([e.render(r['x'][i], r['y'][i], r['z'][i], r['h'][i]) for i in ii])
            out['lag'][L] = float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(v, img[ii + L])]))
    return out


def coef_to_calib(c):
    nk = len(KNOTS)
    return dict(sky=float(c[0]), ground=float(c[1]), tex=[float(v) for v in c[2:2 + nk]], plain=[float(v) for v in c[2 + nk:]])


def rowcorr(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--step', type=int, default=15); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--lag', type=int, default=1, help='image i + lag was rendered at pose i (measured: 1 frame)')
    a = ap.parse_args(); t0 = time.time()
    files = fit_files()
    with Pool(a.workers) as p:
        R = p.map(_feat, [(str(f), a.step, i % 3 == 0, a.lag) for i, f in enumerate(files)])
    train = [r for r in R if r['seed'] in (0, 1)]; test = [r for r in R if r['seed'] == 2]
    X = np.concatenate([r['F'].reshape(-1, r['F'].shape[-1]) for r in train]); y = np.concatenate([r['Y'].ravel() for r in train])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); calib = coef_to_calib(coef)
    c0 = default_calib(); c0v = np.r_[c0['sky'], c0['ground'], c0['tex'], c0['plain']]
    rng = np.random.default_rng(0)

    def stats(rs, c):
        P = np.concatenate([r['F'] @ c for r in rs]); P1 = np.concatenate([r['F1'] @ c for r in rs])
        Y = np.concatenate([r['Y'] for r in rs]); Y1 = np.concatenate([r['Y1'] for r in rs])
        fc = rowcorr(P, Y)
        # ceiling: the logged img itself carries N(0, 0.02) + uint8 rounding -> best possible corr of a perfect render
        Pn = np.clip(np.round((P + rng.normal(0, NOISE, P.shape)) * 255), 0, 255) / 255
        ceil = rowcorr(P, Pn)
        dsp = rowcorr(np.diff(P, axis=1), np.diff(Y, axis=1))                     # spatial contrast pattern
        dt_c = np.corrcoef((P1 - P).ravel(), (Y1 - Y).ravel())[0, 1]                 # frame-to-frame change
        return dict(frames=len(P), frame_corr_mean=float(fc.mean()), frame_corr_median=float(np.median(fc)), frame_corr_p10=float(np.percentile(fc, 10)),
                    pixel_corr=float(np.corrcoef(P.ravel(), Y.ravel())[0, 1]), rmse=float(np.sqrt(np.mean((P - Y) ** 2))), bias=float(np.mean(P - Y)),
                    noise_ceiling_frame_corr=float(ceil.mean()), spatial_diff_corr_mean=float(dsp.mean()), temporal_diff_corr=float(dt_c))

    res = dict(step=a.step, lag=a.lag, files=[r['file'] for r in R], n_train_runs=len(train), n_test_runs=len(test), calib=calib,
               train=stats(train, coef), test=stats(test, coef), test_uncalibrated=stats(test, c0v),
               per_run={r['file']: float(rowcorr(r['F'] @ coef, r['Y']).mean()) for r in R},
               per_preset_test={p: stats([r for r in test if f'_{p}_' in r['world']], coef)['frame_corr_mean'] for p in ('easy', 'medium', 'hard')},
               lag_corr={L: float(np.mean([r['lag'][L] for r in R if 'lag' in r])) for L in range(-3, 4)}, wall_s=round(time.time() - t0, 1))
    calib['source'] = f'fastsim.validate_eye lstsq on {len(train)} seed-0/1 runs (step {a.step}, image lag {a.lag}); held-out frame corr {res["test"]["frame_corr_mean"]:.3f}'
    if not a.no_save: CALIB_FILE.write_text(json.dumps(calib, indent=1))
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'eye_validation.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ('files', 'per_run')}, indent=1))
    plot(R, coef)


def plot(R, coef):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    pick = [r for r in R if r['seed'] == 2][:3]
    fig, axs = plt.subplots(len(pick), 3, figsize=(13, 3.2 * len(pick)), squeeze=False)
    for ax, r in zip(axs, pick):
        P = r['F'] @ coef; m = min(len(P), 240)
        ax[0].imshow(r['Y'][:m].T, aspect='auto', cmap='gray', vmin=0, vmax=1); ax[0].set_title(f"{r['file']}\nPX4/Gazebo img", fontsize=8)
        ax[1].imshow(P[:m].T, aspect='auto', cmap='gray', vmin=0, vmax=1); ax[1].set_title('fastsim ray-cast at logged poses', fontsize=8)
        k = m // 2; ax[2].plot(np.linspace(-165, 165, 120), r['Y'][k], label='Gazebo')
        ax[2].plot(np.linspace(-165, 165, 120), P[k], label='fastsim'); ax[2].set_xlabel('eye angle deg (+ left)'); ax[2].legend(fontsize=7)
        ax[2].set_title(f'frame {r["idx"][k]}  corr {np.corrcoef(P[k], r["Y"][k])[0, 1]:.3f}', fontsize=8)
        for a_ in ax[:2]: a_.set_xlabel('sample (every step frames)'); a_.set_ylabel('ommatidium')
    fig.tight_layout(); fig.savefig(OUT / 'eye_compare.png', dpi=100); plt.close(fig)


if __name__ == '__main__':
    main()
