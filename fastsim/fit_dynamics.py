"""Fit the x500 flight model (fastsim/dynamics.py) to the logged PX4 forest runs.

Commands: yaw rate = logged yawc; forward speed is not logged, so it is rebuilt from the bridge logic of
px4/fly_forest.py and the logged state/looming flags (route follower: speed; fixate: speed, x0.5 while looming;
settling / saccade_start: speed x sac_speed; closed-loop saccade: speed x sac_v; recovery: -0.6 m/s back-off with yaw
0, then 0 m/s with yaw +-2; geofence: 0.6 / 0).
Fit: from every 10th frame, start the model at the logged pose / velocity / yaw rate, feed the logged commands and
compare position and heading 0.5, 1 and 2 s later. Yaw first (heading error; tau_r, rmax, alpha for each integer
delay_r), then speed with that yaw model (position error; tau_v, acc, dec for each integer delay_v), Nelder-Mead.
Seed-0/1 worlds train, seed-2 worlds test.

  .venv/bin/python -m fastsim.fit_dynamics        -> fastsim/dyn_params.json, results/fastsim/dynamics_fit.json + .png"""
import json, time
import numpy as np
from .world import ROOT
from .dynamics import substeps, NAMES_R, NAMES_V, DEFAULT, PARAM_FILE
from px4.forest_params import resolve

RUNS = ROOT / 'px4/runs/forest'
# runs flown with the 2026-11-22 bridge logic (newer configs use px4/forest_pilot.py with other behaviours)
FIT_CONFIGS = ('none-v0.8', 'lplc2-bold-gate-sv0-v0.8', 'lplc2-bold-gate-sv0-gg0-v0.8')


def fit_files():
    return sorted(f for f in RUNS.glob('*_ff_*.json') if f.name.rsplit('_ff_', 1)[0] in FIT_CONFIGS)


OUT = ROOT / 'results/fastsim'
DT, SUB, HOR, STEP = 1 / 30, 4, (15, 30, 60), 10
W_HEAD = 0.5            # m per rad in the cost


def rebuild_v(r, P, field):
    st, lo, yc = np.array(r['state']), np.array(r['looming']), np.array(r['yawc'])
    x, y = np.array(r['x']), np.array(r['y']); sp = P['speed']
    if P['brain'] == 'none': return np.full(len(st), sp)
    v = np.zeros(len(st))
    v[st == 2] = sp * np.where(lo[st == 2] == 1, 0.5, 1.0)
    v[(st == 1) | (st == 4)] = sp * P['sac_speed']
    v[st == 3] = sp * P['sac_v']
    inside = np.minimum.reduce([x - field['xmin'], field['xmax'] - x, y - field['ymin'], field['ymax'] - y])
    w = st == 0
    v[w & (np.abs(yc) < 1e-6)] = -0.6
    v[w & (np.abs(yc) < 1e-6) & (inside < 2.5)] = 0.6
    v[w & (np.abs(np.abs(yc) - 2.0) < 1e-6)] = 0.0
    return v


def load():
    D = []
    for f in fit_files():
        d = json.loads(f.read_text()); s, r = d['summary'], d['run']
        P = resolve(s.get('config') or s['mode']); meta = json.loads((ROOT / f"px4/forest/{s['world']}.json").read_text())
        x, y, h = np.array(r['x']), np.array(r['y']), np.unwrap(np.array(r['h']))
        vc = rebuild_v(r, P, meta['field']); rc = np.array(r['yawc'])
        n = len(x); k = np.arange(40, n - max(HOR) - 4, STEP)
        D.append(dict(file=f.name, seed=s['seed'], brain=P['brain'], x=x, y=y, h=h, vc=vc, rc=rc, k=k,
                      vx0=(x[k + 3] - x[k - 3]) / (6 * DT), vy0=(y[k + 3] - y[k - 3]) / (6 * DT), r0=(h[k + 3] - h[k - 3]) / (6 * DT)))
    return D


def predict(P, d, delay=None):
    """-> errors (pos m, heading rad) at each horizon, arrays (len(HOR), n_segments)"""
    dv, dr = int(P['delay_v']), int(P['delay_r'])
    k = d['k']; h, r, vx, vy = d['h'][k].copy(), d['r0'].copy(), d['vx0'].copy(), d['vy0'].copy()
    px, py = d['x'][k].copy(), d['y'][k].copy(); ep, eh = [], []
    for j in range(max(HOR)):
        h, r, vx, vy, dx, dy = substeps(P, h, r, vx, vy, d['vc'][np.maximum(k + j - dv, 0)], d['rc'][np.maximum(k + j - dr, 0)], DT / SUB, SUB)
        px, py = px + dx, py + dy
        if j + 1 in HOR:
            t = k + j + 1; ep.append(np.hypot(px - d['x'][t], py - d['y'][t])); eh.append(np.abs(h - d['h'][t]))
    return np.array(ep), np.array(eh)


def cost(P, D, what):
    c = 0.0; n = 0
    for d in D:
        ep, eh = predict(P, d); c += ((W_HEAD * eh) ** 2 if what == 'yaw' else ep ** 2).sum(); n += ep.size
    return c / n


def nelder_mead(f, x0, step=0.3, iters=250, tol=1e-7):
    n = len(x0); S = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]
    F = [f(s) for s in S]
    for _ in range(iters):
        o = np.argsort(F); S = [S[i] for i in o]; F = [F[i] for i in o]
        if abs(F[-1] - F[0]) < tol: break
        c = np.mean(S[:-1], 0); xr = c + (c - S[-1]); fr = f(xr)
        if fr < F[0]:
            xe = c + 2 * (c - S[-1]); fe = f(xe); S[-1], F[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < F[-2]: S[-1], F[-1] = xr, fr
        else:
            xc = c + 0.5 * (S[-1] - c); fc = f(xc)
            if fc < F[-1]: S[-1], F[-1] = xc, fc
            else:
                S = [S[0]] + [S[0] + 0.5 * (s - S[0]) for s in S[1:]]; F = [F[0]] + [f(s) for s in S[1:]]
    i = int(np.argmin(F)); return S[i], F[i]


def report(P, D):
    E = [predict(P, d) for d in D]
    ep = np.concatenate([e[0] for e in E], 1); eh = np.concatenate([e[1] for e in E], 1)
    return {f'{h / 30:.1f}s': dict(pos_rmse_m=float(np.sqrt((ep[i] ** 2).mean())), pos_median_m=float(np.median(ep[i])), pos_p90_m=float(np.percentile(ep[i], 90)),
                                   head_rmse_deg=float(np.degrees(np.sqrt((eh[i] ** 2).mean()))), head_median_deg=float(np.degrees(np.median(eh[i]))))
            for i, h in enumerate(HOR)}


def main():
    t0 = time.time(); D = load(); train = [d for d in D if d['seed'] in (0, 1)]; test = [d for d in D if d['seed'] == 2]
    sub = [dict(d, k=d['k'][::2], vx0=d['vx0'][::2], vy0=d['vy0'][::2], r0=d['r0'][::2]) for d in train]   # faster search
    P = dict(DEFAULT); search = {}
    for what, names, dkey in (('yaw', NAMES_R, 'delay_r'), ('speed', NAMES_V, 'delay_v')):
        best = None
        for delay in range(0, 13):
            def f(z):
                Q = dict(P); Q.update({n: float(np.exp(v)) for n, v in zip(names, z)}); Q[dkey] = delay; return cost(Q, sub, what)
            z, c = nelder_mead(f, np.log([P[n] for n in names]), iters=150)
            search[f'{dkey}={delay}'] = dict(cost=round(float(c), 6), **{n: round(float(np.exp(v)), 3) for n, v in zip(names, z)})
            print(what, f'delay {delay}: cost {c:.6f}', search[f'{dkey}={delay}'], flush=True)
            if best is None or c < best[1]: best = (delay, c, z)
        P.update({n: float(np.exp(v)) for n, v in zip(names, best[2])}); P[dkey] = best[0]
    ideal = dict(DEFAULT, delay_v=0, delay_r=0, tau_v=1e-4, acc=1e4, dec=1e4, tau_r=1e-4, rmax=1e4, alpha=1e6)
    res = dict(params=P, train=report(P, train), test=report(P, test),
               test_route_follower=report(P, [d for d in test if d['brain'] == 'none']),
               test_brain=report(P, [d for d in test if d['brain'] != 'none']),
               test_ideal_unicycle=report(ideal, test), search=search, n_segments_train=int(sum(len(d['k']) for d in train)),
               n_segments_test=int(sum(len(d['k']) for d in test)), wall_s=round(time.time() - t0, 1))
    PARAM_FILE.write_text(json.dumps(dict(P, source='fastsim.fit_dynamics on seed-0/1 PX4 forest runs'), indent=1))
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'dynamics_fit.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'search'}, indent=1)); plot(P, test)


def plot(P, D):
    """Step responses: a recovery (back-off + 2 rad/s turn) and a saccade, logged vs model (open loop from the start)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    d = next(d for d in D if d['brain'] != 'none'); x, y, h = d['x'], d['y'], d['h']
    vf = np.gradient(x, DT) * np.cos(h) + np.gradient(y, DT) * np.sin(h); wy = np.gradient(h, DT)
    st = np.argmax(np.abs(d['rc']) > 1.9) - 40
    T = 240; hh, r, vx, vy = h[st], wy[st], np.gradient(x, DT)[st], np.gradient(y, DT)[st]; mh, mr, mv = [], [], []
    for j in range(T):
        hh, r, vx, vy, _, _ = substeps(P, hh, r, vx, vy, d['vc'][max(st + j - P['delay_v'], 0)], d['rc'][max(st + j - P['delay_r'], 0)], DT / SUB, SUB)
        mh.append(hh); mr.append(r); mv.append(vx * np.cos(hh) + vy * np.sin(hh))
    t = np.arange(T) * DT
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t, d['vc'][st:st + T], 'k:', label='cmd v (rebuilt)'); ax[0].plot(t, vf[st + 1:st + T + 1], label='PX4'); ax[0].plot(t, mv, label='model'); ax[0].set_ylabel('forward m/s'); ax[0].legend(fontsize=7)
    ax[1].plot(t, d['rc'][st:st + T], 'k:', label='cmd'); ax[1].plot(t, wy[st + 1:st + T + 1], label='PX4'); ax[1].plot(t, mr, label='model'); ax[1].set_ylabel('yaw rate rad/s')
    ax[2].plot(t, np.degrees(h[st + 1:st + T + 1] - h[st]), label='PX4'); ax[2].plot(t, np.degrees(np.array(mh) - h[st]), label='model'); ax[2].set_ylabel('heading deg'); ax[2].set_xlabel('s (open loop from start)')
    ax[0].set_title(f"{d['file']} frames {st}-{st + T}: logged vs fitted model, delay v/r {P['delay_v']}/{P['delay_r']} frames", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / 'dynamics_fit.png', dpi=100); plt.close(fig)


if __name__ == '__main__':
    main()
