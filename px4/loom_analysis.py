"""Why does looming fire so often in PX4/Gazebo? For every looming episode during fixation (consecutive looming
frames), look at the ground truth at its onset and classify (first match wins):
  real post      a post edge within 3 m ahead (+-35 deg, widened by the post's angular size)
  real wall      the area wall within 3 m ahead
  fixation start fired within 0.2 s of the fixation starting (filters still settling after a turn)
  body turning   |yaw rate| > 20 deg/s
  body tilting   |pitch rate| > 15 deg/s or |pitch| > 8 deg (accelerating / braking)
  altitude       |z - 1.5| > 0.15 m (camera moving up/down)
  unexplained    none of the above
Run from the project root: python -m px4.loom_analysis <mode> <seed> [<seed> ...]"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; FPS = 30.0; L = 40.0


def wall_ahead(x, y, h):
    c, s = np.cos(h), np.sin(h); ts = []
    for d, p, lo in ((c, x, 0), (c, x, L), (s, y, 0), (s, y, L)):
        if abs(d) > 1e-6:
            t = (lo - p) / d
            if t > 0: ts.append(t)
    return min(ts) if ts else 99.0


def analyse(mode, seeds):
    eps, fix = [], dict(wy=[], pitch=[], dpitch=[], dz=[], front=[], loom=[], looming=[])
    for sd in seeds:
        r = json.loads((ROOT / f'px4/runs/{mode}_{sd}.json').read_text())['run']
        st = np.array(r['state']); lm = np.array(r['looming']); n = len(st)
        x, y, h = np.array(r['x']), np.array(r['y']), np.array(r['h'])
        wy = np.degrees(np.abs(r['wy'])); pitch = np.degrees(np.array(r['pitch'])); z = np.array(r['z'])
        dp = np.abs(np.gradient(pitch) * FPS); front = np.array(r['front_d']); loom = np.array(r['loom'])
        since = np.zeros(n)
        for k in range(1, n): since[k] = since[k - 1] + 1 if st[k] == 2 and st[k - 1] == 2 else 0
        fx = st == 2
        for key, v in (('wy', wy), ('pitch', np.abs(pitch)), ('dpitch', dp), ('dz', np.abs(z - 1.5)), ('front', front), ('loom', loom), ('looming', lm)):
            fix[key] += list(v[fx])
        k = 0
        while k < n:
            if fx[k] and lm[k] and (k == 0 or not (fx[k - 1] and lm[k - 1])):
                e = k
                while e + 1 < n and fx[e + 1] and lm[e + 1]: e += 1
                wa = wall_ahead(x[k], y[k], h[k])
                if front[k] < 3.0: cause = 'real post'
                elif wa < 3.0: cause = 'real wall'
                elif since[k] < 0.2 * FPS: cause = 'fixation start'
                elif wy[k] > 20: cause = 'body turning'
                elif dp[k] > 15 or abs(pitch[k]) > 8: cause = 'body tilting'
                elif abs(z[k] - 1.5) > 0.15: cause = 'altitude'
                else: cause = 'unexplained'
                eps.append(dict(seed=sd, k=k, frames=e - k + 1, cause=cause, front=float(front[k]), wall=float(wa), since=float(since[k] / FPS),
                                wy=float(wy[k]), pitch=float(pitch[k]), dpitch=float(dp[k]), dz=float(z[k] - 1.5)))
                k = e + 1
            else: k += 1
    return eps, {k: np.array(v) for k, v in fix.items()}


if __name__ == '__main__':
    mode, seeds = sys.argv[1], [int(a) for a in sys.argv[2:]]
    eps, F = analyse(mode, seeds)
    longs = [e for e in eps if e['frames'] >= 3]
    print(f'looming episodes: {len(eps)} (>= 3 frames, i.e. the ones that become hazards: {len(longs)})')
    for name, group in (('all', eps), ('>=3 frames', longs)):
        c = Counter(e['cause'] for e in group); n = max(1, len(group))
        print(f'  {name:11s}', ', '.join(f'{k} {v} ({100 * v / n:.0f}%)' for k, v in c.most_common()))
    lm = F['looming'].astype(bool)
    print('during fixation, looming vs not looming (median):')
    for key, lab in (('wy', '|yaw rate| deg/s'), ('pitch', '|pitch| deg'), ('dpitch', '|pitch rate| deg/s'), ('dz', '|z-1.5| m'), ('front', 'post ahead m')):
        print(f'  {lab:20s} looming {np.median(F[key][lm]):6.2f}   not {np.median(F[key][~lm]):6.2f}')
    for key in ('wy', 'dpitch', 'pitch'):
        print(f'  corr(loom, {key}) = {np.corrcoef(F["loom"], F[key])[0, 1]:.2f}')
    print(f'  corr(loom, 1/post ahead) = {np.corrcoef(F["loom"], 1 / np.maximum(F["front"], .3))[0, 1]:.2f}')
    un = [e for e in longs if e['cause'] == 'unexplained']
    if un: print('unexplained examples:', [{k: round(v, 2) if isinstance(v, float) else v for k, v in e.items()} for e in un[:4]])
    json.dump(dict(episodes=eps), open(ROOT / f'results/px4/loom_analysis_{mode}.json', 'w'), indent=1)
