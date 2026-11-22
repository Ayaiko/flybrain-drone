"""Crash anatomy for PX4 coverage runs. For each post crash: what was the drone doing in the 1 s before?
Categories (first match wins):
  recrash         < 3.5 s after the previous crash (still in / just out of the back-off-and-turn recovery)
  during turn     inside a closed-loop saccade (turning while creeping forward at 0.4x speed)
  gated           vision ignored by the haltere gate (body yaw > 15 deg/s) at some point in the last 0.5 s
  settling        in the short settle after a turn / fence / recovery (filters restarting)
  side            post hit from > 45 deg off the heading while flying straight
  missed          head-on, looming never fired in the last 1 s
  late            head-on, looming fired but the escape did not clear it
Also: was the planner's waypoint behind that post (goal within 40 deg of the post's bearing)?
Run from the project root: python -m px4.crash_analysis <mode> <seed> [...] [-- <mode2> <seed> ...]"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from flydrone.world import World

ROOT = Path(__file__).resolve().parents[1]; FPS = 30.0


def analyse(mode, seeds):
    rows = []
    for sd in seeds:
        r = json.loads((ROOT / f'px4/runs/{mode}_{sd}.json').read_text())['run']; w = World(seed=sd)
        st = np.array(r['state']); lm = np.array(r['looming']); wy = np.degrees(np.abs(r['wy']))
        x, y, h = np.array(r['x']), np.array(r['y']), np.array(r['h'])
        last = -1e9
        for k, cx, cy in r['crashes']:
            k = min(k, len(st) - 1)
            j = int(np.argmin(np.hypot(w.c[:, 0] - cx, w.c[:, 1] - cy) - w.r))
            b = np.degrees((np.arctan2(w.c[j, 1] - y[k], w.c[j, 0] - x[k]) - h[k] + np.pi) % (2 * np.pi) - np.pi)
            win1 = slice(max(0, k - int(FPS)), k + 1); win05 = slice(max(0, k - int(FPS / 2)), k + 1)
            gated = bool(((st[win05] == 1) & (wy[win05] > 15)).any())
            if (k - last) / FPS < 3.5: cat = 'recrash'
            elif st[k] in (3, 4): cat = 'during turn'
            elif gated: cat = 'gated'
            elif st[k] == 1: cat = 'settling'
            elif st[k] == 0: cat = 'recovery/fence'
            elif abs(b) > 45: cat = 'side'
            elif not lm[win1].any(): cat = 'missed'
            else: cat = 'late'
            last = k
            g = r.get('goal', [None] * len(st))[k]
            goal_behind = None
            if g is not None:
                gb = np.arctan2(g[1] - y[k], g[0] - x[k]); pb = np.arctan2(w.c[j, 1] - y[k], w.c[j, 0] - x[k])
                goal_behind = bool(abs((gb - pb + np.pi) % (2 * np.pi) - np.pi) < np.radians(40))
            turn_frac = float(np.isin(st[win1], [3, 4]).mean())
            rows.append(dict(seed=sd, cat=cat, bearing=round(float(b), 1), goal_behind_post=goal_behind, turn_frac_1s=round(turn_frac, 2),
                             gated=gated, wy=round(float(wy[k]), 1)))
    return rows


if __name__ == '__main__':
    args = sys.argv[1:]; groups, cur = [], []
    for a in args:
        if a == '--': groups.append(cur); cur = []
        else: cur.append(a)
    groups.append(cur); out = {}
    for g in groups:
        mode, seeds = g[0], [int(s) for s in g[1:]]; rows = analyse(mode, seeds); n = max(1, len(rows))
        c = Counter(r['cat'] for r in rows)
        gb = [r['goal_behind_post'] for r in rows if r['goal_behind_post'] is not None]
        print(f'{mode}: {len(rows)} post crashes')
        print('   ', ', '.join(f'{k} {v} ({100 * v / n:.0f}%)' for k, v in c.most_common()))
        print(f'    waypoint behind the post that was hit: {100 * np.mean(gb):.0f}%' if gb else '')
        print(f'    median |bearing| of hit: {np.median(np.abs([r["bearing"] for r in rows])):.0f} deg; turning in the last 1 s (median share): {np.median([r["turn_frac_1s"] for r in rows]):.2f}')
        out[mode] = rows
    json.dump(out, open(ROOT / 'results/px4/crash_analysis.json', 'w'), indent=1)
