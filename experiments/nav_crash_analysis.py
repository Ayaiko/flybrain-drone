"""Why does the fly crash so much more when a planner steers it? Coverage mode (LPLC2 + gap-aware nav + hazard memory),
40 validation worlds x 180 s; for every post crash record the controller state, what the last saccade was
(goal / looming escape / balance), time since it, and whether the same post had already been hit.
Run from the project root: python -m experiments.nav_crash_analysis"""
import numpy as np
from collections import Counter
from multiprocessing import Pool
from flydrone import nav_sim
from flydrone.nav_sim import run_nav

L2 = dict(loom_mode='lplc2', lplc2_thr=2.0, front=35, goal_gap=2.0)


def one(seed):
    rows = []; orig = nav_sim.GoalNav.act
    hist = []
    def act(self, lum, dist, p, h):
        out = orig(self, lum, dist, p, h); d = self.diag
        hist.append((d.get('state'), d.get('goal', False), d.get('big', False), bool(d.get('looming')), p.copy(), h, self.goal.copy()))
        return out
    nav_sim.GoalNav.act = act
    try: r = run_nav('fly', L2, seed, task='cov', T=180.0, record=True, memory=True, loom_d=2.5)
    finally: nav_sim.GoalNav.act = orig
    w = r['world']; hit_posts = Counter()
    for k, x, y, kind in r['rec']['crashes']:
        if kind != 'post': continue
        k = min(k, len(hist)) - 1; st = hist[k][0]
        sac = [i for i in range(k, -1, -1) if hist[i][0] == 'saccade_start']
        last = sac[0] if sac else None
        typ = None if last is None else ('looming' if hist[last][2] else 'goal' if hist[last][1] else 'balance')
        since = None if last is None else (k - last) * nav_sim.DT
        j = int(np.argmin(np.linalg.norm(w.c - np.array([x, y]), axis=1) - w.r)); hit_posts[j] += 1
        loomed = any(hist[i][3] for i in range(max(0, k - 50), k + 1))
        dgoal = float(np.linalg.norm(hist[k][6] - hist[k][4]))
        rows.append(dict(state=st, last=typ, since=since, repeat=hit_posts[j] > 1, loomed=loomed, dgoal=dgoal))
    return rows


if __name__ == '__main__':
    with Pool() as P: rows = sum(P.map(one, range(200, 240)), [])
    n = len(rows); print('post crashes', n)
    print('state at crash   ', Counter(r['state'] for r in rows).most_common())
    print('last saccade type', Counter(r['last'] for r in rows).most_common())
    since = np.array([r['since'] for r in rows if r['since'] is not None])
    print('time since last saccade: median %.2f s, <0.3 s %.0f%%, <0.6 s %.0f%%' % (np.median(since), 100 * (since < .3).mean(), 100 * (since < .6).mean()))
    g = [r for r in rows if r['last'] == 'goal']
    if g: print('after GOAL saccade: <0.6 s %.0f%%' % (100 * np.mean([r['since'] < .6 for r in g])))
    print('same post hit before: %.0f%%' % (100 * np.mean([r['repeat'] for r in rows])))
    print('looming fired in last 1 s: %.0f%%' % (100 * np.mean([r['loomed'] for r in rows])))
    dg = np.array([r['dgoal'] for r in rows]); print('distance to waypoint at crash: median %.1f m, <2 m %.0f%%' % (np.median(dg), 100 * (dg < 2).mean()))
