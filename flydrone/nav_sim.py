"""Goal-reaching and area-coverage episodes (pose known, as from GPS + compass).
task='goal':     fly from the start to one goal point; success when within GOAL_R.
task='coverage': the area is the arena minus a 2 m geofence margin, split in 2x2 m cells; a greedy planner keeps sending
                 the drone to the nearest unvisited cell (gives up on a waypoint after STUCK_T s - the planner does not
                 know where obstacles are). Score = % of cells visited.
memory=True: the planner keeps a hazard map built only from what the drone itself felt - a point ~loom_d m ahead
(biased to the side of the stronger looming unit) each time looming fires, and the contact point of each crash
(a real drone would feel it on the IMU). Waypoints within HAZ_R of a hazard are skipped, and among the rest the
nearest one whose straight line is clear of hazards is preferred.
Crashes bounce like flydrone.sim (turn around, continue) and are split into posts vs walls (walls = geofence edge)."""
import numpy as np
from .sim import DT, OVERSAMPLE, make_eye
from .world import World
from .navigation import GoalNav
from .controllers import FlyVision

GOAL_R, CELL, MARGIN, STUCK_T, HAZ_R = 1.0, 2.0, 2.0, 8.0, 1.2


def crash_kind(world, p, rad=.25):
    L = world.size
    if p[0] < rad or p[1] < rad or p[0] > L - rad or p[1] > L - rad: return 'wall'
    return 'post'


class Planner:
    def __init__(self, world, memory=False, line_check=True, expire=None):
        # expire: looming hazards are forgotten after this many s (crash points are kept); None = never
        self.memory, self.line_check, self.expire = memory, line_check, expire
        self.haz = np.zeros((0, 2)); self.haz_t = []; self.haz_perm = []; self.now = 0.0; self.ttl = {}
        L = world.size; n = int((L - 2 * MARGIN) // CELL); self.n = n
        self.centers = np.array([[MARGIN + (i + .5) * CELL, MARGIN + (j + .5) * CELL] for i in range(n) for j in range(n)])
        self.visited = np.zeros(len(self.centers), bool); self.skip_until = np.zeros(len(self.centers)); self.cur = None; self.t_cur = 0
    def cell_of(self, p):
        i, j = int((p[0] - MARGIN) // CELL), int((p[1] - MARGIN) // CELL)
        return i * self.n + j if 0 <= i < self.n and 0 <= j < self.n else None
    turn_w, min_d = 0.0, 0.0      # heading-aware choice: cost = distance + turn_w * |turn needed| (m per rad); ignore cells closer than min_d

    def update(self, p, t, h=None):
        self.now = t
        self._expire(t)
        c = self.cell_of(p)
        if c is not None: self.visited[c] = True
        if self.cur is None or self.visited[self.cur] or t - self.t_cur > STUCK_T:
            if self.cur is not None and not self.visited[self.cur]: self.skip_until[self.cur] = t + 30
            ok = ~self.visited & (self.skip_until <= t)
            if self.memory and len(self.haz):
                near = np.linalg.norm(self.centers[:, None] - self.haz[None], axis=2).min(1) < HAZ_R
                if (ok & ~near).any(): ok &= ~near
            if not ok.any(): ok = ~self.visited
            if not ok.any(): return None
            d = np.linalg.norm(self.centers - p, axis=1)
            if self.min_d > 0 and (ok & (d >= self.min_d)).any(): ok = ok & (d >= self.min_d)
            if h is not None and self.turn_w > 0:
                turn = np.abs((np.arctan2(self.centers[:, 1] - p[1], self.centers[:, 0] - p[0]) - h + np.pi) % (2 * np.pi) - np.pi)
                d = d + self.turn_w * turn
            d[~ok] = np.inf
            if self.memory and self.line_check and len(self.haz):
                for c in np.argsort(d)[:12]:                       # nearest few: first one with a clear straight line
                    if not np.isfinite(d[c]): break
                    if self.clear(p, self.centers[c]): d[c] -= 1e3; break
            self.cur = int(d.argmin()); self.t_cur = t
        return self.centers[self.cur]
    def coverage(self): return float(self.visited.mean())

    def _expire(self, t):
        if not len(self.haz) or (self.expire is None and not self.ttl): return
        keep = np.array([pm or t - tt < self.ttl.get(i, self.expire if self.expire is not None else np.inf)
                         for i, (tt, pm) in enumerate(zip(self.haz_t, self.haz_perm))])
        if not keep.all():
            old = np.where(keep)[0]; self.ttl = {j: self.ttl[i] for j, i in enumerate(old) if i in self.ttl}
            self.haz = self.haz[keep]; self.haz_t = [v for v, k in zip(self.haz_t, keep) if k]; self.haz_perm = [v for v, k in zip(self.haz_perm, keep) if k]
    def mark(self, q, perm=False, ttl=None):
        if self.memory and (not len(self.haz) or np.linalg.norm(self.haz - q, axis=1).min() > 0.5):   # de-duplicate
            self.haz = np.vstack([self.haz, q]); self.haz_t.append(self.now); self.haz_perm.append(perm)
            if ttl is not None: self.ttl[len(self.haz) - 1] = ttl
    def clear(self, a, b):
        n = max(2, int(np.linalg.norm(b - a) / .5)); pts = a + np.linspace(0, 1, n)[:, None] * (b - a)
        return bool((np.linalg.norm(pts[:, None] - self.haz[None], axis=2) > 0.8).all())


def run_nav(base, kw, seed, task='goal', T=60.0, record=False, wander=False, memory=False, loom_d=2.0, line_check=True,
            persist=1, expire=None, lawn=False, turn_w=0.0, min_d=0.0, flowmap=False, fm_d=2.0, fm_expire=10.0):
    # flowmap: also map SIDE obstacles from translational optic flow (d ~ v|sin phi|/flow, calibrated in
    # experiments/flow_depth_calib.py) so goal turns know what is beside/behind them
    # persist: mark a hazard only after looming has fired for this many consecutive fixation steps
    world = World(seed=seed); eye = make_eye(120); sa = eye[1] - eye[0]
    fine = (eye[:, None] + sa * (np.arange(OVERSAMPLE) / OVERSAMPLE - .5 + .5 / OVERSAMPLE)).ravel()
    rng = np.random.default_rng(seed + 999); grng = np.random.default_rng(seed + 4242)
    p = world.start.copy(); h = np.pi / 2 + rng.normal(0, .2)
    if task == 'goal':
        while True:
            g = np.array([grng.uniform(5, 35), grng.uniform(30, 37)])
            if not len(world.r) or (np.linalg.norm(world.c - g, axis=1) > world.r + 1.0).all(): break
    else:
        planner = (LawnPlanner(world, memory, line_check, expire, start=p) if lawn else Planner(world, memory, line_check, expire)); planner.turn_w, planner.min_d = turn_w, min_d; g = planner.update(p, 0.0, h)
    nav = GoalNav(eye, DT, seed=seed, base=base, goal=g, **kw)
    if wander: nav.c.goal_err = None
    t, dist_flown, posts, walls, reached = 0.0, 0.0, 0, 0, None
    loom_run = 0
    if flowmap:
        import json as _j; from pathlib import Path as _P
        FC = _j.loads((_P(__file__).resolve().parents[1] / 'results/flow_depth_coef.json').read_text())
    rec = dict(x=[], y=[], h=[], state=[], loom=[], thr=[], looming=[], img=[], goal=[], crashes=[])
    while t < T:
        df, lf = world.cast(p, h + fine)
        lum = lf.reshape(-1, OVERSAMPLE).mean(1) + rng.normal(0, .02, 120)
        if wander: yaw, v = nav.c.act(lum, df.reshape(-1, OVERSAMPLE).min(1)); nav.diag = getattr(nav.c, 'diag', {})
        else:
            if task != 'goal' and memory: nav.haz = planner.haz
            yaw, v = nav.act(lum, df.reshape(-1, OVERSAMPLE).min(1), p, h)
        if record:
            dg = nav.diag; rec['x'].append(float(p[0])); rec['y'].append(float(p[1])); rec['h'].append(float(h))
            rec['state'].append(dg.get('state', 'warmup')); rec['loom'].append(float(dg.get('loom', 0)))
            th = dg.get('thr', np.nan); rec['thr'].append(None if th is None or not np.isfinite(th) else float(th))
            rec['looming'].append(int(bool(dg.get('looming', False)))); rec['img'].append(lum); rec['goal'].append(None if g is None else [float(g[0]), float(g[1])])
        loom_run = loom_run + 1 if (nav.diag.get('looming') and nav.diag.get('state') == 'fixate') else 0
        if task != 'goal' and memory and loom_run >= persist:
            off = 0.0
            if getattr(nav.c, 'l2', None) is not None and getattr(nav.c, 'loom_mode', '') == 'lplc2': off = .25 if nav.c.l2[0] > nav.c.l2[1] else -.25
            planner.mark(p + loom_d * np.array([np.cos(h + off), np.sin(h + off)]))
        if task != 'goal' and memory and flowmap and nav.diag.get('state') == 'fixate' and getattr(nav.c, 't', 0) > 0.9:
            c_ = nav.c; side = (np.abs(c_.mid) > np.radians(15)) & (np.abs(c_.mid) < np.radians(150)) & (c_.out_ema > 1e-4)
            idx = np.where(side)[0][::3]
            if len(idx):
                dh = np.exp(FC['a'] + FC['b'] * np.log(max(v, 1e-3) * np.abs(np.sin(c_.mid[idx]))) + FC['c'] * np.log(c_.out_ema[idx]))
                for k in idx[dh < FC['near_thr']]:
                    planner.mark(p + fm_d * np.array([np.cos(h + c_.mid[k]), np.sin(h + c_.mid[k])]), ttl=fm_expire)
        h0 = h
        h += np.clip(yaw, -12, 12) * DT; prev = p; p = p + v * DT * np.array([np.cos(h), np.sin(h)]); t += DT
        dist_flown += v * DT
        if world.collides(p, .25):
            k = crash_kind(world, p); posts += k == 'post'; walls += k == 'wall'
            if record: rec['crashes'].append([len(rec['x']), float(p[0]), float(p[1]), k])
            if task != 'goal': planner.mark(p + .4 * np.array([np.cos(h0), np.sin(h0)]), perm=True)
            p = prev; h += np.pi + rng.uniform(-.5, .5)
        if task == 'goal':
            if np.linalg.norm(p - g) < GOAL_R: reached = t; break
        else:
            g = planner.update(p, t, h)
            if g is None: break
            nav.goal = g
    out = dict(posts=posts, walls=walls, t=t, dist=dist_flown, world=world)
    if task == 'goal': out.update(reached=reached is not None, time=reached, straight=float(np.linalg.norm(g - world.start)))
    else: out.update(coverage=planner.coverage(), hazards=[[round(float(a), 2), round(float(b), 2), int(round(tt / DT))] for (a, b), tt in zip(planner.haz, planner.haz_t)] if record else len(planner.haz))
    if record: out['rec'] = rec
    return out


class LawnPlanner(Planner):
    """Boustrophedon ('lawnmower') coverage: rows CELL m apart, waypoints every WP m along each row, alternating
    direction, starting from the drone's start on the first row. Far, collinear waypoints -> few goal turns.
    A waypoint is skipped when its cell is already visited, when it sits near a remembered hazard (memory=True),
    or after STUCK_ROW s without reaching it. Same interface as Planner (update / mark / coverage / haz)."""
    WP, REACH, STUCK_ROW = 6.0, 1.5, 12.0

    def __init__(self, world, memory=False, line_check=True, expire=None, start=None):
        super().__init__(world, memory, line_check, expire)
        L = world.size; lo, hi = MARGIN + 1.0, L - MARGIN - 1.0
        xs = np.arange(lo, hi + 1e-6, self.WP); xs[-1] = hi
        seq = []
        for j in range(self.n):
            y = MARGIN + (j + .5) * CELL; row = xs if j % 2 == 0 else xs[::-1]
            if j == 0 and start is not None: row = row[row >= start[0]] if j % 2 == 0 else row
            seq += [(x, y) for x in row]
        self.seq = np.array(seq); self.i = 0

    def update(self, p, t):
        self.now = t
        if self.expire is not None and len(self.haz):
            keep = np.array([pm or t - tt < self.expire for tt, pm in zip(self.haz_t, self.haz_perm)])
            if not keep.all():
                self.haz = self.haz[keep]; self.haz_t = [v for v, k in zip(self.haz_t, keep) if k]; self.haz_perm = [v for v, k in zip(self.haz_perm, keep) if k]
        c = self.cell_of(p)
        if c is not None: self.visited[c] = True
        if self.cur is None: self.t_cur = t; self.cur = 0
        while self.i < len(self.seq):
            w = self.seq[self.i]; cw = self.cell_of(w)
            reached = np.linalg.norm(w - p) < self.REACH
            near_h = self.memory and len(self.haz) and np.linalg.norm(self.haz - w, axis=1).min() < HAZ_R
            done = cw is not None and self.visited[cw] and np.linalg.norm(w - p) > 3.0
            if reached or near_h or done or t - self.t_cur > self.STUCK_ROW:
                self.i += 1; self.t_cur = t
            else:
                return w
        return super().update(p, t)          # rows finished: fall back to nearest unvisited cell
