"""Navigation layer: knows the drone pose (as from GPS + compass) and a goal point, and hands the low-level
controller a bearing error. The fly controller turns it into goal saccades; its looming reflex still overrides.
GoalNav(base='fly'|'oracle'|'none')."""
import numpy as np
from .controllers import FlyVision, Oracle


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


class GoalNav:
    def __init__(self, eye, dt, seed=0, base='fly', goal=(20.0, 36.0), speed=2.0, veto_r=0.0, turn_slow=1.0, turn_slow_t=0.6, **kw):
        self.base, self.goal, self.speed = base, np.asarray(goal, float), speed
        # veto_r > 0: never steer toward a remembered hazard closer than veto_r m (haz is set by the planner each step)
        # turn_slow < 1: fly at that fraction of speed for turn_slow_t s after each goal saccade (time for the EMD to settle)
        self.veto_r, self.turn_slow, self.turn_slow_t, self.dt = veto_r, turn_slow, turn_slow_t, dt
        self.haz = np.zeros((0, 2)); self.slow_left = 0.0
        self.c = FlyVision(eye, dt, seed=seed, speed=speed, **kw) if base == 'fly' else Oracle(eye, dt, seed=seed, speed=speed)
        self.diag = {}

    def act(self, lum, dist, p, h):
        d = self.goal - p; err = wrap(np.arctan2(d[1], d[0]) - h)
        if self.base == 'fly':
            if self.veto_r > 0 and len(self.haz):
                v = self.haz - p; dh = np.linalg.norm(v, axis=1); m = dh < self.veto_r
                if m.any():
                    bh = np.arctan2(v[m, 1], v[m, 0]) - h; half = np.arcsin(np.minimum(1, 1.0 / np.maximum(dh[m], 1e-3))) + np.radians(10)
                    phis = np.radians(np.arange(-180, 180, 5))
                    blocked = (np.abs(wrap(phis[:, None] - bh[None])) < half[None]).any(1)
                    if not blocked.all(): err = float(phis[~blocked][np.argmin(np.abs(wrap(phis[~blocked] - err)))])
            self.c.goal_err = err; yaw, v = self.c.act(lum, dist); self.diag = self.c.diag
            if self.diag.get('state') == 'saccade_start' and self.diag.get('goal'): self.slow_left = self.turn_slow_t
            if self.slow_left > 0: self.slow_left -= self.dt; v *= self.turn_slow
            return yaw, v
        if self.base == 'oracle':                     # true depth: steer to the open direction closest to the goal
            eye = self.c.eye; dg = np.linalg.norm(self.goal - p); need = min(3.0, dg)
            win = int(np.ceil(np.radians(15) / (eye[1] - eye[0])))
            free = np.array([dist[max(0, i - win):i + win + 1].min() >= need for i in range(len(eye))])
            front = dist[np.abs(eye) < np.radians(25)].min()
            if not free.any(): return 4.0 * (1 if (np.sin(eye) / np.maximum(dist, .3) ** 2).sum() < 0 else -1), self.speed * .3
            cand = np.where(free, np.abs(wrap(eye - err)), np.inf); tgt = eye[cand.argmin()]
            push = -(np.sin(eye) / np.maximum(dist, .3) ** 2)[np.abs(eye) < np.radians(100)].sum()   # side repulsion
            return float(np.clip(5.0 * tgt + 1.0 * push, -6, 6)), self.speed * float(np.clip(front / 3, .3, 1))
        return float(np.clip(4.0 * err, -6, 6)), self.speed           # 'none': beeline, no avoidance
