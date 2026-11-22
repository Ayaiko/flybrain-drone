"""Controllers. Each gets the eye brightness vector (and, for the oracle only, ray distances) and returns
(yaw_rate rad/s, speed m/s). Eye angles are relative to heading, ordered from right (-) to left (+).

Fly-inspired pieces:
- EMD: Hassenstein-Reichardt correlator between neighbouring ommatidia (the computation T4/T5 cells implement):
  each input is high-passed, one arm low-pass delayed, R_i = D_i * I_{i+1} - I_i * D_{i+1}  (>0 = motion toward +angle).
- Translational flow: during forward flight the image streams outward (left side toward +, right toward -),
  faster for nearer surfaces -> side flow strength ~ nearness.
- Looming (LPLC2-like): strong outward flow on both sides of the frontal field = something approaching ahead.
- Saccade-and-fixate: fly straight while measuring, then turn fast toward the side with weaker flow; image motion
  during the saccade is ignored (efference copy), so self-rotation does not pollute the depth cue.
"""
import numpy as np


class EMD:
    def __init__(self, n, dt, tau_lp=0.035, tau_hp=0.25):
        self.a_lp = dt / (tau_lp + dt); self.a_hp = dt / (tau_hp + dt)
        self.hp_state = None; self.prev = None; self.lp = np.zeros(n)

    def step(self, lum):
        if self.prev is None:
            self.prev = lum.copy(); self.hp_state = lum.copy()
        self.hp_state += self.a_hp * (lum - self.hp_state)
        x = lum - self.hp_state                        # high-passed brightness
        self.lp += self.a_lp * (x - self.lp)           # delayed arm
        r = self.lp[:-1] * x[1:] - x[:-1] * self.lp[1:]
        return r                                       # n-1 local motion signals between neighbours


class Random:
    name = 'no_avoidance'
    def __init__(self, eye, dt, seed=0, speed=2.0):
        self.rng = np.random.default_rng(seed); self.speed = speed; self.yaw = 0.0
    def act(self, lum, dist=None):
        if self.rng.random() < 0.02: self.yaw = self.rng.normal(0, 1.0)
        return self.yaw, self.speed


class Oracle:
    """Upper bound with true distances (like a lidar): repulsion from near surfaces, slow down when close ahead."""
    name = 'oracle_depth'
    def __init__(self, eye, dt, seed=0, speed=2.0):
        self.eye = eye; self.speed = speed
    def act(self, lum, dist):
        w = 1.0 / np.maximum(dist, 0.3) ** 2
        push = -(np.sin(self.eye) * w).sum()            # surfaces on the left push right
        front = dist[np.abs(self.eye) < np.radians(25)].min()
        yaw = np.clip(1.5 * push, -4, 4)
        if front < 2.0: yaw = 4.0 * (1 if (np.sin(self.eye) * w).sum() < 0 else -1)
        return yaw, self.speed * np.clip(front / 3, 0.3, 1)


class FlyVision:
    """Brightness-only controller: EMD optic flow balance + looming escape, saccade-and-fixate or continuous."""
    def __init__(self, eye, dt, seed=0, speed=2.0, saccadic=True, side=(10, 110), front=35,
                 balance_thr=0.25, loom_thr=None, sac_rate=np.radians(600), tau_lp=0.035, tau_hp=0.25,
                 loom_k=2.5, sac_small=35, sac_big=90, cool=0.35, w_lgmd=0.0, slow=0.5,
                 loom_mode='sum', lplc2_thr=2.0, lplc2_dir=True, lplc2_base=True, lplc2_tau=3.0,
                 goal_thr=15, goal_hold=1.0, goal_max=120, goal_gap=0.0,
                 rot_gate=None, sac_speed=0.6, goal_hold_all=0.0):
        self.eye, self.dt, self.speed, self.saccadic = eye, dt, speed, saccadic
        self.name = 'fly_saccade' if saccadic else 'fly_continuous'
        self.tau_lp, self.tau_hp = tau_lp, tau_hp
        self.loom_k, self.sac_small, self.sac_big, self.cool_t, self.w_lgmd, self.slow = loom_k, sac_small, sac_big, cool, w_lgmd, slow
        self.prev_lum = None
        self.emd = EMD(len(eye), dt, tau_lp, tau_hp)
        mid = (eye[:-1] + eye[1:]) / 2
        self.outward = np.sign(mid)                        # outward direction per pair (+ on left, - on right)
        a = np.abs(np.degrees(mid))
        self.left = (mid > 0) & (a > side[0]) & (a < side[1]); self.right = (mid < 0) & (a > side[0]) & (a < side[1])
        self.frontm = a < front
        self.balance_thr = balance_thr; self.loom_thr = loom_thr; self.sac_rate = sac_rate
        self.sac_left = 0.0; self.sac_dir = 0; self.cool = 0.0; self.loom_ref = 1e-6
        self.flow_ema = np.zeros(2); self.t = 0.0
        # 'lplc2' mode: a frontal looming population split into a left and a right unit (outward flow pooled over
        # 0..front deg on each side). Threshold = lplc2_thr x a slowly adapting baseline of the same signal
        # (EMA, only during calm fixation) instead of the old max-ratchet; escape away from the stronger unit.
        self.loom_mode = loom_mode; self.lplc2_thr = lplc2_thr; self.l2_dir, self.l2_base = lplc2_dir, lplc2_base; self.l2_tau = lplc2_tau; self.base = None
        self.fl_m = (mid > 0) & (a < front); self.fr_m = (mid < 0) & (a < front)
        # goal steering (set from outside by a navigation layer): bearing error to the goal, rad (+ = goal on the left).
        # Steering is done with saccades like everything else, so rotation never pollutes the flow measurement.
        self.goal_err = None; self.goal_thr, self.goal_hold, self.goal_max = np.radians(goal_thr), goal_hold, np.radians(goal_max)
        # rot_gate (rad/s): body yaw rate from a gyro (like the fly's halteres), set from outside in self.ext_rot.
        # While the body still rotates faster than this, vision is not trusted (treated like the post-saccade settle).
        self.rot_gate, self.ext_rot, self.sac_speed = rot_gate, 0.0, sac_speed
        self.goal_hold_all = goal_hold_all          # > 0: after ANY turn, fly straight this long before turning back to the goal
        self.hold = 0.0; self.goal_gap = goal_gap; self.mid = mid; self.out_ema = np.zeros(len(mid))
        # gap-aware goal turns: nearness ~ outward flow / |sin(angle)| (translational flow is faster for near surfaces);
        # cost of heading phi = |phi - goal| + goal_gap * normalised nearness around phi  -> steer for the gap nearest the goal
        self.sin_m = np.maximum(np.abs(np.sin(mid)), np.sin(np.radians(10)))

    def act(self, lum, dist=None):
        self.t += self.dt
        r = self.emd.step(lum); self.r = r
        out = np.maximum(r * self.outward, 0)              # outward (translational) component only
        fl, fr = out[self.left].sum(), out[self.right].sum()
        dI = 0.0 if self.prev_lum is None else np.abs(lum - self.prev_lum)[1:][self.frontm].sum()
        self.prev_lum = lum.copy()
        loom = out[self.frontm].sum() + self.w_lgmd * dI   # EMD expansion + LGMD-like temporal contrast
        if self.loom_mode == 'lplc2':
            self.l2 = np.array([out[self.fl_m].sum(), out[self.fr_m].sum()]); loom = float(self.l2.sum())
        self.diag = dict(state='saccade', loom=loom, thr=np.nan, asym=np.nan, looming=False, loom_ref=self.loom_ref)
        if self.saccadic and self.sac_left > 0:            # mid-saccade: ignore vision (efference copy)
            self.sac_left -= self.dt
            return self.sac_dir * self.sac_rate, self.speed * self.sac_speed
        self.flow_ema += 0.15 * (np.array([fl, fr]) - self.flow_ema)
        self.out_ema += 0.15 * (out - self.out_ema)
        fl_s, fr_s = self.flow_ema
        asym = (fl_s - fr_s) / (fl_s + fr_s + 1e-6)        # >0: left side nearer
        self.loom_ref = max(self.loom_ref * 0.999, 1e-6) if self.t < 1 else self.loom_ref
        if self.rot_gate is not None and abs(self.ext_rot) > self.rot_gate and self.t >= 0.6:
            self.diag.update(state='settling', asym=asym, loom_ref=self.loom_ref)   # body still turning: don't trust flow
            return 0.0, self.speed * self.sac_speed
        if self.t < 0.6:                                   # let the filters settle
            self.loom_ref = max(self.loom_ref, loom)
            self.diag.update(state='settling', asym=asym, loom_ref=self.loom_ref)
            return 0.0, self.speed
        loom_thr = self.loom_thr if self.loom_thr is not None else self.loom_k * self.loom_ref
        if self.loom_mode == 'lplc2' and self.l2_base:
            if self.base is None: self.base = loom
            loom_thr = self.lplc2_thr * self.base
        looming = loom > loom_thr
        turn_dir = -1 if asym > 0 else 1                   # turn toward the side with weaker flow
        if self.loom_mode == 'lplc2':
            if looming and self.l2_dir: turn_dir = -1 if self.l2[0] > self.l2[1] else 1   # away from the expanding side
            elif not looming and self.l2_base: self.base += self.dt / self.l2_tau * (loom - self.base)
        self.diag.update(state='fixate', thr=loom_thr, asym=asym, looming=bool(looming))
        if not self.saccadic:
            yaw = -3.0 * asym + (turn_dir * 3.0 if looming else 0.0)
            return yaw, self.speed * (self.slow if looming else 1.0)
        self.cool -= self.dt; self.hold -= self.dt
        goal_target = self.goal_err
        if self.goal_err is not None and self.goal_gap > 0:
            near = np.convolve(self.out_ema / self.sin_m, np.ones(7) / 7, 'same'); near = near / (np.median(near) + 1e-9)
            ok = np.abs(self.mid) < np.radians(120)
            cost = np.abs((self.mid - self.goal_err + np.pi) % (2 * np.pi) - np.pi) + self.goal_gap * near
            if abs(self.goal_err) < np.radians(120):
                goal_target = float(self.mid[ok][np.argmin(cost[ok])])
        goal_turn = goal_target is not None and self.hold <= 0 and abs(goal_target) > self.goal_thr
        if self.cool <= 0 and (looming or goal_turn or abs(asym) > self.balance_thr):
            if looming:                                    # reflex wins; then fly past before heading for the goal again
                angle = np.radians(self.sac_big); self.hold = max(self.goal_hold, self.goal_hold_all)
            elif goal_turn:
                angle = min(abs(goal_target), self.goal_max); turn_dir = 1 if goal_target > 0 else -1
                if self.goal_hold_all > 0: self.hold = self.goal_hold_all
            else:
                angle = np.radians(self.sac_small)
                if self.goal_hold_all > 0: self.hold = max(self.hold, self.goal_hold_all)
            self.sac_dir = turn_dir; self.sac_left = angle / self.sac_rate; self.cool = self.cool_t
            self.emd = EMD(len(self.eye), self.dt, self.tau_lp, self.tau_hp)   # fresh fixation after the turn
            self.prev_lum = None
            self.t = 0.45
            self.diag.update(state='saccade_start', turn=int(turn_dir), big=bool(looming), goal=bool(goal_turn and not looming))
            return self.sac_dir * self.sac_rate, self.speed * self.sac_speed
        return 0.0, self.speed * (self.slow if looming else 1.0)


class Linear:
    """No structural prior: yaw and slow-down are a linear read-out of binned EMD signals (same EMD front end,
    so the comparison isolates the fly *circuit structure*, not the motion sensor)."""
    name = 'linear_scratch'
    NB = 24
    def __init__(self, eye, dt, seed=0, speed=2.0, w=None):
        self.eye, self.speed = eye, speed
        self.emd = EMD(len(eye), dt)
        self.w = np.zeros(2 * (2 * self.NB + 1)) if w is None else np.asarray(w)
        self.bins = np.array_split(np.arange(len(eye) - 1), self.NB)
    def act(self, lum, dist=None):
        r = self.emd.step(lum)
        f = np.array([r[b].sum() for b in self.bins] + [np.abs(r[b]).sum() for b in self.bins] + [1.0])
        n = len(f); yaw = 6 * np.tanh(self.w[:n] @ f); slow = 1 / (1 + np.exp(-(self.w[n:] @ f) - 3))
        return yaw, self.speed * (0.3 + 0.7 * slow)
