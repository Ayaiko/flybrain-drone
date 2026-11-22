"""Pilot for forest flights: everything between the eye and the autopilot, free of any simulator I/O, so the PX4 bridge
(px4/fly_forest.py) and a fast simulator can share exactly the same behaviour.

    pilot = Pilot(P, meta, eye_azimuths, fps=30, seed=0)
    v, yaw_rate, info = pilot.step(now, x, y, z, h, lum, v_meas, bump=False, tof=None)
        v        forward speed m/s (body), yaw_rate rad/s (+ = left / counter-clockwise)
        info     dict(state=..., looming=bool, ...) for logging

P is a resolved parameter dict (px4/forest_params.resolve). Layers, outermost first:
  geofence (position) > crash recovery > brake-before-turn > closed-loop turn > ToF veto > fly brain (FlyVision),
  plus the route carrot / hazard memory that feed the brain its goal bearing, and an optional optic-flow speed governor.
New behaviours (all off by default, so the defaults reproduce the earlier runs):
  recovery='detour'  back off rec_back_s, mark the obstacle, turn 90 deg to the freer side, slide rec_side_m, turn back to
                     the route, and shift the route carrot sideways by bypass_m for bypass_len m of progress
  brake='loom'|'all' before a looming (or any) turn, stop forward motion until speed < brake_v (max brake_max_s)
  flow_speed=true    forward speed x clip(flow_ref / lateral optic flow, v_min, 1): slow down in clutter (bees, flies)
  tof=true           forward range finder (tof_fov deg cone, tof_max m): closer than tof_stop + tof_t * speed -> brake +
                     90 deg escape turn (tof_action='turn') or the detour manoeuvre without the back-off (tof_action='detour')
  reacquire=true     when more than half a corridor width outside the corridor, aim the carrot 2 m ahead on the route
  creep_s > 0        look-then-go: after every turn / manoeuvre, fly at creep_v x speed for creep_s s (optic flow needs
                     motion to see depth, so the new direction is unknown right after turning on the spot)"""
import numpy as np
from flydrone.controllers import FlyVision, EMD


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def fresh(fly):
    """Start a new fixation after a manoeuvre the brain did not order (its filters must not see the turn)."""
    if fly is None: return
    fly.emd = EMD(len(fly.eye), fly.dt, fly.tau_lp, fly.tau_hp); fly.prev_lum = None; fly.t = 0.45; fly.sac_left = 0.0
    fly.flow_ema[:] = 0.0; fly.out_ema[:] = 0.0


class Hazards:
    def __init__(self, expire=None): self.p = np.zeros((0, 2)); self.t, self.perm, self.expire = [], [], expire

    def mark(self, q, now, perm=False):
        if not len(self.p) or np.linalg.norm(self.p - q, axis=1).min() > 0.5:
            self.p = np.vstack([self.p, q]); self.t.append(now); self.perm.append(perm); return True
        return False

    def prune(self, now):
        if self.expire is None or not len(self.p): return
        keep = np.array([pm or now - t < self.expire for t, pm in zip(self.t, self.perm)])
        self.p = self.p[keep]; self.t = [v for v, k in zip(self.t, keep) if k]; self.perm = [v for v, k in zip(self.perm, keep) if k]


class Pilot:
    def __init__(self, P, meta, eye, fps=30.0, seed=0):
        self.P, self.fps, self.dt = P, fps, 1.0 / fps
        A, PER = meta['curve']['amp'], meta['curve']['period']
        self.xc = lambda y: A * np.sin(2 * np.pi * y / PER)
        self.EXIT, self.F = meta['exit_y'], meta['field']; self.hw = meta['cfg']['width'] / 2
        self.rng = np.random.default_rng(seed + 11)
        kw = dict(sac_rate=np.radians(P['sac_rate']), cool=P['cool'], goal_gap=P['goal_gap'], goal_thr=P['goal_thr'], front=P['front'], sac_speed=P['sac_speed'])
        if P['brain'] == 'lplc2': kw.update(loom_mode='lplc2', lplc2_thr=P['lplc2_thr'])
        if P['rot_gate'] is not None: kw.update(rot_gate=np.radians(P['rot_gate']))
        self.fly = FlyVision(eye, self.dt, seed=seed, speed=P['speed'], **kw) if P['brain'] != 'none' else None
        self.haz = Hazards(P['haz_expire']); self.sac_end_wy = np.radians(P['sac_end_wy'])
        self.prev_h, self.loom_run, self.sac_target, self.brake_target, self.brake_t0 = None, 0, None, None, 0.0
        self.fencing, self.fence_events = False, 0
        self.rec = None                         # active recovery: dict(kind, phase, ...)
        self.bypass = None                      # (dx, until_y): route carrot shifted sideways after a detour
        self.v_scale = 1.0; self.t_settle = -1e9   # time the last turn / manoeuvre ended (look-then-go)
        self.turn_kind = dict(looming=0, goal=0, balance=0, tof=0); self.hazards_log = []; self.n_brakes = 0; self.n_recover = 0

    # ------------------------------------------------------------------ helpers
    def goal_err(self, p2, h, goal):
        err = wrap(np.arctan2(goal[1] - p2[1], goal[0] - p2[0]) - h)
        if len(self.haz.p):                     # never steer toward a remembered hazard closer than veto_r
            v_ = self.haz.p - p2; dh = np.linalg.norm(v_, axis=1); m = dh < self.P['veto_r']
            if m.any():
                bh = np.arctan2(v_[m, 1], v_[m, 0]) - h; half = np.arcsin(np.minimum(1, 1.0 / np.maximum(dh[m], 1e-3))) + np.radians(10)
                phis = np.radians(np.arange(-180, 180, 5))
                blocked = (np.abs(wrap(phis[:, None] - bh[None])) < half[None]).any(1)
                if not blocked.all(): err = float(phis[~blocked][np.argmin(np.abs(wrap(phis[~blocked] - err)))])
        return err

    def _turn(self, target, h, wy, v_fwd):
        """Closed-loop turn on the IMU heading (PX4 x500 tops out near 60 deg/s with ~0.3 s lag). -> (v, yaw, done)"""
        err = wrap(target - h)
        yaw = float(np.sign(err) * np.clip(1.8 * abs(err), np.radians(12), np.radians(60)))
        return v_fwd, yaw, abs(err) < np.radians(10) and abs(wy) < self.sac_end_wy

    def _begin_turn(self, target, now, big):
        mode = self.P['brake']
        if mode == 'all' or (mode == 'loom' and big): self.brake_target, self.brake_t0 = target, now; self.n_brakes += 1
        else: self.sac_target = target

    def _side_choice(self, p2, h):
        """Detour side: the one whose sidestep point is farther from remembered hazards and nearer the route."""
        best, best_s = None, 1
        for s in (1, -1):
            q = p2 + self.P['rec_side_m'] * np.array([np.cos(h + s * np.pi / 2), np.sin(h + s * np.pi / 2)])
            dens = float(np.exp(-((self.haz.p - q) ** 2).sum(1) / 0.8).sum()) if len(self.haz.p) else 0.0
            cost = dens + 0.3 * abs(q[0] - self.xc(q[1]))
            if best is None or cost < best: best, best_s = cost, s
        return best_s

    # ------------------------------------------------------------------ main step
    def step(self, now, x, y, z, h, lum, v_meas, bump=False, tof=None):
        P, fly = self.P, self.fly; p2 = np.array([x, y])
        wy = 0.0 if self.prev_h is None else wrap(h - self.prev_h) * self.fps; self.prev_h = h
        self.haz.prune(now)
        off_route = abs(x - self.xc(y)) - self.hw
        ahead = 2.0 if (P.get('reacquire') and off_route > self.hw * 0.5) else P['carrot']
        yg = min(max(y, 0.0) + ahead, self.EXIT + 2.0); gx = self.xc(yg)
        if self.bypass is not None:
            if y < self.bypass[1]: gx += self.bypass[0]
            else: self.bypass = None
        goal = np.array([gx, yg]); to_g = wrap(np.arctan2(goal[1] - y, goal[0] - x) - h)
        info = dict(state='warmup', looming=False, tof=tof)

        if bump and fly is not None:            # contact (a real drone: accelerometer spike) -> recovery
            self.haz.mark(p2 + .4 * np.array([np.cos(h), np.sin(h)]), now, perm=True)
            self.sac_target = self.brake_target = None; self.n_recover += 1
            if P['recovery'] == 'detour': self.rec = dict(kind='detour', phase='back', t0=now, h0=h)
            else: self.rec = dict(kind='flip', left=2.2 + self.rng.uniform(-.25, .25), sign=int(self.rng.choice([-1, 1])))

        # geofence (position layer): near the field edge turn toward the carrot and come back in
        inside = min(x - self.F['xmin'], self.F['xmax'] - x, y - self.F['ymin'], self.F['ymax'] - y)
        if inside < 1.5 and not self.fencing: self.fencing = True; self.fence_events += 1
        if self.fencing and inside >= 2.5 and abs(to_g) < np.radians(60): self.fencing = False; fresh(fly); self.t_settle = now
        if self.fencing:
            info['phase'] = 'geofence'
            return (0.6 if abs(to_g) < np.radians(30) else 0.0), float(np.clip(3.0 * to_g, -2.0, 2.0)), info

        if self.rec is not None:
            r = self.rec
            info['phase'] = r.get('phase', 'flip')
            if r['kind'] == 'flip':                         # old: back off 0.6 s, turn ~180 deg in place
                r['left'] -= self.dt; v, yaw = (-0.6, 0.0) if r['left'] > 1.6 else (0.0, r['sign'] * 2.0)
                if r['left'] <= 0: self.rec = None; fresh(fly); self.t_settle = now
                return v, yaw, info
            if r['phase'] == 'brake':
                if v_meas > P['brake_v'] and now - r['t0'] < P['brake_max_s']: return 0.0, 0.0, info
                s = self._side_choice(p2, r['h0']); r.update(phase='turn1', side=s, target=r['h0'] + s * np.pi / 2)
            if r['phase'] == 'back':
                if now - r['t0'] < P['rec_back_s']: return -P['rec_back_v'], 0.0, info
                s = self._side_choice(p2, r['h0']); r.update(phase='turn1', side=s, target=r['h0'] + s * np.pi / 2)
            if r['phase'] == 'turn1':
                info['phase'] = 'turn1'
                v, yaw, done = self._turn(r['target'], h, wy, 0.0)
                if not done: return v, yaw, info
                r.update(phase='side', start=p2.copy(), t1=now)
            if r['phase'] == 'side' and P.get('guard_side') and tof is not None:
                if tof < P['tof_stop'] + P['tof_t'] * v_meas:
                    r.update(phase='side_brake')
                    self.n_brakes += 1
            if r['phase'] == 'side_brake':
                info['phase'] = 'side_brake'
                if v_meas > P['brake_v']:
                    return 0.0, 0.0, info
                # Abort this leg. Reorient toward the route before selecting a new leg.
                self.bypass = None
                r.update(phase='turn2')
            if r['phase'] == 'side':
                info['phase'] = 'side'
                if np.linalg.norm(p2 - r['start']) < P['rec_side_m'] and now - r['t1'] < 4.0: return 0.4, 0.0, info
                r.update(phase='turn2')
                dx = P['bypass_m'] * np.cos(r['h0'] + r['side'] * np.pi / 2)
                self.bypass = (float(dx), y + P['bypass_len'])
            if r['phase'] == 'turn2':
                info['phase'] = 'turn2'
                tgt = np.arctan2(goal[1] - y, goal[0] - x)
                v, yaw, done = self._turn(tgt, h, wy, 0.0)
                if not done: return v, yaw, info
                self.rec = None; fresh(fly); self.t_settle = now

        info.pop('phase', None)
        if self.brake_target is not None:                    # stop first, then turn
            if v_meas < P['brake_v'] or now - self.brake_t0 > P['brake_max_s']:
                self.sac_target, self.brake_target = self.brake_target, None
            else:
                info['phase'] = 'brake'; info['state'] = 'saccade'; return 0.0, 0.0, info
        if self.sac_target is not None:
            info['state'] = 'saccade'
            v, yaw, done = self._turn(self.sac_target, h, wy, P['sac_v'] * P['speed'])
            if done: self.sac_target = None; fresh(fly); self.t_settle = now
            return v, yaw, info
        if fly is None:                                      # route following, no avoidance
            return P['speed'], float(np.clip(1.8 * to_g, -np.radians(60), np.radians(60))), info

        fly.ext_rot = wy; fly.goal_err = self.goal_err(p2, h, goal)
        yaw, v = fly.act(lum); dg = fly.diag; info.update(state=dg.get('state', 'warmup'), looming=bool(dg.get('looming', False)))
        if P['tof'] and tof is not None and tof < P['tof_stop'] + P.get('tof_t', 0.0) * v_meas and dg.get('state') in ('fixate', 'settling'):
            # independent range veto: something solid ahead that vision may not see (flat wall, plain surface)
            q = p2 + tof * np.array([np.cos(h), np.sin(h)])
            if self.haz.mark(q, now): self.hazards_log.append([round(float(q[0]), 2), round(float(q[1]), 2), 'tof'])
            l2 = getattr(fly, 'l2', None)
            side = (-1 if l2[0] > l2[1] else 1) if l2 is not None and l2.sum() > 0 else (1 if to_g > 0 else -1)
            self.turn_kind['tof'] += 1; fly.sac_left = 0.0; fly.cool = fly.cool_t
            if P.get('tof_action') == 'detour':        # go around it: brake, turn to the freer side, slide, turn back, bypass
                self.rec = dict(kind='detour', phase='brake', t0=now, h0=h); info['state'] = 'saccade_start'; return 0.0, 0.0, info
            self._begin_turn(h + side * np.pi / 2, now, big=True)
            info['state'] = 'saccade_start'; return 0.0, 0.0, info
        self.loom_run = self.loom_run + 1 if (info['looming'] and dg.get('state') == 'fixate') else 0
        if self.loom_run >= P['persist']:
            off = (.25 if fly.l2[0] > fly.l2[1] else -.25) if getattr(fly, 'loom_mode', '') == 'lplc2' else 0.0
            q = p2 + 2.5 * np.array([np.cos(h + off), np.sin(h + off)])
            if self.haz.mark(q, now): self.hazards_log.append([round(float(q[0]), 2), round(float(q[1]), 2), 'loom'])
        if dg.get('state') == 'saccade_start':
            big = bool(dg.get('big'))
            self.turn_kind['looming' if big else 'goal' if dg.get('goal') else 'balance'] += 1
            target = h + fly.sac_dir * fly.sac_left * fly.sac_rate; fly.sac_left = 0.0
            self._begin_turn(target, now, big)
            if self.brake_target is not None: return 0.0, 0.0, info
        if P['flow_speed'] and P['flow_ref'] and dg.get('state') == 'fixate':
            f = float(fly.flow_ema.sum()); want = float(np.clip(P['flow_ref'] / max(f, 1e-9), P['v_min'], 1.0))
            self.v_scale += 0.1 * (want - self.v_scale); v *= self.v_scale
        if P.get('creep_s', 0) > 0 and now - self.t_settle < P['creep_s']: v = min(v, P['creep_v'] * P['speed'])
        info['v_scale'] = round(self.v_scale, 3)
        return v, yaw, info

    def stats(self):
        return dict(geofence_events=self.fence_events, hazards=len(self.hazards_log), turn_kinds=self.turn_kind,
                    brakes=self.n_brakes, recoveries=self.n_recover)
