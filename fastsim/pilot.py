"""Pilot = everything between the eye and the flight controller: the fly brain plus the mission layer around it
(route carrot, geofence, crash recovery, closed-loop saccades, hazard memory, altitude hold).

Interface (what fastsim.runner needs; a shared pilot module can replace BridgePilot by implementing the same):

    pilot = make_pilot(world, params, rng)          # world: fastsim.world.World, params: px4.forest_params.resolve(...)
    v, yaw, vz, diag = pilot.step(now, (x, y, z, h), lum)
        now  sim time s since start, pose = logged-style pose (ENU, heading rad, + = left turn), lum = 120 eye vector
             (already noisy, like fly_forest's eye.vector() + noise)
        ->   forward body speed m/s, yaw rate rad/s (+ = left), vertical speed m/s (+ = up),
             diag dict with at least 'state' (one of STATES) and 'looming'
    pilot.on_collision(now, (x, y, z, h))           # called once per new contact (5 cm hysteresis), after step()
    pilot.stats() -> dict merged into the episode summary (geofence_events, hazards, turn_kinds, ...)

BridgePilot mirrors the main loop of px4/fly_forest.py as of 2026-11-22 (md5 58b63ac18a7694f921ad3d75fa65e191;
a copy is in fastsim/ref/fly_forest_20261122.py) = the logic the PX4 forest results so far were flown with.
SharedPilot adapts the shared module px4/forest_pilot.Pilot (written by the other session; the PX4 bridge now uses it)
to this interface, feeding it v_meas, bump and the ToF range exactly like the new bridge loop does.
Default = 'shared' (the PX4 bridge itself runs px4/forest_pilot.Pilot, so both sims fly identical logic); 'bridge' is
kept only to reproduce the first calibration. make_pilot(..., kind='shared' | 'bridge'), or FASTSIM_PILOT=module:factory."""
import importlib, os
import numpy as np
from .world import ROOT  # noqa: F401  (puts the project root on sys.path)
from flydrone.controllers import FlyVision, EMD

FPS = 30.0
STATES = ['warmup', 'settling', 'fixate', 'saccade', 'saccade_start']


def fresh(fly):
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


class BridgePilot:
    def __init__(self, world, P, rng, eye_angles=None):
        m = world.meta; self.w, self.P, self.rng = world, P, rng
        self.A, self.PER, self.EXIT, self.F, self.ALT = m['curve']['amp'], m['curve']['period'], m['exit_y'], m['field'], m['alt']
        base, speed = P['brain'], P['speed']; self.base, self.speed, self.sac_v, self.CARROT = base, speed, P['sac_v'], P['carrot']
        kw = dict(sac_rate=np.radians(P['sac_rate']), cool=P['cool'], goal_gap=P['goal_gap'], goal_thr=P['goal_thr'], front=P['front'], sac_speed=P['sac_speed'])
        if base == 'lplc2': kw.update(loom_mode='lplc2', lplc2_thr=P['lplc2_thr'])
        if P['rot_gate'] is not None: kw.update(rot_gate=np.radians(P['rot_gate']))
        self.sac_end_wy = np.radians(P['sac_end_wy']); self.persist = P['persist']; self.VETO_R = P['veto_r']
        self.haz = Hazards(expire=P['haz_expire'])
        eye = eye_angles if eye_angles is not None else np.radians(np.linspace(-165, 165, 120))
        self.fly = FlyVision(eye, 1 / FPS, seed=m['seed'], speed=speed, **kw) if base != 'none' else None
        self.recover, self.turn_sign, self.loom_run, self.prev_h, self.sac_target = 0.0, 1, 0, None, None
        self.fence_events, self.fencing = 0, False
        self.turn_kind = dict(looming=0, goal=0, balance=0); self.hazards_log = []; self.n = 0; self._h = 0.0; self._p2 = None

    def xc(self, y): return self.A * np.sin(2 * np.pi * y / self.PER)

    def goal_err(self, p2, h, goal):
        haz = self.haz
        err = (np.arctan2(goal[1] - p2[1], goal[0] - p2[0]) - h + np.pi) % (2 * np.pi) - np.pi
        if len(haz.p):                                # never steer toward a remembered hazard closer than VETO_R
            v_ = haz.p - p2; dh = np.linalg.norm(v_, axis=1); m = dh < self.VETO_R
            if m.any():
                bh = np.arctan2(v_[m, 1], v_[m, 0]) - h; half = np.arcsin(np.minimum(1, 1.0 / np.maximum(dh[m], 1e-3))) + np.radians(10)
                phis = np.radians(np.arange(-180, 180, 5))
                blocked = (np.abs((phis[:, None] - bh[None] + np.pi) % (2 * np.pi) - np.pi) < half[None]).any(1)
                if not blocked.all(): err = float(phis[~blocked][np.argmin(np.abs((phis[~blocked] - err + np.pi) % (2 * np.pi) - np.pi))])
        return err

    def step(self, now, pose, lum):
        x, y, z, h = pose; p2 = np.array([x, y]); fly, F = self.fly, self.F; self._h, self._p2 = h, p2; self.n += 1
        yg = min(max(y, 0.0) + self.CARROT, self.EXIT + 2.0); goal = np.array([self.xc(yg), yg])
        wy = 0.0 if self.prev_h is None else ((h - self.prev_h + np.pi) % (2 * np.pi) - np.pi) * FPS; self.prev_h = h
        self.haz.prune(now)
        inside = min(x - F['xmin'], F['xmax'] - x, y - F['ymin'], F['ymax'] - y)
        to_g = (np.arctan2(goal[1] - y, goal[0] - x) - h + np.pi) % (2 * np.pi) - np.pi
        if inside < 1.5 and not self.fencing: self.fencing = True; self.fence_events += 1
        if self.fencing and inside >= 2.5 and abs(to_g) < np.radians(60): self.fencing = False; fresh(fly)
        if self.fencing:
            dg = {'state': 'warmup'}; yaw = float(np.clip(3.0 * to_g, -2.0, 2.0)); v = 0.6 if abs(to_g) < np.radians(30) else 0.0
        elif self.recover > 0:
            self.recover -= 1 / FPS; dg = {'state': 'warmup'}
            yaw, v = (0.0, -0.6) if self.recover > 1.6 else (self.turn_sign * 2.0, 0.0)
            if self.recover <= 0: fresh(fly)
        elif self.sac_target is not None:
            err = (self.sac_target - h + np.pi) % (2 * np.pi) - np.pi; dg = {'state': 'saccade'}
            yaw, v = float(np.sign(err) * np.clip(1.8 * abs(err), np.radians(12), np.radians(60))), self.sac_v * self.speed
            if abs(err) < np.radians(10) and abs(wy) < self.sac_end_wy: self.sac_target = None; fresh(fly)
        elif fly is not None:
            fly.ext_rot = wy; fly.goal_err = self.goal_err(p2, h, goal)
            yaw, v = fly.act(lum); dg = fly.diag
            self.loom_run = self.loom_run + 1 if (dg.get('looming') and dg.get('state') == 'fixate') else 0
            if self.loom_run >= self.persist:
                off = (.25 if fly.l2[0] > fly.l2[1] else -.25) if getattr(fly, 'loom_mode', '') == 'lplc2' else 0.0
                q = p2 + 2.5 * np.array([np.cos(h + off), np.sin(h + off)])
                if self.haz.mark(q, now): self.hazards_log.append([round(float(q[0]), 2), round(float(q[1]), 2), self.n - 1])
            if dg.get('state') == 'saccade_start':
                self.turn_kind['looming' if dg.get('big') else 'goal' if dg.get('goal') else 'balance'] += 1
                self.sac_target = h + fly.sac_dir * fly.sac_left * fly.sac_rate; fly.sac_left = 0.0
        else:                                          # route following, no avoidance
            yaw, v, dg = float(np.clip(1.8 * to_g, -np.radians(60), np.radians(60))), self.speed, {}
        dz = float(np.clip(1.0 * (self.ALT - z), -0.5, 0.5))
        return float(v), float(yaw), dz, dg

    def on_collision(self, now, pose):
        if self.fly is not None:
            h = pose[3]; p2 = np.array(pose[:2])
            self.recover = 2.2 + self.rng.uniform(-.25, .25); self.turn_sign = int(self.rng.choice([-1, 1])); self.sac_target = None
            self.haz.mark(p2 + .4 * np.array([np.cos(h), np.sin(h)]), now, perm=True)

    def stats(self):
        return dict(geofence_events=self.fence_events, hazards=len(self.hazards_log), turn_kinds=self.turn_kind)


class SharedPilot:
    """px4/forest_pilot.Pilot behind the fastsim interface (mirrors the loop of the current px4/fly_forest.py)."""
    def __init__(self, world, P, rng, eye_angles=None):
        from px4.forest_pilot import Pilot
        from px4.forest_geom import RangeSensor
        m = world.meta; self.ALT = m['alt']
        eye = eye_angles if eye_angles is not None else np.radians(np.linspace(-165, 165, 120))
        self.p = Pilot(P, m, eye, fps=FPS, seed=m['seed'])
        self.tof_s = RangeSensor(world.obs, self.ALT, P['tof_fov'], P['tof_max']) if P.get('tof') else None
        self.last_xy, self.bump = None, False

    def step(self, now, pose, lum):
        x, y, z, h = pose
        v_meas = 0.0 if self.last_xy is None else float(np.hypot(x - self.last_xy[0], y - self.last_xy[1]) * FPS)
        tof = self.tof_s.read(x, y, h)[0] if self.tof_s is not None else None
        v, yaw, info = self.p.step(now, x, y, z, h, lum, v_meas, bump=self.bump, tof=tof)
        self.bump = False; self.last_xy = (x, y)
        return float(v), float(yaw), float(np.clip(1.0 * (self.ALT - z), -0.5, 0.5)), info

    def on_collision(self, now, pose): self.bump = True          # seen by the pilot on the next frame, as in the bridge
    def stats(self): return self.p.stats()


def make_pilot(world, P, rng, kind=None):
    spec = os.environ.get('FASTSIM_PILOT')
    if spec:
        mod, fn = spec.split(':'); return getattr(importlib.import_module(mod), fn)(world, P, rng)
    return BridgePilot(world, P, rng) if kind == 'bridge' else SharedPilot(world, P, rng)
