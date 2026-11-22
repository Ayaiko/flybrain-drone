"""Fly-vision controller for the Webots DJI Mavic 2 Pro.

Vision: a 480x8 px spherical camera ('flyeye', 330 deg) -> grey -> average rows and every 4 columns -> 120 ommatidia,
the same eye as the 2D sim. The FlyVision circuit (flydrone/controllers.py) turns that into a yaw-rate command and a
forward speed. A simple attitude/velocity stabiliser (after Cyberbotics' mavic2pro_patrol example) flies the drone.
Crashes (drone body within 0.3 m of a post or wall) are counted from the world layout file, and the drone is put back
where it was 0.5 s earlier and turned around ('bounce', like the 2D sim). Supervisor rights are used only for that and
for metrics; the controller never sees distances.

controllerArgs: mode ('old' | 'lplc2' | 'none'), seed, duration_s.  Writes webots/runs/<mode>_<seed>.json (viewer format)."""
import base64, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
(ROOT / 'webots/runs').mkdir(exist_ok=True)
from controller import Supervisor                      # noqa: E402
from flydrone.controllers import FlyVision             # noqa: E402
from flydrone.sim import make_eye                      # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else 'lplc2'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DURATION = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
ALT, SPEED, BODY_R = 1.5, 2.0, 0.3
import os
KV, PCL, KYAW, KLAT = (float(os.environ.get(k, d)) for k, d in (('FLY_KV', 1.5), ('FLY_PCL', 4.0), ('FLY_KYAW', 3.0), ('FLY_KLAT', 1.2)))
KI = float(os.environ.get('FLY_KI', 2.0))
STATES = ['warmup', 'settling', 'fixate', 'saccade', 'saccade_start']


def clamp(v, a, b): return min(max(v, a), b)


class FlyMavic(Supervisor):
    K_VERTICAL_THRUST, K_VERTICAL_OFFSET, K_VERTICAL_P, K_ROLL_P, K_PITCH_P = 68.5, 0.6, 3.0, 50.0, 30.0

    def __init__(self):
        super().__init__()
        self.ts = int(self.getBasicTimeStep()); self.vis_every = 3; self.dt = self.ts * self.vis_every / 1000.0   # 24 ms ~ 42 fps
        self.eye_cam = self.getDevice('flyeye'); self.eye_cam.enable(self.ts * self.vis_every)
        for name in ('inertial unit', 'gps', 'gyro'):
            self.getDevice(name).enable(self.ts)
        self.imu, self.gps, self.gyro = self.getDevice('inertial unit'), self.getDevice('gps'), self.getDevice('gyro')
        self.getDevice('camera').enable(64)
        self.getDevice('camera pitch').setPosition(0.0)
        self.motors = [self.getDevice(n) for n in ('front left propeller', 'front right propeller', 'rear left propeller', 'rear right propeller')]
        for m in self.motors: m.setPosition(float('inf')); m.setVelocity(1)
        self.meta = json.loads((ROOT / f'webots/worlds/forest_{SEED}.json').read_text())
        self.posts = np.array([[p['x'], p['y'], p['r']] for p in self.meta['posts']])
        self.node = self.getSelf(); self.tf = self.node.getField('translation'); self.rf = self.node.getField('rotation')
        eye = make_eye(120)
        # MODE = base[-g][-v<speed>]: base old|lplc2|none; g = turn in place + gyro gate (20 deg/s); v = cruise speed m/s
        parts = MODE.split('-'); base = parts[0]; speed = SPEED
        kw = dict(sac_rate=np.radians(200))           # a Mavic cannot yaw at a fly's 600 deg/s
        if base == 'lplc2': kw.update(loom_mode='lplc2', lplc2_thr=2.0, front=35)
        self.klat, self.kyaw = KLAT, KYAW
        for p_ in parts[1:]:
            if p_ == 'g': kw.update(rot_gate=np.radians(20), sac_speed=0.0)
            elif p_.startswith('v'): speed = float(p_[1:])
            elif p_.startswith('L'): self.klat = float(p_[1:].replace('m', '-'))
            elif p_.startswith('Y'): self.kyaw = float(p_[1:])
        self.speed = speed
        self.fly = FlyVision(eye, self.dt, seed=SEED, speed=speed, **kw) if base not in ('none', 'test') else None
        self.test = base == 'test'
        self.rng = np.random.default_rng(SEED + 7)

    def eye_image(self):
        img = np.frombuffer(self.eye_cam.getImage(), np.uint8).reshape(self.eye_cam.getHeight(), self.eye_cam.getWidth(), 4)
        g = img[:, :, :3].mean(2).mean(0) / 255.0                      # grey, averaged over the 8 rows
        g = g[::-1].reshape(120, 4).mean(1)                             # image left = drone left = +angle -> index 119
        return g + self.rng.normal(0, .02, 120)

    def crashed(self, x, y):
        w = self.meta['walls']
        if x < w['xmin'] + BODY_R or x > w['xmax'] - BODY_R or y < w['ymin'] + BODY_R or y > w['ymax'] - BODY_R: return 'wall'
        d = np.hypot(self.posts[:, 0] - x, self.posts[:, 1] - y) - self.posts[:, 2]
        return 'post' if (d < BODY_R).any() else None

    def run(self):
        yaw_cmd, v_cmd, step, airborne, vi = 0.0, 0.0, 0, False, 0.0
        recover, turn_sign, in_contact = 0.0, 1, False
        v_ref, flipped = 0.0, None
        hist, rec, crashes = [], dict(x=[], y=[], h=[], state=[], loom=[], thr=[], looming=[], img=[], pitch=[], roll=[], wy=[], vf=[], front_d=[], near_b=[], yawc=[]), []
        cells, dist, last_xy, t0 = set(), 0.0, None, None
        logf = open(ROOT / 'webots/runs/progress.log', 'a'); logf.write(f'start {MODE} {SEED}\n'); logf.flush()
        while self.step(self.ts) != -1:
            t = self.getTime(); step += 1
            if step % 250 == 0:
                logf.write(f't={t:.1f} z={self.gps.getValues()[2]:.2f} airborne={airborne} frames={len(rec["x"])} crashes={len(crashes)}\n'); logf.flush()
            if t > DURATION + 30: break
            roll, pitch, yaw = self.imu.getRollPitchYaw(); x, y, z = self.gps.getValues(); vx, vy, _ = self.gps.getSpeedVector()
            wr, wp, wy = self.gyro.getValues()
            if not airborne and z > ALT - 0.2: airborne, t0 = True, t
            if airborne and flipped is None and (abs(roll) > 1.0 or abs(pitch) > 1.0): flipped = round(t - t0, 2)   # > 57 deg = lost
            if airborne and step % self.vis_every == 0:
                lum = self.eye_image()
                if recover > 0:                                # after a crash: back off, then turn ~180 deg in place
                    recover -= self.dt
                    yaw_cmd, v_cmd = (0.0, -0.8) if recover > 1.6 else (turn_sign * 2.0, 0.0); dg = {'state': 'warmup'}
                    if self.fly is not None: self.fly.act(lum)  # keep its filters running
                elif self.fly is not None:
                    self.fly.ext_rot = wy                      # gyro (haltere) yaw rate for the rotation gate
                    yaw_cmd, v_cmd = self.fly.act(lum); dg = self.fly.diag
                elif self.test:                                # scripted stabiliser test: fly 3 s, stop+turn 90 deg at 200 deg/s, hold 1.5 s
                    ph = (t - t0) % 5.0; dg = {'state': 'fixate'}
                    if ph < 3.0: yaw_cmd, v_cmd = 0.0, self.speed
                    elif ph < 3.45: yaw_cmd, v_cmd = np.radians(200), 0.0; dg = {'state': 'saccade'}
                    else: yaw_cmd, v_cmd = 0.0, 0.0; dg = {'state': 'settling'}
                else:
                    yaw_cmd, v_cmd, dg = 0.0, self.speed, {}
                rec['x'].append(round(x - 0, 3)); rec['y'].append(round(y, 3)); rec['h'].append(round(yaw, 3))
                rec['state'].append(STATES.index(dg.get('state', 'warmup')) if dg.get('state', 'warmup') in STATES else 0)
                rec['loom'].append(round(float(dg.get('loom', 0)), 4)); th = dg.get('thr', np.nan)
                rec['thr'].append(None if th is None or not np.isfinite(th) else round(float(th), 4))
                rec['looming'].append(int(bool(dg.get('looming', False)))); rec['img'].append(np.clip(lum * 255, 0, 255).astype(np.uint8).tobytes())
                # diagnostics only (truth from the layout file; the fly never sees these)
                c_, s_ = np.cos(yaw), np.sin(yaw); rel = self.posts[:, :2] - [x, y]
                dd = np.hypot(rel[:, 0], rel[:, 1]) - self.posts[:, 2]; bb = np.arctan2(rel[:, 1], rel[:, 0]) - yaw
                bb = (bb + np.pi) % (2 * np.pi) - np.pi; fm = np.abs(bb) < np.radians(30)
                j = int(np.argmin(np.where(fm, dd, np.inf))) if fm.any() else int(np.argmin(dd))
                for k_, v_ in (('pitch', pitch), ('roll', roll), ('wy', wy), ('vf', c_ * vx + s_ * vy), ('front_d', dd[j] if fm.any() else 99.0),
                               ('near_b', bb[int(np.argmin(dd))]), ('yawc', yaw_cmd)):
                    rec[k_].append(round(float(v_), 4))
                hist.append((x, y, yaw))
                if last_xy is not None: dist += np.hypot(x - last_xy[0], y - last_xy[1])
                last_xy = (x, y); cells.add((int((x + self.meta['origin'][0]) // 2), int((y + self.meta['origin'][1]) // 2)))
                kind = self.crashed(x, y)
                if kind and not in_contact and not self.test:                    # count each new contact once (posts have no collision shape)
                    crashes.append([len(rec['x']) - 1, round(x, 3), round(y, 3), kind])
                    recover = 2.2 + self.rng.uniform(-.25, .25); turn_sign = self.rng.choice([-1, 1])
                in_contact = bool(kind)
                if t - t0 > DURATION: break
            # --- stabiliser: yaw-rate loop on the gyro, forward-speed loop on GPS velocity, lateral damping ---
            c, s = np.cos(yaw), np.sin(yaw); v_fwd = c * vx + s * vy; v_lat = -s * vx + c * vy
            if airborne:
                v_ref += clamp(v_cmd - v_ref, -1.5 * self.ts / 1000.0, 1.5 * self.ts / 1000.0)   # accel limit 1.5 m/s^2
                e = v_ref - v_fwd; vi = clamp(vi + e * self.ts / 1000.0, -3.0, 3.0)   # PI speed loop (damping eats a P-only loop)
                pitch_d = clamp(-(KV * e + KI * vi), -PCL, PCL)       # negative pitch disturbance = go forward
                roll_d = clamp(self.klat * v_lat, -1.5, 1.5)                    # cancel sideways drift
                yaw_in = clamp(self.kyaw * (yaw_cmd - wy), -2.0, 2.0)
            else:
                pitch_d = roll_d = yaw_in = 0.0
            roll_in = self.K_ROLL_P * clamp(roll, -1, 1) + wr + roll_d
            pitch_in = self.K_PITCH_P * clamp(pitch, -1, 1) + wp + pitch_d
            vert = self.K_VERTICAL_P * clamp(ALT - z + self.K_VERTICAL_OFFSET, -1, 1) ** 3
            fl = self.K_VERTICAL_THRUST + vert - yaw_in + pitch_in - roll_in
            fr = self.K_VERTICAL_THRUST + vert + yaw_in + pitch_in + roll_in
            rl = self.K_VERTICAL_THRUST + vert + yaw_in - pitch_in - roll_in
            rr = self.K_VERTICAL_THRUST + vert - yaw_in - pitch_in + roll_in
            for m, v in zip(self.motors, (fl, -fr, -rl, rr)): m.setVelocity(v)
        posts = sum(c[3] == 'post' for c in crashes); walls = sum(c[3] == 'wall' for c in crashes)
        T = len(rec['x']) * self.dt
        summary = dict(mode=MODE, seed=SEED, flown_s=round(T, 1), dist_m=round(dist, 1), cells=len(cells),
                       post_crashes=posts, wall_crashes=walls, flipped_at=flipped, post_per_min=round(posts * 60 / max(T, 1e-6), 2))
        print('SUMMARY', json.dumps(summary), flush=True)
        out = ROOT / f'webots/runs/{MODE}_{SEED}.json'; out.parent.mkdir(exist_ok=True)
        run = {k: v for k, v in rec.items() if k != 'img'}
        o = self.meta['origin']
        run['x'] = [round(v + o[0], 3) for v in rec['x']]; run['y'] = [round(v + o[1], 3) for v in rec['y']]   # back to 2D-sim coords
        run['img'] = base64.b64encode(b''.join(rec['img'])).decode()
        run['crashes'] = [[k, round(cx + o[0], 3), round(cy + o[1], 3)] for k, cx, cy, kind in crashes if kind == 'post']
        out.write_text(json.dumps(dict(summary=summary, dt=self.dt, run=run), separators=(',', ':')))
        self.next_world()

    def next_world(self):
        # batch mode: one Webots instance works through webots/runs/queue.json (one window for the whole batch)
        q = ROOT / 'webots/runs/queue.json'
        jobs = json.loads(q.read_text()) if q.exists() else []
        if jobs:
            q.write_text(json.dumps(jobs[1:])); self.worldLoad(jobs[0])
        else:
            self.simulationQuit(0)



_robot = None
try:
    _robot = FlyMavic(); _robot.run()
except Exception:
    import traceback; traceback.print_exc()
    if _robot is not None: _robot.next_world()          # skip to the next job; never leave a paused window behind
