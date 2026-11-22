"""Fly brain -> PX4 bridge (runs inside the flybrain-px4 container, next to PX4 SITL + Gazebo).

Eye:   3 Gazebo cameras (110 deg, 160x8 px, yaw -110/0/+110) -> grey -> each pixel column mapped to its true azimuth
       (pinhole: atan((cx - u) / f)) -> resampled to the 120 evenly spaced ommatidia of the 2D sim (330 deg).
Brain: flydrone.controllers.FlyVision, unchanged; it outputs a yaw rate (rad/s, + = left) and a forward speed.
Body:  PX4 does all the flying. The brain's two numbers go to PX4 as an offboard VelocityBodyYawspeed setpoint
       (forward, right=0, down from an altitude hold, yaw rate in deg/s with PX4's + = clockwise).
Fence: a geofence (position from PX4/GPS in reality) keeps it inside the area: near the edge it turns to the centre.
Truth: the drone pose from Gazebo is used only to count crashes (posts have no collision shape) and for the log.
       On a crash the bridge does what a real drone would: back off 0.6 s, turn ~180 deg in place, carry on.

python3 px4/fly_px4.py <mode: none|old|lplc2>[-v<speed>] <seed> <seconds>   -> px4/runs/<mode>_<seed>.json"""
import asyncio, base64, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from flydrone.controllers import FlyVision, EMD # noqa: E402
from flydrone.sim import make_eye               # noqa: E402
from flydrone.world import World                # noqa: E402
from flydrone.nav_sim import Planner            # noqa: E402
from gz.transport13 import Node                 # noqa: E402
from gz.msgs10.image_pb2 import Image           # noqa: E402
from gz.msgs10.pose_v_pb2 import Pose_V         # noqa: E402
from mavsdk import System                       # noqa: E402
from mavsdk.offboard import VelocityBodyYawspeed, OffboardError   # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else 'lplc2'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DURATION = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
ALT, BODY_R, CAM_W, CAM_FOV, CAM_YAWS, FPS = 1.5, 0.35, 160, np.radians(110), np.radians([-110, 0, 110]), 30.0
import os
SAC_V, COOL = float(os.environ.get('FLY_SAC_V', 0.4)), float(os.environ.get('FLY_COOL', 1.2))
CALIB = 'calib' in (sys.argv[1] if len(sys.argv) > 1 else '')
CLEAR_THR = float(os.environ.get('FLY_CLEAR_THR', 1.5))
STATES = ['warmup', 'settling', 'fixate', 'saccade', 'saccade_start']
MODEL = 'x500_flyeye_0'


class Eye:
    """Collects the latest frame of each camera (with its sim timestamp) and builds the 120-ommatidium vector."""
    def __init__(self):
        self.frames, self.stamp = [None] * 3, [0.0] * 3
        f = (CAM_W / 2) / np.tan(CAM_FOV / 2); u = np.arange(CAM_W) + .5
        az = np.concatenate([y + np.arctan((CAM_W / 2 - u) / f) for y in CAM_YAWS])      # left of image = +azimuth
        self.eye = make_eye(120); half = (self.eye[1] - self.eye[0]) / 2
        self.W = np.array([(np.abs(az - e) <= half).astype(float) for e in self.eye]); self.W /= self.W.sum(1, keepdims=True)
        self.node = Node(); self.pose = None; self.att = (0.0, 0.0)
        for i in range(3): self.node.subscribe(Image, f'/flyeye/cam{i}', self._cb(i))
        self.node.subscribe(Pose_V, f'/world/forest_{SEED}/pose/info', self._pose)

    def _cb(self, i):
        def f(msg):
            self.frames[i] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            self.stamp[i] = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        return f

    def _pose(self, msg):
        for p in msg.pose:
            if p.name == MODEL:
                q = p.orientation; yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
                self.att = (np.arctan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x ** 2 + q.y ** 2)), np.arcsin(np.clip(2 * (q.w * q.y - q.z * q.x), -1, 1)))
                self.pose = (p.position.x, p.position.y, p.position.z, yaw)

    def ready(self, after): return all(f is not None for f in self.frames) and min(self.stamp) > after

    def vector(self):
        g = np.concatenate([f[:, :, :3].mean(2).mean(0) for f in self.frames]) / 255.0
        return self.W @ g


def fresh(fly):
    """After a manoeuvre the brain did not order (geofence / crash recovery), start a new fixation: its motion filters
    must not see the turn (same as after its own saccades - efference copy)."""
    if fly is None: return
    fly.emd = EMD(len(fly.eye), fly.dt, fly.tau_lp, fly.tau_hp); fly.prev_lum = None; fly.t = 0.45; fly.sac_left = 0.0
    fly.flow_ema[:] = 0.0; fly.out_ema[:] = 0.0


async def main():
    w = World(seed=SEED); o = w.start; posts = np.c_[w.c - o, w.r]; L = w.size
    walls = dict(xmin=-o[0], xmax=L - o[0], ymin=-o[1], ymax=L - o[1])
    eye = Eye(); rng = np.random.default_rng(SEED + 7)
    parts = MODE.split('-'); base = parts[0]; speed = 1.0
    kw = dict(sac_rate=np.radians(120), cool=COOL)
    if base == 'lplc2': kw.update(loom_mode='lplc2', lplc2_thr=2.0, front=35)
    cov = 'cov' in parts                       # area coverage: GPS-like planner + hazard memory + heading veto
    bold = 'bold' in parts                     # bolder: looming thr 2.5, hazard only if looming 3 frames in a row, forget after 20 s
    if cov and base != 'none': kw.update(goal_gap=2.0)
    for p_ in parts:
        if p_.startswith('gt'): kw.update(goal_thr=float(p_[2:]))      # gt30 -> only turn toward the goal when > 30 deg off
    if bold and base == 'lplc2': kw.update(lplc2_thr=2.5)
    gate = 'gate' in parts                     # haltere gate: ignore vision while body yaw > 15 deg/s (keep flying); end turns only when < 12 deg/s
    if gate and base != 'none': kw.update(rot_gate=np.radians(15), sac_speed=1.0)
    sac_end_wy = np.radians(12 if gate else 30)
    sac_v = SAC_V
    sac_clear = any(p_.startswith('svc') for p_ in parts)   # middle way: creep while turning only if the swept sector looked clear
    clear_thr = CLEAR_THR
    for p_ in parts:
        if p_.startswith('svc') and len(p_) > 3: clear_thr = float(p_[3:])          # svc3 -> threshold 3x median nearness
        elif p_.startswith('sv') and not p_.startswith('svc'): sac_v = float(p_[2:])   # forward speed fraction while turning
    turn_v = sac_v
    persist, expire = (3, 20.0) if bold else (1, None)
    flowmap = any(p_.startswith('fm') for p_ in parts)                    # map side obstacles from optic flow (2D-calibrated d ~ v|sin phi|/flow) into hazard memory
    FC = json.loads((ROOT / ('results/flow_depth_coef_px4.json' if (ROOT / 'results/flow_depth_coef_px4.json').exists() else 'results/flow_depth_coef.json')).read_text()) if flowmap else None
    for p_ in parts[1:]:
        if p_.startswith('v'): speed = float(p_[1:])
    planner = Planner(w, memory=True, line_check=True, expire=expire) if cov else None
    fly = FlyVision(eye.eye, 1 / FPS, seed=SEED, speed=speed, **kw) if base != 'none' else None

    print('connecting', flush=True)
    drone = System(mavsdk_server_address='localhost', port=50051); await drone.connect()   # server started by run.sh
    async for s in drone.core.connection_state():
        if s.is_connected: break
    for name, val in (('NAV_DLL_ACT', 0), ('COM_RC_IN_MODE', 4), ('NAV_RCL_ACT', 0), ('COM_OBL_RC_ACT', 0)):
        try: await drone.param.set_param_int(name, val)
        except Exception as e: print('param', name, e)
    for name, val in (('MPC_YAWRAUTO_MAX', 200.0), ('MPC_MAN_Y_MAX', 200.0)):
        try: await drone.param.set_param_float(name, val)
        except Exception as e: print('param', name, e)
    async for h in drone.telemetry.health():
        if h.is_armable and h.is_local_position_ok: break
    print('health ok, taking off', flush=True)
    await drone.action.set_takeoff_altitude(ALT); await drone.action.arm(); await drone.action.takeoff()
    alt = [0.0]
    async def alt_watch():
        async for pv in drone.telemetry.position_velocity_ned(): alt[0] = -pv.position.down_m
    asyncio.ensure_future(alt_watch())
    t_w = time.time()
    while alt[0] < ALT - 0.3 and time.time() - t_w < 30: await asyncio.sleep(.1)
    await asyncio.sleep(2)
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    try: await drone.offboard.start()
    except OffboardError as e: print('offboard failed', e); return
    print('offboard started', flush=True)

    rec = dict(x=[], y=[], h=[], state=[], loom=[], thr=[], looming=[], img=[], wy=[], yawc=[], goal=[], roll=[], pitch=[], z=[], front_d=[], near_d=[], l2l=[], l2r=[])
    hazards, goal, loom_run = [], None, 0
    calib = []

    def mark(q, perm=False, ttl=None):
        n0 = len(planner.haz); planner.mark(np.asarray(q, float), perm=perm, ttl=ttl)
        if len(planner.haz) > n0: hazards.append([round(float(q[0]), 2), round(float(q[1]), 2), len(rec['x'])])

    def goal_err(p2, h):
        err = (np.arctan2(goal[1] - p2[1], goal[0] - p2[0]) - h + np.pi) % (2 * np.pi) - np.pi
        if len(planner.haz):                          # never steer toward a remembered hazard closer than 4 m
            v_ = planner.haz - p2; dh = np.linalg.norm(v_, axis=1); m = dh < 4.0
            if m.any():
                bh = np.arctan2(v_[m, 1], v_[m, 0]) - h; half = np.arcsin(np.minimum(1, 1.0 / np.maximum(dh[m], 1e-3))) + np.radians(10)
                phis = np.radians(np.arange(-180, 180, 5))
                blocked = (np.abs((phis[:, None] - bh[None] + np.pi) % (2 * np.pi) - np.pi) < half[None]).any(1)
                if not blocked.all(): err = float(phis[~blocked][np.argmin(np.abs((phis[~blocked] - err + np.pi) % (2 * np.pi) - np.pi))])
        return err
    crashes, recover, turn_sign, in_contact = [], 0.0, 1, False
    t_last, t0, prev_h, dist, last_xy, cells = 0.0, None, None, 0.0, None, set()
    sac_target = None; n_clear, n_turns = [0], [0]; turn_kind = dict(looming=0, goal=0, balance=0)
    GF, fencing, fence_events = 1.5, False, 0; cx, cy = (walls['xmin'] + walls['xmax']) / 2, (walls['ymin'] + walls['ymax']) / 2
    while True:
        await asyncio.sleep(0.003)
        if not eye.ready(t_last) or eye.pose is None: continue
        t_last = max(eye.stamp); t0 = t0 or t_last
        if t_last - t0 > DURATION: break
        lum = eye.vector() + rng.normal(0, .02, 120)
        x, y, z, h = eye.pose; p2 = np.array([x + o[0], y + o[1]])
        v_meas = 0.0 if last_xy is None else float(np.hypot(x - last_xy[0], y - last_xy[1]) * FPS)
        if cov:
            goal = planner.update(p2, t_last - t0)
            if goal is None: break
        wy = 0.0 if prev_h is None else ((h - prev_h + np.pi) % (2 * np.pi) - np.pi) * FPS; prev_h = h
        # geofence (position layer, like a real drone's): within GF m of the area edge -> turn toward the centre and
        # come back in; the fly brain resumes once we are >= GF + 1 m inside and facing inward
        inside = min(x - walls['xmin'], walls['xmax'] - x, y - walls['ymin'], walls['ymax'] - y)
        if inside < GF and not fencing: fencing = True; fence_events += 1
        to_c = (np.arctan2(cy - y, cx - x) - h + np.pi) % (2 * np.pi) - np.pi
        if fencing and inside >= GF + 1.0 and abs(to_c) < np.radians(60): fencing = False; fresh(fly)
        if fencing:
            dg = {'state': 'warmup'}; yaw = float(np.clip(3.0 * to_c, -2.0, 2.0)); v = 0.6 if abs(to_c) < np.radians(30) else 0.0
        elif recover > 0:
            recover -= 1 / FPS; dg = {'state': 'warmup'}
            yaw, v = (0.0, -0.6) if recover > 1.6 else (turn_sign * 2.0, 0.0)
            if recover <= 0: fresh(fly)
        elif sac_target is not None:
            # closed-loop saccade: turn (in place) until the IMU heading reaches the target and the body has stopped
            # rotating, then hand back to the brain with a fresh fixation. A drone turns slower than a fly.
            err = (sac_target - h + np.pi) % (2 * np.pi) - np.pi; dg = {'state': 'saccade'}
            # PX4 x500 tops out near 60 deg/s with ~0.3 s lag: a gentle P loop that does not overshoot
            yaw, v = float(np.sign(err) * np.clip(1.8 * abs(err), np.radians(12), np.radians(60))), turn_v * speed
            if abs(err) < np.radians(10) and abs(wy) < sac_end_wy: sac_target = None; fresh(fly)
        elif fly is not None:
            fly.ext_rot = wy
            if cov: fly.goal_err = goal_err(p2, h)
            yaw, v = fly.act(lum); dg = fly.diag
            loom_run = loom_run + 1 if (dg.get('looming') and dg.get('state') == 'fixate') else 0
            if CALIB and dg.get('state') == 'fixate' and fly.t > 0.9 and v_meas > 0.2 and len(rec['x']) % 3 == 0:
                calib.append(dict(x=float(p2[0]), y=float(p2[1]), h=float(h), v=round(v_meas, 3), out=[round(float(u), 5) for u in fly.out_ema]))
            if cov and flowmap and dg.get('state') == 'fixate' and fly.t > 0.9 and v_meas > 0.2:
                side = (np.abs(fly.mid) > np.radians(15)) & (np.abs(fly.mid) < np.radians(150)) & (fly.out_ema > 1e-4)
                idx = np.where(side)[0][::3]
                if len(idx):
                    dh = np.exp(FC['a'] + FC['b'] * np.log(v_meas * np.abs(np.sin(fly.mid[idx]))) + FC['c'] * np.log(fly.out_ema[idx]))
                    for kk in idx[dh < FC['near_thr']]:
                        mark(p2 + 2.0 * np.array([np.cos(h + fly.mid[kk]), np.sin(h + fly.mid[kk])]), ttl=10.0)
            if cov and loom_run >= persist:
                off = (.25 if fly.l2[0] > fly.l2[1] else -.25) if getattr(fly, 'loom_mode', '') == 'lplc2' else 0.0
                mark(p2 + 2.5 * np.array([np.cos(h + off), np.sin(h + off)]))
            if dg.get('state') == 'saccade_start':
                ang = fly.sac_dir * fly.sac_left * fly.sac_rate
                turn_kind['looming' if dg.get('big') else 'goal' if dg.get('goal') else 'balance'] += 1
                sac_target = h + ang; fly.sac_left = 0.0
                if sac_clear:
                    # nearness ~ outward flow / |sin(azimuth)| measured during the fixation just before the turn; look at the
                    # sector from straight ahead to the new heading (+-20 deg). Clear -> creep 0.4x while turning, else turn in place.
                    near = np.convolve(fly.out_ema / fly.sin_m, np.ones(5) / 5, 'same'); near = near / (np.median(near) + 1e-9)
                    lo, hi = min(0.0, ang) - np.radians(20), max(0.0, ang) + np.radians(20)
                    sector = (fly.mid > lo) & (fly.mid < hi)
                    turn_v = 0.4 if (not sector.any() or near[sector].max() < clear_thr) else 0.0
                    n_clear[0] += turn_v > 0; n_turns[0] += 1
        elif cov:                                      # beeline to the planner's waypoint, no avoidance
            e_ = (np.arctan2(goal[1] - p2[1], goal[0] - p2[0]) - h + np.pi) % (2 * np.pi) - np.pi
            yaw, v, dg = float(np.clip(1.8 * e_, -np.radians(60), np.radians(60))), speed, {}
        else:
            yaw, v, dg = 0.0, speed, {}
        dz = float(np.clip(1.0 * (ALT - z), -0.5, 0.5))
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(float(v), 0.0, -dz, float(-np.degrees(yaw))))
        rec['x'].append(round(x + o[0], 3)); rec['y'].append(round(y + o[1], 3)); rec['h'].append(round(h, 3))
        st = dg.get('state', 'warmup'); rec['state'].append(STATES.index(st) if st in STATES else 0)
        rec['loom'].append(round(float(dg.get('loom', 0)), 4)); th = dg.get('thr', np.nan)
        rec['thr'].append(None if th is None or not np.isfinite(th) else round(float(th), 4))
        rec['looming'].append(int(bool(dg.get('looming', False)))); rec['img'].append(np.clip(lum * 255, 0, 255).astype(np.uint8).tobytes())
        rec['wy'].append(round(float(wy), 3)); rec['yawc'].append(round(float(yaw), 3))
        rec['goal'].append(None if goal is None else [round(float(goal[0]), 2), round(float(goal[1]), 2)])
        # diagnostics only (truth): attitude, altitude, distance to the nearest post edge ahead (+-35 deg) and anywhere
        rel = posts[:, :2] - [x, y]; dd = np.hypot(rel[:, 0], rel[:, 1]) - posts[:, 2]
        bb = (np.arctan2(rel[:, 1], rel[:, 0]) - h + np.pi) % (2 * np.pi) - np.pi; fm = np.abs(bb) < np.radians(35) + np.arcsin(np.clip(posts[:, 2] / np.maximum(dd + posts[:, 2], 1e-3), 0, 1))
        rec['roll'].append(round(float(eye.att[0]), 4)); rec['pitch'].append(round(float(eye.att[1]), 4)); rec['z'].append(round(float(z), 3))
        rec['front_d'].append(round(float(dd[fm].min()) if fm.any() else 99.0, 2)); rec['near_d'].append(round(float(dd.min()), 2))
        l2 = getattr(fly, 'l2', None) if fly is not None else None
        rec['l2l'].append(round(float(l2[0]), 4) if l2 is not None else 0.0); rec['l2r'].append(round(float(l2[1]), 4) if l2 is not None else 0.0)
        if last_xy is not None: dist += np.hypot(x - last_xy[0], y - last_xy[1])
        last_xy = (x, y); cells.add((int((x + o[0]) // 2), int((y + o[1]) // 2)))
        wall = x < walls['xmin'] + BODY_R or x > walls['xmax'] - BODY_R or y < walls['ymin'] + BODY_R or y > walls['ymax'] - BODY_R
        post = (np.hypot(posts[:, 0] - x, posts[:, 1] - y) - posts[:, 2] < BODY_R).any()
        kind = 'wall' if wall else 'post' if post else None
        if kind and not in_contact:
            crashes.append([len(rec['x']) - 1, round(x + o[0], 3), round(y + o[1], 3), kind])
            recover = 2.2 + rng.uniform(-.25, .25); turn_sign = int(rng.choice([-1, 1]))
            if cov: mark(p2 + .4 * np.array([np.cos(h), np.sin(h)]), perm=True)
        in_contact = bool(kind)
        if len(rec['x']) % 150 == 0: print(f't={t_last - t0:5.1f} pos=({x:5.1f},{y:5.1f}) z={z:.2f} crashes={len(crashes)} state={st}', flush=True)

    T = len(rec['x']) / FPS; posts_n = sum(c[3] == 'post' for c in crashes)
    summary = dict(mode=MODE, seed=SEED, flown_s=round(T, 1), dist_m=round(dist, 1), cells=len(cells), post_crashes=posts_n,
                   wall_crashes=sum(c[3] == 'wall' for c in crashes), geofence_events=fence_events, post_per_min=round(posts_n * 60 / max(T, 1e-6), 2))
    if cov: summary.update(coverage=round(planner.coverage() * 100, 1), hazards=len(hazards))
    if sac_clear: summary.update(turns=n_turns[0], turns_moving=n_clear[0])
    summary.update(turn_kinds=turn_kind)
    print('SUMMARY', json.dumps(summary), flush=True)
    run = {k: v for k, v in rec.items() if k != 'img'}; run['img'] = base64.b64encode(b''.join(rec['img'])).decode()
    run['crashes'] = [c[:3] for c in crashes if c[3] == 'post']
    if cov: run['hazards'] = hazards
    else: run.pop('goal')
    out = ROOT / f'px4/runs/{MODE}_{SEED}.json'; out.parent.mkdir(exist_ok=True)
    if CALIB: (ROOT / f'px4/runs/calib_{SEED}.json').write_text(json.dumps(dict(seed=SEED, mid=[float(m) for m in fly.mid], rows=calib)))
    out.write_text(json.dumps(dict(summary=summary, dt=1 / FPS, run=run), separators=(',', ':')))
    try:                                   # save first, then land (the server sometimes drops the link at the end)
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0)); await drone.offboard.stop(); await drone.action.land()
    except Exception as e: print('landing call failed (results already saved):', type(e).__name__)


asyncio.run(main())
