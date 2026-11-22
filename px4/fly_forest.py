"""Fly brain -> PX4 bridge for the forest test field (px4/make_forest.py). Runs inside the flybrain-px4 container.

Same eye / brain / body as px4/fly_px4.py (3 cameras -> 120 ommatidia -> FlyVision -> PX4 offboard body velocity + yaw
rate at 30 Hz; PX4 does all stabilisation). What is new here is the mission and the truth:
  Mission: a GPS-style route down the corridor centre line. The navigation layer gives the brain the bearing to a carrot
           6 m ahead on the route (FlyVision.goal_err); the brain decides when to turn (saccades) and when to dodge.
           Remembered hazards (looming / crashes) veto goal turns toward them, as in the coverage runs.
  Fence:   field edge -> turn toward the carrot (position layer, not vision).
  Truth:   3D obstacle list from the generator (trunks, branches, foliage, shrubs, rock, boulder) vs. a 0.35 m x 0.15 m
           drone body -> collisions, clearance. Obstacles have no physics: 'none' (route following, no avoidance)
           flies through them, which counts what the route alone would hit; brain modes do the usual crash recovery
           (back off 0.6 s, turn ~180 deg in place).
  End:     reaching y >= exit_y (success if zero collisions) or the time limit.

python3 px4/fly_forest.py <config> <world name> <seconds>      e.g. lplc2-bold-gate-sv0-v0.8 ff_medium_0 360
   config = a name in px4/forest_configs.json or an old-style mode string; env FLY_OVERRIDES='{"goal_gap": 0}' overrides
   parameters, env RUN_ID names the output   -> px4/runs/forest/<RUN_ID>_<world>.json
   Early stop (stall_s / stall_m / max_coll in the config) ends hopeless runs; summary.end_reason says why it ended."""
import asyncio, base64, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from flydrone.sim import make_eye                # noqa: E402
from px4.forest_geom import Obstacles, RangeSensor   # noqa: E402
from px4.forest_pilot import Pilot                  # noqa: E402
from gz.transport13 import Node                  # noqa: E402
from gz.msgs10.image_pb2 import Image            # noqa: E402
from gz.msgs10.pose_v_pb2 import Pose_V          # noqa: E402
from mavsdk import System                        # noqa: E402
from mavsdk.offboard import VelocityBodyYawspeed, OffboardError   # noqa: E402

from px4.forest_params import resolve, Stall      # noqa: E402

MODE, WORLD = sys.argv[1], sys.argv[2]; DURATION = float(sys.argv[3]) if len(sys.argv) > 3 else 360.0
OVR = json.loads(os.environ.get('FLY_OVERRIDES') or '{}'); P = resolve(MODE, OVR); RUN_ID = os.environ.get('RUN_ID') or MODE
CAM_W, CAM_FOV, CAM_YAWS, FPS = 160, np.radians(110), np.radians([-110, 0, 110]), 30.0
STATES = ['warmup', 'settling', 'fixate', 'saccade', 'saccade_start']; MODEL = 'x500_flyeye_0'


class Eye:
    def __init__(self, world):
        self.frames, self.stamp = [None] * 3, [0.0] * 3
        f = (CAM_W / 2) / np.tan(CAM_FOV / 2); u = np.arange(CAM_W) + .5
        az = np.concatenate([y + np.arctan((CAM_W / 2 - u) / f) for y in CAM_YAWS])
        self.eye = make_eye(120); half = (self.eye[1] - self.eye[0]) / 2
        self.W = np.array([(np.abs(az - e) <= half).astype(float) for e in self.eye]); self.W /= self.W.sum(1, keepdims=True)
        self.node = Node(); self.pose = None
        for i in range(3): self.node.subscribe(Image, f'/flyeye/cam{i}', self._cb(i))
        self.node.subscribe(Pose_V, f'/world/{world}/pose/info', self._pose)

    def _cb(self, i):
        def f(msg):
            self.frames[i] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            self.stamp[i] = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        return f

    def _pose(self, msg):
        for p in msg.pose:
            if p.name == MODEL:
                q = p.orientation; yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
                self.pose = (p.position.x, p.position.y, p.position.z, yaw)

    def ready(self, after): return all(f is not None for f in self.frames) and min(self.stamp) > after
    def vector(self): return self.W @ (np.concatenate([f[:, :, :3].mean(2).mean(0) for f in self.frames]) / 255.0)


async def main():
    meta = json.loads((ROOT / f'px4/forest/{WORLD}.json').read_text()); obs = Obstacles(meta)
    A, PER, EXIT, F, ALT = meta['curve']['amp'], meta['curve']['period'], meta['exit_y'], meta['field'], meta['alt']
    xc = lambda y: A * np.sin(2 * np.pi * y / PER)
    zone_of = lambda y: next((n for n, a, b in meta['zones'] if a <= y < b), 'outside')
    eye = Eye(WORLD); rng = np.random.default_rng(meta['seed'] + 7)
    print('params', json.dumps(P), flush=True)
    pilot = Pilot(P, meta, eye.eye, fps=FPS, seed=meta['seed'])
    tof_s = RangeSensor(obs, ALT, P['tof_fov'], P['tof_max']) if P['tof'] else None
    stall = Stall(P['stall_s'], P['stall_m']); end_reason = None

    print('connecting', flush=True)
    drone = System(mavsdk_server_address='localhost', port=50051); await drone.connect()
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
    awake, tp = 0.0, time.time()        # 'awake' wall time: a gap > 5 s means the Mac slept (whole VM paused) -> not counted
    while alt[0] < ALT - 0.3 and awake < 120:
        await asyncio.sleep(.1); tn = time.time(); awake += (tn - tp) if tn - tp < 5 else 0.0; tp = tn
    if alt[0] < ALT - 0.5:            # PX4 did not take off (rare sim hiccup): no result file -> forest_eval retries
        print(f'TAKEOFF FAILED: altitude {alt[0]:.2f} m after 120 s', flush=True); return
    await asyncio.sleep(2)
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    try: await drone.offboard.start()
    except OffboardError as e: print('offboard failed', e); return
    print('offboard started', flush=True)

    rec = dict(x=[], y=[], z=[], h=[], state=[], looming=[], gap=[], img=[], yawc=[], v=[], tof=[])
    crashes, in_contact, bump, hit_ids = [], False, False, set()
    t_last, t0, dist, last_xy, reached, t_exit, max_y = 0.0, None, 0.0, None, False, None, -99.0
    zmin = {}                                       # per zone: min surface distance / min body gap

    idle, tp = 0.0, time.time()          # awake wall time without a new camera frame (sleep-aware, see takeoff)
    while True:
        await asyncio.sleep(0.003)
        if not eye.ready(t_last) or eye.pose is None:
            tn = time.time(); idle += (tn - tp) if tn - tp < 5 else 0.0; tp = tn
            if idle > 20:    # sim stalled (overload / lockstep hiccup): no result -> forest_eval retries
                print(f'SIM STALLED: no camera frame for 20 s at sim t={t_last - (t0 or t_last):.1f}', flush=True); return
            continue
        idle, tp = 0.0, time.time()
        t_last = max(eye.stamp); t0 = t0 or t_last; now = t_last - t0
        if now > DURATION: end_reason = 'timeout'; break
        lum = eye.vector() + rng.normal(0, .02, 120)
        x, y, z, h = eye.pose; max_y = max(max_y, y)
        if y >= EXIT: reached, t_exit, end_reason = True, now, 'exit'; break
        if stall.update(now, min(max(y, 0.0), EXIT)): end_reason = 'stalled'; break
        if P['max_coll'] is not None and len(crashes) >= P['max_coll']: end_reason = 'too_many_collisions'; break
        v_meas = 0.0 if last_xy is None else float(np.hypot(x - last_xy[0], y - last_xy[1]) * FPS)
        tof = tof_s.read(x, y, h)[0] if tof_s is not None else None
        v, yaw, dg = pilot.step(now, x, y, z, h, lum, v_meas, bump=bump, tof=tof)
        dz = float(np.clip(1.0 * (ALT - z), -0.5, 0.5))
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(float(v), 0.0, -dz, float(-np.degrees(yaw))))
        # truth
        gap, sd, kind, oid = obs.query([x, y, z]); zn = zone_of(y)
        zm = zmin.setdefault(zn, [np.inf, np.inf]); zm[0] = min(zm[0], sd); zm[1] = min(zm[1], gap)
        bump = gap < 0 and not in_contact
        if bump:
            crashes.append(dict(i=len(rec['x']), t=round(now, 2), x=round(x, 2), y=round(y, 2), z=round(z, 2), zone=zn, kind=kind, oid=oid,
                                rehit=oid in hit_ids, v=round(v_meas, 2), state=dg.get('state', 'warmup')))
            hit_ids.add(oid)
        in_contact = gap < 0 or (in_contact and gap < 0.05)      # 5 cm hysteresis: a graze counts once
        rec['x'].append(round(x, 3)); rec['y'].append(round(y, 3)); rec['z'].append(round(z, 3)); rec['h'].append(round(h, 3))
        st = dg.get('state', 'warmup'); rec['state'].append(STATES.index(st) if st in STATES else 0)
        rec['looming'].append(int(bool(dg.get('looming', False)))); rec['gap'].append(round(gap, 3)); rec['yawc'].append(round(float(yaw), 3))
        rec['v'].append(round(v_meas, 3)); rec['tof'].append(None if tof is None else round(tof, 2))
        rec['img'].append(np.clip(lum * 255, 0, 255).astype(np.uint8).tobytes())
        if last_xy is not None: dist += np.hypot(x - last_xy[0], y - last_xy[1])
        last_xy = (x, y)
        if len(rec['x']) % 300 == 0: print(f't={now:5.1f} pos=({x:5.1f},{y:5.1f}) z={z:.2f} zone={zn} crashes={len(crashes)} state={st}', flush=True)

    by_zone, by_kind = {}, {}
    for c in crashes: by_zone[c['zone']] = by_zone.get(c['zone'], 0) + 1; by_kind[c['kind']] = by_kind.get(c['kind'], 0) + 1
    gaps = np.array(rec['gap']) if rec['gap'] else np.array([np.nan])
    summary = dict(mode=RUN_ID, run_id=RUN_ID, config=MODE, overrides=OVR, params=P, end_reason=end_reason,
                   speed_factor=float(os.environ.get('PX4_SIM_SPEED_FACTOR') or 1), worker=os.environ.get('GZ_PARTITION'), world=WORLD, preset=meta['preset'], seed=meta['seed'], reached=reached, success=bool(reached and not crashes),
                   collisions=len(crashes), collisions_by_zone=by_zone, collisions_by_kind=by_kind,
                   dist_m=round(dist, 1), progress_m=round(min(max_y, EXIT), 1), time_s=round(t_exit if reached else len(rec['x']) / FPS, 1),
                   min_clear_m=round(float(min(z_[0] for z_ in zmin.values())), 3) if zmin else None,
                   min_gap_m=round(float(np.nanmin(gaps)), 3), frac_time_gap_lt_0p2=round(float(np.mean(gaps < 0.2)), 3),
                   min_clear_by_zone={k: round(float(v[0]), 2) for k, v in zmin.items()},
                   first_hits=len(hit_ids), rehits=sum(c['rehit'] for c in crashes), **pilot.stats(),
                   centreline_hits=meta['stats']['centreline_hits'])
    print('SUMMARY', json.dumps(summary), flush=True)
    run = {k: v for k, v in rec.items() if k != 'img'}; run['img'] = base64.b64encode(b''.join(rec['img'])).decode()
    run['crashes'] = crashes; run['hazards'] = pilot.hazards_log
    out = ROOT / f'px4/runs/forest/{RUN_ID}_{WORLD}.json'; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, dt=1 / FPS, run=run), separators=(',', ':')))
    try:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0)); await drone.offboard.stop(); await drone.action.land()
    except Exception as e: print('landing call failed (results already saved):', type(e).__name__)


asyncio.run(main())
