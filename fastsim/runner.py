"""Closed-loop forest episodes in the fast sim: N episodes per process stepping together, groups spread over cores.

One episode = the PX4 forest mission of px4/fly_forest.py with Gazebo + PX4 replaced by the ray-cast eye
(fastsim/kernel.py via eye.RayEye) and the fitted x500 model (fastsim/dynamics.py). Loop at 30 Hz sim time, per drone
in the same order as the bridge:
  timeout -> eye renders the pose of the previous frame (1-frame camera latency, measured) + N(0, 0.02) noise
  -> exit / stall / max-collision checks -> pilot (fastsim/pilot.py) -> truth (grid version of forest_geom.query,
  5 cm contact hysteresis) -> command into the x500 model (fitted delays).
Batching: run_group steps every unfinished episode of one world together - one kernel call renders all eyes, one
truth query and one dynamics step serve all drones; only the pilots (python objects) run in a loop. run_episode is
run_group with one episode. run_batch splits jobs into per-world groups of `group` episodes and runs them on a
process pool (default: all cores, minus 3 while flybrain-px4 containers are busy).

  .venv/bin/python -m fastsim.runner --configs none-v0.8,lplc2-bold-gate-sv0-v0.8 --presets easy,medium,hard \
        --reps 3 [--workers N] [--group 16] [--param goal_gap=0] [--pilot shared|bridge] [--out results/fastsim/runs.jsonl]
  --worlds ff_easy_0,ff_hard_1 instead of --presets / --seeds"""
import argparse, json, os, subprocess, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from .world import World, ROOT
from .eye import RayEye
from .dynamics import X500Batch
from .truth import BatchTruth
from .pilot import make_pilot, STATES, FPS
from px4.forest_params import resolve, run_id, parse_value, Stall

OUT = ROOT / 'results/fastsim'
_WORLD_CACHE = {}


def _world(name):
    if name not in _WORLD_CACHE:
        w = World(name); _WORLD_CACHE[name] = (w, BatchTruth(w.obs), {})
    return _WORLD_CACHE[name]


class _Ep:
    """Book-keeping of one episode inside a group."""
    def __init__(self, w, spec, pilot_kind):
        self.config, self.rep, self.ovr = spec['config'], spec.get('rep', 0), spec.get('overrides') or None
        self.tag = spec.get('tag')                                   # opaque, copied into the summary (e.g. CEM candidate id)
        self.P = resolve(self.config, self.ovr); meta = w.meta
        self.rng = np.random.default_rng([meta['seed'] + 7, self.rep])
        self.pilot_kind = spec.get('pilot', pilot_kind); self.pilot = make_pilot(w, self.P, self.rng, self.pilot_kind)
        self.x0, self.h0 = (0.03, np.pi / 2) if self.rep == 0 else (0.03 + self.rng.normal(0, 0.05), np.pi / 2 + self.rng.normal(0, np.radians(1)))
        self.stall = Stall(self.P['stall_s'], self.P['stall_m']); self.duration = spec.get('duration', 360.0)
        self.rec = dict(x=[], y=[], z=[], h=[], state=[], phase=[], looming=[], gap=[], yawc=[], v=[]); self.img = []
        self.crashes, self.hit_ids, self.in_contact, self.max_y, self.dist = [], set(), False, -99.0, 0.0
        self.reached, self.t_exit, self.end_reason, self.zmin, self.k, self.last = False, None, None, {}, 0, None


def run_group(world, specs, rows=4, image_lag=1, dyn_params=None, pilot='shared', record=False, prof=None):
    """specs: list of dict(config, rep=0, overrides=None, duration=360, pilot=...) on ONE world -> list of results
    (dict(summary=..., run=... if record)). prof: optional dict, receives wall time per stage."""
    t_wall = time.time(); wname = world if isinstance(world, str) else world.name
    w, truth, eyes = _world(wname); meta = w.meta
    if rows not in eyes: eyes[rows] = RayEye(w, rows=rows)
    eye = eyes[rows]; N = len(specs); EXIT = w.exit_y
    E = [_Ep(w, s, pilot) for s in specs]
    dyn = X500Batch(N, dyn_params); dyn.reset([e.x0 for e in E], np.zeros(N), np.full(N, w.alt - 0.09), [e.h0 for e in E])
    hist = [(dyn.x.copy(), dyn.y.copy(), dyn.z.copy(), dyn.h.copy())]       # pose history (for the camera latency)
    T = prof if prof is not None else {}
    for k_ in ('render', 'pilot', 'truth', 'dynamics', 'book'): T.setdefault(k_, 0.0)
    active = np.arange(N); k = 0
    while len(active):
        now = k / FPS; tq = time.perf_counter()
        active = np.array([i for i in active if not (now > E[i].duration and _end(E[i], 'timeout'))], int)
        if not len(active): break
        lp = hist[max(0, len(hist) - 1 - image_lag)]
        LUM = eye.render_batch(lp[0][active], lp[1][active], lp[2][active], lp[3][active]); t1 = time.perf_counter(); T['render'] += t1 - tq
        go, cmd = [], []
        for a, i in enumerate(active):
            e = E[i]; x, y, z, h = float(dyn.x[i]), float(dyn.y[i]), float(dyn.z[i]), float(dyn.h[i])
            lum = LUM[a] + e.rng.normal(0, .02, 120); e.max_y = max(e.max_y, y)
            if y >= EXIT: e.reached, e.t_exit = True, now; _end(e, 'exit'); continue
            if e.stall.update(now, min(max(y, 0.0), EXIT)): _end(e, 'stalled'); continue
            if e.P['max_coll'] is not None and len(e.crashes) >= e.P['max_coll']: _end(e, 'too_many_collisions'); continue
            v, yaw, vz, dg = e.pilot.step(now, (x, y, z, h), lum)
            go.append(i); cmd.append((v, yaw, vz, dg, lum))
        t2 = time.perf_counter(); T['pilot'] += t2 - t1
        if not go: break
        go = np.array(go, int); pos = np.c_[dyn.x[go], dyn.y[go], dyn.z[go]]
        gap, sd, ty, oi = truth.query(pos); t3 = time.perf_counter(); T['truth'] += t3 - t2
        for a, i in enumerate(go):
            e = E[i]; x, y, z, h = pos[a, 0], pos[a, 1], pos[a, 2], float(dyn.h[i]); v, yaw, vz, dg, lum = cmd[a]
            g = float(gap[a]); zn = w.zone_of(y); zm = e.zmin.setdefault(zn, [np.inf, np.inf]); zm[0] = min(zm[0], float(sd[a])); zm[1] = min(zm[1], g)
            if g < 0 and not e.in_contact:
                kind, oid = truth.label(int(ty[a]), int(oi[a]))
                e.crashes.append(dict(i=e.k, t=round(now, 2), x=round(x, 2), y=round(y, 2), z=round(z, 2), zone=zn, kind=kind, oid=oid,
                                      rehit=oid in e.hit_ids, state=dg.get('state', 'warmup'), phase=dg.get('phase', dg.get('state', 'warmup'))))
                e.hit_ids.add(oid); e.pilot.on_collision(now, (x, y, z, h))
            e.in_contact = g < 0 or (e.in_contact and g < 0.05)
            st = dg.get('state', 'warmup'); r = e.rec
            r['x'].append(round(x, 3)); r['y'].append(round(y, 3)); r['z'].append(round(z, 3)); r['h'].append(round(h, 3))
            r['phase'].append(dg.get('phase', st)); r['state'].append(STATES.index(st) if st in STATES else 0); r['looming'].append(int(bool(dg.get('looming', False))))
            r['gap'].append(round(g, 3)); r['yawc'].append(round(yaw, 3)); r['v'].append(round(v, 3))
            if record:
                r.setdefault('tof', []).append(dg.get('tof'))
                e.img.append(np.clip(lum * 255, 0, 255).astype(np.uint8))
            if e.last is not None: e.dist += float(np.hypot(x - e.last[0], y - e.last[1]))
            e.last = (x, y); e.k += 1
        t4 = time.perf_counter(); T['book'] += t4 - t3
        dyn.step(go, np.array([c[0] for c in cmd]), np.array([c[1] for c in cmd]), np.array([c[2] for c in cmd]))
        hist.append((dyn.x.copy(), dyn.y.copy(), dyn.z.copy(), dyn.h.copy()))
        if len(hist) > 8: hist.pop(0)
        active = go; k += 1; T['dynamics'] += time.perf_counter() - t4
    wall = time.time() - t_wall; frames = max(1, sum(e.k for e in E))
    return [_summary(e, w, wall * e.k / frames, record) for e in E]


def _end(e, reason):
    if e.end_reason is None: e.end_reason = reason
    return True


def _summary(e, w, wall, record):
    meta = w.meta; by_zone, by_kind = {}, {}
    for c in e.crashes: by_zone[c['zone']] = by_zone.get(c['zone'], 0) + 1; by_kind[c['kind']] = by_kind.get(c['kind'], 0) + 1
    gaps = np.array(e.rec['gap']) if e.rec['gap'] else np.array([np.nan]); sim_t = e.k / FPS
    s = dict(sim='fastsim', pilot=e.pilot_kind, mode=run_id(e.config, e.ovr), run_id=run_id(e.config, e.ovr), config=e.config, overrides=e.ovr or {},
             rep=e.rep, world=w.name, preset=meta['preset'], seed=meta['seed'], end_reason=e.end_reason, reached=e.reached,
             success=bool(e.reached and not e.crashes), collisions=len(e.crashes), collisions_by_zone=by_zone, collisions_by_kind=by_kind,
             dist_m=round(e.dist, 1), progress_m=round(min(e.max_y, w.exit_y), 1), time_s=round(e.t_exit if e.reached else sim_t, 1),
             min_clear_m=round(float(min(z_[0] for z_ in e.zmin.values())), 3) if e.zmin else None,
             min_gap_m=round(float(np.nanmin(gaps)), 3), frac_time_gap_lt_0p2=round(float(np.mean(gaps < 0.2)), 3),
             min_clear_by_zone={kk: round(float(vv[0]), 2) for kk, vv in e.zmin.items()}, centreline_hits=meta['stats']['centreline_hits'],
             first_contact_y_m=e.crashes[0]['y'] if e.crashes else None,
             contacts_by_phase={p: sum(c['phase'] == p for c in e.crashes) for p in sorted({c['phase'] for c in e.crashes})},
             first_hits=len(e.hit_ids), rehits=sum(bool(c['rehit']) for c in e.crashes), frames=e.k, wall_s=round(wall, 3),
             rtf=round(sim_t / max(wall, 1e-9), 1), **e.pilot.stats())
    if e.tag is not None: s['tag'] = e.tag
    out = dict(summary=s)
    if record:
        r = dict(e.rec); r['crashes'] = e.crashes; r['img'] = np.array(e.img); out['run'] = r
    return out


def run_episode(config, world, rep=0, duration=360.0, overrides=None, record=False, rows=4, image_lag=1, dyn_params=None, pilot='shared'):
    """One episode (= run_group with a single spec) -> dict(summary=..., run=... if record)."""
    spec = dict(config=config, rep=rep, overrides=overrides, duration=duration)
    return run_group(world, [spec], rows=rows, image_lag=image_lag, dyn_params=dyn_params, pilot=pilot, record=record)[0]


# ----------------------------------------------------------------------------- process pool
CAP_FILE = Path(__file__).resolve().parent / 'max_workers.txt'


def worker_cap():
    """Hard cap on worker processes: env FASTSIM_MAX_WORKERS, else fastsim/max_workers.txt (delete it to lift the cap)."""
    for v in (os.environ.get('FASTSIM_MAX_WORKERS'), CAP_FILE.read_text().strip() if CAP_FILE.exists() else None):
        try:
            if v: return max(1, int(v))
        except ValueError: pass
    return None


def default_workers(reserve_if_px4=3):
    """All cores, minus `reserve_if_px4` while any flybrain-px4* container is using CPU (PX4 batches running), never
    above worker_cap()."""
    n = os.cpu_count() or 4; cap = worker_cap() or n
    try:
        out = subprocess.run(['docker', 'stats', '--no-stream', '--format', '{{.Name}} {{.CPUPerc}}'], capture_output=True, text=True, timeout=8).stdout
        busy = any(l.startswith('flybrain-px4') and float(l.split()[-1].rstrip('%')) > 20 for l in out.splitlines() if l.strip())
    except Exception:
        busy = False
    return min(cap, max(1, n - reserve_if_px4) if busy else n)


def _group_job(a):
    world, specs, kw = a
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
    prof = {}; t = time.time(); res = run_group(world, specs, prof=prof, **kw)
    return [r['summary'] for r in res], dict(prof, wall=time.time() - t, n=len(specs))


def run_batch(jobs, workers=None, out=None, group=None, verbose=True, **kw):
    """jobs: list of (config, world, rep) or dict(config, world, rep, overrides, duration). Episodes of the same world are
    grouped `group` at a time (each group = one process step loop; None = auto: up to 32, but at least ~3 groups per
    worker so the pool stays balanced). -> (summaries, wall s, profile dict)"""
    workers = workers or default_workers(); cap = worker_cap()
    if cap and workers > cap: print(f'worker cap {cap} (fastsim/max_workers.txt / FASTSIM_MAX_WORKERS) < {workers} requested', flush=True); workers = cap
    specs = [j if isinstance(j, dict) else dict(config=j[0], world=j[1], rep=j[2]) for j in jobs]
    by_w = {}
    for s in specs:
        s = dict(s); s.setdefault('overrides', kw.get('overrides')); s.setdefault('duration', kw.get('duration', 360.0)); by_w.setdefault(s.pop('world'), []).append(s)
    kw = {k: v for k, v in kw.items() if k not in ('overrides', 'duration')}
    if group is None:
        group = max(4, min(32, int(np.ceil(len(specs) / (3 * workers)))))
    tasks = []
    for wname, ss in by_w.items():
        ss.sort(key=lambda s: resolve(s['config'], s.get('overrides'))['brain'] == 'none')     # similar lengths together
        for i in range(0, len(ss), group): tasks.append((wname, ss[i:i + group], kw))
    tasks.sort(key=lambda t: -sum(resolve(s['config'], s.get('overrides'))['brain'] != 'none' for s in t[1]))   # long groups first
    res, prof, t0 = [], {}, time.time()
    with Pool(min(workers, len(tasks))) as p:
        for summ, pr in p.imap_unordered(_group_job, tasks):
            for k_, v in pr.items(): prof[k_] = prof.get(k_, 0.0) + v
            res.extend(summ)
            if out:
                with open(out, 'a') as f: f.write(''.join(json.dumps(s) + '\n' for s in summ))
            if verbose:
                s = summ[0]
                print(f"[{len(res)}/{len(specs)} {time.time() - t0:5.0f}s] group of {len(summ)} on {s['world']} done in {pr['wall']:.0f}s; "
                      f"e.g. {s['run_id']} rep{s['rep']}: {s['end_reason']} prog {s['progress_m']} coll {s['collisions']}", flush=True)
    return res, time.time() - t0, prof


def report_speed(res, wall, prof, workers):
    sim = sum(s['time_s'] for s in res); frames = sum(s['frames'] for s in res); stage = {k: prof.get(k, 0) for k in ('render', 'pilot', 'truth', 'dynamics', 'book')}
    tot = sum(stage.values()) or 1
    return dict(episodes=len(res), workers=workers, wall_s=round(wall, 1), episodes_per_min=round(60 * len(res) / wall, 1),
                rtf_total=round(sim / wall, 1), rtf_per_worker=round(sim / max(prof.get('wall', wall * workers), 1e-9), 1),
                us_per_drone_frame={k: round(1e6 * v / max(frames, 1), 1) for k, v in stage.items()},
                share={k: round(v / tot, 3) for k, v in stage.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', required=True); ap.add_argument('--worlds'); ap.add_argument('--presets', default='easy,medium,hard')
    ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--reps', type=int, default=1); ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--group', type=int, default=None, help='episodes stepped together per process (default auto, <= 32)')
    ap.add_argument('--param', action='append', default=[]); ap.add_argument('--duration', type=float, default=360.0)
    ap.add_argument('--out', default=str(OUT / 'runs.jsonl')); ap.add_argument('--rows', type=int, default=4)
    ap.add_argument('--pilot', default='shared', choices=['shared', 'bridge'], help='shared = px4/forest_pilot.Pilot (what the PX4 bridge runs); bridge = frozen copy of the 2026-11-22 bridge logic')
    a = ap.parse_args()
    worlds = a.worlds.split(',') if a.worlds else [f'ff_{p}_{s}' for p in a.presets.split(',') for s in a.seeds.split(',')]
    ovr = {k: parse_value(v) for k, v in (p.split('=', 1) for p in a.param)}
    jobs = [(c, w, r) for c in a.configs.split(',') for w in worlds for r in range(a.reps)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); workers = min(a.workers or default_workers(), worker_cap() or 10 ** 6)
    res, wall, prof = run_batch(jobs, workers, a.out, group=a.group, duration=a.duration, overrides=ovr or None, rows=a.rows, pilot=a.pilot)
    print(json.dumps(report_speed(res, wall, prof, workers)))


if __name__ == '__main__':
    main()
