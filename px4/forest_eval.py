"""Forest test field batch (host side): presets x seeds x configs, each flight on a fresh sim inside a flybrain-px4
container (px4/run_forest.sh). With --workers N the jobs are spread over N containers running in parallel (the original
flybrain-px4 plus flybrain-px4-w1.., created on first use from the same image and the same project folder).
Every finished run is appended to results/forest/forest_runs.jsonl right away (with its full parameter set).

Run from the project root:
  python -m px4.forest_eval --configs lplc2-bold-gate-sv0-v0.8 --presets medium --seeds 0,1,2 --workers 3
  python -m px4.forest_eval --configs lplc2-bold-gate-sv0-v0.8 --param goal_gap=0 --param cool=0.8     # one-off variant
  python -m px4.forest_eval ... --speed 2      # faster than real time (check consistency first, see EXPERIMENTS.md)
Configs: px4/forest_configs.json (or old-style mode strings). Already logged (preset, seed, run id) are skipped unless --redo.
Compare afterwards: python -m px4.forest_compare <run id> <run id> ..."""
import argparse, json, queue, subprocess, threading, time
from pathlib import Path
import numpy as np
from px4.make_forest import make, plot_map
from px4.forest_geom import Obstacles
from px4.forest_params import resolve, run_id, parse_value

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'results/forest'; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / 'forest_runs.jsonl'; MAIN = 'flybrain-px4'; WORKER_IMAGE = 'flybrain-px4:worker'
PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf']
_lock = threading.Lock()


def load():
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()] if LOG.exists() else []


def docker(*a, check=False): return subprocess.run(['docker', *a], capture_output=True, text=True, check=check)


def ensure_workers(n):
    names = [MAIN] + [f'{MAIN}-w{i}' for i in range(1, n)]
    if n > 1 and docker('image', 'inspect', WORKER_IMAGE).returncode != 0:   # workers = exact copy of the main container's software
        docker('commit', MAIN, WORKER_IMAGE, check=True); print('created image', WORKER_IMAGE, flush=True)
    for i, name in enumerate(names):
        st = docker('inspect', '-f', '{{.State.Running}}', name)
        if st.returncode != 0:            # create: same image, same bind mount, own Gazebo partition
            docker('run', '-d', '--init', '--name', name, '--hostname', name, '-e', f'FLYBRAIN_WORKER=w{i}', '--shm-size', '1g', '--user', 'px4',
                   '-v', f'{ROOT}:/work', '-w', '/work/px4/PX4-Autopilot', WORKER_IMAGE, 'sleep', 'infinity', check=True)
            print('created worker container', name, flush=True)
        elif st.stdout.strip() != 'true':
            docker('start', name, check=True)
    return names


def run_job(container, job, T, speed):
    preset, seed, cfg, ovr, rid = job; world = f'ff_{preset}_{seed}'
    out = ROOT / f'px4/runs/forest/{rid}_{world}.json'
    if out.exists(): out.unlink()
    for attempt in range(2):
        r = docker('exec', '-e', f'SPEED={speed}', '-e', f'FLY_OVERRIDES={json.dumps(ovr)}', '-e', f'RUN_ID={rid}', container,
                   'bash', '/work/px4/run_forest.sh', cfg, world, str(T))
        if out.exists(): break
        with _lock: print(f'{world} {rid} attempt {attempt + 1} failed on {container}: {r.stdout.strip()[-300:]}', flush=True)
    if not out.exists(): return None
    s = json.loads(out.read_text())['summary']; s['container'] = container
    return s


def paths_png(world, rids):
    meta = json.loads((ROOT / f'px4/forest/{world}.json').read_text()); paths = {}
    for k, rid in enumerate(rids):
        f = ROOT / f'px4/runs/forest/{rid}_{world}.json'
        if not f.exists(): continue
        d = json.loads(f.read_text()); s = d['summary']; run = d['run']
        lab = f"{rid[:38]}: {s['collisions']} coll, {'exit' if s['reached'] else 'y=%.0f %s' % (s['progress_m'], s.get('end_reason') or '')}"
        paths[lab] = (run['x'], run['y'], PALETTE[k % len(PALETTE)], [[c['x'], c['y']] for c in run['crashes']])
    plot_map(meta, Obstacles(meta), OUT / f'paths_{world}.png', paths)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--presets', default='easy,medium,hard'); ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--configs', '--modes', dest='configs', default='none-v0.8,lplc2-bold-gate-sv0-v0.8')
    ap.add_argument('--param', action='append', default=[], help='override for every config, e.g. goal_gap=0 (repeatable)')
    ap.add_argument('--T', type=int, default=360, help='time limit per flight, sim seconds')
    ap.add_argument('--workers', type=int, default=1); ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--redo', action='store_true'); ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--tag', default='', help='suffix for the run ids, e.g. @T600 when the time limit differs')
    a = ap.parse_args(); t0 = time.time()
    ovr = {k: parse_value(v) for k, v in (p.split('=', 1) for p in a.param)}
    cfgs = a.configs.split(','); [resolve(c, ovr) for c in cfgs]            # fail fast on typos
    speed = int(a.speed) if a.speed == int(a.speed) else a.speed
    rids = {c: run_id(c, ovr) + (f'@x{speed}' if speed != 1 else '') + a.tag for c in cfgs}
    done = {(r['preset'], r['seed'], r.get('run_id', r['mode'])) for r in load()} if not a.redo else set()
    jobs, worlds = [], []
    for preset in a.presets.split(','):
        for seed in map(int, a.seeds.split(',')):
            world = f'ff_{preset}_{seed}'; worlds.append(world)
            if not (ROOT / f'px4/forest/{world}.json').exists(): make(preset, seed)
            jobs += [(preset, seed, c, ovr, rids[c]) for c in cfgs if (preset, seed, rids[c]) not in done]
    print(f'{len(jobs)} flights on {a.workers} worker(s), speed x{speed}', flush=True)
    q = queue.Queue(); [q.put(j) for j in jobs]; names = ensure_workers(a.workers) if jobs else []

    def worker(container):
        while True:
            try: job = q.get_nowait()
            except queue.Empty: return
            s = run_job(container, job, a.T, speed)
            with _lock:
                if s is None: print(f'ff_{job[0]}_{job[1]} {job[4]} FAILED', flush=True); continue
                s['wall_s'] = round(time.time() - t0)
                with open(LOG, 'a') as f: f.write(json.dumps(s) + '\n')
                print(f"{s['world']:12s} {s['run_id'][:40]:40s} {'SUCCESS' if s['success'] else 'reached' if s['reached'] else 'FAIL   '} "
                      f"coll {s['collisions']:2d} prog {s['progress_m']:5.1f} m t {s['time_s']:5.1f} s end {s['end_reason']:<20s} "
                      f"zones {s['collisions_by_zone']}  [{container[-2:]} {time.time() - t0:.0f}s]", flush=True)
    th = [threading.Thread(target=worker, args=(n,)) for n in names]; [t.start() for t in th]; [t.join() for t in th]
    if not a.no_plots:
        for w in worlds: paths_png(w, list(rids.values()))
    print(f'wall {time.time() - t0:.0f}s   compare: python -m px4.forest_compare ' + ' '.join(rids.values()), flush=True)


if __name__ == '__main__':
    main()
