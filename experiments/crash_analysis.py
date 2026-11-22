"""Why does the fly controller crash? Fly 100 unseen worlds in bounce mode with full step logging, then put each
crash in one category (first match wins):
  recrash     <1 s after the previous crash (artefact of the bounce reset)
  saccade     hit while mid-turn (vision is ignored during a saccade)
  settling    hit in the 0.15 s blind window right after a saccade (filters reset, no avoidance)
  side        hit from the side/rear (|bearing| > 45 deg) while flying straight
  missed      head-on, looming never fired in the last 1 s
  late        head-on, looming fired in the last 1 s but the escape did not clear the obstacle
Also checks whether the looming threshold drifts upward over an episode.
Run from the project root: python -m experiments.crash_analysis   -> results/crash_analysis/"""
import json
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from flydrone.sim import DT, OVERSAMPLE, make_eye
from flydrone.world import World
from flydrone.controllers import FlyVision
from experiments.h6_finetune import fly_kw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/crash_analysis'; OUT.mkdir(parents=True, exist_ok=True)
TEST = list(range(100)); T = 60.0
CATS = ['recrash', 'saccade', 'settling', 'side', 'missed', 'late']


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def contact(world, p):
    """Bearing (world angle) from p to the surface it touches, and what it is."""
    L = world.size; cands = [(p[0], np.pi, 'wall'), (L - p[0], 0.0, 'wall'), (p[1], -np.pi / 2, 'wall'), (L - p[1], np.pi / 2, 'wall')]
    if len(world.r):
        d = np.linalg.norm(world.c - p, axis=1) - world.r; j = d.argmin()
        cands.append((d[j], np.arctan2(*(world.c[j] - p)[::-1]), 'post'))
    g, ang, kind = min(cands, key=lambda c: c[0]); return ang, kind


def run(args):
    name, kw, seed = args
    world = World(seed=seed); eye = make_eye(120); step_a = eye[1] - eye[0]
    fine = (eye[:, None] + step_a * (np.arange(OVERSAMPLE) / OVERSAMPLE - .5 + .5 / OVERSAMPLE)).ravel()
    ctrl = FlyVision(eye, DT, seed=seed, **kw); rng = np.random.default_rng(seed + 999)
    p = world.start.copy(); h = np.pi / 2 + rng.normal(0, .2); t = 0.0
    log, crashes, last_crash, refs = [], [], -9.0, []
    while t < T:
        dist_f, lum_f = world.cast(p, h + fine)
        lum = lum_f.reshape(-1, OVERSAMPLE).mean(1) + rng.normal(0, .02, len(eye)); dist = dist_f.reshape(-1, OVERSAMPLE).min(1)
        yaw, v = ctrl.act(lum, dist); dg = dict(getattr(ctrl, 'diag', {}))
        front = float(dist[np.abs(eye) < np.radians(20)].min())
        log.append(dict(t=t, x=float(p[0]), y=float(p[1]), h=float(h), state=dg.get('state', 'warmup'),
                        looming=bool(dg.get('looming', False)), turn=dg.get('turn', 0), front=front))
        if dg.get('state') in ('fixate', 'settling'): refs.append((t, float(dg.get('loom_ref', np.nan))))
        h += np.clip(yaw, -12, 12) * DT; prev = p; p = p + v * DT * np.array([np.cos(h), np.sin(h)]); t += DT
        if world.collides(p, .25):
            ang, kind = contact(world, p); bearing = float(np.degrees(wrap(ang - h)))
            recent = [e for e in log if e['t'] > t - 1.0]
            sac = [e for e in log if e['state'] == 'saccade_start']
            last_sac = sac[-1] if sac else None
            state = log[-1]['state']
            if t - last_crash < 1.0: cat = 'recrash'
            elif state in ('saccade', 'saccade_start'): cat = 'saccade'
            elif state == 'settling': cat = 'settling'
            elif abs(bearing) > 45: cat = 'side'
            elif not any(e['looming'] for e in recent): cat = 'missed'
            else: cat = 'late'
            toward = None
            if last_sac is not None and t - last_sac['t'] < 1.5:
                toward = bool(np.sign(last_sac['turn']) == np.sign(bearing))
            crashes.append(dict(ctrl=name, seed=seed, t=t, x=float(p[0]), y=float(p[1]), bearing=bearing, kind=kind, cat=cat, state=state,
                                since_saccade=(t - last_sac['t']) if last_sac else None, last_turn_toward=toward,
                                front_1s_before=float(np.median([e['front'] for e in recent[:10]])) if recent else None))
            last_crash = t; p = prev; h += np.pi + rng.uniform(-.5, .5)
    return dict(name=name, seed=seed, crashes=crashes, refs=refs, path=[(e['x'], e['y'], e['state']) for e in log])


def main():
    crash = json.load(open(ROOT / 'results/crash/crash_results.json'))
    best = int(np.argmin(crash['per_run_crashes_per_min']['fly_tuned_crash_penalty']))
    ctrls = {'fly_innate_saccade': {}, 'fly_tuned_crash_penalty': fly_kw(np.array(crash['params'][best]['fly_u'])),
             'fly_lplc2': dict(loom_mode='lplc2', lplc2_thr=2.0, front=35)}
    with Pool() as pool:
        res = pool.map(run, [(n, kw, s) for n, kw in ctrls.items() for s in TEST])
    report = {}
    for n in ctrls:
        cs = [c for r in res if r['name'] == n for c in r['crashes']]
        counts = {c: sum(x['cat'] == c for x in cs) for c in CATS}
        drift = []
        for r in res:
            if r['name'] == n and len(r['refs']) > 50:
                a = np.array(r['refs']); early = a[a[:, 0] < 5, 1]; late = a[a[:, 0] > 50, 1]
                if len(early) and len(late): drift.append(np.median(late) / np.median(early))
        toward = [c['last_turn_toward'] for c in cs if c['last_turn_toward'] is not None]
        report[n] = dict(n_crashes=len(cs), per_min=len(cs) / (len(TEST) * T / 60), categories=counts,
                         share={c: round(v / max(1, len(cs)), 3) for c, v in counts.items()},
                         walls=sum(c['kind'] == 'wall' for c in cs),
                         turned_toward_obstacle=float(np.mean(toward)) if toward else None,
                         loom_threshold_drift_x=float(np.median(drift)) if drift else None,
                         median_since_saccade=float(np.median([c['since_saccade'] for c in cs if c['since_saccade'] is not None])))
        print(n, json.dumps(report[n], indent=1))
    json.dump(dict(report=report, crashes=[c for r in res for c in r['crashes']], tuned_run=best), open(OUT / 'crash_analysis.json', 'w'), indent=1)
    # keep a few paths for snapshots
    keep = [r for r in res if r['name'] == 'fly_tuned_crash_penalty' and r['crashes']][:6]
    json.dump([dict(seed=r['seed'], path=r['path'], crashes=r['crashes']) for r in keep], open(OUT / 'example_paths.json', 'w'))


if __name__ == '__main__':
    main()
