"""PX4 coverage: LPLC2 'bold' settings vs the earlier LPLC2 coverage runs on worlds 0-3 (180 s each).
Writes results/px4/bold_results.json and viewer/episodes_px4_bold.json. Run from the project root: python -m px4.bold_eval"""
import json, subprocess, time
from pathlib import Path
import numpy as np
from px4.make_gz import world, model
from flydrone.world import World

ROOT = Path(__file__).resolve().parents[1]; SEEDS = [0, 1, 2, 3]; T = 180
OLD, NEW = 'lplc2-cov-v1.0', 'lplc2-cov-bold-v1.0'
if __name__ == '__main__':
    t0 = time.time(); model(); res = {OLD: [], NEW: []}
    viewer = dict(dt=1 / 30, n_eye=120, fov=330, drone_r=.35, cells=dict(margin=2.0, cell=2.0), states=['warmup', 'settling', 'fixate', 'saccade', 'saccade_start'], worlds=[])
    for s in SEEDS:
        world(s); out = ROOT / f'px4/runs/{NEW}_{s}.json'
        if out.exists(): out.unlink()
        for attempt in range(2):
            subprocess.run(['docker', 'exec', 'flybrain-px4', 'bash', '/work/px4/run.sh', NEW, str(s), str(T)], capture_output=True, timeout=T * 4 + 300)
            if out.exists(): break
        w = World(seed=s); ent = dict(seed=s, size=w.size, runs={}, obstacles=[[round(float(a), 3), round(float(b), 3), round(float(r), 3), int(k), round(float(ph), 3)]
                                                                                for (a, b), r, k, ph in zip(w.c, w.r, w.k, w.ph)])
        for m, lab in ((OLD, 'LPLC2 coverage (before)'), (NEW, 'LPLC2 coverage BOLD (after)')):
            f = ROOT / f'px4/runs/{m}_{s}.json'
            if not f.exists(): print(s, m, 'missing'); continue
            d = json.loads(f.read_text()); res[m].append(d['summary']); ent['runs'][lab] = d['run']
            print(s, m, {k: d['summary'][k] for k in ('coverage', 'post_crashes', 'post_per_min', 'hazards', 'dist_m')}, f'{time.time() - t0:.0f}s', flush=True)
        viewer['worlds'].append(ent); json.dump(viewer, open(ROOT / 'viewer/episodes_px4_bold.json', 'w'), separators=(',', ':'))
    S = {m: dict(n=len(rs), coverage=float(np.mean([r['coverage'] for r in rs])), post_per_min=float(np.mean([r['post_per_min'] for r in rs])),
                 hazards=float(np.mean([r['hazards'] for r in rs])), dist_m=float(np.mean([r['dist_m'] for r in rs]))) for m, rs in res.items() if rs}
    for m, v in S.items(): print(f'{m:22s} coverage {v["coverage"]:.1f}%  post crashes/min {v["post_per_min"]:.2f}  hazards {v["hazards"]:.0f}  dist {v["dist_m"]:.0f} m')
    json.dump(S, open(ROOT / 'results/px4/bold_results.json', 'w'), indent=1)
