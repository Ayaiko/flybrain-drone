"""Pre-generate a pool of forest worlds over a range of continuous difficulty levels, for curriculum training
(generation of hard layouts can take ~1 min each because of the passability re-draws, so do it once, in parallel).

  python -m px4.make_pool --levels 0:2:0.25 --seeds 100:110 --procs 3
  -> px4/forest/ff_L<level*100>_<seed>.json + meshes/SDF, and px4/forest/pool.json listing them by level.
Training seeds should stay disjoint from the evaluation worlds (ff_<preset>_0..14)."""
import argparse, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from px4.make_forest import make, OUT


def rng_arg(s):
    a, b, *st = [float(v) for v in s.split(':')]; step = st[0] if st else 1.0
    return [round(v, 4) for v in np.arange(a, b + 1e-9, step)]


def job(args):
    level, seed = args; name = f'ff_L{int(round(level * 100)):03d}_{int(seed)}'
    if not (OUT / f'{name}.json').exists():
        try: make(seed=int(seed), level=level, assets=False)     # JSON only (fast sim); run make_forest for a Gazebo world
        except RuntimeError as e: return level, seed, None, str(e)
    return level, seed, name, None


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--levels', default='0:2:0.25'); ap.add_argument('--seeds', default='100:109')
    ap.add_argument('--procs', type=int, default=3); a = ap.parse_args()
    jobs = [(l, s) for l in rng_arg(a.levels) for s in rng_arg(a.seeds)]
    with Pool(a.procs) as p: res = p.map(job, jobs)
    pool_f = OUT / 'pool.json'; pool = json.loads(pool_f.read_text()) if pool_f.exists() else {}
    for level, seed, name, err in res:
        if name: pool.setdefault(f'{level:.2f}', []); pool[f'{level:.2f}'] = sorted(set(pool[f'{level:.2f}']) | {name})
        else: print('failed', level, seed, err)
    pool_f.write_text(json.dumps(pool, indent=1)); print({k: len(v) for k, v in sorted(pool.items())})
