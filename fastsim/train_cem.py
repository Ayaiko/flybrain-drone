"""Curriculum CEM on the fast sim: tune the shared pilot's parameters (px4/forest_pilot.Pilot via
px4.forest_params.resolve(base, overrides)) on the procedural training pool, raising the difficulty step by step.

Curriculum (the forest-test-field session's plan, 2026-11-22):
  - worlds: the training pool px4/forest/pool.json (levels 0, 0.25, ... 2.0 x seeds 100-109; 0 = easy, 1 = medium,
    2 = hard). The evaluation worlds ff_<easy|medium|hard>_0..5 are held out and never trained on.
  - start at level 0; each generation every candidate flies the same `--worlds` (>= 8) pool worlds of the current level
    plus ~20 % extra episodes on easier levels (anti-forgetting), all with the same noise seeds (common random numbers).
  - promote level += 0.25 when the elite set's success rate (reached the exit with 0 collisions) on the current level
    is >= `--promote` (0.7).
  - every `--eval-every` generations (and at the end) the current mean parameters fly the whole held-out set at all
    levels -> heldout.jsonl.
CEM: candidates are Gaussian in [0, 1]^d (clipped), mapped to SPACE ranges (ints rounded); elite = top `--elite`
fraction; mean / std move toward the elite with smoothing `--alpha`; std floor `--min-std`.
Fitness per episode = progress / exit_y + success - 0.05 x collisions (mean over the candidate's episodes).

  .venv/bin/python -m fastsim.train_cem [--base fix-all3] --pop 24 --gens 60 [--name run1] [--resume] [--workers N] [--dyn drift_sd=0.05]
-> results/fastsim/cem/<name>/: log.jsonl (one line per generation), heldout.jsonl, state.json (resume), best.json
   (mean parameters as overrides: python -m fastsim.runner --configs fix-all --param k=v ... / PX4 forest_eval --param)"""
import argparse, json, time
from pathlib import Path
import numpy as np
from .world import ROOT
from .runner import run_batch, default_workers, worker_cap, report_speed
from px4.forest_params import resolve, load_configs

OUT = ROOT / 'results/fastsim/cem'
POOL = ROOT / 'px4/forest/pool.json'
HELDOUT = [f'ff_{p}_{s}' for p in ('easy', 'medium', 'hard') for s in range(6)]
# name: (low, high, int?) - only keys known to px4/forest_configs.json 'defaults' are used
SPACE = {
    'speed': (0.5, 1.0, False), 'sac_v': (0.0, 0.6, False), 'lplc2_thr': (1.5, 4.0, False), 'front': (20, 50, False),
    'cool': (0.5, 2.0, False), 'goal_gap': (0.0, 3.0, False), 'goal_thr': (5, 30, False), 'rot_gate': (8, 30, False),
    'sac_end_wy': (6, 30, False), 'persist': (1, 5, True), 'veto_r': (2.0, 6.0, False), 'carrot': (3.0, 9.0, False),
    'rec_back_s': (0.5, 2.5, False), 'rec_side_m': (0.5, 1.5, False), 'bypass_m': (0.5, 1.5, False), 'bypass_len': (2.0, 6.0, False),
    'brake_v': (0.05, 0.3, False), 'flow_ref': (0.03, 0.2, False), 'v_min': (0.2, 0.6, False), 'tof_stop': (0.3, 1.5, False),
    'tof_t': (0.0, 2.0, False), 'creep_s': (0.2, 2.0, False), 'creep_v': (0.1, 0.5, False),
}
# the forest session's list for fix-all3 (2026-11-22): pilot parameters + speed; booleans (tof, flow_speed, ...) stay as in the base
DEFAULT_PARAMS = ['creep_s', 'creep_v', 'tof_stop', 'tof_t', 'rec_side_m', 'bypass_m', 'brake_v', 'flow_ref', 'v_min', 'cool', 'goal_thr',
                  'goal_gap', 'lplc2_thr', 'veto_r', 'speed']


def fitness(s, exit_y=80.0):
    return s['progress_m'] / exit_y + float(s['success']) - 0.05 * s['collisions']


def load_pool():
    P = json.loads(POOL.read_text())
    levels = {round(float(k), 2): [w for w in v if (ROOT / f'px4/forest/{w}.json').exists() and w.startswith('ff_L')] for k, v in P.items()}
    return {k: v for k, v in sorted(levels.items()) if v}


class Space:
    def __init__(self, base, names=None):
        known = set(load_configs()['defaults']); self.base = base
        self.names = [n for n in (names or DEFAULT_PARAMS) if n in known and n in SPACE]
        missing = [n for n in (names or DEFAULT_PARAMS) if n not in self.names]
        if missing: print('not tunable (unknown to forest_configs defaults or SPACE):', missing, flush=True)
        self.lo = np.array([SPACE[n][0] for n in self.names], float); self.hi = np.array([SPACE[n][1] for n in self.names], float)
        self.isint = np.array([SPACE[n][2] for n in self.names])
        P0 = resolve(base)                                  # start the mean at the base config's values (clipped into range)
        x0 = [(P0[n] if P0.get(n) is not None else (lo + hi) / 2) for n, lo, hi in zip(self.names, self.lo, self.hi)]
        self.m0 = np.clip((np.array(x0, float) - self.lo) / (self.hi - self.lo), 0, 1)

    def decode(self, u):
        v = self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)
        return {n: (int(round(x)) if i else round(float(x), 4)) for n, x, i in zip(self.names, v, self.isint)}


def evaluate(cands, worlds, rep, a):
    """cands: list of override dicts; worlds: list of names -> (per candidate list of summaries, speed report)."""
    specs = [dict(config=a.base, world=w, rep=rep, overrides=dict(c), tag=i) for i, c in enumerate(cands) for w in worlds]
    dyn = {k: float(v) for k, v in (q.split('=', 1) for q in a.dyn)} or None
    res, wall, prof = run_batch(specs, a.workers, None, verbose=False, duration=a.duration, dyn_params=dyn)
    out = [[] for _ in cands]
    for s in res: out[s['tag']].append(s)
    return out, report_speed(res, wall, prof, a.workers)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default='fix-all3'); ap.add_argument('--name', default=None); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--pop', type=int, default=24); ap.add_argument('--gens', type=int, default=60); ap.add_argument('--elite', type=float, default=0.25)
    ap.add_argument('--alpha', type=float, default=0.7); ap.add_argument('--std0', type=float, default=0.2); ap.add_argument('--min-std', type=float, default=0.03)
    ap.add_argument('--worlds', type=int, default=8, help='current-level pool worlds per generation (>= 8 for promotion)')
    ap.add_argument('--easy-frac', type=float, default=0.2); ap.add_argument('--promote', type=float, default=0.7); ap.add_argument('--step', type=float, default=0.25)
    ap.add_argument('--start-level', type=float, default=0.0); ap.add_argument('--max-level', type=float, default=2.0)
    ap.add_argument('--eval-every', type=int, default=5); ap.add_argument('--eval-reps', type=int, default=2)
    ap.add_argument('--duration', type=float, default=360.0); ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--dyn', action='append', default=[], help='flight-model override for training, e.g. drift_sd=0.06 drift_tau=0.5 (randomisation)')
    ap.add_argument('--params', default=None, help='comma list of parameters to tune (default: DEFAULT_PARAMS; "all" = all of SPACE)'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    a.workers = min(a.workers or default_workers(), worker_cap() or 10 ** 6)
    name = a.name or time.strftime('cem_%Y%m%d_%H%M'); D = OUT / name; D.mkdir(parents=True, exist_ok=True)
    pool = load_pool(); levels = sorted(pool); sp = Space(a.base, list(SPACE) if a.params == 'all' else (a.params.split(',') if a.params else None))
    st_f = D / 'state.json'
    if a.resume and st_f.exists():
        st = json.loads(st_f.read_text()); m, s, level, gen = np.array(st['m']), np.array(st['s']), st['level'], st['gen']
        rng = np.random.default_rng([a.seed, gen])
    else:
        m, s, level, gen = sp.m0.copy(), np.full(len(sp.names), a.std0), a.start_level, 0; rng = np.random.default_rng(a.seed)
        (D / 'config.json').write_text(json.dumps(dict(vars(a), names=sp.names, lo=sp.lo.tolist(), hi=sp.hi.tolist(), pool_levels=levels), indent=1))
    print(f'CEM {name}: base {a.base}, {len(sp.names)} params, pop {a.pop}, workers {a.workers}, level {level}', flush=True)

    def heldout(gen, params):
        ws = [w for w in HELDOUT if (ROOT / f'px4/forest/{w}.json').exists()]
        out = {}
        for r in range(a.eval_reps):
            res, _ = evaluate([params], ws, 1000 + r, a); out.setdefault('runs', []).extend(res[0])
        rows = out['runs']; rep = dict(gen=gen, level=level, params=params, n=len(rows))
        for p in ('easy', 'medium', 'hard'):
            rs = [x for x in rows if f'_{p}_' in x['world']]
            if rs: rep[p] = dict(n=len(rs), success=float(np.mean([x['success'] for x in rs])), reached=float(np.mean([x['reached'] for x in rs])),
                                  progress=float(np.mean([x['progress_m'] for x in rs])), collisions=float(np.mean([x['collisions'] for x in rs])))
        with open(D / 'heldout.jsonl', 'a') as f: f.write(json.dumps(rep) + '\n')
        print('  held-out:', {p: rep[p] for p in ('easy', 'medium', 'hard') if p in rep}, flush=True)

    if gen == 0 and not (D / 'heldout.jsonl').exists(): heldout(0, {})                     # baseline: base config as is
    while gen < a.gens:
        t0 = time.time(); lv = min(levels, key=lambda L: abs(L - level))
        cur = list(rng.choice(pool[lv], size=min(a.worlds, len(pool[lv])), replace=False))
        easier = [L for L in levels if L < lv - 1e-9]; n_easy = int(round(a.easy_frac / (1 - a.easy_frac) * len(cur))) if easier else 0
        old = [str(rng.choice(pool[float(rng.choice(easier))])) for _ in range(n_easy)]
        U = np.clip(m + s * rng.normal(size=(a.pop, len(m))), 0, 1); U[0] = m                 # candidate 0 = current mean
        cands = [sp.decode(u) for u in U]
        res, speed = evaluate(cands, cur + old, gen, a)
        F = np.array([np.mean([fitness(x) for x in r]) for r in res])
        succ_cur = np.array([np.mean([x['success'] for x in r if x['world'] in cur]) for r in res])
        k = max(2, int(round(a.elite * a.pop))); el = np.argsort(-F)[:k]
        m = a.alpha * U[el].mean(0) + (1 - a.alpha) * m; s = np.maximum(a.alpha * U[el].std(0) + (1 - a.alpha) * s, a.min_std)
        elite_succ = float(succ_cur[el].mean()); promoted = elite_succ >= a.promote and len(cur) >= 8 and level < a.max_level
        rec = dict(gen=gen, level=lv, worlds=cur, easier=old, fit_best=float(F.max()), fit_mean=float(F.mean()), fit_mean_cand=float(F[0]),
                   elite_success=elite_succ, best_success=float(succ_cur[el[0]]), promoted=bool(promoted), mean=sp.decode(m),
                   best=cands[int(el[0])], std=[round(float(v), 3) for v in s], wall_s=round(time.time() - t0, 1), speed=speed)
        with open(D / 'log.jsonl', 'a') as f: f.write(json.dumps(rec) + '\n')
        print(f'gen {gen:3d} level {lv:.2f}: fitness best {F.max():.3f} mean {F.mean():.3f} | elite success {elite_succ:.2f}'
              f'{" -> PROMOTE" if promoted else ""} | {rec["wall_s"]:.0f}s, {speed["episodes_per_min"]} ep/min', flush=True)
        if promoted: level = round(min(a.max_level, level + a.step), 2)
        gen += 1
        st_f.write_text(json.dumps(dict(m=m.tolist(), s=s.tolist(), level=level, gen=gen, names=sp.names)))
        (D / 'best.json').write_text(json.dumps(dict(base=a.base, overrides=sp.decode(m), gen=gen, level=level), indent=1))
        if gen % a.eval_every == 0 or gen == a.gens: heldout(gen, sp.decode(m))


if __name__ == '__main__':
    main()
