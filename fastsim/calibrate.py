"""Calibration gate: does the fast sim rank worlds and configs like PX4 + Gazebo does?

For every PX4 run id given (--configs; e.g. none-v0.8, fix-all, fix-all2@T600 = config fix-all2 with a 600 s limit,
name+k=v = overrides) and every held-out world ff_{easy,medium,hard}_{0..5} that has a PX4 log in px4/runs/forest/,
runs N fast-sim reps (different sensor-noise seeds) with the same pilot (default: the shared px4/forest_pilot.Pilot)
and compares per (run id, world) mean progress and collisions with the PX4 run, both under the same early-stop rule
(stall 90 s / 3 m, 20 collisions, the run's time limit). PX4 runs with < 1 m progress (failed takeoff) are excluded.
Gate: Spearman rank correlation >= 0.8, read against the ceiling a single PX4 run allows.

  .venv/bin/python -m fastsim.calibrate [--configs a,b,...] [--reps 5] [--workers N] [--tag _x] [--reuse]
-> results/fastsim/calibration.md, calibration.json, calibration_runs.jsonl, calibration_scatter.png"""
import argparse, json, os, platform, time
from pathlib import Path
import numpy as np
from .world import ROOT
from .runner import run_batch, report_speed, default_workers
from .eye import load_calib
from .dynamics import load_params

OUT = ROOT / 'results/fastsim'; RUNS = ROOT / 'px4/runs/forest'
TAG = ''
CONFIGS = ['none-v0.8', 'lplc2-bold-gate-sv0-v0.8', 'lplc2-bold-gate-sv0-gg0-v0.8']
WORLDS = [f'ff_{p}_{s}' for p in ('easy', 'medium', 'hard') for s in range(6)]      # held-out evaluation worlds
RULE = dict(stall_s=90, stall_m=3.0, max_coll=20, limit_s=360)


# ---- PX4 scoring (same logic as px4/forest_params.debounce / rescore of 2026-11-22, copied so this gate does not move
#      when those files change)
def debounce(run, hyst=0.05):
    g = run['gap']; keep, last = [], None
    for i in sorted(c['i'] for c in run['crashes']):
        if last is None or (i > last and max(g[last:i]) >= hyst): keep.append(i)
        last = i
    return keep


def parse_rid(rid):
    """PX4 run id -> (config, overrides or None, time limit s): 'fix-all2@T600', 'lplc2-...+goal_gap=0'."""
    from px4.forest_params import parse_value
    lim = RULE['limit_s']
    if '@T' in rid: rid, t = rid.rsplit('@T', 1); lim = float(t)
    parts = rid.split('+'); ovr = {k: parse_value(v) for k, v in (q.split('=', 1) for q in parts[1:])}
    return parts[0], ovr or None, lim


def rescore(run, dt, exit_y, reached, logged_end=None, limit_s=None):
    y = run['y']; crash_i = debounce(run); best_t, best_p = 0.0, None; n = len(y); stop, reason = n, None
    limit_s = limit_s or RULE['limit_s']
    for i in range(n):
        t = i * dt
        if t > limit_s: stop, reason = i, 'timeout'; break
        if sum(ci <= i for ci in crash_i) >= RULE['max_coll']: stop, reason = i + 1, 'too_many_collisions'; break
        prog = min(max(y[i], 0.0), exit_y)
        if best_p is None or prog >= best_p + RULE['stall_m']: best_p, best_t = prog, t
        if t - best_t > RULE['stall_s']: stop, reason = i + 1, 'stalled'; break
    reached = reason is None and reached
    return dict(collisions=sum(ci < stop for ci in crash_i), progress_m=round(exit_y if reached else min(max(y[:stop] or [0]), exit_y), 1),
                time_s=round(stop * dt, 1), end_reason=reason or ('exit' if reached else (logged_end or 'timeout')), reached=bool(reached))


def px4_scores():
    S = {}
    for c in CONFIGS:
        for w in WORLDS:
            f = RUNS / f'{c}_{w}.json'
            if not f.exists(): continue
            d = json.loads(f.read_text()); exit_y = json.loads((ROOT / f'px4/forest/{w}.json').read_text())['exit_y']
            S[(c, w)] = rescore(d['run'], d['dt'], exit_y, d['summary']['reached'], d['summary'].get('end_reason'), parse_rid(c)[2])
    return S


def rank(a):
    a = np.asarray(a, float); o = np.argsort(a, kind='mergesort'); r = np.empty(len(a)); r[o] = np.arange(len(a))
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1: r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0: return float('nan')
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main():
    global CONFIGS, TAG
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, default=5); ap.add_argument('--workers', type=int, default=None); ap.add_argument('--group', type=int, default=None)
    ap.add_argument('--reuse', action='store_true', help='score the existing calibration_runs.jsonl instead of running')
    ap.add_argument('--pilot', default='shared', choices=['shared', 'bridge'])
    ap.add_argument('--configs', default=','.join(CONFIGS), help='configs to compare (PX4 runs must exist in px4/runs/forest)')
    ap.add_argument('--dyn', action='append', default=[], help='flight-model override, e.g. drift_sd=0.06')
    ap.add_argument('--tag', default='', help='suffix for the output files, e.g. _batch3')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); runs_f = OUT / f'calibration_runs{a.tag}.jsonl'
    CONFIGS = a.configs.split(','); TAG = a.tag
    px = px4_scores(); bad = {k: v for k, v in px.items() if v['progress_m'] < 1.0}       # failed takeoff / sim stall
    for k in bad: px.pop(k)
    pairs = sorted(px)
    if a.reuse:
        res = [json.loads(l) for l in runs_f.read_text().splitlines() if l.strip()]; wall = None; prof = {}
    else:
        if runs_f.exists(): runs_f.unlink()
        jobs = []
        for r in range(a.reps):
            for (c, w) in pairs:
                cfg, ovr, lim = parse_rid(c); jobs.append(dict(config=cfg, overrides=ovr, duration=lim, world=w, rep=r, tag=c))
        a.workers = a.workers or default_workers()
        dyn = {k: float(v) for k, v in (q.split('=', 1) for q in a.dyn)} or None
        res, wall, prof = run_batch(jobs, a.workers, runs_f, group=a.group, pilot=a.pilot, dyn_params=dyn)
    for s_ in res: s_.setdefault('tag', s_['config'])
    rows = []; brain = None
    for c in CONFIGS:
        for w in WORLDS:
            fs = [s for s in res if s['tag'] == c and s['world'] == w]
            if not fs or (c, w) not in px: continue
            P = px[(c, w)]
            rows.append(dict(config=c, world=w, n=len(fs), px4_prog=P['progress_m'], px4_coll=P['collisions'], px4_end=P['end_reason'],
                             fs_prog=float(np.mean([s['progress_m'] for s in fs])), fs_prog_sd=float(np.std([s['progress_m'] for s in fs])),
                             fs_coll=float(np.mean([s['collisions'] for s in fs])), fs_coll_sd=float(np.std([s['collisions'] for s in fs])),
                             fs_reached=float(np.mean([s['reached'] for s in fs])), fs_ends={e: sum(s['end_reason'] == e for s in fs) for e in set(s['end_reason'] for s in fs)}))
    col = lambda k, rs=rows: [r[k] for r in rs]
    corr = dict(all=dict(progress=spearman(col('px4_prog'), col('fs_prog')), collisions=spearman(col('px4_coll'), col('fs_coll'))))
    for c in CONFIGS:
        rs = [r for r in rows if r['config'] == c]
        corr[c] = dict(progress=spearman(col('px4_prog', rs), col('fs_prog', rs)), collisions=spearman(col('px4_coll', rs), col('fs_coll', rs)))
    brain = [r for r in rows if r['config'] != 'none-v0.8']
    corr['brain'] = dict(progress=spearman(col('px4_prog', brain), col('fs_prog', brain)), collisions=spearman(col('px4_coll', brain), col('fs_coll', brain)))
    # how well can a single PX4 run agree at best? PX4 has one run per (config, world), the fast sim shows how much one
    # run scatters: corr(one fast-sim rep, mean of the other reps), averaged over which rep is held out
    ceil = {}
    reps = sorted(set(s['rep'] for s in res))
    subsets = dict(all=CONFIGS, brain=[c for c in CONFIGS if c != 'none-v0.8'])
    if len(reps) >= 3:
        for name, cfgs in subsets.items():
            ceil[name] = {}
            for k in ('progress_m', 'collisions'):
                vals = []
                for r0 in reps:
                    one, rest = [], []
                    for c in cfgs:
                        for w in WORLDS:
                            a1 = [s[k] for s in res if s['tag'] == c and s['world'] == w and s['rep'] == r0]
                            a2 = [s[k] for s in res if s['tag'] == c and s['world'] == w and s['rep'] != r0]
                            if a1 and a2: one.append(a1[0]); rest.append(np.mean(a2))
                    vals.append(spearman(one, rest))
                ceil[name][k] = float(np.nanmean(vals))
    # where does the PX4 run fall inside the fast-sim distribution of the same (config, world)? calibrated -> ~uniform,
    # mean 0.5; < 0.5 = PX4 lower than the fast sim
    pit = {}
    for k, pk in (('progress_m', 'px4_prog'), ('collisions', 'px4_coll')):
        u = []
        for r in brain:
            fs = np.array([s[k] for s in res if s['tag'] == r['config'] and s['world'] == r['world']])
            u.append(float((np.sum(fs < r[pk]) + 0.5 * np.sum(fs == r[pk])) / len(fs)))
        pit[k] = dict(mean=float(np.mean(u)), frac_outside_range=float(np.mean([(x == 0) or (x == 1) for x in u])), values=u)
    by_preset = {}
    for c in CONFIGS:
        for p in ('easy', 'medium', 'hard'):
            rs = [r for r in rows if r['config'] == c and f'_{p}_' in r['world']]
            if not rs: continue
            by_preset[f'{c} {p}'] = dict(px4_prog=float(np.mean(col('px4_prog', rs))), fs_prog=float(np.mean(col('fs_prog', rs))),
                                         px4_coll=float(np.mean(col('px4_coll', rs))), fs_coll=float(np.mean(col('fs_coll', rs))))
    sim_s = sum(s['time_s'] for s in res); wall_ep = sum(s['wall_s'] for s in res)
    speed = dict(episodes=len(res), sim_s_total=round(sim_s), worker_wall_s_total=round(wall_ep, 1),
                 rtf_per_core=round(sim_s / wall_ep, 1), batch_wall_s=None if wall is None else round(wall, 1),
                 episodes_per_min=None if wall is None else round(60 * len(res) / wall, 1),
                 rtf_total=None if wall is None else round(sim_s / wall, 1), workers=a.workers, cpu=platform.processor() or platform.machine(),
                 batched=None if wall is None else report_speed(res, wall, prof, a.workers),
                 px4_reference='PX4 + Gazebo Docker: ~1x real time per container, 3 containers ~1.8x total (forest-test-field notes)')
    out = dict(configs=CONFIGS, px4_pairs=len(pairs), px4_excluded={f'{c} {w}': v['end_reason'] for (c, w), v in bad.items()}, rule=RULE, reps=len(reps), pilot=a.pilot, corr=corr, split_half_ceiling=ceil, px4_percentile_in_fastsim=pit, speed=speed, by_preset=by_preset, rows=rows,
               eye_calib=load_calib().get('source'), dyn_params=load_params())
    (OUT / f'calibration{TAG}.json').write_text(json.dumps(out, indent=1))
    plot(rows, corr); write_md(out)
    print(json.dumps(dict(corr=corr, ceiling=ceil, pit={k: {a: b for a, b in v.items() if a != 'values'} for k, v in pit.items()}, speed=speed), indent=1))


def plot(rows, corr):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6)); mk = {'easy': 'o', 'medium': 's', 'hard': '^'}
    colr = dict(zip(CONFIGS, ['#7f7f7f', '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']))
    for r in rows:
        p = r['world'].split('_')[1]
        ax[0].errorbar(r['px4_prog'], r['fs_prog'], yerr=r['fs_prog_sd'], fmt=mk[p], color=colr[r['config']], ms=6, alpha=.8, capsize=2)
        ax[1].errorbar(r['px4_coll'], r['fs_coll'], yerr=r['fs_coll_sd'], fmt=mk[p], color=colr[r['config']], ms=6, alpha=.8, capsize=2)
    for a_, k, lab in ((ax[0], 'progress', 'progress m'), (ax[1], 'collisions', 'collisions')):
        lim = a_.get_xlim()[1] if k == 'collisions' else 85
        a_.plot([0, lim], [0, lim], 'k:', lw=.8); a_.set_xlabel(f'PX4 + Gazebo {lab}'); a_.set_ylabel(f'fastsim {lab} (mean +- sd over reps)')
        a_.set_title(f"{lab}: Spearman all = {corr['all'][k]:.2f}, brain configs = {corr['brain'][k]:.2f}", fontsize=9)
    for c, cl in colr.items(): ax[0].plot([], [], 'o', color=cl, label=c)
    for p, m in mk.items(): ax[0].plot([], [], m, color='k', label=p)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / f'calibration_scatter{TAG}.png', dpi=100); plt.close(fig)


def write_md(o):
    """Tables only -> results/fastsim/calibration_tables.md (calibration.md carries the discussion and links here)."""
    L = [f"Rule: {o['rule']}; {o['reps']} fast-sim reps per (config, world); pilot = {o['pilot']}", '',
         '| Spearman | progress | collisions |', '|---|---|---|']
    for k, v in o['corr'].items(): L.append(f"| {k} | {v['progress']:.2f} | {v['collisions']:.2f} |")
    for k, v in o['split_half_ceiling'].items():
        L.append(f"| ceiling {k}: one fast-sim rep vs mean of the others | {v['progress_m']:.2f} | {v['collisions']:.2f} |")
    pt = o['px4_percentile_in_fastsim']
    L += ['', f"| PX4 run as a percentile of the fast-sim reps ({len(pt['progress_m']['values'])} brain pairs) | progress | collisions |", '|---|---|---|',
          f"| mean (0.5 = centred) | {pt['progress_m']['mean']:.2f} | {pt['collisions']['mean']:.2f} |",
          f"| fraction outside the fast-sim range | {pt['progress_m']['frac_outside_range']:.2f} | {pt['collisions']['frac_outside_range']:.2f} |"]
    L += ['', '| config | world | PX4 progress | fastsim progress (sd) | PX4 coll | fastsim coll (sd) | PX4 end | fastsim ends |', '|---|---|---|---|---|---|---|---|']
    for r in o['rows']:
        L.append(f"| {r['config']} | {r['world']} | {r['px4_prog']:.1f} | {r['fs_prog']:.1f} ({r['fs_prog_sd']:.1f}) | {r['px4_coll']} | {r['fs_coll']:.1f} ({r['fs_coll_sd']:.1f}) | "
                 f"{r['px4_end']} | {' '.join(f'{k}:{v}' for k, v in sorted(r['fs_ends'].items()))} |")
    L += ['', '| config / preset | PX4 progress | fastsim progress | PX4 coll | fastsim coll |', '|---|---|---|---|---|']
    for k, v in o['by_preset'].items(): L.append(f"| {k} | {v['px4_prog']:.1f} | {v['fs_prog']:.1f} | {v['px4_coll']:.1f} | {v['fs_coll']:.1f} |")
    sp = o['speed']; L += ['', f"Speed: {sp}"]
    (OUT / f'calibration_tables{TAG}.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
