"""Compare forest runs side by side, on the same worlds, under one early-stop rule.

  python -m px4.forest_compare <run id> <run id> ... [--presets easy,medium,hard] [--seeds 0,1,2] [--md]
                               [--rule stall_s=90,stall_m=3,max_coll=20,limit_s=360 | --rule none]

Reads the full run files (px4/runs/forest/<run id>_ff_<preset>_<seed>.json) and re-applies the stop rule to every run
(default: the one in forest_configs.json 'defaults'), so runs recorded without early stop (batch 1) and with it are
scored the same way. The first run id is the reference for the paired per-world differences.
--md prints Markdown tables ready for EXPERIMENTS.md."""
import argparse, json
from pathlib import Path
import numpy as np
from px4.forest_params import load_configs, rescore, debounce

ROOT = Path(__file__).resolve().parents[1]; ZONES = ['trunks', 'clearing', 'branches', 'lowtex']


def score(rid, world, rule):
    f = ROOT / f'px4/runs/forest/{rid}_{world}.json'
    if not f.exists(): return None
    d = json.loads(f.read_text()); s, run = d['summary'], d['run']; exit_y = json.loads((ROOT / f'px4/forest/{world}.json').read_text())['exit_y']
    if rule is None:
        r = dict(collisions=s['collisions'], progress_m=s['progress_m'], time_s=s['time_s'], end_reason=s.get('end_reason') or ('exit' if s['reached'] else 'timeout'),
                 reached=s['reached'], stop_i=len(run['y']))
    else:
        r = rescore(run, d['dt'], exit_y, rule.get('stall_s'), rule.get('stall_m'), rule.get('max_coll'), rule.get('limit_s'), reached=s['reached'])
        if r['end_reason'] == 'timeout' and s.get('end_reason') in ('stalled', 'too_many_collisions') and r['stop_i'] >= len(run['y']) - 1:
            r['end_reason'] = s['end_reason']          # the live rule fired on the last frame
    keep = set(debounce(run)); cr = [c for c in run['crashes'] if c['i'] < r['stop_i'] and c['i'] in keep]
    r['collisions'] = len(cr)
    seen, first = [], 0                       # first hit of each obstacle: by id when logged, else > 1.5 m from earlier hits
    for c in cr:
        key = c.get('oid')
        new = (key not in seen) if key is not None else all(np.hypot(c['x'] - a_[0], c['y'] - a_[1]) > 1.5 for a_ in seen)
        first += new; seen.append(key if key is not None else (c['x'], c['y']))
    r['first'] = first
    r.update(success=bool(r['reached'] and not cr), zones={z: sum(c['zone'] == z for c in cr) for z in ZONES},
             kinds={}, clear=min((v for k, v in s['min_clear_by_zone'].items()), default=np.nan))
    for c in cr: r['kinds'][c['kind']] = r['kinds'].get(c['kind'], 0) + 1
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rids', nargs='+'); ap.add_argument('--presets', default='easy,medium,hard'); ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--rule', default=None); ap.add_argument('--md', action='store_true')
    a = ap.parse_args(); D = load_configs()['defaults']
    if a.rule == 'none': rule = None
    elif a.rule: rule = {k: (None if v in ('none', 'null') else float(v)) for k, v in (p.split('=') for p in a.rule.split(','))}
    else: rule = dict(stall_s=D['stall_s'], stall_m=D['stall_m'], max_coll=D['max_coll'], limit_s=360)
    presets, seeds = a.presets.split(','), [int(s) for s in a.seeds.split(',')]
    R = {(rid, p, s): score(rid, f'ff_{p}_{s}', rule) for rid in a.rids for p in presets for s in seeds}
    print('stop rule:', rule)
    hdr = ['run', 'level', 'n', 'success', 'reached', 'coll/run', 'first hits/run', 're-hit %', 'coll/10 m', 'progress m', 'time s', 'ended', 'coll by zone (trunk/clear/branch/lowtex)']
    rows = []
    for rid in a.rids:
        for p in presets:
            rs = [R[(rid, p, s)] for s in seeds if R[(rid, p, s)]]
            if not rs: continue
            ends = {}
            for r in rs: ends[r['end_reason']] = ends.get(r['end_reason'], 0) + 1
            coll = sum(r['collisions'] for r in rs); prog = sum(r['progress_m'] for r in rs)
            rows.append([rid, p, len(rs), f"{sum(r['success'] for r in rs)}/{len(rs)}", f"{sum(r['reached'] for r in rs)}/{len(rs)}",
                         f"{coll / len(rs):.1f}", f"{sum(r['first'] for r in rs) / len(rs):.1f}", (f"{100 * (1 - sum(r['first'] for r in rs) / coll):.0f}" if coll else '-'), f"{10 * coll / max(prog, 1e-6):.2f}", f"{prog / len(rs):.0f}", f"{np.mean([r['time_s'] for r in rs]):.0f}",
                         ' '.join(f'{k}:{v}' for k, v in sorted(ends.items())), '/'.join(str(sum(r['zones'][z] for r in rs)) for z in ZONES)])
    if a.md:
        print('| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr))
        for r in rows: print('| ' + ' | '.join(map(str, r)) + ' |')
    else:
        w = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)]
        for r in [hdr] + rows: print('  '.join(str(x).ljust(n) for x, n in zip(r, w)))
    if len(a.rids) > 1:                      # paired, same worlds
        ref = a.rids[0]; print(f'\npaired vs {ref} (same worlds): progress m / collisions')
        for rid in a.rids[1:]:
            for p in presets:
                pairs = [(R[(ref, p, s)], R[(rid, p, s)], s) for s in seeds if R[(ref, p, s)] and R[(rid, p, s)]]
                if not pairs: continue
                dp = [b['progress_m'] - a_['progress_m'] for a_, b, _ in pairs]; dc = [b['collisions'] - a_['collisions'] for a_, b, _ in pairs]
                print(f"  {rid[:40]:40s} {p:7s} Δprogress {np.mean(dp):+6.1f} m ({', '.join(f'{v:+.0f}' for v in dp)})   "
                      f"Δcollisions {np.mean(dc):+5.1f} ({', '.join(f'{v:+d}' for v in dc)})")


if __name__ == '__main__':
    main()
