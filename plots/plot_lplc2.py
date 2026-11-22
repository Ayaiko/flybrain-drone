"""LPLC2 fix: crashes vs exploration trade-off -> results/figures/lplc2.png
Run from the project root: python -m plots.plot_lplc2"""
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / 'results/lplc2/lplc2_results.json')); S = R['summary']
C = json.load(open(ROOT / 'results/crash/crash_results.json'))['summary']
pts = [('no avoidance', S['no_avoidance'], '#999'), ('fly old looming', S['fly_old_looming'], '#e8b04a'),
       ('fly old, tuned crash-pen.', C['fly_tuned_crash_penalty'], '#2a9d4b'), ('linear crash-pen.', C['linear_crash_penalty'], '#e0812b'),
       ('LPLC2: direction only', S['lplc2_direction_only'], '#c9a0dc'), ('LPLC2: threshold only', S['lplc2_threshold_only'], '#7fb3ff'),
       ('LPLC2 full (thr 2.0)', S['lplc2_full'], '#2a6fdb'), ('LPLC2 full bolder (thr 2.5)', S['lplc2_full_bolder'], '#1b3f8f'),
       ('oracle (true depth)', S['oracle_depth'], '#d9534f')]
TU = ROOT / 'results/lplc2/lplc2_tune.json'
if TU.exists():
    tu = json.load(open(TU)); pts.append(('LPLC2 tuned crash-pen. (3 runs)', tu['summary']['lplc2_tuned'], '#111'))
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for name, s, c in pts:
    ax.errorbar(s['cells'], s['crashes_per_min'], xerr=[[s['cells'] - s['cells_ci'][0]], [s['cells_ci'][1] - s['cells']]],
                yerr=[[s['crashes_per_min'] - s['crashes_ci'][0]], [s['crashes_ci'][1] - s['crashes_per_min']]], fmt='o', color=c, ms=8, capsize=3, label=name)
ax.set_yscale('symlog', linthresh=0.5); ax.set_xlabel('cells explored in 60 s (higher = explores more)'); ax.set_ylabel('crashes per minute (lower = safer)')
p = R['paired']
ax.set_title(f'100 unseen worlds, bounce mode. LPLC2 full - old: {p["mean_diff"]:+.2f} crashes/min '
             f'[95% CI {p["ci"][0]:+.2f}, {p["ci"][1]:+.2f}]', fontsize=10)
if TU.exists():
    for r in tu['per_run']: ax.scatter(r['cells'], r['crashes_per_min'], color='#111', marker='x', s=30, zorder=3)
ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=.3)
fig.tight_layout(); out = ROOT / 'results/figures/lplc2.png'; fig.savefig(out, dpi=90); print('saved', out)
