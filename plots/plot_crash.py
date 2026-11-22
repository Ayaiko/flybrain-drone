"""Crash-penalty comparison figure -> results/figures/crash_penalty.png
Run from the project root: python -m plots.plot_crash"""
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / 'results/crash/crash_results.json')); S = R['summary']
order = ['no_avoidance', 'linear_crash_penalty', 'fly_innate_continuous', 'fly_innate_saccade',
         'fly_tuned_explore_only', 'fly_tuned_crash_penalty', 'oracle_depth']
lab = ['no\navoidance', 'linear\ncrash-pen.', 'fly innate\ncontinuous', 'fly innate\nsaccade',
       'fly tuned\nexplore only', 'fly tuned\ncrash-pen.', 'oracle\n(depth)']
col = ['#999', '#e0812b', '#b04bd6', '#e8b04a', '#8fcf9f', '#2a9d4b', '#2a6fdb']
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8)); x = np.arange(len(order))
v = np.array([S[k]['crashes_per_min'] for k in order]); lo = np.array([S[k]['crashes_ci'][0] for k in order]); hi = np.array([S[k]['crashes_ci'][1] for k in order])
ax[0].bar(x, v, color=col, yerr=[v - lo, hi - v], capsize=4); ax[0].set_title('Crashes per minute (lower is better)')
for k, xs in R['per_run_crashes_per_min'].items(): ax[0].scatter([order.index(k)] * len(xs), xs, color='k', s=14, zorder=3)
v = np.array([S[k]['cells'] for k in order]); lo = np.array([S[k]['cells_ci'][0] for k in order]); hi = np.array([S[k]['cells_ci'][1] for k in order])
ax[1].bar(x, v, color=col, yerr=[v - lo, hi - v], capsize=4); ax[1].set_title('Cells explored in 60 s')
ax[2].bar(x, [S[k]['m_per_crash'] for k in order], color=col); ax[2].set_yscale('log'); ax[2].set_title('Metres flown per crash (log, higher is better)')
for a in ax: a.set_xticks(x); a.set_xticklabels(lab, fontsize=8)
b = R['budget']
fig.suptitle(f'Bounce mode (crash -> turn around, keep flying), {b["test"]} unseen worlds x 60 s, {b["runs"]} trainings each '
             f'(dots), fitness = cells - {R["lambda_"]:.0f} x crashes', fontsize=10)
fig.tight_layout(); out = ROOT / 'results/figures/crash_penalty.png'; fig.savefig(out, dpi=85); print('saved', out)
