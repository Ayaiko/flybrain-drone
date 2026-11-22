"""Robust-eval figure with 95% bootstrap CIs -> results/figures/robust.png
Run from the project root: python -m plots.plot_robust"""
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / 'results/robust/robust_results.json'))
S = R['summary']
order = ['no_avoidance', 'fly_innate_continuous', 'linear_scratch_tuned', 'fly_innate_saccade', 'fly_tuned_saccade', 'oracle_depth']
lab = ['no\navoidance', 'fly innate\ncontinuous', 'linear\nscratch', 'fly innate\nsaccade', 'fly tuned\nsaccade', 'oracle\n(depth)']
col = ['#999', '#b04bd6', '#e0812b', '#e8b04a', '#2a9d4b', '#2a6fdb']
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
for a, key, ci, title, scale in [(ax[0], 'cells', 'cells_ci', 'Cells explored before first crash', 1),
                                 (ax[1], 'collision_rate', 'collision_rate_ci', 'Collision rate (%)', 100),
                                 (ax[2], 'm_per_collision', 'm_per_collision_ci', 'Metres flown per collision (log)', 1)]:
    v = np.array([S[k][key] for k in order]) * scale
    lo = np.array([S[k][ci][0] for k in order]) * scale; hi = np.array([S[k][ci][1] for k in order]) * scale
    a.bar(range(len(order)), v, color=col, yerr=[v - lo, hi - v], capsize=4, error_kw=dict(lw=1))
    a.set_xticks(range(len(order))); a.set_xticklabels(lab, fontsize=8); a.set_title(title)
    if key == 'm_per_collision': a.set_yscale('log')
for k, xs in R['per_run_cells'].items():
    ax[0].scatter([order.index(k)] * len(xs), xs, color='k', s=14, zorder=3)
p = R['paired']; b = R['budget']
fig.suptitle(f'{b["test"]} unseen worlds, {b["runs"]} independent trainings (dots = each run). '
             f'Fly tuned - linear: {p["mean_diff"]:+.1f} cells [95% CI {p["ci"][0]:+.1f}, {p["ci"][1]:+.1f}], '
             f'fly better on {p["worlds_fly_better"] * 100:.0f}% of worlds', fontsize=10)
fig.tight_layout(); out = ROOT / 'results/figures/robust.png'; fig.savefig(out, dpi=85); print('saved', out)
