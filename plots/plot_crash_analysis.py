"""Crash anatomy figure -> results/figures/crash_analysis.png
Run from the project root: python -m plots.plot_crash_analysis"""
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / 'results/crash_analysis/crash_analysis.json'))
CATS = ['missed', 'side', 'settling', 'saccade', 'recrash', 'late']
LAB = ['head-on,\nlooming\nnever fired', 'from side\n(>45°)', 'blind window\nafter saccade', 'mid-\nsaccade', 're-crash\n(<1 s)', 'looming fired\ntoo late']
names = list(R['report']); col = {'fly_innate_saccade': '#e8b04a', 'fly_tuned_crash_penalty': '#2a9d4b', 'fly_lplc2': '#2a6fdb'}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
x = np.arange(len(CATS)); w = .8 / len(names)
for i, n in enumerate(names):
    s = R['report'][n]['share']; ax[0].bar(x + (i - (len(names) - 1) / 2) * w, [100 * s[c] for c in CATS], w, color=col[n], label=n.replace('_', ' '))
ax[0].set_xticks(x); ax[0].set_xticklabels(LAB, fontsize=8); ax[0].set_ylabel('% of crashes (each controller)'); ax[0].legend(fontsize=8)
ax[0].set_title('What kind of crash (100 unseen worlds x 60 s)')
cs = [c for c in R['crashes'] if c.get('ctrl', 'fly_lplc2') != 'fly_lplc2']
ax[1].hist([c['bearing'] for c in cs if c['cat'] != 'recrash'], bins=np.arange(-180, 181, 15), color='#555')
ax[1].axvspan(-45, 45, color='#e8b04a', alpha=.2, label='front ±45°'); ax[1].axvspan(-20, 20, color='#d9534f', alpha=.2, label='front ±20°')
ax[1].set_xlabel('where the obstacle was hit, relative to heading (°)'); ax[1].set_title('Hit bearing (both controllers)'); ax[1].legend(fontsize=8)
t = np.array([c['t'] for c in cs if c['cat'] == 'missed']); ta = np.array([c['t'] for c in cs if c['cat'] != 'recrash'])
bins = np.arange(0, 61, 10); a, _ = np.histogram(t, bins); b, _ = np.histogram(ta, bins)
ax[2].bar(bins[:-1] + 5, 100 * a / np.maximum(b, 1), 8, color='#d9534f'); ax[2].set_ylim(0, 100)
ax[2].set_xlabel('time in episode (s)'); ax[2].set_ylabel('% of crashes that were "never fired"')
d = R['report'][names[0]]['loom_threshold_drift_x']
ax[2].set_title(f'Old looming threshold grows x{d:.2f} over 60 s')
fig.tight_layout(); out = ROOT / 'results/figures/crash_analysis.png'; fig.savefig(out, dpi=85); print('saved', out)
