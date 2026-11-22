"""Summary figure from saved results -> results/figures/summary.png
Run from the project root: python -m plots.plot_summary"""
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
h6 = json.load(open(ROOT / 'results/h6/h6_results.json'))
cb = json.load(open(ROOT / 'results/compute/compute_bench.json'))['controllers']

# Measured controllers + order-of-magnitude references (NOT measured here; hatched in the plot):
#  small obstacle-avoidance CNN (DroNet-class ResNet-8 on a small grey image): ~1e7 multiply-adds per frame
#  whole FlyWire connectome as a spiking sim: ~140k neurons x 200 steps (0.1 ms) per 20 ms frame + synaptic events ~ 3e7
measured = {'fly circuit': cb['fly (EMD + flow balance + looming)'], 'linear (scratch)': cb['linear read-out (EMD + 98 weights)'],
            'oracle (depth)': cb['oracle (needs depth sensor)']}
refs = {'small CNN\n(estimate)': 1e7, 'whole fly brain\nsim (estimate)': 3e7}

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
a = ax[0, 0]
names = list(measured) + list(refs); vals = [m['mac_per_frame'] for m in measured.values()] + list(refs.values())
cols = ['#2a9d4b', '#e0812b', '#2a6fdb', '#999', '#999']
bars = a.bar(names, vals, color=cols)
for b in bars[len(measured):]: b.set_hatch('//'); b.set_edgecolor('#555')
a.set_yscale('log'); a.set_ylabel('multiply-adds per frame (log)'); a.set_title('Brain compute per decision (50 frames/s)')
for b, v in zip(bars, vals): a.text(b.get_x() + b.get_width() / 2, v * 1.3, f'{v:,.0f}', ha='center', fontsize=8)

a = ax[0, 1]
mn = list(measured); us = [measured[k]['us_per_frame_python'] for k in mn]
bs = a.bar(mn, us, color=cols[:3]); a.set_ylabel('microseconds per frame'); a.set_title('Measured time per decision (Python/numpy, 1 core, Apple M5)')
for b, v in zip(bs, us): a.text(b.get_x() + b.get_width() / 2, v + .5, f'{v:.1f} us', ha='center', fontsize=9)
a.text(.98, .82, 'linear is slow only because of a Python loop;\nits arithmetic is the same size as the fly circuit',
       transform=a.transAxes, ha='right', va='top', fontsize=8, color='#555')

a = ax[1, 0]
order = ['no_avoidance', 'fly_innate_continuous', 'linear_scratch_tuned', 'fly_innate_saccade', 'fly_tuned_saccade', 'oracle_depth']
lab = ['no avoidance', 'fly innate\ncontinuous', 'linear\nscratch tuned', 'fly innate\nsaccade', 'fly tuned\nsaccade', 'oracle\n(depth)']
cells = [h6['test'][k]['cells'] for k in order]; coll = [h6['test'][k]['collision_rate'] * 100 for k in order]
x = np.arange(len(order))
b1 = a.bar(x - .2, cells, .4, label='cells explored (2x2 m)', color='#2a9d4b')
a2 = a.twinx(); b2 = a2.bar(x + .2, coll, .4, label='collision rate %', color='#d0453b', alpha=.75)
a.set_xticks(x); a.set_xticklabels(lab, fontsize=8); a.set_ylabel('cells explored'); a2.set_ylabel('collision rate (%)'); a2.set_ylim(0, 110)
a.set_title(f'Held-out test worlds (n={h6["budget"]["test"].__len__()}, 60 s each)')
a.legend(handles=[b1, b2], loc='upper left', fontsize=8)

a = ax[1, 1]
g = np.arange(1, len(h6['fly_curve']) + 1)
a.plot(g, h6['fly_curve'], 'o-', color='#2a9d4b', label='fly circuit (starts from innate values)')
a.plot(g, h6['lin_curve'], 's-', color='#e0812b', label='linear from scratch')
a.set_xlabel('CEM generation (same budget for both)'); a.set_ylabel('best cells on training worlds')
a.set_title('H6: fine-tuning an innate prior vs learning from scratch'); a.legend(fontsize=8)
a.text(.98, .05, f'test: fly tuned {h6["test"]["fly_tuned_saccade"]["cells"]:.1f} vs linear {h6["test"]["linear_scratch_tuned"]["cells"]:.1f} cells',
       transform=a.transAxes, ha='right', fontsize=9)
fig.tight_layout(); out = ROOT / 'results/figures/summary.png'; fig.savefig(out, dpi=85); print('saved', out)
