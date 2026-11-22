"""Top-down flight paths on held-out worlds for each controller -> results/paths.png"""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from flydrone.sim import run_episode
from flydrone.controllers import Random, Oracle, FlyVision
from experiments.h6_finetune import fly_kw

ROOT = Path(__file__).resolve().parents[1]
fly_u = np.load(ROOT / 'results/h6/fly_tuned_u.npy')
CTRL = [('no avoidance', Random, {}, '#888'), ('fly innate (saccade)', FlyVision, {}, '#e0812b'),
        ('fly innate (continuous)', FlyVision, {'saccadic': False}, '#b04bd6'),
        ('fly tuned (saccade)', FlyVision, fly_kw(fly_u), '#2a9d4b'), ('oracle depth', Oracle, {}, '#2a6fdb')]
seeds = [0, 1, 2, 3]
fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 5.4))
for ax, sd in zip(axes, seeds):
    for name, cls, kw, colr in CTRL:
        r = run_episode(cls, sd, T=60, ctrl_kw=kw, record=True)
        ax.plot(*r['path'].T, color=colr, lw=1.3, label=f"{name}: {r['cells']} cells" + (' ✕' if r['collided'] else ''))
        if r['collided']: ax.plot(*r['path'][-1], 'x', color=colr, ms=9, mew=2)
    w = r['world']
    for c, rad in zip(w.c, w.r): ax.add_patch(plt.Circle(c, rad, color='#333', alpha=.55))
    ax.set_xlim(0, w.size); ax.set_ylim(0, w.size); ax.set_aspect('equal'); ax.set_title(f'test world {sd}')
    ax.plot(*w.start, 'k^', ms=8); ax.legend(fontsize=7, loc='upper left')
fig.tight_layout(); fig.savefig(ROOT / 'results/figures/paths.png', dpi=80); print('saved results/figures/paths.png')
