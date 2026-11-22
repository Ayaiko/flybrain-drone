"""Offline clearance map around the stalled route, using simulation geometry only."""
from collections import deque
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastsim.world import World
from fastsim.truth import BatchTruth


def connected_path(grid, start, goal_row, margin):
    queue = deque([start])
    parent = {start:None}
    while queue:
        y,x = queue.popleft()
        if y >= goal_row:
            path = []
            pos = (y,x)
            while pos is not None:
                path.append(pos)
                pos = parent[pos]
            return path[::-1]
        for dy,dx in ((1,0),(0,1),(0,-1),(-1,0)):
            q = (y+dy,x+dx)
            if 0<=q[0]<grid.shape[0] and 0<=q[1]<grid.shape[1] and q not in parent and grid[q]>=margin:
                parent[q] = (y,x)
                queue.append(q)
    return []


def make(run,output):
    world = World('ff_hard_0')
    truth = BatchTruth(world.obs)
    x = np.arange(-6,9.01,.05)
    y = np.arange(42,53.01,.05)
    xx,yy = np.meshgrid(x,y)
    poses = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size,world.meta['alt'])])
    grid = np.concatenate([truth.query(chunk)[0] for chunk in np.array_split(poses,256)]).reshape(xx.shape)
    i = int(np.argmax(run['y']))
    start = int(np.argmin(abs(y-run['y'][i]))), int(np.argmin(abs(x-run['x'][i])))
    goal = int(np.argmin(abs(y-52)))
    paths = {}
    for margin in (.05,.10,.20):
        result = connected_path(grid,start,goal,margin) if grid[start]>=margin else []
        paths[margin] = result
    fig,ax = plt.subplots(figsize=(8,6))
    cf = ax.contourf(xx,yy,np.clip(grid,-.1,1),levels=[-.11,0,.1,.2,.4,1.01],colors=['#333a40','#a3aaac','#c8c6b8','#dfded2','#f3f1e9'])
    fig.colorbar(cf,ax=ax,label='Modelled body clearance (m)')
    ax.plot(run['x'][-2700:],run['y'][-2700:],color='#c56334',label='Final 90 s',lw=1.8)
    ok=max((k for k,v in paths.items() if v),default=None)
    if ok is not None:
        a,b=np.array(paths[ok]).T
        ax.plot(x[b],y[a],color='#087875',label=f'Grid path with ≥{ok:.2f} m at nodes',lw=2)
    ax.scatter([run['x'][i]],[run['y'][i]],color='black',s=20,label='Farthest point')
    ax.axhline(52,color='#888',linestyle=':',lw=1)
    ax.set(xlim=(x.min(),x.max()),ylim=(y.min(),y.max()),aspect='equal',xlabel='Across course (m)',ylabel='Along course (m)',title='Hard 0, start 1   /   branch section at 1.50 m altitude')
    ax.legend(loc='lower left',framealpha=.95,fontsize=9)
    fig.tight_layout()
    fig.savefig(output/'departure-clearance.png',dpi=160)
    plt.close(fig)
    (output/'departure-clearance.json').write_text(json.dumps(dict(start_xy=[run['x'][i],run['y'][i]],nearest_grid_clearance_m=float(grid[start]),node_spacing_m=.05,goal_y_m=52,reachable={str(k):bool(v) for k,v in paths.items()}),indent=2)+'\n')


if __name__=='__main__':
    make(json.loads(Path(sys.argv[1]).read_text())['run'],ROOT/'wiki/llm')
