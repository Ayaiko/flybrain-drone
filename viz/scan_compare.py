"""Rebuild the figure or synchronized flight excerpt from the scan experiment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REPORT = ROOT/'wiki/llm'
CONFIGS = ['fix-all2','fix-all2-side-guard','side-scan','side-scan-creep']
LABELS = ['Baseline','Brake on detour','Scan','Scan + slow start']
COLORS = ['#7b858e','#4366a3','#d37629','#087e76']

def figure(offset=False):
    configs = ['side-scan', 'scan-offset', 'scan-offset-long', 'scan-front', 'scan-memory'] if offset else CONFIGS
    labels = ['Scan baseline', 'Keep offset', 'Offset + longer leg', 'Scan at front brake', 'Remember turns'] if offset else LABELS
    prefix = 'offset' if offset else 'scan'
    rows = [json.loads(line) for line in (REPORT/f'{prefix}-eval.jsonl').read_text().splitlines()]
    worlds = ['ff_medium_1','ff_hard_1','ff_medium_4','ff_hard_2']
    fig, axes = plt.subplots(1,2,figsize=(11,5.2))
    for ax,key,title in zip(axes,['progress_m','collisions'],['Farthest progress (m)','Contacts per flight']):
        for i,(cfg,label,col) in enumerate(zip(configs,labels,COLORS + ['#8a4787'])):
            for j,world in enumerate(worlds):
                vals = [r[key] for r in rows if r['world']==world and r['config']==cfg]
                assert len(vals)==2
                x = j+(i-(len(configs)-1)/2)*.16
                ax.plot([x,x],[min(vals),max(vals)],color=col,lw=2)
                ax.scatter([x]*2,vals,color=col,s=30,label=label if j==0 else '',zorder=3)
        ax.set_xticks(range(4),['Medium 1','Hard 1','Medium 4','Hard 2'])
        ax.set_title(title,loc='left')
        ax.grid(axis='y',alpha=.25)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.axvline(1.5,color='#aaa',ls=':',lw=1)
        ax.set_ylim(bottom=0)
    axes[0].axhline(80,color='#555',ls='--',lw=.9)
    axes[0].set_ylim(0,85)
    fig.suptitle(('After the scan: route recovery' if offset else 'Forest recovery trials'),x=.07,ha='left',fontsize=19,weight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=len(configs),frameon=False)
    fig.text(.07,.10,'Each dot is one run. Lines connect two starts. Divider separates the original world pairs. Finish = 80 m.',fontsize=10)
    fig.subplots_adjust(left=.07,right=.98,bottom=.22,top=.82,wspace=.22)
    fig.savefig(REPORT/f'{prefix}-comparison.png',dpi=160)
    plt.close(fig)

def movie(offset=False):
    from fastsim.runner import run_group
    from viz.common import Encoder
    from viz.replay2d import forest_raster
    world='ff_hard_1'
    pair = ('side-scan', 'scan-memory', 'scan-front') if offset else (CONFIGS[1], CONFIGS[3])
    names = ['Scan baseline', 'Remember previous turns', 'Scan at the front brake'] if offset else [LABELS[1], LABELS[3]]
    prefix = 'offset' if offset else 'scan'
    specs=[dict(config=c,rep=0,duration=160) for c in pair]
    flights=run_group(world,specs,record=True)
    runs=[f['run'] for f in flights]
    if offset:
        phases = runs[0]['phase']
        moves = [i for i in range(1, len(phases)) if phases[i] == 'scan_move' and phases[i-1] != 'scan_move']
        scan = moves[1]
    else:
        scan = runs[1]['phase'].index('scan')
    start=max(0,scan-3*30)
    end=min(start+48*30,min(len(r['x']) for r in runs)-1)
    raster,extent=forest_raster(json.loads((ROOT/'px4/forest'/f'{world}.json').read_text()))
    width = 14 if offset else 10
    fig,axes=plt.subplots(1,len(pair),figsize=(width,5.6),dpi=100)
    fig.subplots_adjust(left=.07,right=.98,bottom=.13,top=.79,wspace=.17)
    fig.suptitle('What happens after braking?',x=.07,ha='left',weight='bold',fontsize=19)
    clock=fig.text(.07,.88,'',fontsize=11)
    fig.text(.07,.035,'Top view at flight height   /   arrow = heading   /   red cross = contact   /   axes in metres',fontsize=10)
    artists=[]
    for ax,r,label,col in zip(axes,runs,names,[COLORS[1],COLORS[3],COLORS[2]]):
        ax.imshow(raster,origin='lower',extent=extent,interpolation='nearest')
        ax.set_title(label,loc='left',weight='bold')
        line,=ax.plot([],[],color=col,lw=2)
        body,=ax.plot([],[],'o',color=col,ms=9)
        arrow=ax.quiver([0.],[0.],[1.],[0.],angles='xy',scale_units='xy',scale=1,color='#14212a',width=.014,zorder=5)
        hits=ax.scatter([],[],marker='x',color='#bb2222',s=50,zorder=6)
        phase=ax.text(.02,.02,'',transform=ax.transAxes,bbox=dict(facecolor='white',alpha=.93,edgecolor='none'),fontsize=11)
        artists.append((line,body,arrow,hits,phase))
    enc=Encoder(ROOT/f'results/videos/{prefix}-comparison.mp4',width*100,560,15,channels=4)
    try:
        for i in range(start,end+1,4):
            clock.set_text(f'Hard forest 1   /   start 0   /   sim time {i/30:.1f} s   /   synchronized replay at 2x')
            for ax,r,(line,body,arrow,hits,phase) in zip(axes,runs,artists):
                x,y,h=r['x'][i],r['y'][i],r['h'][i]
                line.set_data(r['x'][max(0,i-300):i+1],r['y'][max(0,i-300):i+1])
                body.set_data([x],[y]);arrow.set_offsets([[x,y]]);arrow.set_UVC([np.cos(h)],[np.sin(h)])
                ax.set_xlim(x-4.5,x+4.5);ax.set_ylim(y-4.5,y+4.5)
                collisions=[(c['x'],c['y']) for c in r['crashes'] if c['i']<=i]
                hits.set_offsets(np.array(collisions).reshape(-1,2))
                phase.set_text(f"{r['phase'][i]}   |   y {y:.1f} m   |   contacts {len(collisions)}")
            fig.canvas.draw()
            enc.add(np.asarray(fig.canvas.buffer_rgba()))
    finally:
        enc.close()
        plt.close(fig)


def validation_figure():
    """Six paired starts, grouped by environment. Every cell is a complete flight."""
    rows = [json.loads(line) for line in (REPORT/'offset-validation.jsonl').read_text().splitlines()]
    configs = ('side-scan', 'scan-memory', 'scan-front')
    worlds = ('ff_easy_0', 'ff_medium_2', 'ff_hard_0')
    keys = [(world, rep) for world in worlds for rep in range(2)]
    values = {(r['world'], r['rep'], r['config']): r for r in rows}
    assert len(values) == len(rows) == 18
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), facecolor='#f8f7f2')
    fig.subplots_adjust(left=.18, right=.97, top=.77, bottom=.15, wspace=.30)
    for ax, metric, title, cmap, limit in zip(
        axes, ('progress_m','collisions'), ('Farthest progress / m','Contacts / flight'),
        ('GnBu','Oranges'), (80, max(1, max(r['collisions'] for r in rows)))
    ):
        data = np.array([[values[w, rep, c][metric] for c in configs] for w, rep in keys])
        ax.imshow(data, cmap=cmap, vmin=0, vmax=limit, aspect='auto')
        for (i, j), val in np.ndenumerate(data):
            label = f'{val:.1f}' if metric == 'progress_m' else str(int(val))
            ax.text(j, i, label, ha='center', va='center', weight='bold',
                    color='white' if val/limit > .65 else '#182b32')
        ax.set_xticks(range(3), ['Baseline', 'Memory', 'Front scan'])
        ax.set_yticks(range(6), [w.replace('ff_','').replace('_',' ').title()+f'   /   {rep}' for w, rep in keys]
                      if ax is axes[0] else ['']*6)
        for y in (1.5,3.5):
            ax.axhline(y, color='white', linewidth=3)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        ax.set_title(title, loc='left', fontsize=13, pad=10)
    fig.text(.055,.91,'Do the gains carry over?',weight='bold',fontsize=23,color='#182b32')
    fig.text(.055,.845,'Three additional forests   /   Two matched starts each   /   Exit at 80 m',fontsize=12)
    fig.text(.055,.045,'Rows show forest seed / start. Higher progress and fewer contacts are preferable.',fontsize=10)
    fig.savefig(REPORT/'offset-validation.png', dpi=160)
    plt.close(fig)



def finish_movie():
    from fastsim.runner import run_episode
    from viz.common import Encoder
    from viz.replay2d import forest_raster

    world = 'ff_easy_0'
    flight = run_episode('scan-front', world, rep=0, duration=600, record=True)
    assert flight['summary']['success'], 'the replay must reproduce the clean exit'
    run = flight['run']
    raster, extent = forest_raster(json.loads((ROOT/'px4/forest'/f'{world}.json').read_text()))
    fig, (overview, detail) = plt.subplots(1, 2, figsize=(9.6, 6), dpi=100,
                                           gridspec_kw={'width_ratios':[1, 1.7]})
    fig.subplots_adjust(left=.06, right=.98, bottom=.11, top=.79, wspace=.20)
    fig.suptitle('A clean flight through the easy forest', x=.06, ha='left', weight='bold', fontsize=19)
    status = fig.text(.06,.865,'',fontsize=12)
    fig.text(.06,.03,'Easy forest 0  /  start 0  /  scan at front brake  /  overhead replay  /  positions in metres')
    tracks, markers = [], []
    for ax, label in ((overview,'Full route'), (detail,'Around the aircraft')):
        ax.imshow(raster, origin='lower', extent=extent, interpolation='nearest')
        ax.axhline(80, color='#ca5734', ls='--', lw=1.5)
        ax.set_title(label, loc='left', fontsize=11)
        tracks.append(ax.plot([], [], color='#287a79', lw=1.8)[0])
        markers.append(ax.plot([], [], 'o', color='#d46b26', ms=7)[0])
    overview.set_ylim(-2,84)
    overview.set_xlim(min(run['x'])-7,max(run['x'])+7)
    heading = detail.quiver([0.],[0.],[1.],[0.],angles='xy',scale_units='xy',scale=.6,color='#192e35',zorder=5)
    frames = list(range(0,len(run['x']),30))
    if frames[-1] != len(run['x'])-1:
        frames.append(len(run['x'])-1)
    encoder = Encoder(ROOT/'results/videos/scan-front-clean-exit.mp4',960,600,12,channels=4)
    try:
        for i in frames:
            x, y, h = run['x'][i], run['y'][i], run['h'][i]
            for path, dot in zip(tracks,markers):
                path.set_data(run['x'][:i+1],run['y'][:i+1])
                dot.set_data([x],[y])
            detail.set_xlim(x-5,x+5)
            detail.set_ylim(y-5,y+5)
            heading.set_offsets([[x,y]])
            heading.set_UVC([np.cos(h)],[np.sin(h)])
            phase = run['phase'][i] or 'route following'
            status.set_text(f'{i/30:5.1f} s   /   y {y:.1f} m   /   {phase}   /   0 contacts   /   12x')
            fig.canvas.draw()
            encoder.add(np.asarray(fig.canvas.buffer_rgba()))
    finally:
        encoder.close()
        plt.close(fig)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('output',choices=('figure','movie','validation','finish'))
    cli.add_argument('--offset', action='store_true', help='compare the post-scan route experiments')
    args=cli.parse_args()
    if args.output == 'finish':
        finish_movie()
    elif args.output == 'validation':
        validation_figure()
    else:
        (movie if args.output=='movie' else figure)(args.offset)
