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

def figure():
    rows = [json.loads(line) for line in (REPORT/'scan-eval.jsonl').read_text().splitlines()]
    worlds = ['ff_medium_1','ff_hard_1','ff_medium_4','ff_hard_2']
    fig, axes = plt.subplots(1,2,figsize=(11,5.2))
    for ax,key,title in zip(axes,['progress_m','collisions'],['Farthest progress (m)','Contacts per flight']):
        for i,(cfg,label,col) in enumerate(zip(CONFIGS,LABELS,COLORS)):
            for j,world in enumerate(worlds):
                vals = [r[key] for r in rows if r['world']==world and r['config']==cfg]
                assert len(vals)==2
                x = j+(i-1.5)*.16
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
    fig.suptitle('Forest recovery trials',x=.07,ha='left',fontsize=19,weight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.text(.07,.10,'Each dot is one run. Lines connect two starts. Right of divider: held out for this change. Finish = 80 m.',fontsize=10)
    fig.subplots_adjust(left=.07,right=.98,bottom=.22,top=.82,wspace=.22)
    fig.savefig(REPORT/'scan-comparison.png',dpi=160)
    plt.close(fig)

def movie():
    from fastsim.runner import run_group
    from viz.common import Encoder
    from viz.replay2d import forest_raster
    world='ff_hard_1'
    specs=[dict(config=c,rep=0,duration=160) for c in (CONFIGS[1],CONFIGS[3])]
    flights=run_group(world,specs,record=True)
    runs=[f['run'] for f in flights]
    scan=runs[1]['phase'].index('scan')
    start=max(0,scan-3*30)
    end=min(start+48*30,min(len(r['x']) for r in runs)-1)
    raster,extent=forest_raster(json.loads((ROOT/'px4/forest'/f'{world}.json').read_text()))
    fig,axes=plt.subplots(1,2,figsize=(10,5.6),dpi=100)
    fig.subplots_adjust(left=.07,right=.98,bottom=.13,top=.79,wspace=.17)
    fig.suptitle('What happens after braking?',x=.07,ha='left',weight='bold',fontsize=19)
    clock=fig.text(.07,.88,'',fontsize=11)
    fig.text(.07,.035,'Top view at flight height   /   arrow = heading   /   red cross = contact   /   axes in metres',fontsize=10)
    artists=[]
    for ax,r,label,col in zip(axes,runs,[LABELS[1],LABELS[3]],[COLORS[1],COLORS[3]]):
        ax.imshow(raster,origin='lower',extent=extent,interpolation='nearest')
        ax.set_title(label,loc='left',weight='bold')
        line,=ax.plot([],[],color=col,lw=2)
        body,=ax.plot([],[],'o',color=col,ms=9)
        arrow=ax.quiver([0.],[0.],[1.],[0.],angles='xy',scale_units='xy',scale=1,color='#14212a',width=.014,zorder=5)
        hits=ax.scatter([],[],marker='x',color='#bb2222',s=50,zorder=6)
        phase=ax.text(.02,.02,'',transform=ax.transAxes,bbox=dict(facecolor='white',alpha=.93,edgecolor='none'),fontsize=11)
        artists.append((line,body,arrow,hits,phase))
    enc=Encoder(ROOT/'results/videos/scan-comparison.mp4',1000,560,15,channels=4)
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
                phase.set_text(f"{r['phase'][i]}   |   contacts {len(collisions)}")
            fig.canvas.draw()
            enc.add(np.asarray(fig.canvas.buffer_rgba()))
    finally:
        enc.close()
        plt.close(fig)

if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('output',choices=('figure','movie'))
    args=cli.parse_args()
    (movie if args.output=='movie' else figure)()
