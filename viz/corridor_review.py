"""Visualize measured flight trajectories from the corridor experiment."""
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from viz.common import Encoder
from viz.hard_contact_3d import geometry, panel, TITLE, SMALL, W

CONFIGS = ['depart-loom35','corridor-depth','corridor-preview','corridor-forward','corridor-follow','corridor-approach']
LABELS = ['Original','Depth scoring','Depth + early scan','Forward depth','Follow corridor','Follow + early scan']
COLORS = ['#676e76','#c37238','#177c79','#5753a3','#008194','#a54067']


def bundle(directory):
    rows, pairs = [], {'full':{}, 'entry':{}}
    for p in sorted(directory.glob('corr26-*.json')):
        if not p.name.startswith(('corr26-ff_', 'corr26-entry-')):
            continue
        f=json.loads(p.read_text())
        s=f['summary']
        mode='entry' if p.name.startswith('corr26-entry-') else 'full'
        rows.append(dict(test=mode, **{k:s[k] for k in ('world','rep','config','collisions','progress_m','time_s','end_reason','preview_count','scan_count','dist_m')}))
        if s['world']=='ff_hard_0' and s['rep']==1:
            pairs[mode][s['config']]=dict(run=f['run'],events=f.get('events',[]))
    return dict(rows=rows,pairs=pairs)


def graph(data, chosen, out):
    fig,axes=plt.subplots(2,1,figsize=(9,6.6),sharex=True,layout='constrained')
    for ax,(mode,title) in zip(axes,[('full','Planner active from takeoff'),('entry','Identical approach to the branch section')]):
        for cfg,label,color in zip(CONFIGS,LABELS,COLORS):
            if cfg not in {CONFIGS[0],chosen,'corridor-follow'}:
                continue
            r=data['pairs'][mode][cfg]['run']
            t=np.arange(len(r['y']))/30
            ax.plot(t,r['y'],color=color,label=label,lw=1.8)
            ax.scatter([c['t'] for c in r['crashes']],[c['y'] for c in r['crashes']],color=color,marker='x',s=24)
        for y in (47,52):
            ax.axhline(y,color='#444',ls=':',lw=.9)
            ax.text(.99,y+.35,f'{y} m',transform=ax.get_yaxis_transform(),ha='right',fontsize=8)
        if mode=='entry':
            events=data['pairs'][mode][CONFIGS[0]]['events']
            ax.axvline(next(e['t'] for e in events if e['event']=='entry'),color='#999',ls='--')
        ax.set(title=title,ylabel='Along-course position (m)',ylim=(0,84))
        ax.spines[['right','top']].set_visible(False)
    axes[0].legend(ncol=3,frameon=False)
    axes[1].set_xlabel('Simulation time (s)   /   crosses = contacts')
    fig.suptitle('Hard forest 0  /  start 1',weight='bold')
    fig.savefig(out,dpi=160)
    plt.close(fig)


def movie(data, chosen, out):
    objects=geometry(json.loads((ROOT/'px4/forest/ff_hard_0.json').read_text())['obstacles'])
    enc=Encoder(out,1280,656,15,channels=3)
    try:
        frames=[]
        for mode in ('full','entry'):
            pair=[data['pairs'][mode][c] for c in (CONFIGS[0],chosen)]
            if mode=='entry':
                start=round(next(e['t'] for e in pair[0]['events'] if e['event']=='entry')*30)
            else:
                start=min(i for f in pair for i,y in enumerate(f['run']['y']) if y>=38)
            end=max(len(f['run']['x']) for f in pair)-1
            frames.extend((mode,pair,start,k) for k in [*range(start,end,36),end])
        for mode,pair,start,k in frames:
            im=Image.new('L',(1280,656),248)
            d=ImageDraw.Draw(im)
            chapter='active from takeoff' if mode=='full' else 'same entry at 38 m'
            d.text((16,7),f'Hard 0 / start 1    |    {chapter}    |    {k/30:.1f} s    |    18x',font=TITLE,fill=20)
            for j,(f,cfg) in enumerate(zip(pair,[CONFIGS[0],chosen])):
                r=f['run'];i=min(k,len(r['x'])-1);off=j*W
                im.paste(panel(r,i,objects,cfg),(off,42))
                d.text((off+18,487),f"Position {r['y'][i]:.2f} m    /    {'stopped' if k>i else r['phase'][i]}",font=TITLE,fill=32)
                recent=[e for e in f['events'] if e['event']=='selected' and start/30<=e['t']<=i/30]
                desc=(f"Last chosen heading: supported to {recent[-1]['depth']:.1f} m" if recent else f"Peak progress: {max(r['y'][:i+1]):.2f} m")
                d.text((off+18,530),desc,font=SMALL,fill=44)
                for y,label in [(47,'47'),(52,'52')]:
                    bx=off+20+(y-38)*18;d.line((bx,574,bx,604),fill=100,width=1);d.text((bx,602),label,font=SMALL,fill=40)
                d.line((off+20,587,off+580,587),fill=145,width=2)
                bx=off+20+np.clip(r['y'][i]-38,0,31)*18;d.ellipse((bx-5,582,bx+5,592),fill=30)
            if mode=='full' and k==start: im.save(ROOT/'wiki/llm/corridor-preview-3d.png')
            enc.add(np.asarray(im.convert('RGB')))
    finally:
        enc.close()


if __name__=='__main__':
    source=Path(sys.argv[1])
    chosen=sys.argv[2] if len(sys.argv)>2 else 'corridor-preview'
    wiki=ROOT/'wiki/llm'
    if source.is_dir():
        data=bundle(source)
        with gzip.open(wiki/'corridor-traces.json.gz','wt') as f:
            json.dump(data,f,allow_nan=False,separators=(',',':'))
    else:
        with gzip.open(source,'rt') as f: data=json.load(f)
    (wiki/'corridor-trials.json').write_text(json.dumps(data['rows'],indent=2,allow_nan=False)+'\n')
    graph(data,chosen,wiki/'corridor-trials.png')
    movie(data,chosen,ROOT/'results/videos/corridor-before-after-3d.mp4')
