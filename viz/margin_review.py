"""Run or visualize the forest brake-margin comparison. See wiki/llm/margin-review.md."""
import argparse
import inspect
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from viz.hard_contact_3d import geometry,panel,TITLE,SMALL,W
from viz.common import Encoder

CASES = {
    'baseline': {},
    'fixed': {'tof_stop': .70},
    'balanced': {'tof_stop': .55, 'tof_t': 1.25},
    'moderate': {'tof_stop': .60},
}


def simulate(traces, world, rep):
    from fastsim.runner import run_group

    traces.mkdir(parents=True, exist_ok=True)
    specs = [dict(config='scan-front', rep=rep, duration=600, overrides=v) for v in CASES.values()]
    timing = {'pose_tick': 0} if 'pose_tick' in inspect.signature(run_group).parameters else {}
    flights = run_group(world, specs, record=True, **timing)
    for name, flight in zip(CASES, flights):
        flight['run'] = {key: flight['run'][key] for key in ('x','y','z','h','phase','tof','crashes')}
        (traces / f'{world}-{rep}-{name}.json').write_text(json.dumps(flight, allow_nan=False))


def render(traces,choice):
    names=list(CASES)
    colors=['#7d8590','#bb633c','#067d77','#534993']
    rows=[]
    for path in sorted(traces.glob('ff_*.json')):
        world,rep,name=path.stem.split('-')
        s=json.loads(path.read_text())['summary']
        rows.append(dict(world=world,rep=int(rep),variant=name,**{k:s[k] for k in ('progress_m','collisions','dist_m','end_reason','reached')}))
    keys=sorted({(r['world'],r['rep']) for r in rows})
    report=ROOT/'wiki/llm'
    (report/'margin-review.json').write_text(json.dumps(rows,indent=2)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(9,5.5),sharex=True)
    for ax,key,title in zip(axes,['collisions','progress_m'],['Contacts','Farthest progress (m)']):
        for j,(name,color) in enumerate(zip(names,colors)):
            values=[next(r[key] for r in rows if (r['world'],r['rep'],r['variant'])==(w,i,name)) for w,i in keys]
            ax.bar(np.arange(len(keys))+(j-1.5)*.19,values,.18,color=color,label=name)
        ax.set_ylabel(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['right','top']].set_visible(False)
    axes[0].legend(ncol=4,frameon=False)
    axes[1].set_xticks(range(len(keys)),[f"{w.removeprefix('ff_').replace('_',' ')}\nstart {i}" for w,i in keys])
    fig.suptitle('Brake margin across matched starts',fontsize=17,weight='bold')
    fig.tight_layout()
    fig.savefig(report/'margin-review.png',dpi=160);plt.close(fig)
    runs=[json.loads((traces/f'ff_hard_0-0-{n}.json').read_text())['run'] for n in ['fixed',choice]]
    obs=geometry(json.loads((ROOT/'px4/forest/ff_hard_0.json').read_text())['obstacles'])
    first=runs[0]['crashes'][0]['i']
    runs1=[json.loads((traces/f'ff_hard_0-1-{n}.json').read_text())['run'] for n in ['baseline',choice]]
    original = json.loads((traces/'ff_hard_0-0-baseline.json').read_text())['run']
    phases = zip(original['phase'], runs[0]['phase'])
    divergence = next(i for i,(a,b) in enumerate(phases) if a != b)
    scenes=[(0,runs,'fixed',divergence,'First brake decision'),(0,runs,'fixed',first,'Fixed-margin contact'),
            (1,runs1,'baseline',runs1[0]['crashes'][0]['i'],'Original baseline contact')]
    enc=Encoder(ROOT/'results/videos/margin-review-3d.mp4',1280,512,15,channels=3)
    try:
        for rep,pair,reference,center,desc in scenes:
            for i in range(center-90,center+121,2):
                im=Image.new('L',(1280,512),250);draw=ImageDraw.Draw(im)
                draw.text((16,8),f'Hard 0 / start {rep}     {i/30:.2f} s     /     {desc}',font=TITLE,fill=25)
                for j,(r,name) in enumerate(zip(pair,[reference,choice])):
                    frame=min(i,len(r['x'])-1)
                    im.paste(panel(r,frame,obs,name),(j*W,44))
                draw.text((16,483),'Circle = aircraft centre   /   same scale and time   /   grid = 1 m',font=SMALL,fill=40)
                if rep==0 and i==first: im.save(report/'margin-review-3d.png')
                enc.add(np.asarray(im.convert('RGB')))
    finally:
        enc.close()

if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('traces', type=Path)
    cli.add_argument('--world')
    cli.add_argument('--rep', type=int, default=0)
    cli.add_argument('--choice', choices=['balanced','moderate'], default='moderate')
    args = cli.parse_args()
    if args.world:
        simulate(args.traces, args.world, args.rep)
    else:
        render(args.traces, args.choice)
