"""Render the departure experiment from recorded fastsim runs."""
import argparse
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

NAMES = {'ref':'Original', 'loom35':'Higher threshold', 'both':'Higher + direct goal', 'far35':'Higher at max range'}
COLORS = ['#808790','#bf682e','#355baa','#137c70']


def load(traces):
    if traces.is_file():
        with gzip.open(traces, 'rt') as f:
            bundle = json.load(f)
        return bundle['rows'], bundle['target']
    rows, target = [], {}
    for path in sorted(traces.glob('trace-*.json')):
        _,world,rep,duration,name = path.stem.split('-')
        flight = json.loads(path.read_text())
        s = flight['summary']
        rows.append(dict(world=world,rep=int(rep),duration=int(duration),variant=name,**{
            k:s[k] for k in ('progress_m','dist_m','time_s','collisions','reached','end_reason','min_clear_m','turn_kinds','scan_count')}))
        if world=='ff_hard_0' and rep=='1' and duration=='600' and name in NAMES:
            target[name]=flight['run']
    expected = {(world, rep, name) for world,rep in [('ff_hard_0',i) for i in range(4)]+[('ff_medium_2',1),('ff_easy_0',0)] for name in NAMES}
    keys = [(r['world'],r['rep'],r['variant']) for r in rows if r['duration']==600 and (r['world'],r['rep'],r['variant']) in expected]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError('Incomplete full-length comparison')
    return rows,target


def graph(rows, target, output):
    fig, (path, hits) = plt.subplots(2,1,figsize=(9,6.4),layout='constrained')
    keys=sorted({(r['world'],r['rep']) for r in rows if r['duration']==600 and r['variant']=='far35'})
    for j,((name,label),color) in enumerate(zip(NAMES.items(),COLORS)):
        run=target[name]
        path.plot(np.arange(len(run['y']))/30,run['y'],color=color,label=label,linewidth=1.6)
        h=run['crashes']
        path.scatter([c['t'] for c in h],[c['y'] for c in h],color=color,s=20,marker='x')
        by_key={(r['world'],r['rep']):r for r in rows if r['duration']==600 and r['variant']==name}
        hits.bar(np.arange(len(keys))+(j-1.5)*.20,[by_key[k]['collisions'] for k in keys],.195,color=color)
    path.set(title='Hard 0, start 1   /   crosses mark contacts',ylabel='Course position (m)',xlabel='Simulation time (s)')
    path.legend(frameon=False,ncol=2)
    hits.set(title='Matched starts   /   full flight, up to 600 seconds',ylabel='Contacts per flight')
    hits.set_xticks(range(len(keys)),[f"{w.replace('ff_','').replace('_',' ')}\nstart {rep}" for w,rep in keys])
    for ax in (path,hits):
        ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(output,dpi=160)
    plt.close(fig)


def movie(target, choice, report):
    objects=geometry(json.loads((ROOT/'px4/forest/ff_hard_0.json').read_text())['obstacles'])
    runs=[target['ref'],target[choice]]
    # Keep the same ground coordinate system for both miniature route maps.
    speed, fps = 6, 15
    last=min(180*30,max(len(r['x']) for r in runs)-1)
    enc=Encoder(ROOT/'results/videos/departure-50hz-3d.mp4',1280,656,fps,channels=3)
    try:
        for k in range(0,last+1,round(30*speed/fps)):
            im=Image.new('L',(1280,656),247); d=ImageDraw.Draw(im)
            d.text((16,6),f'Hard forest 0 / start 1     {k/30:.1f} s     /     replay {speed}x',font=TITLE,fill=20)
            for j,(run,label) in enumerate(zip(runs,['Before / Original','After / '+NAMES[choice]])):
                i=min(k,len(run['x'])-1);x0=j*W
                im.paste(panel(run,i,objects,label),(x0,42))
                center=(x0+128,605)
                # x rightwards, y upwards at 3 px/m; fixed framing on both sides.
                d.line([(x0+12,center[1]),(x0+242,center[1])],fill=170)
                d.line([(center[0],478),(center[0],632)],fill=190)
                points=[(center[0]+x*3,center[1]-y*3) for x,y in zip(run['x'][:i+1:8],run['y'][:i+1:8])]
                end=(center[0]+run['x'][i]*3,center[1]-run['y'][i]*3)
                d.line(points+[end],fill=40,width=2)
                d.ellipse((end[0]-3,end[1]-3,end[0]+3,end[1]+3),fill=0)
                d.text((x0+258,493),f"Course position  {run['y'][i]:.1f} m",font=TITLE,fill=24)
                d.text((x0+258,533),f"Forward command  {run['v'][i]:.2f} m/s",font=SMALL,fill=45)
                d.text((x0+258,572),'Flight ended / last pose' if k>i else 'Flight running',font=SMALL,fill=45)
            d.text((12,624),'Top view: start at axis crossing; forward is up.',font=SMALL,fill=40)
            if k==90*30: im.save(report/'departure-50hz-3d.png')
            enc.add(np.asarray(im.convert('RGB')))
    finally:
        enc.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('traces',type=Path)
    ap.add_argument('choice',choices=NAMES)
    ap.add_argument('--bundle', type=Path, help='save source rows and target trajectories for replay')
    args=ap.parse_args()
    rows,target=load(args.traces)
    if args.bundle:
        with gzip.open(args.bundle, 'wt') as f:
            json.dump(dict(rows=rows,target=target),f,allow_nan=False,separators=(',',':'))
    out=ROOT/'wiki/llm'
    (out/'departure-50hz.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    graph(rows,target,out/'departure-50hz.png')
    movie(target,args.choice,out)
