"""Paired lighting perturbations. Camera-only frozen pilot, identical pedestrian paths."""
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image, ImageDraw
import image_room as sim
from wall_review import ReviewPilot, draw_walls, X, Z

OUT = Path(__file__).resolve().parent
PROFILES = ('reference','shadow','shirt','both')
SEEDS = [*range(12), *range(80,92)]

def camera_for(profile):
    def render(p, solids, rng, light, background):
        def room(p, pixels):
            background(p, pixels)
            if profile not in ('shadow', 'both'):
                return
            # Oblique soft footprint cast by the pedestrian on the ground plane.
            depth = -p[2] / Z
            x = p[0] + X * depth
            y = p[1] + depth
            cx = solids[0][0][0]
            along = (x-cx)*.8 - (y-1.8)*.6
            across = (x-cx)*.6 + (y-1.8)*.8
            radial = ((along-.45)/.78)**2 + (across/.3)**2
            alpha = np.clip((1.3-radial)/.5,0,1)
            ground = (depth>0)&(abs(x)<1.9)&(y<4.8)
            pixels[ground] *= (1-.55*alpha[ground])
        if profile in ('shirt','both'):
            solids=list(solids)
            c,h,_=solids[1]
            solids[1]=(c,h,210)
        return sim.capture(p,solids,rng,light,room)
    return render

def metrics(row,h):
    e=row['event_tick']
    prev=np.array(h[e-1][0])
    stopped=brake=None
    for i in range(e,len(h)):
        p,_,_,mode,_=h[i]
        if p[1]>=1.54:
            break
        if mode in ('brake / observe','yield / observe','reversal / yield') and brake is None:
            brake=i
        if brake is not None and stopped is None and np.linalg.norm(p-prev)/sim.DT<=.02:
            stopped=i
        prev=p
    row.update(brake_tick=brake, stopped_tick=stopped,
               recognized=any(f[3]=='reversal / yield' for f in h[e:]),
               sidestepped_before_reversal=any(f[3]=='sidestep' for f in h[:e]),
               sidestepped=any(f[3]=='sidestep' for f in h))

def pair(seed, runs):
    with TemporaryDirectory() as tmp:
        clips=[]
        for j,(r,h) in enumerate(runs):
            path=Path(tmp)/f'{j}.gif'
            sim.replay(r,h,path,shadow=j==1,pale_shirt=j==1)
            clips.append((Image.open(path), sorted(set(range(0,len(h),3))|{len(h)-1})))
        frames=[]
        for tick in range(0,max(len(h) for r,h in runs)+3,3):
            canvas=Image.new('RGB',(1220,328),'white')
            d=ImageDraw.Draw(canvas)
            for j,(gif,times) in enumerate(clips):
                idx=int(np.searchsorted(times,tick,side='right'))-1
                gif.seek(min(gif.n_frames-1,max(0,idx)))
                canvas.paste(gif.convert('RGB'),(610*j,28))
                d.text((610*j+8,7),['Original scene','Moving shadow + low-contrast shirt'][j],fill='black')
            frames.append(canvas)
        frames[0].save(OUT/f'lighting-{seed}.gif',save_all=True,append_images=frames[1:],duration=150,loop=0)
        for gif,_ in clips:
            gif.close()

def main():
    archived={r['seed']:r for r in json.loads((OUT/'wall-review-results.json').read_text())
              if r['scene']=='walk_reverse' and r['variant']=='review'}
    rows=[]
    for seed in SEEDS:
        examples=[]
        for profile in PROFILES:
            row,h=sim.simulate('walk_reverse',seed,background=draw_walls,pilot_factory=ReviewPilot,
                               camera=camera_for(profile),reversal_tick=archived[seed]['event_tick'])
            if profile=='reference':
                assert all(row[k]==archived[seed][k] for k in row)
            assert row['event_tick']==archived[seed]['event_tick']
            row['profile']=profile
            metrics(row,h)
            rows.append(row)
            if profile in ('reference','both'):
                examples.append((row,h))
        if seed in (3,8):
            pair(seed,examples)
    # Independent stress check: change direction only once this pilot actually sidesteps.
    triggered=[]
    for seed in SEEDS:
        row,h=sim.simulate('walk_reverse',seed,background=draw_walls,pilot_factory=ReviewPilot,
                           camera=camera_for('both'))
        if row['event_tick'] is not None:
            metrics(row,h)
        triggered.append(row)
    (OUT/'lighting-triggered.json').write_text(json.dumps(triggered,indent=2))
    (OUT/'lighting-results.json').write_text(json.dumps(rows,indent=2))
    for profile in PROFILES:
        rs=[r for r in rows if r['profile']==profile]
        print(profile,Counter(r['outcome'] for r in rs),Counter(r['contact'] for r in rs),
              'stopped',sum(r['stopped_tick'] is not None for r in rs),
              'recognized',sum(r['recognized'] for r in rs),'sidestep',sum(r['sidestepped'] for r in rs),
              'before',sum(r['sidestepped_before_reversal'] for r in rs),flush=True)

if __name__=='__main__':
    main()
