"""3-D geometry upper bound. Obstacle tracks and room bounds are assumed known."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
DT, SPEED, RADIUS = .1, .2, .10
START = np.array([0., 0., .85])
GOAL = np.array([0., 3.6, .85])
ROOM = np.array([1.9, 2.3])  # half width, ceiling
SCENES = {  # centre, half-extents and constant walking velocity (m, m/s)
    'floor box': ([0,1.8,.5], [.37,.22,.5], [0,0,0]),
    'wide floor box': ([0,1.8,.46], [1.8,.22,.46], [0,0,0]),
    'suspended panel': ([0,1.8,1.4], [1.8,.22,.67], [0,0,0]),
    'person crossing': ([-2.25,1.8,.85], [.25,.18,.85], [.25,0,0]),
    'person walking ahead': ([0,1.65,.85], [.25,.18,.85], [0,.025,0]),
}

def gap(points, centers, half):
    return np.linalg.norm(np.maximum(np.abs(points-centers)-half, 0), axis=-1)-RADIUS

def plan(p, center, half, walk, human):
    margin = .55 if human else .18
    routes = [('straight', [GOAL])]
    for name, offset in [('left', [-1.2,0,0]), ('right',[1.2,0,0]),
                         ('above',[0,0,.62]), ('below',[0,0,-.50])]:
        if human and name in ('above','below'):
            continue
        offset = np.array(offset)
        routes.append((name, [START + [0,.65,0]+offset, GOAL-[0,.65,0]+offset, GOAL]))
    valid = []
    for name, route in routes:
        ends = np.vstack([p, *route])
        legs = np.diff(ends,axis=0)
        dist = np.r_[0., np.cumsum(np.linalg.norm(legs,axis=1))]
        s = np.linspace(0,dist[-1],int(dist[-1]/.025)+2)
        pts = np.stack([np.interp(s,dist,ends[:,i]) for i in range(3)],axis=1)
        if np.any(np.abs(pts[:,0])+RADIUS > ROOM[0]) or np.any(pts[:,2]<RADIUS+.12) or np.any(pts[:,2]+RADIUS>ROOM[1]):
            continue
        if np.min(gap(pts, center + s[:,None]/SPEED*walk, half)) >= margin:
            valid.append((dist[-1] + (2.0 if name in ('above','below') else 0.0), name, route))
        elif name == 'straight' and human and abs(walk[0])>.1:
            return 'yield', []
    if not valid:
        return 'hold', []
    _, name, route = min(valid,key=lambda item:item[0])
    return name, list(route)

def simulate(scene, seed, tracks=True):
    origin, half, walk = map(np.array, SCENES[scene])
    human = scene.startswith('person')
    rng = np.random.default_rng(seed)
    p, v = START.copy(), np.zeros(3)
    route, history, modes = [], [], set()
    mingap, last_seen = np.inf, -10
    held = 0
    for i in range(500):
        t = i*DT
        center = origin + t*walk
        if tracks and i%2==0 and rng.random()>.08:
            age = rng.uniform(.08,.18)
            measured = center-age*walk+rng.normal(0,.018,3)
            last_seen=t-age
        fresh = t-last_seen<=.45
        if fresh and not route:
            mode,route = plan(p, measured+(t-last_seen)*walk, half, walk, human)
            modes.add(mode)
        if fresh and route and np.linalg.norm(route[0]-p)<.035:
            route.pop(0)
        target = np.zeros(3)
        if fresh and route:
            delta = route[0]-p
            target = delta * min(1.6,SPEED/max(np.linalg.norm(delta),1e-8))
        else:
            held += 1
        step = target-v
        step *= min(.35,.055/max(np.linalg.norm(step),1e-8))
        p += DT*(v+step/2)
        v += step
        center = origin+(t+DT)*walk
        clearance = float(gap(p,center,half))
        mingap = min(mingap,clearance)
        history.append((p.copy(),center.copy()))
        if clearance<=0 or np.linalg.norm(p-GOAL)<.04:
            break
    outcome = 'contact' if mingap<=0 else 'goal' if np.linalg.norm(p-GOAL)<.04 else 'held/timeout'
    return dict(scene=scene,seed=seed,tracks=tracks,outcome=outcome,min_gap_m=round(mingap,3),
                modes=sorted(modes),hold_seconds=round(held*DT,1)), history

def replay(paths):
    frames=[]
    for i in range(0,max(map(len,paths.values())),5):
        frame=Image.new('RGB',(680,455),'white')
        draw=ImageDraw.Draw(frame)
        draw.text((12,4),f'Geometry reference | left: top view / right: side view | {i*DT:.1f} s',fill='black')
        for row,(scene,hist) in enumerate(paths.items()):
            idx=min(i,len(hist)-1)
            p,c=hist[idx]
            half=np.array(SCENES[scene][1])
            y=32+row*84
            draw.text((6,y+26),scene,fill='black')
            for col,axis in enumerate((0,2)):
                x=155+col*260
                def uv(point):
                    height = point[axis] if axis==2 else point[axis]+1.6
                    return (x+point[1]*58,y+65-height*20)
                a,b=uv(c-half),uv(c+half)
                draw.rectangle((min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])),fill='gray')
                line=[uv(h[0]) for h in hist[:idx+1:2]]
                if len(line)>1:
                    draw.line(line,fill='black',width=2)
                px,py=uv(p)
                draw.ellipse((px-3,py-3,px+3,py+3),fill='black')
                draw.line((x,y+66,x+215,y+66),fill='silver')
        frames.append(frame)
    frames[0].save(OUT/'room-replay.gif', save_all=True,append_images=frames[1:],duration=100,loop=0)

if __name__=='__main__':
    rows,paths=[],{}
    for scene in SCENES:
        for seed in range(12):
            row,hist=simulate(scene,seed)
            rows.append(row)
            if seed==0:
                paths[scene]=hist
        print(scene, Counter(r['outcome'] for r in rows[-12:]), rows[-1]['modes'])
    row,_=simulate('person crossing',0,False)
    rows.append(row)
    (OUT/'room-results.json').write_text(json.dumps(rows,indent=2))
    replay(paths)
