"""Pixel-only reactive indoor benchmark. AABB geometry is private to renderer/scorer."""
import io
import json
from pathlib import Path
from itertools import product

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
DT, R = .1, .10
SCENES = ('walk_stop', 'walk_reverse', 'over_box', 'under_panel')
W,H=160,120
corners=np.array(list(product((-1,1), repeat=3)))
faces=[[i for i,c in enumerate(corners) if c[axis]==sign] for axis in range(3) for sign in (-1,1)]

def objects(scene, cx):
    if scene=='over_box':
        return [(np.array([0,1.8,.51]),np.array([1.8,.24,.51]),72)]
    if scene=='under_panel':
        return [(np.array([0,1.8,1.55]),np.array([1.8,.24,.75]),72)]
    return [(np.array([cx+x,1.8,z]), np.array(half),tone) for x,z,half,tone in
            [(0,1.55,(.13,.13,.15),91),(0,1.04,(.24,.16,.36),64),
             (-.12,.34,(.09,.12,.34),79),(.12,.34,(.09,.12,.34),79)]]

def capture(p, solids, noise, light, background=None):
    # Fixed level forward pinhole, front-facing rectangles projected into QQVGA.
    # With fixed viewing axes, the complete projected box is the hull of its corners.
    pixels=np.full((H,W),180,dtype=np.float32)
    pixels[H//2:]=156
    if background is not None:
        background(p, pixels)
    image=Image.fromarray(pixels.astype('uint8'))
    draw=ImageDraw.Draw(image)
    for c,h,tone in sorted(solids,key=lambda s:-s[0][1]):
        points=c+h*corners-p
        if points[:,1].max()<=.01:
            continue
        points[:,1]=points[:,1].clip(.01)
        uv=np.column_stack((80+80*points[:,0]/points[:,1],60-80*points[:,2]/points[:,1]))
        for a,b,c1,d in faces:
            draw.polygon([tuple(uv[j]) for j in (a,b,d,c1)],fill=tone)
    noisy=(np.asarray(image)+light+noise.normal(0,4,(H,W))).clip(0,255).astype('uint8')
    buf=io.BytesIO()
    Image.fromarray(noisy).save(buf,format='JPEG',quality=50)
    buf.seek(0)
    with Image.open(buf) as jpeg:
        return jpeg.crop((0,12,160,108)).resize((80,48),Image.Resampling.BOX)

class Pilot:
    def __init__(self):
        self.side=0

    def command(self,image):
        mask=np.asarray(image)<115
        strip=mask[19:29,32:48]
        if np.mean(strip)<.09:
            return np.array([0,.20,0]),'forward'
        ys,xs=np.where(mask)
        span=xs.max()-xs.min()+1
        if span>=57:
            top=np.mean(mask[2:15,30:50])
            bottom=np.mean(mask[33:46,30:50])
            if abs(top-bottom)<.12:
                return np.zeros(3),'hold'
            up=top<bottom
            return np.array([0,.035,.20 if up else -.20]),'climb' if up else 'descend'
        if span<16:
            return np.array([0,.20,0]),'forward'
        if not self.side:
            self.side=-1 if xs.mean()>=39.5 else 1
        return np.array([self.side*.24,.085,0]),'sidestep'

def simulate(scene,seed, *, background=None, pilot_factory=Pilot, camera=capture, reversal_tick=None):
    rng=np.random.default_rng(seed)
    noise=np.random.default_rng([seed,111])
    p=np.array([rng.uniform(-.06,.06),0.,.85+rng.uniform(-.04,.04)])
    vel=np.zeros(3)
    cx=rng.uniform(-.65,-.47)
    walk=rng.uniform(.15,.27)
    light=rng.uniform(-12,12)
    event_delay=int(rng.integers(2,9))
    event=first=None
    min_gap=4.
    object_gap=np.inf
    contact=None
    min_z,max_z=p[2],p[2]
    plane=[]
    history=[]
    pilot=pilot_factory()
    queue=[]
    image=Image.new('L',(80,48),155)
    stamp=-100
    target=np.zeros(3)
    mode='no_frame'
    for i in range(440):
        solids=objects(scene,cx)
        if rng.random()>.06:
            queue.append((i+int(rng.integers(1,3)),i,camera(p,solids,noise,light,background)))
        ready=[q for q in queue if q[0]<=i and q[1]>stamp]
        queue=[q for q in queue if q[0]>i]
        if ready:
            _,stamp,image=max(ready,key=lambda q:q[1])
            target,mode=pilot.command(image)
        if i-stamp>2:
            target,mode=np.zeros(3),'stale/hold'
        if scene.startswith('walk') and first is None and mode=='sidestep' and abs(vel[0])>.045:
            first=i
        if i==reversal_tick or (reversal_tick is None and first is not None and i==first+event_delay):
            walk=0 if scene=='walk_stop' else -walk
            event=i
        for _ in range(5):
            dv=target-vel
            vel+=dv*min(.07,.011/max(np.linalg.norm(dv),1e-9))
            p+=vel*.02
            cx+=walk*.02
            solids=objects(scene,cx)
            dist=min(np.linalg.norm(np.maximum(abs(p-c)-h,0))-R for c,h,_ in solids)
            object_gap=min(object_gap,dist)
            boundaries={'obstacle':dist,'floor':p[2]-R,'ceiling':2.3-p[2]-R,'side wall':1.9-abs(p[0])-R}
            nearest=min(boundaries,key=boundaries.get)
            min_gap=min(min_gap,boundaries[nearest])
            if min_gap<=0:
                contact=nearest
            if 1.56<=p[1]<=2.04:
                plane.append(float(p[2]))
            if min_gap<=0:
                break
        min_z,max_z=min(min_z,p[2]),max(max_z,p[2])
        history.append((p.copy(),cx,image,mode,stamp))
        if min_gap<=0 or p[1]>=3.6:
            break
    outcome='contact' if min_gap<=0 else 'finish' if p[1]>=3.6 else 'timeout'
    return dict(scene=scene,seed=seed,outcome=outcome,min_gap_m=round(float(min_gap),3),
                first_sidestep_tick=first,event_tick=event,last_tick=i,contact=contact,
                object_gap_m=round(float(object_gap),3),
                obstacle_z_range=[round(min(plane),3),round(max(plane),3)] if plane else None),history

def view(pt):
    x,y,z=pt
    return (int(132+x*51+y*40),int(220+x*15-y*18-z*63))

def replay(row,history,filename, *, shadow=False, pale_shirt=False):
    frames=[]
    for i in sorted(set(range(0,len(history),3))|{len(history)-1}):
        p,cx,img,mode,stamp=history[i]
        canvas=Image.new('RGB',(610,300),'white')
        d=ImageDraw.Draw(canvas)
        for x in (-1.9,-1,0,1,1.9):
            d.line([view((x,0,0)),view((x,3.6,0))],fill='#aaaaaa')
            if abs(x)==1.9:
                d.line([view((x,0,0)),view((x,0,2.3)),view((x,3.6,2.3)),view((x,3.6,0))],fill='#cccccc')
        for y in (0,1,2,3,3.6):
            d.line([view((-1.9,y,0)),view((1.9,y,0))],fill='#aaaaaa')
        if shadow:
            outline=[]
            for angle in np.linspace(0,2*np.pi,41):
                along=.45+.78*np.cos(angle)
                across=.30*np.sin(angle)
                outline.append(view((cx+.8*along+.6*across,1.8-.6*along+.8*across,0)))
            d.polygon(outline,fill='#999999')
        # Painter order from the camera at positive x, negative y, positive z.
        drawlist=[]
        for c,h,tone in objects(row['scene'],cx):
            verts=c+corners*h
            for axis,sign,shade in [(0,1,155),(1,-1,125),(2,1,190)]:
                ids=[j for j,v in enumerate(corners) if v[axis]==sign]
                pts=[view(verts[j]) for j in ids]
                mid=np.mean(pts,axis=0)
                pts.sort(key=lambda v:np.arctan2(v[1]-mid[1],v[0]-mid[0]))
                depth=np.dot(verts[ids].mean(axis=0),[1,-1,1])
                drawlist.append((depth,pts, min(250,shade+67) if pale_shirt and tone==64 else shade))
        for _,pts,shade in sorted(drawlist,key=lambda item:item[0]):
            d.polygon(pts,fill=(shade,)*3,outline='black')
        line=[view(h[0]) for h in history[:i+1:2]]
        if len(line)>1:
            d.line(line,fill='black',width=2)
        u,v=view(p)
        d.ellipse([u-4,v-4,u+4,v+4],fill='white',outline='black',width=2)
        canvas.paste(img.resize((240,144),Image.Resampling.NEAREST),(360,65))
        d.text((8,8),f"{row['scene']}  |  seed {row['seed']}  |  t={i/10:.1f}s",fill='black')
        d.text((360,44),'Last received camera frame',fill='black')
        d.text((360,220),f'Command: {mode}',fill='black')
        if row['event_tick'] is not None and i>=row['event_tick']:
            d.text((12,260),'Pedestrian changed motion',fill='black')
        if i==len(history)-1:
            d.text((360,250),(row['contact'] or row['outcome']).upper(),fill='black')
        frames.append(canvas)
    frames[0].save(filename,save_all=True,append_images=frames[1:],duration=100,loop=0)

if __name__=='__main__':
    rows=[]
    links=[]
    for scene in SCENES:
        runs=[simulate(scene,seed) for seed in range(12)]
        rows.extend(r for r,h in runs)
        # Show a contact if one occurred, then the closest pass otherwise.
        row,hist=min(runs,key=lambda pair:pair[0]['min_gap_m'])
        replay(row,hist,OUT/(scene+'.gif'))
        links.append(f'<h2>{scene}</h2><img src="{scene}.gif" alt="{scene} 3D and camera replay">')
        print(scene,[(r['outcome'],r['event_tick'],r['min_gap_m']) for r,h in runs],flush=True)
    (OUT/'image-room-results.json').write_text(json.dumps(rows,indent=2))
    (OUT/'image-room-replay.html').write_text('<!doctype html><meta charset="utf-8"><title>Image-only 3D replay</title><style>body{font:16px system-ui;margin:2em}img{max-width:100%}</style><h1>Image-only 3D replay</h1>'+''.join(links))
