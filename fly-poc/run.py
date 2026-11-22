"""Offline 3-D camera experiment. Run with Python 3 + NumPy and a C compiler."""
import base64
import ctypes
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib

import numpy as np

from planner import Airframe, Command, plan

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'model'
META = json.loads((MODEL / 'flyvis.json').read_text())
N, E = META['N'], META['E']
with (MODEL / 'flyvis.bin').open('rb') as f:
    def take(dtype, size):
        return np.fromfile(f, dtype=dtype, count=size)
    tid, u, v = take('u1', N), take('<i2', N), take('<i2', N)
    bias, tau = take('<f4', N), take('<f4', N)
    ptr, dst, weight = take('<i4', N+1), take('<i4', E), take('<f4', E)
    assert f.read() == b'' and ptr[-1] == E
inputs = np.array(list(json.loads((MODEL / 'flyvis_inputs.json').read_text()).values()))
cols = inputs[0]
hx, hy = u + v * .5, v * math.sqrt(3)/2
W, H = 80, 48
px = np.rint((hx[cols]/30+.5)*(W-1)).astype(int)
py = np.rint((.5-hy[cols]/26)*(H-1)).astype(int).clip(0,H-1)
groups = []
for name in ('L1','L2','T4a','T4b','T5a','T5b'):
    cells = np.flatnonzero(tid == META['types'].index(name))
    for band in (cells[hy[cells] > 0], cells[hy[cells] <= 0]):
        ordered = band[np.argsort(hx[band])]
        groups.extend(np.array_split(ordered, 9))
scratch = tempfile.TemporaryDirectory()
libfile = Path(scratch.name)/'step.so'
subprocess.run(['cc','-shared','-fPIC','-O3',str(ROOT/'step.c'),'-o',str(libfile)], check=True)
kernel = ctypes.CDLL(str(libfile)).step
kernel.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*8
gain = (.02/np.maximum(.02,tau)).astype('f4')
def read_eye(img, state):
    inp = np.zeros(N, 'f4')
    inp[inputs] = img[py,px]/255.
    acc = np.empty_like(inp)
    args = [a.ctypes.data for a in (ptr,dst,weight,bias,gain,inp,state,acc)]
    for _ in range(5):  # camera at 10 Hz, model integration at 50 Hz
        kernel(N,*args)
    return [state[g].mean() for g in groups]

# Pinhole camera faces positive y, at fixed height 0.85 m. Nearest ray hit wins.
dx, dz = np.meshgrid((np.arange(W)+.5-W/2)/(W/2),(H/2-np.arange(H)-.5)/(W/2))
def frame(box,x,y,rng,obstacle_tone=42):
    return scene_frame([(box,obstacle_tone)] if box else [],x,y,rng)

def scene_frame(obstacles,x,y,rng):
    floor = np.full(dx.shape,np.inf)
    np.divide(-.85,dz,out=floor,where=dz<0)
    ground = np.isfinite(floor)
    img = np.full(dx.shape,158.)
    img[ground] = 82+12*((np.floor(x+dx[ground]*floor[ground])+np.floor(y+floor[ground]))%2)
    nearest = floor.copy()
    for box, obstacle_tone in obstacles:
        lo,hi,front,back,height = box
        a,b = (lo-x)/dx,(hi-x)/dx
        c,d = -.85/dz,(height-.85)/dz
        near = np.maximum.reduce([np.minimum(a,b),np.minimum(c,d),np.full(dx.shape,front-y)])
        far = np.minimum.reduce([np.maximum(a,b),np.maximum(c,d),np.full(dx.shape,back-y)])
        mask = (near>0)&(near<=far)&(near<nearest)
        nearest[mask] = near[mask]
        img[mask] = obstacle_tone+10*((np.floor((x+dx[mask]*near[mask])*5)+np.floor((.85+dz[mask]*near[mask])*5))%2)
    return (img + rng.normal(0,4,img.shape)).clip(0,255).astype('u1')

names = ['Forward','Left + forward','Right + forward','Stop']
def teacher(box,x,y):
    if not box:
        return 0
    lo,hi,front,back,_ = box
    if y > back or front-y>2.9 or not lo-.45<x<hi+.45:
        return 0
    if hi-lo>6:
        return 3
    return 1 if x<(lo+hi)/2 else 2

def layout(kind,rng):
    if kind==0:
        return None
    center = (1 if kind==1 else -1)*rng.uniform(.25,.68)
    width = rng.uniform(.82,1.13)
    lo,hi = (-5.,5.) if kind==3 else (center-width,center+width)
    front = rng.uniform(3.7,4.5)
    return lo,hi,front,front+.65,rng.uniform(1.45,1.95)

def learn():
    rng = np.random.default_rng(217)
    features, labels = [], []
    tone_rng = np.random.default_rng(192)
    for episode in range(240):
        tone = int(tone_rng.uniform(30, 145))
        box = layout(episode%4,rng)
        x,y = rng.uniform(-1.4,1.4),rng.uniform(0,4)
        vx = rng.uniform(-.08,.08)
        state = bias.copy()
        for tick in range(12):
            f = read_eye(frame(box,x,y,rng,tone),state)
            if tick>=5:
                features.append(f)
                labels.append(teacher(box,x,y))
            x += vx
            y += .055
    features = np.asarray(features,dtype=float)
    mean,sd = features.mean(0),features.std(0).clip(.001)
    a = np.column_stack(((features-mean)/sd,np.ones(len(features))))
    labels = np.array(labels)
    weights = 1/np.bincount(labels)[labels]
    coef = np.linalg.solve(a.T@(weights[:,None]*a)+.012*np.eye(a.shape[1]),a.T@(weights[:,None]*np.eye(4)[labels]))
    return mean,sd,coef

def png(img):
    def chunk(tag,data):
        return struct.pack('>I',len(data))+tag+data+struct.pack('>I',zlib.crc32(tag+data))
    raw = b''.join(b'\0'+row.tobytes() for row in img)
    data = b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',W,H,8,0,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
    return base64.b64encode(data).decode()

def evaluate(model,kind,seed,save,obstacle_tone=42,fault=None,course=None,gate_controller=None):
    rng = np.random.default_rng(seed)
    box = layout(kind,rng) if course is None else None
    obstacles = course if course is not None else ([(box,obstacle_tone)] if box else [])
    finish = 13.5 if course is not None else 6.8
    minimum_clearance = None
    hit = None
    state = bias.copy()
    x,y = rng.uniform(-.15,.15),0.
    frames,stops = [],0
    mean,sd,coef = model
    vehicle = Airframe()
    velocity = np.zeros(2)
    dt = .1
    for tick in range(350 if course is not None else 200):
        now = tick * dt
        if fault != 'freeze' or tick < 25:
            img = scene_frame(obstacles,x,y,rng)
            feature = read_eye(img,state)
            scores = np.r_[(feature-mean)/sd,1.]@coef
            captured_at = now
        # Optional single forward beam, limited to 4 m. Empty space returns max range.
        distance = None
        if fault == 'range':
            distance = 4.
            if box and box[0] <= x <= box[1] and y <= box[2]:
                distance = min(distance,box[2]-y)
            # Fault injection: the vision readout incorrectly requests forward flight.
            scores = np.array([1.,0.,0.,0.])
        cmd = plan(scores,captured_at,now,velocity,vehicle,distance)
        if gate_controller is not None:
            cmd = gate_controller(img,x,y)
        if tick < 6:
            cmd = Command(0,0,'รอภาพเริ่มต้น')
        if save:
            frames.append([round(x,2),round(y,2),names[int(np.argmax(scores))],png(img),
                           cmd.reason,round(cmd.lateral,2),round(cmd.forward,2),
                           round(float(np.linalg.norm(velocity)),2),round(now-captured_at,2)])
        target = np.array([cmd.lateral,cmd.forward])
        delta = target - velocity
        limit = vehicle.brake if not target.any() else vehicle.acceleration
        factor = min(1.,limit*dt/max(np.linalg.norm(delta),1e-9))
        previous = velocity.copy()
        velocity += delta * factor
        x,y = np.array([x,y]) + .5*(previous+velocity)*dt
        stops = stops+1 if tick>=6 and not target.any() and np.linalg.norm(velocity)<.001 else 0
        for index,(shape,_) in enumerate(obstacles):
            lo,hi,front,back,_ = shape
            clearance = math.hypot(max(lo-x,0,x-hi),max(front-y,0,y-back))-vehicle.radius
            minimum_clearance = clearance if minimum_clearance is None else min(clearance,minimum_clearance)
            if clearance <= 0:
                hit = index+1
                break
        if hit is not None:
            result = 'Collision'
            break
        if y>=finish or abs(x)>=3.5 or stops>=12:
            result = 'Goal' if y>=finish else ('Stopped' if stops>=12 else 'Out of bounds')
            break
    else:
        result = 'Timeout'
    return dict(obstacles=obstacles,finish=finish,end_y=round(y,3),end_x=round(x,3),
                hit=hit,min_clearance=minimum_clearance,kind=kind,seed=seed,box=box,frames=frames,result=result,
                obstacle_tone=obstacle_tone,fault=fault,final_reason=cmd.reason,
                final_speed=round(float(np.linalg.norm(velocity)),4))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain',action='store_true')
    args = parser.parse_args()
    if args.retrain or not (ROOT/'readout.npz').exists():
        model = learn()
        np.savez(ROOT/'readout.npz',mean=model[0],sd=model[1],coef=model[2])
    else:
        with np.load(ROOT/'readout.npz') as archive:
            model = tuple(archive[k] for k in ('mean','sd','coef'))
    trials = [evaluate(model,k,7800+4*s+k,s==0) for s in range(5) for k in range(4)]
    harder = [evaluate(model,k,7800+4*s+k,s==0 and k==1,125) for s in range(5) for k in (1,2,3)]
    fresh = [evaluate(model, 1, 9100+s, False, 125) for s in range(10)]
    faults = [evaluate(model,0,7800,True,fault='freeze'),
              evaluate(model,3,7803,True,fault='range')]
    summary = [{k:v for k,v in trial.items() if k!='frames'} for trial in trials+harder+fresh+faults]
    (ROOT/'results.json').write_text(json.dumps(summary,indent=2))
    template = (ROOT/'viewer.template').read_text()
    (ROOT/'replay.html').write_text(template.replace('REPLAY_DATA',json.dumps(trials[:4]+harder[:1]+[json.loads((ROOT/'baseline-light.json').read_text())]+faults)))
    for k in range(4):
        print(k,[t['result'] for t in trials if t['kind']==k], [t['result'] for t in harder if t['kind']==k])
