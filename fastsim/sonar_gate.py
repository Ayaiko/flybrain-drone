"""Paired stationary sonar / gyro bench scans of a 1.30 m gate."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from px4.forest_pilot import corridor_depth

FPS = 15
DT = 1 / FPS
N = 181
TIMES = np.arange(N)*DT
TRUE = np.deg2rad(-180 + 30*TIMES)  # zero is the centreline, positive toward +x
WIDTH = 1.30
GATE_Y = 1.4
BRANCH_Y = 1.55
RADIUS = .01
BACK = 3.4
LIMIT = 4.0
BIN = np.pi/18
PROBS = (0., .03, .15, .60, 1.)
SEEDS = 200


def scene_ranges(half_angle, width=WIDTH, gate_y=GATE_Y):
    # Horizontal slice at the common sensor/branch height. All objects span that slice.
    beam = TRUE[:,None] + np.deg2rad(np.linspace(-half_angle,half_angle,65))
    dx,dy = np.sin(beam), np.cos(beam)
    dx = np.where(abs(dx)<1e-12,1e-12,dx)
    dy = np.where(abs(dy)<1e-12,1e-12,dy)
    def slab(x0,x1,y0,y1):
        xlo,xhi = sorted_ranges(x0/dx,x1/dx)
        ylo,yhi = sorted_ranges(y0/dy,y1/dy)
        near,far = np.maximum(xlo,ylo),np.minimum(xhi,yhi)
        return np.where((near>=.03)&(near<=far)&(near<=LIMIT),near,np.inf)
    left=slab(-width/2-1,-width/2,gate_y,gate_y+.5)
    right=slab(width/2,width/2+1,gate_y,gate_y+.5)
    back=slab(-3,3,BACK,BACK+.05)
    # Side view of a horizontal round branch on its central horizontal plane.
    branch=slab(-width/2,width/2,gate_y+.15-RADIUS,gate_y+.15+RADIUS)
    solid=np.minimum(np.minimum(left,right),back)
    return solid.min(1), np.where(branch<solid,branch,np.inf).min(1)


def sorted_ranges(a,b):
    return np.minimum(a,b),np.maximum(a,b)


def noise(seed):
    rng=np.random.default_rng(seed)
    bias=rng.uniform(-.4,.4)
    # Integrated relative yaw. Starting yaw is set by aiming at a marked centreline.
    gyro_hz=120
    rates=30+bias+rng.normal(0,.35,(N-1)*gyro_hz//FPS)
    angles=-180+np.r_[0.,np.cumsum(rates/gyro_hz)[gyro_hz//FPS-1::gyro_hz//FPS]]
    return np.deg2rad(angles),rng.random(N),rng.random(N),rng.normal(0,.012,N)


def decide(angles,ranges):
    bins=np.full(36,np.inf)
    seen=np.full(36,-np.inf)
    slots=np.rint(angles/BIN).astype(int)%36
    valid=np.isfinite(ranges)
    np.minimum.at(bins,slots[valid],ranges[valid])
    np.maximum.at(seen,slots[valid],TIMES[valid])
    fresh=TIMES[-1]-seen<15
    depth=float(corridor_depth(bins,fresh,BIN)[0])
    # Missing coverage gets its own outcome; a measured short return can veto immediately.
    max_span=int(np.ceil(np.arctan2(.45,.8)/BIN))
    required=range(-max_span,max_span+1)
    known_short=any(fresh[k] and bins[k]<=1.4 for k in range(-2,3))
    for plane in (.8,1.4,2.,2.6):
        span=int(np.ceil(np.arctan2(.45,plane)/BIN))
        for k in range(-span,span+1):
            if fresh[k] and bins[k]*np.cos(k*BIN)<plane:
                known_short=True
    if known_short:
        return 'HOLD',depth
    if not all(fresh[k] for k in required):
        return 'UNKNOWN',depth
    return ('PASS' if depth>=2.6 else 'UNKNOWN'),depth


def flight(seed, geometry, p=None):
    angles,detect,drop,jitter=noise(seed)
    solid,branch=geometry
    branch_echo=np.zeros(N,dtype=bool) if p is None else (detect<p)&(branch<solid)
    ranges=np.where(branch_echo,branch,solid)+jitter
    ranges[(drop<.025)|(ranges>LIMIT)]=np.inf
    decision,depth=decide(angles,ranges)
    return decision, depth, angles, ranges, branch_echo & np.isfinite(ranges)


def main():
    out=ROOT/'wiki/llm'
    rows=[]
    for width in (1.30,2.20,2.60):
        for half in (5,8,12):
            geom=scene_ranges(half,width)
            for seed in range(SEEDS):
                for p in (None,*PROBS):
                    action,depth,angle,ranges,echo=flight(seed,geom,p)
                    rows.append(dict(width_m=width,half_angle_deg=half,seed=seed,scene='clear' if p is None else 'branch',
                                     p_echo=p,action=action,supported_m=depth,branch_returns=int(echo.sum()),
                                     yaw_error_deg=round(float(np.rad2deg(angle[-1]-TRUE[-1])),3)))
    with (out/'sonar-gate-trials.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    summary=[]
    for width in (1.30,2.20,2.60):
        for half in (5,8,12):
            for p in (None,*PROBS):
                selected=[r for r in rows if r['width_m']==width and r['half_angle_deg']==half and r['p_echo']==p]
                summary.append(dict(width=width,half=half,p=p,**{k:sum(r['action']==k for r in selected) for k in ('PASS','HOLD','UNKNOWN')}))
    (out/'sonar-gate-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    print(main())
