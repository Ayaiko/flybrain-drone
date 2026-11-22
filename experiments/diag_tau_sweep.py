"""How well does each cue track true nearness during straight flight? Sweep the EMD delay time constant."""
import numpy as np
from flydrone import sim
from flydrone.controllers import FlyVision, EMD
def run(tau, seeds=range(6)):
    rows=[]
    for sd in seeds:
        log=[]
        class P(FlyVision):
            def __init__(s,*a,**k):super().__init__(*a,**k);s.emd=EMD(len(s.eye),s.dt,tau_lp=tau);s.prev=None
            def act(s,lum,dist=None):
                r=s.emd.step(lum);out=np.maximum(r*s.outward,0)
                dI=np.zeros_like(lum) if s.prev is None else np.abs(lum-s.prev); s.prev=lum.copy()
                fr=np.abs(s.eye)<np.radians(35)
                log.append((out[s.left].sum(),out[s.right].sum(),out[s.frontm].sum(),dI[fr].mean(),
                    dist[(s.eye>np.radians(10))&(s.eye<np.radians(110))].min(),dist[(s.eye<-np.radians(10))&(s.eye>-np.radians(110))].min(),dist[fr].min()))
                return 0.0,2.0
        sim.run_episode(P,seed=sd,T=30);L=np.array(log)[30:]
        if len(L)>20:rows.append(L)
    L=np.concatenate(rows);nl,nr,nf=[1/np.maximum(L[:,k],.2) for k in (4,5,6)]
    c=lambda a,b:np.corrcoef(a,b)[0,1]
    return c(L[:,0],nl),c(L[:,1],nr),c(L[:,2],nf),c(L[:,3],nf)
print('tau_ms  left   right  front_EMD  front_LGMD(|dI/dt|)')
for tau in (.005,.01,.02,.035):
    print(f'{tau*1000:5.0f}  '+'  '.join(f'{v:6.2f}' for v in run(tau)))
