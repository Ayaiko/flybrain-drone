"""Diagnostic: fly straight (no steering) through a world and log the fly-vision signals vs true distances."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from flydrone import sim
from flydrone.controllers import FlyVision
ROOT = Path(__file__).resolve().parents[1]
log=[]
class Probe(FlyVision):
    def act(self, lum, dist=None):
        r=self.emd.step(lum); out=np.maximum(r*self.outward,0)
        log.append((out[self.left].sum(), out[self.right].sum(), out[self.frontm].sum(),
                    dist[(self.eye>np.radians(10))&(self.eye<np.radians(110))].min(),
                    dist[(self.eye<-np.radians(10))&(self.eye>-np.radians(110))].min(),
                    dist[np.abs(self.eye)<np.radians(35)].min()))
        return 0.0, 2.0
r=sim.run_episode(Probe, seed=3, T=30, record=True)
L=np.array(log); t=np.arange(len(L))*sim.DT
fig,ax=plt.subplots(3,1,figsize=(9,7),sharex=True)
for i,(n,j) in enumerate([('left',0),('right',1),('front/looming',2)]):
    ax[i].plot(t,L[:,j],label=f'{n} outward flow'); a2=ax[i].twinx(); a2.plot(t,1/np.maximum(L[:,3+j],.2),'r',alpha=.6,label='true nearness 1/d'); ax[i].set_ylabel(n)
ax[-1].set_xlabel('s'); fig.suptitle(f'straight flight seed 3, collided={r["collided"]} at {r["time"]:.1f}s'); fig.tight_layout()
fig.savefig(ROOT/'results/diag/diag_signals.png',dpi=90)
m=t>0.6
for j,n in enumerate(['left','right','front']):print(n,'corr(flow, nearness)=',np.corrcoef(L[m,j],1/np.maximum(L[m,3+j],.2))[0,1].round(3))
