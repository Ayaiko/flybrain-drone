"""Reproduce and review the hard-0 brake-margin experiment in fastsim."""
import inspect
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastsim.runner import run_group
from viz.common import Encoder
from viz.replay2d import forest_raster

BASE = 'scan-front'
FIX = 'scan-front-margin'
REPORT = ROOT / 'wiki/llm'
COLORS = ['#af542d', '#087b77']


def contact_rows(run):
    result = []
    for hit in run['crashes']:
        i = hit['i']
        start = max(0, i-15)
        speed = 30*np.hypot(run['x'][i]-run['x'][start], run['y'][i]-run['y'][start])/max(1,i-start)
        result.append(dict(hit, tof_m=run['tof'][i], command_m_s=run['v'][i],
                           displacement_speed_05s=round(float(speed),3), heading_deg=round(float(np.degrees(run['h'][i])),1)))
    return result


def graph(flights):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for f, col in zip(flights,COLORS):
        run, s = f['run'], f['summary']
        t = np.arange(len(run['y']))/30
        hits = run['crashes']
        axes[0].plot(t,run['y'],color=col,label=s['config'])
        axes[0].scatter([h['t'] for h in hits],[h['y'] for h in hits],marker='x',color=col,zorder=3)
        axes[1].step([0,*[h['t'] for h in hits],t[-1]],[0,*range(1,len(hits)+1),len(hits)],where='post',color=col)
    axes[0].legend(frameon=False)
    axes[0].set_ylabel('Position along course (m)')
    axes[1].set_ylabel('Cumulative contacts')
    axes[1].set_xlabel('Simulation time (s)')
    for ax in axes:
        ax.grid(axis='y',alpha=.23)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Hard forest 0  /  start 1',weight='bold')
    fig.text(.12,.91,'Full flights at matched simulation time. Crosses mark first contact after separation.')
    fig.subplots_adjust(top=.84,bottom=.12,hspace=.13)
    fig.savefig(REPORT/'hard0-braking.png',dpi=160)
    plt.close(fig)


def movie(flights):
    meta=json.loads((ROOT/'px4/forest/ff_hard_0.json').read_text())
    raster,extent=forest_raster(meta)
    fig,axes=plt.subplots(1,2,figsize=(10,5.6),dpi=100)
    fig.subplots_adjust(left=.07,right=.98,top=.79,bottom=.14)
    fig.suptitle('Hard forest 0  /  start 1',x=.07,ha='left',weight='bold',fontsize=19)
    clock=fig.text(.07,.87,'',fontsize=12)
    fig.text(.07,.035,'Top view in metres  /  arrow = heading  /  cross = contact  /  each camera follows its aircraft')
    artists=[]
    for ax,f,col in zip(axes,flights,COLORS):
        ax.imshow(raster,extent=extent,origin='lower',interpolation='nearest')
        ax.set_title(f['summary']['config'],loc='left',weight='bold')
        path,=ax.plot([],[],color=col,lw=2)
        arrow=ax.quiver([0.],[0.],[1.],[0.],angles='xy',scale_units='xy',scale=.8,color=col,width=.018,zorder=6)
        marks=ax.scatter([],[],marker='x',color='#a02c24',s=64,zorder=5)
        label=ax.text(.02,.02,'',transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=.95,edgecolor='none'))
        artists.append((path,arrow,marks,label))
    first=flights[0]['run']['crashes'][0]['i']
    end=max(len(f['run']['x']) for f in flights)-1
    enc=Encoder(ROOT/'results/videos/hard0-braking.mp4',1000,560,15,channels=4)
    try:
        k=0
        while True:
            slow=first-120<=k<=first+210
            clock.set_text(f'Simulation {k/30:.1f} s   /   playback {2 if slow else 16}x')
            for ax,f,(path,arrow,marks,label) in zip(axes,flights,artists):
                r=f['run']
                i=min(k,len(r['x'])-1)
                x,y,h=r['x'][i],r['y'][i],r['h'][i]
                hits=[(c['x'],c['y']) for c in r['crashes'] if c['i']<=i]
                path.set_data(r['x'][max(0,i-600):i+1],r['y'][max(0,i-600):i+1])
                arrow.set_offsets([[x,y]])
                arrow.set_UVC([np.cos(h)],[np.sin(h)])
                marks.set_offsets(np.array(hits).reshape(-1,2))
                ax.set_xlim(x-4,x+4)
                ax.set_ylim(y-4,y+4)
                phase=r['phase'][i] if i==k else f['summary']['end_reason']
                label.set_text(f"{phase}   /   y {y:.1f} m\nToF {r['tof'][i]:.2f} m   /   contacts {len(hits)}")
            fig.canvas.draw()
            enc.add(np.asarray(fig.canvas.buffer_rgba()))
            if k==end:
                break
            k=min(end,k+(4 if slow else 32))
    finally:
        enc.close()
        plt.close(fig)


if __name__=='__main__':
    timing = {'pose_tick': 0} if 'pose_tick' in inspect.signature(run_group).parameters else {}
    flights=run_group('ff_hard_0',[dict(config=c,rep=1,duration=600) for c in (BASE,FIX)],record=True,**timing)
    records=[dict(summary=f['summary'],contacts=contact_rows(f['run'])) for f in flights]
    (REPORT/'hard0-braking.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    graph(flights)
    movie(flights)
    from viz.hard_contact_3d import movie3d
    movie3d(flights,json.loads((ROOT/'px4/forest/ff_hard_0.json').read_text()),ROOT)
