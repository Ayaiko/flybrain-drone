"""Render paired seed 302 traces and the holdout comparison."""
import csv
from pathlib import Path
from bisect import bisect_right
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent
def rows(name):
    with (out/name).open() as stream:
        return list(csv.DictReader(stream))

results=rows('validation.csv')
fig,ax=plt.subplots(figsize=(6,3.3),layout='constrained')
for j,(v,col) in enumerate([('before','#969fa7'),('after','#126c61')]):
    counts=[sum(r['result']=='finish' for r in results if r['version']==v and int(r['loss_pct'])==loss) for loss in (5,18)]
    bars=ax.bar([i+(j-.5)*.33 for i in (0,1)],counts,.32,label=v.title(),color=col)
    ax.bar_label(bars,labels=[f'{n}/64' for n in counts])
ax.set(xticks=[0,1],xticklabels=['5% thrust loss','18% thrust loss'],ylim=(0,78),ylabel='Completed missions',title='62 g quad | 18 gf per motor at start')
ax.legend(frameon=False);fig.savefig(out/'finish.png',dpi=170)
stages=['Takeoff / hover','Approach','Brake','Sidestep','Pass']
for loss in (5,18):
    traces=[rows(f'trace-{loss}-{j}.csv') for j in range(2)]
    times=[[float(r['time']) for r in tr] for tr in traces]
    frames=[]
    end=max(ts[-1] for ts in times)
    timeline=[i*.8 for i in range(int(min(40,end)/.8)+1)]
    timeline.extend(40+i*.16 for i in range(1,int(max(0,end-40)/.16)+1))
    for t in timeline+[end]*8:
        im=Image.new('RGB',(880,320),'#fafafa');d=ImageDraw.Draw(im)
        for j,tr in enumerate(traces):
            def pt(x,y,z):
                return (j*440+122+x*110+y*75,243+x*19-y*34-z*112)
            def path(coords,color,width=1): d.line([pt(*q) for q in coords],fill=color,width=width)
            for y in (0,.5,1,1.5,2,2.5):path([(-1,y,0),(1,y,0)],'#cccccc')
            for x in (-1,-.5,0,.5,1):path([(x,0,0),(x,2.5,0)],'#cccccc')
            path([(-1,2.15,0),(1,2.15,0)],'#292929',3)
            for z in (0,1.1):
                path([(x,y,z) for x,y in [(-.19,1.18),(.19,1.18),(.19,1.6),(-.19,1.6),(-.19,1.18)]],'#777777')
            for x in (-.19,.19):
                for y in (1.18,1.6):path([(x,y,0),(x,y,1.1)],'#777777')
            idx=bisect_right(times[j],t)-1;r=tr[idx]
            coords=[tuple(float(row[k]) for k in ('x','y','z')) for row in tr[:idx+1]]
            if len(coords)>1:path(coords,'#353535',2)
            x,y,z=coords[-1]
            path([(x,y,0),(x,y,z)],'#aaaaaa')
            px,py=pt(x,y,z);d.ellipse([px-4,py-4,px+4,py+4],fill='black')
            d.text((j*440+14,15),f"{['BEFORE','AFTER'][j]}   |   loss {loss}%   |   seed 302",fill='black')
            label=r['status'] if idx==len(tr)-1 else stages[int(r['stage'])]
            d.text((j*440+14,280),f"{float(r['time']):.1f} s    {label}    height {z:.2f} m",fill='black')
        frames.append(im)
    frames[0].save(out/f'replay-{loss}.gif',save_all=True,append_images=frames[1:],duration=95,loop=0)
    with Image.open(out/f'replay-{loss}.gif') as check: print(loss,check.size,check.n_frames)
