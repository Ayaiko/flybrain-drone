"""Full-charge paired test. Front ToF commands braking; fresh camera verifies bypass corridor."""
from pathlib import Path
import io
import json
import numpy as np
from PIL import Image, ImageDraw
import voltage_box as sim

OUT=Path(__file__).resolve().parent
U,V=np.meshgrid((np.arange(160)+.5-80)/80,(60-np.arange(120)-.5)/80)

class Camera:
    def __init__(self,seed):
        self.rng=np.random.default_rng([seed,716])
        self.pending=[]
        self.frames=[]
        self.latest=None

    def update(self,tick,p,yaw):
        if tick%2==0 and self.rng.random()>.06:
            dx=np.sin(yaw)+U*np.cos(yaw)
            dy=np.cos(yaw)-U*np.sin(yaw)
            base=np.where(V<0,156.,180.)
            l,r,a,b,z=sim.BOX
            near=np.full(U.shape,.005)
            far=np.full(U.shape,10.)
            for vec,orig,lo,hi in [(dx,p[0],l,r),(dy,p[1],a,b),(V,p[2],0,z)]:
                safe=np.where(abs(vec)<1e-10,1e-10,vec)
                t1,t2=(lo-orig)/safe,(hi-orig)/safe
                near=np.maximum(near,np.minimum(t1,t2))
                far=np.minimum(far,np.maximum(t1,t2))
            base[near<=far]=72.
            pix=(base+self.rng.normal(0,4,base.shape)).clip(0,255).astype('uint8')
            buf=io.BytesIO()
            Image.fromarray(pix).save(buf,format='JPEG',quality=50)
            buf.seek(0)
            with Image.open(buf) as raw:
                img=raw.crop((0,12,160,108)).resize((80,48),Image.Resampling.BOX)
            self.pending.append((tick+int(self.rng.integers(2,5)),tick,yaw,img))
        ready=[f for f in self.pending if f[0]<=tick]
        self.pending=[f for f in self.pending if f[0]>tick]
        if ready:self.latest=max(ready+([self.latest] if self.latest else []),key=lambda f:f[1])
        self.frames.append(self.latest[3] if self.latest else None)
        if not self.latest or tick-self.latest[1]>6 or abs(yaw-self.latest[2])>.08:return None
        return bool(np.mean(np.asarray(self.latest[3])[18:30,32:48]<115)<.10)

def fly(m,s):
    camera=Camera(s)
    row,h=sim.trial(m,4.2,2.,s,camera=camera)
    return row,h,camera.frames

def main():
    rows=[fly(m,s)[0] for s in sim.SEEDS for m in (45,48)]
    (OUT/'full-cam-results.json').write_text(json.dumps(rows,indent=2))
    seed=min(rows,key=lambda r:r['min_box_gap_m'])['seed']
    runs=[fly(m,seed) for m in (45,48)]
    frames=[]
    for t in range(0,max(len(h) for _,h,_ in runs)+3,4):
        page=Image.new('RGB',(740,365),'white')
        d=ImageDraw.Draw(page)
        for j,(row,h,images) in enumerate(runs):
            page.paste(sim.panel(h,t,row),(370*j,0))
            img=images[min(t,len(h)-1)]
            if img:page.paste(img.resize((160,96),Image.Resampling.NEAREST),(370*j+8,251))
            d.text((370*j+180,268),'Received OV2640 image',fill='black')
            d.text((370*j+180,293),'ToF: brake / scan',fill='black')
            d.text((370*j+180,313),'Image: clear path check',fill='black')
        frames.append(page)
    frames[0].save(OUT/'full-cam-45-48.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)
    for m in (45,48):
        rs=[r for r in rows if r['mass_g']==m]
        print(m,[r['result'] for r in rs],{k:(min(r[k] for r in rs),max(r[k] for r in rs)) for k in ('stop_gap_m','min_box_gap_m','seconds','braking_travel_m')})
    print('seed',seed)
if __name__=='__main__': main()
