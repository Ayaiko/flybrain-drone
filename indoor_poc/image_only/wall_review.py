"""Paired image-only controller comparison; world pose is used only by the renderer/scorer."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image, ImageDraw
import image_room as sim

OUT=Path(__file__).resolve().parent
X,Z=np.meshgrid((np.arange(160)+.5-80)/80,(60-np.arange(120)-.5)/80)

def draw_walls(p, pixels):
    side=np.where(X<0,-1.9,1.9)
    depth=(side-p[0])/X
    height=p[2]+Z*depth
    along=p[1]+depth
    visible=(along<=4.8)&(height>0)&(height<2.3)
    seam=(np.mod(along,.5)<.025)|(np.mod(height,.5)<.022)
    pixels[visible]=np.where(seam[visible],187,218)

class ReviewPilot(sim.Pilot):
    def __init__(self):
        super().__init__()
        self.centers=[]
        self.initial_motion=0
        self.steps=0
        self.state='cruise'
        self.look=[]
        self.clear=0

    def command(self,image):
        gray=np.asarray(image)
        mask=gray<115
        xs=np.where(mask[12:36])[1]
        center=float(np.median(xs)) if len(xs) else None
        lateral_target,mode=super().command(image)
        if self.state=='yield':
            self.clear=self.clear+1 if not mask[15:33,26:54].any() else 0
            if self.clear<6:
                return np.zeros(3),'yield / observe'
            self.state='cruise'
            self.steps=self.side=0
            self.centers.clear()
        if self.state=='look':
            self.look.append(center)
            if len(self.look)<14:
                return np.zeros(3),'brake / observe'
            values=[x for x in self.look[5:] if x is not None]
            shift=values[-1]-values[0] if len(values)>1 else 0
            self.state='cruise'
            self.steps=0
            if self.initial_motion*shift < -1:
                self.state='yield'
                self.clear=0
                return np.zeros(3),'reversal / yield'
            self.side=0
        if mode=='sidestep':
            if self.steps==0 and len(self.centers)>=3:
                self.initial_motion=np.sign(self.centers[-1]-self.centers[0])
            self.steps+=1
            edge=gray[:,-16:] if lateral_target[0]>0 else gray[:,:16]
            wall_close=np.mean(edge>200)>.88
            if self.steps>=10 or wall_close and self.steps>=5:
                self.state='look'
                self.look=[]
                return np.zeros(3),'brake / observe'
        elif center is not None and self.steps==0:
            self.centers=(self.centers+[center])[-18:]
        return lateral_target,mode

def compare_gif(seed, runs):
    with TemporaryDirectory() as tmp:
        imgs=[]
        for k,(r,h) in enumerate(runs):
            path=Path(tmp)/f'{k}.gif'
            sim.replay(r,h,path)
            imgs.append((Image.open(path),sorted(set(range(0,len(h),3))|{len(h)-1})))
        frames=[]
        for tick in range(0,max(len(h) for r,h in runs)+3,3):
            canvas=Image.new('RGB',(1220,327),'white')
            pen=ImageDraw.Draw(canvas)
            for k,(gif,times) in enumerate(imgs):
                index=int(np.searchsorted(times,tick,side='right'))-1
                gif.seek(min(gif.n_frames-1,max(index,0)))
                canvas.paste(gif.convert('RGB'),(610*k,27))
                pen.text((610*k+8,6),['Original pilot | walls visible','Revised pilot | walls visible'][k],fill='black')
            frames.append(canvas)
        frames[0].save(OUT/f'wall-pair-{seed}.gif',save_all=True,append_images=frames[1:],duration=120,loop=0)

if __name__=='__main__':
    rows=[]
    for scene in sim.SCENES:
        for seed in (list(range(12))+list(range(80,92)) if scene=="walk_reverse" else range(12)):
            examples=[]
            for variant,background,pilot in [('original',None,sim.Pilot),('walls',draw_walls,sim.Pilot),('review',draw_walls,ReviewPilot)]:
                row,h=sim.simulate(scene,seed,background=background,pilot_factory=pilot)
                row['variant']=variant
                for label,action in [('brake','brake / observe'),('reversal','reversal / yield')]:
                    row[label+'_ticks']=[i for i,f in enumerate(h) if f[3]==action and (i==0 or h[i-1][3]!=action)]
                rows.append(row)
                if variant!='original':
                    examples.append((row,h))
            if scene=='walk_reverse' and seed in (3,8):
                compare_gif(seed,examples)
            if scene=='walk_reverse':
                print(seed,row['outcome'],row['event_tick'],row['brake_ticks'][:3],row['reversal_ticks'],flush=True)
    (OUT/'wall-review-results.json').write_text(json.dumps(rows,indent=2))
