"""Render paired bench scenes and plots from the sonar/gyro scan model."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastsim import sonar_gate as sim
from viz.common import Encoder
from viz.hard_contact_3d import tube

W,H=640,448
FONT=font_manager.findfont('DejaVu Sans')
TITLE=ImageFont.truetype(FONT,24)
SMALL=ImageFont.truetype(FONT,17)
MONO=ImageFont.truetype(font_manager.findfont('DejaVu Sans Mono'),15)
RIGHT=np.array([.94,.34,0.])
RIGHT/=np.linalg.norm(RIGHT)
VIEW=np.array([-RIGHT[1]*.82,RIGHT[0]*.82,-.57])
VIEW/=np.linalg.norm(VIEW)
UP=-np.cross(VIEW,RIGHT)
BASIS=np.stack([RIGHT,-UP,VIEW],axis=1)
MOUNT=.8


def prism(x0,x1,y0,y1,z0,z1):
    p=np.array([[x,y,z] for x in (x0,x1) for y in (y0,y1) for z in (z0,z1)])
    return [p[ix] for ix in ([0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5])]


def project(points):
    q=(np.asarray(points)-[0,1.55,.55])@BASIS
    return q[:,:2]*97+[320,217],q[:,2].mean()


def line(d,pts,fill,width=1):
    pixels,_=project(pts)
    d.line(list(map(tuple,pixels)),fill=fill,width=width)


def static(width,branch):
    im=Image.new('RGB',(W,H),(250,250,248))
    d=ImageDraw.Draw(im)
    for t in np.arange(-3,4,.5):
        line(d,[[t,-.8,0],[t,3.8,0]],'#ddddda')
        line(d,[[-3,t,0],[3,t,0]],'#ddddda')
    shapes=[]
    def add(faces,colors):
        for k,face in enumerate(faces):
            pts,depth=project(face)
            shapes.append((depth,list(map(tuple,pts)),colors[k%len(colors)]))
    add(prism(-3,3,sim.BACK,sim.BACK+.05,0,1.35),[222,217,228,235,219,245])
    for x0 in (-width/2-1,width/2):
        add(prism(x0,x0+1,sim.GATE_Y,sim.GATE_Y+.5,0,1.2),[139,148,175,186,159,202])
    if branch:
        add(tube([-width/2,sim.BRANCH_Y,MOUNT],[width/2,sim.BRANCH_Y,MOUNT],sim.RADIUS,12),[38,44,50])
    add(prism(-.16,.16,-.14,.14,0,.67),[160,142,153])
    for _,poly,c in sorted(shapes,key=lambda f:-f[0]):
        d.polygon(poly,fill=(c,c,c))
    if branch: line(d,[[-width/2,sim.BRANCH_Y,MOUNT],[width/2,sim.BRANCH_Y,MOUNT]],'#272727',2)
    return im


def scene_frame(bg,trace,i,finished,label):
    action,depth,angles,ranges,_=trace
    im=bg.copy();d=ImageDraw.Draw(im)
    origin=np.array([0.,0.,MOUNT])
    direction=lambda a: np.array([np.sin(a),np.cos(a),0.])
    for a in (sim.TRUE[i]-np.radians(8),sim.TRUE[i]+np.radians(8)):
        line(d,[origin,origin+.7*direction(a)],'#666666')
    a=sim.TRUE[i]
    for sign in (-1,1):
        off=sign*.09*direction(a+np.pi/2)
        line(d,[origin+off,origin+off+.16*direction(a)],'#111111',7)
    if np.isfinite(ranges[i]):
        end=origin+ranges[i]*direction(angles[i])
        line(d,[origin,end],'#555555',2)
        xy,_=project([end]);u,v=xy[0]
        d.ellipse((u-3,v-3,u+3,v+3),fill='black',outline='white')
    d.rectangle((0,0,W,37),fill='#282828');d.text((14,4),label,font=TITLE,fill='white')
    # On-screen range plots contain instrument readings only, never ground truth.
    d.rectangle((0,310,W,H),fill='#f5f5f3')
    d.text((14,316),'Echo range vs gyro yaw   /   front sector -40 to +40 deg',font=MONO,fill='#333333')
    d.line([(30,420),(605,420)],fill='#999999')
    for val in (1.5,3.4):
        yy=420-val*18
        d.line([(30,yy),(605,yy)],fill='#dededa')
        d.text((4,yy-8),str(val),font=MONO,fill='#555555')
    for angle,rr in zip(np.degrees(angles[:i+1]),ranges[:i+1]):
        if np.isfinite(rr) and abs(angle)<=40:
            x=30+(angle+40)*575/80;y=420-rr*18
            d.ellipse((x-2,y-2,x+2,y+2),fill='#171717')
    d.text((220,422),'0 deg = straight through gate',font=MONO,fill='#555555')
    return im


def example_seed():
    g=sim.scene_ranges(8,2.6)
    for seed in range(sim.SEEDS):
        good=sim.flight(seed,g,.6)
        weak=sim.flight(seed,g,.03)
        if good[0]=='HOLD' and weak[0]=='PASS' and sim.flight(seed,g)[0]=='PASS':
            return seed
    raise RuntimeError('No paired illustration of a false pass')


def movie(seed,out):
    fps=15
    enc=Encoder(out,1280,576,fps,channels=3)
    try:
        for n,(width,p) in enumerate([(1.30,.60),(2.60,.60),(2.60,.03)],1):
            geom=sim.scene_ranges(8,width)
            pair=[sim.flight(seed,geom),sim.flight(seed,geom,p)]
            bgs=[static(width,b) for b in (False,True)]
            for frame in range(91):
                i=min(frame*3,sim.N-1)
                finished=i==sim.N-1
                im=Image.new('RGB',(1280,576),'#fafaf8');d=ImageDraw.Draw(im)
                d.text((18,6),f'{n}/3    Gate {width:.2f} m    /    branch echo chance {p:.0%} per eligible ping    /    scan at 3x',font=TITLE,fill='#171717')
                for j,(bg,r,lbl) in enumerate(zip(bgs,pair,('Open gate','Same gate + 20 mm branch'))):
                    im.paste(scene_frame(bg,r,i,finished,lbl),(j*W,43))
                    measured=f'{r[3][i]:.2f} m' if np.isfinite(r[3][i]) else 'no return'
                    d.text((j*W+15,504),f'{"Decision: "+r[0] if finished else "Scanning"}    /    supported depth {r[1]:.1f} m' if finished else f'Scanning    /    echo {measured}',font=TITLE,fill='#222222')
                d.text((18,548),f'Stationary scan   /   yaw from simulated gyro   /   t {sim.TIMES[i]:.1f} s   /   paired seed {seed}',font=SMALL,fill='#454545')
                if n==2 and frame==32: im.save(ROOT/'wiki/llm/sonar-gate-3d.png')
                enc.add(np.asarray(im))
    finally:
        enc.close()


def graph(summary,seed,out):
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    ax=axes[0]
    geom=sim.scene_ranges(8,2.6)
    for p,marker,col,lab in [(None,'o','#929aa1','Open'),(.6,'x','#157a74','Branch, strong'),(.03,'+','#ae592c','Branch, weak')]:
        _,_,angles,ranges,_=sim.flight(seed,geom,p)
        x=np.degrees(angles);mask=(abs(x)<41)&np.isfinite(ranges)
        ax.scatter(x[mask],ranges[mask],s=26,color=col,marker=marker,label=lab)
    ax.set(title=f'One paired scan / 2.60 m gate / seed {seed}',xlabel='Gyro yaw relative to gate centre (deg)',ylabel='Measured echo range (m)',ylim=(0,4.1))
    ax.legend(frameon=False)
    ax=axes[1]
    ticks=['Open','0%','3%','15%','60%','100%']
    for w,col in zip((1.3,2.2,2.6),('#5e6671','#b0652d','#18756c')):
        a=[s for s in summary if s['half']==8 and s['width']==w]
        ax.plot(range(6),[s['PASS']/sim.SEEDS*100 for s in a],marker='o',label=f'{w:.2f} m gate',color=col)
    ax.set_xticks(range(6),ticks)
    ax.set(title='Scans that permit passage / 200 per condition',xlabel='Open gate or branch echo chance per ping',ylabel='PASS decisions (%)',ylim=(-6,106))
    ax.axvline(.5,color='#999',ls=':');ax.legend(frameon=False)
    for ax in axes:
        ax.spines[['right','top']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.22)
    fig.savefig(out,dpi=160);plt.close(fig)


if __name__=='__main__':
    s=json.loads((ROOT/'wiki/llm/sonar-gate-summary.json').read_text())
    seed=example_seed()
    graph(s,seed,ROOT/'wiki/llm/sonar-gate.png')
    movie(seed,ROOT/'results/videos/sonar-gate-3d.mp4')
    print('video seed',seed)
