"""Monochrome orthographic replay of paired fastsim poses and forest collision solids."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from viz.common import Encoder

W,H=640,430
RIGHT=np.array([.86,.51,0.])
RIGHT/=np.linalg.norm(RIGHT)
VIEW=np.array([-RIGHT[1]*.85,RIGHT[0]*.85,-.52])
VIEW/=np.linalg.norm(VIEW)
UP=np.cross(VIEW,RIGHT)*-1
UP/=np.linalg.norm(UP)
BASIS=np.stack((RIGHT,-UP,VIEW),axis=1)
FONT=font_manager.findfont('DejaVu Sans')
TITLE=ImageFont.truetype(FONT,23)
SMALL=ImageFont.truetype(FONT,17)


def tube(a,b,r,sides=12):
    a,b=np.asarray(a),np.asarray(b)
    axis=b-a
    axis=axis/np.linalg.norm(axis)
    u=np.cross(axis,[1.,0,0] if abs(axis[0])<.8 else [0,1.,0])
    u/=np.linalg.norm(u)
    v=np.cross(axis,u)
    angle=np.arange(sides)*2*np.pi/sides
    ring=r*(np.outer(np.cos(angle),u)+np.outer(np.sin(angle),v))
    lo,hi=a+ring,b+ring
    return [np.asarray([lo[i],lo[(i+1)%sides],hi[(i+1)%sides],hi[i]]) for i in range(sides)]+[hi]


def geometry(obs):
    objects=[]
    def add(faces,shade):
        vertices=np.concatenate(faces)
        objects.append((vertices.min(0),vertices.max(0),faces,shade))
    for x,y,r,z0,z1,_ in obs['cyl']:
        add(tube([x,y,z0],[x,y,z1],r),155)
    for *pts,r,_ in obs['seg']:
        add(tube(pts[:3],pts[3:],r,8),120)
    for x,y,z,r,_ in obs['sph']:
        lat=np.linspace(-np.pi/2,np.pi/2,7)
        lon=np.arange(10)*2*np.pi/10
        rings=np.stack([np.c_[np.cos(lon)*np.cos(a),np.sin(lon)*np.cos(a),np.full(10,np.sin(a))] for a in lat])*r+[x,y,z]
        add([rings[[j,j,j+1,j+1],[i,(i+1)%10,(i+1)%10,i]] for j in range(6) for i in range(10)],175)
    for x,y,z,dx,dy,dz,_ in obs['box']:
        verts=np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)])*[dx/2,dy/2,dz/2]+[x,y,z]
        add([verts[ix] for ix in ([0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5])],145)
    return objects


def panel(run,i,objects,title):
    p=np.array([run[k][i] for k in ('x','y','z')])
    def project(points):
        q=(np.asarray(points)-p)@BASIS
        return q[:,:2]*68+[W/2,H*.54],q[:,2].mean()
    im=Image.new('L',(W,H),242)
    d=ImageDraw.Draw(im)
    for j in range(-8,9):
        x,y=np.floor(p[:2])
        for a,b in (([x+j,y-8,0],[x+j,y+8,0]),([x-8,y+j,0],[x+8,y+j,0])):
            pts,_=project([a,b]);d.line(list(map(tuple,pts)),fill=210)
    faces=[]
    def face(poly,color):
        pts,depth=project(poly)
        faces.append((depth,list(map(tuple,pts)),color))
    for lo,hi,polys,shade in objects:
        if np.any(lo[:2]>p[:2]+4.8) or np.any(hi[:2]<p[:2]-4.8):
            continue
        for k,poly in enumerate(polys):
            face(poly,int(shade+22*np.cos(k*.55)))
    h=run['h'][i]
    f=np.array([np.cos(h),np.sin(h),0])
    l=np.array([-f[1],f[0],0])
    for a in (-1,1):
        for b in (-1,1):
            tip=p+.21*(a*f+b*l)
            for poly in tube(p,tip,.035,6):
                face(poly,42)
            angle=np.arange(16)*np.pi/8
            rotor=tip+[0,0,.05]+.125*(np.outer(np.cos(angle),f)+np.outer(np.sin(angle),l))
            face(rotor,67)
    face([p+.22*f,p-.13*f+.13*l,p-.13*f-.13*l],18)
    for _,pts,color in sorted(faces,key=lambda t:-t[0]):
        d.polygon(pts,fill=color)
    # Ring marks the logged centre even if a foreground trunk hides the body.
    cx,cy=W/2,H*.54
    d.ellipse((cx-26,cy-26,cx+26,cy+26),outline=255,width=4)
    d.ellipse((cx-27,cy-27,cx+27,cy+27),outline=35,width=1)
    d.rectangle((0,0,W,38),fill=33);d.text((14,6),title,font=TITLE,fill=249)
    hits=[c for c in run['crashes'] if c['i']<=i]
    active=hits and i-hits[-1]['i']<30
    bar=40 if active else 232
    d.rectangle((0,H-39,W,H),fill=bar)
    msg=f"{run['phase'][i]}   /   ToF {run['tof'][i]:.2f} m   /   contacts {len(hits)}"
    if active: msg+='   /   HIT '+hits[-1]['kind']
    d.text((12,H-30),msg,font=SMALL,fill=250 if active else 28)
    return im


def movie3d(flights,meta,root):
    objects=geometry(meta['obstacles'])
    runs=[f['run'] for f in flights]
    hits=runs[0]['crashes']
    events=[hits[0],next(h for h in hits if h['kind']=='branch')]
    enc=Encoder(root/'results/videos/hard0-braking-3d.mp4',1280,512,15,channels=3)
    try:
        for hit in events:
            for i in range(hit['i']-120,hit['i']+91,2):
                im=Image.new('L',(1280,512),250);d=ImageDraw.Draw(im)
                d.text((16,10),f"Hard 0 / start 1     {i/30:.2f} s     /     {hit['kind']} encounter     /     1x",font=TITLE,fill=25)
                for j,(r,title) in enumerate(zip(runs,['Before  /  scan-front','After  /  scan-front-margin'])):
                    im.paste(panel(r,i,objects,title),(j*W,44))
                d.text((16,483),'Circle = aircraft centre   /   same camera angle and scale   /   floor grid = 1 m',font=SMALL,fill=40)
                if i==events[0]['i']:
                    im.save(root/'wiki/llm/hard0-braking-3d.png')
                enc.add(np.asarray(im.convert('RGB')))
    finally:
        enc.close()
