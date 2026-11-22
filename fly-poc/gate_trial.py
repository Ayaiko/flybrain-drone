"""Paired gate experiment with a map reference and an image-only controller."""
import json
from collections import Counter

import numpy as np
from courses import courses
from planner import Command
from run import ROOT, evaluate

GATES = courses['ประตูคู่เยื้องแนว']

def steer(error, reason):
    return Command(float(np.clip(.8*error,-.48,.48)),.52,reason)

def reference(x,y):
    for left,right in zip(GATES[::2],GATES[1::2]):
        if y < left[0][3]+.24:
            center = (left[0][1]+right[0][0])/2
            return steer(center-x,'เล็งกลางช่องจากแผนที่')
    return steer(0,'พ้นประตูแล้ว')

def opening(image):
    # Locate the nearest pair of panel tops against the bright background.
    # Higher scanlines favor nearby panels over the offset gate behind them.
    h,w = image.shape
    for row in range(1,h//2-2):
        bright = np.median(image[row-1:row+2],axis=0)>141
        if np.sum(~bright)<6:
            continue
        edges = np.diff(np.r_[0,bright,0])
        spans = [(a,b) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))
                 if b-a>=4 and a>=2 and b<=w-2]
        if spans:
            a,b = max(spans,key=lambda pair:pair[1]-pair[0])
            bearing = ((a+b)/2-w/2)/(w/2)
            return bearing,(b-a)/w
    return None

class VisionGate:
    def __init__(self):
        self.hold_frames = 0

    def command(self,image):
        if self.hold_frames:
            self.hold_frames -= 1
            return steer(0,'รักษาแนวระหว่างผ่านช่อง')
        gap = opening(image)
        if gap is None:
            if np.median(image[1:image.shape[0]//2-1],axis=0).min()>141:
                return steer(0,'เบื้องหน้าว่างในภาพ')
            return Command(0,0,'ไม่พบขอบช่องครบสองด้าน')
        bearing,width = gap
        if width >= .6 and abs(bearing)<.1:
            # Fixed 10 Hz, 0.52 m/s: maintain heading over roughly 2.1 m.
            self.hold_frames = 40
        return steer(bearing*2.8,'เล็งกลางช่องระหว่างขอบแผงในภาพ')

def main():
    with np.load(ROOT/'readout.npz') as archive:
        model = tuple(archive[k] for k in ('mean','sd','coef'))
    options = {'อ้างอิงจากพิกัด':lambda img,x,y:reference(x,y),
               'หาช่องจากภาพ':None}
    logs,examples = [],[]
    for name,controller in options.items():
        for seed in [*range(12000,12008), *range(16200,16208)]:
            camera = VisionGate()
            active = controller or (lambda img,x,y:camera.command(img))
            run = evaluate(model,None,seed,True,course=GATES,gate_controller=active)
            run['name'] = name
            run['split'] = 'original' if seed < 16200 else 'fresh'
            # Confirm actual crossing inside both openings, including body radius.
            path = [(f[0],f[1]) for f in run['frames']]
            crossed = []
            for left,right in zip(GATES[::2],GATES[1::2]):
                strip = [x for x,y in path if left[0][2]<=y<=left[0][3]]
                crossed.append(bool(bool(strip) and run['end_y']>left[0][3]+.24 and
                               all(left[0][1]+.24<x<right[0][0]-.24 for x in strip)))
            run['crossed'] = crossed
            logs.append({k:v for k,v in run.items() if k!='frames'})
            if seed==12000:
                examples.append(run)
        batch = [r for r in logs if r['name']==name]
        print(name,Counter(r['result'] for r in batch),min(r['min_clearance'] for r in batch),[r['crossed'] for r in batch])
    (ROOT/'gate-results.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2))
    template = (ROOT/'viewer.template').read_text()
    start,end = template.index('<option>'),template.index('</select>')
    template = template[:start]+''.join(f'<option>{n}</option>' for n in options)+template[end:]
    template = template.replace('กล้องหน้าของโดรนจำลอง','ลอดประตูคู่เยื้องแนว').replace(' s · Model ',' s · FlyVis (บันทึกเปรียบเทียบ) ')
    (ROOT/'gate-replay.html').write_text(template.replace('REPLAY_DATA',json.dumps(examples)))

if __name__=='__main__':
    main()
