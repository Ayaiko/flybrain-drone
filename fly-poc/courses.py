"""Multi-obstacle benchmark. Dimensions in metres; x lateral, y forward."""
import json
from collections import Counter
import numpy as np
from run import ROOT, evaluate

# For each obstacle: (left, right, near, far, height), grayscale tone.
courses = {
    'กล่องสามใบ': [
        ((-.45,1.55,3.5,4.15,1.65),42),
        ((-2.25,-.55,7.2,7.85,1.8),125),
        ((.05,1.45,10.8,11.45,1.5),72)],
    'สลาลอมสลับฝั่ง': [
        ((-.35,3.8,3.5,4.05,1.7),52),
        ((-3.8,.35,7.1,7.65,1.6),115),
        ((-.35,3.8,10.7,11.25,1.75),65)],
    'ประตูคู่เยื้องแนว': [
        ((-3.8,-.12,4.2,4.8,1.65),60),
        ((1.18,3.8,4.2,4.8,1.65),105),
        ((-3.8,-1.18,9.2,9.8,1.65),105),
        ((.12,3.8,9.2,9.8,1.65),60)]
}

def main():
    with np.load(ROOT/'readout.npz') as archive:
        model = tuple(archive[k] for k in ('mean','sd','coef'))
    logs, replays = [], []
    for name, obstacles in courses.items():
        outcomes = Counter()
        for seed in range(12000,12008):
            run = evaluate(model,None,seed,seed==12000,course=obstacles)
            run['name'] = name
            logs.append({k:v for k,v in run.items() if k!='frames'})
            outcomes[run['result']] += 1
            if run['frames']:
                replays.append(run)
        print(name,dict(outcomes),flush=True)
    (ROOT/'course-results.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2))
    template = (ROOT/'viewer.template').read_text()
    start,end = template.index('<option>'),template.index('</select>')
    options = ''.join(f'<option>{name}</option>' for name in courses)
    template = template[:start]+options+template[end:]
    template = template.replace('กล้องหน้าของโดรนจำลอง','สนามทดสอบหลายสิ่งกีดขวาง')
    template = template.replace('<main>','<p><a href="gate-replay.html">เปรียบเทียบวิธีลอดประตูคู่</a></p><main>',1)
    (ROOT/'courses.html').write_text(template.replace('REPLAY_DATA',json.dumps(replays)))

if __name__=='__main__':
    main()
