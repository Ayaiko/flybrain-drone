"""Score the same fresh simulated frames with ReviewPilot and pretrained PULP-DroNet v2.
Usage: python dronet_replay.py PATH_TO_model_original_himax.onnx
Dependencies: numpy, pillow, onnxruntime. The model URL and checksum are recorded below.
"""
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter_ns
import numpy as np
import onnxruntime as ort
from PIL import Image
from lighting_trial import sim, draw_walls, camera_for, ReviewPilot, SEEDS

OUT = Path(__file__).resolve().parent
SOURCE = 'https://raw.githubusercontent.com/pulp-platform/pulp-dronet/98f15cfe2e9c34c3a546358e22e45b3b09f11a79/pulp-dronet-v2/gapflow/nntool_input/models_onnx/model_original_himax.onnx'
SHA = 'b764df96067210f55fd9a1ae3804c84be9f60dc51329fbab6277d49d6b7074d6'
STOP = {'reversal / yield','yield / observe','brake / observe','hold'}

def main():
    path = Path(sys.argv[1])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    model = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
    name = model.get_inputs()[0].name

    def risk(image):
        square = image.resize((200,200),Image.Resampling.BILINEAR)
        x = np.asarray(square,dtype=np.float32)[None,None,:,:]/255
        return float(model.run(None,{name:x})[1].item())

    for _ in range(12):
        risk(Image.new('L',(80,48),160))
    archive = json.loads((OUT/'lighting-results.json').read_text())
    times=[[],[]]
    rows=[]
    for case in archive:
        if case['profile'] not in ('reference','both'):
            continue
        seed,profile,event = case['seed'],case['profile'],case['event_tick']
        row,h = sim.simulate('walk_reverse',seed,background=draw_walls,pilot_factory=ReviewPilot,
                             camera=camera_for(profile),reversal_tick=event)
        assert row['last_tick']==case['last_tick']
        pilot=ReviewPilot()
        last=-100
        first=[None,None]
        early=[False,False]
        for i,(pos,_,img,mode,stamp) in enumerate(h):
            if stamp==last: continue
            last=stamp
            start=perf_counter_ns()
            _,cmd=pilot.command(img)
            mid=perf_counter_ns()
            prob=risk(img)
            end=perf_counter_ns()
            times[0].append((mid-start)/1e6)
            times[1].append((end-mid)/1e6)
            if pos[1]>=1.54: break
            for j,flag in enumerate([cmd in STOP,prob>=.5]):
                if flag and i<event: early[j]=True
                if flag and i>=event and first[j] is None: first[j]=i
        rows.append(dict(profile=profile,seed=seed,event_tick=event,early=early,first=first))
    # Empty matched room at the same initial pose distribution, one frame per seed.
    empty=[risk(sim.capture(np.array([0.,0.,.85]),[],np.random.default_rng(s),0,draw_walls)) for s in SEEDS]
    info=dict(source=SOURCE,sha256=SHA,onnxruntime=ort.__version__,threshold=.5,
              method_order=['ReviewPilot','DroNet'],rows=rows,empty_scores=empty,
              timing_ms=[np.percentile(t,[50,95,99]).tolist() for t in times],frames=len(times[0]))
    (OUT/'dronet-replay-results.json').write_text(json.dumps(info,indent=2))
    for prof in ('reference','both'):
        rr=[r for r in rows if r['profile']==prof]
        print(prof,[(sum(r['first'][j] is not None for r in rr),sum(r['early'][j] for r in rr)) for j in (0,1)])
    print('empty',sum(v>=.5 for v in empty),'timing',info['timing_ms'],'frames',info['frames'])

if __name__=='__main__':
    main()
