"""Run paired fastsim corridor trials or a controlled switch just before the branch section."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastsim.runner import run_group
from px4.forest_pilot import Pilot, corridor_depth

CONFIGS=('depart-loom35','corridor-depth','corridor-preview','corridor-forward','corridor-follow','corridor-approach')


def trials(output,world,rep,entry,configs):
    details={}
    original=Pilot.step
    keys=('corridor_weight','preview_range','corridor_forward','corridor_follow','scan_remember')
    def observed(p,now,x,y,z,h,lum,v_meas,bump=False,tof=None):
        if p not in details:
            details[p]=dict(recipe={k:p.P[k] for k in keys},events=[],entered=False)
            if entry:
                p.P.update(corridor_weight=0,preview_range=0,corridor_forward=False,corridor_follow=False,scan_remember=False)
        trace=details[p]
        if entry and not trace['entered'] and y>=38:
            trace['entered']=True
            p.P.update(trace['recipe'])
            trace['events'].append(dict(event='entry',t=now,x=x,y=y))
        count=p.preview_count
        rec=p.rec
        phase=rec.get('phase') if rec else None
        result=original(p,now,x,y,z,h,lum,v_meas,bump,tof)
        if count!=p.preview_count:
            trace['events'].append(dict(event='preview',t=now,x=x,y=y,tof=tof))
        if rec and phase=='scan' and rec.get('phase') in ('scan_move','scan_align'):
            k=int(round(rec['target']/p.bin_width))%36
            depth=corridor_depth(rec['bins'],now-rec['seen']<15,p.bin_width)[k]
            trace['events'].append(dict(event='selected',t=now,x=x,y=y,depth=float(depth),angle_rad=float(rec['target'])))
        return result

    specs=[dict(config=c,rep=rep,duration=600) for c in configs]
    with patch.object(Pilot,'step',observed):
        flights=run_group(world,specs,record=True,pose_tick=.02)
    output.mkdir(parents=True,exist_ok=True)
    for f,trace in zip(flights,details.values()):
        run=f['run']
        f['run']={k:run[k] for k in ('x','y','z','h','phase','tof','v','crashes')}
        f['events']=trace['events']
        name='entry' if entry else world
        (output/f"corr26-{name}-{rep}-{f['summary']['config']}.json").write_text(json.dumps(f,allow_nan=False))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    cli.add_argument('--world',default='ff_hard_0')
    cli.add_argument('--rep',type=int,default=1)
    cli.add_argument('--entry',action='store_true')
    cli.add_argument('--configs',nargs='+',choices=CONFIGS,default=CONFIGS)
    a=cli.parse_args()
    if a.entry and a.world!='ff_hard_0': cli.error('the controlled branch-entry comparison uses ff_hard_0')
    trials(a.output,a.world,a.rep,a.entry,a.configs)
