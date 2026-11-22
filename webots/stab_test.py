"""Stabiliser test (no fly brain): repeated fly 3 s -> stop + 90 deg turn -> hold, in an empty part of world 0.
Reports flips, yaw-rate leak while holding, roll excursions and speed tracking for several gain sets.
Run from the project root: python -m webots.stab_test"""
import json
from pathlib import Path
import numpy as np
from webots.run_sim import run_batch

ROOT = Path(__file__).resolve().parents[1]
CFG = ['test-v1.0-L1.2-Y1.3', 'test-v1.0-Lm1.2-Y1.3', 'test-v1.0-L0-Y1.3', 'test-v1.0-Lm1.2-Y3', 'test-v1.0-L1.2-Y3', 'test-v1.0-Lm2.5-Y3']
if __name__ == '__main__':
    run_batch([(m, 0, 40) for m in CFG], per_job=120)
    for m in CFG:
        f = ROOT / f'webots/runs/{m}_0.json'
        if not f.exists(): print(m, 'no result'); continue
        r = json.loads(f.read_text())['run']; st = np.array(r['state']); roll = np.degrees(np.abs(r['roll'])); wy = np.degrees(np.abs(r['wy']))
        vf = np.array(r['vf']); hold = st == 1; fly = st == 2
        flip = bool((roll > 60).any())
        print(f'{m:22s} flipped {flip!s:5s}  max roll {roll.max():5.1f}  hold |yaw rate| median {np.median(wy[hold]):5.1f} deg/s  '
              f'cruise speed median {np.median(vf[fly][len(vf[fly])//4:]):.2f} m/s  hold speed median {np.median(np.abs(vf[hold])):.2f}')
