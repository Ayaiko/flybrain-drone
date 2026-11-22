"""Closed-loop FlyVis with queued OV2640 images and lagged velocity tracking."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fly-poc"))
import run
from ov2640_trial import jpeg_frame

def trial(model, kind, seed, stress):
    scene = np.random.default_rng(seed)
    channel = np.random.default_rng(seed + 501)
    box = run.layout(kind, scene)
    obstacles = [(box, 42)] if box else []
    pos = np.array([scene.uniform(-.15, .15), 0.])
    vel = np.zeros(2)
    vehicle = run.Airframe(cruise=.2)
    state = run.bias.copy()
    queue, idle, stale_count = [], 0, 0
    captured, scores = -10, np.zeros(4)
    minimum = None
    mean, sd, coef = model
    for tick in range(560):
        now = tick / 10
        img = jpeg_frame(obstacles, *pos, scene)
        if not stress or channel.random() >= .08:
            delay = channel.uniform(.04, .18) if stress else 0
            queue.append((now + delay, now, img))
        ready = [item for item in queue if item[0] <= now]
        queue = [item for item in queue if item[0] > now]
        if ready:
            _, stamp, image = max(ready, key=lambda r: r[1])
            if stamp > captured:
                captured = stamp
                scores = np.r_[(run.read_eye(image, state)-mean)/sd, 1.] @ coef
        cmd = run.plan(scores, captured, now, vel, vehicle)
        stale_count += tick >= 6 and now - captured > vehicle.frame_timeout
        target = np.array([cmd.lateral, cmd.forward]) if tick >= 6 else np.zeros(2)
        change = target - vel
        limit = .55 if stress else 1.6
        step = change * min(.28 if stress else 1., limit*.1/max(np.linalg.norm(change), 1e-9))
        pos += .1 * (vel + step / 2)
        vel += step
        if box:
            x, y = pos
            gap = np.hypot(max(box[0]-x, x-box[1], 0), max(box[2]-y, y-box[3], 0)) - vehicle.radius
            minimum = float(gap) if minimum is None else min(minimum, float(gap))
            if gap <= 0:
                outcome = "contact"
                break
        idle = idle + 1 if tick >= 6 and not target.any() and np.linalg.norm(vel)<.005 else 0
        if idle >= 12 or pos[1]>=6.8:
            outcome = "stopped" if idle >= 12 else "goal"
            break
    else:
        outcome = "timeout"
    return dict(kind=kind, seed=seed, stress=stress, outcome=outcome, min_gap_m=minimum,
                stale_ticks=stale_count, end_y=float(pos[1]))

if __name__ == "__main__":
    with np.load(run.ROOT / 'readout.npz') as archive:
        model = tuple(archive[k] for k in ('mean','sd','coef'))
    rows = [trial(model, k, seed, stress) for k in (0,1,3) for seed in (23000,23001)
            for stress in (False, True)]
    (Path(__file__).with_name('camera_stress.json')).write_text(json.dumps(rows, indent=2))
