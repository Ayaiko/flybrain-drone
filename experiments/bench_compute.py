"""Compute cost of each controller's 'brain' only (one decision = one 50 Hz frame), excluding world/raycasting.
Times act() on recorded real eye frames; op counts are analytic (multiply-adds per frame)."""
import time, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from flydrone.sim import make_eye, DT, OVERSAMPLE
from flydrone.world import World
from flydrone.controllers import Oracle, FlyVision, Linear

eye = make_eye(120); n = len(eye)
w = World(seed=5); rng = np.random.default_rng(0)
# record 600 real frames along a straight line so EMD/branches see realistic input
step_a = eye[1] - eye[0]
fine = (eye[:, None] + step_a * (np.arange(OVERSAMPLE) / OVERSAMPLE - .5 + .5 / OVERSAMPLE)).ravel()
frames = []
for k in range(600):
    p = w.start + np.array([0, 0.004 * k]); d, l = w.cast(p, np.pi / 2 + fine)
    frames.append((l.reshape(-1, OVERSAMPLE).mean(1) + rng.normal(0, .02, n), d.reshape(-1, OVERSAMPLE).min(1)))
t0 = time.perf_counter()
for _ in range(600): w.cast(w.start, np.pi / 2 + fine)
sensor_us = (time.perf_counter() - t0) / 600 * 1e6

ops = {  # multiply-adds per frame, from the code
    'fly (EMD + flow balance + looming)': n * 4 + (n - 1) * 3 + 3 * (n - 1) + 20,
    'linear read-out (EMD + 98 weights)': n * 4 + (n - 1) * 3 + 2 * (n - 1) + 2 * 49,
    'oracle (needs depth sensor)': n * 6 + 20,
}
res = {}
for name, cls, kw in [('fly (EMD + flow balance + looming)', FlyVision, {}), ('linear read-out (EMD + 98 weights)', Linear, {}),
                      ('oracle (needs depth sensor)', Oracle, {})]:
    c = cls(eye, DT, **kw)
    for lum, d in frames[:50]: c.act(lum, d)                       # warm up
    t0 = time.perf_counter()
    for _ in range(5):
        for lum, d in frames: c.act(lum, d)
    us = (time.perf_counter() - t0) / (5 * len(frames)) * 1e6
    res[name] = dict(us_per_frame_python=round(us, 1), mac_per_frame=ops[name])
    print(f'{name:38s} {us:7.1f} us/frame (numpy, 1 core)   ~{ops[name]:5d} multiply-adds/frame')
print(f'(simulator raycasting, not the brain: {sensor_us:.0f} us/frame)')
json.dump(dict(controllers=res, sim_raycast_us=round(sensor_us, 1)), open(ROOT / 'results/compute/compute_bench.json', 'w'), indent=1)
