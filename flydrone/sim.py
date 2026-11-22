"""Episode runner: unicycle drone (yaw rate + forward speed).
crash='end': episode ends at the first collision.  crash='bounce': the drone is put back where it was, turned
around (180 deg +- 30), and keeps flying until T; every collision is counted (measures avoidance, not luck)."""
import numpy as np
from .world import World

DT = 0.02
OVERSAMPLE = 4


def make_eye(n=120, fov_deg=330):
    return np.radians(np.linspace(-fov_deg / 2, fov_deg / 2, n))


def run_episode(ctrl_cls, seed, T=60.0, density=60, n_eye=120, noise=0.02, ctrl_kw=None, record=False, drone_r=0.25, crash='end'):
    world = World(seed=seed, n_obstacles=density)
    eye = make_eye(n_eye)
    step_a = eye[1] - eye[0]
    eye_fine = (eye[:, None] + step_a * (np.arange(OVERSAMPLE) / OVERSAMPLE - .5 + .5 / OVERSAMPLE)).ravel()
    ctrl = ctrl_cls(eye, DT, seed=seed, **(ctrl_kw or {}))
    rng = np.random.default_rng(seed + 999)
    p = world.start.copy(); h = np.pi / 2 + rng.normal(0, .2)
    dist_flown, t, path, crashes, crash_pts = 0.0, 0.0, [p.copy()], 0, []
    cells = {(int(p[0] // 2), int(p[1] // 2))}          # 2x2 m cells visited = exploration (no credit for circling)
    while t < T:
        dist_f, lum_f = world.cast(p, h + eye_fine)          # 4 rays per ommatidium, averaged = optical blur
        lum = lum_f.reshape(-1, OVERSAMPLE).mean(1); dist = dist_f.reshape(-1, OVERSAMPLE).min(1)
        lum = lum + rng.normal(0, noise, lum.shape)
        yaw, v = ctrl.act(lum, dist)
        h += np.clip(yaw, -12, 12) * DT
        step = v * DT * np.array([np.cos(h), np.sin(h)])
        prev = p; p = p + step; dist_flown += np.linalg.norm(step); t += DT
        if record: path.append(p.copy())
        cells.add((int(p[0] // 2), int(p[1] // 2)))
        if world.collides(p, drone_r):
            crashes += 1; crash_pts.append(p.copy())
            if crash == 'end':
                return dict(collided=True, crashes=1, time=t, dist=dist_flown, cells=len(cells), crash_pts=crash_pts,
                            path=np.array(path) if record else None, world=world)
            p = prev; h += np.pi + rng.uniform(-.5, .5)
            if record: path.append(p.copy())
    return dict(collided=crashes > 0, crashes=crashes, time=t, dist=dist_flown, cells=len(cells), crash_pts=crash_pts,
                path=np.array(path) if record else None, world=world)


def evaluate(ctrl_cls, seeds, **kw):
    res = [run_episode(ctrl_cls, s, **kw) for s in seeds]
    t = np.array([r['time'] for r in res]); d = np.array([r['dist'] for r in res])
    c = np.array([r['collided'] for r in res]); cl = np.array([r['cells'] for r in res])
    return dict(cells=float(cl.mean()), survival_time=float(t.mean()), dist=float(d.mean()), collision_rate=float(c.mean()),
                dist_per_collision=float(d.sum() / max(1, c.sum())), n=len(seeds))
