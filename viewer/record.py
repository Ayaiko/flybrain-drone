"""Record full sim episodes (pose, eye image, looming signal, controller state, crashes) for the browser viewer.
Same loop as flydrone.sim.run_episode in bounce mode. Run from the project root: python -m viewer.record
-> viewer/episodes.json  (then serve the project root and open /viewer/)"""
import base64, json
from pathlib import Path
import numpy as np
from flydrone.sim import DT, OVERSAMPLE, make_eye
from flydrone.world import World
from flydrone.controllers import FlyVision, Oracle

ROOT = Path(__file__).resolve().parents[1]
WORLDS = [0, 1, 2, 3, 4, 5]; T = 60.0
CTRL = {'old looming (sum)': (FlyVision, {}), 'LPLC2 (new threshold)': (FlyVision, dict(loom_mode='lplc2', lplc2_thr=2.0, front=35)),
        'oracle (true depth)': (Oracle, {})}
STATES = ['warmup', 'settling', 'fixate', 'saccade', 'saccade_start']


def record(cls, kw, seed):
    w = World(seed=seed); eye = make_eye(120); sa = eye[1] - eye[0]
    fine = (eye[:, None] + sa * (np.arange(OVERSAMPLE) / OVERSAMPLE - .5 + .5 / OVERSAMPLE)).ravel()
    c = cls(eye, DT, seed=seed, **kw); rng = np.random.default_rng(seed + 999)
    p = w.start.copy(); h = np.pi / 2 + rng.normal(0, .2); t = 0.0
    X, Y, H, S, Lo, Th, Lm, Im, crashes = [], [], [], [], [], [], [], [], []
    while t < T:
        dist_f, lum_f = w.cast(p, h + fine)
        lum = lum_f.reshape(-1, OVERSAMPLE).mean(1) + rng.normal(0, .02, 120)
        yaw, v = c.act(lum, dist_f.reshape(-1, OVERSAMPLE).min(1)); dg = getattr(c, 'diag', {})
        X.append(round(float(p[0]), 3)); Y.append(round(float(p[1]), 3)); H.append(round(float(h), 3))
        S.append(STATES.index(dg.get('state', 'warmup'))); Lo.append(round(float(dg.get('loom', 0)), 4))
        th = dg.get('thr', np.nan); Th.append(None if th is None or not np.isfinite(th) else round(float(th), 4))
        Lm.append(int(bool(dg.get('looming', False)))); Im.append(np.clip(lum * 255, 0, 255).astype(np.uint8).tobytes())
        h += np.clip(yaw, -12, 12) * DT; prev = p; p = p + v * DT * np.array([np.cos(h), np.sin(h)]); t += DT
        if w.collides(p, .25):
            crashes.append([len(X), round(float(p[0]), 3), round(float(p[1]), 3)]); p = prev; h += np.pi + rng.uniform(-.5, .5)
    return dict(x=X, y=Y, h=H, state=S, loom=Lo, thr=Th, looming=Lm, img=base64.b64encode(b''.join(Im)).decode(), crashes=crashes)


def main():
    out = dict(dt=DT, n_eye=120, fov=330, drone_r=.25, states=STATES, worlds=[])
    for s in WORLDS:
        w = World(seed=s)
        ent = dict(seed=s, size=w.size, obstacles=[[round(float(a), 3), round(float(b), 3), round(float(r), 3), int(k), round(float(ph), 3)]
                                                   for (a, b), r, k, ph in zip(w.c, w.r, w.k, w.ph)], runs={})
        for name, (cls, kw) in CTRL.items():
            ent['runs'][name] = record(cls, kw, s); print(s, name, 'crashes', len(ent['runs'][name]['crashes']), flush=True)
        out['worlds'].append(ent)
    p = ROOT / 'viewer/episodes.json'; json.dump(out, open(p, 'w'), separators=(',', ':')); print('saved', p, p.stat().st_size // 1024, 'KB')




def main_nav():
    """Coverage episodes (GPS-like planner + controller) -> viewer/episodes_nav.json; open /?f=episodes_nav.json"""
    from flydrone.nav_sim import run_nav, MARGIN, CELL
    import sys
    mem = json.load(open(ROOT / 'results/nav/nav_memory.json'))['picked'][0] if (ROOT / 'results/nav/nav_memory.json').exists() else 'LPLC2 mem d2.5 line1'
    d = float(mem.split(' d')[1].split()[0]); lc = mem.endswith('line1')
    L2 = dict(loom_mode='lplc2', lplc2_thr=2.0, front=35, goal_gap=2.0)
    cfg = {'LPLC2 + nav, no memory': ('fly', L2, {}), 'LPLC2 + nav + memory + heading veto': ('fly', {**L2, 'veto_r': 4.0}, dict(memory=True, loom_d=d, line_check=lc)),
           'oracle + nav': ('oracle', {}, {})}
    out = dict(dt=DT, n_eye=120, fov=330, drone_r=.25, states=STATES, worlds=[], cells=dict(margin=MARGIN, cell=CELL))
    for s in [0, 1, 2]:
        w = World(seed=s)
        ent = dict(seed=s, size=w.size, obstacles=[[round(float(a), 3), round(float(b), 3), round(float(r), 3), int(k), round(float(ph), 3)]
                                                   for (a, b), r, k, ph in zip(w.c, w.r, w.k, w.ph)], runs={})
        for name, (base, kw, nk) in cfg.items():
            r = run_nav(base, kw, s, task='cov', T=180.0, record=True, **nk); rec = r['rec']
            img = np.clip(np.array(rec['img']) * 255, 0, 255).astype(np.uint8).tobytes()
            ent['runs'][name] = dict(x=[round(v, 3) for v in rec['x']], y=[round(v, 3) for v in rec['y']], h=[round(v, 3) for v in rec['h']],
                                     state=[STATES.index(v) if v in STATES else 0 for v in rec['state']], loom=[round(v, 4) for v in rec['loom']],
                                     thr=[None if v is None else round(v, 4) for v in rec['thr']], looming=rec['looming'],
                                     goal=[None if g is None else [round(g[0], 2), round(g[1], 2)] for g in rec['goal']],
                                     img=base64.b64encode(img).decode(), crashes=[c[:3] for c in rec['crashes'] if c[3] == 'post'],
                                     wall_crashes=[c[:3] for c in rec['crashes'] if c[3] == 'wall'], hazards=r['hazards'])
            print(s, name, 'coverage', round(r['coverage'] * 100, 1), 'posts', r['posts'], flush=True)
        out['worlds'].append(ent)
    p = ROOT / 'viewer/episodes_nav.json'; json.dump(out, open(p, 'w'), separators=(',', ':')); print('saved', p, p.stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    import sys
    main_nav() if '--nav' in sys.argv else main()
