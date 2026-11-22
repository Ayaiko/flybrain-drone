"""Forest test field for the PX4 + Gazebo sim: a procedural, seeded forest with a long under-canopy corridor split into
zones, each probing one thing a fly-style eye cares about. Everything (geometry + textures) is generated here with numpy,
so there are no third-party assets or licences involved.

Corridor runs along +y (the drone spawns at the origin facing +y). Zones (y, metres):
  entry     0-8    open clearing, baseline
  trunks    8-30   dense trunks lining the corridor at varying spacing + trunks standing inside it + background forest
  clearing  30-40  round gap in the forest
  branches  40-60  low branches / twigs / leaf clumps reaching into the corridor at eye height + shrubs
  lowtex    60-76  low texture: plain trunks the same grey as the sky, a featureless rock wall and a boulder jutting in
  exit      76-84  exit clearing, goal at y = 80
The corridor centre line is x = curve_amp * sin(2 pi y / 40).

Difficulty presets easy / medium / hard; any parameter can be overridden: --set width=2.8 --set bg_density=0.2
Every layout is checked to be passable for the drone (0.35 m body radius) inside the corridor at flight altitude;
if not, it is re-drawn.

Outputs: px4/forest/<name>.json (config, zones, obstacle list for crash/clearance truth), px4/forest/<name>/meshes/*.obj,
px4/forest/<name>_map.png (top-down view at flight height), PX4 world PX4-Autopilot/.../worlds/<name>.sdf

Run from the project root (host): python -m px4.make_forest easy 0   [--set key=value ...]"""
import argparse, json
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image
from px4.forest_geom import Obstacles, BODY_R

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'px4/forest'; TEXD = OUT / 'textures'
PX = ROOT / 'px4/PX4-Autopilot/Tools/simulation/gz'; CONT = '/work/px4/forest'      # same folder inside the container
ALT = 1.5
ZONES = [('entry', -6, 8), ('trunks', 8, 30), ('clearing', 30, 40), ('branches', 40, 60), ('lowtex', 60, 76), ('exit', 76, 90)]
FIELD = dict(xmin=-16.0, xmax=16.0, ymin=-6.0, ymax=90.0)
EXIT_Y, PERIOD = 80.0, 40.0
PRESETS = {
    #            corridor  edge-trunk  trunks inside  background   P(branch)  branch reach   shrubs     corridor  rock wall
    #            width m   spacing m   per 10 m       trunks/m2    per edge   (x width)      per 10 m   bend m    jut (x width)
    'easy':   dict(width=4.0, edge_spacing=2.6, intruders_per_10m=0.4, bg_density=0.05, branch_prob=0.30, branch_reach=0.30, shrubs_per_10m=1.5, curve_amp=0.0, rock_jut=0.30),
    'medium': dict(width=3.0, edge_spacing=2.0, intruders_per_10m=0.8, bg_density=0.09, branch_prob=0.50, branch_reach=0.40, shrubs_per_10m=3.0, curve_amp=1.5, rock_jut=0.40),
    'hard':   dict(width=2.4, edge_spacing=1.5, intruders_per_10m=1.3, bg_density=0.14, branch_prob=0.70, branch_reach=0.50, shrubs_per_10m=5.0, curve_amp=2.5, rock_jut=0.45),
}
PLAIN = 0.66          # grey of the low-texture surfaces, close to Gazebo's 0.7 background


# ---------------------------------------------------------------- textures (periodic, so they tile around a trunk)
def pnoise(h, w, sx, sy, rng):
    f = np.fft.fft2(rng.normal(size=(h, w))); ky = np.fft.fftfreq(h)[:, None] * h; kx = np.fft.fftfreq(w)[None] * w
    n = np.real(np.fft.ifft2(f * np.exp(-(kx / sx) ** 2 - (ky / sy) ** 2))); return n / (np.abs(n).max() + 1e-9)


def textures():
    TEXD.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(1234); x = np.arange(512) / 512
    for i in range(4):          # bark: ridges around the trunk + vertical furrow noise; dark brown-grey
        k1, k2 = rng.integers(5, 13, 2)
        lum = (0.30 + 0.04 * i + 0.10 * np.sin(2 * np.pi * k1 * x + rng.uniform(0, 6))[None] + 0.05 * np.sin(2 * np.pi * k2 * x + rng.uniform(0, 6))[None]
               + 0.10 * pnoise(256, 512, 40, 4, rng) + 0.05 * pnoise(256, 512, 120, 60, rng))
        rgb = np.clip(lum[..., None] * np.array([1.08, 0.96, 0.84]), 0, 1)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(TEXD / f'bark{i}.png')
    for name, base, col in (('leaves', 0.42, (0.62, 1.0, 0.45)), ('crown', 0.32, (0.55, 0.95, 0.42))):
        n = pnoise(512, 512, 30, 30, rng); gaps = (pnoise(512, 512, 12, 12, rng) > 0.35) * 0.25
        lum = np.clip(base + 0.22 * n - gaps, 0.05, 1)
        Image.fromarray((np.clip(lum[..., None] * np.array(col), 0, 1) * 255).astype(np.uint8)).save(TEXD / f'{name}.png')


# ---------------------------------------------------------------- meshes
class Mesh:
    def __init__(self): self.V, self.T, self.N, self.F, self.n = [], [], [], [], 0

    def add(self, V, T, N, F):
        self.V.append(V); self.T.append(T); self.N.append(N); self.F.append(F + self.n); self.n += len(V)

    def cylinder(self, a, b, r, seg=12, rep=1.0, vs=0.5):
        a, b = np.asarray(a, float), np.asarray(b, float); ax = b - a; L = np.linalg.norm(ax); ax /= L
        u1 = np.cross(ax, [0, 0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0, 0]); u1 /= np.linalg.norm(u1); u2 = np.cross(ax, u1)
        th = np.linspace(0, 2 * np.pi, seg + 1); rad = np.cos(th)[:, None] * u1 + np.sin(th)[:, None] * u2
        V = np.vstack([a + r * rad, b + r * rad]); N = np.vstack([rad, rad])
        T = np.vstack([np.c_[th / (2 * np.pi) * rep, np.zeros(seg + 1)], np.c_[th / (2 * np.pi) * rep, np.full(seg + 1, L * vs)]])
        j = np.arange(seg); F = np.vstack([np.c_[j, j + 1, j + seg + 2], np.c_[j, j + seg + 2, j + seg + 1]])
        self.add(V, T, N, F)

    def sphere(self, c, r, nu=12, nv=8):
        r = np.broadcast_to(np.asarray(r, float), 3); lon = np.linspace(0, 2 * np.pi, nu + 1); lat = np.linspace(-np.pi / 2, np.pi / 2, nv + 1)
        LO, LA = np.meshgrid(lon, lat); n = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
        V = np.asarray(c) + n * r; N = n / r; N /= np.linalg.norm(N, axis=1, keepdims=True)
        T = np.c_[LO.ravel() / np.pi * max(1, r[0]), (LA.ravel() + np.pi / 2) / np.pi * max(1, r[2])]
        F = []
        for i in range(nv):
            for j in range(nu):
                p = i * (nu + 1) + j; F += [[p, p + 1, p + nu + 2], [p, p + nu + 2, p + nu + 1]]
        self.add(V, T, N, np.array(F))

    def box(self, c, s):
        c, s = np.asarray(c, float), np.asarray(s, float) / 2
        for ax in range(3):
            for sg in (-1, 1):
                o1, o2 = [k for k in range(3) if k != ax]; q = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
                V = np.zeros((4, 3)); V[:, ax] = sg * s[ax]; V[:, o1] = q[:, 0] * s[o1]; V[:, o2] = q[:, 1] * s[o2]
                N = np.zeros((4, 3)); N[:, ax] = sg
                F = np.array([[0, 1, 2], [0, 2, 3]])
                if np.dot(np.cross(V[1] - V[0], V[2] - V[0]), N[0]) < 0: F = F[:, ::-1]
                self.add(c + V, (q + 1) * s[[o1, o2]] / 2, N, F)

    def write(self, path):
        V, T, N, F = (np.vstack(a) for a in (self.V, self.T, self.N, self.F))
        with open(path, 'w') as f:
            f.write(''.join(f'v {a:.4f} {b:.4f} {c:.4f}\n' for a, b, c in V)); f.write(''.join(f'vt {a:.4f} {b:.4f}\n' for a, b in T))
            f.write(''.join(f'vn {a:.4f} {b:.4f} {c:.4f}\n' for a, b, c in N))
            f.write(''.join(f'f {a + 1}/{a + 1}/{a + 1} {b + 1}/{b + 1}/{b + 1} {c + 1}/{c + 1}/{c + 1}\n' for a, b, c in F))
        return len(V)


# ---------------------------------------------------------------- layout
def zone_of(y):
    for n, a, b in ZONES:
        if a <= y < b: return n
    return 'outside'


def generate(cfg, seed, attempt=0):
    rng = np.random.default_rng([seed, attempt, int(cfg['width'] * 100), int(cfg['bg_density'] * 1000)])
    A, hw = cfg['curve_amp'], cfg['width'] / 2
    xc = lambda y: A * np.sin(2 * np.pi * y / PERIOD)
    clearings = [(0.0, 2.0, 7.0), (xc(35.0), 35.0, 6.0), (0.0, 82.0, 6.0)]
    in_clear = lambda x, y, m=0.0: any(np.hypot(x - a, y - b) < r + m for a, b, r in clearings)
    trunks, segs, sphs, boxes = [], [], [], []       # trunk: [x, y, r, h, kind, tex]

    def free(x, y, r, gap=0.3): return all(np.hypot(x - t[0], y - t[1]) > r + t[2] + gap for t in trunks)
    def tkind(y): return 'trunk_plain' if zone_of(y) == 'lowtex' else 'trunk'

    # 1. trunks lining both corridor edges, varying spacing
    for side in (-1, 1):
        y = 8 + rng.uniform(0, cfg['edge_spacing'])
        while y < 76:
            r = rng.uniform(0.12, 0.30); x = xc(y) + side * (hw + r + rng.uniform(0.05, 0.5))
            if zone_of(y) in ('trunks', 'branches', 'lowtex') and not in_clear(x, y, 0.5) and free(x, y, r, 0.15):
                trunks.append([x, y, r, rng.uniform(7, 12), tkind(y), 'edge'])
            y += cfg['edge_spacing'] * rng.uniform(0.6, 1.4)
    # 2. trunks standing inside the corridor (zones trunks + lowtex), always leaving >= 1.1 m on one side
    for zn in ('trunks', 'lowtex'):
        _, a, b = next(z for z in ZONES if z[0] == zn); ys = []
        for _ in range(rng.poisson(cfg['intruders_per_10m'] * (b - a) / 10)):
            for _try in range(30):
                y = rng.uniform(a + 1, b - 1); r = rng.uniform(0.12, 0.25)
                lo, hi = max(0.0, 1.1 + r - hw), min(hw - r, r + 0.5)    # stands in the drone's straight path
                if lo > hi or in_clear(xc(y), y, 0.5) or any(abs(y - q) < 4 for q in ys): continue
                x = xc(y) + rng.choice([-1, 1]) * rng.uniform(lo, hi)
                if free(x, y, r, 0.9): trunks.append([x, y, r, rng.uniform(7, 12), tkind(y), 'intruder']); ys.append(y); break
    # 3. background forest
    area = (FIELD['xmax'] - FIELD['xmin']) * (FIELD['ymax'] - FIELD['ymin'])
    for _ in range(rng.poisson(cfg['bg_density'] * area * 1.6)):
        x, y = rng.uniform(FIELD['xmin'], FIELD['xmax']), rng.uniform(FIELD['ymin'], FIELD['ymax']); r = rng.uniform(0.1, 0.4)
        if abs(x - xc(y)) < hw + r + 0.6 and -6 <= y <= 90: continue
        if in_clear(x, y, r) or not free(x, y, r, 0.3): continue
        trunks.append([x, y, r, rng.uniform(7, 13), tkind(y), 'bg'])
        if sum(t[5] == 'bg' for t in trunks) >= cfg['bg_density'] * area: break
    # 4. low-texture zone: featureless rock wall + boulder jutting into the corridor from opposite sides
    side = rng.choice([-1, 1]); jut = cfg['rock_jut'] * cfg['width']; y0, y1 = 63.0, 70.0; ym = (y0 + y1) / 2
    inner = xc(ym) + side * (hw - jut); outer = inner + side * (jut + 1.2)
    box = [(inner + outer) / 2, ym, 1.6, abs(outer - inner), y1 - y0, 3.2, 'rock']; boxes.append(box)
    br = 0.8; bx = xc(73.5) - side * (hw - 0.8 * jut + br); sphs.append([bx, 73.5, 1.4, br, 'boulder'])
    def hits_rock(t):
        return (abs(t[0] - box[0]) < box[3] / 2 + t[2] + 0.2 and abs(t[1] - box[1]) < box[4] / 2 + t[2] + 0.2) or np.hypot(t[0] - bx, t[1] - 73.5) < br + t[2] + 0.2
    trunks = [t for t in trunks if not hits_rock(t)]
    # 5. branches zone: low branches + twigs + leaf clumps reaching into the corridor; branches on background trunks too
    for t in trunks:
        x, y, r, h, k, role = t
        if zone_of(y) != 'branches': continue
        if role == 'edge' and rng.random() < cfg['branch_prob']:
            sd = 1 if x > xc(y) else -1; z = rng.uniform(1.15, 1.9); ye = y + rng.uniform(-0.6, 0.6)
            reach = cfg['branch_reach'] * cfg['width'] * rng.uniform(0.7, 1.0)
            a_ = np.array([x - sd * r * 0.8, y, z]); e_ = np.array([xc(ye) + sd * (hw - reach), ye, z + rng.uniform(-0.1, 0.3)])
            segs.append([*a_, *e_, rng.uniform(0.02, 0.05), 'branch'])
            m = (a_ + e_) / 2; d = rng.normal(size=3) * [0.4, 0.4, 0.15]; segs.append([*m, *(m + d), 0.012, 'branch'])
            if rng.random() < 0.7: sphs.append([*e_, rng.uniform(0.15, 0.30), 'foliage'])
        elif role == 'bg':
            for _ in range(rng.integers(0, 3)):
                az = rng.uniform(0, 2 * np.pi); z = rng.uniform(0.8, 2.6); L = rng.uniform(0.5, 1.5)
                a_ = np.array([x + r * np.cos(az), y + r * np.sin(az), z]); e_ = a_ + [L * np.cos(az), L * np.sin(az), rng.uniform(-0.1, 0.4)]
                segs.append([*a_, *e_, rng.uniform(0.02, 0.04), 'branch'])
                if rng.random() < 0.5: sphs.append([*e_, rng.uniform(0.15, 0.35), 'foliage'])
    # 6. shrubs along the corridor edges (dense in the branches zone, sparse among the trunks)
    for zn, rate in (('branches', 1.0), ('trunks', 0.4)):
        _, a, b = next(z for z in ZONES if z[0] == zn)
        for _ in range(rng.poisson(cfg['shrubs_per_10m'] * rate * (b - a) / 10)):
            y = rng.uniform(a, b); sr = rng.uniform(0.4, 0.7); sd = rng.choice([-1, 1])
            x = xc(y) + sd * (hw + sr - rng.uniform(0.1, 0.8 * cfg['branch_reach'] * cfg['width']))
            if not in_clear(x, y, sr): sphs.append([x, y, rng.uniform(0.5, 1.1), sr, 'shrub'])
    # 7. canopy crowns (visual; far above the eye band)
    for t in trunks:
        if rng.random() < 0.75:
            cr = rng.uniform(1.2, 2.5); sphs.append([t[0] + rng.normal(0, .3), t[1] + rng.normal(0, .3), t[3] - 0.3 * cr, cr, 'crown'])
    return dict(trunks=trunks, segs=segs, sphs=sphs, boxes=boxes, xc=xc, clearings=clearings)


def obstacles_meta(L):
    return dict(cyl=[[round(t[0], 3), round(t[1], 3), round(t[2], 3), 0.0, round(t[3], 2), t[4]] for t in L['trunks']],
                seg=[[*(round(v, 3) for v in s[:7]), s[7]] for s in L['segs']],
                sph=[[*(round(v, 3) for v in s[:4]), s[4]] for s in L['sphs']],
                box=[[*(round(v, 3) for v in b[:6]), b[6]] for b in L['boxes']])


def passable(obs, cfg, xc, clearings, res=0.1):
    """BFS on a 0.1 m grid of the flight-height cross-section, inflated by the drone radius, restricted to the corridor
    and clearings. Returns (ok, narrowest free width along the corridor in m)."""
    xs = np.arange(FIELD['xmin'], FIELD['xmax'], res) + res / 2; ys = np.arange(-2, EXIT_Y + 0.5, res) + res / 2
    X, Y = np.meshgrid(xs, ys); allowed = np.abs(X - xc(Y)) <= cfg['width'] / 2 + 0.3
    for a, b, r in clearings: allowed |= np.hypot(X - a, Y - b) < r
    blocked = np.zeros_like(allowed); raw = np.zeros_like(allowed)
    for sh in obs.footprint(ALT - 0.15, ALT + 0.15):
        if sh[0] == 'c':
            _, x, y, r, _k = sh; m = np.hypot(X - x, Y - y); blocked |= m < r + BODY_R + 0.02; raw |= m < r
        else:
            _, x, y, sx, sy, _k = sh; dx = np.maximum(np.abs(X - x) - sx / 2, 0); dy = np.maximum(np.abs(Y - y) - sy / 2, 0)
            m = np.hypot(dx, dy); blocked |= m < BODY_R + 0.02; raw |= m <= 0
    ok = allowed & ~blocked; H, W = ok.shape
    i0, j0 = int((0 - ys[0]) / res), int((0 - xs[0]) / res); seen = np.zeros_like(ok); q = deque()
    if ok[i0, j0]: q.append((i0, j0)); seen[i0, j0] = True
    while q:
        i, j = q.popleft()
        if ys[i] >= EXIT_Y: break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < H and 0 <= b < W and ok[a, b] and not seen[a, b]: seen[a, b] = True; q.append((a, b))
    else:
        return False, 0.0
    corr = (np.abs(X - xc(Y)) <= cfg['width'] / 2 + 0.3) & (Y > 8) & (Y < 76)
    widths = [(corr[i] & ~raw[i]).sum() * res for i in range(H) if corr[i].any() and 8 < ys[i] < 76]
    return True, float(min(widths))


LEVELS = ['easy', 'medium', 'hard']


def level_cfg(level):
    """Continuous difficulty for curriculum training: 0 = easy, 1 = medium, 2 = hard, values in between interpolate
    every parameter linearly between the neighbouring presets (above 2 extrapolates the medium->hard step, capped)."""
    level = float(np.clip(level, 0.0, 2.5)); i = min(int(level), 1); f = level - i
    a, b = PRESETS[LEVELS[i]], PRESETS[LEVELS[i + 1]]
    cfg = {k: a[k] + f * (b[k] - a[k]) for k in a}
    cfg['width'] = max(cfg['width'], 1.6); cfg['edge_spacing'] = max(cfg['edge_spacing'], 1.0)
    return cfg


def make(preset='easy', seed=0, overrides=None, name=None, level=None, assets=True):
    """preset = easy|medium|hard, or level = continuous difficulty (see level_cfg) -> world ff_L<level*100>_<seed>."""
    if level is not None:
        cfg = level_cfg(level); preset = f'L{int(round(level * 100)):03d}'; cfg.update(overrides or {})
    else:
        cfg = dict(PRESETS[preset]); cfg.update(overrides or {})
    name = name or f'ff_{preset}_{seed}'; textures()
    for attempt in range(150):
        L = generate(cfg, seed, attempt); meta_obs = obstacles_meta(L); obs = Obstacles(dict(obstacles=meta_obs))
        ok, narrow = passable(obs, cfg, L['xc'], L['clearings'])
        if ok: break
    else:
        raise RuntimeError('no passable layout in 150 attempts - loosen the parameters')
    xc = L['xc']
    # what a drone flying the centre line blind would touch (baseline difficulty)
    yy = np.arange(0, EXIT_Y, 0.05); route_hits, prev = [], None
    for y in yy:
        g, _, k, _ = obs.query([xc(y), y, ALT])
        if g < 0 and prev is None: route_hits.append([round(float(y), 1), k])
        prev = k if g < 0 else None
    zone_counts = {}
    for t in L['trunks']: zone_counts.setdefault(zone_of(t[1]), {}).setdefault(t[4], 0); zone_counts[zone_of(t[1])][t[4]] += 1
    stats = dict(attempt=attempt, trunks=len(L['trunks']), branches=sum(s[7] == 'branch' for s in L['segs']),
                 foliage=sum(s[4] == 'foliage' for s in L['sphs']), shrubs=sum(s[4] == 'shrub' for s in L['sphs']),
                 narrowest_gap_m=round(narrow, 2), centreline_hits=len(route_hits), centreline_hit_kinds=route_hits)
    if assets:
        # meshes, grouped by material (one draw call each)
        d = OUT / name / 'meshes'; d.mkdir(parents=True, exist_ok=True); meshes = {}
        def m(k): return meshes.setdefault(k, Mesh())
        for i, t in enumerate(L['trunks']):
            tex = 'plain' if t[4] == 'trunk_plain' else f'bark{i % 4}'
            m(tex).cylinder([t[0], t[1], -0.1], [t[0], t[1], t[3]], t[2], seg=14 if t[2] > 0.2 else 10, rep=1.0, vs=0.5)
        for s in L['segs']: m('bark0').cylinder(s[:3], s[3:6], s[6], seg=6, rep=0.25, vs=0.5)
        for s in L['sphs']:
            if s[4] == 'boulder': m('plain').sphere(s[:3], s[3], nu=24, nv=16)
            elif s[4] == 'shrub': m('leaves').sphere(s[:3], [s[3], s[3], s[3] * 0.9], nu=14, nv=10)
            elif s[4] == 'foliage': m('leaves').sphere(s[:3], s[3], nu=10, nv=6)
            else: m('crown').sphere(s[:3], [s[3], s[3], s[3] * 0.8], nu=12, nv=8)
        for b in L['boxes']: m('plain').box(b[:3], b[3:6])
        nverts = {k: v.write(d / f'{k}.obj') for k, v in meshes.items()}
        stats['vertices'] = int(sum(nverts.values()))
        visuals = []
        for k in meshes:
            if k == 'plain':
                mat = f'<material><ambient>{PLAIN} {PLAIN} {PLAIN} 1</ambient><diffuse>{PLAIN} {PLAIN} {PLAIN} 1</diffuse><specular>0 0 0 1</specular></material>'
            else:
                mat = (f'<material><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular><pbr><metal><albedo_map>{CONT}/textures/{k}.png</albedo_map>'
                       f'<roughness>1</roughness><metalness>0</metalness></metal></pbr></material>')
            visuals.append(f'<visual name="{k}"><geometry><mesh><uri>file://{CONT}/{name}/meshes/{k}.obj</uri></mesh></geometry>{mat}</visual>')
        base = (PX / 'worlds/default.sdf').read_text(encoding='utf-8-sig').replace('<world name="default">', f'<world name="{name}">')
        base = base.replace('<shadows>true</shadows>', '<shadows>false</shadows>').replace('<cast_shadows>true</cast_shadows>', '<cast_shadows>false</cast_shadows>')
        model = f'<model name="forest"><static>true</static><pose>0 0 0 0 0 0</pose><link name="l">\n  ' + '\n  '.join(visuals) + '\n</link></model>'
        (PX / f'worlds/{name}.sdf').write_text(base.replace('</world>', model + '\n  </world>'))
    meta = dict(name=name, preset=preset, seed=seed, cfg=cfg, alt=ALT, zones=ZONES, field=FIELD, exit_y=EXIT_Y, curve=dict(amp=cfg['curve_amp'], period=PERIOD),
                clearings=[[round(float(v), 3) for v in c] for c in L['clearings']], stats=stats, obstacles=meta_obs)
    (OUT / f'{name}.json').write_text(json.dumps(meta, separators=(',', ':')))
    if assets: plot_map(meta, obs, OUT / f'{name}_map.png')
    print(name, json.dumps(stats))
    return meta


def preview_world(name, views=((0, 6, 0), (0, 41, 0), (0, 58, 0))):
    """<name>_preview.sdf: the same world plus static 640x360 cameras at eye height looking down the corridor
    (topics /preview/cam<i>), to look at what the eye band sees. Usage: WORLD=<name>_preview start_sim.sh + px4/grab_preview.py"""
    meta = json.loads((OUT / f'{name}.json').read_text()); xc = lambda y: meta['curve']['amp'] * np.sin(2 * np.pi * y / meta['curve']['period'])
    s = (PX / f'worlds/{name}.sdf').read_text().replace(f'<world name="{name}">', f'<world name="{name}_preview">'); cams = []
    for i, (dx, y, yaw) in enumerate(views):
        x = xc(y) + dx; yaw = np.arctan2(xc(y + 3) - xc(y), 3) if yaw == 0 else yaw   # yaw from +y axis
        cams.append(f'''<model name="pcam{i}"><static>true</static><pose>{x:.2f} {y:.2f} {ALT} 0 0 {np.pi / 2 - yaw:.3f}</pose><link name="l">
  <sensor name="c" type="camera"><camera><horizontal_fov>1.6</horizontal_fov><image><width>640</width><height>360</height></image><clip><near>0.05</near><far>120</far></clip></camera>
  <always_on>1</always_on><update_rate>2</update_rate><topic>preview/cam{i}</topic></sensor></link></model>''')
    (PX / f'worlds/{name}_preview.sdf').write_text(s.replace('</world>', '\n'.join(cams) + '\n  </world>'))


COLORS = dict(trunk='#5b4a3a', trunk_plain='#b8b8b8', branch='#8a6a3a', foliage='#4f8a3a', shrub='#2f6a2a', rock='#9a9a9a', boulder='#9a9a9a', crown='#cfe3c0')


def plot_map(meta, obs, path, paths=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    fig, ax = plt.subplots(figsize=(5.5, 13)); xc = lambda y: meta['curve']['amp'] * np.sin(2 * np.pi * y / meta['curve']['period'])
    shade = dict(trunks='#f3efe8', branches='#eef5ea', lowtex='#efefef')
    for n, a, b in meta['zones']:
        if n in shade: ax.add_patch(Rectangle((meta['field']['xmin'], a), 32, b - a, color=shade[n], zorder=0))
        ax.text(meta['field']['xmin'] + 0.3, (max(a, -4) + b) / 2, n, fontsize=8, rotation=90, va='center', color='#555')
    for a, b, r in meta['clearings']: ax.add_patch(Circle((a, b), r, fill=False, ls=':', color='#999'))
    yy = np.linspace(-2, meta['exit_y'], 300); w = meta['cfg']['width'] / 2
    ax.plot(xc(yy) - w, yy, color='#c9a', lw=.6); ax.plot(xc(yy) + w, yy, color='#c9a', lw=.6); ax.plot(xc(yy), yy, color='#c9a', lw=.5, ls='--')
    for sh in obs.footprint(meta['alt'] - 0.15, meta['alt'] + 0.15):
        if sh[0] == 'c': ax.add_patch(Circle(sh[1:3], sh[3], color=COLORS[sh[4]], lw=0))
        else: ax.add_patch(Rectangle((sh[1] - sh[3] / 2, sh[2] - sh[4] / 2), sh[3], sh[4], color=COLORS[sh[5]]))
    for lab, (px, py, col, cr) in (paths or {}).items():
        ax.plot(px, py, color=col, lw=1.1, label=lab)
        if cr: ax.scatter(*np.array(cr).T, marker='x', color=col, s=30, zorder=5)
    ax.axhline(meta['exit_y'], color='g', lw=.8, ls='--'); ax.set_aspect('equal'); ax.set_xlim(meta['field']['xmin'], meta['field']['xmax']); ax.set_ylim(-4, 86)
    s = meta['stats']; ax.set_title(f"{meta['name']}  width {meta['cfg']['width']} m  narrowest {s['narrowest_gap_m']} m\n"
                                    f"{s['trunks']} trunks, {s['branches']} branches, {s['shrubs']} shrubs  (cross-section at {meta['alt']} m)", fontsize=8)
    if paths: ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('preset', help='easy | medium | hard, or a continuous level like 1.25 (curriculum)')
    ap.add_argument('seed', type=int, nargs='?', default=0)
    ap.add_argument('--set', action='append', default=[], help='override, e.g. width=2.8'); ap.add_argument('--name')
    a = ap.parse_args(); ovr = {k: float(v) for k, v in (s.split('=') for s in a.set)}
    if a.preset in PRESETS: make(a.preset, a.seed, ovr, a.name)
    else: make(seed=a.seed, overrides=ovr, name=a.name, level=float(a.preset))
