"""Forest world for the fast sim: obstacle list (px4/forest/<world>.json, px4/forest_geom.py format), the procedural
textures as grey mip pyramids, and the mission constants. numpy only."""
import json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from px4.forest_geom import Obstacles  # noqa: E402

TEXD = ROOT / 'px4/forest/textures'
TEX_NAMES = ['bark0', 'bark1', 'bark2', 'bark3', 'leaves', 'crown']
PLAIN = 0.66                      # make_forest.PLAIN: grey of trunk_plain, rock, boulder (untextured material)


class Textures:
    """Grey albedo (mean of RGB, 0..1) of every texture as a box-filtered mip pyramid, flattened into one array so a
    vectorised lookup is one fancy-index: value = flat[off[t, l] + iv * w[t, l] + iu]."""
    NL = 10

    def __init__(self):
        flat, off, ww, hh, pos = [], np.zeros((len(TEX_NAMES), self.NL), np.int64), np.zeros((len(TEX_NAMES), self.NL), np.int64), \
            np.zeros((len(TEX_NAMES), self.NL), np.int64), 0
        self.mean = np.zeros(len(TEX_NAMES))
        for t, n in enumerate(TEX_NAMES):
            a = load_grey(n)
            self.mean[t] = a.mean()
            for l in range(self.NL):
                off[t, l], hh[t, l], ww[t, l] = pos, a.shape[0], a.shape[1]; flat.append(a.ravel()); pos += a.size
                h2, w2 = max(1, a.shape[0] // 2), max(1, a.shape[1] // 2)
                a = a[:h2 * (a.shape[0] // h2), :w2 * (a.shape[1] // w2)].reshape(h2, a.shape[0] // h2, w2, a.shape[1] // w2).mean((1, 3))
        self.flat, self.off, self.w, self.h = np.concatenate(flat).astype(np.float32), off, ww, hh
        self.base_w, self.base_h = ww[:, 0].astype(float), hh[:, 0].astype(float)

    def sample(self, tex, texels, u, v):
        """tex: texture ids, texels: footprint of one pixel in level-0 texels (picks the mip level), u, v: texture
        coordinates in repeats (wrap)."""
        lvl = np.clip(np.round(np.log2(np.maximum(texels, 1.0))), 0, self.NL - 1).astype(np.int64)
        w, h = self.w[tex, lvl], self.h[tex, lvl]
        iu = np.floor(u * w).astype(np.int64) % w; iv = np.floor(v * h).astype(np.int64) % h
        return self.flat[self.off[tex, lvl] + iv * w + iu]


_TEX = None
CACHE = Path(__file__).resolve().parent / 'cache'


def load_grey(name, tries=10):
    """Grey texture (mean RGB / 255). px4/make_forest.py rewrites the PNGs whenever it builds a world (same content,
    seed 1234), so a read can hit a half-written file: retry, and fall back to the last good copy in fastsim/cache/."""
    CACHE.mkdir(exist_ok=True); cf = CACHE / f'{name}.npy'
    for i in range(tries):
        try:
            a = np.asarray(Image.open(TEXD / f'{name}.png').convert('RGB'), np.float64)[..., :3].mean(2) / 255.0
            try:
                tmp = cf.with_suffix(f'.{os.getpid()}.npy'); np.save(tmp, a); os.replace(tmp, cf)     # atomic
            except OSError: pass
            return a
        except Exception:
            if cf.exists() and i >= 2:
                try: return np.load(cf)
                except Exception: pass
            time.sleep(0.3)
    raise RuntimeError(f'cannot read texture {name}')


def textures():
    global _TEX
    if _TEX is None: _TEX = Textures()
    return _TEX


class World:
    """Everything one forest needs: meta (mission), Obstacles (collision truth, from px4/forest_geom.py) and
    render arrays (texture id / kind per obstacle)."""

    def __init__(self, name):
        self.name = name
        self.meta = json.loads((ROOT / f'px4/forest/{name}.json').read_text())
        self.obs = Obstacles(self.meta)
        o = self.obs
        # trunks: texture bark{i % 4} by index in the trunk list (= cyl order), plain material for trunk_plain
        self.cyl_tex = np.array([-1 if k == 'trunk_plain' else i % 4 for i, k in enumerate(o.cyl_k)], np.int64)
        # spheres: shrub ellipsoid z 0.9, crown z 0.8; leaves / crown textures; boulder plain
        self.sph_kz = np.array([0.9 if k == 'shrub' else 0.8 if k == 'crown' else 1.0 for k in o.sph_k])
        self.sph_tex = np.array([4 if k in ('shrub', 'foliage') else 5 if k == 'crown' else -1 for k in o.sph_k], np.int64)
        m = self.meta
        self.alt, self.exit_y, self.field = m['alt'], m['exit_y'], m['field']
        self.amp, self.period = m['curve']['amp'], m['curve']['period']

    def xc(self, y): return self.amp * np.sin(2 * np.pi * y / self.period)

    def zone_of(self, y): return next((n for n, a, b in self.meta['zones'] if a <= y < b), 'outside')
