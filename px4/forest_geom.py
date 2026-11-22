"""Obstacle geometry of the forest test field (numpy only, shared by the generator on the host and the bridge in the
container). Obstacles have no collision shapes in Gazebo; crashes and clearance are computed from this list.

Drone body = flat ellipsoid, horizontal radius BODY_R (props), half height BODY_H.
For a drone centre p the nearest obstacle surface point q gives  gap = |p - q| - r_body(elevation of p - q);
gap < 0 -> contact (crash)."""
import json
import numpy as np

BODY_R, BODY_H = 0.35, 0.15
KINDS = ['trunk', 'trunk_plain', 'branch', 'foliage', 'shrub', 'rock', 'boulder', 'crown']


def r_body(dh, dz):
    th = np.arctan2(np.abs(dz), np.maximum(dh, 1e-9))
    return 1.0 / np.sqrt(np.cos(th) ** 2 / BODY_R ** 2 + np.sin(th) ** 2 / BODY_H ** 2)


class Obstacles:
    def __init__(self, meta):
        if isinstance(meta, str): meta = json.loads(open(meta).read())
        def arr(key, n):
            rows = meta['obstacles'][key]
            return np.array([r[:n] for r in rows], float).reshape(-1, n), [r[n] for r in rows]
        self.cyl, self.cyl_k = arr('cyl', 5)          # x y r z0 z1
        self.seg, self.seg_k = arr('seg', 7)          # x0 y0 z0 x1 y1 z1 r
        self.sph, self.sph_k = arr('sph', 4)          # x y z r
        self.box, self.box_k = arr('box', 6)          # cx cy cz sx sy sz (axis aligned)
        self._near_at, self._near = None, None

    def _candidates(self, p):
        """Obstacles whose footprint is within 5 m (cached, refreshed after moving 1 m)."""
        if self._near_at is None or np.hypot(*(p[:2] - self._near_at)) > 1.0:
            R = 5.0
            c = np.where(np.hypot(self.cyl[:, 0] - p[0], self.cyl[:, 1] - p[1]) - self.cyl[:, 2] < R)[0]
            mid = (self.seg[:, :2] + self.seg[:, 3:5]) / 2; half = np.hypot(*(self.seg[:, 3:5] - self.seg[:, :2]).T) / 2
            s = np.where(np.hypot(mid[:, 0] - p[0], mid[:, 1] - p[1]) - half < R)[0]
            h = np.where(np.hypot(self.sph[:, 0] - p[0], self.sph[:, 1] - p[1]) - self.sph[:, 3] < R)[0]
            b = np.where(np.hypot(self.box[:, 0] - p[0], self.box[:, 1] - p[1]) - np.hypot(self.box[:, 3], self.box[:, 4]) / 2 < R)[0]
            self._near_at, self._near = p[:2].copy(), (c, s, h, b)
        return self._near

    def query(self, p):
        """-> (gap, surface distance, kind, obstacle id) of the obstacle with the smallest gap to the drone body at p=(x,y,z)."""
        p = np.asarray(p, float); c, s, h, b = self._candidates(p); best = (np.inf, np.inf, None, None)
        if len(c):
            o = self.cyl[c]; dh = np.hypot(o[:, 0] - p[0], o[:, 1] - p[1]) - o[:, 2]
            dz = np.maximum.reduce([o[:, 3] - p[2], np.zeros(len(o)), p[2] - o[:, 4]])
            d = np.hypot(np.maximum(dh, 0), dz); d = np.where((dh < 0) & (dz == 0), dh, d)
            g = d - r_body(np.maximum(dh, 0), dz); i = int(np.argmin(g))
            if g[i] < best[0]: best = (float(g[i]), float(d[i]), self.cyl_k[c[i]], f'c{c[i]}')
        if len(s):
            o = self.seg[s]; a, e = o[:, :3], o[:, 3:6]; ab = e - a
            t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1); q = a + t[:, None] * ab
            v = p - q; d = np.linalg.norm(v, axis=1) - o[:, 6]
            g = d - r_body(np.hypot(v[:, 0], v[:, 1]), v[:, 2]); i = int(np.argmin(g))
            if g[i] < best[0]: best = (float(g[i]), float(d[i]), self.seg_k[s[i]], f's{s[i]}')
        if len(h):
            o = self.sph[h]; v = p - o[:, :3]; d = np.linalg.norm(v, axis=1) - o[:, 3]
            g = d - r_body(np.hypot(v[:, 0], v[:, 1]), v[:, 2]); i = int(np.argmin(g))
            if g[i] < best[0]: best = (float(g[i]), float(d[i]), self.sph_k[h[i]], f'h{h[i]}')
        if len(b):
            o = self.box[b]; q = np.clip(p, o[:, :3] - o[:, 3:6] / 2, o[:, :3] + o[:, 3:6] / 2); v = p - q
            d = np.linalg.norm(v, axis=1); g = d - r_body(np.hypot(v[:, 0], v[:, 1]), v[:, 2]); i = int(np.argmin(g))
            if g[i] < best[0]: best = (float(g[i]), float(d[i]), self.box_k[b[i]], f'b{b[i]}')
        return best

    def footprint(self, z0, z1):
        """2D cross-section of everything that overlaps the height band [z0, z1]: list of (kind, shape...) for plotting /
        passability. Segments are sampled into circles."""
        out = []
        for (x, y, r, a, e), k in zip(self.cyl, self.cyl_k):
            if a < z1 and e > z0: out.append(('c', x, y, r, k))
        for (x0, y0, za, x1, y1, zb, r), k in zip(self.seg, self.seg_k):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 0.05))
            for t in np.linspace(0, 1, n):
                z = za + t * (zb - za)
                if z0 - r < z < z1 + r: out.append(('c', x0 + t * (x1 - x0), y0 + t * (y1 - y0), r, k))
        for (x, y, z, r), k in zip(self.sph, self.sph_k):
            dz = 0 if z0 <= z <= z1 else min(abs(z - z0), abs(z - z1))
            if dz < r: out.append(('c', x, y, np.sqrt(r * r - dz * dz), k))
        for (x, y, z, sx, sy, sz), k in zip(self.box, self.box_k):
            if z - sz / 2 < z1 and z + sz / 2 > z0: out.append(('b', x, y, sx, sy, k))
        return out


class RangeSensor:
    """Forward 1D time-of-flight range finder (VL53L1X-like: ~27 deg cone, 4 m max) at flight height: 2D ray casts
    against the cross-section of all obstacles in [z - 0.2, z + 0.2] (the drone holds altitude within a few cm)."""
    def __init__(self, obs, z, fov_deg=27.0, max_r=4.0, n_rays=7):
        self.max_r = max_r; self.off = np.radians(np.linspace(-fov_deg / 2, fov_deg / 2, n_rays))
        fp = obs.footprint(z - 0.2, z + 0.2)
        self.C = np.array([[s[1], s[2], s[3]] for s in fp if s[0] == 'c'], float).reshape(-1, 3)
        self.B = np.array([[s[1] - s[3] / 2, s[2] - s[4] / 2, s[1] + s[3] / 2, s[2] + s[4] / 2] for s in fp if s[0] == 'b'], float).reshape(-1, 4)

    def read(self, x, y, h):
        """-> (min range over the cone, per-ray ranges); max_r when nothing is in range."""
        p = np.array([x, y]); ang = h + self.off; d = np.c_[np.cos(ang), np.sin(ang)]; t = np.full(len(ang), self.max_r)
        if len(self.C):
            C = self.C[np.hypot(self.C[:, 0] - x, self.C[:, 1] - y) < self.max_r + self.C[:, 2]]
            if len(C):
                rel = C[:, :2] - p; proj = d @ rel.T                                     # (rays, circles)
                perp2 = (rel ** 2).sum(1)[None] - proj ** 2; disc = C[:, 2][None] ** 2 - perp2
                hit = np.where((disc >= 0) & (proj > 0), proj - np.sqrt(np.maximum(disc, 0)), np.inf)
                t = np.minimum(t, np.maximum(hit.min(1), 0))
        if len(self.B):
            with np.errstate(divide='ignore', invalid='ignore'):
                inv = 1.0 / d
                t1 = (self.B[None, :, 0] - x) * inv[:, :1]; t2 = (self.B[None, :, 2] - x) * inv[:, :1]
                t3 = (self.B[None, :, 1] - y) * inv[:, 1:]; t4 = (self.B[None, :, 3] - y) * inv[:, 1:]
                tmin = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4)); tmax = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
                hit = np.where((tmax >= np.maximum(tmin, 0)), np.maximum(tmin, 0), np.inf)
            t = np.minimum(t, hit.min(1))
        return float(t.min()), t
