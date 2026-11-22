"""Collision truth for N drones at once: the formulas of px4/forest_geom.Obstacles.query (same body ellipsoid, same
gap / surface distance), with candidates from a uniform 1 m grid instead of a per-drone distance search.

Each grid cell lists every primitive whose 2D bounding circle comes within INCL m of the cell, padded to a dense
(cells, K) index array, so a query is a handful of array ops over (N, K). Exact (identical to Obstacles.query) whenever
the true gap is below ~INCL - 0.7 m; farther obstacles may be missed, which only affects clearance statistics above
that distance, never contact detection."""
import numpy as np
from px4.forest_geom import r_body

CELL, INCL = 1.0, 4.0


class BatchTruth:
    def __init__(self, obs):
        self.o = obs; F = np.array([np.r_[obs.cyl[:, :2].min(0), obs.cyl[:, :2].max(0)]]) if len(obs.cyl) else None
        allxy = np.vstack([a for a in (obs.cyl[:, :2], obs.seg[:, :2], obs.seg[:, 3:5], obs.sph[:, :2], obs.box[:, :2]) if len(a)])
        self.x0, self.y0 = allxy.min(0) - INCL - 2; x1, y1 = allxy.max(0) + INCL + 2
        self.nx, self.ny = int(np.ceil((x1 - self.x0) / CELL)), int(np.ceil((y1 - self.y0) / CELL))
        cx = self.x0 + (np.arange(self.nx) + .5) * CELL; cy = self.y0 + (np.arange(self.ny) + .5) * CELL
        CX, CY = np.meshgrid(cx, cy, indexing='ij'); CX, CY = CX.ravel(), CY.ravel(); half = CELL * np.sqrt(2) / 2
        def table(xy, rad):
            if len(xy) == 0: return np.zeros((len(CX), 0), np.int64)
            near = np.hypot(CX[:, None] - xy[None, :, 0], CY[:, None] - xy[None, :, 1]) - half - rad[None] < INCL
            K = max(1, int(near.sum(1).max())); T = np.full((len(CX), K), -1, np.int64)
            for i in range(len(CX)):
                idx = np.where(near[i])[0]; T[i, :len(idx)] = idx
            return T
        smid = (obs.seg[:, :2] + obs.seg[:, 3:5]) / 2 if len(obs.seg) else np.zeros((0, 2))
        shalf = np.hypot(*(obs.seg[:, 3:5] - obs.seg[:, :2]).T) / 2 + obs.seg[:, 6] if len(obs.seg) else np.zeros(0)
        self.Tc = table(obs.cyl[:, :2], obs.cyl[:, 2]); self.Ts = table(smid, shalf)
        self.Th = table(obs.sph[:, :2], obs.sph[:, 3]); self.Tb = table(obs.box[:, :2], np.hypot(obs.box[:, 3], obs.box[:, 4]) / 2 if len(obs.box) else np.zeros(0))
        self.kinds = [obs.cyl_k, obs.seg_k, obs.sph_k, obs.box_k]; self.prefix = 'cshb'

    def cell(self, p):
        i = np.clip(((p[:, 0] - self.x0) / CELL).astype(int), 0, self.nx - 1); j = np.clip(((p[:, 1] - self.y0) / CELL).astype(int), 0, self.ny - 1)
        return i * self.ny + j

    def query(self, p):
        """p (N, 3) -> gap (N,), surface distance (N,), type (N,) 0..3 cyl/seg/sph/box (-1 none), index (N,)"""
        p = np.asarray(p, float).reshape(-1, 3); N = len(p); ci = self.cell(p); o = self.o
        G = np.full((4, N), np.inf); D = np.full((4, N), np.inf); I = np.full((4, N), -1, np.int64); ar = np.arange(N)
        px, py, pz = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        if self.Tc.shape[1]:
            idx = self.Tc[ci]; v = idx >= 0; c = o.cyl[np.maximum(idx, 0)]
            dh = np.hypot(c[..., 0] - px, c[..., 1] - py) - c[..., 2]
            dz = np.maximum(np.maximum(c[..., 3] - pz, 0.0), pz - c[..., 4])
            d = np.hypot(np.maximum(dh, 0), dz); d = np.where((dh < 0) & (dz == 0), dh, d)
            g = np.where(v, d - r_body(np.maximum(dh, 0), dz), np.inf); k = np.argmin(g, 1)
            G[0], D[0], I[0] = g[ar, k], d[ar, k], np.where(v[ar, k], idx[ar, k], -1)
        if self.Ts.shape[1]:
            idx = self.Ts[ci]; v = idx >= 0; s = o.seg[np.maximum(idx, 0)]; a, e = s[..., :3], s[..., 3:6]; ab = e - a
            t = np.clip(((p[:, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1); q = a + t[..., None] * ab
            w = p[:, None] - q; d = np.linalg.norm(w, axis=-1) - s[..., 6]
            g = np.where(v, d - r_body(np.hypot(w[..., 0], w[..., 1]), w[..., 2]), np.inf); k = np.argmin(g, 1)
            G[1], D[1], I[1] = g[ar, k], d[ar, k], np.where(v[ar, k], idx[ar, k], -1)
        if self.Th.shape[1]:
            idx = self.Th[ci]; v = idx >= 0; s = o.sph[np.maximum(idx, 0)]; w = p[:, None] - s[..., :3]; d = np.linalg.norm(w, axis=-1) - s[..., 3]
            g = np.where(v, d - r_body(np.hypot(w[..., 0], w[..., 1]), w[..., 2]), np.inf); k = np.argmin(g, 1)
            G[2], D[2], I[2] = g[ar, k], d[ar, k], np.where(v[ar, k], idx[ar, k], -1)
        if self.Tb.shape[1]:
            idx = self.Tb[ci]; v = idx >= 0; b = o.box[np.maximum(idx, 0)]
            q = np.clip(p[:, None], b[..., :3] - b[..., 3:6] / 2, b[..., :3] + b[..., 3:6] / 2); w = p[:, None] - q
            d = np.linalg.norm(w, axis=-1); g = np.where(v, d - r_body(np.hypot(w[..., 0], w[..., 1]), w[..., 2]), np.inf); k = np.argmin(g, 1)
            G[3], D[3], I[3] = g[ar, k], d[ar, k], np.where(v[ar, k], idx[ar, k], -1)
        ty = np.argmin(G, 0); gap = G[ty, ar]; ty = np.where(np.isfinite(gap), ty, -1)
        return gap, np.where(ty >= 0, D[np.maximum(ty, 0), ar], np.inf), ty, np.where(ty >= 0, I[np.maximum(ty, 0), ar], -1)

    def label(self, ty, i):
        """-> (kind, oid) in forest_geom's format ('trunk', 'c12')"""
        if ty < 0: return None, None
        return self.kinds[ty][i], f'{self.prefix[ty]}{i}'
