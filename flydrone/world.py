"""2D top-down world: a walled arena with a 'forest' of textured cylinders, and a drone seen as a point.

The drone's only vision is a ring of ommatidia (like a fly's compound eye). Each one reports the brightness of the
surface its ray hits: cylinders carry vertical stripes, walls carry a stripe pattern along their length. No depth
is given to the bio controller; depth (ray distance) is exposed only for the oracle baseline and for metrics.
"""
import numpy as np


class World:
    def __init__(self, seed=0, size=40.0, n_obstacles=60, r_range=(0.3, 0.9), clear_start=3.0):
        rng = np.random.default_rng(seed)
        self.size = size
        self.start = np.array([size / 2, 2.0])
        centers, radii = [], []
        while len(centers) < n_obstacles:
            c = rng.uniform(1.5, size - 1.5, 2)
            r = rng.uniform(*r_range)
            if np.linalg.norm(c - self.start) < clear_start + r:
                continue
            if any(np.linalg.norm(c - c2) < r + r2 + 0.6 for c2, r2 in zip(centers, radii)):
                continue
            centers.append(c); radii.append(r)
        self.c = np.array(centers) if centers else np.zeros((0, 2))
        self.r = np.array(radii) if radii else np.zeros(0)
        self.k = rng.integers(3, 7, len(self.r))           # stripes around each cylinder (coarse enough not to alias)
        self.ph = rng.uniform(0, 2 * np.pi, len(self.r))

    def cast(self, p, angles):
        """Rays from p at world angles -> (distance, brightness) per ray."""
        d = np.stack([np.cos(angles), np.sin(angles)], 1)               # (R,2)
        dist = np.full(len(angles), np.inf); lum = np.full(len(angles), 0.5)
        # walls x=0, x=L, y=0, y=L
        L = self.size
        with np.errstate(divide='ignore', invalid='ignore'):
            for axis, plane in ((0, 0.0), (0, L), (1, 0.0), (1, L)):
                t = (plane - p[axis]) / d[:, axis]
                ok = (t > 1e-6) & (t < dist)
                hit = p + t[:, None] * d
                along = hit[:, 1 - axis]
                dist = np.where(ok, t, dist)
                lum = np.where(ok, 0.5 + 0.35 * np.sin(2 * np.pi * along / 1.2 + 3 * axis + plane), lum)
        if len(self.r):
            oc = p - self.c                                               # (C,2)
            with np.errstate(all='ignore'):
                b = d @ oc.T                                              # (R,C)
            cc = (oc ** 2).sum(1) - self.r ** 2                           # (C,)
            disc = b ** 2 - cc
            with np.errstate(invalid='ignore'):
                t = -b - np.sqrt(disc)
            t = np.where((disc > 0) & (t > 1e-6), t, np.inf)
            j = t.argmin(1); tj = t[np.arange(len(angles)), j]
            ok = tj < dist
            hit = p + tj[:, None] * d
            ang = np.arctan2(hit[:, 1] - self.c[j, 1], hit[:, 0] - self.c[j, 0])
            dist = np.where(ok, tj, dist)
            lum = np.where(ok, 0.5 + 0.4 * np.sin(self.k[j] * ang + self.ph[j]), lum)
        return dist, lum

    def collides(self, p, radius):
        L = self.size
        if p[0] < radius or p[1] < radius or p[0] > L - radius or p[1] > L - radius:
            return True
        return bool(len(self.r) and (np.linalg.norm(self.c - p, axis=1) < self.r + radius).any())
