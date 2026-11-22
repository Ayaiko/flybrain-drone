"""Ray-cast 120-ommatidium eye: the same optics as the Gazebo eye in px4/fly_forest.py, without a renderer.

Gazebo eye (x500_flyeye model): 3 pinhole cameras 160 x 8 px, 110 deg horizontal FOV, yaw -110 / 0 / +110 deg, 0.18 m
above base_link, far clip 80 m (depth), near 0.05 m. The bridge averages each camera column over its 8 rows (= an
~8 deg elevation band at the image centre, narrower toward the image edges) and then each ommatidium averages the
columns within +-half a bin of its direction (fly_forest.Eye.W).

Here every camera column (480, exact azimuths of the real pixels) is ray-cast at `rows` elevations (pixel rows of the
real camera, default 4 of the 8 -> pairs of rows), against the obstacle list:
  vertical cylinders (trunks), capsules (branches), ellipsoids (foliage, shrubs, crowns, boulder), boxes (rock),
  the ground plane z = 0, and the background.
Brightness = texture albedo (mip-mapped, same procedural PNGs as Gazebo) x a lighting curve of n.L (sun direction
0.001 0.625 -0.78), plus constants for untextured grey, ground and background. The lighting curve and the constants are
fitted to real Gazebo frames by fastsim/validate_eye.py and stored in fastsim/eye_calib.json; defaults below are the
SDF values (ambient 0.4, sun 0.904, background 0.7, ground 0.8) for an uncalibrated start."""
import json
from pathlib import Path
import numpy as np
from .world import textures, PLAIN
from . import kernel

HERE = Path(__file__).resolve().parent
CALIB_FILE = HERE / 'eye_calib.json'
CAM_W, CAM_H, CAM_FOV, CAM_YAWS, EYE_DZ, FAR, NEAR = 160, 8, np.radians(110), np.radians([-110, 0, 110]), 0.18, 80.0, 0.05
SUN = np.array([0.001, 0.625, -0.78]); LDIR = -SUN / np.linalg.norm(SUN)          # direction toward the sun
KNOTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])                                       # lighting curve knots in n.L
SKY, GROUND, TEX, PLN = 0, 1, 2, 3
BRANCH_CULL = 25.0                                                                  # branches (r <= 5 cm) farther than this are sub-pixel


def make_eye(n=120, fov_deg=330):                                                  # = flydrone.sim.make_eye
    return np.radians(np.linspace(-fov_deg / 2, fov_deg / 2, n))


def default_calib():
    amb, sun = 0.4, 0.904
    return dict(sky=0.7, ground=0.8 * (amb + sun * LDIR[2]), tex=list(amb + sun * KNOTS), plain=list(PLAIN * (amb + sun * KNOTS)),
                source='SDF defaults (uncalibrated)')


def load_calib():
    return json.loads(CALIB_FILE.read_text()) if CALIB_FILE.exists() else default_calib()


class RayEye:
    def __init__(self, world, rows=4, calib=None, n_omm=120):
        self.world, self.tx, self.rows = world, textures(), rows
        f = (CAM_W / 2) / np.tan(CAM_FOV / 2); u = np.arange(CAM_W) + .5; xo = CAM_W / 2 - u
        off = np.arctan(xo / f)
        self.az = np.concatenate([y + off for y in CAM_YAWS])                      # 480 column azimuths rel. heading (+ = left)
        self.cosoff = np.tile(np.cos(off), 3)                                       # depth = horizontal t * cos(offset from cam axis)
        vr = (np.arange(rows) + .5) * CAM_H / rows                                  # sample rows in pixel units
        self.tan_el = np.tile((CAM_H / 2 - vr)[None, :] / np.sqrt(f ** 2 + xo ** 2)[:, None], (3, 1))   # (480, rows)
        self.dcol = np.tile(f / (f ** 2 + xo ** 2), 3)                              # column angular width (rad / px)
        self.drow = np.tile(1.0 / np.sqrt(f ** 2 + xo ** 2), 3)                     # pixel angular height
        self.eye = make_eye(n_omm); half = (self.eye[1] - self.eye[0]) / 2
        W = np.array([(np.abs(self.az - e) <= half).astype(float) for e in self.eye]); self.W = W / W.sum(1, keepdims=True)
        self.n_col = len(self.az)
        self.set_calib(calib or load_calib())
        o = world.obs
        self.cyl = o.cyl; self.sph = o.sph; self.seg = o.seg; self.box = o.box
        self.sph_R = np.c_[o.sph[:, 3], o.sph[:, 3], o.sph[:, 3] * world.sph_kz] if len(o.sph) else np.zeros((0, 3))
        self.tm = float(np.abs(self.tan_el).max())
        self.cam = kernel.make_cam(rows); self.scene = kernel.pack_scene(world, self.tx)
        assert np.allclose(self.cam.tan_el, self.tan_el) and np.allclose(self.cam.az, self.az)

    def set_calib(self, c):
        self.calib = c; self.k_tex = np.asarray(c['tex'], float); self.k_plain = np.asarray(c['plain'], float)
        self.sky_v, self.ground_v = float(c['sky']), float(c['ground'])

    # ------------------------------------------------------------------ ray casting
    def cast_ref(self, x, y, z, h):
        """Brute-force reference of cast() (every ray against every obstacle); slower, same result."""
        ex, ey, ez = x, y, z + EYE_DZ; R = self.rows; tx = self.tx
        a = h + self.az; ca, sa = np.cos(a), np.sin(a)
        K = self.n_col * R
        tan_el = self.tan_el                                                        # (C, R)
        best = np.full((self.n_col, R), np.inf); cls = np.zeros((self.n_col, R), np.int8)
        alb = np.zeros((self.n_col, R)); ndl = np.zeros((self.n_col, R))
        sec = np.sqrt(1 + tan_el ** 2)                                              # 3D distance per unit horizontal t

        # ground (z = 0) and background
        g = tan_el < 0
        tg = np.where(g, ez / np.maximum(-tan_el, 1e-9), np.inf)
        best = np.where(tg * self.cosoff[:, None] < FAR, tg, np.inf)
        cls[np.isfinite(best)] = GROUND

        # vertical cylinders (trunks): 2D ray-circle per column, z range per row
        c = self.cyl
        if len(c):
            mx, my = c[:, 0] - ex, c[:, 1] - ey; d2 = mx * mx + my * my
            keep = np.where((d2 < (FAR / 0.55 + 1) ** 2))[0]
            if len(keep):
                c = c[keep]; mx, my = mx[keep], my[keep]
                b = ca[:, None] * mx + sa[:, None] * my                              # (C, N)
                disc = b * b - (mx * mx + my * my - c[:, 2] ** 2)[None]
                t = np.where((disc > 0) & (b > 0), b - np.sqrt(np.maximum(disc, 0)), np.inf)
                t = np.where(t > NEAR, t, np.where((disc > 0) & (b > 0), NEAR, np.inf))   # inside a trunk: NEAR
                j = np.argmin(t, 1); tc = t[np.arange(self.n_col), j]               # nearest trunk per column
                zc = ez + tc[:, None] * tan_el                                      # (C, R)
                ok = np.isfinite(tc)[:, None] & (zc >= c[j, 3][:, None]) & (zc <= c[j, 4][:, None]) & \
                     ((tc * self.cosoff)[:, None] < FAR) & (tc[:, None] < best)
                if ok.any():
                    tcb = np.broadcast_to(tc[:, None], ok.shape); tc = np.where(np.isfinite(tc), tc, 0.0)
                    px, py = ex + tc * ca, ey + tc * sa
                    nx, ny = (px - c[j, 0]) / c[j, 2], (py - c[j, 1]) / c[j, 2]
                    nl = np.maximum(0, nx * LDIR[0] + ny * LDIR[1])
                    texid = self.world.cyl_tex[keep[j]]
                    cosi = np.maximum(np.abs(nx * ca + ny * sa), 0.15)
                    t3 = tc[:, None] * sec
                    plain = texid < 0; tid = np.maximum(texid, 0)
                    texu = t3 * (self.dcol / cosi)[:, None] * tx.base_w[tid][:, None] / (2 * np.pi * c[j, 2])[:, None]
                    texv = t3 * self.drow[:, None] * 0.5 * tx.base_h[tid][:, None]
                    uu = ((np.arctan2(ny, nx) - np.pi / 2) / (2 * np.pi)) % 1.0
                    val = tx.sample(np.broadcast_to(tid[:, None], ok.shape), np.maximum(texu, texv),
                                    np.broadcast_to(uu[:, None], ok.shape), 0.5 * zc)
                    val = np.where(plain[:, None], PLAIN, val)
                    best = np.where(ok, tcb, best); cls = np.where(ok, np.where(plain[:, None], PLN, TEX), cls).astype(np.int8)
                    alb = np.where(ok, val, alb); ndl = np.where(ok, np.broadcast_to(nl[:, None], ok.shape), ndl)

        # rays as 3D directions parametrised by horizontal distance t: p = e + t * (ca, sa, tan_el)
        Dx = np.broadcast_to(ca[:, None], (self.n_col, R)).ravel(); Dy = np.broadcast_to(sa[:, None], (self.n_col, R)).ravel()
        Dz = tan_el.ravel()
        best = best.ravel(); cls = cls.ravel(); alb = alb.ravel(); ndl = ndl.ravel()
        secf = sec.ravel(); colw = np.repeat(self.dcol, R); rowh = np.repeat(self.drow, R)

        # ellipsoids (foliage, shrubs, crowns, boulder): cull by distance and by the elevation band
        s = self.sph
        if len(s):
            dh = np.hypot(s[:, 0] - ex, s[:, 1] - ey); Rz = self.sph_R[:, 2]; reach = (dh + s[:, 3]) * self.tm
            keep = np.where((dh - s[:, 3] < FAR / 0.55) & (s[:, 2] - Rz - ez < reach) & (s[:, 2] + Rz - ez > -reach))[0]
            if len(keep):
                Rr = self.sph_R[keep]; o = (np.array([ex, ey, ez])[None] - s[keep, :3]) / Rr     # (M, 3)
                dx = Dx[:, None] / Rr[None, :, 0]; dy = Dy[:, None] / Rr[None, :, 1]; dz = Dz[:, None] / Rr[None, :, 2]   # (K, M)
                A = dx * dx + dy * dy + dz * dz; B = dx * o[:, 0] + dy * o[:, 1] + dz * o[:, 2]; Cq = (o * o).sum(1) - 1
                disc = B * B - A * Cq[None]
                t = np.where(disc > 0, (-B - np.sqrt(np.maximum(disc, 0))) / A, np.inf)
                t = np.where(t > NEAR, t, np.where((disc > 0) & ((-B + np.sqrt(np.maximum(disc, 0))) / A > NEAR), NEAR, np.inf))
                j = np.argmin(t, 1); ts = t[np.arange(K), j]
                ok = (ts < best) & np.isfinite(ts)
                if ok.any():
                    i = np.where(ok)[0]; jj = j[i]; k = keep[jj]; tt = ts[i]
                    q = (o[jj] + tt[:, None] * np.c_[Dx[i], Dy[i], Dz[i]] / Rr[jj])                  # unit-sphere point
                    n = q / Rr[jj]; n /= np.linalg.norm(n, axis=1, keepdims=True)
                    texid = self.world.sph_tex[k]; plain = texid < 0; tid = np.maximum(texid, 0)
                    r, rz = Rr[jj, 0], Rr[jj, 2]
                    lo = np.arctan2(q[:, 1], q[:, 0]) % (2 * np.pi); la = np.arcsin(np.clip(q[:, 2], -1, 1))
                    uu = lo / np.pi * np.maximum(1, r); vv = (la + np.pi / 2) / np.pi * np.maximum(1, rz)
                    d3 = np.c_[Dx[i], Dy[i], Dz[i]] / secf[i, None]
                    cosi = np.maximum(np.abs((n * d3).sum(1)), 0.15); t3 = tt * secf[i]
                    texu = t3 * colw[i] / cosi * tx.base_w[tid] * np.maximum(1, r) / (np.pi * r)
                    texv = t3 * rowh[i] / cosi * tx.base_h[tid] * np.maximum(1, rz) / (np.pi * rz)
                    val = np.where(plain, PLAIN, tx.sample(tid, np.maximum(texu, texv), uu, vv))
                    best[i] = tt; cls[i] = np.where(plain, PLN, TEX); alb[i] = val; ndl[i] = np.maximum(0, n @ LDIR)

        # branches (capsules rendered as thin bark cylinders): only near ones
        sg = self.seg
        if len(sg):
            mid = (sg[:, :3] + sg[:, 3:6]) / 2; half = np.linalg.norm(sg[:, 3:6] - sg[:, :3], axis=1) / 2
            keep = np.where(np.hypot(mid[:, 0] - ex, mid[:, 1] - ey) - half < BRANCH_CULL)[0]
            if len(keep):
                a0 = sg[keep, :3]; ax = sg[keep, 3:6] - a0; L = np.linalg.norm(ax, axis=1); ua = ax / L[:, None]; rr = sg[keep, 6]
                w = np.array([ex, ey, ez])[None] - a0                                  # (M, 3)
                dU = Dx[:, None] * ua[:, 0] + Dy[:, None] * ua[:, 1] + Dz[:, None] * ua[:, 2]   # (K, M)
                wU = (w * ua).sum(1)
                px_ = Dx[:, None] - dU * ua[:, 0]; py_ = Dy[:, None] - dU * ua[:, 1]; pz_ = Dz[:, None] - dU * ua[:, 2]
                wx, wy, wz = w[:, 0] - wU * ua[:, 0], w[:, 1] - wU * ua[:, 1], w[:, 2] - wU * ua[:, 2]
                A = px_ * px_ + py_ * py_ + pz_ * pz_; B = px_ * wx + py_ * wy + pz_ * wz; Cq = wx * wx + wy * wy + wz * wz - rr * rr
                disc = B * B - A * Cq[None]
                t = np.where((disc > 0) & (A > 1e-12), (-B - np.sqrt(np.maximum(disc, 0))) / np.maximum(A, 1e-12), np.inf)
                sax = wU[None] + t * dU
                t = np.where((t > NEAR) & (sax >= 0) & (sax <= L[None]), t, np.inf)
                j = np.argmin(t, 1); tb = t[np.arange(K), j]; ok = (tb < best) & np.isfinite(tb)
                if ok.any():
                    i = np.where(ok)[0]; jj = j[i]; tt = tb[i]
                    p = np.array([ex, ey, ez])[None] + tt[:, None] * np.c_[Dx[i], Dy[i], Dz[i]]
                    rel = p - a0[jj]; n = rel - (rel * ua[jj]).sum(1, keepdims=True) * ua[jj]; n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
                    best[i] = tt; cls[i] = TEX; alb[i] = tx.mean[0]; ndl[i] = np.maximum(0, n @ LDIR)

        # boxes (rock wall), axis aligned
        for bx in self.box:
            lo_, hi_ = bx[:3] - bx[3:6] / 2, bx[:3] + bx[3:6] / 2; e = np.array([ex, ey, ez])
            D = np.c_[Dx, Dy, Dz]
            with np.errstate(divide='ignore', invalid='ignore'):
                t1 = (lo_[None] - e[None]) / D; t2 = (hi_[None] - e[None]) / D
            tmin = np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2)); tmax = np.where(np.isnan(t1), np.inf, np.maximum(t1, t2))
            tn = tmin.max(1); tf = tmax.min(1); face = tmin.argmax(1)
            ok = (tn <= tf) & (tf > NEAR) & (np.maximum(tn, NEAR) < best)
            if ok.any():
                i = np.where(ok)[0]; f_ = face[i]; n = np.zeros((len(i), 3)); n[np.arange(len(i)), f_] = -np.sign(D[i, f_])
                best[i] = np.maximum(tn[i], NEAR); cls[i] = PLN; alb[i] = PLAIN; ndl[i] = np.maximum(0, n @ LDIR)

        # depth clip for everything (Gazebo far clip is camera depth)
        far = best * np.repeat(self.cosoff, R) >= FAR
        cls = np.where(far, SKY, cls)
        return cls, alb, ndl

    def cast(self, x, y, z, h):
        """-> per ray (480 columns x rows, column-major): class, albedo, n.L for one pose (base_link z). Uses the
        shared kernel (fastsim/kernel.py); cast_ref is the brute-force reference."""
        c, a, n = kernel.cast(self.scene, self.cam, [x], [y], [z + EYE_DZ], [h])
        return c[0], a[0], n[0]

    def render_batch(self, X, Y, Z, H):
        """N poses (base_link z) -> (N, 120) brightness, one kernel call."""
        cls, alb, ndl = kernel.cast(self.scene, self.cam, X, Y, np.asarray(Z, float) + EYE_DZ, H)
        v = self.shade(cls, alb, ndl).reshape(len(cls), self.n_col, self.rows).mean(2)
        return v @ self.W.T

    # ------------------------------------------------------------------ shading
    def shade(self, cls, alb, ndl):
        v = np.where(cls == SKY, self.sky_v, self.ground_v)
        v = np.where(cls == TEX, alb * np.interp(ndl, KNOTS, self.k_tex), v)
        v = np.where(cls == PLN, np.interp(ndl, KNOTS, self.k_plain), v)
        return v

    def columns(self, ray_vals):
        return ray_vals.reshape(self.n_col, self.rows).mean(1)

    def render(self, x, y, z, h):
        """-> 120-vector of brightness (0..1, no noise), like fly_forest.Eye.vector()."""
        return self.W @ self.columns(self.shade(*self.cast(x, y, z, h)))

    def features(self, x, y, z, h):
        """Linear basis for fitting the lighting: (120, 2 + 2*len(KNOTS)) = [sky, ground, albedo*hat_k(n.L) .., hat_k(n.L) ..]
        (the plain grey is folded into its own curve)."""
        cls, alb, ndl = self.cast(x, y, z, h); nk = len(KNOTS)
        hat = np.maximum(0, 1 - np.abs(ndl[:, None] - KNOTS[None]) / (KNOTS[1] - KNOTS[0]))       # (K, nk)
        F = np.zeros((len(cls), 2 + 2 * nk))
        F[:, 0] = cls == SKY; F[:, 1] = cls == GROUND
        F[:, 2:2 + nk] = (cls == TEX)[:, None] * alb[:, None] * hat; F[:, 2 + nk:] = (cls == PLN)[:, None] * hat
        Fc = F.reshape(self.n_col, self.rows, -1).mean(1)
        return self.W @ Fc
