"""The ray-cast kernel: one pure function, arrays in -> arrays out, for N drones at once.

    cls, alb, ndl = cast(scene, cam, ex, ey, ez, h)
        scene  Scene  (packed float/int arrays of one world, see pack_scene)
        cam    Cam    (eye geometry, see make_cam)
        ex, ey, ez, h  (N,) eye positions (eye height, not base_link) and headings, rad
        ->  cls (N, C*R) int8   0 background, 1 ground, 2 textured, 3 plain grey
            alb (N, C*R) float  albedo (texture grey, mip-mapped) for textured / plain hits
            ndl (N, C*R) float  max(0, n . sun) at the hit
        C = 480 * ss ray columns (ss sub-rays per pixel column), R = sample rows; ray index = ray column * R + row.

Everything else (shading, ommatidium averaging, pilot, dynamics) is outside, so this function alone can be replaced
by a numba / Rust (PyO3) version with the same signature. Scene and Cam are NamedTuples of plain numpy arrays and
scalars for that reason.

Culling: every obstacle (and the rock box, by its bounding circle) is only tested against the columns inside its
angular footprint +-asin(radius / distance) (exact, no misses), ellipsoids also by the elevation band, branches beyond
BRANCH_CULL m or thinner than BRANCH_PIX of a pixel are dropped (sub-pixel; they would only be hit by chance). Nearest hit per ray/column = one argsort on (key, t).
All drones x obstacles x columns pairs of a step are built and solved in one set of array ops."""
from typing import NamedTuple
import numpy as np

CAM_W, CAM_H, CAM_FOV, CAM_YAWS, EYE_DZ, FAR, NEAR = 160, 8, np.radians(110), np.radians([-110, 0, 110]), 0.18, 80.0, 0.05
SUN = np.array([0.001, 0.625, -0.78]); LDIR = -SUN / np.linalg.norm(SUN)
SKY, GROUND, TEX, PLN = 0, 1, 2, 3
PLAIN = 0.66
BRANCH_CULL = 25.0
BRANCH_PIX = 0.3          # drop branches thinner than this fraction of a centre pixel (1.02 deg) at their nearest point


class Cam(NamedTuple):
    az: np.ndarray        # (C,) column azimuth rel. heading, + = left
    az_s: np.ndarray      # (C,) sorted
    order: np.ndarray     # (C,) column index of az_s
    cosoff: np.ndarray    # (C,) cos(offset from the camera axis): depth = horizontal t * cosoff
    tan_el: np.ndarray    # (C, R)
    sec: np.ndarray       # (C, R) sqrt(1 + tan_el^2)
    dcol: np.ndarray      # (C,) column angular width
    drow: np.ndarray      # (C,) pixel angular height
    tm: float             # max |tan_el|


class Scene(NamedTuple):
    cyl: np.ndarray; cyl_tex: np.ndarray                                   # (M,5) x y r z0 z1 ; texture id (-1 plain)
    sph: np.ndarray; sph_R: np.ndarray; sph_tex: np.ndarray                # (M,4) x y z r ; (M,3) radii ; tex id
    seg: np.ndarray; seg_mid: np.ndarray; seg_rad: np.ndarray              # (M,7) ; (M,3) ; (M,) bounding radius
    box: np.ndarray                                                        # (B,6) cx cy cz sx sy sz
    t_flat: np.ndarray; t_off: np.ndarray; t_w: np.ndarray; t_h: np.ndarray  # mip pyramids
    t_bw: np.ndarray; t_bh: np.ndarray; t_mean: np.ndarray; t_nl: int


def make_cam(rows=2, ss=4):
    """rows = sample rows per pixel column (of the camera's 8), ss = sub-rays across each pixel column (horizontal
    anti-aliasing: Gazebo's frames behave like a supersampled render, see fastsim/README). C = 480 * ss ray columns,
    ordered pixel column-major (ray column = pixel column * ss + sub-ray)."""
    f = (CAM_W / 2) / np.tan(CAM_FOV / 2); u = np.arange(CAM_W) + .5; xo = CAM_W / 2 - u; off = np.arctan(xo / f)
    az = np.concatenate([y + off for y in CAM_YAWS]); vr = (np.arange(rows) + .5) * CAM_H / rows
    tan_el = np.tile((CAM_H / 2 - vr)[None, :] / np.sqrt(f ** 2 + xo ** 2)[:, None], (3, 1))
    dcol = np.tile(f / (f ** 2 + xo ** 2), 3); cosoff = np.tile(np.cos(off), 3); drow = np.tile(1.0 / np.sqrt(f ** 2 + xo ** 2), 3)
    if ss > 1:
        sub = (np.arange(ss) + 0.5) / ss - 0.5
        az = (az[:, None] + sub[None] * dcol[:, None]).ravel()
        rep = lambda a: np.repeat(a, ss, axis=0)
        tan_el, cosoff, drow, dcol = rep(tan_el), rep(cosoff), rep(drow), rep(dcol) / ss
    order = np.argsort(az)
    return Cam(az=az, az_s=az[order], order=order, cosoff=cosoff, tan_el=tan_el, sec=np.sqrt(1 + tan_el ** 2),
               dcol=dcol, drow=drow, tm=float(np.abs(tan_el).max()))


def pack_scene(world, tx):
    o = world.obs
    seg_mid = (o.seg[:, :3] + o.seg[:, 3:6]) / 2 if len(o.seg) else np.zeros((0, 3))
    seg_rad = np.linalg.norm(o.seg[:, 3:6] - o.seg[:, :3], axis=1) / 2 + o.seg[:, 6] if len(o.seg) else np.zeros(0)
    sph_R = np.c_[o.sph[:, 3], o.sph[:, 3], o.sph[:, 3] * world.sph_kz] if len(o.sph) else np.zeros((0, 3))
    return Scene(cyl=o.cyl, cyl_tex=world.cyl_tex, sph=o.sph, sph_R=sph_R, sph_tex=world.sph_tex, seg=o.seg, seg_mid=seg_mid,
                 seg_rad=seg_rad, box=o.box, t_flat=tx.flat, t_off=tx.off, t_w=tx.w, t_h=tx.h, t_bw=tx.base_w, t_bh=tx.base_h,
                 t_mean=tx.mean, t_nl=tx.NL)


def tex_sample(S, tex, texels, u, v):
    lvl = np.clip(np.round(np.log2(np.maximum(texels, 1.0))), 0, S.t_nl - 1).astype(np.int64)
    w, h = S.t_w[tex, lvl], S.t_h[tex, lvl]
    iu = np.floor(u * w).astype(np.int64) % w; iv = np.floor(v * h).astype(np.int64) % h
    return S.t_flat[S.t_off[tex, lvl] + iv * w + iu]


_E = np.zeros(0, np.int64)


def _pairs(cam, n, dx, dy, rad, h):
    """Candidate (drone n, obstacle) entries (flat arrays) -> (entry index, column) for every column inside the
    obstacle's angular footprint."""
    if len(n) == 0: return _E, _E
    d = np.hypot(dx, dy); th = (np.arctan2(dy, dx) - h[n] + np.pi) % (2 * np.pi) - np.pi
    al = np.where(d > rad, np.arcsin(np.minimum(1.0, rad / np.maximum(d, 1e-9))), np.pi) + 1e-9
    lo = np.concatenate([np.searchsorted(cam.az_s, th - al + s) for s in (0.0, 2 * np.pi, -2 * np.pi)])
    hi = np.concatenate([np.searchsorted(cam.az_s, th + al + s, 'right') for s in (0.0, 2 * np.pi, -2 * np.pi)])
    cnt = np.maximum(hi - lo, 0); tot = int(cnt.sum())
    if tot == 0: return _E, _E
    oi = np.repeat(np.arange(3 * len(n)), cnt); first = np.cumsum(cnt) - cnt
    ci = np.repeat(lo - first, cnt) + np.arange(tot)
    return oi % len(n), cam.order[ci]


def _nearest(key, t):
    """Pairs with hit distance t -> (pair index, key) of the nearest finite hit per key."""
    f = np.isfinite(t)
    if not f.any(): return _E, _E
    idx = np.where(f)[0]; o = idx[np.argsort(key[idx] * 1024.0 + np.minimum(t[idx], 1000.0))]   # one sort: key, then t
    k = key[o]; firsts = np.r_[True, k[1:] != k[:-1]]
    return o[firsts], k[firsts]


def cast(S, cam, ex, ey, ez, h):
    ex, ey, ez, h = (np.asarray(v, float).reshape(-1) for v in (ex, ey, ez, h))
    N = len(ex); C, R = cam.tan_el.shape; tan_el, sec = cam.tan_el, cam.sec
    a = h[:, None] + cam.az[None]; ca, sa = np.cos(a), np.sin(a)                         # (N, C)
    cls = np.zeros((N, C, R), np.int8); alb = np.zeros((N, C, R)); ndl = np.zeros((N, C, R))
    tg = np.where(tan_el[None] < 0, ez[:, None, None] / np.maximum(-tan_el[None], 1e-9), np.inf)
    best = np.where(tg * cam.cosoff[None, :, None] < FAR, tg, np.inf); cls[np.isfinite(best)] = GROUND

    # ---- vertical cylinders: nearest per column (2D), then z range per row
    c = S.cyl
    if len(c):
        mxa, mya = c[None, :, 0] - ex[:, None], c[None, :, 1] - ey[:, None]                # (N, M)
        n_, j_ = np.nonzero(mxa * mxa + mya * mya < (FAR / 0.55 + 1) ** 2)
        mx, my = mxa[n_, j_], mya[n_, j_]
        ei, col = _pairs(cam, n_, mx, my, c[j_, 2], h)
        if len(ei):
            nn, jj = n_[ei], j_[ei]; pmx, pmy = mx[ei], my[ei]
            b = ca[nn, col] * pmx + sa[nn, col] * pmy; disc = b * b - (pmx * pmx + pmy * pmy - c[jj, 2] ** 2)
            hit = (disc > 0) & (b > 0); t = np.where(hit, b - np.sqrt(np.maximum(disc, 0)), np.inf)
            t = np.where(t > NEAR, t, np.where(hit, NEAR, np.inf))
            sel, keys = _nearest(nn * C + col, t)
            tc = np.full(N * C, np.inf); j = np.zeros(N * C, np.int64); tc[keys] = t[sel]; j[keys] = jj[sel]
            tc = tc.reshape(N, C); j = j.reshape(N, C)
            fin = np.isfinite(tc); tc0 = np.where(fin, tc, 0.0)
            zc = ez[:, None, None] + tc0[:, :, None] * tan_el[None]
            ok = fin[:, :, None] & (zc >= c[j, 3][:, :, None]) & (zc <= c[j, 4][:, :, None]) & \
                ((tc * cam.cosoff[None])[:, :, None] < FAR) & (tc[:, :, None] < best)
            if ok.any():
                px, py = ex[:, None] + tc0 * ca, ey[:, None] + tc0 * sa
                nx, ny = (px - c[j, 0]) / c[j, 2], (py - c[j, 1]) / c[j, 2]
                nl = np.maximum(0, nx * LDIR[0] + ny * LDIR[1])
                texid = S.cyl_tex[j]; cosi = np.maximum(np.abs(nx * ca + ny * sa), 0.15)
                t3 = tc0[:, :, None] * sec[None]; plain = texid < 0; tid = np.maximum(texid, 0)
                texu = t3 * (cam.dcol[None] / cosi * S.t_bw[tid] / (2 * np.pi * c[j, 2]))[:, :, None]
                texv = t3 * (cam.drow[None] * 0.5 * S.t_bh[tid])[:, :, None]
                uu = ((np.arctan2(ny, nx) - np.pi / 2) / (2 * np.pi)) % 1.0
                shp = ok.shape
                val = tex_sample(S, np.broadcast_to(tid[:, :, None], shp), np.maximum(texu, texv), np.broadcast_to(uu[:, :, None], shp), 0.5 * zc)
                val = np.where(plain[:, :, None], PLAIN, val)
                best = np.where(ok, tc[:, :, None], best); cls = np.where(ok, np.where(plain[:, :, None], PLN, TEX), cls).astype(np.int8)
                alb = np.where(ok, val, alb); ndl = np.where(ok, nl[:, :, None], ndl)

    K = C * R
    best = best.reshape(N, K).ravel(); cls = cls.reshape(N, K).ravel(); alb = alb.reshape(N, K).ravel(); ndl = ndl.reshape(N, K).ravel()
    rr = np.arange(R); tel = tan_el.ravel(); secf = sec.ravel(); colw = np.repeat(cam.dcol, R); rowh = np.repeat(cam.drow, R)

    # ---- ellipsoids (foliage, shrubs, crowns, boulder)
    s = S.sph
    if len(s):
        dxa, dya = s[None, :, 0] - ex[:, None], s[None, :, 1] - ey[:, None]; dh = np.hypot(dxa, dya)
        Rz = S.sph_R[None, :, 2]; reach = (dh + s[None, :, 3]) * cam.tm; zrel = s[None, :, 2] - ez[:, None]
        n_, j_ = np.nonzero((dh - s[None, :, 3] < FAR / 0.55) & (zrel - Rz < reach) & (zrel + Rz > -reach))
        ei, col = _pairs(cam, n_, dxa[n_, j_], dya[n_, j_], s[j_, 3], h)
        if len(ei):
            nn, jj = n_[ei], j_[ei]; Rr = S.sph_R[jj]
            o = (np.c_[ex[nn], ey[nn], ez[nn]] - s[jj, :3]) / Rr                               # (P, 3)
            dx = (ca[nn, col] / Rr[:, 0])[:, None]; dy = (sa[nn, col] / Rr[:, 1])[:, None]; dz = tan_el[col] / Rr[:, 2:3]
            A = dx * dx + dy * dy + dz * dz; B = dx * o[:, :1] + dy * o[:, 1:2] + dz * o[:, 2:3]; Cq = (o * o).sum(1, keepdims=True) - 1
            disc = B * B - A * Cq; sq = np.sqrt(np.maximum(disc, 0))
            t = np.where(disc > 0, (-B - sq) / A, np.inf)
            t = np.where(t > NEAR, t, np.where((disc > 0) & ((-B + sq) / A > NEAR), NEAR, np.inf)).ravel()
            ray = ((nn * C + col)[:, None] * R + rr[None]).ravel(); pid = np.repeat(np.arange(len(ei)), R)
            sel, rays = _nearest(ray, t); ok = t[sel] < best[rays]
            if ok.any():
                i = rays[ok]; p_ = pid[sel[ok]]; tt = t[sel[ok]]; jp = jj[p_]; Rp = Rr[p_]
                di, ci_ = i // K, (i % K) // R
                Dv = np.c_[ca[di, ci_], sa[di, ci_], tel[i % K]]
                q = o[p_] + tt[:, None] * Dv / Rp
                nrm = q / Rp; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
                texid = S.sph_tex[jp]; plain = texid < 0; tid = np.maximum(texid, 0); r_, rz = Rp[:, 0], Rp[:, 2]
                lo_ = np.arctan2(q[:, 1], q[:, 0]) % (2 * np.pi); la = np.arcsin(np.clip(q[:, 2], -1, 1))
                uu = lo_ / np.pi * np.maximum(1, r_); vv = (la + np.pi / 2) / np.pi * np.maximum(1, rz)
                sf = secf[i % K]; cosi = np.maximum(np.abs((nrm * Dv / sf[:, None]).sum(1)), 0.15); t3 = tt * sf
                texu = t3 * colw[i % K] / cosi * S.t_bw[tid] * np.maximum(1, r_) / (np.pi * r_)
                texv = t3 * rowh[i % K] / cosi * S.t_bh[tid] * np.maximum(1, rz) / (np.pi * rz)
                val = np.where(plain, PLAIN, tex_sample(S, tid, np.maximum(texu, texv), uu, vv))
                best[i] = tt; cls[i] = np.where(plain, PLN, TEX); alb[i] = val; ndl[i] = np.maximum(0, nrm @ LDIR)

    # ---- branches (thin capsules rendered as bark cylinders), near ones only
    sg = S.seg
    if len(sg):
        dxa, dya = S.seg_mid[None, :, 0] - ex[:, None], S.seg_mid[None, :, 1] - ey[:, None]
        dn = np.maximum(np.hypot(dxa, dya) - S.seg_rad[None], 0.1)
        n_, j_ = np.nonzero((dn < BRANCH_CULL) & (2 * sg[None, :, 6] / dn > BRANCH_PIX * 0.0179))
        ei, col = _pairs(cam, n_, dxa[n_, j_], dya[n_, j_], S.seg_rad[j_], h)
        if len(ei):
            nn, jj = n_[ei], j_[ei]; a0 = sg[jj, :3]; ax = sg[jj, 3:6] - a0; L = np.linalg.norm(ax, axis=1); ua = ax / L[:, None]; rad = sg[jj, 6]
            e3 = np.c_[ex[nn], ey[nn], ez[nn]]; w = e3 - a0; wU = (w * ua).sum(1)
            Dx = ca[nn, col][:, None]; Dy = sa[nn, col][:, None]; Dz = tan_el[col]
            dU = Dx * ua[:, :1] + Dy * ua[:, 1:2] + Dz * ua[:, 2:3]
            px_ = Dx - dU * ua[:, :1]; py_ = Dy - dU * ua[:, 1:2]; pz_ = Dz - dU * ua[:, 2:3]
            wp = w - wU[:, None] * ua
            A = px_ * px_ + py_ * py_ + pz_ * pz_; B = px_ * wp[:, :1] + py_ * wp[:, 1:2] + pz_ * wp[:, 2:3]
            disc = B * B - A * ((wp * wp).sum(1) - rad * rad)[:, None]
            t = np.where((disc > 0) & (A > 1e-12), (-B - np.sqrt(np.maximum(disc, 0))) / np.maximum(A, 1e-12), np.inf)
            sax = wU[:, None] + t * dU
            t = np.where((t > NEAR) & (sax >= 0) & (sax <= L[:, None]), t, np.inf).ravel()
            ray = ((nn * C + col)[:, None] * R + rr[None]).ravel(); pid = np.repeat(np.arange(len(ei)), R)
            sel, rays = _nearest(ray, t); ok = t[sel] < best[rays]
            if ok.any():
                i = rays[ok]; p_ = pid[sel[ok]]; tt = t[sel[ok]]; di, ci_ = i // K, (i % K) // R
                Dv = np.c_[ca[di, ci_], sa[di, ci_], tel[i % K]]
                rel = e3[p_] + tt[:, None] * Dv - a0[p_]; nrm = rel - (rel * ua[p_]).sum(1, keepdims=True) * ua[p_]
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
                best[i] = tt; cls[i] = TEX; alb[i] = S.t_mean[0]; ndl[i] = np.maximum(0, nrm @ LDIR)

    # ---- boxes (rock): footprint pairs, slab test per ray
    for bx in S.box:
        lo_, hi_ = bx[:3] - bx[3:6] / 2, bx[:3] + bx[3:6] / 2; rad = float(np.hypot(bx[3], bx[4]) / 2)
        nn_all = np.arange(N); ei, col = _pairs(cam, nn_all, bx[0] - ex, bx[1] - ey, np.full(N, rad), h)
        if not len(ei): continue
        ray = ((ei * C + col)[:, None] * R + rr[None]).ravel(); nn = ray // K
        D = np.c_[ca[nn, (ray % K) // R], sa[nn, (ray % K) // R], tel[ray % K]]; E = np.c_[ex[nn], ey[nn], ez[nn]]
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (lo_[None] - E) / D; t2 = (hi_[None] - E) / D
        tmin = np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2)); tmax = np.where(np.isnan(t1), np.inf, np.maximum(t1, t2))
        tn = tmin.max(1); tf = tmax.min(1); face = tmin.argmax(1)
        ok = (tn <= tf) & (tf > NEAR) & (np.maximum(tn, NEAR) < best[ray])
        if ok.any():
            q = np.where(ok)[0]; i = ray[q]; f_ = face[q]; nrm = np.zeros((len(q), 3)); nrm[np.arange(len(q)), f_] = -np.sign(D[q, f_])
            best[i] = np.maximum(tn[q], NEAR); cls[i] = PLN; alb[i] = PLAIN; ndl[i] = np.maximum(0, nrm @ LDIR)

    far = best.reshape(N, K) * np.repeat(cam.cosoff, R)[None] >= FAR
    return np.where(far, SKY, cls.reshape(N, K)).astype(np.int8), alb.reshape(N, K), ndl.reshape(N, K)
