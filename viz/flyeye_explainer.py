"""Explainer video for outsiders: what the fly's eye sees and what its neurons do as the drone meets an obstacle.

    python viz/flyeye_explainer.py --clip a          # one of the built-in clips (a, b, c), see CLIPS
    python viz/flyeye_explainer.py --clip all
    python viz/flyeye_explainer.py <run.json> --key 92.9 [--pre 7 --post 3.5 --slow-pre 2 --slow-post 2] --title "..." --out x.mp4

1280x720, one panel per step of the pipeline, labelled in Thai with the English term in brackets:
  1 what a person sees     first-person camera at eye height, rendered by Gazebo from the logged poses (chasecam.py --cam fpv,
                           container flybrain-px4-viz), with the 8 deg band the fly eye samples marked
  2 what the fly eye sees  the logged 120-value eye vector as a pixelated 330 deg panorama (front / left / right / back marked)
  3 motion detectors       per-ommatidium outward motion, recomputed from the logged eye with the brain's own EMD
                           (flydrone.controllers.EMD, tau_lp 0.035, tau_hp 0.25): outward = max(r * sign(mid), 0)
  4 looming neurons        LPLC2 left / right = outward motion summed over the front 0-35 deg on each side; threshold =
                           lplc2_thr (2.5) x a slowly adapting baseline (EMA, tau 3 s, only while calm); logged looming flag
  5 top-down map           drone, heading, the obstacle ahead (or nearest), true distance
The EMD is reset where the real brain resets it (saccade start, and the fresh start after a turn / recovery: logged
state 4, 1, 0); the baseline only adapts during fixation, as in FlyVision.act. The recomputed bars follow the brain's
maths but are not bit-identical to the flight (the brain's gating has extra rules), so the logged flag is shown too.
Each clip: a short title card, then the run at 1x with --slow-pre/--slow-post seconds around the key moment at 0.5x."""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
from viz.common import Run, Encoder, EYE_AZ, zone_of                     # noqa: E402
from viz.replay2d import forest_raster                                    # noqa: E402
from flydrone.controllers import EMD                                      # noqa: E402
from flydrone.sim import make_eye                                         # noqa: E402
from px4.forest_geom import BODY_R                                        # noqa: E402

OUTD = ROOT / 'results/videos/explainer'
CLIPS = {
    'a': dict(run='px4/runs/forest/lplc2-bold-gate-sv0-v0.8_ff_easy_0.json', key=2786, pre=7.0, post=3.5, slow_pre=2.0, slow_post=2.2,
              out='a_success_trunk_escape.mp4',
              title='(ก) เห็นต้นไม้พุ่งเข้ามา แล้วเลี้ยวหลบทัน', title_en='(a) A trunk looms, the looming neurons fire, the drone turns away in time',
              intro=['โดรนใช้ "ตาแมลง" อย่างเดียว ไม่มีเซนเซอร์วัดระยะ', 'ดูว่าต้นไม้ข้างหน้าทำให้ภาพ "ขยายออก" อย่างไร',
                     'และเซลล์ประสาท LPLC2 สั่งให้เลี้ยวหลบได้อย่างไร'],
              intro_en='Vision only: watch the trunk expand on the eye, the motion detectors point outward, LPLC2 cross its threshold, and the escape turn.'),
    'b': dict(run='px4/runs/forest/lplc2-bold-gate-sv0-v0.8_ff_easy_2.json', key=9584, pre=6.0, post=2.0, slow_pre=2.6, slow_post=1.0,
              out='b_fail_flat_rock_wall.mp4',
              title='(ข) กำแพงหินเรียบ: ไม่มีลาย ภาพแทบไม่ขยาย จึงชน', title_en='(b) A featureless rock wall: almost no texture, weak expansion, no alarm, crash',
              intro=['กำแพงหินสีเรียบ เกือบเหมือนสีท้องฟ้า', 'ตาแมลงวัดได้แค่ "ความสว่างที่เปลี่ยน"', 'ถ้าพื้นผิวไม่มีลาย ตัวตรวจจับการเคลื่อนไหวก็แทบไม่ตอบสนอง'],
              intro_en='A plain grey wall gives the motion detectors almost nothing to track: the looming neurons stay below threshold until it is too late.'),
    'c': dict(run='px4/runs/forest/lplc2-bold-gate-sv0-v0.8_ff_medium_3.json', key=7276, pre=6.0, post=2.0, slow_pre=2.6, slow_post=0.8,
              out='c_fail_thin_branch.mp4',
              title='(ค) กิ่งไม้บาง: เล็กเกินกว่าตา 120 จุดจะเห็น', title_en='(c) A thin branch: too small for the 120-point eye band to register',
              intro=['ตาแมลงมีแค่ 120 จุด แต่ละจุดกว้างเกือบ 3 องศา', 'และมองเห็นแค่แถบแคบ ๆ สูง 8 องศา', 'กิ่งไม้บาง ๆ จึงแทบไม่ปรากฏในภาพของตาแมลง'],
              intro_en='Each ommatidium is ~2.8 deg wide and the band only 8 deg tall: a thin branch barely changes any of the 120 values.'),
}
STATE_TH = ['เตรียมตัว (warm-up)', 'ตั้งหลัก (settling)', 'จ้องตรงไป (fixate)', 'กำลังเลี้ยว (saccade)', 'เริ่มเลี้ยว (saccade start)']
C_L, C_R, C_THR, C_LOOM, C_CRASH = '#1e88e5', '#fb8c00', '#424242', '#d81b60', '#e53935'


# ---------------------------------------------------------------- neuron model (recomputed from the logged eye)
def neurons(R, thr_k=2.5, front=35.0, tau_base=3.0):
    eye = make_eye(120); mid = (eye[:-1] + eye[1:]) / 2; sgn = np.sign(mid); a = np.abs(np.degrees(mid))
    fl, fr = (mid > 0) & (a < front), (mid < 0) & (a < front)
    out = np.zeros((R.n, 119)); l2 = np.zeros((R.n, 2)); base = np.full(R.n, np.nan); loom = np.zeros(R.n, bool)
    emd, b, prev = EMD(120, R.dt, 0.035, 0.25), None, -1
    for i in range(R.n):
        st = int(R.state[i])
        if st != prev and st in (0, 1, 4): emd = EMD(120, R.dt, 0.035, 0.25)        # where FlyVision / the pilot reset it
        prev = st
        r = emd.step(R.eye[i].astype(float) / 255.0); o = np.maximum(r * sgn, 0); out[i] = o
        l2[i] = o[fl].sum(), o[fr].sum(); s = l2[i].sum()
        if st == 2:                                                                 # fixating: the only time it is used / adapts
            if b is None: b = s
            loom[i] = s > thr_k * b
            if not loom[i]: b += R.dt / tau_base * (s - b)
        base[i] = np.nan if b is None else b
    return mid, out, l2, base * thr_k, loom


# ---------------------------------------------------------------- geometry for the map / distance readout
class Shapes:
    """Obstacle cross-section at flight height: circles (trunks, sampled branches, spheres) and axis-aligned boxes (rock wall)."""
    def __init__(self, meta):
        from px4.forest_geom import Obstacles
        fp = Obstacles(meta).footprint(meta['alt'] - 0.15, meta['alt'] + 0.15)
        self.C = np.array([s[1:4] for s in fp if s[0] == 'c']).reshape(-1, 3); self.Ck = [s[4] for s in fp if s[0] == 'c']
        self.B = np.array([s[1:5] for s in fp if s[0] == 'b']).reshape(-1, 4); self.Bk = [s[5] for s in fp if s[0] == 'b']

    def ahead(self, x, y, h, hw=0.25):
        """First obstacle along the heading within the body half-width: (distance, kind, shape) or (inf, None, None)."""
        best = (np.inf, None, None); d = np.array([np.cos(h), np.sin(h)])
        if len(self.C):
            rel = self.C[:, :2] - [x, y]; t = rel @ d; perp = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]); R_ = self.C[:, 2] + hw
            ok = (t > 0) & (perp < R_); dist = np.where(ok, t - np.sqrt(np.maximum(R_ ** 2 - perp ** 2, 0)), np.inf); j = int(np.argmin(dist))
            if dist[j] < best[0]: best = (float(dist[j]), self.Ck[j], ('c', *self.C[j]))
        for (bx, by, sx, sy), k in zip(self.B, self.Bk):                     # ray vs box inflated by the half-width (slab test)
            lo, hi = np.array([bx - sx / 2 - hw, by - sy / 2 - hw]), np.array([bx + sx / 2 + hw, by + sy / 2 + hw])
            with np.errstate(divide='ignore', invalid='ignore'):
                t1, t2 = (lo - [x, y]) / d, (hi - [x, y]) / d
            tn, tf = np.nanmax(np.minimum(t1, t2)), np.nanmin(np.maximum(t1, t2))
            if tf >= max(tn, 0) and max(tn, 0) < best[0]: best = (float(max(tn, 0)), k, ('b', bx, by, sx, sy))
        return best

    def nearest(self, x, y):
        best = (np.inf, None, None)
        if len(self.C):
            dd = np.hypot(self.C[:, 0] - x, self.C[:, 1] - y) - self.C[:, 2]; j = int(np.argmin(dd)); best = (float(dd[j]), self.Ck[j], ('c', *self.C[j]))
        for (bx, by, sx, sy), k in zip(self.B, self.Bk):
            dd = np.hypot(max(abs(x - bx) - sx / 2, 0), max(abs(y - by) - sy / 2, 0))
            if dd < best[0]: best = (float(dd), k, ('b', bx, by, sx, sy))
        return best


def closest_point(shape, x, y):
    if shape[0] == 'c':
        _, cx, cy, r = shape; v = np.array([x - cx, y - cy]); n = np.linalg.norm(v) + 1e-9; return cx + r * v[0] / n, cy + r * v[1] / n
    _, bx, by, sx, sy = shape; return float(np.clip(x, bx - sx / 2, bx + sx / 2)), float(np.clip(y, by - sy / 2, by + sy / 2))


# ---------------------------------------------------------------- first-person frames from Gazebo
def fpv_frames(run, t0, t1, frames, threads=4):
    if frames.exists() and len(list(frames.glob('*.jpg'))) > 0: return frames
    cmd = [sys.executable, str(HERE / 'chasecam.py'), str(run), '--cam', 'fpv', '--no-hud', '--frames-only', '--frames', str(frames),
           '--speed', '1', '--fps', f'{1 / Run(run).dt:.6f}', '--t0', f'{t0}', '--t1', f'{t1}', '--threads', str(threads)]
    subprocess.run(cmd, check=True); return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', nargs='?'); ap.add_argument('--clip', help='a, b, c or all'); ap.add_argument('--key', type=float, help='key moment, s')
    ap.add_argument('--pre', type=float, default=6); ap.add_argument('--post', type=float, default=3)
    ap.add_argument('--slow-pre', type=float, default=2); ap.add_argument('--slow-post', type=float, default=1.5); ap.add_argument('--slow', type=float, default=0.5)
    ap.add_argument('--title', default=''); ap.add_argument('--out'); ap.add_argument('--intro-secs', type=float, default=4.0)
    ap.add_argument('--no-3d', action='store_true', help='skip the Gazebo first-person panel'); ap.add_argument('--keep-frames', action='store_true')
    a = ap.parse_args()
    if a.clip:
        for c in (CLIPS if a.clip == 'all' else a.clip.split(',')): make(dict(CLIPS[c]), a)
    else:
        if not (a.run and a.key is not None): ap.error('give --clip or a run plus --key')
        R = Run(a.run)
        make(dict(run=a.run, key=int(round(a.key / R.dt)), pre=a.pre, post=a.post, slow_pre=a.slow_pre, slow_post=a.slow_post,
                  out=a.out or f'{Path(a.run).stem}_t{a.key:.0f}_explainer.mp4', title=a.title or R.mode, title_en=R.label(), intro=[], intro_en=''), a)


def make(c, a):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, Wedge, FancyBboxPatch
    plt.rcParams['font.family'] = ['Sarabun', 'Thonburi', 'Tahoma', 'DejaVu Sans']
    run = (ROOT / c['run']).resolve(); R = Run(run); meta = R.meta; dt = R.dt; fps = 1 / dt
    try:
        from px4.forest_params import resolve
        P = resolve(R.mode, {}); thr_k = P.get('lplc2_thr') or 2.5; front = P.get('front') or 35
    except Exception:
        thr_k, front = 2.5, 35
    mid, out, l2, thr, cloom = neurons(R, thr_k, front)
    k0 = c['key']; i0 = max(0, k0 - int(c['pre'] / dt)); i1 = min(R.n - 1, k0 + int(c['post'] / dt))
    s0, s1 = k0 - int(c['slow_pre'] / dt), k0 + int(c['slow_post'] / dt)
    sched = []                                                     # log index per output frame (0.5x = each log frame twice)
    for i in range(i0, i1 + 1): sched += [i] * (int(round(1 / a.slow)) if s0 <= i <= s1 else 1)
    OUTD.mkdir(parents=True, exist_ok=True); out_path = Path(a.out).resolve() if (a.out and not a.clip) else OUTD / c['out']
    fdir = OUTD / f'_fpv_{Path(c["out"]).stem}_{i0}_{i1}'
    fpv = None
    if not a.no_3d:
        t_render = time.time(); fpv_frames(run, (i0 + 0.5) * dt, (i1 + 0.5) * dt, fdir)   # +0.5: chasecam floors t/dt
        files = sorted(fdir.glob('*.jpg')); print(f'  {len(files)} first-person frames ({time.time() - t_render:.0f} s)', flush=True)
        idx = np.unique(np.round(np.arange(i0, i1 + 1e-9, 1.0)).astype(int))        # chasecam's own frame -> log index mapping
        fpv = {int(i): f for i, f in zip(idx, files)}
    raster, ext = forest_raster(meta); SH = Shapes(meta)
    crash_i = np.array([cr['i'] for cr in R.crashes], int)
    turn_after = {}                                                # true turn direction after each looming onset (from the heading)
    hu = np.unwrap(R.h)
    W, H, dpi = 1280, 720, 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor='#fafafa')
    enc = Encoder(out_path, W, H, fps, channels=4)

    # ---------------- title card
    fig.text(0.5, 0.80, c['title'], ha='center', fontsize=26, weight='bold', color='#212121')
    fig.text(0.5, 0.735, c['title_en'], ha='center', fontsize=13, color='#555')
    for j, line in enumerate(c.get('intro', [])): fig.text(0.5, 0.62 - j * 0.065, line, ha='center', fontsize=18, color='#333')
    if c.get('intro_en'): fig.text(0.5, 0.40, c['intro_en'], ha='center', fontsize=11.5, color='#666', wrap=True)
    steps = ['1 สิ่งที่คนเห็น\n(human view)', '2 ตาแมลง 120 จุด\n(ommatidia)', '3 ตัวตรวจจับการเคลื่อนไหว\n(T4/T5 → EMD)',
             '4 เซลล์ตรวจจับการพุ่งเข้าหา\n(LPLC2)', '→ เลี้ยวหนี\n(escape turn)']
    for j, s in enumerate(steps):
        x = 0.06 + j * 0.185
        fig.patches.append(FancyBboxPatch((x, 0.19), 0.165, 0.11, boxstyle='round,pad=0.005', transform=fig.transFigure,
                                          fc=['#e3f2fd', '#eeeeee', '#fff3e0', '#fce4ec', '#e8f5e9'][j], ec='#bbb'))
        fig.text(x + 0.0825, 0.245, s, ha='center', va='center', fontsize=11, linespacing=1.4)
    fig.text(0.5, 0.08, f'{R.mode} · {R.world} · t = {k0 * dt:.1f} s · ภาพช้า {a.slow:g}× ช่วงสำคัญ (slow motion around the key moment)',
             ha='center', fontsize=9, color='#888')
    fig.canvas.draw(); card = np.asarray(fig.canvas.buffer_rgba()).copy()
    for _ in range(int(a.intro_secs * fps)): enc.add(card)
    fig.clf()

    # ---------------- layout
    title = fig.text(0.012, 0.975, c['title'], fontsize=15, weight='bold', va='top')
    clock = fig.text(0.988, 0.975, '', fontsize=11, ha='right', va='top', family=['DejaVu Sans Mono', 'Sarabun'])
    slowtag = fig.text(0.988, 0.94, '', fontsize=11, ha='right', va='top', color='white', weight='bold',
                       bbox=dict(fc='#6a1b9a', ec='none', boxstyle='round,pad=0.25'))
    caption = fig.text(0.012, 0.928, '', fontsize=12.5, va='top', color='#263238')

    def label(ax, s): ax.set_title(s, fontsize=10.5, loc='left', pad=3, weight='bold', color='#37474f')

    # 1 first-person view
    a1 = fig.add_axes([0.008, 0.36, 0.5, 0.5]); a1.set_xticks([]); a1.set_yticks([]); label(a1, '1. สิ่งที่คนเห็น (human view, กล้อง 100°)')
    im1 = a1.imshow(np.full((360, 640, 3), 200, np.uint8), extent=[0, 640, 360, 0], aspect='auto')
    vf = 2 * np.arctan(np.tan(np.radians(50)) * 360 / 640); bh = np.tan(np.radians(4)) / np.tan(vf / 2) * 180
    for yb in (180 - bh, 180 + bh): a1.axhline(yb, color='#ffeb3b', lw=1, ls='--', alpha=.9)
    a1.text(6, 180 - bh - 4, 'แถบที่ตาแมลงมองเห็น สูง 8° (eye band)', color='#ffeb3b', fontsize=8.5, va='bottom',
            bbox=dict(fc='black', ec='none', alpha=.45, pad=1.5))
    a1.set_xlim(0, 640); a1.set_ylim(360, 0)
    if fpv is None: a1.text(320, 180, 'first-person view not rendered (--no-3d)', ha='center', color='#555')
    crash_box = a1.text(320, 300, '', ha='center', fontsize=20, weight='bold', color='white', bbox=dict(fc=C_CRASH, ec='none', boxstyle='round,pad=0.3'))

    # 4 looming neurons
    a4 = fig.add_axes([0.555, 0.625, 0.17, 0.235]); label(a4, '4. เซลล์ตรวจจับการพุ่งเข้าหา (LPLC2)')
    win = slice(i0, i1 + 1); fixw = R.state[win] == 2; sums = l2[win].sum(1)[fixw] if fixw.any() else l2[win].sum(1)
    top = max(np.nanmax(thr[win]) if np.isfinite(thr[win]).any() else 0, np.max(sums)) * 1.3 + 1e-6   # fixation frames only: rotation during turns is ignored by the brain
    a4.set_xlim(-0.6, 2.6); a4.set_ylim(0, top); a4.set_xticks([0, 1, 2]); a4.set_xticklabels(['ซ้าย (L)', 'ขวา (R)', 'รวม (Σ)'], fontsize=9)
    a4.set_yticks([]); bars = a4.bar([0, 1, 2], [0, 0, 0], color=[C_L, C_R, '#7e57c2'], width=0.7)
    thr_line, = a4.plot([1.6, 2.4], [0, 0], color=C_THR, lw=2.2, ls='--'); thr_txt = a4.text(0.03, 0.97, '- - เกณฑ์ (threshold)', transform=a4.transAxes, fontsize=8, va='top', color=C_THR)
    a4h = fig.add_axes([0.755, 0.625, 0.235, 0.235]); a4h.set_title('ย้อนหลัง 4 วินาที: รวม vs เกณฑ์ (last 4 s)', fontsize=9, loc='left', pad=3, color='#555')
    a4h.set_xlim(-4, 0); a4h.set_ylim(0, top); a4h.set_yticks([]); a4h.tick_params(labelsize=7.5); a4h.set_xlabel('s   | = สัญญาณเตือนจริง (logged flag)   ช่องว่าง = เลี้ยว (turning)', fontsize=7, labelpad=1)
    h_sum, = a4h.plot([], [], color='#7e57c2', lw=1.6); h_thr, = a4h.plot([], [], color=C_THR, lw=1.3, ls='--')
    h_flag = a4h.scatter([], [], marker='|', s=60, color=C_LOOM, lw=2)

    # 5 map
    a5 = fig.add_axes([0.555, 0.36, 0.175, 0.19]); label(a5, '5. แผนที่จากด้านบน (top-down)')
    a5.imshow(raster, extent=ext, origin='lower', interpolation='nearest'); a5.set_aspect('equal'); a5.set_xticks([]); a5.set_yticks([])
    trail, = a5.plot([], [], color='#1565c0', lw=1); body = Circle((0, 0), 0.35, fc='#1565c0', ec='white', zorder=5); a5.add_patch(body)
    fov = Wedge((0, 0), 2.2, 0, 1, fc='#ffeb3b', alpha=.35, ec='none', zorder=4); a5.add_patch(fov)
    hd, = a5.plot([], [], color='#0d1b4a', lw=2, zorder=6); tgt = Circle((0, 0), 0.3, fill=False, ec=C_CRASH, lw=2, zorder=6); a5.add_patch(tgt)
    tgtb = Rectangle((0, 0), 1, 1, fill=False, ec=C_CRASH, lw=2, zorder=6); a5.add_patch(tgtb)
    tline, = a5.plot([], [], color=C_CRASH, lw=1, ls=':', zorder=6)
    info = fig.text(0.745, 0.548, '', fontsize=10.5, va='top', linespacing=1.55)
    banner = fig.text(0.255, 0.80, '', fontsize=15, weight='bold', color='white', ha='center', va='center',
                      bbox=dict(fc=C_LOOM, ec='none', boxstyle='round,pad=0.35'))

    # 2 eye band
    a2 = fig.add_axes([0.04, 0.215, 0.92, 0.085]); label(a2, '2. สิ่งที่ตาแมลงเห็น: ความสว่าง 120 จุด รอบตัว 330° (ommatidia)')
    az = np.degrees(EYE_AZ); half = (az[1] - az[0]) / 2
    im2 = a2.imshow(np.zeros((1, 120)), cmap='gray', vmin=0, vmax=255, aspect='auto', interpolation='nearest', extent=[az[0] - half, az[-1] + half, 0, 1])
    a2.set_xlim(az[-1] + half, az[0] - half); a2.set_yticks([])
    a2.set_xticks([165, 90, 50, 35, 0, -35, -50, -90, -165])
    a2.set_xticklabels(['หลัง (back)', 'ซ้าย 90° (left)', '', '', 'หน้า (front)', '', '', 'ขวา 90° (right)', 'หลัง (back)'], fontsize=8.5)
    for xv in np.arange(-165, 166, 30): a2.axvline(xv, color='white', lw=.3, alpha=.25)
    a2.plot([50, -50], [1.12, 1.12], color='#f9a825', lw=2.5, clip_on=False, solid_capstyle='butt'); a2.text(0, 1.2, 'มุมกล้องด้านบน (camera view 100°)', ha='center', fontsize=7.5, color='#b28704', transform=a2.transData, clip_on=False)
    a2.plot([35, -35], [-0.08, -0.08], color=C_LOOM, lw=3, clip_on=False, solid_capstyle='butt')

    # 3 motion detectors
    a3 = fig.add_axes([0.04, 0.025, 0.92, 0.1]); label(a3, '3. ตัวตรวจจับการเคลื่อนไหว: ภาพไหลออกจากด้านหน้า = มีของเข้าใกล้ (T4/T5 → EMD, outward motion)')
    md = np.degrees(mid); otop = np.percentile(out[win][fixw] if fixw.any() else out[win], 99.0) * 1.1 + 1e-6   # strongest 1% clip at the top
    a3.set_xlim(az[-1] + half, az[0] - half); a3.set_ylim(0, otop); a3.set_yticks([]); a3.set_xticks([])
    ebars = a3.bar(md, np.zeros(119), width=2 * half * 0.85, color=[C_L if m > 0 else C_R for m in md])
    a3.axvspan(-front, front, color=C_LOOM, alpha=.07, lw=0)
    a3.text(0.0, 0.92, '◀ ไหลออกทางซ้าย', transform=a3.transAxes, fontsize=8.5, color=C_L, va='top')
    a3.text(1.0, 0.92, 'ไหลออกทางขวา ▶', transform=a3.transAxes, fontsize=8.5, color=C_R, va='top', ha='right')
    a3.text(0.5, 0.92, 'ช่วงที่ LPLC2 รวมสัญญาณ (front 0–35°)', transform=a3.transAxes, fontsize=8, color=C_LOOM, va='top', ha='center')

    # ---------------- frames
    last_onset, t_start = -10 ** 9, time.time(); prev_flag = 0
    for kf, i in enumerate(sched):
        x, y, h = R.x[i], R.y[i], R.h[i]; st = int(R.state[i]); flag = int(R.looming[i])
        if fpv is not None and i in fpv:
            from PIL import Image
            im1.set_data(np.asarray(Image.open(fpv[i])))
        # 2 + 3
        im2.set_data(R.eye[i][None, :].astype(float))
        use = st == 2                                              # the brain only uses vision while fixating
        for b, v in zip(ebars, out[i]): b.set_height(min(v, otop)); b.set_alpha(1.0 if use else 0.25)
        # 4
        L_, R_ = l2[i]; S = L_ + R_; th = thr[i]
        for b, v in zip(bars, (L_, R_, S)): b.set_height(min(v, top)); b.set_alpha(1.0 if use else 0.25)
        if np.isfinite(th): thr_line.set_data([1.6, 2.4], [min(th, top)] * 2)
        bars[2].set_color(C_LOOM if (use and np.isfinite(th) and S > th) else '#7e57c2')
        w0 = max(i0 - int(4 / dt), i - int(4 / dt)); tt = (np.arange(w0, i + 1) - i) * dt
        h_sum.set_data(tt, np.where(R.state[w0:i + 1] == 2, np.minimum(l2[w0:i + 1].sum(1), top), np.nan)); h_thr.set_data(tt, np.minimum(thr[w0:i + 1], top))
        fl = np.where(R.looming[w0:i + 1] > 0)[0]; h_flag.set_offsets(np.c_[tt[fl], np.full(len(fl), top * 0.9)] if len(fl) else np.zeros((0, 2)))
        # banner: logged looming onset -> escape turn, direction from the heading change that follows
        if flag and not prev_flag: last_onset = i
        prev_flag = flag
        age = (i - last_onset) * dt
        if age < 1.6:
            if last_onset not in turn_after:
                dh = hu[min(R.n - 1, last_onset + int(1.2 / dt))] - hu[last_onset]; turn_after[last_onset] = 'ซ้าย (left)' if dh > 0 else 'ขวา (right)'
            banner.set_text(f'เลี้ยวหนี! (ESCAPE TURN) → {turn_after[last_onset]}'); banner.set_visible(True)
        else:
            banner.set_visible(False)
        # 5 map + distance
        hw = 3.2; vh = hw * (0.19 * 720) / (0.175 * 1280); a5.set_xlim(x - hw, x + hw); a5.set_ylim(y - vh, y + vh)
        r0 = max(i0, i - int(8 / dt)); trail.set_data(R.x[r0:i + 1], R.y[r0:i + 1]); body.center = (x, y)
        fov.set_center((x, y)); fov.set_theta1(np.degrees(h) - 50); fov.set_theta2(np.degrees(h) + 50)
        hd.set_data([x, x + 0.9 * np.cos(h)], [y, y + 0.9 * np.sin(h)])
        dist, kd, shp = SH.ahead(x, y, h); what = 'ข้างหน้า (ahead)'
        if shp is None: dist, kd, shp = SH.nearest(x, y); what = 'ใกล้สุด (nearest)'          # nothing ahead: nearest instead
        if shp[0] == 'c':
            tgt.center = tuple(shp[1:3]); tgt.set_radius(shp[3] + 0.08); tgt.set_visible(True); tgtb.set_visible(False)
        else:
            _, bx, by, sx, sy = shp; tgtb.set_bounds(bx - sx / 2 - .05, by - sy / 2 - .05, sx + .1, sy + .1); tgtb.set_visible(True); tgt.set_visible(False)
        px_, py_ = closest_point(shp, x, y); tline.set_data([x, px_], [y, py_])
        dist = np.hypot(px_ - x, py_ - y) - BODY_R                   # clearance from the drone body, like the logged gap
        kind = {'trunk': 'ลำต้น (trunk)', 'trunk_plain': 'ลำต้นสีเรียบ (plain trunk)', 'branch': 'กิ่งไม้ (branch)', 'foliage': 'พุ่มใบ (foliage)',
                'shrub': 'พุ่มไม้ (shrub)', 'rock': 'กำแพงหิน (rock wall)', 'boulder': 'ก้อนหิน (boulder)'}.get(kd, kd)
        nc = int(np.searchsorted(crash_i, i, 'right')); c_age = (i - crash_i[nc - 1]) * dt if nc else 1e9
        info.set_text(f'สิ่งกีดขวาง{what}:\n{kind}\nระยะจริงจากตัวโดรน (true clearance): {max(dist, 0):.2f} m\n'
                      f'ความเร็ว (speed): {R.speed[i]:.2f} m/s\nสถานะ (state): {STATE_TH[st]}')
        crash_box.set_text('ชน! (CRASH)' if c_age < 1.5 else ''); crash_box.set_visible(c_age < 1.5)
        # caption
        if c_age < 1.5:
            had = R.looming[max(0, crash_i[nc - 1] - int(2 / dt)):crash_i[nc - 1]].any()
            cap = 'ชน! ' + ('สัญญาณเตือนมาช้าเกินไป (alarm came too late)' if had else 'ไม่มีสัญญาณเตือนเลย: ภาพขยายตัวไม่พอให้ LPLC2 เกินเกณฑ์ (no looming alarm)')
        elif age < 1.6:
            cap = 'LPLC2 เกินเกณฑ์ → สั่งเลี้ยวหนีทันที (looming neurons fire → escape turn)'
        elif st in (3, 4):
            cap = 'กำลังเลี้ยว: สมองไม่ใช้ภาพชั่วคราว ตัวตรวจจับถูกรีเซ็ต (turning: vision ignored, detectors reset)'
        elif st == 2 and np.isfinite(th) and S > 0.45 * th:
            cap = 'ของข้างหน้าขยายใหญ่ขึ้น → ตัวตรวจจับชี้ออกด้านนอก สัญญาณ LPLC2 สูงขึ้น (expansion → outward motion → LPLC2 rises)'
        elif st == 2:
            cap = 'บินตรงไปข้างหน้า ตาแมลงเห็นแค่ความสว่าง 120 จุดรอบตัว (flying straight: the eye only sees 120 brightness values)'
        else:
            cap = 'ตั้งหลักหลังเลี้ยว ให้ตัวตรวจจับเริ่มใหม่ (settling after the turn)'
        caption.set_text(cap)
        clock.set_text(f't = {i * dt:6.2f} s   {zone_of(y, R.zones)}'); slowtag.set_text(f'ช้า {a.slow:g}× (slow motion)' if s0 <= i <= s1 else '')
        slowtag.set_visible(s0 <= i <= s1)
        fig.canvas.draw(); enc.add(np.asarray(fig.canvas.buffer_rgba()))
        if kf % 150 == 0: print(f'  frame {kf}/{len(sched)}  ({time.time() - t_start:.0f} s)', flush=True)
    for _ in range(int(0.8 * fps)): enc.add(np.asarray(fig.canvas.buffer_rgba()))      # short hold on the last frame
    p = enc.close(); plt.close(fig)
    if fpv is not None and not a.keep_frames: shutil.rmtree(fdir, ignore_errors=True)
    print(f'wrote {p}  ({(len(sched) + a.intro_secs * fps) / fps:.1f} s)')


if __name__ == '__main__':
    main()
