"""2D replay video of a logged forest flight - from the log alone, no simulator needed.

    python viz/replay2d.py px4/runs/forest/<run id>_<world>.json [--out results/videos/x.mp4] [--speed 4] [--t0 0 --t1 120]

Panels: full-field overview (path so far, crashes, view box) | follow map (forest cross-section at flight height, trail,
heading, the 120 ommatidia drawn as a ring around the drone, crashes / remembered hazards as they happen) | eye strip
(scrolling time x azimuth image of the 120-px eye with the brain state alongside) | clearance + looming timeline | text
(time, zone, progress, speed, state, collisions).
--speed N plays N x faster than real time (every N-th log frame at --fps). Writes mp4 via ffmpeg if one is installed
(PATH, $FFMPEG or a known local copy - nothing is downloaded), otherwise a GIF via Pillow (or ask for .gif with --out)."""
import argparse, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viz.common import Run, Encoder, STATES, STATE_COLORS, EYE_AZ, ROOT, zone_of   # noqa: E402

FOREST_COLORS = dict(trunk='#5b4a3a', trunk_plain='#b8b8b8', branch='#8a6a3a', foliage='#4f8a3a', shrub='#2f6a2a',
                     rock='#9a9a9a', boulder='#9a9a9a', crown='#cfe3c0')        # = px4/make_forest.COLORS
ZONE_SHADE = dict(entry='#fafaf7', trunks='#f3efe8', clearing='#fafaf7', branches='#eef5ea', lowtex='#ececec', exit='#fafaf7')


def hex2rgb(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)


def forest_raster(meta, res=0.03):
    """Top-down RGB image of the obstacle cross-section at flight height (same band as make_forest.plot_map)."""
    from px4.forest_geom import Obstacles
    F = meta['field']; W, H = int((F['xmax'] - F['xmin']) / res), int((F['ymax'] - F['ymin']) / res)
    img = np.full((H, W, 3), 250, np.uint8); ys = F['ymin'] + (np.arange(H) + .5) * res; xs = F['xmin'] + (np.arange(W) + .5) * res
    for n, a, b in meta['zones']: img[(ys >= a) & (ys < b)] = hex2rgb(ZONE_SHADE.get(n, '#fafaf7'))
    obs = Obstacles(meta)
    for sh in obs.footprint(meta['alt'] - 0.15, meta['alt'] + 0.15):
        if sh[0] == 'c':
            _, x, y, r, k = sh; i0, i1 = np.searchsorted(ys, [y - r, y + r]); j0, j1 = np.searchsorted(xs, [x - r, x + r])
            if i1 <= i0 or j1 <= j0: continue
            m = np.hypot(xs[None, j0:j1] - x, ys[i0:i1, None] - y) < r
        else:
            _, x, y, sx, sy, k = sh; i0, i1 = np.searchsorted(ys, [y - sy / 2, y + sy / 2]); j0, j1 = np.searchsorted(xs, [x - sx / 2, x + sx / 2])
            if i1 <= i0 or j1 <= j0: continue
            m = np.ones((i1 - i0, j1 - j0), bool)
        img[i0:i1, j0:j1][m] = hex2rgb(FOREST_COLORS.get(k, '#555555'))
    return img, [F['xmin'], F['xmax'], F['ymin'], F['ymax']]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run'); ap.add_argument('--out'); ap.add_argument('--speed', type=float, default=None, help='x real time (default: 2, or 4 for runs > 120 s)')
    ap.add_argument('--fps', type=float, default=30); ap.add_argument('--t0', type=float, default=0); ap.add_argument('--t1', type=float, default=None)
    ap.add_argument('--window', type=float, default=12, help='follow-map width in metres'); ap.add_argument('--eye-secs', type=float, default=10)
    ap.add_argument('--size', default='1280x720'); ap.add_argument('--gif-every', type=int, default=2, help='GIF fallback: keep every n-th frame')
    a = ap.parse_args()

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    R = Run(a.run); meta = R.meta
    if meta is None: sys.exit(f'world metadata px4/forest/{R.world}.json not found')
    speed = a.speed or (4.0 if R.n * R.dt > 120 else 2.0)
    out = Path(a.out) if a.out else ROOT / 'results/videos' / f'{R.path.stem}_2d.mp4'
    i_end = R.n - 1 if a.t1 is None else min(R.n - 1, int(a.t1 / R.dt)); i_start = max(0, int(a.t0 / R.dt))
    idx = np.unique(np.round(np.arange(i_start, i_end + 1e-9, speed / (a.fps * R.dt))).astype(int)); idx = idx[idx <= i_end]
    Wpx, Hpx = (int(v) for v in a.size.split('x')); dpi = 100
    fig = plt.figure(figsize=(Wpx / dpi, Hpx / dpi), dpi=dpi, facecolor='white')
    xc = lambda y: meta['curve']['amp'] * np.sin(2 * np.pi * y / meta['curve']['period']); wid = meta['cfg']['width'] / 2
    ycl = np.linspace(-4, meta['exit_y'] + 4, 400)
    raster, ext = forest_raster(meta)
    crash_i = np.array([c['i'] for c in R.crashes], int); crash_xy = np.array([[c['x'], c['y']] for c in R.crashes]).reshape(-1, 2)
    haz_i = np.array([h[2] for h in R.hazards], int); haz_xy = np.array([h[:2] for h in R.hazards], float).reshape(-1, 2)
    fig.text(0.01, 0.975, R.label(), fontsize=10.5, weight='bold', va='top')
    fig.text(0.01, 0.945, f'{meta["preset"]} forest, seed {meta["seed"]}, corridor {meta["cfg"]["width"]} m   |   replay {speed:g}x real time   |   '
             f'map = obstacle cross-section at {meta["alt"]} m', fontsize=8, va='top', color='#555')

    # ---- overview
    ao = fig.add_axes([0.005, 0.03, 0.12, 0.88]); ao.imshow(raster, extent=ext, origin='lower', interpolation='nearest')
    for s in (-wid, wid): ao.plot(xc(ycl) + s, ycl, color='#c9a', lw=.5)
    ao.axhline(meta['exit_y'], color='g', lw=.8, ls='--'); ao.set_xlim(-10, 10); ao.set_ylim(-4, meta['exit_y'] + 6); ao.set_aspect('equal')
    ao.set_xticks([]); ao.set_yticks([])
    for n, lo, hi in meta['zones']:
        if lo > -4: ao.axhline(lo, color='#bbb', lw=.4)
        ao.text(-9.6, (max(lo, -4) + min(hi, meta['exit_y'] + 6)) / 2, n, rotation=90, fontsize=6, va='center', color='#666')
    o_trail, = ao.plot([], [], color='#1565c0', lw=.8); o_dot, = ao.plot([], [], 'o', color='#1565c0', ms=4)
    o_cr = ao.scatter([], [], marker='x', color='#d32f2f', s=14, lw=1, zorder=5)
    o_box = Rectangle((0, 0), 1, 1, fill=False, ec='#333', lw=.6); ao.add_patch(o_box)

    # ---- follow map
    am = fig.add_axes([0.13, 0.03, 0.45, 0.88]); am.imshow(raster, extent=ext, origin='lower', interpolation='nearest')
    for s in (-wid, wid): am.plot(xc(ycl) + s, ycl, color='#c9a', lw=.8)
    am.plot(xc(ycl), ycl, color='#c9a', lw=.6, ls='--'); am.axhline(meta['exit_y'], color='g', lw=1.2, ls='--')
    am.set_aspect('equal'); am.tick_params(labelsize=7)
    m_trail, = am.plot([], [], color='#1565c0', lw=1.2, alpha=.8)
    m_recent, = am.plot([], [], color='#0d47a1', lw=2.4)
    ring = am.scatter(np.zeros(120), np.zeros(120), c=np.zeros(120), cmap='gray', vmin=0, vmax=255, s=9, lw=0, zorder=6)
    body = Circle((0, 0), 0.35, fc='#1565c0', ec='white', lw=1, alpha=.85, zorder=7); am.add_patch(body)
    head, = am.plot([], [], color='#0d1b4a', lw=2.2, zorder=8, solid_capstyle='round')
    m_cr = am.scatter([], [], marker='x', color='#d32f2f', s=60, lw=2, zorder=9)
    m_hz = am.scatter([], [], marker='^', color='#ff9800', ec='#7a4a00', s=30, lw=.5, zorder=8)
    flash = Circle((0, 0), 0.5, fill=False, ec='#d32f2f', lw=3, alpha=0, zorder=10); am.add_patch(flash)
    am.text(0.01, 0.995, 'follow view  (ring = the 120 ommatidia, brightness as seen;  x = crash,  ^ = remembered hazard)',
            transform=am.transAxes, fontsize=7, va='top', color='#333', bbox=dict(fc='white', ec='none', alpha=.7, pad=1))
    aspect = (0.88 * Hpx) / (0.45 * Wpx); hw = a.window / 2; hh = hw * aspect

    # ---- eye strip (time x azimuth, newest at the top) + state stripe
    neye = int(round(a.eye_secs / R.dt)); az_deg = np.degrees(EYE_AZ)
    ae = fig.add_axes([0.635, 0.43, 0.33, 0.47])
    eye_im = ae.imshow(np.zeros((neye, 120), np.uint8), cmap='gray', vmin=0, vmax=255, aspect='auto', interpolation='nearest',
                       extent=[az_deg[-1] + 1.4, az_deg[0] - 1.4, a.eye_secs, 0])     # left of the drone on the left, newest row at top
    ae.set_title('eye: 120 ommatidia over time (left | front | right)', fontsize=8, pad=3); ae.tick_params(labelsize=7)
    ae.set_xticks([150, 90, 0, -90, -150]); ae.set_xticklabels(['150L', '90L', '0', '90R', '150R']); ae.set_ylabel('seconds ago', fontsize=7)
    ast = fig.add_axes([0.980, 0.43, 0.012, 0.47]); from matplotlib.colors import ListedColormap
    st_im = ast.imshow(np.zeros((neye, 1)), cmap=ListedColormap(STATE_COLORS), vmin=-.5, vmax=4.5, aspect='auto', interpolation='nearest')
    ast.set_xticks([]); ast.set_yticks([])
    lm_ax = fig.add_axes([0.967, 0.43, 0.010, 0.47]); lm_im = lm_ax.imshow(np.zeros((neye, 1)), cmap=ListedColormap(['#ffffff', '#d81b60']), vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    lm_ax.set_xticks([]); lm_ax.set_yticks([])
    fig.text(0.972, 0.905, 'loom', fontsize=6, color='#d81b60', ha='right'); fig.text(0.996, 0.905, 'state', fontsize=6, ha='right')

    # ---- clearance / looming timeline
    tw = 20.0; at = fig.add_axes([0.665, 0.205, 0.325, 0.165]); at.set_xlim(-tw, 0); gmax = 3.0; at.set_ylim(-0.6, gmax)
    at.axhline(0, color='#d32f2f', lw=.8); at.tick_params(labelsize=7); at.set_ylabel('clearance m', fontsize=7); at.set_xlabel('seconds', fontsize=7, labelpad=1)
    g_line, = at.plot([], [], color='#1565c0', lw=1.1); l_line, = at.plot([], [], color='#d81b60', lw=1.5, drawstyle='steps-post')
    c_lines = at.scatter([], [], marker='x', color='#d32f2f', s=30, lw=1.5, zorder=5)
    at.text(0.01, 0.96, 'blue = clearance to nearest obstacle (truth), pink = looming flag', transform=at.transAxes, fontsize=6.5, va='top')

    # ---- text
    txt = fig.text(0.635, 0.155, '', fontsize=9.5, va='top', family='monospace', linespacing=1.45)
    badge = fig.text(0.95, 0.155, '', fontsize=10, va='top', ha='right', weight='bold', color='white', bbox=dict(fc='#999', ec='none', boxstyle='round,pad=0.3'))
    loom_b = fig.text(0.95, 0.10, 'LOOMING', fontsize=10, va='top', ha='right', weight='bold', color='white', bbox=dict(fc='#d81b60', ec='none', boxstyle='round,pad=0.3'))
    crash_t = fig.text(0.95, 0.045, '', fontsize=10, va='top', ha='right', weight='bold', color='white', bbox=dict(fc='#d32f2f', ec='none', boxstyle='round,pad=0.3'))
    prog = fig.add_axes([0.635, 0.015, 0.2, 0.012]); prog.set_xlim(0, 1); prog.set_ylim(0, 1); prog.axis('off')
    prog.add_patch(Rectangle((0, 0), 1, 1, fc='#e0e0e0', ec='none')); pbar = Rectangle((0, 0), 0, 1, fc='#43a047', ec='none'); prog.add_patch(pbar)

    enc = Encoder(out, Wpx, Hpx, a.fps, channels=4, gif_every=a.gif_every); t_start = time.time()
    best_y = -1e9; ys_run = R.y
    for k, i in enumerate(idx):
        x, y, h = R.x[i], R.y[i], R.h[i]; best_y = max(best_y, ys_run[max(i_start, 0):i + 1].max())
        o_trail.set_data(R.x[:i + 1], R.y[:i + 1]); o_dot.set_data([x], [y]); o_box.set_bounds(x - hw, y - hh, 2 * hw, 2 * hh)
        nc = int(np.searchsorted(crash_i, i, 'right')); o_cr.set_offsets(crash_xy[:nc]); m_cr.set_offsets(crash_xy[:nc])
        nh = int((haz_i <= i).sum()); m_hz.set_offsets(haz_xy[haz_i <= i] if nh else np.zeros((0, 2)))
        am.set_xlim(x - hw, x + hw); am.set_ylim(y - hh, y + hh)
        m_trail.set_data(R.x[:i + 1], R.y[:i + 1]); r0 = max(0, i - int(3 / R.dt)); m_recent.set_data(R.x[r0:i + 1], R.y[r0:i + 1])
        ang = h + EYE_AZ; ring.set_offsets(np.c_[x + 0.85 * np.cos(ang), y + 0.85 * np.sin(ang)]); ring.set_array(R.eye[i].astype(float))
        body.center = (x, y); head.set_data([x, x + 1.3 * np.cos(h)], [y, y + 1.3 * np.sin(h)])
        st = R.state[i] if len(R.state) > i else 0; body.set_facecolor(STATE_COLORS[st])
        # crash flash: 1.2 s expanding ring + label
        last = crash_i[:nc][-1] if nc else -10 ** 9; age = (i - last) * R.dt
        if age < 1.2: flash.center = tuple(crash_xy[nc - 1]); flash.set_radius(0.4 + 1.5 * age); flash.set_alpha(1 - age / 1.2)
        else: flash.set_alpha(0)
        c = R.crashes[nc - 1] if nc else None
        crash_t.set_text(f"CRASH #{nc}: {c['kind']}" if (c and age < 2.0) else ''); crash_t.set_visible(bool(c and age < 2.0))
        # eye strip
        j0 = i - neye + 1; blk = R.eye[max(0, j0):i + 1][::-1]; stb = R.state[max(0, j0):i + 1][::-1]; lmb = R.looming[max(0, j0):i + 1][::-1]
        if len(blk) < neye:
            pad = neye - len(blk); blk = np.vstack([blk, np.zeros((pad, 120), np.uint8)]); stb = np.r_[stb, np.zeros(pad, int)]; lmb = np.r_[lmb, np.zeros(pad, int)]
        eye_im.set_data(blk[:, ::-1]); st_im.set_data(stb[:, None]); lm_im.set_data(lmb[:, None])
        # timeline
        w0 = max(0, i - int(tw / R.dt)); tt = R.t[w0:i + 1] - R.t[i]
        g_line.set_data(tt, np.clip(R.gap[w0:i + 1], -0.6, gmax)); l_line.set_data(tt, R.looming[w0:i + 1] * (gmax * .9))
        cm = (crash_i >= w0) & (crash_i <= i); c_lines.set_offsets(np.c_[R.t[crash_i[cm]] - R.t[i], np.zeros(cm.sum())] if cm.any() else np.zeros((0, 2)))
        # text
        zn = zone_of(y, R.zones); frac = float(np.clip(best_y / R.exit_y, 0, 1))
        txt.set_text(f't {R.t[i]:6.1f} s   zone {zn:<8s}\n'
                     f'y {y:5.1f} m  best {best_y:5.1f} / {R.exit_y:.0f} m\n'
                     f'speed {R.speed[i]:4.2f} m/s  alt {R.z[i]:4.2f} m\n'
                     f'collisions {nc:3d}   hazards {nh:3d}')
        badge.set_text(STATES[st].replace('_', ' ')); badge.get_bbox_patch().set_facecolor(STATE_COLORS[st])
        loom_b.set_visible(bool(R.looming[i])); pbar.set_width(frac)
        fig.canvas.draw(); enc.add(np.asarray(fig.canvas.buffer_rgba()))
        if k % 300 == 0: print(f'  frame {k}/{len(idx)}  ({time.time() - t_start:.0f} s)', flush=True)
    path = enc.close(); plt.close(fig)
    print(f'wrote {path}  ({len(idx)} frames, {len(idx) / a.fps:.0f} s at {speed:g}x, {time.time() - t_start:.0f} s to render)')


if __name__ == '__main__':
    main()
