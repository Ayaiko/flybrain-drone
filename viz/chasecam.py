"""3D chase-cam video of a logged forest flight, rendered by Gazebo from the logged poses (no PX4, no physics).

Host:   python viz/chasecam.py px4/runs/forest/<run id>_<world>.json [--out results/videos/x.mp4] [--speed 2] [--fps 24]
                               [--t0 40 --t1 120] [--cam chase|high|side] [--size 640x360] [--keep-frames]
  Starts (or reuses) the container flybrain-px4-viz (image flybrain-px4:worker, GZ partition fb_viz, no ports), runs this
  same script inside it with --inside, then encodes the frames with ffmpeg on the host (GIF via Pillow if no ffmpeg).

Inside: a copy of the forest world (viz/_worlds/<world>_chase.sdf: gravity off, only Physics / UserCommands / Sensors
  systems, an x500 body and one camera model) runs paused under gz sim server (software ogre2 on Xvfb). For every video
  frame the drone and camera poses are set with /world/<w>/set_pose_vector, the world is stepped one iteration and the
  camera image is saved with a HUD (time, zone, progress, state, looming, collisions, crash flash, the 120-px eye band).
  The drone's roll / pitch are not logged; they are approximated from the logged acceleration (cosmetic only).
  The camera is a smoothed chase cam (heading low-passed, so saccades and crash turn-arounds swing it round gently);
  its near clip plane is pushed out so trunks between camera and drone are cut away instead of blocking the view.
Software rendering is slow (~0.1-0.3 s / frame at 640x360): long runs -> use --speed or --t0/--t1."""
import argparse, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
from viz.common import Run, STATES, STATE_COLORS, zone_of, find_ffmpeg, h264_args   # noqa: E402

CONTAINER, IMAGE = 'flybrain-px4-viz', 'flybrain-px4:worker'
GZ = '/work/px4/PX4-Autopilot/Tools/simulation/gz'
CAMS = {  # back (m), up (m), look-ahead (m), near clip (m), fov (rad), heading time-constant (s)
    'chase': dict(back=2.6, up=1.1, ahead=1.2, near=1.1, fov=1.45, tau=0.8),
    'high':  dict(back=4.0, up=3.2, ahead=1.5, near=2.2, fov=1.35, tau=1.0),
    'side':  dict(back=0.0, up=0.8, ahead=0.0, near=1.6, fov=1.40, tau=1.5, side=3.2),
}


# ---------------------------------------------------------------- host side
def host(a, argv):
    run = Path(a.run).resolve(); rel = run.relative_to(ROOT)
    out = Path(a.out).resolve() if a.out else ROOT / 'results/videos' / f'{run.stem}_chase.mp4'
    frames = ROOT / 'results/videos' / f'_frames_{out.stem}'
    if frames.exists(): shutil.rmtree(frames)
    st = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', CONTAINER], capture_output=True, text=True)
    if st.returncode:
        print(f'starting container {CONTAINER}', flush=True)
        subprocess.run(['docker', 'run', '-d', '--name', CONTAINER, '--user', 'px4', '-e', 'FLYBRAIN_WORKER=viz', '--shm-size=1g',
                        '-v', f'{ROOT}:/work', IMAGE, 'sleep', 'infinity'], check=True)
    elif st.stdout.strip() != 'true':
        subprocess.run(['docker', 'start', CONTAINER], check=True)
    rest = [x for x in argv if x not in (a.run,)]
    inner = ['python3', '/work/viz/chasecam.py', f'/work/{rel}', '--inside', '--frames', f'/work/{frames.relative_to(ROOT)}', *rest]
    cmd = f'export LP_NUM_THREADS={a.threads} && source /work/px4/sim_env.sh && ' + ' '.join(f"'{x}'" for x in inner)
    t0 = time.time(); r = subprocess.run(['docker', 'exec', CONTAINER, 'bash', '-lc', cmd])
    if r.returncode: sys.exit(f'render failed ({r.returncode}); frames left in {frames}')
    n = len(list(frames.glob('*.jpg'))); print(f'rendered {n} frames in {time.time() - t0:.0f} s; encoding', flush=True)
    ff = find_ffmpeg(); out.parent.mkdir(parents=True, exist_ok=True)
    if ff and out.suffix != '.gif':
        subprocess.run([ff, '-y', '-hide_banner', '-loglevel', 'error', '-framerate', f'{a.fps}', '-i', str(frames / '%06d.jpg'),
                        *h264_args(ff, '1.5M'), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)], check=True)
    else:
        from PIL import Image
        out = out.with_suffix('.gif'); fs = sorted(frames.glob('*.jpg'))[::2]
        ims = [Image.open(f).convert('P', palette=Image.ADAPTIVE, colors=128) for f in fs]
        ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(2000 / a.fps), loop=0)
    if not a.keep_frames: shutil.rmtree(frames)
    print(f'wrote {out}')


# ---------------------------------------------------------------- inside the container
def quat(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2), np.cos(yaw / 2), np.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)  # w x y z


def lowpass(v, dt, tau):
    out = np.empty_like(v); out[0] = v[0]; k = dt / (tau + dt)
    for i in range(1, len(v)): out[i] = out[i - 1] + k * (v[i] - out[i - 1])
    return out


def trajectory(R, cam):
    """Per-log-frame drone pose (x, y, z, roll, pitch, yaw) and camera pose (x, y, z, pitch, yaw)."""
    dt = R.dt; h = np.unwrap(R.h)
    px, py, pz = (lowpass(v, dt, 0.12) for v in (R.x, R.y, R.z))
    vx, vy = np.gradient(lowpass(R.x, dt, 0.25)) / dt, np.gradient(lowpass(R.y, dt, 0.25)) / dt
    ax, ay = lowpass(np.gradient(vx) / dt, dt, 0.2), lowpass(np.gradient(vy) / dt, dt, 0.2)
    af = ax * np.cos(h) + ay * np.sin(h); al = -ax * np.sin(h) + ay * np.cos(h)          # body-frame forward / left accel
    pitch = np.clip(np.arctan2(af, 9.81), -.35, .35); roll = np.clip(np.arctan2(-al, 9.81), -.35, .35)
    drone = np.c_[R.x, R.y, R.z, roll, pitch, h]
    hs = lowpass(h, dt, cam['tau']); fx, fy = np.cos(hs), np.sin(hs)
    side = cam.get('side', 0.0)
    cx = lowpass(px, dt, 0.35) - cam['back'] * fx - side * fy; cy = lowpass(py, dt, 0.35) - cam['back'] * fy + side * fx
    cz = lowpass(pz, dt, 0.5) + cam['up']
    tx, ty, tz = px + cam['ahead'] * fx, py + cam['ahead'] * fy, pz - 0.1
    cyaw = np.arctan2(ty - cy, tx - cx); cpitch = np.arctan2(cz - tz, np.hypot(tx - cx, ty - cy))
    return drone, np.c_[cx, cy, cz, cpitch, cyaw]


def write_world(world, cam, W, H):
    src = Path(f'{GZ}/worlds/{world}.sdf').read_text()
    s = src.replace(f'<world name="{world}">', f'<world name="{world}_chase">').replace('<gravity>0 0 -9.8</gravity>', '<gravity>0 0 0</gravity>')
    plugins = '''
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>'''
    extra = f'''
    <include><uri>model://x500_base</uri><name>drone</name><pose>0 0 1.5 0 0 1.5708</pose></include>
    <model name="chasecam"><pose>0 -3 2.5 0 0.3 1.5708</pose><link name="l"><gravity>false</gravity>
      <inertial><mass>0.1</mass><inertia><ixx>1e-3</ixx><iyy>1e-3</iyy><izz>1e-3</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <sensor name="c" type="camera"><camera><horizontal_fov>{cam['fov']}</horizontal_fov><image><width>{W}</width><height>{H}</height></image>
        <clip><near>{cam['near']}</near><far>150</far></clip></camera><always_on>1</always_on><update_rate>250</update_rate><topic>chase/cam</topic></sensor>
    </link></model>'''
    s = s.replace(f'<world name="{world}_chase">', f'<world name="{world}_chase">' + plugins, 1).replace('</world>', extra + '\n  </world>')
    d = HERE / '_worlds'; d.mkdir(exist_ok=True); p = d / f'{world}_chase.sdf'; p.write_text(s); return p


def font(sz):
    from PIL import ImageFont
    for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        if Path(f).exists(): return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def hud(img, R, i, nc, crash_age, best_y, W, H):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img); d = ImageDraw.Draw(im, 'RGBA'); f1, f2 = font(max(11, H // 26)), font(max(10, H // 32))
    zn = zone_of(R.y[i], R.zones); st = int(R.state[i]) if len(R.state) > i else 0
    d.rectangle([0, 0, W, H // 9], fill=(0, 0, 0, 120))
    d.text((8, 5), f't {R.t[i]:5.1f} s   {zn}   y {R.y[i]:4.1f} m (best {best_y:4.1f} / {R.exit_y:.0f})', font=f1, fill='white')
    d.text((8, 5 + H // 22), f'{R.mode}   collisions {nc}   {R.speed[i]:.2f} m/s', font=f2, fill=(220, 220, 220))
    lab = STATES[st].replace('_', ' '); tw = d.textlength(lab, font=f1); col = STATE_COLORS[st]
    d.rounded_rectangle([W - tw - 20, 6, W - 6, 8 + H // 20], 5, fill=col); d.text((W - tw - 13, 7), lab, font=f1, fill='white')
    if R.looming[i]:
        lt = 'LOOMING'; w2 = d.textlength(lt, font=f2)
        d.rounded_rectangle([W - w2 - 20, 12 + H // 20, W - 6, 14 + H // 20 + H // 24], 5, fill='#d81b60'); d.text((W - w2 - 13, 13 + H // 20), lt, font=f2, fill='white')
    # eye band: the 120 ommatidia, left of the drone on the left
    bw, bh = int(W * 0.7), max(8, H // 28); x0 = (W - bw) // 2; y0 = H - bh - 8
    band = Image.fromarray(np.repeat(R.eye[i][::-1][None, :], 1, 0)).resize((bw, bh), Image.NEAREST).convert('RGB')
    d.rectangle([x0 - 2, y0 - 2, x0 + bw + 1, y0 + bh + 1], outline='white'); im.paste(band, (x0, y0)); d = ImageDraw.Draw(im, 'RGBA')
    d.line([W // 2, y0 - 4, W // 2, y0 + bh + 3], fill=(255, 220, 0), width=1)
    d.text((x0 - 4, y0 + bh // 2), 'L', font=f2, fill='white', anchor='rm'); d.text((x0 + bw + 4, y0 + bh // 2), 'R', font=f2, fill='white', anchor='lm')
    pw = int(W * 0.7 * np.clip(best_y / R.exit_y, 0, 1)); d.rectangle([x0, y0 - 7, x0 + pw, y0 - 5], fill='#43a047')
    if crash_age < 1.2:
        a = int(255 * (1 - crash_age / 1.2)); c = R.crashes[nc - 1]
        for k in range(6): d.rectangle([k, k, W - 1 - k, H - 1 - k], outline=(229, 57, 53, a))
        msg = f"CRASH #{nc}: {c['kind']}"; tw = d.textlength(msg, font=f1)
        d.rounded_rectangle([(W - tw) / 2 - 10, H * .42, (W + tw) / 2 + 10, H * .42 + H // 15], 6, fill=(229, 57, 53, a))
        d.text(((W - tw) / 2, H * .42 + 3), msg, font=f1, fill=(255, 255, 255, a))
    return im


def inside(a):
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image as GzImage
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.world_control_pb2 import WorldControl
    R = Run(a.run); cam = CAMS[a.cam]; W, H = (int(v) for v in a.size.split('x')); speed = a.speed or (4.0 if R.n * R.dt > 120 else 2.0)
    i_end = R.n - 1 if a.t1 is None else min(R.n - 1, int(a.t1 / R.dt)); i_start = max(0, int(a.t0 / R.dt))
    idx = np.unique(np.round(np.arange(i_start, i_end + 1e-9, speed / (a.fps * R.dt))).astype(int)); idx = idx[idx <= i_end]
    drone, cpose = trajectory(R, cam); sdf = write_world(R.world, cam, W, H); wname = f'{R.world}_chase'
    frames = Path(a.frames); frames.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DISPLAY=':99', LIBGL_ALWAYS_SOFTWARE='1', GZ_SIM_RESOURCE_PATH=f'{GZ}/models:{GZ}/worlds')
    if subprocess.run(['pgrep', '-x', 'Xvfb'], capture_output=True).returncode:
        subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1280x800x24'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1.5)
    subprocess.run(['pkill', '-f', 'gz sim'], capture_output=True); time.sleep(.5)
    log = open('/tmp/chasecam_gz.log', 'w')
    gz = subprocess.Popen(['gz', 'sim', '-s', '-v', '2', str(sdf)], env=env, stdout=log, stderr=subprocess.STDOUT)
    node = Node(); last = {'stamp': -1.0, 'img': None}

    def cb(msg):
        last['img'] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3].copy()
        last['stamp'] = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
    node.subscribe(GzImage, '/chase/cam', cb)

    def set_poses(i):
        req = Pose_V()
        for name, (x, y, z, r, p, yw) in (('drone', drone[i]), ('chasecam', (*cpose[i][:3], 0.0, *cpose[i][3:]))):
            ps = req.pose.add(); ps.name = name; ps.position.x, ps.position.y, ps.position.z = float(x), float(y), float(z)
            w_, x_, y_, z_ = quat(r, p, yw); ps.orientation.w, ps.orientation.x, ps.orientation.y, ps.orientation.z = w_, x_, y_, z_
        ok, _ = node.request(f'/world/{wname}/set_pose_vector', req, Pose_V, Boolean, 3000)
        return ok

    def step():
        wc = WorldControl(); wc.pause = True; wc.multi_step = 1
        node.request(f'/world/{wname}/control', wc, WorldControl, Boolean, 3000)

    t0 = time.time()
    while not set_poses(idx[0]):                                   # wait until the server is up
        if time.time() - t0 > 120 or gz.poll() is not None: sys.exit('gz sim did not come up; see /tmp/chasecam_gz.log')
        time.sleep(1)
    print(f'gz up in {time.time() - t0:.0f} s; rendering {len(idx)} frames ({speed:g}x, {a.fps:g} fps, cam {a.cam})', flush=True)
    def render():                                                  # one iteration; returns once its image has arrived
        prev = last['stamp']; step(); tw = time.time()
        while last['stamp'] <= prev and time.time() - tw < 20: time.sleep(0.002)
    for _ in range(3):                                             # warm-up: first renders load textures
        set_poses(idx[0]); render()
    crash_i = np.array([c['i'] for c in R.crashes], int); best_y = -1e9; t1 = time.time()
    try:
        for k, i in enumerate(idx):
            set_poses(i); render()            # a pose set before an iteration is in that iteration's image (checked against a settled reference)
            if last['img'] is None: sys.exit('no camera image; see /tmp/chasecam_gz.log')
            best_y = max(best_y, R.y[i_start:i + 1].max()); nc = int(np.searchsorted(crash_i, i, 'right'))
            age = (i - crash_i[nc - 1]) * R.dt if nc else 1e9
            hud(last['img'], R, i, nc, age, best_y, W, H).save(frames / f'{k:06d}.jpg', quality=90)
            if k % 100 == 0:
                el = time.time() - t1; print(f'  frame {k}/{len(idx)}  {el:.0f} s  (~{el / max(k, 1) * (len(idx) - k) / 60:.1f} min left)', flush=True)
    finally:
        gz.terminate()
        try: gz.wait(10)
        except subprocess.TimeoutExpired: gz.kill()
        subprocess.run(['pkill', '-f', 'gz sim'], capture_output=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run'); ap.add_argument('--out'); ap.add_argument('--speed', type=float, default=None, help='x real time (default 2, or 4 for runs > 120 s)')
    ap.add_argument('--fps', type=float, default=24); ap.add_argument('--t0', type=float, default=0); ap.add_argument('--t1', type=float, default=None)
    ap.add_argument('--cam', choices=list(CAMS), default='chase'); ap.add_argument('--size', default='640x360')
    ap.add_argument('--threads', type=int, default=4, help='llvmpipe render threads in the container')
    ap.add_argument('--keep-frames', action='store_true'); ap.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--frames', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.inside: inside(a)
    else: host(a, sys.argv[1:])


if __name__ == '__main__':
    main()
