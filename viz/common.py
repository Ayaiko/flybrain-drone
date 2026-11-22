"""Shared helpers for the replay tools: load a logged forest flight, decode the eye stream, find ffmpeg, encode frames."""
import base64, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATES = ['warmup', 'settling', 'fixate', 'saccade', 'saccade_start']           # same order as px4/fly_forest.py
STATE_COLORS = ['#9e9e9e', '#64b5f6', '#43a047', '#fb8c00', '#e53935']
EYE_N, EYE_FOV = 120, 330.0                                                     # flydrone.sim.make_eye(120): -165..+165 deg, +az = left
EYE_AZ = np.radians(np.linspace(-EYE_FOV / 2, EYE_FOV / 2, EYE_N))
ZONES = [('entry', -6, 8), ('trunks', 8, 30), ('clearing', 30, 40), ('branches', 40, 60), ('lowtex', 60, 76), ('exit', 76, 90)]


def zone_of(y, zones=ZONES):
    for n, a, b in zones:
        if a <= y < b: return n
    return 'outside'


class Run:
    """A logged flight: px4/runs/forest/<run id>_<world>.json (+ the world's px4/forest/<world>.json)."""
    def __init__(self, path):
        self.path = Path(path); d = json.loads(self.path.read_text())
        self.summary, self.dt = d['summary'], float(d.get('dt', 1 / 30)); r = d['run']
        for k in ('x', 'y', 'z', 'h', 'gap', 'yawc'): setattr(self, k, np.asarray(r.get(k, []), float))
        self.state = np.asarray(r.get('state', []), int); self.looming = np.asarray(r.get('looming', []), int)
        self.n = len(self.x); self.t = np.arange(self.n) * self.dt
        raw = base64.b64decode(r.get('img', '')) if r.get('img') else b''
        self.eye = (np.frombuffer(raw, np.uint8).reshape(-1, EYE_N) if raw else np.zeros((self.n, EYE_N), np.uint8))[:self.n]
        if len(self.eye) < self.n: self.eye = np.vstack([self.eye, np.zeros((self.n - len(self.eye), EYE_N), np.uint8)])
        self.crashes = r.get('crashes', []); self.hazards = r.get('hazards', [])
        self.world = self.summary.get('world') or self.path.stem.rsplit('_ff_', 1)[-1]
        self.mode = self.summary.get('mode', self.path.stem)
        wj = ROOT / 'px4/forest' / f'{self.world}.json'
        self.meta = json.loads(wj.read_text()) if wj.exists() else None
        self.zones = self.meta['zones'] if self.meta else ZONES
        self.exit_y = self.meta['exit_y'] if self.meta else 80.0
        v = np.hypot(np.gradient(self.x), np.gradient(self.y)) / self.dt if self.n > 1 else np.zeros(self.n)
        self.speed = np.convolve(v, np.ones(9) / 9, 'same')                       # light smoothing, log positions are quantised

    def label(self):
        s = self.summary
        return f"{self.mode} on {self.world}  -  {'reached exit' if s.get('reached') else 'did not reach exit'}, {s.get('collisions', len(self.crashes))} collisions, progress {s.get('progress_m', '?')} m"


# ---------------------------------------------------------------- ffmpeg
FFMPEG_CANDIDATES = ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/usr/bin/ffmpeg',
                     '/Applications/Live2D Cubism 5.3/tools/ffmpeg/ffmpeg']          # already-installed copies only; nothing is downloaded


def find_ffmpeg():
    for c in [os.environ.get('FFMPEG'), shutil.which('ffmpeg'), *FFMPEG_CANDIDATES]:
        if c and Path(c).is_file() and os.access(c, os.X_OK): return c
    return None


def h264_args(ffmpeg, bitrate='2.5M'):
    enc = subprocess.run([ffmpeg, '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    if ' libx264 ' in enc: return ['-c:v', 'libx264', '-preset', 'medium', '-crf', '20']
    if ' h264_videotoolbox ' in enc: return ['-c:v', 'h264_videotoolbox', '-b:v', bitrate, '-allow_sw', '1']
    if ' libopenh264 ' in enc: return ['-c:v', 'libopenh264', '-b:v', bitrate]
    return ['-c:v', 'mpeg4', '-q:v', '3']


class Encoder:
    """Feed RGB(A) uint8 frames; writes an mp4 through ffmpeg, or a GIF through Pillow when ffmpeg is missing / out is .gif."""
    def __init__(self, out, w, h, fps, channels=3, gif_every=1):
        self.out, self.w, self.h, self.fps, self.ch = Path(out), w, h, fps, channels
        self.out.parent.mkdir(parents=True, exist_ok=True); self.ff = None if self.out.suffix == '.gif' else find_ffmpeg()
        if self.out.suffix != '.gif' and not self.ff:
            print('ffmpeg not found (set FFMPEG=/path/to/ffmpeg) - writing a GIF instead', file=sys.stderr); self.out = self.out.with_suffix('.gif')
        if self.ff:
            pix = 'rgba' if channels == 4 else 'rgb24'
            self.p = subprocess.Popen([self.ff, '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', pix, '-s', f'{w}x{h}',
                                       '-r', f'{fps}', '-i', '-', *h264_args(self.ff), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.out)],
                                      stdin=subprocess.PIPE)
        else:
            self.frames, self.k, self.every = [], 0, max(1, gif_every)

    def add(self, rgb):
        if self.ff: self.p.stdin.write(np.ascontiguousarray(rgb).tobytes()); return
        if self.k % self.every == 0:
            from PIL import Image
            self.frames.append(Image.fromarray(np.ascontiguousarray(rgb[:, :, :3])).convert('P', palette=Image.ADAPTIVE, colors=128))
        self.k += 1

    def close(self):
        if self.ff:
            self.p.stdin.close(); rc = self.p.wait()
            if rc: raise RuntimeError(f'ffmpeg exited with {rc}')
        elif self.frames:
            self.frames[0].save(self.out, save_all=True, append_images=self.frames[1:], duration=int(1000 * self.every / self.fps), loop=0, optimize=False)
        return self.out
