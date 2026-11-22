"""x500 under PX4 offboard body-velocity + yaw-rate control, reduced to what the fly brain can feel.

State: position, world-frame horizontal velocity, heading, yaw rate, altitude. Per frame (1/30 s) the command
(forward speed v, yaw rate r, vertical speed vz) is delayed by `delay_v` / `delay_r` frames (PX4 / MAVLink / pose-stream
latency, fitted separately for speed and yaw), then, in `sub` sub-steps:
  yaw rate  -> first order (tau_r) toward clip(r, +-rmax), yaw acceleration limited to alpha
  velocity  -> the body-frame setpoint v * (cos h, sin h) is a world vector that rotates with the heading; the actual
               velocity follows it first order (tau_v), with the change limited to acc (speeding up) / dec (slowing
               down) m/s^2 -> lag, side-slip in turns, braking distance
  altitude  -> first order toward vz (tau_z)
  vel_off   -> optional fixed angle (rad) between the true heading and the direction PX4 flies the body-x velocity in
               (seen only in the PX4 route-follower runs: -2..-6 deg in 8 of 9; 0 in all 18 brain runs) - default 0
Parameters are fitted to the logged PX4 forest runs by fastsim/fit_dynamics.py (fastsim/dyn_params.json)."""
import json
from pathlib import Path
import numpy as np

PARAM_FILE = Path(__file__).resolve().parent / 'dyn_params.json'
DEFAULT = dict(delay_v=4, delay_r=4, tau_v=0.35, acc=2.0, dec=2.0, tau_r=0.15, rmax=1.9, alpha=8.0, tau_z=0.5, vel_off=0.0)
NAMES_R = ['tau_r', 'rmax', 'alpha']; NAMES_V = ['tau_v', 'acc', 'dec']


def load_params():
    P = dict(DEFAULT)
    if PARAM_FILE.exists(): P.update({k: v for k, v in json.loads(PARAM_FILE.read_text()).items() if k in DEFAULT})
    return P


def substeps(P, h, r, vx, vy, vc, rc, ds, n):
    """Vectorised core (works on floats or arrays): n sub-steps of length ds with fixed commands."""
    tv, acc, dec, tr, rmax, alpha = P['tau_v'], P['acc'], P['dec'], P['tau_r'], P['rmax'], P['alpha']
    rt = np.clip(rc, -rmax, rmax); px = py = 0.0; vo = P.get('vel_off', 0.0)
    kr, kv = 1 - np.exp(-ds / tr), 1 - np.exp(-ds / tv)          # exact first-order step (stable for any tau)
    for _ in range(n):
        r = r + np.clip((rt - r) * kr, -alpha * ds, alpha * ds)
        h = h + r * ds
        sx, sy = vc * np.cos(h + vo), vc * np.sin(h + vo)
        dx, dy = (sx - vx) * kv, (sy - vy) * kv
        lim = np.where(dx * vx + dy * vy >= 0, acc, dec) * ds
        m = np.sqrt(dx * dx + dy * dy); s = np.minimum(1.0, lim / np.maximum(m, 1e-12))
        vx = vx + dx * s; vy = vy + dy * s
        px = px + vx * ds; py = py + vy * ds
    return h, r, vx, vy, px, py


class X500:
    def __init__(self, params=None, dt=1 / 30, sub=4):
        self.P = dict(load_params()); self.P.update(params or {}); self.dt, self.sub = dt, sub

    def reset(self, x, y, z, h, v=0.0):
        self.x, self.y, self.z, self.h = float(x), float(y), float(z), float(h)
        self.vx, self.vy, self.r, self.vz = v * np.cos(h), v * np.sin(h), 0.0, 0.0
        self.dv, self.dr = int(self.P['delay_v']), int(self.P['delay_r'])
        self.q = [(0.0, 0.0, 0.0)] * (max(self.dv, self.dr) + 1)

    def step(self, v, yaw, vz=0.0):
        """Apply this frame's command; advance one frame. Returns the new pose (x, y, z, h)."""
        self.q.append((float(v), float(yaw), float(vz))); self.q.pop(0)
        vc, vzc = self.q[-1 - self.dv][0], self.q[-1 - self.dv][2]; rc = self.q[-1 - self.dr][1]
        h, r, vx, vy, px, py = substeps(self.P, self.h, self.r, self.vx, self.vy, vc, rc, self.dt / self.sub, self.sub)
        self.h = float((h + np.pi) % (2 * np.pi) - np.pi); self.r, self.vx, self.vy = float(r), float(vx), float(vy)
        self.x += float(px); self.y += float(py)
        self.vz += (vzc - self.vz) * min(1.0, self.dt / self.P['tau_z']); self.z += self.vz * self.dt
        return self.x, self.y, self.z, self.h


class X500Batch:
    """N independent x500 models stepped together (same equations as X500; arrays of shape (N,))."""
    def __init__(self, n, params=None, dt=1 / 30, sub=4):
        self.P = dict(load_params()); self.P.update(params or {}); self.dt, self.sub, self.n = dt, sub, n
        self.dv, self.dr = int(self.P['delay_v']), int(self.P['delay_r'])

    def reset(self, x, y, z, h):
        self.x, self.y, self.z, self.h = (np.array(a, float) for a in (x, y, z, h))
        self.vx, self.vy, self.r, self.vz = (np.zeros(self.n) for _ in range(4))
        self.q = np.zeros((max(self.dv, self.dr) + 1, self.n, 3))

    def step(self, idx, v, yaw, vz):
        """Advance the drones idx (int array) by one frame with this frame's commands (arrays like idx)."""
        self.q[:, idx] = np.roll(self.q[:, idx], -1, axis=0); self.q[-1, idx] = np.c_[v, yaw, vz]
        vc, vzc = self.q[-1 - self.dv, idx, 0], self.q[-1 - self.dv, idx, 2]; rc = self.q[-1 - self.dr, idx, 1]
        h, r, vx, vy, px, py = substeps(self.P, self.h[idx], self.r[idx], self.vx[idx], self.vy[idx], vc, rc, self.dt / self.sub, self.sub)
        self.h[idx] = (h + np.pi) % (2 * np.pi) - np.pi; self.r[idx], self.vx[idx], self.vy[idx] = r, vx, vy
        self.x[idx] += px; self.y[idx] += py
        self.vz[idx] += (vzc - self.vz[idx]) * min(1.0, self.dt / self.P['tau_z']); self.z[idx] += self.vz[idx] * self.dt
