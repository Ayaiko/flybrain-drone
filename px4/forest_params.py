"""Parameter sets and early-stop rule for forest flights (stdlib + numpy only; used in the container and on the host).

resolve(name, overrides) -> full parameter dict: px4/forest_configs.json 'defaults' <- named set (with 'base'
inheritance) <- overrides. A name that is not in the file is parsed as an old-style mode string
(e.g. 'lplc2-bold-gate-sv0-gg0-v0.8'), so earlier runs stay reproducible."""
import json
from pathlib import Path

CFG_FILE = Path(__file__).with_name('forest_configs.json')


def load_configs():
    return json.loads(CFG_FILE.read_text())


def parse_mode(s):
    parts = s.split('-'); P = dict(brain=parts[0])
    if 'bold' in parts: P.update(lplc2_thr=2.5, persist=3, haz_expire=20)
    if 'gate' in parts: P.update(rot_gate=15, sac_speed=1.0, sac_end_wy=12)
    for p in parts[1:]:
        if p.startswith('sv'): P['sac_v'] = float(p[2:])
        elif p.startswith('gg'): P['goal_gap'] = float(p[2:])
        elif p.startswith('gt'): P['goal_thr'] = float(p[2:])
        elif p.startswith('v'): P['speed'] = float(p[1:])
    return P


def resolve(name, overrides=None):
    C = load_configs(); P = dict(C['defaults'])
    def apply(n, depth=0):
        if depth > 10: raise ValueError('config base loop at ' + n)
        if n in C and not n.startswith('_') and n != 'defaults':
            c = dict(C[n]); b = c.pop('base', None)
            if b: apply(b, depth + 1)
            P.update(c)
        else:
            P.update(parse_mode(n))
    apply(name); P.update(overrides or {})
    unknown = set(P) - set(C['defaults'])
    if unknown: raise ValueError(f'unknown parameter(s) {sorted(unknown)}; known: {sorted(C["defaults"])}')
    return P


def run_id(name, overrides=None):
    return name if not overrides else name + '+' + '+'.join(f'{k}={v}' for k, v in sorted(overrides.items()))


def parse_value(v):
    try: return json.loads(v)
    except ValueError: return v


class Stall:
    """Early stop: True once progress along the route has grown by less than stall_m during the last stall_s seconds."""
    def __init__(self, stall_s, stall_m):
        self.s, self.m, self.best, self.t_best = stall_s, stall_m, None, 0.0

    def update(self, now, prog):
        if self.s is None: return False
        if self.best is None or prog >= self.best + self.m: self.best, self.t_best = prog, now
        return now - self.t_best > self.s


def debounce(run, hyst=0.05):
    """Indices of contact events with the live 5 cm hysteresis applied (old runs counted every re-touch of a graze)."""
    g = run['gap']; keep, last = [], None
    for i in sorted(c['i'] for c in run['crashes']):
        if last is None or (i > last and max(g[last:i]) >= hyst): keep.append(i)
        last = i
    return keep


def rescore(run, dt, exit_y, stall_s, stall_m, max_coll, limit_s=None, reached=False):
    """Apply an early-stop rule after the fact to a run recorded without it (same logic as live) ->
    dict(collisions, progress_m, time_s, end_reason, reached). Lets old full-length runs be compared with new ones."""
    y = run['y']; crash_i = debounce(run); st = Stall(stall_s, stall_m); best = -99.0
    n = len(y); stop, reason = n, None
    for i in range(n):
        t = i * dt; best = max(best, y[i])
        if limit_s is not None and t > limit_s: stop, reason = i, 'timeout'; break
        if max_coll is not None and sum(ci <= i for ci in crash_i) >= max_coll: stop, reason = i + 1, 'too_many_collisions'; break
        if st.update(t, min(max(y[i], 0.0), exit_y)): stop, reason = i + 1, 'stalled'; break
    reached = reason is None and reached
    return dict(collisions=sum(ci < stop for ci in crash_i), progress_m=round(exit_y if reached else min(max(y[:stop] or [0]), exit_y), 1), time_s=round(stop * dt, 1),
                end_reason=reason or ('exit' if reached else 'timeout'), reached=bool(reached), stop_i=stop)
