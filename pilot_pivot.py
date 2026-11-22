"""pilot_map.py + experimental "pivot-and-go" for alternating spike beds.

Same Scan Map pilot as pilot_map.py. One addition, switched by P_PIVOT (default 1):
when the next spike bed blocks the robot's side and the gap to the previous bed is
too short for the normal lane change, the robot
  1. PIVOT : stops and turns toward a point on the other side, past the bed it just passed
  2. GO    : walks straight (diagonally) to that point
  3. ALIGN : turns back parallel to the corridor and resumes normal driving.
No lateral (vy) command is used, so the pretrained policy is unchanged.

Run with P_PIVOT=0 to get exactly the pilot_map.py behaviour.
EXPERIMENTAL: not for submission until it passes the seed 3/7/11/13 benchmark.
"""

import math
import os

import torch

from g1_maze_lab.autopilot import MazeAutopilot


def _env(name, default):
    return float(os.environ.get(name, default))


SPEED = _env("P_SPEED", 0.8)
SPIKE_SPEED = _env("P_SPIKE_SPEED", 0.8)
MIN_SPEED = _env("P_MIN_SPEED", 0.45)
TURN_EARLY = _env("P_TURN_EARLY", 1.0)
MAX_TURN = _env("P_MAX_TURN", 1.4)
LANE = _env("P_LANE", 0.6)
LOOK_SPIKE = _env("P_LOOK_SPIKE", 2.0)
CARROT = _env("P_CARROT", 0.9)
CLUSTER = _env("P_CLUSTER", 0.8)
DANGER_SPEED = _env("P_DANGER_SPEED", 0.5)
DANGER_LANE = _env("P_DANGER_LANE", 0.6)
AVOID_AFTER = int(_env("P_AVOID_AFTER", 2))
AVOID_COST = int(_env("P_AVOID_COST", 8))

TRENCH = int(_env("P_TRENCH", 1))
TRENCH_SPEED = _env("P_TRENCH_SPEED", 0.9)
TRENCH_HOLD = _env("P_TRENCH_HOLD", 0.75)
TRENCH_DROP = _env("P_TRENCH_DROP", 0.105)
TRENCH_ALIGN = _env("P_TRENCH_ALIGN", 0.25)
# run-up for wide trenches (experimental)
TRENCH_LOOK = _env("P_TRENCH_LOOK", 1.6)            # look this far ahead for the trench [m] (was 1.0)
TRENCH_WIDE = _env("P_TRENCH_WIDE", 0.45)           # trench wider than this counts as "wide" [m]
TRENCH_SPEED_WIDE = _env("P_TRENCH_SPEED_WIDE", 1.1)  # speed for wide trenches [m/s]
TRENCH_EXIT = _env("P_TRENCH_EXIT", 0.4)            # keep the speed this far past the far edge [m]
TRENCH_MAX_W = _env("P_TRENCH_MAX_W", 0.3)          # max yaw rate while crossing: go straight [rad/s]
TRENCH_LOG = int(_env("P_TRENCH_LOG", 1))

SLALOM = int(_env("P_SLALOM", 0))         # default 0 = the configuration that was benchmarked
FRONT = _env("P_FRONT", 0.1)
BESIDE_BACK = _env("P_BESIDE_BACK", 0.2)
BESIDE_FRONT = _env("P_BESIDE_FRONT", 0.25)
CLEAR = _env("P_CLEAR", 0.4)
SLALOM_SPEED = _env("P_SLALOM_SPEED", 0.5)
SLALOM_HOLD = _env("P_SLALOM_HOLD", 1.0)
SWITCH_RATE = _env("P_SWITCH_RATE", 0.5)

GATE = int(_env("P_GATE", 1))
GATE_DIST = _env("P_GATE_DIST", 0.5)
GATE_MARGIN = _env("P_GATE_MARGIN", 0.3)
GATE_SPEED = _env("P_GATE_SPEED", 0.1)
GATE_CARROT = _env("P_GATE_CARROT", 0.45)
GATE_MAX_WIDTH = _env("P_GATE_MAX_WIDTH", 1.2)

# ---- pivot-and-go ----------------------------------------------------------
PIVOT = int(_env("P_PIVOT", 1))
PIVOT_DIST = _env("P_PIVOT_DIST", 1.0)        # trigger when the blocking bed starts closer than this [m]
PIVOT_MARGIN = _env("P_PIVOT_MARGIN", 0.25)   # keep this far past the old bed / before the new bed [m]
PIVOT_MIN_GAP = _env("P_PIVOT_MIN_GAP", 0.45)     # gap between the two beds must be at least this [m]
PIVOT_CENTER_TOL = _env("P_PIVOT_CENTER_TOL", 0.05)  # start stopping this far before the turn point [m]
PIVOT_APPROACH_SPEED = _env("P_PIVOT_APPROACH_SPEED", 0.2)  # creep speed toward the turn point [m/s]
PIVOT_TURN_AFTER = _env("P_PIVOT_TURN_AFTER", 0.2)  # turn point = this far past the end of bed 1 [m]
PIVOT_KEEP = _env("P_PIVOT_KEEP", 0.45)             # turn point at least this far before bed 2 [m]
PIVOT_BED_CLEAR = _env("P_PIVOT_BED_CLEAR", 0.4)    # end the crossing this far (sideways) from bed 2 [m]
PIVOT_WALL_CLEAR = _env("P_PIVOT_WALL_CLEAR", 0.5)  # ...and at least this far from a wall [m]
PIVOT_MAX_W = _env("P_PIVOT_MAX_W", 0.8)            # max yaw rate while turning in place [rad/s]
PIVOT_ALIGN_GAIN = _env("P_PIVOT_ALIGN_GAIN", 1.0)  # softer gain for the turn back
PIVOT_SETTLE = _env("P_PIVOT_SETTLE", 0.4)          # stand still this long before each turn [s]
PIVOT_SPEED = _env("P_PIVOT_SPEED", 0.35)     # forward speed while walking diagonally [m/s]
PIVOT_GAIN = _env("P_PIVOT_GAIN", 2.0)        # yaw gain while turning
PIVOT_AIM_TOL = _env("P_PIVOT_AIM_TOL", 0.15) # PIVOT -> GO when heading error below this [rad]
PIVOT_ALIGN_TOL = _env("P_PIVOT_ALIGN_TOL", 0.15)
PIVOT_REACH = _env("P_PIVOT_REACH", 0.15)     # GO -> ALIGN when this close to the waypoint [m]
PIVOT_TIMEOUT = _env("P_PIVOT_TIMEOUT", 12.0)  # give up a pivot after this long [s] (all phases)
PIVOT_COOLDOWN = _env("P_PIVOT_COOLDOWN", 1.5)
PIVOT_LOG = int(_env("P_PIVOT_LOG", 1))

PIVOT_STOP_V = _env("P_PIVOT_STOP_V", 0.05)  # "stopped" when |vx cmd| below this [m/s]
PIVOT_STOP_W = _env("P_PIVOT_STOP_W", 0.15)  # "not turning" when |wz cmd| below this [rad/s]
PIVOT_HOLD_GAIN = _env("P_PIVOT_HOLD_GAIN", 0.6)  # small heading hold while walking straight
# Keep the gait alive: a tiny forward command instead of a dead stop while braking/turning.
# Many RL walkers switch to a standing pose at (0,0,0) and lose balance when restarting to turn.
PIVOT_KEEPALIVE = _env("P_PIVOT_KEEPALIVE", 0.06)  # [m/s], 0 = old behaviour (full stop)

# creep to turn point -> stop -> turn in place -> walk straight -> stop -> turn back
IDLE, APPROACH, STOP, TURN, GO, STOP2, ALIGN = 0, 1, 2, 3, 4, 5, 6


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


class ScanWallMap:
    """Observe shared cell edges anywhere in the 3.2 m scan footprint."""

    def __init__(self, num_cells, cell_size, device):
        n, c = num_cells, cell_size
        edges = []
        for i in range(n):
            for j in range(n):
                if i + 1 < n:
                    edges.append((i, j, 0, i + 1, j, (i + 0.5) * c, j * c))
                if j + 1 < n:
                    edges.append((i, j, 1, i, j + 1, i * c, (j + 0.5) * c))
        self.i = torch.tensor([e[0] for e in edges], device=device)
        self.j = torch.tensor([e[1] for e in edges], device=device)
        self.d = torch.tensor([e[2] for e in edges], device=device)
        self.ni = torch.tensor([e[3] for e in edges], device=device)
        self.nj = torch.tensor([e[4] for e in edges], device=device)
        self.x = torch.tensor([e[5] for e in edges], device=device)
        self.y = torch.tensor([e[6] for e in edges], device=device)
        self.vertical = self.d == 0

    @torch.no_grad()
    def update(self, walls, scan_xy, scan_z):
        x = scan_xy[..., 0, None] - self.x
        y = scan_xy[..., 1, None] - self.y
        along = torch.where(self.vertical, x, y)
        across = torch.where(self.vertical, y, x)
        valid = torch.isfinite(scan_z) & (scan_z > -2) & (scan_z < 4)
        valid &= (scan_xy.abs().sum(-1) > 0.01)
        strip = (along.abs() < 0.16) & (across.abs() < 0.58) & valid[..., None]
        samples = strip.sum(dim=1)
        tall = (strip & (scan_z[..., None] > 0.55)).sum(dim=1)
        covered = samples >= 8
        is_wall = tall >= 3
        old = walls[:, self.i, self.j, self.d]
        new = torch.where(covered & is_wall, 1,
                          torch.where(covered & (old != 1), 0, old)).to(torch.int8)
        walls[:, self.i, self.j, self.d] = new
        walls[:, self.ni, self.nj, (self.d + 2) % 4] = new
        return (new != old).any(dim=1)


class LanePilot(MazeAutopilot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        N, n = self.num_envs, self.n
        self.lane = torch.zeros(N, 2, device=self.device)
        self.near_spikes = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.falls = torch.zeros(N, n, n, dtype=torch.long, device=self.device)
        self.cur_cell = torch.tensor(self.start, device=self.device).repeat(N, 1)
        self.trench_hold = torch.zeros(N, device=self.device)
        self.trench_speed = torch.full((N,), TRENCH_SPEED, device=self.device)
        self.slalom_hold = torch.zeros(N, device=self.device)
        self.gate = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.scan_map = ScanWallMap(n, self.c, self.device)
        self.map_ticks = 0
        self.map_changed = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.max_acc, self.max_ang_acc = 3.0, 6.0
        # pivot-and-go state
        self.pivot_state = torch.zeros(N, dtype=torch.long, device=self.device)
        self.pivot_goal = torch.zeros(N, 2, device=self.device)
        self.pivot_turn_pt = torch.zeros(N, 2, device=self.device)   # latched world point to turn at
        self.pivot_yaw = torch.zeros(N, device=self.device)      # corridor heading at trigger
        self.pivot_timer = torch.zeros(N, device=self.device)
        self.pivot_phase_t = torch.zeros(N, device=self.device)     # time spent in the current phase
        self.pivot_cool = torch.zeros(N, device=self.device)
        self.pivot_pending = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.pivot_count = 0

    def observe_map(self, scan_xy, scan_z):
        self.map_ticks += 1
        if self.map_ticks % 2 == 1:
            self.map_changed = self.scan_map.update(self.walls, scan_xy, scan_z)

    def step(self, pos_xy, yaw, walls_now, dt):
        cell = self.cell_of(pos_xy)
        reached = (torch.linalg.norm(pos_xy - self.centre_of(cell), dim=1) < self.turn_early) & (
            (cell == self.target).all(dim=1))
        origin_here = (cell == self.origin).all(dim=1)
        idx = torch.arange(self.num_envs, device=self.device)
        blocked_exit = self.walls[idx, cell[:, 0], cell[:, 1], self.heading] == 1
        replan = origin_here & ~self.need_decision & blocked_exit
        if self.map_changed.any():
            possible = self.walls[idx, cell[:, 0], cell[:, 1]] != 1
            preferred = self.choose_direction(cell, possible, self.heading, self.walls)
            near_junction = torch.linalg.norm(pos_xy - self.centre_of(self.origin), dim=1) < 0.80
            replan |= origin_here & near_junction & (preferred != self.heading) & ~self.need_decision
        self.map_changed[:] = False
        decide = self.need_decision | reached | replan
        if decide.any():
            ix = torch.nonzero(decide).flatten()
            possible = self.walls[ix, cell[ix, 0], cell[ix, 1]] != 1
            direction = self.choose_direction(cell[ix], possible, self.heading[ix], self.walls[ix])
            self.cells_walked[ix] += (~self.need_decision[ix] & ~replan[ix]).long()
            self.heading[ix] = direction
            self.target[ix] = (cell[ix] + self._dirs[direction]).clamp(0, self.n - 1)
            self.origin[ix] = cell[ix]
            self.need_decision[ix] = False
        return self.velocity_command(pos_xy, yaw, dt)

    def flood_fill(self, walls):
        n = self.n
        big = 10**6
        pen = torch.where(self.falls >= AVOID_AFTER, AVOID_COST, 0)
        if pen.shape[0] != walls.shape[0]:
            pen = torch.zeros(walls.shape[:3], dtype=torch.long, device=self.device)
        pen[:, self.goal[0], self.goal[1]] = 0
        pen[:, self.start[0], self.start[1]] = 0
        dist = torch.full(walls.shape[:3], big, dtype=torch.long, device=self.device)
        dist[:, self.goal[0], self.goal[1]] = 0
        blocked = walls == 1
        for _ in range(4 * n * n):
            best = torch.full_like(dist, big)
            best[:, :-1, :] = torch.minimum(best[:, :-1, :], torch.where(blocked[:, :-1, :, 0], big, dist[:, 1:, :] + 1))
            best[:, 1:, :] = torch.minimum(best[:, 1:, :], torch.where(blocked[:, 1:, :, 2], big, dist[:, :-1, :] + 1))
            best[:, :, :-1] = torch.minimum(best[:, :, :-1], torch.where(blocked[:, :, :-1, 1], big, dist[:, :, 1:] + 1))
            best[:, :, 1:] = torch.minimum(best[:, :, 1:], torch.where(blocked[:, :, 1:, 3], big, dist[:, :, :-1] + 1))
            new = torch.minimum(dist, (best + pen).clamp(max=big))
            if torch.equal(new, dist):
                break
            dist = new
        return dist

    def update_lane(self, pos_xy, scan_xy, scan_z, dt):
        self.slalom_hold = (self.slalom_hold - dt).clamp(min=0.0)
        self.pivot_cool = (self.pivot_cool - dt).clamp(min=0.0)
        d = self._dirs[self.heading].float()
        perp = torch.stack((-d[:, 1], d[:, 0]), dim=1)
        rel = scan_xy - self.centre_of(self.target).unsqueeze(1)
        ahead = ((scan_xy - pos_xy.unsqueeze(1)) * d.unsqueeze(1)).sum(-1)
        across = (rel * perp.unsqueeze(1)).sum(-1)
        cone = (scan_z > 0.08) & (scan_z < 0.45) & (across.abs() < 0.9)
        spiky = cone & (ahead > -0.5) & (ahead < LOOK_SPIKE)
        any_spikes = spiky.sum(dim=1) >= 5
        pick = (cone & (ahead > FRONT) & (ahead < LOOK_SPIKE)) if SLALOM else spiky
        big = torch.full_like(ahead, 1e3)
        first = torch.where(pick, ahead, big).min(dim=1).values
        bed = pick & (ahead < first.unsqueeze(1) + CLUSTER)
        count = bed.sum(dim=1)
        mean_across = (across * bed).sum(dim=1) / count.clamp(min=1)
        seen = (count >= 5) & (mean_across.abs() > 0.15)
        danger = self._danger()
        lane_w = torch.where(danger, DANGER_LANE, LANE)
        side = torch.where(seen, -torch.sign(mean_across), torch.zeros_like(mean_across))
        want = side * lane_w
        rate = torch.where(seen, 0.3, 0.05)
        if SLALOM:
            second = pick & (ahead >= first.unsqueeze(1) + CLUSTER)
            c2 = second.sum(dim=1)
            m2 = (across * second).sum(dim=1) / c2.clamp(min=1)
            slalom = seen & (c2 >= 5) & (m2.abs() > 0.15) & (torch.sign(m2) != torch.sign(mean_across))
            self.slalom_hold = torch.where(slalom, torch.full_like(self.slalom_hold, SLALOM_HOLD), self.slalom_hold)
            beside = cone & (ahead > -BESIDE_BACK) & (ahead < BESIDE_FRONT)
            left_min = torch.where(beside & (across > 0), across, big).min(dim=1).values
            right_max = torch.where(beside & (across < 0), across, -big).max(dim=1).values
            one_side = (left_min < 1e2) ^ (right_max > -1e2)
            want = torch.where(one_side, torch.minimum(want, left_min - CLEAR), want)
            want = torch.where(one_side, torch.maximum(want, right_max + CLEAR), want)
            cur = (self.lane * perp).sum(-1)
            switching = seen & (torch.sign(want) != torch.sign(cur)) & (cur.abs() > 0.1)
            rate = torch.where(switching, SWITCH_RATE, rate)
        robot_lat = ((pos_xy - self.centre_of(self.target)) * perp).sum(-1)
        if PIVOT:
            # detect alternating beds once and latch the turn point / crossing goal in world coords
            self._maybe_start_pivot(pos_xy, d, perp, ahead, across, cone, robot_lat, big, scan_z)
            # hold the lane while approaching, pivoting and during the cooldown, otherwise the
            # lane offset would steer the robot diagonally into a bed
            frozen = (self.pivot_state != IDLE) | (self.pivot_cool > 0)
            rate = torch.where(frozen, torch.zeros_like(rate), rate)
        self.lane += rate.unsqueeze(1) * (want.unsqueeze(1) * perp - self.lane)
        self.near_spikes = any_spikes

        # bed geometry for the gate
        bed_lo = torch.where(bed, across, big).min(dim=1).values
        bed_hi = torch.where(bed, across, -big).max(dim=1).values
        inside = (robot_lat > bed_lo - GATE_MARGIN) & (robot_lat < bed_hi + GATE_MARGIN)
        narrow = (bed_hi - bed_lo) < GATE_MAX_WIDTH
        blocked = seen & narrow & inside
        self.gate = (blocked & (first < GATE_DIST)) if GATE else torch.zeros_like(self.gate)

    def _maybe_start_pivot(self, pos_xy, d, perp, ahead, across, cone, robot_lat, big, scan_z):
        """When alternating beds are seen, latch a turn point and a crossing goal (world coords)."""
        # Own "next bed strictly ahead" detector (independent of P_SLALOM), so the bed we are
        # passing right now is not mixed into the bed that blocks us.
        nxt = cone & (ahead > FRONT) & (ahead < LOOK_SPIKE)
        first = torch.where(nxt, ahead, big).min(dim=1).values
        bed = nxt & (ahead < first.unsqueeze(1) + CLUSTER)
        count = bed.sum(dim=1)
        mean_across = (across * bed).sum(dim=1) / count.clamp(min=1)
        lo_b = torch.where(bed, across, big).min(dim=1).values
        hi_b = torch.where(bed, across, -big).max(dim=1).values
        seen = (count >= 5) & (mean_across.abs() > 0.15) & ((hi_b - lo_b) < GATE_MAX_WIDTH)
        inside = (robot_lat > lo_b - GATE_MARGIN) & (robot_lat < hi_b + GATE_MARGIN)
        cand = seen & inside & (first < PIVOT_DIST) & (self.pivot_state == IDLE) & (self.pivot_cool <= 0)
        for i in torch.nonzero(cand).flatten().tolist():
            new_side = -float(torch.sign(mean_across[i]))   # go to the side opposite the blocking bed
            f = float(first[i])
            # the bed we are passing: cones on new_side, from just behind us up to the blocking bed
            old_bed = cone[i] & (across[i] * new_side > 0.15) & (ahead[i] > -0.6) & (ahead[i] < f)
            if int(old_bed.sum()) < 3:
                continue        # single bed, not alternating: the normal lane offset handles it
            old_end = float(ahead[i][old_bed].max())
            gap = f - old_end
            if gap < PIVOT_MIN_GAP:
                if PIVOT_LOG >= 2:   # can repeat every tick, so only at log level 2
                    print(f"[PIVOT] env{i} skip: gap {gap:.2f} m < {PIVOT_MIN_GAP} "
                          f"(old_end={old_end:.2f} next_bed={f:.2f})", flush=True)
                continue
            gap_mid = 0.5 * (old_end + f)
            # turn just after bed 1 ends (bed 1 is on the far side, so turning near it is safe),
            # but never closer than PIVOT_KEEP to bed 2, which is straight ahead on our side
            turn_fwd = min(old_end + PIVOT_TURN_AFTER, f - PIVOT_KEEP)
            if turn_fwd < old_end + 0.05:        # both rules cannot hold: use the gap centre
                turn_fwd = gap_mid
            turn_fwd = max(turn_fwd, 0.0)        # already past it: stop where we are
            goal_fwd = max(gap_mid, turn_fwd)    # cross toward the gap centre line, never backwards
            # how far to cross (measured toward new_side): just clear bed 2, stay off the wall
            bed_edge = float(lo_b[i]) if new_side < 0 else float(hi_b[i])   # bed 2 edge nearest the centre
            need = -new_side * bed_edge + PIVOT_BED_CLEAR                     # min distance toward new_side
            tall = ((scan_z[i] > 0.55) & (across[i] * new_side > 0) &
                    (ahead[i] > turn_fwd - 0.5) & (ahead[i] < goal_fwd + 0.8))
            wall = float((across[i][tall] * new_side).min()) if int(tall.sum()) >= 3 else 9.9
            limit = min(wall - PIVOT_WALL_CLEAR, LANE)                       # max distance toward new_side
            s = need if need <= limit else 0.5 * (need + limit)
            want_lat = new_side * s
            if PIVOT_LOG and need > limit:
                print(f"[PIVOT] env{i} tight: need {need:.2f} m from centre, wall allows {limit:.2f} m "
                      f"-> using {s:.2f} m", flush=True)
            p = pos_xy[i]
            self.pivot_turn_pt[i] = p + d[i] * turn_fwd
            self.pivot_goal[i] = p + d[i] * goal_fwd + perp[i] * (want_lat - float(robot_lat[i]))
            self.pivot_yaw[i] = math.atan2(float(d[i, 1]), float(d[i, 0]))
            self.pivot_state[i] = APPROACH if turn_fwd > PIVOT_CENTER_TOL else STOP
            self.pivot_timer[i] = 0.0
            self.pivot_phase_t[i] = 0.0
            self.lane[i] = want_lat * perp[i]          # keep the normal lane on the new side afterwards
            self.pivot_count += 1
            if PIVOT_LOG:
                ang = math.degrees(math.atan2(want_lat - float(robot_lat[i]), goal_fwd - turn_fwd))
                print(f"[PIVOT] #{self.pivot_count} env{i} start: bed1_end={old_end:.2f} bed2={f:.2f} "
                      f"gap={gap:.2f} | turn at {turn_fwd:.2f} m ({f - turn_fwd:.2f} m before bed2), "
                      f"cross to fwd={goal_fwd:.2f} lat {float(robot_lat[i]):+.2f}->{want_lat:+.2f} "
                      f"(~{ang:+.0f} deg)", flush=True)

    def update_trench(self, pos_xy, yaw, scan_xy, scan_z, dt):
        self.trench_hold = (self.trench_hold - dt).clamp(min=0.0)
        if not TRENCH:
            return
        d = self._dirs[self.heading].float()
        heading_err = torch.atan2(d[:, 1], d[:, 0]) - yaw
        heading_err = torch.atan2(torch.sin(heading_err), torch.cos(heading_err)).abs()
        rel = scan_xy - pos_xy.unsqueeze(1)
        fwd = (rel * d.unsqueeze(1)).sum(-1)
        lat = rel[..., 0] * -d[:, 1:2] + rel[..., 1] * d[:, 0:1]
        finite = torch.isfinite(scan_z)
        for i in range(self.num_envs):
            if float(heading_err[i]) > TRENCH_ALIGN or bool(self.near_spikes[i]):
                continue
            base = finite[i] & (fwd[i] > -0.25) & (fwd[i] < 0.15) & (lat[i].abs() < 0.48)
            if int(base.sum()) < 5:
                continue
            ground = scan_z[i][base].median()
            ahead = finite[i] & (fwd[i] > 0.32) & (fwd[i] < TRENCH_LOOK)
            bands = 0
            for lo, hi in ((-0.46, -0.16), (-0.15, 0.15), (0.16, 0.46)):
                sample = ahead & (lat[i] >= lo) & (lat[i] < hi)
                if int(sample.sum()) < 4:
                    break
                if int((scan_z[i][sample] < ground - TRENCH_DROP).sum()) >= 2:
                    bands += 1
            if bands == 3:
                # measure the trench along the centre line: near edge, far edge, width
                centre = finite[i] & (fwd[i] > 0.2) & (fwd[i] < TRENCH_LOOK) & (lat[i].abs() < 0.15)
                low = centre & (scan_z[i] < ground - TRENCH_DROP)
                if int(low.sum()) >= 2:
                    near = float(fwd[i][low].min())
                    far = float(fwd[i][low].max())
                else:
                    near, far = 0.5, 0.8
                width = max(far - near, 0.0)
                speed = TRENCH_SPEED_WIDE if width > TRENCH_WIDE else TRENCH_SPEED
                # hold long enough to run up, cross and clear the far edge
                hold = max(TRENCH_HOLD, (far + TRENCH_EXIT) / max(speed, 0.1))
                if TRENCH_LOG and float(self.trench_hold[i]) <= 0.0:
                    print(f"[TRENCH] env{i} near={near:.2f} far={far:.2f} width={width:.2f} m "
                          f"-> speed {speed:.2f} m/s for {hold:.2f} s", flush=True)
                self.trench_speed[i] = speed
                self.trench_hold[i] = max(float(self.trench_hold[i]), hold)

    def _danger(self):
        idx = torch.arange(self.num_envs, device=self.device)
        here = self.falls[idx, self.cur_cell[:, 0], self.cur_cell[:, 1]]
        nxt = self.falls[idx, self.target[:, 0], self.target[:, 1]]
        return (here > 0) | (nxt > 0)

    def _pivot_command(self, pos_xy, yaw, dt, v_des, w_des):
        """Override (v, w) for envs that are pivoting. Returns updated tensors."""
        active = self.pivot_state != IDLE
        if not active.any():
            return v_des, w_des
        self.pivot_timer = torch.where(active, self.pivot_timer + dt, self.pivot_timer)
        to_goal = self.pivot_goal - pos_xy
        dist = torch.linalg.norm(to_goal, dim=1)
        err_goal = _wrap(torch.atan2(to_goal[:, 1], to_goal[:, 0]) - yaw)
        err_head = _wrap(self.pivot_yaw - yaw)
        corr = torch.stack((torch.cos(self.pivot_yaw), torch.sin(self.pivot_yaw)), dim=1)
        left_along = (to_goal * corr).sum(-1)       # forward distance still to go along the corridor
        to_turn = ((self.pivot_turn_pt - pos_xy) * corr).sum(-1)   # forward distance to the turn point

        self.pivot_phase_t = torch.where(active, self.pivot_phase_t + dt, self.pivot_phase_t)
        for i in torch.nonzero(active).flatten().tolist():
            s = int(self.pivot_state[i])
            settled = float(self.pivot_phase_t[i]) >= PIVOT_SETTLE
            if float(self.pivot_timer[i]) > PIVOT_TIMEOUT:
                if PIVOT_LOG:
                    print(f"[PIVOT] env{i} timeout in state {s}, back to normal", flush=True)
                self.pivot_state[i] = IDLE
                self.pivot_cool[i] = PIVOT_COOLDOWN
                continue
            vx_now = abs(float(self.cmd[i, 0]))
            wz_now = abs(float(self.cmd[i, 2]))
            if s == APPROACH:                   # 0. creep straight along the corridor to the turn point
                v = PIVOT_APPROACH_SPEED
                w = PIVOT_HOLD_GAIN * float(err_head[i])
                if float(to_turn[i]) <= PIVOT_CENTER_TOL:
                    self.pivot_state[i] = STOP
                    if PIVOT_LOG:
                        print(f"[PIVOT] env{i} at turn point (overshoot {-float(to_turn[i]):+.2f} m)", flush=True)
            elif s == STOP:                     # 1. brake to (almost) a stop, keep stepping, no turning yet
                v, w = PIVOT_KEEPALIVE, 0.0
                if vx_now < PIVOT_KEEPALIVE + PIVOT_STOP_V and wz_now < PIVOT_STOP_W and settled:
                    self.pivot_state[i] = TURN
            elif s == TURN:                     # 2. turn (almost) in place toward the waypoint, slowly
                v = PIVOT_KEEPALIVE
                w = max(-PIVOT_MAX_W, min(PIVOT_MAX_W, PIVOT_GAIN * float(err_goal[i])))
                if abs(float(err_goal[i])) < PIVOT_AIM_TOL and wz_now < 2 * PIVOT_STOP_W:
                    self.pivot_state[i] = GO
            elif s == GO:                       # 3. walk straight, only a small heading hold
                v = PIVOT_SPEED
                w = PIVOT_HOLD_GAIN * float(err_goal[i])
                if float(dist[i]) < PIVOT_REACH or float(left_along[i]) < 0.02:
                    self.pivot_state[i] = STOP2
            elif s == STOP2:                    # 4. slow down again, keep stepping
                v, w = PIVOT_KEEPALIVE, 0.0
                if vx_now < PIVOT_KEEPALIVE + PIVOT_STOP_V and wz_now < PIVOT_STOP_W and settled:
                    self.pivot_state[i] = ALIGN
            else:                               # 5. turn (almost) in place back to the corridor heading, softly
                v = PIVOT_KEEPALIVE
                w = max(-PIVOT_MAX_W, min(PIVOT_MAX_W, PIVOT_ALIGN_GAIN * float(err_head[i])))
                if abs(float(err_head[i])) < PIVOT_ALIGN_TOL:
                    self.pivot_state[i] = IDLE
                    self.pivot_cool[i] = PIVOT_COOLDOWN
                    if PIVOT_LOG:
                        print(f"[PIVOT] env{i} done in {float(self.pivot_timer[i]):.1f} s", flush=True)
            if int(self.pivot_state[i]) != s:   # phase changed: restart the phase timer
                self.pivot_phase_t[i] = 0.0
            v_des[i] = v
            w_des[i] = max(-self.max_turn, min(self.max_turn, w))
        return v_des, w_des

    def velocity_command(self, pos_xy, yaw, dt):
        dt = float(torch.as_tensor(dt).reshape(-1)[0])
        c = self.centre_of(self.target)
        d = self._dirs[self.heading].float()
        along = ((c - pos_xy) * d).sum(-1)
        look = torch.minimum(along, torch.where(self.gate, GATE_CARROT, CARROT))
        carrot = c - d * (along - look).unsqueeze(1) + self.lane
        to_target = carrot - pos_xy
        yaw_err = torch.atan2(to_target[:, 1], to_target[:, 0]) - yaw
        yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))
        top = torch.where(self.near_spikes, SPIKE_SPEED, self.speed)
        top = torch.where(self._danger(), torch.clamp(top, max=DANGER_SPEED), top)
        top = torch.where(self.slalom_hold > 0, torch.clamp(top, max=SLALOM_SPEED), top)
        lo = torch.clamp(torch.full_like(top, self.min_speed), max=top)
        v_des = lo + (top - lo) * torch.clamp(torch.cos(yaw_err), min=0.0) ** 2
        crossing = (self.trench_hold > 0) & (yaw_err.abs() < TRENCH_ALIGN)
        v_des = torch.where(crossing, torch.maximum(v_des, self.trench_speed), v_des)
        v_des = torch.where(self.gate & ~crossing, torch.clamp(v_des, max=GATE_SPEED), v_des)
        w_des = torch.clamp(1.5 * yaw_err, -self.max_turn, self.max_turn)
        # go straight while crossing: no big steering mid-jump
        w_des = torch.where(crossing, torch.clamp(w_des, -TRENCH_MAX_W, TRENCH_MAX_W), w_des)
        if PIVOT:
            v_des, w_des = self._pivot_command(pos_xy, yaw, dt, v_des.clone(), w_des.clone())
        self.cmd[:, 0] += torch.clamp(v_des - self.cmd[:, 0], -self.max_acc * dt, self.max_acc * dt)
        self.cmd[:, 2] += torch.clamp(w_des - self.cmd[:, 2], -self.max_ang_acc * dt, self.max_ang_acc * dt)
        return self.cmd.clone()

    def reset(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device).long().reshape(-1)
        ci, cj = self.cur_cell[env_ids, 0], self.cur_cell[env_ids, 1]
        at_start = (ci == self.start[0]) & (cj == self.start[1])
        k = env_ids[~at_start]
        self.falls[k, ci[~at_start], cj[~at_start]] += 1
        start = torch.tensor(self.start, dtype=torch.long, device=self.device)
        self.target[env_ids] = start
        self.origin[env_ids] = start
        self.cur_cell[env_ids] = start
        self.heading[env_ids] = 0
        self.need_decision[env_ids] = True
        self.cells_walked[env_ids] = 0
        self.cmd[env_ids] = 0.0
        self.lane[env_ids] = 0.0
        self.near_spikes[env_ids] = False
        self.trench_hold[env_ids] = 0.0
        self.trench_speed[env_ids] = TRENCH_SPEED
        self.slalom_hold[env_ids] = 0.0
        self.gate[env_ids] = False
        self.pivot_state[env_ids] = IDLE
        self.pivot_timer[env_ids] = 0.0
        self.pivot_cool[env_ids] = 0.0
        self.pivot_pending[env_ids] = False


class Pilot:
    def __init__(self, num_envs, num_cells, cell_size, device, start_cell=(0, 0), goal_cell=None):
        print(f"[PILOT PIVOT] pivot={PIVOT} dist={PIVOT_DIST} margin={PIVOT_MARGIN} speed={PIVOT_SPEED} "
              f"gain={PIVOT_GAIN} timeout={PIVOT_TIMEOUT} | speed={SPEED} danger={DANGER_SPEED} "
              f"trench={TRENCH} trench_speed={TRENCH_SPEED}/{TRENCH_SPEED_WIDE} wide>{TRENCH_WIDE} "
              f"look={TRENCH_LOOK} slalom={SLALOM} gate={GATE} gate_dist={GATE_DIST}", flush=True)
        self._map_viewer = None
        self._map_ui_attempts = 0
        self._map_ui_active = False
        self._map_frame = 0
        self._map_snapshot = None
        self.ap = LanePilot(
            num_envs, num_cells, cell_size, device, strategy="flood", speed=SPEED,
            turn_early=TURN_EARLY, min_speed=MIN_SPEED, max_turn=MAX_TURN, start=start_cell, goal=goal_cell,
        )

    def act(self, obs):
        cell = self.ap.cell_of(obs["pos_xy"])
        self.ap.cur_cell = cell.clone()
        self.ap.observe_map(obs["scan_xy"], obs["scan_z"])
        dt = float(torch.as_tensor(obs["dt"]).reshape(-1)[0])
        self.ap.update_lane(obs["pos_xy"], obs["scan_xy"], obs["scan_z"], dt)
        self.ap.update_trench(obs["pos_xy"], obs["yaw"], obs["scan_xy"], obs["scan_z"], dt)
        command = self.ap.step(obs["pos_xy"], obs["yaw"], None, obs["dt"])
        self._update_map_view(obs)
        return command

    def map_snapshot(self, obs):
        walls = self.ap.walls[0].detach().cpu()
        cell = self.ap.cell_of(obs["pos_xy"])[0].tolist()
        return {
            "walls": walls.tolist(), "n": self.ap.n,
            "cell": cell, "target": self.ap.target[0].tolist(),
            "start": self.ap.start, "goal": self.ap.goal,
            "heading": int(self.ap.heading[0].item()),
            "known": int((walls == 0).sum().item() + (walls == 1).sum().item()),
            "pivot": int(self.ap.pivot_state[0].item()),
        }

    def _update_map_view(self, obs):
        if os.environ.get("P_MAP_VIEW", "1") == "0":
            return
        self._map_frame += 1
        if self._map_frame % 10 == 0:
            self._map_snapshot = self.map_snapshot(obs)
        if self._map_ui_active or self._map_ui_attempts >= 5:
            return
        if self._map_frame % 30 != 1:
            return
        self._map_ui_attempts += 1
        try:
            from isaaclab.sim import SimulationContext
            ctx = SimulationContext.instance()
            for viz in getattr(ctx, "visualizers", []):
                viewer = getattr(viz, "_viewer", None)
                if viewer is not None and hasattr(viewer, "register_ui_callback"):
                    viewer.register_ui_callback(self._draw_map_ui, position="free")
                    self._map_viewer = viewer
                    self._map_ui_active = True
                    print("[MAP] Newton live map HUD registered", flush=True)
                    return
        except (ImportError, AttributeError, RuntimeError) as exc:
            if self._map_ui_attempts == 5:
                print(f"[MAP] HUD unavailable: {exc}", flush=True)

    def _draw_map_ui(self, imgui):
        s = self._map_snapshot
        if s is None:
            return
        n, walls = s["n"], s["walls"]
        size = 22.0 if n <= 10 else 16.0
        v = imgui.ImVec2
        color = lambda r, g, b: imgui.get_color_u32(imgui.ImVec4(r, g, b, 1.0))
        red, gray, blue = color(.96, .32, .27), color(.45, .48, .52), color(.20, .76, .99)
        bg, dead = color(.15, .17, .20), color(.38, .24, .18)
        imgui.set_next_window_pos(v(12.0, 12.0), imgui.Cond_.always)
        imgui.set_next_window_bg_alpha(.88)
        flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.always_auto_resize |
                 imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings |
                 imgui.WindowFlags_.no_focus_on_appearing | imgui.WindowFlags_.no_nav)
        imgui.begin("Observed maze map", None, flags)
        imgui.text("HEIGHT-SCAN MAP (observed only)")
        imgui.text("Red wall | Gray unknown | Gap open")
        p = imgui.get_cursor_screen_pos()
        dl = imgui.get_window_draw_list()
        for i in range(n):
            for j in range(n):
                x, y = p.x + i * size, p.y + (n - 1 - j) * size
                state = walls[i][j]
                dl.add_rect_filled(v(x + 1, y + 1), v(x + size - 1, y + size - 1),
                                   dead if state.count(1) >= 3 else bg)
                for dd in range(4):
                    if (dd == 2 and i != 0) or (dd == 3 and j != 0) or state[dd] == 0:
                        continue
                    a, b = ((v(x + size, y), v(x + size, y + size)) if dd == 0 else
                            (v(x, y), v(x + size, y)) if dd == 1 else
                            (v(x, y), v(x, y + size)) if dd == 2 else
                            (v(x, y + size), v(x + size, y + size)))
                    dl.add_line(a, b, red if state[dd] == 1 else gray, 3.0 if state[dd] == 1 else 1.0)
        ci, cj = s["cell"]
        dl.add_circle_filled(v(p.x + (ci + .5) * size, p.y + (n - cj - .5) * size), 5.5, blue)
        ti, tj = s["target"]
        dl.add_circle(v(p.x + (ti + .5) * size, p.y + (n - tj - .5) * size), 7.0, color(.98, .82, .29), 0, 2.0)
        imgui.dummy(v(n * size + 4, n * size + 4))
        names = {0: "-", 1: "APPROACH", 2: "STOP", 3: "TURN", 4: "GO", 5: "STOP", 6: "ALIGN"}
        imgui.text(f"Robot {tuple(s['cell'])}  Target {tuple(s['target'])}  Pivot {names.get(s['pivot'], '?')}")
        imgui.end()

    def reset(self, env_ids):
        self.ap.reset(env_ids)
