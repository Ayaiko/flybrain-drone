# fastsim — fast headless forest flight sim (calibrated against PX4)

Training-speed stand-in for the PX4 v1.17 SITL + Gazebo Harmonic forest runs (`px4/fly_forest.py`). Gazebo renders the
three eye cameras in software, which caps PX4 at about 1x real time. Here the eye is ray-cast directly against the
obstacle list, and PX4 is replaced by a flight model fitted to the logged PX4 runs. numpy only. Use it for search and
training (CEM etc.), and use PX4 only to confirm.

Built for throughput: N episodes per process step together (one kernel call renders every eye), episodes are grouped per
world, and groups are spread over all cores. When the PX4 containers are busy, 3 cores are left free. The ray cast is
one array-in/array-out function (`kernel.cast`), so a numba or Rust (PyO3) version can replace it without touching
anything else.

Numbers below are from 2026-11-22. The full report is `results/fastsim/calibration.md`.

## Pieces

| file | what |
|---|---|
| `kernel.py` | **the ray-cast kernel**: `cast(scene, cam, ex, ey, ez, h)` for N drones -> per-ray class / albedo / n.L arrays. `Scene` and `Cam` are NamedTuples of plain arrays. Each obstacle is only tested against the columns inside its angular footprint (exact). Nearest hit = one argsort. Branches thinner than 0.3 px are culled. |
| `truth.py` | `BatchTruth`: collision truth for N drones, with the formulas of `forest_geom.Obstacles.query` and candidates from a 1 m grid. Identical to `Obstacles.query` (checked on 9000 points with gap < 3 m), ~15 µs per drone |
| `world.py` | loads `px4/forest/<world>.json` (obstacles via `px4/forest_geom.Obstacles`), the procedural textures (`px4/forest/textures`) as grey mip pyramids |
| `eye.py` | `RayEye`: the real camera geometry (3 x 160 x 8 px, 110 deg, yaw -110/0/+110, eye 0.18 m above base_link, far clip 80 m depth). Ray-casts all 480 pixel columns at 4 of the 8 pixel rows against trunks (cylinders), branches (capsules), foliage/shrubs/crowns/boulder (ellipsoids), the rock (box), the ground and the background. Brightness = texture albedo x fitted lighting curve of n.L (sun 0.001 0.625 -0.78), plus constants for untextured grey, ground and background. Then the same column -> 120-ommatidium averaging matrix as the bridge. `render_batch` handles N poses in one call; `cast` / `render` handle one pose. All of them go through `kernel.cast`. `cast_ref` is the slow brute-force reference used in tests. |
| `eye_calib.json` | lighting curve fitted by `validate_eye.py` |
| `dynamics.py` | `X500` / `X500Batch` (N drones): PX4 offboard body velocity + yaw rate. Command delay (speed 8 frames, yaw 4 frames), first-order yaw rate with acceleration limit, world-frame velocity that follows the rotating body setpoint (lag, side-slip) with accel/brake limits, first-order altitude |
| `dyn_params.json` | parameters fitted by `fit_dynamics.py` |
| `pilot.py` | the pilot interface (see below). **Default `SharedPilot`** = the shared `px4/forest_pilot.Pilot`, the exact logic the PX4 bridge runs. `BridgePilot` is a frozen copy of the 2026-11-22 bridge logic, kept only to reproduce the first calibration |
| `train_cem.py` | **curriculum CEM** on the training pool (`px4/forest/pool.json`, levels 0–2 in 0.25 steps): promotes by +0.25 when the elite succeed >= 70% on >= 8 worlds of the current level, keeps ~20% of episodes on easier levels, and reports on the held-out `ff_<easy/medium/hard>_0..5` set every few generations |
| `max_workers.txt` | optional hard cap on worker processes (also env `FASTSIM_MAX_WORKERS`). Currently 3, at the forest session's request while PX4 batches run; delete it to lift the cap |
| `runner.py` | `run_group` (N episodes of one world stepping together), `run_episode` (= a group of 1, same numbers), `run_batch` (per-world groups on a process pool), `default_workers`, CLI |
| `validate_eye.py` | renders the eye at the poses of the logged PX4 runs and compares it with the logged `img`; fits the lighting |
| `fit_dynamics.py` | fits the flight model to the logged PX4 runs |
| `calibrate.py` | calibration gate: same configs x worlds as PX4, rank correlation of progress and collisions |
| `ref/fly_forest_20261122.py` | snapshot of `px4/fly_forest.py` (md5 58b63ac1…) that `BridgePilot` mirrors |

## Loop (`runner.run_group`)

Runs at 30 Hz sim time; for each drone, in the same order as the bridge:
1. The eye renders the pose of the **previous** frame (Gazebo's measured 1-frame camera latency) and adds N(0, 0.02) noise.
2. Exit, stall and max-collision checks run (the live early stop from the config).
3. `pilot.step` returns a command, which goes into the `X500` model with its fitted delays.
4. Truth = `BatchTruth.query` (the same formulas as `forest_geom.Obstacles.query`) on the current pose, with 5 cm
   contact hysteresis. A new contact calls `pilot.on_collision`.

Per step, one `kernel.cast` covers every unfinished drone of the group, as do one truth query and one dynamics step.
Only the pilots run in a Python loop. A group of N gives exactly the same episodes as N single runs (checked).

The start pose is (0.03, 0, alt − 0.09, +y) at rest. Rep 0 is noise-free. Reps > 0 jitter the start (5 cm, 1 deg) and
use a different sensor-noise seed.

## Pilot interface

```python
pilot = make_pilot(world, P, rng, kind='bridge' | 'shared')   # or env FASTSIM_PILOT=module:factory(world, P, rng)
v, yaw_rate, vz, diag = pilot.step(now, (x, y, z, h), lum)    # diag: at least 'state', 'looming'
pilot.on_collision(now, (x, y, z, h))
pilot.stats()                                                 # merged into the episode summary
```

`kind='shared'` (the default) runs `px4/forest_pilot.Pilot` unchanged. It is fed `v_meas` (pose difference x 30), `bump` (on the
next frame, as in the bridge) and the ToF range (`forest_geom.RangeSensor`) when `P['tof']` is set. With
`recovery=detour`, `brake`, `flow_speed` and `tof` it is the new pilot. With default parameters it lands within noise of
`bridge` (checked on 27 episodes).

## Usage (from the project root)

```bash
.venv/bin/python -m fastsim.runner --configs lplc2-bold-gate-sv0-v0.8 --presets easy,medium,hard --seeds 0,1,2 --reps 3   # workers auto, group auto (<= 32)
.venv/bin/python -m fastsim.runner --configs fix-all --pilot shared --worlds ff_medium_0 --param flow_ref=0.1
.venv/bin/python -m fastsim.calibrate --reps 10                  # calibration gate -> results/fastsim/calibration*.{md,json,png}
.venv/bin/python -m fastsim.calibrate --configs fix-detour,fix-all --tag _batch3   # any configs that have PX4 runs
.venv/bin/python -m fastsim.train_cem --base fix-all --pop 24 --gens 60 --name run1  # curriculum CEM -> results/fastsim/cem/run1/ (--resume)
.venv/bin/python -m fastsim.validate_eye                         # re-fit / check the eye against logged PX4 frames
.venv/bin/python -m fastsim.fit_dynamics                         # re-fit the flight model (≈3 min)
```

From Python: `from fastsim.runner import run_episode; s = run_episode('none-v0.8', 'ff_hard_1', rep=0)['summary']`.
For training, evaluate a whole population at once: `run_batch([dict(config=..., world=..., rep=..., overrides={...}), ...])`
returns (summaries, wall, profile).
Add `record=True` for the per-frame trace and eye images. Batches append one summary per line to
`results/fastsim/runs.jsonl` (the same keys as `results/forest/forest_runs.jsonl`, plus `sim`, `pilot`, `rep`, `frames`,
`wall_s`, `rtf`).

## Curriculum training (`train_cem.py`)

- Training worlds come only from the pool `ff_L<level*100>_<seed>` (seeds 100–109). The held-out evaluation worlds
  `ff_<easy|medium|hard>_0..5` are never trained on.
- Each generation, every candidate flies the same 8 current-level worlds plus 2 easier-level worlds (20%), with the same
  noise seed (common random numbers). Fitness = progress/80 + success − 0.05 × collisions.
- Promotion: elite success on the current level >= 0.7, then +0.25. Held-out report on all levels at generation 0
  (the base config as is) and every `--eval-every` generations.
- Search space: 20 continuous parameters of the shared pilot (`SPACE` in the file; the base config's values are the
  start). `best.json` holds the current mean as overrides, so it can be flown in PX4 with `forest_eval --param ...`.
- Cost: pop 24 × 10 episodes = 240 episodes per generation. That is ~10–15 min with the current 3-worker cap under
  PX4 load (110x real time in total), and an estimated ~3 min on the idle machine with 10 workers. Not measured idle yet.

## Calibration in short

- **Eye vs Gazebo** (9 held-out seed-2 runs, 4678 frames): per-frame correlation mean 0.936 (median 0.953, p10 0.89).
  The noise ceiling is 0.994, set by the logged img's own noise. Frame-to-frame change correlation is 0.77; neighbour-contrast correlation is 0.90.
- **Flight model** (held out): position RMSE 4.5 cm at 1 s and 8.5 cm at 2 s; heading RMSE 1.1 deg at 1 s and 1.6 deg at 2 s
  (an ideal no-lag unicycle gives 29 cm / 14 deg). PX4 follows yaw-rate commands up to the 2 rad/s used in recovery.
  The ~60 deg/s limit seen in saccades is the bridge's own clip, not the autopilot.
- **Closed loop** (3 configs x 9 worlds, 10 reps vs the single PX4 runs): Spearman progress 0.89 (gate 0.8 passed),
  collisions 0.74. Both are at the ceiling a single run allows (0.83 / 0.70, measured from rep-to-rep scatter). See
  `results/fastsim/calibration.md`.
- **Speed** (batched): one process runs ~42x real time with groups of 32 (787 µs per drone-frame: render 80%, pilot 13%).
  Full batch: 37.8 episodes/min, 144x real time on 7 workers while PX4 containers used ~7 of the 10 cores. PX4 itself
  manages ~1.8x with 3 containers.

## Caveats

- Body pitch/roll are not modelled for the eye: the cameras stay level. Gazebo's anti-aliasing and filtering are only
  approximated, by nearest-mip sampling.
- The forward-speed command was not logged in the PX4 runs, so it was rebuilt from the logged state. That rebuild partly
  explains the long fitted speed delay.
- Obstacles have no physics in either sim: the drone flies through what it hits, and contacts are counted from the truth.
- `px4/forest_params.resolve` and `Stall` are imported live, so new configs work. `calibrate.py` keeps its own copy of
  the 2026-11-22 re-scoring rule, so the gate stays fixed.
