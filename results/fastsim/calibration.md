# fastsim calibration against PX4 + Gazebo (2026-11-22)

`fastsim/` is a numpy ray-cast eye plus a fitted x500 model, flying the same forest mission as `px4/fly_forest.py`.
This report checks it in three steps: the eye against real Gazebo frames, the flight model against PX4 trajectories,
and closed-loop outcomes against the PX4 forest batch. Reproduce with `fastsim/validate_eye.py`,
`fastsim/fit_dynamics.py` and `fastsim/calibrate.py`. Raw numbers are in `eye_validation.json`, `dynamics_fit.json`,
`calibration.json`, `calibration_tables.md` and `calibration_runs.jsonl` in this folder.

## Verdict

| check | result | target |
|---|---|---|
| eye vs Gazebo frames, held-out per-frame correlation | **0.936** (median 0.953; noise ceiling 0.994) | - |
| flight model, held-out 1 s / 2 s prediction | **4.5 / 8.5 cm, 1.1 / 1.6 deg** (no-lag unicycle: 29 cm / 14 deg) | - |
| rank correlation, all 27 (config, world), progress | **0.89** | >= 0.8 pass |
| rank correlation, all 27, collisions | **0.74** | >= 0.8 miss, but the ceiling is 0.70 (below) |
| rank correlation, 18 brain pairs, progress / collisions | 0.65 / 0.56 | ceiling 0.45 / 0.52 |
| speed (batched runner) | **37.8 episodes/min, 144x real time** on 7 workers *while the PX4 containers used ~7 of the 10 cores*; one process alone: ~42x real time (groups of 32) | PX4: ~1x per container, ~1.8x with 3 |

The progress gate passes. The collisions gate misses 0.8, but no sim could pass it against this reference. PX4 has
**one** run per (config, world), and a brain run is chaotic: in the fast sim the same (config, world) spreads by
±10–24 m of progress and ±3–8 collisions across noise seeds. As a ceiling, take one fast-sim rep against the mean of
the other nine. That reaches only 0.83 (progress) and 0.70 (collisions) over the 27 pairs, and 0.45 / 0.52 over the 18
brain pairs. The fast sim's agreement with PX4 equals those ceilings, so it agrees with a single PX4 run as well as
another fast-sim run would. PX4 runs also sit inside the fast-sim spread the way a calibrated sim predicts: their mean
percentile among the 10 reps is 0.48 (progress) and 0.54 (collisions), and 11% fall outside the rep range, where 18%
is expected for an exchangeable draw. A tighter gate needs more PX4 runs per pair, not a better fast sim.

## 1. Eye

Method (`validate_eye.py`): the 27 logged PX4 forest runs store the pose and the 120-px eye vector the brain saw (Gazebo
camera average + N(0, 0.02) noise). The ray-cast eye is rendered at every 15th logged pose. A linear lighting model is
fitted by least squares on the seed-0/1 worlds and tested on the seed-2 worlds: background, ground, albedo x a
piecewise-linear curve of n.L, and a curve for untextured grey.

- **Camera latency.** Image i+1 matches pose i (correlation 0.92 at lag +1, 0.84 at lag 0, 0.83 at lag +2). Gazebo's
  image is one frame older than the pose the bridge reads. The fast sim reproduces this: the eye renders the previous frame's pose.
- **Held out** (9 runs, 4678 frames): per-frame correlation mean 0.936, median 0.953, 10th percentile 0.887; pixel
  correlation 0.946, RMSE 0.079. Neighbour-difference (spatial contrast) correlation is 0.90. Frame-to-frame change
  (what the EMDs see) correlation is 0.77. By preset: easy 0.946, medium 0.912, hard 0.951.
- The SDF defaults alone, before fitting, already give 0.931. The fit mainly removes a −0.02 brightness bias. Fitted
  values: background 0.80, ground 0.78; the textured curve runs 0.45 (unlit) to 0.72 (n.L = 0.5).
- Sampling 4 of the 8 pixel rows costs almost nothing: 0.897 vs 0.900 on ff_medium_2. 2 rows gives 0.886.
- Remaining differences: Gazebo's far background is not flat (0.7–0.87 near the horizon; the fast sim uses a constant
  0.80), texture filtering and anti-aliasing are approximated by nearest-mip sampling, and body pitch/roll are ignored.
- Picture: `eye_compare.png` (Gazebo vs fast-sim eye over time for 3 held-out runs).

## 2. Flight model

Method (`fit_dynamics.py`): the model is started at the logged pose, velocity and yaw rate of every 10th frame and fed
the logged commands. It is compared with the logged position and heading 0.5, 1 and 2 s later; seed-0/1 worlds train,
seed-2 worlds test. 12657 / 6931 segments. Yaw is fitted first, then speed. The forward-speed command was not logged, so
it was rebuilt from the logged pilot state with the bridge's rules.

| fitted (x500, PX4 v1.17 offboard) | value |
|---|---|
| yaw: delay, lag, acceleration limit | 4 frames (133 ms), tau 0.20 s, 13 rad/s² (rate limit not reached; ≥ 2.4 rad/s) |
| speed: delay, lag, accel / brake limit | 8 frames (267 ms), tau 0.43 s, 2.0 / 3.4 m/s² |

| held out | pos RMSE 0.5 / 1 / 2 s | heading RMSE 0.5 / 1 / 2 s |
|---|---|---|
| all | 2.1 / 4.5 / 8.5 cm | 0.8 / 1.1 / 1.6 deg |
| brain runs | 2.1 / 4.5 / 8.0 cm | 0.9 / 1.2 / 1.7 deg |
| route follower | 1.8 / 4.6 / 11.4 cm | 0.04 / 0.05 / 0.08 deg |
| ideal unicycle (no lag) | 18 / 29 / 34 cm | 9 / 14 / 17 deg |

- **The ~60 deg/s yaw limit is the bridge's**, not PX4's. The closed-loop saccade clips its command at 60 deg/s.
  Recovery turns command 2.0 rad/s, and PX4 reaches about 1.8–1.9 rad/s.
- Picture: `dynamics_fit.png` (8 s open-loop replay of speed, yaw rate and heading; the model tracks PX4 throughout).
- **PX4 artefact in the route-follower runs.** In 8 of 9 `none-v0.8` runs PX4 flies the body-x velocity 2–6 deg to the
  right of the true heading, so the drone settles 0.3–0.5 m east of the centre line. The 18 brain runs show 0 ± 0.5 deg.
  They hold the yaw-rate command at exactly 0 while fixating; the route follower sends small non-zero rates all the time.
  It is likely a yaw-setpoint/estimate effect inside PX4. `dynamics.py` has `vel_off` for it (default 0). Rerunning the
  route follower with each run's measured offset brings collisions to 0/1/2/5/4/4/12/11/19, against PX4's 0/1/3/6/6/4/10/15/18
  (without the offset: 0/1/2/3/5/3/9/10/12). This is the main reason the route follower hits less in medium/hard below.

## 3. Closed loop vs PX4 (calibration gate)

Setup (`calibrate.py`): configs `none-v0.8`, `lplc2-bold-gate-sv0-v0.8`, `lplc2-bold-gate-sv0-gg0-v0.8` on
ff_{easy,medium,hard}_{0,1,2}, 10 reps each (different sensor-noise seeds and a 5 cm / 1 deg start jitter), with
`BridgePilot` (the mirror of the bridge logic the PX4 runs used). Both sims use the same live early stop: stall 90 s /
3 m, 20 collisions, 360 s. PX4 runs from batch 1, recorded without early stop, were re-scored under it, as
`px4/forest_compare.py` does.

Per preset (means over 3 worlds; fast sim also over 10 reps):

| config / preset | PX4 progress m | fastsim progress m | PX4 coll | fastsim coll |
|---|---|---|---|---|
| none easy | 80 | 80 | 1.0 | 1.0 |
| none medium | 80 | 80 | 5.3 | 3.7 |
| none hard | 80 | 80 | 14.0 | 10.2 |
| lplc2-bold easy | 54.9 | 51.0 | 8.0 | 5.7 |
| lplc2-bold medium | 49.3 | 41.8 | 13.3 | 11.6 |
| lplc2-bold hard | 13.3 | 24.7 | 7.3 | 12.7 |
| lplc2-bold-gg0 easy | 48.6 | 51.8 | 5.3 | 6.3 |
| lplc2-bold-gg0 medium | 40.9 | 44.2 | 14.0 | 10.8 |
| lplc2-bold-gg0 hard | 28.2 | 25.8 | 18.0 | 15.6 |

Per-world table: `calibration_tables.md`. Scatter: `calibration_scatter.png`.

- Behaviour matches in detail. Time share per pilot state is nearly identical: for lplc2-bold on ff_hard_2, warm-up,
  settling, fixate and turn are 13/7/33/47 % in both sims, and ff_easy_0 is 6/6/37/51 % vs 6/5/35/53 %. The median
  closed-loop turn lasts 1.5 s in both.
- Largest systematic difference: lplc2-bold on **hard**. PX4 stalled in all 3 worlds at 11–17 m. The fast sim reaches
  25 m on average (sd 11–18 m per world). The gg0 variant on hard goes the other way (PX4 28 vs fast sim 26 m). With one
  PX4 run per world, this is not yet distinguishable from chance. Worth 2–3 PX4 repeats of lplc2-bold on ff_hard_* to settle it.
- Kept: route follower vs brain, and easy > medium > hard for both brain configs (PX4 lplc2-bold hard is the exception
  above). **Not reproduced:** the goal_gap effect (gg0 vs the default goal_gap 2). PX4 favours gg2 on easy (+6 m) and
  medium (+8 m), and gg0 on hard (+15 m). The fast sim favours gg0 slightly on all three (+0.8 / +2.4 / +1.1 m). All of
  these gaps are within one world's run-to-run spread. Small config effects like this need many episodes in either sim
  before they are trusted.

## 4. Speed and where the time goes

Final numbers are from the batched runner (second pass, done at the request of the forest-test-field session). The
first-pass numbers are kept below for comparison. The re-run gate in sections 3 and Verdict used the batched runner.

Design for throughput:
- **One ray-cast kernel** (`kernel.cast(scene, cam, ex, ey, ez, h)`): arrays in, arrays out, for N drones. A numba or
  Rust (PyO3) version can replace it without touching anything else.
- **N episodes per process** (`runner.run_group`). One kernel call renders every unfinished drone of a world, and one
  grid truth query and one dynamics step serve them all. Only the pilots loop in Python. A group gives exactly the same
  episodes as single runs (checked frame by frame against the first-pass runner).
- **Culling.** Each primitive is only tested against the columns inside its angular footprint. This is exact, and
  better than a grid for 80 m rays through an open forest: a grid would need ray marching, which is slow in numpy.
  Branches thinner than 0.3 px are dropped (frame correlation 0.999 vs keeping them; 33% faster in the hard branch
  zone). The **1 m uniform grid** is used for collision truth (`truth.BatchTruth`): identical to `Obstacles.query` on
  9000 test points, 11–15 µs per drone instead of ~80 µs.
- **Cores:** `default_workers()` uses all cores, minus 3 while any flybrain-px4 container uses > 20% CPU (`docker stats`).
  Groups (auto size ≤ 32, about 3 per worker) go to a process pool, longest first.

One process, hard world, lplc2-bold, µs per drone per frame:

| group size | total | render | pilot | truth | dynamics | other | real time per process |
|---|---|---|---|---|---|---|---|
| 1 (first pass) | 2165 | 1495 | 160 | 283 | 186 | 39 | 15x |
| 8 | 1058 | 819 | 129 | 54 | 29 | 27 | 32x |
| 16 | 839 | 663 | 110 | 31 | 14 | 22 | 40x |
| 32 | 787 | 631 | 104 | 21 | 8 | 22 | 42x |
| 64 | 751 | 610 | 99 | 16 | 5 | 21 | 44x |

Inside the render, at N = 32 on a hard world: trunks ~0.5 ms per drone, branches ~0.5 ms (in the branch zone, before the
thin-branch cull), ellipsoids ~0.3, rock box ~0.03. What is left is numpy work on arrays of ~10^5 elements (shading
every column at 4 rows, pair building, one sort); there are no Python loops left in the kernel.

Full gate batch (270 episodes, 61,599 s of flight): **428.6 s on 7 workers = 37.8 episodes/min, 144x real time**,
measured while the 4 flybrain-px4 containers were using ~700% CPU. Under that contention render rose to 1270 µs per
drone-frame (80%), with pilot 159 (10%), truth 73, dynamics 40, bookkeeping 40. First pass, single-drone runner with
8 workers and lighter load: 24.8 episodes/min, 93x. On an idle machine with 10 workers, expect roughly 10 x 40 =
**~400x real time, ~100 episodes/min** (estimate, not measured).

A CEM generation of 32 candidates x 9 worlds (288 episodes) now takes about 8 min under the current PX4 load, and an
estimated ~3 min on an idle machine.

What would go faster:
- A compiled kernel (numba or Rust) for the render: 80% of the time, embarrassingly parallel per ray. Expect 5–20x on
  the render. Then the pilot (FlyVision in Python, ~100–160 µs per drone-frame) becomes the limit. A batched FlyVision
  would need changes in `flydrone/controllers.py` and `px4/forest_pilot.py`, which are owned by the other session.
  numba/Rust are **not installed**; this waits for the user's approval.
- Fewer eye rows (2 instead of 4) save ~30% of the render, at a cost of 0.011 in eye correlation.

## 5. Caveats and next steps

- Body pitch/roll are not modelled in the eye, and the background is flat. Contacts have no physics in either sim.
- The forward-speed command was rebuilt, not logged, for the dynamics fit. Logging it, which the new bridge does
  (`rec['v']` is v_meas, not the command), would sharpen the speed-delay estimate.
- `vel_off` (PX4 velocity/heading offset) only appears with continuous small yaw-rate commands. If a new pilot steers
  continuously, check its PX4 runs for it.
- To make the gate decisive, re-run 2–3 PX4 repeats of the brain configs on a few worlds (especially ff_hard_*), then
  compare distributions rather than single runs.
- `SharedPilot` (`--pilot shared`, `px4/forest_pilot.Pilot`) with default parameters lands within noise of
  `BridgePilot` (lplc2-bold, 3 reps x 9 worlds: easy 57.5 / 6.9, medium 40.7 / 12.2, hard 24.8 / 15.4 progress /
  collisions vs bridge 51.0 / 5.7, 41.8 / 11.6, 24.7 / 12.7). So the new recovery/brake/flow/ToF configs can be trained here directly.

## 6. Status after the curriculum / shared-pilot update (2026-11-22 01:25)

- The fast sim now flies `px4/forest_pilot.Pilot` by default, the same module the PX4 bridge runs. The calibration above
  used `BridgePilot` (the frozen pre-module logic of the PX4 runs it is compared with). With default parameters the two
  land within noise (section 5).
- Pending: calibration against PX4 batch 3 (`fix-detour`, `fix-all`) with
  `python -m fastsim.calibrate --configs fix-detour,fix-all --tag _batch3`, once that batch has finished. Several of
  its runs were affected by CPU overload (e.g. fix-all ff_medium_0 never took off; runs with < 1 m progress are excluded).
- Pending: re-measure throughput on an idle machine. Every speed number here was taken with PX4 containers busy.
  The fast sim is capped at 3 workers (`fastsim/max_workers.txt`) until the PX4 batch is done.

## 7. Calibration v2: shared pilot vs the full PX4 target set (2026-11-22 afternoon)

`python -m fastsim.calibrate --configs none-v0.8,lplc2-bold-gate-sv0-v0.8,lplc2-bold-gate-sv0-gg0-v0.8,fix-detour,fix-all,fix-all2@T600 --reps 6 --tag _v2`.
All configs were flown with the shared `px4/forest_pilot.Pilot`, on 69 PX4 pairs (easy 0–2, medium 0–5, hard 0–2;
gg0 on 0–2 only; fix-all2 with its 600 s limit), with 6 fast-sim reps each. Tables: `calibration_tables_v2.md`;
scatter: `calibration_scatter_v2.png`; data: `calibration_v2.json`, `calibration_runs_v2.jsonl`.

| Spearman (69 pairs) | progress | collisions |
|---|---|---|
| fast sim vs PX4 | **0.76** | **0.81** |
| ceiling (one rep vs mean of the others) | 0.69 | 0.68 |
| brain configs only (57 pairs) | 0.60 | 0.77 |
| ceiling, brain configs | 0.47 | 0.64 |

- Both are above the single-run ceiling. Collisions pass 0.8; progress misses it nominally, but no single-run comparison
  could do better than ~0.69 here.
- PX4 runs as a percentile of the fast-sim reps (57 brain pairs): progress 0.49 (centred), collisions **0.62** (PX4
  collides somewhat more). 21% / 19% fall outside the 6-rep range; an exchangeable draw would give 29%.
- Per preset (PX4 vs fast sim, progress m / collisions):

| run id | easy | medium | hard |
|---|---|---|---|
| fix-detour | 53.6 / 8.3 vs 62.2 / 5.9 | 44.5 / 12.7 vs 38.0 / 8.4 | 17.4 / 15.7 vs 25.4 / 10.3 |
| fix-all | 45.1 / 2.3 vs 37.1 / 0.4 | 36.7 / 1.5 vs 31.4 / 1.0 | 23.2 / 0.0 vs 19.1 / 1.5 |
| fix-all2@T600 | 74.3 / 3.0 vs 54.2 / 1.9 | 45.2 / 6.8 vs 41.9 / 4.4 | 22.2 / 6.3 vs 25.8 / 8.0 |

The relative order of the configs is kept: fix-all collides least by far in both sims, fix-detour most, and
fix-all2 gets furthest. The largest gaps are fix-all2 on easy (PX4 74 m vs 54 m; PX4 reached the exit in 2 of 3) and
fix-detour collisions (PX4 ~1.5x).

What the diagnostics show (fix-detour / fix-all / fix-all2 on 4 worlds, 4 reps each):
- **Forward speed while fixating matches** (PX4 0.45–0.50 m/s, fast sim 0.44–0.49). So the optic-flow magnitude
  that the flow-speed governor reads is right in the ray-cast eye.
- **Contacts by phase**: PX4 has more contacts in every phase for fix-detour. Per 12 worlds: turns 69 vs 44, recovery
  32 vs 17, fixation 46 vs 38. For fix-all2 they are close (recovery 55 vs 47).
- **Drift during in-place turns**: PX4 median 0.15 m/s (sideways 0.10) vs fast sim 0.10 (0.065). An optional
  Ornstein-Uhlenbeck velocity disturbance (`--dyn drift_sd=0.08 --dyn drift_tau=0.5`) matches PX4's turn speed and raises
  fix-detour collisions from 8.0 to 10.2 per world (PX4 13.0). But that much drift alone would give ~6 cm of 1 s
  position error, more than the whole 4.5 cm residual of the PX4 trajectory fit. So part of PX4's "turn speed" is
  probably pose-sampling jitter in the log (next point), not motion. It stays **off by default**; it is useful as
  randomisation when training (`train_cem --dyn drift_sd=0.05 --dyn drift_tau=0.5`).
- **PX4 pose sampling**: frame-to-frame displacements in the PX4 logs jump between 0.6x and 1.2x of the true step (a
  50 Hz pose stream read at 30 Hz). The bridge's `v_meas` and yaw rate are therefore noisy. Feeding the fast-sim pilot
  the same kind of stale pose (`--pose-tick 0.02`) made agreement worse: fixation speed fell to 0.40 m/s (PX4 0.50)
  and looming detections rose from 13% to 21% of fixation. So it is also **off by default**. Its effect in PX4 is
  evidently different from this simple model; the timing of camera frames vs pose is not logged.
- Speed of this batch: 414 episodes (104,128 s of flight) in 966 s on **3 workers** (the cap) while 3 PX4 sims ran:
  25.7 episodes/min, 108x real time, 36x per worker (render 79%, pilot 14%).

## 8. Anti-aliased eye + calibration v3 (8 PX4 run ids, 2026-11-22 evening)

**What was wrong.** On the new creep configs (`fix-detour-creep`, `fix-all3`) the fast sim got only 15–20 m on easy
worlds (PX4 49–51 m). Trace: in the open entry clearing, both sims wander with alternating ±60° gap-seeking turns. PX4
leaves the clearing after a median 25–50 s, but the fast sim needed 54–106 s for every fix config, and with creep it
often hit the 90 s stall rule. Replaying identical pose sequences through the EMD front end showed the cause. The
fast-sim nearness map (what goal_gap uses, normalised by its median) was much peakier than Gazebo's (peak/median 9.5 vs
7.9 in the clearing, 4,000+ vs 350–900 in the forest). Gazebo's frames carry faint, continuous lines from distant thin
trunks, and the fast sim missed them with one ray per pixel column. **Gazebo renders anti-aliased.**

**Fix.** The eye now casts **4 sub-rays across each pixel column x 2 rows** (`RayEye(ss=4, rows=2)`, the new default).
The background and ground are now Fourier series in world azimuth (the ground plane is specular), refitted on all 90
logged PX4 runs:

| eye vs Gazebo, held out (seed 2–5 worlds, 22,225 frames) | before (1 ray x 4 rows) | 4 sub-rays x 2 rows |
|---|---|---|
| frame correlation mean / median / p10 | 0.940 / 0.953 / 0.896 | **0.954 / 0.969 / 0.909** |
| frame-to-frame change correlation (what the EMDs see) | 0.788 | **0.862** |
| neighbour contrast correlation | 0.907 | **0.933** |
| nearness peak/median in the clearing (Gazebo 7.9) | 9.5 | 7.7–8.1 |
| time to leave the clearing, fix-all3 / fix-detour (PX4 50 / 25 s) | 106 / 54 s | **41 / 26 s** |

`--rows 1 --ss 4` behaves the same in the clearing test (frame correlation 0.946) and costs about the same as the old
eye.

**Calibration v3** (`--configs none-v0.8,lplc2-bold-gate-sv0-v0.8,lplc2-bold-gate-sv0-gg0-v0.8,fix-detour,fix-all,fix-all2@T600,fix-detour-creep,fix-all3 --reps 6 --tag _v3`;
87 PX4 pairs, shared Pilot):

| Spearman | progress | collisions |
|---|---|---|
| all 87 pairs | **0.80** (passes 0.8) | **0.85** (passes) |
| ceiling (one rep vs mean of the others) | 0.87 | 0.89 |
| brain configs (75 pairs) | 0.69 | 0.84 |

- Collisions agree well, including the creep effect (fix-detour medium 12.7 PX4 vs 12.1; fix-detour-creep hard 6.7 vs
  6.8). Collisions are centred on PX4 (percentile 0.55).
- **Remaining bias: fast-sim progress is optimistic.** The PX4 percentile mean is 0.34, and 41% of PX4 runs fall
  outside the fast-sim range. Examples: fix-detour easy 54 vs 70 m, fix-all easy 45 vs 64 m, fix-detour-creep medium
  26 vs 50 m. Exceptions where PX4 is better: fix-all3 hard 35 vs 25 m, fix-all hard 23 vs 16 m. Treat fast-sim
  progress as an upper estimate and rank configs with it; confirm absolute progress in PX4.
- Per preset and per world: `calibration_tables_v3.md`; scatter: `calibration_scatter_v3.png`.

**Speed** (default eye, fix-all3, 120 s episodes). Not a clean benchmark: other sessions ran video renders and
`camera_stress.py`, and the Mac slept once mid-run (`caffeinate -i` does not stop lid-close sleep).
- 9 workers: **65 episodes/min, 130x real time** (`--rows 1`: 80 episodes/min, 160x).
- One process: `--rows 1` 49x real time; the default eye costs ~1040 µs per drone-frame for render in small groups.
- Last idle measurement, old eye, 1 ray x 4 rows: 91x in one process, 267x with 6 workers. The new default doubles the
  render rays.
- A clean number needs an idle machine on mains power with the lid open.
