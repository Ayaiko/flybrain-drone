# fastsim: fast ray-cast forest sim, calibrated against PX4 (2026-11-22)

Where: `~/Developer/flybrain-drone/fastsim/` (README) and `results/fastsim/calibration.md` (full report). numpy only;
nothing installed, no existing project files touched.

- **Eye**: ray-casts the real camera geometry (480 columns x 4 rows) against `px4/forest/<world>.json`, using the same
  textures, fitted lighting and the bridge's ommatidium averaging. Held-out correlation with logged Gazebo frames is
  0.936 per frame (noise ceiling 0.994). Gazebo's image is **1 frame older than the pose** the bridge reads; fastsim
  reproduces this.
- **x500 model**: fitted to the logged runs. Yaw delay 133 ms, tau 0.20 s; speed delay 267 ms, tau 0.43 s, accel 2.0
  m/s², brake 3.4 m/s². Held-out error at 1 s is 4.5 cm and 1.1 deg. PX4 follows yaw rates up to ≥ 2 rad/s; the 60 deg/s
  cap is the pilot's own clip.
- **Gate** (3 configs x 9 worlds, 10 reps, vs the single PX4 runs; re-run on the batched runner): Spearman progress
  **0.89** (passes 0.8), collisions **0.74**. Both are at the ceiling one PX4 run per pair allows (0.83 / 0.70). Pilot-state time shares match
  PX4 within 1–2%. Open: lplc2-bold on hard is too optimistic (25 vs 13 m; could be chance), and the goal_gap effect is
  not reproduced.
- **PX4 quirk found**: in 8 of 9 route-follower runs PX4 flies 2–6 deg right of the true heading (never in brain runs).
  `dynamics.vel_off` models it (default 0).
- **Throughput redesign (as requested)**:
  - `fastsim/kernel.py` `cast(scene, cam, ex, ey, ez, h)` is the single array-in/array-out ray-cast kernel for N
    drones, ready to swap for numba or Rust.
  - `runner.run_group` steps N episodes of a world together. It gives the same episodes as single runs, checked frame
    by frame.
  - `truth.BatchTruth` is a 1 m grid collision truth, identical to `Obstacles.query`.
  - `default_workers()` uses all cores, minus 3 while flybrain-px4 containers are busy.
  - Per drone-frame at group 32: 787 µs (render 80%, pilot 13%, truth 3%, dynamics 1%), so ~42x real time per process.
  - Full gate batch: **37.8 episodes/min, 144x real time** on 7 workers while PX4 used ~7 cores. On an idle machine,
    expect ~100 episodes/min (estimate).
  - Next step is a compiled render kernel (numba/Rust): 5–20x on 80% of the time. Then the Python FlyVision pilot
    (~100–160 µs) becomes the limit. Nothing installed.
- **Shared pilot**: `--pilot shared` runs `px4/forest_pilot.Pilot` unchanged (v_meas, bump on the next frame, ToF via
  RangeSensor). With defaults it is within noise of the bridge mirror, so fix-detour / brake / flow / tof configs can be
  trained in fastsim now.
  Example: `.venv/bin/python -m fastsim.runner --configs fix-all --pilot shared --presets easy,medium,hard --reps 3 --workers 8`
- Contract for whoever owns `forest_pilot`: keep `Pilot(P, meta, eye, fps, seed).step(now, x, y, z, h, lum, v_meas,
  bump, tof) -> (v, yaw, info)` and `stats()`, or adapt `fastsim/pilot.py:SharedPilot`. `forest_geom.query` may return 3
  or 4 values; both work.
- Suggested next PX4 work: 2–3 repeats of the brain configs on ff_hard_* to settle the hard-preset gap. Log the
  commanded forward speed too, which sharpens the speed-delay fit.

## Update 2026-11-22 01:25: curriculum + shared pilot folded in
- The fast sim now drives the shared `px4/forest_pilot.Pilot` **by default** (`--pilot shared`), with the same loop as
  the current bridge: v_meas, bump on the next frame, ToF via RangeSensor. `bridge` is kept only to reproduce the first
  calibration.
- `fastsim/train_cem.py`: curriculum CEM on the pool (px4/forest/pool.json), following your rules: start at 0, +0.25
  when elite success >= 0.7 on >= 8 worlds, 20% easier-level episodes, held-out `ff_*_0..5` reported every N
  generations. It writes `results/fastsim/cem/<name>/{log,heldout}.jsonl`, `best.json` (overrides for PX4) and
  `state.json` (resume). Smoke-tested only; no training run started.
- `fastsim.calibrate --configs fix-detour,fix-all --tag _batch3` compares any configs that have PX4 runs. PX4 runs with
  < 1 m progress (failed takeoff / stall) are excluded automatically. **Waiting for the PX4 batch to finish** before
  running it, because some batch-3 runs were hurt by the overload (fix-all ff_medium_0 = 0 m).
- Worker cap of 3 is in force (`fastsim/max_workers.txt`). Throughput gets re-measured on an idle machine after your go.
