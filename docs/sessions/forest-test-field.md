# Forest test field for flybrain-drone — hand-over (2026-11-22)

Repo: ~/Developer/flybrain-drone (no git). Sim: PX4 v1.17 SITL + Gazebo Harmonic in Docker (flybrain-px4 + workers -w1, -w2).
Files: px4/make_forest.py (generator, presets easy/medium/hard, --set overrides), px4/forest_geom.py (3D obstacle truth),
px4/fly_forest.py (bridge), px4/run_forest.sh, px4/forest_eval.py (--workers N, --param k=v), px4/forest_compare.py (--md),
px4/forest_configs.json, px4/forest_params.py. Results: results/forest/forest_runs.jsonl, paths_ff_*.png; log in EXPERIMENTS.md.

Findings: fly brain 0/9 exits (easy 55 m, medium 49 m, hard 13 m avg progress); blind route follower 9/9 exits but hits
route obstacles (1 / 5.3 / 14 per run). Crash causes: 56–62% re-hits after recovery, 58% no looming warning, 0.6 m/s impact
speed. Rock wall in low-texture zone = reproducible optic-flow failure. gg0 (goal_gap 0) mixed.
Tooling: 3 workers ≈ 1.8× throughput; speed factor >1 does not work (camera rendering caps RTF ~1); early stop added.
Pending: user approval of re-plan (recovery/brake/speed-by-flow/ToF fixes, fast ray-cast sim + replay videos, CEM training).
