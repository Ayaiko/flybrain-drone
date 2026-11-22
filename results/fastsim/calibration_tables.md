Rule: {'stall_s': 90, 'stall_m': 3.0, 'max_coll': 20, 'limit_s': 360}; 10 fast-sim reps per (config, world); pilot = bridge

| Spearman | progress | collisions |
|---|---|---|
| all_27 | 0.89 | 0.74 |
| none-v0.8 | nan | 0.99 |
| lplc2-bold-gate-sv0-v0.8 | 0.63 | 0.19 |
| lplc2-bold-gate-sv0-gg0-v0.8 | 0.52 | 0.80 |
| brain_18 | 0.65 | 0.56 |
| ceiling all_27: one fast-sim rep vs mean of the others | 0.83 | 0.70 |
| ceiling brain_18: one fast-sim rep vs mean of the others | 0.45 | 0.52 |

| PX4 run as a percentile of the fast-sim reps (18 brain pairs) | progress | collisions |
|---|---|---|
| mean (0.5 = centred) | 0.48 | 0.54 |
| fraction outside the fast-sim range | 0.11 | 0.11 |

| config | world | PX4 progress | fastsim progress (sd) | PX4 coll | fastsim coll (sd) | PX4 end | fastsim ends |
|---|---|---|---|---|---|---|---|
| none-v0.8 | ff_easy_0 | 80.0 | 80.0 (0.0) | 0 | 0.0 (0.0) | exit | exit:10 |
| none-v0.8 | ff_easy_1 | 80.0 | 80.0 (0.0) | 1 | 1.0 (0.0) | exit | exit:10 |
| none-v0.8 | ff_easy_2 | 80.0 | 80.0 (0.0) | 2 | 2.0 (0.0) | exit | exit:10 |
| none-v0.8 | ff_medium_0 | 80.0 | 80.0 (0.0) | 6 | 3.0 (0.0) | exit | exit:10 |
| none-v0.8 | ff_medium_1 | 80.0 | 80.0 (0.0) | 6 | 5.0 (0.0) | exit | exit:10 |
| none-v0.8 | ff_medium_2 | 80.0 | 80.0 (0.0) | 4 | 3.0 (0.0) | exit | exit:10 |
| none-v0.8 | ff_hard_0 | 80.0 | 80.0 (0.0) | 10 | 8.6 (0.5) | exit | exit:10 |
| none-v0.8 | ff_hard_1 | 80.0 | 80.0 (0.0) | 14 | 10.0 (0.0) | exit | exit:10 |
| none-v0.8 | ff_hard_2 | 80.0 | 80.0 (0.0) | 18 | 12.0 (0.0) | exit | exit:10 |
| lplc2-bold-gate-sv0-v0.8 | ff_easy_0 | 49.2 | 48.7 (17.9) | 9 | 5.8 (3.3) | timeout | stalled:2 timeout:8 |
| lplc2-bold-gate-sv0-v0.8 | ff_easy_1 | 52.6 | 52.3 (17.1) | 4 | 5.4 (3.8) | timeout | stalled:1 timeout:9 |
| lplc2-bold-gate-sv0-v0.8 | ff_easy_2 | 63.0 | 52.1 (24.0) | 11 | 5.8 (2.5) | stalled | stalled:2 timeout:8 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_0 | 52.7 | 31.5 (22.5) | 17 | 8.5 (6.6) | timeout | stalled:5 timeout:4 too_many_collisions:1 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_1 | 42.2 | 46.5 (11.2) | 14 | 15.1 (4.0) | stalled | stalled:2 timeout:8 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_2 | 53.1 | 47.4 (15.6) | 9 | 11.2 (4.8) | stalled | stalled:6 timeout:4 |
| lplc2-bold-gate-sv0-v0.8 | ff_hard_0 | 12.0 | 34.0 (15.3) | 5 | 13.2 (5.8) | stalled | stalled:6 timeout:1 too_many_collisions:3 |
| lplc2-bold-gate-sv0-v0.8 | ff_hard_1 | 17.1 | 18.1 (5.9) | 11 | 11.5 (4.9) | stalled | stalled:9 too_many_collisions:1 |
| lplc2-bold-gate-sv0-v0.8 | ff_hard_2 | 10.8 | 22.1 (12.9) | 6 | 13.4 (7.0) | stalled | stalled:6 timeout:1 too_many_collisions:3 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_easy_0 | 63.0 | 47.9 (21.9) | 3 | 6.2 (3.8) | timeout | stalled:2 timeout:8 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_easy_1 | 14.6 | 47.4 (22.6) | 6 | 5.9 (3.4) | timeout | exit:1 stalled:3 timeout:6 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_easy_2 | 68.2 | 60.0 (4.6) | 7 | 6.7 (2.9) | timeout | timeout:10 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_medium_0 | 14.9 | 42.9 (21.1) | 11 | 8.5 (5.3) | timeout | stalled:4 timeout:6 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_medium_1 | 54.6 | 42.0 (22.2) | 20 | 11.8 (6.2) | too_many_collisions | stalled:4 timeout:6 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_medium_2 | 53.2 | 47.8 (12.7) | 11 | 12.0 (2.8) | stalled | stalled:3 timeout:7 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_hard_0 | 48.8 | 28.6 (15.9) | 20 | 15.4 (6.3) | too_many_collisions | stalled:5 too_many_collisions:5 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_hard_1 | 15.3 | 19.0 (9.9) | 14 | 16.2 (5.0) | timeout | stalled:4 too_many_collisions:6 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_hard_2 | 20.6 | 29.7 (14.3) | 20 | 15.3 (5.6) | too_many_collisions | stalled:5 timeout:3 too_many_collisions:2 |

| config / preset | PX4 progress | fastsim progress | PX4 coll | fastsim coll |
|---|---|---|---|---|
| none-v0.8 easy | 80.0 | 80.0 | 1.0 | 1.0 |
| none-v0.8 medium | 80.0 | 80.0 | 5.3 | 3.7 |
| none-v0.8 hard | 80.0 | 80.0 | 14.0 | 10.2 |
| lplc2-bold-gate-sv0-v0.8 easy | 54.9 | 51.0 | 8.0 | 5.7 |
| lplc2-bold-gate-sv0-v0.8 medium | 49.3 | 41.8 | 13.3 | 11.6 |
| lplc2-bold-gate-sv0-v0.8 hard | 13.3 | 24.7 | 7.3 | 12.7 |
| lplc2-bold-gate-sv0-gg0-v0.8 easy | 48.6 | 51.8 | 5.3 | 6.3 |
| lplc2-bold-gate-sv0-gg0-v0.8 medium | 40.9 | 44.2 | 14.0 | 10.8 |
| lplc2-bold-gate-sv0-gg0-v0.8 hard | 28.2 | 25.8 | 18.0 | 15.6 |

Speed: {'episodes': 270, 'sim_s_total': 61599, 'worker_wall_s_total': 2927.3, 'rtf_per_core': 21.0, 'batch_wall_s': 428.6, 'episodes_per_min': 37.8, 'rtf_total': 143.7, 'workers': 7, 'cpu': 'arm', 'batched': {'episodes': 270, 'workers': 7, 'wall_s': 428.6, 'episodes_per_min': 37.8, 'rtf_total': 143.7, 'rtf_per_worker': 21.0, 'us_per_drone_frame': {'render': 1270.0, 'pilot': 158.5, 'truth': 73.0, 'dynamics': 40.1, 'book': 39.7}, 'share': {'render': 0.803, 'pilot': 0.1, 'truth': 0.046, 'dynamics': 0.025, 'book': 0.025}}, 'px4_reference': 'PX4 + Gazebo Docker: ~1x real time per container, 3 containers ~1.8x total (forest-test-field notes)'}
