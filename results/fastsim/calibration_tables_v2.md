Rule: {'stall_s': 90, 'stall_m': 3.0, 'max_coll': 20, 'limit_s': 360}; 6 fast-sim reps per (config, world); pilot = shared

| Spearman | progress | collisions |
|---|---|---|
| all | 0.76 | 0.81 |
| none-v0.8 | nan | 0.89 |
| lplc2-bold-gate-sv0-v0.8 | 0.49 | 0.01 |
| lplc2-bold-gate-sv0-gg0-v0.8 | 0.23 | 0.89 |
| fix-detour | 0.52 | 0.34 |
| fix-all | 0.70 | -0.36 |
| fix-all2@T600 | 0.58 | 0.48 |
| brain | 0.60 | 0.77 |
| ceiling all: one fast-sim rep vs mean of the others | 0.69 | 0.68 |
| ceiling brain: one fast-sim rep vs mean of the others | 0.47 | 0.64 |

| PX4 run as a percentile of the fast-sim reps (57 brain pairs) | progress | collisions |
|---|---|---|
| mean (0.5 = centred) | 0.49 | 0.62 |
| fraction outside the fast-sim range | 0.21 | 0.19 |

| config | world | PX4 progress | fastsim progress (sd) | PX4 coll | fastsim coll (sd) | PX4 end | fastsim ends |
|---|---|---|---|---|---|---|---|
| none-v0.8 | ff_easy_0 | 80.0 | 80.0 (0.0) | 0 | 0.0 (0.0) | exit | exit:6 |
| none-v0.8 | ff_easy_1 | 80.0 | 80.0 (0.0) | 1 | 1.0 (0.0) | exit | exit:6 |
| none-v0.8 | ff_easy_2 | 80.0 | 80.0 (0.0) | 2 | 2.0 (0.0) | exit | exit:6 |
| none-v0.8 | ff_medium_0 | 80.0 | 80.0 (0.0) | 6 | 3.0 (0.0) | exit | exit:6 |
| none-v0.8 | ff_medium_1 | 80.0 | 80.0 (0.0) | 6 | 5.0 (0.0) | exit | exit:6 |
| none-v0.8 | ff_medium_2 | 80.0 | 80.0 (0.0) | 4 | 3.0 (0.0) | exit | exit:6 |
| none-v0.8 | ff_medium_3 | 80.0 | 80.0 (0.0) | 6 | 6.0 (0.0) | exit | exit:6 |
| none-v0.8 | ff_medium_4 | 80.0 | 80.0 (0.0) | 4 | 6.0 (0.0) | exit | exit:6 |
| none-v0.8 | ff_medium_5 | 80.0 | 80.0 (0.0) | 3 | 5.0 (0.0) | exit | exit:6 |
| none-v0.8 | ff_hard_0 | 80.0 | 80.0 (0.0) | 10 | 8.7 (0.5) | exit | exit:6 |
| none-v0.8 | ff_hard_1 | 80.0 | 80.0 (0.0) | 14 | 10.0 (0.0) | exit | exit:6 |
| none-v0.8 | ff_hard_2 | 80.0 | 80.0 (0.0) | 18 | 12.0 (0.0) | exit | exit:6 |
| lplc2-bold-gate-sv0-v0.8 | ff_easy_0 | 49.2 | 55.9 (8.7) | 9 | 6.2 (2.1) | timeout | timeout:6 |
| lplc2-bold-gate-sv0-v0.8 | ff_easy_1 | 52.6 | 50.7 (18.3) | 4 | 6.3 (3.0) | timeout | stalled:1 timeout:5 |
| lplc2-bold-gate-sv0-v0.8 | ff_easy_2 | 63.0 | 61.8 (6.8) | 11 | 9.5 (3.3) | stalled | stalled:2 timeout:4 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_0 | 52.7 | 37.3 (24.9) | 17 | 8.8 (6.4) | timeout | stalled:2 timeout:4 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_1 | 42.2 | 48.4 (4.6) | 14 | 15.3 (3.6) | stalled | stalled:3 timeout:1 too_many_collisions:2 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_2 | 53.1 | 42.6 (23.5) | 9 | 11.5 (7.0) | stalled | stalled:2 timeout:2 too_many_collisions:2 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_3 | 44.8 | 44.8 (20.2) | 19 | 11.7 (6.3) | stalled | stalled:1 timeout:5 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_4 | 52.5 | 23.8 (22.6) | 16 | 5.3 (6.0) | timeout | stalled:5 timeout:1 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_5 | 63.9 | 38.8 (18.7) | 14 | 13.8 (6.8) | timeout | stalled:2 timeout:2 too_many_collisions:2 |
| lplc2-bold-gate-sv0-v0.8 | ff_hard_0 | 12.0 | 27.6 (13.3) | 5 | 16.3 (3.5) | stalled | stalled:3 timeout:1 too_many_collisions:2 |
| lplc2-bold-gate-sv0-v0.8 | ff_hard_1 | 17.1 | 19.6 (4.3) | 11 | 18.2 (3.7) | stalled | stalled:2 too_many_collisions:4 |
| lplc2-bold-gate-sv0-v0.8 | ff_hard_2 | 10.8 | 12.7 (6.4) | 6 | 9.3 (6.0) | stalled | stalled:5 too_many_collisions:1 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_easy_0 | 63.0 | 40.0 (23.0) | 3 | 4.2 (3.7) | timeout | stalled:2 timeout:4 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_easy_1 | 14.6 | 50.0 (21.8) | 6 | 5.3 (3.0) | stalled | stalled:1 timeout:5 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_easy_2 | 68.2 | 60.8 (5.9) | 7 | 7.8 (4.4) | timeout | stalled:1 timeout:5 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_medium_0 | 14.9 | 34.2 (20.4) | 11 | 9.3 (5.0) | stalled | stalled:3 timeout:3 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_medium_1 | 54.6 | 27.9 (22.3) | 20 | 10.8 (8.8) | too_many_collisions | stalled:3 timeout:2 too_many_collisions:1 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_medium_2 | 53.2 | 49.2 (7.7) | 11 | 11.8 (4.2) | stalled | stalled:2 timeout:3 too_many_collisions:1 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_hard_0 | 48.8 | 25.1 (15.3) | 20 | 16.0 (7.2) | too_many_collisions | stalled:2 timeout:1 too_many_collisions:3 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_hard_1 | 15.3 | 24.4 (13.6) | 14 | 13.2 (5.7) | stalled | stalled:3 timeout:2 too_many_collisions:1 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_hard_2 | 20.6 | 32.4 (17.1) | 20 | 14.3 (7.2) | too_many_collisions | stalled:2 timeout:1 too_many_collisions:3 |
| fix-detour | ff_easy_0 | 63.0 | 61.8 (6.9) | 7 | 6.3 (2.2) | timeout | timeout:6 |
| fix-detour | ff_easy_1 | 42.9 | 59.1 (5.5) | 8 | 5.5 (2.1) | timeout | timeout:6 |
| fix-detour | ff_easy_2 | 54.8 | 65.7 (9.0) | 10 | 6.0 (2.2) | timeout | exit:1 timeout:5 |
| fix-detour | ff_medium_0 | 55.7 | 38.2 (25.7) | 16 | 6.2 (4.5) | timeout | stalled:3 timeout:3 |
| fix-detour | ff_medium_1 | 65.5 | 48.2 (3.6) | 15 | 12.3 (3.9) | timeout | stalled:2 timeout:4 |
| fix-detour | ff_medium_2 | 53.1 | 44.8 (18.7) | 9 | 6.7 (3.3) | stalled | stalled:6 |
| fix-detour | ff_medium_3 | 29.4 | 34.5 (19.9) | 19 | 11.2 (6.2) | timeout | stalled:3 timeout:3 |
| fix-detour | ff_medium_4 | 43.4 | 18.5 (18.8) | 10 | 4.8 (5.0) | stalled | stalled:5 timeout:1 |
| fix-detour | ff_medium_5 | 19.7 | 43.8 (19.0) | 7 | 9.3 (5.3) | stalled | stalled:2 timeout:4 |
| fix-detour | ff_hard_0 | 10.2 | 41.4 (10.7) | 11 | 13.7 (3.4) | stalled | stalled:3 timeout:2 too_many_collisions:1 |
| fix-detour | ff_hard_1 | 22.4 | 14.7 (6.2) | 20 | 8.7 (4.9) | too_many_collisions | stalled:5 timeout:1 |
| fix-detour | ff_hard_2 | 19.5 | 20.0 (15.5) | 16 | 8.5 (5.3) | stalled | stalled:5 timeout:1 |
| fix-all | ff_easy_0 | 57.7 | 46.4 (16.1) | 4 | 0.7 (0.9) | timeout | stalled:1 timeout:5 |
| fix-all | ff_easy_1 | 16.8 | 21.5 (25.9) | 1 | 0.2 (0.4) | stalled | stalled:4 timeout:2 |
| fix-all | ff_easy_2 | 60.8 | 43.5 (23.4) | 2 | 0.5 (0.8) | timeout | stalled:2 timeout:4 |
| fix-all | ff_medium_0 | 14.1 | 30.3 (23.8) | 0 | 1.0 (1.2) | stalled | stalled:4 timeout:2 |
| fix-all | ff_medium_1 | 43.7 | 37.4 (15.4) | 1 | 1.2 (1.1) | stalled | stalled:5 timeout:1 |
| fix-all | ff_medium_2 | 52.1 | 53.3 (7.5) | 1 | 1.0 (1.0) | stalled | stalled:5 timeout:1 |
| fix-all | ff_medium_3 | 47.7 | 32.8 (19.8) | 5 | 1.7 (2.6) | stalled | stalled:3 timeout:3 |
| fix-all | ff_medium_4 | 20.3 | 14.3 (14.6) | 2 | 0.0 (0.0) | stalled | stalled:6 |
| fix-all | ff_medium_5 | 42.4 | 20.4 (18.5) | 0 | 1.0 (1.5) | stalled | stalled:5 timeout:1 |
| fix-all | ff_hard_0 | 41.9 | 24.9 (14.0) | 0 | 1.2 (1.5) | stalled | stalled:6 |
| fix-all | ff_hard_1 | 17.6 | 9.2 (3.2) | 0 | 1.2 (1.8) | stalled | stalled:6 |
| fix-all | ff_hard_2 | 10.2 | 23.2 (10.6) | 0 | 2.2 (2.2) | stalled | stalled:5 timeout:1 |
| fix-all2@T600 | ff_easy_0 | 80.0 | 54.5 (36.0) | 3 | 1.3 (1.4) | exit | exit:4 stalled:2 |
| fix-all2@T600 | ff_easy_1 | 80.0 | 41.1 (39.0) | 2 | 1.7 (2.2) | exit | exit:3 stalled:3 |
| fix-all2@T600 | ff_easy_2 | 62.9 | 67.1 (28.8) | 4 | 2.8 (2.0) | stalled | exit:5 stalled:1 |
| fix-all2@T600 | ff_medium_0 | 18.9 | 43.4 (30.9) | 7 | 3.8 (3.4) | stalled | stalled:5 timeout:1 |
| fix-all2@T600 | ff_medium_1 | 48.3 | 56.8 (8.8) | 6 | 6.8 (3.3) | stalled | stalled:6 |
| fix-all2@T600 | ff_medium_2 | 52.6 | 52.8 (0.2) | 3 | 4.0 (1.6) | stalled | stalled:6 |
| fix-all2@T600 | ff_medium_3 | 50.9 | 45.1 (23.5) | 12 | 4.5 (2.4) | stalled | stalled:5 timeout:1 |
| fix-all2@T600 | ff_medium_4 | 43.0 | 20.1 (20.7) | 9 | 2.8 (3.3) | stalled | stalled:6 |
| fix-all2@T600 | ff_medium_5 | 57.7 | 33.3 (32.1) | 4 | 4.5 (4.8) | stalled | stalled:5 timeout:1 |
| fix-all2@T600 | ff_hard_0 | 44.7 | 45.7 (2.7) | 9 | 11.7 (4.5) | stalled | stalled:6 |
| fix-all2@T600 | ff_hard_1 | 11.3 | 12.4 (6.3) | 4 | 5.2 (2.8) | stalled | stalled:6 |
| fix-all2@T600 | ff_hard_2 | 10.5 | 19.3 (14.0) | 6 | 7.2 (5.0) | stalled | stalled:6 |

| config / preset | PX4 progress | fastsim progress | PX4 coll | fastsim coll |
|---|---|---|---|---|
| none-v0.8 easy | 80.0 | 80.0 | 1.0 | 1.0 |
| none-v0.8 medium | 80.0 | 80.0 | 4.8 | 4.7 |
| none-v0.8 hard | 80.0 | 80.0 | 14.0 | 10.2 |
| lplc2-bold-gate-sv0-v0.8 easy | 54.9 | 56.1 | 8.0 | 7.3 |
| lplc2-bold-gate-sv0-v0.8 medium | 51.5 | 39.3 | 14.8 | 11.1 |
| lplc2-bold-gate-sv0-v0.8 hard | 13.3 | 20.0 | 7.3 | 14.6 |
| lplc2-bold-gate-sv0-gg0-v0.8 easy | 48.6 | 50.3 | 5.3 | 5.8 |
| lplc2-bold-gate-sv0-gg0-v0.8 medium | 40.9 | 37.1 | 14.0 | 10.7 |
| lplc2-bold-gate-sv0-gg0-v0.8 hard | 28.2 | 27.3 | 18.0 | 14.5 |
| fix-detour easy | 53.6 | 62.2 | 8.3 | 5.9 |
| fix-detour medium | 44.5 | 38.0 | 12.7 | 8.4 |
| fix-detour hard | 17.4 | 25.4 | 15.7 | 10.3 |
| fix-all easy | 45.1 | 37.1 | 2.3 | 0.4 |
| fix-all medium | 36.7 | 31.4 | 1.5 | 1.0 |
| fix-all hard | 23.2 | 19.1 | 0.0 | 1.5 |
| fix-all2@T600 easy | 74.3 | 54.2 | 3.0 | 1.9 |
| fix-all2@T600 medium | 45.2 | 41.9 | 6.8 | 4.4 |
| fix-all2@T600 hard | 22.2 | 25.8 | 6.3 | 8.0 |

Speed: {'episodes': 414, 'sim_s_total': 104128, 'worker_wall_s_total': 2848.6, 'rtf_per_core': 36.6, 'batch_wall_s': 966.1, 'episodes_per_min': 25.7, 'rtf_total': 107.8, 'workers': 3, 'cpu': 'arm', 'batched': {'episodes': 414, 'workers': 3, 'wall_s': 966.1, 'episodes_per_min': 25.7, 'rtf_total': 107.8, 'rtf_per_worker': 36.5, 'us_per_drone_frame': {'render': 715.6, 'pilot': 131.2, 'truth': 28.3, 'dynamics': 14.7, 'book': 21.4}, 'share': {'render': 0.785, 'pilot': 0.144, 'truth': 0.031, 'dynamics': 0.016, 'book': 0.023}}, 'px4_reference': 'PX4 + Gazebo Docker: ~1x real time per container, 3 containers ~1.8x total (forest-test-field notes)'}
