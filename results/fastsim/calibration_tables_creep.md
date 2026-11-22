Rule: {'stall_s': 90, 'stall_m': 3.0, 'max_coll': 20, 'limit_s': 360}; 10 fast-sim reps per (config, world); pilot = shared

| Spearman | progress | collisions |
|---|---|---|
| all | 0.43 | 0.79 |
| fix-detour-creep | 0.25 | 0.35 |
| fix-all3 | 0.08 | 0.49 |
| fix-detour | 0.52 | 0.43 |
| fix-all | 0.76 | -0.48 |
| brain | 0.43 | 0.79 |
| ceiling all: one fast-sim rep vs mean of the others | 0.50 | 0.69 |
| ceiling brain: one fast-sim rep vs mean of the others | 0.50 | 0.69 |

| PX4 run as a percentile of the fast-sim reps (42 brain pairs) | progress | collisions |
|---|---|---|
| mean (0.5 = centred) | 0.62 | 0.66 |
| fraction outside the fast-sim range | 0.26 | 0.17 |

| config | world | PX4 progress | fastsim progress (sd) | PX4 coll | fastsim coll (sd) | PX4 end | fastsim ends |
|---|---|---|---|---|---|---|---|
| fix-detour-creep | ff_easy_0 | 45.4 | 14.0 (12.3) | 6 | 1.2 (1.7) | timeout | stalled:6 timeout:4 |
| fix-detour-creep | ff_easy_1 | 52.4 | 21.4 (13.6) | 2 | 2.1 (2.2) | timeout | stalled:3 timeout:7 |
| fix-detour-creep | ff_easy_2 | 48.9 | 25.5 (13.8) | 1 | 2.3 (2.2) | timeout | stalled:3 timeout:7 |
| fix-detour-creep | ff_medium_0 | 18.5 | 20.6 (12.7) | 3 | 3.2 (2.3) | stalled | stalled:7 timeout:3 |
| fix-detour-creep | ff_medium_1 | 15.7 | 24.4 (19.0) | 5 | 3.1 (2.7) | stalled | stalled:6 timeout:4 |
| fix-detour-creep | ff_medium_2 | 44.2 | 34.5 (16.6) | 5 | 2.6 (1.5) | timeout | stalled:2 timeout:8 |
| fix-detour-creep | ff_hard_0 | 42.3 | 30.1 (15.0) | 6 | 6.8 (2.6) | stalled | stalled:4 timeout:6 |
| fix-detour-creep | ff_hard_1 | 10.8 | 15.6 (9.3) | 5 | 7.1 (3.9) | stalled | stalled:9 timeout:1 |
| fix-detour-creep | ff_hard_2 | 19.4 | 19.1 (14.8) | 9 | 6.6 (4.4) | stalled | stalled:8 timeout:2 |
| fix-all3 | ff_easy_0 | 49.5 | 10.8 (15.4) | 2 | 0.1 (0.3) | timeout | stalled:8 timeout:2 |
| fix-all3 | ff_easy_1 | 44.5 | 8.9 (9.7) | 3 | 0.2 (0.6) | timeout | stalled:8 timeout:2 |
| fix-all3 | ff_easy_2 | 60.2 | 24.6 (16.6) | 1 | 0.3 (0.5) | timeout | stalled:3 timeout:7 |
| fix-all3 | ff_medium_0 | 42.8 | 17.3 (14.2) | 0 | 1.9 (2.0) | timeout | stalled:6 timeout:4 |
| fix-all3 | ff_medium_1 | 40.8 | 24.9 (18.8) | 1 | 0.6 (0.8) | timeout | stalled:4 timeout:6 |
| fix-all3 | ff_medium_2 | 52.9 | 37.3 (7.1) | 2 | 1.8 (1.2) | timeout | timeout:10 |
| fix-all3 | ff_hard_0 | 42.5 | 25.1 (14.7) | 7 | 4.4 (2.2) | timeout | stalled:6 timeout:4 |
| fix-all3 | ff_hard_1 | 22.2 | 16.5 (8.6) | 7 | 4.3 (2.9) | stalled | stalled:9 timeout:1 |
| fix-all3 | ff_hard_2 | 41.2 | 23.5 (13.9) | 6 | 4.4 (2.3) | timeout | stalled:6 timeout:4 |
| fix-detour | ff_easy_0 | 63.0 | 56.3 (19.3) | 7 | 5.5 (2.8) | timeout | stalled:1 timeout:9 |
| fix-detour | ff_easy_1 | 42.9 | 54.5 (18.1) | 8 | 4.8 (2.4) | timeout | stalled:1 timeout:9 |
| fix-detour | ff_easy_2 | 54.8 | 58.0 (20.3) | 10 | 5.6 (2.7) | timeout | exit:1 stalled:1 timeout:8 |
| fix-detour | ff_medium_0 | 55.7 | 37.5 (26.5) | 16 | 6.9 (4.8) | timeout | stalled:5 timeout:5 |
| fix-detour | ff_medium_1 | 65.5 | 48.3 (3.4) | 15 | 12.2 (3.2) | timeout | stalled:4 timeout:6 |
| fix-detour | ff_medium_2 | 53.1 | 45.8 (16.5) | 9 | 9.3 (5.1) | stalled | stalled:7 timeout:2 too_many_collisions:1 |
| fix-detour | ff_medium_3 | 29.4 | 31.6 (22.6) | 19 | 9.2 (6.7) | timeout | stalled:6 timeout:4 |
| fix-detour | ff_medium_4 | 43.4 | 26.2 (18.5) | 10 | 6.5 (4.5) | stalled | stalled:9 timeout:1 |
| fix-detour | ff_medium_5 | 19.7 | 47.5 (15.6) | 7 | 10.9 (5.2) | stalled | stalled:3 timeout:6 too_many_collisions:1 |
| fix-detour | ff_hard_0 | 10.2 | 43.2 (8.7) | 11 | 13.6 (3.0) | stalled | stalled:5 timeout:4 too_many_collisions:1 |
| fix-detour | ff_hard_1 | 22.4 | 17.6 (6.5) | 20 | 11.2 (5.4) | too_many_collisions | stalled:8 timeout:1 too_many_collisions:1 |
| fix-detour | ff_hard_2 | 19.5 | 21.8 (14.5) | 16 | 10.1 (5.4) | stalled | stalled:8 timeout:2 |
| fix-all | ff_easy_0 | 57.7 | 44.7 (19.6) | 4 | 0.6 (0.9) | timeout | stalled:2 timeout:8 |
| fix-all | ff_easy_1 | 16.8 | 29.4 (27.1) | 1 | 0.1 (0.3) | stalled | stalled:5 timeout:5 |
| fix-all | ff_easy_2 | 60.8 | 45.7 (18.4) | 2 | 0.5 (0.7) | timeout | stalled:2 timeout:8 |
| fix-all | ff_medium_0 | 14.1 | 31.7 (24.0) | 0 | 0.8 (1.0) | stalled | stalled:8 timeout:2 |
| fix-all | ff_medium_1 | 43.7 | 34.3 (17.6) | 1 | 0.7 (1.0) | stalled | stalled:9 timeout:1 |
| fix-all | ff_medium_2 | 52.1 | 51.9 (6.3) | 1 | 0.8 (0.9) | stalled | stalled:9 timeout:1 |
| fix-all | ff_medium_3 | 47.7 | 32.9 (19.9) | 5 | 1.1 (2.2) | stalled | stalled:6 timeout:4 |
| fix-all | ff_medium_4 | 20.3 | 25.6 (17.9) | 2 | 0.6 (1.3) | stalled | stalled:10 |
| fix-all | ff_medium_5 | 42.4 | 23.6 (19.1) | 0 | 1.1 (1.4) | stalled | stalled:9 timeout:1 |
| fix-all | ff_hard_0 | 41.9 | 26.2 (14.7) | 0 | 0.9 (1.3) | stalled | stalled:10 |
| fix-all | ff_hard_1 | 17.6 | 9.8 (2.6) | 0 | 1.0 (1.5) | stalled | stalled:10 |
| fix-all | ff_hard_2 | 10.2 | 21.9 (8.7) | 0 | 2.8 (3.6) | stalled | stalled:9 timeout:1 |

| config / preset | PX4 progress | fastsim progress | PX4 coll | fastsim coll |
|---|---|---|---|---|
| fix-detour-creep easy | 48.9 | 20.3 | 3.0 | 1.9 |
| fix-detour-creep medium | 26.1 | 26.5 | 4.3 | 3.0 |
| fix-detour-creep hard | 24.2 | 21.6 | 6.7 | 6.8 |
| fix-all3 easy | 51.4 | 14.7 | 2.0 | 0.2 |
| fix-all3 medium | 45.5 | 26.5 | 1.0 | 1.4 |
| fix-all3 hard | 35.3 | 21.7 | 6.7 | 4.4 |
| fix-detour easy | 53.6 | 56.3 | 8.3 | 5.3 |
| fix-detour medium | 44.5 | 39.5 | 12.7 | 9.2 |
| fix-detour hard | 17.4 | 27.5 | 15.7 | 11.6 |
| fix-all easy | 45.1 | 40.0 | 2.3 | 0.4 |
| fix-all medium | 36.7 | 33.3 | 1.5 | 0.8 |
| fix-all hard | 23.2 | 19.3 | 0.0 | 1.6 |

Speed: {'episodes': 420, 'sim_s_total': 108572, 'worker_wall_s_total': 3970.6, 'rtf_per_core': 27.3, 'batch_wall_s': 596.0, 'episodes_per_min': 42.3, 'rtf_total': 182.2, 'workers': 7, 'cpu': 'arm', 'batched': {'episodes': 420, 'workers': 7, 'wall_s': 596.0, 'episodes_per_min': 42.3, 'rtf_total': 182.2, 'rtf_per_worker': 27.3, 'us_per_drone_frame': {'render': 909.3, 'pilot': 216.1, 'truth': 40.1, 'dynamics': 21.1, 'book': 30.3}, 'share': {'render': 0.747, 'pilot': 0.178, 'truth': 0.033, 'dynamics': 0.017, 'book': 0.025}}, 'px4_reference': 'PX4 + Gazebo Docker: ~1x real time per container, 3 containers ~1.8x total (forest-test-field notes)'}
