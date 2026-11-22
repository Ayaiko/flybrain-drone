Rule: {'stall_s': 90, 'stall_m': 3.0, 'max_coll': 20, 'limit_s': 360}; 6 fast-sim reps per (config, world); pilot = shared

| Spearman | progress | collisions |
|---|---|---|
| all | 0.80 | 0.85 |
| none-v0.8 | nan | 0.89 |
| lplc2-bold-gate-sv0-v0.8 | 0.73 | 0.14 |
| lplc2-bold-gate-sv0-gg0-v0.8 | 0.72 | 0.89 |
| fix-detour | 0.57 | 0.62 |
| fix-all | 0.48 | -0.24 |
| fix-all2@T600 | 0.82 | 0.81 |
| fix-detour-creep | 0.77 | 0.58 |
| fix-all3 | 0.83 | 0.62 |
| brain | 0.69 | 0.84 |
| ceiling all: one fast-sim rep vs mean of the others | 0.87 | 0.89 |
| ceiling brain: one fast-sim rep vs mean of the others | 0.81 | 0.87 |

| PX4 run as a percentile of the fast-sim reps (75 brain pairs) | progress | collisions |
|---|---|---|
| mean (0.5 = centred) | 0.34 | 0.55 |
| fraction outside the fast-sim range | 0.41 | 0.19 |

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
| lplc2-bold-gate-sv0-v0.8 | ff_easy_0 | 49.2 | 66.7 (3.6) | 9 | 7.3 (2.1) | timeout | timeout:6 |
| lplc2-bold-gate-sv0-v0.8 | ff_easy_1 | 52.6 | 57.6 (19.4) | 4 | 6.2 (1.3) | timeout | stalled:1 timeout:5 |
| lplc2-bold-gate-sv0-v0.8 | ff_easy_2 | 63.0 | 64.4 (6.0) | 11 | 7.7 (2.6) | stalled | stalled:1 timeout:5 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_0 | 52.7 | 50.0 (11.1) | 17 | 15.0 (2.9) | timeout | stalled:2 timeout:3 too_many_collisions:1 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_1 | 42.2 | 49.8 (5.7) | 14 | 12.0 (2.2) | stalled | stalled:3 timeout:3 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_2 | 53.1 | 56.8 (7.3) | 9 | 12.5 (2.6) | stalled | stalled:4 timeout:2 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_3 | 44.8 | 37.8 (14.3) | 19 | 14.3 (4.7) | stalled | stalled:2 timeout:3 too_many_collisions:1 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_4 | 52.5 | 43.1 (10.8) | 16 | 14.2 (4.4) | timeout | stalled:4 timeout:1 too_many_collisions:1 |
| lplc2-bold-gate-sv0-v0.8 | ff_medium_5 | 63.9 | 50.6 (3.4) | 14 | 12.5 (3.8) | timeout | stalled:2 timeout:3 too_many_collisions:1 |
| lplc2-bold-gate-sv0-v0.8 | ff_hard_0 | 12.0 | 37.4 (10.9) | 5 | 20.0 (0.0) | stalled | too_many_collisions:6 |
| lplc2-bold-gate-sv0-v0.8 | ff_hard_1 | 17.1 | 19.0 (6.6) | 11 | 14.7 (4.5) | stalled | stalled:4 too_many_collisions:2 |
| lplc2-bold-gate-sv0-v0.8 | ff_hard_2 | 10.8 | 31.4 (12.8) | 6 | 17.2 (4.3) | stalled | stalled:2 too_many_collisions:4 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_easy_0 | 63.0 | 63.4 (4.9) | 3 | 7.5 (3.2) | timeout | timeout:6 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_easy_1 | 14.6 | 50.0 (24.7) | 6 | 6.2 (3.3) | stalled | exit:1 stalled:2 timeout:3 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_easy_2 | 68.2 | 66.4 (3.9) | 7 | 7.8 (2.4) | timeout | stalled:1 timeout:5 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_medium_0 | 14.9 | 47.1 (14.0) | 11 | 13.5 (3.6) | stalled | stalled:3 timeout:3 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_medium_1 | 54.6 | 50.0 (3.0) | 20 | 14.0 (4.4) | too_many_collisions | stalled:2 timeout:3 too_many_collisions:1 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_medium_2 | 53.2 | 55.9 (5.3) | 11 | 13.3 (4.7) | stalled | stalled:4 timeout:1 too_many_collisions:1 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_hard_0 | 48.8 | 33.4 (15.0) | 20 | 15.2 (5.4) | too_many_collisions | stalled:3 too_many_collisions:3 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_hard_1 | 15.3 | 20.5 (12.1) | 14 | 15.3 (6.0) | stalled | stalled:2 timeout:1 too_many_collisions:3 |
| lplc2-bold-gate-sv0-gg0-v0.8 | ff_hard_2 | 20.6 | 27.4 (12.3) | 20 | 16.0 (3.2) | too_many_collisions | stalled:4 too_many_collisions:2 |
| fix-detour | ff_easy_0 | 63.0 | 73.3 (5.4) | 7 | 5.5 (1.1) | timeout | exit:2 timeout:4 |
| fix-detour | ff_easy_1 | 42.9 | 64.9 (8.7) | 8 | 6.3 (1.4) | timeout | exit:1 timeout:5 |
| fix-detour | ff_easy_2 | 54.8 | 72.4 (9.7) | 10 | 4.2 (1.1) | timeout | exit:3 stalled:1 timeout:2 |
| fix-detour | ff_medium_0 | 55.7 | 57.8 (4.0) | 16 | 13.2 (1.6) | timeout | timeout:6 |
| fix-detour | ff_medium_1 | 65.5 | 44.4 (13.0) | 15 | 10.8 (3.5) | timeout | stalled:4 timeout:2 |
| fix-detour | ff_medium_2 | 53.1 | 53.3 (0.7) | 9 | 11.0 (2.4) | stalled | stalled:4 timeout:2 |
| fix-detour | ff_medium_3 | 29.4 | 46.7 (9.2) | 19 | 13.8 (2.8) | timeout | stalled:1 timeout:5 |
| fix-detour | ff_medium_4 | 43.4 | 49.7 (14.7) | 10 | 12.0 (3.6) | stalled | stalled:2 timeout:4 |
| fix-detour | ff_medium_5 | 19.7 | 57.8 (8.6) | 7 | 11.5 (2.7) | stalled | stalled:1 timeout:5 |
| fix-detour | ff_hard_0 | 10.2 | 39.7 (12.7) | 11 | 14.7 (3.9) | stalled | stalled:4 timeout:1 too_many_collisions:1 |
| fix-detour | ff_hard_1 | 22.4 | 26.4 (13.9) | 20 | 16.0 (4.6) | too_many_collisions | stalled:4 timeout:1 too_many_collisions:1 |
| fix-detour | ff_hard_2 | 19.5 | 17.4 (4.6) | 16 | 11.0 (5.9) | stalled | stalled:6 |
| fix-all | ff_easy_0 | 57.7 | 61.6 (7.8) | 4 | 0.2 (0.4) | timeout | stalled:1 timeout:5 |
| fix-all | ff_easy_1 | 16.8 | 61.7 (7.1) | 1 | 0.2 (0.4) | stalled | timeout:6 |
| fix-all | ff_easy_2 | 60.8 | 67.8 (10.3) | 2 | 0.8 (0.9) | timeout | exit:2 stalled:1 timeout:3 |
| fix-all | ff_medium_0 | 14.1 | 44.3 (15.7) | 0 | 1.7 (0.9) | stalled | stalled:4 timeout:2 |
| fix-all | ff_medium_1 | 43.7 | 42.3 (6.5) | 1 | 0.7 (0.9) | stalled | stalled:6 |
| fix-all | ff_medium_2 | 52.1 | 54.0 (3.0) | 1 | 1.2 (1.1) | stalled | stalled:5 timeout:1 |
| fix-all | ff_medium_3 | 47.7 | 33.8 (16.9) | 5 | 1.7 (1.1) | stalled | stalled:3 timeout:3 |
| fix-all | ff_medium_4 | 20.3 | 23.6 (9.0) | 2 | 0.3 (0.5) | stalled | stalled:6 |
| fix-all | ff_medium_5 | 42.4 | 41.6 (10.9) | 0 | 0.8 (1.2) | stalled | stalled:6 |
| fix-all | ff_hard_0 | 41.9 | 16.8 (1.6) | 0 | 1.5 (1.3) | stalled | stalled:6 |
| fix-all | ff_hard_1 | 17.6 | 13.4 (3.2) | 0 | 0.3 (0.5) | stalled | stalled:6 |
| fix-all | ff_hard_2 | 10.2 | 17.9 (4.8) | 0 | 1.2 (2.2) | stalled | stalled:6 |
| fix-all2@T600 | ff_easy_0 | 80.0 | 80.0 (0.0) | 3 | 1.0 (0.8) | exit | exit:6 |
| fix-all2@T600 | ff_easy_1 | 80.0 | 63.5 (11.7) | 2 | 1.7 (0.7) | exit | exit:2 stalled:4 |
| fix-all2@T600 | ff_easy_2 | 62.9 | 77.1 (6.4) | 4 | 3.0 (1.8) | stalled | exit:5 stalled:1 |
| fix-all2@T600 | ff_medium_0 | 18.9 | 58.9 (10.1) | 7 | 9.8 (3.6) | stalled | exit:1 stalled:5 |
| fix-all2@T600 | ff_medium_1 | 48.3 | 51.1 (7.2) | 6 | 6.3 (4.4) | stalled | stalled:6 |
| fix-all2@T600 | ff_medium_2 | 52.6 | 58.5 (7.5) | 3 | 4.2 (2.5) | stalled | stalled:6 |
| fix-all2@T600 | ff_medium_3 | 50.9 | 56.4 (19.1) | 12 | 7.0 (2.5) | stalled | exit:2 stalled:3 timeout:1 |
| fix-all2@T600 | ff_medium_4 | 43.0 | 41.8 (20.0) | 9 | 6.2 (2.6) | stalled | exit:1 stalled:5 |
| fix-all2@T600 | ff_medium_5 | 57.7 | 58.2 (0.8) | 4 | 5.2 (0.7) | stalled | stalled:6 |
| fix-all2@T600 | ff_hard_0 | 44.7 | 40.4 (13.9) | 9 | 8.0 (4.9) | stalled | stalled:6 |
| fix-all2@T600 | ff_hard_1 | 11.3 | 13.2 (3.0) | 4 | 4.8 (2.2) | stalled | stalled:6 |
| fix-all2@T600 | ff_hard_2 | 10.5 | 36.9 (14.4) | 6 | 9.8 (4.6) | stalled | stalled:6 |
| fix-detour-creep | ff_easy_0 | 45.4 | 58.5 (4.0) | 6 | 2.3 (1.7) | timeout | timeout:6 |
| fix-detour-creep | ff_easy_1 | 52.4 | 58.7 (3.2) | 2 | 2.3 (1.1) | timeout | timeout:6 |
| fix-detour-creep | ff_easy_2 | 48.9 | 62.9 (1.3) | 1 | 1.8 (0.7) | timeout | timeout:6 |
| fix-detour-creep | ff_medium_0 | 18.5 | 48.4 (14.0) | 3 | 6.7 (1.6) | stalled | stalled:1 timeout:5 |
| fix-detour-creep | ff_medium_1 | 15.7 | 51.7 (5.6) | 5 | 7.7 (1.1) | stalled | stalled:1 timeout:5 |
| fix-detour-creep | ff_medium_2 | 44.2 | 50.2 (6.6) | 5 | 5.2 (1.3) | timeout | stalled:4 timeout:2 |
| fix-detour-creep | ff_hard_0 | 42.3 | 28.7 (14.9) | 6 | 6.7 (1.2) | stalled | stalled:5 timeout:1 |
| fix-detour-creep | ff_hard_1 | 10.8 | 14.9 (4.5) | 5 | 5.3 (1.6) | stalled | stalled:6 |
| fix-detour-creep | ff_hard_2 | 19.4 | 26.5 (13.4) | 9 | 8.3 (2.9) | stalled | stalled:4 timeout:2 |
| fix-all3 | ff_easy_0 | 49.5 | 57.0 (2.8) | 2 | 1.0 (0.8) | timeout | timeout:6 |
| fix-all3 | ff_easy_1 | 44.5 | 58.6 (3.8) | 3 | 0.7 (0.7) | timeout | timeout:6 |
| fix-all3 | ff_easy_2 | 60.2 | 62.4 (2.6) | 1 | 0.5 (0.5) | timeout | timeout:6 |
| fix-all3 | ff_medium_0 | 42.8 | 56.7 (4.6) | 0 | 2.8 (1.6) | timeout | timeout:6 |
| fix-all3 | ff_medium_1 | 40.8 | 47.9 (3.3) | 1 | 3.3 (1.4) | timeout | stalled:2 timeout:4 |
| fix-all3 | ff_medium_2 | 52.9 | 52.1 (2.1) | 2 | 4.0 (1.6) | timeout | stalled:2 timeout:4 |
| fix-all3 | ff_hard_0 | 42.5 | 40.6 (11.6) | 7 | 5.3 (2.6) | timeout | stalled:3 timeout:3 |
| fix-all3 | ff_hard_1 | 22.2 | 13.3 (3.9) | 7 | 4.0 (1.6) | stalled | stalled:6 |
| fix-all3 | ff_hard_2 | 41.2 | 19.6 (10.8) | 6 | 4.3 (1.5) | timeout | stalled:5 timeout:1 |

| config / preset | PX4 progress | fastsim progress | PX4 coll | fastsim coll |
|---|---|---|---|---|
| none-v0.8 easy | 80.0 | 80.0 | 1.0 | 1.0 |
| none-v0.8 medium | 80.0 | 80.0 | 4.8 | 4.7 |
| none-v0.8 hard | 80.0 | 80.0 | 14.0 | 10.2 |
| lplc2-bold-gate-sv0-v0.8 easy | 54.9 | 62.9 | 8.0 | 7.1 |
| lplc2-bold-gate-sv0-v0.8 medium | 51.5 | 48.0 | 14.8 | 13.4 |
| lplc2-bold-gate-sv0-v0.8 hard | 13.3 | 29.2 | 7.3 | 17.3 |
| lplc2-bold-gate-sv0-gg0-v0.8 easy | 48.6 | 59.9 | 5.3 | 7.2 |
| lplc2-bold-gate-sv0-gg0-v0.8 medium | 40.9 | 51.0 | 14.0 | 13.6 |
| lplc2-bold-gate-sv0-gg0-v0.8 hard | 28.2 | 27.1 | 18.0 | 15.5 |
| fix-detour easy | 53.6 | 70.2 | 8.3 | 5.3 |
| fix-detour medium | 44.5 | 51.6 | 12.7 | 12.1 |
| fix-detour hard | 17.4 | 27.8 | 15.7 | 13.9 |
| fix-all easy | 45.1 | 63.7 | 2.3 | 0.4 |
| fix-all medium | 36.7 | 39.9 | 1.5 | 1.1 |
| fix-all hard | 23.2 | 16.0 | 0.0 | 1.0 |
| fix-all2@T600 easy | 74.3 | 73.5 | 3.0 | 1.9 |
| fix-all2@T600 medium | 45.2 | 54.1 | 6.8 | 6.4 |
| fix-all2@T600 hard | 22.2 | 30.2 | 6.3 | 7.6 |
| fix-detour-creep easy | 48.9 | 60.0 | 3.0 | 2.2 |
| fix-detour-creep medium | 26.1 | 50.1 | 4.3 | 6.5 |
| fix-detour-creep hard | 24.2 | 23.4 | 6.7 | 6.8 |
| fix-all3 easy | 51.4 | 59.3 | 2.0 | 0.7 |
| fix-all3 medium | 45.5 | 52.2 | 1.0 | 3.4 |
| fix-all3 hard | 35.3 | 24.5 | 6.7 | 4.6 |

Speed: {'episodes': 522, 'sim_s_total': 145916, 'worker_wall_s_total': 18554.2, 'rtf_per_core': 7.9, 'batch_wall_s': None, 'episodes_per_min': None, 'rtf_total': None, 'workers': None, 'cpu': 'arm', 'batched': None, 'px4_reference': 'PX4 + Gazebo Docker: ~1x real time per container, 3 containers ~1.8x total (forest-test-field notes)'}
