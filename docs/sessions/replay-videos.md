# Replay videos for forest flights: hand-over (2026-11-22)

All new files are in ~/Developer/flybrain-drone/viz/ (README.md has the full usage) and results/videos/. No existing files were edited, and nothing was committed (the project is not a git repo).

## Tools (run from the project root with .venv/bin/python)

- **viz/replay2d.py `<run json>` `[--speed N --t0 --t1 --out x.mp4|x.gif]`**
  - Builds the video from the log alone, at 1280×720:
    - an overview map and a follow map (the obstacle cross-section at flight height, trail, heading, the drone coloured by brain state, the 120 ommatidia as a ring at their true azimuths, crashes as an expanding red ring with a banner, remembered hazards as ▲)
    - a scrolling eye strip (time × azimuth) with looming and state columns
    - a 20 s timeline of true clearance and looming
    - text: zone, y / best / exit, speed, collisions, and badges
  - Renders at about 8–30 frames/s depending on load.
  - Default speed is 2×, or 4× for runs longer than 120 s.
- **viz/chasecam.py `<run json>` `[--speed --t0 --t1 --cam chase|high|side --size 640x360]`**
  - Renders a 3D chase cam in Gazebo.
  - Starts or reuses its own container **flybrain-px4-viz** (image flybrain-px4:worker, `--user px4`, `FLYBRAIN_WORKER=viz` so GZ_PARTITION is fb_viz, no ports). It does not touch flybrain-px4, -w1 or -w2.
  - Inside, `gz sim -s` runs a generated copy of the world (viz/_worlds/<w>_chase.sdf: gravity off, only the Physics, UserCommands and Sensors systems, an x500_base body and a camera model), paused, with no PX4.
  - Per frame it calls set_pose_vector, steps once and grabs the image. I verified there is no pose→image lag against a settled reference.
  - Each frame gets a HUD (state, LOOMING, crash flash, eye band, progress), then the host encodes it.
  - Speed is about 0.15–0.3 s per frame at 640×360 with 4 llvmpipe threads.
  - The camera heading is low-passed, and the near clip plane is pushed out so trunks between the camera and the drone are cut away.
  - Drone roll and pitch are not logged; they are approximated from acceleration, for looks only.
- **Encoding:** neither the host PATH nor the container has ffmpeg. The tools find the copy already installed with Live2D Cubism (/Applications/Live2D Cubism 5.3/tools/ffmpeg/ffmpeg, h264_videotoolbox). You can override it with `$FFMPEG`. Nothing was downloaded. Without ffmpeg they fall back to a Pillow GIF.

## Example videos in results/videos/

All 2D files are 1280×720 at about 2.5 Mb/s. All chase files are 640×360 at 24 fps and about 1.5 Mb/s.

| file | run | length |
|---|---|---|
| lplc2-bold-gate-sv0-v0.8_ff_medium_0_2d.mp4 / _chase.mp4 | full 360 s at 4× | 90 s each (28 MB / 17 MB) |
| lplc2-bold-gate-sv0-v0.8_ff_hard_1_2d.mp4 / _chase.mp4 | full 360 s at 4× | 90 s each (28 / 17 MB) |
| lplc2-bold-gate-sv0-v0.8_ff_hard_1_traploop_2d.mp4 / _chase.mp4 | t = 95–192 s at 2× (the trap loop) | 49 s each (15 / 9 MB) |
| none-v0.8_ff_hard_0_2d.mp4 / _chase.mp4 | full 103 s at 2× | 52 s each (16 / 10 MB) |

Render times: 2D took 2.5–8 min per full run, slowed by CPU contention. Chase took 9–16 min per file at 0.28–0.47 s/frame while the batch containers were busy.

## Notes for the next steps

- The 2D replay costs nothing to make. Consider adding it to forest_eval (for example the worst run per config) once that file is free.
- The chase cam competes for CPU with batch runs. Leave the viz container idle or remove it (`docker rm -f flybrain-px4-viz`) during long batches.
- The trap loop is easy to see in both views: 25 hits on the same trunks at y ≈ 14–17 between t = 99 and 185 s. After each crash recovery the drone heads back into the same spot, and the remembered hazards (▲) don't keep it away. Around t = 135–155 s the 2D timeline shows true clearance ≈ 0 for most of 20 s: the drone stays pressed against one trunk near (−0.1, 14.4) instead of backing clear. This is worth checking in the recovery logic.
- `none-*` runs log state 0 throughout, so their HUD badge always reads "warmup". That comes from the log, not the replay.
- The container flybrain-px4-viz is still up and idle; no gz is running in it.
