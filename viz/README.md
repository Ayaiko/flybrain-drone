# viz/ — replay videos of logged forest flights

Two tools that turn a forest flight log (`px4/runs/forest/<run id>_<world>.json`, written by `px4/fly_forest.py`) into a video.
Run them from the project root on the host, using the project venv (`.venv/bin/python`). Outputs go to `results/videos/` by default.

| tool | needs | speed | shows |
|---|---|---|---|
| `replay2d.py` | log only (matplotlib + ffmpeg/Pillow) | ~10–30 frames/s | top-down map, eye strip, state / looming, clearance, text |
| `chasecam.py` | Docker image `flybrain-px4:worker` (Gazebo, software render) | ~4–6 frames/s at 640×360 | 3D chase camera in the real forest world, with a heads-up display (HUD) |

## replay2d.py: 2D replay from the log alone

```bash
.venv/bin/python viz/replay2d.py px4/runs/forest/lplc2-bold-gate-sv0-v0.8_ff_hard_1.json
.venv/bin/python viz/replay2d.py <run.json> --speed 2 --t0 95 --t1 192 --out results/videos/clip.mp4
```

The 1280×720 layout has these panels:
- **Overview:** the whole field, the path so far, crashes (red ×), and a box showing the follow window.
- **Follow map:** a 12 m window centred on the drone, drawn from the obstacle cross-section at flight height (the same band as `make_forest.plot_map`). It also shows:
  - the corridor, the trail (with the last 3 s drawn bold), and the heading line
  - the drone body coloured by brain state
  - the 120 ommatidia as a grey ring around the drone at their true azimuths
  - crashes as they happen (an expanding red ring), and remembered hazards (orange ▲)
- **Eye strip:** the logged 120-px eye vector over the last 10 s, as time × azimuth with left on the left and the newest row at the top. Next to it are a looming flag column and a brain-state column (grey warmup, blue settling, green fixate, orange saccade, red saccade_start).
- **Timeline:** the last 20 s of true clearance to the nearest obstacle (blue), the looming flag (pink) and crashes (×).
- **Text:** time, zone, y / best y / exit, speed, altitude, collision and hazard counts, a state badge, a LOOMING badge, a CRASH banner, and a progress bar.

Options:
- `--speed N` plays N× real time (default 2, or 4 for runs over 120 s).
- `--fps` sets the frame rate (default 30).
- `--t0` / `--t1` cut a time window.
- `--window` sets the follow-map width in metres.
- `--eye-secs` sets the length of the eye-strip history.
- `--size` sets the frame size.

Encoding uses ffmpeg if one is installed: on `PATH`, in `$FFMPEG`, or in a known local copy (this Mac has one inside Live2D Cubism, which supports h264_videotoolbox). Nothing is downloaded. Without ffmpeg, or with `--out x.gif`, it writes a GIF with Pillow, keeping every 2nd frame (`--gif-every`).

## chasecam.py: 3D chase cam rendered by Gazebo from the logged poses

```bash
.venv/bin/python viz/chasecam.py px4/runs/forest/none-v0.8_ff_hard_0.json --speed 2
.venv/bin/python viz/chasecam.py <run.json> --t0 95 --t1 192 --speed 2 --cam chase --out results/videos/clip_chase.mp4
```

How it works:
- **Container:** the host script starts or reuses the container `flybrain-px4-viz`. It is created from image `flybrain-px4:worker` with `-v <project>:/work`, `--user px4` and `FLYBRAIN_WORKER=viz`, which gives GZ partition `fb_viz`. It has no ports. The script then re-runs itself inside the container with `--inside`.
- **World:** inside, it writes `viz/_worlds/<world>_chase.sdf`, a generated copy of the forest world with these changes:
  - gravity off
  - only the Physics, UserCommands and Sensors (ogre2) systems
  - an `x500_base` body named `drone`
  - a camera model `chasecam` (topic `/chase/cam`)

  No PX4 is involved. `gz sim -s` runs paused on Xvfb with `LIBGL_ALWAYS_SOFTWARE=1`.
- **Per frame:** it calls `/world/<w>_chase/set_pose_vector` for the drone and camera, steps one iteration through `/world/<w>_chase/control` (`multi_step=1`), and waits for that iteration's image. I checked this against a slow, settled reference: the image matches the pose that was just set exactly, with no lag. Each frame gets a HUD (time, zone, progress, mode, collisions, state badge, LOOMING, a red crash flash with its kind, the 120-px eye band with L/R and a centre tick, and a progress bar) and is saved as a JPEG under `results/videos/_frames_<out>/`.
- **Encoding:** the host encodes the frames with ffmpeg (GIF fallback) and deletes them unless you pass `--keep-frames`.
- **Camera:** the camera follows a low-passed copy of the heading (τ = 0.8 s for `chase`), so saccades and the 180° crash turn-arounds swing it round smoothly. Its near clip plane is pushed out (1.1 m for `chase`), so trunks between the camera and the drone are cut away instead of filling the frame.
  - `--cam chase` (default): 2.6 m behind and 1.1 m above
  - `--cam high`: 4 m behind and 3.2 m above
  - `--cam side`: 3.2 m to the side
  - `--cam fpv`: first person from the logged pose, 100° FOV, drone body hidden. `flyeye_explainer.py` uses this mode.
- **Frame-only output:** `--no-hud` saves raw images. `--frames-only --frames DIR` renders into DIR and stops without encoding.
- **Drone attitude:** roll and pitch are not logged. They are approximated from the logged acceleration (clipped to ±20°), for looks only.

Options:
- `--speed` / `--fps` / `--t0` / `--t1` work as in `replay2d.py`; the default fps here is 24.
- `--size` sets the frame size (default 640×360). Rendering is software-only, so keep it modest.
- `--threads` sets the number of llvmpipe threads (default 4).

Cost: about 0.15–0.3 s per frame at 640×360 with 4 threads on an idle machine. With the PX4 batch containers busy, measured times were 0.28–0.47 s per frame, so a full 360 s run at 4× and 24 fps (2160 frames) took 10–16 min. For long runs, use `--speed` or a time window. The script kills its own `gz sim` when it finishes; the container stays up and idle. Remove it with `docker rm -f flybrain-px4-viz`.

The script doesn't use the other `flybrain-px4*` containers.

Note: `none-*` runs (route following with no brain) log state 0 throughout, so their state badge always reads "warmup" and the looming flag is always off. That is how the log is written, not a bug in the replay.

## flyeye_explainer.py: explainer video for people outside the project

```bash
.venv/bin/python viz/flyeye_explainer.py --clip all          # the built-in clips a, b, c -> results/videos/explainer/
.venv/bin/python viz/flyeye_explainer.py <run.json> --key 92.9 --pre 7 --post 3.5 --slow-pre 2 --slow-post 2 --title "..." --out x.mp4
```

The video is 1280×720. Each panel is one step of the pipeline, labelled in Thai with the English term in brackets:
1. **What a person sees:** a first-person Gazebo camera at eye height with a 100° field of view, rendered from the logged poses (`chasecam.py --cam fpv`). The 8°-tall band that the fly eye samples is marked.
2. **What the fly eye sees:** the logged 120-value eye vector as a pixelated 330° panorama, marked front / left / right / back. Brackets show the camera's 100° view and the LPLC2 field (front 0–35°).
3. **Motion detectors (T4/T5 → EMD):** outward motion for each pair of neighbouring ommatidia, recomputed from the logged eye with `flydrone.controllers.EMD(120, 1/30, 0.035, 0.25)`.
4. **Looming neurons (LPLC2):** left, right and sum bars, the threshold (`lplc2_thr` × a slowly adapting baseline), a 4 s history, the logged looming flag, and an escape-turn banner. The banner's direction comes from the heading change that actually follows.
5. **Top-down map:** the obstacle ahead along the heading (or the nearest one), outlined. The readout is the true clearance from the drone body, measured the same way as the logged `gap`.

How the neuron signals are recomputed:
- The EMD is reset where the brain resets it: at saccade start, and at the fresh start after a turn or recovery (logged state 4, 1, 0).
- The baseline adapts only during calm fixation.
- Outside fixation the bars are faded and the history has gaps, because the brain ignores vision while turning.
- The recomputed flag matches the logged one well: on the three runs used, 268/309, 303/387 and 175/220 of the logged looming frames during fixation are reproduced, and the onset in clip (a) is exact. The logged flag is still drawn as well.

Each clip starts with a 4 s title card, then plays at 1×, slowing to 0.5× around the key moment. First-person frames take about 0.2–0.3 s each to render in the viz container.

| clip | run, key time | what it shows |
|---|---|---|
| `a_success_trunk_escape.mp4` (19 s) | `lplc2-bold-gate-sv0-v0.8_ff_easy_0`, t = 92.9 s | After a saccade the drone fixates with a trunk front-right 1.8 m ahead and speeds up. The right LPLC2 unit rises and the sum crosses threshold at 1.2 m clearance. It turns 72° left and passes the trunk with about 0.8 m to spare. This run uses vision only; the `fix-all*` configs also use a ToF sensor, so they were avoided here. |
| `b_fail_flat_rock_wall.mp4` (16 s) | `lplc2-bold-gate-sv0-v0.8_ff_easy_2`, t = 319.5 s | The drone fixates for 1.1 s at 0.7 m/s toward the plain grey rock wall. LPLC2 peaks at 82% of threshold, there is no alarm, and it crashes. |
| `c_fail_thin_branch.mp4` (16 s) | `lplc2-bold-gate-sv0-v0.8_ff_medium_3`, t = 242.5 s | A branch at eye height barely changes the 120 values. LPLC2 stays under threshold and the drone hits the branch. |

## Files

- `common.py`: run loader (`Run`: per-frame arrays, decoded eye stream `eye[n,120]`, crashes, hazards, world metadata, speed), the ffmpeg finder and the `Encoder` (mp4 pipe, or GIF fallback).
- `replay2d.py`, `chasecam.py`: the two tools.
- `_worlds/`: generated SDFs. They are safe to delete; they are rewritten on every chasecam run.

## Example videos (`results/videos/`)

| file | run | what to look at |
|---|---|---|
| `lplc2-bold-gate-sv0-v0.8_ff_medium_0_{2d,chase}.mp4` | fly brain, medium forest, 360 s at 4× | slow progress to y ≈ 53 m, 17 trunk hits |
| `lplc2-bold-gate-sv0-v0.8_ff_hard_1_{2d,chase}.mp4` | fly brain, hard forest, 360 s at 4× | stuck at y ≈ 12–17 m, 45 hits |
| `lplc2-bold-gate-sv0-v0.8_ff_hard_1_traploop_{2d,chase}.mp4` | the same run, t = 95–192 s at 2× | the trap loop: 25 hits between t = 99 and 185 s on the same few trunks around x ≈ −2…0.4, y ≈ 14–17; after each crash recovery it heads back into the same spot |
| `none-v0.8_ff_hard_0_{2d,chase}.mp4` | blind route follower, hard forest, 103 s at 2× | reaches the exit through 10 obstacles: trunks, foliage, a branch, plain trunks and the boulder |
