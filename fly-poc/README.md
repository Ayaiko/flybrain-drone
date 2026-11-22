# FlyVis camera experiment

Open replay.html in a browser for the recorded camera frames, commands, and overhead trajectory. results.json contains all trial outcomes. readout.npz stores the trained linear classifier and feature normalization.

The camera ray casts a 3D ground plane and axis-aligned boxes. Vehicle motion is kinematic at fixed altitude and heading. Side commands move diagonally. The neural input is one grayscale camera sampled on the 721-column lattice. The teacher sees geometry during training. The tested controller sees camera luminance only. Goal means crossing the finish line. Stop detection requires 12 consecutive stop commands. Collision detection uses a 0.24 m horizontal body radius.

The frozen FlyVis flow/0000/000 parameter export comes from [Lulzx/fly-brain](https://github.com/Lulzx/fly-brain/tree/main/public/vision), based on [TuragaLab/flyvis](https://github.com/TuragaLab/flyvis). The C kernel implements the dynamics specified in flyvis.json. The readout averages six cell types in nine horizontal groups in each of two vertical bands and fits class-weighted ridge regression. Training uses 240 short simulated trajectories with obstacle brightness sampled independently of layout (integer gray levels 30–144). Evaluation uses separate seeds. The brighter obstacles use the same test seeds as their dark counterparts. Ten additional right-offset light-box trials use new seeds. The viewer includes an archived pre-change collision from baseline-light.json for direct comparison.

Reproduce with Python 3, NumPy and a C compiler by running python3 run.py from this directory. On the machine used for this test:

```sh
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk /Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13 run.py
```

## Local planner and motion simulation

The readout sends four scores with the camera capture time to planner.py. The fixed mission is to cross the forward finish line. The planner outputs lateral and forward velocity in the fixed-heading vehicle frame. Simulated state feedback supplies current velocity. A stale frame (over 250 ms), invalid scores, or invalid velocity commands a stop. When a forward range reading is supplied, the clearance check includes body radius, margin, motion during frame age and the next tick, and braking distance. A single forward beam cannot certify side clearance. The normal camera-only trials have no range input.

Vehicle response is acceleration-limited at 1.6 m/s² in this experiment. This is a planar simulation with ideal velocity feedback. It does not implement IMU estimation, PX4, or a hardware connection. The replay includes two injected faults: frozen camera timestamps while moving, and a forced forward proposal toward a wall with a simulated front range sensor. The viewer shows requested velocities separately from measured speed. The archived before-fix replay predates the planner and has no planner telemetry.

The default run loads the existing readout and evaluates 45 navigation trials plus 2 fault trials. Pass --retrain to regenerate the readout.

## Working from the combined repository

Use the repository virtual environment with NumPy and a C compiler. Run the following from the repository root. The binary weight export is copied locally during migration and is ignored by Git. On a fresh clone, fetch it and verify its expected hash first.

```sh
curl -fL https://raw.githubusercontent.com/Lulzx/fly-brain/main/public/vision/flyvis.bin -o fly-poc/model/flyvis.bin
echo '5ddc6fda8f207ce60d372ca6a11eed727ee3179de83c47f5c2eea15317b59315  fly-poc/model/flyvis.bin' | shasum -a 256 -c -
.venv/bin/python fly-poc/run.py
.venv/bin/python fly-poc/courses.py
.venv/bin/python fly-poc/gate_trial.py
```

The scripts regenerate the basic, multi-obstacle and gate replays. The recorded office clip and its local replay are also available under real-desk after migration. To regenerate those from TUM recording files, see real_video.py --help. In the two-gate comparison, the map and image controllers command the vehicle while FlyVis scores are recorded for comparison. The forest baseline and evaluation pipeline live in ../px4.
