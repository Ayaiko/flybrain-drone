#!/bin/bash
# Start headless Gazebo + PX4 SITL (x500 with fly eye) inside the container.
# Usage: start_sim.sh <seed> [gui]   gui = also show the Gazebo GUI on a virtual screen, viewable at http://localhost:6080/vnc.html
# Env: WORLD=<world name> (default forest_<seed>), SPEED=<sim speed factor> (faster than real time, default 1)
SEED=${1:-0}; GUI=${2:-}
source /work/px4/sim_env.sh
for n in px4 ruby; do pkill -9 -x $n; done; sleep 1
export DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1
pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1280x800x24 >/tmp/xvfb.log 2>&1 &); sleep 1
if [ -n "$GUI" ]; then
  pgrep -x x11vnc >/dev/null || (x11vnc -display :99 -forever -shared -nopw -quiet >/tmp/x11vnc.log 2>&1 &)
  pgrep -x websockify >/dev/null && false || (websockify --web /usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &)
fi
PX=/work/px4/PX4-Autopilot; B=$PX/build/px4_sitl_default
export PX4_GZ_WORLD=${WORLD:-forest_${SEED}} PX4_SIM_MODEL=gz_x500_flyeye PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="0,0,0.25,0,0,1.5708"
[ -n "$SPEED" ] && [ "$SPEED" != "1" ] && export PX4_SIM_SPEED_FACTOR=$SPEED
[ -z "$GUI" ] && export HEADLESS=1
# private copy of the rootfs per container: the build folder is shared by all worker containers through the bind mount,
# and PX4 writes parameters / dataman / logs there
# (never copy rootfs/log: it holds GBs of old flight logs, and copying them over the bind mount on every run, in 3
#  containers at once, saturated disk I/O and caused sensor timeouts / failed takeoffs)
WD=/tmp/px4_rootfs; rm -rf $WD; mkdir -p $WD
for f in $B/rootfs/* $B/rootfs/.[!.]*; do [ "$(basename "$f")" = log ] || cp -a "$f" $WD/; done
cd $WD && nohup $B/bin/px4 -d $B/etc -w $WD > /tmp/px4.log 2>&1 &
echo started
