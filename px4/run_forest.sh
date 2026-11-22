#!/bin/bash
# Fresh sim on a forest test-field world + one flight. Usage (inside a container):
#   px4/run_forest.sh <config or mode string> <world> <seconds> [gui]
# Env (docker exec -e): SPEED=2 (faster than real time), FLY_OVERRIDES='{"goal_gap":0}', RUN_ID=<output name>
CFG=$1; WORLD=$2; DUR=$3; GUI=$4; RUN_ID=${RUN_ID:-$CFG}
source /work/px4/sim_env.sh
pkill -9 -x mavsdk_server; pkill -9 -f "python3 -u px4/fly_"; sleep 1
WORLD=$WORLD /work/px4/start_sim.sh 0 $GUI >/dev/null
for i in $(seq 1 90); do sleep 1; grep -q "Ready for takeoff" /tmp/px4.log && break; done
grep -q "Ready for takeoff" /tmp/px4.log || { echo "FAILED $RUN_ID $WORLD: PX4 not ready"; tail -5 /tmp/px4.log; exit 1; }
if [ -n "$GUI" ]; then   # GUI camera follows the drone from behind once the GUI is up
  ( for i in $(seq 1 60); do gz service -l 2>/dev/null | grep -q /gui/follow && break; sleep 1; done
    gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 3000 --req 'data: "x500_flyeye_0"' >/dev/null
    gz service -s /gui/follow/offset --reqtype gz.msgs.Vector3d --reptype gz.msgs.Boolean --timeout 3000 --req 'x: 0, y: -3.0, z: 1.6' >/dev/null ) &
fi
SRV=$(python3 -c 'import mavsdk,os;print(os.path.join(os.path.dirname(mavsdk.__file__),"bin","mavsdk_server"))')
$SRV -p 50051 udpin://0.0.0.0:14540 > /tmp/mavsdk_server.log 2>&1 &
sleep 2
LOG=/tmp/fly_${RUN_ID}_${WORLD}.log
cd /work && RUN_ID=$RUN_ID timeout $((DUR * 20 + 600)) python3 -u px4/fly_forest.py "$CFG" "$WORLD" "$DUR" > $LOG 2>&1
pkill -9 -x mavsdk_server; for n in px4 ruby; do pkill -9 -x $n; done
grep -E "SUMMARY|offboard failed" $LOG || (echo "FAILED $RUN_ID $WORLD"; grep -v "^\s*$" $LOG | tail -8)
