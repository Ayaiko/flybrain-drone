#!/bin/bash
# Fresh sim + one fly-brain flight. Usage (inside container): px4/run.sh <mode> <seed> <seconds> [gui]
MODE=$1; SEED=$2; DUR=$3; GUI=$4
source /work/px4/sim_env.sh
pkill -9 -x mavsdk_server
/work/px4/start_sim.sh $SEED $GUI >/dev/null
for i in $(seq 1 60); do sleep 2; grep -q "Startup script returned successfully" /tmp/px4.log && break; done
sleep 15
SRV=$(python3 -c 'import mavsdk,os;print(os.path.join(os.path.dirname(mavsdk.__file__),"bin","mavsdk_server"))')
$SRV -p 50051 udpin://0.0.0.0:14540 > /tmp/mavsdk_server.log 2>&1 &
sleep 2
cd /work && timeout $((DUR * 3 + 120)) python3 -u px4/fly_px4.py $MODE $SEED $DUR > /tmp/fly_${MODE}_${SEED}.log 2>&1
pkill -9 -x mavsdk_server
grep -E "SUMMARY|offboard failed" /tmp/fly_${MODE}_${SEED}.log || (echo "FAILED $MODE $SEED"; grep -v "^\s*$" /tmp/fly_${MODE}_${SEED}.log | head -5)
