#!/bin/bash
# Diagnostic: one flight with sim-speed (RTF) sampled every 5 s from inside the container.
# Usage (inside container): px4/diag_run.sh <config> <world> <seconds> <tag>
source /work/px4/sim_env.sh
( for i in $(seq 1 80); do sleep 5; echo "$(date +%T) $(timeout 3 gz topic -e -t /stats -n 1 2>/dev/null | grep real_time_factor | cut -c19-24)"; done > /tmp/rtf_$4.log 2>&1 ) &
MON=$!
RUN_ID=diag_$4 bash /work/px4/run_forest.sh $1 $2 $3 | grep -oE 'SUMMARY.{0,40}|FAILED.*|TAKEOFF.*|STALLED.*'
kill $MON 2>/dev/null
grep -E "STALLED|TAKEOFF|offboard" /tmp/fly_diag_$4_$2.log | head -3
echo "RTF: $(tr '\n' ' ' < /tmp/rtf_$4.log)"
