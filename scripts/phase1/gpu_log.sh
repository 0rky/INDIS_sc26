#!/bin/bash
# gpu_log.sh <rundir> — 1 Hz GPU utilization/memory/power log on THIS node.
# Started/stopped by run_cell.sh over ssh. Appends until killed.
RUNDIR=$1; H=$(hostname -s)
mkdir -p "$RUNDIR"
exec nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv,noheader -l 1 > "$RUNDIR/${H}_gpu.csv" 2>/dev/null
