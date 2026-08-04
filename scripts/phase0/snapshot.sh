#!/bin/bash
# snapshot.sh <rundir> <tag>
# Snapshot NIC + RDMA counters for BOTH rails on THIS node.
# Runs on every node; called directly, by capture_local.sh, or via snap_all.sh.
source /opt/phase0/env.sh
RUNDIR=$1; TAG=$2; H=$(hostname -s)
mkdir -p "$RUNDIR"

for IF in $IFA $IFB; do
  ethtool -S "$IF" > "$RUNDIR/${H}_ethtool_${IF}_${TAG}.txt" 2>/dev/null
done

for DEV in $DEVA $DEVB; do
  for D in counters hw_counters; do
    OUT="$RUNDIR/${H}_${DEV}_${D}_${TAG}.txt"; : > "$OUT"
    for f in /sys/class/infiniband/$DEV/ports/1/$D/*; do
      echo "$(basename "$f") $(cat "$f" 2>/dev/null)" >> "$OUT"
    done
  done
done

date +%s.%N > "$RUNDIR/${H}_ts_${TAG}.txt"
