#!/bin/bash
# net_log.sh <rundir> — 1 Hz RDMA byte-counter sampler on THIS node.
# Gives a TIME SERIES of network traffic during a cell (the before/after
# snapshots only give one aggregate number). Started/stopped by run_cell.sh.
# Columns: ts, dev, xmit_bytes_cumulative, rcv_bytes_cumulative
# (counters are in 4-byte words on the wire; we convert to bytes here.)
source /opt/phase1/phase1_env.sh
RUNDIR=$1; H=$(hostname -s); OUT="$RUNDIR/${H}_net.csv"
mkdir -p "$RUNDIR"
echo "ts,dev,xmit_bytes,rcv_bytes" > "$OUT"
while true; do
  T=$(date +%s.%N)
  for D in $DEVA $DEVB; do
    X=$(cat /sys/class/infiniband/$D/ports/1/counters/port_xmit_data 2>/dev/null || echo 0)
    R=$(cat /sys/class/infiniband/$D/ports/1/counters/port_rcv_data 2>/dev/null || echo 0)
    echo "$T,$D,$((X*4)),$((R*4))" >> "$OUT"
  done
  sleep 1
done
