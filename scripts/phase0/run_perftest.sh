#!/bin/bash
# run_perftest.sh — Experiment B (RDMA point-to-point). Run from the launch node.
# Starts the perftest server on the destination via SSH, then runs the client
# on the source under capture_local.sh. Fresh control port per test.
# Assumes GID identical on all nodes (verified in Stage 1.7).
set -u
source /opt/phase0/env.sh
declare -A IPA=([node1]=192.168.100.1 [node2]=192.168.100.2 [node3]=192.168.100.3 [node4]=192.168.100.4)
declare -A IPB=([node1]=192.168.101.1 [node2]=192.168.101.2 [node3]=192.168.101.3 [node4]=192.168.101.4)
PAIRS=("node1 node2" "node1 node3" "node1 node4" "node2 node3" "node2 node4" "node3 node4")
PORT=18000

run_test () {  # run_test <SRC> <DST> <DEV> <IP> <TEST> <EXTRA> <RUNID>
  local SRC=$1 DST=$2 DEV=$3 IP=$4 TEST=$5 EXTRA=$6 RUNID=$7
  PORT=$((PORT+1))
  ssh "$DST" "source /opt/phase0/env.sh; nohup $TEST -d $DEV -x \$GID -a $EXTRA -F -p $PORT \
      > /opt/phase0/runs/${RUNID}_server.log 2>&1 &"
  sleep 2
  ssh "$SRC" "source /opt/phase0/env.sh; /opt/phase0/capture_local.sh $RUNID -- \
      $TEST -d $DEV -x \$GID -a $EXTRA -F -p $PORT $IP"
  sleep 1
}

echo "=== Part 1: all pairs x both rails — bandwidth + latency sweeps ==="
for pair in "${PAIRS[@]}"; do
  set -- $pair; SRC=$1; DST=$2
  # rail A
  run_test $SRC $DST $DEVA ${IPA[$DST]} ib_write_bw  "--report_gbits" pt_wbw_${SRC}${DST}_rA
  run_test $SRC $DST $DEVA ${IPA[$DST]} ib_write_lat ""               pt_wlat_${SRC}${DST}_rA
  run_test $SRC $DST $DEVA ${IPA[$DST]} ib_send_lat  ""               pt_slat_${SRC}${DST}_rA
  # rail B
  run_test $SRC $DST $DEVB ${IPB[$DST]} ib_write_bw  "--report_gbits" pt_wbw_${SRC}${DST}_rB
  run_test $SRC $DST $DEVB ${IPB[$DST]} ib_write_lat ""               pt_wlat_${SRC}${DST}_rB
  run_test $SRC $DST $DEVB ${IPB[$DST]} ib_send_lat  ""               pt_slat_${SRC}${DST}_rB
done

echo "=== Part 2: deep dive node1 -> node2 ==="
SRC=node1; DST=node2

echo "--- dual-rail aggregate: both devices concurrently, 1 MB, 20 s ---"
P1=$((PORT+1)); P2=$((PORT+2)); PORT=$((PORT+2))
ssh "$DST" "source /opt/phase0/env.sh; \
  nohup ib_write_bw -d \$DEVA -x \$GID -s 1048576 -D 20 --report_gbits -F -p $P1 >/dev/null 2>&1 & \
  nohup ib_write_bw -d \$DEVB -x \$GID -s 1048576 -D 20 --report_gbits -F -p $P2 >/dev/null 2>&1 &"
sleep 2
ssh "$SRC" "source /opt/phase0/env.sh; /opt/phase0/capture_local.sh pt_dual_n1n2 -- bash -c '
  ib_write_bw -d \$DEVA -x \$GID -s 1048576 -D 20 --report_gbits -F -p $P1 ${IPA[$DST]} \
      > /opt/phase0/runs/pt_dual_n1n2/railA.txt &
  ib_write_bw -d \$DEVB -x \$GID -s 1048576 -D 20 --report_gbits -F -p $P2 ${IPB[$DST]} \
      > /opt/phase0/runs/pt_dual_n1n2/railB.txt &
  wait'"

echo "--- 4 QPs, rail A, 1 MB ---"
PORT=$((PORT+1))
ssh "$DST" "source /opt/phase0/env.sh; nohup ib_write_bw -d \$DEVA -x \$GID -s 1048576 -q 4 -D 20 \
    --report_gbits -F -p $PORT >/dev/null 2>&1 &"
sleep 2
ssh "$SRC" "source /opt/phase0/env.sh; /opt/phase0/capture_local.sh pt_q4_n1n2 -- \
    ib_write_bw -d \$DEVA -x \$GID -s 1048576 -q 4 -D 20 --report_gbits -F -p $PORT ${IPA[$DST]}"

echo "Experiment B complete. Results in /opt/phase0/runs/ on the SOURCE nodes."
