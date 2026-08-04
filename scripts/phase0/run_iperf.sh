#!/bin/bash
# run_iperf.sh — Experiment A (TCP envelope). Run from the launch node.
# Prereq: iperf3 servers on all nodes:  iperf3 -s -p 5201 -D ; iperf3 -s -p 5202 -D
# Rail A clients target port 5201, rail B clients target port 5202.
set -u
NODES=(node1 node2 node3 node4)
declare -A IPA=([node1]=192.168.100.1 [node2]=192.168.100.2 [node3]=192.168.100.3 [node4]=192.168.100.4)
declare -A IPB=([node1]=192.168.101.1 [node2]=192.168.101.2 [node3]=192.168.101.3 [node4]=192.168.101.4)

echo "=== Part 1: symmetry sweep — all ordered pairs x both rails, P=8, 20 s ==="
for SRC in "${NODES[@]}"; do
  for DST in "${NODES[@]}"; do
    [ "$SRC" = "$DST" ] && continue
    ssh "$SRC" "/opt/phase0/capture_local.sh iperf_${SRC}_${DST}_rA_P8 -- \
      iperf3 -c ${IPA[$DST]} -p 5201 -t 20 -P 8 -J"
    ssh "$SRC" "/opt/phase0/capture_local.sh iperf_${SRC}_${DST}_rB_P8 -- \
      iperf3 -c ${IPB[$DST]} -p 5202 -t 20 -P 8 -J"
  done
done

echo "=== Part 2: deep dive node1 -> node2 ==="
SRC=node1; DST=node2
for P in 1 4 8; do
  for RAIL in A B; do
    if [ $RAIL = A ]; then IP=${IPA[$DST]}; PORT=5201; else IP=${IPB[$DST]}; PORT=5202; fi
    for r in 1 2 3; do
      ssh "$SRC" "sar -u 1 30 > /opt/phase0/runs/iperf_dd_r${RAIL}_P${P}_rep${r}_cpu.log 2>&1 &
        /opt/phase0/capture_local.sh iperf_dd_r${RAIL}_P${P}_rep${r} -- \
        iperf3 -c $IP -p $PORT -t 30 -P $P -J"
    done
  done
done

echo "--- bidirectional, rail A ---"
ssh "$SRC" "/opt/phase0/capture_local.sh iperf_dd_rA_bidir -- \
  iperf3 -c ${IPA[$DST]} -p 5201 -t 30 -P 8 --bidir -J"

echo "--- dual-rail aggregate: both rails concurrently, P=8 each ---"
ssh "$SRC" "/opt/phase0/capture_local.sh iperf_dd_dual -- bash -c '
  iperf3 -c ${IPA[$DST]} -p 5201 -t 30 -P 8 -J > /opt/phase0/runs/iperf_dd_dual/railA.json &
  iperf3 -c ${IPB[$DST]} -p 5202 -t 30 -P 8 -J > /opt/phase0/runs/iperf_dd_dual/railB.json &
  wait'"

echo "Experiment A complete. Results in /opt/phase0/runs/ on the SOURCE nodes."
