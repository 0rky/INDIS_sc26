#!/bin/bash
# run_nccl.sh — Experiment C (NCCL collectives). Run from the launch node.
# Matrix: {all_reduce,alltoall,sendrecv} x {2,4} nodes x {dual,single,tcp} x 3 reps
# Binaries: /opt/nccl-tests/build (must exist on ALL nodes at the same path).
# Hostfiles: ~/hosts4.txt (node1-4), ~/hosts2.txt (node1-2).
set -u
source /opt/phase0/env.sh

MPI="mpirun --map-by node \
  --mca btl_tcp_if_include 192.168.100.0/24 \
  --mca oob_tcp_if_include 192.168.100.0/24"

COMMON="-x NCCL_SOCKET_IFNAME=$IFA -x NCCL_IB_GID_INDEX=$GID \
  -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,NET,TUNING"

SWEEP="-b 1K -e 1G -f 2 -g 1 -w 20 -n 100"

for COLL in all_reduce_perf alltoall_perf sendrecv_perf; do
  for NP in 2 4; do
    HF=~/hosts${NP}.txt
    for MODE in dual single tcp; do
      case $MODE in
        dual)   ENV="-x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=${DEVA}:1,${DEVB}:1 $COMMON" ;;
        single) ENV="-x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=${DEVA}:1 $COMMON" ;;
        tcp)    ENV="-x NCCL_IB_DISABLE=1 $COMMON" ;;
      esac
      for r in 1 2 3; do
        RUN=nccl_${COLL}_np${NP}_${MODE}_rep${r}
        echo "=== $RUN ==="
        /opt/phase0/snap_all.sh "$RUN" before "$NP"          # also creates run dir on all nodes
        $MPI -np "$NP" -hostfile "$HF" $ENV \
          -x NCCL_DEBUG_FILE=/opt/phase0/runs/$RUN/nccl.%h.log \
          $NCCL_BIN/$COLL $SWEEP 2>&1 | tee /opt/phase0/runs/$RUN/stdout.log
        /opt/phase0/snap_all.sh "$RUN" after "$NP"
        # transport validity check — do not batch blind runs
        T=$(grep -h "via NET" /opt/phase0/runs/$RUN/nccl.*.log 2>/dev/null | sort -u | head -1)
        echo "  transport: ${T:-NOT FOUND IN LOCAL LOG (check remote logs)}"
      done
    done
  done
done

echo "Experiment C complete."
echo "Verify: dual/single runs used NET/IB, tcp runs used NET/Socket (nccl.*.log on each node)."
