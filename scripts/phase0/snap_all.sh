#!/bin/bash
# snap_all.sh <run_id> <tag> [num_nodes]
# Snapshot counters on the first N nodes (default 4), in parallel, into the
# same run directory path on each node. Also creates the run dir everywhere
# (needed before mpirun so NCCL_DEBUG_FILE has a place to write).
RUN=$1; TAG=$2; NP=${3:-4}
ALL=(node1 node2 node3 node4)

for n in "${ALL[@]:0:$NP}"; do
  ssh "$n" "/opt/phase0/snapshot.sh /opt/phase0/runs/$RUN $TAG" &
done
wait
