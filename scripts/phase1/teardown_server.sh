#!/bin/bash
# teardown_server.sh — stop the running vLLM server and WAIT until GPU memory
# is actually released on every node. Leaves the Ray cluster up.
#
# Why this exists: vLLM pre-allocates its KV-cache pool, so a live server holds
# ~80% of unified memory on every TP rank. If the next config starts before the
# previous workers have died, it OOMs. run_matrix.sh calls this between configs.
#
# Usage: teardown_server.sh [free_threshold_MiB]   (default 20000 = 20 GiB used)
set -u
NODES=(node1 node2 node3 node4)
THRESH=${1:-20000}     # consider a node "released" below this MiB used

echo "=== killing vllm serve (driver on node1) ==="
ssh node1 'docker exec p1vllm pkill -f "vllm serve" 2>/dev/null' || true
sleep 10

used_mib () {  # echo integer MiB used on $1
  ssh "$1" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | head -1
}

echo "=== waiting for memory release (up to 120s) ==="
for attempt in $(seq 1 12); do
  BUSY=0
  for n in "${NODES[@]}"; do
    U=$(used_mib "$n"); U=${U:-0}
    [ "$U" -gt "$THRESH" ] && BUSY=1
  done
  [ $BUSY -eq 0 ] && break
  # after 40s, escalate: kill lingering Ray worker processes holding the pool
  if [ "$attempt" = "4" ]; then
    echo "  still held — killing lingering Ray workers on all nodes"
    for n in "${NODES[@]}"; do
      ssh "$n" 'docker exec p1vllm bash -lc "pkill -f RayWorkerProc; pkill -f VLLM::Worker; pkill -f EngineCore" 2>/dev/null' || true
    done
  fi
  sleep 10
done

echo "=== final GPU memory used (MiB) ==="
OK=1
for n in "${NODES[@]}"; do
  U=$(used_mib "$n"); U=${U:-0}
  printf "  %-7s %8s MiB" "$n" "$U"
  if [ "$U" -gt "$THRESH" ]; then echo "   <-- STILL HELD"; OK=0; else echo "   ok"; fi
done

if [ $OK -eq 0 ]; then
  echo
  echo "WARNING: memory not fully released. Options:"
  echo "  ssh nodeX 'docker exec p1vllm nvidia-smi'      # see what holds it"
  echo "  ./stop_ray.sh && ./start_ray.sh                # hard reset (safe)"
  exit 1
fi
echo "all nodes released."


