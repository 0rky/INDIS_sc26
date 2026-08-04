#!/bin/bash
# stop_ray.sh — remove the vLLM containers (kills Ray + any server) on all nodes.
for n in node1 node2 node3 node4; do
  echo "=== stopping container on $n ==="
  ssh $n "docker rm -f p1vllm 2>/dev/null" || true
done
