#!/bin/bash
# setup_container.sh — build the eugr/spark-vllm-docker image on node1 and
# distribute it to all nodes. Run from the launch node. Rerun only when you
# want a newer vLLM build.
#
# Prereqs on every node (DGX OS usually ships both):
#   docker + NVIDIA Container Toolkit, and your user in the docker group.
#   NVIDIA driver 580.x (NOT 590.x — known CUDAGraph deadlock on GB10).
set -eu

echo "=== 0) prereq checks on all nodes ==="
for n in node1 node2 node3 node4; do
  ssh $n 'docker info >/dev/null || { echo "docker not usable on $(hostname)"; exit 1; }
          nvidia-smi --query-gpu=driver_version --format=csv,noheader'
done
echo "(driver must be 580.x on all nodes; 590.x deadlocks CUDAGraph capture on GB10)"

echo "=== 1) clone + build on node1 (~10 min with prebuilt SM121 wheels) ==="
ssh node1 '
  [ -d ~/spark-vllm-docker ] || git clone https://github.com/eugr/spark-vllm-docker.git ~/spark-vllm-docker
  cd ~/spark-vllm-docker && git pull
  ./build-and-copy.sh
'
# "No host specified, skipping copy" at the end of the build is normal —
# we distribute the image ourselves in the next step for determinism.

echo "=== 2) distribute image to node2-4 (large; run while testbed is idle) ==="
for n in node2 node3 node4; do
  echo "--- $n ---"
  ssh node1 "docker save vllm-node" | ssh $n "docker load"
done

echo "=== 3) verify GPU + vLLM import inside the container on every node ==="
for n in node1 node2 node3 node4; do
  echo "--- $n ---"
  ssh $n 'docker run --rm --gpus all vllm-node python3 -c "
import torch, vllm
print(\"torch\", torch.__version__, \"cuda_ok\", torch.cuda.is_available(),
      torch.cuda.get_device_name(0)); print(\"vllm\", vllm.__version__)"'
done
echo
echo "All four nodes must print cuda_ok True and IDENTICAL torch/vllm versions."
