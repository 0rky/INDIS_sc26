#!/bin/bash
# start_ray.sh — start one persistent vLLM container per node, then form the
# Ray cluster inside them via docker exec. Run from the launch node.
#
# KEY DESIGN: the container's main process is `sleep infinity`, NOT ray. If ray
# fails to start, the container STAYS ALIVE so you can inspect it
# (docker exec p1vllm bash). This is the difference between a debuggable
# cluster and a container that vanishes with its error message.
#
# Transport MODE (dual|single|tcp) is baked in at container start:
#   ./stop_ray.sh && MODE=single ./start_ray.sh
set -u
MODE=${MODE:-dual}
IMAGE=${IMAGE:-vllm-node}
HEAD_IP=192.168.100.1
NODES=(node1 node2 node3 node4)

# --- 1) launch the containers (bare shell; --entrypoint bash defeats any
#        ENTRYPOINT in the image that would swallow our command) ---
for n in "${NODES[@]}"; do
  # compute the rail-A IP ON THE HOST: the image may not ship iproute2
  IP=$(ssh "$n" "ip -4 -o addr show enp1s0f0np0 | awk '{print \$4}' | cut -d/ -f1")
  if [ -z "$IP" ]; then
    echo "ERROR: could not determine rail A IP on $n"; exit 1
  fi
  echo "=== $n : container (rail A IP $IP, MODE=$MODE) ==="
  ssh "$n" "docker rm -f p1vllm >/dev/null 2>&1 || true
    docker run -d --name p1vllm --entrypoint bash \
      --network host --ipc host --gpus all \
      --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
      -v /opt/models:/opt/models -v /opt/phase1:/opt/phase1 \
      -e MODE=$MODE -e VLLM_HOST_IP=$IP \
      $IMAGE -c 'sleep infinity'" || { echo "docker run failed on $n"; exit 1; }
done

sleep 3

# --- 2) verify every container is actually alive before touching Ray ---
echo "=== container health ==="
BAD=0
for n in "${NODES[@]}"; do
  ST=$(ssh "$n" "docker inspect -f '{{.State.Running}}' p1vllm 2>/dev/null")
  echo "  $n running=$ST"
  [ "$ST" = "true" ] || { BAD=1; echo "    --- last log lines ---";
                          ssh "$n" "docker logs p1vllm 2>&1 | tail -20"; }
done
[ $BAD -eq 0 ] || { echo "ABORT: a container is not running (see logs above)"; exit 1; }

# --- 3) sanity: GPU + ray visible inside the container on EVERY node ---
echo "=== in-container sanity (all nodes) ==="
GPUBAD=0
for n in "${NODES[@]}"; do
  printf "  %-7s " "$n"
  OUT=$(ssh "$n" "docker exec p1vllm bash -lc '
    nvidia-smi -L 2>&1 | head -1
    python3 -c \"import torch;print(\\\"cuda_ok\\\", torch.cuda.is_available())\" 2>&1 | tail -1'")
  echo "$OUT" | tr '\n' ' '; echo
  echo "$OUT" | grep -q "cuda_ok True" || GPUBAD=1
done
if [ $GPUBAD -eq 1 ]; then
  echo
  echo "ABORT: a container cannot see the GPU. This is usually a lost NVIDIA"
  echo "device cgroup (e.g. after systemctl daemon-reload). Re-run:"
  echo "    ./stop_ray.sh && ./start_ray.sh"
  echo "If it persists, check the toolkit on that node:"
  echo "    docker run --rm --gpus all \$IMAGE nvidia-smi -L"
  exit 1
fi
ssh node1 "docker exec p1vllm bash -lc '
  command -v ray >/dev/null && echo \"ray: \$(ray --version 2>&1 | head -1)\" || echo \"ERROR: ray not on PATH in image\"'"

# --- 4) start Ray head, then workers ---
echo "=== ray head on node1 ==="
ssh node1 "docker exec p1vllm bash -lc '
  source /opt/phase1/phase1_env.sh
  ray stop --force >/dev/null 2>&1
  ray start --head --port=6379 --node-ip-address=\$VLLM_HOST_IP \
    --object-store-memory=4000000000 \
    --disable-usage-stats --include-dashboard=false'" \
  || { echo "ray head FAILED — inspect with: ssh node1 docker exec -it p1vllm bash"; exit 1; }

sleep 5
for n in node2 node3 node4; do
  echo "=== ray worker on $n ==="
  ssh "$n" "docker exec p1vllm bash -lc '
    source /opt/phase1/phase1_env.sh
    ray stop --force >/dev/null 2>&1
    ray start --address=$HEAD_IP:6379 --node-ip-address=\$VLLM_HOST_IP \
      --object-store-memory=4000000000 \
      --disable-usage-stats'" \
    || echo "  WARNING: worker $n failed to join (continuing; check ray status)"
done

sleep 5
echo "=== cluster status ==="
ssh node1 "docker exec p1vllm ray status"
echo
echo "Expect 4 nodes / 4 GPUs. Containers stay alive even if Ray failed:"
echo "  ssh nodeX docker exec -it p1vllm bash    # interactive debugging"
