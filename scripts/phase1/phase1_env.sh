#!/bin/bash
# /opt/phase1/phase1_env.sh — sourced on EVERY node before starting Ray or vLLM.
# Copy to /opt/phase1/ on all nodes. Edit GID / MODE as needed.

# ---- testbed constants (same as Phase 0) ----
export DEVA=rocep1s0f0
export DEVB=roceP2p1s0f0
export IFA=enp1s0f0np0            # rail A netdev, 192.168.100.0/24
export IFB=enP2p1s0f0np0          # rail B netdev, 192.168.101.0/24
export GID=3                      # RoCE v2 GID index (from Phase 0 Stage 1.7)

# ---- transport mode: dual | single | tcp ----
# Changing MODE requires a full Ray restart on all nodes (stop_ray, edit, start_ray).
export MODE=${MODE:-dual}
case "$MODE" in
  dual)   export NCCL_IB_DISABLE=0; export NCCL_IB_HCA=${DEVA}:1,${DEVB}:1 ;;
  single) export NCCL_IB_DISABLE=0; export NCCL_IB_HCA=${DEVA}:1 ;;
  tcp)    export NCCL_IB_DISABLE=1; unset NCCL_IB_HCA ;;
esac
export NCCL_IB_GID_INDEX=$GID
export NCCL_SOCKET_IFNAME=$IFA
export GLOO_SOCKET_IFNAME=$IFA     # vLLM uses gloo for CPU-side groups

# ---- this node's data-plane IP ----
# If VLLM_HOST_IP is already set (e.g. passed into the container with -e by
# start_ray.sh), keep it. Otherwise compute it from the rail A interface.
# This matters because the container image may not ship iproute2.
export VLLM_HOST_IP=${VLLM_HOST_IP:-$(ip -4 -o addr show $IFA 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}

# ---- cluster layout ----
export RAY_HEAD_IP=192.168.100.1   # node1 is the Ray head + vLLM server
export RAY_PORT=6379
export SERVE_PORT=8000

# ---- Ray on UNIFIED memory (critical for GB10) ----
# On GB10 the "GPU" memory IS host RAM. Ray's memory monitor therefore sees
# vLLM's allocation as enormous host-RAM usage and OOM-kills the worker
# ("Workers killed due to memory pressure"). Disable the monitor; vLLM already
# bounds its own usage via --gpu-memory-utilization.
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

# ---- container (eugr/spark-vllm-docker image) ----
export IMAGE=${IMAGE:-vllm-node}      # built by eugr's build-and-copy.sh
export CONTAINER=p1vllm               # persistent per-node container name

# ---- paths ----
export P1=/opt/phase1
export MODEL_ROOT=/opt/models      # weights live at the SAME path on all nodes

# ---- hygiene ----
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=INFO


