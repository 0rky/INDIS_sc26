#!/bin/bash
# serve_model.sh — launch vLLM inside the node1 container (Ray cluster must be
# up via start_ray.sh). RUN THIS ON node1 (run_matrix.sh does it over ssh).
#
# Usage: serve_model.sh <MODEL_PATH> <TP> <PP> <EP:0|1> [MAX_LEN] [LABEL]
# Env overrides:
#   GPU_MEM_UTIL   fraction of unified memory for vLLM (default 0.80).
#                  GB10 shares ONE 128GB pool with the OS + container runtime,
#                  so ~10GB is already gone before vLLM starts. Values above
#                  ~0.88 typically fail with "Free memory ... is less than
#                  desired GPU memory utilization".
#   EXTRA_ARGS     extra vLLM flags appended verbatim.
#
# Robustness:
#  1) CORE FLAGS ARE ALWAYS PASSED. Optional/renamed flags are probed against
#     `vllm serve --help`. If the help probe fails we say so LOUDLY rather than
#     silently dropping flags (which previously caused vLLM to fall back to its
#     own defaults, e.g. gpu-memory-utilization=0.92, and crash).
#  2) FAST FAIL: if the vllm process dies we stop immediately and print the log.
set -u
source /opt/phase1/phase1_env.sh
MODEL=$1; TP=$2; PP=$3; EP=$4; MAXLEN=${5:-8192}; LABEL=${6:-run}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}

mkdir -p $P1/server_logs
LOG=$P1/server_logs/server_${LABEL}.log

docker exec $CONTAINER pkill -f "vllm serve" 2>/dev/null; sleep 5
: > "$LOG"

# ---- PRECHECK: is the GPU actually visible inside the container? ----
# Containers can silently lose their NVIDIA device cgroup (e.g. after a
# systemctl daemon-reload), after which vLLM dies with an unhelpful
# "Failed to infer device type" pydantic traceback. Catch it here instead.
if ! docker exec $CONTAINER nvidia-smi -L >/dev/null 2>&1; then
  echo "FATAL: no GPU visible inside container '$CONTAINER' on $(hostname -s)."
  echo "       docker exec $CONTAINER nvidia-smi -L   ->"
  docker exec $CONTAINER nvidia-smi -L 2>&1 | head -5
  echo
  echo "       The container lost its NVIDIA devices (common after a systemd"
  echo "       daemon-reload). Fix from the launch node:"
  echo "           cd /opt/phase1 && ./stop_ray.sh && ./start_ray.sh"
  exit 1
fi

# ---- probe the CLI (advisory only; core flags are passed regardless) ----
HELP=$(docker exec $CONTAINER vllm serve --help 2>&1 || true)
HELP_OK=0
printf '%s' "$HELP" | grep -q -- "--tensor-parallel-size" && HELP_OK=1
has() { [ $HELP_OK -eq 1 ] && printf '%s' "$HELP" | grep -q -- "$1"; }

if [ $HELP_OK -eq 0 ]; then
  echo "WARNING: could not read 'vllm serve --help' in the container."
  echo "         Passing the standard flag set unprobed. If the server rejects"
  echo "         a flag, check the CLI manually:"
  echo "           docker exec $CONTAINER vllm serve --help | less"
fi

# ---- CORE flags: always passed, never dropped ----
ARGS="--host 0.0.0.0 --port $SERVE_PORT"
ARGS="$ARGS --tensor-parallel-size $TP --pipeline-parallel-size $PP"
ARGS="$ARGS --distributed-executor-backend ray"
ARGS="$ARGS --gpu-memory-utilization $GPU_MEM_UTIL"
ARGS="$ARGS --max-model-len $MAXLEN"

# ---- OPTIONAL / version-dependent flags ----
if [ "$EP" = "1" ]; then
  if has "--enable-expert-parallel"; then
    ARGS="$ARGS --enable-expert-parallel"
  elif [ $HELP_OK -eq 0 ]; then
    ARGS="$ARGS --enable-expert-parallel"     # assume present; will error loudly
  else
    echo "WARNING: this build has no --enable-expert-parallel."
    echo "         MoE would run under plain TP (WRONG for the EP experiment)."
    echo "         Find the new spelling:  docker exec $CONTAINER vllm serve --help | grep -i expert"
  fi
fi
# request logging: old CLI had --disable-log-requests; new CLI is quiet by default
has "--disable-log-requests" && ARGS="$ARGS --disable-log-requests"
ARGS="$ARGS ${EXTRA_ARGS:-}"

echo "launching in container: $MODEL"
echo "  flags: $ARGS"
echo "  log:   $LOG"
docker exec -d $CONTAINER bash -lc "source /opt/phase1/phase1_env.sh && \
  NCCL_DEBUG=INFO vllm serve '$MODEL' $ARGS > '$LOG' 2>&1"

sleep 5
echo "waiting for /health (multi-node weight loading can take many minutes)..."
for i in $(seq 1 240); do
  if curl -sf http://localhost:$SERVE_PORT/health >/dev/null 2>&1; then
    echo "server healthy after ~$((i*10))s"
    grep -m2 -E "NCCL INFO.*via NET" "$LOG" || echo "(no transport line yet; check $LOG later)"
    exit 0
  fi
  if ! docker exec $CONTAINER pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo
    echo "=== vLLM PROCESS EXITED after ~$((i*10))s — last 40 log lines ==="
    tail -40 "$LOG"
    echo
    echo "=== common fixes ==="
    echo " * 'Free memory ... less than desired GPU memory utilization'"
    echo "     -> GPU_MEM_UTIL=0.75 $0 $* (unified memory: OS+runtime take ~10GB)"
    echo " * 'unrecognized arguments: --X'  -> CLI drift; check vllm serve --help"
    echo " * 'No available memory for the cache blocks' -> lower MAX_LEN"
    echo "=== full log: $LOG ==="
    exit 1
  fi
  sleep 10
done

echo "TIMEOUT: server never became healthy — last 40 log lines:"
tail -40 "$LOG"
exit 1
