#!/bin/bash
# run_cell.sh — execute ONE measurement cell against an already-running server.
# Brackets the benchmark client with counter snapshots (both rails, all nodes,
# reusing Phase 0's /opt/phase0/snapshot.sh) and per-node GPU logs, and writes
# a manifest describing the cell. Run from the launch node.
#
# Required env (set by run_matrix.sh): MODEL_PATH TP PP EP MODE LABEL
# Usage: run_cell.sh <RUN_ID> <WORKLOAD> <INPUT_LEN> <OUTPUT_LEN> <CONCURRENCY> <NUM_PROMPTS>
set -u
source /opt/phase1/phase1_env.sh
RUN=$1; WORKLOAD=$2; INLEN=$3; OUTLEN=$4; CONC=$5; NUM=$6
NODES=(node1 node2 node3 node4)
OUT=$P1/runs/$RUN
mkdir -p "$OUT"

# manifest first, so even a failed run is identifiable
cat > "$OUT/manifest.json" << EOF
{
  "run_id": "$RUN",
  "model_path": "$MODEL_PATH",
  "tp": $TP, "pp": $PP, "ep": $EP,
  "mode": "$MODE", "label": "$LABEL",
  "workload": "$WORKLOAD",
  "input_len": $INLEN, "output_len": $OUTLEN,
  "concurrency": $CONC, "num_prompts": $NUM,
  "start_ts": "$(date -Is)"
}
EOF

# counter snapshot BEFORE + start GPU loggers (all nodes; idle nodes read ~0)
for n in "${NODES[@]}"; do
  ssh -n "$n" "/opt/phase0/snapshot.sh $OUT before; \
            nohup /opt/phase1/gpu_log.sh $OUT >/dev/null 2>&1 & echo \$! > /tmp/gpulog_$RUN.pid; \
            nohup /opt/phase1/net_log.sh $OUT >/dev/null 2>&1 & echo \$! > /tmp/netlog_$RUN.pid" &
done
wait

# benchmark client (launch-node local; talks to node1 over rail A HTTP —
# HTTP bytes are negligible vs NCCL traffic but noted in the paper's methods)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate phase1-client
python /opt/phase1/bench_client.py \
  --base-url "http://${RAY_HEAD_IP}:${SERVE_PORT}" \
  --tokenizer "$MODEL_PATH" \
  --input-len "$INLEN" --output-len "$OUTLEN" \
  --num-prompts "$NUM" --concurrency "$CONC" --warmup 4 \
  --out "$OUT/client.json" 2>&1 | tee "$OUT/client_stdout.log"
RC=${PIPESTATUS[0]}

# stop GPU loggers + snapshot AFTER
for n in "${NODES[@]}"; do
  ssh -n "$n" "kill \$(cat /tmp/gpulog_$RUN.pid) 2>/dev/null; rm -f /tmp/gpulog_$RUN.pid; \
            kill \$(cat /tmp/netlog_$RUN.pid) 2>/dev/null; rm -f /tmp/netlog_$RUN.pid; \
            /opt/phase0/snapshot.sh $OUT after" &
done
wait
echo "$RC" > "$OUT/exit_code.txt"
echo "cell $RUN done (exit $RC)"
