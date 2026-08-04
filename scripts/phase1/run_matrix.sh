#!/bin/bash
# run_matrix.sh — unattended Phase 1 campaign driver.
#
#   ./run_matrix.sh [configs.tsv]
#
# Env:
#   MODE=dual|single|tcp   transport (must match how Ray was started)
#   GPU_MEM_UTIL=0.80      memory fraction; a failed server is retried at RETRY_MEM
#   RETRY_MEM=0.70         fallback fraction for the one automatic retry
#   FORCE=1                re-run configs whose results already exist
#   CELLS_ONLY="decode"    run only decode (or only prefill) cells
#
# Features for long runs:
#   * RESUMABLE  — configs whose cells all have client.json are skipped, so an
#                  interrupted campaign can simply be restarted.
#   * PREFLIGHT  — verifies Ray + GPU visibility on all nodes before each config.
#   * RETRY      — a server that fails to start is retried once at RETRY_MEM.
#   * TEARDOWN   — verified memory release between configs (no cascade OOMs).
#   * SUMMARY    — per-config OK/FAILED/SKIPPED table with durations at the end.
#
# Run it detached so an SSH drop doesn't kill the campaign:
#   nohup ./run_matrix.sh > /dev/null 2>&1 &      (output still goes to the log)
#   tail -f /opt/phase1/campaign_*.log
set -u
CFG=${1:-/opt/phase1/configs.tsv}
source /opt/phase1/phase1_env.sh
MODE=${MODE:-dual}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}
RETRY_MEM=${RETRY_MEM:-0.70}
FORCE=${FORCE:-0}
CELLS_ONLY=${CELLS_ONLY:-}
NODES=(node1 node2 node3 node4)

# workload  input  output  concurrency  num_prompts
CELLS=(
  "decode  128   1024  1   8"
  "decode  128   1024  8   32"
  "decode  128   1024  32  64"
  "prefill 4096  32    1   8"
  "prefill 4096  32    8   32"
)

mkdir -p "$P1/runs" "$P1/server_logs"
LOG="$P1/campaign_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date +%H:%M:%S)] $*"; }

trap 'say "INTERRUPTED — tearing down"; /opt/phase1/teardown_server.sh || true; exit 130' INT TERM

# ---------------------------------------------------------------- preflight
preflight() {
  ssh -n node1 "docker exec $CONTAINER ray status" >/dev/null 2>&1 || {
    say "FATAL: Ray cluster not responding. Run ./start_ray.sh first."; return 1; }
  for n in "${NODES[@]}"; do
    ssh -n "$n" "docker exec $CONTAINER nvidia-smi -L" >/dev/null 2>&1 || {
      say "FATAL: no GPU in container on $n. Run ./stop_ray.sh && ./start_ray.sh"; return 1; }
  done
  return 0
}

# ------------------------------------------------------------------- resume
run_id_for() { echo "p1_${1}_${MODE}_${2}_in${3}_out${4}_c${5}"; }

config_done() {   # $1 = LABEL ; true if every selected cell already has results
  local LABEL=$1 cell W IN OUT C N RID
  for cell in "${CELLS[@]}"; do
    read -r W IN OUT C N <<< "$cell"
    [ -n "$CELLS_ONLY" ] && [ "$W" != "$CELLS_ONLY" ] && continue
    RID=$(run_id_for "$LABEL" "$W" "$IN" "$OUT" "$C")
    [ -s "$P1/runs/$RID/client.json" ] || return 1
  done
  return 0
}

# --------------------------------------------------------------------- main
mapfile -t CFG_LINES < <(grep -Ev '^[[:space:]]*(#|$)' "$CFG")
TOTAL=${#CFG_LINES[@]}
say "campaign start: $TOTAL configs from $CFG"
say "  MODE=$MODE  GPU_MEM_UTIL=$GPU_MEM_UTIL  FORCE=$FORCE  log=$LOG"

preflight || exit 1

declare -a STATUS_LABEL STATUS_RESULT STATUS_MINS
IDX=0
CAMPAIGN_START=$(date +%s)

for LINE in "${CFG_LINES[@]}"; do
  read -r LABEL MODEL TP PP EP MAXLEN <<< "$LINE"
  IDX=$((IDX+1))
  echo
  say "########## [$IDX/$TOTAL] $LABEL  (TP=$TP PP=$PP EP=$EP) ##########"
  T0=$(date +%s)

  # ---- resume ----
  if [ "$FORCE" != "1" ] && config_done "$LABEL"; then
    say "already complete — skipping (FORCE=1 to redo)"
    STATUS_LABEL+=("$LABEL"); STATUS_RESULT+=("SKIPPED"); STATUS_MINS+=("0")
    continue
  fi

  # ---- weights present on every node? ----
  MISSING=""
  for n in "${NODES[@]}"; do
    ssh -n "$n" "test -d '$MODEL'" || MISSING="$MISSING $n"
  done
  if [ -n "$MISSING" ]; then
    say "SKIP: model dir missing on:$MISSING"
    STATUS_LABEL+=("$LABEL"); STATUS_RESULT+=("NO_WEIGHTS"); STATUS_MINS+=("0")
    continue
  fi

  # ---- preflight before each config (catches mid-campaign GPU loss) ----
  preflight || { say "preflight failed — aborting campaign"; break; }

  # ---- start server (one automatic retry at lower memory) ----
  SERVED=0
  for MEM in "$GPU_MEM_UTIL" "$RETRY_MEM"; do
    say "starting server at gpu-memory-utilization=$MEM"
    if ssh -n node1 "GPU_MEM_UTIL=$MEM /opt/phase1/serve_model.sh '$MODEL' $TP $PP $EP $MAXLEN $LABEL"; then
      SERVED=1; break
    fi
    say "server failed at $MEM; tearing down before retry"
    /opt/phase1/teardown_server.sh >/dev/null 2>&1 || true
  done
  if [ $SERVED -eq 0 ]; then
    say "SERVER FAILED for $LABEL — see $P1/server_logs/server_${LABEL}.log"
    STATUS_LABEL+=("$LABEL"); STATUS_RESULT+=("SERVER_FAIL"); STATUS_MINS+=("$(( ($(date +%s)-T0)/60 ))")
    /opt/phase1/teardown_server.sh >/dev/null 2>&1 || true
    continue
  fi

  # ---- workload cells ----
  export MODEL_PATH=$MODEL TP PP EP LABEL MODE
  CELL_FAIL=0
  for cell in "${CELLS[@]}"; do
    read -r W IN OUT C N <<< "$cell"
    [ -n "$CELLS_ONLY" ] && [ "$W" != "$CELLS_ONLY" ] && continue
    RID=$(run_id_for "$LABEL" "$W" "$IN" "$OUT" "$C")
    if [ "$FORCE" != "1" ] && [ -s "$P1/runs/$RID/client.json" ]; then
      say "  cell $RID already done — skipping"; continue
    fi
    say "  cell $RID"
    /opt/phase1/run_cell.sh "$RID" "$W" "$IN" "$OUT" "$C" "$N" || CELL_FAIL=$((CELL_FAIL+1))
    sleep 10
  done

  # ---- verified teardown ----
  /opt/phase1/teardown_server.sh || say "WARNING: memory not fully released"

  MINS=$(( ($(date +%s)-T0)/60 ))
  if [ $CELL_FAIL -eq 0 ]; then
    say "$LABEL COMPLETE in ${MINS}m"
    STATUS_LABEL+=("$LABEL"); STATUS_RESULT+=("OK"); STATUS_MINS+=("$MINS")
  else
    say "$LABEL finished with $CELL_FAIL failed cell(s) in ${MINS}m"
    STATUS_LABEL+=("$LABEL"); STATUS_RESULT+=("CELLS_FAILED($CELL_FAIL)"); STATUS_MINS+=("$MINS")
  fi
done

# ------------------------------------------------------------------ summary
TOTMIN=$(( ($(date +%s)-CAMPAIGN_START)/60 ))
echo
say "================ CAMPAIGN SUMMARY ($TOTMIN min total) ================"
printf "  %-18s %-20s %s\n" "CONFIG" "RESULT" "MINUTES"
for i in "${!STATUS_LABEL[@]}"; do
  printf "  %-18s %-20s %s\n" "${STATUS_LABEL[$i]}" "${STATUS_RESULT[$i]}" "${STATUS_MINS[$i]}"
done
echo
say "next steps:"
say "  ./collect_phase1.sh"
say "  python3 analyze_phase1.py --root ~/phase1_results"
say "  python3 plot_phase1.py    --root ~/phase1_results"
say "full log: $LOG"
