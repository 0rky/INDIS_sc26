#!/bin/bash
# verify_run.sh — confirm a completed cell captured EVERYTHING the paper needs.
# Run it right after your first config finishes; don't wait until the end.
#
# Usage:
#   ./verify_run.sh <RUN_ID>                 # checks all 4 nodes over ssh
#   ./verify_run.sh <RUN_ID> --collected ~/phase1_results   # checks a collected tree
#
# Example:
#   ./verify_run.sh p1_l8_tp4_dual_decode_in128_out1024_c32
set -u
RUN=$1; shift || true
MODE_LOCAL=""; ROOT=""
if [ "${1:-}" = "--collected" ]; then MODE_LOCAL=1; ROOT=${2:-$HOME/phase1_results}; fi
NODES=(node1 node2 node3 node4)
DEVA=rocep1s0f0; DEVB=roceP2p1s0f0

pass=0; fail=0
ok()   { echo "  [ OK ] $1"; pass=$((pass+1)); }
bad()  { echo "  [FAIL] $1"; fail=$((fail+1)); }

# ---- helper: list files for this run (either via ssh or from collected tree) ----
list_files () {   # $1 = node
  if [ -n "$MODE_LOCAL" ]; then
    ls "$ROOT/$1/runs/$RUN" 2>/dev/null
  else
    ssh -n "$1" "ls /opt/phase1/runs/$RUN 2>/dev/null"
  fi
}
cat_file () {     # $1 = node, $2 = filename
  if [ -n "$MODE_LOCAL" ]; then
    cat "$ROOT/$1/runs/$RUN/$2" 2>/dev/null
  else
    ssh -n "$1" "cat /opt/phase1/runs/$RUN/$2 2>/dev/null"
  fi
}

echo "=============================================================="
echo "verifying run: $RUN"
echo "=============================================================="

# ---- 1. launch-node artifacts: manifest + client results ----
echo "[1] application-level results"
FOUND_MANIFEST=""; FOUND_CLIENT=""
for n in "${NODES[@]}"; do
  F=$(list_files "$n")
  echo "$F" | grep -q '^manifest.json$' && FOUND_MANIFEST=$n
  echo "$F" | grep -q '^client.json$'   && FOUND_CLIENT=$n
done
[ -n "$FOUND_MANIFEST" ] && ok "manifest.json (on $FOUND_MANIFEST)" || bad "manifest.json MISSING"
if [ -n "$FOUND_CLIENT" ]; then
  J=$(cat_file "$FOUND_CLIENT" client.json)
  TOK=$(printf '%s' "$J" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['aggregates'].get('total_output_tokens',0))" 2>/dev/null || echo 0)
  FAILED=$(printf '%s' "$J" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['aggregates'].get('failed',0))" 2>/dev/null || echo 0)
  ITL=$(printf '%s' "$J" | python3 -c "import sys,json;d=json.load(sys.stdin);print(round(d['aggregates'].get('itl_ms_p50') or 0,2))" 2>/dev/null || echo 0)
  if [ "${TOK:-0}" -gt 0 ] 2>/dev/null; then
    ok "client.json  (output tokens=$TOK, failed=$FAILED, ITL p50=${ITL}ms)"
  else
    bad "client.json present but has NO output tokens"
  fi
else
  bad "client.json MISSING (benchmark did not complete)"
fi

# ---- 2. per-node measurement artifacts ----
echo "[2] per-node capture (counters / GPU / network time series)"
for n in "${NODES[@]}"; do
  F=$(list_files "$n")
  if [ -z "$F" ]; then bad "$n: run directory missing entirely"; continue; fi
  MISS=""
  for pat in "_${DEVA}_counters_before.txt" "_${DEVA}_counters_after.txt" \
             "_${DEVB}_counters_before.txt" "_${DEVB}_counters_after.txt" \
             "_${DEVA}_hw_counters_before.txt" "_${DEVA}_hw_counters_after.txt"; do
    echo "$F" | grep -q -- "$pat" || MISS="$MISS $pat"
  done
  echo "$F" | grep -q "_gpu.csv" || MISS="$MISS gpu.csv"
  echo "$F" | grep -q "_net.csv" || MISS="$MISS net.csv(NEW)"
  if [ -z "$MISS" ]; then
    GN=$(cat_file "$n" "$(echo "$F" | grep -m1 '_gpu.csv')" | wc -l)
    NN=$(cat_file "$n" "$(echo "$F" | grep -m1 '_net.csv')" | wc -l)
    ok "$n: all artifacts (gpu samples=$GN, net samples=$NN)"
    [ "${GN:-0}" -lt 5 ] 2>/dev/null && bad "$n: gpu.csv has <5 samples (cell too short?)"
    [ "${NN:-0}" -lt 5 ] 2>/dev/null && bad "$n: net.csv has <5 samples (cell too short?)"
  else
    bad "$n: missing:$MISS"
  fi
done

# ---- 3. did any bytes actually move? ----
echo "[3] wire traffic sanity"
TOTAL=0
for n in "${NODES[@]}"; do
  F=$(list_files "$n"); [ -z "$F" ] && continue
  for DEV in $DEVA $DEVB; do
    B=$(cat_file "$n" "$(echo "$F" | grep -m1 "_${DEV}_counters_before.txt")" | awk '/port_xmit_data/{print $2}')
    A=$(cat_file "$n" "$(echo "$F" | grep -m1 "_${DEV}_counters_after.txt")"  | awk '/port_xmit_data/{print $2}')
    if [ -n "${A:-}" ] && [ -n "${B:-}" ]; then
      D=$(( (A - B) * 4 ))
      [ "$D" -gt 0 ] && TOTAL=$((TOTAL + D))
    fi
  done
done
MB=$(python3 -c "print(round($TOTAL/1e6,1))" 2>/dev/null || echo "?")
case "$RUN" in
  *_tp1_*) if [ "$TOTAL" -lt 100000000 ]; then
             ok "TP=1 null control: only ${MB} MB moved (expected ~0)"
           else bad "TP=1 should move ~no data but moved ${MB} MB — investigate"; fi ;;
  *)       if [ "$TOTAL" -gt 0 ]; then ok "cluster TX during cell: ${MB} MB"
           else bad "NO bytes recorded on the wire — counters or RoCE path broken"; fi ;;
esac

echo "=============================================================="
echo "PASS: $pass   FAIL: $fail"
[ "$fail" -eq 0 ] && echo "This run is complete and paper-ready." \
                  || echo "Fix the FAILs above before running the full matrix."
echo "=============================================================="
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
