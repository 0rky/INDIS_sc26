#!/bin/bash
# collect_all.sh — gather EVERY node's Phase 1 results onto this node (node2)
# and verify the collection is complete before you start analysing.
#
# Run ON node2:   ./collect_all.sh
# Output tree:    ~/phase1_results/<node>/runs/<run_id>/...
#
# Safe to re-run. Use --fresh to wipe the destination first (recommended after a
# FORCE=1 re-run, so stale files from earlier campaigns cannot linger).
set -u
DEST=${DEST:-$HOME/phase1_results}
NODES=(node1 node2 node3 node4)

if [ "${1:-}" = "--fresh" ]; then
  echo "wiping $DEST"; rm -rf "$DEST"
fi
mkdir -p "$DEST"

echo "=== copying results from all nodes -> $DEST ==="
for n in "${NODES[@]}"; do
  echo "--- $n ---"
  mkdir -p "$DEST/$n"
  # runs/ holds manifests, client.json, counters, gpu.csv, net.csv
  rsync -a --info=stats1 "$n:/opt/phase1/runs/" "$DEST/$n/runs/" || \
      echo "  WARNING: rsync failed for $n"
  # server logs (needed to confirm transport = NET/IB per config)
  rsync -a "$n:/opt/phase1/server_logs/" "$DEST/$n/server_logs/" 2>/dev/null || true
  # campaign logs (durations, retries, skips - useful for the paper's methods)
  rsync -a "$n:/opt/phase1/campaign_"*.log "$DEST/$n/" 2>/dev/null || true
done

echo
echo "=== completeness check ==="
# every cell should have: manifest.json + client.json (launch node) and
# counters + gpu.csv + net.csv on each participating node
TOTAL=0; OK=0; NONET=0
for d in "$DEST"/*/runs/*/; do
  RID=$(basename "$d")
  # count each run_id once, using whichever node has the manifest
  [ -f "$d/manifest.json" ] || continue
  TOTAL=$((TOTAL+1))
  if [ -s "$d/client.json" ]; then OK=$((OK+1)); else echo "  NO client.json: $RID"; fi
  # net.csv presence anywhere for this run
  if ! ls "$DEST"/*/runs/"$RID"/*_net.csv >/dev/null 2>&1; then
    NONET=$((NONET+1)); echo "  NO net.csv    : $RID"
  fi
done
echo
echo "  cells with manifest : $TOTAL"
echo "  cells with results  : $OK"
echo "  cells missing net   : $NONET"
echo
echo "=== transport check (must be NET/IB for MODE=dual) ==="
grep -ho "via NET/[A-Za-z]*" "$DEST"/*/server_logs/*.log 2>/dev/null | sort | uniq -c
echo
echo "next:  python3 analyze_phase1.py --root $DEST"
echo "       python3 plot_phase1.py    --root $DEST"
echo "       python3 phase1_figures.py --root $DEST --phase0 <T6_nccl_full.csv>"

