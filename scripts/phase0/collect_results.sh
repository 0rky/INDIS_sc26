#!/bin/bash
# collect_results.sh — pull /opt/phase0/runs from every node to the launch node.
# Run AFTER experiments finish; never during a measurement (file copies would
# ride rail A and pollute counters).
set -u
DEST=~/phase0_results
mkdir -p "$DEST"

for n in node1 node2 node3 node4; do
  echo "=== collecting from $n ==="
  mkdir -p "$DEST/$n"
  rsync -a "$n:/opt/phase0/runs/" "$DEST/$n/runs/"
  rsync -a "$n:/opt/phase0/versions_*.txt" "$DEST/$n/" 2>/dev/null || true
done

echo "Done. Everything under $DEST/"
echo "Reminder: also save the SNMP CSV from the polling host and the RouterOS"
echo "'/interface ethernet print stats' captures from the switch."
