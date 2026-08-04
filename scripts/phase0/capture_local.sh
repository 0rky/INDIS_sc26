#!/bin/bash
# capture_local.sh <run_id> -- <command...>
# Wraps any command with before/after counter snapshots on THIS node.
# Results land in /opt/phase0/runs/<run_id>/ (stdout.log + counter files).
RUN=$1; shift
[ "$1" = "--" ] && shift
OUT=/opt/phase0/runs/$RUN
mkdir -p "$OUT"

/opt/phase0/snapshot.sh "$OUT" before
"$@" 2>&1 | tee "$OUT/stdout.log"
RC=${PIPESTATUS[0]}
/opt/phase0/snapshot.sh "$OUT" after

echo "$RC" > "$OUT/exit_code.txt"
exit $RC
