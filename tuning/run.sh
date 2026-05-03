#!/bin/bash
# Start the tuning sweep under nohup so it survives SSH disconnects.
# Logs to sweep.log; status visible in status.json; intervene via intervention.flag.
#
# Usage:
#   ./run.sh                 # full sweep
#   ./run.sh --resume        # skip already-completed experiments
#   ./run.sh --only T1.3     # one experiment
#
# Stops only on: clean completion, signal, or intervention=abort.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Sanity: ensure preflight passes before going live
echo "==> Preflight before launch..."
if ! python3 sweep.py --preflight; then
  echo
  echo "Preflight FAILED. Refusing to start sweep." >&2
  echo "Fix the issues above, then re-run." >&2
  exit 1
fi

# Sanity: lockfile
if [[ -f .sweep.lock ]]; then
  pid=$(cat .sweep.lock 2>/dev/null || echo "")
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Sweep already running (PID $pid). Use 'kill $pid' or write 'abort' to intervention.flag." >&2
    exit 1
  fi
fi

ts=$(date +%Y%m%d-%H%M%S)
log="sweep-${ts}.log"
ln -sfn "$log" sweep.log

echo "==> Starting sweep in background. Log: $SCRIPT_DIR/$log"
nohup python3 sweep.py "$@" > "$log" 2>&1 &
pid=$!
echo "==> PID: $pid"
echo "==> Tail logs:    tail -f $SCRIPT_DIR/$log"
echo "==> Status:       cat $SCRIPT_DIR/status.json | python3 -m json.tool"
echo "==> Pause/abort:  echo pause > $SCRIPT_DIR/intervention.flag"
echo "==> Pause/abort:  echo abort > $SCRIPT_DIR/intervention.flag"
disown $pid 2>/dev/null || true
