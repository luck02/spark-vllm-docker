#!/bin/bash
# Deploy the tuning harness to spark1.
#
# This script:
#   1. SCPs sweep.py, experiments.yaml, baseline.json, run.sh, README.md, tests/ to spark1
#   2. Refuses to overwrite if a sweep is currently running on spark1 (lockfile check)
#   3. Refuses to overwrite results/, status.json, sweep.log (preserves prior runs)
#   4. Runs `python3 sweep.py --preflight` on spark1 and prints the result
#   5. Does NOT start any sweep — that's a separate `./run.sh` invocation
#
# Usage:
#   ./install.sh                       # default target: gary_lucas@10.10.0.11
#   TARGET=user@host ./install.sh      # override target
#   FORCE=1 ./install.sh               # overwrite even if results/ exists (still locks out live sweep)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${TARGET:-10.10.0.11}"
REMOTE_DIR="${REMOTE_DIR:-/home/gary_lucas/dev/spark-vllm-docker/tuning}"
FORCE="${FORCE:-0}"

echo "==> Target: $TARGET"
echo "==> Remote dir: $REMOTE_DIR"

# Pre-deploy: lockfile check on remote
echo "==> Checking for live sweep on $TARGET..."
if ssh -o BatchMode=yes -o ConnectTimeout=10 "$TARGET" "test -f $REMOTE_DIR/.sweep.lock" 2>/dev/null; then
  pid=$(ssh "$TARGET" "cat $REMOTE_DIR/.sweep.lock 2>/dev/null || echo")
  if [[ -n "$pid" ]] && ssh "$TARGET" "kill -0 $pid 2>/dev/null"; then
    echo "ERROR: A sweep is currently running on $TARGET (PID $pid). Refusing to deploy." >&2
    echo "       To stop it cleanly: ssh $TARGET 'echo abort > $REMOTE_DIR/intervention.flag'" >&2
    exit 1
  fi
fi

# Check for results/ that we'd potentially clobber
if ssh "$TARGET" "test -d $REMOTE_DIR/results" 2>/dev/null; then
  count=$(ssh "$TARGET" "ls $REMOTE_DIR/results 2>/dev/null | wc -l")
  if [[ "$count" -gt 0 && "$FORCE" != "1" ]]; then
    echo "WARNING: $REMOTE_DIR/results/ exists with $count entries on $TARGET." >&2
    echo "         The deploy preserves results/, status.json, sweep.log, baseline.json." >&2
    echo "         To force overwrite of NON-DATA files only: FORCE=1 ./install.sh" >&2
  fi
fi

# Ensure remote dir exists
ssh "$TARGET" "mkdir -p $REMOTE_DIR/tests"

# Deploy code files (does NOT overwrite results/, status.json, sweep.log)
echo "==> Copying files..."
scp -p "$SCRIPT_DIR/sweep.py" "$TARGET:$REMOTE_DIR/sweep.py"
scp -p "$SCRIPT_DIR/experiments.yaml" "$TARGET:$REMOTE_DIR/experiments.yaml"
scp -p "$SCRIPT_DIR/run.sh" "$TARGET:$REMOTE_DIR/run.sh"
scp -p "$SCRIPT_DIR/README.md" "$TARGET:$REMOTE_DIR/README.md" 2>/dev/null || true
scp -rp "$SCRIPT_DIR/tests" "$TARGET:$REMOTE_DIR/"

# Only copy baseline.json if it doesn't exist on remote (preserve any updates from a prior sweep)
if ! ssh "$TARGET" "test -f $REMOTE_DIR/baseline.json" 2>/dev/null; then
  echo "==> Seeding baseline.json (none on remote)..."
  scp -p "$SCRIPT_DIR/baseline.json" "$TARGET:$REMOTE_DIR/baseline.json"
else
  echo "==> Preserving existing baseline.json on remote"
fi

ssh "$TARGET" "chmod +x $REMOTE_DIR/run.sh $REMOTE_DIR/sweep.py"

# Run preflight on remote
echo
echo "==> Running preflight on $TARGET..."
echo
if ssh "$TARGET" "cd $REMOTE_DIR && python3 sweep.py --preflight"; then
  echo
  echo "==> Deploy successful. Preflight PASSED."
  echo
  echo "Next steps (on $TARGET):"
  echo "  ssh $TARGET"
  echo "  cd $REMOTE_DIR"
  echo "  python3 sweep.py --dry-run     # see what each experiment would do"
  echo "  python3 sweep.py --smoke-test  # full pipeline, ~10 min, 5 prompts"
  echo "  ./run.sh                       # start the sweep when ready"
else
  echo
  echo "==> Deploy successful but preflight FAILED. Fix issues above before running." >&2
  exit 1
fi
