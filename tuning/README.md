# MiniMax-M2.7-AWQ Tuning Sweep Harness

Self-contained Python harness that runs the remaining tier-step optimization
experiments (T1.3 through T4.4) on dual DGX Spark unattended. Replaces the
manual claude-orchestrated workflow.

## Files

| File | Role |
|------|------|
| `sweep.py` | Main harness (preflight, smoke-test, dry-run, real sweep) |
| `experiments.yaml` | 11 experiment definitions (recipe deltas + thresholds + timeouts) |
| `baseline.json` | Current accepted baseline metrics (seeded from T1.2) |
| `run.sh` | nohup wrapper — start the live sweep |
| `install.sh` | Deploy from PC to spark1 + run remote preflight |
| `README.md` | This file |
| `tests/` | Unit tests (pytest) for the pure-function pieces |
| `tests/fixtures/` | Real T1.2 recipe and bench output for testing |
| `results/` | Per-experiment output (created at runtime) |
| `status.json` | Live progress (poll for monitoring) |
| `sweep.log` | Symlink to most recent run log |
| `.sweep.lock` | PID lockfile (auto-managed) |
| `.baseline-recipe.yaml` | Snapshot of baseline (signal-restore target) |
| `intervention.flag` | Write `pause`/`abort`/`skip`/`resume` to control |

## Validation pipeline

Run these in order. Stop at any failure.

```bash
# 1. Unit tests (PC or spark1)
cd spark-vllm-docker/tuning
uv run --with pyyaml --with pytest pytest tests/ -v
# OR if you have pytest+pyyaml installed: python3 -m pytest tests/ -v

# 2. Preflight (PC for code sanity; rerun on spark1 after deploy for env sanity)
python3 sweep.py --preflight

# 3. Deploy to spark1 (runs remote preflight automatically)
./install.sh

# 4. SSH to spark1 and run the safety checks there
ssh 10.10.0.11
cd ~/dev/spark-vllm-docker/tuning
python3 sweep.py --dry-run         # see what each experiment would do
python3 sweep.py --smoke-test      # full pipeline, ~10 min

# 5. When you're satisfied, go live:
./run.sh
```

## Modes

| Command | Time | What it does |
|---------|------|--------------|
| `python3 sweep.py --preflight` | ~10 sec | Static checks; safe anywhere |
| `python3 sweep.py --dry-run` | ~5 sec | Render each experiment's recipe, no cluster touch |
| `python3 sweep.py --smoke-test` | ~10–15 min | Full pipeline with empty delta + 5 prompts at c=1 |
| `python3 sweep.py --only T1.3` | ~2.5 h | One real experiment |
| `python3 sweep.py` | ~25 h | All 10 RUNable experiments (T2.1 may skip on old image) |
| `python3 sweep.py --resume` | varies | Skip experiments already completed in status.json |
| `python3 sweep.py --seed-baseline DIR` | <1 sec | Rebuild baseline.json from a bench-results dir |
| `./run.sh [args]` | bg | Same as above, under nohup, surviving SSH disconnect |

## Monitoring (from anywhere)

```bash
# Live status
ssh 10.10.0.11 'cat ~/dev/spark-vllm-docker/tuning/status.json | python3 -m json.tool'

# Tail the log
ssh 10.10.0.11 'tail -f ~/dev/spark-vllm-docker/tuning/sweep.log'

# Heartbeat freshness — if 'heartbeat' is older than ~60 seconds, the sweep is stuck/dead
ssh 10.10.0.11 'jq .heartbeat ~/dev/spark-vllm-docker/tuning/status.json'
```

## Intervention

Drop a single word into `intervention.flag`. The harness checks before each
experiment and obeys.

```bash
# Pause after current experiment
ssh 10.10.0.11 'echo pause > ~/dev/spark-vllm-docker/tuning/intervention.flag'

# Resume from pause
ssh 10.10.0.11 'echo resume > ~/dev/spark-vllm-docker/tuning/intervention.flag'

# Skip the next experiment, then continue
ssh 10.10.0.11 'echo skip > ~/dev/spark-vllm-docker/tuning/intervention.flag'

# Abort the sweep cleanly (signal handler restores baseline recipe)
ssh 10.10.0.11 'echo abort > ~/dev/spark-vllm-docker/tuning/intervention.flag'
```

For a hard-stop emergency:

```bash
# Find PID, send SIGTERM (signal handler does the cleanup)
ssh 10.10.0.11 'kill $(cat ~/dev/spark-vllm-docker/tuning/.sweep.lock)'
```

## What gets modified outside the tuning directory

The harness writes to `../recipes/minimax-m2.7-awq.yaml` during each
experiment (it has to — that's the recipe `run-recipe.sh` reads). It always
restores the baseline copy (`.baseline-recipe.yaml`) on REJECT, FAILED,
REVIEW, signal-trap, or abort.

Nothing else outside `tuning/` is touched.

## Acceptance / rejection rules

Per-concurrency comparison vs current baseline. An experiment is:

- **ACCEPTED** if any concurrency shows ≥3% throughput win AND no concurrency
  has TPOT regression > 5ms or throughput drop > 2%. Becomes new baseline.
- **REJECTED** if any concurrency has TPOT regression > 5ms OR throughput
  drop > 2%. Baseline restored.
- **REVIEW** if all changes are within the noise band (no clear win, no
  regression). Baseline restored — promote manually if you want it.
- **FAILED** if cluster won't load or bench crashes after `max_retries` (2)
  attempts. Diagnostics + docker logs saved to results dir.
- **SKIPPED** if a dependency wasn't accepted, image is too old, or
  intervention=skip.

Thresholds live in `experiments.yaml`.

## Reliability features

| Risk | Mitigation |
|------|------------|
| Concurrent sweeps | PID lockfile (`.sweep.lock`) |
| Crash leaves cluster broken | SIGTERM/INT/HUP handlers restore baseline recipe + stop cluster |
| Bench process hangs | Watchdog: kills bench if output file hasn't grown for 600s |
| Bench wall-clock blows up | Hard timeout per concurrency (7200s) |
| Stale `vllm_node` container blocks docker run | Pre-stop sweep: `docker rm -f` any stale container |
| Disk fills up | Per-experiment gate: refuse if <5 GiB free |
| Hardware partial failure | Circuit breaker: 3 consecutive FAILED → abort sweep |
| status.json frozen, sweep dead/alive ambiguous | Heartbeat thread updates every 30s |
| Recipe template syntax error mid-sweep | Preflight renders every experiment's recipe |
| Bench dataset missing | Preflight checks |
| JSON-valued flags break str.format | apply_delta auto-escapes braces |

## Resuming after a crash

```bash
ssh 10.10.0.11
cd ~/dev/spark-vllm-docker/tuning

# Verify nothing weird is left running
docker ps -a | grep vllm_node
# If so:
docker rm -f vllm_node

# Verify the active recipe is the last accepted baseline
cat ../recipes/minimax-m2.7-awq.yaml | grep -E "max-num-seqs|gpu_memory"

# Resume — picks up after the last terminal-decision experiment
./run.sh --resume
```

## Adding a new experiment

Edit `experiments.yaml`:

```yaml
experiments:
  - id: T5.7              # any unique id
    label: my-new-thing   # short slug, becomes the results/ directory name
    description: "What this is testing"
    recipe_delta:
      command_append: ["--my-flag value"]
      command_replace: {"--existing-flag": "--existing-flag new-value"}
      env_add: {MY_ENV: "1"}
    depends_on: T1.3                  # optional
    requires_image_after: "2026-05-01"  # optional ISO date
```

Re-run preflight to validate (it renders every delta).

## Why the harness lives separately from `bench-minimax.sh`

`bench-minimax.sh` runs from the PC and SSHes to spark1 — fragile
across SSH timeouts and Claude outages. The Python harness runs **on**
spark1 and uses `docker exec` locally, eliminating that whole class of
failure. It also adds the lifecycle/retry/comparison layer that the bash
script doesn't have.

## Updating baseline manually

```bash
# After a manual win you want to promote to baseline:
cd ~/dev/spark-vllm-docker/tuning
python3 sweep.py --seed-baseline ../../bench-results/<dir-with-c1.out-etc>
```
