#!/usr/bin/env python3
"""
MiniMax-M2.7-AWQ Tuning Sweep Harness (v2 — hardened)

Runs directly on spark1. Reads experiments from experiments.yaml, applies each
recipe delta, launches the cluster, benchmarks, compares to baseline, and
accepts/rejects. Writes status.json for remote monitoring.

Modes:
    python3 sweep.py --preflight       # static checks, no cluster touch (~10s)
    python3 sweep.py --dry-run         # show what each experiment would do (~5s)
    python3 sweep.py --smoke-test      # full pipeline, 5 prompts, ~10 min
    python3 sweep.py --only T1.3       # one real experiment (~2.5h)
    python3 sweep.py                   # full sweep (~24-30h)
    python3 sweep.py --resume          # skip already-completed experiments
    python3 sweep.py --seed-baseline DIR  # rebuild baseline.json from bench results
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
RECIPE_PATH = REPO_DIR / "recipes" / "minimax-m2.7-awq.yaml"
RESULTS_DIR = SCRIPT_DIR / "results"
STATUS_PATH = SCRIPT_DIR / "status.json"
BASELINE_PATH = SCRIPT_DIR / "baseline.json"
INTERVENTION_FLAG = SCRIPT_DIR / "intervention.flag"
SWEEP_LOG = SCRIPT_DIR / "sweep.log"
EXPERIMENTS_PATH = SCRIPT_DIR / "experiments.yaml"
LOCKFILE = SCRIPT_DIR / ".sweep.lock"
SAVED_BASELINE_RECIPE = SCRIPT_DIR / ".baseline-recipe.yaml"

ENDPOINT_URL = "http://localhost:8000/v1/models"
METRICS_URL = "http://localhost:8000/metrics"
CONTAINER_NAME = "vllm_node"
IMAGE_NAME = "vllm-node"

DEFAULT_THRESHOLDS = {
    "tpot_p50_max_regression_ms": 5.0,
    "throughput_min_improvement_pct": -2.0,
    "throughput_clear_win_pct": 3.0,
}

CIRCUIT_BREAKER_CONSECUTIVE_FAILURES = 3
HEARTBEAT_INTERVAL_S = 30
WATCHDOG_NO_PROGRESS_S = 600
DISK_SPACE_MIN_GIB = 5
DISK_SPACE_PREFLIGHT_GIB = 20

log = logging.getLogger("sweep")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Clear existing handlers to allow re-init in tests
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    SWEEP_LOG.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(SWEEP_LOG, mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Lockfile (PID-based)
# ---------------------------------------------------------------------------

def acquire_lock() -> None:
    """Acquire the sweep lock. Exit if another sweep is alive; remove stale locks."""
    if LOCKFILE.exists():
        try:
            pid = int(LOCKFILE.read_text().strip())
            os.kill(pid, 0)  # signal 0 = check alive
            log.error(f"Another sweep is running (PID {pid}). Refusing to start.")
            log.error(f"If certain it's dead, delete {LOCKFILE}")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            log.warning(f"Stale lockfile (PID not alive). Removing.")
            LOCKFILE.unlink()
        except PermissionError:
            log.error(f"Lockfile owned by PID {pid}, signal-check denied. Refusing to start.")
            sys.exit(1)
    LOCKFILE.write_text(str(os.getpid()) + "\n")


def release_lock() -> None:
    if LOCKFILE.exists():
        try:
            content = LOCKFILE.read_text().strip()
            if content == str(os.getpid()):
                LOCKFILE.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Atomic file writes
# ---------------------------------------------------------------------------

def save_json_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.rename(path)


def save_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Status file (heartbeat-aware)
# ---------------------------------------------------------------------------

_status: dict[str, Any] = {}
_status_lock = threading.Lock()
_heartbeat_stop = threading.Event()


def init_status(sweep_id: str, total: int, mode: str) -> None:
    global _status
    with _status_lock:
        _status = {
            "sweep_id": sweep_id,
            "mode": mode,
            "started": _now_iso(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "current_experiment": None,
            "current_phase": "starting",
            "current_concurrency": None,
            "current_attempt": 0,
            "experiments_total": total,
            "experiments_completed": 0,
            "experiments_accepted": 0,
            "experiments_rejected": 0,
            "experiments_review": 0,
            "experiments_failed": 0,
            "experiments_skipped": 0,
            "consecutive_failures": 0,
            "last_update": _now_iso(),
            "heartbeat": _now_iso(),
            "intervention": None,
            "results_so_far": {},
        }
    _write_status()


def update_status(exp_id: Optional[str] = None, phase: Optional[str] = None,
                  concurrency: Optional[int] = None, attempt: Optional[int] = None,
                  **extra) -> None:
    with _status_lock:
        if exp_id is not None:
            _status["current_experiment"] = exp_id
        if phase is not None:
            _status["current_phase"] = phase
        if concurrency is not None:
            _status["current_concurrency"] = concurrency
        if attempt is not None:
            _status["current_attempt"] = attempt
        _status["last_update"] = _now_iso()
        _status.update(extra)
    _write_status()


def record_result(exp_id: str, decision: str, summary: Optional[dict] = None) -> None:
    with _status_lock:
        entry: dict[str, Any] = {"decision": decision, "completed": _now_iso()}
        if summary:
            for key in ("output_tps_c1", "tpot_p50_c1", "interactive_tps_c1", "reason"):
                if key in summary:
                    entry[key] = summary[key]
        _status["results_so_far"][exp_id] = entry
        _status["experiments_completed"] += 1
        if decision == "ACCEPTED":
            _status["experiments_accepted"] += 1
            _status["consecutive_failures"] = 0
        elif decision == "REJECTED":
            _status["experiments_rejected"] += 1
            _status["consecutive_failures"] = 0
        elif decision == "REVIEW":
            _status["experiments_review"] += 1
            _status["consecutive_failures"] = 0
        elif decision == "FAILED":
            _status["experiments_failed"] += 1
            _status["consecutive_failures"] += 1
        elif decision == "SKIPPED":
            _status["experiments_skipped"] += 1
        _status["last_update"] = _now_iso()
    _write_status()


def _write_status() -> None:
    with _status_lock:
        snapshot = copy.deepcopy(_status)
    save_json_atomic(snapshot, STATUS_PATH)


def consecutive_failures() -> int:
    with _status_lock:
        return _status.get("consecutive_failures", 0)


def heartbeat_thread() -> threading.Thread:
    def run():
        while not _heartbeat_stop.wait(HEARTBEAT_INTERVAL_S):
            with _status_lock:
                _status["heartbeat"] = _now_iso()
            _write_status()
    t = threading.Thread(target=run, name="heartbeat", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------

def check_intervention() -> Optional[str]:
    """Return 'pause', 'skip', 'abort', or None."""
    if INTERVENTION_FLAG.exists():
        cmd = INTERVENTION_FLAG.read_text().strip().lower()
        if cmd in ("pause", "skip", "abort", "resume"):
            return cmd
    return None


def clear_intervention() -> None:
    if INTERVENTION_FLAG.exists():
        INTERVENTION_FLAG.unlink()


def wait_until_unpaused() -> None:
    log.info("PAUSED — write 'resume' or 'abort' to intervention.flag")
    while True:
        time.sleep(10)
        cmd = check_intervention()
        if cmd != "pause":
            log.info(f"Resuming (intervention: {cmd or 'cleared'})")
            if cmd == "resume":
                clear_intervention()
            return


# ---------------------------------------------------------------------------
# Recipe IO
# ---------------------------------------------------------------------------

def load_recipe(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def save_recipe(recipe: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(recipe, default_flow_style=False, sort_keys=False, width=200)
    save_text_atomic(text, path)


def _escape_braces(s: str) -> str:
    """Double-up curly braces so str.format() treats them as literal.

    Recipe commands use Python str.format() to substitute {port}, {host}, etc.
    Delta-added content (e.g. JSON values like '{"mode":3}') would otherwise
    be misinterpreted as format placeholders. This escape lets deltas contain
    literal braces without escaping them in YAML.
    """
    return s.replace("{", "{{").replace("}", "}}")


def apply_delta(recipe: dict, delta: dict) -> dict:
    """Apply a recipe_delta. Returns a NEW recipe (does not mutate input)."""
    r = copy.deepcopy(recipe)

    # command_append: add flags to the end of the command block.
    # Braces in the new content are escaped so str.format() treats them literally.
    for flag in delta.get("command_append", []) or []:
        cmd = r["command"].rstrip()
        if not cmd.endswith("\\"):
            cmd += " \\"
        cmd += f"\n      {_escape_braces(flag)}"
        r["command"] = cmd + "\n"

    # command_replace: find a flag prefix, swap the line containing it.
    for prefix, replacement in (delta.get("command_replace", {}) or {}).items():
        escaped = _escape_braces(replacement)
        lines = r["command"].split("\n")
        found = False
        new_lines = []
        for line in lines:
            stripped = line.strip().rstrip("\\").strip()
            if stripped.startswith(prefix):
                indent = line[: len(line) - len(line.lstrip())]
                # Preserve trailing backslash if original had one
                trailing = " \\" if line.rstrip().endswith("\\") else ""
                new_lines.append(f"{indent}{escaped}{trailing}")
                found = True
            else:
                new_lines.append(line)
        r["command"] = "\n".join(new_lines)
        if not found:
            # Append-if-missing
            cmd = r["command"].rstrip()
            if not cmd.endswith("\\"):
                cmd += " \\"
            cmd += f"\n      {escaped}"
            r["command"] = cmd + "\n"

    # env_add: merge env vars (no escaping — env values aren't format-rendered)
    for key, value in (delta.get("env_add", {}) or {}).items():
        if r.get("env") is None:
            r["env"] = {}
        r["env"][key] = value

    return r


def render_recipe_command(recipe: dict) -> str:
    """Substitute template variables in the recipe command. Raises KeyError if missing."""
    params = dict(recipe.get("defaults", {}))
    return recipe["command"].format(**params)


# ---------------------------------------------------------------------------
# Cluster lifecycle
# ---------------------------------------------------------------------------

def cleanup_stale_containers() -> None:
    """Force-remove any leftover vllm_node containers (running or exited)."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}",
             "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=15,
        )
        for line in result.stdout.strip().splitlines():
            if line:
                log.warning(f"Removing stale container: {line}")
                subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                               capture_output=True, timeout=30)
    except Exception as e:
        log.warning(f"cleanup_stale_containers failed (non-fatal): {e}")


def stop_cluster(timeout: int = 120) -> None:
    log.info("Stopping cluster...")
    try:
        subprocess.run(
            ["./launch-cluster.sh", "stop"],
            cwd=REPO_DIR, timeout=timeout,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        log.warning("Cluster stop timed out; forcing container removal")
    cleanup_stale_containers()
    time.sleep(3)


def start_cluster() -> None:
    log.info("Launching cluster...")
    result = subprocess.run(
        ["./run-recipe.sh", "minimax-m2.7-awq", "-d"],
        cwd=REPO_DIR, timeout=120,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error(f"Cluster launch failed (exit {result.returncode}):")
        log.error(result.stderr[-1000:] if result.stderr else "(no stderr)")
        raise RuntimeError("Cluster launch failed")
    log.info("Cluster launch dispatched")


def wait_for_endpoint(timeout: int = 900) -> bool:
    log.info(f"Waiting for endpoint (timeout {timeout}s)...")
    start = time.time()
    deadline = start + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(ENDPOINT_URL, timeout=3)
            if r.status == 200:
                elapsed = int(time.time() - start)
                log.info(f"Endpoint ready ({elapsed}s)")
                return True
        except Exception:
            pass
        time.sleep(15)
    log.error(f"Endpoint did not come up within {timeout}s")
    return False


def capture_docker_logs(exp_dir: Path, label: str = "final") -> None:
    out = exp_dir / f"docker-logs-{label}.txt"
    try:
        result = subprocess.run(
            ["docker", "logs", CONTAINER_NAME],
            capture_output=True, text=True, timeout=30,
        )
        out.write_text(result.stdout + "\n---STDERR---\n" + result.stderr)
    except Exception as e:
        out.write_text(f"Failed to capture docker logs: {e}\n")


def capture_diagnostics(exp_dir: Path, label: str) -> None:
    """Snapshot system state when something goes wrong."""
    out = exp_dir / f"diagnostics-{label}.txt"
    sections = []
    for name, cmd in [
        ("free", ["free", "-h"]),
        ("uptime", ["uptime"]),
        ("docker ps -a", ["docker", "ps", "-a"]),
        ("docker stats (no-stream)", ["docker", "stats", "--no-stream"]),
        ("dmesg tail", ["bash", "-c", "dmesg | tail -50"]),
    ]:
        sections.append(f"=== {name} ===")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            sections.append(r.stdout or r.stderr or "(empty)")
        except Exception as e:
            sections.append(f"(failed: {e})")
        sections.append("")
    out.write_text("\n".join(sections))


def check_image_date() -> Optional[str]:
    """Return vllm-node image creation timestamp or None."""
    try:
        r = subprocess.run(
            ["docker", "inspect", IMAGE_NAME, "--format", "{{.Created}}"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def scrape_metrics(out_path: Path) -> None:
    try:
        r = urllib.request.urlopen(METRICS_URL, timeout=5)
        out_path.write_bytes(r.read())
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------

def free_gib(path: Path) -> float:
    s = shutil.disk_usage(path)
    return s.free / (1024 ** 3)


# ---------------------------------------------------------------------------
# Bench runner
# ---------------------------------------------------------------------------

class BenchCrash(Exception):
    pass


def run_bench(bench_cfg: dict, exp_dir: Path, timeouts: dict,
              num_prompts_override: Optional[int] = None,
              concurrencies_override: Optional[list[int]] = None) -> dict:
    """Run vllm bench serve at all concurrencies. Watchdog kills bench if no progress."""
    container = bench_cfg["container"]
    num_prompts = num_prompts_override or bench_cfg["num_prompts"]
    concurrencies = concurrencies_override or bench_cfg["concurrencies"]
    bench_timeout = timeouts.get("bench_per_concurrency_s", 7200)

    results: dict[int, dict] = {}

    for c in concurrencies:
        log.info(f"  Benchmarking c={c} ({num_prompts} prompts, timeout {bench_timeout}s)")
        update_status(concurrency=c)

        scrape_metrics(exp_dir / f"metrics-c{c}-before.prom")

        cmd = (
            f"docker exec {container} vllm bench serve"
            f" --backend openai-chat"
            f" --model '{bench_cfg['model']}'"
            f" --base-url http://localhost:8000"
            f" --endpoint /v1/chat/completions"
            f" --dataset-name sharegpt"
            f" --dataset-path '{bench_cfg['dataset']}'"
            f" --num-prompts {num_prompts}"
            f" --max-concurrency {c}"
            f" --temperature {bench_cfg.get('temperature', 0)}"
            f" --seed {bench_cfg.get('seed', 0)}"
            f" --percentile-metrics ttft,tpot,itl,e2el"
            f" --metric-percentiles '50,95,99'"
        )

        out_file = exp_dir / f"c{c}.out"
        # Stream to file so the watchdog can monitor progress
        with open(out_file, "wb") as fh:
            proc = subprocess.Popen(cmd, shell=True, stdout=fh,
                                    stderr=subprocess.STDOUT)
            try:
                _wait_with_watchdog(proc, out_file, exp_dir, c, bench_timeout)
            except BenchCrash:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                raise

        if proc.returncode != 0:
            log.error(f"  Bench c={c} exited {proc.returncode}")
            raise BenchCrash(f"c={c} exited {proc.returncode}")

        scrape_metrics(exp_dir / f"metrics-c{c}-after.prom")

        text = out_file.read_text(errors="replace")
        parsed = parse_bench_output(text)
        if not parsed.get("output_tps"):
            raise BenchCrash(f"c={c} produced no parseable metrics")
        results[c] = parsed

        log.info(f"  c={c}: output_tps={parsed.get('output_tps')}, "
                 f"tpot_p50={parsed.get('tpot_p50')}ms, "
                 f"interactive={parsed.get('interactive_tps')} tok/s")

    return results


def _wait_with_watchdog(proc: subprocess.Popen, out_file: Path,
                        exp_dir: Path, concurrency: int,
                        hard_timeout: int) -> None:
    """Block until proc exits. Kill if no file growth for WATCHDOG_NO_PROGRESS_S."""
    start = time.time()
    last_size = 0
    last_growth = start
    while True:
        rc = proc.poll()
        if rc is not None:
            return
        elapsed = time.time() - start
        if elapsed > hard_timeout:
            log.error(f"  Bench c={concurrency} hit hard timeout {hard_timeout}s")
            capture_diagnostics(exp_dir, f"hard-timeout-c{concurrency}")
            raise BenchCrash(f"c={concurrency} hard timeout")
        try:
            sz = out_file.stat().st_size
            if sz > last_size:
                last_size = sz
                last_growth = time.time()
            elif time.time() - last_growth > WATCHDOG_NO_PROGRESS_S:
                log.error(f"  Bench c={concurrency} stalled (no output growth for "
                          f"{WATCHDOG_NO_PROGRESS_S}s)")
                capture_diagnostics(exp_dir, f"watchdog-c{concurrency}")
                raise BenchCrash(f"c={concurrency} watchdog stall")
        except FileNotFoundError:
            pass
        time.sleep(15)


def parse_bench_output(text: str) -> dict:
    """Extract metrics from `vllm bench serve` stdout."""
    metrics: dict[str, float] = {}
    patterns = {
        "output_tps": r"Output token throughput \(tok/s\):\s+([\d.]+)",
        "total_tps": r"Total token throughput \(tok/s\):\s+([\d.]+)",
        "peak_tps": r"Peak output token throughput \(tok/s\):\s+([\d.]+)",
        "ttft_p50": r"P50 TTFT \(ms\):\s+([\d.]+)",
        "ttft_p95": r"P95 TTFT \(ms\):\s+([\d.]+)",
        "ttft_p99": r"P99 TTFT \(ms\):\s+([\d.]+)",
        "tpot_p50": r"P50 TPOT \(ms\):\s+([\d.]+)",
        "tpot_p95": r"P95 TPOT \(ms\):\s+([\d.]+)",
        "itl_p50": r"P50 ITL \(ms\):\s+([\d.]+)",
        "itl_p95": r"P95 ITL \(ms\):\s+([\d.]+)",
        "duration_s": r"Benchmark duration \(s\):\s+([\d.]+)",
        "total_input_tokens": r"Total input tokens:\s+(\d+)",
        "total_generated_tokens": r"Total generated tokens:\s+(\d+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            metrics[key] = float(m.group(1))
    if "tpot_p50" in metrics and metrics["tpot_p50"] > 0:
        metrics["interactive_tps"] = round(1000.0 / metrics["tpot_p50"], 1)
    return metrics


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(results: dict, baseline: dict, thresholds: Optional[dict] = None) -> tuple[str, str]:
    """Return (decision, reason) where decision in {ACCEPTED, REJECTED, REVIEW}."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    tpot_max_reg = t["tpot_p50_max_regression_ms"]
    tp_min = t["throughput_min_improvement_pct"]
    tp_win = t["throughput_clear_win_pct"]

    any_clear_win = False
    details = []

    for c in sorted(results.keys()):
        if c not in baseline:
            continue
        r = results[c]
        b = baseline[c]
        tpot_delta = r.get("tpot_p50", 0) - b.get("tpot_p50", 0)
        b_tps = b.get("output_tps", 1) or 1
        tp_delta_pct = ((r.get("output_tps", 0) - b_tps) / b_tps * 100)

        details.append(f"c={c}: tpot {tpot_delta:+.1f}ms, throughput {tp_delta_pct:+.1f}%")

        if tpot_delta > tpot_max_reg:
            return "REJECTED", f"TPOT regression at c={c}: {tpot_delta:+.1f}ms > {tpot_max_reg}ms"
        if tp_delta_pct < tp_min:
            return "REJECTED", f"Throughput drop at c={c}: {tp_delta_pct:+.1f}% < {tp_min}%"
        if tp_delta_pct >= tp_win:
            any_clear_win = True

    reason = "; ".join(details)
    if any_clear_win:
        return "ACCEPTED", reason
    return "REVIEW", f"Within noise band — {reason}"


# ---------------------------------------------------------------------------
# Baseline seeding
# ---------------------------------------------------------------------------

def seed_baseline_from_bench_results(bench_dir: Path) -> dict:
    """Parse c*.out files in a bench result dir into baseline metrics."""
    baseline = {}
    for f in sorted(bench_dir.glob("c*.out")):
        m = re.match(r"c(\d+)\.out", f.name)
        if not m:
            continue
        c = int(m.group(1))
        parsed = parse_bench_output(f.read_text())
        if parsed:
            baseline[c] = parsed
    return baseline


def load_baseline(path: Path) -> dict:
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def preflight(experiments_yaml: Path = EXPERIMENTS_PATH) -> bool:
    """Run static checks. Print PASS/FAIL per check. Return True if all gating checks PASS.

    Some checks (e.g. endpoint state) are informational only — they print INFO and don't
    affect the return value.
    """
    gating_checks: list[bool] = []

    def check(name: str, ok: bool, detail: str = ""):
        sym = "PASS" if ok else "FAIL"
        msg = f"[{sym}] {name}"
        if detail:
            msg += f" — {detail}"
        log.info(msg)
        gating_checks.append(ok)
        return ok

    def check_info(name: str, ok: bool, detail: str = ""):
        sym = "OK" if ok else "INFO"
        msg = f"[{sym}] {name}"
        if detail:
            msg += f" — {detail}"
        log.info(msg)
        return ok

    # Lockfile
    if LOCKFILE.exists():
        try:
            pid = int(LOCKFILE.read_text().strip())
            os.kill(pid, 0)
            check("Lockfile not held", False, f"PID {pid} alive")
        except Exception:
            check("Lockfile not held", True, "stale lockfile present (will be cleaned)")
    else:
        check("Lockfile not held", True)

    # Layout
    expected = {
        "sweep.py": SCRIPT_DIR / "sweep.py",
        "experiments.yaml": EXPERIMENTS_PATH,
        "baseline.json": BASELINE_PATH,
    }
    missing = [name for name, path in expected.items() if not path.exists()]
    check("tuning/ layout", not missing, f"missing: {missing}" if missing else "")

    # Python version
    pyver = sys.version_info
    check("Python >= 3.10", pyver >= (3, 10), f"have {pyver.major}.{pyver.minor}")

    # PyYAML
    try:
        import yaml as _y
        check("PyYAML installed", True, _y.__version__)
    except ImportError:
        check("PyYAML installed", False, "pip install pyyaml")

    # Active recipe parses
    if RECIPE_PATH.exists():
        try:
            recipe = load_recipe(RECIPE_PATH)
            check("Active recipe parses", True)
            try:
                render_recipe_command(recipe)
                check("Active recipe template substitutes", True)
            except KeyError as e:
                check("Active recipe template substitutes", False, f"missing var: {e}")
        except Exception as e:
            check("Active recipe parses", False, str(e))
            recipe = None
    else:
        check("Active recipe present", False, str(RECIPE_PATH))
        recipe = None

    # Experiments YAML
    config = None
    if experiments_yaml.exists():
        try:
            config = yaml.safe_load(experiments_yaml.read_text())
            check("experiments.yaml parses", True,
                  f"{len(config.get('experiments', []))} experiments")
        except Exception as e:
            check("experiments.yaml parses", False, str(e))
    else:
        check("experiments.yaml present", False)

    # Baseline JSON
    if BASELINE_PATH.exists():
        try:
            baseline = load_baseline(BASELINE_PATH)
            need_keys = {"output_tps", "tpot_p50"}
            ok = all(need_keys.issubset(set(v.keys())) for v in baseline.values())
            check("baseline.json schema", ok,
                  f"{len(baseline)} concurrencies present")
        except Exception as e:
            check("baseline.json schema", False, str(e))

    # Apply each delta + render
    if recipe and config:
        all_deltas_ok = True
        for exp in config.get("experiments", []):
            try:
                modified = apply_delta(recipe, exp.get("recipe_delta", {}))
                render_recipe_command(modified)
            except Exception as e:
                check(f"Delta for {exp['id']}", False, str(e))
                all_deltas_ok = False
        if all_deltas_ok:
            check("All recipe deltas apply + substitute", True,
                  f"{len(config['experiments'])} experiments")

    # Disk space
    free = free_gib(SCRIPT_DIR)
    check(f"Free disk >= {DISK_SPACE_PREFLIGHT_GIB} GiB",
          free >= DISK_SPACE_PREFLIGHT_GIB, f"{free:.1f} GiB free")

    # launch-cluster.sh patch
    lc = REPO_DIR / "launch-cluster.sh"
    if lc.exists():
        text = lc.read_text()
        check("launch-cluster.sh patched", "auto_mount_symlink_targets" in text)
    else:
        check("launch-cluster.sh present", False)

    # Image — informational on the PC (image lives on Sparks); gating on spark1
    img_date = check_image_date()
    if img_date:
        check_info("vllm-node image present", True, img_date[:19])
    else:
        check_info("vllm-node image present", False,
                   "missing (OK during PC validation; required on spark1)")

    # Endpoint check (informational — only checked if cluster is up).
    # On the PC during local validation, the cluster is on spark1, not localhost,
    # so this is expected to be down. On spark1, this should be UP for a real go-live.
    try:
        urllib.request.urlopen(ENDPOINT_URL, timeout=2)
        check_info("Endpoint currently up", True)
        # Container running?
        r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                           capture_output=True, text=True, timeout=5)
        running = CONTAINER_NAME in r.stdout.split()
        if running:
            # Dataset present in container — IS a hard gate when we have a container
            r2 = subprocess.run(
                ["docker", "exec", CONTAINER_NAME, "test", "-f",
                 "/root/.cache/huggingface/sharegpt.json"],
                capture_output=True, timeout=10,
            )
            check("Dataset present in container", r2.returncode == 0,
                  "/root/.cache/huggingface/sharegpt.json")
            # vllm CLI in container
            r3 = subprocess.run(
                ["docker", "exec", CONTAINER_NAME, "which", "vllm"],
                capture_output=True, text=True, timeout=10,
            )
            check("vllm CLI in container", r3.returncode == 0)
    except Exception:
        check_info("Endpoint currently up", False,
                   "cluster down (OK during PC validation; required on spark1)")

    # Forecast
    if config and recipe:
        log.info("")
        log.info(f"Forecast ({len(config['experiments'])} experiments queued):")
        will_run = 0
        for exp in config["experiments"]:
            req_after = exp.get("requires_image_after")
            verdict = "WILL RUN"
            reason = ""
            if req_after and img_date:
                if img_date[:10] < req_after:
                    verdict = "WILL SKIP"
                    reason = f"image {img_date[:10]} < required {req_after}"
            if exp.get("depends_on"):
                verdict = "CONDITIONAL"
                reason = f"depends on {exp['depends_on']} ACCEPTED"
            line = f"  {exp['id']:>5} {exp.get('label',''):<32} — {verdict}"
            if reason:
                line += f" ({reason})"
            log.info(line)
            if verdict == "WILL RUN":
                will_run += 1
        log.info("")
        log.info(f"Estimated wall time: ~{will_run * 2.5:.0f} hours "
                 f"({will_run} experiments x ~2.5 h each)")

    all_ok = all(gating_checks)
    log.info("")
    log.info(f"Preflight: {'ALL PASS' if all_ok else 'FAILED'} "
             f"({sum(gating_checks)}/{len(gating_checks)} gating checks)")
    if not img_date:
        log.info("NOTE: vllm-node image and endpoint must also pass on spark1 before go-live.")
    return all_ok


# ---------------------------------------------------------------------------
# Single-experiment runner (used by sweep + smoke + --only)
# ---------------------------------------------------------------------------

def run_single_experiment(exp: dict, baseline_recipe: dict, baseline_metrics: dict,
                          bench_cfg: dict, thresholds: dict, timeouts: dict,
                          max_retries: int,
                          num_prompts_override: Optional[int] = None,
                          concurrencies_override: Optional[list[int]] = None
                          ) -> tuple[str, Optional[dict], dict]:
    """Run one experiment to completion. Returns (decision, results, summary)."""
    exp_id = exp["id"]
    exp_label = exp.get("label", exp_id)
    delta = exp.get("recipe_delta", {})

    exp_recipe = apply_delta(baseline_recipe, delta)
    exp_dir = RESULTS_DIR / f"{exp_id}-{exp_label}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_recipe(exp_recipe, exp_dir / "recipe.yaml")

    # Disk gate
    free = free_gib(SCRIPT_DIR)
    if free < DISK_SPACE_MIN_GIB:
        reason = f"disk too low: {free:.1f} GiB free < {DISK_SPACE_MIN_GIB} GiB"
        log.error(f"  {reason}")
        return "FAILED", None, {"reason": reason}

    bench_results = None
    last_error = None
    for attempt in range(max_retries + 1):
        update_status(exp_id, "stopping", attempt=attempt)
        stop_cluster(timeout=timeouts.get("cluster_stop_s", 120))

        # Write the experiment recipe as the active recipe
        save_recipe(exp_recipe, RECIPE_PATH)

        update_status(exp_id, "launching")
        try:
            start_cluster()
        except Exception as e:
            last_error = f"launch: {e}"
            capture_docker_logs(exp_dir, f"launch-fail-{attempt}")
            log.error(f"  Launch failed (attempt {attempt + 1}): {e}")
            continue

        update_status(exp_id, "loading")
        if not wait_for_endpoint(timeout=timeouts.get("model_load_s", 900)):
            last_error = "endpoint did not come up"
            capture_docker_logs(exp_dir, f"load-fail-{attempt}")
            continue

        update_status(exp_id, "benchmarking")
        try:
            bench_results = run_bench(
                bench_cfg, exp_dir, timeouts,
                num_prompts_override=num_prompts_override,
                concurrencies_override=concurrencies_override,
            )
            break
        except BenchCrash as e:
            last_error = f"bench: {e}"
            capture_docker_logs(exp_dir, f"bench-crash-{attempt}")
            log.error(f"  Bench crashed (attempt {attempt + 1}): {e}")

    if bench_results is None:
        capture_docker_logs(exp_dir, "final-fail")
        return "FAILED", None, {"reason": last_error or "all retries failed"}

    update_status(exp_id, "analyzing")
    capture_docker_logs(exp_dir, "final")
    decision, reason = compare(bench_results, baseline_metrics, thresholds)

    summary = {
        "experiment_id": exp_id,
        "label": exp_label,
        "description": exp.get("description", ""),
        "decision": decision,
        "reason": reason,
        "timestamp": _now_iso(),
        "results": {str(c): m for c, m in bench_results.items()},
        "baseline": {str(c): m for c, m in baseline_metrics.items()},
        "delta": delta,
    }
    if 1 in bench_results:
        summary["output_tps_c1"] = bench_results[1].get("output_tps")
        summary["tpot_p50_c1"] = bench_results[1].get("tpot_p50")
        summary["interactive_tps_c1"] = bench_results[1].get("interactive_tps")
    save_json_atomic(summary, exp_dir / "summary.json")

    return decision, bench_results, summary


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(bench_cfg: dict, timeouts: dict) -> bool:
    """Run a synthetic no-op experiment with 5 prompts at c=1. Returns True on success."""
    log.info("=" * 60)
    log.info("SMOKE TEST")
    log.info("=" * 60)

    baseline_recipe = load_recipe(RECIPE_PATH)
    save_recipe(baseline_recipe, SAVED_BASELINE_RECIPE)
    baseline_metrics = load_baseline(BASELINE_PATH) if BASELINE_PATH.exists() else {}

    smoke_exp = {
        "id": "SMOKE",
        "label": f"smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "description": "no-op delta, 5 prompts, c=1",
        "recipe_delta": {},
    }

    decision, results, summary = run_single_experiment(
        smoke_exp, baseline_recipe, baseline_metrics,
        bench_cfg, DEFAULT_THRESHOLDS, timeouts,
        max_retries=0,
        num_prompts_override=5,
        concurrencies_override=[1],
    )

    # Always restore baseline recipe
    save_recipe(baseline_recipe, RECIPE_PATH)

    if decision == "FAILED":
        log.error(f"SMOKE FAIL: {summary.get('reason', 'unknown')}")
        return False

    log.info("")
    log.info(f"SMOKE PASS — decision: {decision}")
    if results and 1 in results:
        r = results[1]
        log.info(f"  output_tps: {r.get('output_tps')}")
        log.info(f"  tpot_p50:   {r.get('tpot_p50')} ms")
        log.info(f"  interactive: {r.get('interactive_tps')} tok/s")
    return True


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def install_signal_handlers(get_baseline_recipe) -> None:
    def handler(sig, frame):
        log.warning(f"Received signal {sig} — restoring baseline recipe and stopping")
        try:
            br = get_baseline_recipe()
            if br:
                save_recipe(br, RECIPE_PATH)
            stop_cluster(timeout=60)
        finally:
            release_lock()
            _heartbeat_stop.set()
        sys.exit(130)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, handler)


def run_sweep(config: dict, args) -> None:
    bench_cfg = config["bench"]
    thresholds = config.get("thresholds", DEFAULT_THRESHOLDS)
    timeouts = config.get("timeouts", {})
    max_retries = config.get("max_retries", 2)

    if not BASELINE_PATH.exists():
        log.error(f"No baseline.json at {BASELINE_PATH}. Run --seed-baseline first.")
        sys.exit(1)
    baseline_metrics = load_baseline(BASELINE_PATH)

    baseline_recipe = load_recipe(RECIPE_PATH)
    save_recipe(baseline_recipe, SAVED_BASELINE_RECIPE)

    install_signal_handlers(lambda: baseline_recipe)

    accepted_ids: set[str] = set()
    completed_ids: set[str] = set()

    if args.resume and STATUS_PATH.exists():
        try:
            prev = json.loads(STATUS_PATH.read_text())
            for eid, info in prev.get("results_so_far", {}).items():
                if info.get("decision") in ("ACCEPTED", "REJECTED", "REVIEW",
                                            "FAILED", "SKIPPED"):
                    completed_ids.add(eid)
                if info.get("decision") == "ACCEPTED":
                    accepted_ids.add(eid)
        except Exception as e:
            log.warning(f"Could not load previous status.json for resume: {e}")

    experiments = config["experiments"]
    if args.only:
        experiments = [e for e in experiments if e["id"] == args.only]
        if not experiments:
            log.error(f"No experiment with id '{args.only}'")
            sys.exit(1)

    sweep_id = f"minimax-m2.7-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    init_status(sweep_id, len(experiments),
                mode=("dry-run" if args.dry_run else
                      ("only:" + args.only if args.only else "full")))
    heartbeat_thread()

    log.info(f"Starting sweep: {sweep_id}")
    log.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    log.info(f"Baseline output_tps c=1: {baseline_metrics.get(1, {}).get('output_tps', '?')}")
    log.info(f"Experiments queued: {len(experiments)}")

    for exp in experiments:
        exp_id = exp["id"]
        exp_label = exp.get("label", exp_id)

        if args.resume and exp_id in completed_ids:
            log.info(f"=== {exp_id} {exp_label} — already completed, skipping ===")
            continue

        # Circuit breaker
        if consecutive_failures() >= CIRCUIT_BREAKER_CONSECUTIVE_FAILURES:
            log.error(f"CIRCUIT BREAKER: {CIRCUIT_BREAKER_CONSECUTIVE_FAILURES} "
                      f"consecutive failures. Aborting sweep.")
            update_status(phase="aborted-circuit-breaker")
            sys.exit(2)

        # Intervention
        intervention = check_intervention()
        if intervention == "pause":
            wait_until_unpaused()
        elif intervention == "skip":
            log.info(f"SKIPPING {exp_id} per intervention")
            clear_intervention()
            record_result(exp_id, "SKIPPED", {"reason": "intervention"})
            continue
        elif intervention == "abort":
            log.info("ABORTING sweep per intervention")
            break

        log.info("=" * 60)
        log.info(f"=== {exp_id} {exp_label} ===")
        log.info("=" * 60)
        log.info(f"  {exp.get('description', '')}")

        # Dependency
        dep = exp.get("depends_on")
        if dep and dep not in accepted_ids:
            log.info(f"  SKIPPING — depends on {dep} which was not accepted")
            record_result(exp_id, "SKIPPED", {"reason": f"depends on {dep}"})
            continue

        # Image date
        req_after = exp.get("requires_image_after")
        if req_after:
            img_date = check_image_date()
            if img_date and img_date[:10] < req_after:
                log.info(f"  SKIPPING — needs image after {req_after}, have {img_date[:10]}")
                record_result(exp_id, "SKIPPED",
                              {"reason": f"image too old: {img_date[:10]} < {req_after}"})
                continue

        # Dry-run: just save the experiment recipe and stop
        if args.dry_run:
            delta = exp.get("recipe_delta", {})
            exp_recipe = apply_delta(baseline_recipe, delta)
            exp_dir = RESULTS_DIR / f"{exp_id}-{exp_label}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            save_recipe(exp_recipe, exp_dir / "recipe.yaml")
            log.info(f"  [DRY RUN] Recipe written to {exp_dir / 'recipe.yaml'}")
            log.info(f"  [DRY RUN] Delta: {json.dumps(delta)}")
            record_result(exp_id, "SKIPPED", {"reason": "dry-run"})
            continue

        # Real run
        decision, results, summary = run_single_experiment(
            exp, baseline_recipe, baseline_metrics,
            bench_cfg, thresholds, timeouts, max_retries,
        )

        log.info(f"  Decision: {decision}")
        if "reason" in summary:
            log.info(f"  Reason: {summary['reason']}")

        if decision == "ACCEPTED":
            log.info(f"  {exp_id} ACCEPTED — promoting to baseline")
            baseline_recipe = apply_delta(baseline_recipe, exp.get("recipe_delta", {}))
            save_recipe(baseline_recipe, SAVED_BASELINE_RECIPE)
            baseline_metrics = results
            save_json_atomic({str(c): m for c, m in baseline_metrics.items()}, BASELINE_PATH)
            accepted_ids.add(exp_id)
            # Active recipe stays as the experiment recipe (which IS the new baseline)
        else:
            # Restore baseline (whether REJECTED, REVIEW, or FAILED)
            log.info(f"  {exp_id} {decision} — restoring baseline recipe")
            save_recipe(baseline_recipe, RECIPE_PATH)

        record_result(exp_id, decision, summary)

    update_status(phase="completed")
    log.info("")
    log.info("=" * 60)
    log.info("SWEEP COMPLETE")
    log.info("=" * 60)
    log.info(f"  Accepted: {_status['experiments_accepted']}")
    log.info(f"  Rejected: {_status['experiments_rejected']}")
    log.info(f"  Review:   {_status['experiments_review']}")
    log.info(f"  Failed:   {_status['experiments_failed']}")
    log.info(f"  Skipped:  {_status['experiments_skipped']}")
    log.info(f"  Final baseline output_tps c=1: "
             f"{baseline_metrics.get(1, {}).get('output_tps', '?')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MiniMax-M2.7 tuning sweep harness")
    parser.add_argument("--preflight", action="store_true",
                        help="Run static checks only (no cluster touch)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Full pipeline with 5 prompts at c=1 (~10 min)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what each experiment would do, no cluster")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this experiment id (e.g. T1.3)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed experiments per status.json")
    parser.add_argument("--seed-baseline", type=str, default=None, metavar="DIR",
                        help="Rebuild baseline.json from a bench-results directory")
    args = parser.parse_args()

    setup_logging()

    # Seed-baseline mode (no lock, no sweep)
    if args.seed_baseline:
        bench_dir = Path(args.seed_baseline)
        if not bench_dir.exists():
            log.error(f"Bench dir not found: {bench_dir}")
            sys.exit(1)
        baseline = seed_baseline_from_bench_results(bench_dir)
        if not baseline:
            log.error(f"No c*.out files in {bench_dir}")
            sys.exit(1)
        save_json_atomic({str(k): v for k, v in baseline.items()}, BASELINE_PATH)
        log.info(f"Baseline seeded from {bench_dir}")
        for c, m in sorted(baseline.items()):
            log.info(f"  c={c}: output_tps={m.get('output_tps')}, "
                     f"tpot_p50={m.get('tpot_p50')}ms, "
                     f"interactive={m.get('interactive_tps')} tok/s")
        return

    # Preflight (no lock — read-only)
    if args.preflight:
        ok = preflight()
        sys.exit(0 if ok else 1)

    # All other modes need the lock
    acquire_lock()
    atexit.register(release_lock)

    # Smoke test mode
    if args.smoke_test:
        if not EXPERIMENTS_PATH.exists():
            log.error("experiments.yaml missing")
            sys.exit(1)
        config = yaml.safe_load(EXPERIMENTS_PATH.read_text())
        ok = smoke_test(config["bench"], config.get("timeouts", {}))
        sys.exit(0 if ok else 1)

    # Real sweep / dry-run / single-experiment
    if not EXPERIMENTS_PATH.exists():
        log.error(f"experiments.yaml not found at {EXPERIMENTS_PATH}")
        sys.exit(1)
    config = yaml.safe_load(EXPERIMENTS_PATH.read_text())
    run_sweep(config, args)


if __name__ == "__main__":
    main()
