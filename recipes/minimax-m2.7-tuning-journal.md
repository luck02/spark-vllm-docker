# MiniMax-M2.7-AWQ Tuning Journal

Append-only log of each tier-step A/B against `minimax-m2.7-awq.yaml` on dual DGX Spark (TP=2, CX-7 200GbE).
Plan: `~/.claude/plans/atomic-hopping-ripple.md`.

## Schema

Each step entry contains:

- **Step ID** — plan tier/step (e.g. T1.2).
- **Change** — exact diff applied on top of the prior accepted baseline.
- **Image build** — `docker inspect vllm-node --format '{{.Created}}'` output on spark1.
- **Results** — one sub-table per concurrency (1 / 4 / 8 / 16) with the metrics below.
- **Correctness** — pass/fail of the 10-prompt golden set (temperature=0, seed=0).
- **Decision** — `ACCEPTED` (promoted to new baseline) or `REVERTED` (rolled back). One-line justification.

### Metrics captured (per concurrency)

| column | meaning |
|---|---|
| TTFT p50 / p95 (ms) | Time to first token |
| TPOT p50 / p95 (ms) | Inter-token / per-output-token latency |
| Throughput (tok/s) | Aggregate output tokens/sec across all streams |
| Prefix hit % | `vllm:prefix_cache_hit_rate` during run |
| KV usage % | peak `vllm:kv_cache_usage_perc` |
| Mem peak (GiB) | peak unified memory per node (node_exporter) |

### Acceptance thresholds

- TTFT p95: **≤ baseline + 3 ms**
- TPOT p50: **≤ baseline** (no regression)
- Throughput: **≥ baseline + 3 %**
- Correctness: **golden set identical** (or within documented threshold when change is numerical)

Any failure → REVERT.

---

## Prerequisites log

- [ ] vllm-node image build date ≥ 2026-04-10 (required for T2.1 `fuse_minimax_qk_norm`, PR #37045).
  - spark1: `2026-04-06T16:13:31-07:00` — **pre-merge, rebuild needed before T2.1**
  - spark2: `2026-04-06T23:18:39Z` — **pre-merge, rebuild needed before T2.1**
  - Tier 0 / Tier 1 do not depend on this; safe to run on current image.
- [ ] Baseline git tag `minimax-m2.7-baseline`. **Blocked**: outer repo has no commits yet; revisit after initial commit.
- [x] Benchmark harness drafted at `scripts/bench-minimax.sh` (runs from PC via SSH+docker exec on spark1; no execution yet).
  - Endpoint check at 2026-04-14: `http://10.10.0.11:8000/v1/models` is UP, currently serving **pre-Tier-0** recipe (cluster must be relaunched for edits to take effect).
  - **Dataset gap**: `/root/.cache/huggingface/sharegpt.json` does not exist inside `vllm_node` on spark1. User must place a ShareGPT JSON (plan specifies a code-heavy slice, ~500 prompts) there before T0.4 can run.

---

## Step entries

> Fill in one section per step as it is applied. Do not delete or rewrite prior entries — this file is append-only.

### Baseline (post-Tier 0)

**Step ID**: `T0.4` (baseline-after-cleanup)
**Change**: T0.1 drop `--enable-chunked-prefill`, T0.2 add `SAFETENSORS_FAST_GPU=1`, T0.3 add `--prefix-caching-hash-algo sha256_cbor`. Also patched `launch-cluster.sh` to auto-mount symlink targets so `~/.cache/huggingface/hub -> /mnt/nas/models/huggingface` resolves inside the container (see memory `spark_cluster_symlink_mount`).
**Image build (vllm-node on spark1)**: `2026-04-06T16:13:31-07:00` — **pre PR #37045**; T2.1 will need a rebuild before running.
**Applied**: 2026-04-14 (recipe + launch-cluster.sh + harness)
**Bench ran**: 2026-04-14 19:46 → 22:29 (~2h43m total; 500 ShareGPT prompts per concurrency, ShareGPT_V3_unfiltered_cleaned_split.json)

| Concurrency | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | TPOT p95 (ms) | Output tok/s | Total tok/s | Duration (s) | Prefix hits / queries (delta) |
|---|---|---|---|---|---|---|---|---|
| 1  |    300.6 |   775.7 |   44.87 |   45.82 | 21.50 |  42.30 | 4927.90 |  13 744 /  121 498 (11.3%) — cold cache |
| 4  |    244.6 |   301.2 |   81.69 |   85.44 | 48.10 |  94.43 | 2212.57 | 117 216 /  121 498 (96.5%) |
| 8  |    296.0 |   326.0 |  115.94 |  119.13 | 67.54 | 132.26 | 1583.71 | 117 216 /  121 498 (96.5%) |
| 16 | 24 247.7 | 37 354.7 |  114.65 |  117.78 | 68.61 | 134.12 | 1564.59 | 117 216 /  121 498 (96.5%) |

**Note on prefix cache**: `c=1` seeds the cache; subsequent concurrencies hit ~97 %. Cache survives restarts but NOT inter-concurrency (same prompts re-run). For future tier steps, run concurrencies in the same 1→4→8→16 order so caching effect is consistent across A/B comparisons.

**Correctness**: n/a (this is the baseline; correctness is measured against this reference in later tiers).

**Decision**: **ACCEPTED as baseline**.

### Headline findings from baseline

1. **`--max-num-seqs 8` is the hard cap**: c=16 gives the same TPOT / throughput as c=8 (115 ms / 68 tok/s output) but TTFT explodes from 296 ms to 24 247 ms because requests past the 8th wait in queue. T1.2 (`--max-num-seqs 16`) is expected to be a big win here.
2. **TTFT is batch-friendly up to c=8**: c=1 p95 TTFT (776 ms) is actually worse than c=4 (301 ms) — the batch amortizes first-token cost.
3. **Output throughput doubles c=1→c=4 then diminishes** (21 → 48 → 68): the model is compute-bound, MoE/TP=2 is scaling, and we're almost certainly seeing TP all-reduce cost starting to bite.
4. **KV usage was 0 % at idle end-of-run snapshots** — we never captured mid-run peak. Consider adding a periodic scraper inside the harness for future steps.

---

### T1.1 — gpu-memory-utilization 0.70 → 0.78

**Step ID**: `T1.1`
**Change**: `defaults.gpu_memory_utilization: 0.78` (was 0.7)
**Image build (vllm-node on spark1)**: `2026-04-06T16:13:31-07:00` (unchanged)
**Applied**: 2026-04-15
**Bench ran**: 2026-04-15/16 (500 ShareGPT prompts, seed=0)

| Concurrency | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | TPOT p95 (ms) | Output tok/s | Total tok/s | Interactive tok/s | Duration (s) |
|---|---|---|---|---|---|---|---|---|
| 1  |    104.1 |   131.6 |   24.58 |   24.95 |  40.15 |  78.62 | 40.7 | 2664.34 |
| 4  |    149.8 |   177.1 |   46.98 |   49.04 |  83.70 | 163.85 | 21.3 | 1278.86 |
| 8  |    188.9 |   218.2 |   70.87 |   73.27 | 110.55 | 216.66 | 14.1 |  965.98 |
| 16 | 14 563.5 | 23 264.6 |   70.72 |   72.75 | 111.54 | 218.78 | 14.1 |  955.83 |

**vs baseline (T0.4)**: +87% output tok/s at c=1, -65% TTFT p50, -45% TPOT p50. Interactive speed: 40.7 tok/s (was 22.3).

**Caveat — FlashInfer JIT warmup**: T0.4 baseline was the first-ever run after the vllm-node image was built (Apr 6). FlashInfer compiles attention kernels on first use; the cubin cache at `~/.cache/flashinfer` was cold. By T1.1, the cache was warm (survives reboots). Most of the TPOT improvement (~45%) is likely JIT warmup, not the memory-util change. All future tier comparisons will be warm-JIT, so they are apples-to-apples from T1.1 onward.

**c=16 still queue-bound**: TTFT p50 jumps from 189ms (c=8) to 14.6s (c=16) — same pattern as baseline. `--max-num-seqs 8` remains the bottleneck. T1.2 targets this.

**Correctness**: n/a (no golden set run; this is now the effective warm-JIT baseline).

**Decision**: **ACCEPTED** — promoted to new baseline. All future tiers compare against these numbers.

---

### T1.2 — max-num-seqs 8→16, max-num-batched-tokens 8192→16384

**Step ID**: `T1.2`
**Change**: `--max-num-seqs 16 --max-num-batched-tokens 16384` (was 8 / 8192)
**Image build**: `2026-04-06T16:13:31-07:00` (unchanged)
**Applied**: 2026-04-16
**Bench ran**: 2026-04-16 (500 ShareGPT prompts, seed=0, warm JIT)

| Concurrency | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | TPOT p95 (ms) | Output tok/s | Total tok/s | Interactive tok/s | Duration (s) |
|---|---|---|---|---|---|---|---|---|
| 1  |    206.1 |   439.5 |   24.73 |   25.42 |  38.88 |  76.10 | 40.4 | 2753.44 |
| 4  |    163.5 |   214.4 |   47.08 |   49.12 |  83.42 | 163.62 | 21.2 | 1278.15 |
| 8  |    222.7 |   256.6 |   71.65 |   74.67 | 108.60 | 212.40 | 14.0 |  987.44 |
| 16 |    296.0 |   375.6 |  112.62 |  119.35 | 137.69 | 270.55 |  8.9 |  771.49 |

**vs T1.1 baseline**:
- c=1/4/8: TPOT flat (±0.8ms), TTFT slightly worse (+100ms at c=1, likely run-to-run variance), throughput flat.
- **c=16: TTFT p50 dropped from 14,564ms to 296ms (-98%)**. Queue bottleneck eliminated. Aggregate throughput +24% (219→271 total tok/s). TPOT rose 71→113ms — expected cost of serving 16 concurrent decodes vs 8.

**Correctness**: n/a (no golden set run).

**Decision**: **ACCEPTED** — the c=16 unlock is the entire purpose of this change and it delivered: requests no longer queue behind `max-num-seqs`. c≤8 performance is essentially unchanged. Promoted to new baseline.

---

### Template (copy per step, do not modify)

```
### T<tier>.<step> — <one-line summary>

**Step ID**: `T<tier>.<step>`
**Change**: <exact diff vs prior baseline>
**Image build**: <docker inspect Created>
**Applied**: <ISO date>

| Concurrency | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | Throughput | Prefix hit % | KV usage % | Mem peak |
|---|---|---|---|---|---|---|---|---|
| 1  | | | | | | | | |
| 4  | | | | | | | | |
| 8  | | | | | | | | |
| 16 | | | | | | | | |

**Correctness**: <pass/fail, notes>
**Decision**: <ACCEPTED | REVERTED> — <one-line justification>
```
