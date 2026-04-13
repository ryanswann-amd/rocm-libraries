# Tritonblas 262K Benchmark Infrastructure

Slurm infrastructure for running the full 262K-problem tritonblas benchmark on supported AMD Instinct GPUs (gfx942, gfx950). Designed for the benchmarking team — supply the origami install path and config, this handles the rest.

## Quick Start

```bash
# 1. Generate the 262K problem corpus
python3 generate_problem_set.py --output problems_262k.csv

# 2. Dry-run to validate batching (no Slurm submission)
python3 submit_benchmark.py --problems problems_262k.csv --dry-run

# 3. Test with 100-problem subset
python3 submit_benchmark.py --problems problems_262k.csv --total 100 --gpu mi300x

# 4. Full 262K run (wave-based, handles all 262 batches)
python3 submit_benchmark.py --problems problems_262k.csv --gpu mi300x

# 5. Resume after preemption (skips completed batches)
python3 submit_benchmark.py --problems problems_262k.csv --resume --gpu mi300x

# 6. Aggregate results
python3 aggregate_results.py --results-dir results/ --output combined_results.csv
```

## Components

| Script | Purpose |
|---|---|
| `generate_problem_set.py` | Generates the 262K problem corpus CSV |
| `bench_runner.py` | Per-batch runner with checkpointing (runs inside Slurm job) |
| `submit_benchmark.py` | Wave-based batch submission via slurm_gpu_run.py bridge |
| `aggregate_results.py` | Merges batch results, computes statistics and regret |
| `kill_stale.py` | Zombie detection, stale job cleanup, and scancel integration |

## Architecture

```
262K problems ──split──> 262 batches (1000 each)
                              │
                    ┌─────────┴─────────┐
                    │ Wave-based submit  │  (max-concurrent per wave, polls before next)
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  slurm_gpu_run.py  │  (bridge: routing, health, rate limiting)
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │   bench_runner.py  │  (per-batch, SIGTERM checkpoint, resume)
                    └─────────┬─────────┘
                              │
              results/batch_XXXX.csv + checkpoints/
                              │
                    ┌─────────┴─────────┐
                    │ aggregate_results  │  (merge, dedup, stats, regret)
                    └───────────────────┘
```

## Key Features

- **Wave-based submission**: Submits `--max-concurrent` jobs per wave (default 16), polls for 50% completion, then fires next wave. Covers all 262 batches without manual re-invocation.
- **Pre-flight validation**: Checks CSV exists, has required columns (idx, m, n, k, batch, a_dtype, etc.), and reports row count before any submission.
- **Checkpointing**: Each batch writes progress every 100 problems. On preemption (SIGTERM), flushes immediately. Resume picks up from last checkpoint.
- **Zombie prevention**: Job registry tracks all submissions with dedup. Stale sweep auto-detects jobs stuck beyond `--stale-timeout` (default 2h).
- **Kill-on-stale**: `kill_stale.py --cancel` invokes `scancel` for hung jobs and marks them in registry.
- **set -euo pipefail**: All sbatch wrapper commands include pipefail so failures are surfaced.
- **python3**: All commands use `python3` (never bare `python`) for remote node compatibility.

## Wall-clock Estimate

At ~500 problems/sec (origami analytical model, no GPU kernel launch):
- **Serial**: ~8.7 min for full 262K
- **Parallel (8 jobs)**: ~1.1 min

With Slurm scheduling overhead (queue wait, node allocation): estimate 30-60 min total.

## Environment Requirements

The origami Python wheel must be installed in the venv on compute nodes:
```bash
source /scratch/users/$USER/origami-bench-venv/bin/activate
pip install origami-*.whl
```
