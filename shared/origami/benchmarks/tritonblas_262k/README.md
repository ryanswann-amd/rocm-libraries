# Tritonblas 262K Benchmark Infrastructure

Slurm infrastructure for running the full 262K-problem tritonblas benchmark on MI300X/MI355X. Designed for the benchmarking team — just supply the origami binary path and config.

## Quick Start

```bash
# 1. Generate the 262K problem corpus
python3 generate_problem_set.py --output problems_262k.csv

# 2. Dry-run to validate batching (no Slurm submission)
python3 submit_benchmark.py --problems problems_262k.csv --dry-run

# 3. Test with 100-problem subset
python3 submit_benchmark.py --problems problems_262k.csv --total 100 --gpu mi300x

# 4. Full 262K run
python3 submit_benchmark.py --problems problems_262k.csv --gpu mi300x

# 5. Aggregate results
python3 aggregate_results.py --results-dir results/ --output combined_results.csv
```

## Components

| Script | Purpose |
|---|---|
| `generate_problem_set.py` | Generates the 262K problem corpus CSV |
| `bench_runner.py` | Per-batch runner with checkpointing (runs inside Slurm job) |
| `submit_benchmark.py` | Orchestrates batch submission via slurm_gpu_run.py bridge |
| `aggregate_results.py` | Merges batch results, computes statistics and regret |
| `kill_stale.py` | Zombie detection and stale job cleanup |

## Architecture

```
262K problems ──split──> 262 batches (1000 each)
                              │
                    ┌─────────┴─────────┐
                    │  slurm_gpu_run.py  │   (bridge handles routing, health checks)
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │   bench_runner.py  │   (per-batch, with SIGTERM checkpoint)
                    └─────────┬─────────┘
                              │
              results/batch_XXXX.csv + checkpoints/
                              │
                    ┌─────────┴─────────┐
                    │ aggregate_results  │   (merge, dedup, stats, regret)
                    └───────────────────┘
```

## Key Features

- **Checkpointing**: Each batch writes progress every 100 problems. On preemption (SIGTERM), flushes immediately. Resume picks up from last checkpoint.
- **Zombie Prevention**: Job registry tracks all submissions. Duplicate batch IDs are rejected. `kill_stale.py` detects hung jobs.
- **Parameterized**: Benchmarking team only needs to supply origami install path. All arch/GPU/batch params are CLI flags.
- **Dedup on Aggregation**: If a batch reruns after preemption, the aggregator deduplicates by problem index.

## Wall-clock Estimate

At ~500 problems/sec (origami analytical model, no GPU kernel launch):
- **Serial**: ~8.7 min for full 262K
- **Parallel (8 jobs)**: ~1.1 min

With Slurm scheduling overhead (queue wait, node allocation): estimate 30-60 min total.
