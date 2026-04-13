#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Tritonblas 262K benchmark — Slurm job array submission orchestrator.

Partitions the 262K-problem corpus into batches and submits them as
Slurm jobs via the slurm_gpu_run.py bridge. Includes:

  - Configurable batch size (default 1000 problems/job)
  - Zombie dispatch prevention: checks for existing jobs, dedup logging
  - Dry-run mode for validation
  - Automatic wall-clock estimation

Usage:
    # Dry-run (no submission, just prints what would happen):
    python3 submit_benchmark.py --dry-run --problems problems_262k.csv

    # Submit first 100 problems as a test:
    python3 submit_benchmark.py --problems problems_262k.csv --total 100

    # Full 262K submission:
    python3 submit_benchmark.py --problems problems_262k.csv --gpu mi300x

    # Resume after preemption (skips batches with existing results):
    python3 submit_benchmark.py --problems problems_262k.csv --resume
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Job registry for zombie prevention
JOB_REGISTRY = Path("job_registry.jsonl")


class ZombieGuard:
    """Prevents duplicate/zombie job submissions."""

    def __init__(self, registry_path=None):
        self.registry = registry_path or JOB_REGISTRY
        self._submitted = set()
        self._load()

    def _load(self):
        """Load previously submitted batch IDs."""
        if self.registry.exists():
            with open(self.registry, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("status") in ("submitted", "running"):
                            self._submitted.add(entry["batch_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue

    def is_duplicate(self, batch_id):
        """Check if this batch was already submitted."""
        return batch_id in self._submitted

    def register(self, batch_id, slurm_job_id, metadata=None):
        """Record a submitted job."""
        entry = {
            "batch_id": batch_id,
            "slurm_job_id": slurm_job_id,
            "status": "submitted",
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }
        with open(self.registry, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._submitted.add(batch_id)

    def mark_stale(self, batch_id, reason="stale"):
        """Mark a previously submitted job as stale (for kill-on-stale)."""
        entries = []
        if self.registry.exists():
            with open(self.registry, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry["batch_id"] == batch_id:
                            entry["status"] = reason
                            entry["stale_at"] = datetime.now().isoformat()
                        entries.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        continue
        with open(self.registry, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        self._submitted.discard(batch_id)


def count_problems(csv_path):
    """Count problems in the CSV file."""
    with open(csv_path, "r") as f:
        return sum(1 for _ in f) - 1  # subtract header


def check_batch_complete(results_dir, batch_id, expected_count):
    """Check if a batch already has complete results."""
    result_file = Path(results_dir) / f"batch_{batch_id:04d}.csv"
    if not result_file.exists():
        return False
    with open(result_file, "r") as f:
        actual = sum(1 for _ in f) - 1  # subtract header
    return actual >= expected_count


def build_batch_command(args, batch_id, start_idx, end_idx):
    """Build the bench_runner.py command for one batch."""
    bench_dir = args.bench_dir
    cmd_parts = [
        f"cd {bench_dir}",
        "&&",
        f"python3 bench_runner.py",
        f"--problems {args.problems}",
        f"--output {args.results_dir}/batch_{batch_id:04d}.csv",
        f"--checkpoint {args.checkpoint_dir}/batch_{batch_id:04d}.ckpt",
        f"--start-idx {start_idx}",
        f"--end-idx {end_idx}",
        f"--arch {args.arch}",
    ]
    if args.n_cu:
        cmd_parts.append(f"--n-cu {args.n_cu}")
    return " ".join(cmd_parts)


def submit_via_bridge(command, args, batch_id):
    """Submit a job via slurm_gpu_run.py bridge."""
    bridge_cmd = [
        sys.executable, str(args.bridge_path),
        command,
        "--gpu", args.gpu,
        "--timelimit", args.timelimit,
        "--task", "K-016",
        "--description", f"tritonblas-262k-batch-{batch_id:04d}",
        "--json",
    ]

    result = subprocess.run(bridge_cmd, capture_output=True, text=True,
                            timeout=120)
    if result.returncode != 0:
        print(f"  [ERROR] Bridge submission failed: {result.stderr[:200]}")
        return None

    try:
        data = json.loads(result.stdout)
        return data.get("job_id") or data.get("slurm_job_id")
    except json.JSONDecodeError:
        # Try to extract job ID from output
        for line in result.stdout.splitlines():
            if "job" in line.lower() and any(c.isdigit() for c in line):
                import re
                m = re.search(r'(\d{4,})', line)
                if m:
                    return m.group(1)
        print(f"  [WARN] Could not parse job ID from bridge output")
        return "unknown"


def estimate_wallclock(n_problems, batch_size, rate_per_sec=500):
    """Estimate total wall-clock time for the full benchmark.

    Assumes ~500 problems/sec (origami analytical model is fast, no GPU kernel
    launch needed for selection). Conservative estimate.
    """
    n_batches = (n_problems + batch_size - 1) // batch_size
    time_per_batch_sec = batch_size / rate_per_sec
    # Assume jobs run in parallel up to available nodes
    serial_time = n_batches * time_per_batch_sec
    # With ~8 concurrent jobs, parallel time is much less
    parallel_time = serial_time / 8
    return {
        "n_batches": n_batches,
        "serial_estimate_min": serial_time / 60,
        "parallel_8x_estimate_min": parallel_time / 60,
        "rate_assumption": f"{rate_per_sec} problems/sec",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Submit 262K tritonblas benchmark as Slurm job array")
    parser.add_argument("--problems", required=True,
                        help="Path to problems CSV")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Problems per Slurm job (default: 1000)")
    parser.add_argument("--total", type=int, default=-1,
                        help="Total problems to run (-1=all)")
    parser.add_argument("--gpu", default="mi300x",
                        help="GPU type for Slurm submission")
    parser.add_argument("--arch", default="gfx942",
                        help="GPU architecture for origami")
    parser.add_argument("--n-cu", type=int, default=None,
                        help="Override CU count")
    parser.add_argument("--timelimit", default="01:00:00",
                        help="Wall-clock limit per job (HH:MM:SS)")
    parser.add_argument("--bench-dir", default=None,
                        help="Path to benchmark scripts on compute node")
    parser.add_argument("--results-dir", default="results",
                        help="Directory for result CSVs")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for checkpoint files")
    parser.add_argument("--bridge-path",
                        default=str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent
                                    / "global_orchestrator" / "tools" / "slurm_gpu_run.py"),
                        help="Path to slurm_gpu_run.py")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")
    parser.add_argument("--resume", action="store_true",
                        help="Skip batches with existing complete results")
    parser.add_argument("--max-concurrent", type=int, default=16,
                        help="Max concurrent Slurm jobs")
    parser.add_argument("--registry", default=None,
                        help="Job registry file for zombie prevention")
    args = parser.parse_args()

    if args.bench_dir is None:
        args.bench_dir = str(Path(__file__).resolve().parent)

    # Count problems
    n_total = count_problems(args.problems)
    if args.total > 0:
        n_total = min(n_total, args.total)
    print(f"[SUBMIT] Problems: {n_total}")
    print(f"[SUBMIT] Batch size: {args.batch_size}")

    # Wall-clock estimate
    est = estimate_wallclock(n_total, args.batch_size)
    print(f"[SUBMIT] Batches: {est['n_batches']}")
    print(f"[SUBMIT] Serial estimate: {est['serial_estimate_min']:.1f} min")
    print(f"[SUBMIT] Parallel (8x) estimate: {est['parallel_8x_estimate_min']:.1f} min")

    # Setup directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Zombie guard
    registry_path = Path(args.registry) if args.registry else JOB_REGISTRY
    guard = ZombieGuard(registry_path)

    # Generate batches
    n_submitted = 0
    n_skipped = 0
    n_duplicate = 0

    for batch_id in range(est["n_batches"]):
        start_idx = batch_id * args.batch_size
        end_idx = min(start_idx + args.batch_size, n_total)

        # Resume check
        if args.resume and check_batch_complete(args.results_dir, batch_id,
                                                 end_idx - start_idx):
            n_skipped += 1
            continue

        # Zombie/duplicate check
        batch_key = f"262k_b{batch_id:04d}_{start_idx}_{end_idx}"
        if guard.is_duplicate(batch_key):
            print(f"  [SKIP] Batch {batch_id:04d} already submitted (zombie guard)")
            n_duplicate += 1
            continue

        # Max concurrent check
        if n_submitted >= args.max_concurrent:
            print(f"  [LIMIT] Reached max concurrent jobs ({args.max_concurrent}). "
                  f"Submit remaining batches after current jobs complete.")
            break

        cmd = build_batch_command(args, batch_id, start_idx, end_idx)

        if args.dry_run:
            print(f"  [DRY-RUN] Batch {batch_id:04d}: [{start_idx}:{end_idx}] "
                  f"({end_idx - start_idx} problems)")
            print(f"    CMD: {cmd}")
            n_submitted += 1
            continue

        print(f"  [SUBMIT] Batch {batch_id:04d}: [{start_idx}:{end_idx}] "
              f"({end_idx - start_idx} problems)")
        job_id = submit_via_bridge(cmd, args, batch_id)
        if job_id:
            guard.register(batch_key, job_id, {
                "start_idx": start_idx, "end_idx": end_idx,
                "gpu": args.gpu, "arch": args.arch,
            })
            n_submitted += 1
            print(f"    -> Job {job_id}")
        else:
            print(f"    -> FAILED to submit")

        # Small delay between submissions to avoid overwhelming scheduler
        if not args.dry_run:
            time.sleep(2)

    print(f"\n[SUBMIT] Summary: {n_submitted} submitted, "
          f"{n_skipped} skipped (complete), {n_duplicate} skipped (duplicate)")

    if args.dry_run:
        print("[SUBMIT] Dry-run complete — no jobs were submitted.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
