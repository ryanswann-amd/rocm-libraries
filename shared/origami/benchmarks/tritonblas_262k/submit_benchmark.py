#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Tritonblas 262K benchmark — Slurm job array submission orchestrator.

Partitions the 262K-problem corpus into batches and submits them as
Slurm jobs via the slurm_gpu_run.py bridge. Includes:

  - Configurable batch size (default 1000 problems/job)
  - Wave-based submission: submits up to --max-concurrent jobs per wave,
    polls for completions, then submits next wave (covers all 262 batches)
  - Zombie dispatch prevention: checks for existing jobs, dedup logging
  - Pre-flight CSV validation: checks header and row count before submitting
  - Checkpointed resume: skips batches with existing complete results
  - Automatic wall-clock estimation
  - Kill-on-stale: detects hung jobs exceeding --stale-timeout

Usage:
    # Dry-run (no submission, just prints what would happen):
    python3 submit_benchmark.py --dry-run --problems problems_262k.csv

    # Submit first 100 problems as a test:
    python3 submit_benchmark.py --problems problems_262k.csv --total 100

    # Full 262K submission with wave-based dispatch:
    python3 submit_benchmark.py --problems problems_262k.csv --gpu mi300x

    # Resume after preemption (skips batches with existing results):
    python3 submit_benchmark.py --problems problems_262k.csv --resume
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Required CSV columns in the problems file
REQUIRED_COLUMNS = {"idx", "m", "n", "k", "batch", "a_dtype", "b_dtype",
                    "out_dtype", "a_trans", "b_trans"}

# Job registry for zombie prevention
JOB_REGISTRY = Path("job_registry.jsonl")


def _find_bridge_path():
    """Locate slurm_gpu_run.py bridge via known paths or environment."""
    # 1. Environment variable (most reliable)
    env_path = os.environ.get("SLURM_BRIDGE_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    # 2. Standard global_orchestrator location
    for base in [Path.home() / "global_orchestrator",
                 Path("/home/ryaswann/global_orchestrator")]:
        candidate = base / "tools" / "slurm_gpu_run.py"
        if candidate.exists():
            return str(candidate)
    # 3. Relative from this script (worktree layout)
    rel = Path(__file__).resolve()
    for _ in range(8):
        rel = rel.parent
        candidate = rel / "global_orchestrator" / "tools" / "slurm_gpu_run.py"
        if candidate.exists():
            return str(candidate)
    return None


class ZombieGuard:
    """Prevents duplicate/zombie job submissions with stale-job detection."""

    def __init__(self, registry_path=None):
        self.registry = registry_path or JOB_REGISTRY
        self._submitted = set()
        self._entries = []
        self._load()

    def _load(self):
        """Load previously submitted batch IDs."""
        if self.registry.exists():
            with open(self.registry, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        self._entries.append(entry)
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
        self._entries.append(entry)

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

    def get_active_job_ids(self):
        """Return slurm_job_ids for all active (submitted/running) entries."""
        return [e.get("slurm_job_id") for e in self._entries
                if e.get("status") in ("submitted", "running")
                and e.get("slurm_job_id") not in (None, "unknown")]

    def stale_sweep(self, stale_timeout_sec=7200):
        """Find and return batch_ids that have been 'submitted' too long."""
        now = time.time()
        stale = []
        for entry in self._entries:
            if entry.get("status") != "submitted":
                continue
            ts_str = entry.get("timestamp")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str).timestamp()
                if (now - ts) > stale_timeout_sec:
                    stale.append(entry)
            except (ValueError, TypeError):
                continue
        return stale


def validate_problems_csv(csv_path):
    """Pre-flight validation: check CSV exists, has required columns, row count."""
    p = Path(csv_path)
    if not p.exists():
        print(f"[ERROR] Problems CSV not found: {csv_path}")
        return -1
    with open(p, "r") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"[ERROR] Problems CSV is empty or has no header: {csv_path}")
            return -1
        header_set = set(reader.fieldnames)
        missing = REQUIRED_COLUMNS - header_set
        if missing:
            print(f"[ERROR] Problems CSV missing columns: {missing}")
            print(f"  Found: {reader.fieldnames}")
            return -1
        count = sum(1 for _ in reader)
    return count


def count_problems(csv_path):
    """Count problems in the CSV file (row count minus header)."""
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
    """Build the bench_runner.py command for one batch.

    Uses set -euo pipefail in the wrapper so sbatch surfaces failures.
    The bench_dir, problems path, results/checkpoint dirs must all exist
    on the compute node (same filesystem or pre-copied via SCP).
    """
    bench_dir = args.bench_dir
    results_path = f"{args.results_dir}/batch_{batch_id:04d}.csv"
    ckpt_path = f"{args.checkpoint_dir}/batch_{batch_id:04d}.ckpt"

    # Build command with set -euo pipefail for safety
    cmd_parts = [
        f"set -euo pipefail &&",
        f"cd {bench_dir} &&",
        f"python3 bench_runner.py",
        f"--problems {args.problems}",
        f"--output {results_path}",
        f"--checkpoint {ckpt_path}",
        f"--start-idx {start_idx}",
        f"--end-idx {end_idx}",
        f"--arch {args.arch}",
    ]
    if args.n_cu:
        cmd_parts.append(f"--n-cu {args.n_cu}")
    return " ".join(cmd_parts)


def submit_via_bridge(command, args, batch_id):
    """Submit a job via slurm_gpu_run.py bridge.

    Uses python3 explicitly (not sys.executable) since the bridge must
    run on the head node where python3 is guaranteed.
    """
    bridge_cmd = [
        "python3", str(args.bridge_path),
        command,
        "--gpu", args.gpu,
        "--timelimit", args.timelimit,
        "--task", "K-016",
        "--description", f"tritonblas-262k-batch-{batch_id:04d}",
        "--json",
    ]

    try:
        result = subprocess.run(bridge_cmd, capture_output=True, text=True,
                                timeout=120)
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Bridge submission timed out (120s)")
        return None

    if result.returncode != 0:
        stderr_snippet = result.stderr[:300] if result.stderr else "(no stderr)"
        print(f"  [ERROR] Bridge submission failed (rc={result.returncode}): "
              f"{stderr_snippet}")
        return None

    try:
        data = json.loads(result.stdout)
        return data.get("job_id") or data.get("slurm_job_id")
    except json.JSONDecodeError:
        # Try to extract job ID from output
        for line in result.stdout.splitlines():
            if "job" in line.lower() and any(c.isdigit() for c in line):
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
    serial_time = n_batches * time_per_batch_sec
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
    parser.add_argument("--bridge-path", default=None,
                        help="Path to slurm_gpu_run.py (auto-detected if omitted)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")
    parser.add_argument("--resume", action="store_true",
                        help="Skip batches with existing complete results")
    parser.add_argument("--max-concurrent", type=int, default=16,
                        help="Max concurrent Slurm jobs per wave")
    parser.add_argument("--wave-poll-interval", type=int, default=30,
                        help="Seconds between polling for wave completion")
    parser.add_argument("--stale-timeout", type=int, default=7200,
                        help="Mark jobs as stale after this many seconds (default 2h)")
    parser.add_argument("--registry", default=None,
                        help="Job registry file for zombie prevention")
    args = parser.parse_args()

    # --- Pre-flight checks ---

    # 1. Validate problems CSV
    n_validated = validate_problems_csv(args.problems)
    if n_validated < 0:
        return 1
    print(f"[PREFLIGHT] Problems CSV validated: {n_validated} rows, "
          f"all required columns present")

    # 2. Locate bridge
    if args.bridge_path is None:
        args.bridge_path = _find_bridge_path()
    if args.bridge_path is None or not Path(args.bridge_path).exists():
        print(f"[ERROR] Cannot find slurm_gpu_run.py bridge. "
              f"Set --bridge-path or SLURM_BRIDGE_PATH env var.")
        return 1
    print(f"[PREFLIGHT] Bridge: {args.bridge_path}")

    # 3. Default bench_dir
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
    print(f"[SUBMIT] Parallel (8x) estimate: "
          f"{est['parallel_8x_estimate_min']:.1f} min")

    # Setup directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Zombie guard
    registry_path = Path(args.registry) if args.registry else JOB_REGISTRY
    guard = ZombieGuard(registry_path)

    # --- Stale sweep before submitting new work ---
    stale = guard.stale_sweep(args.stale_timeout)
    if stale:
        print(f"[STALE] Found {len(stale)} stale jobs (>{args.stale_timeout}s old):")
        for s in stale:
            print(f"  batch={s['batch_id']}, job={s.get('slurm_job_id')}")
            guard.mark_stale(s["batch_id"], "stale-auto-sweep")
        print(f"[STALE] Marked {len(stale)} jobs as stale — "
              f"their batch IDs are now eligible for resubmission")

    # --- Build batch queue ---
    pending_batches = []
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
            n_duplicate += 1
            continue

        pending_batches.append((batch_id, start_idx, end_idx, batch_key))

    print(f"[SUBMIT] Pending: {len(pending_batches)} batches, "
          f"{n_skipped} complete (skipped), {n_duplicate} duplicate (skipped)")

    if not pending_batches:
        print("[SUBMIT] Nothing to submit.")
        return 0

    # --- Wave-based submission ---
    # Submit up to max_concurrent batches per wave. In dry-run mode,
    # all batches are processed at once (no polling).
    n_submitted = 0
    n_failed = 0
    wave_num = 0
    idx = 0

    while idx < len(pending_batches):
        wave_num += 1
        wave_end = min(idx + args.max_concurrent, len(pending_batches))
        wave_batches = pending_batches[idx:wave_end]
        wave_job_ids = []

        print(f"\n[WAVE {wave_num}] Submitting batches {idx}..{wave_end - 1} "
              f"({len(wave_batches)} jobs)")

        for batch_id, start_idx, end_idx, batch_key in wave_batches:
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
                    "wave": wave_num,
                })
                n_submitted += 1
                wave_job_ids.append(job_id)
                print(f"    -> Job {job_id}")
            else:
                n_failed += 1
                print(f"    -> FAILED to submit")

            # Small delay between submissions to avoid overwhelming scheduler
            time.sleep(2)

        idx = wave_end

        # If there are more batches and this isn't dry-run, poll for
        # wave completion before submitting the next wave.
        if idx < len(pending_batches) and not args.dry_run and wave_job_ids:
            print(f"[WAVE {wave_num}] Waiting for wave to complete before "
                  f"submitting next wave ({len(pending_batches) - idx} batches remaining)...")
            _wait_for_wave_results(args, wave_batches, args.wave_poll_interval)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"[SUBMIT] Summary: {n_submitted} submitted, {n_failed} failed, "
          f"{n_skipped} skipped (complete), {n_duplicate} skipped (duplicate)")
    print(f"[SUBMIT] Waves: {wave_num}")
    if args.dry_run:
        print("[SUBMIT] Dry-run complete — no jobs were submitted.")
    else:
        print(f"[SUBMIT] Results dir: {args.results_dir}")
        print(f"[SUBMIT] Registry: {registry_path}")
        print(f"[SUBMIT] Aggregate with: python3 aggregate_results.py "
              f"--results-dir {args.results_dir}")

    return 0


def _wait_for_wave_results(args, wave_batches, poll_interval):
    """Poll until at least half the wave's batches have result files.

    This allows the next wave to start before all jobs complete, keeping
    the cluster pipeline full while avoiding burst oversubscription.
    """
    batch_ids = [b[0] for b in wave_batches]
    expected_per_batch = [(b[0], b[2] - b[1]) for b in wave_batches]
    threshold = max(1, len(batch_ids) // 2)  # wait for 50% completion
    max_wait = 3600  # 1 hour max wait per wave
    waited = 0

    while waited < max_wait:
        n_done = 0
        for batch_id, expected in expected_per_batch:
            if check_batch_complete(args.results_dir, batch_id, expected):
                n_done += 1
        if n_done >= threshold:
            print(f"  [WAVE] {n_done}/{len(batch_ids)} batches complete — "
                  f"proceeding to next wave")
            return
        time.sleep(poll_interval)
        waited += poll_interval
        if waited % 120 == 0:
            print(f"  [WAVE] Still waiting... {n_done}/{len(batch_ids)} done "
                  f"(need {threshold}, {waited}s elapsed)")

    print(f"  [WAVE] Timed out after {max_wait}s — proceeding anyway")


if __name__ == "__main__":
    sys.exit(main())
