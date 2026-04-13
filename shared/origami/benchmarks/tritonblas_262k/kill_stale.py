#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Kill-on-stale logic for the 262K tritonblas benchmark.

Scans the job registry for stale jobs (submitted but no progress after a
configurable timeout) and provides options to cancel them. Also detects
duplicate submissions for the same batch.

Usage:
    # Check for stale jobs (report only):
    python3 kill_stale.py --registry job_registry.jsonl --report

    # Cancel stale jobs (jobs with no progress after 2 hours):
    python3 kill_stale.py --registry job_registry.jsonl --cancel --stale-timeout 7200

    # Detect and report duplicate batch submissions:
    python3 kill_stale.py --registry job_registry.jsonl --check-duplicates
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_registry(path):
    """Load job registry entries."""
    entries = []
    if not Path(path).exists():
        return entries
    with open(path, "r") as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return entries


def check_stale(entries, timeout_sec=7200):
    """Find jobs that have been in 'submitted' status for too long."""
    now = time.time()
    stale = []
    for entry in entries:
        if entry.get("status") != "submitted":
            continue
        ts_str = entry.get("timestamp")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str).timestamp()
            age = now - ts
            if age > timeout_sec:
                entry["_age_hours"] = round(age / 3600, 1)
                stale.append(entry)
        except (ValueError, TypeError):
            continue
    return stale


def check_duplicates(entries):
    """Find batch IDs with multiple active submissions."""
    batch_jobs = defaultdict(list)
    for entry in entries:
        if entry.get("status") in ("submitted", "running"):
            batch_jobs[entry.get("batch_id", "unknown")].append(entry)

    duplicates = {k: v for k, v in batch_jobs.items() if len(v) > 1}
    return duplicates


def check_checkpoint_progress(checkpoint_dir, batch_id):
    """Check if a batch has made any checkpoint progress."""
    ckpt_path = Path(checkpoint_dir) / f"batch_{batch_id:04d}.ckpt"
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r") as f:
                data = json.load(f)
            return data.get("last_completed_idx", -1)
        except (json.JSONDecodeError, KeyError):
            pass
    return -1


def main():
    parser = argparse.ArgumentParser(
        description="Stale job detection and cleanup for 262K benchmark")
    parser.add_argument("--registry", default="job_registry.jsonl",
                        help="Job registry file")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--stale-timeout", type=int, default=7200,
                        help="Stale timeout in seconds (default: 2 hours)")
    parser.add_argument("--report", action="store_true",
                        help="Report stale jobs without cancelling")
    parser.add_argument("--cancel", action="store_true",
                        help="Cancel stale jobs (marks in registry)")
    parser.add_argument("--check-duplicates", action="store_true",
                        help="Check for duplicate batch submissions")
    args = parser.parse_args()

    entries = load_registry(args.registry)
    if not entries:
        print("[STALE] No entries in job registry")
        return 0

    # Status summary
    from collections import Counter
    status_counts = Counter(e.get("status", "unknown") for e in entries)
    print(f"[REGISTRY] {len(entries)} entries: {dict(status_counts)}")

    # Stale check
    if args.report or args.cancel:
        stale = check_stale(entries, args.stale_timeout)
        if stale:
            print(f"\n[STALE] {len(stale)} stale jobs "
                  f"(>{args.stale_timeout / 3600:.1f}h old):")
            for s in stale:
                print(f"  batch={s['batch_id']}, job={s.get('slurm_job_id')}, "
                      f"age={s['_age_hours']}h")

            if args.cancel:
                # Mark as stale in registry (actual scancel must be done
                # via slurm_gpu_run.py bridge or manually)
                print("\n[ACTION] Marking stale jobs in registry...")
                # We need to update the registry file
                updated_entries = []
                stale_ids = {s["batch_id"] for s in stale}
                with open(args.registry, "r") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if (entry.get("batch_id") in stale_ids
                                    and entry.get("status") == "submitted"):
                                entry["status"] = "stale"
                                entry["stale_at"] = datetime.now().isoformat()
                                job_id = entry.get("slurm_job_id")
                                if job_id and job_id != "unknown":
                                    print(f"  -> Cancel job {job_id}: "
                                          f"python3 tools/slurm_gpu_run.py "
                                          f"cancel {job_id}")
                            updated_entries.append(entry)
                        except json.JSONDecodeError:
                            continue
                with open(args.registry, "w") as f:
                    for entry in updated_entries:
                        f.write(json.dumps(entry) + "\n")
                print(f"[ACTION] {len(stale)} jobs marked as stale")
        else:
            print(f"[STALE] No stale jobs found "
                  f"(threshold: {args.stale_timeout / 3600:.1f}h)")

    # Duplicate check
    if args.check_duplicates:
        dups = check_duplicates(entries)
        if dups:
            print(f"\n[DUPLICATE] {len(dups)} batch IDs have multiple "
                  f"active submissions:")
            for batch_id, jobs in dups.items():
                job_ids = [j.get("slurm_job_id") for j in jobs]
                print(f"  batch={batch_id}: jobs={job_ids}")
                print(f"    -> Keep newest, cancel older submissions")
        else:
            print("[DUPLICATE] No duplicate submissions found")

    return 0


if __name__ == "__main__":
    sys.exit(main())
