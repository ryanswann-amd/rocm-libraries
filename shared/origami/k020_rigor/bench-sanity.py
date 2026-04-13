#!/usr/bin/env python3
"""
bench-sanity.py — Benchmark measurement sanity checker.

Catches stale-flag bugs, warm-state artifacts, and measurement errors by
comparing single-iteration vs multi-iteration timing.

Usage:
  python3 tools/bench-sanity.py check <benchmark_cmd> [--warmup=N] [--iters=N] [--threshold=0.20]
  python3 tools/bench-sanity.py validate <value1> <value2> [--threshold=0.20]
  python3 tools/bench-sanity.py report <results_json>

Examples:
  # Run a benchmark with sanity checking
  python3 tools/bench-sanity.py check "python3 bench.py --shape=4096x4096" --warmup=5 --iters=50

  # Validate two measurements against each other
  python3 tools/bench-sanity.py validate 17.3 25.1 --threshold=0.20

  # Check a results.json for suspicious entries
  python3 tools/bench-sanity.py report dashboards/iris/data/results.json

Exit codes:
  0 = measurements look sane
  1 = suspicious measurement detected (>threshold difference)
  2 = usage error
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def validate_pair(val1, val2, threshold=0.20, label=""):
    """Check if two measurements differ by more than threshold."""
    if val1 == 0 and val2 == 0:
        return True, "Both zero"
    if val1 == 0 or val2 == 0:
        return False, f"One value is zero: {val1} vs {val2}"

    ratio = abs(val1 - val2) / max(abs(val1), abs(val2))
    status = ratio <= threshold

    prefix = f"[{label}] " if label else ""
    if status:
        msg = f"{prefix}OK: {val1:.4f} vs {val2:.4f} (diff={ratio:.1%}, threshold={threshold:.0%})"
    else:
        msg = (
            f"{prefix}WARNING: {val1:.4f} vs {val2:.4f} "
            f"(diff={ratio:.1%} > threshold={threshold:.0%})\n"
            f"  Possible causes:\n"
            f"  - Stale benchmark flags not reset between iterations\n"
            f"  - Warm-state artifact (cache/TLB warm from prior run)\n"
            f"  - GPU clock throttling during multi-iteration run\n"
            f"  - Background GPU workload interference\n"
            f"  Action: Re-run with GPU idle check (rocm-smi --showuse < 5%)"
        )

    return status, msg


def check_benchmark(cmd, warmup=5, iters=50, threshold=0.20):
    """Run a benchmark in single-iter and multi-iter modes, compare results."""
    print(f"──────────────────────────────────────────────────────────────────")
    print(f"  BENCHMARK SANITY CHECK")
    print(f"──────────────────────────────────────────────────────────────────")
    print(f"  Command:   {cmd}")
    print(f"  Warmup:    {warmup}")
    print(f"  Iters:     {iters}")
    print(f"  Threshold: {threshold:.0%}")
    print(f"──────────────────────────────────────────────────────────────────")

    # Run single iteration
    print("\n  Running single iteration...", flush=True)
    single_cmd = f"{cmd} --warmup={warmup} --iters=1"
    t1 = time.time()
    r1 = subprocess.run(single_cmd, shell=True, capture_output=True, text=True, timeout=300)
    t1 = time.time() - t1

    # Run multi iteration
    print(f"  Running {iters} iterations...", flush=True)
    multi_cmd = f"{cmd} --warmup={warmup} --iters={iters}"
    t2 = time.time()
    r2 = subprocess.run(multi_cmd, shell=True, capture_output=True, text=True, timeout=600)
    t2 = time.time() - t2

    # Parse timing from output (look for common patterns)
    def extract_timing(output):
        """Extract timing value from benchmark output."""
        patterns = [
            r'(\d+\.?\d*)\s*ms',           # N ms
            r'(\d+\.?\d*)\s*us',            # N us (convert)
            r'time[:\s=]+(\d+\.?\d*)',       # time: N or time=N
            r'latency[:\s=]+(\d+\.?\d*)',    # latency: N
            r'(\d+\.?\d*)\s*TFLOPS',        # N TFLOPS
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 'us' in pattern:
                    val /= 1000  # convert us to ms
                return val
        return None

    single_val = extract_timing(r1.stdout) or extract_timing(r1.stderr)
    multi_val = extract_timing(r2.stdout) or extract_timing(r2.stderr)

    if single_val is None:
        print(f"\n  ERROR: Could not parse timing from single-iteration output")
        print(f"  stdout: {r1.stdout[:200]}")
        return False
    if multi_val is None:
        print(f"\n  ERROR: Could not parse timing from multi-iteration output")
        print(f"  stdout: {r2.stdout[:200]}")
        return False

    passed, msg = validate_pair(single_val, multi_val, threshold, "iter1 vs mean")
    print(f"\n  {msg}")
    print(f"──────────────────────────────────────────────────────────────────")

    return passed


def validate_command(val1, val2, threshold):
    """Validate two values from command line."""
    v1, v2 = float(val1), float(val2)
    passed, msg = validate_pair(v1, v2, threshold)
    print(msg)
    return passed


def report_results(results_file, threshold=0.20):
    """Scan a results.json for suspicious entries."""
    with open(results_file) as f:
        data = json.load(f)

    print(f"──────────────────────────────────────────────────────────────────")
    print(f"  RESULTS SANITY REPORT: {results_file}")
    print(f"──────────────────────────────────────────────────────────────────")

    issues = []

    # Handle different data formats
    experiments = data if isinstance(data, list) else data.get("experiments", [])

    for i, entry in enumerate(experiments):
        # Check for missing metadata (self-documenting rule)
        missing = []
        for field in ["timestamp", "hostname", "gpu_model", "git_commit"]:
            if field not in entry:
                missing.append(field)
        if missing:
            issues.append(f"  Entry {i}: Missing metadata: {', '.join(missing)}")

        # Check for suspiciously identical consecutive values
        if i > 0:
            prev = experiments[i - 1]
            for key in ["value", "metric_value", "tflops", "time_ms", "latency"]:
                if key in entry and key in prev:
                    if entry[key] == prev[key] and entry[key] != 0:
                        issues.append(
                            f"  Entry {i}: Identical {key}={entry[key]} as previous entry "
                            f"(copy-paste or stale cache?)"
                        )

    if issues:
        print(f"  Found {len(issues)} issue(s):")
        for issue in issues:
            print(issue)
        print(f"──────────────────────────────────────────────────────────────────")
        return False
    else:
        print(f"  All {len(experiments)} entries look sane.")
        print(f"──────────────────────────────────────────────────────────────────")
        return True


def main():
    parser = argparse.ArgumentParser(description="Benchmark sanity checker")
    sub = parser.add_subparsers(dest="command")

    # check subcommand
    p_check = sub.add_parser("check", help="Run and compare single vs multi iteration")
    p_check.add_argument("benchmark_cmd", help="Benchmark command to run")
    p_check.add_argument("--warmup", type=int, default=5)
    p_check.add_argument("--iters", type=int, default=50)
    p_check.add_argument("--threshold", type=float, default=0.20)

    # validate subcommand
    p_val = sub.add_parser("validate", help="Compare two values")
    p_val.add_argument("value1", type=float)
    p_val.add_argument("value2", type=float)
    p_val.add_argument("--threshold", type=float, default=0.20)

    # report subcommand
    p_rep = sub.add_parser("report", help="Scan results file for issues")
    p_rep.add_argument("results_file", help="Path to results.json")
    p_rep.add_argument("--threshold", type=float, default=0.20)

    args = parser.parse_args()

    if args.command == "check":
        ok = check_benchmark(args.benchmark_cmd, args.warmup, args.iters, args.threshold)
    elif args.command == "validate":
        ok = validate_command(args.value1, args.value2, args.threshold)
    elif args.command == "report":
        ok = report_results(args.results_file, args.threshold)
    else:
        parser.print_help()
        sys.exit(2)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
