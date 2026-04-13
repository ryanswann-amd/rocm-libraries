#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Aggregate per-batch results from the 262K tritonblas benchmark into a single
dataset and compute regret analysis.

Merges all batch_XXXX.csv files into one combined dataset, computes:
  - Coverage: fraction of problems with successful selection
  - Median/p95/p99 selection time (ms)
  - LDS-pruned-to-zero count (shapes where all configs were filtered)
  - Error breakdown by type

Usage:
    python3 aggregate_results.py --results-dir results/ --output combined_results.csv

    # With regret analysis against a hipblaslt baseline:
    python3 aggregate_results.py --results-dir results/ --output combined.csv \\
        --baseline hipblaslt_baseline.csv --regret-output regret_analysis.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


def collect_batch_files(results_dir):
    """Find and sort all batch result CSV files."""
    batch_files = sorted(Path(results_dir).glob("batch_*.csv"))
    return batch_files


def merge_results(batch_files, output_path):
    """Merge all batch CSVs into a single output CSV."""
    all_rows = []
    seen_idx = set()
    fieldnames = None

    for bf in batch_files:
        with open(bf, "r") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                idx = row.get("idx")
                if idx in seen_idx:
                    continue  # dedup across checkpointed re-runs
                seen_idx.add(idx)
                all_rows.append(row)

    # Sort by index
    all_rows.sort(key=lambda r: int(r.get("idx", 0)))

    # Write merged output
    if fieldnames and all_rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    return all_rows, fieldnames


def compute_statistics(rows):
    """Compute summary statistics from merged results."""
    total = len(rows)
    if total == 0:
        return {"total": 0, "error": "no results"}

    status_counts = Counter(r.get("status", "unknown") for r in rows)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    error_rows = [r for r in rows if r.get("status") == "error"]
    no_config_rows = [r for r in rows if r.get("status") == "no_configs"]

    # Selection time stats (only for successful selections)
    sel_times = []
    for r in ok_rows:
        try:
            sel_times.append(float(r["selection_time_ms"]))
        except (KeyError, ValueError):
            pass

    sel_times.sort()
    n = len(sel_times)

    stats = {
        "total_problems": total,
        "ok": len(ok_rows),
        "errors": len(error_rows),
        "no_configs": len(no_config_rows),
        "coverage_pct": round(100.0 * len(ok_rows) / total, 2) if total > 0 else 0,
        "status_breakdown": dict(status_counts),
    }

    if n > 0:
        stats["selection_time_ms"] = {
            "median": sel_times[n // 2],
            "p95": sel_times[int(n * 0.95)],
            "p99": sel_times[int(n * 0.99)],
            "min": sel_times[0],
            "max": sel_times[-1],
            "mean": round(sum(sel_times) / n, 3),
        }

    # Error type breakdown
    error_types = Counter()
    for r in error_rows:
        err = r.get("error", "unknown")
        # Truncate long error messages
        if len(err) > 80:
            err = err[:77] + "..."
        error_types[err] += 1
    if error_types:
        stats["error_types"] = dict(error_types.most_common(10))

    # Config distribution (most commonly selected tile sizes)
    tile_counts = Counter()
    for r in ok_rows:
        try:
            tile = f"{r['mt_m']}x{r['mt_n']}x{r['mt_k']}"
            tile_counts[tile] += 1
        except KeyError:
            pass
    stats["top_tiles"] = dict(tile_counts.most_common(10))

    # Dtype coverage
    dtype_ok = Counter()
    dtype_total = Counter()
    for r in rows:
        dt = r.get("a_dtype", "unknown")
        dtype_total[dt] += 1
        if r.get("status") == "ok":
            dtype_ok[dt] += 1
    stats["dtype_coverage"] = {
        dt: f"{dtype_ok.get(dt, 0)}/{dtype_total[dt]}"
        for dt in sorted(dtype_total.keys())
    }

    return stats


def compute_regret(results_rows, baseline_path):
    """Compute regret vs hipblaslt baseline.

    Baseline CSV expected to have columns: idx, m, n, k, latency_us
    Regret = (origami_latency - baseline_latency) / baseline_latency
    """
    # Load baseline
    baseline = {}
    with open(baseline_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["m"]), int(row["n"]), int(row["k"]))
            baseline[key] = float(row["latency_us"])

    # Match and compute regret
    regrets = []
    for r in results_rows:
        if r.get("status") != "ok":
            continue
        key = (int(r["m"]), int(r["n"]), int(r["k"]))
        if key not in baseline:
            continue
        try:
            origami_lat = float(r["predicted_latency_us"])
            baseline_lat = baseline[key]
            if baseline_lat > 0:
                regret = (origami_lat - baseline_lat) / baseline_lat
                regrets.append({
                    "m": r["m"], "n": r["n"], "k": r["k"],
                    "origami_latency_us": origami_lat,
                    "baseline_latency_us": baseline_lat,
                    "regret": round(regret, 6),
                })
        except (KeyError, ValueError):
            continue

    if not regrets:
        return {"matched": 0, "error": "no matching problems in baseline"}

    regret_vals = sorted([r["regret"] for r in regrets])
    n = len(regret_vals)

    return {
        "matched_problems": n,
        "median_regret": regret_vals[n // 2],
        "mean_regret": round(sum(regret_vals) / n, 6),
        "p95_regret": regret_vals[int(n * 0.95)],
        "p99_regret": regret_vals[int(n * 0.99)],
        "min_regret": regret_vals[0],
        "max_regret": regret_vals[-1],
        "pct_within_5pct": round(100.0 * sum(1 for v in regret_vals if v < 0.05) / n, 2),
        "pct_within_10pct": round(100.0 * sum(1 for v in regret_vals if v < 0.10) / n, 2),
    }, regrets


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate 262K benchmark results and compute regret")
    parser.add_argument("--results-dir", required=True,
                        help="Directory with batch_XXXX.csv files")
    parser.add_argument("--output", default="combined_results.csv",
                        help="Combined output CSV")
    parser.add_argument("--baseline", default=None,
                        help="Hipblaslt baseline CSV for regret analysis")
    parser.add_argument("--regret-output", default="regret_analysis.csv",
                        help="Per-problem regret CSV")
    parser.add_argument("--summary-output", default="summary.json",
                        help="JSON summary output")
    args = parser.parse_args()

    # Collect and merge
    batch_files = collect_batch_files(args.results_dir)
    if not batch_files:
        print(f"[ERROR] No batch_*.csv files found in {args.results_dir}")
        return 1

    print(f"[AGGREGATE] Found {len(batch_files)} batch files")
    rows, fieldnames = merge_results(batch_files, args.output)
    print(f"[AGGREGATE] Merged {len(rows)} unique problems -> {args.output}")

    # Statistics
    stats = compute_statistics(rows)
    print(f"\n[STATS] Coverage: {stats['coverage_pct']}% "
          f"({stats['ok']}/{stats['total_problems']})")
    if "selection_time_ms" in stats:
        st = stats["selection_time_ms"]
        print(f"[STATS] Selection time: median={st['median']:.3f}ms, "
              f"p95={st['p95']:.3f}ms, p99={st['p99']:.3f}ms")
    if "top_tiles" in stats:
        print(f"[STATS] Top tiles: {list(stats['top_tiles'].keys())[:5]}")
    if "dtype_coverage" in stats:
        print(f"[STATS] Dtype coverage: {stats['dtype_coverage']}")

    # Regret analysis
    if args.baseline:
        print(f"\n[REGRET] Computing regret vs {args.baseline}")
        regret_stats, regret_rows = compute_regret(rows, args.baseline)
        print(f"[REGRET] Matched problems: {regret_stats.get('matched_problems', 0)}")
        if "median_regret" in regret_stats:
            print(f"[REGRET] Median regret: {regret_stats['median_regret']:.4f} "
                  f"({regret_stats['median_regret']*100:.2f}%)")
            print(f"[REGRET] p95 regret: {regret_stats['p95_regret']:.4f}")
            print(f"[REGRET] p99 regret: {regret_stats['p99_regret']:.4f}")
            print(f"[REGRET] Within 5%: {regret_stats['pct_within_5pct']}%")
        stats["regret"] = regret_stats

        # Write per-problem regret CSV
        if regret_rows:
            with open(args.regret_output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(regret_rows[0].keys()))
                writer.writeheader()
                writer.writerows(regret_rows)
            print(f"[REGRET] Per-problem regret -> {args.regret_output}")

    # Write summary JSON
    with open(args.summary_output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[AGGREGATE] Summary -> {args.summary_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
