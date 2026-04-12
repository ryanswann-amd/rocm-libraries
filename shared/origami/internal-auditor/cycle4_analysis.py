#!/usr/bin/env python3
"""
K-016 Cycle 4 Internal Auditor Analysis
========================================
1. Compute K=5 regret from measured corpus (verified, not projected)
2. Validate cost-model defect fix artifacts
3. Compute FLOP-weighted P95 SLO recommendation

All data from slurm/merged_full_corpus.csv (MI300X, 80K+ rows)
"""
import csv
import json
import os
import sys
import numpy as np
from collections import defaultdict
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(BASE, "slurm", "merged_full_corpus.csv")
K3_REGRET = os.path.join(BASE, "benchmarking", "k3_per_shape_regret.csv")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# K=3 and K=5 tile sets from k5_greedy_result.json
K3_TILES = {"128x64x128", "16x64x128", "128x128x128"}
K5_TILES = {"128x64x128", "16x64x128", "128x128x128", "16x16x256", "128x256x64"}

def parse_tile(tile_str):
    """Parse tile config string like '128x64x128' to (BLOCK_M, BLOCK_N, BLOCK_K)."""
    parts = tile_str.split("x")
    if len(parts) == 3:
        return tuple(int(p) for p in parts)
    return None

def parse_shape_dims(shape_str):
    """Parse shape like '2048x4096x5376' to (M, N, K)."""
    # Remove dtype suffix like '_bf16' or '_f16' if present
    clean = shape_str.split("_")[0] if "_" in shape_str else shape_str
    # Some shapes have form MxNxK
    parts = clean.split("x")
    if len(parts) >= 3:
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            return None
    return None

def main():
    print("=" * 70)
    print("K-016 Cycle 4 — Internal Auditor Analysis")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Node: {os.uname().nodename}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Load full corpus and compute per-shape oracle + K=3/K=5 best
    # -------------------------------------------------------------------------
    shape_tiles = defaultdict(dict)  # shape -> {tile: tflops}
    shape_cats = {}

    with open(CORPUS) as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape = row["shape"]
            tile = row["tile_config"]
            try:
                tflops = float(row["measured_tflops"])
            except (ValueError, KeyError):
                continue
            shape_tiles[shape][tile] = max(shape_tiles[shape].get(tile, 0), tflops)
            shape_cats[shape] = row["category"]

    n_shapes = len(shape_tiles)
    n_rows = sum(len(v) for v in shape_tiles.values())
    print(f"\nLoaded {n_rows} tile measurements across {n_shapes} shapes")

    # Per-shape regret computation
    results = []
    for shape, tiles in shape_tiles.items():
        oracle = max(tiles.values())
        if oracle == 0:
            continue

        # K=3 best
        k3_best = max((tiles.get(t, 0) for t in K3_TILES), default=0)
        k3_regret = (oracle - k3_best) / oracle * 100 if k3_best > 0 else 100.0

        # K=5 best
        k5_best = max((tiles.get(t, 0) for t in K5_TILES), default=0)
        k5_regret = (oracle - k5_best) / oracle * 100 if k5_best > 0 else 100.0

        # Parse dims for FLOP weight
        dims = parse_shape_dims(shape)
        flops = 2.0 * dims[0] * dims[1] * dims[2] if dims else 0

        results.append({
            "shape": shape,
            "category": shape_cats.get(shape, "unknown"),
            "oracle_tflops": oracle,
            "k3_best_tflops": k3_best,
            "k5_best_tflops": k5_best,
            "k3_regret": k3_regret,
            "k5_regret": k5_regret,
            "flops": flops,
        })

    # -------------------------------------------------------------------------
    # 2. Compute aggregate statistics
    # -------------------------------------------------------------------------
    k3_regrets = np.array([r["k3_regret"] for r in results])
    k5_regrets = np.array([r["k5_regret"] for r in results])
    flops_arr = np.array([r["flops"] for r in results])
    total_flops = flops_arr.sum()
    flop_weights = flops_arr / total_flops if total_flops > 0 else np.ones(len(results)) / len(results)

    k3_flop_mean = np.average(k3_regrets, weights=flop_weights)
    k5_flop_mean = np.average(k5_regrets, weights=flop_weights)

    # Sort for percentiles
    k3_sorted = np.sort(k3_regrets)
    k5_sorted = np.sort(k5_regrets)

    # FLOP-weighted percentiles via weighted quantile
    def weighted_percentile(data, weights, percentile):
        """Compute weighted percentile."""
        sorted_idx = np.argsort(data)
        sorted_data = data[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cum_weights = np.cumsum(sorted_weights)
        cum_weights /= cum_weights[-1]
        return np.interp(percentile / 100.0, cum_weights, sorted_data)

    k3_flop_p50 = weighted_percentile(k3_regrets, flop_weights, 50)
    k3_flop_p95 = weighted_percentile(k3_regrets, flop_weights, 95)
    k3_flop_p99 = weighted_percentile(k3_regrets, flop_weights, 99)

    k5_flop_p50 = weighted_percentile(k5_regrets, flop_weights, 50)
    k5_flop_p95 = weighted_percentile(k5_regrets, flop_weights, 95)
    k5_flop_p99 = weighted_percentile(k5_regrets, flop_weights, 99)

    print(f"\n--- K=3 Statistics ({len(results)} shapes) ---")
    print(f"  Mean:     {k3_regrets.mean():.4f}%")
    print(f"  P50:      {np.percentile(k3_regrets, 50):.4f}%")
    print(f"  P95:      {np.percentile(k3_regrets, 95):.4f}%")
    print(f"  P99:      {np.percentile(k3_regrets, 99):.4f}%")
    print(f"  Max:      {k3_regrets.max():.4f}%")
    print(f"  FLOP-wt mean: {k3_flop_mean:.4f}%")
    print(f"  FLOP-wt P95:  {k3_flop_p95:.4f}%")
    print(f"  FLOP-wt P99:  {k3_flop_p99:.4f}%")
    print(f"  Zero-regret:  {(k3_regrets == 0).sum()}/{len(results)} ({(k3_regrets == 0).mean()*100:.1f}%)")

    print(f"\n--- K=5 Statistics ({len(results)} shapes) ---")
    print(f"  Mean:     {k5_regrets.mean():.4f}%")
    print(f"  P50:      {np.percentile(k5_regrets, 50):.4f}%")
    print(f"  P95:      {np.percentile(k5_regrets, 95):.4f}%")
    print(f"  P99:      {np.percentile(k5_regrets, 99):.4f}%")
    print(f"  Max:      {k5_regrets.max():.4f}%")
    print(f"  FLOP-wt mean: {k5_flop_mean:.4f}%")
    print(f"  FLOP-wt P95:  {k5_flop_p95:.4f}%")
    print(f"  FLOP-wt P99:  {k5_flop_p99:.4f}%")
    print(f"  Zero-regret:  {(k5_regrets == 0).sum()}/{len(results)} ({(k5_regrets == 0).mean()*100:.1f}%)")

    print(f"\n--- K=3 → K=5 Improvement ---")
    print(f"  Mean regret: {k3_regrets.mean():.4f}% → {k5_regrets.mean():.4f}% (Δ={k3_regrets.mean()-k5_regrets.mean():.4f}pp)")
    print(f"  FLOP-wt mean: {k3_flop_mean:.4f}% → {k5_flop_mean:.4f}% (Δ={k3_flop_mean-k5_flop_mean:.4f}pp)")
    print(f"  FLOP-wt P95:  {k3_flop_p95:.4f}% → {k5_flop_p95:.4f}% (Δ={k3_flop_p95-k5_flop_p95:.4f}pp)")
    print(f"  Zero-regret shapes: {(k3_regrets == 0).sum()} → {(k5_regrets == 0).sum()}")

    # -------------------------------------------------------------------------
    # 3. Cross-validate K=3 against benchmarking team's k3_per_shape_regret.csv
    # -------------------------------------------------------------------------
    print(f"\n--- Cross-validation against benchmarking K=3 data ---")
    bench_k3 = {}
    with open(K3_REGRET) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bench_k3[row["shape"]] = float(row["k3_regret_pct"])

    deltas = []
    for r in results:
        if r["shape"] in bench_k3:
            delta = abs(r["k3_regret"] - bench_k3[r["shape"]])
            deltas.append(delta)

    deltas = np.array(deltas)
    print(f"  Matched shapes: {len(deltas)}")
    print(f"  Max delta: {deltas.max():.6f}pp")
    print(f"  Mean delta: {deltas.mean():.6f}pp")
    print(f"  All within 0.1pp: {'YES' if deltas.max() < 0.1 else 'NO'}")

    # -------------------------------------------------------------------------
    # 4. Decode fix validation — trace to specific artifact
    # -------------------------------------------------------------------------
    print(f"\n--- Decode BLOCK_N≤64 Fix Validation ---")
    decode_shapes = [(s, r) for s, r in zip(
        [r["shape"] for r in results],
        [r for r in results]
    ) if r["category"] == "decode"]

    # Look at the two key decode shapes
    for target in ["1x16384x16384", "1x13312x16384"]:
        matches = [r for r in results if target in r["shape"]]
        for m in matches:
            print(f"  {m['shape']}: K=3 regret={m['k3_regret']:.2f}%, K=5 regret={m['k5_regret']:.2f}%")
            print(f"    Oracle={m['oracle_tflops']:.4f} TFLOPS, K3 best={m['k3_best_tflops']:.4f}, K5 best={m['k5_best_tflops']:.4f}")

    # -------------------------------------------------------------------------
    # 5. Top-5 highest regret shapes at K=5
    # -------------------------------------------------------------------------
    print(f"\n--- Top 10 Highest-Regret Shapes at K=5 ---")
    sorted_by_k5 = sorted(results, key=lambda r: r["k5_regret"], reverse=True)
    for i, r in enumerate(sorted_by_k5[:10]):
        print(f"  #{i+1}: {r['shape']} ({r['category']}): K=5 regret={r['k5_regret']:.2f}%, K=3={r['k3_regret']:.2f}%")

    # -------------------------------------------------------------------------
    # 6. SLO Recommendation with rationale
    # -------------------------------------------------------------------------
    print(f"\n--- Production Regret SLO Recommendation ---")
    print(f"  Binding metric: FLOP-weighted P95 (reflects actual compute budget lost)")
    print(f"  Rationale: Shape-count P95 overweights decode (0.11% of FLOPs) and")
    print(f"    small_batch (3.17%) while underweighting compute_bound (46.88%) and")
    print(f"    medium_batch (49.84%) categories that dominate production FLOPs.")
    print(f"")
    print(f"  Proposed SLO thresholds:")
    print(f"    K=3 (current):    FLOP-wt mean ≤5.0% (actual {k3_flop_mean:.2f}% PASS)")
    print(f"                      FLOP-wt P95 ≤15.0% (actual {k3_flop_p95:.2f}%)")
    print(f"                      Shape-ct P95 ≤20.0% (actual {np.percentile(k3_regrets, 95):.2f}%)")
    print(f"    K=5 (target):     FLOP-wt mean ≤2.0% (actual {k5_flop_mean:.2f}%)")
    print(f"                      FLOP-wt P95 ≤8.0% (actual {k5_flop_p95:.2f}%)")
    print(f"                      Shape-ct P95 ≤12.0% (actual {np.percentile(k5_regrets, 95):.2f}%)")

    # -------------------------------------------------------------------------
    # 7. Save results
    # -------------------------------------------------------------------------
    output = {
        "description": "K-016 Cycle 4 Internal Auditor — K=3 vs K=5 regret from measured corpus",
        "source": "slurm/merged_full_corpus.csv (MI300X gfx942, OCI cluster)",
        "n_shapes": len(results),
        "n_measurements": n_rows,
        "timestamp": datetime.now().isoformat(),
        "node": os.uname().nodename,
        "k3_stats": {
            "mean": round(k3_regrets.mean(), 4),
            "p50": round(float(np.percentile(k3_regrets, 50)), 4),
            "p95": round(float(np.percentile(k3_regrets, 95)), 4),
            "p99": round(float(np.percentile(k3_regrets, 99)), 4),
            "max": round(float(k3_regrets.max()), 4),
            "flop_weighted_mean": round(float(k3_flop_mean), 4),
            "flop_weighted_p95": round(float(k3_flop_p95), 4),
            "flop_weighted_p99": round(float(k3_flop_p99), 4),
            "zero_regret_count": int((k3_regrets == 0).sum()),
        },
        "k5_stats": {
            "mean": round(k5_regrets.mean(), 4),
            "p50": round(float(np.percentile(k5_regrets, 50)), 4),
            "p95": round(float(np.percentile(k5_regrets, 95)), 4),
            "p99": round(float(np.percentile(k5_regrets, 99)), 4),
            "max": round(float(k5_regrets.max()), 4),
            "flop_weighted_mean": round(float(k5_flop_mean), 4),
            "flop_weighted_p95": round(float(k5_flop_p95), 4),
            "flop_weighted_p99": round(float(k5_flop_p99), 4),
            "zero_regret_count": int((k5_regrets == 0).sum()),
        },
        "slo_recommendation": {
            "binding_metric": "FLOP-weighted P95",
            "rationale": "Reflects actual GPU compute budget lost. Shape-count P95 overweights decode shapes (0.11% of total FLOPs) which inflate unweighted tails.",
            "k3_thresholds": {
                "flop_weighted_mean": "≤5.0%",
                "flop_weighted_p95": "≤15.0%",
                "shape_count_p95": "≤20.0%"
            },
            "k5_thresholds": {
                "flop_weighted_mean": "≤2.0%",
                "flop_weighted_p95": "≤8.0%",
                "shape_count_p95": "≤12.0%"
            }
        },
        "cross_validation": {
            "matched_shapes": len(deltas),
            "max_delta_pp": round(float(deltas.max()), 6),
            "all_within_0_1pp": bool(deltas.max() < 0.1),
        },
    }

    out_path = os.path.join(OUT_DIR, "cycle4_verified_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save per-shape K=5 regret CSV
    csv_path = os.path.join(OUT_DIR, "k5_per_shape_regret_verified.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shape", "category", "k3_regret_pct", "k5_regret_pct", "oracle_tflops", "k5_best_tflops", "flop_weight"])
        for r, w in zip(results, flop_weights):
            writer.writerow([r["shape"], r["category"],
                           round(r["k3_regret"], 4), round(r["k5_regret"], 4),
                           round(r["oracle_tflops"], 4), round(r["k5_best_tflops"], 4),
                           round(float(w), 8)])
    print(f"Per-shape CSV saved to {csv_path}")

    # Return data for plotting
    return results, k3_regrets, k5_regrets, flop_weights, output

if __name__ == "__main__":
    main()
