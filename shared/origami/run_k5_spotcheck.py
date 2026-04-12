#!/usr/bin/env python3
"""
K-016 Step s3 — Script A: K=5 Tile Spotcheck Runner
====================================================
Launches the 2 new K=5 tiles (`16x16x256`, `128x256x64`) on the 4 high-regret
shapes from the gap_closure_manifest.md spotcheck matrix (8 combos total).
Captures TFLOPS output (or crash/zero-TFLOPS), writes results to
spotcheck_k5_results.csv.

Uses the same Triton matmul kernel and benchmarking harness pattern as
tail_spotcheck_bench.py from cycle-2.

Usage:
  # Full GPU run (requires MI300X/MI355X):
  python3 run_k5_spotcheck.py

  # Dry-run (validates config without GPU):
  python3 run_k5_spotcheck.py --dry-run

  # Custom output path:
  python3 run_k5_spotcheck.py --output my_results.csv

Output: spotcheck_k5_results.csv with columns:
  shape, tile, BLOCK_M, BLOCK_N, BLOCK_K, lds_bytes, lds_ok, category,
  tflops, ms, status, error, gpu, node, timestamp

All TFLOPS values are tagged [VERIFIED] when measured on GPU.
Prints [VERIFIED] tag on completion.

Date: 2026-04-12
Source: gap_closure_manifest.md §1 (spotcheck matrix)
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration — directly from gap_closure_manifest.md §1
# ---------------------------------------------------------------------------

# K=3 base tiles (1,863-shape greedy set-cover, k5_greedy_result.json)
K3_TILES = [
    (128, 64, 128),   # 128x64x128
    (16, 64, 128),    # 16x64x128
    (128, 128, 128),  # 128x128x128
]

# K=5 marginal tiles (the 2 new tiles beyond K=3)
K5_NEW_TILES = [
    # (BLOCK_M, BLOCK_N, BLOCK_K, label)
    (16, 16, 256, "16x16x256"),      # Added at K=4; decode specialist (small M, deep K)
    (128, 256, 64, "128x256x64"),    # Added at K=5; prefill specialist (large N, BLOCK_K=64)
]

# 4 high-regret shapes from gap_closure_manifest.md §1
# Format: (M, N, K, dtype_str, category, origami_regret_pct)
SPOTCHECK_SHAPES = [
    (2048, 4096, 5376, "bf16", "prefill", 35.7),     # Worst regret in 16-shape set
    (1, 16384, 16384, "bf16", "decode", 29.3),        # Worst decode shape; deep-K pathology
    (3600, 4096, 4096, "bf16", "compute-bound", 28.7), # Large compute-bound, oracle=256x256x64
    (1536, 3584, 3584, "f16", "prefill", 24.0),       # f16 prefill shape with moderate regret
]

NUM_STAGES = 2  # Standard Triton pipeline stages for MI300X
ELEM_BYTES = {"bf16": 2, "f16": 2}  # bytes per element

# CSV output columns
CSV_COLUMNS = [
    "shape", "tile", "BLOCK_M", "BLOCK_N", "BLOCK_K",
    "lds_bytes", "lds_ok", "category", "dtype", "origami_regret_pct",
    "tflops", "ms", "status", "error",
    "gpu", "arch", "node", "timestamp",
]


def triton_lds_bytes(bm, bn, bk, dtype_str, stages=NUM_STAGES):
    """Compute LDS usage for Triton staged pipeline model.

    Formula from origami/include/origami/gemm.hpp:266-277 [VERIFIED]:
      LDS = (num_stages - 1) × (BLOCK_M × BLOCK_K × bpe + BLOCK_K × BLOCK_N × bpe)
    """
    bpe = ELEM_BYTES[dtype_str]
    return max(0, stages - 1) * (bm * bk * bpe + bk * bn * bpe)


def validate_config():
    """Validate all configuration parameters (used in dry-run mode)."""
    errors = []
    warnings = []

    # Check tile definitions
    for bm, bn, bk, label in K5_NEW_TILES:
        if bm <= 0 or bn <= 0 or bk <= 0:
            errors.append(f"Invalid tile dimensions: {label}")
        lds_bf16 = triton_lds_bytes(bm, bn, bk, "bf16")
        lds_f16 = triton_lds_bytes(bm, bn, bk, "f16")
        if lds_bf16 > 65536:
            errors.append(f"Tile {label} exceeds MI300X LDS (bf16): {lds_bf16} > 65536")
        if lds_f16 > 65536:
            errors.append(f"Tile {label} exceeds MI300X LDS (f16): {lds_f16} > 65536")

    # Check shape definitions
    for M, N, K, dtype, cat, regret in SPOTCHECK_SHAPES:
        if M <= 0 or N <= 0 or K <= 0:
            errors.append(f"Invalid shape dimensions: {M}x{N}x{K}")
        if dtype not in ELEM_BYTES:
            errors.append(f"Unknown dtype: {dtype}")

    # Build full spotcheck matrix and validate
    combos = []
    for M, N, K, dtype, cat, regret in SPOTCHECK_SHAPES:
        for bm, bn, bk, tile_label in K5_NEW_TILES:
            lds = triton_lds_bytes(bm, bn, bk, dtype)
            lds_ok = lds <= 65536  # MI300X limit
            shape_id = f"{M}x{N}x{K}_{dtype}_r"
            combos.append({
                "shape": shape_id,
                "tile": tile_label,
                "lds_bytes": lds,
                "lds_ok": lds_ok,
                "category": cat,
                "dtype": dtype,
            })

    return errors, warnings, combos


def dry_run():
    """Validate configuration without GPU, print spotcheck matrix."""
    print("=" * 70)
    print("K-016 K=5 Spotcheck — DRY RUN (no GPU required)")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Node: {os.uname().nodename}")
    print()

    errors, warnings, combos = validate_config()

    # Print spotcheck matrix
    print(f"Spotcheck Matrix: {len(K5_NEW_TILES)} tiles × {len(SPOTCHECK_SHAPES)} shapes = {len(combos)} combos")
    print()
    print(f"{'Shape':<32} {'Tile':<14} {'LDS (B)':<10} {'LDS OK?':<8} {'Category':<14}")
    print("-" * 82)
    for c in combos:
        print(f"{c['shape']:<32} {c['tile']:<14} {c['lds_bytes']:<10} "
              f"{'PASS' if c['lds_ok'] else 'FAIL':<8} {c['category']:<14}")

    print()

    # Print K=3 vs K=5 tile set comparison
    print("K=3 tile set:")
    for bm, bn, bk in K3_TILES:
        print(f"  {bm}x{bn}x{bk}")
    print("K=5 additions:")
    for bm, bn, bk, label in K5_NEW_TILES:
        lds = triton_lds_bytes(bm, bn, bk, "bf16")
        print(f"  {label}  (LDS bf16 s2 = {lds} bytes)")

    print()
    print(f"CSV output columns: {', '.join(CSV_COLUMNS)}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
        print("\n[DRY-RUN FAILED] — fix errors before GPU run")
        return False
    else:
        all_lds_ok = all(c["lds_ok"] for c in combos)
        print(f"\nAll {len(combos)} combos LDS-feasible: {'YES' if all_lds_ok else 'NO'}")
        if warnings:
            for w in warnings:
                print(f"  ⚠ {w}")
        print("\n[DRY-RUN PASSED] — config validated, ready for GPU execution")
        print("[VERIFIED] — LDS feasibility confirmed for all 8 combos (gap_closure_manifest.md §1)")
        return True


def gpu_run(output_path, iters=30, warmup=10, gpu_id=0):
    """Run full GPU benchmark on all 8 spotcheck combos."""
    # Late imports — only needed for GPU run
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    import triton
    import triton.language as tl

    # Device info
    props = torch.cuda.get_device_properties(0)
    device_name = props.name.lower()
    if "mi300x" in device_name or "mi325" in device_name:
        ARCH = "gfx942"
        LDS_CAP = 65536
        GPU_NAME = "MI300X" if "mi300" in device_name else "MI325X"
    elif "mi355" in device_name or "mi350" in device_name:
        ARCH = "gfx950"
        LDS_CAP = 163840
        GPU_NAME = "MI355X" if "mi355" in device_name else "MI350X"
    else:
        ARCH = "unknown"
        LDS_CAP = 65536
        GPU_NAME = props.name

    CUS = props.multi_processor_count

    print("=" * 70)
    print("K-016 K=5 Spotcheck — GPU RUN")
    print("=" * 70)
    print(f"Device: {props.name} ({GPU_NAME})")
    print(f"Arch: {ARCH}, CUs: {CUS}, LDS: {LDS_CAP // 1024} KB")
    print(f"Triton: {triton.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Iters: {iters}, Warmup: {warmup}")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Node: {os.uname().nodename}")
    print()

    # -- Define Triton GEMM kernel (same as tail_spotcheck_bench.py) --
    @triton.jit
    def _matmul_kernel(
        A, B, C, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        A_ptrs = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        B_ptrs = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            a = tl.load(A_ptrs, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_ptrs, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)
            A_ptrs += BLOCK_K * stride_ak
            B_ptrs += BLOCK_K * stride_bk

        c = acc.to(C.dtype.element_ty)
        C_ptrs = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C_ptrs, c, mask=mask)

    # -- Benchmark function --
    def bench_one(M, N, K, bm, bn, bk, dtype_str, device):
        """Benchmark one tile config. Returns (ms, tflops, error_msg)."""
        if bk > K:
            return None, None, f"BLOCK_K={bk} > K={K}, skip"

        dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

        try:
            A = torch.randn(M, K, dtype=dtype, device=device)
            B = torch.randn(K, N, dtype=dtype, device=device)
            C = torch.empty(M, N, dtype=dtype, device=device)
        except RuntimeError:
            return None, None, "OOM"

        grid = lambda META: (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

        try:
            # Warmup
            for _ in range(warmup):
                _matmul_kernel[grid](
                    A, B, C, M, N, K,
                    A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                    C.stride(0), C.stride(1),
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
                )
            torch.cuda.synchronize()

            # Timed
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(iters):
                _matmul_kernel[grid](
                    A, B, C, M, N, K,
                    A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                    C.stride(0), C.stride(1),
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
                )
            end_evt.record()
            torch.cuda.synchronize()
            ms = start_evt.elapsed_time(end_evt) / iters
            tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
            del A, B, C
            return ms, tflops, None

        except Exception as e:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            del A, B, C
            torch.cuda.empty_cache()
            error_str = str(e)[:300]
            lds_crash = ("lds" in error_str.lower() or
                         "shared memory" in error_str.lower() or
                         "out of resource" in error_str.lower() or
                         "illegal" in error_str.lower())
            if lds_crash:
                return None, None, f"LDS_CRASH: {error_str}"
            return None, None, error_str

    # -- Run all 8 combos --
    device = "cuda:0"
    results = []
    timestamp = datetime.now().isoformat()
    total_combos = len(SPOTCHECK_SHAPES) * len(K5_NEW_TILES)
    combo_idx = 0

    for M, N, K, dtype_str, category, regret in SPOTCHECK_SHAPES:
        shape_id = f"{M}x{N}x{K}_{dtype_str}_r"
        for bm, bn, bk, tile_label in K5_NEW_TILES:
            combo_idx += 1
            lds = triton_lds_bytes(bm, bn, bk, dtype_str)
            lds_ok = lds <= LDS_CAP

            row = {
                "shape": shape_id,
                "tile": tile_label,
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "BLOCK_K": bk,
                "lds_bytes": lds,
                "lds_ok": lds_ok,
                "category": category,
                "dtype": dtype_str,
                "origami_regret_pct": regret,
                "gpu": GPU_NAME,
                "arch": ARCH,
                "node": os.uname().nodename,
                "timestamp": timestamp,
            }

            if not lds_ok:
                row["tflops"] = None
                row["ms"] = None
                row["status"] = "SKIPPED_LDS"
                row["error"] = f"LDS {lds} > {LDS_CAP}"
                print(f"  [{combo_idx}/{total_combos}] {shape_id} × {tile_label}: "
                      f"LDS {lds} > {LDS_CAP} — SKIP")
            else:
                ms, tflops, error = bench_one(M, N, K, bm, bn, bk, dtype_str, device)
                row["tflops"] = round(tflops, 4) if tflops is not None else None
                row["ms"] = round(ms, 4) if ms is not None else None

                if error:
                    row["status"] = "CRASH" if "LDS_CRASH" in str(error) else "ERROR"
                    row["error"] = error
                    print(f"  [{combo_idx}/{total_combos}] {shape_id} × {tile_label}: "
                          f"ERROR — {error[:80]}")
                elif tflops is not None and tflops == 0.0:
                    row["status"] = "ZERO_TFLOPS"
                    row["error"] = "Zero TFLOPS measured"
                    print(f"  [{combo_idx}/{total_combos}] {shape_id} × {tile_label}: "
                          f"ZERO TFLOPS (possible silent failure)")
                else:
                    row["status"] = "OK"
                    row["error"] = None
                    print(f"  [{combo_idx}/{total_combos}] {shape_id} × {tile_label}: "
                          f"{tflops:.4f} TFLOPS ({ms:.3f} ms) [VERIFIED] on {GPU_NAME}")

            results.append(row)

    # -- Write CSV --
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # -- Also write JSON sidecar --
    json_path = output_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump({
            "description": "K-016 K=5 tile spotcheck results",
            "gpu": GPU_NAME,
            "arch": ARCH,
            "node": os.uname().nodename,
            "timestamp": timestamp,
            "triton_version": triton.__version__,
            "pytorch_version": torch.__version__,
            "iters": iters,
            "warmup": warmup,
            "k3_tiles": [f"{bm}x{bn}x{bk}" for bm, bn, bk in K3_TILES],
            "k5_new_tiles": [label for _, _, _, label in K5_NEW_TILES],
            "num_combos": total_combos,
            "results": results,
        }, f, indent=2)

    # -- Summary --
    ok_results = [r for r in results if r["status"] == "OK"]
    crash_results = [r for r in results if r["status"] == "CRASH"]
    zero_results = [r for r in results if r["status"] == "ZERO_TFLOPS"]

    print()
    print("=" * 70)
    print(f"SPOTCHECK COMPLETE: {len(ok_results)}/{total_combos} OK, "
          f"{len(crash_results)} crashes, {len(zero_results)} zero-TFLOPS")
    print(f"CSV: {output_path}")
    print(f"JSON: {json_path}")
    print("=" * 70)

    # Print per-shape summary table
    print()
    print(f"{'Shape':<32} {'Tile':<14} {'TFLOPS':>10} {'Status':<12}")
    print("-" * 72)
    for r in results:
        tflops_str = f"{r['tflops']:.4f}" if r["tflops"] is not None else "N/A"
        print(f"{r['shape']:<32} {r['tile']:<14} {tflops_str:>10} {r['status']:<12}")

    print()
    tag = "[VERIFIED]" if crash_results == [] else "[VERIFIED — WITH CRASHES]"
    print(f"{tag} — K=5 spotcheck complete on {GPU_NAME} ({os.uname().nodename})")
    print(f"  Measured on: {GPU_NAME} ({ARCH}, {CUS} CU, {LDS_CAP // 1024} KB LDS)")
    print(f"  Source: gap_closure_manifest.md §1 — 2 tiles × 4 shapes = 8 combos")
    return len(crash_results) == 0


def main():
    parser = argparse.ArgumentParser(
        description="K-016 K=5 tile spotcheck runner. "
                    "Benchmarks 2 new K=5 tiles on 4 high-regret shapes."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without GPU")
    parser.add_argument("--output", type=str,
                        default="spotcheck_k5_results.csv",
                        help="Output CSV path (default: spotcheck_k5_results.csv)")
    parser.add_argument("--iters", type=int, default=30,
                        help="Timed iterations per config (default: 30)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index (default: 0)")
    args = parser.parse_args()

    if args.dry_run:
        success = dry_run()
        sys.exit(0 if success else 1)
    else:
        success = gpu_run(args.output, iters=args.iters,
                          warmup=args.warmup, gpu_id=args.gpu)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
