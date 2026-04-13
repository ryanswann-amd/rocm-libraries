#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Origami Attention Test Harness

Invokes the origami analytical model to evaluate attention kernel configurations.
Accepts attention problem dimensions and kernel config parameters as CLI arguments,
reports predicted latency, throughput, memory, and selects optimal configuration.

Usage:
    python attention_harness.py --q-seq-len 2048 --kv-seq-len 2048 --head-dim 128 \
        --q-heads 32 --batch 1 --dtype f16 --arch gfx942 --n-cu 304 \
        --configs "128,128,64,1;64,64,64,2;256,128,64,1"

    python attention_harness.py --sweep  # Run a predefined sweep of common shapes
"""

import argparse
import json
import sys
import time

import origami


ARCH_MAP = {
    "gfx90a":  origami.architecture_t.gfx90a,
    "gfx942":  origami.architecture_t.gfx942,
    "gfx950":  origami.architecture_t.gfx950,
    "gfx1100": origami.architecture_t.gfx1100,
    "gfx1151": origami.architecture_t.gfx1151,
    "gfx1201": origami.architecture_t.gfx1201,
}

DEFAULT_HW_PARAMS = {
    "gfx942": {"N_CU": 304, "lds_capacity": 65536, "L2_capacity": 4_000_000, "clock_khz": 2_100_000},
    "gfx90a": {"N_CU": 110, "lds_capacity": 65536, "L2_capacity": 8_000_000, "clock_khz": 1_700_000},
    "gfx950": {"N_CU": 256, "lds_capacity": 65536, "L2_capacity": 4_000_000, "clock_khz": 2_100_000},
    "gfx906": {"N_CU": 60,  "lds_capacity": 65536, "L2_capacity": 4_000_000, "clock_khz": 1_802_000},
}


def create_hardware(arch_str, n_cu=None):
    """Create hardware descriptor from arch string."""
    arch_enum = ARCH_MAP.get(arch_str)
    if arch_enum is None:
        raise ValueError(f"Unknown arch: {arch_str}. Supported: {list(ARCH_MAP.keys())}")

    defaults = DEFAULT_HW_PARAMS.get(arch_str, DEFAULT_HW_PARAMS["gfx942"])
    cu = n_cu or defaults["N_CU"]

    hw = origami.get_hardware_for_arch(
        arch=arch_enum,
        N_CU=cu,
        lds_capacity=defaults["lds_capacity"],
        L2_capacity=defaults["L2_capacity"],
        compute_clock_khz=defaults["clock_khz"],
    )
    return hw


def create_problem(q_seq_len, kv_seq_len, head_dim, q_heads, batch, dtype_str):
    """Create an attention problem specification."""
    problem = origami.problem_t()
    problem.size = origami.dim3_t(q_seq_len, kv_seq_len, head_dim)
    problem.batch = batch
    problem.q_heads = q_heads
    problem.a_transpose = origami.transpose_t.N
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype(dtype_str)
    problem.b_dtype = origami.string_to_datatype(dtype_str)
    problem.d_dtype = origami.string_to_datatype(dtype_str)
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = problem.a_dtype
    problem.a_mx_block_size = 0
    problem.b_mx_block_size = 0
    return problem


def parse_configs(config_str, hw, problem):
    """Parse config string 'mt_m,mt_n,mt_k,occ;...' into origami config_t list."""
    mi = hw.get_recommended_matrix_instruction(problem.mi_dtype)
    configs = []
    for spec in config_str.split(";"):
        parts = [int(x.strip()) for x in spec.split(",")]
        if len(parts) < 3:
            raise ValueError(f"Config spec needs at least 3 values (mt_m,mt_n,mt_k): {spec}")
        mt_m, mt_n, mt_k = parts[0], parts[1], parts[2]
        occ = parts[3] if len(parts) > 3 else 1

        cfg = origami.config_t()
        cfg.mt = origami.dim3_t(mt_m, mt_n, mt_k)
        cfg.mi = mi
        cfg.occupancy = occ
        cfg.workgroup_mapping = 1
        configs.append(cfg)
    return configs


def bytes_per_element(dtype_str):
    """Get bytes per element for a dtype string."""
    mapping = {"f16": 2, "bf16": 2, "f32": 4, "f8": 1, "f64": 8, "i8": 1}
    return mapping.get(dtype_str, 2)


def evaluate_config(problem, hw, cfg, dtype_str):
    """Evaluate a single config against the attention model. Returns dict of metrics."""
    q_seq = problem.size.m
    kv_seq = problem.size.n
    head_dim = problem.size.k
    batch = problem.batch
    q_heads = problem.q_heads
    bpe = bytes_per_element(dtype_str)

    # Core latency metrics
    total_latency_cyc = origami.att_compute_total_latency(problem, hw, cfg, hw.N_CU)
    work_util = origami.att_calculate_work_utilization(problem, cfg)

    num_wgs, active_cus, timesteps, split_factor = origami.att_compute_cu_occupancy(
        problem, hw, cfg, origami.grid_selection_t.k_split_aware, hw.N_CU, 1
    )

    # Memory metrics
    lds_fits = origami.att_check_lds_capacity(hw, cfg.mt, problem.a_dtype, problem.b_dtype)
    l2_hit = origami.att_estimate_l2_hit(problem, hw, cfg, 1)
    mem_lat = origami.att_compute_memory_latency(problem, hw, cfg, active_cus, 1)
    compute_lat = origami.att_compute_mt_compute_latency(problem, hw, cfg)

    # Convert cycles to microseconds
    clock_ghz = hw.compute_clock_ghz
    total_latency_us = total_latency_cyc / (clock_ghz * 1e3)  # cycles / (GHz * 1e3) = us

    # Flash Attention FLOPs: 2 * batch * q_heads * q_seq * kv_seq * head_dim * 2
    # (2x for QK^T matmul + AV matmul)
    flops = 2.0 * batch * q_heads * q_seq * kv_seq * head_dim * 2
    tflops = flops / (total_latency_us * 1e-6) / 1e12 if total_latency_us > 0 else 0

    # Memory footprint estimate (Q + K + V + O)
    mem_bytes = batch * q_heads * (q_seq * head_dim + kv_seq * head_dim + kv_seq * head_dim + q_seq * head_dim) * bpe

    return {
        "mt": f"({cfg.mt.m},{cfg.mt.n},{cfg.mt.k})",
        "occupancy": cfg.occupancy,
        "total_latency_cyc": total_latency_cyc,
        "total_latency_us": total_latency_us,
        "tflops": tflops,
        "work_utilization": work_util,
        "workgroups": num_wgs,
        "active_cus": active_cus,
        "timesteps": timesteps,
        "split_factor": split_factor,
        "lds_fits": lds_fits,
        "l2_hit_rate": l2_hit,
        "mem_latency_cyc": mem_lat,
        "compute_latency_cyc": compute_lat,
        "mem_bytes": mem_bytes,
        "tflops_predicted": tflops,
    }


def run_evaluation(args):
    """Run evaluation for given args."""
    hw = create_hardware(args.arch, args.n_cu)
    problem = create_problem(args.q_seq_len, args.kv_seq_len, args.head_dim,
                             args.q_heads, args.batch, args.dtype)
    configs = parse_configs(args.configs, hw, problem)

    results = []
    for i, cfg in enumerate(configs):
        metrics = evaluate_config(problem, hw, cfg, args.dtype)
        metrics["index"] = i
        results.append(metrics)

    # Also run select_config to get the analytically optimal choice
    selected = origami.select_config(problem, hw, configs)
    selected_mt = f"({selected.config.mt.m},{selected.config.mt.n},{selected.config.mt.k})"

    return results, selected_mt


def print_results(results, selected_mt, args):
    """Pretty-print evaluation results."""
    print(f"\n{'='*80}")
    print(f"Origami Attention Config Evaluation")
    print(f"{'='*80}")
    print(f"Problem: q_seq={args.q_seq_len} kv_seq={args.kv_seq_len} head_dim={args.head_dim}")
    print(f"         q_heads={args.q_heads} batch={args.batch} dtype={args.dtype}")
    print(f"Hardware: arch={args.arch} N_CU={args.n_cu or 'default'}")
    print(f"{'='*80}\n")

    # Header
    header = f"{'Config':>18} {'Occ':>3} {'Latency(us)':>12} {'TFLOPS':>8} {'WorkUtil':>8} {'WGs':>6} {'CUs':>4} {'LDS':>4} {'L2Hit':>6}"
    print(header)
    print("-" * len(header))

    best_idx = min(range(len(results)), key=lambda i: results[i]["total_latency_us"])

    for r in results:
        marker = " *" if r["mt"] == selected_mt else "  "
        best = " <-- BEST" if r["index"] == best_idx else ""
        lds = "OK" if r["lds_fits"] else "NO"
        print(f"{r['mt']:>18} {r['occupancy']:>3} {r['total_latency_us']:>12.2f} "
              f"{r['tflops']:>8.1f} {r['work_utilization']:>8.4f} {r['workgroups']:>6} "
              f"{r['active_cus']:>4} {lds:>4} {r['l2_hit_rate']:>6.3f}{marker}{best}")

    print(f"\nOrigami selected: {selected_mt}")
    print(f"Lowest latency:   {results[best_idx]['mt']} ({results[best_idx]['total_latency_us']:.2f} us)")

    if args.json:
        print(f"\n--- JSON OUTPUT ---")
        print(json.dumps({"results": results, "selected": selected_mt}, indent=2, default=str))


SWEEP_SHAPES = [
    # (q_seq, kv_seq, head_dim, q_heads, batch, dtype)
    (512,   512,   64,  8,  1,  "f16"),
    (1024, 1024,  128, 16,  1,  "f16"),
    (2048, 2048,  128, 32,  1,  "f16"),
    (4096, 4096,  128, 32,  1,  "f16"),
    (8192, 8192,  128, 32,  1,  "f16"),
    (2048, 2048,  128, 32,  1,  "bf16"),
    (2048, 2048,   64, 16,  4,  "f16"),
    (1024, 2048,  128, 32,  1,  "f16"),  # cross-attention
    (2048, 2048,  128, 32,  8,  "f16"),  # GQA q_heads=32, batch simulates kv_heads implicitly
]

DEFAULT_CONFIGS = "128,128,64,1;64,64,64,2;256,128,64,1;128,64,32,1;64,128,64,1;256,256,64,1"


def run_sweep(args):
    """Run sweep over predefined shapes."""
    print(f"\n{'='*80}")
    print(f"Origami Attention Config Sweep — arch={args.arch}")
    print(f"{'='*80}\n")

    all_results = []
    for q_seq, kv_seq, hd, qh, batch, dtype in SWEEP_SHAPES:
        args.q_seq_len = q_seq
        args.kv_seq_len = kv_seq
        args.head_dim = hd
        args.q_heads = qh
        args.batch = batch
        args.dtype = dtype

        hw = create_hardware(args.arch, args.n_cu)
        problem = create_problem(q_seq, kv_seq, hd, qh, batch, dtype)
        configs = parse_configs(args.configs, hw, problem)

        results = []
        for cfg in configs:
            m = evaluate_config(problem, hw, cfg, dtype)
            results.append(m)

        selected = origami.select_config(problem, hw, configs)
        sel_mt = f"({selected.config.mt.m},{selected.config.mt.n},{selected.config.mt.k})"

        best = min(results, key=lambda r: r["total_latency_us"])
        shape_str = f"q={q_seq:>5} kv={kv_seq:>5} hd={hd:>3} qh={qh:>2} b={batch} {dtype:>4}"
        print(f"{shape_str} | selected={sel_mt:>18} latency={best['total_latency_us']:>10.2f} us "
              f"TFLOPS={best['tflops']:>8.1f} wu={best['work_utilization']:.4f}")

        all_results.append({
            "shape": shape_str,
            "selected": sel_mt,
            "best_latency_us": best["total_latency_us"],
            "best_tflops": best["tflops"],
        })

    if args.json:
        print(f"\n--- JSON OUTPUT ---")
        print(json.dumps(all_results, indent=2, default=str))

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Origami Attention Test Harness — evaluate kernel configs analytically"
    )
    parser.add_argument("--q-seq-len", type=int, default=2048, help="Query sequence length")
    parser.add_argument("--kv-seq-len", type=int, default=2048, help="KV sequence length")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--q-heads", type=int, default=32, help="Number of query heads")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--dtype", type=str, default="f16", choices=["f16", "bf16", "f32", "f8"])
    parser.add_argument("--arch", type=str, default="gfx942",
                        help=f"GPU architecture. Supported: {list(ARCH_MAP.keys())}")
    parser.add_argument("--n-cu", type=int, default=None, help="Number of CUs (default: per-arch default)")
    parser.add_argument("--configs", type=str, default=DEFAULT_CONFIGS,
                        help="Semicolon-separated configs: mt_m,mt_n,mt_k,occupancy")
    parser.add_argument("--sweep", action="store_true", help="Run sweep over predefined shapes")
    parser.add_argument("--json", action="store_true", help="Also output JSON")
    parser.add_argument("--output", type=str, default=None, help="Write JSON results to file")

    args = parser.parse_args()

    if args.sweep:
        results = run_sweep(args)
    else:
        results, selected = run_evaluation(args)
        print_results(results, selected, args)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
