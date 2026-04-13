#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Tritonblas 262K benchmark runner with checkpointing and resume support.

Processes a batch of problems from the 262K corpus, running origami config
selection with target_t=triton and recording predictions. Supports:

  - Batch slicing: --start-idx / --end-idx to process a subset
  - Checkpointing: writes progress every N problems; resumes from last checkpoint
  - Preemption recovery: SIGTERM handler flushes checkpoint before exit
  - Result output: per-problem CSV with selected config, predicted latency, etc.

Usage:
    python3 bench_runner.py \\
        --problems problems_262k.csv \\
        --output results/batch_0000.csv \\
        --checkpoint checkpoints/batch_0000.ckpt \\
        --start-idx 0 --end-idx 1000 \\
        --arch gfx942 --n-cu 304

Designed to run inside a Slurm job array — each array task processes one batch.
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
from pathlib import Path

# Try importing origami — if not available, provide helpful error
try:
    import origami
except ImportError:
    print("ERROR: origami not importable. Ensure the origami wheel is installed "
          "in your venv: pip install origami-*.whl", file=sys.stderr)
    sys.exit(1)


# --- Checkpoint Management ---

class CheckpointManager:
    """Manages benchmark checkpointing for preemption recovery."""

    def __init__(self, path, flush_interval=100):
        self.path = Path(path)
        self.flush_interval = flush_interval
        self._last_completed_idx = -1
        self._results_buffer = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self):
        """Load checkpoint, return last completed global index or -1."""
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self._last_completed_idx = data.get("last_completed_idx", -1)
                print(f"[CHECKPOINT] Resuming from idx {self._last_completed_idx + 1}")
                return self._last_completed_idx
            except (json.JSONDecodeError, KeyError):
                print("[CHECKPOINT] Corrupt checkpoint, starting fresh")
        return -1

    def update(self, global_idx):
        """Mark a problem as completed."""
        self._last_completed_idx = global_idx
        self._results_buffer.append(global_idx)
        if len(self._results_buffer) >= self.flush_interval:
            self.flush()

    def flush(self):
        """Write checkpoint to disk."""
        if self._last_completed_idx >= 0:
            with open(self.path, "w") as f:
                json.dump({"last_completed_idx": self._last_completed_idx,
                           "timestamp": time.time()}, f)
            self._results_buffer.clear()

    @property
    def last_completed(self):
        return self._last_completed_idx


# --- Hardware Setup ---

ARCH_MAP = {
    "gfx90a": origami.architecture_t.gfx90a,
    "gfx942": origami.architecture_t.gfx942,
    "gfx950": origami.architecture_t.gfx950,
}

# Default hardware parameters per arch (N_CU, LDS_bytes, L2_bytes, clk_MHz)
ARCH_DEFAULTS = {
    "gfx90a": (110, 64 * 1024, 8 * 1024 * 1024, 1700000),
    "gfx942": (304, 64 * 1024, 24 * 1024 * 1024, 1700000),
    "gfx950": (304, 64 * 1024, 32 * 1024 * 1024, 2100000),
}


def get_hardware(arch_str, n_cu=None):
    """Create hardware object for the given architecture."""
    if arch_str not in ARCH_MAP:
        raise ValueError(f"Unsupported arch: {arch_str}. Supported: {list(ARCH_MAP.keys())}")
    defaults = ARCH_DEFAULTS[arch_str]
    cu = n_cu if n_cu else defaults[0]
    return origami.get_hardware_for_arch(ARCH_MAP[arch_str], cu, defaults[1],
                                         defaults[2], defaults[3])


# --- Config Generation (Triton-targeted) ---

def generate_triton_configs(hardware, dtype_str):
    """Generate config_t objects with target_t=triton for benchmarking.

    This generates the same config space as create_config_list in helpers.py
    but sets target=triton on every config.
    """
    dtype_enum = origami.string_to_datatype(dtype_str)
    mi_list = hardware.get_valid_matrix_instructions(dtype_enum)
    if not mi_list:
        return []

    depth_unroll = [16, 32, 64, 128, 256, 512, 1024]
    occupancy_values = [1]
    waves = [[4, 1], [2, 2], [1, 4], [1, 2], [2, 1], [1, 1]]
    max_mt = 512
    min_mt = 16

    configs = []
    for mi in mi_list:
        mi_m, mi_n, mi_k = mi.m, mi.n, mi.k
        # Generate wave-based MT pairs
        for wave in waves:
            wave_tile_m = 0
            while True:
                wave_tile_m += 1
                mt_m = mi_m * wave_tile_m * wave[0]
                if mt_m < min_mt:
                    continue
                if mt_m > max_mt:
                    break
                wave_tile_n = 0
                while True:
                    wave_tile_n += 1
                    mt_n = mi_n * wave_tile_n * wave[1]
                    if mt_n < min_mt:
                        continue
                    if mt_n > max_mt:
                        break
                    for mt_k in depth_unroll:
                        for occ in occupancy_values:
                            config = origami.config_t()
                            config.mt = origami.dim3_t(mt_m, mt_n, mt_k)
                            config.mi = origami.dim3_t(mi_m, mi_n, mi_k)
                            config.occupancy = occ
                            config.target = origami.target_t.triton
                            configs.append(config)
    return configs


# --- Problem Processing ---

DTYPE_MAP = {
    "f16": "f16", "f32": "f32", "bf16": "bf16", "f8": "f8",
    "i8": "i8",
}

TRANS_MAP = {
    "T": origami.transpose_t.T,
    "N": origami.transpose_t.N,
}


def process_problem(row, hardware, configs_cache, fallback_idx=0):
    """Run origami config selection for a single problem.

    Returns dict with results or error info.
    """
    m, n, k = int(row["m"]), int(row["n"]), int(row["k"])
    batch = int(row.get("batch", 1))
    a_dt = row.get("a_dtype", "f16")
    b_dt = row.get("b_dtype", "f16")
    out_dt = row.get("out_dtype", "f16")
    a_trans = row.get("a_trans", "T")
    b_trans = row.get("b_trans", "N")
    # Use idx from CSV if present, otherwise use fallback
    problem_idx = row.get("idx", str(fallback_idx))

    # Use cached configs for this dtype, or generate
    cache_key = a_dt  # configs depend on input dtype for MI selection
    if cache_key not in configs_cache:
        configs_cache[cache_key] = generate_triton_configs(hardware, a_dt)
    configs = configs_cache[cache_key]

    if not configs:
        return {"idx": problem_idx, "m": m, "n": n, "k": k, "batch": batch,
                "a_dtype": a_dt, "b_dtype": b_dt, "out_dtype": out_dt,
                "a_trans": a_trans, "b_trans": b_trans,
                "status": "no_configs", "error": f"No configs for dtype {a_dt}"}

    # Build problem_t
    problem = origami.problem_t()
    problem.size = origami.dim3_t(m, n, k)
    problem.batch = batch
    problem.a_transpose = TRANS_MAP.get(a_trans, origami.transpose_t.T)
    problem.b_transpose = TRANS_MAP.get(b_trans, origami.transpose_t.N)
    problem.a_dtype = origami.string_to_datatype(a_dt)
    problem.b_dtype = origami.string_to_datatype(b_dt)
    problem.d_dtype = origami.string_to_datatype(out_dt)
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = problem.a_dtype
    problem.a_mx_block_size = 0
    problem.b_mx_block_size = 0

    try:
        t0 = time.monotonic()
        result = origami.select_config(problem, hardware, configs)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        cfg = result.config
        return {
            "idx": problem_idx, "m": m, "n": n, "k": k, "batch": batch,
            "a_dtype": a_dt, "b_dtype": b_dt, "out_dtype": out_dt,
            "a_trans": a_trans, "b_trans": b_trans,
            "status": "ok",
            "mt_m": cfg.mt.m, "mt_n": cfg.mt.n, "mt_k": cfg.mt.k,
            "mi_m": cfg.mi.m, "mi_n": cfg.mi.n, "mi_k": cfg.mi.k,
            "occupancy": cfg.occupancy,
            "predicted_latency_us": result.latency,
            "selection_time_ms": round(elapsed_ms, 3),
            "n_configs_evaluated": len(configs),
            "error": "",
        }
    except Exception as e:
        return {
            "idx": problem_idx, "m": m, "n": n, "k": k, "batch": batch,
            "a_dtype": a_dt, "b_dtype": b_dt, "out_dtype": out_dt,
            "a_trans": a_trans, "b_trans": b_trans,
            "status": "error", "error": str(e),
        }


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Tritonblas 262K benchmark runner with checkpointing")
    parser.add_argument("--problems", required=True, help="Input problems CSV")
    parser.add_argument("--output", required=True, help="Output results CSV")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file path")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index in problems CSV (inclusive)")
    parser.add_argument("--end-idx", type=int, default=-1,
                        help="End index in problems CSV (exclusive, -1=all)")
    parser.add_argument("--arch", default="gfx942",
                        help="GPU architecture (gfx90a, gfx942, gfx950)")
    parser.add_argument("--n-cu", type=int, default=None,
                        help="Number of CUs (overrides arch default)")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Flush checkpoint every N problems")
    args = parser.parse_args()

    # Setup hardware
    hardware = get_hardware(args.arch, args.n_cu)
    print(f"[BENCH] Hardware: {args.arch}, CUs: {hardware.N_CU}")

    # Load problems
    with open(args.problems, "r") as f:
        reader = csv.DictReader(f)
        all_problems = list(reader)

    end = args.end_idx if args.end_idx > 0 else len(all_problems)
    problems = all_problems[args.start_idx:end]
    print(f"[BENCH] Processing problems [{args.start_idx}:{end}] "
          f"({len(problems)} problems)")

    # Setup checkpoint
    ckpt = CheckpointManager(args.checkpoint, args.checkpoint_interval)
    resume_from = ckpt.load()

    # Setup output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Result columns
    fieldnames = ["idx", "m", "n", "k", "batch", "a_dtype", "b_dtype",
                  "out_dtype", "a_trans", "b_trans", "status",
                  "mt_m", "mt_n", "mt_k", "mi_m", "mi_n", "mi_k",
                  "occupancy", "predicted_latency_us", "selection_time_ms",
                  "n_configs_evaluated", "error"]

    # If resuming, append; otherwise write fresh
    write_mode = "a" if resume_from >= args.start_idx else "w"
    write_header = (write_mode == "w")

    # SIGTERM handler for graceful preemption
    def sigterm_handler(signum, frame):
        print(f"\n[SIGTERM] Preempted! Flushing checkpoint at idx "
              f"{ckpt.last_completed}...")
        ckpt.flush()
        sys.exit(143)  # 128 + 15

    signal.signal(signal.SIGTERM, sigterm_handler)

    # Config cache (keyed by dtype)
    configs_cache = {}

    # Process
    t_start = time.monotonic()
    n_processed = 0
    n_errors = 0

    with open(out_path, write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for i, row in enumerate(problems):
            global_idx = args.start_idx + i
            # Skip already-checkpointed problems
            if global_idx <= resume_from:
                continue

            result = process_problem(row, hardware, configs_cache,
                                     fallback_idx=global_idx)
            writer.writerow(result)
            n_processed += 1

            if result.get("status") == "error":
                n_errors += 1

            ckpt.update(global_idx)

            # Progress reporting every 1000 problems
            if n_processed % 1000 == 0:
                elapsed = time.monotonic() - t_start
                rate = n_processed / elapsed if elapsed > 0 else 0
                print(f"[BENCH] {n_processed}/{len(problems)} problems "
                      f"({rate:.1f}/s, {n_errors} errors)")

    # Final flush
    ckpt.flush()

    elapsed = time.monotonic() - t_start
    rate = n_processed / elapsed if elapsed > 0 else 0
    print(f"\n[BENCH] Complete: {n_processed} problems in {elapsed:.1f}s "
          f"({rate:.1f}/s)")
    print(f"[BENCH] Errors: {n_errors}/{n_processed}")
    print(f"[BENCH] Output: {args.output}")
    print(f"[BENCH] Checkpoint: {args.checkpoint}")

    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
