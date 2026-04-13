#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Generate the 262K-problem tritonblas benchmark corpus.

Produces a CSV of (m, n, k, batch, a_dtype, b_dtype, out_dtype, a_trans, b_trans)
covering the realistic tritonblas workload surface: powers-of-2, non-powers-of-2,
skinny/tall/fat shapes, batch dimensions, and mixed dtypes.

Usage:
    python3 generate_problem_set.py --output problems_262k.csv [--count 262144]
"""

import argparse
import csv
import itertools
import random
import sys


# --- Shape generators ---

# Powers of 2 commonly seen in transformer / LLM workloads
POWERS_OF_2 = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Non-power-of-2 common in real models (vocabulary, hidden dims, etc.)
NON_POW2 = [48, 96, 112, 160, 192, 224, 320, 384, 448, 576, 640, 768,
            896, 1152, 1280, 1536, 1792, 2304, 2560, 3072, 3584, 4608,
            5120, 6144, 7168, 7680, 8960, 10240, 12288, 14336]

# Tiny/skinny dims (attention heads, batch-1 inference, etc.)
SKINNY = [1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 24]

# Supported dtypes for tritonblas
DTYPES = [
    ("f16", "f16", "f16"),
    ("bf16", "bf16", "bf16"),
    ("f8", "f8", "f16"),
    ("f8", "f8", "bf16"),
    ("f32", "f32", "f32"),
]

# Transpose combos: (a_trans, b_trans)
TRANSPOSES = [("T", "N"), ("N", "N"), ("T", "T"), ("N", "T")]

# Batch sizes
BATCHES = [1, 2, 4, 8, 16]


def generate_shape_pool():
    """Build a diverse pool of (m, n, k) shapes."""
    shapes = []

    # 1) Cubic: m=n=k from standard sizes
    for d in POWERS_OF_2 + NON_POW2:
        shapes.append((d, d, d))

    # 2) Rectangular: all combos of M,N from standard, K from standard
    m_n_pool = POWERS_OF_2[:8] + NON_POW2[:15]  # keep reasonable
    k_pool = POWERS_OF_2[:9] + NON_POW2[:10]
    for m, n in itertools.product(m_n_pool, repeat=2):
        for k in k_pool:
            shapes.append((m, n, k))

    # 3) Skinny shapes (inference-like)
    for s in SKINNY:
        for d in POWERS_OF_2[:8] + NON_POW2[:10]:
            shapes.append((s, d, d))
            shapes.append((d, s, d))
            shapes.append((d, d, s))

    # 4) Transformer-specific shapes (attention, MLP, etc.)
    # Common hidden sizes: 768, 1024, 2048, 4096, 5120, 8192, 12288
    # Seq lengths: 128, 256, 512, 1024, 2048, 4096
    hidden = [768, 1024, 2048, 4096, 5120, 8192, 12288]
    seqlen = [128, 256, 512, 1024, 2048, 4096]
    for h in hidden:
        for s in seqlen:
            shapes.append((s, h, h))       # attention QKV projection
            shapes.append((s, 4 * h, h))   # MLP up-projection
            shapes.append((s, h, 4 * h))   # MLP down-projection

    return shapes


def generate_problems(target_count=262144, seed=42):
    """Generate target_count problems with full diversity."""
    random.seed(seed)
    shapes = generate_shape_pool()
    problems = []

    # Systematic: each shape × each dtype (primary transpose T,N only)
    for m, n, k in shapes:
        for a_dt, b_dt, out_dt in DTYPES:
            problems.append((m, n, k, 1, a_dt, b_dt, out_dt, "T", "N"))

    # Add transpose and batch diversity
    for m, n, k in shapes[:2000]:
        for a_t, b_t in TRANSPOSES[1:]:  # skip T,N already covered
            problems.append((m, n, k, 1, "f16", "f16", "f16", a_t, b_t))

    for m, n, k in shapes[:1000]:
        for batch in BATCHES[1:]:  # skip 1 already covered
            problems.append((m, n, k, batch, "f16", "f16", "f16", "T", "N"))

    # Deduplicate
    seen = set(problems)
    problems = list(seen)

    # If we have more than target, downsample
    if len(problems) > target_count:
        random.shuffle(problems)
        problems = problems[:target_count]
        seen = set(problems)

    # If we need more, generate random problems to fill
    all_m = POWERS_OF_2 + NON_POW2 + SKINNY
    all_nk = POWERS_OF_2 + NON_POW2
    while len(problems) < target_count:
        m = random.choice(all_m)
        n = random.choice(all_nk)
        k = random.choice(all_nk)
        batch = random.choice(BATCHES)
        dt_triple = random.choice(DTYPES)
        trans = random.choice(TRANSPOSES)
        p = (m, n, k, batch, dt_triple[0], dt_triple[1], dt_triple[2],
             trans[0], trans[1])
        if p not in seen:
            problems.append(p)
            seen.add(p)

    random.shuffle(problems)
    return problems[:target_count]


def main():
    parser = argparse.ArgumentParser(
        description="Generate 262K-problem tritonblas benchmark corpus")
    parser.add_argument("--output", "-o", default="problems_262k.csv",
                        help="Output CSV path")
    parser.add_argument("--count", type=int, default=262144,
                        help="Target number of problems (default: 262144)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    problems = generate_problems(target_count=args.count, seed=args.seed)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "m", "n", "k", "batch", "a_dtype", "b_dtype",
                         "out_dtype", "a_trans", "b_trans"])
        for i, p in enumerate(problems):
            writer.writerow([i] + list(p))

    print(f"Generated {len(problems)} problems -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
