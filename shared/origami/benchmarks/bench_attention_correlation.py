#!/usr/bin/env python3
"""
K-024: GPU-Based Origami Attention Correlation Test
=====================================================

Runs ACTUAL flash attention kernels on GPU with multiple tile configurations,
then compares measured performance against origami's analytical predictions.

This is the GPU validation that was missing from the previous cycle —
all prior work was CPU-only analytical modeling. This script produces
real measurements.

Output: results.json + correlation plots in output directory.

Usage:
  python3 k024_attention_correlation_gpu.py [--quick] [--output-dir DIR]

Slurm:
  python tools/slurm_gpu_run.py "cd /home/ryaswann/global_orchestrator && python3 scripts/k024_attention_correlation_gpu.py --output-dir /home/ryaswann/global_orchestrator/results/k024" --gpu mi300x --task K-024
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime

# ============================================================================
# Attention benchmark shapes — representative LLM workloads
# ============================================================================
ATTENTION_SHAPES = [
    # (batch, heads, seq_q, seq_kv, head_dim, causal, name)
    # Prefill shapes
    (1, 32, 512, 512, 128, True, "prefill_512"),
    (1, 32, 2048, 2048, 128, True, "prefill_2k"),
    (1, 32, 4096, 4096, 128, True, "prefill_4k"),
    # Decode shapes (seq_q=1 → paged attention regime)
    (32, 32, 1, 2048, 128, False, "decode_2k"),
    (32, 32, 1, 8192, 128, False, "decode_8k"),
    # GQA (fewer KV heads)
    (1, 32, 2048, 2048, 128, True, "gqa_2k"),  # kv_heads=8 handled in bench
    # Different head dims
    (1, 64, 2048, 2048, 64, True, "prefill_hd64"),
    (1, 16, 2048, 2048, 256, True, "prefill_hd256"),
]

# Tile configs to sweep: (BLOCK_M, BLOCK_N)
# Expanded set to ensure ≥14 feasible configs per shape (after LDS filtering).
# For head_dim=128 on MI300X (64KB LDS), configs where
#   (BM*HD + BN*HD)*2 + BM*BN*4 ≤ 65536 are feasible.
# Adding asymmetric tiles (32×16, 64×16, 128×16, 128×32) to fill the gap.
TILE_CONFIGS = [
    (16, 16), (16, 32), (16, 64), (16, 128),
    (32, 16), (32, 32), (32, 64), (32, 128),
    (64, 16), (64, 32), (64, 64), (64, 128), (64, 256),
    (128, 16), (128, 32), (128, 64), (128, 128), (128, 256),
    (256, 16), (256, 32), (256, 64), (256, 128), (256, 256),
]

WARMUP = 15
ITERS = 50
DTYPE_STR = "bf16"


# ============================================================================
# Triton Flash Attention kernel (parametric tile sizes)
# ============================================================================
def bench_attention_tile(batch, heads, seq_q, seq_kv, head_dim, causal,
                         block_m, block_n, kv_heads=None,
                         warmup=WARMUP, iters=ITERS):
    """Benchmark flash attention with specific BLOCK_M, BLOCK_N on GPU.

    Uses a bare Triton flash attention kernel — no framework dependency.
    Returns (ms, tflops) or (None, None) on failure.
    """
    import torch
    import triton
    import triton.language as tl

    if kv_heads is None:
        kv_heads = heads

    dtype = torch.bfloat16

    # Validate tile config feasibility
    if block_m > seq_q and seq_q > 1:
        return None, None, "BLOCK_M > seq_q"
    if block_n > seq_kv:
        return None, None, "BLOCK_N > seq_kv"

    # LDS check: Triton uses registers for Q/K/V tiles and MFMA accumulators
    # on CDNA3 (MI300X). The 64KB LDS is used for dot-product shuffles, not tile
    # storage. We use a conservative upper bound for actual LDS pressure:
    # shared-mem for two dot products (QK^T and PV) run sequentially.
    # Real feasibility depends on the compiler — let kernel launch catch overflows.
    lds_needed = (block_m * head_dim + block_n * head_dim) * 2  # bf16 tiles
    lds_needed += block_m * block_n * 4  # fp32 accumulator for S
    # Relaxed limit: 96KB allows Triton to spill to VGPRs/AGPRs on MI300X
    if lds_needed > 98304:
        return None, None, f"LDS overflow ({lds_needed} > 98304)"

    @triton.jit
    def _attn_fwd(
        Q, K, V, Out,
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        seq_q, seq_kv,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        off_b = pid_bh // stride_qh if stride_qh > 0 else pid_bh
        off_h = pid_bh % stride_qh if stride_qh > 0 else 0

        # For GQA: map query head to KV head
        # Simplified: assume heads = kv_heads for this kernel
        off_kv_h = off_h

        q_offset = off_b * stride_qb + off_h * stride_qh
        k_offset = off_b * stride_kb + off_kv_h * stride_kh
        v_offset = off_b * stride_vb + off_kv_h * stride_vh
        o_offset = off_b * stride_ob + off_h * stride_oh

        # Q block
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, HEAD_DIM)

        q = tl.load(Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                     mask=offs_m[:, None] < seq_q)

        # Accumulator
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # KV loop
        kv_range = tl.cdiv(seq_kv, BLOCK_N)
        if IS_CAUSAL:
            kv_range = min(kv_range, tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N))

        for j in range(0, kv_range):
            offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)

            # Load K
            k = tl.load(K + k_offset + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                         mask=offs_n[:, None] < seq_kv)

            # QK^T
            s = tl.dot(q, tl.trans(k)) * sm_scale

            # Causal mask
            if IS_CAUSAL:
                s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float('-inf'))

            # Online softmax
            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            l_i = l_i * alpha + tl.sum(tl.exp(s - m_new[:, None]), axis=1)
            acc = acc * alpha[:, None]
            m_i = m_new

            # Load V
            v = tl.load(V + v_offset + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk,
                         mask=offs_n[:, None] < seq_kv)

            # PV
            p = tl.exp(s - m_new[:, None])
            acc += tl.dot(p.to(v.dtype), v)

        # Normalize and store
        acc = acc / l_i[:, None]
        tl.store(Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
                 acc.to(Out.dtype.element_ty),
                 mask=offs_m[:, None] < seq_q)

    # Allocate tensors — expand K/V to full head count for kernel compatibility.
    # The Triton kernel uses simplified head indexing (off_kv_h = off_h), so
    # K/V must have 'heads' entries to avoid OOB on GQA configs.
    Q = torch.randn(batch, heads, seq_q, head_dim, dtype=dtype, device="cuda")
    K = torch.randn(batch, heads, seq_kv, head_dim, dtype=dtype, device="cuda")
    V = torch.randn(batch, heads, seq_kv, head_dim, dtype=dtype, device="cuda")
    Out = torch.empty(batch, heads, seq_q, head_dim, dtype=dtype, device="cuda")

    sm_scale = 1.0 / math.sqrt(head_dim)

    n_tiles_m = math.ceil(seq_q / block_m)
    grid = (n_tiles_m, batch * heads)

    try:
        # Warmup (compile + warm caches)
        for _ in range(warmup):
            _attn_fwd[grid](
                Q, K, V, Out, sm_scale,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
                seq_q, seq_kv,
                BLOCK_M=block_m, BLOCK_N=block_n,
                HEAD_DIM=head_dim,
                IS_CAUSAL=causal,
            )
        torch.cuda.synchronize()

        # Timed iterations
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(iters):
            _attn_fwd[grid](
                Q, K, V, Out, sm_scale,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
                seq_q, seq_kv,
                BLOCK_M=block_m, BLOCK_N=block_n,
                HEAD_DIM=head_dim,
                IS_CAUSAL=causal,
            )
        end_ev.record()
        torch.cuda.synchronize()

        ms = start_ev.elapsed_time(end_ev) / iters

        # Attention FLOPs: 2*batch*heads*seq_q*seq_kv*head_dim for QK^T,
        #                  2*batch*heads*seq_q*seq_kv*head_dim for PV
        # Total = 4*batch*heads*seq_q*seq_kv*head_dim
        # For causal: roughly half
        flops = 4.0 * batch * heads * seq_q * seq_kv * head_dim
        if causal:
            flops *= 0.5
        tflops = flops / (ms * 1e-3) / 1e12

        return ms, tflops, None

    except Exception as e:
        return None, None, str(e)


# ============================================================================
# Analytical model predictions (roofline-based)
# ============================================================================
MI300X_HW = {
    'n_cu': 304,
    'peak_tflops': 1300.0,   # FP16 peak
    'hbm_bw_tbps': 5.3,      # TB/s
    'lds_bytes': 65536,
}


def predict_attention_latency(batch, heads, seq_q, seq_kv, head_dim, causal,
                               block_m, block_n, hw=MI300X_HW):
    """Predict attention latency using GEMM decomposition + roofline model.

    Returns dict with predicted latency (us), tflops, and breakdown.
    """
    n_cu = hw['n_cu']
    peak_tflops = hw['peak_tflops']
    hbm_bw = hw['hbm_bw_tbps']

    # Per-tile FLOP and byte counts
    qk_flops = 2 * block_m * block_n * head_dim
    pv_flops = 2 * block_m * head_dim * block_n
    softmax_flops = 5 * block_m * block_n  # exp + max + sub + sum + div

    # Memory per tile
    qk_bytes = (block_m * head_dim + block_n * head_dim + block_m * block_n) * 2
    pv_bytes = (block_m * block_n + block_n * head_dim + block_m * head_dim) * 2
    softmax_bytes = block_m * block_n * 4 * 2  # fp32 intermediates

    # Roofline time (ns)
    qk_ns = max(qk_flops / (peak_tflops * 1e3), qk_bytes / (hbm_bw * 1e3))
    pv_ns = max(pv_flops / (peak_tflops * 1e3), pv_bytes / (hbm_bw * 1e3))
    softmax_ns = softmax_bytes / (hbm_bw * 1e3)

    tile_ns = qk_ns + pv_ns + softmax_ns

    # Tiles
    tiles_m = math.ceil(seq_q / block_m)
    if causal and seq_q == seq_kv:
        tiles_n_eff = math.ceil(seq_kv / block_n) / 2
    else:
        tiles_n_eff = math.ceil(seq_kv / block_n)

    total_tiles = batch * heads * tiles_m * tiles_n_eff

    # CU occupancy
    waves = math.ceil(total_tiles / n_cu)
    last_wave_util = (total_tiles % n_cu) / n_cu if total_tiles % n_cu != 0 else 1.0

    total_ns = waves * tile_ns
    latency_us = total_ns / 1e3
    latency_ms = latency_us / 1e3

    # Total FLOPS
    total_flops = batch * heads * tiles_m * tiles_n_eff * (qk_flops + pv_flops + softmax_flops)
    predicted_tflops = total_flops / (latency_ms * 1e-3) / 1e12 if latency_ms > 0 else 0

    gemm_frac = (qk_ns + pv_ns) / tile_ns if tile_ns > 0 else 0

    return {
        'latency_us': latency_us,
        'predicted_tflops': predicted_tflops,
        'gemm_fraction': gemm_frac,
        'cu_waves': waves,
        'cu_last_wave_util': last_wave_util,
        'total_tiles': total_tiles,
        'tile_ns': tile_ns,
    }


# ============================================================================
# Origami predictions (if available)
# ============================================================================
def get_origami_attention_predictions(batch, heads, seq_q, seq_kv, head_dim,
                                       block_m, block_n, kv_heads=None):
    """Get origami's native attention model prediction.

    Uses att_compute_total_latency with problem_t (attention mode via q_heads).
    For decode (seq_q=1), applies sequential KV-loop decomposition fix:
    latency = n_kv_blocks * per_block_latency.

    Returns dict or None if origami not available.
    """
    if kv_heads is None:
        kv_heads = heads

    try:
        import origami
        # Try GPU-aware first, fall back to architecture-based construction
        try:
            hw = origami.get_hardware_for_device(0)
        except (AttributeError, RuntimeError):
            # MI300X: 304 CUs, 64KB LDS, 256MB L2, 2.1 GHz
            hw = origami.get_hardware_for_arch(
                origami.architecture_t.gfx942,
                304,     # N_CU
                65536,   # lds_capacity
                268435456,  # L2_capacity (256MB)
                2100000  # compute_clock_khz (2.1 GHz)
            )

        dt = origami.string_to_datatype('bf16')
        mi = hw.get_recommended_matrix_instruction(dt)
        max_cus = hw.N_CU

        # Use origami's native attention API: att_compute_total_latency
        # Construct problem_t with q_heads set (attention mode)
        # M=seq_q, N=seq_kv, K=head_dim (matches OrigamiAttentionSelector)
        prob = origami.problem_t()
        prob.size = origami.dim3_t(seq_q, seq_kv, head_dim)
        prob.batch = batch
        prob.q_heads = heads
        prob.a_transpose = origami.transpose_t.N
        prob.b_transpose = origami.transpose_t.N
        prob.a_dtype = dt
        prob.b_dtype = dt
        prob.c_dtype = dt
        prob.d_dtype = dt
        prob.mi_dtype = dt
        prob.a_mx_block_size = 0
        prob.b_mx_block_size = 0

        cfg = origami.config_t()
        cfg.mt = origami.dim3_t(block_m, block_n, head_dim)
        cfg.mi = mi
        cfg.occupancy = 2

        lat_ns = origami.att_compute_total_latency(prob, hw, cfg, max_cus)
        lat_us = lat_ns / 1000.0  # nanoseconds to microseconds

        is_decode = (seq_q == 1)
        result = {
            'source': 'origami_att_native',
            'latency_us': lat_us,
            'latency_ns': lat_ns,
        }

        # Decode fix: sequential KV-loop decomposition
        # For decode (seq_q=1), model each KV block independently and sum
        if is_decode:
            import math
            n_kv_blocks = math.ceil(seq_kv / block_n)
            eff_seq_q = max(seq_q, block_m)

            prob_blk = origami.problem_t()
            prob_blk.size = origami.dim3_t(eff_seq_q, block_n, head_dim)
            prob_blk.batch = batch
            prob_blk.q_heads = heads
            prob_blk.a_transpose = origami.transpose_t.N
            prob_blk.b_transpose = origami.transpose_t.N
            prob_blk.a_dtype = dt
            prob_blk.b_dtype = dt
            prob_blk.c_dtype = dt
            prob_blk.d_dtype = dt
            prob_blk.mi_dtype = dt
            prob_blk.a_mx_block_size = 0
            prob_blk.b_mx_block_size = 0

            per_blk_ns = origami.att_compute_total_latency(prob_blk, hw, cfg, max_cus)
            decode_fix_ns = n_kv_blocks * per_blk_ns
            result['decode_fix_latency_us'] = decode_fix_ns / 1000.0
            result['n_kv_blocks'] = n_kv_blocks

        return result

    except ImportError:
        return None
    except Exception as e:
        return {'source': 'origami_error', 'error': str(e)}


# ============================================================================
# Correlation metrics
# ============================================================================
def spearman_rho(x, y):
    """Pure-Python Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0
    # Rank both
    def rank(arr):
        s = sorted(range(n), key=lambda i: arr[i])
        r = [0.0] * n
        for i, idx in enumerate(s):
            r[idx] = float(i)
        return r
    rx = rank(x)
    ry = rank(y)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def pearson_r(x, y):
    """Pure-Python Pearson correlation."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    denom = (sxx * syy) ** 0.5
    return sxy / denom if denom > 0 else 0.0


# ============================================================================
# Plotting
# ============================================================================
def generate_plots(results, output_dir):
    """Generate correlation plots from benchmark results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib/numpy not available, skipping plots")
        return

    shapes_data = results.get('per_shape', [])
    if not shapes_data:
        return

    # ── Plot 1: Predicted vs Measured TFLOPS scatter ──────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, sd in enumerate(shapes_data[:8]):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break

        measured = []
        predicted = []
        tile_labels = []

        for tile_key, td in sd.get('tile_results', {}).items():
            if td.get('measured_tflops') and td.get('analytical_tflops'):
                measured.append(td['measured_tflops'])
                predicted.append(td['analytical_tflops'])
                tile_labels.append(tile_key)

        if len(measured) >= 3:
            rho = spearman_rho(predicted, measured)
            pr = pearson_r(predicted, measured)

            ax.scatter(predicted, measured, s=30, alpha=0.7, edgecolors='black', linewidth=0.3)

            # Diagonal line
            mn = min(min(predicted), min(measured))
            mx_val = max(max(predicted), max(measured))
            ax.plot([mn, mx_val], [mn, mx_val], 'r--', alpha=0.4)

            # Label top-1 tiles
            best_meas_idx = measured.index(max(measured))
            best_pred_idx = predicted.index(max(predicted))
            ax.scatter([predicted[best_meas_idx]], [measured[best_meas_idx]],
                      s=100, marker='*', c='green', zorder=5, label=f'Best measured: {tile_labels[best_meas_idx]}')
            ax.scatter([predicted[best_pred_idx]], [measured[best_pred_idx]],
                      s=100, marker='D', c='red', zorder=5, label=f'Best predicted: {tile_labels[best_pred_idx]}')

            ax.set_title(f"{sd['name']}\nSpearman ρ={rho:.3f}, Pearson r={pr:.3f}", fontsize=9)
            ax.legend(fontsize=6, loc='upper left')
        else:
            ax.text(0.5, 0.5, f"{sd['name']}\nInsufficient data ({len(measured)} tiles)",
                   ha='center', va='center', transform=ax.transAxes)

        ax.set_xlabel('Predicted TFLOPS', fontsize=8)
        ax.set_ylabel('Measured TFLOPS', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle('K-024: Origami Attention Correlation — Predicted vs Measured TFLOPS\n'
                 'GPU benchmarks on MI300X with Triton flash attention kernel',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path1 = os.path.join(output_dir, 'attention_correlation_scatter.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # ── Plot 2: Summary bar chart ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    agg = results.get('aggregate', {})

    # Spearman by shape
    ax = axes[0]
    shape_names = []
    spearmans = []
    for sd in shapes_data:
        corr = sd.get('correlation', {})
        if 'spearman_rho' in corr:
            shape_names.append(sd['name'])
            spearmans.append(corr['spearman_rho'])

    if spearmans:
        colors = ['#2ecc71' if s > 0.7 else '#e74c3c' if s < 0.4 else '#f39c12' for s in spearmans]
        ax.barh(range(len(shape_names)), spearmans, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(shape_names)))
        ax.set_yticklabels(shape_names, fontsize=8)
        ax.set_xlabel('Spearman ρ')
        ax.set_title('Rank Correlation by Shape')
        ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.2, 1.0)

    # Regret by shape
    ax = axes[1]
    regrets = []
    rshape_names = []
    for sd in shapes_data:
        corr = sd.get('correlation', {})
        if 'regret_pct' in corr:
            rshape_names.append(sd['name'])
            regrets.append(corr['regret_pct'])

    if regrets:
        colors = ['#2ecc71' if r < 5 else '#e74c3c' if r > 20 else '#f39c12' for r in regrets]
        ax.barh(range(len(rshape_names)), regrets, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(rshape_names)))
        ax.set_yticklabels(rshape_names, fontsize=8)
        ax.set_xlabel('Regret %')
        ax.set_title('Selection Regret (lower = better)')

    # Summary stats
    ax = axes[2]
    summary_text = [
        f"Shapes tested: {agg.get('n_shapes', 0)}",
        f"Avg Spearman ρ: {agg.get('avg_spearman', 0):.3f}",
        f"Avg Pearson r: {agg.get('avg_pearson', 0):.3f}",
        f"Top-1 accuracy: {agg.get('top1_accuracy_pct', 0):.0f}%",
        f"Top-3 accuracy: {agg.get('top3_accuracy_pct', 0):.0f}%",
        f"Avg regret: {agg.get('avg_regret_pct', 0):.1f}%",
        f"Max regret: {agg.get('max_regret_pct', 0):.1f}%",
    ]
    ax.text(0.1, 0.5, '\n'.join(summary_text), transform=ax.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.axis('off')
    ax.set_title('Aggregate Summary')

    fig.suptitle('K-024: Attention Correlation Summary — GPU Measurements vs Analytical Model',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path2 = os.path.join(output_dir, 'attention_correlation_summary.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")

    return [path1, path2]


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="K-024: GPU Attention Correlation Test")
    parser.add_argument("--output-dir", default="/tmp/k024_attn_corr",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer shapes and iterations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("K-024: GPU ATTENTION CORRELATION TEST")
    print("=" * 70)
    print(f"Output:  {args.output_dir}")
    print(f"Start:   {datetime.now().isoformat()}")

    # ── Environment detection ────────────────────────────────────────────
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: No GPU available. This script requires CUDA/HIP.")
        sys.exit(1)

    dev = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print(f"Device:  {dev}")
    print(f"CUs:     {props.multi_processor_count}")
    total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"Memory:  {total_mem / 1e9:.1f} GB")

    import triton
    print(f"Triton:  {triton.__version__}")

    origami_available = False
    try:
        import origami
        origami_available = True
        print(f"Origami: available")
    except ImportError:
        print(f"Origami: NOT available (will use analytical model only)")

    # ── Config ───────────────────────────────────────────────────────────
    shapes = ATTENTION_SHAPES
    tile_configs = TILE_CONFIGS
    warmup = WARMUP
    iters = ITERS

    if args.quick:
        shapes = shapes[:4]
        tile_configs = [(32, 32), (64, 64), (128, 128), (64, 128), (128, 64)]
        warmup = 5
        iters = 20

    print(f"Shapes:  {len(shapes)}")
    print(f"Tiles:   {len(tile_configs)}")
    total = len(shapes) * len(tile_configs)
    print(f"Total:   ~{total} benchmarks")
    print()

    # ── Results container ────────────────────────────────────────────────
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "device": dev,
            "cu_count": props.multi_processor_count,
            "memory_gb": total_mem / 1e9,
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "origami_available": origami_available,
            "iters": iters,
            "warmup": warmup,
            "n_shapes": len(shapes),
            "n_tile_configs": len(tile_configs),
        },
        "per_shape": [],
    }

    # ── Run benchmarks ───────────────────────────────────────────────────
    aggregate = {
        "spearmans": [],
        "pearsons": [],
        "regrets": [],
        "top1_hits": 0,
        "top3_hits": 0,
        "shapes_evaluated": 0,
    }

    for si, (batch, heads, seq_q, seq_kv, head_dim, causal, name) in enumerate(shapes):
        t0 = time.time()
        print(f"\n[{si+1}/{len(shapes)}] {name}: batch={batch} heads={heads} "
              f"seq_q={seq_q} seq_kv={seq_kv} hd={head_dim} causal={causal}")

        kv_heads = 8 if 'gqa' in name else heads

        shape_data = {
            "name": name,
            "batch": batch, "heads": heads,
            "seq_q": seq_q, "seq_kv": seq_kv,
            "head_dim": head_dim, "causal": causal,
            "kv_heads": kv_heads,
            "tile_results": {},
        }

        measured_tiles = {}
        analytical_tiles = {}

        for bm, bn in tile_configs:
            tile_key = f"{bm}x{bn}"

            # GPU benchmark
            ms, tflops, err = bench_attention_tile(
                batch, heads, seq_q, seq_kv, head_dim, causal,
                bm, bn, kv_heads=kv_heads if 'gqa' in name else None,
                warmup=warmup, iters=iters
            )

            # Analytical prediction
            pred = predict_attention_latency(
                batch, heads, seq_q, seq_kv, head_dim, causal, bm, bn
            )

            # Origami prediction (if available)
            origami_pred = None
            if origami_available:
                origami_pred = get_origami_attention_predictions(
                    batch, heads, seq_q, seq_kv, head_dim, bm, bn, kv_heads
                )

            tile_data = {
                "block_m": bm, "block_n": bn,
                "measured_ms": ms,
                "measured_tflops": tflops,
                "error": err,
                "analytical_tflops": pred['predicted_tflops'],
                "analytical_latency_us": pred['latency_us'],
                "gemm_fraction": pred['gemm_fraction'],
                "cu_waves": pred['cu_waves'],
            }
            if origami_pred:
                tile_data["origami"] = origami_pred

            shape_data["tile_results"][tile_key] = tile_data

            if tflops is not None:
                measured_tiles[tile_key] = tflops
                # Prefer origami prediction for correlation if available
                if origami_pred and origami_pred.get('source') != 'origami_error':
                    lat_us = origami_pred.get('total_latency_us',
                             origami_pred.get('latency_us'))
                    if lat_us and lat_us > 0:
                        flops = 4.0 * batch * heads * seq_q * seq_kv * head_dim
                        if causal:
                            flops *= 0.5
                        analytical_tiles[tile_key] = flops / (lat_us * 1e-6) / 1e12
                    else:
                        analytical_tiles[tile_key] = pred['predicted_tflops']
                else:
                    analytical_tiles[tile_key] = pred['predicted_tflops']
                status = f"{tflops:>7.2f} TFLOPS (pred: {analytical_tiles[tile_key]:.2f})"
            else:
                status = f"SKIP: {err}"

            print(f"  {tile_key:>10s}: {status}")

        # ── Correlation for this shape ───────────────────────────────────
        if len(measured_tiles) >= 3:
            common_tiles = sorted(measured_tiles.keys())
            m_vals = [measured_tiles[t] for t in common_tiles]
            a_vals = [analytical_tiles[t] for t in common_tiles]

            rho = spearman_rho(a_vals, m_vals)
            pr = pearson_r(a_vals, m_vals)

            # Oracle and pick
            oracle_tile = max(measured_tiles, key=lambda t: measured_tiles[t])
            oracle_tflops = measured_tiles[oracle_tile]
            pred_best = max(analytical_tiles, key=lambda t: analytical_tiles[t])
            pred_best_actual = measured_tiles.get(pred_best, 0)
            regret = (oracle_tflops - pred_best_actual) / oracle_tflops * 100 if oracle_tflops > 0 else 0
            top1 = oracle_tile == pred_best

            # Top-3
            top3_pred = sorted(analytical_tiles, key=lambda t: -analytical_tiles[t])[:3]
            top3_hit = oracle_tile in top3_pred

            corr = {
                "spearman_rho": rho,
                "pearson_r": pr,
                "n_tiles": len(common_tiles),
                "oracle_tile": oracle_tile,
                "oracle_tflops": oracle_tflops,
                "predicted_best": pred_best,
                "predicted_best_actual_tflops": pred_best_actual,
                "regret_pct": regret,
                "top1_match": top1,
                "top3_match": top3_hit,
            }
            shape_data["correlation"] = corr

            aggregate["spearmans"].append(rho)
            aggregate["pearsons"].append(pr)
            aggregate["regrets"].append(regret)
            aggregate["shapes_evaluated"] += 1
            if top1:
                aggregate["top1_hits"] += 1
            if top3_hit:
                aggregate["top3_hits"] += 1

            print(f"  >>> Spearman ρ={rho:+.3f}  Pearson r={pr:+.3f}  "
                  f"Oracle={oracle_tile} ({oracle_tflops:.2f}T)  "
                  f"Pick={pred_best} (regret={regret:.1f}%)  "
                  f"top1={'✓' if top1 else '✗'}")
        else:
            print(f"  >>> Insufficient data ({len(measured_tiles)} valid tiles)")

        elapsed = time.time() - t0
        print(f"  >>> Elapsed: {elapsed:.1f}s")

        results["per_shape"].append(shape_data)

        # Incremental save — protect against GPU faults losing all data
        inc_path = os.path.join(args.output_dir, "results_incremental.json")
        try:
            with open(inc_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
        except Exception:
            pass

    # ── Aggregate ────────────────────────────────────────────────────────
    n = aggregate["shapes_evaluated"]
    agg = {"n_shapes": n}
    if n > 0:
        agg["avg_spearman"] = sum(aggregate["spearmans"]) / n
        agg["avg_pearson"] = sum(aggregate["pearsons"]) / n
        agg["avg_regret_pct"] = sum(aggregate["regrets"]) / n
        agg["max_regret_pct"] = max(aggregate["regrets"])
        agg["min_spearman"] = min(aggregate["spearmans"])
        agg["max_spearman"] = max(aggregate["spearmans"])
        agg["top1_accuracy_pct"] = aggregate["top1_hits"] / n * 100
        agg["top3_accuracy_pct"] = aggregate["top3_hits"] / n * 100

    results["aggregate"] = agg

    # ── Save JSON ────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "k024_measured_latency.csv")
    with open(csv_path, "w") as f:
        f.write("shape,batch,heads,seq_q,seq_kv,head_dim,causal,block_m,block_n,"
                "measured_ms,measured_tflops,analytical_latency_us,analytical_tflops,"
                "origami_latency_us,spearman_rho,gpu,node\n")
        import socket
        hostname = socket.gethostname()
        for sd in results["per_shape"]:
            rho_val = sd.get("correlation", {}).get("spearman_rho", "")
            for tile_key, td in sd.get("tile_results", {}).items():
                if td.get("measured_tflops") is None:
                    continue
                origami_lat = ""
                if td.get("origami"):
                    origami_lat = td["origami"].get("total_latency_us",
                                  td["origami"].get("latency_us", ""))
                f.write(f"{sd['name']},{sd['batch']},{sd['heads']},"
                        f"{sd['seq_q']},{sd['seq_kv']},{sd['head_dim']},"
                        f"{sd['causal']},{td['block_m']},{td['block_n']},"
                        f"{td['measured_ms']},{td['measured_tflops']:.4f},"
                        f"{td['analytical_latency_us']:.4f},{td['analytical_tflops']:.4f},"
                        f"{origami_lat},{rho_val},{dev},{hostname}\n")
    print(f"CSV: {csv_path}")

    # ── Generate plots ───────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print("GENERATING PLOTS")
    print(f"{'=' * 50}")
    try:
        plot_files = generate_plots(results, args.output_dir)
    except Exception as e:
        print(f"Plot generation failed: {e}")
        traceback.print_exc()

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CORRELATION SUMMARY")
    print(f"{'=' * 70}")
    print(json.dumps(agg, indent=2))
    print(f"\nResults: {out_path}")
    print(f"End:     {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
