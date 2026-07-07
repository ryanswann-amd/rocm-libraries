"""Tests for the Tensor Collective layer.

Two-fold coverage:
  1. The convention table — wire-bytes-per-rank and msg_bytes lowering
     match the rccl-tests / NCCL busbw formulas for every op.
  2. 2D-shape parity — predicting a 2D tensor must agree with the
     equivalent flat 1D buffer when the bytes are the same, since at the
     RCCL transport level shape carries no information beyond cacheline
     padding. (Where padding diverges, the difference is small and
     accounted for by `gpu_tile.cacheline_efficiency`.)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import prod

import pytest

from model.collective import predict_row
from model.hardware import MI300X, MI300X_COMM
from model.tensor_collective import (
    SUPPORTED_OPS,
    TensorCollectivePrediction,
    _msg_bytes_for_predict_row,
    _normalize_dtype,
    _per_rank_shape_to_full_mn,
    _wire_factor,
    predict_tensor_collective,
)
from model.types import DataType, dtype_bytes


# ───────────────────────────────────────────────────────────────
# Convention table
# ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("W", [2, 4, 8, 16])
def test_wire_factor_matches_busbw_formulas(W):
    """Per-rank wire factor matches the rccl-tests busbw formula for each op.

    These are the same formulas we verified empirically against the CSV's
    `busbw_gbps` column.
    """
    assert _wire_factor("all_reduce", W)     == pytest.approx(2 * (W - 1) / W)
    assert _wire_factor("all_gather", W)     == pytest.approx(W - 1)
    assert _wire_factor("reduce_scatter", W) == pytest.approx(W - 1)
    assert _wire_factor("broadcast", W)      == pytest.approx(1.0)
    assert _wire_factor("all_to_all", W)     == pytest.approx((W - 1) / W)


@pytest.mark.parametrize("W", [2, 4, 8])
def test_msg_bytes_csv_convention(W):
    per_rank = 1024 * 1024
    for op in ("all_reduce", "all_gather", "broadcast", "all_to_all"):
        assert _msg_bytes_for_predict_row(op, per_rank, W) == per_rank
    # RS is the one outlier: CSV reports the total input.
    assert _msg_bytes_for_predict_row("reduce_scatter", per_rank, W) == per_rank * W


def test_unknown_op_raises():
    with pytest.raises(ValueError):
        _wire_factor("scatter_reduce_v2", 4)
    with pytest.raises(ValueError):
        predict_tensor_collective("not_a_real_op", (1024,), DataType.BF16, 8)


# ───────────────────────────────────────────────────────────────
# dtype normalization
# ───────────────────────────────────────────────────────────────

def test_dtype_accepts_enum():
    assert _normalize_dtype(DataType.BF16) is DataType.BF16


@pytest.mark.parametrize("alias,expected", [
    ("bf16",       DataType.BF16),
    ("bfloat16",   DataType.BF16),
    ("torch.bfloat16", DataType.BF16),
    ("fp16",       DataType.FP16),
    ("half",       DataType.FP16),
    ("fp32",       DataType.FP32),
    ("float",      DataType.FP32),
    ("float64",    DataType.FP64),
    ("fp8",        DataType.FP8),
    ("int8",       DataType.INT8),
])
def test_dtype_accepts_string_aliases(alias, expected):
    assert _normalize_dtype(alias) is expected


def test_dtype_rejects_garbage():
    with pytest.raises(ValueError):
        _normalize_dtype("complex128")


# ───────────────────────────────────────────────────────────────
# Shape → (M, N, split_dim) lowering
# ───────────────────────────────────────────────────────────────

def test_lower_outer_dim_is_split_dim_0():
    M, N, sd = _per_rank_shape_to_full_mn((64, 128, 256), dim=0, world_size=8)
    # per_rank N = 256, per_rank M = 64*128 = 8192
    # dim=0 (outermost) → split_dim=0, M_full = M_per_rank * W
    assert (M, N, sd) == (8192 * 8, 256, 0)


def test_lower_inner_dim_is_split_dim_1():
    M, N, sd = _per_rank_shape_to_full_mn((64, 128, 256), dim=-1, world_size=8)
    # dim==last → split_dim=1, N_full = N_per_rank * W
    assert (M, N, sd) == (64 * 128, 256 * 8, 1)


def test_lower_middle_dim_treated_as_outer():
    # Middle dim isn't innermost → still split_dim=0 (model is 2D, outer dims
    # all flatten into M).
    M, N, sd = _per_rank_shape_to_full_mn((64, 128, 256), dim=1, world_size=8)
    assert sd == 0
    assert N == 256


def test_lower_1d_shape():
    M, N, sd = _per_rank_shape_to_full_mn((4096,), dim=0, world_size=4)
    # 1D, dim=0 is also innermost → split_dim=1, N_full = 4096*4.
    assert (M, N, sd) == (1, 4096 * 4, 1)


def test_lower_rejects_negative_dim_out_of_range():
    with pytest.raises(ValueError):
        _per_rank_shape_to_full_mn((4, 8), dim=-3, world_size=2)


def test_lower_rejects_empty_shape():
    with pytest.raises(ValueError):
        _per_rank_shape_to_full_mn((), dim=0, world_size=2)


def test_lower_rejects_nonpositive_dims():
    with pytest.raises(ValueError):
        _per_rank_shape_to_full_mn((4, 0, 8), dim=0, world_size=2)


# ───────────────────────────────────────────────────────────────
# End-to-end predict_tensor_collective
# ───────────────────────────────────────────────────────────────

def test_returns_prediction_struct():
    p = predict_tensor_collective("all_gather", (1024, 4096), DataType.BF16, 8)
    assert isinstance(p, TensorCollectivePrediction)
    assert p.op == "all_gather"
    assert p.world_size == 8
    assert p.per_rank_bytes == 1024 * 4096 * 2
    assert p.wire_bytes_per_rank == (8 - 1) * 1024 * 4096 * 2
    # `float(p)` returns the predicted us — convenient for ad-hoc usage.
    assert float(p) == p.predicted_us


def test_world_size_1_is_zero_us():
    p = predict_tensor_collective("all_reduce", (1024, 4096), "bf16", world_size=1)
    assert p.predicted_us == 0.0
    assert p.wire_bytes_per_rank == 0


# ───────────────────────────────────────────────────────────────
# 2D-shape parity with the 1D fallback path
# ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("op", SUPPORTED_OPS)
@pytest.mark.parametrize("W", [2, 4, 8])
def test_2d_shape_matches_1d_for_aligned_buffers(op, W):
    """For shapes whose `n_per_rank * dtype` is a multiple of CACHELINE_BYTES
    and `m_per_rank` is divisible by num_wgs and chunks, the 2D path must
    produce the SAME prediction as the 1D-degenerate path used by validate.py.

    This is the safety net pinning that the Tensor Collective layer doesn't
    silently shift CCL numbers — it only adds shape carrying.
    """
    # Pick a shape where everything aligns cleanly:
    #   - per_rank bytes = 1 MB at BF16
    #   - cl_per_row, m_per_rank, num_wgs all powers of two
    dtype = DataType.BF16
    eb = dtype_bytes(dtype)
    per_rank_bytes = 1024 * 1024
    per_rank_elements = per_rank_bytes // eb

    # Compare 1D vs 2D representations of the same buffer
    pred_1d = predict_tensor_collective(
        op, (per_rank_elements,), dtype, W, dim=0, nchannels=32,
    )
    # 2D variant: (1024, per_rank_elements // 1024) with dim=0
    m_per_rank = 1024
    n_per_rank = per_rank_elements // m_per_rank
    pred_2d = predict_tensor_collective(
        op, (m_per_rank, n_per_rank), dtype, W, dim=0, nchannels=32,
    )

    assert pred_1d.wire_bytes_per_rank == pred_2d.wire_bytes_per_rank
    assert pred_1d.msg_bytes == pred_2d.msg_bytes
    assert pred_1d.predicted_us == pytest.approx(pred_2d.predicted_us, rel=0.01)


@pytest.mark.parametrize("W", [2, 4, 8])
def test_predict_tensor_collective_matches_predict_row_direct(W):
    """Calling the API for each op must agree with calling predict_row
    directly with M=0, N=0 (the path validate.py exercises)."""
    per_rank_bytes = 1024 * 1024
    nch = 32
    for op in SUPPORTED_OPS:
        msg = _msg_bytes_for_predict_row(op, per_rank_bytes, W)
        direct = predict_row(op, msg, W, nch, MI300X, MI300X_COMM)

        elements = per_rank_bytes // dtype_bytes(DataType.BF16)
        api = predict_tensor_collective(
            op, (elements,), DataType.BF16, W, dim=0, nchannels=nch,
        )
        assert api.predicted_us == pytest.approx(direct, rel=1e-6), (
            f"{op} W={W}: api={api.predicted_us}, direct={direct}"
        )


# ───────────────────────────────────────────────────────────────
# Wire-volume sanity vs RCCL conventions
# ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("op,W,shape,expected_factor", [
    # AR: 2(W-1)/W × per_rank_bytes
    ("all_reduce",     8, (1024, 2048), 2 * 7 / 8),
    # AG: (W-1) × per_rank_bytes
    ("all_gather",     4, (1024, 2048), 3),
    # RS: (W-1) × per_rank_bytes
    ("reduce_scatter", 8, (1024, 2048), 7),
    # BC: 1 × per_rank_bytes (one push per rank)
    ("broadcast",      4, (1024, 2048), 1),
    # A2A: (W-1)/W × per_rank_bytes
    ("all_to_all",     8, (1024, 2048), 7 / 8),
])
def test_wire_bytes_matches_busbw_factor(op, W, shape, expected_factor):
    eb = 2  # BF16
    per_rank_bytes = prod(shape) * eb
    p = predict_tensor_collective(op, shape, DataType.BF16, W)
    assert p.wire_bytes_per_rank == int(expected_factor * per_rank_bytes)
