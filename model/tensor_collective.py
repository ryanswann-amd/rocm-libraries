"""Tensor Collective layer — shape-aware frontend over the Collective layer.

This is the top of the three-layer architecture:

    Tensor Collective layer    (shape-aware: tensor + op + dim + W)
        ↓ lowers via convention table
    Collective layer            (byte-level: primitive + msg_bytes + W + NCH)
        ↓ lowers via CollectiveLayout
    Workgroup layer             (heterogeneous WG graph) — not yet built

A `predict_tensor_collective` call takes the kind of inputs a framework user
naturally has (a per-rank tensor shape, dtype, world size, gather/scatter dim)
and lowers them onto the validated byte-level `predict_row` model. The 2D
shape flows through as a `TileShape`, so cacheline padding at the
gpu_tile / gpu_timestep_tile / wg_tile levels is modeled correctly — not
just the flat byte count.

The lowering is framework-neutral: the same convention table applies to
torch.distributed, JAX, MPI, NCCL/RCCL, or any library that exposes the
standard collective APIs. Framework-specific measurement scripts (e.g.,
microbench/sweep_torch_shapes.py) write into the same neutral CSV that
this module's predictions are validated against.
"""

from dataclasses import dataclass
from math import prod
from typing import Optional, Tuple, Union

from .collective import predict_row
from .hardware import MI300X, MI300X_COMM, CommHardware, Hardware
from .heuristics import DEFAULT_HEURISTICS, Heuristics
from .types import DataType, TileShape, dtype_bytes


SUPPORTED_OPS = ("all_reduce", "all_gather", "reduce_scatter", "broadcast", "all_to_all")


# ────────────────────────────────────────────────────────────────────
# Convention table — per-op wire-bytes-per-rank and msg_bytes lowering
# ────────────────────────────────────────────────────────────────────
#
# Per-rank wire bytes follow the rccl-tests / NCCL `busbw` formulas:
#   busbw = wire_factor(n) * msg / time
# where `msg` is the per-rank user buffer size (= prod(shape) * elem_bytes).
#
# `msg_bytes_for_predict_row` is the CSV's `msg_bytes` convention, which is
# what `predict_row` and the existing rccl_master_sweep.csv use. It differs
# from the per-rank buffer size only for reduce_scatter, where the CSV
# reports the *total input* (= W × per-rank output).

def _wire_factor(op: str, world_size: int) -> float:
    """Per-rank wire bytes = factor × per_rank_buffer_bytes."""
    n = world_size
    if op == "all_reduce":      return 2 * (n - 1) / n
    if op == "all_gather":      return (n - 1)
    if op == "reduce_scatter":  return (n - 1)
    if op == "broadcast":       return 1.0
    if op == "all_to_all":      return (n - 1) / n
    raise ValueError(f"unknown op: {op}")


def _msg_bytes_for_predict_row(op: str, per_rank_bytes: int, world_size: int) -> int:
    """How many bytes to feed into the CSV-convention `msg_bytes` argument."""
    if op == "reduce_scatter":
        # CSV reports total input = W × per-rank output for RS.
        return per_rank_bytes * world_size
    return per_rank_bytes


# ────────────────────────────────────────────────────────────────────
# dtype normalization
# ────────────────────────────────────────────────────────────────────

_DTYPE_ALIASES = {
    "bf16": DataType.BF16, "bfloat16": DataType.BF16,
    "fp16": DataType.FP16, "float16": DataType.FP16, "half": DataType.FP16,
    "fp32": DataType.FP32, "float32": DataType.FP32, "float": DataType.FP32,
    "fp64": DataType.FP64, "float64": DataType.FP64, "double": DataType.FP64,
    "fp8":  DataType.FP8,
    "int8": DataType.INT8,
}


def _normalize_dtype(dt) -> DataType:
    """Accept DataType, lowercase string alias, or framework dtype objects."""
    if isinstance(dt, DataType):
        return dt
    if isinstance(dt, str):
        key = dt.lower().replace("torch.", "").replace("np.", "").replace("numpy.", "")
        if key in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[key]
    # Framework dtype objects expose a string-ish repr; try to recover.
    name = getattr(dt, "name", None) or str(dt)
    key = name.lower().replace("torch.", "").replace("np.", "").replace("numpy.", "")
    if key in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[key]
    raise ValueError(f"unsupported dtype: {dt!r}")


# ────────────────────────────────────────────────────────────────────
# Shape → (M, N, split_dim) lowering
# ────────────────────────────────────────────────────────────────────

def _per_rank_shape_to_full_mn(
    input_shape: Tuple[int, ...],
    dim: int,
    world_size: int,
) -> Tuple[int, int, int]:
    """Lower a multi-dim per-rank shape into the model's 2D (M_full, N_full, split_dim).

    Rule:
      - `n_per_rank` = innermost dim (input_shape[-1]).
      - `m_per_rank` = product of all outer dims.
      - If `dim` is the innermost dim, the W replicas of the full tensor live
        along that dim → split_dim=1, N_full = n_per_rank * W, M_full = m_per_rank.
      - Otherwise (dim is an outer dim, or scalar shape), the W replicas live
        in an outer dim that flattens into M → split_dim=0,
        M_full = m_per_rank * W, N_full = n_per_rank.
    """
    if not input_shape:
        raise ValueError("input_shape must have at least one dimension")
    if any(d <= 0 for d in input_shape):
        raise ValueError(f"input_shape has non-positive entries: {input_shape}")
    if world_size < 1:
        raise ValueError(f"world_size must be ≥ 1, got {world_size}")

    rank_ndim = len(input_shape)
    # Normalize negative dim and clamp.
    norm_dim = dim if dim >= 0 else dim + rank_ndim
    if not (0 <= norm_dim < rank_ndim):
        raise ValueError(
            f"dim={dim} out of range for shape {input_shape} (ndim={rank_ndim})"
        )

    n_per_rank = input_shape[-1]
    m_per_rank = prod(input_shape) // n_per_rank
    last_dim = rank_ndim - 1

    if norm_dim == last_dim:
        return m_per_rank, n_per_rank * world_size, 1
    return m_per_rank * world_size, n_per_rank, 0


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TensorCollectivePrediction:
    """Result of `predict_tensor_collective`.

    The bare `predicted_us` matches `predict_row`'s scalar return *plus*
    any framework overhead floor (see `framework`/`heuristics`). The
    extra fields make the lowering and any added overhead inspectable —
    useful for dashboards, debugging, and any tooling that needs to
    attribute cost to a particular convention, shape, or frontend.
    """
    predicted_us: float
    op: str
    input_shape: Tuple[int, ...]
    dim: int
    world_size: int
    nchannels: int
    dtype: DataType
    # Derived
    per_rank_bytes: int
    wire_bytes_per_rank: int
    msg_bytes: int
    gpu_tile: TileShape
    # Frontend bookkeeping
    framework: str = "raw"
    framework_overhead_us: float = 0.0

    @property
    def backend_us(self) -> float:
        """Predicted latency excluding any framework overhead floor."""
        return self.predicted_us - self.framework_overhead_us

    def __float__(self) -> float:
        return self.predicted_us


def predict_tensor_collective(
    op: str,
    input_shape: Tuple[int, ...],
    dtype: Union[DataType, str],
    world_size: int,
    *,
    dim: int = 0,
    nchannels: int = 32,
    hw: Hardware = MI300X,
    comm_hw: CommHardware = MI300X_COMM,
    framework: str = "raw",
    heuristics: Heuristics = DEFAULT_HEURISTICS,
) -> TensorCollectivePrediction:
    """Predict the latency of a tensor-shape collective on `(input_shape, dtype)`.

    `input_shape` follows the standard framework convention for each op:
      - all_reduce:     per-rank buffer (every rank has the full tensor)
      - all_gather:     per-rank send tensor   (output = shape × W along `dim`)
      - reduce_scatter: per-rank output tensor (input  = shape × W along `dim`)
      - broadcast:      the broadcast tensor   (all ranks end with this shape)
      - all_to_all:     per-rank buffer split into W parts along `dim`

    `framework` selects the host-side overhead floor pulled from
    `heuristics.framework_overhead_ns`. Use `"raw"` (default) for bare
    rccl-tests / ncclbench numbers, `"torch"` to add the
    `torch.distributed` floor (~400 µs on MI300X), etc. Unknown names
    add 0 µs.

    Returns a `TensorCollectivePrediction` whose `predicted_us` is the
    per-rank latency estimate *including* the framework overhead;
    `.backend_us` exposes the underlying backend-only prediction.
    """
    if op not in SUPPORTED_OPS:
        raise ValueError(
            f"unsupported op {op!r}; supported: {SUPPORTED_OPS}"
        )

    dtype_enum = _normalize_dtype(dtype)
    eb = dtype_bytes(dtype_enum)

    if world_size < 1:
        raise ValueError(f"world_size must be ≥ 1, got {world_size}")

    overhead_us = heuristics.framework_overhead_us(framework)

    # Degenerate: single rank is a no-op for every collective. We still
    # apply framework overhead because frameworks pay it even for W=1.
    if world_size == 1:
        per_rank_bytes = prod(input_shape) * eb
        gpu_tile = TileShape(
            m=prod(input_shape) // input_shape[-1],
            n=input_shape[-1],
            dtype=dtype_enum,
            split_dim=0,
        )
        return TensorCollectivePrediction(
            predicted_us=overhead_us, op=op, input_shape=tuple(input_shape),
            dim=dim, world_size=1, nchannels=nchannels, dtype=dtype_enum,
            per_rank_bytes=per_rank_bytes, wire_bytes_per_rank=0,
            msg_bytes=per_rank_bytes, gpu_tile=gpu_tile,
            framework=framework, framework_overhead_us=overhead_us,
        )

    per_rank_elements = prod(input_shape)
    per_rank_bytes = per_rank_elements * eb

    wire_bytes_per_rank = int(_wire_factor(op, world_size) * per_rank_bytes)

    M_full, N_full, split_dim = _per_rank_shape_to_full_mn(
        input_shape, dim, world_size
    )
    msg_bytes = _msg_bytes_for_predict_row(op, per_rank_bytes, world_size)

    backend_us = predict_row(
        primitive=op,
        msg_bytes=msg_bytes,
        world_size=world_size,
        nchannels=nchannels,
        hw=hw, comm_hw=comm_hw,
        M=M_full, N=N_full, split_dim=split_dim,
    )

    predicted_us = backend_us + overhead_us

    gpu_tile = TileShape(
        m=M_full // world_size if split_dim == 0 else M_full,
        n=N_full // world_size if split_dim == 1 else N_full,
        dtype=dtype_enum,
        split_dim=split_dim,
    )

    return TensorCollectivePrediction(
        predicted_us=predicted_us,
        op=op,
        input_shape=tuple(input_shape),
        dim=dim,
        world_size=world_size,
        nchannels=nchannels,
        dtype=dtype_enum,
        per_rank_bytes=per_rank_bytes,
        wire_bytes_per_rank=wire_bytes_per_rank,
        msg_bytes=msg_bytes,
        gpu_tile=gpu_tile,
        framework=framework,
        framework_overhead_us=overhead_us,
    )
