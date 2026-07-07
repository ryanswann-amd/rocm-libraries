from dataclasses import dataclass, field, replace
from enum import Enum, auto
from math import ceil
from typing import List, Dict, Optional

from .heuristics import DEFAULT_HEURISTICS


class DataType(Enum):
    FP16 = auto()
    BF16 = auto()
    FP32 = auto()
    FP64 = auto()
    FP8 = auto()
    INT8 = auto()

DTYPE_BYTES = {
    DataType.FP8: 1, DataType.INT8: 1,
    DataType.FP16: 2, DataType.BF16: 2,
    DataType.FP32: 4, DataType.FP64: 8,
}

def dtype_bytes(dt: DataType) -> int:
    return DTYPE_BYTES[dt]


class LoadWidth(Enum):
    DWORD = 4
    DWORDX4 = 16
    DWORDX16 = 64

    @property
    def bytes(self):
        return self.value

    @property
    def instrs_per_cacheline(self):
        return 64 // self.value


class Direction(Enum):
    PUSH = auto()
    PULL = auto()


class ReduceOp(Enum):
    SUM = auto()
    MAX = auto()
    MIN = auto()
    PROD = auto()


CACHELINE_BYTES = 64


@dataclass(frozen=True)
class TileShape:
    """
    A 2D rectangular tile of a row-major tensor, carried through every level
    of the tile hierarchy (gpu_tile → gpu_timestep_tile → wg_tile).

    Stores the shape (m, n), the element dtype, and two layout bits:

      `split_dim`  — which dimension of the parent tensor this tile was
                     carved out of. Mostly informational for downstream
                     consumers (e.g., the wire-volume formula in collectives).

      `contiguous` — whether the tile's rows sit in contiguous memory.
                     This determines whether per-row cacheline padding is a
                     real physical cost or just a 2D index convention.

    Two contiguity regimes:

      `contiguous=True` (default): the tile is a dense `(m × n × eb)`-byte
        block (e.g., a per-rank `torch.Tensor`, or a row-stripe of one).
        The hardware streams it as a flat byte sequence — partial cache lines
        at "row boundaries" don't exist physically because the next row starts
        in the very next byte. Use:
            cachelines  = ceil(m * n * eb / CACHELINE_BYTES)
            cacheline_shape = (1, cachelines)

      `contiguous=False`: the tile is a column stripe of a larger tensor
        (e.g., produced by an axis=1 `divide()`), so each row of the tile is
        separated from the next by sibling stripes in memory. Each row's
        payload must therefore round up to its own cacheline boundary:
            cachelines  = m * ceil(n * eb / CACHELINE_BYTES)
            cacheline_shape = (m, cl_per_row)
        This regime captures real cacheline-padding waste for strided access.

    Methods:
      divide(factor, axis)        — explicit per-axis subdivision (ceil-div).
                                    axis=0 preserves contiguity; axis=1
                                    breaks it (column stripe of parent).
      divide_byte_equal(factor)   — pick the axis that yields byte-equal chunks,
                                    preferring axis=0 (row-stripes) so cacheline
                                    alignment is preserved when possible.
    """
    m: int
    n: int
    dtype: "DataType" = field(default=None)  # type: ignore[assignment]
    split_dim: int = 0
    contiguous: bool = True

    def __post_init__(self):
        # Frozen dataclass — bypass setattr to enforce minimum size 1.
        if self.m < 1:
            object.__setattr__(self, "m", 1)
        if self.n < 1:
            object.__setattr__(self, "n", 1)

    @property
    def element_bytes(self) -> int:
        return dtype_bytes(self.dtype)

    @property
    def elements(self) -> int:
        return self.m * self.n

    @property
    def bytes(self) -> int:
        return self.elements * self.element_bytes

    @property
    def cl_per_row(self) -> int:
        """Cache lines per row of this tile, *if rows were independent*.

        Always rounds the row payload (`n × element_bytes`) up to a full
        cacheline. This is the right per-row count for non-contiguous tiles
        (column stripes). For contiguous tiles `cachelines` does NOT use this
        because the per-row padding is fictional — rows are end-to-end in
        memory and the kernel streams flat bytes.
        """
        return ceil(self.n * self.element_bytes / CACHELINE_BYTES)

    @property
    def cacheline_shape(self) -> tuple[int, int]:
        """The 2D cache-line footprint of this tile.

        - Contiguous tile: (1, total_cachelines). The kernel sees a flat
          stream; the "2D" structure is purely logical.
        - Non-contiguous tile (column stripe): (m, cl_per_row). Each row
          really is an independent cacheline cohort because sibling columns
          live between rows in physical memory.

        Downstream cost models reason about this shape directly:
          - HBM coalescing prefers contiguous cls within a row (axis=1).
          - L2/MALL reuse follows row-stride patterns (axis=0).
          - Inner-loop iteration counts may walk this 2D space
            (iters_per_row × rows) when the tile is non-contiguous,
            so intra-row iteration waste is captured.
        """
        if self.contiguous:
            return (1, self.cachelines)
        return (self.m, self.cl_per_row)

    @property
    def cachelines(self) -> int:
        """Total cache lines transferred for this 2D tile.

        Dense (contiguous) tile: flat-byte count, `ceil(total_bytes / CL)`.
        Non-contiguous (column stripe of a larger tensor): per-row count,
        `m × ceil(n*eb / CL)`, so partial-CL padding compounds per row.

        Empirically: for collectives operating on per-rank `torch.Tensor`s,
        the per-rank buffer is always dense, so the flat-byte form matches
        RCCL/torch.distributed behavior (rccl-tests + torch sweep agree).
        """
        if self.contiguous:
            return max(ceil(self.bytes / CACHELINE_BYTES), 1)
        return self.m * self.cl_per_row

    @property
    def cacheline_efficiency(self) -> float:
        useful = self.bytes
        transferred = self.cachelines * CACHELINE_BYTES
        return useful / transferred if transferred > 0 else 1.0

    def divide(self, factor: int, axis: int = 0) -> "TileShape":
        """Subdivide along `axis` (0 = rows, 1 = cols) by `factor`, ceil-divided.

        axis=0 (row-stripe):  rows shrink; contiguity is *preserved* (a row
                              stripe of a contiguous tile is itself a smaller
                              contiguous block).
        axis=1 (column-stripe): columns shrink; contiguity is *broken* — the
                              child's rows are interleaved with sibling
                              stripes of the parent in physical memory.
        """
        factor = max(int(factor), 1)
        if axis == 0:
            return replace(self, m=max(ceil(self.m / factor), 1))
        return replace(
            self,
            n=max(ceil(self.n / factor), 1),
            split_dim=1,
            contiguous=False,
        )

    def divide_byte_equal(self, factor: int) -> "TileShape":
        """Subdivide into `factor` ≈byte-equal pieces, preferring axis=0.

        Axis=0 (row-stripe) preserves cacheline alignment AND contiguity for
        the child tile. If the parent has fewer rows than `factor`, fall back
        to axis=1, which is the only option that produces non-degenerate
        chunks for a tile that is effectively 1D (e.g., when `M=1`).
        """
        factor = max(int(factor), 1)
        if factor == 1:
            return self
        if self.m >= factor:
            return self.divide(factor, axis=0)
        return self.divide(factor, axis=1)


@dataclass
class CommProblem:
    """
    2D tensor [M, N] distributed across num_gpus.

    M = rows, N = columns (contiguous dimension for row-major layout).
    split_dim determines which dimension the collective splits along:
      0 = split M: each GPU handles [M/W, N] — rows contiguous, good cacheline usage
      1 = split N: each GPU handles [M, N/W] — may have partial cachelines per row

    The split dimension affects cacheline efficiency:
      split_dim=0: gpu_tile is M/W rows of N contiguous elements → ceil(M/W × N × dtype / 64) cachelines
      split_dim=1: gpu_tile is M rows of N/W elements → M × ceil(N/W × dtype / 64) cachelines
                   (each row's chunk may not fill a full cacheline → padding waste)

    Naming hierarchy (see also CollectiveLayout). Every level except iter_tile
    is now a 2D TileShape (m × n); iter_tile is still a scalar cacheline count
    because it is consumed by the per-iteration work graph.

        msg                  full input across all ranks
        gpu_tile             one rank's data after the algorithm's split
                             (this dataclass; access via .gpu_tile_shape)
        gpu_timestep_tile    = gpu_tile.divide_byte_equal(chunks_per_timestep)
        wg_tile              = gpu_timestep_tile.divide_byte_equal(num_wgs)
        iter_tile            one inner-loop iteration of a wg_tile (= cl_per_iter)
    """
    M: int
    N: int
    num_gpus: int
    dtype: DataType = DataType.BF16
    split_dim: int = 0              # 0 = split rows, 1 = split columns

    @property
    def element_bytes(self):
        return dtype_bytes(self.dtype)

    @property
    def message_bytes(self):
        return self.M * self.N * self.element_bytes

    @property
    def total_elements(self):
        return self.M * self.N

    @property
    def gpu_tile_m(self):
        """Per-GPU rows after splitting."""
        if self.split_dim == 0:
            return ceil(self.M / self.num_gpus)
        return self.M

    @property
    def gpu_tile_n(self):
        """Per-GPU columns after splitting."""
        if self.split_dim == 1:
            return ceil(self.N / self.num_gpus)
        return self.N

    @property
    def gpu_tile_bytes(self):
        """Bytes per GPU after splitting."""
        return self.gpu_tile_m * self.gpu_tile_n * self.element_bytes

    @property
    def gpu_tile_elements(self):
        return self.gpu_tile_m * self.gpu_tile_n

    @property
    def gpu_tile_cachelines(self):
        """
        Cache lines for the per-GPU tile.

        The per-rank tile is *always* dense in the rank's local memory — it is
        a regular `torch.Tensor` (or equivalent) of shape `(gpu_tile_m,
        gpu_tile_n)`, contiguous, row-major. Collectives stream this buffer
        as a flat byte sequence (verified empirically by the tensor-shape
        sweep: max/min latency ratio across same-byte shapes is 1.00-1.08x).

        So cache-line count is the flat byte total:
            ceil(gpu_tile_m × gpu_tile_n × element_bytes / CACHELINE_BYTES).

        `split_dim` controls which dimension of the *full* tensor the rank
        owns (and thus the wire-traffic accounting in the collective layer);
        it does NOT change the local tile's contiguity in memory.
        """
        return self.gpu_tile_shape.cachelines

    @property
    def cacheline_efficiency(self):
        """Fraction of cacheline bytes that are useful data (1.0 = no waste)."""
        useful = self.gpu_tile_bytes
        transferred = self.gpu_tile_cachelines * CACHELINE_BYTES
        return useful / transferred if transferred > 0 else 1.0

    @property
    def gpu_tile_shape(self) -> TileShape:
        """The per-GPU tile as a 2D TileShape.

        This is the root of the tile hierarchy carried through the cost model:
        successive `divide_byte_equal()` calls produce gpu_timestep_tile and
        wg_tile, each remaining 2D so downstream models (cacheline padding,
        future GEMM↔comm interop) keep full shape fidelity.

        The root tile is `contiguous=True` because the rank's local buffer is
        dense in memory. Sub-tiles produced by `divide()` along axis=1
        (column stripe of the parent) inherit `contiguous=False`, which is
        the regime where partial-cacheline-per-row padding becomes real.
        """
        return TileShape(
            m=self.gpu_tile_m, n=self.gpu_tile_n,
            dtype=self.dtype, split_dim=self.split_dim,
            contiguous=True,
        )


@dataclass
class CommConfig:
    """Workgroup-level execution configuration.

    `num_wgs` is the *requested* parallelism (≡ NCCL channels). The
    *effective* parallelism is capped at runtime by the data available
    per timestep — see `effective_num_wgs()`.

    `min_bytes_per_wg` is the minimum byte budget below which a workgroup
    is considered "idle" rather than a real consumer of the timestep's
    data. When the per-timestep tile is small enough that
    `num_wgs × min_bytes_per_wg` exceeds the tile, the model treats only
    `⌈tile_bytes / min_bytes_per_wg⌉` WGs as active for the purposes of:
      - HBM/L2 BW polynomial scaling (fewer active CUs share the bus
        → higher per-CU bandwidth)
      - xGMI link sharing (`wgs_on_link` shrinks proportionally)
      - wg_tile sizing (each active WG gets ≥ 1 unit of real work)
    Launch overhead and per-step sync still scale with the *requested*
    `num_wgs` — NCCL launches all channels regardless.

    The default is sourced from `model.heuristics.DEFAULT_HEURISTICS` so
    that all calibration knobs live in one file. To override per study,
    pass an explicit value here (set to 0 to disable the cap entirely).
    """
    num_wgs: int
    load_width: LoadWidth = LoadWidth.DWORDX16
    vgprs_for_data: int = 128
    min_bytes_per_wg: int = field(
        default_factory=lambda: DEFAULT_HEURISTICS.min_bytes_per_wg,
    )

    @property
    def bytes_per_iter(self):
        return self.vgprs_for_data * 4

    @property
    def cl_per_iter(self):
        return self.bytes_per_iter // CACHELINE_BYTES

    @property
    def instrs_per_cl(self):
        return self.load_width.instrs_per_cacheline

    def effective_num_wgs(self, tile_bytes: int) -> int:
        """Number of WGs that have ≥ `min_bytes_per_wg` of real work.

        Capped at `num_wgs` (can't have more effective than launched) and
        floored at 1 (at least one WG owns the tile, even if tiny).
        """
        if self.min_bytes_per_wg <= 0:
            return self.num_wgs
        fit = max(tile_bytes // self.min_bytes_per_wg, 1)
        return min(self.num_wgs, fit)


@dataclass
class FunctionalUnitWork:
    vmem_read_instrs: int = 0
    vmem_write_instrs: int = 0
    tcp_read_cl: int = 0
    tcp_write_cl: int = 0
    l2_read_cl: int = 0
    l2_write_cl: int = 0
    mall_read_cl: int = 0
    mall_write_cl: int = 0
    hbm_read_cl: int = 0
    hbm_write_cl: int = 0
    xgmi_read_cl: int = 0
    xgmi_write_cl: int = 0
    valu_ops: int = 0
    atomic_count: int = 0

    def __iadd__(self, other):
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))
        return self

    def __add__(self, other):
        result = FunctionalUnitWork()
        for f in self.__dataclass_fields__:
            setattr(result, f, getattr(self, f) + getattr(other, f))
        return result

    @staticmethod
    def zero():
        return FunctionalUnitWork()


@dataclass
class WgTileLatencyBreakdown:
    """
    Full wg_tile transfer latency breakdown — one workgroup's slice of one
    gpu_timestep_tile.

    All `*_cycles` fields are in **GPU cycles** — the model's native unit.
    Each one has a corresponding `T_*` property that returns the value in
    nanoseconds for human-facing display (dashboards, logging). The
    cycles→µs conversion at the public API boundary (`predict_row`) is
    independent of these display helpers.

    T_wlt_cycles = one iter_tile time (one workgroup-link-tile atom) = max of all functional units
    T_total_cycles = prologue + (num_iters - 1) × T_wlt + epilogue + sync
    """
    T_total_cycles: float
    T_wlt_cycles: float
    T_prologue_cycles: float
    T_epilogue_cycles: float
    T_sync_cycles: float
    num_iters: int
    # Per-WLT functional unit breakdown (all in cycles):
    T_vmem_cycles: float = 0.0
    T_tcp_cycles: float = 0.0
    T_l2_cycles: float = 0.0
    T_mall_cycles: float = 0.0
    T_hbm_read_cycles: float = 0.0
    T_hbm_write_cycles: float = 0.0
    T_xgmi_read_cycles: float = 0.0
    T_xgmi_write_cycles: float = 0.0
    T_valu_cycles: float = 0.0
    bottleneck: str = ""

    # Clock used by the *_ns display helpers below. Populated by
    # `compute_wg_tile_latency` from `hw.clock_ghz`; falls back to 2.0
    # (MI300X). The cycle fields are the source of truth.
    clock_ghz: float = 2.0

    @property
    def clock_hz(self) -> float:
        """GPU clock frequency in Hz."""
        return self.clock_ghz * 1e9

    def _cycles_to_ns(self, cycles: float) -> float:
        """cycles ÷ frequency (Hz) → seconds × 1e9 → nanoseconds."""
        return cycles / self.clock_hz * 1e9

    # ── Backward-compat ns accessors (legacy name = nanosecond value) ──
    # These let dashboard/UI code that historically read `T_total` etc.
    # continue to work with ns semantics, while internal math stays in
    # cycles. Conversion goes through the GPU frequency.
    @property
    def T_total(self) -> float:        return self._cycles_to_ns(self.T_total_cycles)
    @property
    def T_wlt(self) -> float:          return self._cycles_to_ns(self.T_wlt_cycles)
    @property
    def T_prologue(self) -> float:     return self._cycles_to_ns(self.T_prologue_cycles)
    @property
    def T_epilogue(self) -> float:     return self._cycles_to_ns(self.T_epilogue_cycles)
    @property
    def T_sync(self) -> float:         return self._cycles_to_ns(self.T_sync_cycles)
    @property
    def T_vmem(self) -> float:         return self._cycles_to_ns(self.T_vmem_cycles)
    @property
    def T_tcp(self) -> float:          return self._cycles_to_ns(self.T_tcp_cycles)
    @property
    def T_l2(self) -> float:           return self._cycles_to_ns(self.T_l2_cycles)
    @property
    def T_mall(self) -> float:         return self._cycles_to_ns(self.T_mall_cycles)
    @property
    def T_hbm_read(self) -> float:     return self._cycles_to_ns(self.T_hbm_read_cycles)
    @property
    def T_hbm_write(self) -> float:    return self._cycles_to_ns(self.T_hbm_write_cycles)
    @property
    def T_xgmi_read(self) -> float:    return self._cycles_to_ns(self.T_xgmi_read_cycles)
    @property
    def T_xgmi_write(self) -> float:   return self._cycles_to_ns(self.T_xgmi_write_cycles)
    @property
    def T_valu(self) -> float:         return self._cycles_to_ns(self.T_valu_cycles)
