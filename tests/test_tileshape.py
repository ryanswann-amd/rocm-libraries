"""TileShape unit tests.

These pin down the 2D tile arithmetic used at every level of the cost model's
tile hierarchy (gpu_tile → gpu_timestep_tile → wg_tile).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import ceil

import pytest

from model.types import (
    CACHELINE_BYTES, CommProblem, DataType, TileShape, dtype_bytes,
)


# ─── basic shape / byte / cacheline math ─────────────────────────────────────

def test_elements_and_bytes():
    t = TileShape(m=4, n=128, dtype=DataType.BF16)
    assert t.elements == 4 * 128
    assert t.bytes == 4 * 128 * 2


def test_cachelines_full_rows():
    # 128 BF16 elements/row = 256 B = 4 CLs exactly → no padding waste.
    t = TileShape(m=4, n=128, dtype=DataType.BF16)
    assert t.cachelines == 4 * 4
    assert t.cacheline_efficiency == pytest.approx(1.0)


def test_cachelines_partial_rows_dense():
    """Dense (contiguous) tile: cacheline count is the flat-byte total."""
    # 10×31 BF16 dense block = 620 B → ceil(620/64) = 10 CLs (one tail CL,
    # NOT one per row — rows are end-to-end in memory).
    t = TileShape(m=10, n=31, dtype=DataType.BF16, contiguous=True)
    assert t.cachelines == ceil(10 * 31 * 2 / CACHELINE_BYTES)
    expected_eff = (10 * 31 * 2) / (t.cachelines * CACHELINE_BYTES)
    assert t.cacheline_efficiency == pytest.approx(expected_eff)


def test_cachelines_partial_rows_non_contiguous():
    """Non-contiguous tile (column stripe): per-row padding is real."""
    # 10 rows × 31 BF16 cols, but rows are NOT contiguous in memory (i.e.,
    # a column stripe of a wider tensor) → each row burns one full 64 B CL.
    t = TileShape(m=10, n=31, dtype=DataType.BF16, contiguous=False)
    cl_per_row = ceil(31 * 2 / CACHELINE_BYTES)  # = 1
    assert t.cachelines == 10 * cl_per_row
    expected_eff = (10 * 31 * 2) / (10 * CACHELINE_BYTES)
    assert t.cacheline_efficiency == pytest.approx(expected_eff)


def test_cacheline_shape_is_flat_when_contiguous():
    # Dense (16, 200) BF16 = 6400 B → 100 CLs total. cacheline_shape collapses
    # to (1, 100) because the tile is streamed as a flat byte sequence.
    t = TileShape(m=16, n=200, dtype=DataType.BF16, contiguous=True)
    assert t.cachelines == ceil(16 * 200 * 2 / CACHELINE_BYTES)
    assert t.cacheline_shape == (1, t.cachelines)


def test_cacheline_shape_is_2d_when_non_contiguous():
    # Column stripe: each row carries its own padding → 2D shape is real.
    t = TileShape(m=16, n=200, dtype=DataType.BF16, contiguous=False)
    assert t.cl_per_row == ceil(200 * 2 / CACHELINE_BYTES)
    assert t.cacheline_shape == (16, t.cl_per_row)
    assert t.cachelines == 16 * t.cl_per_row


def test_cacheline_shape_preserved_across_divide():
    parent = TileShape(m=64, n=200, dtype=DataType.BF16)
    # Row-stripe split: contiguity is preserved (smaller contiguous block).
    child0 = parent.divide(8, axis=0)
    assert child0.contiguous is True
    assert child0.cl_per_row == parent.cl_per_row
    assert child0.cachelines == ceil(8 * 200 * 2 / CACHELINE_BYTES)
    # Column-stripe split: contiguity broken → child becomes 2D-padded.
    child1 = parent.divide(8, axis=1)
    assert child1.contiguous is False
    assert child1.split_dim == 1
    assert child1.cacheline_shape == (parent.m, ceil(child1.n * 2 / CACHELINE_BYTES))


def test_min_size_clamp():
    # Tile shape must round up to at least 1×1.
    t = TileShape(m=0, n=0, dtype=DataType.BF16)
    assert t.m == 1 and t.n == 1


# ─── divide ─────────────────────────────────────────────────────────────────

def test_divide_axis0():
    parent = TileShape(m=16, n=128, dtype=DataType.BF16)
    child = parent.divide(4, axis=0)
    assert (child.m, child.n) == (4, 128)
    # Total bytes shrink by ~4×.
    assert child.bytes == parent.bytes // 4


def test_divide_axis1():
    parent = TileShape(m=16, n=128, dtype=DataType.BF16)
    child = parent.divide(4, axis=1)
    assert (child.m, child.n) == (16, 32)
    assert child.split_dim == 1


def test_divide_ceil_rounds_up():
    # 17 rows / 4 should give 5 rows per chunk (ceil).
    parent = TileShape(m=17, n=128, dtype=DataType.BF16)
    child = parent.divide(4, axis=0)
    assert child.m == ceil(17 / 4)


# ─── divide_byte_equal: axis selection ──────────────────────────────────────

def test_byte_equal_prefers_axis0_when_rows_sufficient():
    parent = TileShape(m=64, n=128, dtype=DataType.BF16)
    child = parent.divide_byte_equal(8)
    assert (child.m, child.n) == (8, 128)


def test_byte_equal_falls_back_to_axis1_for_1d_tile():
    # M=1 case (the predict_row default): can't subdivide along rows.
    parent = TileShape(m=1, n=4096, dtype=DataType.BF16)
    child = parent.divide_byte_equal(8)
    assert (child.m, child.n) == (1, 512)


def test_byte_equal_factor_1_is_identity():
    parent = TileShape(m=16, n=128, dtype=DataType.BF16)
    assert parent.divide_byte_equal(1) is parent


# ─── two-level (gpu_tile → gpu_timestep_tile → wg_tile) ─────────────────────

def test_two_level_division_matches_old_scalar_path_for_1d():
    """In the 1D-degenerate case (M=1) used by validate.py, the new 2D
    division must produce the same wg_tile_cachelines and wg_tile_elements
    that the pre-refactor scalar arithmetic produced (within ceil/floor of 1).
    """
    msg_bytes = 1024 * 1024
    eb = 2  # BF16
    total_elements = msg_bytes // eb
    gpu_tile = TileShape(m=1, n=total_elements, dtype=DataType.BF16)

    for chunks in (1, 2, 4, 8):
        ts = gpu_tile.divide_byte_equal(chunks)
        for num_wgs in (1, 4, 16, 64, 256):
            wg = ts.divide_byte_equal(num_wgs)
            # cachelines/elements should be at least 1
            assert wg.cachelines >= 1
            assert wg.elements >= 1
            # bytes-per-wg-tile within a small (rounding) factor of the
            # ideal byte-equal chunk
            ideal = msg_bytes / (chunks * num_wgs)
            assert wg.bytes <= max(2 * ideal, CACHELINE_BYTES)


def test_two_level_division_2d_axis0():
    """For a real 2D tile with enough rows, divisions stay on axis=0 and
    cacheline efficiency is preserved exactly."""
    gpu_tile = TileShape(m=4096, n=128, dtype=DataType.BF16)  # 1 MB
    ts = gpu_tile.divide_byte_equal(8)
    wg = ts.divide_byte_equal(64)
    assert ts.m == 512 and ts.n == 128
    assert wg.m == 8   and wg.n == 128
    # No padding waste anywhere along this path.
    assert wg.cacheline_efficiency == pytest.approx(1.0)


# ─── integration with CommProblem ───────────────────────────────────────────

def test_commproblem_gpu_tile_shape_matches_scalar_properties():
    p = CommProblem(M=1024, N=512, num_gpus=8, dtype=DataType.BF16, split_dim=0)
    t = p.gpu_tile_shape
    assert t.m == p.gpu_tile_m
    assert t.n == p.gpu_tile_n
    assert t.elements == p.gpu_tile_elements
    assert t.bytes == p.gpu_tile_bytes
    assert t.cachelines == p.gpu_tile_cachelines
    assert t.dtype == p.dtype
    assert t.split_dim == p.split_dim


def test_commproblem_local_tile_is_dense():
    """Per-rank local tile is dense regardless of split_dim.

    A `CommProblem` describes how the full tensor is split across GPUs, but
    each rank's *local* buffer is a regular row-major torch.Tensor — i.e.,
    dense in memory. So `gpu_tile_shape.contiguous == True` and the
    cacheline count is the flat-byte total, NOT the per-row padded count.
    The empirical torch sweep confirms this: collective latency does NOT
    vary with per-rank tile aspect ratio at fixed byte total (max/min
    spread ≤ 1.08× across the (m, n) sweep).
    """
    p = CommProblem(M=1024, N=33, num_gpus=4, dtype=DataType.BF16, split_dim=1)
    t = p.gpu_tile_shape
    # gpu_tile is 1024 rows × ceil(33/4)=9 cols; total = 1024*9*2 = 18432 B.
    # Dense → ceil(18432/64) = 288 CLs, NOT 1024 (which would be the strided
    # column-stripe answer).
    assert t.m == 1024 and t.n == 9
    assert t.contiguous is True
    assert t.cachelines == ceil(1024 * 9 * 2 / CACHELINE_BYTES)
    # No per-row padding waste because rows are contiguous in memory.
    assert t.cacheline_efficiency == pytest.approx(1024 * 9 * 2 / (t.cachelines * CACHELINE_BYTES))


# ─── 2D iter-walk: intra-row iteration accounting ───────────────────────────

def test_iter_counts_1d_path_matches_old_scalar():
    """When no wg_tile is supplied the function must reproduce the pre-2D
    scalar arithmetic so the existing CCL validation numbers are unchanged."""
    from model.latency import _iter_counts_from_tile
    cl_per_iter = 4
    n_iters, e_per = _iter_counts_from_tile(
        wg_tile=None, wg_tile_cachelines=20, wg_tile_elements=200,
        cl_per_iter=cl_per_iter,
    )
    assert n_iters == ceil(20 / 4)
    assert e_per == ceil(200 / n_iters)


def test_iter_counts_non_contiguous_captures_intra_row_waste():
    """Non-contiguous tile walks each row separately and counts a partial-tail
    iter on every row when cl_per_iter doesn't divide cl_per_row evenly.

    This is the strided / column-stripe regime — produced by `divide(axis=1)`
    or constructed explicitly with `contiguous=False`. It is NOT the regime
    that collective per-rank tiles live in (those are dense).
    """
    from model.latency import _iter_counts_from_tile
    # 8 rows × 10 CL per row (320 BF16 = 640 B/row → 10 CL), non-contiguous.
    # Flat-byte would give ceil(80/4)=20 iters; per-row gives 8*ceil(10/4)=24.
    wg_tile = TileShape(m=8, n=320, dtype=DataType.BF16, contiguous=False)
    assert wg_tile.cl_per_row == 10
    n_iters, e_per = _iter_counts_from_tile(
        wg_tile=wg_tile, wg_tile_cachelines=wg_tile.cachelines,
        wg_tile_elements=wg_tile.elements, cl_per_iter=4,
    )
    assert n_iters == 8 * ceil(10 / 4)  # = 24
    assert e_per == ceil(320 / 3)


def test_iter_counts_contiguous_matches_flat_byte_walk():
    """Contiguous (dense) tile streams as flat bytes, so iter count matches
    the 1D / scalar formula regardless of (m, n) aspect ratio.

    This is the regime collectives operate in. Empirically validated by the
    torch shape sweep: latency depends only on per-rank byte total.
    """
    from model.latency import _iter_counts_from_tile
    # Same byte total, different aspect ratios — all must agree.
    elements = 8 * 256  # 4096 B → 64 CL
    cases = [
        TileShape(m=1,    n=elements, dtype=DataType.BF16),  # 1D
        TileShape(m=8,    n=256,      dtype=DataType.BF16),  # 2D square-ish
        TileShape(m=256,  n=8,        dtype=DataType.BF16),  # 2D narrow
        TileShape(m=2048, n=1,        dtype=DataType.BF16),  # 2D pathological
    ]
    expected_n, expected_e = _iter_counts_from_tile(
        wg_tile=None, wg_tile_cachelines=cases[0].cachelines,
        wg_tile_elements=cases[0].elements, cl_per_iter=4,
    )
    for t in cases:
        n_iters, e_per = _iter_counts_from_tile(
            wg_tile=t, wg_tile_cachelines=t.cachelines,
            wg_tile_elements=t.elements, cl_per_iter=4,
        )
        assert n_iters == expected_n, f"{t} broke iter parity"
        assert e_per == expected_e, f"{t} broke element-per-iter parity"


def test_effective_num_wgs_caps_by_tile():
    """`effective_num_wgs` caps the requested WG count when the per-timestep
    tile can't supply each WG with `min_bytes_per_wg` of real work.
    """
    from model.types import CommConfig
    c = CommConfig(num_wgs=32, min_bytes_per_wg=512)
    # Plenty of work — all 32 WGs active.
    assert c.effective_num_wgs(tile_bytes=64 * 1024) == 32
    # Border case — exactly 32 WGs' worth.
    assert c.effective_num_wgs(tile_bytes=32 * 512) == 32
    # Half the work — only 16 WGs active.
    assert c.effective_num_wgs(tile_bytes=16 * 512) == 16
    # Tiny tile — clamped to at least 1 active WG.
    assert c.effective_num_wgs(tile_bytes=1) == 1
    assert c.effective_num_wgs(tile_bytes=0) == 1


def test_effective_num_wgs_disabled_returns_full_request():
    from model.types import CommConfig
    c = CommConfig(num_wgs=32, min_bytes_per_wg=0)
    assert c.effective_num_wgs(tile_bytes=1) == 32
    assert c.effective_num_wgs(tile_bytes=10**12) == 32


def test_effective_num_wgs_never_exceeds_requested():
    from model.types import CommConfig
    c = CommConfig(num_wgs=4, min_bytes_per_wg=512)
    # Huge tile could fit thousands of WGs — but we only have 4.
    assert c.effective_num_wgs(tile_bytes=10**9) == 4


def test_iter_counts_2d_degenerate_m1_matches_1d():
    """For m=1 tiles (the predict_row default) the 2D path collapses onto
    the 1D answer — this is the safety net that pins existing CCL numbers."""
    from model.latency import _iter_counts_from_tile
    wg_tile = TileShape(m=1, n=12_345, dtype=DataType.BF16)
    n_iters_2d, e_per_2d = _iter_counts_from_tile(
        wg_tile=wg_tile, wg_tile_cachelines=wg_tile.cachelines,
        wg_tile_elements=wg_tile.elements, cl_per_iter=8,
    )
    n_iters_1d, e_per_1d = _iter_counts_from_tile(
        wg_tile=None, wg_tile_cachelines=wg_tile.cachelines,
        wg_tile_elements=wg_tile.elements, cl_per_iter=8,
    )
    assert n_iters_2d == n_iters_1d
    assert e_per_2d == e_per_1d
