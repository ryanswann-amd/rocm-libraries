"""
wg_tile latency computation. All times are in **GPU cycles**.

The atom (T_wlt): cycles for one WG to transfer one iter_tile over one
link, given all WGs contending for that link. The full wg_tile transfer
time is a software-pipelined loop of num_iters such atoms, mirroring
Origami's per-WG-tile latency structure:

    T_total = T_prologue + (num_iters - 1) × T_wlt + T_epilogue + T_sync

All `T_*` quantities here are cycle counts; the conversion to seconds
happens once at the top of the model (`predict_row` in collective.py).

`wg_tile_cachelines` is the one WG's share of a single gpu_timestep_tile
(see model/types.py for the full data hierarchy).
"""

from math import ceil, exp
from typing import List, Optional

from .types import (
    CommProblem, CommConfig, FunctionalUnitWork, TileShape,
    WgTileLatencyBreakdown, CACHELINE_BYTES, dtype_bytes,
)
from .hardware import Hardware, CommHardware
from .heuristics import Heuristics, DEFAULT_HEURISTICS
from .primitives import Op, Signal, Wait


def resolve_work_graph(
    ops: List[Op],
    cl_per_iter: int,
    instrs_per_cl: int,
    elements_per_iter: int,
) -> tuple[FunctionalUnitWork, FunctionalUnitWork]:
    """Resolve a work graph into per-iteration and sync work."""
    iter_work = FunctionalUnitWork.zero()
    sync_work = FunctionalUnitWork.zero()
    for op in ops:
        w = op.resolve(cl_per_iter, instrs_per_cl, elements_per_iter)
        if isinstance(op, (Signal, Wait)):
            sync_work += w
        else:
            iter_work += w
    return iter_work, sync_work


def compute_iter_times(
    work: FunctionalUnitWork,
    hw: Hardware,
    comm_hw: CommHardware,
    bw_per_wg: float,
    active_cus: int,
    heuristics: Heuristics = DEFAULT_HEURISTICS,
    primitive: Optional[str] = None,
) -> dict[str, float]:
    """Compute per-iteration time at each level of the hierarchy, in **cycles**.

    All hardware fields (`hw.*`, `comm_hw.*`) are cycle-based: bandwidths
    are bytes/cycle, latencies are cycles. The returned dict therefore
    holds cycle counts. Conversion to seconds happens once at the top
    of the model (`predict_row` in collective.py).

    Uses active_cus to scale HBM/L2 bandwidth via Origami's polynomial model
    (more CUs active → less per-CU bandwidth due to sharing).

    `heuristics` supplies the xGMI write concentration constant; the
    `primitive` name selects a per-collective k value if available.
    """
    CL = CACHELINE_BYTES

    vmem_total = work.vmem_read_instrs + work.vmem_write_instrs
    T_vmem = vmem_total / hw.vmem_issue_rate if hw.vmem_issue_rate > 0 else 0.0

    T_tcp = (work.tcp_read_cl + work.tcp_write_cl) * CL / hw.tcp_bw if hw.tcp_bw > 0 else 0.0

    # L2: scale by active CUs on this XCD
    active_per_xcd = max(ceil(active_cus / hw.num_xcd), 1)
    l2_bw = hw.l2_bw_per_cu_scaled(active_per_xcd)
    T_l2 = (work.l2_read_cl + work.l2_write_cl) * CL / l2_bw if l2_bw > 0 else 0.0

    T_mall = (work.mall_read_cl + work.mall_write_cl) * CL / hw.mall_bw if hw.mall_bw > 0 else 0.0

    # HBM: scale by active CUs via polynomial model
    hbm_r_per_cu = hw.hbm_read_bw_per_cu(active_cus)
    hbm_w_per_cu = hw.hbm_write_bw_per_cu(active_cus)
    T_hbm_read = work.hbm_read_cl * CL / hbm_r_per_cu if work.hbm_read_cl > 0 and hbm_r_per_cu > 0 else 0.0
    T_hbm_write = work.hbm_write_cl * CL / hbm_w_per_cu if work.hbm_write_cl > 0 and hbm_w_per_cu > 0 else 0.0

    # xGMI: limited by BOTH link bandwidth sharing AND MSHR depth (latency-bandwidth product)
    # Per-WG remote read throughput is min of:
    #   1. link_bw / wgs_on_link (bandwidth sharing) — bytes/cycle
    #   2. MSHR_depth × waves × cacheline / xGMI_latency (outstanding request limit)
    #      — bytes/cycle because xgmi_latency_cycles is in cycles
    mshr_limited_bw = (hw.mshr_depth_per_wave * hw.waves_per_wg * CL) / hw.xgmi_latency_cycles
    effective_remote_read_bw = min(bw_per_wg, mshr_limited_bw)
    T_xgmi_read = work.xgmi_read_cl * CL / effective_remote_read_bw if work.xgmi_read_cl > 0 and effective_remote_read_bw > 0 else 0.0

    # xGMI write: per-WG throughput scales with WG concentration on the
    # link via 1 - exp(-wgs/k). Empirically (rccl AG sweep, MI300X):
    #   1 WG/link → ~21% of payload, 2 → 38%, 5 → 65%, 9+ → 83%
    # The MSHR-cap-for-writes heuristic this replaces missed both the
    # single-WG floor (it overestimated by ~5x) and the multi-WG ramp.
    # See heuristics.xgmi_write_concentration_k and
    # wiki/pages/ag-outlier-investigation.md.
    #
    # Back out wgs_on_link from bw_per_wg — the layout has already
    # distributed WGs across links upstream:
    #   bw_per_wg = comm_hw.link_bw / wgs_on_link
    if work.xgmi_write_cl > 0 and bw_per_wg > 0:
        wgs_on_link = max(comm_hw.link_bw / bw_per_wg, 1.0)
        k = heuristics.k_xgmi_write(primitive)
        util = 1.0 - exp(-wgs_on_link / k)
        eff_link_bw = comm_hw.link_bw * util
        eff_write_bw_per_wg = eff_link_bw / wgs_on_link
        T_xgmi_write = work.xgmi_write_cl * CL / eff_write_bw_per_wg
    else:
        T_xgmi_write = 0.0

    T_valu = work.valu_ops / hw.valu_rate if hw.valu_rate > 0 and work.valu_ops > 0 else 0.0

    return {
        "vmem": T_vmem, "tcp": T_tcp, "l2": T_l2, "mall": T_mall,
        "hbm_read": T_hbm_read, "hbm_write": T_hbm_write,
        "xgmi_read": T_xgmi_read, "xgmi_write": T_xgmi_write,
        "valu": T_valu,
    }


def _iter_counts_from_tile(
    wg_tile: Optional[TileShape],
    wg_tile_cachelines: int,
    wg_tile_elements: int,
    cl_per_iter: int,
) -> tuple[int, int]:
    """Resolve (num_iters, elements_per_iter) for one wg_tile.

    Two cases when a 2D `wg_tile` is provided:

      - `contiguous=True` (default for dense per-rank tiles): the kernel
        streams the tile as a flat byte sequence — iteration count is the
        same as the 1D scalar path:
            num_iters         = ceil(cachelines / cl_per_iter)
            elements_per_iter = ceil(elements / num_iters)

      - `contiguous=False` (column stripe of a larger tile, post axis=1
        divide): the kernel walks each row separately and the partial-tail
        CL on every row costs a real iteration:
            iters_per_row     = ceil(cl_per_row / cl_per_iter)
            num_iters         = m × iters_per_row
            elements_per_iter = ceil(n / iters_per_row)

    For the degenerate 1D case (m=1) the contiguous-path collapses onto the
    scalar 1D answer, so existing CCL validation numbers are unaffected.
    """
    if wg_tile is not None and not wg_tile.contiguous:
        iters_per_row = max(ceil(wg_tile.cl_per_row / cl_per_iter), 1)
        num_iters = max(wg_tile.m * iters_per_row, 1)
        elements_per_iter = max(ceil(wg_tile.n / iters_per_row), 1)
        return num_iters, elements_per_iter
    # Contiguous tile (or no wg_tile supplied) — flat-byte iteration walk.
    num_iters = max(ceil(wg_tile_cachelines / cl_per_iter), 1)
    elements_per_iter = max(ceil(wg_tile_elements / num_iters), 1)
    return num_iters, elements_per_iter


def compute_wg_tile_latency(
    work_graph: List[Op],
    wg_tile_cachelines: int,
    config: CommConfig,
    hw: Hardware,
    comm_hw: CommHardware,
    bw_per_wg: float,
    wg_tile_elements: int,
    wg_tile: Optional[TileShape] = None,
    active_cus: Optional[int] = None,
    heuristics: Heuristics = DEFAULT_HEURISTICS,
    primitive: Optional[str] = None,
) -> WgTileLatencyBreakdown:
    """
    Compute the full wg_tile transfer latency for one timestep.

    T_wlt = one iter_tile = max(functional unit times) — the atom
    T_total = T_prologue + (num_iters - 1) × T_wlt + T_epilogue + T_sync

    If `wg_tile` (a 2D TileShape) is supplied, iteration counts are derived
    row-aware (m × ceil(cl_per_row / cl_per_iter)); otherwise the scalar
    `wg_tile_cachelines` / `wg_tile_elements` are used as a flat 1D chunk.

    `active_cus` controls the HBM/L2 bandwidth polynomial. When omitted,
    falls back to `config.num_wgs` (the requested parallelism); callers
    that know the effective WG count (e.g., after applying
    `CommConfig.effective_num_wgs`) should pass it explicitly so that
    over-launched WGs don't artificially depress per-CU bandwidth.
    """
    cl_per_iter = config.cl_per_iter
    instrs_per_cl = config.instrs_per_cl
    num_iters, elements_per_iter = _iter_counts_from_tile(
        wg_tile, wg_tile_cachelines, wg_tile_elements, cl_per_iter
    )

    iter_work, sync_work = resolve_work_graph(
        work_graph, cl_per_iter, instrs_per_cl, elements_per_iter
    )

    if active_cus is None:
        active_cus = config.num_wgs
    times = compute_iter_times(iter_work, hw, comm_hw, bw_per_wg, active_cus, heuristics, primitive)

    T_wlt = max(times.values()) if times else 0.0  # one iteration = one WLT

    read_times = [times["hbm_read"], times["xgmi_read"], times["mall"]]
    T_prologue = max(t for t in read_times if t > 0) if any(t > 0 for t in read_times) else 0.0

    write_times = [times["hbm_write"], times["xgmi_write"]]
    T_epilogue = max(t for t in write_times if t > 0) if any(t > 0 for t in write_times) else 0.0

    T_sync = sync_work.atomic_count * comm_hw.atomic_latency_cycles

    T_total = T_prologue + max(num_iters - 1, 0) * T_wlt + T_epilogue + T_sync

    bottleneck = max(times, key=times.get) if times else ""

    return WgTileLatencyBreakdown(
        T_total_cycles=T_total,
        T_wlt_cycles=T_wlt,
        T_prologue_cycles=T_prologue,
        T_epilogue_cycles=T_epilogue,
        T_sync_cycles=T_sync,
        num_iters=num_iters,
        T_vmem_cycles=times.get("vmem", 0),
        T_tcp_cycles=times.get("tcp", 0),
        T_l2_cycles=times.get("l2", 0),
        T_mall_cycles=times.get("mall", 0),
        T_hbm_read_cycles=times.get("hbm_read", 0),
        T_hbm_write_cycles=times.get("hbm_write", 0),
        T_xgmi_read_cycles=times.get("xgmi_read", 0),
        T_xgmi_write_cycles=times.get("xgmi_write", 0),
        T_valu_cycles=times.get("valu", 0),
        bottleneck=bottleneck,
        clock_ghz=hw.clock_ghz,
    )
