"""
End-to-end collective latency prediction.

Wires together: layout → contention → wg_tile cost → total latency.

Two computation modes based on layout structure:
1. Sequential timesteps: each timestep uses a (potentially) different link
   (e.g., one_shot AR visiting each peer in turn). T = Σ T_timestep.
2. Pipelined ring: all timesteps use the same link, but multiple WGs (channels)
   pipeline through ring steps. T = total_data / aggregate_throughput + sync.

Naming: gpu_tile → gpu_timestep_tile → wg_tile → iter_tile.
See model/types.py CommProblem docstring for the full hierarchy.
"""

from math import ceil
from typing import Optional

from .types import CommProblem, CommConfig, CACHELINE_BYTES, dtype_bytes, DataType
from .hardware import Hardware, CommHardware
from .heuristics import Heuristics, DEFAULT_HEURISTICS
from .layouts import (
    CollectiveLayout,
    RingAllReduceLayout,
    RingAllGatherLayout,
    RingReduceScatterLayout,
    RingFixedLayout,
    allgather_layout,
    reduce_scatter_layout,
    allreduce_one_shot_layout,
    allreduce_two_shot_layout,
    allreduce_ring_layout,
    alltoall_layout,
    broadcast_layout,
)
from .latency import compute_wg_tile_latency
from .primitives import Signal, Wait


# Ring-style layouts charged the per-step proxy/sync overhead heuristic.
# Non-ring layouts (TwoShot AR, A2A, PidStaggered) get 0 by default.
RING_LAYOUT_CLASSES = (
    RingAllReduceLayout,
    RingAllGatherLayout,
    RingReduceScatterLayout,
    RingFixedLayout,
)


def _ring_step_overhead_cycles(primitive: Optional[str], layout, heuristics: Heuristics) -> float:
    """Heuristic per-step proxy/sync overhead in cycles, summed over all steps.

    Returns 0 when the layout isn't ring-style or the primitive has no
    fitted overhead. See Heuristics.ring_step_overhead_cycles.
    """
    if not isinstance(layout, RING_LAYOUT_CLASSES):
        return 0.0
    if primitive is None:
        return 0.0
    per_step = heuristics.ring_step_overhead_cycles.get(primitive, 0.0)
    return per_step * layout.num_timesteps


COLLECTIVE_LAYOUTS = {
    "all_gather": allgather_layout,
    "reduce_scatter": reduce_scatter_layout,
    "all_reduce": allreduce_two_shot_layout,  # pid-staggered: channels spread across peers
    "all_to_all": alltoall_layout,
    "broadcast": broadcast_layout,
}


def _is_ring_layout(layout):
    return isinstance(layout, (RingAllReduceLayout, RingFixedLayout))


def compute_collective_latency(
    collective: str,
    problem: CommProblem,
    config: CommConfig,
    hw: Hardware,
    comm_hw: CommHardware,
    layout: Optional[CollectiveLayout] = None,
    my_rank: int = 0,
    heuristics: Heuristics = DEFAULT_HEURISTICS,
) -> float:
    """
    Predict total collective latency in **GPU cycles**.

    The conversion to seconds happens once at the top of the model
    (`predict_row`), dividing by `hw.clock_ghz * 1000` to produce µs.

    `heuristics` supplies xGMI write concentration and per-ring-step
    overhead constants (see Heuristics).
    """
    if layout is None:
        layout_fn = COLLECTIVE_LAYOUTS.get(collective)
        if layout_fn is None:
            raise ValueError(f"Unknown collective: {collective}")
        layout = layout_fn(problem.num_gpus)

    if _is_ring_layout(layout):
        return _compute_ring_latency(layout, problem, config, hw, comm_hw, my_rank, heuristics, collective)
    else:
        return _compute_sequential_latency(layout, problem, config, hw, comm_hw, my_rank, heuristics, collective)


def _compute_ring_latency(
    layout, problem, config, hw, comm_hw, my_rank,
    heuristics: Heuristics = DEFAULT_HEURISTICS,
    primitive: Optional[str] = None,
) -> float:
    """
    Ring latency: all timesteps use the same link. Multiple WGs (channels)
    pipeline through the ring steps, so the aggregate throughput approaches
    link bandwidth when enough WGs are active.

    Total wire data crossing the link = gpu_timestep_tile_bytes × num_timesteps.
    Each WG handles gpu_timestep_tile_bytes / effective_num_wgs bytes per step.
    The link is shared by `effective_num_wgs` WGs simultaneously — over-launched
    channels (more WGs than the timestep can feed) are idle for aggregate
    throughput purposes (see `CommConfig.effective_num_wgs`).

    Aggregate link throughput = min(link_bw, effective_num_wgs × mshr_bw_per_wg).
    T_transfer = total_wire_data / aggregate_throughput.
    T_sync = num_timesteps × per_step_sync (sync still costs full num_wgs).
    """
    N = problem.num_gpus
    num_wgs = config.num_wgs
    num_timesteps = layout.num_timesteps  # 2(N-1) for AllReduce, N-1 for AG/RS

    # Per-timestep data crossing the link per GPU (not per WG).
    # NOTE: with chunks_per_timestep=1 (the current default), this equals the
    # whole gpu_tile per timestep — behaviorally preserved from the pre-rename
    # code. The bug fix is to set chunks_per_timestep=N on chunked layouts and
    # let the division below shrink each timestep's data accordingly.
    gpu_timestep_tile_bytes = (
        problem.gpu_tile_cachelines * CACHELINE_BYTES // layout.chunks_per_timestep
    )
    total_wire_bytes = gpu_timestep_tile_bytes * num_timesteps

    # Cap WGs to the amount of work actually available per timestep — see
    # CommConfig.effective_num_wgs and `min_bytes_per_wg`.
    eff_wgs = config.effective_num_wgs(gpu_timestep_tile_bytes)

    # Per-WG MSHR-limited throughput (bytes/cycle)
    CL = CACHELINE_BYTES
    mshr_bw_per_wg = (hw.mshr_depth_per_wave * hw.waves_per_wg * CL) / hw.xgmi_latency_cycles

    # Aggregate throughput: only ACTIVE WGs contribute to throughput, but the
    # per-link physical cap is independent of WG count.
    aggregate_bw = min(
        comm_hw.link_bw,                    # per-link physical cap
        eff_wgs * mshr_bw_per_wg,           # CU-limited aggregate (active only)
    )

    T_transfer = total_wire_bytes / aggregate_bw

    # Per-timestep sync: each of the num_timesteps logical steps has signal/wait
    # overhead. With pipelining, sync is overlapped. Only the first and last
    # steps aren't overlapped. Model as: num_timesteps × sync_per_step, where
    # sync_per_step is the atomic RTT (not the full step time, since sync
    # overlaps with transfer).
    entry = layout.link_of(0, 0, my_rank, N)
    sync_ops = sum(1 for op in entry.work_graph if isinstance(op, (Signal, Wait)))
    T_sync_per_step = sync_ops * comm_hw.atomic_latency_cycles
    T_sync_total = num_timesteps * T_sync_per_step

    # Also account for local HBM read/write (Load + Store in work graph).
    # For RS phase: each step reads msg/N from local HBM and writes msg/N.
    # For AG phase: each step writes msg/N.
    # This traffic shares HBM BW with the xGMI traffic.
    # At low WG count, HBM per-CU BW may be the bottleneck.
    # Use `eff_wgs` so HBM polynomial scaling reflects real consumer count.
    hbm_bw_agg = hw.hbm_read_bw * hw._bw_fraction(eff_wgs)
    T_hbm = total_wire_bytes / hbm_bw_agg  # local read side

    T_transfer_total = max(T_transfer, T_hbm)

    # Per-ring-step proxy/sync overhead (heuristic). Additive to T_sync_total
    # (which covers in-graph atomic RTTs from Signal/Wait ops).
    T_step_overhead = _ring_step_overhead_cycles(primitive, layout, heuristics)

    return comm_hw.launch_overhead_cycles + T_transfer_total + T_sync_total + T_step_overhead


def _compute_sequential_latency(
    layout, problem, config, hw, comm_hw, my_rank,
    heuristics: Heuristics = DEFAULT_HEURISTICS,
    primitive: Optional[str] = None,
) -> float:
    """
    Sequential timesteps: each timestep uses a (potentially) different link.
    T = Σ T_timestep across all timesteps.
    """
    N = problem.num_gpus
    num_wgs = config.num_wgs

    # 2D tile hierarchy. Each level is a TileShape (m × n + dtype + split_dim)
    # so that downstream consumers (cacheline-padding accounting, future
    # heterogeneous gemm↔comm interop) can keep the shape; the scalar
    # `cachelines` / `elements` values fed into `compute_wg_tile_latency` are
    # just projections of the 2D tile.
    chunks_per_timestep = layout.chunks_per_timestep
    gpu_tile = problem.gpu_tile_shape
    gpu_timestep_tile = gpu_tile.divide_byte_equal(chunks_per_timestep)

    # Cap WG count by available data — over-launched WGs sit idle and don't
    # contend for HBM/L2/xGMI bandwidth. See `CommConfig.effective_num_wgs`.
    eff_wgs = config.effective_num_wgs(gpu_timestep_tile.bytes)

    wg_tile = gpu_timestep_tile.divide_byte_equal(eff_wgs)
    wg_tile_cachelines = max(wg_tile.cachelines, 1)
    wg_tile_elements = max(wg_tile.elements, 1)

    T_timesteps = 0.0
    for timestep in range(layout.num_timesteps):
        entry = layout.link_of(0, timestep, my_rank, N)

        if entry.is_self:
            # Self-timestep: local HBM only, no xGMI. Per-CU HBM BW uses the
            # effective WG count so that low-active-CU regimes get the full
            # per-CU share rather than a polynomial-discounted slice.
            bw_per_wg = hw.hbm_read_bw_per_cu(eff_wgs)
            breakdown = compute_wg_tile_latency(
                work_graph=entry.work_graph,
                wg_tile_cachelines=wg_tile_cachelines,
                config=config,
                hw=hw,
                comm_hw=comm_hw,
                bw_per_wg=bw_per_wg,
                wg_tile_elements=wg_tile_elements,
                wg_tile=wg_tile,
                active_cus=eff_wgs,
                heuristics=heuristics,
                primitive=primitive,
            )
            T_timesteps += breakdown.T_total_cycles
        else:
            # Layout distributes ACTIVE WGs across links — over-launched
            # idle WGs don't share link BW.
            link_wg_counts = layout.active_links(timestep, eff_wgs, N)

            T_links = []
            for link_id, wgs_on_link in link_wg_counts.items():
                bw_per_wg = comm_hw.link_bw / max(wgs_on_link, 1)

                breakdown = compute_wg_tile_latency(
                    work_graph=entry.work_graph,
                    wg_tile_cachelines=wg_tile_cachelines,
                    config=config,
                    hw=hw,
                    comm_hw=comm_hw,
                    bw_per_wg=bw_per_wg,
                    wg_tile_elements=wg_tile_elements,
                    wg_tile=wg_tile,
                    active_cus=eff_wgs,
                    heuristics=heuristics,
                    primitive=primitive,
                )
                T_links.append(breakdown.T_total_cycles)

            T_timesteps += max(T_links) if T_links else 0.0

    # Per-ring-step proxy/sync overhead (heuristic). Zero for non-ring layouts.
    T_step_overhead = _ring_step_overhead_cycles(primitive, layout, heuristics)

    return comm_hw.launch_overhead_cycles + T_timesteps + T_step_overhead


def predict_row(
    primitive: str,
    msg_bytes: int,
    world_size: int,
    nchannels: int,
    hw: Hardware,
    comm_hw: CommHardware,
    M: int = 0,
    N: int = 0,
    split_dim: int = 0,
    heuristics: Heuristics = DEFAULT_HEURISTICS,
) -> float:
    """Predict latency in microseconds for one row of rccl_master_sweep.csv.

    If M and N are not provided, assumes 1D contiguous layout
    (M=1, N=per_rank_elements). `msg_bytes` follows the RCCL/NCCL CSV
    convention, which is per-op:
        AR / AG / BC / A2A  →  `msg_bytes` is the per-rank tile size
        RS                  →  `msg_bytes` is the *total input* (= W × per-rank)

    The internal `gpu_tile` is normalized to the per-rank tile for every op,
    so RS's `msg_bytes` is divided by `world_size` here. This keeps the
    contract uniform: `gpu_tile` always == per-rank, `chunks_per_timestep`
    handles the algorithm's internal subdivision.
    """
    dtype = DataType.BF16
    if M == 0 or N == 0:
        total_elements = msg_bytes // dtype_bytes(dtype)
        if primitive == "reduce_scatter":
            # CSV/NCCL convention: msg_bytes for RS is the total input across
            # all ranks; the per-rank tile is total / W.
            per_rank_elements = max(total_elements // world_size, 1)
        else:
            per_rank_elements = total_elements
        M = 1
        N = per_rank_elements

    problem = CommProblem(M=M, N=N, num_gpus=world_size, dtype=dtype, split_dim=split_dim)
    config = CommConfig(num_wgs=nchannels)

    # `compute_collective_latency` returns GPU cycles. The cycles → time
    # boundary is governed by the GPU frequency: seconds = cycles / clock_hz.
    # `hw.cycles_to_us` applies this conversion and scales to microseconds.
    T_cycles = compute_collective_latency(
        primitive, problem, config, hw, comm_hw, heuristics=heuristics,
    )
    return hw.cycles_to_us(T_cycles)
