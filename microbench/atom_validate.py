"""
Atom validation microbench (Day 2 of pre-port spike).

Validates the per-WG iteration cost predicted by
`model.latency.compute_iter_times` against direct measurement on MI300X.

The model atom: cycles for one WG to do one iter_tile of work (cl_per_iter
cachelines, often 8 = 1 KB on MI300X). Phase A validates the HBM-write
atom; Phase B validates the [Pull, Store] (xGMI-read + HBM-write) atom.

Outputs (per phase):
- Aggregate hipEvent ns vs model-predicted aggregate ns
- Per-WG s_memrealtime ticks → ns (median, p10, p90, max)
- Atom-alignment % per (num_wgs, bytes_per_wg) cell
- A JSON dump for downstream analysis

Decision criteria (per cpp-port-plan.md Day 3):
  - alignment within 20%  → port with confidence
  - 20–30% misaligned     → port but plan recalibration
  - > 30% misaligned      → defer port

Counter calibration (K-2834): s_memrealtime = 99.77 MHz (10.023 ns/tick)
on MI300X (gfx942). SCLK is ~2.1 GHz under load → ~21.05 cycles/tick if
the kernel runs at boost clock. We always report **ns** to sidestep the
SCLK-drift uncertainty.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch
import triton
import triton.language as tl

# ── model imports (parent of microbench/ is the repo root) ─────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from model.hardware import MI300X, MI300X_COMM
from model.heuristics import DEFAULT_HEURISTICS
from model.latency import compute_iter_times
from model.types import CACHELINE_BYTES, CommConfig, FunctionalUnitWork


# ── s_memrealtime calibration ──────────────────────────────────────
# K-2834: ten cross-host measurements agree on 99.77 MHz / 10.023 ns/tick.
MEMREALTIME_NS_PER_TICK = 10.023


# =====================================================================
# Phase A: store-only (HBM write atom)
# =====================================================================

@triton.jit
def store_only_kernel(
    out_ptr,                              # *fp32 [num_wgs * elements_per_wg]
    ticks_ptr,                            # *int64 [num_wgs]
    elements_per_wg: tl.constexpr,        # contiguous fp32 slice per WG
    elements_per_iter: tl.constexpr,      # = cl_per_iter * CL / 4 (fp32)
    num_iters: tl.constexpr,              # = elements_per_wg / elements_per_iter
):
    """One WG streams `elements_per_wg` fp32 values to its own slice of HBM.

    Brackets the inner store loop with two `s_memrealtime` reads (via
    `tl.extra.hip.memrealtime`). `s_waitcnt vmcnt(0)` after the last
    store ensures the second timer read does not issue before write
    completion (the matrix-pipe-fence equivalent for VMEM stores).
    """
    pid = tl.program_id(0)
    base = pid * elements_per_wg

    offs = tl.arange(0, elements_per_iter)
    val = tl.zeros([elements_per_iter], dtype=tl.float32)

    t0 = tl.extra.hip.memrealtime()

    for i in tl.range(0, num_iters):
        ptr = out_ptr + base + i * elements_per_iter + offs
        tl.store(ptr, val)

    # memrealtime is marked side-effecting in the Triton AMD backend
    # (K-4554 precedent uses it bare). Triton's lowering inserts the
    # required s_waitcnt before issuing the timer read.
    t1 = tl.extra.hip.memrealtime()

    tl.store(ticks_ptr + pid, t1 - t0)


@dataclass
class StoreOnlyResult:
    num_wgs: int
    bytes_per_wg: int
    elements_per_iter: int
    num_iters: int

    # measured
    agg_kernel_ns: float           # hipEvent
    per_wg_median_ns: float        # s_memrealtime
    per_wg_p10_ns: float
    per_wg_p90_ns: float
    per_wg_max_ns: float

    # model prediction (per-WG)
    model_per_wg_ns: float
    model_per_iter_ns: float

    # diff
    alignment_pct: float           # 100 * model / measured_median (1.0 = perfect)


def run_store_only(
    bytes_per_wg: int,
    num_wgs: int,
    elements_per_iter: int = 1024,    # 8 CL = 1 KB (cl_per_iter=8)
    warmup_iters: int = 5,
    timed_iters: int = 20,
) -> StoreOnlyResult:
    """Run one (num_wgs, bytes_per_wg) cell, return measured + model."""
    elements_per_wg = bytes_per_wg // 4
    num_iters = elements_per_wg // elements_per_iter
    assert num_iters * elements_per_iter == elements_per_wg, (
        f"bytes_per_wg={bytes_per_wg} must divide evenly into "
        f"{elements_per_iter}-element iters"
    )

    out = torch.empty(num_wgs * elements_per_wg, device="cuda", dtype=torch.float32)
    ticks = torch.empty(num_wgs, device="cuda", dtype=torch.int64)

    grid = (num_wgs,)

    # warmup
    for _ in range(warmup_iters):
        store_only_kernel[grid](
            out, ticks,
            elements_per_wg=elements_per_wg,
            elements_per_iter=elements_per_iter,
            num_iters=num_iters,
        )
    torch.cuda.synchronize()

    # timed via hipEvent for aggregate
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(timed_iters):
        store_only_kernel[grid](
            out, ticks,
            elements_per_wg=elements_per_wg,
            elements_per_iter=elements_per_iter,
            num_iters=num_iters,
        )
    end_evt.record()
    torch.cuda.synchronize()
    agg_ms = start_evt.elapsed_time(end_evt) / timed_iters
    agg_ns = agg_ms * 1e6

    # per-WG ticks (from the last run; the warmup pattern is steady-state)
    ticks_host = ticks.cpu().numpy().astype("float64")
    per_wg_ns = ticks_host * MEMREALTIME_NS_PER_TICK

    median = float(sorted(per_wg_ns)[len(per_wg_ns) // 2])
    p10 = float(sorted(per_wg_ns)[max(int(0.1 * len(per_wg_ns)) - 1, 0)])
    p90 = float(sorted(per_wg_ns)[min(int(0.9 * len(per_wg_ns)), len(per_wg_ns) - 1)])
    pmax = float(max(per_wg_ns))

    # model prediction: per-WG ns = num_iters × T_hbm_write_cycles → ns
    cl_per_iter = elements_per_iter * 4 // CACHELINE_BYTES   # 1024*4 / 128 = 32 CL
    work = FunctionalUnitWork.zero()
    work.hbm_write_cl = cl_per_iter

    iter_times_cycles = compute_iter_times(
        work=work,
        hw=MI300X,
        comm_hw=MI300X_COMM,
        bw_per_wg=MI300X_COMM.link_bw,    # irrelevant for store-only
        active_cus=num_wgs,
        heuristics=DEFAULT_HEURISTICS,
        primitive=None,
    )
    T_iter_cycles = iter_times_cycles["hbm_write"]
    T_iter_ns = MI300X.cycles_to_ns(T_iter_cycles)
    T_per_wg_ns = T_iter_ns * num_iters

    alignment = 100.0 * T_per_wg_ns / median if median > 0 else 0.0

    return StoreOnlyResult(
        num_wgs=num_wgs,
        bytes_per_wg=bytes_per_wg,
        elements_per_iter=elements_per_iter,
        num_iters=num_iters,
        agg_kernel_ns=agg_ns,
        per_wg_median_ns=median,
        per_wg_p10_ns=p10,
        per_wg_p90_ns=p90,
        per_wg_max_ns=pmax,
        model_per_wg_ns=T_per_wg_ns,
        model_per_iter_ns=T_iter_ns,
        alignment_pct=alignment,
    )


def phase_a_sweep(out_json: Path):
    """Sweep (num_wgs, bytes_per_wg). HBM-write atom validation."""
    print("=" * 78)
    print("Phase A — HBM write atom (store-only, 1 GPU)")
    print("=" * 78)
    print(f"{'WGs':>4} {'B/WG':>10} {'iters':>6} "
          f"{'agg(µs)':>9} {'meas/WG':>9} {'p10/p90':>14} "
          f"{'model/WG':>9} {'align%':>7}")
    print("-" * 78)

    results: List[StoreOnlyResult] = []
    wg_grid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 304]
    bpw_grid = [
        64 * 1024,           # 64 KB
        256 * 1024,          # 256 KB
        1024 * 1024,         # 1 MB
        4 * 1024 * 1024,     # 4 MB
    ]

    for bpw in bpw_grid:
        for nwg in wg_grid:
            try:
                r = run_store_only(bpw, nwg)
            except Exception as e:
                print(f"  skip nwg={nwg} bpw={bpw}: {e}")
                continue
            results.append(r)
            print(
                f"{r.num_wgs:>4} {r.bytes_per_wg:>10} {r.num_iters:>6} "
                f"{r.agg_kernel_ns/1000:>9.2f} {r.per_wg_median_ns/1000:>9.2f} "
                f"{r.per_wg_p10_ns/1000:>5.1f}/{r.per_wg_p90_ns/1000:<7.1f} "
                f"{r.model_per_wg_ns/1000:>9.2f} {r.alignment_pct:>7.1f}"
            )

    # summary statistics
    aligns = [r.alignment_pct for r in results]
    if aligns:
        aligns_sorted = sorted(aligns)
        n = len(aligns_sorted)
        print("-" * 78)
        print(f"Phase A summary over {n} cells:")
        print(f"  median alignment%: {aligns_sorted[n // 2]:.1f}")
        print(f"  p10/p90:           {aligns_sorted[max(int(0.1*n)-1,0)]:.1f} / "
              f"{aligns_sorted[min(int(0.9*n),n-1)]:.1f}")
        print(f"  range:             {min(aligns):.1f} → {max(aligns):.1f}")

    with open(out_json, "w") as f:
        json.dump({"phase_a": [asdict(r) for r in results]}, f, indent=2)
    print(f"\nWrote {out_json}")
    return results


# =====================================================================
# Phase B: [Pull, Store] (xGMI-read + HBM-write atom)
#
# Needs 2 GPUs. Uses iris symmetric-heap pointer translation if iris is
# installed; otherwise falls back to direct cross-device pointer with
# hipDeviceEnablePeerAccess (Phase B0).
# =====================================================================

@triton.jit
def pull_store_kernel(
    src_ptr,                              # remote-GPU buffer pointer
    dst_ptr,                              # local-GPU buffer pointer
    ticks_ptr,
    elements_per_wg: tl.constexpr,
    elements_per_iter: tl.constexpr,
    num_iters: tl.constexpr,
):
    """One WG pulls a slice from `src_ptr` (remote, via P2P) and stores
    it to `dst_ptr` (local HBM)."""
    pid = tl.program_id(0)
    base = pid * elements_per_wg
    offs = tl.arange(0, elements_per_iter)

    t0 = tl.extra.hip.memrealtime()

    for i in tl.range(0, num_iters):
        src_off = src_ptr + base + i * elements_per_iter + offs
        dst_off = dst_ptr + base + i * elements_per_iter + offs
        x = tl.load(src_off)
        tl.store(dst_off, x)

    t1 = tl.extra.hip.memrealtime()

    tl.store(ticks_ptr + pid, t1 - t0)


@dataclass
class PullStoreResult:
    num_wgs: int
    bytes_per_wg: int
    num_iters: int

    agg_kernel_ns: float
    per_wg_median_ns: float
    per_wg_p10_ns: float
    per_wg_p90_ns: float

    model_per_wg_ns: float
    model_per_iter_ns: float
    model_bottleneck: str          # "hbm_write" / "xgmi_read" / etc.
    alignment_pct: float


def _enable_peer_access():
    """ROCm/HIP: peer access is enabled-by-default in many configs; this is
    idempotent. Returns (src=0, dst=1) device pair."""
    assert torch.cuda.device_count() >= 2, "Need ≥ 2 GPUs for Phase B"
    # No explicit hipDeviceEnablePeerAccess from torch; modern rocm-pytorch
    # auto-enables peer access on cross-device pointer use.
    return 0, 1


def run_pull_store(
    bytes_per_wg: int,
    num_wgs: int,
    elements_per_iter: int = 1024,
    warmup_iters: int = 5,
    timed_iters: int = 20,
) -> PullStoreResult:
    src_dev, dst_dev = _enable_peer_access()

    elements_per_wg = bytes_per_wg // 4
    num_iters = elements_per_wg // elements_per_iter
    assert num_iters * elements_per_iter == elements_per_wg

    src = torch.randn(
        num_wgs * elements_per_wg,
        device=f"cuda:{src_dev}", dtype=torch.float32,
    )
    torch.cuda.set_device(dst_dev)
    dst = torch.empty(
        num_wgs * elements_per_wg,
        device=f"cuda:{dst_dev}", dtype=torch.float32,
    )
    ticks = torch.empty(num_wgs, device=f"cuda:{dst_dev}", dtype=torch.int64)

    grid = (num_wgs,)
    for _ in range(warmup_iters):
        pull_store_kernel[grid](
            src, dst, ticks,
            elements_per_wg=elements_per_wg,
            elements_per_iter=elements_per_iter,
            num_iters=num_iters,
        )
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(timed_iters):
        pull_store_kernel[grid](
            src, dst, ticks,
            elements_per_wg=elements_per_wg,
            elements_per_iter=elements_per_iter,
            num_iters=num_iters,
        )
    end_evt.record()
    torch.cuda.synchronize()
    agg_ns = (start_evt.elapsed_time(end_evt) / timed_iters) * 1e6

    ticks_host = ticks.cpu().numpy().astype("float64")
    per_wg_ns = ticks_host * MEMREALTIME_NS_PER_TICK
    s = sorted(per_wg_ns)
    median = float(s[len(s) // 2])
    p10 = float(s[max(int(0.1 * len(s)) - 1, 0)])
    p90 = float(s[min(int(0.9 * len(s)), len(s) - 1)])

    # model prediction
    cl_per_iter = elements_per_iter * 4 // CACHELINE_BYTES
    work = FunctionalUnitWork.zero()
    work.hbm_write_cl = cl_per_iter
    work.xgmi_read_cl = cl_per_iter

    # bw_per_wg = link_bw / wgs_on_link. With both endpoints on one link
    # (single direct P2P), wgs_on_link = num_wgs.
    bw_per_wg = MI300X_COMM.link_bw / num_wgs
    iter_times = compute_iter_times(
        work=work,
        hw=MI300X,
        comm_hw=MI300X_COMM,
        bw_per_wg=bw_per_wg,
        active_cus=num_wgs,
        heuristics=DEFAULT_HEURISTICS,
        primitive="all_gather",   # uses calibrated k for write conc.
    )
    # the atom is one iter = max over FUs (concurrent issue assumption)
    T_iter_cycles = max(iter_times.values())
    bottleneck = max(iter_times, key=iter_times.get)
    T_iter_ns = MI300X.cycles_to_ns(T_iter_cycles)
    T_per_wg_ns = T_iter_ns * num_iters
    alignment = 100.0 * T_per_wg_ns / median if median > 0 else 0.0

    return PullStoreResult(
        num_wgs=num_wgs,
        bytes_per_wg=bytes_per_wg,
        num_iters=num_iters,
        agg_kernel_ns=agg_ns,
        per_wg_median_ns=median,
        per_wg_p10_ns=p10,
        per_wg_p90_ns=p90,
        model_per_wg_ns=T_per_wg_ns,
        model_per_iter_ns=T_iter_ns,
        model_bottleneck=bottleneck,
        alignment_pct=alignment,
    )


def phase_b_sweep(out_json: Path):
    print("=" * 78)
    print("Phase B — [Pull, Store] atom (xGMI read + HBM write, 2 GPUs)")
    print("=" * 78)
    print(f"{'WGs':>4} {'B/WG':>10} {'iters':>6} "
          f"{'agg(µs)':>9} {'meas/WG':>9} {'p10/p90':>14} "
          f"{'model/WG':>9} {'bot':>10} {'align%':>7}")
    print("-" * 90)

    results: List[PullStoreResult] = []
    wg_grid = [1, 2, 4, 8, 16, 32, 64]
    bpw_grid = [64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024]

    for bpw in bpw_grid:
        for nwg in wg_grid:
            try:
                r = run_pull_store(bpw, nwg)
            except Exception as e:
                print(f"  skip nwg={nwg} bpw={bpw}: {e}")
                continue
            results.append(r)
            print(
                f"{r.num_wgs:>4} {r.bytes_per_wg:>10} {r.num_iters:>6} "
                f"{r.agg_kernel_ns/1000:>9.2f} {r.per_wg_median_ns/1000:>9.2f} "
                f"{r.per_wg_p10_ns/1000:>5.1f}/{r.per_wg_p90_ns/1000:<7.1f} "
                f"{r.model_per_wg_ns/1000:>9.2f} {r.model_bottleneck:>10} "
                f"{r.alignment_pct:>7.1f}"
            )

    aligns = [r.alignment_pct for r in results]
    if aligns:
        s = sorted(aligns)
        n = len(s)
        print("-" * 90)
        print(f"Phase B summary over {n} cells:")
        print(f"  median alignment%: {s[n // 2]:.1f}")
        print(f"  p10/p90:           {s[max(int(0.1*n)-1,0)]:.1f} / "
              f"{s[min(int(0.9*n),n-1)]:.1f}")
        print(f"  range:             {min(aligns):.1f} → {max(aligns):.1f}")

    # append to JSON
    blob = {}
    if out_json.exists():
        blob = json.loads(out_json.read_text())
    blob["phase_b"] = [asdict(r) for r in results]
    out_json.write_text(json.dumps(blob, indent=2))
    print(f"\nAppended Phase B to {out_json}")
    return results


# =====================================================================
# main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["a", "b", "ab"], default="ab")
    ap.add_argument("--out", type=Path, default=Path("microbench/atom_validate.json"))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "ROCm/CUDA not available"
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Triton: {triton.__version__}")
    print(f"Torch:  {torch.__version__}")
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if "a" in args.phase:
        phase_a_sweep(args.out)
    if "b" in args.phase:
        if torch.cuda.device_count() < 2:
            print("Phase B skipped: need ≥ 2 GPUs")
        else:
            phase_b_sweep(args.out)


if __name__ == "__main__":
    main()
