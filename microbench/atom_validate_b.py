"""
Phase B of Day 2 atom validation: xGMI-read + HBM-write atom on 2 GPUs.

Uses iris symmetric heap for safe cross-GPU pointer translation.
Launch with:
  torchrun --nproc-per-node=2 microbench/atom_validate_b.py \
      [--out OUT.json] [--world-size N]

Each rank participates in the symmetric heap. Rank 1 launches the
benchmark kernel that does iris.load(remote=rank 0) + local store, with
s_memrealtime brackets per WG. Per-WG measured ns are compared against
the model's prediction for one full wg_tile (num_iters × T_iter, where
T_iter is the max of T_xgmi_read and T_hbm_write).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from model.hardware import MI300X, MI300X_COMM
from model.heuristics import DEFAULT_HEURISTICS
from model.latency import compute_iter_times
from model.types import CACHELINE_BYTES, FunctionalUnitWork


MEMREALTIME_NS_PER_TICK = 10.023   # K-2834, MI300X gfx942


@triton.jit
def pull_store_kernel(
    src_ptr,                              # symmetric buffer (remote-side data)
    dst_ptr,                              # symmetric buffer (local write target)
    ticks_ptr,                            # local int64[num_wgs]
    heap_bases,
    cur_rank: tl.constexpr,
    remote_rank: tl.constexpr,
    elements_per_wg: tl.constexpr,
    elements_per_iter: tl.constexpr,
    num_iters: tl.constexpr,
):
    """One WG pulls a slice from remote_rank's `src_ptr` slice into
    registers, then stores it into the local rank's `dst_ptr` slice.

    Bracket: s_memrealtime via tl.extra.hip.memrealtime."""
    pid = tl.program_id(0)
    base = pid * elements_per_wg
    offs = tl.arange(0, elements_per_iter)

    t0 = tl.extra.hip.memrealtime()

    for i in tl.range(0, num_iters):
        src_off = src_ptr + base + i * elements_per_iter + offs
        dst_off = dst_ptr + base + i * elements_per_iter + offs
        x = iris.load(src_off, remote_rank, cur_rank, heap_bases)
        tl.store(dst_off, x)

    t1 = tl.extra.hip.memrealtime()
    tl.store(ticks_ptr + pid, t1 - t0)


@dataclass
class PullStoreResult:
    num_wgs: int
    bytes_per_wg: int
    num_iters: int
    elements_per_iter: int

    agg_kernel_ns: float
    per_wg_median_ns: float
    per_wg_p10_ns: float
    per_wg_p90_ns: float

    model_per_wg_ns: float
    model_per_iter_ns: float
    model_bottleneck: str
    alignment_pct: float


def run_pull_store_iris(
    iris_ctx,
    bytes_per_wg: int,
    num_wgs: int,
    elements_per_iter: int = 1024,
    warmup_iters: int = 5,
    timed_iters: int = 20,
) -> PullStoreResult:
    elements_per_wg = bytes_per_wg // 4
    num_iters = elements_per_wg // elements_per_iter
    assert num_iters * elements_per_iter == elements_per_wg, \
        f"bpw={bytes_per_wg} must divide {4*elements_per_iter}"

    # symmetric allocation — same shape on every rank
    src = iris_ctx.randn(num_wgs * elements_per_wg, device="cuda", dtype=torch.float32)
    dst = iris_ctx.zeros(num_wgs * elements_per_wg, device="cuda", dtype=torch.float32)
    ticks = torch.zeros(num_wgs, device="cuda", dtype=torch.int64)

    cur = iris_ctx.get_rank()
    # producer = rank 0, consumer = rank 1. Only consumer launches.
    cur_rank = 1
    remote_rank = 0

    if cur != cur_rank:
        iris_ctx.barrier()
        iris_ctx.barrier()  # match the warmup+timed barriers in the consumer
        # consumer-side results are returned only from cur_rank
        return None  # type: ignore

    grid = (num_wgs,)
    for _ in range(warmup_iters):
        pull_store_kernel[grid](
            src, dst, ticks,
            iris_ctx.get_heap_bases(),
            cur_rank=cur_rank,
            remote_rank=remote_rank,
            elements_per_wg=elements_per_wg,
            elements_per_iter=elements_per_iter,
            num_iters=num_iters,
        )
    torch.cuda.synchronize()
    iris_ctx.barrier()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(timed_iters):
        pull_store_kernel[grid](
            src, dst, ticks,
            iris_ctx.get_heap_bases(),
            cur_rank=cur_rank,
            remote_rank=remote_rank,
            elements_per_wg=elements_per_wg,
            elements_per_iter=elements_per_iter,
            num_iters=num_iters,
        )
    end_evt.record()
    torch.cuda.synchronize()
    iris_ctx.barrier()
    agg_ns = (start_evt.elapsed_time(end_evt) / timed_iters) * 1e6

    ticks_host = ticks.cpu().numpy().astype("float64")
    per_wg_ns = ticks_host * MEMREALTIME_NS_PER_TICK
    s = sorted(per_wg_ns)
    n = len(s)
    median = float(s[n // 2])
    p10 = float(s[max(int(0.1 * n) - 1, 0)])
    p90 = float(s[min(int(0.9 * n), n - 1)])

    # model prediction (per-WG, one tile = num_iters × T_iter atoms)
    cl_per_iter = elements_per_iter * 4 // CACHELINE_BYTES
    work = FunctionalUnitWork.zero()
    work.hbm_write_cl = cl_per_iter
    work.xgmi_read_cl = cl_per_iter
    bw_per_wg = MI300X_COMM.link_bw / num_wgs   # single P2P link, all WGs share
    iter_times = compute_iter_times(
        work=work,
        hw=MI300X,
        comm_hw=MI300X_COMM,
        bw_per_wg=bw_per_wg,
        active_cus=num_wgs,
        heuristics=DEFAULT_HEURISTICS,
        primitive="all_gather",
    )
    T_iter_cycles = max(iter_times.values())
    bottleneck = max(iter_times, key=iter_times.get)
    T_iter_ns = MI300X.cycles_to_ns(T_iter_cycles)
    T_per_wg_ns = T_iter_ns * num_iters
    alignment = 100.0 * T_per_wg_ns / median if median > 0 else 0.0

    return PullStoreResult(
        num_wgs=num_wgs,
        bytes_per_wg=bytes_per_wg,
        num_iters=num_iters,
        elements_per_iter=elements_per_iter,
        agg_kernel_ns=agg_ns,
        per_wg_median_ns=median,
        per_wg_p10_ns=p10,
        per_wg_p90_ns=p90,
        model_per_wg_ns=T_per_wg_ns,
        model_per_iter_ns=T_iter_ns,
        model_bottleneck=bottleneck,
        alignment_pct=alignment,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("microbench/atom_validate.json"))
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29501",
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    heap_size = 1 << 30  # 1 GiB
    iris_ctx = iris.iris(heap_size)

    if rank == 1:
        print(f"[rank{rank}] starting Phase B sweep on {torch.cuda.get_device_name(local_rank)}")
        print(f"  triton={triton.__version__}  torch={torch.__version__}")
        print("=" * 88)
        print("Phase B — [Pull, Store] atom (xGMI read + HBM write, 2 GPUs)")
        print("=" * 88)
        print(f"{'WGs':>4} {'B/WG':>10} {'iters':>6} "
              f"{'agg(µs)':>9} {'meas/WG':>9} {'p10/p90':>14} "
              f"{'model/WG':>9} {'bot':>10} {'align%':>7}")
        print("-" * 95)

    results: List[PullStoreResult] = []
    wg_grid = [1, 2, 4, 8, 16, 32, 64]
    bpw_grid = [64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024]

    for bpw in bpw_grid:
        for nwg in wg_grid:
            try:
                r = run_pull_store_iris(iris_ctx, bpw, nwg)
            except Exception as e:
                if rank == 1:
                    print(f"  skip nwg={nwg} bpw={bpw}: {type(e).__name__}: {e}")
                iris_ctx.barrier()
                iris_ctx.barrier()
                continue
            if rank == 1 and r is not None:
                results.append(r)
                print(
                    f"{r.num_wgs:>4} {r.bytes_per_wg:>10} {r.num_iters:>6} "
                    f"{r.agg_kernel_ns/1000:>9.2f} {r.per_wg_median_ns/1000:>9.2f} "
                    f"{r.per_wg_p10_ns/1000:>5.1f}/{r.per_wg_p90_ns/1000:<7.1f} "
                    f"{r.model_per_wg_ns/1000:>9.2f} {r.model_bottleneck:>10} "
                    f"{r.alignment_pct:>7.1f}"
                )

    if rank == 1:
        aligns = [r.alignment_pct for r in results]
        if aligns:
            s = sorted(aligns)
            n = len(s)
            print("-" * 95)
            print(f"Phase B summary over {n} cells:")
            print(f"  median alignment%: {s[n // 2]:.1f}")
            print(f"  p10/p90:           {s[max(int(0.1*n)-1,0)]:.1f} / "
                  f"{s[min(int(0.9*n),n-1)]:.1f}")
            print(f"  range:             {min(aligns):.1f} → {max(aligns):.1f}")

        blob = {}
        if args.out.exists():
            blob = json.loads(args.out.read_text())
        blob["phase_b"] = [asdict(r) for r in results]
        args.out.write_text(json.dumps(blob, indent=2))
        print(f"\nAppended Phase B to {args.out}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
