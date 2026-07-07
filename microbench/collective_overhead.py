"""
Microbenchmark 3+4: Collective static overhead and single-hop transfer.

Measures:
- Zero-payload overhead: AllReduce/AllGather with minimum message size
  → isolates launch + setup + sync cost
- Single-hop transfer: W=2 AllGather at various sizes
  → subtract zero-payload overhead → pure transfer time
  → compare against size / link_bw to validate bandwidth assumption

Uses torch.distributed (RCCL backend) — no custom kernels needed.
"""

import os
import torch
import torch.distributed as dist
import time
import argparse


def setup_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def bench_collective(collective, tensor, warmup=10, iters=50):
    """Time a collective operation, return median latency in microseconds."""
    torch.cuda.synchronize()

    for _ in range(warmup):
        if collective == "allreduce":
            dist.all_reduce(tensor)
        elif collective == "allgather":
            out = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(out, tensor)
        elif collective == "reduce_scatter":
            out = torch.empty_like(tensor)
            dist.reduce_scatter(out, [tensor] * dist.get_world_size())
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        if collective == "allreduce":
            dist.all_reduce(tensor)
        elif collective == "allgather":
            out = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(out, tensor)
        elif collective == "reduce_scatter":
            out = torch.empty_like(tensor)
            dist.reduce_scatter(out, [tensor] * dist.get_world_size())
        torch.cuda.synchronize()
        elapsed_ns = time.perf_counter_ns() - start
        times.append(elapsed_ns / 1000)  # to microseconds

    times.sort()
    return times[len(times) // 2]  # median


def main():
    rank, world_size = setup_dist()

    if rank == 0:
        print(f"Collective overhead measurement: {world_size} GPUs")
        print()

    sizes = [0, 64, 256, 1024, 4096, 16384, 65536, 262144,
             1048576, 4194304, 16777216, 67108864, 268435456]

    for collective in ["allreduce", "allgather"]:
        if rank == 0:
            print(f"=== {collective} W={world_size} ===")
            print(f"{'msg_bytes':>12} {'latency_us':>12} {'bw_GBps':>10} {'overhead_us':>12}")
            print("-" * 50)

        baseline_us = None
        for msg_bytes in sizes:
            n_elements = max(msg_bytes // 2, 1)  # bf16
            tensor = torch.zeros(n_elements, device=f"cuda:{rank}", dtype=torch.bfloat16)

            lat_us = bench_collective(collective, tensor)

            if msg_bytes == 0:
                baseline_us = lat_us

            bw = msg_bytes / (lat_us * 1000) if lat_us > 0 and msg_bytes > 0 else 0
            overhead = lat_us - (msg_bytes / (49.2 * 1000)) if msg_bytes > 0 else lat_us

            if rank == 0:
                print(f"{msg_bytes:>12} {lat_us:>12.1f} {bw:>10.2f} {overhead:>12.1f}")

        if rank == 0:
            print(f"\nZero-payload baseline: {baseline_us:.1f} us")
            print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
