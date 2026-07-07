"""
Fallback collective sweep using torch.distributed if rccl-tests unavailable.
Run with: torchrun --nproc_per_node=8 sweep_torch.py
"""
import os
import torch
import torch.distributed as dist
import time
import csv

def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, ws

def bench(fn, warmup=20, iters=100):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1000)  # µs
    times.sort()
    return times[len(times)//2]

def main():
    rank, ws = setup()

    AMP = {
        "all_reduce": 2 * (ws-1) / ws,
        "all_gather": (ws-1),  # RCCL busbw formula
        "reduce_scatter": (ws-1) / ws,
    }

    sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304,
             16777216, 67108864, 268435456, 1073741824]

    outdir = "/home/ryaswann/mc2-workspaces/origami-comms-bench/results"
    os.makedirs(outdir, exist_ok=True)

    for coll in ["all_reduce", "all_gather", "reduce_scatter"]:
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"  {coll}  W={ws}")
            print(f"{'='*50}")
            print(f"{'size':>12} {'lat_us':>10} {'algobw':>10} {'busbw':>10}")
            f = open(f"{outdir}/{coll}_torch.csv", "w")
            writer = csv.writer(f)
            writer.writerow(["msg_bytes", "world_size", "latency_us", "algobw_gbps", "busbw_gbps"])

        for msg_bytes in sizes:
            n_elem = msg_bytes // 2  # bfloat16
            tensor = torch.zeros(n_elem, device=f"cuda:{rank}", dtype=torch.bfloat16)

            if coll == "all_reduce":
                lat = bench(lambda: dist.all_reduce(tensor))
            elif coll == "all_gather":
                out = [torch.empty_like(tensor) for _ in range(ws)]
                lat = bench(lambda: dist.all_gather(out, tensor))
            elif coll == "reduce_scatter":
                inp = [torch.empty_like(tensor) for _ in range(ws)]
                out = torch.empty_like(tensor)
                lat = bench(lambda: dist.reduce_scatter(out, inp))

            algobw = msg_bytes / (lat * 1000)  # GB/s
            busbw = algobw * AMP[coll]

            if rank == 0:
                print(f"{msg_bytes:>12} {lat:>10.1f} {algobw:>10.2f} {busbw:>10.2f}")
                writer.writerow([msg_bytes, ws, f"{lat:.1f}", f"{algobw:.4f}", f"{busbw:.4f}"])

        if rank == 0:
            f.close()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
