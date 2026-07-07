"""Tensor-shape collective sweep for the Tensor Collective layer.

Generates `data/tensor_shapes_sweep.csv` — the validation corpus for
`model.tensor_collective.predict_tensor_collective`. Each row carries the
full (op, world_size, shape, dim, dtype) call and the measured per-rank
latency (median + p10 + p90), so the dashboard can plot predicted vs
measured straight from the file.

Run via torchrun, e.g.:

    torchrun --nproc_per_node=8 microbench/sweep_torch_shapes.py \
        --out data/tensor_shapes_sweep.csv

The CSV schema is a strict superset of `rccl_master_sweep.csv`, so the
byte-level dashboard pages can also read this file (they will just see
extra columns they don't use).

Output schema (one row per (op, world_size, shape, dim, dtype) sample):

    primitive, world_size, nchannels, dtype, dim,
    shape, shape_ndim, shape_m, shape_n,
    M_full, N_full, split_dim,
    per_rank_elements, per_rank_bytes, msg_bytes, wire_bytes_per_rank,
    latency_us, latency_p10_us, latency_p90_us,
    algobw_gbps, busbw_gbps,
    iters, warmup, hostname, timestamp,
    # legacy aliases (for byte-level dashboard compat)
    bandwidth_gbps, algorithm

`msg_bytes` follows the rccl-tests / CSV convention (per-rank input for
all_reduce/all_gather/broadcast/all_to_all; total input for
reduce_scatter) so byte-level prediction tooling stays a single code path.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from math import prod
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


# ────────────────────────────────────────────────────────────────────────────
# Sweep configuration
# ────────────────────────────────────────────────────────────────────────────

# Per-rank shape templates. Realistic transformer geometries plus a few
# stress points that exercise cacheline-padding regimes the 1D byte model
# can't see.
#
# Tag layout:
#   1d_<bytes>        — flat 1D baseline (parity check vs RCCL CSV)
#   tf_<M>x<H>        — transformer-style (B*S, H)
#   wide_<M>x<H>      — wide tensor (M small, H large)
#   tall_<M>x<H>      — tall-skinny tensor (M large, H small)
#   subcl_<M>x<H>     — sub-cacheline row width (exercises 2D padding model)

@dataclass(frozen=True)
class ShapeSpec:
    tag: str
    shape: Tuple[int, ...]

    @property
    def elements(self) -> int:
        return prod(self.shape)


def _flat_shapes() -> List[ShapeSpec]:
    """1D buffers matching the byte-level corpus message-size sweep."""
    # Powers of 4 from 1 KiB to 256 MiB, BF16 element count.
    out = []
    bytes_list = [1 << k for k in range(10, 29)]  # 1 KiB … 256 MiB
    for nb in bytes_list:
        n_elem = max(nb // 2, 1)  # bf16
        out.append(ShapeSpec(tag=f"1d_{nb}B", shape=(n_elem,)))
    return out


def _transformer_shapes() -> List[ShapeSpec]:
    """(BS, H) per-rank shapes: typical transformer activations / weights."""
    # B*S ∈ {1024, 4096, 16384}; H ∈ {1024, 2048, 4096, 8192, 12288, 16384}.
    shapes = []
    for bs in (1024, 4096, 16384):
        for h in (1024, 2048, 4096, 8192, 12288, 16384):
            shapes.append(ShapeSpec(tag=f"tf_{bs}x{h}", shape=(bs, h)))
    return shapes


def _aspect_ratio_shapes() -> List[ShapeSpec]:
    """Same total elements, varied M/N aspect ratio. ~4 MiB per shape."""
    # 2M elements total ≈ 4 MiB BF16. Walk aspect from very-tall to very-wide.
    pairs = [
        (1,         2_097_152),   # 1×N  → flat
        (8,         262_144),
        (64,         32_768),
        (512,         4_096),
        (4_096,         512),
        (32_768,         64),     # sub-cacheline rows (128 B → 2 CLs, fine)
        (262_144,         8),     # 16 B rows → sub-CL
        (2_097_152,       1),     # 1-column → degenerate
    ]
    return [ShapeSpec(tag=f"ar_{m}x{n}", shape=(m, n)) for (m, n) in pairs]


def _subcacheline_shapes() -> List[ShapeSpec]:
    """Shapes whose row payload is sub-cacheline — stresses 2D padding."""
    # 256 KiB total, BF16; row counts that put row bytes ≤ 64 B.
    out = []
    for m, n in [(8192, 16), (16384, 8), (32768, 4), (65536, 2)]:
        out.append(ShapeSpec(tag=f"subcl_{m}x{n}", shape=(m, n)))
    return out


def default_shape_sweep() -> List[ShapeSpec]:
    return [
        *_flat_shapes(),
        *_transformer_shapes(),
        *_aspect_ratio_shapes(),
        *_subcacheline_shapes(),
    ]


# Per-op valid `dim` arguments. `dim=-1` is normalized later.
#   - all_reduce / broadcast: dim irrelevant (whole tensor moves) → only dim=0
#   - all_gather / reduce_scatter / all_to_all: try both dim=0 and dim=-1
DIMS_FOR_OP = {
    "all_reduce":     (0,),
    "broadcast":      (0,),
    "all_gather":     (0, -1),
    "reduce_scatter": (0, -1),
    "all_to_all":     (0, -1),
}


# ────────────────────────────────────────────────────────────────────────────
# Torch dtype handling
# ────────────────────────────────────────────────────────────────────────────

_TORCH_DTYPE = {
    "bf16":     torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16":     torch.float16,
    "float16":  torch.float16,
    "half":     torch.float16,
    "fp32":     torch.float32,
    "float32":  torch.float32,
    "float":    torch.float32,
}


def _dtype_obj(name: str) -> torch.dtype:
    if name not in _TORCH_DTYPE:
        raise ValueError(f"unsupported dtype: {name!r}")
    return _TORCH_DTYPE[name]


def _dtype_bytes(name: str) -> int:
    return torch.empty((), dtype=_dtype_obj(name)).element_size()


# ────────────────────────────────────────────────────────────────────────────
# CUDA-event timing
# ────────────────────────────────────────────────────────────────────────────

def _bench_cuda(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> Tuple[float, float, float]:
    """Time `fn` with CUDA events. Returns (median_us, p10_us, p90_us)."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Each iter gets its own pair of events so we capture per-call timing
    # without a host sync between them (lets the queue overlap host work).
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()

    times_us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, stops))
    n = len(times_us)
    median = times_us[n // 2]
    p10 = times_us[int(0.10 * n)]
    p90 = times_us[int(0.90 * n) - 1 if n > 0 else 0]
    return median, p10, p90


# ────────────────────────────────────────────────────────────────────────────
# Collective dispatch
# ────────────────────────────────────────────────────────────────────────────

def _make_fn(
    op: str,
    shape: Sequence[int],
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    world_size: int,
) -> Tuple[Callable[[], None], int, int]:
    """Build a callable `fn()` for the given collective shape + dim.

    Returns (fn, per_rank_bytes, msg_bytes_csv_convention).
    """
    ndim = len(shape)
    norm_dim = dim if dim >= 0 else dim + ndim
    if not (0 <= norm_dim < ndim):
        raise ValueError(f"dim={dim} out of range for shape {shape}")

    per_rank_elements = prod(shape)
    eb = torch.empty((), dtype=dtype).element_size()
    per_rank_bytes = per_rank_elements * eb

    if op == "all_reduce":
        tensor = torch.zeros(*shape, device=device, dtype=dtype)
        def fn() -> None:
            dist.all_reduce(tensor)
        return fn, per_rank_bytes, per_rank_bytes

    if op == "broadcast":
        tensor = torch.zeros(*shape, device=device, dtype=dtype)
        def fn() -> None:
            dist.broadcast(tensor, src=0)
        return fn, per_rank_bytes, per_rank_bytes

    if op == "all_gather":
        # Output buffer is shape × W along `dim`.
        send = torch.zeros(*shape, device=device, dtype=dtype)
        out_shape = list(shape)
        out_shape[norm_dim] *= world_size
        recv = torch.zeros(*out_shape, device=device, dtype=dtype)
        # all_gather_into_tensor wants contiguous along dim=0; if dim>0 we
        # need the list form. Use the list form universally for simplicity.
        chunks = list(torch.zeros_like(send) for _ in range(world_size))
        def fn() -> None:
            dist.all_gather(chunks, send)
        return fn, per_rank_bytes, per_rank_bytes

    if op == "reduce_scatter":
        # Input = per-rank output × W along `dim`; output = per-rank shape.
        in_shape = list(shape)
        in_shape[norm_dim] *= world_size
        in_buf = torch.zeros(*in_shape, device=device, dtype=dtype)
        out_buf = torch.zeros(*shape, device=device, dtype=dtype)
        # reduce_scatter wants a list of W shards, each per-rank-sized.
        in_list = list(torch.zeros(*shape, device=device, dtype=dtype) for _ in range(world_size))
        del in_buf  # we built it for size accounting, but use the list form
        # CSV msg_bytes convention for RS is the total input.
        msg_bytes_csv = per_rank_bytes * world_size
        def fn() -> None:
            dist.reduce_scatter(out_buf, in_list)
        return fn, per_rank_bytes, msg_bytes_csv

    if op == "all_to_all":
        # Split per-rank tensor into W shards along `dim`.
        if shape[norm_dim] % world_size != 0:
            raise ValueError(
                f"all_to_all needs shape[{norm_dim}]={shape[norm_dim]} "
                f"divisible by world_size={world_size}"
            )
        in_t = torch.zeros(*shape, device=device, dtype=dtype)
        out_t = torch.zeros_like(in_t)
        in_chunks = list(torch.chunk(in_t, world_size, dim=norm_dim))
        out_chunks = list(torch.chunk(out_t, world_size, dim=norm_dim))
        # all_to_all needs contiguous chunks
        in_chunks = [c.contiguous() for c in in_chunks]
        out_chunks = [c.contiguous() for c in out_chunks]
        def fn() -> None:
            dist.all_to_all(out_chunks, in_chunks)
        return fn, per_rank_bytes, per_rank_bytes

    raise ValueError(f"unsupported op: {op}")


# ────────────────────────────────────────────────────────────────────────────
# Wire factor (mirror of model.tensor_collective)
# ────────────────────────────────────────────────────────────────────────────

def _wire_factor(op: str, world_size: int) -> float:
    n = world_size
    if op == "all_reduce":      return 2 * (n - 1) / n
    if op == "all_gather":      return (n - 1)
    if op == "reduce_scatter":  return (n - 1)
    if op == "broadcast":       return 1.0
    if op == "all_to_all":      return (n - 1) / n
    raise ValueError(f"unknown op: {op}")


# ────────────────────────────────────────────────────────────────────────────
# Sweep driver
# ────────────────────────────────────────────────────────────────────────────

def _build_sweep_points(
    ops: Sequence[str],
    shapes: Sequence[ShapeSpec],
    dtypes: Sequence[str],
    world_size: int,
    max_per_rank_bytes: int,
) -> List[Tuple[str, ShapeSpec, int, str]]:
    """Enumerate all (op, shape, dim, dtype) points eligible for this run."""
    points: List[Tuple[str, ShapeSpec, int, str]] = []
    for op in ops:
        for dt in dtypes:
            eb = _dtype_bytes(dt)
            for spec in shapes:
                per_rank_bytes = spec.elements * eb
                if per_rank_bytes > max_per_rank_bytes:
                    continue
                # Skip 1D shapes for ops where dim=-1 is the same as dim=0
                for dim in DIMS_FOR_OP[op]:
                    norm_dim = dim if dim >= 0 else dim + len(spec.shape)
                    # For 1D shapes, dim=0 and dim=-1 are equivalent — keep only one.
                    if len(spec.shape) == 1 and dim != 0:
                        continue
                    # For all_to_all, need shape[dim] divisible by world_size.
                    if op == "all_to_all" and spec.shape[norm_dim] % world_size != 0:
                        continue
                    points.append((op, spec, dim, dt))
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tensor_shapes_sweep.csv",
                    help="Output CSV path (appended to if it exists).")
    ap.add_argument("--ops", default="all_reduce,all_gather,reduce_scatter,all_to_all,broadcast",
                    help="Comma-separated ops to sweep.")
    ap.add_argument("--dtypes", default="bf16",
                    help="Comma-separated dtypes (bf16, fp16, fp32).")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--max-per-rank-mb", type=float, default=256.0,
                    help="Skip shapes whose per-rank bytes exceed this (MiB).")
    ap.add_argument("--smoke", action="store_true",
                    help="Use a tiny shape list (sanity check). Does not need torchrun.")
    ap.add_argument("--nchannels", type=int, default=0,
                    help="Record this `nchannels` in the CSV (0 = unknown/auto).")
    args = ap.parse_args()

    is_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_torchrun:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())
    else:
        # Single-process smoke mode (gloo backend, world_size=1 — collectives
        # are no-ops but the dispatch + timing + CSV schema get exercised).
        if not args.smoke:
            print("Not launched via torchrun. Pass --smoke for a single-process dry run.", file=sys.stderr)
            sys.exit(1)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29499")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank, world_size = 0, 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")

    ops = [s.strip() for s in args.ops.split(",") if s.strip()]
    dtypes = [s.strip() for s in args.dtypes.split(",") if s.strip()]

    if args.smoke:
        shapes = [
            ShapeSpec(tag="1d_4KiB", shape=(2048,)),
            ShapeSpec(tag="tf_1024x1024", shape=(1024, 1024)),
            ShapeSpec(tag="subcl_8192x16", shape=(8192, 16)),
        ]
    else:
        shapes = default_shape_sweep()

    max_per_rank_bytes = int(args.max_per_rank_mb * (1 << 20))
    points = _build_sweep_points(ops, shapes, dtypes, world_size, max_per_rank_bytes)

    if rank == 0:
        print(f"# Host: {socket.gethostname()}  device: {device}", flush=True)
        print(f"# World size: {world_size}   ops: {ops}   dtypes: {dtypes}", flush=True)
        print(f"# Shapes: {len(shapes)}   total points: {len(points)}", flush=True)

    # Open CSV on rank 0 only
    out_path = args.out
    writer = None
    csv_fp = None
    if rank == 0:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
        csv_fp = open(out_path, "a", newline="")
        writer = csv.writer(csv_fp)
        if write_header:
            writer.writerow([
                "nchannels", "world_size", "msg_bytes", "primitive",
                "bandwidth_gbps", "latency_us", "busbw_gbps",
                "algorithm", "hostname", "timestamp",
                # Tensor-collective extensions
                "dtype", "dim", "shape", "shape_ndim",
                "shape_m", "shape_n",
                "per_rank_elements", "per_rank_bytes",
                "wire_bytes_per_rank",
                "latency_p10_us", "latency_p90_us",
                "iters", "warmup", "tag",
            ])

    hostname = socket.gethostname()
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Header for live console output
    if rank == 0:
        print(f"\n{'op':>15} {'W':>2} {'shape':>20} {'dim':>3} {'dt':>5} "
              f"{'lat_us':>9} {'algobw':>8} {'busbw':>8} {'tag':>22}", flush=True)

    for i, (op, spec, dim, dt_name) in enumerate(points):
        dtype = _dtype_obj(dt_name)
        eb = _dtype_bytes(dt_name)
        norm_dim = dim if dim >= 0 else dim + len(spec.shape)

        try:
            fn, per_rank_bytes, msg_bytes_csv = _make_fn(
                op, spec.shape, dim, dtype, device, world_size
            )
        except Exception as e:
            if rank == 0:
                print(f"# skip {op} {spec.tag} dim={dim}: {e}", flush=True)
            continue

        try:
            if world_size > 1:
                dist.barrier()
            lat_us, lat_p10, lat_p90 = _bench_cuda(
                fn, warmup=args.warmup, iters=args.iters
            )
            if world_size > 1:
                # Reduce timing to a max across ranks (slowest rank = effective latency).
                t = torch.tensor([lat_us, lat_p10, lat_p90], device=device, dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                lat_us, lat_p10, lat_p90 = t[0].item(), t[1].item(), t[2].item()
        except RuntimeError as e:
            if rank == 0:
                print(f"# OOM/runtime err {op} {spec.tag} dim={dim}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue

        wire_bytes_per_rank = int(_wire_factor(op, world_size) * per_rank_bytes)
        algobw = per_rank_bytes / (lat_us * 1e3) if lat_us > 0 else 0.0  # GB/s
        busbw = wire_bytes_per_rank / (lat_us * 1e3) if lat_us > 0 else 0.0  # GB/s

        # split into (m, n) for the CSV: m = product of leading dims, n = last.
        n_last = spec.shape[-1]
        m_lead = prod(spec.shape) // n_last

        if rank == 0:
            shape_str = "x".join(str(s) for s in spec.shape)
            writer.writerow([
                args.nchannels, world_size, msg_bytes_csv, op,
                f"{algobw:.4f}", f"{lat_us:.2f}", f"{busbw:.4f}",
                "torch", hostname, timestamp,
                dt_name, dim, shape_str, len(spec.shape),
                m_lead, n_last,
                spec.elements, per_rank_bytes, wire_bytes_per_rank,
                f"{lat_p10:.2f}", f"{lat_p90:.2f}",
                args.iters, args.warmup, spec.tag,
            ])
            csv_fp.flush()
            print(f"{op:>15} {world_size:>2} {shape_str:>20} {dim:>3} {dt_name:>5} "
                  f"{lat_us:>9.1f} {algobw:>8.2f} {busbw:>8.2f} {spec.tag:>22}",
                  flush=True)

    if rank == 0 and csv_fp is not None:
        csv_fp.close()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
