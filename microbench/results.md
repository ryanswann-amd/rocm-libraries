# Microbenchmark Results — 2026-05-26, d19u19 (MI300X)

## P2P Bandwidth (torch.copy_, SDMA path)

64 MB GPU 0 → GPU 1:

| WGs | Agg BW (GB/s) | Per-WG BW (GB/s) | Time (µs) |
|-----|--------------|------------------|-----------|
| 1 | 37.12 | 37.12 | 1808 |
| 8 | 37.10 | 4.64 | 1809 |
| 32 | 37.11 | 1.16 | 1808 |
| 256 | 37.12 | 0.14 | 1808 |

WG count has zero effect — this is the copy engine, not CU stores.

## P2P Fanout (torch.copy_, SDMA path)

64 MB per peer from GPU 0:

| Peers | Per-Link BW | Agg Egress BW | Time (µs) |
|-------|-------------|--------------|-----------|
| 1 | 37.13 | 37.13 | 1807 |
| 2 | 19.45 | 38.90 | 3450 |
| 4 | 9.65 | 38.60 | 6954 |
| 7 | 5.57 | 39.01 | 12043 |

Aggregate egress caps at ~39 GB/s regardless of fan-out. This is the SDMA per-GPU cap.
RCCL uses kernel-initiated stores (CU path), not SDMA — different cap.

## RCCL Zero-Payload Overhead

| Collective | W=2 (µs) | W=8 (µs) |
|-----------|---------|---------|
| AllReduce | 484 | 662 |
| AllGather | 625 | 1125 |

## RCCL Bandwidth-Bound (default NCH, auto)

AllReduce W=8 at large messages:

| msg_bytes | latency (µs) | raw BW (GB/s) | busbw (GB/s) |
|-----------|-------------|--------------|-------------|
| 4,194,304 | 1,379 | 3.04 | 5.32 |
| 16,777,216 | 2,604 | 6.44 | 11.27 |
| 67,108,864 | 5,584 | 12.02 | 21.03 |
| 268,435,456 | 17,024 | 15.77 | 27.60 |

Default NCH (auto) achieves much lower busbw than the master sweep at NCH=32 (239 GB/s).

## What's Missing

Need a custom Triton kernel that does `tl.store` / `iris.store` to remote GPU memory
from CU-launched workgroups (not the copy engine). This measures the kernel-initiated
store path that RCCL actually uses, which may have a different per-GPU aggregate cap.
