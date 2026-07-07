# Compute-Communication Overlap

Techniques and analysis for running GEMM and communication concurrently on AMD GPUs, and the interference effects that must be modeled.

## Why Overlap?

In bulk-synchronous execution, communication sits on the critical path:
```
T_serial = T_gemm + T_comm
```

With perfect overlap:
```
T_overlap = max(T_gemm, T_comm)
```

Ideal speedup = `(T_gemm + T_comm) / max(T_gemm, T_comm)`, ranging from 1.0× (one dominates completely) to 2.0× (equal durations).

In practice, interference reduces this. The C3 paper (arXiv:2412.14335) reports:
- Baseline concurrent execution: **21% of ideal speedup** (1.13×)
- With schedule prioritization: **42%** of ideal
- With DMA offload + CU tuning: **72%** of ideal (up to 1.67×)

## Three Overlap Approaches

### 1. Stream-Based Concurrent Kernels
GEMM on stream A, RCCL collective on stream B. GPU scheduler distributes CUs.

**Pros**: Simple, no code changes
**Cons**: Uncontrolled CU allocation → GEMM can starve communication; baseline achieves only 21% of ideal

### 2. CU-Partitioned Concurrent Kernels
Explicitly reserve CUs for each kernel using MI300X stream-CU reservation.

**Pros**: Predictable resource allocation
**Cons**: Compute-bound GEMMs degrade with fewer CUs; requires tuning partition point
See [[cu-partitioning]]

### 3. DMA-Offloaded Communication
Use SDMA engines for data movement, freeing all CUs for GEMM.

**Pros**: Zero CU interference, zero L2 interference
**Cons**: No arithmetic (can't do AllReduce), CPU-driven launch overhead, HBM BW still contended
See [[dma-offload]]

### 4. Fused Kernels
Single kernel with workgroups that alternate between compute and communication roles.

**Pros**: Fine-grained tile-level overlap, no kernel launch overhead
**Cons**: Complex programming model, lower occupancy from extra register pressure
See [[fused-kernels]]

## Interference Analysis (C3 Paper)

### GEMM Slowdown Under CU Reduction

| GEMM Type | 32 CUs Lost | 64 CUs Lost | Mechanism |
|-----------|-------------|-------------|-----------|
| Compute-bound | 17% slower | 27% slower | Fewer CUs → fewer parallel tiles |
| Memory-bound | 0–5% *faster* | 0–10% *faster* | Fewer threads → better cache behavior |

### Communication Slowdown

| Collective | CU Saturation | Beyond Saturation |
|------------|--------------|-------------------|
| AllGather | 32 CUs | No benefit from more CUs |
| AllToAll | 64 CUs | No benefit from more CUs |
| AllReduce | Varies | Depends on decomposition |

### Contended Resources

| Resource | Contention Pattern |
|----------|-------------------|
| CUs | Primary bottleneck for compute-bound GEMMs |
| L2 Cache (4 MB/XCD) | Both kernels share per-XCD L2; thrashing possible |
| Infinity Cache (256 MB) | All traffic passes through; moderate contention |
| HBM (5.3 TB/s) | Memory-bound GEMMs consume most bandwidth |

### Taxonomy

Three C3 scenario types based on isolated execution times:
- **G-long**: T_gemm_isolated > 1.15 × T_comm_isolated
- **C-long**: T_comm_isolated > 1.15 × T_gemm_isolated
- **GC-equal**: Within 15% of each other

GC-equal scenarios benefit most from overlap (up to 2× ideal speedup).

## First-Principles Cost Modeling Approach

The C3 paper used empirical lookup tables for CU-slowdown factors. Our approach replaces these with analytical prediction from architectural parameters, extending [[origami]]'s existing model:

For each CU split point k:
- `T_gemm(k)` = Origami `compute_total_latency()` with `hardware_t.N_CU = N_total − k`
- `T_comm(k)` = analytical communication model from link bandwidth, message size, and collective algorithm

Select k minimizing `max(T_gemm(k), T_comm(k))`.

Origami already parameterizes its cache and bandwidth models by active CU count (`mem_bw_per_wg_coefficients`, wave count, L2 hit rates). The extension is to also analytically model communication latency from the same kind of architectural parameters — link bandwidth, protocol overhead, arbitration effects — rather than profiling-based lookup tables.

## What Needs to Be Modeled

For [[origami]] to support overlap, the cost model must predict:
1. **GEMM latency under reduced CUs** — Origami already models wave count; needs CU reduction effects on cache
2. **Communication latency** — new model component (α + data/β, with CU and BW interference)
3. **Cache interference** — concurrent workloads polluting each other's L2 working sets
4. **HBM bandwidth sharing** — partitioning 5.3 TB/s between GEMM and communication
5. **Tile completion timing** — for producer-consumer patterns, when does each tile finish?
