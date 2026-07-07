# DMA Offload

Using GPU DMA (SDMA) engines for collective communication to eliminate CU and cache interference with concurrent GEMM execution.

## Mechanism

MI300X has **14 SDMA copy engines** on the I/O dies, positioned beyond the L2 caches:

```
CUs ←→ L1 ←→ L2 (per XCD) ←→ Infinity Cache (MALL) ←→ HBM
                                        ↑
                              SDMA Engines (on IODs)
```

DMA engines access Infinity Cache and HBM directly, bypassing L1/L2 entirely.

## API

```c
hsa_amd_memory_async_copy_on_engine(dst, dst_agent, src, src_agent, size, ...)
```

CPU places command packets in a DMA queue → SDMA engine fetches → decodes → issues reads/writes directly to/from HBM.

## ConCCL (C3 Paper)

The C3 paper introduces **ConCCL** (Concurrent Communication Collectives):
- **Algorithm**: Direct/flat — each GPU writes its buffer to all N−1 peers in a single step
- **Parallelism**: Each point-to-point transfer dispatched to a separate SDMA engine (up to 14)
- **Orchestration**: CPU-driven (unlike RCCL's GPU kernel orchestration)

### Limitations
- **No arithmetic**: SDMA engines can only move data, cannot reduce — AllReduce must be decomposed into AllGather + local reduction, or only the gather step is offloaded
- **Launch overhead**: CPU-driven DMA dispatch is slow for small transfers — **4× slower than RCCL for < 32 MB**
- **Bandwidth contention**: DMA still competes for Infinity Cache and HBM bandwidth

## Interference Profile

| Resource | CU-Based Comm | DMA-Based Comm |
|----------|---------------|----------------|
| CUs | Consumes 32–64 CUs | **Zero** |
| L2 cache | Pollutes per-XCD L2 | **Zero** (bypasses L2) |
| Infinity Cache | Contended | **Contended** (DMA accesses MALL) |
| HBM bandwidth | Contended | **Contended** |

Key insight: DMA eliminates CU and L2 interference but HBM bandwidth sharing remains.

## Performance (C3 Paper)

| Config | % of Ideal Speedup | Notes |
|--------|-------------------|-------|
| CU-based baseline | 21% | GEMM + RCCL concurrent |
| CU-based + schedule priority | 42% | Communication launched first |
| ConCCL (DMA) | 66% | All CUs free for GEMM |
| ConCCL + CU reduction | 72% | Remove 8 CUs from memory-bound GEMM |

## DMA Collectives Paper (arXiv:2511.06605)

Extends the DMA approach further:
- Evaluated on MI300X 8-GPU Infinity Platform
- Leverages existing SDMA engines — no hardware changes required
- Synchronization with producer GEMM kernels tested using rocBLAS GEMMs
- Addresses the "who signals completion?" problem for producer-consumer patterns

## Implications for [[cost-model-extension]]

When modeling DMA-based communication:
1. **CU cost = 0** — full N_CU available for GEMM
2. **L2 interference = 0** — no cache pollution from communication
3. **HBM BW contention remains** — must model bandwidth sharing
4. **Communication latency** = `launch_overhead + data_size / effective_dma_bandwidth`
5. **Effective DMA BW** < peak HBM BW due to Infinity Cache sharing with GEMM

The Origami memory model's `mem_bw_per_wg_coefficients` may need a DMA bandwidth term:
```
BW_gemm_effective = BW_total - BW_dma_active
```

Or model as reduced `mem3_perf_ratio` (DRAM bandwidth ratio) when DMA is active.
