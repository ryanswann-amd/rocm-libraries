# Stream-K

Work-centric parallel decomposition for GEMM that partitions aggregate inner-loop iterations evenly across CUs, eliminating wave quantization waste.

## Paper

**Stream-K: Work-Centric Parallel Decomposition for Dense Matrix-Matrix Multiply on the GPU**
Osama, Merrill (2023)

## The Wave Quantization Problem

Data-parallel GEMM maps one output tile per workgroup. When tile count doesn't divide evenly by CU count, the last wave is under-occupied:

```
304 CUs, 305 tiles → 2 waves
  Wave 1: 304 tiles (100% utilization)
  Wave 2: 1 tile (0.3% utilization)
```

This can cause up to **2× slowdown** for pathological shapes.

## How Stream-K Works

Instead of mapping tiles to CUs, Stream-K partitions the **aggregate K-dimension iterations** evenly:

```
total_work = num_tiles × K_iters_per_tile
work_per_CU = total_work / N_CU
```

Each CU processes a continuous stream of work, potentially spanning multiple output tiles. When a CU's work range crosses a tile boundary, it:
1. Writes partial results for the finishing tile
2. Starts accumulation for the next tile

Partial results from tiles split across CUs are combined via:
- Atomic adds
- Explicit fixup kernels
- Tree reduction

## Why Stream-K Enables Communication Overlap

### 1. Uniform CU Utilization
No "tail wave" with idle CUs. In data-parallel GEMM, the tail wave leaves CUs idle that could otherwise run communication kernels. Stream-K eliminates this waste.

### 2. Persistent Kernel Compatibility
Stream-K naturally maps to persistent kernels where workgroups stay alive and pull work dynamically — exactly the pattern needed for [[fused-kernels]].

### 3. Flexible CU Partitioning
When CUs are partitioned between GEMM and communication ([[cu-partitioning]]), Stream-K ensures the GEMM portion maintains high utilization despite reduced CU count:

```
Data-parallel (272 CUs, 300 tiles): 2 waves, tail = 28 CUs idle (90.7%)
Stream-K (272 CUs, 300 tiles): all CUs equally loaded (100%)
```

### 4. Predictable Tile Completion Timing
For producer-consumer patterns, Stream-K provides more uniform tile completion times than data-parallel scheduling, making it easier to pipeline communication with computation.

## tritonBLAS Work-Stealing Persistent Kernel

tritonBLAS (March 2026) added a work-stealing persistent kernel that guarantees **linear performance scaling with active CU count**. This avoids the nonlinear decay that data-parallel GEMM suffers when CUs are partitioned for concurrent communication.

This is directly relevant to [[cost-model-extension]] — the cost model can assume:
```
T_gemm(N_CU) ≈ T_gemm(304) × (304 / N_CU)  # linear scaling
```

Rather than the nonlinear behavior observed with data-parallel dispatch.

## Stream-K++ (arXiv:2408.11417)

Extends Stream-K with 7 scheduling policies and Bloom-filter-based configuration selection. Up to 43% improvement on MI250X.

## Integration in hipBLASLt

[[hipblaslt]] enables Stream-K via `TENSILE_SOLUTION_SELECTION_METHOD=2`. On MI350, it's the only option.

Grid selection algorithms in [[origami]]:
1. `number_of_cus` — use all CUs
2. `min_resources` — `min(N_CU, num_tiles)`
3. `energy_aware` — scale down for power savings
4. `reduction_cost_aware` — analytical cost with α/β/c/d coefficients
5. `data_parallel` — no splitting
6. `analytical` — sweep split factors via `compute_total_latency()`
7. `k_split_aware` (default) — distributes tiles evenly, tries fractional splits (1/2, 1/3, 1/4, 1/5, 1/8)

## Relevance to This Project

Stream-K is critical infrastructure for communication overlap because:
1. It makes `TENSILE_STREAMK_MAX_CUS` work well (linear scaling, no tail waste)
2. It enables persistent kernel patterns for [[fused-kernels]]
3. The `k_split_aware` algorithm in [[origami]] already handles CU-aware grid selection — extending it to account for CUs reserved for communication is natural
4. Predictable tile completion feeds producer-consumer cost modeling
