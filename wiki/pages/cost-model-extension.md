# Cost Model Extension

How to extend [[origami]]'s analytical GEMM cost model to predict the cost of concurrent and pipelined GEMM+communication patterns.

## Current State

Origami predicts single-GPU GEMM latency with 94.7% accuracy vs. exhaustive autotuning. It models:
- Compute latency (MFMA instruction count × latency)
- Memory latency (3-level cache hierarchy: L2, MALL, DRAM)
- Wave quantization (tail occupancy)
- Epilogue cost (stores, reductions)
- Stream-K work distribution

It does **not** model: communication, concurrent kernels, CU partitioning effects, or bandwidth contention.

## Extension Tiers

The extension follows [[iris]]'s 5-pattern taxonomy, from simplest to most complex:

### Tier 0: Bulk-Synchronous (Baseline)
```
T_total = T_gemm + T_comm
T_gemm = origami.compute_total_latency(problem, hardware, config)
T_comm = α + data_size / β_link
```

No interference, no extension needed beyond a communication latency model.

### Tier 1: Concurrent Separate Kernels (CU-Partitioned)
```
T_total = max(T_gemm(N_CU_gemm), T_comm(N_CU_comm))
```

Where:
- `N_CU_gemm = N_CU_total - N_CU_comm`
- `T_gemm(N_CU_gemm)` = Origami with modified `hardware_t.N_CU`
- `T_comm(N_CU_comm)` = communication model (see below)

**What changes in Origami:**
1. `N_CU` → `N_CU_gemm` (fewer waves needed → but more waves per tile)
2. `mem_bw_per_wg_coefficients` → evaluate at reduced CU count (changes effective BW)
3. L2 cache model → fewer CUs per XCD means less cache pressure (may improve hit rates)
4. HBM bandwidth → shared between GEMM and communication

**Optimization**: sweep CU_comm from 8 to 64 (in multiples of 8), select the split minimizing `T_total`.

### Tier 2: DMA-Offloaded Communication
```
T_total = max(T_gemm(N_CU_total), T_comm_dma)
```

Where:
- `T_gemm` uses full CU count (DMA consumes zero CUs)
- No L2 interference (DMA bypasses L2)
- HBM bandwidth contention remains

**What changes in Origami:**
- `mem3_perf_ratio` (DRAM ratio) reduced by DMA bandwidth consumption
- Or: `BW_gemm_effective = BW_total - BW_dma_active`

For memory-bound GEMMs, optionally reduce CU count by 8 (one per XCD) to improve cache behavior.

### Tier 3: Sequential Fused (Tile-Level Pipeline)
```
T_total = T_gemm_tile × num_tiles + T_comm_tile × (pipeline_depth − 1)
```

Approximation: each tile's communication overlaps with next tile's compute. The pipeline reaches steady state after `pipeline_depth` tiles.

**What changes in Origami:**
- Per-tile latency (`compute_tile_latency`) gains a communication epilogue
- `L_epilogue` extended with: `L_comm_tile = α_tile + tile_data / β_link`
- Effective tile latency: `max(L_tile_compute, L_comm_tile)` in steady state

### Tier 4: Workgroup-Specialized Fused Kernel
```
T_total = max(T_gemm(N_CU_gemm), T_comm_fused(N_CU_comm))
```

Similar to Tier 1, but within a single kernel. Additional factors:
- Extra register pressure from communication → reduced occupancy → `config_t.occupancy` decreases
- Atomic synchronization overhead between compute and comm WGs
- Tile-level lock contention (acquire/release semantics)

## Communication Latency Model

First-principles model built from architectural parameters (link bandwidth, topology, protocol overhead), analogous to how Origami models GEMM from cache hierarchy and instruction latencies.

### For CU-Based Communication (RCCL-style)

```
T_comm = T_startup + n_steps × (T_step_latency + chunk_size / BW_link_effective)
```

Where:
- `T_startup` = kernel launch + synchronization overhead (derived from dispatch pipeline parameters)
- `n_steps` = algorithm-dependent: 2(N-1) for ring AllReduce, (N-1) for ring AllGather/ReduceScatter
- `BW_link_effective` = min(per-link wire BW, per-GPU egress ceiling / active_links) — from [[mi300x-architecture]] measured link parameters
- `chunk_size` = message_size / (n_steps × n_channels)

The model must account for the per-GPU egress ceiling (49.2 GB/s on MI300X) which caps aggregate outbound regardless of fan-out.

### For DMA-Based Communication

```
T_comm_dma = T_cpu_dispatch + data_size / BW_sdma_effective
```

Where `BW_sdma_effective` is limited by:
- Per-link SDMA write asymmetry (23.6 GB/s write vs 49.5 GB/s read)
- HBM bandwidth contention with concurrent GEMM (SDMA bypasses L2 but shares HBM)
- Per-GCD stream saturation (hard cap at 10 streams/GCD)

### For Per-Tile Communication (Fused Patterns)

```
T_comm_tile = T_sync + tile_output_bytes / BW_link_effective
tile_output_bytes = MT_M × MT_N × sizeof(dtype)
```

`T_sync` includes cross-XCD atomic signaling cost — modeled from atomic latency and coherence scope (see [[mi300x-architecture]] coherence contracts).

### Per-Tile Communication (for Fused Patterns)

```
T_comm_tile = α_tile + tile_output_bytes / β_link
tile_output_bytes = MT_M × MT_N × sizeof(dtype)
```

## Key Parameters to Add to hardware_t

```cpp
// Inter-GPU communication
float if_link_bw;           // Infinity Fabric per-link bandwidth (64 GB/s on MI300X)
int num_if_links;            // number of peer links (7 for 8-GPU fully connected)
int num_sdma_engines;        // DMA engines (14 on MI300X)

// Communication kernel resource profile
int comm_cu_saturation;      // CUs where comm saturates (32 for AG, 64 for A2A)
float comm_bw_fraction;      // fraction of HBM BW consumed by communication

// CU partitioning
int min_cu_allocation;       // minimum CUs per partition (8 on MI300X)
```

## Key Parameters to Add to problem_t

```cpp
// Communication overlay
collective_type_t collective; // AllGather, ReduceScatter, AllReduce, AllToAll
size_t comm_data_size;        // bytes to communicate
int num_gpus;                 // GPU count
overlap_pattern_t pattern;    // bulk_sync, concurrent, dma_offload, fused_sequential, fused_wg_specialized
```

## Validation Strategy

1. **Tier 1 validation**: Compare predicted vs. measured T_overlap for concurrent GEMM + RCCL collective across the C3 paper's 30 scenarios
2. **Tier 2 validation**: Compare against ConCCL results (DMA offload)
3. **Tier 3/4 validation**: Compare against [[triton-distributed]] and [[iris]] fused kernel results

The goal is analytical prediction from architectural parameters alone — no profiling-based lookup tables or regression models.

## Implementation Path

1. Add `comm_hardware_t` struct to Origami with inter-GPU topology
2. Implement `compute_comm_latency()` for each collective type
3. Modify `hardware_t.N_CU` to be a parameter, not a constant
4. Add `compute_overlap_latency(problem, hardware, config, comm_config)` that:
   - Sweeps CU partitions (Tier 1)
   - Evaluates DMA option (Tier 2)
   - Returns `(T_overlap, best_cu_split, best_strategy)`
5. Add tile-level pipeline modeling (Tier 3) as optional refinement
6. Validate against measured data from [[test-vehicles]]
