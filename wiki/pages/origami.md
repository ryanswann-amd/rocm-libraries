# Origami

Analytical, deterministic GEMM kernel configuration selector that uses GPU architectural parameters to predict near-optimal kernel configurations without runtime autotuning.

## Location

`ROCm/rocm-libraries` → `shared/origami/` ([GitHub](https://github.com/ROCm/rocm-libraries/tree/develop/shared/origami))

## What It Does

Given a GEMM problem (M, N, K, dtypes, transposes) and a target GPU, Origami evaluates candidate kernel configurations and returns the predicted-fastest one. It replaces exhaustive autotuning with an analytical cost model that runs in 50–80 µs (vs. seconds–hours for autotuning).

Origami achieves 94.7% of exhaustive autotuning performance across 150,000 random GEMM shapes (arXiv:2512.04226).

## Architecture

```
problem_t + hardware_t + config_t
        │
        ▼
    context_t          ← derived values (grid dims, StreamK params, bandwidth limits)
        │
        ▼
  compute_total_latency()
        │
        ├── compute_tile_latency()
        │      ├── compute_latency()      ← N_MI × L_MI
        │      ├── memory_latency()       ← 3-level cache model (L2, MALL, DRAM)
        │      ├── prologue_latency()     ← first tile load
        │      ├── epilogue_latency()     ← ACC→VGPR, stores, edge tiles
        │      └── loop_overhead()
        │
        └── parallel_reduction_latency()  ← split-K reduction kernel
```

## Key Data Structures

### `problem_t`
```
size: dim3_t (M, N, K)
batch: int
a/b_transpose: T or N
a/b/c/d_dtype: data_type_t (23 types including FP8, BF8, FP4, MX formats)
mi_dtype: data_type_t
```

### `hardware_t`
```
arch: architecture_t (gfx90a, gfx942, gfx950, gfx1100, gfx1201, ...)
N_CU: int                          ← total compute units
NUM_XCD: int                       ← chiplet count (8 on MI300X)
lds_capacity: int                  ← Local Data Share per CU (bytes)
L2_capacity: int                   ← L2 cache per XCD
CU_per_L2: int                     ← CUs sharing an L2 partition
compute_clock_ghz: float
parallel_mi_cu: int                ← parallel matrix instructions per CU (4 CDNA, 2 RDNA)
mem1/2/3_perf_ratio: float         ← L2/MALL/DRAM bandwidth ratios
mem_bw_per_wg_coefficients: float[3]  ← quadratic polynomial for BW vs active CUs
```

### `config_t`
```
mt: dim3_t (macro tile M, N, K)
mi: dim3_t (matrix instruction M, N, K)
occupancy: int (wavefronts per CU)
workgroup_mapping: int
reduction_strategy: spinlock | tree | parallel | atomic
grid_selection: number_of_cus | min_resources | energy_aware | ... | k_split_aware
prediction_mode: estimation | simulation
target: tensilelite | triton | composable_kernel | rocroller
```

## Cost Model

### Total Latency
```
L_total = L_timestep × num_timesteps + L_parallel_reduce
```

### Per-Timestep (Tile) Latency
```
L_tile = L_tile_single × num_k_iters
       + w_prologue × L_prologue
       + w_epilogue × L_epilogue
       + w_wg_setup × L_WG_setup
       + w_loop_overhead × num_k_iters
```

### Compute vs Memory Balance
```
L_tile_single = max(L_compute × w_compute, L_mem × w_memory)
              × main_loop_efficiency × effective_tile_penalty + L_cvt
```

### Compute Latency
```
L_compute = N_MI × L_MI
N_MI = ⌈MT_M/MI_M⌉ × ⌈MT_N/MI_N⌉ × ⌈MT_K/MI_K⌉
```

### Memory Latency (3-Level Hierarchy)
```
L_mem = max(L_mem_l2 × w_l2, L_mem_mall × w_mall, L_mem_dram × w_dram)
```

Each level: `remaining_bytes / bandwidth_at_level`

Cache hit rates computed via:
1. **Spatial reuse** — shared data within XCD's tile set (L2) or all CUs (MALL)
2. **K-depth warmup** — sigmoid: `floor + (1-floor) × k²/(k² + depth²)`
3. **L2 residency** — interference-based working set estimation
4. **Pollution penalty** — cross-operand eviction (0.7 factor)
5. **Request amplification** — for batched/split-K/skinny GEMMs

### Epilogue Latency
Models ACC→VGPR transfer, edge bounds checking, global stores (bandwidth-limited), and in-kernel serial reduction for split-K.

## Heuristics System

~40 trainable parameters in `heuristic_params_t`, organized as:
- **Latency weights** (10): `weight_compute`, `weight_memory`, `weight_prologue` (1.5), `weight_epilogue` (2.0), etc.
- **Empirical constants** (~25): `main_memory_load_latency` (200 cycles), cache hit rate parameters, occupancy decay, etc.
- **Architecture-specific overrides** via `heuristics_database_t` with hierarchical key-based lookup

## Selection Algorithm

`select_config(problem, hardware, configs)`:
1. Compute `compute_total_latency()` for each candidate
2. Filter by LDS capacity
3. Sort by predicted latency (stable)
4. Tie-break: arithmetic intensity → dimension alignment → deterministic (largest tiles)

Additional selection steps:
- `select_workgroup_mapping()` — sweeps WGM candidates to minimize L2 load bytes across XCDs
- `select_staggerU()` — K-offset staggering to reduce L2 bank conflicts

## Stream-K Integration

Seven grid selection algorithms, default `k_split_aware`:
- Distributes tiles evenly across CUs
- Tries fractional tile splits (1/2, 1/3, 1/4, 1/5, 1/8)
- Validates workspace limits and cache-line alignment
- See [[stream-k]] for details

## What Origami Does NOT Model

- **Inter-GPU communication** — no RCCL/NCCL collective modeling
- **Concurrent kernel execution** — no resource contention between overlapping kernels
- **Memory bandwidth contention** — assumes sole ownership of HBM/cache
- **CU partitioning effects** — no model for reduced-CU execution
- **Network topology** — no PCIe/xGMI/Infinity Fabric modeling

These are exactly the gaps this project (origami_comms) aims to fill. See [[cost-model-extension]].

## Python API

```python
from origami import OrigamiMatmulSelector
selector = OrigamiMatmulSelector(problem, hardware, configs)
best = selector.select()
```

Full nanobind bindings expose all C++ types and functions. `OrigamiMatmulSelector` wraps for Triton/PyTorch integration.

## Supported GPUs

| Arch | GPUs | XCDs |
|------|------|------|
| gfx90a | MI210, MI250/X | 1 |
| gfx942 | MI300A/X, MI325X | 8 |
| gfx950 | MI350X, MI355X | 8 |
| gfx1100 | RX 7900 series | 1 |
| gfx1201 | RX 9070 | 1 |

## Build

CMake 3.25+, C++17, requires HIP runtime. Produces `roc::origami` library. Python via scikit-build-core + nanobind.
