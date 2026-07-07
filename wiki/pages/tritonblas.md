# tritonBLAS

Analytical GEMM kernel parameter selection framework implemented in Triton, described in arXiv:2512.04226. The cost model underlying [[origami]].

## Paper

**tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection**
Swann, Osama, Guo, Nelson, Zhang, Brown, Ong, Yazdani, Siddens, Dasika, Underwood (AMD, 2025)
[arXiv:2512.04226](https://arxiv.org/abs/2512.04226)

## Core Idea

Replace empirical autotuning (O(T×M×N×K) per problem) with an analytical model (O(T) where T = candidate tile count) that uses architectural parameters to predict GEMM kernel performance. Selection time: 50–80 µs vs. 12–1384 seconds for autotuning.

## Cost Model Formulation

### Five-Level Tiling Hierarchy
1. **Instruction tile** — MFMA/WMMA atom (e.g., 32×32×16)
2. **Wavefront/register tile** — SIMD register-level
3. **Workgroup/shared memory tile** — MT_M × MT_N × MT_K
4. **Cache tile** — group of workgroup tiles within L2 scope (per XCD)
5. **Global problem** — parallelized across all CUs, serialized in K

Output-stationary dataflow: M×N parallelized spatially, K accumulated sequentially.

### Compute Latency (Algorithm 3)
```
N_MI = ⌈MT_M/MI_M⌉ × ⌈MT_N/MI_N⌉ × ⌈MT_K/MI_K⌉
L_MT = L_MI × N_MI
```

### Memory Latency (Algorithm 7)
```
L_CU_lat = Ld_CU / R_L1
T = Ld_CU × C                    (total loads, C = active CUs)
L1 = T / R1
T2 = (1 − H1) × T               (L2 miss traffic)
L2 = T2 / R2
T_M = (1 − H2) × T2             (HBM miss traffic)
L_MEM = T_M / R_MEM + L_lat
L_mem = max(L_CU_lat, L1, L2, L_MEM)
```

### Cache Hit Rate (Algorithm 5)
```
U = (mt·MT + nt·NT) · KT         (uncached reads)
R = (mt·nt)(MT + NT) · KT        (total reads)
h = 1 − U/R                      (hit rate)
```

Cache tile shape selected via factorizations of N_CU_Cache (Algorithm 6).

### Tile Latency (Algorithm 8)
```
L_prologue = L_mem
L_epilogue = (a · mt · nt) / H.R_mem
L_loopiter = max(L_compute, L_mem)
I = ⌈K/kt⌉ − 1
L_tile = L_prologue + L_epilogue + (L_loopiter × I)
```

### Total GEMM Latency (Algorithm 9)
```
N_waves = ⌈(nm × nn) / H.N_CU⌉
L_total = N_waves × L_tile
```

## Architectural Parameters Required

| Parameter | Description | Example (MI300X) |
|-----------|-------------|-----------------|
| R_L1 | L1/shared memory bandwidth | — |
| R1 | L2 bandwidth | — |
| R2 | LLC bandwidth | — |
| R_MEM | HBM bandwidth | 5.3 TB/s |
| L_lat | Memory latency constant | — |
| MI_M, MI_N, MI_K | Matrix instruction shape | 32, 32, 16 |
| L_MI | Instruction latency (cycles) | 32 |
| N_CU | Compute units | 304 |
| N_CU_Cache | CUs per cache scope (XCD) | 38 |

## Wave Quantization

The "tail occupancy" problem: only the last wave is under-occupied.
```
active_cu = T_out % N_CU
ω = ⌈T_out / N_CU⌉
```

[[stream-k]] directly addresses this by distributing inner-loop iterations evenly across CUs.

## Key Results

| Metric | Value |
|--------|-------|
| Selection efficiency vs. exhaustive autotune | 94.7% |
| Test problems | 150,000 random (dims < 8193, multiples of 128) |
| vs. torch.matmul() average | +3% faster |
| Selection time (all sizes) | 50–80 µs |
| Autotune time (75 configs, 512³) | ~12 seconds |
| Autotune time (75 configs, 16384³) | ~1384 seconds |

## Bottleneck Categories

The model classifies each GEMM into:
- **Load/store issue rate bound**
- **Shared memory bandwidth bound**
- **Cache bandwidth bound**
- **Under-occupied compute bound**
- **Max parallelism compute bound** (ideal)

## Relationship to This Project

tritonBLAS/Origami models single-GPU GEMM execution only. Extending it to communication requires modeling:
1. CU reduction effects (fewer CUs → compute-bound GEMMs slow down; memory-bound may speed up)
2. Cache interference from concurrent communication kernels
3. HBM bandwidth contention between GEMM and communication
4. Tile-level completion timing for producer-consumer patterns

See [[cost-model-extension]] and [[cu-partitioning]].
