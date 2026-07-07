# CU Partitioning

Strategies for dividing compute units between concurrent GEMM and communication kernels on [[mi300x-architecture]].

## Hardware Support

MI300X provides **stream-CU reservation**: specific CUs can be exclusively assigned to a work stream. This prevents the GPU scheduler from unfairly allocating CUs.

**Constraints:**
- Minimum allocation: **8 CUs** (1 per XCD on 8-XCD MI300X)
- Total available: 304 CUs (38 per XCD)
- Typical sweep: powers of two (8, 16, 32, 64, 128, ...)

## hipBLASLt Controls

hipBLASLt exposes CU limiting for concurrent execution via environment variables:
```bash
export TENSILE_STREAMK_MAX_CUS=272     # limit GEMM to 272 CUs (leave 32 for comm)
export TENSILE_STREAMK_FIXED_GRID=64   # limit GEMM to 64 workgroups
```

These prevent GEMM from monopolizing GPU resources during concurrent execution.

## Partitioning Effects on GEMM

From the C3 paper (arXiv:2412.14335):

### Compute-Bound GEMMs
- Monotonically degrade with fewer CUs
- 32 CUs lost → 17% slower
- 64 CUs lost → 27% slower
- More CUs lost → proportionally worse
- Mechanism: fewer parallel tiles per wave → more waves → higher latency

### Memory-Bound GEMMs
- **Resilient** to CU loss — can even speed up
- 8 CUs lost → up to 10% faster
- Mechanism: fewer concurrent threads → less cache thrashing → higher L2 hit rate
- Sweet spot on MI300X: remove 8 CUs (1 per XCD)

### Impact on [[origami]] Model
Origami computes `N_waves = ⌈num_tiles / N_CU⌉`. With CU partitioning:
- `N_CU_effective = N_CU - CUs_reserved_for_comm`
- More waves needed → higher total latency
- But cache behavior changes too — `mem_bw_per_wg_coefficients` polynomial changes with active CU count
- L2 hit rates may improve with fewer concurrent requestors

## Partitioning Effects on Communication

| Collective | CU Saturation | Recommendation |
|------------|--------------|----------------|
| AllGather | 32 CUs | Reserve ≥32 CUs |
| AllToAll | 64 CUs | Reserve ≥64 CUs |
| ReduceScatter | ~32 CUs | Reserve ≥32 CUs |

Beyond the saturation point, additional CUs provide no benefit for communication.

## Optimal Partition Strategy

### Schedule Prioritization (c3_sp)
Launch communication kernel **first**, then GEMM. The GPU scheduler gives early-launched kernels priority CU access.

Heuristic: schedule kernels in ascending order of workgroup count (communication has far fewer workgroups than GEMM).

Result: 42% of ideal speedup (vs. 21% baseline).

### Explicit Reservation (c3_rp)
Reserve CUs explicitly for communication stream. Sweep to find optimal split.

Result: 41% of ideal speedup — matches schedule prioritization.

### Combined
No additional benefit from combining sp + rp (42% either way).

### DMA + Reduced GEMM CUs
When communication uses [[dma-offload]] (zero CUs), deliberately reducing GEMM CUs by 8 (one per XCD) can improve memory-bound GEMM performance.

Result: 72% of ideal speedup.

## Modeling CU Partitioning in [[origami]]

The key extension: `hardware_t.N_CU` becomes a variable, not a constant.

```
For a given overlap scenario:
  for each candidate CU_split:
    N_CU_gemm = N_CU_total - CU_comm
    N_CU_comm = CU_comm

    T_gemm = origami.compute_total_latency(problem, hardware(N_CU=N_CU_gemm), config)
    T_comm = comm_model.compute_latency(collective, data_size, N_CU_comm)

    T_overlap = max(T_gemm, T_comm)
  select CU_split minimizing T_overlap
```

Origami already has the infrastructure for this — `mem_bw_per_wg_coefficients` and cache models are parameterized by active CU count. The missing piece is the communication latency model.
