# Test Vehicles

Concrete implementations that can serve as validation targets for [[cost-model-extension]].

## 1. tritonBLAS + RCCL Concurrent (Tier 1)

**What**: Run tritonBLAS GEMM on one stream, RCCL collective on another, measure total latency.

**Why**: Simplest overlap pattern. Tests CU partitioning model.

**Controls**:
- `TENSILE_STREAMK_MAX_CUS` — limit GEMM CU count
- `GPU_MAX_HW_QUEUES=2` — limit HW queues
- `TORCH_NCCL_HIGH_PRIORITY=1` — priority scheduling

**Validation**: Predict T_overlap = max(T_gemm(N_CU_gemm), T_comm(N_CU_comm)) for each CU split. Compare against measured.

**Shapes**: Use C3 paper's 30 scenarios (compute-bound and memory-bound GEMMs × AllGather and AllToAll collectives at various sizes).

## 2. Triton-Distributed GEMM+ReduceScatter (Tier 3/4)

**Repo**: [ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)

**What**: Fused GEMM+ReduceScatter with tile-level overlap on MI300X.

**Why**: Tests tile-level pipeline model. Known SM allocation for communication stages.

**Benchmark**:
```bash
bash ./third_party/distributed/launch_amd.sh \
  ./third_party/distributed/distributed/test/amd/test_gemm_rs_intra_node.py \
  ${m} ${n} ${k} --warmup 5 --iters 20
```

**Shapes tested**: `8192×4096×12288`, `8192×8192×28672`, etc.

**Validation**: Predict T_fused from per-tile compute latency (Origami) + per-tile comm latency + pipeline overlap. Compare against measured.

## 3. Iris Overlap Patterns (Tiers 1–4)

**Repo**: [ROCm/iris](https://github.com/ROCm/iris)

**What**: 5 distinct overlap patterns in a single framework, all on AMD GPUs.

**Why**: Each pattern isolates a different overlap mechanism. The 5-pattern taxonomy maps directly to the cost model tiers in [[cost-model-extension]]:

| Iris Pattern | Cost Model Tier | Validation Target |
|-------------|----------------|-------------------|
| Bulk-synchronous | Tier 0 | T_gemm + T_comm |
| Producer-consumer (streams) | Tier 1 | max(T_gemm(CU_g), T_comm(CU_c)) |
| Sequential fused | Tier 3 | Tile pipeline model |
| WG specialization | Tier 4 | CU-partitioned fused model |
| Wave specialization | Future | — |

**Best result**: `8192×4608×36864` at 1.8× over PyTorch+RCCL.

## 4. C3 Paper Benchmarks (Tiers 1–2)

**Paper**: arXiv:2412.14335

**What**: Systematic characterization of concurrent GEMM + collective on MI300X with 30 scenarios.

**Why**: Published ground truth data for CU partitioning effects and DMA offload performance.

**Key data**:
- GEMM slowdown vs. CUs lost (compute-bound and memory-bound)
- Communication slowdown vs. CUs allocated
- Optimal CU split for each scenario
- DMA (ConCCL) results

**Validation**: Compare Origami's analytical prediction against measured optimal CU splits across the full scenario matrix.

## 5. Yotta Labs MI300X Kernels (Tiers 3–4)

**Source**: [AMD Developer Challenge 2025](https://www.yottalabs.ai/post/optimizing-distributed-inference-kernels-for-amd-developer-challenge-2025)

**What**: GEMM+ReduceScatter and AllGather+GEMM with XCD-aware scheduling.

**Why**: Tests XCD-aware tile mapping — each XCD handles 1/8 of output rows, utilizing all 7 Infinity Fabric links simultaneously.

**Key details**:
- Thread blocks remapped across 8 XCDs for topology-aware scheduling
- Symmetric heap with `.cg` cache modifier for remote writes
- Two-phase: GEMM+scatter epilogue, then reduce kernel after global barrier
- Launch overhead reduced from ~120 µs to ~40 µs
- WG specialization: 56 communication CTAs + remaining compute CTAs

## 6. tritonBLAS Work-Stealing Persistent Kernel (CU Scaling Validation)

**What**: tritonBLAS's March 2026 persistent kernel with work-stealing.

**Why**: Guarantees **linear performance scaling with active CU count**. Validates the critical assumption that `T_gemm(N_CU) ≈ T_gemm(304) × (304 / N_CU)`.

**Validation**: Run the persistent kernel with varying CU counts (via `TENSILE_STREAMK_MAX_CUS`), measure actual vs. predicted linear scaling. This validates Origami's CU reduction modeling.

## Recommended Validation Order

1. **C3 paper benchmarks** (published data, no setup needed, Tier 1–2)
2. **tritonBLAS persistent kernel CU scaling** (single-GPU, validates core assumption)
3. **tritonBLAS + RCCL concurrent** (simple multi-GPU, Tier 1)
4. **Iris patterns** (comprehensive, all tiers)
5. **Triton-Distributed** (fused, Tier 3–4)
6. **Yotta Labs** (XCD-aware, advanced Tier 4)
