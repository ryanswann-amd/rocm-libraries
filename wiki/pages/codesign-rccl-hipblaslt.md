# Codesign: RCCL + hipBLASLt

How RCCL and hipBLASLt can be made mutually aware for better concurrent performance, enabled by [[origami]]'s extended cost model.

## Current State: No Coordination

Today, RCCL and hipBLASLt are independent libraries:
- hipBLASLt selects GEMM kernels assuming sole GPU ownership
- RCCL selects collective algorithms assuming sole GPU ownership
- When they run concurrently, both suffer uncontrolled interference
- Result: only 21% of ideal speedup (C3 paper baseline)

## Codesign Opportunities

### 1. CU Budget Negotiation

RCCL could expose its CU requirements, and hipBLASLt could set `TENSILE_STREAMK_MAX_CUS` accordingly.

```
RCCL: "AllGather of 128 MB needs 32 CUs"
hipBLASLt: sets TENSILE_STREAMK_MAX_CUS = 304 - 32 = 272
```

CU requirements are deterministic based on message size and collective type — no runtime negotiation needed, just a lookup.

### 2. Shared Bandwidth Model

Both libraries model bandwidth internally:
- Origami: `mem_bw_per_wg_coefficients` polynomial for HBM bandwidth vs. active CUs
- RCCL: algorithm-dependent step counts and per-link bandwidth consumption

A unified analytical model could predict total HBM consumption from architectural parameters (link bandwidths, active CU count, collective algorithm structure) and throttle when approaching the HBM ceiling.

### 3. Tile-Communication Alignment

hipBLASLt (via [[stream-k]]) controls tile execution order. RCCL/communication could prioritize gathering data for tiles that execute soonest, minimizing GEMM stalls in AllGather→GEMM patterns.

The [[origami]] workgroup mapping (`select_workgroup_mapping`) already optimizes tile order for L2 reuse — extending this to consider communication data arrival order is natural.

### 4. Protocol-Aware GEMM Selection

RCCL protocol selection is deterministic based on message size:

| Message Size | Protocol | CU Impact |
|-------------|----------|-----------|
| Small (< ~1 KB) | LL | Very low CU usage (4B data + 4B flag per element) |
| Medium (~1 KB – ~64 KB) | LL128 | Low CU usage (120B data + 8B flag) |
| Large (> ~64 KB) | SIMPLE | Higher CU usage (bulk transfers) |

When RCCL uses LL protocol, GEMM can use more CUs. When SIMPLE, GEMM should throttle. Since protocol selection is deterministic, this can be pre-computed.

### 5. RCCL WarpSpeed Interaction

RCCL's WarpSpeed optimization (gfx950, ≥ 64 MB messages) reduces CU usage by **50%** via traffic shaping. When WarpSpeed is active, more CUs are available for GEMM — the cost model should account for this.

```
if warpspeed_active:
    comm_cu_need = comm_cu_saturation / 2
```

### 6. Dynamic CU Rebalancing

As communication progresses (more data gathered = fewer remaining transfers), CUs could be released back to GEMM. Requires runtime feedback — more complex but higher theoretical ceiling.

## Integration Architecture

```
┌─────────────────────────────────────┐
│         Overlap Controller          │
│  (uses extended Origami cost model) │
├─────────────────────────────────────┤
│                                     │
│  Inputs:                            │
│    - GEMM problem (M, N, K, dtype)  │
│    - Collective (type, size, GPUs)  │
│    - Hardware (MI300X, MI350X, ...) │
│                                     │
│  Outputs:                           │
│    - CU partition (GEMM vs comm)   │
│    - hipBLASLt config + grid size  │
│    - RCCL algorithm/protocol hint  │
│    - Expected T_overlap            │
│    - Overlap strategy              │
│      (concurrent / DMA / fused)    │
│                                     │
├────────────┬────────────────────────┤
│ hipBLASLt  │         RCCL          │
│ (GEMM)     │   (Communication)     │
│            │                        │
│ MAX_CUS=N  │   CU reservation=M    │
│ StreamK    │   Priority scheduling  │
└────────────┴────────────────────────┘
```

## Implementation Path

### Phase 1: Offline Cost Prediction
- Extend [[origami]] with communication latency model
- Given (GEMM problem, collective spec), predict best overlap strategy and CU split
- No runtime changes to RCCL or hipBLASLt — just environment variable recommendations

### Phase 2: Library-Level Integration
- hipBLASLt API extension: `hipblasLtMatmulPreference_setConcurrentComm()`
- RCCL hint API: `ncclSetCUBudget()` or similar
- Both query the extended Origami model at launch time

### Phase 3: Runtime Coordination
- Shared memory region for CU budget negotiation
- Dynamic rebalancing as communication progresses
- Kernel-level awareness (fused patterns via [[triton-distributed]] or [[iris]])

## Validation Plan

1. Sweep GEMM shapes from LLaMA-70B/405B (realistic workload)
2. For each shape + collective: predict vs. measure T_overlap across CU splits
3. Compare against C3 paper baselines (21% → 72% of ideal)
4. Target: Origami prediction within 5% of measured optimal CU split
