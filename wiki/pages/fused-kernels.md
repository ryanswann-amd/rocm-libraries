# Fused Kernels

Single GPU kernels that combine GEMM computation with collective communication, enabling tile-level overlap without kernel launch boundaries.

## Paper

**Optimizing Distributed ML Communication with Fused Computation-Collective Operations** (SC'24)
arXiv:2305.06942

## Core Mechanism

Instead of:
```
launch(GEMM) → wait → launch(AllReduce) → wait
```

A single persistent kernel runs workgroups that:
1. Compute an output tile (GEMM partial)
2. Immediately communicate the result to remote GPUs
3. Loop to next tile while communication is in flight

## Persistent Kernel Design

- **Grid size fixed** at maximum occupancy (from `hipOccupancyMaxActiveBlocksPerMultiprocessor`)
- Each physical WG runs a **task loop** — each iteration handles one logical WG's work
- Logical WGs computing tiles destined for **remote peers execute first** — maximizes communication hiding

## Implemented Operators

### GEMV + AllReduce
- Two-phase direct AllReduce: ReduceScatter then AllGather
- Each GPU computes full GEMV output, reduces only 1/N tiles locally
- WGs handling same output tiles across GPUs coordinate reduction
- **Scale-up optimization**: zero-copy — WG threads store directly into peer GPU's buffer via Infinity Fabric
- Result: **13% lower latency** average, up to 22% vs. RCCL baseline

### GEMM + AllToAll (MoE)
- Implemented in Triton with ROC_SHMEM extensions
- WGs compute GEMM tiles then AllToAll-communicate to destination GPUs
- Result: **12% lower latency** average, up to 20% vs. RCCL baseline

### Embedding + AllToAll
- Persistent HIP kernel using ROC_SHMEM for intra-kernel RDMA
- Organized in slices; last-completing WG per slice triggers remote PUT
- Result: **20% lower latency** (intra-node), **31%** (inter-node)

## Communication Mechanisms

### Scale-Up (Intra-Node, Infinity Fabric)
- Native load/store to peer GPU memory — no RDMA library needed
- All WG threads store results directly to destination buffer (zero-copy)
- One ready flag per peer GPU per physical WG
- After stores complete, WG sets flag; peer WG polls flag, then reduces

### Scale-Out (Inter-Node, RDMA)
- ROC_SHMEM (`roc_shmem_malloc`) for NIC-registered symmetric heap memory
- Single thread from last-completing WG per slice issues non-blocking PUT
- Remote fence ensures data arrives before flag is set
- Receiver polls `sliceRdy` flags

### Last-WG Detection
- `WG_Done` bitmask per slice; each WG sets its bit on completion
- Cross-lane operations (GCN ISA) reduce bitmask efficiently
- Avoids inter-WG barrier instructions

## Communication-Aware Scheduling

Critical optimization: execute remote-bound tiles **before** local tiles.

Without scheduling: up to 7% execution skew between nodes (one node delays sending what another is waiting for).
With scheduling: skew drops to ~1%.

## Occupancy Impact

ROC_SHMEM API calls consume extra registers → **12.5% lower occupancy** vs. unfused kernel. Despite this, communication overlap more than compensates.

## Relationship to This Project

Fused kernels represent the most aggressive overlap strategy. For [[cost-model-extension]], modeling fused execution requires:

1. **Tile completion ordering** — which tiles finish when? Stream-K ([[stream-k]]) vs. data-parallel scheduling affects this
2. **Communication latency per tile** — α + tile_data / β_link for each remote transfer
3. **Pipeline depth** — how many tiles of communication can be in-flight while compute continues?
4. **Occupancy penalty** — extra registers for communication reduce GEMM occupancy
5. **CU allocation within kernel** — some WGs do compute, some do communication, same CU pool

This is harder to model than concurrent separate kernels because the compute-communication interleaving happens at the workgroup level, not the kernel level.
