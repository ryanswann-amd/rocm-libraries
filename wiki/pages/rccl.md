# RCCL

ROCm Communication Collectives Library — AMD's equivalent of NVIDIA's NCCL. Implements multi-GPU and multi-node collective communication operations optimized for AMD GPU interconnects.

## Repository

[GitHub: ROCm/rccl](https://github.com/ROCm/rccl)

## Collective Operations

| Operation | Description | Communication Pattern |
|-----------|-------------|----------------------|
| AllReduce | Reduce across all GPUs, result on all | All-to-all + reduction |
| AllGather | Gather from all GPUs, result on all | All-to-all (no reduction) |
| ReduceScatter | Reduce + scatter result chunks | All-to-all + reduction + scatter |
| Broadcast | One GPU → all GPUs | One-to-all |
| Reduce | All GPUs → one GPU (reduced) | All-to-one + reduction |
| AllToAll | Each GPU sends different data to each other | All-to-all (personalized) |
| Send/Recv | Point-to-point | Peer-to-peer |

## Execution Model

- RCCL collectives are **GPU kernel-based** — they launch kernels onto GPU CUs to orchestrate data movement
- Communication is **triggered by the host CPU** at kernel boundaries
- Uses GPU streams — collectives can overlap with compute if placed on different streams
- Internal algorithms: ring, tree, recursive halving-doubling, direct (fully-connected)

## Transport Layers

| Transport | Use Case | Bandwidth |
|-----------|----------|-----------|
| Infinity Fabric | Intra-node GPU-GPU (MI300X) | 64 GB/s per link |
| PCIe | GPU-CPU or cross-socket | ~32 GB/s |
| RDMA (ROCnRDMA) | Inter-node via InfiniBand/RoCE | NIC-dependent |
| Blit kernels | Intra-node, GPU-initiated DMA | Uses SDMA engines |

## Resource Consumption

RCCL collectives consume GPU resources when running:
- **CUs**: AllGather saturates at ~32 CUs; AllToAll saturates at ~64 CUs (see [[cu-partitioning]])
- **HBM bandwidth**: communication kernels read/write data through HBM
- **L2 cache**: communication kernels may pollute L2 with non-reusable data
- **Infinity Cache**: all traffic passes through MALL on MI300X

## Interaction with GEMM

In bulk-synchronous execution:
```
GEMM kernel completes → RCCL collective launches → next GEMM waits for collective
```

This serialization leaves potential performance on the table. Overlapping strategies:
1. **Stream-based overlap**: GEMM and RCCL on different streams — limited by GPU scheduler fairness
2. **CU partitioning**: Reserve CUs for RCCL via [[cu-partitioning]]
3. **DMA offload**: Bypass RCCL entirely, use SDMA engines (see [[dma-offload]])
4. **Kernel fusion**: Fuse GEMM + collective into single kernel (see [[fused-kernels]])

## Relationship to NCCL

RCCL tracks NCCL API compatibility. Same collective function signatures, same communicator model. Key differences are in transport layers (Infinity Fabric vs. NVLink) and internal kernel implementations.

## Codesign Opportunity

RCCL currently launches collectives without awareness of concurrent compute workloads. hipBLASLt launches GEMMs without awareness of concurrent communication. The [[codesign-rccl-hipblaslt]] page describes how mutual awareness could improve total throughput by:
1. RCCL selecting algorithms that use complementary resources
2. hipBLASLt/Origami selecting GEMM configs that leave headroom for communication
3. Both agreeing on CU partitioning and stream scheduling

## Key Papers

- C3 (arXiv:2412.14335) — characterizes RCCL collective interference with compute on MI300X
- Fused Computation-Collectives (arXiv:2305.06942) — bypasses RCCL with GPU-initiated communication
