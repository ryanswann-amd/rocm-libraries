# Key Papers

Papers and publications forming the knowledge base for this project.

## Core Papers

### tritonBLAS (arXiv:2512.04226)
**tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection**
Swann, Osama, Guo, Nelson, Zhang, Brown, Ong, Yazdani, Siddens, Dasika, Underwood (2025)
- Analytical GEMM cost model achieving 94.7% of autotuning
- 5-level tiling hierarchy, 3-level cache model
- Basis for [[origami]]

### C3 (arXiv:2412.14335)
**Optimizing ML Concurrent Computation and Communication with GPU DMA Engines**
- Concurrent compute-communication on MI300X
- CU partitioning, schedule prioritization, DMA offload
- ConCCL: 72% of ideal speedup with DMA + CU tuning
- CU partitioning characterization and interference measurements on MI300X

### Fused Computation-Collectives (arXiv:2305.06942, SC'24)
**Optimizing Distributed ML Communication with Fused Computation-Collective Operations**
- GPU-initiated communication within GEMM kernels
- GEMV+AllReduce (13% faster), GEMM+AllToAll (12%), Embedding+AllToAll (31%)
- Persistent kernel design, communication-aware WG scheduling
- ROC_SHMEM for GPU-initiated RDMA

### Iris (arXiv:2511.12500)
**Iris: A Multi-GPU Communication Library in Python and Triton**
- 5-pattern overlap taxonomy (bulk-sync, producer-consumer, sequential fused, WG specialization, wave specialization)
- Pure Python/Triton with SHMEM-like RMA APIs
- Average 1.21× over PyTorch+RCCL

### Triton-Distributed (arXiv:2504.19442)
**Triton-Distributed: Enabling Compute-Communication Overlap in Triton**
- MPMD programming model (symmetric memory, signal exchange, async-tasks)
- AllGather+GEMM and GEMM+ReduceScatter overlap
- 1.09–1.16× over PyTorch+RCCL on MI300X

### DMA Collectives (arXiv:2511.06605)
**DMA Collectives for Efficient ML Communication Offloads**
- Leverages existing SDMA engines, no hardware changes
- Synchronization with producer GEMM kernels
- Evaluated on MI300X 8-GPU Infinity Platform

## Additional References

### Flux (arXiv:2406.06858)
**Flux: GEMM Kernel Fusion with Communication for Fast LLM Inference**
ByteDance. CTA-level fusion on NVIDIA GPUs. COMET for MoE workloads.

### Stream-K (Osama, Merrill 2023)
Work-centric parallel decomposition eliminating wave quantization.

### Stream-K++ (arXiv:2408.11417)
7 scheduling policies, Bloom-filter configuration selection. 43% improvement on MI250X.

### TokenWeave (arXiv:2505.11329)
Efficient compute-communication overlap for distributed LLM inference.

### mKernel (UCCL Project)
Single persistent kernel: intra-node NVLink + inter-node RDMA + compute. Built on libibverbs.

### Characterizing Compute-Communication Overlap (arXiv:2507.03114)
Power and performance analysis of overlap on NVIDIA and AMD GPUs.
