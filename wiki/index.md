# Wiki Index

Content catalog for origami_comms. Updated on every ingest.

## Project Context

- [Cost Model Extension](pages/cost-model-extension.md) — extending Origami for communication-aware GEMM cost prediction
- [C++ Port Plan](pages/cpp-port-plan.md) — original plan for porting the Python comm cost model into `origami::comm` (executed; see status)
- [C++ Port Status](pages/cpp-port-status.md) — as-built record of `origami_comms_cpp/`: M0–M9 complete, 34,599 bit-identical sweep cells, dashboard routed through it at 8.5× speedup; upstream into `rocm-libraries` is the next session
- [C++ Upstream Integration Plan](pages/cpp-upstream-integration-plan.md) — Stage 2 scope: land `origami::comm` into `rocm-libraries/shared/origami/`; audit vs the real upstream tree (HIP-required, C++17, nanobind, Catch2, rocm-cmake); sibling-types-first, ~2 PRs, ~2 weeks
- [AllGather Outlier Investigation](pages/ag-outlier-investigation.md) — root cause of the AG 41% MdAPE; per-ring-step sync + xGMI write concentration; closes to 6.1% MdAPE with two heuristics
- [Day 2 Atom Validation](pages/day2-atom-validation.md) — Triton `s_memrealtime` microbench on 8× MI300X validating per-WG HBM-write atom; 40 cells, alignment 101–358% (median 175.8%); pre-port spike Day 2 gate **green** for C++ port
- [Codesign: RCCL + hipBLASLt](pages/codesign-rccl-hipblaslt.md) — mutual awareness between communication and GEMM libraries
- [Test Vehicles](pages/test-vehicles.md) — concrete implementations for validating the cost model

## Core Components

- [Origami](pages/origami.md) — analytical GEMM cost model (the foundation being extended)
- [tritonBLAS](pages/tritonblas.md) — Triton-based GEMM framework and the paper behind Origami
- [hipBLASLt](pages/hipblaslt.md) — production GEMM library consuming Origami for kernel selection
- [RCCL](pages/rccl.md) — ROCm Communication Collectives Library

## Overlap Techniques

- [Compute-Communication Overlap](pages/compute-comm-overlap.md) — overview of approaches and interference analysis
- [CU Partitioning](pages/cu-partitioning.md) — dividing compute units between GEMM and communication
- [DMA Offload](pages/dma-offload.md) — using SDMA engines for interference-free communication
- [Fused Kernels](pages/fused-kernels.md) — single kernels combining GEMM + collective operations
- [Stream-K](pages/stream-k.md) — work-centric GEMM scheduling enabling flexible CU partitioning

## Frameworks

- [Triton-Distributed](pages/triton-distributed.md) — Triton extension for fused GEMM+comm overlap
- [Iris](pages/iris.md) — Python/Triton multi-GPU communication library with 5-pattern overlap taxonomy

## Hardware

- [MI300X / MI350X Architecture](pages/mi300x-architecture.md) — verified hardware specs, measured bandwidths, memory hierarchy, dispatch pipeline, xGMI, MFMA, coherence contracts, Origami calibration gaps

## Reference

- [Communication Primitives](pages/communication-primitives.md) — collective operations and their cost characteristics
- [Key Papers](pages/key-papers.md) — papers and publications forming the knowledge base
