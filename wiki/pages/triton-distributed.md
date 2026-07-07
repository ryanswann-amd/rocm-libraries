# Triton-Distributed

Extension of OpenAI Triton enabling concurrent GEMM computation and cross-GPU data transfers within a single kernel, providing fine-grained tile-level overlap.

## Repository

[GitHub: ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) | [Paper: arXiv:2504.19442](https://arxiv.org/abs/2504.19442) | [ROCm Blog](https://rocm.blogs.amd.com/software-tools-optimization/triton-distributed-c/README.html)

## Programming Model: MPMD

Three abstractions:

### Symmetric Memory
Each rank allocates equal-sized buffers; remote access via explicit primitives.

### Signal Exchange
Coordinates tasks via set/increment/spin-lock operations on signals in symmetric memory.

### Async-Tasks
Communication and compute run concurrently, synchronized through signals.

## API

Low-level primitives in `triton.distributed.language`:

```python
rank(axis=-1)              # current rank
num_ranks(axis=-1)         # total ranks
symm_at(ptr, rank)         # symmetric memory pointer for rank

wait(barrierPtrs, numBarriers, scope, semantic)   # wait on signal
consume_token(value, token)                        # create data dependency
notify(ptr, rank, signal=1, sig_op="set", comm_scope="inter_node")  # signal peer
```

Compose these with standard Triton compute blocks. The compiler handles shared-memory management, tensor core usage, and warp-level parallelism.

## Overlap Patterns

### AllGather → GEMM
- Communication threadblocks handle data transfers
- GEMM threadblocks augmented with `wait` + `consume_token`
- Each GEMM tile waits only for its specific data chunk
- Tiles computed in order of data arrival

### GEMM → ReduceScatter
Decomposed into 3 stages:
1. **Intra-node scatter** — via copy engine, zero SM cost
2. **Cross-node P2P** — ~1 SM
3. **Local reduction** — minimum SMs computed analytically

For 8×H800: if local reduction bandwidth exceeds ~470 GB/s, "perfect overlap" using as few as 15 SMs.

## Tile-Level Swizzling

Topology-aware tile scheduling:
- **NVSwitch (H800)**: Each rank gathers from one peer at a time, starts from local data
- **Full-mesh (MI300X)**: Must gather sub-chunks from all peers simultaneously; tiles sub-divided per step

## Performance

On AMD MI300X (8-GPU, vs. PyTorch+RCCL):

| Kernel | Speedup |
|--------|---------|
| AllGather + GEMM | 1.09× |
| GEMM + ReduceScatter | 1.16× |

On NVIDIA H800 (8-GPU, vs. PyTorch+NCCL):

| Kernel | vs NCCL | vs Flux |
|--------|---------|---------|
| AG+GEMM intra | 1.42× | 1.09× |
| GEMM+RS intra | 1.28× | 1.30× |
| AG+GEMM inter (16 GPU) | 1.33× | 0.96× |
| GEMM+RS inter (16 GPU) | 1.42× | 0.96× |

## Relevance to This Project

Triton-Distributed provides concrete test cases for validating [[cost-model-extension]]:
- Known GEMM configurations + known collective operations
- Measurable overlap efficiency
- AMD MI300X support (CDNA3)
- The SM/CU allocation for communication stages is analytically determined — can be compared against Origami predictions

See [[test-vehicles]].
