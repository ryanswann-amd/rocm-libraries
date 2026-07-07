# Communication Primitives

Collective communication operations used in distributed ML training and inference, and their cost characteristics relevant to [[cost-model-extension]].

## Operations and Their Costs

### AllReduce
- **Pattern**: Every GPU contributes data, every GPU gets the reduced result
- **Data volume**: Each GPU sends and receives `(N-1)/N × D` bytes (ring algorithm), where D = data size, N = GPU count
- **Decomposition**: ReduceScatter + AllGather
- **Latency**: `2(N-1) × α + 2(N-1)/N × D/β` (ring), where α = startup latency, β = link bandwidth
- **Use case**: Gradient synchronization in data-parallel training

### ReduceScatter
- **Pattern**: Reduce across GPUs, each GPU gets 1/N of the result
- **Data volume**: Each GPU sends `(N-1)/N × D` bytes
- **Latency**: `(N-1) × α + (N-1)/N × D/β` (ring)
- **Use case**: First half of AllReduce; output of GEMM in tensor-parallel inference

### AllGather
- **Pattern**: Each GPU contributes 1/N, every GPU gets the full result
- **Data volume**: Each GPU sends `(N-1)/N × D` bytes
- **Latency**: `(N-1) × α + (N-1)/N × D/β` (ring)
- **Use case**: Second half of AllReduce; input gather before GEMM in tensor-parallel

### AllToAll
- **Pattern**: Each GPU sends different data to each other GPU (personalized exchange)
- **Data volume**: Each GPU sends `(N-1)/N × D` bytes
- **Latency**: `(N-1) × α + (N-1)/N × D/β`
- **Use case**: MoE expert dispatch/combine, sequence-parallel redistribution

## Resource Profiles on MI300X

From the C3 paper (arXiv:2412.14335):

| Collective | CU Saturation Point | Memory BW Profile | Latency-Bound Threshold |
|------------|---------------------|--------------------|-----------------------|
| AllGather | 32 CUs | Low relative to GEMM | < 32 MB |
| AllToAll | 64 CUs | Moderate | < 64 MB |
| AllReduce | Varies (decomposed) | Sum of RS + AG | Varies |

Key insight: communication kernels have **much lower CU requirements** than compute-bound GEMMs, making them natural candidates for concurrent execution with [[cu-partitioning]].

## Bandwidth vs. Latency Regimes

- **Latency-bound**: Small messages where startup cost dominates (< tens of MB)
- **Bandwidth-bound**: Large messages where link bandwidth is the bottleneck
- Cost modeling must handle both regimes — the overlap benefit differs dramatically

## Producer-Consumer Patterns

### GEMM → ReduceScatter (gemm-producer, comm-consumer)
```
GEMM computes output tiles → each tile immediately sent to remote GPUs for reduction
```
- Benefit: communication starts before GEMM finishes, hiding latency
- Tile completion order matters — [[stream-k]] provides more predictable ordering than data-parallel
- See [[fused-kernels]] for implementation

### AllGather → GEMM (comm-producer, gemm-consumer)
```
AllGather brings remote data → GEMM consumes tiles as they arrive
```
- Benefit: GEMM starts before AllGather finishes
- Requires GEMM to handle partial input (tiles arriving incrementally)
- See [[triton-distributed]] for implementation

## Modeling Communication Cost

For extending [[origami]], communication cost has three components:
1. **Startup latency** (α): fixed per-operation overhead, kernel launch, synchronization
2. **Transfer time**: `data_volume / effective_bandwidth`
3. **Interference cost**: slowdown when running concurrently with compute (see [[compute-comm-overlap]])

The interference cost is the hard part — it depends on CU allocation, cache contention, and HBM bandwidth sharing.
