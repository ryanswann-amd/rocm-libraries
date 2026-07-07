# Iris

Multi-GPU communication library implemented entirely in Python and Triton, providing SHMEM-like RMA APIs with full compiler visibility into both computation and communication.

## Repository

[GitHub: ROCm/iris](https://github.com/ROCm/iris) | [Paper: arXiv:2511.12500](https://arxiv.org/abs/2511.12500) | [Docs](https://rocm.github.io/iris/)

## Key Difference from RCCL

RCCL is an opaque C++ binary — the compiler cannot see inside its collective implementations. Iris is pure Python/Triton, so the compiler has full visibility into both GEMM and communication code. This enables the compiler to optimize across the compute-communication boundary.

## Symmetric Memory

Allocated via `hipMalloc`, shared across GPUs using HIP IPC handles. Pointer translation adds negligible overhead — heap_bases array is 64 bytes, stays L1-cached.

## Device-Side APIs

```python
iris.load(ptr, rank)            # pull from remote GPU
iris.store(ptr, value, rank)    # push to remote GPU
iris.get(dst, src, rank)        # RMA get
iris.put(dst, src, rank)        # RMA put
iris.copy(dst, src, rank)       # remote copy

iris.atomic_add(ptr, val, rank, sem="relaxed", scope="gpu")
iris.atomic_cas(ptr, cmp, val, rank, sem="acquire", scope="gpu")
iris.atomic_xchg(ptr, val, rank, sem="release", scope="gpu")
```

## 5-Pattern Overlap Taxonomy

This taxonomy is critical for [[cost-model-extension]] — each pattern has different cost characteristics:

### 1. Bulk-Synchronous (Unfused)
```
GEMM completes → barrier → scatter starts
```
- Simplest model: `T = T_gemm + T_comm`
- No overlap, no interference

### 2. Producer-Consumer via Streams (Unfused)
```
Two kernels on separate HIP streams; CUs partitioned
(e.g., 256 GEMM + 48 scatter on MI300X's 304 CUs)
```
- Cost: `T = max(T_gemm(256 CUs), T_comm(48 CUs))`
- Overlap via hardware concurrency
- See [[cu-partitioning]]

### 3. Sequential Fused
```python
for tile_id in range(pid, total_tiles, NUM_SMS):
    c = gemm_loop(A, B, C)
    for remote_rank in range(world_size):
        iris.store(C + offset, c, cur_rank, remote_rank, heap_bases, mask=mask)
```
- GEMM tile computed → immediately scattered within same kernel
- Minimal code change, no intermediate global memory
- Cost: each tile's communication overlaps with next tile's compute

### 4. Workgroup Specialization (Fused)
```python
if pid < GEMM_SMS:
    # Compute GEMM tile
    tl.store(C + offset, c, cache_modifier=".wt")
    tl.atomic_cas(locks + tile_id, 0, 1, sem="release", scope="gpu")
else:
    # Communication: wait for tile, then scatter
    while tl.atomic_cas(locks + tile_id, 1, 0, sem="acquire", scope="gpu") == 0:
        pass
    iris.put(C + offset, C + offset, cur_rank, remote_rank, heap_bases, mask=mask)
```
- Single persistent kernel; `pid < GEMM_SMS` computes, `pid >= GEMM_SMS` communicates
- True overlap, one kernel launch
- Uses atomic_cas with release/acquire semantics for tile-level sync

### 5. Wave Specialization (Experimental)
- Work split at wavefront granularity (64 threads on AMD)
- Not well suited to Triton's programming model

## Performance

Average 1.21× speedup over PyTorch+RCCL (range 0.93–1.79×).

Best case: `8192×4608×36864` at 1.8× faster on 4 GPUs.

At `8192×3584×14336` on 8 GPUs, producer-consumer and WG specialization nearly fully hide communication.

## Relevance to This Project

Iris's 5-pattern taxonomy maps directly to the cost model tiers in [[cost-model-extension]]:
1. Bulk-sync → baseline (no extension needed)
2. Producer-consumer → [[cu-partitioning]] model
3. Sequential fused → tile-level pipeline model
4. WG specialization → CU-partitioned fused model
5. Wave specialization → future work

Each pattern isolates a different overlap mechanism, providing clean validation targets.

See [[test-vehicles]].
