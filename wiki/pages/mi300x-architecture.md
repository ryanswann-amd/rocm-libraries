# MI300X / MI350X Architecture

Verified hardware architecture details for AMD Instinct MI300X (CDNA3, gfx942) and MI350X/MI355X (CDNA4, gfx950), focused on parameters relevant to modeling concurrent compute-communication costs.

Data sourced from global wiki experiments (K-series tickets) and AMD documentation. All bandwidth/latency numbers are measured unless noted otherwise.

## Die Structure

Both architectures use 8 XCDs (Accelerator Complex Dies) over 4 AIDs/IODs (I/O Dies), 2 XCDs per IOD.

| Property | MI300X (CDNA3, gfx942) | MI355X (CDNA4, gfx950) |
|----------|------------------------|------------------------|
| CUs per XCD | 38 | 32 |
| Total CUs | 304 | 256 |
| SIMDs per CU | 4 | 4 |
| Total SIMDs | 1,216 | 1,024 |
| Clock (sclk_max) | ~2.0 GHz | 2.4 GHz |
| TDP | 750 W | 750 W (MI355X); 1200 W+ (MI375X) |

MI350X variant: 7.1 TB/s HBM (vs 8.0 for MI355X/MI375X). MI350P: half-die, 4 XCDs, 128 CUs, PCIe only (no xGMI).

## Dispatch Pipeline

Two-stage pipeline with asymmetric rates — a critical constraint for short kernels:

- **CPC** (Command Processor Complex): 1 workgroup/cycle enumeration
- **SPI** (Shader Processor Input): 4 cycles/wave dispatch
- Workgroups < 8 waves starve execution units (CPC outruns SPI)
- Dispatch-bound threshold: `Tp ≈ #CUs × Wo / WG × 4` cycles. For MI300X at max occupancy (32 waves/CU, 4 waves/WG): **~2,432 cycles**
- AQL dispatch overhead: 1.5–2.0 µs floor; HIP graphs achieve 25–40% speedup via fence elision

**Per-CU resource allocation (all-or-nothing at dispatch):**

| Resource | MI300X | MI350X |
|----------|--------|--------|
| VGPRs per SIMD | 512 (blocks of 16) | 512 (blocks of 16) |
| SGPRs per SIMD | 112 (blocks of 8) | 112 (blocks of 8) |
| Wave slots per SIMD | 8 (max 32 waves/CU) | 8 |
| LDS per workgroup | 64 KB (512-byte granularity) | 160 KB (1,280-byte granularity) |
| LDS banks | 32 × 4 B/cycle = 128 B/cycle | 64 banks = 256 B/cycle (2×) |
| Barriers per CU | 32–64 | 32–64 |
| Bulky WG slot | 1 (for LDS > 32 KB) | 1 |

## Compute: MFMA Pipeline

### Measured Initiation Intervals (CDNA3/gfx942)

| Instruction | II (cycles) | Depth | RAW Latency | Source |
|-------------|-------------|-------|-------------|--------|
| `mfma_f32_16x16x16_f16/bf16` | 16 | 2 | 32 | K-4434, K-5285 |
| `mfma_f32_32x32x8_f16/bf16` | 32 | 2 | 64 | K-4564, K-4862 |
| `mfma_f32_4x4x4_f16` | 8 | — | — | K-3938 |
| `mfma_i32_16x16x32_i8` | 16 | — | — | K-5349 |
| `mfma_i32_32x32x16_i8` | 32 (+6 cy stall) | — | — | K-3934 |
| `mfma_f32_16x16x4_f32` | 16 | 1 | 128 | K-5085 |

**Accumulator forwarding is free**: back-to-back MFMAs reusing the same AGPR run at exactly II with zero NOPs. AGPR-to-VGPR round-trip costs 19–50 cycles (5–12× costlier).

**MFMA-VMEM dual-issue**: Independent pipes, additive throughput. Optimal ratio 8:1 MFMA-to-VMEM for F16 32×32×16 sustains 1,092 TFLOPS (95.5% peak) with 17 TB/s VMEM throughput.

**Per-CU saturation**: Minimum 4 waves (1 per SIMD); optimal 8 waves yields 91.5% of peak; beyond 8 waves < 2% gain (K-5290).

**Dense peak reconciliation**: FP16: 1.27 PFLOPS (spec 1.31); FP32: 0.164 PFLOPS (spec 0.163). FP16:FP32 ratio = 7.78× = 4× (K-packing) × 1.94× (instruction-rate gap).

### MI350X Compute Changes

**XDL2x instructions**: Higher-density MFMA variants. Doubled K-dimension (e.g., 32×32×16 bf16 instead of 32×32×8) at same 32-cycle II = 2× throughput. Introduce new hazards: XDL-to-XDL2x transitions require 2 NOPs; overlapping SrcC requires 4–18 software NOPs (vs 0–2 on MI300X).

**New MFMA shapes**: `MFMA_F32_16X16X32_F16`, `MFMA_F32_32X32X16_F16`, sparse variants (`SMFMAC`), native FP4/FP6 scaled operations.

**Dense peak**: FP16 ~2 PFLOPS (vs ~1.3 MI300X, +54%); FP4 10.1 PFLOPS.

**Precision format shift**: Native IEEE FP8 (E5M2, E4M3) replaces AMD FNUZ format. Native MXFP4/MXFP6 as first-class. TF32 deprecated.

## Memory Hierarchy

### Measured Latency Ladder (MI300X, K-7197)

| Level | Latency | Capacity per CU/XCD | Total |
|-------|---------|---------------------|-------|
| vL1D (TCP) | 50–82 ns | 32 KB per CU | 9.7 MB |
| L2 (TCC) | 125 ns | 4 MB per XCD | 32 MB |
| MALL (Infinity Cache) | 304 ns | — | 256 MB |
| HBM3 | 391 ns | — | 192 GB |

### TCP/vL1D (Critical Bottleneck)

- **32 KB per CU, constant across all MI-series**
- **TCP FIFO saturation**: After ~5 wide `buffer_load`s per wave, stalls occur from TCP capacity fill-up (not vmcnt exhaustion). This is the real memory bottleneck for vector operations, not LDS bank conflicts.
- vL1D is a **single dynamically-shared pool** per CU, NOT per-wave partitioned (K-7809: static way-partition hypothesis FALSIFIED). Two wavefronts share until union WSS exceeds 32 KiB, then both drop to L2 simultaneously.
- Per-wave MSHR: 12 outstanding `global_load_dwordx4` (~12 KiB in-flight); sharp cliff at N=13
- Per-CU MSHR: 128 cache-line misses (scatter) vs 768 wave-coalesced instructions (streaming) — 24× gap
- MI350X: Per-wave VMEM allocation ~50% smaller; stall onset at 5th buffer_load

### L2 Cache (TCC)

| Property | MI300X | MI350X |
|----------|--------|--------|
| Capacity per XCD | 4 MB | 8 MB (2×) |
| Total | 32 MB | ~64 MB |
| TCC slices | 32 total | — |
| Read/write BW | baseline | 2× |

**No hardware cross-XCD coherency.** A `.cg` store on XCD0 is invisible to XCD1. Cross-XCD producer-consumer requires the three-component protocol:
1. **Write-through** (`.wt`) data stores — bypass L2 to HBM
2. **Atomic release** on completion flags (`sem="release", scope="gpu"`)
3. **Atomic acquire** on flag reads (`sem="acquire", scope="gpu"`)

Scale-dependent bug: problems with M < 65K often fit single-XCD, masking coherency bugs that surface only at production scale.

**Cache flush**: Probabilistic eviction on 32-slice TCC requires writing 16× L2 capacity (256M elements) to guarantee P(survive) < 0.001.

### HBM

| Property | MI300X | MI350X |
|----------|--------|--------|
| Technology | HBM3 | HBM3E |
| Capacity | 192 GB (8 stacks × 24 GB) | 288 GB |
| Peak BW (spec) | 5.3 TB/s | 8.0 TB/s (MI355X) |
| Measured peak read | 4.73 TB/s (saturates at 144 CUs) | — |
| Measured peak write | 5.14 TB/s (saturates at 280 CUs) | — |
| Per-CU coalesced | 110 GB/s | — |
| Per-CU scatter | 3.33 GB/s (33× penalty) | — |

**HBM arbiter saturation** (critical finding): The TCC arbiter queue saturates at single-XCD occupancy for non-sequential patterns. Aggregate ceiling ~4.1–4.3 TB/s (77–81% of spec) regardless of XCD count. Adding XCD pairs degrades aggregate: 7-pair → 59% of peak. Only sequential access scales multi-XCD (4.3× at 8 XCDs, 54% efficiency).

**Bidirectional R+W collapse**: 64% below naive sum due to HBM3 row-buffer thrashing.

**Address interleaving**: 64B × 256 channels (16 KiB period). Shared regions outperform per-XCD partitioning by 7–15% for non-sequential patterns — hardware interleaving provides better load balancing than software partitioning.

## xGMI Interconnect (MI300X 8-GPU)

### Link Performance (Measured)

| Metric | Value |
|--------|-------|
| Per-link unidirectional | 49.1 GiB/s (<2% cross-cluster variance) |
| Full-duplex bidirectional (symmetric) | 95.7 GB/s (47.8 per direction, 96% of theoretical) |
| Full-duplex (single-initiator) | 50.8 GB/s (47% collapse from credit-pool exhaustion) |
| Mixed W+R same channel | 44 GB/s (56% loss vs unidirectional) |
| Wire-to-payload overhead | 1.23–1.24× (framing, ECC, flit boundaries) |
| Ring topology efficiency | 98.7% per-link |
| All-to-all fabric efficiency | 12.6% per-link (87% collapse from arbiter saturation) |

### Scaling Regimes

| Regime | N (concurrent sources) | Behavior |
|--------|------------------------|----------|
| Linear | ≤ 6–8 | 49N GB/s, both constraints slack |
| Per-source saturation | 8–14 | Individual GPUs hit 49 GB/s egress ceiling |
| Fabric saturation | ≥ 14–20 | Aggregate plateaus at **334 GB/s** |

- **Per-GPU egress ceiling**: 49.2 GB/s aggregate, independent of link count or destination count (validated N=1 to N=7; per-link BW = 49/N)
- **Bidirectional saturation knee**: 16 concurrent streams per direction → 88.9 GiB/s
- Jain fairness at saturation: 0.977

### SDMA Engine Performance

| Metric | Value |
|--------|-------|
| SDMA read | 49.5 GB/s per link |
| SDMA write | 23.6 GB/s per link (asymmetric throttling) |
| Per-GCD saturation | 10 streams/GCD hard cap |
| SDMA-kernel contention (shared link) | η=0.209 SDMA, η=0.91 kernel (4.3× asymmetric starvation) |
| Disjoint flows | η=1.013 (zero mutual interference — controller-local arbitration) |
| Compute-SDMA overlap | 95% (5% residual from shared HBM bank aliasing) |

### Direction-Asymmetric Interference

- **Peer-READ destroys local memory BW 30–43× harder than peer-WRITE**
- READ egress collapses cross-IOD MALL BW by -64.5% at P=7 (slope -176.89 GB/s/peer)
- WRITE ingress costs only -1.4%/peer on local HBM reads (capped at -17%)
- Root cause: WRITE ingress bypasses MALL; READ egress shares AID DF read-bandwidth arbitration

### MI350X Interconnect Changes

- 34% higher per-link bandwidth (MI355X); 27% higher (MI350X)
- Mesh bandwidth: ~1,075 GB/s (MI350X 8-GPU) vs ~896 GB/s (MI300X)
- MI350P: No xGMI — PCIe Gen5 only

## Data Fabric Arbitration

**Per-cacheline round-robin**: Memoryless, Jain fairness ≥ 0.99 for temporal patterns. Breaks down for structural heterogeneity — large bursts suffer tail-latency amplification from interleaving with small bursts.

**Request-proportional sharing prevents CU-based isolation**: CU masking cannot partition crossbar bandwidth. A 4-CU victim issues ~10× fewer requests and receives proportionally fewer arbiter slots (Hill coefficient α ∈ [0.94, 1.18], pure Michaelis-Menten saturation). **No resource partitioning strategy isolates via CU count.**

**Coherence collapse under saturation**: 11× bandwidth reduction when DF crossbar saturation coincides with coherence operations. Coherence probes compete in the proportional-share arbiter; multiplicative protocol overhead cascades.

### Four-Layer Ingress Contention Hierarchy

| Layer | Peak Degradation | Mechanism |
|-------|-----------------|-----------|
| CU scheduling | 28–40% | Wavefront dispatch competition |
| Cache fabric (TCC/LLC) | 86% | L3-resident aggressors saturate cache bisection BW |
| DF arbitration | 11–42% | Bidirectional DF R+W creates 6.13× super-additive congestion |
| HBM write-buffer | 10–99% | ILP32 causes 98.97% collapse |

## Communication Kernel Resource Profiles (Measured)

### RCCL AllReduce on 8×MI300X

```
T_allreduce(N bytes) = 55.045 µs + 5.42 ns/byte × N
Effective BW: 184.5 GB/s per GPU
Wire traffic: 322 GB/s per GPU (1.75× ring amplification)
```

- HBM utilization: only 15.3% (6.5× headroom)
- Reduction ALU: 0.7% utilization (141× headroom)
- **Bottleneck is exclusively xGMI fabric transport**
- Operator invariance: sum/max/min/prod differ by 0.27%
- Dtype invariance: FP8–FP64 differ by 0.43% (byte-pipeline-bound)

### CU Sweet Spots for Collectives

| Collective | CU Sweet Spot | Peak BW | Per-CU Efficiency Collapse |
|-----------|--------------|---------|---------------------------|
| AllReduce | 60 CUs | 167 GB/s | 6.7× from 32→304 CUs |
| ReduceScatter | 32 CUs | — | Over-allocation penalty 14–31% |
| AllToAll | 32 CUs | — | Over-allocation penalty 14–31% |

- Latency floors are CU-insensitive (±0.2 µs across CU∈{32, 60, 104})
- RCCL dispatches exactly 1 kernel per AllReduce call (not 1 per ring step)

### Multi-Primitive Interference

- **Super-additive**: Pairwise models underestimate N≥3 concurrent collective slowdown by 12–48%
- AllGather acts as a binary anchor — its presence dominates slowdown regardless of what else runs concurrently
- Fused AllReduce beats AG+RS decomposition by **~2.5×** at 8 GPUs; gap grows with world size
- Barrier+atomic coupling exhibits extreme tail-latency amplification under saturation

### Comm+GEMM Overlap

Naive stream-based overlap consistently underperforms expectations. Root causes: HBM bandwidth contention, XCD crossbar saturation, and launch-queue serialization — not compute capacity. GEMM efficiency remains 73–90% during contention (compute is never the bottleneck; the memory fabric is).

Viable overlap requires either small-message collectives, high AG/GEMM time ratios (> 0.8), or moving to [[fused-kernels]] / [[dma-offload]] approaches that avoid the fabric contention path entirely.

## MI350X Atomic and Memory Ordering

| Change | MI300X | MI350X |
|--------|--------|--------|
| CAS release fence | 10.16 µs | 0.12 µs (85× collapse — fence folded into instruction) |
| Ordering tetrad spread | 10.32 µs | 1.12 µs (12× compression) |
| XCHG acq_rel anomaly | acq_rel faster than relaxed (0.75×) | Inverted: acq_rel 1.10× slower (intuitive ordering restored) |
| Bitwise atomic fence fusion | 6–12× penalty vs CAS | Closed — L2 atomic engine fusion |
| Native BF16 atomics | Missing (CMPSWAP emulation, 10× cost) | `global_atomic_add_bf16` native |
| LDS load transpose | Not available | `ds_load_b128` + transpose instructions |

CDNA4 atomic contention: 1.69× the contention pressure despite 1.6–2× faster solo atomics (faster engines saturate arbiter sooner).

## Cross-XCD Memory Coherence Contracts

Coherence requires alignment of three independent control surfaces:

### Instruction Modifiers (SC[1:0])
- `00` Wave, `01` Workgroup, `10` Device, `11` System
- Scope controls where the operation resolves, NOT whether the underlying memory allows coherent access

### Memory Pool Policies (allocation-time MTYPE)
- **Coarse-grained** (`hipMalloc`): RW-cached locally, NC remotely. Cross-GPU atomics serialize within each L2 but not across. Stale reads at 1–5% rate.
- **Fine-grained** (`hipDeviceMallocFinegrained`): True atomic serialization across all GPUs/XCDs. ~20% local bandwidth penalty from coherence traffic.
- **Uncached** (`hipDeviceMallocUncached`): Bypasses all caches. Eliminates stale-read risk but doubles latency for small updates.

### Hierarchical Synchronization (100× traffic reduction)
Two-level barrier: all WGs per XCD atomic-add to per-XCD counter (fine-grained), then one elected leader per XCD atomic-adds to global counter. Only 8 threads contend on xGMI links (vs hundreds without hierarchy). Reduces multi-microsecond serialization to sub-100 ns.

**Packed atomic signaling**: 64-bit atomics with bit fields for flags + numeric fields for reduction values. Fence-per-operation cost drops from 800 ns to 50 ns when 16 flags pack into one atomic word.

### Persistent Kernel Grid Sizing
Max safe grid: `#CUs × waves_per_CU × threads_per_wave / threads_per_WG`. For MI300X: ~2,400 workgroups of 256 threads. Oversubscription at cooperative barrier points → deadlock.

## Origami Calibration Status

### MI300X (gfx942): Calibrated
All bandwidth probes, MFMA latencies, and heuristic parameters tuned. 94.7% of exhaustive autotuning across 150K shapes.

### MI350X (gfx950): Major Gaps
- All 9 bandwidth probes missing (hbm_read/write, l1_read, l2_read/write, lfifo_stride/width, xgmi_bw, rccl_busbw)
- `mem_bw_per_wg_coefficients` known wrong (saturates at 125 CUs; MI350X has 256 CUs with linear scaling)
- Clock bug: calibration scripts hardcoded 2.0 GHz but MI355X runs at 2,400 MHz
- Best measured TFLOPS: 1,454–1,493 BF16 (64.9% of peak)
- `main_loop_efficiency` for gfx950: 1.30 general BF16, 1.10–1.15 CMS-optimized (not in DB)

## Key Numbers for Cost Modeling

| Resource | MI300X | MI350X | Notes |
|----------|--------|--------|-------|
| Total CUs | 304 | 256 | 38/XCD vs 32/XCD |
| Min CU allocation | 8 | 8 | 1 per XCD |
| L2 per XCD | 4 MB | 8 MB | No cross-XCD coherency |
| MALL | 256 MB | — | Shared, memory-side |
| HBM BW (measured peak read) | 4.73 TB/s | ~8 TB/s (spec) | Arbiter ceiling ~4.1 TB/s (non-seq) |
| xGMI per-link | 49.1 GiB/s | ~65 GiB/s | 34% higher on MI355X |
| Per-GPU egress ceiling | 49.2 GB/s | — | Independent of fan-out |
| Fabric aggregate ceiling | 334 GB/s | ~450 GB/s | 8-GPU fully connected |
| SDMA engines | 14 | 14 | Zero CU cost; 95% compute overlap |
| AllReduce CU sweet spot | 60 CUs | — | Over-allocation degrades 14–31% |
| AllReduce model | 55 µs + 5.42 ns/B | — | xGMI-fabric-bound |
| LDS per WG | 64 KB | 160 KB | 2.5× increase on MI350X |
| TCP per CU | 32 KB | 32 KB | Unchanged; saturation at ~5 wide loads |
| MFMA 32×32 II | 32 cycles | 32 cycles | MI350X: doubled K-dim at same II |
| vL1D hit latency | 50–82 ns | — | Dynamic shared pool, not per-wave |
| L2 hit latency | 125 ns | — | — |
| MALL hit latency | 304 ns | — | — |
| HBM latency | 391 ns | — | — |
