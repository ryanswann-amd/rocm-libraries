# Communication Cost Model — Design Document

First-principles analytical model for predicting collective communication latency on AMD MI300X/MI350X, designed as a companion to Origami's GEMM cost model.

## Design Philosophy

1. **First-principles only.** Every parameter traces to an architectural constant (link bandwidth, CU count, cache capacity) or a microarchitectural measurement (dispatch latency, protocol overhead). No regression models, no lookup tables, no correction factors.

2. **Mirror Origami's patterns.** The API uses the same `(problem, hardware, config) → context → latency` hierarchy. A communication workgroup is the atomic unit, paralleling how Origami treats the MFMA instruction as the compute atom.

3. **Compose from workgroup level up.** The model builds device-level predictions from workgroup-level costs, just as Origami builds GEMM predictions from per-tile costs scaled by `num_timesteps`.

4. **Composable primitives, not hard-coded collectives.** A collective is defined as a per-WG work graph of primitive operations (load, push, pull, reduce, signal, wait) that each map to specific functional units. Symbolic analysis of the work graph determines which functional unit binds. This follows the Iris API pattern and enables modeling novel/fused collectives without new code.

---

## Type Definitions

### `comm_problem_t` — What to communicate

```python
@dataclass
class comm_problem_t:
    message_bytes: int          # total message size across all ranks
    num_gpus: int               # world size (N)
    dtype: DataType             # element type (for reduction arithmetic)
    # No collective enum — the collective is defined by its work graph (see below)
```

Parallel to `problem_t` (M, N, K, dtypes). Defines the workload. The collective operation itself is expressed as a composition of primitive operations in a WG work graph — not as an enum. This means the model handles arbitrary collectives, fused patterns, and custom protocols without modification.

Standard collectives are provided as named work graph constructors (e.g., `allgather_ring_step()`, `reduce_scatter_ring_step()`).

### `comm_hardware_t` — What the fabric provides

```cpp
struct comm_hardware_t {
    // xGMI link parameters (from microarch measurement)
    double link_bw;                 // per-link unidirectional BW (49.1 GiB/s MI300X)
    double gpu_egress_ceiling;      // per-GPU aggregate outbound cap (49.2 GB/s MI300X)
    size_t num_peer_links;          // links to other GPUs (7 for 8-GPU fully connected)

    // SDMA engines
    size_t num_sdma_engines;        // 14 on MI300X
    double sdma_read_bw;            // per-link SDMA read (49.5 GB/s)
    double sdma_write_bw;           // per-link SDMA write (23.6 GB/s)

    // Protocol overhead (from microarch measurement)
    double step_sync_latency_ns;    // per-step synchronization cost
    double launch_overhead_ns;      // kernel dispatch overhead

    // Topology
    topology_t topology;            // fully_connected, ring, tree
};
```

Extends `hardware_t` with inter-GPU parameters. Constructed from architecture constants the same way `hardware_t` is constructed from `hipDeviceProp_t`.

### `comm_config_t` — How to communicate

```cpp
struct comm_config_t {
    algorithm_t algorithm;      // ring, tree, direct
    protocol_t protocol;        // simple, ll128, ll
    size_t num_channels;        // RCCL channel count (1–128)
    size_t num_cus;             // CUs allocated to communication
};
```

Parallel to `config_t` (tile sizes, occupancy, reduction strategy). Defines the execution strategy.

### `comm_context_t` — Derived execution parameters

```cpp
struct comm_context_t {
    // Derived from (problem, hardware, config) — computed once, passed through hierarchy
    size_t num_steps;               // algorithm-dependent step count
    size_t tile_bytes;             // bytes transferred per step per channel
    size_t active_links_per_step;   // links active in each step
    double effective_bw_per_step;   // bandwidth available per step (after ceiling/contention)
    double protocol_efficiency;     // payload fraction (Simple: ~1.0, LL128: 0.9375, LL: 0.5)
    size_t active_cus;              // CUs actually used (min of allocated and saturated)
    double hbm_bw_consumed;         // HBM bandwidth consumed by communication
};
```

Parallel to `context_t`. All derived values computed in the constructor from the three input structs.

### `comm_prediction_result_t` — Output

```cpp
struct comm_prediction_result_t {
    double latency_ns;              // predicted total latency
    comm_config_t config;           // configuration used
    bottleneck_t bottleneck;        // which resource limits: link_bw, egress_ceiling, hbm_bw, protocol
};
```

Parallel to `prediction_result_t`. Adds bottleneck attribution for diagnostics.

---

## The Workgroup-Link-Chunk: The Model's Atom

The fundamental unit of cost prediction is the **workgroup-link-tile (WLT)**: the time for one workgroup to transfer one tile of data over one xGMI link, given all workgroups contending for that link.

A tile has structure — it is `BLOCK_M × BLOCK_N` elements of a given dtype, not just a byte count. This structure determines cache line count, load width efficiency, and reduction element count.

```
T_wlt = time for 1 WG to move one tile over 1 link
        while sharing that link with (wgs_on_link - 1) other WGs
        and sharing the GPU egress ceiling with (concurrent_links - 1) other active links
```

### Definition

```python
@dataclass
class WorkgroupLinkTile:
    """The atom: one workgroup, one link, one tile."""

    # Tile shape (structured, not just bytes)
    block_m: int                    # tile rows
    block_n: int                    # tile columns
    dtype: DataType                 # element type
    tile_bytes: int                 # = block_m × block_n × dtype_bytes
    tile_cl: int                    # = ceil(tile_bytes / 64), cache lines to move
    elements: int                   # = block_m × block_n

    # Where
    link_id: int                    # which physical xGMI link (0..6 for 8-GPU)
    peer_rank: int                  # remote GPU at the other end
    direction: str                  # "push" (local→remote, egress) or "pull" (remote→local, ingress)

    # Contention
    wgs_on_link: int                # total WGs sharing this link simultaneously
    concurrent_links: int           # total links active on this GPU simultaneously
    total_comm_wgs: int             # total comm WGs active on this GPU

    # CU execution parameters (from config)
    load_width: int                 # bytes per VMEM instruction (4, 16, or 64)
    vgprs_for_data: int             # VGPRs available for data per iteration
```

### Derived Parameters

```python
    @property
    def instrs_per_cl(self):
        """VMEM instructions needed per cache line."""
        return 64 // self.load_width        # dwordx16=1, dwordx4=4, dword=16

    @property
    def cl_per_iter(self):
        """Cache lines processed per loop iteration (register budget)."""
        return (self.vgprs_for_data * 4) // 64

    @property
    def num_iters(self):
        """Loop iterations to move the full tile."""
        return ceil(self.tile_cl / self.cl_per_iter)

    @property
    def bw_per_wg(self):
        """Effective xGMI bandwidth available to this WG (bytes/ns)."""
        egress_per_link = gpu_egress_ceiling / self.concurrent_links
        link_cap = min(link_bw, egress_per_link)
        return link_cap / self.wgs_on_link
```

### The WLC Cost Equation

The cost of one WLC is a software-pipelined loop over iterations, where each iteration's time is the max of all functional units in the data path:

```
T_wlt = T_prologue + (num_iters - 1) × T_steady_state + T_epilogue + T_sync

T_steady_state = max(T_vmem,          ← CU: instruction issue rate
                     T_tcp,           ← CU: vL1D/TCP throughput (32 KB)
                     T_l2,            ← XCD: L2/TCC bandwidth (4 MB per XCD)
                     T_mall,          ← Device: MALL/Infinity Cache (256 MB)
                     T_hbm_read,      ← Device: HBM read bandwidth
                     T_hbm_write,     ← Device: HBM write bandwidth
                     T_xgmi_read,     ← Fabric: xGMI ingress (for Pull)
                     T_xgmi_write,    ← Fabric: xGMI egress (for Push)
                     T_valu)          ← CU: reduction ALU

T_prologue     = max(read-side times)     ← fill the pipeline (first iter memory-only)
T_epilogue     = max(write-side times)    ← drain the pipeline (last iter write-only)
T_sync         = N_atomics × atomic_latency  ← signal/wait at step boundaries
```

Each `T_*` for one iteration:

```
T_vmem       = (cl_per_iter × instrs_per_cl) / vmem_issue_rate
T_tcp        = (cl_per_iter × 64)            / tcp_bw_per_cu
T_l2         = (cl_per_iter × 64)            / l2_bw_per_cu
T_mall       = (cl_per_iter × 64)            / mall_bw
T_hbm_read   = (cl_per_iter × 64)            / hbm_read_bw_per_cu
T_hbm_write  = (cl_per_iter × 64)            / hbm_write_bw_per_cu
T_xgmi_read  = (cl_per_iter × 64)            / bw_per_wg          ← contention-adjusted
T_xgmi_write = (cl_per_iter × 64)            / bw_per_wg          ← contention-adjusted
T_valu       = elements_per_iter              / valu_rate
```

Not all terms are active for every primitive. The work graph determines which are nonzero.

**Push/Pull are two-sided operations.** A `Push` consumes xGMI **write** (egress) on the sending GPU and xGMI **read** (ingress) on the receiving GPU. A `Pull` consumes xGMI **read** (ingress) on the requesting GPU and xGMI **write** (egress) on the serving GPU. The model evaluates from one GPU's perspective — but the peer GPU has a symmetric WLC with the complementary xGMI direction. Both sides must be below their respective ceilings for the transfer to proceed at the predicted rate.

This matters because direction has asymmetric interference: peer-READ (Pull from sender's perspective) degrades local MALL bandwidth 30–43× harder than peer-WRITE (Push from sender's perspective). The work graph captures this: a `Push` on GPU A corresponds to a receive on GPU B, and the interference cost depends on which direction each GPU experiences.

### The Schedule Layout: `(pid, i) → link`

A collective kernel has two loop levels:

```python
for tile_id in range(pid, total_tiles, COMM_SMS):       # outer: persistent tile loop
    for peer_visit in range(num_peer_visits):             # inner: peer visit loop
        link = link_of(pid, peer_visit)                   # which xGMI link
        ops  = work_at(peer_visit)                        # what primitives to execute
        execute(ops, tile_at(tile_id), link)
```

**Outer loop (tile assignment):** Each WG (`pid = tl.program_id(0)`) processes a stripe of tiles spaced `COMM_SMS` apart. This determines **which data** moves but doesn't affect **which link** is used. The tile has shape `BLOCK_M × BLOCK_N` elements.

**Inner loop (peer visits):** Each iteration is one **peer visit** — the WG interacts with one remote GPU over one xGMI link. A peer visit might read from the peer (Pull), write to the peer (Push), or both (Pull + Reduce + Push). The total number of peer visits is typically `world_size` or `world_size - 1`. Each peer visit is sequential — the WG finishes with one peer before moving to the next.

The schedule is a layout: `(pid, peer_visit) → link`. Contention is: "how many `pid`s map to the same `link` at the same `peer_visit`?"

```
┌──────────────────────────────────────────────────────────────────────┐
│  CONTENTION LAYOUT: (pid, i) → link                                  │
│                                                                      │
│  For contention, tile_id doesn't matter — only (pid, i) determines   │
│  which link is active. The outer tile loop affects data volume only.  │
│                                                                      │
│       peer_visit →   0     1     2     3     4     5     6       │
│  pid ↓                                                               │
│    0                     L₃    L₃    L₃    L₃    L₃    L₃    L₃     │  ← ring: constant
│    1                     L₃    L₃    L₃    L₃    L₃    L₃    L₃     │
│   ...                                                                │
│    31                    L₃    L₃    L₃    L₃    L₃    L₃    L₃     │
│                                                                      │
│  wgs_on_link(L₃, any i) = 32 = COMM_SMS                             │
│                                                                      │
│       peer_visit →   0     1     2     3     4     5     6       │
│  pid ↓                                                               │
│    0                     L₁    L₂    L₃    L₄    L₅    L₆    L₇     │  ← two_shot:
│    1                     L₂    L₃    L₄    L₅    L₆    L₇    L₁     │    pid offsets start
│    2                     L₃    L₄    L₅    L₆    L₇    L₁    L₂     │
│   ...                                                                │
│    31                    L₄    L₅    L₆    L₇    L₁    L₂    L₃     │
│                                                                      │
│  wgs_on_link(any L, any i) ≈ COMM_SMS / (N-1)                       │
└──────────────────────────────────────────────────────────────────────┘
```

```python
@dataclass
class ScheduleEntry:
    link_id: int                # physical xGMI link
    peer_rank: int              # remote GPU
    direction: str              # "push" or "pull"
    work_graph: List[Op]        # primitives for this peer visit

class CollectiveLayout:
    """
    A collective = a layout mapping (pid, peer_visit) → link + work_graph.
    Closed-form for all standard algorithms.

    The outer tile loop (persistent kernel) determines data volume
    but doesn't affect the link layout or contention.
    """

    def link_of(self, pid: int, peer_visit: int,
                topology: Topology, my_rank: int) -> ScheduleEntry:
        """What link does this pid use at this peer visit? O(1)."""
        raise NotImplementedError

    def wgs_on_link(self, link_id: int, peer_visit: int, num_wgs: int) -> int:
        """How many pids map to this link at this peer visit? O(1)."""
        raise NotImplementedError

    def active_links(self, peer_visit: int, num_wgs: int) -> dict[int, int]:
        """All active links and WG counts at this peer visit. O(1)."""
        raise NotImplementedError

    @property
    def num_peer_visits(self) -> int:
        """Number of peers visited in the inner loop (usually world_size or world_size-1)."""
        raise NotImplementedError

    @property
    def total_tiles(self) -> int:
        """Total tiles in the outer persistent loop."""
        raise NotImplementedError
```

### Layouts from Iris Kernels

The following layouts are derived directly from the Iris CCL kernel source (`iris/ccl/triton/`). Each kernel's inner loop maps `(pid, i)` → link in a specific pattern.

**Pattern 1: All-to-same-link** (ring allreduce, one_shot, spinlock, atomic)

Every `pid` iterates over the same peers in the same order. `link_of` is independent of `pid`.

Iris kernels: `one_shot`, `spinlock`, `atomic` allreduce; `persistent` allgather; `all_to_all`.

```python
# Kernel loop:
for tile_id in range(pid, total_tiles, COMM_SMS):
    for i in range(world_size):
        target = rank_start + i * rank_stride
        iris.store/load(..., target, ...)   # same target for all pids at same i

def link_of(pid, peer_visit, topology, my_rank, N):
    return topology.link_to((my_rank + i + 1) % N)   # pid unused

def wgs_on_link(link, peer_visit, COMM_SMS):
    return COMM_SMS  # all WGs on same link

def active_links(peer_visit, COMM_SMS):
    return {link_of(0, i, ...): COMM_SMS}  # 1 link active
```

```
       peer_visit →   0     1     2     3     4     5     6
  pid ↓
    0        L₁    L₂    L₃    L₄    L₅    L₆    L₇
    1        L₁    L₂    L₃    L₄    L₅    L₆    L₇
   ...       L₁    L₂    L₃    L₄    L₅    L₆    L₇
    31       L₁    L₂    L₃    L₄    L₅    L₆    L₇

  Every pid hits the same link at the same i.
  wgs_on_link = COMM_SMS at every (i, link).
  concurrent_links = 1 per peer visit.
```

**Pattern 2: pid-staggered start** (two_shot allreduce, two_shot reduce_scatter)

`pid % world_size` offsets which peer is read first. Different pids start at different links, distributing fabric pressure.

Iris kernels: `two_shot` allreduce, `two_shot` reduce_scatter.

```python
# Kernel loop:
for tile_offset in range(pid, max_tile_offset, COMM_SMS):
    start_rank_idx = pid % world_size             # <-- pid determines start peer
    acc = iris.load(..., start_rank_global, ...)
    for i in tl.static_range(1, world_size):
        remote_rank_idx = (start_rank_idx + i) % world_size
        acc += iris.load(..., remote_rank, ...)

def link_of(pid, peer_visit, topology, my_rank, N):
    start = pid % N
    peer_idx = (start + i) % N
    peer = (my_rank + peer_idx + 1) % N  # skip self
    return topology.link_to(peer)

def wgs_on_link(link, peer_visit, COMM_SMS, N):
    return COMM_SMS // (N - 1)  # uniform: pids distribute across N-1 links

def active_links(peer_visit, COMM_SMS, N):
    per_link = COMM_SMS // (N - 1)
    return {topology.link_to(peer): per_link for peer in all_peers}
```

```
       peer_visit →   0     1     2     3     4     5     6
  pid ↓      (start_rank_idx = pid % 8)
    0        L₁    L₂    L₃    L₄    L₅    L₆    L₇
    1        L₂    L₃    L₄    L₅    L₆    L₇    L₁
    2        L₃    L₄    L₅    L₆    L₇    L₁    L₂
    3        L₄    L₅    L₆    L₇    L₁    L₂    L₃
   ...
    31       L₁    L₂    L₃    L₄    L₅    L₆    L₇

  At any column i: pids spread across all 7 links.
  wgs_on_link ≈ COMM_SMS / 7 per link.
  concurrent_links = 7.
  bw_per_wg = min(link_bw, egress / 7) / (COMM_SMS / 7)
```

**Pattern 3: pid-partitioned** (partitioned allgather)

No inner loop at all. `pid // pids_per_rank` permanently assigns each WG to one link.

Iris kernel: `partitioned` allgather.

```python
# Kernel loop:
pids_per_rank = COMM_SMS // world_size
dest_rank_idx = pid // pids_per_rank   # <-- pid determines link (no inner loop)
for tile_id in range(pid % pids_per_rank, total_tiles, pids_per_rank):
    iris.store(..., dest_rank, ...)

def link_of(pid, peer_visit, topology, my_rank, N):
    # i is always 0 (no inner loop)
    dest = pid // (COMM_SMS // N)
    return topology.link_to((my_rank + dest + 1) % N)

def wgs_on_link(link, peer_visit, COMM_SMS, N):
    return COMM_SMS // N  # static partition

def active_links(peer_visit, COMM_SMS, N):
    per_link = COMM_SMS // N
    return {topology.link_to(peer): per_link for peer in all_peers}
```

```
       peer_visit →   0  (only 1 peer visit)
  pid ↓
    0-3      L₁   ← pids 0-3 assigned to GPU1
    4-7      L₂   ← pids 4-7 assigned to GPU2
    8-11     L₃
   ...
    28-31    L₇

  No inner loop. Each pid is permanently on one link.
  wgs_on_link = COMM_SMS / N = 4 (for 32 WGs, 8 GPUs).
  concurrent_links = N-1 = 7.
  bw_per_wg = min(link_bw, egress / 7) / 4
```

**Pattern 4: Ring with fixed next_rank** (ring allreduce)

All hops go to the same `next_rank`. The inner loop is `world_size - 1` steps, each doing push+receive on the same link. Multiple concurrent rings possible (`NUM_RINGS`).

Iris kernel: `ring` allreduce.

```python
# Kernel loop:
ring_id = pid % NUM_RINGS
cta_in_ring = pid // NUM_RINGS
next_rank = (group_rank + 1) % world_size    # fixed for all steps
for tile_index in range(cta_in_ring, tiles_per_ring, ctas_per_ring):
    for _step in range(world_size - 1):
        iris.store(ring_buffer, send_data, ..., next_rank, ...)   # push to next
        recv = tl.load(ring_buffer, ...)                          # receive from prev

def link_of(pid, peer_visit, topology, my_rank, N, NUM_RINGS):
    # i (_step) doesn't change the link — always next_rank
    next_rank = (my_rank + 1) % N
    return topology.link_to(next_rank)   # constant across pid and i

def wgs_on_link(link, peer_visit, COMM_SMS, NUM_RINGS):
    return COMM_SMS // NUM_RINGS  # CTAs per ring, all on same link

def active_links(peer_visit, COMM_SMS, NUM_RINGS):
    # Each ring uses the same link (next_rank)
    return {link_to_next: COMM_SMS // NUM_RINGS}
    # If NUM_RINGS > 1, each ring could use a different link
    # (but in Iris, all rings use the same next_rank)
```

```
       i (_step) →  0     1     2     3     4     5     6
  pid ↓
    0 (ring 0)     L₁    L₁    L₁    L₁    L₁    L₁    L₁
    1 (ring 1)     L₁    L₁    L₁    L₁    L₁    L₁    L₁
   ...
    31             L₁    L₁    L₁    L₁    L₁    L₁    L₁

  Constant link L₁ (to next_rank) across all pid and i.
  wgs_on_link = COMM_SMS (or COMM_SMS/NUM_RINGS per ring).
  concurrent_links = 1.
```

### Summary: Four Layout Patterns

| Pattern | `link_of(pid, i)` | `wgs_on_link` | `concurrent_links` | Iris kernels |
|---------|-------------------|---------------|---------------------|-------------|
| All-to-same | `f(i)` only | COMM_SMS | 1 | one_shot, spinlock, atomic AR; persistent AG; A2A |
| pid-staggered | `(pid%N + i) % N` | COMM_SMS / (N-1) | N-1 | two_shot AR, two_shot RS |
| pid-partitioned | `pid // (SMS/N)` | COMM_SMS / N | N-1 | partitioned AG |
| Ring (fixed link) | constant | COMM_SMS | 1 | ring AR |

All four are closed-form — `wgs_on_link` is computable in O(1) from `(COMM_SMS, N, NUM_RINGS)` without enumerating pids.

### From Layout to WLC Cost

```python
def compute_collective_latency(layout, problem, hardware, comm_hw, config):
    """Total collective latency from the layout."""

    T_total = comm_hw.launch_overhead

    for pv in range(layout.num_peer_visits):
        # Closed-form contention at this peer visit
        link_wg_counts = layout.active_links(pv, config.comm_sms)
        concurrent_links = len(link_wg_counts)

        # Each link group's WLT cost (WGs on same link finish together)
        T_links = []
        for link_id, wgs_on_link in link_wg_counts.items():
            entry = layout.link_of(0, pv, topology, my_rank)  # work graph
            bw_per_wg = min(comm_hw.link_bw,
                           comm_hw.gpu_egress_ceiling / concurrent_links) / wgs_on_link
            T_links.append(compute_wlt_latency(
                entry.work_graph, problem, hardware, comm_hw, config, bw_per_wg))

        T_total += max(T_links)  # slowest link group this peer visit

    # Multiply by outer tile loop iterations (persistent kernel)
    tiles_per_wg = ceil(layout.total_tiles / config.comm_sms)
    T_total *= tiles_per_wg

    return T_total
```

### Example: Ring AllGather, 1 MB, 8 GPUs, 32 WGs, dwordx16

```
Layout: Pattern 1 (all-to-same) with ring work graph
  num_peer_visits = N - 1 = 7 (one visit per peer in the ring)
  link_of(any pid, any i) = L₃ (constant: always pull from prev)
  wgs_on_link = 32, concurrent_links = 1

Per peer visit:
  bw_per_wg = min(49.1 GiB/s, 49.2 / 1) / 32 = 1.53 GB/s
  tile_bytes per WG per iter = 1 MB / 8 / 32 = 4 KB → 64 cache lines
  T_wlt ≈ 64 × 64 B / 1.53 GB/s ≈ 2.67 µs

  Aggregate: 32 WGs × 1.53 GB/s = 49.1 GB/s (full link) ✓

T_collective = 7 × 2.67 µs + T_launch + 7 × T_sync ≈ 18.7 µs + overhead
```

---

## WG Work Graph: Composable Primitives

The work graph defines **which primitives** execute within each WLC. Each primitive's `resolve()` populates the `FunctionalUnitWork` at every hierarchy level, and the WLC cost equation takes the max.

This follows the Iris API pattern (load/store/put/get/atomic) and enables modeling arbitrary collectives — including fused and custom patterns — without new cost model code.

### FunctionalUnitWork: The Full Data Path

Every memory operation traverses the cache hierarchy. The `FunctionalUnitWork` struct tracks cache lines at **every level** a request touches, plus CU-side instruction counts:

```python
@dataclass
class FunctionalUnitWork:
    """Cache-line traffic at each level of the hierarchy, per iteration."""

    # CU-level: instruction counts (depends on load width)
    vmem_read_instrs: int = 0       # VMEM read instructions issued
    vmem_write_instrs: int = 0      # VMEM write instructions issued

    # TCP/vL1D: first level cache (32 KB per CU)
    tcp_read_cl: int = 0            # cache lines read through TCP
    tcp_write_cl: int = 0           # cache lines written through TCP

    # L2/TCC: per-XCD cache (4 MB MI300X, 8 MB MI350X)
    l2_read_cl: int = 0             # cache lines read from L2
    l2_write_cl: int = 0            # cache lines written to L2

    # MALL/Infinity Cache: device-level LLC (256 MB MI300X)
    mall_read_cl: int = 0           # cache lines read from MALL
    mall_write_cl: int = 0          # cache lines written/evicted through MALL

    # HBM: main memory
    hbm_read_cl: int = 0            # cache lines read from HBM
    hbm_write_cl: int = 0           # cache lines written to HBM

    # xGMI: inter-GPU fabric
    xgmi_read_cl: int = 0           # cache lines read from remote GPU (ingress)
    xgmi_write_cl: int = 0          # cache lines written to remote GPU (egress)

    # ALU
    valu_ops: int = 0               # VALU instructions (for reductions)

    # Synchronization
    atomic_count: int = 0           # atomic operations (signal + wait)
```

### Data Path Per Primitive

Each primitive's `resolve()` traces the full path a cache line takes through the hierarchy:

```python
@dataclass
class Load:
    """Read from local HBM into registers."""
    source: MemorySpace         # local_hbm, lds

    def resolve(self, problem, hardware, config, iter_ctx):
        """
        Data path: VMEM instr → TCP (likely miss, streaming) → L2 (likely miss)
                   → MALL (hit or miss) → HBM (if MALL miss)

        For streaming comm traffic: TCP miss, L2 miss, MALL hit rate depends
        on working set vs 256 MB capacity. Conservatively assume MALL miss
        for large transfers (data > MALL capacity).
        """
        cl = iter_ctx.cl_per_iter
        mall_hit_rate = estimate_mall_hit_rate(problem, hardware, config)

        return FunctionalUnitWork(
            vmem_read_instrs = cl * iter_ctx.instrs_per_cl,
            tcp_read_cl      = cl,          # every cl passes through TCP
            l2_read_cl       = cl,          # TCP miss → L2 (streaming, no reuse)
            mall_read_cl     = cl,          # L2 miss → MALL
            hbm_read_cl      = int(cl * (1 - mall_hit_rate)),  # MALL miss → HBM
        )

@dataclass
class Store:
    """Write from registers to local HBM."""
    dest: MemorySpace
    cache_modifier: str         # ".cg", ".wt" (write-through), ".cs" (streaming)

    def resolve(self, problem, hardware, config, iter_ctx):
        """
        Data path depends on cache modifier:
          .cg: VMEM → TCP → L2 (write-allocate) → evict through MALL → HBM
          .wt: VMEM → TCP → bypass L2 → HBM directly (used for cross-XCD visibility)
          .cs: VMEM → TCP → L2 (streaming, low-priority eviction) → MALL → HBM
        """
        cl = iter_ctx.cl_per_iter
        bypass_l2 = (self.cache_modifier == ".wt")

        return FunctionalUnitWork(
            vmem_write_instrs = cl * iter_ctx.instrs_per_cl,
            tcp_write_cl      = cl,
            l2_write_cl       = 0 if bypass_l2 else cl,
            mall_write_cl     = cl,         # all writes eventually reach MALL/HBM
            hbm_write_cl      = cl,         # all writes reach HBM
        )

@dataclass
class Pull:
    """Read from a remote GPU's HBM via xGMI (ingress)."""
    peer: int

    def resolve(self, problem, hardware, config, iter_ctx):
        """
        Data path: VMEM instr → TCP → L2 (miss, not in local cache)
                   → DF crossbar → xGMI read from peer → peer's HBM
                   → data arrives at local L2 → TCP → registers

        xGMI read traffic: cl cache lines traverse the fabric inbound.
        L2 is populated on the read path (unlike Push which originates locally).
        MALL is NOT in the read path for remote data — xGMI reads bypass MALL,
        going directly through DF to the remote GPU's memory controller.
        """
        cl = iter_ctx.cl_per_iter

        return FunctionalUnitWork(
            vmem_read_instrs = cl * iter_ctx.instrs_per_cl,
            tcp_read_cl      = cl,          # data arrives into TCP
            l2_read_cl       = cl,          # populated in L2 on arrival
            mall_read_cl     = 0,           # xGMI bypasses local MALL
            hbm_read_cl      = 0,           # not reading local HBM
            xgmi_read_cl     = cl,          # cl cache lines from remote GPU
        )

@dataclass
class Push:
    """Write to a remote GPU's HBM via xGMI (egress)."""
    peer: int

    def resolve(self, problem, hardware, config, iter_ctx):
        """
        Data path: VMEM read instr → TCP → L2 (read source data, likely miss)
                   → MALL (hit or miss) → HBM (if miss) → got data in registers
                   → DF crossbar → xGMI write to peer → peer's HBM

        Push = local read + remote write. The local read path is the same as Load.
        The xGMI write path adds egress traffic.
        """
        cl = iter_ctx.cl_per_iter
        mall_hit_rate = estimate_mall_hit_rate(problem, hardware, config)

        return FunctionalUnitWork(
            vmem_read_instrs = cl * iter_ctx.instrs_per_cl,  # read source locally
            tcp_read_cl      = cl,          # source data through TCP
            l2_read_cl       = cl,          # TCP miss → L2
            mall_read_cl     = cl,          # L2 miss → MALL
            hbm_read_cl      = int(cl * (1 - mall_hit_rate)),  # MALL miss → HBM
            xgmi_write_cl    = cl,          # write all cl to remote via xGMI
        )

@dataclass
class Reduce:
    """Element-wise reduction on data already in registers."""
    op: ReduceOp                # sum, max, min, prod

    def resolve(self, problem, hardware, config, iter_ctx):
        """Pure ALU — no memory hierarchy traffic."""
        return FunctionalUnitWork(
            valu_ops=iter_ctx.elements_per_iter,
        )

@dataclass
class Signal:
    """Notify a peer that data is ready (atomic write to remote flag)."""
    peer: int
    scope: Scope                # xcd, gpu, system
    semantic: str               # "release"

    def resolve(self, problem, hardware, config, iter_ctx):
        """1 atomic store → 1 cache line through xGMI (flag + padding)."""
        return FunctionalUnitWork(
            atomic_count=1,
            xgmi_write_cl=1,        # flag write to remote, padded to 1 cache line
        )

@dataclass
class Wait:
    """Spin-wait for a peer's signal (atomic read poll)."""
    peer: int
    scope: Scope
    semantic: str               # "acquire"

    def resolve(self, problem, hardware, config, iter_ctx):
        """Atomic read poll — L2 read (fine-grained memory) or xGMI read."""
        return FunctionalUnitWork(
            atomic_count=1,
            l2_read_cl=1,           # poll from L2 (fine-grained pool, coherent)
        )
```

### The Atom: Workgroup-Link-Chunk

The smallest unit of work is one WG transferring one tile over one specific xGMI link (or local HBM path). This granularity matters because:

1. **Link contention**: multiple WGs sharing a link split its bandwidth; different links may have different loads
2. **Egress ceiling sharing**: when multiple links are active simultaneously, egress bandwidth divides across them
3. **Direction asymmetry**: peer-READ degrades local MALL bandwidth 30-43× harder than peer-WRITE (from measured data)

The actual loop structure in many collectives is:

```
for step in collective_steps:           # outer: algorithm steps (ring=N-1, etc.)
  for link in active_links_this_step:    # inner: which xGMI links are active
    for iter in tile_iterations:          # innermost: cache-line iterations through registers
      transfer cl_per_iter cache lines over this link
```

The iteration context captures this three-level structure:

```python
@dataclass
class IterationContext:
    """Derived per-iteration parameters — the workgroup-link-tile atom."""

    # Which link this iteration targets
    link_id: int                # which xGMI link (0..6 for 8-GPU)
    link_direction: str         # "push" (egress) or "pull" (ingress)
    peer_rank: int              # remote GPU rank

    # How many links are active simultaneously this step
    concurrent_links: int       # links sharing the egress ceiling this step
    wgs_sharing_link: int       # WGs sharing this specific link

    # Per-iteration data volume
    cl_per_iter: int            # cache lines processed per loop iteration
    elements_per_iter: int      # elements per iteration (for reduce ops)
    instrs_per_cl: int          # VMEM instructions per cache line (64 / load_width)
    num_iters: int              # total iterations = ceil(N_cl_per_link / cl_per_iter)

    @staticmethod
    def from_config(problem, hardware, comm_hw, config, step_desc):
        load_width = config.load_width
        instrs_per_cl = 64 // load_width

        # Register budget determines cl_per_iter
        vgprs_for_data = config.vgprs_per_iter
        bytes_per_iter = vgprs_for_data * 4
        cl_per_iter = bytes_per_iter // 64
        elements_per_iter = bytes_per_iter // dtype_bytes(problem.dtype)

        # Total cache lines this WG must move over this link
        tile_bytes = problem.message_bytes // (problem.num_gpus * config.num_channels)
        N_cl_total = ceil(tile_bytes / 64)
        N_cl_per_link = N_cl_total // step_desc.links_per_step  # if work split across links
        num_iters = ceil(N_cl_per_link / cl_per_iter)

        # Link bandwidth available to this WG
        # Egress ceiling shared across concurrent links, then across WGs per link
        egress_per_link = comm_hw.gpu_egress_ceiling / step_desc.concurrent_links
        bw_per_wg = min(comm_hw.link_bw, egress_per_link) / step_desc.wgs_per_link

        return IterationContext(
            link_id=step_desc.link_id,
            link_direction=step_desc.direction,
            peer_rank=step_desc.peer,
            concurrent_links=step_desc.concurrent_links,
            wgs_sharing_link=step_desc.wgs_per_link,
            cl_per_iter=cl_per_iter,
            elements_per_iter=elements_per_iter,
            instrs_per_cl=instrs_per_cl,
            num_iters=num_iters,
        )
```

### Step Description

Each step of a collective describes which links are active and how work distributes:

```python
@dataclass
class StepDesc:
    """Describes one step of a collective algorithm — which links, what direction."""
    link_id: int                # specific link being used
    peer: int                   # remote GPU rank
    direction: str              # "push" or "pull"
    concurrent_links: int       # how many links active simultaneously this step
    links_per_step: int         # links this WG operates across (usually 1)
    wgs_per_link: int           # WGs sharing this link
```

For ring AllReduce: each step uses 1 link, `concurrent_links=1`, direction alternates by phase.
For direct AllToAll: each step uses 1 link but different steps target different peers.
For tree algorithms: `concurrent_links` may be > 1 when fan-out > 1.

### WG Latency: Software-Pipelined Loop

Like Origami's `L_tile = prologue + (num_iters × L_iter) + epilogue`, the WG's total latency is a software-pipelined loop:

```
T_wg = T_prologue + (num_iters - 1) × T_steady_state + T_epilogue + T_sync
```

Where:

```
T_prologue    = T_first_load                (fill the pipeline — first iteration is memory-only)

T_steady_state = max(T_vmem, T_xgmi, T_hbm, T_l2, T_valu)
                 ↑ per-iteration, all functional units overlap
                 ↑ the slowest unit determines the iteration time
                 ↑ (exactly like Origami's max(L_compute, L_mem) per K-iter)

T_epilogue    = T_last_store                (drain the pipeline — last iteration is write-only)

T_sync        = atomic_count × atomic_latency  (signal/wait at step boundaries)
```

Each term is computed by resolving the work graph's primitives for one iteration:

```python
def compute_wg_latency(work_graph: List[Op], problem, hardware, config,
                       iter_ctx: IterationContext,
                       num_wgs_per_link: int) -> WGLatencyBreakdown:
    """
    Compute T_wg from a work graph using software-pipelined loop model.

    Analogous to Origami's compute_tile_latency():
      L_tile = prologue + (K_iters - 1) × max(L_compute, L_mem) + epilogue
    """

    # Resolve each primitive for one iteration
    iter_work = FunctionalUnitWork.zero()
    sync_work = FunctionalUnitWork.zero()
    for op in work_graph:
        w = op.resolve(problem, hardware, config, iter_ctx)
        if isinstance(op, (Signal, Wait)):
            sync_work += w   # sync ops happen once per step, not per iteration
        else:
            iter_work += w   # data movement ops happen every iteration

    # Per-iteration time at each level of the hierarchy (all overlap — take the max)
    #
    # CU level: instruction issue
    T_vmem = iter_work.vmem_total_instrs / hardware.vmem_issue_rate_per_cu

    # TCP/vL1D level: 32 KB, ~5 wide loads before stall
    T_tcp  = (iter_work.tcp_total_cl * 64) / hardware.tcp_bw_per_cu

    # L2/TCC level: per-XCD, shared across CUs
    T_l2   = (iter_work.l2_total_cl * 64) / hardware.l2_bw_per_cu

    # MALL level: device-wide LLC, 304 ns/hit
    T_mall = (iter_work.mall_total_cl * 64) / hardware.mall_bw

    # HBM level: main memory
    T_hbm_r = (iter_work.hbm_read_cl * 64) / hardware.hbm_read_bw_per_cu
    T_hbm_w = (iter_work.hbm_write_cl * 64) / hardware.hbm_write_bw_per_cu

    # xGMI level: inter-GPU fabric
    T_xgmi_r = (iter_work.xgmi_read_cl * 64) / (comm_hw.link_bw / num_wgs_per_link)
    T_xgmi_w = (iter_work.xgmi_write_cl * 64) / (comm_hw.link_bw / num_wgs_per_link)

    # VALU: reduction arithmetic
    T_valu = iter_work.valu_ops / hardware.valu_rate_per_cu

    T_steady_state = max(T_vmem, T_tcp, T_l2, T_mall,
                         T_hbm_r, T_hbm_w, T_xgmi_r, T_xgmi_w, T_valu)

    # Prologue: first iteration loads (memory only, pipeline not yet full)
    T_prologue = max(T_hbm_r, T_xgmi_r, T_mall)  # whichever read path is slowest

    # Epilogue: last iteration stores/pushes (drain the pipeline)
    T_epilogue = max(T_hbm_w, T_xgmi_w)

    # Sync: once per step (not per iteration)
    T_sync = sync_work.atomic_count * config.comm_hardware.atomic_latency_ns

    # Total WG latency (software-pipelined loop)
    T_wg = (T_prologue
            + max(iter_ctx.num_iters - 1, 0) * T_steady_state
            + T_epilogue
            + T_sync)

    bottleneck = argmax_name(
        T_vmem, T_tcp, T_l2, T_mall,
        T_hbm_r, T_hbm_w, T_xgmi_r, T_xgmi_w, T_valu
    )

    return WGLatencyBreakdown(
        T_wg=T_wg,
        T_steady_state=T_steady_state,
        T_prologue=T_prologue,
        T_epilogue=T_epilogue,
        T_sync=T_sync,
        num_iters=iter_ctx.num_iters,
        # Per-iter breakdown at every hierarchy level (for roofline visualization):
        T_vmem=T_vmem,
        T_tcp=T_tcp,
        T_l2=T_l2,
        T_mall=T_mall,
        T_hbm_read=T_hbm_r,
        T_hbm_write=T_hbm_w,
        T_xgmi_read=T_xgmi_r,
        T_xgmi_write=T_xgmi_w,
        T_valu=T_valu,
        bottleneck=bottleneck,
    )
```

### Composing Collectives from Primitives

Each collective is a per-step work graph. Primitives take `(problem, hardware, config)` — the resolution to cache lines and instructions happens inside `resolve()`:

```python
def allgather_ring_step(peer_prev, peer_next):
    """AllGather ring: wait, pull from prev, store locally, signal next."""
    return [
        Wait(peer_prev, scope="gpu", semantic="acquire"),
        Pull(peer_prev),
        Store(dest="local_hbm", cache_modifier=".cs"),
        Signal(peer_next, scope="gpu", semantic="release"),
    ]

def reduce_scatter_ring_step(peer_prev, peer_next):
    """ReduceScatter ring: load local, wait, pull from prev, reduce, store, signal."""
    return [
        Load(source="local_hbm"),
        Wait(peer_prev, scope="gpu", semantic="acquire"),
        Pull(peer_prev),
        Reduce(op="sum"),
        Store(dest="local_hbm", cache_modifier=".cs"),
        Signal(peer_next, scope="gpu", semantic="release"),
    ]

def allreduce_ring(peer_prev, peer_next, phase):
    """AllReduce = ReduceScatter phase + AllGather phase."""
    if phase == "reduce_scatter":
        return reduce_scatter_ring_step(peer_prev, peer_next)
    else:
        return allgather_ring_step(peer_prev, peer_next)

def alltoall_direct_step(peer):
    """AllToAll direct: load local, push to peer."""
    return [
        Load(source="local_hbm"),
        Push(peer),
    ]
```

The `N_cl_total` (total cache lines per step) comes from `problem` and `config`:
```
N_cl_total = ceil(tile_bytes / 64)
tile_bytes = problem.message_bytes / (problem.num_gpus × config.num_channels)
```

And `iter_ctx = IterationContext.from_config(problem, hardware, config, N_cl_total)` determines how many iterations the loop runs and how much work per iteration.

### Why This Works

This is structurally identical to how Origami models a GEMM tile:

| GEMM Tile (Origami) | Comm WG Timestep (this model) |
|---------------------|-------------------------------|
| K-iteration: one MT_K slice | Loop iteration: cl_per_iter cache lines |
| Prologue: first load (memory only) | Prologue: first load/pull (memory only) |
| Steady state: max(L_compute, L_mem) | Steady state: max(T_vmem, T_tcp, T_l2, T_mall, T_hbm_r, T_hbm_w, T_xgmi_r, T_xgmi_w, T_valu) |
| L_mem = max(L_l2, L_mall, L_dram) | Same hierarchy: every cache level has a throughput ceiling |
| Epilogue: store C | Epilogue: final store/push |
| N_MI × L_MI per iter | N_cl_per_iter × instrs_per_cl per iter |
| num_k_iters = ceil(K / MT_K) | num_iters = ceil(N_cl_total / cl_per_iter) |
| L_tile = pro + (iters-1)×SS + epi | T_wg = pro + (iters-1)×SS + epi + sync |
| config_t determines tile shape | config determines load_width, vgprs_per_iter |
| problem_t determines total work | problem determines N_cl_total |

The work graph makes the model **collective-agnostic** — any sequence of primitives can be analyzed:
- Standard collectives are just named compositions
- Fused patterns (GEMM→ReduceScatter within a kernel) compose GEMM tile ops + comm ops
- Custom patterns (Iris's WG specialization) are directly expressible
- Extending config (beyond CU params) just adds new fields that `resolve()` can read

---

## The Model: Workgroup Level → Device Level

### Cache Line as the Unit of Transfer

Above the CU level, all data movement is denominated in **cache lines**, not raw bytes. On MI300X/MI350X, a cache line is **64 bytes**. If a workgroup requests fewer than 64 bytes, the hardware pads the request to a full cache line. This has two consequences:

1. **Effective data volume** at L2/MALL/HBM is always `ceil(requested_bytes / 64) × 64`
2. **Load width selection matters** — a `global_load_dwordx4` (16 B) wastes 75% of the cache line it fetches; a `global_load_dwordx16` (64 B) uses the full cache line. Wider loads are strictly better for streaming communication traffic.

```
Cache Line Padding Effect:

  Requested:  ┌──16B──┐
  Transferred: ┌──────────────64B──────────────┐    ← 4× amplification (dwordx4)

  Requested:  ┌──────────────64B──────────────┐
  Transferred: ┌──────────────64B──────────────┐    ← 1× (dwordx16, optimal)
```

Throughout the model below, `N_cachelines` = `ceil(data_bytes / CACHELINE_SIZE)` is the fundamental unit of work above the CU.

### Level 1: The Communication Workgroup (Atomic Unit)

A single communication workgroup performs one tile transfer within one step of a collective. Its cost is determined by the **slowest parallel functional unit** for the work it must do:

```
T_wg = max(T_vmem_read, T_lds, T_xgmi_write, T_sync)
```

Each functional unit operates in parallel. The WG's latency is the slowest:

| Functional Unit | Work per WG | Time | Notes |
|----------------|-------------|------|-------|
| VMEM read | `N_cl` cache lines from local HBM | `N_cl / vmem_read_rate` | Rate depends on load width: dwordx4 vs dwordx16 |
| LDS | Buffer/stage (protocol-dependent) | `N_cl × 64 / lds_bw` | Only for protocols requiring staging |
| xGMI write | `N_cl` cache lines to remote GPU | `N_cl × 64 / xgmi_write_bw_per_wg` | Share of link BW |
| Sync/atomic | Protocol flags, barriers | `T_sync` | Per-WG overhead |

Where:
- `N_cl = ceil(tile_bytes_per_wg / 64)` — cache lines this WG must move
- `tile_bytes_per_wg = tile_bytes / num_wgs_per_channel`
- `vmem_read_rate` — depends on load width:
  - `global_load_dwordx16` (64 B): 1 cache line per load, optimal
  - `global_load_dwordx4` (16 B): 4 loads per cache line, 4× more VMEM instructions
  - `global_load_dword` (4 B): 16 loads per cache line, 16× more instructions
- `xgmi_write_bw_per_wg = link_bw / wgs_sharing_this_link`

The workgroup's data path (cache-line-denominated):

```
┌──────────────────────────────────────────────────────────────────────┐
│ Communication Workgroup (1 CU)                                       │
│                                                                      │
│  ┌──────────────┐   ┌──────────┐   ┌───────────┐   ┌─────────────┐  │
│  │ VMEM Read     │──>│ vL1D/TCP │──>│ Data      │──>│ xGMI Link   │  │
│  │ N_cl loads    │   │ (32 KB)  │   │ Fabric    │   │             │  │
│  │               │   │ streams  │   │ Crossbar  │   │ N_cl × 64 B │  │
│  │ Width matters:│   │ through  │   │           │   │ to remote   │  │
│  │ dwordx16 best│   │ (no reuse│   │           │   │             │  │
│  └──────────────┘   └──────────┘   └───────────┘   └──────┬──────┘  │
│                                                            │          │
│  Per-WG resource consumption:                              ▼          │
│  • 1 CU (waves resident)                          ┌──────────────┐   │
│  • N_cl VMEM read instructions                    │ Remote GPU   │   │
│  • N_cl cache lines through TCP (streaming)       │ HBM Write    │   │
│  • N_cl cache lines through L2 (miss to HBM)     │ (N_cl × 64B) │   │
│  • N_cl × 64 bytes xGMI egress                   └──────────────┘   │
│  • DF arbiter slots (proportional to N_cl)                            │
└──────────────────────────────────────────────────────────────────────┘
```

### Level 2: XCD Composition (cache lines through shared L2)

Multiple comm WGs on the same XCD generate aggregate cache-line traffic through the shared L2:

```
┌───────────────────────────────────────────────────────────────────┐
│ XCD (38 CUs, 4 MB L2 = 65,536 cache lines)                       │
│                                                                   │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      ┌──────────┐  │
│  │Comm WG │ │Comm WG │ │Comm WG │ │Comm WG │ ...  │ GEMM WGs │  │
│  │N_cl ea │ │N_cl ea │ │N_cl ea │ │N_cl ea │      │(tile data)│  │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘      └────┬─────┘  │
│      │          │          │          │                  │         │
│      └──────────┴──────────┴──────────┴──────────────────┘         │
│                             │                                      │
│                    ┌────────▼─────────┐                             │
│                    │    L2 (TCC)      │                             │
│                    │   4 MB capacity  │                             │
│                    │                  │  Comm WGs: N_cl_total       │
│                    │  Throughput:     │  cache lines streaming      │
│                    │  L2 read BW     │  through (no reuse,         │
│                    │  (cachelines/ns)│  pollutes GEMM tile data)   │
│                    └────────┬────────┘                             │
│                             │                                      │
│          cache lines that miss L2 → MALL/HBM                      │
│                             │                                      │
│                    ┌────────▼────────┐                             │
│                    │   DF Port       │  N_cl_total × 64 B          │
│                    │   to fabric     │  aggregate egress           │
│                    └─────────────────┘  (request-proportional)     │
└───────────────────────────────────────────────────────────────────┘
```

At XCD level, the throughput ceilings are in **cache lines per unit time**:

| Resource | Ceiling | Unit |
|----------|---------|------|
| L2 read bandwidth | L2_read_BW / 64 | cachelines/ns per XCD |
| L2 capacity | 65,536 | cachelines (4 MB / 64 B) |
| DF port bandwidth | DF_port_BW / 64 | cachelines/ns per XCD |

Comm WGs stream through L2 without reuse — every cache line is a miss that passes through to MALL/HBM. This streaming traffic competes with GEMM tile data for L2 capacity and DF port bandwidth.

**Aggregate cache lines from this XCD per step:**
```
N_cl_xcd = N_comm_cus_on_xcd × N_cl_per_wg
T_xcd = N_cl_xcd × 64 / min(L2_read_BW, DF_port_BW, link_BW_share)
```

### Level 3: Device Composition (cache lines through HBM/MALL → egress)

All 8 XCDs' cache-line traffic aggregates toward HBM and the per-GPU xGMI egress:

```
┌──────────────────────────────────────────────────────────────────────┐
│ GPU (304 CUs across 8 XCDs)                                         │
│                                                                      │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  ...  ┌──────┐      │
│  │XCD 0 │ │XCD 1 │ │XCD 2 │ │XCD 3 │ │XCD 4 │       │XCD 7 │      │
│  │N_cl  │ │N_cl  │ │N_cl  │ │N_cl  │ │N_cl  │       │N_cl  │      │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘       └──┬───┘      │
│     │        │        │        │        │               │           │
│     └────────┴────────┴────────┴────────┴───────────────┘           │
│                              │                                       │
│               N_cl_total = Σ(N_cl_xcd) cache lines                  │
│                              │                                       │
│            ┌─────────────────▼──────────────────┐                    │
│            │          HBM Read                   │                    │
│            │   4.73 TB/s = 73.9M cachelines/ms  │                    │
│            │   (measured peak, MI300X)           │                    │
│            └─────────────────┬──────────────────┘                    │
│                              │                                       │
│            ┌─────────────────▼──────────────────┐                    │
│            │       MALL (Infinity Cache)         │                    │
│            │   256 MB = 4M cache lines capacity │                    │
│            │   304 ns per hit                    │                    │
│            └─────────────────┬──────────────────┘                    │
│                              │                                       │
│            ┌─────────────────▼──────────────────┐                    │
│            │       Per-GPU xGMI Egress           │                    │
│            │   49.2 GB/s = 769K cachelines/ms   │  ← hard cap       │
│            │   (independent of fan-out)          │                    │
│            └─────────────────┬──────────────────┘                    │
│                              │                                       │
│              ┌───────────────┼───────────────┐                       │
│              ▼               ▼               ▼                       │
│         ┌────────┐     ┌────────┐     ┌────────┐                    │
│         │Link →G1│     │Link →G2│     │Link →G7│                    │
│         │49 GiB/s│     │49 GiB/s│     │49 GiB/s│                    │
│         └────────┘     └────────┘     └────────┘                    │
└──────────────────────────────────────────────────────────────────────┘
```

**Device-level model in cache lines:**

```
N_cl_total = ceil(tile_bytes / 64)              ← cache lines per step

T_hbm_read  = N_cl_total / HBM_read_rate_cl     ← HBM read (cachelines/ns)
T_mall      = N_cl_total / MALL_rate_cl          ← MALL throughput (if resident)
T_egress    = N_cl_total × 64 / egress_ceiling   ← xGMI egress cap
T_link      = N_cl_total × 64 / link_bw          ← per-link bandwidth

T_step_transfer = max(T_hbm_read, T_egress, T_link)   ← slowest pipe
```

The xGMI egress ceiling (49.2 GB/s = 769K cachelines/ms) is typically the binding constraint. HBM read (73.9M cachelines/ms) has ~96× headroom over egress for communication traffic alone. But under concurrent GEMM, HBM headroom shrinks — see overlap model.

**Load width impact on cache-line efficiency:**

```
                    Useful bytes    Cache lines    Amplification
                    ──────────      ───────────    ─────────────
dwordx16 (64 B)     64 B/load       1 cl/load      1.0× (optimal)
dwordx4  (16 B)     16 B/load       1 cl/load      4.0× (¾ wasted)
dword    (4 B)       4 B/load       1 cl/load     16.0× (15/16 wasted)

For a 1 MB tile (16,384 cache lines):
  dwordx16: 16,384 VMEM instructions
  dwordx4:  65,536 VMEM instructions  ← 4× more CU work, same cache-line traffic
  dword:   262,144 VMEM instructions  ← 16× more CU work
```

The cache-line traffic through L2/MALL/HBM is identical regardless of load width — but narrower loads require more VMEM instructions per CU, which can make the CU's functional units (not the memory hierarchy) the bottleneck for small load widths.

### Level 4: Collective Algorithm (Multi-GPU)

The collective algorithm determines the step structure:

```
Ring AllReduce on 8 GPUs:
════════════════════════

Phase 1: ReduceScatter (7 steps)          Phase 2: AllGather (7 steps)
─────────────────────────────             ────────────────────────────
Step 1: G0→G1→G2→...→G7→G0               Step 8:  Same ring, gather
Step 2: rotate tiles                      Step 9:  ...
...                                       ...
Step 7: final reduce tile                 Step 14: all tiles gathered

Total: 14 steps
Per-step data: msg_bytes / 8 / num_channels
Per-step BW: 49.1 GiB/s × protocol_efficiency
```

---

## The Complete Latency Equation

### Device-Level (single GPU's view of the collective)

```
T_comm = T_launch + num_steps × T_step
```

Where:

```
T_step = max(T_transfer, T_sync)

T_transfer = tile_bytes / BW_effective

tile_bytes = message_bytes / (N × num_channels)     [for ring-based collectives]

BW_effective = min(link_bw × active_links_per_step, gpu_egress_ceiling) × protocol_efficiency

T_sync = step_sync_latency                            [per-step protocol overhead]

T_launch = launch_overhead                            [kernel dispatch cost]
```

### Algorithm-Specific Parameters

| Collective | num_steps | tile_bytes | active_links_per_step | Notes |
|-----------|-----------|-------------|----------------------|-------|
| AllReduce (ring) | 2(N-1) | msg / N / nch | 1 | ReduceScatter + AllGather |
| AllGather (ring) | N-1 | msg / N / nch | 1 | Each step: one tile around ring |
| ReduceScatter (ring) | N-1 | msg / N / nch | 1 | Each step: reduce + forward |
| AllToAll (direct) | N-1 | msg / N / nch | 1 | Each step: exchange with one peer |

Where `N = num_gpus`, `nch = num_channels`.

### Protocol Efficiency

| Protocol | Payload Fraction | Sync Mechanism | Message Size Regime |
|----------|-----------------|----------------|-------------------|
| Simple | ~1.0 | Step-based memory fences | > ~64 KB |
| LL128 | 120/128 = 0.9375 | 128B-aligned atomic flags | ~1 KB – ~64 KB |
| LL | 4/8 = 0.5 | Per-element 4B flag | < ~1 KB |

### CU Saturation (Derived, Not Hard-Coded)

CU saturation is not a per-collective constant — it falls out of the WLC model. Each WG can sustain a certain xGMI throughput determined by its work graph (how much VMEM, VALU, and sync work competes with xGMI transfers). The saturation point is where adding more WGs no longer increases aggregate link throughput:

```
bw_per_wg = 1 / T_wlt_per_byte
          = rate at which one WG can push/pull data through xGMI
          (limited by the slowest non-xGMI functional unit in the work graph)

wgs_to_saturate_link = ceil(link_bw / bw_per_wg)

cu_saturation = wgs_to_saturate_link × concurrent_links
```

Different work graphs → different `bw_per_wg` → different saturation points:

```
AllGather step:   [Wait, Pull, Store, Signal]
  Non-xGMI work per WG: Wait + Store (fast)
  → high bw_per_wg → fewer WGs needed → lower saturation

AllReduce step:   [Load, Wait, Pull, Reduce, Store, Signal]
  Non-xGMI work per WG: Load + Wait + Reduce + Store (more work)
  → lower bw_per_wg → more WGs needed → higher saturation
```

The measured saturation points (AllGather ~32, AllReduce ~60, AllToAll ~64) are validation targets, not inputs. If the model correctly predicts `bw_per_wg` from the work graph, the saturation points should emerge.

---

## Function Hierarchy

Mirrors Origami's `compute_total_latency` → `compute_tile_latency` → `compute_*_latency` pattern:

```
compute_comm_latency(comm_problem, hardware, comm_hardware, comm_config)
│
├── comm_context_t ctx(comm_problem, hardware, comm_hardware, comm_config)
│   ├── compute_num_steps(collective, algorithm, num_gpus)
│   ├── compute_tile_bytes(message_bytes, num_gpus, num_steps, num_channels)
│   ├── compute_active_links(algorithm, topology, step)
│   ├── compute_effective_bw(link_bw, active_links, egress_ceiling, protocol)
│   ├── compute_bw_per_wg(work_graph, hardware)  → derives saturation from work graph
│   └── compute_hbm_bw_consumed(tile_bytes, num_steps, effective_bw)
│
├── compute_launch_latency(hardware, comm_config)
│   └── kernel dispatch overhead from hardware constants
│
├── compute_step_latency(comm_problem, hardware, comm_hardware, ctx)
│   ├── compute_transfer_latency(ctx)
│   │   └── tile_bytes / effective_bw_per_step
│   └── compute_sync_latency(comm_hardware, comm_config)
│       └── protocol-dependent per-step overhead
│
└── total: T_launch + num_steps × T_step
```

### Top-Level API

```cpp
namespace origami::comm {

    // Primary: predict latency for a standalone collective
    comm_prediction_result_t predict_latency(
        const comm_problem_t& problem,
        const hardware_t& hardware,
        const comm_hardware_t& comm_hw,
        const comm_config_t& config);

    // Select best config (sweep channels, algorithm, protocol)
    comm_prediction_result_t select_config(
        const comm_problem_t& problem,
        const hardware_t& hardware,
        const comm_hardware_t& comm_hw,
        const std::vector<comm_config_t>& candidates);

    // Rank all candidates
    std::vector<comm_prediction_result_t> rank_configs(
        const comm_problem_t& problem,
        const hardware_t& hardware,
        const comm_hardware_t& comm_hw,
        const std::vector<comm_config_t>& candidates);

    // Overlap: predict concurrent comm+gemm latency
    overlap_result_t predict_overlap(
        const problem_t& gemm_problem,
        const comm_problem_t& comm_problem,
        const hardware_t& hardware,
        const comm_hardware_t& comm_hw,
        const config_t& gemm_config,
        const comm_config_t& comm_config);
}
```

---

## Overlap Model

When GEMM and communication run concurrently, resources are shared:

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Resources                          │
│                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐       │
│  │     GEMM Kernel      │    │    Comm Kernel        │       │
│  │  N_CU_gemm CUs       │    │  N_CU_comm CUs       │       │
│  │                      │    │                      │       │
│  │  Consumes:           │    │  Consumes:           │       │
│  │  • CUs (compute)     │    │  • CUs (< 64)       │       │
│  │  • HBM BW (tiles)    │    │  • HBM BW (tiles)   │       │
│  │  • L2 (tile data)    │    │  • L2 (streaming)   │       │
│  │  • MALL (spill)      │    │  • xGMI link BW     │       │
│  └──────────┬───────────┘    └──────────┬───────────┘       │
│             │                           │                    │
│             └───────────┬───────────────┘                    │
│                         ▼                                    │
│              ┌──────────────────┐                            │
│              │  HBM: 5.3 TB/s  │ ← shared, not partitioned  │
│              │  MALL: 256 MB   │ ← shared                   │
│              │  DF: proportional│ ← no CU-based isolation    │
│              └──────────────────┘                            │
│                                                              │
│  T_overlap = max(T_gemm(N_CU_gemm), T_comm(N_CU_comm))     │
│                                                              │
│  Sweep CU split: k ∈ {8, 16, 32, 64}                       │
│  Select k minimizing T_overlap                              │
└─────────────────────────────────────────────────────────────┘
```

### Interference Terms

```
BW_hbm_gemm = BW_hbm_total - BW_hbm_comm
```

Where `BW_hbm_comm` is derived from the communication model: comm WGs read source data from HBM and the xGMI fabric writes to remote HBM. The HBM read bandwidth consumed by communication is:

```
BW_hbm_comm = message_bytes × traffic_amplification / T_comm
```

This reduces the effective HBM bandwidth available to GEMM, affecting Origami's memory latency calculation via the `mem_bw_per_wg_coefficients` polynomial evaluated at reduced effective bandwidth.

---

## Architectural Parameters (MI300X Ground Truth)

Every parameter in the model traces to a measured architectural constant:

| Parameter | Value | Source | Used In |
|-----------|-------|--------|---------|
| `link_bw` | 49.1 GiB/s | Measured (< 2% cross-cluster variance) | `BW_effective` |
| `gpu_egress_ceiling` | 49.2 GB/s | Measured (N=1..7 fan-out invariant) | `BW_effective` cap |
| `wire_overhead` | 1.23× | Measured (framing, ECC, flit) | Derate `link_bw` to payload |
| `step_sync_simple` | ~5.4 µs | Measured (K-series, chunk-setup overhead) | `T_sync` |
| `launch_overhead` | 1.5–2.0 µs | Measured (AQL dispatch) | `T_launch` |
| *(CU saturation)* | *(derived)* | *(emerges from WLC model: ceil(link_bw / bw_per_wg))* | *validation target, not input* |
| `protocol_eff_simple` | ~1.0 | Protocol spec | `BW_effective` scale |
| `protocol_eff_ll128` | 0.9375 | Protocol spec (120/128) | `BW_effective` scale |
| `protocol_eff_ll` | 0.5 | Protocol spec (4/8) | `BW_effective` scale |
| `fabric_ceiling` | 334 GB/s | Measured (8-GPU aggregate) | Multi-GPU cap |

---

## Validation Integration

### Primary Dataset

`~/comm_data/comm_only/rccl_master_sweep.csv` — 33,690 rows

Schema:
```
nchannels, world_size, msg_bytes, primitive, bandwidth_gbps,
latency_us, busbw_gbps, algorithm, hostname, timestamp
```

Coverage: AllReduce, AllGather, ReduceScatter, AllToAll, Broadcast
GPU counts: 2, 4, 8 | Channels: 1–128 | Sizes: 1 KB – 1 GiB

### Validation Procedure

For each row in the master sweep:

```python
predicted = compute_comm_latency(
    comm_problem_t(collective, msg_bytes, world_size, bf16),
    hardware_mi300x,
    comm_hardware_mi300x,
    comm_config_t(ring, simple, nchannels, 304)
)

error_pct = (predicted.latency_ns/1000 - latency_us) / latency_us × 100
```

### Validation Metrics

1. **Accuracy**: median absolute percentage error (MdAPE) across all rows
2. **Regime coverage**: separate accuracy for latency-bound (< 1 MB) vs bandwidth-bound (> 10 MB) regimes
3. **Bottleneck correctness**: does the predicted bottleneck match observed scaling behavior?

### Secondary Datasets

| Dataset | Rows | Validates |
|---------|------|-----------|
| `rccl_channel_sweep.csv` | 5,760 | Channel scaling (does model predict optimal nch?) |
| `rccl_cu_occupancy.csv` | 1,009 | Validates derived CU saturation (does model predict correct knee?) |
| `rccl_reduce_scatter.csv` | 825 | RS-specific behavior, 0 B – 4 GiB range |

---

## Bottleneck Regimes

The model predicts which architectural resource limits performance:

```
                    Latency vs Message Size (log-log)
                    
    T_comm │
    (µs)   │
           │  ┌─────────────────┐
           │  │ Latency-bound   │   Protocol overhead dominates
           │  │ (flat region)   │   T ≈ T_launch + num_steps × T_sync
           │  └────────┬────────┘
           │           │
           │           │  ┌────────────────────┐
           │           │  │ Transition         │   Both contribute
           │           │  └────────┬───────────┘
           │           │           │
           │           │           │  ┌──────────────────────┐
           │           │           │  │ Bandwidth-bound      │
           │           │           │  │ (linear region)      │
           │           │           │  │ T ≈ msg / BW_eff     │
           │           │           │  └──────────────────────┘
           │
           └──────────────────────────────────────────────── msg_bytes
                   1KB      64KB      1MB       64MB      1GB
```

The knee point (transition from latency-bound to bandwidth-bound) is:

```
msg_knee = T_sync × BW_effective × N × num_channels
```

Below the knee, optimizing protocol overhead matters. Above, link bandwidth is all that matters.
