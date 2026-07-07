# C++ Port Plan: `origami::comm`

Plan for porting the Python communication cost model into Origami's C++ codebase, alongside the existing GEMM model, sharing infrastructure under one `origami::` namespace.

> **Status (2026-06-04): plan executed.** M0–M9 are complete in the standalone workspace at `origami_comms_cpp/`; the port is byte-identical with Python across 34,599 production-sweep cells and the dashboard now routes through it (8.5× faster `predict_all`). The as-built record, including design deltas from this plan and the open questions for upstream integration, is at [[cpp-port-status]]. Stage 2 (upstream into `rocm-libraries/shared/origami/`) has not started.

## Goal

Stand up a C++ implementation of the comm cost model at byte-identical fidelity to the current Python model, integrated into `rocm-libraries/shared/origami/` as a sibling of `origami::gemm`. The port carries the recent unit refactor (cycles bottom-up, frequency-based conversion) and the calibrated heuristics ([[cost-model-extension]]) directly into Origami's vocabulary.

## Non-goals (deferred to later stages)

- **DAG / producer-consumer scheduler**. No multi-stage critical-path solver.
- **`Compute` primitive bridging into `origami::gemm`**. No cross-model orchestration.
- **Workgroup-graph integration**. The two models exist side-by-side; they don't yet compose.
- **Triton-Distributed kernel integration**. The port produces predictions; consuming them in fused-kernel codegen is downstream.

These are explicitly Stage 2+. Stage 1 = comm model lives in Origami alongside GEMM, callable in parallel.

## Pre-port spike (3 days)

The model has ~30% MdAPE byte-level overall but a 41% outlier on `all_gather`. A short de-risk pass before committing to the port:

### Day 1 — Investigate `all_gather` 41% outlier

The AG correlation is materially worse than AR/RS at ~25% MdAPE. Hypothesis bench:
1. AG uses `RingAllGatherLayout` (single-ring, all channels on same link) per [[rccl]]'s direct-copy semantics. Verify `num_timesteps`, `chunks_per_timestep`, and the per-WG xGMI write cap. The latency model currently uses the read-side MSHR cap for xGMI writes with a comment flagging this as conservative; AG is the most write-bound collective.
2. Compare predicted vs measured curves for AG specifically across the size sweep.
3. Decision: a one-line fix → apply; structural mismatch → defer port until layout is rewritten.

### Day 2 — Per-WG atom validation (Triton microbench)

Build one Triton kernel realizing the work_graph `[Pull, Store]` looped, with `s_memrealtime` brackets per iter. Compare:
- Measured median per-WG total cycles vs `compute_wg_tile_latency().T_total_cycles`
- Measured per-segment cycles vs `T_xgmi_read_cycles`, `T_hbm_write_cycles`

Resolves whether the 30% MdAPE comes from atom-level calibration (fix constants) or collective-level orchestration (fix layout/timestep math). High information, ~150 LOC of Triton + Python.

### Day 3 — Go/no-go decision

- **Both checks clean** → port with confidence; current MdAPE is the floor and improvable later via shared `Heuristics`.
- **Atoms misaligned by ≤ 20 %** → port; treat first month after port as constant-recalibration phase.
- **AG fix is structural OR atoms misaligned > 30 %** → defer port; fix Python first.

## Architecture: target directory layout

Mirror Origami's existing structure under `rocm-libraries/shared/origami/`:

```
shared/origami/
├── include/origami/
│   ├── hardware.hpp           ← EXTEND (add xGMI fields if missing; non-breaking)
│   ├── types.hpp              ← EXTEND (TileShape.contiguous, FunctionalUnitWork)
│   ├── heuristics.hpp         ← EXTEND (framework_overhead_ns, min_bytes_per_wg)
│   ├── gemm/                  ← unchanged
│   └── comm/                  ← NEW
│       ├── hardware.hpp           — CommHardware + cycle/freq helpers
│       ├── types.hpp              — CommProblem, CommConfig, WgTileLatencyBreakdown
│       ├── primitives.hpp         — Op = std::variant<Load, Store, Pull, Push, Signal, Wait, Reduce>
│       ├── layouts.hpp            — CollectiveLayout interface + 5 RCCL layouts
│       ├── latency.hpp            — compute_iter_times, compute_wg_tile_latency
│       ├── collective.hpp         — compute_collective_latency, predict_row
│       └── tensor_collective.hpp  — predict_tensor_collective + framework overhead
├── src/comm/
│   ├── layouts.cpp                — concrete layout implementations
│   ├── latency.cpp                — FU-time formulas, MSHR cap, polynomial scaling
│   └── collective.cpp             — ring + sequential timestep solvers
├── tests/comm/
│   ├── test_byte_identical.cpp    — 243 cases from Python baseline (regression gate)
│   ├── test_layouts.cpp           — layout structural checks
│   ├── test_latency.cpp           — atom-level checks
│   └── test_heuristics.cpp        — heuristic injection / override
└── python/ (or Origami's pybind11 location)
    └── comm_bindings.cpp          — exposes C++ API to Python (dashboard continues working)
```

## Shared infrastructure (reused, not duplicated)

| Abstraction          | Where it lives             | Stage-1 work |
|----------------------|----------------------------|--------------|
| `Hardware`           | `origami/hardware.hpp`     | Extend with `xgmi_latency_cycles`, possibly more |
| `TileShape`          | `origami/types.hpp`        | Add `contiguous` flag (per the cacheline accounting fix) |
| `FunctionalUnitWork` | `origami/types.hpp`        | New if absent; reused by both gemm and comm |
| `Heuristics`         | `origami/heuristics.hpp`   | Extend with `framework_overhead_ns`, `min_bytes_per_wg` |
| Cycle units + freq   | `origami/hardware.hpp`     | Add `clock_hz()`, `cycles_to_seconds/ns/us`, inverse helpers |
| `CommHardware`       | `origami/comm/hardware.hpp` | **New** — comm-specific (link_bw, atomic_latency, launch_overhead) |

## Milestones

### M1 — Origami source audit + skeleton (1 day)

- Read `rocm-libraries/shared/origami/include/origami/{hardware,types,heuristics}.hpp` and the `gemm/` subtree.
- Write a short audit note: what's already there, what we extend, what's new under `comm/`.
- Draft empty headers for all `comm/` files with module-level docstrings matching Origami's style.
- CMake hookup: `comm/` as a new compile unit; tests under `tests/comm/`.

**Acceptance:** `cmake --build .` succeeds with the new (empty) headers and one `tests/comm/test_smoke.cpp` that includes them all.

### M2 — Hardware + types (1 day)

- `CommHardware` struct with constants for MI300X, byte-identical to Python `MI300X_COMM`.
- `clock_hz()`, `cycles_to_seconds/ns/us`, `seconds_to_cycles/...`, `rate_per_second/ns`, `rate_per_cycle_from_per_*`. Match Python conventions in `model/hardware.py`.
- `TileShape` with `contiguous` flag + `cachelines` / `elements` / `bytes` accessors.
- `CommProblem` (`M, N, num_gpus, dtype, split_dim`) with `gpu_tile_shape`, `message_bytes` properties.
- `CommConfig` (`num_wgs`, `cl_per_iter`, `min_bytes_per_wg`) with `effective_num_wgs(tile_bytes)`.
- `FunctionalUnitWork` struct.
- `WgTileLatencyBreakdown` with `*_cycles` fields + `T_*` accessors that go through frequency.

**Acceptance:** gtest constructing `CommHardware{MI300X}`, calling `cycles_to_us(20'274'088)` → asserts ≈10.137 µs (matches Python `all_gather W=8 nch=1 mb=16M`).

### M3 — Primitives (0.5 day)

- `std::variant<Load, Store, Pull, Push, Signal, Wait, Reduce>` (no `Compute` yet — deferred to Stage 2).
- `resolve()` overloads via `std::visit` that produce `FunctionalUnitWork` per the Python `primitives.py`.
- `Signal` and `Wait` carry `peer` only (no `tile_id`; that's Stage 2).

**Acceptance:** gtest that resolves each primitive on a fixed `(cl_per_iter, instrs_per_cl, elements_per_iter)` and asserts the FU counts match Python `Op.resolve(...)` exactly.

### M4 — Layouts (3 days, largest single piece)

Layouts are **standalone structs with `constexpr` methods**, not derivatives of a common base. Each layout exposes a fixed concept (`Layout`):

```cpp
template <typename Layout>
concept LayoutConcept = requires (int N, int ts, int rank, int nch) {
    { Layout::is_ring }                  -> std::convertible_to<bool>;
    { Layout::num_timesteps(N) }         -> std::convertible_to<int>;
    { Layout::chunks_per_timestep() }    -> std::convertible_to<int>;
    { Layout::link_of(0, ts, rank, N) }  -> std::convertible_to<LinkEntry>;
    { Layout::active_links(ts, nch, N) } -> std::convertible_to<LinkWgMap>;
};
```

Five concrete implementations matching `model/layouts.py`:
- `RingAllReduceLayout` — N-1 RS phase + N-1 AG phase, `is_ring=true`.
- `RingAllGatherLayout` — N-1 steps, single-ring direct-copy, `is_ring=false` (sequential dispatch per Python).
- `RingFixedLayout` — fixed-next-rank ring (reduce_scatter, broadcast), `is_ring=true`.
- `AllToAllLayout` — full bipartite, one timestep per peer, `is_ring=false`.
- `SequentialLayout` — fallback for ad-hoc tests, `is_ring=false`.

`LinkEntry` carries `work_graph` (vector of Ops or `std::span<const Op>`), `is_self`, `link_id`.

**Acceptance:**
- For each layout, gtest enumerates all timesteps × all peers and asserts the `(work_graph, link_id, is_self)` triple matches Python's output byte-for-byte.
- Stress test: parameterized over W ∈ {2, 4, 8} and 5 primitives.
- `concept LayoutConcept` is satisfied by all 5 implementations (compile-time check).

### M5 — Latency engine (2 days)

- `compute_iter_times(work, hw, comm_hw, bw_per_wg, active_cus)` → `std::map<std::string, double>` of cycle counts per FU.
- `compute_wg_tile_latency(work_graph, wg_tile_cachelines, config, hw, comm_hw, bw_per_wg, wg_tile_elements, wg_tile, active_cus)` → `WgTileLatencyBreakdown`.
- HBM polynomial scaling reused from `origami::Hardware` (existing GEMM uses it; comm just calls the same method).
- MSHR cap formula: `(mshr_depth × waves × CL) / xgmi_latency_cycles`.

**Acceptance:** parameterized gtest sweeps `(work_graph, wg_tile, active_cus)` over a sample set; asserts each FU's cycle count matches Python within 1e-9 relative.

### M6 — Collective engine (1 day)

- `template <LayoutConcept Layout> double compute_collective_latency(const CommProblem&, const CommConfig&, const Hardware&, const CommHardware&, int my_rank=0)` returning total cycles.
- Internal split at compile time: `if constexpr (Layout::is_ring)` selects ring vs sequential math — both paths fully inline for each layout.
- `template <LayoutConcept Layout> double predict_row(size_t msg_bytes, int W, int nch, const Hardware&, const CommHardware&)` returning µs via `hw.cycles_to_us`.
- **Runtime dispatcher** (non-template):
  ```cpp
  double predict_row(std::string_view op, size_t msg_bytes, int W, int nch,
                     const Hardware& hw, const CommHardware& chw);
  ```
  Single switch on `op` → calls the appropriate template instantiation. This is the only string-keyed entry point in the API; everything internal is type-resolved.

**Acceptance:**
- gtest runs all 225 baseline cases from `/tmp/baseline_predictions.json` (the Python pre-refactor snapshot).
- Tolerance: 1e-9 µs absolute. Test fails build if any prediction diverges.
- Generated assembly check (sample one instantiation): the per-iter cycle formula should constant-fold the layout-specific values (no virtual-call indirection, no hash-map lookup).

### M7 — Tensor layer + heuristics (1 day)

- `Heuristics` struct: `min_bytes_per_wg`, `framework_overhead_ns` (std::unordered_map<string, double>).
- `DEFAULT_HEURISTICS` constant matching Python defaults exactly.
- `predict_tensor_collective(op, shape, dtype, world_size, ..., framework, heuristics)` returning `TensorCollectivePrediction{predicted_us, backend_us, framework_overhead_us}`.
- Framework overhead remains in **host wall-time ns** (not cycles) — same rationale as Python.

**Acceptance:** gtest replays the 18 tensor-collective baseline cases byte-identical.

### M8 — pybind11 bindings (1.5 days)

- `origami_comms` Python module exposing: `Hardware`, `CommHardware`, `TileShape`, `CommProblem`, `CommConfig`, `predict_row`, `predict_tensor_collective`, `compute_collective_latency`, all layout classes, all primitives.
- Drop-in compatible with current Python API: existing `model/` imports could be replaced with `origami_comms` and tests/dashboard should pass unchanged.

**Acceptance:**
- Existing `tests/` (Python) run against the C++ implementation via `origami_comms` and **all 87 pass**.
- Dashboard renders with no Python errors when imports are flipped to C++.

### M9 — Documentation + landing prep (1 day)

- README under `shared/origami/comm/` summarizing API + linking to this plan.
- Per-header doxygen-style comments.
- A `wiki/pages/cpp-port-status.md` page tracking what's landed.
- If upstreaming (see "Open decisions"): prepare PR draft against `rocm-libraries`.

## Validation strategy

Two layered gates ensure correctness:

1. **Byte-identical regression** (M6, M7) — every prediction the Python model produces today must be reproduced by C++ within float roundoff (1e-9 absolute). This is the contract.
2. **pybind11 round-trip** (M8) — the existing 87-test Python suite executes against C++. Catches integration / API-level regressions.

Both gates are CI-enforceable. The byte-identical test is the linchpin; once it's green and stays green, every later optimization or refactor is bounded.

## Open decisions

### Decision 1 — Upstream vs parallel

- **Option A: Direct PR to `rocm-libraries`.** Lands in Origami immediately; available for production use. Cost: coordinate with maintainers, more rigorous review, longer cycle to first commit.
- **Option B: Build in our workspace mirroring upstream paths.** Faster iteration; we control structure; upstream as one PR when solid.

**Recommendation:** B → A. Build at `/home/ryaswann/origami_comms/origami_cpp/` mirroring `shared/origami/include/origami/comm/`, get all gates green, then submit one PR.

### Decision 2 — When to bring in upstream `origami::Hardware`

The Python `Hardware` struct may not match upstream's exactly. Two options:

- **Option A: Adapt to upstream `Hardware` immediately** — extend with comm-relevant fields via non-breaking additions.
- **Option B: Use a `comm::Hardware` first, reconcile later.**

**Recommendation:** A. The point of "same framework" is shared types. M1's audit determines what fields need adding. Better to do this once during the port than retrofit later.

### Decision 3 — Polymorphism style for layouts

**Decided: templates + tag dispatch.** Every layout is a standalone struct with `constexpr` / compile-time-resolvable methods. The latency and collective engines are function templates parameterized on the layout type. Rationale: the cost-model math is all constants once a layout is fixed; making layouts template parameters lets the compiler inline and constant-fold the whole `compute_collective_latency` path.

```cpp
struct RingAllReduceLayout {
    static constexpr bool is_ring = true;
    static constexpr int  num_timesteps(int N) { return 2 * (N - 1); }
    static constexpr int  chunks_per_timestep() { return 1; }
    static constexpr LinkEntry link_of(int pid, int ts, int rank, int N);
    static constexpr LinkWgMap active_links(int ts, int nch, int N);
    // ...
};

template <typename Layout>
double compute_collective_latency(const CommProblem& p, const CommConfig& c,
                                  const Hardware& hw, const CommHardware& chw);

template <typename Layout>
double predict_row(size_t msg_bytes, int W, int nch,
                   const Hardware& hw, const CommHardware& chw);
```

**Runtime → compile-time boundary at the public API.** Callers that pass a layout by string name (dashboard, pybind11) go through one explicit dispatcher:

```cpp
double predict_row(std::string_view op, size_t msg_bytes, int W, int nch,
                   const Hardware& hw, const CommHardware& chw) {
    if (op == "all_reduce")     return predict_row<RingAllReduceLayout>(msg_bytes, W, nch, hw, chw);
    if (op == "all_gather")     return predict_row<RingAllGatherLayout>(msg_bytes, W, nch, hw, chw);
    if (op == "reduce_scatter") return predict_row<RingFixedLayout>(msg_bytes, W, nch, hw, chw);
    if (op == "broadcast")      return predict_row<RingFixedLayout>(msg_bytes, W, nch, hw, chw);
    if (op == "all_to_all")     return predict_row<AllToAllLayout>(msg_bytes, W, nch, hw, chw);
    throw std::invalid_argument(...);
}
```

The dispatcher is the only runtime cost; everything inside each branch is compile-time-specialized. New layouts require a recompile (acceptable for a model with a stable, finite set of collectives).

### Decision 4 — Heuristics representation

Python uses a dataclass with a `dict[str, float]` for framework overheads. C++ equivalents:
- `std::unordered_map<std::string, double>` — straightforward; runtime lookup.
- Enum-keyed `std::array<double, NumFrameworks>` — compile-time; faster lookup.

**Recommendation:** `unordered_map`. Frameworks are user-extensible (we'll add more as we measure them). Map is more honest.

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| AG 41% issue is structural | Low (more likely a constant) | Medium — refactor port mid-flight | Pre-port spike Day 1 catches it. |
| Atom math is off by > 20% | Low-Medium | Medium — port goes ahead but with constant-recalibration phase after | Pre-port spike Day 2 catches it; doesn't block port. |
| Origami's `Hardware` is incompatible with our cycle convention | Low | Low — accept divergence in `comm::Hardware` | M1 audit identifies; M2 reconciles. |
| pybind11 build complexity in Origami's CMake | Medium | Low — port still useful via C++ tests | M8 can defer if needed; not on critical path for the prediction value. |
| Origami maintainers reject upstream PR due to MdAPE | Low | Medium — keeps port in our workspace longer | Decision 1 says we build in parallel first; only upstream when calibration is tight. |
| Calibration improvements diverge between Python and C++ | Medium | High — silent drift | Mitigated by shared `Heuristics` struct convention and a periodic byte-identical re-validation. |

## Total estimate

**Pre-port spike:** 3 days
**Port (M1–M9):** 11.5 days
**Total:** ~3 weeks elapsed, assuming focused work and the spike comes back green.

Buffer of ~1 week for unforeseen friction (Origami CMake quirks, pybind11 build, layout edge cases) → **~4 weeks to a landed, byte-identical, pybind11-bound C++ port.**

## What this enables (after Stage 1 lands)

- Any caller in `rocm-libraries` can compute collective cost via `origami::comm::predict_row(...)`.
- hipBLASLt kernel-selection logic can read comm cost alongside gemm cost — first step toward [[codesign-rccl-hipblaslt]].
- Calibration improvements (per-WG correlation work, AG fix, sync-cost refinement) live in shared `Heuristics` and propagate to both Python and C++ via byte-identical regression.
- Stage 2 (workgroup graphs, producer-consumer DAG) sits on top: same types, same units, no glue layer. Both halves of the model live in one namespace; composing them is the natural next move.

## Cross-references

- [[cost-model-extension]] — design rationale for the comm model
- [[origami]] — existing C++ GEMM model (the foundation we're joining)
- [[mi300x-architecture]] — source of hardware constants the C++ values must match
- [[rccl]] — collective algorithms the layouts mirror
- [[codesign-rccl-hipblaslt]] — the downstream use case Stage 1 enables
