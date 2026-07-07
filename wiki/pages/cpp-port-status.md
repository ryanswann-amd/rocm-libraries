# C++ Port Status: `origami::comm` (as built)

As-built record of the `origami_comms_cpp/` workspace — Stage 1 complete, byte-identical with Python, dashboard wired through, awaiting upstream integration into `rocm-libraries/shared/origami/`.

## TL;DR

| | status | evidence |
|---|---|---|
| C++ port (M0–M9) | ✅ done | 11/11 CTest tests pass in <20 s |
| Byte-identity with Python | ✅ proven | **34,599 / 34,599** production sweep cells IEEE-754-bit-identical |
| pybind11 bindings | ✅ done | importable as `origami_comm`, full `Hardware`/`CommHardware` display surface |
| Dashboard integration | ✅ done | correlation pages route through `origami_comm` with Python fallback |
| Production speedup | ✅ measured | **150× predict_row inner loop, 8.5× full dashboard predict_all** |
| Upstream into `rocm-libraries` | ⏳ next session | scoped in [[cpp-port-plan]] but deferred |

## Repository

Standalone workspace at `origami_comms_cpp/` (sibling of `model/`), git-initialized, header-only library:

```
origami_comms_cpp/
├── include/origami/comm/      # the entire cost model, header-only
│   ├── types.hpp              # DataType, TileShape, CommProblem, CommConfig, FU work
│   ├── hardware.hpp           # Hardware, CommHardware + MI300X / MI300X_COMM constants
│   ├── heuristics.hpp         # Heuristics + DEFAULT_HEURISTICS
│   ├── primitives.hpp         # Load, Store, Pull, Push, Signal, Wait, Reduce, Op variant
│   ├── layouts.hpp            # CollectiveLayout base + 9 concrete layouts (see "Design deltas")
│   ├── latency.hpp            # compute_iter_times, compute_wg_tile_latency
│   ├── collective.hpp         # compute_collective_latency, predict_row
│   ├── tensor.hpp             # predict_tensor_collective + TensorCollectivePrediction
│   └── origami_comm.hpp       # umbrella include
├── bindings/origami_comm_py.cpp  # pybind11 module
├── tests/                     # 11 CTest targets (8 C++, 2 Python regression, 1 sweep)
├── tests/golden/              # CSVs dumped from Python; ~10,500 grid cells
├── scripts/                   # dump_golden.py, validate.py, bench_predict_all.py
└── CMakeLists.txt             # C++20, optional ORIGAMI_COMM_PYTHON
```

## Milestone summary

| # | Scope | Tests | Status |
|---|---|---|---|
| M0 | workspace scaffold + git init | n/a | ✓ |
| M1 | (collapsed into M2 — workspace is standalone, no upstream audit yet) | — | ✓ |
| M2 | types + hardware + heuristics | `test_types`, `test_hardware`, `test_heuristics` (38 assertions) | ✓ |
| M3 | primitives + `resolve_work_graph` | `test_primitives`, `test_byte_identity` (50) | ✓ |
| M4 | 9 collective layouts | `test_layouts` (5,936-row grid) | ✓ |
| M5 | latency engine | `test_latency` (1,590-row grid) | ✓ |
| M6 | collective engine + `predict_row` | `test_collective` (990-row grid) | ✓ |
| M7 | tensor frontend + framework overhead | `test_tensor` (1,920-row grid) | ✓ |
| M8 | pybind11 bindings | `test_bindings` (14,081 bit-identity assertions) | ✓ |
| M9 | full production-sweep regression | `test_full_sweep` (34,599 bit-identity assertions) | ✓ |

Aggregate: ~48,680 bit-identity checks across the whole stack, full ctest run < 20 s.

## Design deltas vs the plan

The original [[cpp-port-plan]] (drafted 2026-06-04) had decisions on a few axes that didn't survive contact with the actual port. Recording them here so the eventual upstream PR carries the right rationale:

### 1. Layouts: `CollectiveLayout` base class, not templates + `constexpr`

**Plan said:** layouts are standalone structs with `constexpr` methods, latency/collective engines are function templates parameterized on the layout — so the layout-specific math constant-folds.

**As-built:** virtual polymorphic base `CollectiveLayout` with derived classes for each of the 9 layouts (`AllToSameLayout`, `PidStaggeredLayout`, `PidPartitionedLayout`, `RingFixedLayout`, `RingAllGatherLayout`, `RingReduceScatterLayout`, `TwoShotAllReduceLayout`, `RingAllReduceLayout`, `RingBroadcastLayout`). Per-step work graphs carried via `std::function`-shaped closures.

**Why:** the Python layouts build their per-step `work_graph`s with closures capturing `(rank, peer, step_index, N)` at construction time. `constexpr`-ing that across all 9 layouts hit two walls: (a) `std::vector<Op>` of variants isn't `constexpr` in C++20, (b) the work-graph builder closures are intrinsically dynamic. A virtual `link_of(...)` returning a `ScheduleEntry` with a `WorkGraphFn` closure recovers the closure-capturing pattern exactly, at the cost of one indirection per timestep — which is amortized away by `compute_collective_latency`'s outer loops.

**Consequence for upstream:** the runtime is one virtual call per timestep, not zero. Across the production sweep this is invisible (the Python loop is 150× more expensive than the C++ kernel either way), but a templated rewrite is still possible for a hot inner path. Not pursued because bit-identity, not raw speed, was the M0–M9 contract.

### 2. Headers-only

**Plan said:** split into `include/` headers + `src/comm/*.cpp` translation units.

**As-built:** header-only `INTERFACE` library. Every type, every constant, every function lives under `include/origami/comm/`.

**Why:** the cost model is small enough (~2,000 lines total) that header-only is the simplest CMake story — `target_link_libraries(... origami::comm)` is the entire integration surface for downstream consumers. The dashboard pybind11 module compiles in 8 seconds from a clean state. No `.so` to ship except the bindings.

### 3. Heuristics: `std::array<double, N>` enum-indexed, not `std::unordered_map`

**Plan said:** `std::unordered_map<std::string, double>` for framework / primitive heuristics — "frameworks are user-extensible."

**As-built:** fixed-size `std::array<double, 5>` for primitive-keyed heuristics (`ring_step_overhead_cycles`, `xgmi_write_concentration_k_by_primitive`) and `std::array<double, 6>` for framework overhead, indexed by `Primitive` and `Framework` enums. String-keyed lookup is a fallback that scans the name table.

**Why:** the actual heuristic set is fixed in the model (5 primitives, 6 frameworks); `unordered_map` was over-engineered. The enum/array form is `constexpr`, cache-friendly, and the string overload preserves the Python-side API exactly.

### 4. Public `predict_row` is a single function taking `std::string_view`, no per-layout templates

**Plan said:** templated `predict_row<Layout>` with a runtime dispatcher in front. The dispatcher would be the only string-keyed entry point.

**As-built:** non-template `predict_row(std::string_view primitive, …)`. Internally calls `compute_collective_latency`, which dispatches on the layout's virtual methods.

**Why:** dropped together with delta #1. Once layouts are virtual, the case for templating the engine evaporates.

### 5. Module name is `origami_comm`, not `origami_comms`

**Plan said:** `origami_comms` Python module.

**As-built:** `origami_comm` (singular). The C++ library lives under `origami::comm`; matching the namespace seemed cleaner than the original repo name.

## Validation evidence

### Bit-identity (the M9 contract)

| Source | Rows | Bit-identical | Mismatches |
|---|---:|---:|---:|
| `data/rccl_master_sweep.csv` | 33,690 | 33,690 / 33,690 | 0 |
| `data/tensor_shapes_sweep.csv` | 909 | 909 / 909 | 0 |
| Synthetic predict_row grid (M8) | 630 | 630 / 630 | 0 |
| Synthetic tensor grid (M8) | 1,920 | 1,920 / 1,920 | 0 |
| Layouts grid (M4) | 5,936 | 5,936 / 5,936 | 0 |
| Latency engine grid (M5) | 1,590 | 1,590 / 1,590 | 0 |
| Collective engine grid (M6) | 990 | 990 / 990 | 0 |

KPIs computed from C++ predictions vs from Python predictions match to the last decimal — asserted explicitly by `scripts/validate.py`.

### Correlation against measurements

Run by `scripts/validate.py` (which is also the answer to "does the model still correlate well?"):

```text
predict_row sweep (RCCL benchmarks, 33,690 rows)
  ALL primitives    MdAPE= 20.7%  MeanAPE= 25.5%
                    ≤20%= 48.8%   ≤50%= 89.8%
                    bias=-13.9%   log-log r=0.985
  AG  MdAPE=15.9%   AR  MdAPE=23.7%
  RS  MdAPE=14.1%   BC  MdAPE=22.9%   A2A MdAPE=30.3%

predict_tensor_collective sweep (torch.distributed, 909 rows)
  ALL primitives    MdAPE= 53.8%  log-log r=0.915  bias=-53.2%
```

Same numbers as Python (necessarily — they're the same predictions). The byte-level model is in good shape; the tensor-collective ~53% under-prediction is the documented `framework_overhead_us["torch"] = 400 µs` calibration gap (see [[ag-outlier-investigation]] log entry — same caveat about cluster-to-cluster differences).

### Speedup (dashboard predict-all sweep)

Benchmarked by `scripts/bench_predict_all.py`:

| Variant | Python | C++ | Speedup |
|---|---:|---:|---:|
| Isolated `predict_row` inner loop (1,320 calls) | 675 ms (511 µs/call) | 4.5 ms (3.4 µs/call) | **150×** |
| Full dashboard `predict_all()` (with pandas) | 795 ms | 93 ms | **8.5×** |

The inner-loop figure is the headroom; the full-path figure is bounded by `pd.DataFrame.iterrows` + per-row dict construction + final `df.merge`. Vectorizing `predict_all` would close most of the remaining gap.

## Dashboard integration

`dashboard/_origami_backend.py` is the single re-export point. Pages now import:

```python
from _origami_backend import (
    predict_row, predict_tensor_collective,
    MI300X, MI300X_COMM, DataType,
    BACKEND_NAME, BACKEND_KIND, backend_caption,
)
```

The helper prefers `origami_comm` (C++) and falls back to `model/*` (Python) transparently. The selected backend is displayed in the sidebar. Set `ORIGAMI_FORCE_PY_BACKEND=1` to force the Python reference for A/B verification.

**Pages swapped:** `1_Correlation.py`, `2_Tensor_Correlation.py`.

**Page left on Python:** `app.py` (home). It constructs custom `Hardware` instances via slider knobs and calls into `compute_collective_latency`, `compute_wg_tile_latency`, `resolve_work_graph`, `compute_iter_times` directly. Exposing all of that via pybind11 was out of scope for the dashboard-acceleration milestone — the home page is the modeling sandbox, not the prediction frontend.

## Build & test recipe

```bash
cd origami_comms_cpp
cmake -B build
cmake --build build -j
cd build && ctest --output-on-failure
```

CTest targets:

```text
test_types        test_hardware       test_heuristics    test_primitives
test_byte_identity test_layouts       test_latency       test_collective
test_tensor       test_bindings(Py)   test_full_sweep(Py, ~17 s)
```

Python tests require `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` when run from conda (conda Pythons ship an older libstdc++ than the system toolchain's GLIBCXX_3.4.31). The CMake target sets this for `ctest` automatically.

## Known limitations

1. **libstdc++ ABI mismatch with conda Python.** Documented and worked around via `LD_PRELOAD`. Once we have an installable wheel (scikit-build-core) this disappears.
2. **No Python-callable `Hardware` / `CommHardware` constructors.** Read access only. Adding them is one block of pybind11 — held back because the home page (`app.py`) is the only consumer and it's still on the Python backend.
3. **No internal-engine bindings.** `compute_iter_times`, `compute_wg_tile_latency`, `resolve_work_graph`, and the layout classes themselves are not exposed via pybind11; only the two public `predict_*` functions are. Stage 1's premise was "predict the same number"; deeper introspection is upstream-integration work.

## What's next: upstream into `rocm-libraries/shared/origami/`

This is the integration session. Order of operations (drafted, not yet executed):

1. **Audit upstream.** Read `rocm-libraries/shared/origami/include/origami/{hardware,types,heuristics}.hpp` and reconcile field-by-field with `origami_comms_cpp/include/origami/comm/*.hpp`. The Origami `hardware_t` has fields like `mem1/2/3_perf_ratio`, `mem_bw_per_wg_coefficients`, `compute_clock_ghz` that overlap conceptually with ours but with different names — the PR has to decide field-by-field whether to extend, rename, or hold separately under `comm::Hardware`.
2. **Decide on shared types.** Pick which of `Hardware`, `TileShape`, `FunctionalUnitWork`, `Heuristics` get hoisted to the `origami::` root vs stay under `origami::comm::`. The plan recommended sharing as much as possible; the audit determines feasibility.
3. **Reconcile CMake.** Origami builds `roc::origami` and has nanobind bindings, not pybind11. Decide whether to keep our header-only INTERFACE library inside the existing CMakeLists or stand up a `shared/origami/comm/CMakeLists.txt` that the parent includes.
4. **Port the bindings.** Re-implement the pybind11 module under Origami's nanobind setup (or get a pybind11-friendly carve-out alongside it). The bit-identity test suite goes with it.
5. **PR draft.** One self-contained PR introducing `shared/origami/include/origami/comm/`, the test suite, the binding, and a README pointing back to this status page.

Open questions to settle before PR:

- **Layout polymorphism reconciliation.** Upstream Origami's GEMM is template-heavy; do we templatize layouts on the way in, accepting a partial rewrite? Or accept the as-built virtual interface for now and revisit once the model gets a hot inner-loop user?
- **Heuristics ownership.** Origami already has `heuristic_params_t` for GEMM. Does `comm::Heuristics` live alongside, merge into a shared `heuristics_t`, or stay as a sibling? Merging is cleaner long-term; keeping separate is simpler for the first PR.
- **Bindings strategy.** Switch to nanobind to match Origami's existing Python surface, or keep pybind11 and live with two binding generators in the same repo?

These are intentionally deferred; the next chat session will pick them up against the actual upstream tree.

## Cross-references

- [[cpp-port-plan]] — the plan; this page is the audit against it.
- [[origami]] — upstream's GEMM model (the target integration site).
- [[ag-outlier-investigation]] — the source of the calibrated heuristics the port preserves.
- [[day2-atom-validation]] — the Triton microbenchmark that gated the port.
- [[cost-model-extension]] — design rationale for the comm model overall.
