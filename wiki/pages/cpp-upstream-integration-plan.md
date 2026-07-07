# C++ Upstream Integration Plan: `origami::comm` → `rocm-libraries`

Stage 2 scope — land the as-built `origami_comms_cpp/` header-only model into `rocm-libraries/shared/origami/` as a sibling of the existing `origami::gemm` model, reconciling build system, naming, types, bindings, and tests against the actual upstream tree.

> **Prereq:** Stage 1 complete — see [[cpp-port-status]]. This page is the audit of `origami_comms_cpp/` against the real `shared/origami/` tree (read 2026-06-04 from a local `rocm-libraries` checkout), and supersedes the speculative "Open decisions" in [[cpp-port-plan]] where the audit contradicts them.

## TL;DR

The comm model is a **self-contained, HIP-free, header-only island**; upstream Origami is a **HIP-required, `.hpp`/`.cpp`-split, C++17, nanobind + Catch2 + rocm-cmake** library. The two share *zero* code today and overlap only conceptually (hardware fields). The lowest-risk landing keeps `comm::` types **as a sibling, not merged** into GEMM types, and adapts comm's *packaging* (naming, license headers, clang-format, C++ standard, test framework, bindings) to upstream's conventions. Type unification with `hardware_t` is a deliberate follow-up, not PR1.

## Upstream audit (what we're integrating into)

Source: `shared/origami/` — `CMakeLists.txt`, `include/origami/{hardware,types,heuristics,origami}.hpp`, `python/{CMakeLists.txt,pyproject.toml,src/origami/bindings.cpp}`, `tests/`.

| Axis | Upstream `origami` (GEMM) | Our `origami_comms_cpp` (comm) | Gap |
|---|---|---|---|
| Layout | `include/origami/*.hpp` + `src/origami/*.cpp` | header-only `include/origami/comm/*.hpp` | comm needs no `src/`; fits as INTERFACE headers |
| Targets | `roc::origami` (SHARED/STATIC) + `roc::origami-headers` (INTERFACE) | `origami::comm` (INTERFACE) | add comm headers to `origami-headers` or a new sibling target |
| HIP | **Required** — `<hip/hip_runtime.h>` in `hardware.hpp`, `find_package(hip REQUIRED)`, links `hip::host` | **None** — pure std | keep comm HIP-free; don't let it pull `origami/hardware.hpp` |
| C++ std | `cxx_std_17` | `cxx_std_20` (only dep: `std::span` in `primitives.hpp`) | backport `std::span` → C++17 shim, or bump shared std (affects downstream) |
| Naming | `snake_case_t` (`hardware_t`, `data_type_t`, `config_t`, `problem_t`, `heuristic_params_t`) | PascalCase (`Hardware`, `CommHardware`, `DataType`, `TileShape`, `CommProblem`, `Heuristics`) | mechanical rename across headers + tests + bindings |
| License | full MIT AMD block on every file | single-line `// SPDX-License-Identifier: MIT` | prepend full header block |
| Format | `.clang-format` enforced (2-space) | ad-hoc | run clang-format |
| Bindings | **nanobind** v2.0.0, scikit-build-core wheel, module `origami`, target `_pyorigami` | **pybind11**, module `origami_comm` | port to nanobind under the `origami` package, or ship pybind11 separately |
| Tests | **Catch2** (FetchContent), `origami-tests` target | custom `test_harness.hpp` + asserts, + Python golden-CSV sweeps | convert C++ layers to Catch2; vendor golden data |
| Packaging | rocm-cmake (`rocm_install`, `rocm_export_targets`, `rocm_create_package`, CPack) | plain CMake | hook comm headers into existing rocm-cmake install |

### Type-level reconciliation

- **`hardware_t` vs `comm::Hardware`/`CommHardware`.** Conceptual overlap only: `num_cu`≈`N_CU`, `num_xcd`≈`NUM_XCD`, `clock_ghz`≈`compute_clock_ghz`, `l2_capacity_bytes`≈`L2_capacity`, `mem_bw_coeffs`≈`mem_bw_per_wg_coefficients`. But **parameterized differently**: `hardware_t` derives bandwidths from `mem{1,2,3}_perf_ratio` + a MALL base and is **constructed from `hipDeviceProp_t` at runtime**; `comm::Hardware` carries explicit **bytes/cycle `constexpr` constants** (`hbm_read_bw`, `tcp_bw`, `link_bw`, `xgmi_latency_cycles`). Merging would force comm to inherit HIP + lose `constexpr`, and force `hardware_t` to grow xGMI/SDMA fields it never uses. **Recommendation: keep separate for PR1.**
- **`data_type_t` vs `comm::DataType`.** Upstream `data_type_t` is HIP-free (lives in `types.hpp`, includes only `math.hpp`) and is a strict superset of comm's 6 types. **Reusing it removes a duplicate enum in the same namespace** at the cost of mapping comm's `FP16/BF16/FP32/FP64/FP8/INT8` onto `Half/BFloat16/Float/Double/Float8/Int8`. Low risk; recommended (decision-gated — only touches the dtype path and the byte-identity harness enum labels).
- **`TileShape`, `FunctionalUnitWork`, `CommProblem`, `CommConfig`.** No upstream equivalent → land under `comm/` (renamed `tile_shape_t`, etc.).
- **`Heuristics` vs `heuristic_params_t` + `heuristics_database_t`.** Entirely different shapes: comm uses enum-indexed `std::array` of framework/primitive overheads; GEMM uses a key→params singleton with specificity-ordered lookup. **Recommendation: keep `comm::Heuristics` separate for PR1**; a shared heuristics façade is a later refactor.

### Golden-data provenance (test contract risk)

The Stage-1 byte-identity oracle (`tests/golden/*.csv`, ~10,500 cells) is **dumped from the Python `model/` package**, which does **not exist in rocm-libraries**. On landing, the golden CSVs become the frozen oracle. Vendor a trimmed golden set into `shared/origami/tests/comm/golden/` and keep `scripts/dump_golden.py` in the `origami_comms` workspace as the (external) regeneration tool, referenced in a README.

## Integration decisions

| # | Decision | Recommendation | Rationale |
|---|---|---|---|
| D1 | Merge vs sibling types | **Sibling** (`comm::` types separate from GEMM `_t` types) for PR1 | preserves byte-identity, HIP-free, `constexpr`; unification is a scoped follow-up |
| D2 | C++ standard | **Backport `std::span` → C++17** so comm builds at `cxx_std_17` | bumping the shared standard risks downstream consumers (hipBLASLt, RCCL) pinned to C++17 |
| D3 | dtype | **Reuse `origami::data_type_t`** | kills a duplicate enum in one namespace; low blast radius |
| D4 | Bindings | **nanobind into the `origami` package** (`origami.comm.*`) long-term; **defer to PR2** to keep PR1 reviewable | single wheel/module matches upstream; dashboard shim updates with it |
| D5 | Tests | **Catch2 for C++ layers**, vendor golden CSVs, byte-identity sweep is the gate | matches upstream test infra; the sweep is the non-negotiable contract |
| D6 | PR shape | **PR1 = C++ + tests + docs (no bindings); PR2 = nanobind + dashboard re-wire** | smaller, mergeable first PR; bindings are orthogonal |

> **Supersedes [[cpp-port-plan]] Decision 2** (which recommended adapting to upstream `Hardware` immediately). The audit shows the two hardware structs are parameterized too differently to merge cheaply without breaking byte-identity or HIP-freedom — sibling-first is safer.

## Milestones

### S0 — Checkout + baseline (0.5 d)
Clean `rocm-libraries` checkout, feature branch, confirm baseline builds with HIP toolchain (`-DORIGAMI_BUILD_TESTING=ON -DORIGAMI_ENABLE_PYTHON=ON`) and Catch2/nanobind fetch succeed. **Acceptance:** upstream `ctest` green before any change.

### S1 — Convention conversion in the workspace (2 d) — ✅ done (2026-06-04)
Still in `origami_comms_cpp/` (so byte-identity re-verifies after each change): rename PascalCase → `snake_case_t`; prepend full MIT AMD license headers; run upstream `.clang-format`; backport the two `std::span` uses in `primitives.hpp` to pointer+size (or a tiny C++17 span shim); set the standalone target to `cxx_std_17`. **Acceptance:** full Stage-1 byte-identity suite still passes at C++17.

**As-built:** the only C++20 dependency was `std::span` (the internal-only span overload of `resolve_work_graph`); folded into the `std::vector<op_t>` overload. Rename done via a string-literal-protecting codemod so the pybind11 Python-facing names (`py::class_<hardware_t>(m, "Hardware")`) and model string keys stayed verbatim — dashboard + Python regression tests unchanged. Vendored upstream `.clang-format`. **Verified:** clean rebuild at `-std=c++17`, 11/11 CTest green including the 34,599-cell `test_full_sweep`, 0 warnings under `-Wall -Wextra -Wpedantic`, 0 residual PascalCase type refs.

### S2 — Drop into the tree + CMake wiring (1 d) — ✅ done (2026-06-04)
Copy headers to `shared/origami/include/origami/comm/`; add them to `origami-headers` `target_sources` (or stand up `roc::origami-comm-headers`); verify **no HIP leak** (comm headers compile in a HIP-free TU). **Acceptance:** `#include <origami/comm/origami_comm.hpp>` compiles in-tree; existing GEMM build untouched.

**As-built:** worked in an isolated `git worktree` (`origami-comm-integration` branch off the `k016` checkout at `~/worktrees/origami-comm-integration`, sparse `shared/origami`) so the live k016 tree was never disturbed. Stood up a dedicated **`roc::origami-comm-headers`** INTERFACE target that intentionally does **not** link `hip::host` — comm stays HIP-free and independently consumable — and registered it in `rocm_install(TARGETS …)` + `rocm_export_targets`. The existing recursive `rocm_install(DIRECTORY include/ … PATTERN *.hpp)` already installs `comm/*.hpp`, so no install-rule change was needed. **Verified:** (1) HIP-free TU including `origami/comm/origami_comm.hpp` compiles with plain `g++ -std=c++17 -Iinclude` (no ROCm on the include path), runs, and `-E` preprocessing pulls **zero** `hip_runtime`/`/opt/rocm` headers; (2) full origami `cmake` configure succeeds with the new target (hipcc + `CMAKE_PREFIX_PATH=/opt/rocm`); (3) the GEMM `origami` shared lib rebuilds clean (0 errors) — untouched. Diff: +9 headers, `CMakeLists.txt` +28/−1. Not yet committed.

### S3 — dtype reconciliation (1 d, gated on D3) — ✅ done (2026-06-04)
Map `comm::DataType` onto `origami::data_type_t`; update the byte-identity harness enum labels; re-run sweep. **Acceptance:** byte-identity preserved.

**As-built:** comm `types.hpp` now `#include "origami/types.hpp"` and `using origami::data_type_t;` (resolves unqualified via enclosing-namespace lookup — `origami::comm` is nested in `origami`), deleting the duplicate 6-value `enum class data_type_t`. The label remap is `FP16→Half, BF16→BFloat16, FP32→Float, FP64→Double, FP8→Float8, INT8→Int8`, applied across `types.hpp` (dtype defaults), `tensor.hpp` (`normalize_dtype` string table), `collective.hpp` (`predict_row` default), and the three tests that name dtypes. **Key constraint:** comm keeps its own header-only `dtype_bytes()` switch rather than calling `origami::data_type_to_bytes()` — the latter is backed by `datatype_to_bits()` in the HIP-linked `origami` lib, so using it would force comm consumers to link `roc::origami`. Reusing the *enum* (a pure header type) unifies the type without breaking the header-only / HIP-free contract. **Verified:** rebuilt clean, all 72 comm Catch2 cases pass, all six golden sweeps re-priced with **0 mismatches** (predict_row 630, collective_grid 360, layouts_grid 5936, tensor_collective 1920, wg_tile_latency 960, iter_times 630 = 10,436 cells); test binary `ldd` shows no HIP/ROCm runtime libs.

### S4 — Catch2 tests + golden data (2 d) — ✅ done (2026-06-04)
Port the 8 C++ layer tests + the golden-CSV byte-identity sweep into `shared/origami/tests/comm/` Catch2 style (`catch_discover_tests`); vendor trimmed golden CSVs. **Acceptance:** `ctest` runs comm tests alongside GEMM; sweep is bit-identical.

**As-built:** vendored all six golden CSVs (+README) into `shared/origami/tests/comm/golden/` and ported all **9** comm test files (72 cases) verbatim by shimming the custom harness onto Catch2: a new `tests/comm/test_harness.hpp` re-defines `TEST(name)→TEST_CASE("comm: name")`, `CHECK_NEAR→` absolute-tolerance `CHECK`, `ORIGAMI_TEST_MAIN()→` no-op, and relies on Catch2's own `CHECK` (same non-fatal semantics) — so **zero edits to the test bodies** beyond the S3 enum relabel. New `tests/comm/CMakeLists.txt` builds a standalone `origami-comm-tests` exe linking **only** `roc::origami-comm-headers` + `Catch2::Catch2WithMain` (no HIP), with `catch_discover_tests(... WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})` so the golden CSVs resolve at run time. Wired via `add_subdirectory(comm)` in `tests/CMakeLists.txt`. **Verified:** full `cmake -DORIGAMI_BUILD_TESTING=ON` configure (Catch2 via FetchContent), `cmake --build --target origami-comm-tests` clean, `ctest -R comm:` → 72/72 pass, 10,436 golden cells with 0 mismatches. Diff: +9 test `.cpp`, +6 CSV + README, +`test_harness.hpp`, +`tests/comm/CMakeLists.txt`, `tests/CMakeLists.txt` +3. Not yet committed. **Note for S6:** the installed `CTestTestfile.cmake.install` still references only `origami-tests`; add `origami-comm-tests` there if comm tests should ship in the packaged test component.

### S5 — Bindings (2–3 d, gated on D4/D6 → PR2)
Add comm functions to the nanobind `origami` module (`origami.comm.predict_row`, `predict_tensor_collective`, hardware constants); update `dashboard/_origami_backend.py` to import from `origami.comm` with the existing Python fallback. **Acceptance:** dashboard renders through the in-tree module; `ORIGAMI_FORCE_PY_BACKEND=1` still works.

### S6 — Docs + PR (1 d)
`shared/origami/comm/README.md` (API + link to this page + golden-regeneration note); header doxygen blocks to match GEMM style; PR draft. **Acceptance:** PR opened against `rocm-libraries`.

## Estimate

- **PR1** (S0–S4 + S6, no bindings): **~7 days** — the reviewable core.
- **PR2** (S5 + dashboard re-wire): **~3 days**.
- Buffer ~3 days for rocm-cmake install quirks, clang-format churn, and maintainer review cycles.
- **Total: ~2–2.5 weeks** to a landed, byte-identical, bound comm model in upstream.

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Maintainers require full `hardware_t` unification | Medium | High — scope balloon | Propose sibling-first PR1 with a written unification follow-up; D1 framing |
| C++ standard bump rejected by downstream | Medium | Medium | D2 backports comm to C++17; no shared-standard change |
| nanobind/pybind11 coexistence friction | Medium | Low | D6 defers bindings to PR2; PR1 needs no Python |
| Golden oracle depends on out-of-tree Python | High (structural) | Low | Freeze vendored golden CSVs; document external regeneration script |
| Comm tests drag in HIP toolchain unnecessarily | Low | Low | Gate comm tests so they build without HIP (comm is HIP-free) |
| clang-format / license churn obscures review | Medium | Low | Do S1 conversion as its own commit, re-verified byte-identical, before the move |

## Cross-references

- [[cpp-port-status]] — Stage 1 as-built; the thing being upstreamed.
- [[cpp-port-plan]] — original plan; this page supersedes its Decision 2 and the deferred "What's next".
- [[origami]] — upstream GEMM model (the integration site).
- [[codesign-rccl-hipblaslt]] — the downstream use case landing enables.
