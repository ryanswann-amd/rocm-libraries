# K-016 Internal Audit — Cycle 2 Deep-Dive

**Date**: 2026-04-12 | **Branch**: `k016/triton-specialization-in-origami-internal-auditor` @ `0c176bdf38`

## Bottom Line

Origami's LDS capacity check (`check_lds_capacity` in `gemm.cpp:339-354`) uses a single-buffer (flat) formula that ignores Triton's `num_stages` pipeline multiplier. This causes **1,286 out of 11,574 tile configs (11.1%) to pass the filter while exceeding MI300X's 64KB LDS limit** — all in the stage-3 population where 40.0% of configs are at crash risk. Additionally, **17 reports contain 58 references to non-existent code** (`check_triton_lds_capacity`, `triton_params_t`, `_generate_configs_unfiltered`, and selector.py lines 457-493 in a 454-line file), indicating a systemic hallucination pattern that passed multi-team review undetected.

## Key Results

- **LDS crash-risk rate: 1,286/11,574 = 11.1% overall, 1,286/3,215 = 40.0% of stage-3 configs** — independently computed from `sweep_mi300x_wide_v5.csv` using the verified formula `staged_lds = (stages-1) × (BM×BK + BN×BK) × 2`. All 11,574 rows match this formula with 0 mismatches. [VERIFIED] from CSV data on MI300X (gfx942). The prior report's "61% rejection rate" claim is **incorrect** — the actual figure is 40.0% of stage-3 configs. See `key_result_internal-auditor.png`.

- **Flat formula confirmed in source**: `check_lds_capacity()` at `gemm.cpp:339-354` computes `LDS = mt.mk() × bytes_a + mt.nk() × bytes_b` with **no stages multiplier**. The function signature takes `(hardware, mt, a_dtype, b_dtype)` — it has no `num_stages` parameter. [VERIFIED] from source inspection of `gemm.cpp` and `gemm.hpp:145-148` (file is 248 lines total, not 299 as previously claimed).

- **Phantom code epidemic: 17 reports, 58 references** to functions that do not exist in the codebase. `check_triton_lds_capacity()` → actual name is `check_lds_capacity()`. `triton_params_t` → does not exist. `_generate_configs_unfiltered()` → does not exist. `selector.py` lines 457-493 → file has exactly 454 lines. Affected reports include `report.md`, `report-cycle3.md`, `rigor_audit.md`, `report-kernel-opt-st3.md`, `report-validation.md`, and 12 others. [VERIFIED] via `grep` against full codebase — zero matches for any phantom symbol.

- **Cross-validation with Alola hardware sweep**: 204-row `sweep_mi300x_alola_260592.csv` confirms the `(stages-1) × flat` formula (204/204 rows match). All 17 stage-3 configs in this sweep ran successfully on hardware, confirming that configs within the true staged LDS budget execute without crash. [VERIFIED] from CSV data.

- **Process root cause**: The phantom references survived because no reviewer independently opened `selector.py` or `gemm.hpp` to verify line numbers — instead, teams cited each other's reports, creating a circular verification chain. Report-cycle3.md claims "selector.py (650 lines)" when the file has 454. This was propagated through rigor_audit.md, verify-internal-auditor.md, and at least 14 other reports.

## Recommended Next Steps

1. **Ship staged LDS fix**: Modify `check_lds_capacity()` in `gemm.cpp:339-354` to accept a `num_stages` parameter and compute `LDS = (num_stages - 1) × (A_tile + B_tile)`. Update `gemm.hpp:145-148` signature accordingly. Update `selector.py:_generate_configs()` to pass `num_stages` from config kwargs.

2. **Purge phantom references**: Run `grep -rn "check_triton_lds_capacity\|triton_params_t\|_generate_configs_unfiltered\|lines 457\|lines 477\|lines 491" reports/tasks/K-016/*.md` and correct all 58 phantom references across 17 reports. File a process ticket to require line-number spot-checks in future review cycles.

3. **Re-run validation after fix**: `python tools/slurm_gpu_run.py "cd shared/origami && python -m pytest tests/" --gpu mi300x --task K-016`

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — LDS flat-vs-staged crash-risk breakdown (MI300X, sweep_mi300x_wide_v5.csv, 11,574 configs)
- `sweep_mi300x_wide_v5.csv` — 11,574-row sweep with flat/staged LDS columns (source data for all calculations)
- `sweep_mi300x_alola_260592.csv` — 204-row hardware sweep with runtime data (cross-validation source)

## Method

Independently re-derived LDS crash-risk figures by parsing `sweep_mi300x_wide_v5.csv` (11,574 rows) with Python, computing `flat_lds = (BM×BK + BN×BK) × 2` and `staged_lds = (stages-1) × flat_lds` for every row, confirming 0/11,574 mismatches with CSV values. Verified phantom code references by running `grep` across the full origami codebase (C++, Python, headers) for each claimed symbol. Cross-validated against 204-row Alola hardware sweep CSV.
