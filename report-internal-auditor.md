# K-016 Internal Audit — Cycle 2: Row-Level Reconciliation Report

**Date**: 2026-04-12 | **Branch**: `k016/triton-specialization-in-origami-internal-auditor` | **GPU**: MI300X (Banff cluster, 3 GPUs)

## Bottom Line

The Section 2 per-shape table in `analysis_results.md` contains **8 discrepant cells across 4 shapes** (out of 80 cells audited), not just the 2 previously identified. Root cause: the table was sourced from `comprehensive_analysis_results.json` which used LDS-unfiltered tile picks, while `correlation_results.json` (with `n_lds_filtered=25`) has the correct hardware-measured, LDS-filtered values. The K=3 lookup table (1,863 shapes) is arithmetically faithful (12/12 spot-checks pass). Top-5 worst-regret membership is unaffected by corrections, but ordering shifts and the corrected avg regret is 17.51% vs 17.88% reported.

## Key Results

- **Section 2 discrepancy count: 8/80 cells in 4/16 shapes** — Exhaustive diff of all 16 rows × 5 numeric columns against `correlation_results.json` raw GPU data. 72 cells match, 8 do not. Affected shapes: `1x16384x16384_bf16_r` (regret 29.3→32.2%, tile 16x256x128→16x256x64), `1x13312x16384_bf16_r` (29.1→30.8%, 16x256x128→16x256x64), `1536x3584x3584_f16_r` (24.0→31.4%, 192x96x128→256x224x64), `128x13312x16384_bf16_r` (20.8→3.0%, 128x192x128→128x224x64). [VERIFIED] from `correlation_results.json` hardware measurements on MI300X.

- **LDS filter is the root cause** — All 3 incorrect picked tiles (`16x256x128`=69,632B, `192x96x128`=73,728B, `128x192x128`=81,920B) exceed the MI300X 64KB LDS limit. The correlation harness in `comprehensive_analysis_results.json` never called `check_lds_capacity()`, per the `lds_audit_summary.json` note. [VERIFIED] from LDS byte computation (formula: `(M*K + N*K) * dtype_bytes * stages`).

- **K=3 lookup table is arithmetically correct: 12/12 spot-checks pass** — Random sample (seed=42) of 12 entries from the 1,863-row `k3_per_shape_regret.csv`. Recomputed regret = `(oracle_tflops - k3_tflops) / oracle_tflops × 100` matches CSV values to <0.01pp for all 12. Total row count confirmed = 1,863. [VERIFIED] from `k3_per_shape_regret.csv`.

- **Top-5 ranking: membership stable, ordering shifts** — After correction, top-5 worst-regret shapes are the same 5, but `1536x3584x3584_f16_r` moves from #5 (24.0%) to #3 (31.4%) and `128x13312x16384_bf16_r` drops entirely from high-regret concern (20.8%→3.0%). [VERIFIED] from corrected ranking.

- **Aggregate impact is modest: −0.37pp** — Corrected mean regret = 17.51% vs reported 17.88%. Category-level: decode 21.9%, prefill 23.4%, compute-bound 13.6%, small-batch 10.2%. See `key_result_internal-auditor.png`. [VERIFIED] from `correlation_results.json`.

## Recommended Next Steps

1. **Patch `analysis_results.md` Section 2 table** — Replace the 4 affected rows with values from `correlation_results.json`. The corrected data is in `section2_reconciliation.json`. Copy-paste corrections:
   - `1x16384x16384_bf16_r`: regret=32.2%, picked=16x256x64
   - `1x13312x16384_bf16_r`: regret=30.8%, picked=16x256x64
   - `1536x3584x3584_f16_r`: regret=31.4%, picked=256x224x64
   - `128x13312x16384_bf16_r`: regret=3.0%, picked=128x224x64

2. **Fix the correlation harness to call LDS filter before tile selection** — The `comprehensive_analysis_results.json` pipeline omits the LDS capacity check. Add `check_lds_capacity()` gating to prevent future LDS-invalid tile picks.

3. None further — K=3 lookup table and LDS filtering logic are verified correct.

## Evidence Files

- `reports/tasks/K-016/internal-auditor/key_result_internal-auditor.png` — Bar chart comparing report vs raw JSON regret per shape (16 Banff shapes, MI300X)
- `reports/tasks/K-016/internal-auditor/section2_reconciliation.json` — Full reconciliation data: 8 discrepancies, root cause, top-5 impact, K=3 spot-check results

## Method

Ran exhaustive Python diff of all 16 Section 2 rows (80 cells) against `correlation_results.json` raw GPU measurements; randomly sampled 12/1,863 entries from `k3_per_shape_regret.csv` and recomputed regret from oracle/picked TFLOPS; verified LDS byte computation for all 3 invalid tiles against MI300X 64KB limit using `(M*K + N*K) * dtype_bytes * stages` formula; quantified top-5 ranking impact by re-sorting corrected regret values.
