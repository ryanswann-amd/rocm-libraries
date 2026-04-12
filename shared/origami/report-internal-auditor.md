# K-016 Internal Audit: Executive Re-Evaluation Verdict

**Verdict: CONDITIONAL-GO** | **Date**: 2026-04-12 | **Hardware**: MI300X gfx942 (Banff, OCI)
**Branch**: `k016/triton-specialization-in-origami-internal-auditor`

## Bottom Line

K-016 delivers a necessary crash-prevention gate — LDS filtering eliminates 10/27 unsafe tile/stage/dtype combinations at stage-3+ [VERIFIED] from `rigor/expanded_lds_analysis.json` on MI300X gfx942 — but contributes zero tile-selection quality improvement. The cost-model regret remains 17.88% on 16 Banff shapes [VERIFIED] from `banff_data/correlation_results_no_lds_filter.json` on MI300X gfx942. Ship K-016 as a safety prerequisite, but fast-track K-018 to address the regret gap that K-016 was perceived to solve.

## Key Results

- **Avg Spearman correlation: 0.8438** — recomputed as mean of 16 per-shape values from `banff_data/correlation_results.json`, delta = 4.4x10-5 from report [VERIFIED] on MI300X gfx942 (Banff, 3 GPUs, 16 shapes x 259 tile configs).

- **K=3 cross-validated regret: 3.06% +/- 6.07%** — recomputed from 1,863 shapes in `benchmarking/k3_per_shape_regret.csv`; 58.7% of shapes achieve zero regret, 39 shapes (2.1%) exceed 25% [VERIFIED] on MI300X gfx942. Production-weighted regret (FMA-volume): 4.54% [VERIFIED] from same source.

- **LDS crash prevention: 10/27 combos saved** — 3 K=3 tiles x 3 stages x 3 dtypes; all 9 stage-2 combos pass, 10/27 total fail at stage-3+ (e.g., 128x128x128 bf16 stage-3 requires 131,072 bytes vs 65,536 limit). Only 3/21 tiles pass 3-stage LDS [VERIFIED] from `internal-auditor/lds_audit.csv` on MI300X gfx942.

- **Process regression: 4 -> 43 unverified tags across rounds** — Round 1 gate failed with 4 unverified tags; Round 2 re-dispatch produced 43 (a 10.75x regression). Total cost through 2 rounds: $69.80. Three team-rounds hit max cycles without acceptance [VERIFIED] from system dispatch logs. See `key_result_internal-auditor.png` Panel 3.

- **Scope gap confirmed: K-016 = LDS filtering only** — Of five tile-selection quality capabilities (LDS prevention, cost-model regret, decode specialization, compute-bound tail, top-1 accuracy), K-016 delivers only LDS prevention. The remaining four require K-018 [VERIFIED] from `cross_validation_report.md` scope notices and `k016_executive_report.md` Section 3. See `key_result_internal-auditor.png` Panel 4.

## Recommended Next Steps

1. **Ship K-016 as LDS crash-prevention gate** — rename ticket scope to "LDS Filtering Integration & Validation" to prevent stakeholder confusion that tile-selection quality was improved.
2. **Fast-track K-018 with decode M-threshold** — switching M<=1 shapes to `16x64x128_s2` reduces decode regret from 23.3% to 3.7% [VERIFIED] from `cross_validation_report.md` Table 2 on MI300X gfx942, 1,863-shape corpus. This is a one-line code change: `if M <= 1: tile = 16x64x128_s2`.
3. **Investigate round-2 tag regression** — the 4-to-43 unverified-tag increase across re-dispatch rounds indicates a systemic issue where teams introduced new approximate claims when asked to fix old ones. Recommend adding a pre-commit linter that rejects any report containing the string "ESTIMATED" in brackets.

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 4-panel audit dashboard: K=3 regret by category, regret comparison across methods, hard gate compliance by round, scope gap K-016 vs K-018. MI300X gfx942, `python3 audit_dashboard.py`.
- `internal-auditor/k016_audit_ledger.csv` — 16-entry cross-team claim verification ledger with source files, recomputed values, and deltas.
- `internal-auditor/fat_tail_autopsy.csv` — 5-row root-cause analysis of worst-regret shapes with BLOCK_N overvaluation diagnosis.
- `internal-auditor/lds_audit.csv` — 21-tile LDS validity summary with per-tile byte computations.

## Method

Synthesized findings from 4 team reports (rigor, alignment, benchmarking, optimization) by independently re-deriving 13 key numerical claims from their source data files using Python scripts. Verified process metrics (gate failures, costs, cycle counts) from system dispatch logs. Generated audit dashboard from `k3_per_shape_regret.csv` (1,863 shapes) and `correlation_results.json` (16 Banff shapes). All computations performed on CPU; all underlying measurements are from MI300X gfx942 hardware.
