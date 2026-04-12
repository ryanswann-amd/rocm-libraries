# K-016 Internal Audit: Executive Re-Evaluation Verdict

**Verdict: CONDITIONAL-GO** | **Date**: 2026-04-12 | **Hardware**: MI300X gfx942 (Banff, OCI)
**Branch**: `k016/triton-specialization-in-origami-internal-auditor`

## Bottom Line

K-016 delivers a crash-prevention gate — LDS filtering correctly rejects 18/21 tile configs at 3-stage, with 3 valid tiles confirmed by independent formula re-derivation from Triton compiler source [VERIFIED] from `lds_21tile_computation_trace.json` on MI300X gfx942. However, K-016 contributes zero tile-selection quality improvement: the cost-model regret remains 17.88% on 16 Banff shapes [VERIFIED] from `correlation_results_no_lds_filter.json` on MI300X gfx942, while K=3 lookup regret is 3.06% ± 6.07% on 1,863 shapes [VERIFIED] from `k3_per_shape_regret.csv` on MI300X gfx942. These are different metrics — cost-model ranking accuracy vs lookup-table coverage — not contradictory numbers. The 39 fat-tail shapes (>25% regret) are 100% prefill/small-batch workloads; the proposed M≤1 decode threshold covers none of them. Ship K-016 as a safety prerequisite; fast-track K-018 for regret reduction.

## Key Results

- **Avg Spearman correlation: 0.8438** — independently recomputed as mean of 16 per-shape values from `correlation_results.json`; delta = 4.4×10⁻⁵ from reported value [VERIFIED] on MI300X gfx942 (Banff, 16 shapes × 259 tile configs). See `key_result_internal-auditor.png` Panel 2.

- **K=3 cross-validated regret: 3.06% ± 6.07%** — recomputed from 1,863 shapes in `k3_per_shape_regret.csv`. Median = 0.00%, 58.7% of shapes achieve zero regret, 39 shapes (2.09%) exceed 25% [VERIFIED] on MI300X gfx942. This measures lookup-table coverage, NOT cost-model ranking accuracy. See Panel 2.

- **Banff cost-model regret: 17.88%** — recomputed from 16 shapes in `correlation_results_no_lds_filter.json`. This measures how well the cost model ranks tiles, a fundamentally different question from K=3 lookup coverage. Both numbers are correct; the discrepancy is semantic, not numerical [VERIFIED] on MI300X gfx942.

- **Fat-tail autopsy: 0/39 worst shapes covered by M≤1 threshold** — expanded autopsy from prior cycle's 5 rows to all 39 shapes with >25% K=3 regret. Clustering: 36 prefill (M≥128), 3 small_batch (M=64), 0 decode (M≤1). The proposed decode specialization threshold is irrelevant to the fat-tail problem. Top M values: 192 (7 shapes), 384 (16 shapes), 256 (4 shapes). Total TFLOPS gap: 3.47 TFLOPS [VERIFIED] from `fat_tail_autopsy_full.csv` derived from `k3_per_shape_regret.csv` on MI300X gfx942. See Panel 1.

- **LDS borderline risk: 5 tiles at exact 65536-byte boundary** — 4 tiles at exactly 65536 bytes (2-stage) and 1 tile at exactly 65536 bytes (3-stage: 16×16×512). The origami formula matches the Triton compiler source with 0% delta across all 21 tiles [VERIFIED] from `lds_21tile_computation_trace.json`, formula verified against `gemm.hpp:266-277` and `Allocation.cpp:184-212` on MI300X gfx942. False-negative risk is low (formula is exact) but a 1KB safety margin would eliminate boundary tiles. See Panel 3.

## Recommended Next Steps

1. **Ship K-016 as LDS crash-prevention gate** — rename ticket scope to "LDS Filtering Integration" to prevent stakeholder confusion. The regret metrics are untouched by K-016.
2. **Fast-track K-018 with prefill tile-selection fix** — the M≤1 decode threshold reduces decode regret by 3.94pp [VERIFIED] from `regret_reduction_verification.txt` on MI300X gfx942, but all 39 fat-tail shapes are prefill (M≥128). K-018 must also target the prefill regime (M=192–448) where 36/39 worst shapes cluster.
3. **Add 1KB LDS safety margin** — change the LDS limit from 65536 to 64512 bytes to eliminate 4 borderline 2-stage tiles (`128x128x128`, `256x256x64`, `288x224x64`, `64x64x256`) and 1 borderline 3-stage tile (`16x16x512`). One-line change in `gemm.hpp`.

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 4-panel dashboard: (1) all 39 fat-tail shapes by op_type, (2) K=3 regret distribution with Banff overlay, (3) LDS budget for 21 tiles with 64KB limit, (4) scope gap K-016 vs K-018. MI300X gfx942.
- `internal-auditor/fat_tail_autopsy_full.csv` — 39-row full autopsy of all shapes >25% regret, with M/N/K dimensions, operation type, and TFLOPS gap. Expanded from prior cycle's 5 rows.
- `internal-auditor/k016_audit_ledger.csv` — 13-entry cross-team claim verification ledger with source files, recomputed values, and deltas.

## Method

Independently re-derived 13 key numerical claims from raw data files (`correlation_results.json`, `k3_per_shape_regret.csv`, `lds_21tile_computation_trace.json`) using Python analysis scripts. Expanded the fat-tail autopsy from 5 shapes (Banff subset) to all 39 shapes >25% regret in the 1,863-shape corpus, clustered by operation type (decode/prefill/small_batch) to test M≤1 threshold coverage. Verified LDS formula correctness by checking all 21 K=3 tiles against the 65,536-byte MI300X limit and identifying 5 boundary-condition tiles. All underlying measurements are from MI300X gfx942 hardware sweeps; audit computations performed on CPU.
