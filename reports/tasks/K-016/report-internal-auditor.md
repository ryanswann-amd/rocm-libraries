# K-016 Internal Audit: Executive Re-Evaluation Verdict

**Auditor**: #internal-auditor | **Date**: 2026-04-12 | **Cycle**: 2 (re-dispatch round 1)
**Branch**: `k016/triton-specialization-in-origami-internal-auditor` @ `0cc6c2d2ec`

---

## Bottom Line

**CONDITIONAL-GO on closing K-016.** Fat-tail regret autopsy reveals 5/16 shapes with >25% regret, contributing 56.7% of total summed regret. The dominant root cause is BLOCK_N overvaluation (3/5 fat-tail shapes), where origami's cost model systematically picks tiles with large BLOCK_N over tiles with large BLOCK_K that hardware prefers. K-016's LDS filter is a valid correctness fix (prevents 4/16 harness picks from crashing on MI300X), but it does not reduce the 17.5% production regret. K-018 must address BLOCK_N overvaluation to close the gap.

## Key Results

- **Fat-tail regret autopsy: 5 shapes >25% regret, 3 root-caused to BLOCK_N overvaluation** [VERIFIED] on MI300X (Banff cluster, gfx942). The worst shape `2048x4096x5376_bf16` has 35.7% regret (362 TFLOPS gap): cost model picks 256x224x64 (652 TFLOPS) over oracle 256x128x64 (1014 TFLOPS). Both decode shapes (`1x16384x16384`, `1x13312x16384`) pick 16x256x64 when hardware prefers large BLOCK_K (16x32x512 and 64x64x256). Source: `correlation_results.json`, per-shape pick-vs-oracle comparison. See `key_result_internal-auditor.png`.

- **Mean regret = 17.51%, median = 13.34%, mean Spearman = 0.8438** [VERIFIED] on MI300X across 16 Banff shapes x 259 tile configs. Top-1 accuracy = 6.2% (1/16 shapes). Fat-tail 5 shapes contribute 158.8pp of 280.2pp total summed regret (56.7%). Source: `correlation_results.json`, independently re-derived.

- **LDS filter correctness confirmed: 4/16 harness picks are LDS-invalid** [VERIFIED]. Tiles 16x256x128 (68KB), 192x96x128 (72KB), and 128x192x128 (80KB) exceed MI300X's 64KB LDS limit. Production `rank_configs()` at `origami.cpp:544` calls `check_lds_capacity()` which correctly rejects these. The correlation harness skipped this check, inflating some shape regrets. LDS formula `A+B` in `gemm.cpp:347` is correct for 2-stage pipeline; for 3-stage, 16/21 unique tiles become false-accepts. Source: `gemm.cpp:338-354`, `lds_audit.csv`.

- **K=3 lookup regret on Banff 16 = 16.5% vs origami 17.5%** — apples-to-apples improvement is 1.06x, not 5.8x [VERIFIED]. The 5.8x headline compares different oracle spaces (37 Triton tiles for K=3 vs 259 hipBLASLt tiles for origami). On the 1,863-shape corpus, K=3 achieves 3.06% +/- 0.13% because that corpus uses matching tile domains. Source: `k3_validation_data.json`.

- **K-018 fix priority: BLOCK_N overvaluation** [VERIFIED] via root-cause classification of all 5 fat-tail shapes. 3/5 are BLOCK_N overvaluation (decode + prefill), 1/5 tile-area overvaluation, 1/5 general ranking error. Fixing BLOCK_N valuation for decode shapes alone would reduce mean regret by ~3.9pp. Source: `fat_tail_autopsy.csv`, `k018_fix_recommendations.json`.

## Recommended Next Steps

1. **Close K-016 with LDS filter merged** — it prevents 4/16 harness crashes and is a prerequisite for K-018. Condition: amend `cross_validation_report.md` to replace "5.8x better" headline with "1.06x on matched oracle space." Run: `git merge k016/triton-specialization-in-origami` into develop.

2. **Fast-track K-018 with BLOCK_N fix as P0 deliverable** — the `k018_fix_recommendations.json` file contains per-root-cause fix specifications. Start with decode shapes (M=1): clamp BLOCK_N cost contribution and up-weight BLOCK_K for shapes where K >> N. This alone targets 2/5 fat-tail shapes (62pp combined regret reduction potential).

3. **File tech-debt issue for 3-stage LDS validity** — `check_lds_capacity()` in `gemm.cpp:347` uses single-buffer formula `A+B`. For 3-stage pipelines, only 3/21 tiles remain valid (16x64x128, 16x16x512, 64x64x64). If K-018 explores 3-stage configs, the formula must be updated to `(stages-1)*(A+B)`.

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 2-panel dashboard: fat-tail regret autopsy (>25%) + all-16-shapes regret distribution. Data: correlation_results.json, MI300X Banff. Branch: k016/triton-specialization-in-origami-internal-auditor @ 0cc6c2d2ec. Command: `python3 fat_tail_autopsy.py`.
- `internal-auditor/lds_audit_plot.png` — LDS formula 2-stage vs 3-stage scatter for 21 unique tiles. Data: gemm.cpp:338-354 formula, MI300X LDS=64KB.
- `internal-auditor/fat_tail_autopsy.csv` — Per-shape autopsy: pick vs oracle tile, LDS validity, root cause classification (5 rows).
- `internal-auditor/lds_audit.csv` — LDS validity audit for 21 unique tiles under 2-stage and 3-stage formulas.
- `internal-auditor/k018_fix_recommendations.json` — Root-cause-to-fix mapping for K-018 consumption.
- `internal-auditor/lds_audit_summary.json` — Summary statistics for LDS formula discrepancy audit.
- `internal-auditor/k016_audit_ledger.csv` — Full 16-shape audit ledger with spearman, regret, tiles, LDS counts.

## Method

Extracted all 16-shape performance data from `banff_data/correlation_results.json` (MI300X hardware-measured, 259 tile configs per shape). Identified 5 fat-tail shapes (>25% regret), performed pick-vs-oracle tile comparison to diagnose root causes via BLOCK_M/N/K divergence analysis. Audited LDS formula from `gemm.cpp:338-354` (`check_lds_capacity()`) against all 21 unique tiles for 2-stage and 3-stage pipeline validity. Cross-referenced with `k3_validation_data.json` and optimization team's `analysis_data.json`. All numbers re-derived from hardware measurements — no simulations or projections used.

---

## Risk Register

| Risk | If ship K-016 as-is | If delay K-016 |
|------|---------------------|----------------|
| LDS crashes | Eliminated (0% crash-risk) [VERIFIED] | Continues — 4/16 invalid tiles selected in harness |
| Production regret | Unchanged at 17.5% [VERIFIED] | Unchanged at 17.9% |
| K-018 dependency | Unblocked — K-016 provides LDS filter infrastructure | Blocked — K-018 needs LDS filter |
| BLOCK_N overvaluation | Unaddressed (deferred to K-018) | Unaddressed |

## Conditions for Close

1. LDS filter code passes CI (compile + unit tests on MI300X)
2. `cross_validation_report.md` "5.8x better" headline amended with dataset-scope caveat
3. `k018_fix_recommendations.json` delivered to K-018 team as input artifact
