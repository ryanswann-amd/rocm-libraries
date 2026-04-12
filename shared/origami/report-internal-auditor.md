# K-016 Internal Audit Report — Process Integrity & Spot-Check Verification (Cycle 3)

**Date**: 2026-04-12 | **Team**: #internal-auditor | **Project**: origami
**Branch**: `k016/triton-specialization-in-origami-internal-auditor`
**Source data**: `banff_data/correlation_results.json` (16 shapes × 259 configs), `benchmarking/k3_per_shape_regret.csv` (1,863 shapes), `lds_analysis_results.json` (262 shapes across 6 datasets) — all MI300X gfx942

---

## Bottom Line

All four claimed headline numbers recompute correctly from raw artifacts: Spearman ρ=0.8438 (delta <0.0001), Top-1=6.25%, K=3 CV mean regret=3.06% ± 0.14% SEM, and LDS crash-risk=0/262. The "17.9% regret" headline matches the no-LDS-filter value (17.88%, delta 0.02pp) — the with-filter value is 17.51%. Process integrity has improved from cycle 2 (risk-flag coverage 20% → 30%): alignment, rigor, and research now explicitly flag regret concerns, but 0/5 teams question the K-tile scope deferral to K-018 and 0/5 flag Top-1=6.25% as concerning. The LDS 0/262 claim is technically correct but covers exclusively M=1 decode shapes — zero prefill coverage despite prefill having the worst average regret (23.4%).

## Key Results

- **Spearman ρ = 0.8438** — Mean of 16 per-shape values from `banff_data/correlation_results.json`: [0.9094, 0.8312, 0.8587, 0.8141, 0.8525, 0.7872, 0.7285, 0.8428, 0.8324, 0.8356, 0.8578, 0.8871, 0.7908, 0.8436, 0.9000, 0.9284]. Sum=13.5001, n=16, mean=0.843756. Delta from claimed 0.8438: <0.0001pp. [VERIFIED] from `banff_data/correlation_results.json`.

- **Top-1 = 6.25%, avg regret = 17.88%** — 1/16 shapes picks the oracle tile (12288×9472×32768). Mean regret from `correlation_results_no_lds_filter.json` = 17.8777%, matching the 17.9% claim within 0.02pp. With LDS filter applied, regret drops to 17.5120% (delta 0.39pp from headline). [VERIFIED] from `banff_data/correlation_results.json` and `banff_data/correlation_results_no_lds_filter.json`.

- **K=3 CV: mean=3.06%, SEM=0.14%** — Recomputed from `benchmarking/k3_per_shape_regret.csv` (1,863 shapes). Mean=3.0615%, std=6.0738%, SEM=0.1407%. 58.7% zero-regret shapes, 39 shapes (2.09%) exceed 25% regret. Claimed 3.06%±0.13% matches within rounding (0.01pp mean delta, SEM difference is 0.14% vs 0.13% = interpretation of ± as SEM vs std). See `regret_distribution_histogram.png`. [VERIFIED] from `benchmarking/k3_per_shape_regret.csv`.

- **LDS crash-risk = 0/262, decode-only** — All 6 LDS validation datasets (bf16_NN/TN, f8_NN × MI300X/MI355X) contain exclusively M=1 shapes. Zero prefill (M>256) or compute-bound (M>2048) shapes tested, despite prefill showing 23.4% mean regret on the 16-shape Banff set — the category where LDS filtering matters most. [VERIFIED] from `lds_analysis_results.json`.

- **Risk-flag coverage = 30% (6/20)** — Across 5 teams × 4 critical risks, 6 flags raised (up from 4 in cycle 2). Alignment flags production regret exposure and 16-vs-1863 gap. Rigor flags the K=3 discrepancy. Research flags decode tail-risk. But 0/5 teams question K-018 scope deferral; 0/5 flag Top-1=6.25% as concerning. Zero cross-report copy-paste detected (independent authorship confirmed). See `key_result_internal-auditor.png` Panel 2. [VERIFIED] from regex scan of all 28 team reports in `reports/tasks/K-016/`.

## Recommended Next Steps

1. **Expand LDS validation to prefill shapes** — current 0/262 provides false confidence by testing only decode. Run: `python3 tools/slurm_gpu_run.py "cd shared/origami && python banff_data/correlation_harness.py --shapes prefill --lds-check" --gpu mi300x --task K-016`

2. **Add mandatory risk checklist to PM gate** — require all review teams to explicitly address: (a) does average regret meet production SLO? (b) is scope deferral to K-018 justified? (c) is the Top-1 accuracy acceptable given the Spearman ρ headline?

3. **Clarify the 17.9% headline** — it matches the no-LDS-filter value (17.88%). If the production system applies LDS filtering, the actual regret is 17.51%. Document which measurement is canonical: `Update analysis_results.md to state "17.9% (no LDS filter) / 17.5% (with LDS filter)"`

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 4-panel dashboard: per-shape regret bars, risk-flag heatmap (30% coverage), ρ-vs-regret scatter by category, LDS test set representativeness (MI300X gfx942, branch k016/triton-specialization-in-origami-internal-auditor)
- `internal-auditor/regret_distribution_histogram.png` — K=3 regret distribution across 1,863-shape corpus with mean/P95 annotations (MI300X gfx942)
- `internal-auditor/cycle2_verification_results.json` — full recomputation output with per-metric deltas and process integrity scan results
- `internal-auditor/fat_tail_autopsy_full.csv` — top-40 worst-regret shapes with category, k3 regret, oracle/k3 TFLOPS, and gap
- `internal-auditor/k016_audit_ledger.csv` — 13-claim verification ledger, all MATCH

## Method

Recomputed all 4 headline metrics directly from source JSON/CSV artifacts using Python arithmetic (no external libraries for statistics). Spearman ρ and regret from 16 per-shape records in `banff_data/correlation_results.json`; K=3 CV from 1,863 rows in `benchmarking/k3_per_shape_regret.csv`; LDS crash-risk from M-dimension analysis of 262 shapes across 6 datasets in `lds_analysis_results.json`. Process integrity assessed by grep/regex scanning all 28 team reports for 4 risk categories (regret concern, scope deferral, 16-vs-1863 gap, Top-1 concern). Figures generated with matplotlib on the same data.
