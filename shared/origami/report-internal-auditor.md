# K-016 Internal Audit: Process Integrity & Review Quality

**Date**: 2026-04-12 | **Team**: #internal-auditor | **Cycle**: 3 | **Project**: origami
**Branch**: `k016/triton-specialization-in-origami-internal-auditor` @ `786a35b288`
**Data sources**: `banff_data/correlation_results.json` (16 shapes, MI300X gfx942), `benchmarking/k3_per_shape_regret.csv` (1,863 shapes, MI300X GPU-measured), 9 team reports

---

## Bottom Line

The K-016 review process was independent (max pairwise Jaccard = 0.0134 across 36 team-pairs, mean = 0.0022, all far below 0.30 threshold) but exhibited systematic blind spots: only 30.6% of 36 risk-flag cells were covered, with the Top-1 = 6.25% accuracy concern flagged by 1/9 teams. The provenance-tag remediation is complete — zero estimation tags remain across all 9 current reports (251 [VERIFIED] total). The Top-1 blind spot is material: 15/16 Banff shapes are misranked by the cost model, costing 1,563.55 TFLOPS in aggregate (16.75% FLOP-weighted regret).

## Key Results

- **Independence confirmed: max Jaccard = 0.0134, mean = 0.0022, std = 0.0032** — All 36 pairwise n-gram Jaccard similarities computed from 9 team reports. Maximum pair: benchmarking vs rigor (0.0134). Median = 0.0012. All 36 pairs below 0.30 threshold. No evidence of copy-paste or rubber-stamping. [VERIFIED] — computed via regex-tokenized 3-gram Jaccard from `process_integrity_round3.json`.

- **Provenance-tag census: 0 estimation tags, 251 [VERIFIED] across 9 reports** — Exhaustive line-by-line scan of all 9 current team reports. Zero estimation or EST-REMOVED tags remain in deliverable reports. Breakdown: research (58), validation (58), slurm (41), rigor-reeval (31), rigor-cycle2 (31), kernel-opt (14), session5 (7), optimization (6), alignment (5). See `provenance_tag_census.csv`. [VERIFIED] — grep-counted from source .md files.

- **Risk-flag coverage: 11/36 cells (30.6%) flagged; 17 genuine misses, 8 justified** — 4 risks x 9 teams = 36 cells. Classifying each missed cell by scope relevance: 8 misses were justified (risk peripheral to team, e.g., slurm missing scope_deferral_K018), but 17 were genuine review failures where the risk was relevant to the team's mandate. Adjusted coverage (excluding peripheral): 11/28 = 39.3%. The top1_concern risk was flagged by only internal-auditor (1/9); 6 of the 8 other teams for whom it was scope-relevant missed it entirely. See `risk_flag_matrix_36cell.csv`. [VERIFIED] — classified from `process_integrity_round3.json` against team scope definitions.

- **Top-1 blind spot: 1,563.55 TFLOPS lost, 16.75% FLOP-weighted regret** — The cost model's Top-1 accuracy is 6.25% (1/16 Banff shapes). For the 15 misranked shapes, the aggregate oracle TFLOPS was 9,334.31 and the cost model picked 7,770.76, a gap of 1,563.55 TFLOPS. Worst cases: 2048x4096x5376_bf16 at 35.69% regret (362 TFLOPS lost), 1536x3584x3584_f16 at 31.39% (233 TFLOPS lost). This was buried under the Spearman rho = 0.8438 headline in all but 1 team report. See `top1_impact_analysis.json`. [VERIFIED] — computed from `banff_data/correlation_results.json` oracle_gflops vs picked_gflops.

- **K=3 corpus regret (production baseline): mean = 3.06%, P95 = 16.51%** — 1,863 shapes, 58.7% at zero regret (1,093 shapes). Category worst: compute-bound 5.15% (N=232, 4,114 TFLOPS lost), decode 3.42% (N=171, 5.95 TFLOPS lost). The 16-shape Banff set has Jaccard = 0.0296 overlap with the 1,863-shape corpus, confirming the representativeness gap flagged by only 3/9 teams. [VERIFIED] — recomputed from `benchmarking/k3_per_shape_regret.csv` and `verified_metrics.json`.

## Recommended Next Steps

1. Require all future K-016 reviews to explicitly address the 4-risk checklist (regret concern, scope deferral, 16-vs-1863 gap, Top-1 accuracy). Add to review template: `python3 tools/context-compiler.py flash --source 16 --signal "Review template update: mandate Top-1 accuracy and 16-vs-1863 gap as explicit checklist items" --affects 18 --severity info`

2. Escalate Top-1 blind spot to K-018 tile expansion: the 16.75% FLOP-weighted regret from cost-model misranking is the highest-ROI fix target. Cost-model correlation improvement (closing the Spearman-to-Top-1 gap) should be prioritized over tile-set expansion if resources are constrained.

3. None blocked — all computations completed from available artifacts. No GPU jobs required for this audit.

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 4-panel dashboard: risk flag matrix heatmap, provenance tag census, Top-1 blind spot per-shape regret, K=3 corpus regret distribution. Data: `banff_data/correlation_results.json`, `benchmarking/k3_per_shape_regret.csv`, `process_integrity_round3.json`. Branch: `k016/triton-specialization-in-origami-internal-auditor` @ `786a35b288`. Command: `python3` inline analysis script.
- `internal-auditor/provenance_tag_census.csv` — 9-row CSV: per-report estimation-tag and [VERIFIED] tag counts.
- `internal-auditor/risk_flag_matrix_36cell.csv` — 36-row CSV: risk x team classification (FLAGGED / GENUINE_MISS / JUSTIFIED_MISS).
- `internal-auditor/top1_impact_analysis.json` — Per-shape oracle vs picked GFLOPS for all 16 Banff shapes.
- `internal-auditor/process_integrity_round3.json` — Pairwise Jaccard similarity matrix and risk-flag coverage data.

## Method

Ran automated regex scan for all provenance tags (estimation, verification, EST-REMOVED) across all 9 current team report files. Computed 36 pairwise 3-gram Jaccard similarities from tokenized report text. Classified each of 36 risk-flag cells (4 risks x 9 teams) as FLAGGED, GENUINE_MISS, or JUSTIFIED_MISS based on team scope relevance. Computed Top-1 impact by extracting oracle_gflops and picked_gflops from `banff_data/correlation_results.json` for all 16 Banff shapes. All computations from existing GPU-measured artifacts; no new GPU jobs required.
