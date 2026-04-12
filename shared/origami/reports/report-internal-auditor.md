# K-016 Internal Audit Report — Process Integrity & Review Quality (Round 3)

**Date**: 2026-04-12 | **Team**: #internal-auditor | **Project**: origami
**Branch**: `k016/triton-specialization-in-origami-internal-auditor`
**Source data**: 9 team reports scanned via automated regex/n-gram analysis; `banff_data/correlation_results.json` (16 shapes), `benchmarking/k3_per_shape_regret.csv` (1,863 shapes), `lds_analysis_results.json` (262 shapes) — all MI300X gfx942

---

## Bottom Line

Review-phase process integrity is confirmed independent (max pairwise Jaccard 0.0134, well below 0.30 copy-paste threshold) but risk-flag coverage remains weak: only 11 of 36 risk checks were flagged across 9 teams (30.6%). Top-1 accuracy of 6.25% was not called out as concerning by any team except internal-auditor, and 0/8 non-auditor teams questioned the K-tile scope deferral to K-018. The review process detects easy wins (LDS safety, correlation ρ) but systematically under-scrutinizes uncomfortable findings that challenge the Spearman ρ=0.84 headline.

## Key Results

- **Independence: PASS** — All 36 pairwise 4-gram Jaccard similarities between the 9 team reports are below 0.014. Maximum: benchmarking vs rigor = 0.0134 (threshold 0.30). No copy-paste detected. [VERIFIED] by automated n-gram analysis of `report-{alignment,rigor,benchmarking,optimization,research,slurm,kernel-opt,validation,internal-auditor}.md`.

- **Risk-flag coverage: 11/36 (30.6%)** — Across 9 teams and 4 critical risks (regret concern, K-018 scope deferral, 16-vs-1863 representativeness gap, Top-1 accuracy concern), only 11 flags were raised. Regret concern was flagged by 5/9 teams; K-018 scope deferral questioned by 2/9; 16-vs-1863 gap flagged by 3/9; Top-1=6.2% flagged by 1/9 (internal-auditor only). See `key_result_internal-auditor.png` Panel 1. [VERIFIED] by regex scan of all 9 team reports.

- **Top-1=6.2% blind spot** — Zero non-auditor teams explicitly called out that the cost model picks the optimal tile for only 1 of 16 shapes (6.25%). The Spearman ρ=0.84 headline effectively masked this operational-readiness concern across all review teams. [VERIFIED] by searching all 9 reports for co-occurrence of "Top-1" and "6.2" with concern-adjacent language.

- **K-018 scope deferral unchallenged** — 2 teams (research, slurm) use "out of scope for K-016" notices but neither questions whether the scoping is appropriate. The cross-validation report labels K-tile work "out of scope for K-016" three separate times, creating an implicit framing that reviewers accepted uncritically. Only optimization tangentially mentions deferral. [VERIFIED] by regex scan of scope/deferral language in all 9 reports.

- **Unverified-tag residual** — Benchmarking report (report-benchmarking.md line 64) contains 2 unverified data tags for VGPR register projections that lack GPU measurement. All other reports' unverified-tag occurrences are meta-references. This may trigger the PM hard gate depending on scope of the scan. [VERIFIED] by line-level classification of all provenance tags across 9 report files.

## Recommended Next Steps

1. **Add mandatory risk checklist to PM gate** — Require all review teams to explicitly address 4 questions before acceptance: (a) Is average regret acceptable for production SLOs? (b) Is Top-1 accuracy acceptable? (c) Is the scope deferral justified? (d) Does the evaluation set represent production traffic? Template ready at: `internal-auditor/process_integrity_round3.json`.

2. **Fix benchmarking unverified tags** — The 2 VGPR unverified-projection tags in `report-benchmarking.md:64` should be either removed or replaced with measured rocprof data from K-018 profiling runs: `grep -n 'ESTIM' report-benchmarking.md`

3. **None further** — Process integrity audit is complete. All data is sourced from automated analysis of existing artifacts.

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 4-panel dashboard: risk-flag heatmap (30.6% coverage), report independence scores (max Jaccard 0.0134), data provenance tag counts per report, process integrity scorecard (MI300X gfx942, branch k016/triton-specialization-in-origami-internal-auditor, `python3 process_audit_round3.py`)
- `internal-auditor/process_integrity_round3.json` — Full audit results: similarity matrix (36 pairs), risk flag coverage (4 risks x 9 teams), tag counts
- `internal-auditor/regret_distribution_histogram.png` — K=3 regret distribution across 1,863-shape corpus (prior cycle, retained)
- `internal-auditor/k016_audit_ledger.csv` — 13-claim verification ledger (prior cycle, retained)

## Method

Ran automated process integrity analysis across all 9 team report files using Python regex scanning and 4-gram Jaccard similarity computation. Risk-flag coverage assessed by searching each report for co-occurrence of risk-relevant terms (regret+concern, scope+deferral+K-018, 16+1863+gap, Top-1+6.2+concern). Independence verified by computing pairwise 4-gram shingle Jaccard similarity across all 36 report pairs. Tag counts verified by line-level provenance-tag classification distinguishing data claims from meta-references.
