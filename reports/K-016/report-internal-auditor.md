# K-016 Process Integrity Audit — Internal Auditor Report

## Bottom Line

The 5-team review produced genuinely independent work (0 copy-paste duplicates across 9,449 words), but suffered a critical blind spot: **only 20% of key risks were flagged** (4/20 possible flags). No team called out the 17.9% average regret as explicitly concerning, and no team questioned whether deferring K-tile selection to K-018 was appropriate scoping or convenient scope exclusion. The cross_validation_report.md contains 3 "out of scope for K-016" notices that went unchallenged by all 5 reviewers. The headline numbers (Spearman ρ=0.8438, Top-1=6.2%, K=3 CV 3.06% ± 0.13%, LDS 0/42) all verify correctly from source artifacts.

## Key Results

- **Independence: 0 exact sentence duplicates** across 5 team reports (9,449 total words) [VERIFIED] — parsed all lines >50 chars from report-alignment.md, report-rigor.md, report-benchmarking.md, report-optimization.md, report-internal-auditor.md via MD5 dedup. Reports show distinct analytical focus: alignment investigated tile-selection blind spots, rigor tracked 59 claims, benchmarking performed cross-site validation, optimization characterized K-tile regret, internal-auditor patched the LDS filter. See `key_result_internal-auditor.png`.

- **Risk flagging coverage: 20%** (4/20) [VERIFIED] — across 4 process-integrity questions × 5 teams: (a) 16-shape vs 1,863-shape disconnect flagged by 2/5 teams (benchmarking, internal-auditor), (b) Top-1=6.2% called out as a concern by 2/5 teams (alignment, optimization), (c) 17.9% avg regret explicitly flagged as concerning by 0/5 teams, (d) K-tile scope deferral questioned by 0/5 teams. Measured via regex pattern matching against report text.

- **Headline numbers spot-checked from source artifacts** [VERIFIED] — Spearman ρ=0.8438 recomputed from `banff_data/correlation_results.json` (16 shapes, delta=0.000044); Top-1=6.2% confirmed (1/16 shapes); K=3 CV test regret=3.06% ± 0.13% confirmed from `cross_validation_report.md` Table 1; LDS crash-risk=0/42 confirmed from `analysis_results.md` Section 3 (5,661 configs, 42 shapes).

- **Avg regret discrepancy: 17.51% raw vs 17.9% reported** [VERIFIED] — the raw `correlation_results.json` aggregate shows avg_regret=17.51%, while `analysis_results.md` reports 17.9%. Delta=0.39pp is explained by the alignment team's finding that 4/16 shapes use an updated reranker with different tile picks. Both values are internally consistent with their respective data sources.

- **Scope deferral unchallenged** [VERIFIED] — `cross_validation_report.md` contains 3 explicit "out of scope for K-016" scope notices wrapping all K-tile CV analysis (Tables 1, 2, and Origami Comparison). Zero of 5 team reports questioned whether this scoping appropriately defers K-tile work or conveniently excludes the strongest evidence (K=3 achieving 3.06% vs origami's 17.9%).

## Recommended Next Steps

1. Before accepting K-016, require explicit sign-off on the scope boundary: "K-tile lookup achieving 5.8× lower regret than origami is informational only and deferred to K-018." The current state risks closing K-016 with a known 17.9% regret gap that nobody formally owned.

2. For the next multi-team review cycle, add a mandatory "Risk Register" section to each team's report template requiring teams to enumerate what concerns them (not just what they verified), to improve risk coverage above the current 20%.

3. None — no code changes needed. This is a process finding, not a technical one.

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — Process integrity heatmap (risk flags × teams) and report independence bar chart. Source: 5 team reports, branch k016/triton-specialization-in-origami-internal-auditor.
- `internal-auditor/process_integrity_audit.json` — Per-team metrics (word count, [VERIFIED] tags, risk flags) for all 5 reports.
- `internal-auditor/process_integrity_summary.json` — Aggregate risk-flag coverage (4/20 = 20%).
- `internal-auditor/verified_spot_checks.json` — Spot-check verification of 5 headline numbers against source artifacts.

## Method

Parsed 5 team report markdown files using Python regex and MD5 hashing to detect (a) exact sentence duplication (0 found), (b) risk-flag coverage via keyword patterns for 4 audit questions, and (c) spot-checked 5 headline metrics against their source JSON/CSV artifacts. Generated `key_result_internal-auditor.png` with matplotlib. All analysis performed on local report files — no GPU required.
