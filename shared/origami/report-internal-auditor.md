# K-016 Internal Audit Report — Process Integrity & Spot-Check Verification (Cycle 2)

**Date**: 2026-04-12 | **Team**: #internal-auditor | **Project**: origami
**Branch**: `k016/triton-specialization-in-origami-internal-auditor`
**Source data**: `banff_data/correlation_results.json` (16 shapes × 259 configs, MI300X gfx942)

---

## Bottom Line

The K-016 review process has a serious risk-flagging gap: 0/5 teams called out the 17.9% average regret as concerning, and 0/5 questioned the K-tile scope deferral to K-018. Claimed metrics recompute correctly from raw data (Spearman ρ=0.8438 [VERIFIED], Top-1=6.2% [VERIFIED], K=3 CV=3.06%±0.13% [VERIFIED]), but the average regret recomputes to 17.51% — a 0.39pp delta from the claimed 17.9%, attributable to reranker adjustments on 4 shapes. The LDS 0/42 crash-risk claim is technically correct but misleading: all 131 shapes in the LDS validation set are decode (M≤1), providing zero coverage of prefill workloads where regret is worst.

## Key Results

- **Spearman ρ = 0.8438** — recomputed from 16 per-shape ρ values in `banff_data/correlation_results.json`. Mean of [0.9094, 0.8312, 0.8587, 0.8141, 0.8525, 0.7872, 0.7285, 0.8428, 0.8324, 0.8356, 0.8578, 0.8871, 0.7908, 0.8436, 0.9000, 0.9284] = 0.8438. Delta from claim: <0.0001. [VERIFIED]

- **Top-1 = 6.2%, regret = 17.51%** — only 1/16 shapes (12288×9472×32768) picks the oracle tile; recomputed mean regret is 17.51%, not 17.9% as headline-claimed. The 0.39pp delta traces to 4 shapes using the updated analytical reranker. This is within the >0.1pp flagging threshold. [VERIFIED]

- **Risk-flag coverage = 20% (4/20)** — across 5 teams × 4 critical risks, only 4 flags were raised. Zero teams flagged 17.9% regret as concerning. Zero questioned the scope deferral. See `key_result_internal-auditor.png` risk heatmap (Panel 2). [VERIFIED] from regex scan of all team report text in `process_integrity_audit.json`.

- **LDS 0/42 = decode-only** — the 42-shape bf16_NN_mi300x LDS validation set contains exclusively M=1 shapes. The full 131-shape LDS set across 6 datasets is 100% decode. No prefill (M>256) shapes were tested despite prefill having the highest mean regret (18.0%). [VERIFIED] from `lds_analysis_results.json`.

- **BLOCK_N overvaluation root cause** — 4/5 worst-regret shapes show origami picking tiles with larger BLOCK_N than the hardware oracle (e.g., picked 256×224×64 vs oracle 256×128×64 for the 35.7% worst case at 2048×4096×5376). The cost model systematically overvalues BLOCK_N, penalizing latency-sensitive shapes. [VERIFIED] from per-shape tile comparison in `fat_tail_autopsy.csv`.

## Recommended Next Steps

1. **Flag the 17.51% vs 17.9% regret discrepancy** — the 0.39pp delta exceeds the 0.1pp threshold. The headline number should be corrected to 17.5% or the reranker adjustment methodology documented: `Update analysis_results.md to reflect recomputed 17.51% regret from correlation_results.json`.

2. **Expand LDS validation to include prefill shapes** — the current 0/42 claim provides false confidence by testing only decode. Run: `python3 tools/slurm_gpu_run.py "cd shared/origami && python banff_data/correlation_harness.py --shapes prefill --lds-check" --gpu mi300x --task K-016`

3. **Require all teams to explicitly address regret and scope deferral in future reviews** — add a mandatory risk-checklist to the PM gate: "Does 17.9% regret meet production SLO? Is scope deferral to K-018 justified?"

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 4-panel dashboard: per-shape regret, risk-flag heatmap, ρ-vs-regret scatter, test set representativeness
- `internal-auditor/regret_distribution_histogram.png` — regret distribution with K=3 CV baseline overlay
- `internal-auditor/cycle2_verification_results.json` — full recomputation output with deltas
- `internal-auditor/process_integrity_audit.json` — per-team risk-flag scan results
- `internal-auditor/lds_audit_summary.json` — 21-tile LDS budget verification
- `internal-auditor/fat_tail_autopsy.csv` — top-5 worst-regret shapes with root causes

## Method

Recomputed all headline metrics (Spearman ρ, Top-1, regret) directly from `banff_data/correlation_results.json` using Python arithmetic on the 16 per-shape metric records. Process integrity assessed by regex-scanning all 59 team reports for risk-flag keywords (16-vs-1863 disconnect, Top-1 concern, regret concern, scope deferral). LDS representativeness evaluated by parsing the shape M-dimension distribution in `lds_analysis_results.json` across all 6 datasets (131 unique shapes, 100% M≤1).
