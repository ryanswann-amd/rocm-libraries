# K-016 Internal Audit Report — Triton Specialization in Origami

**Auditor**: #internal-auditor | **Date**: 2026-04-12 | **Cycle**: 2 (PM round 1)
**Branch**: `k016/triton-specialization-in-origami-internal-auditor`

---

## Bottom Line

**CONDITIONAL-GO on K-016 closure.** All 13 audited numerical claims reproduce exactly from GPU-measured MI300X data (0/13 deltas exceed 0.1pp). The LDS staged filtering formula is correctly integrated into origami's tile-selection codepath and produces 0% crash risk at stage-2 for the K=3 tile set. However, the cost model's 6.2% Top-1 accuracy and 17.9% Banff regret confirm K-016 delivers LDS safety filtering—not a production tile selector. K-018 must deliver K≥4 tile selection before origami tile selection is production-ready. Process cost across 3 rounds: $120.74 with 46.2% acceptance rate (6/13 team-rounds).

## Key Results

- **13/13 claims verified, zero delta >0.1pp** — Spearman ρ=0.8438, K=3 mean regret=3.06%, LDS crash risk=0/42 tiles, Top-1=6.2%, CV std=0.13% all reproduced from source artifacts (`correlation_results.json`, `k3_per_shape_regret.csv`, `lds_21tile_computation_trace.json`) [VERIFIED] on MI300X gfx942 (banff-cyxtera, OCI). Full ledger: `k016_audit_ledger.csv`.

- **Cost-model root causes identified by kernel-opt** — Three defects drive the top-5 highest-regret shapes (19–36%): wave quantization blindness (shapes #1, #5), deep-K blindness for M=1 decode (shapes #2, #3), and missing 256×256 tile in K=3 set (shape #4, LDS-filtered at 128KB). All root causes traced to specific cost-model proxy inversions in `regret_decomposition_raw.csv` [VERIFIED] on MI300X banff-cyxtera-s70 (259-tile sweeps × 3 GPUs).

- **FLOP-weighted regret 1.48× worse than shape-weighted** — K=3 FLOP-weighted regret is 4.54% vs 3.06% shape-weighted, because compute-bound shapes (5.94% regret, 46.9% of production FLOPs) dominate real-world cost. K=5 reduces FLOP-weighted regret to 1.35%, saving $1.63/GPU/day at $2/GPU-hr [VERIFIED] from `benchmarking/final_verified_analysis.json` (80,109 MI300X measurements, 1,863 shapes). See `key_result_internal-auditor.png`.

- **Process cost: $120.74 across 3 rounds, 46.2% acceptance rate** — R0a: $34.13 (0/4 accepted, all exhausted); R0b: $34.94 (3/4 accepted); R1: $51.67 (3/5 accepted, benchmarking and kernel-opt exhausted at cycle 2). R0a wipeout caused by hard-gate rejection on 22 estimation tags; 6 were unverified data claims in `alignment/stage2_tile_validation.md`, remainder were meta-references [VERIFIED] from task chat history system messages with timestamps.

- **Cross-team risk-flag coverage: 11/36 (30.6%)** — Top-1=6.2% flagged by only 1/9 teams (internal-auditor). K-018 scope deferral flagged by 2/9 teams. Max pairwise Jaccard=0.0134 (benchmarking vs rigor), confirming team independence [VERIFIED] from `process_integrity_round3.json`.

## Recommended Next Steps

1. **Accept K-016 with two conditions**: (a) K-018 must deliver K≥4 tile set with <5% mean Banff regret before declaring origami tile selection production-ready; (b) define a production regret SLO (current K=3 baseline: 3.06% mean, P95=16.51%). Owner: Product/SRE.

2. **Prioritize K=5 tile expansion in K-018**: Adding `16x16x256` + `128x256x64` cuts FLOP-weighted regret from 4.54% to 1.35% and tail shapes >10% from 211 to 37. Ready-to-validate script: `python benchmarks/run_k5_spotcheck.py`

3. **Fix wave quantization + deep-K cost-model defects in K-018**: These two defects account for 4/5 top-regret shapes. Concrete fix: add grid-size/CU occupancy penalty and M-threshold deep-K heuristic for decode shapes (prior validation: decode regret 23.3% → 3.7% on 171 M=1 shapes).

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 4-panel dashboard: risk register, Banff regret distribution, K-scaling curve, process cost waterfall. MI300X gfx942, branch k016/triton-specialization-in-origami-internal-auditor. Command: `python3 audit_dashboard.py`.
- `internal-auditor/k016_audit_ledger.csv` — 13 claims with claimed vs recomputed values, deltas, sources.
- `internal-auditor/cost_ledger_verified.json` — Per-round, per-team cost breakdown across 3 rounds ($120.74 total).
- `internal-auditor/cross_team_reconciliation.csv` — 5-team verdict summary with conditions and acceptance status.
- `internal-auditor/process_integrity_round3.json` — Cross-team independence matrix and risk-flag coverage (11/36).

## Method

Synthesized findings from alignment (CONDITIONAL-GO, 5 metrics verified), rigor (ALL 4 CHECKPOINTS PASS, 17,683 rows formula-verified), benchmarking (FLOP-weighted cost analysis, 80,109 measurements), and kernel-opt (3 cost-model defects root-caused across 259-tile sweeps). All headline numbers cross-verified against `k016_audit_ledger.csv` (13 claims, 0 deltas >0.1pp). Process costs tallied from orchestrator system messages. Figure generated from `correlation_results.json` and `k3_per_shape_regret.csv`.
