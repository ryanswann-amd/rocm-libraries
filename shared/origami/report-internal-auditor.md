# K-016 Internal Audit Report — Triton Specialization in Origami

**Auditor**: #internal-auditor | **Date**: 2026-04-12 | **Cycle**: 1 (executive re-evaluation)
**Branch**: `k016/triton-specialization-in-origami-internal-auditor`

---

## Bottom Line

K-016's technical deliverables are sound: all 13 audited numerical claims reproduce exactly from GPU-measured data on MI300X gfx942 [VERIFIED]. The LDS filtering formula in `gemm.cpp:339-354` matches Triton's compiler allocation and is correctly called during tile selection at `origami.cpp:544`. However, the orchestration process has been expensive — $144.40 across 4 dispatch rounds, with round 1b ($40.39) consuming 28% of total cost on cycles_exhausted outcomes for all 4 teams. The top-of-ranking failure (6.2% Top-1 accuracy) remains the primary unresolved performance gap and is the highest-impact item for K-018.

## Key Results

- **13/13 claims verified with zero delta** — Every headline number (Spearman ρ=0.8438, K=3 mean regret=3.06%, LDS crash risk=0/42, Top-1=6.2%, CV std=0.13%) reproduced from source artifacts (`correlation_results.json`, `k3_per_shape_regret.csv`, `lds_validation_results.json`) [VERIFIED] on MI300X gfx942. Full trace in `k016_audit_ledger.csv`.

- **LDS formula correctness confirmed** — Independent computation of all 21 tile LDS budgets matches `check_lds_capacity()` output exactly. 3/21 tiles pass 3-stage on MI300X (64KB limit). The formula `(stages-1) * (M*K*elem_bytes + K*N*elem_bytes)` at `gemm.cpp:344-347` is identical to Triton's `MatmulLoopPipeline.cpp:402` [VERIFIED]. See `lds_21tile_computation_trace.json`.

- **Top-of-ranking failure mechanistically confirmed** — Top-5 Spearman is negative or near-zero in 5/6 worst-regret Banff shapes while full Spearman stays >0.8. Worst case: shape `1536x3584x3584_f16_r` has predicted rank-1 tile at actual rank 83 (rank error=82) [VERIFIED] from `regret_decomposition_raw.csv` + benchmark chunks. Root cause: cost-model's wide-N/shallow-K bias at `origami.cpp:570-676`. See key result figure.

- **Process cost: $144.40 total across 4 rounds** — Round 0a ($34.13) and 0b ($34.94) reached acceptance for rigor/benchmarking/internal-auditor. Round 1 triggered PM hard gate fail on 16 estimation-tagged claims (all were meta-references, not data claims). Round 1b ($40.39) exhausted all 4 teams' cycles without new acceptance [VERIFIED] from task chat history timestamps.

- **Risk flag coverage: 11/36 cells (30.6%)** — Only internal-auditor flagged Top-1 accuracy as a concern. 0/9 teams flagged high regret independently. Cross-team independence is healthy (max Jaccard=0.0134 between benchmarking/rigor) but risk-flag coverage is low [VERIFIED] from `process_integrity_round3.json`.

## Recommended Next Steps

1. **Accept K-016 deliverables** — All numerical claims are verified, LDS filtering is correctly integrated, and the K=3 tile set generalizes (0% CV gap). The remaining regret (17.9% on Banff 16 shapes, 3.06% on 1,863-shape corpus) is a cost-model accuracy issue, not a correctness bug.

2. **Open K-018 for top-of-ranking improvement** — The 6.2% Top-1 accuracy and wide-N bias are the dominant regret drivers. Concrete starting point: the 5/6 worst shapes with negative Top-5 Spearman in `k016_cycle3_rank_analysis.csv`, and the 39 fat-tail shapes (>25% regret, 36/39 prefill-class) in `fat_tail_autopsy_full.csv`.

3. **Reduce re-dispatch waste** — R1 hard gate rejection was caused by meta-references to estimation tags in policy statements, not actual estimated data. Recommend the PM gate scanner exclude patterns like "Zero [EST...]" to avoid $40+ false-positive re-dispatch costs.

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — Cost waterfall + claim verification matrix (MI300X gfx942, `python3 audit_dashboard.py`)
- `internal-auditor/k016_audit_ledger.csv` — 13 claims with claimed vs recomputed values, deltas, sources
- `internal-auditor/k016_cycle3_rank_analysis.png` — Predicted vs actual rank scatter for 6 worst-regret shapes
- `internal-auditor/k016_cycle3_rank_analysis.csv` — 30-row per-shape top-5 tile rank comparison
- `internal-auditor/lds_21tile_computation_trace.json` — Full LDS computation for all 21 tiles
- `internal-auditor/cost_ledger_verified.json` — Per-round, per-team cost breakdown
- `internal-auditor/process_integrity_round3.json` — Cross-team independence and risk flag coverage
- `internal-auditor/top1_impact_analysis.json` — Per-shape Top-1 hit/miss with TFLOPS impact
- `internal-auditor/verified_spot_checks.json` — Spot-check verification of 5 headline numbers

## Method

Audited K-016 by (1) extracting all 13 verifiable numerical claims from team reports into `k016_audit_ledger.csv`, (2) recomputing each from source data files (`correlation_results.json`, `k3_per_shape_regret.csv`, `lds_21tile_computation_trace.json`, `regret_decomposition_raw.csv`), and (3) comparing deltas. Verified LDS formula by reading `gemm.cpp:339-354` source and independently computing LDS for all 21 tiles. Cost accounting derived from task chat history system messages with timestamps. All data traces to GPU-measured TFLOPS on MI300X gfx942 (banff-cyxtera, OCI).
