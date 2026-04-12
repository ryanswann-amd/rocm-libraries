# K-016: Triton Specialization — Executive Re-evaluation Verdict

**Branch:** `k016/triton-specialization-in-origami-internal-auditor` | **Commit:** see git log
**Hardware:** MI300X gfx942 (304 CU, 64KB LDS), OCI cluster
**Corpus:** 1,863 shapes × 37 tiles = 80,109 hardware measurements

## Bottom Line

**CONDITIONAL GO on K-016 closure** with 3 blocking conditions. All 4 teams converge on headline metrics (K=3 mean regret=3.06%, worst=33.0%, Spearman ρ=0.8438) — cross-team agreement within 0.01pp on every number [VERIFIED]. However, 3 of 7 acceptance gates FAIL: worst-case regret exceeds 25% threshold (33.0%), scope delivery is 1/10 charter steps, and the Banff 16-shape baseline covers only 0.86% of the corpus. K-016 should be closed as a correctness-only deliverable, not as the full Triton specialization milestone.

## Key Results

- **Cross-team metric reconciliation: 6/6 claims match** — K=3 mean regret (#rigor: 3.06%, #benchmarking: 3.06%, auditor recomputed: 3.0615%), K=3 worst (#rigor: 33.0%, #benchmarking: 32.99%), K=5 mean (1.06% both teams), Banff Spearman (0.8438 both), LDS bug confirmed by both #alignment and #kernel-opt, scope 1/10 confirmed by #alignment. Zero cross-team contradictions found [VERIFIED] from `k016_audit_ledger.csv` (13 claims, 13/13 MATCH) and `executive_reconciliation.json`.

- **GO/NO-GO gate: 2 PASS, 3 FAIL, 1 CONDITIONAL, 1 INFO** — PASS: K=3 mean regret 3.06% ≤ 5% threshold; CV generalization gap 0.00pp ≤ 1pp. FAIL: worst-case 33.0% > 25%; scope 10% < 80%; Banff coverage 0.86% < 10%. CONDITIONAL: LDS crash risk is 0% with flat formula but 38.3% of oracle tile selections fail under staged pipelining. See Panel D of key figure [VERIFIED] from `spot_check_full_population.json` (1,863/1,863 shapes checked, 0 failures on 4 check types).

- **Cost impact: K=3 leaves $2.28/GPU/day on the table; K=5 reduces to $0.66/GPU/day** — at $2/GPU-hr, the K=3→K=5 upgrade saves $390/month for an 8-GPU node. FLOP-weighted regret drops from 4.54% to 1.35% [VERIFIED] from `benchmarking/dollar_cost_analysis.json` (deterministic arithmetic from 80,109 MI300X measurements).

- **Process cost: $120.74 total across 13 team-rounds, 46.2% acceptance rate** — Round 0a: 4/4 teams exhausted ($34.13); Round 0b: 3/4 accepted ($34.94); Round 1: 3/5 accepted ($51.67). The #kernel-opt and #internal-auditor teams hit max-cycles in round 1, driving 54% of rejections [VERIFIED] from task chat history system messages (timestamps 11:27:19–13:04:42 on 2026-04-12).

See `key_result_internal-auditor.png` for the 4-panel dashboard (K-scaling curve, category regret waterfall, risk register, GO/NO-GO gate check).

## Recommended Next Steps

1. **Close K-016 as CONDITIONAL GO** — merge the LDS crash-risk fix (`gemm.cpp:347`, 5-line change adding `num_stages` parameter to `check_lds_capacity()`) and adopt K=5 tile set `{128×64×128, 16×64×128, 128×128×128, 16×16×256, 128×256×64}`. Both changes are specified in #alignment's `c6_alignment_audit_results.json` and #rigor's `worst_case_regret_analysis.json`.

2. **File follow-up ticket for the remaining 9/10 Design 0009 charter steps** — K-016 delivers only Step 3 (Triton LDS model). Steps 1–2, 4–10 (config_t, heuristics, work-stealing grid, Python bindings, A/B comparison, tritonblas migration, gfx950) are deferred. Scope this as K-018 successor work.

3. **Expand Banff baseline from 16 to ≥30 shapes** — add ≥10 shapes from `medium_batch` category (currently zero representation for 39% of corpus). Run: `python tools/slurm_gpu_run.py "cd shared/origami && python validate_lds_filtering.py --shapes medium_batch" --gpu mi300x --task K-016`

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 4-panel executive dashboard (K-scaling, category waterfall, risk register, GO/NO-GO gates). MI300X gfx942 OCI, `python3 executive_synthesis.py`.
- `internal-auditor/executive_reconciliation.json` — Cross-team metric agreement matrix with 6/6 matches.
- `internal-auditor/k016_audit_ledger.csv` — 13-claim verification ledger, all MATCH.
- `internal-auditor/spot_check_full_population.json` — 1,863-shape full-population spot-check (4 check types × 1,863 shapes = 7,452 checks, 0 failures).
- `internal-auditor/cost_ledger_verified.json` — Agent cost tracking across all rounds.

## Method

Cross-referenced all 4 team reports (alignment, rigor, benchmarking, kernel-opt) against their source artifacts. Recomputed K=3 mean regret independently from `merged_full_corpus.csv` (80,109 MI300X rows) and confirmed 3.0615% matching all teams within 0.01pp. Ran `executive_synthesis.py` to generate the 4-panel dashboard from verified JSON artifacts. Process costs extracted from task chat history system messages. All numbers are deterministic aggregations over hardware-measured TFLOPS — no projections or estimates.
