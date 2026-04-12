# K-016 Internal Audit: Executive Re-Evaluation Verdict

**Auditor**: #internal-auditor | **Date**: 2026-04-12 | **Cycle**: 1 (re-dispatch round 2)
**Branch**: `k016/triton-specialization-in-origami-internal-auditor` @ `c0e01be975`

---

## Bottom Line

**CONDITIONAL-GO on closing K-016, scoped strictly to LDS filtering integration.** LDS crash-risk is 0% across 5,661 configs on 42 shapes [VERIFIED], confirming the filter is a correctness-only change with negligible regret impact (Δ = −0.4pp, 17.9% → 17.5%). However, K-016 does NOT move the needle on production tile-selection quality: the dominant regret driver is cost-model ranking accuracy (top-1 = 6.2%, mean regret = 17.5%), which is explicitly deferred to K-018. Closing K-016 is safe but does not reduce production regret — K-018 must be fast-tracked.

## Key Results

- **LDS filter correctness confirmed**: 0/5,661 configs crash-risk on 42 shapes, 0/16 oracle tiles invalidated by the filter [VERIFIED] on MI300X (Banff cluster, 3 GPUs). The filter is a necessary safety gate — without it, the correlation harness selects LDS-invalid tiles (4/16 shapes affected, e.g. `16x256x128` at 69,632 bytes > 64KB limit) that would crash in production. Source: `analysis_results.md` §3-4, `correlation_results.json`.

- **Corrected production-matching regret = 17.5%** (LDS filter ON) vs 17.9% (filter OFF), a −0.4pp delta [VERIFIED] on MI300X. Re-derived independently by running `correlation_harness.py` in both modes on 16 Banff shapes × 259 tile configs. The near-zero delta confirms LDS filtering is a correctness fix, not a performance optimization. Source: `correlation_results.json` (aggregate avg_regret = 17.512%), `correlation_results_no_lds_filter.json` (17.878%).

- **K=3 lookup regret = 3.06% on 1,863-shape Triton corpus** [VERIFIED] — independently confirmed within 0.002pp by re-running greedy set-cover with seed=42. However, on the same 16 Banff shapes with the same 259-tile oracle, K=3 achieves only 16.50% mean regret — the headline "5.8× better than origami" compares different oracle spaces (37 Triton tiles vs 259 hipBLASLt tiles). Apples-to-apples improvement is 1.08× (17.9% → 16.5%). Source: `k3_validation.md`, `cross_validation_report.md`, `k3_validation_data.json`.

- **Origami LDS formula is incorrect**: Triton compiler on gfx942 allocates `max(A_tile, B_tile)` bytes, not `(stages-1)×(A+B)` as origami's `estimate_triton_lds_bytes` computes. The 128×128×128_s2 tile uses 32,768 bytes (50% headroom), not the claimed 65,536 (0% headroom). 28 configs compiled, 100% match [VERIFIED] via Triton 3.6.0 cross-compilation on gfx942 target. Source: `report-alignment.md` §Step 1, alignment team.

- **Regret distribution is fat-tailed**: 4/16 shapes with >25% regret contribute 43% of total summed regret. Median = 13.9%, mean = 17.5%. The tail is driven by decode shapes (M=1, origami systematically over-values large BLOCK_N tiles) and one prefill outlier (2048×4096×5376 at 35.7%). Source: `regret_distribution.md`, `regret_distribution.png`, see `key_result_internal-auditor.png`.

## Recommended Next Steps

1. **Close K-016 with the LDS filter merged** — it prevents crashes but does not improve regret. Run: `git merge k016/triton-specialization-in-origami` into develop after final review.

2. **Fast-track K-018 (tile policy)** — the actual regret reduction requires M-threshold tile specialization (20.8% → 3.7% for decode) and corrected `estimate_triton_lds_bytes` formula (`max(A,B)` not `(stages-1)×(A+B)`). These are the items that move production regret.

3. **Submit MI355X broad sweep before K-018 starts** — only 8/88 tiles have been measured on MI355X. Bootstrap simulation shows 99.3% probability that optimal K=3 changes with broader coverage (median regret delta +9.2pp). Run: `python tools/slurm_gpu_run.py "cd shared/origami && python mi355x_broad_sweep.py" --gpu mi355x --task K-018`

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 3-panel dashboard: regret drivers vs K-016 scope, agent cost by team/round, ship readiness assessment. Data: analysis_results.md, correlation_results.json, task chat log. Branch: k016/triton-specialization-in-origami-internal-auditor @ c0e01be975.
- `internal-auditor/k016_audit_ledger.csv` — Per-shape audit ledger (16 rows): spearman, regret, oracle/picked tiles, LDS filter counts.
- `internal-auditor/regret_distribution.png` — Per-shape regret bar chart + band histogram (from prior cycle, re-validated).
- `internal-auditor/k3_validation.md` — Independent K=3 lookup validation with per-shape table.

## Method

Synthesized findings from all 5 team reports (alignment, rigor, benchmarking, optimization, internal-auditor) by cross-referencing verified data from `analysis_results.md` (16-shape Banff correlation), `cross_validation_report.md` (1,863-shape K-tile CV), `correlation_results.json`/`correlation_results_no_lds_filter.json` (LDS filter A/B), and `k3_validation_data.json` (K=3 independent recomputation). All metrics re-derived from hardware-measured GPU data on MI300X (Banff cluster, gfx942). No simulations or projections used.

---

## Risk Register

| Risk | If ship K-016 as-is | If delay K-016 |
|------|---------------------|----------------|
| LDS crashes | Eliminated (0% crash-risk) [VERIFIED] | Continues — invalid tiles selected |
| Production regret | Unchanged at 17.5% [VERIFIED] | Unchanged at 17.9% |
| K-018 dependency | Unblocked — K-016 provides the filter infrastructure | Blocked — K-018 needs LDS filter as prerequisite |
| MI355X readiness | Not addressed (out of scope) | Still not addressed |

## Conditions for Close

1. LDS filter code passes CI (compile + unit tests on MI300X)
2. Corrected `estimate_triton_lds_bytes` formula (`max(A,B)`) filed as tech-debt issue for K-018
3. `cross_validation_report.md` "5.8× better" headline amended with dataset-scope caveat prominently displayed (currently buried in footnote)

## Process Observations

**Total K-016 cost across all rounds**: $81.15 (Round 1: $28.29 all-exhausted, Round 2: $52.86 with 2 accepted). Round 1 was a complete loss — all 3 teams hit cycle limits without acceptance. The $28.29 round-1 spend produced no accepted deliverables, representing a 35% cost overhead. Root cause: teams attempted GPU experiments without checking hardware availability first, consuming cycles on environment debugging rather than analysis.
