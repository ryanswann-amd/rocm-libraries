# K-016 Internal Audit — Cycle 4 Final

**Branch**: `k016/triton-specialization-in-origami-internal-auditor` @ `4817fdabb4`
**Hardware**: MI300X gfx942 (304 CU, 64KB LDS) — OCI cluster (banff-cyxtera)
**Data**: 80,109+ GPU-measured rows, 1,863 shapes × 37 tiles; K=5 tiles measured in slurm/merged_full_corpus.csv
**Date**: 2026-04-12

---

## Bottom Line

K-016 is a **CONDITIONAL GO** for closure. All 13 headline claims reproduce within <0.1pp delta. K=5 tile expansion delivers mean regret 1.04% (vs K=3's 3.06%) and FLOP-weighted mean 1.34% (vs 4.54%) — both computed from MI300X hardware measurements across 1,863 shapes [VERIFIED]. The two actionable fixes (K=5 tile set + decode BLOCK_N≤64 guard) are code-ready on this branch and should ship via K-018.

## Key Results

- **K=5 mean regret = 1.04%, FLOP-weighted mean = 1.34%** — computed from `slurm/merged_full_corpus.csv` (MI300X measurements) using `benchmarking/k3_per_shape_regret.csv` oracle [VERIFIED]. K=3→K=5 reduces mean by 2.02pp, FLOP-weighted mean by 3.19pp, and adds 260 zero-regret shapes (1,093→1,353, 72.6%). See `key_result_internal-auditor.png`.

- **K=5 FLOP-weighted P95 = 8.54%** (vs K=3 = 27.26%), shape-count P95 = 6.15% (vs 16.51%) [VERIFIED]. The 39 former fat-tail shapes (K=3 mean 28.27%) collapse to K=5 mean 2.14% — 22/27 compute-bound shapes reach 0.0% regret. Source: `k5_regret_verified_1863shapes.json`.

- **Decode BLOCK_N≤64 fix validated** — measured on MI300X (commit `23a8648665`, `decode_fix_results.md`): top-2 decode shapes 29.2%→1.7% mean regret [VERIFIED]. Full 171-shape decode category: K=3 mean 3.42%→K=5 mean 0.78% (Δ=2.64pp) [VERIFIED] from corpus. All 50 decode shapes with >5% K=3 regret reach 0.0% at K=5 via the 16×16×256 tile.

- **13/13 claims reproduced** — all entries in `k016_audit_ledger.csv` match within <0.1pp delta [VERIFIED]. Cross-validation: 1,863 shapes matched between benchmarking's K=3 CSV and raw corpus, K=3 mean 3.0615% matches exactly.

- **Production regret SLO proposal** — Binding metric: **FLOP-weighted P95** because shape-count P95 overweights decode (0.11% of total FLOPs) and small-batch (3.17%) while underweighting compute-bound (46.88%) and medium-batch (49.84%) categories that dominate production workloads. Proposed thresholds: K=3 current — FLOP-wt mean ≤5.0% (actual 4.54%, PASS), FLOP-wt P95 ≤15.0% (actual 27.26%, FAIL → drives K=5 upgrade). K=5 target — FLOP-wt mean ≤2.0% (actual 1.34%, PASS), FLOP-wt P95 ≤10.0% (actual 8.54%, PASS), shape-ct P95 ≤10.0% (actual 6.15%, PASS). Remaining K=5 tail: 10 shapes >10% regret, all in medium-batch (M=192–384), root-caused to 64×128×64 oracle tiles not in K=5 set.

## Recommended Next Steps

1. **Ship K=5 tile expansion + decode guard to K-018**: Run `git cherry-pick 23a8648665` into the K-018 branch. K=5 tiles: {128×64×128, 16×64×128, 128×128×128, 16×16×256, 128×256×64}. Decode guard: 4-line clamp at `origami.cpp:549`. Both are validated on MI300X.

2. **Codify SLO in CI**: Add `origami/tests/test_regret_slo.py` asserting FLOP-weighted mean ≤2.0% and FLOP-weighted P95 ≤10.0% on the 1,863-shape corpus at K=5. Run: `python3 -c "import json; d=json.load(open('internal-auditor/k5_regret_verified_1863shapes.json')); assert d['k5']['flop_wt_mean'] <= 2.0; assert d['k5']['flop_wt_p95'] <= 10.0; print('SLO PASS')"`

3. **K=6 investigation for medium-batch tail**: The 10 residual >10% shapes (M=192, N=22016/27648) need 64×128×64 — evaluate adding this as K=6 tile if <5% additional cost-model overhead is acceptable.

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 3-panel dashboard: K-scaling curve, regret distribution (K=3 vs K=5), per-category FLOP-weighted regret. GPU: MI300X gfx942, OCI cluster. Branch: k016/triton-specialization-in-origami-internal-auditor @ 4817fdabb4. Command: `python3 internal-auditor/cycle4_figure.py`.
- `internal-auditor/k5_regret_verified_1863shapes.json` — K=3 vs K=5 aggregate statistics from MI300X corpus
- `internal-auditor/k016_audit_ledger.csv` — 13-row claim verification ledger, all MATCH
- `internal-auditor/decode_fix_results.md` — BLOCK_N≤64 decode fix before/after (commit 23a8648665)
- `internal-auditor/cycle4_analysis.py` — Recomputation script for K=5 regret from corpus
- `internal-auditor/regret_reduction_verification.txt` — 3.94pp decode fix trace

## Method

Computed K=5 per-shape regret from `slurm/merged_full_corpus.csv` (MI300X measured TFLOPS, 80K+ rows) using benchmarking team's 37-tile oracle from `k3_per_shape_regret.csv`. Cross-validated K=3 means match exactly (3.0615%). Validated decode fix via `decode_fix_results.md` (measured at commit 23a8648665, MI300X banff-cyxtera). K=5 dry-run spotcheck confirmed all 8 tile×shape combos LDS-feasible; GPU Slurm run attempted but blocked on node allocation (2 jobs queued/cancelled due to node failures). FLOP-weighted P95 SLO derived from per-shape FLOP weights (2×M×N×K) across the 1,863-shape corpus.
