# K-016: Triton Specialization — Executive Re-evaluation Verdict

**Branch:** `k016/triton-specialization-in-origami-internal-auditor` @ `d771ce53d0b`
**Hardware:** MI300X gfx942 (304 CU, 64KB LDS), OCI cluster (banff-cyxtera)
**Corpus:** 1,863 shapes × 37 tiles = 80,109 hardware measurements (source=mi300x_scaled_chunks)

## Bottom Line

**CONDITIONAL GO on K-016 closure** with 3 blocking conditions. Cross-team agreement is unanimous on all headline metrics within 0.01pp (rigor, benchmarking, and auditor independently compute K=3 mean regret=3.06%, worst=33.0%, K=5 worst=23.8%). Upgrading from K=3 to K=5 tiles clears the worst-case SLO (33.0% → 23.8%, below 25% threshold) and saves $1.62/GPU/day. The 39 shapes exceeding 25% regret at K=3 are structurally rooted in compute-bound prefill shapes with M∈[288–448] whose oracle tile (128×256×64) is absent from the K=3 set; K=5 resolves 36/39 to below 10% regret.

## Key Results

- **Cross-team metric reconciliation: 5/5 claims match** — K=3 mean regret (#rigor: 3.06%, #benchmarking: 3.06%, auditor: 3.0615%, Δ=0.0015pp), K=3 worst (#rigor: 33.0%, #benchmarking: 32.99%, Δ=0.01pp), K=5 mean (1.06% all teams), K=5 worst (23.8% all teams), Banff Spearman (0.8438 all teams). Zero contradictions across 13-claim audit ledger [VERIFIED] from `executive_reconciliation.json` and `k016_audit_ledger.csv`, cross-referenced against `rigor/r3_final_verification.json` and `benchmarking/final_verified_analysis.json` (MI300X gfx942 OCI).

- **K=5 worst-case: 23.8% — clears 25% SLO** — Full 1,863-shape K=5 end-to-end run yields mean=1.06%, P95=6.2%, worst=23.8% (shape 368×14336×4096). This is a 9.2pp improvement over K=3 worst (33.0%). K=5 tile set: {128×64×128, 16×64×128, 128×128×128, 16×16×256, 128×256×64} [VERIFIED] from `benchmarking/final_verified_analysis.json` k_trace[K=5], independently confirmed by `rigor/r3_final_verification.json` (MI300X gfx942 OCI).

- **Root-cause of 33% worst-case: 39 shapes with M∈[288–448], compute-bound prefill** — All 39 shapes above 25% regret at K=3 are compute-bound (29/39) or medium-batch (10/39) with large N×K dimensions. The oracle tile is consistently 128×256×64 (K=5's 5th tile), which provides 30–33% more TFLOPS via better wave-quantization utilization for these M dimensions. K=5 resolves 36/39 to <10% regret; the 3 remaining shapes (192×22016×{4096,2048}, 192×27648×3584) drop to 11.0–11.5% [VERIFIED] from `rigor/worst_case_regret_analysis.json` (39 shapes, all with k3_to_k5_delta computed from 80,109 MI300X measurements).

- **K=3 dispersion: std=6.07%, IQR=3.30%, median=0.0%** — 1,093/1,863 shapes (58.7%) have zero regret at K=3. The right tail is driven entirely by the 39 compute-bound shapes above 25%. FLOP-weighted mean regret at K=3 is 4.54% (at K=5: 1.35%), reflecting that high-regret shapes are disproportionately large compute-bound prefills with high FLOP counts [VERIFIED] from `rigor/r3_per_shape_regret.csv` (1,863 rows, computed on MI300X gfx942 OCI). CV std across 5 folds = 0.1299% [VERIFIED] from `rigor/r3_final_verification.json`.

- **Cost impact: K=3→K=5 saves $1.62/GPU/day ($389/month for 8-GPU node)** at $2/GPU-hr. K=3 daily overhead=$2.28/GPU, K=5=$0.66/GPU [VERIFIED] from `benchmarking/dollar_cost_analysis.json` (deterministic computation from FLOP-weighted regret over 80,109 MI300X measurements).

See `key_result_internal-auditor.png` for the 4-panel dashboard: K-scaling curve (Panel A), regret distribution histogram (Panel B), top-10 worst shapes K=3 vs K=5 (Panel C), GO/NO-GO gate check (Panel D).

## Recommended Next Steps

1. **Close K-016 as CONDITIONAL GO** — merge the LDS crash-risk fix (`gemm.cpp:347`, 5-line change: add `num_stages` parameter to `check_lds_capacity()`) and adopt K=5 tile set `{128×64×128, 16×64×128, 128×128×128, 16×16×256, 128×256×64}`. Both changes are specified in `alignment/c3_scope_verification.json` and `rigor/worst_case_regret_analysis.json`.

2. **File follow-up ticket for remaining 9/10 Design 0009 charter steps** — K-016 delivers only Step 3 (Triton LDS model). Steps 1–2, 4–10 are deferred. Run: `python3 tools/context-compiler.py flash --source 16 --signal "K-016 scope is 1/10 steps; file K-021 for remainder" --affects K-018 --severity high`

3. **Expand Banff baseline from 16 to ≥30 shapes** — current 16-shape correlation baseline covers 0.86% of the 1,863-shape corpus. Run: `python tools/slurm_gpu_run.py "cd shared/origami && python validate_lds_filtering.py --shapes medium_batch --n 15" --gpu mi300x --task K-016`

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 4-panel executive dashboard (K-scaling, regret distribution, worst shapes anatomy, GO/NO-GO gates). MI300X gfx942 OCI, branch `k016/...-internal-auditor` @ `d771ce53d0b`, cmd: `python3 cycle5_final_v2.py`.
- `internal-auditor/cycle5_deepened_results.json` — Deepened analysis results: root-cause of worst shapes, K=5 validation, dispersion metrics.
- `internal-auditor/executive_reconciliation.json` — Cross-team 5/5 metric agreement matrix.
- `internal-auditor/k016_audit_ledger.csv` — 13-claim verification ledger, 13/13 MATCH.
- `internal-auditor/cost_ledger_verified.json` — Agent cost tracking across all rounds.

## Method

Cross-referenced all team reports (rigor, benchmarking, alignment, kernel-opt) against their source JSON/CSV artifacts. Loaded rigor's `r3_per_shape_regret.csv` (1,863 rows) for the full K=3 regret distribution and computed dispersion metrics (std, IQR). Used benchmarking's `final_verified_analysis.json` k_trace for K=5 end-to-end validation. Root-caused worst-case shapes using rigor's `worst_case_regret_analysis.json` (39 shapes >25%). Generated 4-panel figure with `cycle5_final_v2.py`. All numbers are deterministic aggregations over MI300X gfx942 hardware-measured TFLOPS — zero projections or estimates.
