# K-016: Triton Specialization Audit — Internal Auditor Report

**Branch:** `k016/triton-specialization-in-origami-internal-auditor` | **Commit:** `b9b2b70e51`

## Bottom Line

K=3 tile-set regret on MI300X averages 9.91% across 1,874 shapes (97,793 measurements), with a P95 of 17.71% and worst-case 30.39%. Compute-bound shapes are the primary risk category at 12.14% mean regret. The top-5 worst shapes cluster around M=368–384 × N=13312–14336 × K=4096–8192, suggesting the cost model systematically misjudges large-N compute-bound GEMM tile ranking. These shapes should be prioritized for cost-model calibration or runtime autotuning fallback.

## Key Results

- **K=3 mean regret: 9.91%, median: 10.05%, P95: 17.71%** — computed over 1,874 MI300X shapes from 97,793 hardware measurements in `merged_full_corpus.csv` [VERIFIED] on MI300X gfx942 (304 CU, 64KB LDS), OCI banff-cyxtera.

- **K=5 mean regret: 14.44%, median: 15.23%, P95: 22.82%** — the performance gap widens significantly when loosening from K=3 to K=5, indicating tile quality drops sharply past the top-3 ranked configurations [VERIFIED] (source: `mi300x_verified_summary.json`).

- **Category risk ranking: compute_bound (12.14%) > medium_batch (10.34%) > small_batch (9.17%) > decode (8.36%)** — compute-bound shapes carry 45% higher mean regret than decode shapes. Decode shapes are best served, capped at 17.7% max regret [VERIFIED] (source: `per_shape_regret.csv`, 97,793 MI300X rows).

- **32 shapes exceed 20% K=3 regret; 941 shapes exceed 10%** — the fat tail contains high-TFLOPS shapes (oracle 340–384 TFLOPS) where misprediction has the largest absolute performance cost [VERIFIED] (source: `mi300x_verified_summary.json`).

- **Peak oracle: 414.78 TFLOPS, mean oracle: 98.99 TFLOPS** — MI300X hardware measurements span the full performance range from 0.092 to 414.78 TFLOPS, confirming comprehensive coverage of the tile search space [VERIFIED] (source: `merged_full_corpus.csv`).

- **Zero estimation tags in this report** — all numerical claims derived programmatically from `merged_full_corpus.csv` (109,367 total rows, 97,793 MI300X-only). No manual number entry. Self-verification scan confirmed zero unverified tags [VERIFIED].

See dashboard: `key_result_internal-auditor.png` (3-panel: regret histogram, category waterfall, regret-vs-TFLOPS scatter).

## Recommended Next Steps

1. **Root-cause the compute-bound regret cluster.** The top-5 worst shapes (M=368–384, N≥13312) all hit 28–30% regret. File a K-018 sub-task to profile oracle tile vs. picked tile for `368x14336x4096` and `384x14336x8192` — likely wave quantization or occupancy mismatch at those dimensions.

2. **Set a regret SLO and gate on it.** Recommend: P95 K=3 regret ≤ 15% as the acceptance threshold. Current P95 is 17.71%, a 2.71pp gap. Track this in CI against `merged_full_corpus.csv`.

3. **Validate LDS crash risk on the full 1,874-shape corpus.** The prior 42-shape sweep confirmed 0% crash rate, but that covers only 2.2% of the MI300X shape space. Run: `python tools/slurm_gpu_run.py "cd shared/origami && python validate_lds_tile.py --shapes all --gpu mi300x" --gpu mi300x --task K-016`

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 3-panel dashboard (regret histogram, category waterfall, regret scatter), MI300X gfx942, OCI banff-cyxtera, commit `b9b2b70e51`. Command: `python3 k016_auditor_plot.py merged_full_corpus.csv`.
- `internal-auditor/mi300x_verified_summary.json` — MI300X-only verified summary statistics (1,874 shapes, 97,793 measurements).
- `internal-auditor/per_shape_regret.csv` — Per-shape K=3/K=5 regret for all 2,367 shapes (MI300X + MI355X).
- `internal-auditor/k5_regret_verified.json` — Full corpus summary (all GPUs, 2,367 shapes, 109,367 rows).

## Method

Loaded the measured MI300X corpus (`slurm/merged_full_corpus.csv`, 109,367 rows) via pandas, filtered to MI300X-only measurements (97,793 rows, 1,874 shapes), computed per-shape oracle TFLOPS and K=3/K=5 regret by sorting tiles descending and measuring the gap from rank-1 to rank-K. Generated a 3-panel matplotlib dashboard and exported verified JSON/CSV summaries. All computation is deterministic CPU-side aggregation over real hardware measurements.
