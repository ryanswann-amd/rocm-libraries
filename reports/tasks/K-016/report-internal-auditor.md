# K-016 Internal Audit: Cycle 2 — Fat-Tail Root-Cause Decomposition

**Auditor**: #internal-auditor | **Date**: 2026-04-12 | **Cycle**: 2
**Branch**: `k016/triton-specialization-in-origami-internal-auditor` @ `8166ef64b7`
**Hardware**: MI300X gfx942 (Banff cluster, 3 GPUs, 304 CUs)

---

## Bottom Line

The origami cost model leaves **1,563 TFLOPS on the table** across 16 measured Banff shapes (17.5% mean regret) [VERIFIED]. Five fat-tail shapes (>25% regret) account for 913 TFLOPS of that gap — dominated by two root causes: (1) the latency model undervalues memory-traffic differences between tiles with different BLOCK_N/BLOCK_K ratios (`gemm.cpp:862`: bottleneck = `max(L_compute, L_mem)` conflates tiles that differ in cache behavior), and (2) decode shapes (M≤1) get the wrong BLOCK_K entirely because the model picks wide-N/shallow-K tiles (16×256×64) over narrow-N/deep-K tiles (16×32×512) that hardware strongly prefers. K-016's LDS filter is a valid correctness fix; the performance improvement roadmap belongs in K-018.

## Key Results

- **Total TFLOPS left on table: 1,563T across 16 shapes, 913T from 5 fat-tail shapes** [VERIFIED] on MI300X (Banff gfx942). Fat-tail shapes (>25% regret) contribute 56.7% of total summed regret (158.8pp of 280.2pp). Worst case: `2048x4096x5376_bf16` — model picks 256×224×64 at 652T, oracle is 256×128×64 at 1,014T, a 362T gap. Source: `correlation_results.json`, per-shape pick/oracle GFLOPS measurements. See `key_result_internal-auditor.png`.

- **Systematic wide-N bias: model picks higher BLOCK_N/BLOCK_K ratio in 11/16 shapes** [VERIFIED]. Pearson correlation between (picked N/K − oracle N/K) and regret = 0.28. Three decode shapes show the starkest pattern: picked tile 16×256×64 (N/K=4.0) vs oracle 16×32×512 (N/K=0.06). Root cause: `gemm.cpp:862` computes `L_tile_single = max(L_compute * w_compute, L_mem * w_memory)`, which does not penalize the larger B-matrix loads of wide-N tiles when the operation is compute-bound on paper but memory-bound in practice. Source: `origami.cpp:530-676`, `gemm.cpp:718-896`.

- **CU occupancy overvaluation confirmed**: for `3600x4096x4096_bf16` (28.7% regret), the picked tile 192×256×64 achieves 100% CU occupancy (304/304 WGs) vs oracle 256×256×64 at 79% (240/304). Despite this occupancy advantage and better output utilization (0.987 vs 0.938), the oracle outperforms by 313.8 TFLOPS on hardware [VERIFIED]. The cost model's occupancy-weighted latency formula overweights grid coverage vs actual per-CU instruction efficiency. Source: `gemm.cpp:994`, `correlation_results.json`.

- **Risk-flag coverage is 26/36 (72%) under standard thresholds, not 30.6%** [VERIFIED]. The cycle 1 "11/36 = 30.6%" used lenient thresholds. Using standard quality thresholds (Spearman < 0.85, regret > 15%, top-1 = 0, NDCG < 0.85), every category except small-batch raises ≥7/9 flags. Decode is worst: 8/9 flags, 0/4 top-1, 21.9% mean regret. No prior K-series audits (K-019, K-020) used the same risk-flag matrix, so no cross-task baseline exists. Source: `correlation_results.json`, independently classified.

- **Root-cause taxonomy across all 15 non-zero-regret shapes** [VERIFIED]: tile-area overvaluation (4 shapes, 70.7pp), BLOCK_N over + BLOCK_K under (5 shapes, 119.6pp), BLOCK_N overvaluation alone (2 shapes, 55.1pp), general misrank (3 shapes, 38.0pp), BLOCK_M misrank (1 shape, 28.7pp). Source: `cycle2_analysis.json`.

## Recommended Next Steps

1. **Open K-018 with decode-shape BLOCK_K fix as P0**: clamp BLOCK_N ≤ 64 for M≤1 shapes in `rank_configs()` at `origami.cpp:544` (or add a heuristic override in `heuristics.cpp`). This targets the 2 decode fat-tail shapes (63pp combined regret). Run: `python3 tools/slurm_gpu_run.py "cd shared/origami && python3 banff_data/correlation_harness.py --shape 1x16384x16384 --block-n-cap 64" --gpu mi300x --task K-018`

2. **Merge K-016 LDS filter as-is** — it prevents 4/16 invalid tile selections and is a prerequisite for any K-018 tile-space changes. Condition: amend `cross_validation_report.md` "5.8x better" headline with dataset-scope caveat. Run: `git merge k016/triton-specialization-in-origami` into develop.

3. **Investigate CU-occupancy overweight in cost model** for K-018: the `3600x4096x4096` case proves the model overvalues grid coverage. Add `gemm.cpp:994` `compute_cu_occupancy()` to the K-018 investigation scope.

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 4-panel dashboard: per-shape regret waterfall, absolute TFLOPS gap (top 7), N/K ratio bias scatter, root cause pie chart. Data: correlation_results.json, MI300X Banff gfx942. Branch: k016/triton-specialization-in-origami-internal-auditor @ 8166ef64b7. Command: `python3 cycle2_analysis.py`.
- `internal-auditor/cycle2_analysis.json` — Machine-readable analysis output with root cause taxonomy and aggregate statistics.
- `internal-auditor/cycle2_analysis.py` — Reproducible analysis script.
- `internal-auditor/fat_tail_autopsy.csv` — Per-shape autopsy with pick/oracle tiles, LDS validity, root cause (from cycle 1, still valid).

## Method

Extracted 16-shape performance data from `banff_data/correlation_results.json` (MI300X hardware-measured, 259 tile configs per shape). Performed code-level decomposition of `gemm.cpp:718-896` (tile latency model) and `origami.cpp:530-676` (ranking + tie-breaking) to trace how each scoring component (L_compute, L_mem, utilization, occupancy) contributes to tile misranking. Classified root causes for all 15 non-zero-regret shapes by comparing picked vs oracle tile dimensions. Reconstructed risk-flag matrix under standard quality thresholds across 4 shape categories × 9 metrics.
