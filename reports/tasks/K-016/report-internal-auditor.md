# K-016 Cycle 2 Internal Audit — Triton Specialization in Origami

**Date**: 2026-04-12 | **Branch**: `k016/triton-specialization-in-origami-internal-auditor`

## Bottom Line

K-016's origami cost-model delivers Spearman ρ=0.8438 (strong ranking) but 17.9% mean regret and 6.25% top-1 accuracy — the latter is **below random baseline** (7.69% for 13-tile space) on 16 Banff shapes [VERIFIED]. The $301.02 total spend has a 43.3% waste rate ($130.21 on cycles-exhausted agents), driven by a systemic estimated-tag gate-blocker loop that caused 5 consecutive PM gate failures and 4 of 8 executive oversight triggers [VERIFIED]. The technical results are real and reproducible, but the 17.9% regret leaves significant TFLOPS on the table — the K=3 lookup table (3.06% regret on 1,863 shapes) is 5.8× better and should be the production path forward.

## Key Results

- **Spearman ρ = 0.8438 [ADEQUATE]** — measures rank correlation between origami's predicted GFLOPS and hardware-measured GFLOPS across ~259 tile configs per shape on MI300X (Banff, 3 GPUs). Strong for ranking, but ranking quality doesn't translate to tile-selection accuracy. [VERIFIED] from `correlation_harness.py` line 361: `stats.spearmanr(pred_v, act_v)`.

- **Top-1 accuracy = 6.25% [INADEQUATE]** — 1/16 shapes. Random baseline with 13 candidate tiles = 7.69%. Lift over random = 0.8× (i.e., **worse than random**). With ~259 tile configs evaluated per shape, random per-tile baseline is 0.39%, giving 16× lift per-tile — but the operational metric is per-shape top-1, and origami fails it. [VERIFIED] from `analysis_results.md` Table 2: only `12288x9472x32768_bf16_r` achieves top-1.

- **Mean regret = 17.9% [MARGINAL]** — origami's picked tile delivers 82.1% of oracle TFLOPS. Worst category: prefill at 21.6%, decode at 20.8%. The K=3 lookup table achieves 3.06% mean regret on 1,863 shapes (5-fold CV, seed=42, Jaccard=1.0) — a 5.8× improvement [VERIFIED] from `cross_validation_report.md`. See `key_result_internal-auditor.png`.

- **Cost efficiency: $301.02 total, 43.3% wasted** — $130.21 spent on 15 agent runs that hit cycles_exhausted without delivering accepted work. All 5 PM Hard Gate failures were caused by estimated-data tags in reports — a single, recurring pattern. Peer K-tasks: K-018 ($107.67), K-019 ($666.55), K-020 ($514.49), K-021 ($551.15). K-016's cost/agent-run ($8.60) is efficient; the waste comes from repetition, not per-unit cost. [VERIFIED] from `K-016.jsonl` (35 cost entries summed).

- **Intervention pattern: SYSTEMIC (gate-blocker loop)** — 8 executive oversight triggers: 4 gate-blocker-recovery, 2 stall-recovery (restart storm), 2 user-escalation. The gate-blocker loop (estimated-data tags → gate fail → re-dispatch → agents reproduce estimated-data tags) is the dominant cost driver. Teams cannot reliably produce [VERIFIED]-only reports when blocked from GPU access. [VERIFIED] from `K-016.jsonl` event timeline.

## Recommended Next Steps

1. **Integrate K=3 lookup table into production selector.py** — the 3-tile set (`128x64x128_s2`, `128x128x128_s2`, `16x64x128_s2`) reduces regret from 17.9% to 3.06%. This is the highest-impact change available. Tracked as K-018 scope.

2. **Fix the estimated-tag gate-blocker loop** — add an agent-level pre-submission check that scans for estimated-data tags before submitting. This single guard would have prevented all 5 gate failures and saved ~$130 in wasted re-dispatches.

3. **None for K-016 closure** — K-016's chartered deliverable (Triton LDS filtering integration) is complete with 0% crash-risk post-filter [VERIFIED]. The remaining regret gap is a cost-model quality issue properly scoped to K-018.

## Evidence Files

- `internal-auditor/key_result_internal-auditor.png` — 3-panel audit dashboard: regret by category, cost breakdown (productive vs wasted), intervention trigger classification. Data: K-016.jsonl + sweep_mi300x_wide_v5.csv + analysis_results.md. MI300X Banff. Branch `k016/triton-specialization-in-origami`.

## Method

Parsed all 116 events from `K-016.jsonl` to reconstruct the full intervention timeline, summed 35 cost entries ($301.02 total), and classified 8 executive oversight triggers by root cause. Validated Spearman ρ interpretation by reading `correlation_harness.py` (line 361: correlates origami predicted GFLOPS vs actual hardware GFLOPS). Cross-referenced top-1 accuracy against tile-space cardinality (13 tiles in wide sweep, ~259 configs per shape in Banff data) to establish random baselines. All metrics sourced from `analysis_results.md` and `cross_validation_report.md`, which carry [VERIFIED] tags from GPU hardware measurements on MI300X (Banff, 3 GPUs).
