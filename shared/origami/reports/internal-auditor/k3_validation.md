# S3: Independent Validation of K=3 Lookup Table's 3.06% Regret Claim

**Date**: 2026-04-12 | **Step**: s3 of 3 (component layer)

## Purpose

Independently validate the K=3 lookup table's claimed 3.06% mean regret by:
1. Identifying the exact 3-tile set
2. Computing per-shape tile assignment and regret for the 16 Banff shapes
3. Determining whether the independently-computed regret matches the claim

## The K=3 Tile Set

From `cross_validation_report.md` (greedy set-cover order):

| Order | Tile | Role |
|-------|------|------|
| 1 | `128x64x128` | Best single-tile universal (covers most M>1 shapes) |
| 2 | `128x128x128` | Covers large compute-bound shapes |
| 3 | `16x64x128` | Covers decode (M=1) and small-batch shapes |

**Selection method**: Greedy set-cover on 1,863 shapes × 37 Triton GEMM tiles, minimizing mean regret at each step. Jaccard stability = 1.0 across 5-fold CV. [VERIFIED]

## Tile Selection Logic

The K-tile lookup system operates as: **"Run all K tiles for a given shape on hardware, keep the fastest."** This is a pure best-of-K selection — no cost model, no prediction. For a given shape and K=3 tile set, the selected tile = argmax(TFLOPS) over the 3 candidates.

Regret = (oracle_TFLOPS − best_of_K_TFLOPS) / oracle_TFLOPS × 100%

## Critical Scope Discovery: Two Different Oracle Spaces

The claimed 3.06% and the Banff 16-shape regret operate in **different measurement spaces**:

| Property | Corpus (3.06% claim) | Banff (16 shapes) |
|----------|---------------------|-------------------|
| Shapes | 1,863 | 16 |
| Tile space | 37 Triton GEMM tiles | ~259 hipBLASLt/Tensile tiles |
| Oracle | Best of 37 Triton tiles | Best of ~259 Tensile tiles |
| Kernel backend | Triton | hipBLASLt/Tensile |
| Data source | `merged_full_corpus.csv` (Slurm) | `origami_correlation/preds/*.csv` (Banff cluster) |
| Unit | TFLOPS | GFLOPS |

**This means the oracle is different.** The Banff oracle has access to 259 tile configs including fractional and non-power-of-2 tiles (e.g., `128x192x64`, `224x224x64`, `288x224x64`) that don't exist in the 37-tile Triton corpus.

## Validation 1: K=3 on the 1,863-Shape Triton Corpus [VERIFIED]

**Independently recomputed** from `merged_full_corpus.csv` using the greedy set-cover tile selection logic from `k5_greedy_setcover.py`:

| Metric | Value |
|--------|-------|
| Mean K=3 regret | **3.0615%** |
| Median K=3 regret | 0.0000% |
| P95 | 16.5099% |
| Max | 32.9882% |
| Shapes with 0% regret | 1,093 / 1,863 (58.7%) |
| Shapes with <5% regret | 1,500 / 1,863 (80.5%) |

### Per-Tile Selection Frequency

| Tile | # Shapes Selected | % | Mean Regret When Selected |
|------|-------------------|---|--------------------------|
| `16x64x128` | 1,125 | 60.4% | 2.56% |
| `128x64x128` | 515 | 27.6% | 2.87% |
| `128x128x128` | 223 | 12.0% | 6.06% |

### **VERDICT: 3.06% CONFIRMED** ✓ (independently computed: 3.0615%, within 0.002pp of claim)

The K=3 regret of 3.06% on the 1,863-shape Triton corpus is independently verified. The claim is correct.

## Validation 2: K=3 Applied to 16 Banff Shapes (hipBLASLt Oracle)

**This is the novel contribution of S3**: applying the K=3 tile set to the Banff shapes using the Banff hipBLASLt measurement data, with the 259-tile hipBLASLt oracle.

### Per-Shape Results [VERIFIED on MI300X, Banff cluster]

| Shape | Oracle (GFLOPS) | 128x64x128 | 128x128x128 | 16x64x128 | K3 Best | K3 Tile | K3 Regret | Origami Regret |
|-------|----------------|------------|-------------|-----------|---------|---------|-----------|----------------|
| 128x16384x6656 bf16 | 525,947 | 525,749 | **525,947** | 158,100 | 525,947 | 128x128x128 | **0.00%** | 19.40% |
| 128x13312x16384 bf16 | 532,976 | 443,589 | **524,416** | 212,202 | 524,416 | 128x128x128 | **1.61%** | 20.82% |
| 32768x128x8192 bf16 | 574,770 | 455,548 | **563,274** | 184,497 | 563,274 | 128x128x128 | **2.00%** | 13.15% |
| 128x16384x16384 bf16 | 585,694 | 478,082 | **531,967** | 152,642 | 531,967 | 128x128x128 | **9.17%** | 8.36% |
| 2048x1336x6176 bf16 | 607,853 | 452,795 | **545,371** | 206,581 | 545,371 | 128x128x128 | **10.28%** | 13.45% |
| 1x1280x8192 bf16 | 1,949 | 1,558 | 1,279 | **1,726** | 1,726 | 16x64x128 | **11.44%** | 11.44% |
| 1x1280x8192 f16 | 1,997 | 1,572 | 1,290 | **1,733** | 1,733 | 16x64x128 | **13.22%** | 13.22% |
| 2048x4096x5376 f16 | 918,850 | 613,316 | **769,970** | 266,516 | 769,970 | 128x128x128 | **16.20%** | 13.15% |
| 1536x3584x3584 f16 | 743,126 | 452,575 | **607,544** | 246,194 | 607,544 | 128x128x128 | **18.24%** | 23.98% |
| 2048x4096x5376 bf16 | 1,014,450 | 624,225 | **814,102** | 273,947 | 814,102 | 128x128x128 | **19.75%** | 35.69% |
| 1x16384x16384 bf16 | 6,576 | 4,561 | 4,363 | **5,094** | 5,094 | 16x64x128 | **22.54%** | 29.32% |
| 4196x4196x4196 f16 | 743,786 | 457,397 | **587,791** | 209,757 | 587,791 | 128x128x128 | **20.97%** | 11.91% |
| 4196x4196x4196 bf16 | 811,741 | 467,351 | **624,749** | 217,723 | 624,749 | 128x128x128 | **23.04%** | 14.34% |
| 1x13312x16384 bf16 | 6,342 | 3,957 | 4,452 | **4,610** | 4,610 | 16x64x128 | **27.32%** | 29.09% |
| 3600x4096x4096 bf16 | 1,092,680 | 601,304 | **724,718** | 228,958 | 724,718 | 128x128x128 | **33.68%** | 28.72% |
| 12288x9472x32768 bf16 | 1,165,570 | 587,155 | **763,928** | 239,433 | 763,928 | 128x128x128 | **34.46%** | 0.00% |

### Banff K=3 Aggregate Statistics

| Metric | K=3 Lookup (Banff) | Origami (Banff) |
|--------|-------------------|-----------------|
| **Mean regret** | **16.50%** | **17.88%** |
| Median regret | 17.22% | 13.89% |
| Min regret | 0.00% | 0.00% |
| Max regret | 34.46% | 35.69% |

### Per-Tile Assignment on Banff

| K3 Tile | # Shapes Assigned | Shapes |
|---------|-------------------|--------|
| `128x128x128` | 12 | All M>1 shapes + 128×N×K shapes |
| `16x64x128` | 4 | All M=1 decode shapes |
| `128x64x128` | 0 | Never selected (always dominated by 128x128x128) |

**Note**: `128x64x128` is never the best of the 3 for any Banff shape, despite being the first tile selected by greedy set-cover on the 1,863-shape corpus. This is because the 16 Banff shapes are biased toward large (M,N,K) where `128x128x128` dominates.

## Root Cause Analysis: Why 16.50% ≠ 3.06%

The 13.44 percentage-point gap has **two independent causes**:

### Cause 1: Different Oracle Space (dominant)

The Banff oracle has access to ~259 hipBLASLt tiles including high-performance non-standard tiles like:
- `256x256x64` (oracle for 12288×9472×32768 — 1,165,570 GFLOPS)
- `128x192x64` (oracle for 1536×3584×3584)
- `224x224x64` (oracle for 2048×1336×6176)
- `192x192x64` (oracle for 4196×4196×4196)
- `256x128x64` (oracle for 2048×4096×5376)

These tiles are **not in the K=3 set** (and not even in the 37-tile Triton corpus). The K=3 tiles can never achieve 0% regret against these oracles.

**Example**: 12288×9472×32768 has oracle `256x256x64` at 1,165,570 GFLOPS. The best K=3 tile is `128x128x128` at 763,928 GFLOPS — a 34.46% regret. In the Triton corpus (if this shape existed), the oracle might well be `128x128x128`, giving 0% regret.

### Cause 2: Different Shape Composition (secondary)

The 16 Banff shapes are **not representative** of the 1,863-shape corpus:
- Banff is biased toward large compute-bound shapes (M > 1000) with 11/16 shapes
- The corpus is 60% decode/small-batch where `16x64x128` achieves near-0% regret
- Only 6 of the 16 Banff shapes appear in the corpus; 10 are unique to Banff

**In the corpus**, 58.7% of shapes have exactly 0% K=3 regret. The Banff shapes include the hardest shapes that drove the origami regret discussion.

## 6-Shape Cross-Check: Banff Shapes in Both Data Sources

For the 6 shapes that exist in both datasets:

| Shape | K=3 Regret (Corpus/Triton) | K=3 Regret (Banff/hipBLASLt) | Oracle Source |
|-------|---------------------------|------------------------------|--------------|
| 128x13312x16384 | 0.00% | 1.61% | Different tiles |
| 128x16384x16384 | 0.00% | 9.17% | Different tiles |
| 128x16384x6656 | 0.00% | 0.00% | Same tile (128x128x128) |
| 1x1280x8192 | 17.32% | 11.44% | Different tiles |
| 1x13312x16384 | 0.00% | 27.32% | Different tiles |
| 1x16384x16384 | 0.00% | 22.54% | Different tiles |

The same shapes have dramatically different regret depending on oracle space. This confirms that the oracle tile set — not the K=3 tiles — is the primary driver.

## Verdict

### Claim: "K=3 lookup table achieves 3.06% mean regret"

| Assessment | Details |
|------------|---------|
| **CONFIRMED on claimed corpus** | 3.0615% independently computed on 1,863 shapes × 37 Triton tiles. Match within 0.002pp. ✓ |
| **NOT TRANSFERABLE to Banff 16** | 16.50% mean regret on 16 Banff shapes against 259-tile hipBLASLt oracle. 5.4× higher. |
| **Root cause of gap** | Different oracle space (259 hipBLASLt tiles vs 37 Triton tiles) + non-representative shape sample. |
| **Is the 3.06% claim misleading?** | **Partially.** The claim is technically correct but applies only to the Triton-tile-only selection problem. When comparing K=3 to origami (17.88% on Banff), the comparison mixes oracle spaces: K=3's 3.06% uses a 37-tile oracle; origami's 17.88% uses a 259-tile oracle. A fair comparison requires the same oracle. |
| **Fair comparison on Banff** | K=3 = 16.50% vs Origami = 17.88% → K=3 is only 1.38pp better (1.08× improvement, not 5.8×). |

### The "5.8× better" claim in the cross_validation_report.md is an apples-to-oranges comparison.

The report's Table ("Origami Comparison") explicitly notes the "Dataset-scope warning" about different evaluation sets. However, the bottom-line claim "5.8× better" conflates two distinct problems:
- K=3's 3.06% is regret against a **37-tile Triton oracle** (easy: fewer alternatives to miss)
- Origami's 17.88% is regret against a **259-tile hipBLASLt oracle** (hard: more alternatives, including non-standard fractional tiles)

On the same 16 Banff shapes with the same 259-tile oracle, K=3 achieves 16.50% — only marginally better than origami's 17.88%.

## Evidence Files

- `k3_validation_data.json` — full per-shape data with all tile GFLOPS values
- Source: `origami_correlation/preds/*.csv` (16 Banff shapes, MI300X)
- Source: `merged_full_corpus.csv` (1,863 shapes, MI300X)
- Source: `cross_validation_report.md` (K=3 tile set and claim)
- Source: `correlation_results.json` (oracle data, origami regret)
- Hardware: MI300X (gfx942), Banff cluster, 304 CUs
