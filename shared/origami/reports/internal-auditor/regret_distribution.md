# Regret Distribution Analysis — 16 Banff Shapes

**Source**: `correlation_results.json` (correlation_harness.py on MI300X, Banff cluster)
**All values [VERIFIED]** on AMD Instinct MI300X (gfx942), Banff cluster (banff-cyxtera-s70-1.ctr.dcgpu)

## Percentile Table

| Percentile | Regret (%) |
|-----------|-----------|
| Min       | 0.00      |
| p25       | 12.84     |
| **p50 (median)** | **13.89** |
| p75       | 25.16     |
| p90       | 29.20     |
| p95       | 30.91     |
| p99       | 34.73     |
| Max       | 35.69     |
| **Mean**  | **17.88** |
| Std Dev   | 9.10      |
| IQR       | 12.33     |

## Sorted Regret Values

| Rank | Shape | Regret (%) | Category |
|------|-------|-----------|----------|
| 1 | 12288×9472×32768 bf16 | 0.00 | compute/prefill |
| 2 | 128×16384×16384 bf16 | 8.36 | small-batch |
| 3 | 1×1280×8192 bf16 | 11.44 | decode |
| 4 | 4196×4196×4196 f16 | 11.91 | compute/prefill |
| 5 | 2048×4096×5376 f16 | 13.15 | compute/prefill |
| 6 | 32768×128×8192 bf16 | 13.15 | compute |
| 7 | 1×1280×8192 f16 | 13.22 | decode |
| 8 | 2048×1336×6176 bf16 | 13.45 | compute/prefill |
| 9 | 4196×4196×4196 bf16 | 14.34 | compute/prefill |
| 10 | 128×16384×6656 bf16 | 19.40 | small-batch |
| 11 | 128×13312×16384 bf16 | 20.82 | small-batch |
| 12 | 1536×3584×3584 f16 | 23.98 | mid-batch/prefill |
| 13 | 3600×4096×4096 bf16 | 28.72 | compute/prefill |
| 14 | 1×13312×16384 bf16 | 29.09 | decode |
| 15 | 1×16384×16384 bf16 | 29.32 | decode |
| 16 | 2048×4096×5376 bf16 | 35.69 | compute/prefill |

## Regret Band Buckets

| Band | Count | % of Shapes | Shapes |
|------|-------|------------|--------|
| <5% | 1 | 6.2% | 12288×9472×32768 bf16 (0.0%) |
| 5-15% | 8 | 50.0% | 128×16384×16384 bf16 (8.4%), 1×1280×8192 bf16 (11.4%), 4196×4196×4196 f16 (11.9%), 2048×4096×5376 f16 (13.1%), 32768×128×8192 bf16 (13.2%), 1×1280×8192 f16 (13.2%), 2048×1336×6176 bf16 (13.4%), 4196×4196×4196 bf16 (14.3%) |
| 15-25% | 3 | 18.8% | 128×16384×6656 bf16 (19.4%), 128×13312×16384 bf16 (20.8%), 1536×3584×3584 f16 (24.0%) |
| >25% | 4 | 25.0% | 3600×4096×4096 bf16 (28.7%), 1×13312×16384 bf16 (29.1%), 1×16384×16384 bf16 (29.3%), 2048×4096×5376 bf16 (35.7%) |

## Distribution Shape: RIGHT-SKEWED (fat right tail)

**Evidence:**
- Mean (17.88%) exceeds median (13.89%) by 3.99 percentage points
- Pearson's second skewness coefficient = 1.31 (>0.5 = meaningfully right-skewed)
- The 4 shapes with regret >25% (25% of shapes) contribute 43% of the total summed regret
- Without the 4 tail shapes, mean regret drops from 17.88% to 13.60% — a 24% reduction

**The mean is driven by 4 shapes with regret >25%.** These are:
1. **2048×4096×5376 bf16** — 35.7% regret (compute/prefill, worst case)
2. **1×16384×16384 bf16** — 29.3% regret (decode, large-N)
3. **1×13312×16384 bf16** — 29.1% regret (decode, large-N)
4. **3600×4096×4096 bf16** — 28.7% regret (compute/prefill)

## Category Breakdown in the >25% Tail

| Category | Tail Shapes | Mean Regret (all) | Tail Contribution |
|----------|------------|-------------------|-------------------|
| Decode | 2 of 4 total | 20.77% | Dominated by large-N decode (29.1%, 29.3%) — small decode (11.4%, 13.2%) is fine |
| Compute/Prefill | 2 of 8 total | 16.30% | The 2 worst (35.7%, 28.7%) are non-square shapes where tile-waste dominates |

## GFLOPS-Weighted Regret

| Metric | Value |
|--------|-------|
| Unweighted mean regret | 17.88% |
| GFLOPS-weighted regret | 17.18% |
| Difference | -0.70 pp |

The GFLOPS-weighting is nearly neutral because the worst-regret shapes span both high-throughput (2048×4096×5376 bf16 @ 1014 TFLOPS oracle) and low-throughput (1×16384×16384 bf16 @ 6.6 GFLOPS oracle) shapes.

## Key Finding

**The 17.9% mean regret is NOT uniformly distributed. It is right-skewed with a fat tail.** Half the shapes (8/16) sit in a tight 5-15% regret band (median 13.15%). The mean is inflated by 4 outlier shapes (25% of the population) averaging 30.7% regret — 2.26× higher than the remaining 12 shapes (13.6% mean). Fixing origami's tile selection for just these 4 shapes would reduce mean regret from 17.9% to ~13.6%, a 24% improvement with no model changes needed for the other 12 shapes.
