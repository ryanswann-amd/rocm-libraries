# Prefill BLOCK_M Alignment Heuristic Test

## Methodology

**Heuristic rule**: For a shape with M-dimension value `M`, filter the candidate tile set
to only tiles where `BLOCK_M` evenly divides `M` (i.e., `M % BLOCK_M == 0`) or `BLOCK_M`
evenly divides `M` (i.e., `BLOCK_M % M == 0`). From the filtered set, pick the tile with
the highest measured TFLOPS. Compare against the oracle-best tile (unrestricted).

**Candidate tile set**: 88 unique tile configs from the MI300X hardware sweep corpus.
Per shape, 37–39 tiles have valid measurements (some excluded by M < BLOCK_M).

**Data sources**: `slurm/merged_full_corpus.csv` (MI300X rows only) and
`benchmarking/results_mi300x_chunk*.csv` — all measurements from MI300X hardware sweeps
on AMD Instinct MI300X (gfx942, 304 CUs).

## Summary

- **Heuristic matched oracle on 25/39 shapes** (64.1%) [VERIFIED]
- **Mean regret (heuristic): 10.66%** vs **K=3 lookup: 28.27%** [VERIFIED]
- Heuristic beats K=3 on: **29/39** shapes [VERIFIED]
- Heuristic worse than K=3 on: **10/39** shapes [VERIFIED]
- Median heuristic regret: **0.00%** [VERIFIED]
- Max heuristic regret: **53.42%** [VERIFIED]

### By M-Alignment Class

| M Alignment | # Shapes | Oracle Match | Mean Regret | Status |
|---|---|---|---|---|
| M % 128 == 0 | 20 | 20/20 (100%) | 0.00% | [VERIFIED] |
| M % 64 == 0 (not 128) | 11 | 5/11 (45%) | 7.25% | [VERIFIED] |
| M % 32 == 0 (not 64) | 3 | 0/3 (0%) | 33.73% | [VERIFIED] |
| M % 16 only | 5 | 0/5 (0%) | 46.95% | [VERIFIED] |

## Full Results Table

| Shape | M | #Total | #Aligned | Heuristic Tile | H TFLOPS | Oracle Tile | O TFLOPS | Regret% | K3 Regret% | Match | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 400x14336x4096 | 400 | 37 | 12 | 16x128x64 | 167.6165 | 128x256x64 | 359.8193 | 53.42 | 28.25 | NO | [VERIFIED] |
| 368x14336x4096 | 368 | 37 | 12 | 16x128x64 | 190.5056 | 128x256x64 | 370.3510 | 48.56 | 32.99 | NO | [VERIFIED] |
| 336x14336x4096 | 336 | 37 | 12 | 16x128x64 | 173.4506 | 256x128x64 | 316.4243 | 45.18 | 27.35 | NO | [VERIFIED] |
| 304x14336x4096 | 304 | 37 | 12 | 16x128x64 | 165.8687 | 64x256x64 | 301.5950 | 45.00 | 28.22 | NO | [VERIFIED] |
| 272x14336x4096 | 272 | 37 | 12 | 16x128x64 | 157.9329 | 64x256x64 | 275.0600 | 42.58 | 27.92 | NO | [VERIFIED] |
| 288x14336x4096 | 288 | 37 | 21 | 32x128x64 | 189.4329 | 64x128x64 | 294.5960 | 35.70 | 29.67 | NO | [VERIFIED] |
| 416x14336x4096 | 416 | 37 | 21 | 32x128x64 | 240.8751 | 128x256x64 | 373.6368 | 35.53 | 28.18 | NO | [VERIFIED] |
| 192x27648x3584 | 192 | 37 | 27 | 64x128x128 | 220.1258 | 256x128x64 | 317.3538 | 30.64 | 25.90 | NO | [VERIFIED] |
| 352x14336x4096 | 352 | 37 | 21 | 32x128x64 | 235.1710 | 128x256x64 | 335.8033 | 29.97 | 28.43 | NO | [VERIFIED] |
| 448x14336x4096 | 448 | 37 | 27 | 64x128x128 | 280.3937 | 256x128x64 | 393.2238 | 28.69 | 26.30 | NO | [VERIFIED] |
| 192x22016x5120 | 192 | 37 | 27 | 64x128x64 | 263.7959 | 256x128x64 | 286.9878 | 8.08 | 27.65 | NO | [VERIFIED] |
| 192x22016x3584 | 192 | 37 | 27 | 64x128x64 | 264.3695 | 256x128x64 | 281.3835 | 6.05 | 29.17 | NO | [VERIFIED] |
| 192x22016x4096 | 192 | 37 | 27 | 64x128x64 | 269.1197 | 256x128x64 | 285.3766 | 5.70 | 29.20 | NO | [VERIFIED] |
| 192x22016x8192 | 192 | 37 | 27 | 64x128x64 | 247.0089 | 256x128x64 | 248.5719 | 0.63 | 25.39 | NO | [VERIFIED] |
| 384x13312x4096 | 384 | 37 | 32 | 128x256x64 | 348.3275 | 128x256x64 | 348.3275 | 0.00 | 31.90 | YES | [VERIFIED] |
| 384x13312x3584 | 384 | 37 | 32 | 128x256x64 | 340.9108 | 128x256x64 | 340.9108 | 0.00 | 31.87 | YES | [VERIFIED] |
| 384x13312x5120 | 384 | 37 | 32 | 128x256x64 | 350.7831 | 128x256x64 | 350.7831 | 0.00 | 31.08 | YES | [VERIFIED] |
| 320x14336x4096 | 320 | 37 | 27 | 64x256x64 | 318.4369 | 64x256x64 | 318.4369 | 0.00 | 30.45 | YES | [VERIFIED] |
| 384x14336x4096 | 384 | 37 | 32 | 128x256x64 | 370.3105 | 128x256x64 | 370.3105 | 0.00 | 30.03 | YES | [VERIFIED] |
| 64x1408x7168 | 64 | 37 | 37 | 16x16x256 | 38.4821 | 16x16x256 | 38.4821 | 0.00 | 29.99 | YES | [VERIFIED] |
| 384x14336x5120 | 384 | 37 | 32 | 128x256x64 | 366.5458 | 128x256x64 | 366.5458 | 0.00 | 29.70 | YES | [VERIFIED] |
| 384x14336x2048 | 384 | 37 | 32 | 128x256x64 | 324.4166 | 128x256x64 | 324.4166 | 0.00 | 29.41 | YES | [VERIFIED] |
| 64x1408x8192 | 64 | 37 | 37 | 16x16x256 | 38.8760 | 16x16x256 | 38.8760 | 0.00 | 29.09 | YES | [VERIFIED] |
| 384x13312x8192 | 384 | 37 | 32 | 128x256x64 | 349.2571 | 128x256x64 | 349.2571 | 0.00 | 28.56 | YES | [VERIFIED] |
| 192x22016x2048 | 192 | 37 | 27 | 64x128x64 | 254.0883 | 64x128x64 | 254.0883 | 0.00 | 28.41 | YES | [VERIFIED] |
| 384x14336x8192 | 384 | 37 | 32 | 128x256x64 | 384.2111 | 128x256x64 | 384.2111 | 0.00 | 28.24 | YES | [VERIFIED] |
| 384x13312x7168 | 384 | 37 | 32 | 128x256x64 | 344.8298 | 128x256x64 | 344.8298 | 0.00 | 28.21 | YES | [VERIFIED] |
| 384x16384x4096 | 384 | 37 | 32 | 128x256x64 | 399.8015 | 128x256x64 | 399.8015 | 0.00 | 27.96 | YES | [VERIFIED] |
| 384x14336x3584 | 384 | 37 | 32 | 128x256x64 | 353.4355 | 128x256x64 | 353.4355 | 0.00 | 27.85 | YES | [VERIFIED] |
| 384x16384x5120 | 384 | 37 | 32 | 128x256x64 | 401.6491 | 128x256x64 | 401.6491 | 0.00 | 27.66 | YES | [VERIFIED] |
| 384x16384x7168 | 384 | 37 | 32 | 128x256x64 | 405.8182 | 128x256x64 | 405.8182 | 0.00 | 27.29 | YES | [VERIFIED] |
| 64x1408x4096 | 64 | 37 | 37 | 16x16x256 | 30.6611 | 16x16x256 | 30.6611 | 0.00 | 27.24 | YES | [VERIFIED] |
| 384x13312x2048 | 384 | 37 | 32 | 128x256x64 | 298.4912 | 128x256x64 | 298.4912 | 0.00 | 26.95 | YES | [VERIFIED] |
| 384x16384x3584 | 384 | 37 | 32 | 128x256x64 | 384.7388 | 128x256x64 | 384.7388 | 0.00 | 26.85 | YES | [VERIFIED] |
| 384x16384x2048 | 384 | 37 | 32 | 128x256x64 | 345.6621 | 128x256x64 | 345.6621 | 0.00 | 26.76 | YES | [VERIFIED] |
| 384x14336x7168 | 384 | 37 | 32 | 128x256x64 | 375.0328 | 128x256x64 | 375.0328 | 0.00 | 26.27 | YES | [VERIFIED] |
| 256x22016x4096 | 256 | 37 | 37 | 256x128x64 | 344.6618 | 256x128x64 | 344.6618 | 0.00 | 25.56 | YES | [VERIFIED] |
| 256x22016x3584 | 256 | 37 | 37 | 256x128x64 | 341.0481 | 256x128x64 | 341.0481 | 0.00 | 25.33 | YES | [VERIFIED] |
| 256x22016x5120 | 256 | 37 | 37 | 256x128x64 | 352.5034 | 256x128x64 | 352.5034 | 0.00 | 25.17 | YES | [VERIFIED] |

## Key Findings

1. **The heuristic is highly effective for M-aligned shapes**: When M is divisible by 128,
   the heuristic matches the oracle 20/20 times with 0.00% mean regret [VERIFIED].

2. **The heuristic catastrophically fails for non-power-of-2 M values**: For M values
   divisible only by 16 (e.g., M=272, 304, 336, 368, 400), the aligned tile set is
   restricted to only 12 tiles with small BLOCK_M values (16), leading to 43-53% regret [VERIFIED].

3. **Overall, the heuristic reduces mean regret from 28.27% (K=3) to 10.66%** [VERIFIED],
   but this average is misleading — it works perfectly on easy shapes (M%128==0)
   and terribly on hard shapes (M%16 only).

4. **The bimodal behavior makes this heuristic unreliable as a standalone policy**:
   - On 25/39 shapes: 0.00% regret (perfect) [VERIFIED]
   - On 14/39 shapes: 7-53% regret (often worse than K=3) [VERIFIED]

## Conclusion for K-018 Scoping

**YES, the heuristic is useful — but only as a conditional component, not standalone.**
The BLOCK_M alignment filter is a valid first-pass heuristic for shapes where M is a
multiple of 64 or 128 (31/39 shapes, 79.5%). For the remaining shapes with non-aligned M,
a different strategy is needed (e.g., nearest-M rounding, minimum-waste selection).

**Data provenance**: All TFLOPS measurements from MI300X hardware sweeps [VERIFIED].
Sources: `slurm/merged_full_corpus.csv` and `benchmarking/results_mi300x_chunk*.csv`.
GPU: AMD Instinct MI300X (gfx942, 304 CUs).
