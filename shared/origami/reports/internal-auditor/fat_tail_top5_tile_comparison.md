# Fat-Tail Top-5 Worst-Regret Prefill Shapes: K=3 vs Oracle Tile Comparison

**Hardware**: MI300X gfx942, 304 CU, 64KB LDS (AMD Instinct MI300X VF)
**K=3 Tile Set**: {128x64x128_s2, 16x64x128_s2, 128x128x128_s2} [VERIFIED] from `regret_stats_verified.csv`, commit f9bae275
**Corpus**: 1,863 shapes, 39 fat-tail shapes (>25% K=3 regret) [VERIFIED] from `benchmarking/k3_per_shape_regret.csv`
**Source Sweep Data**: `benchmarking/results_mi300x_chunk8.csv`, `benchmarking/results_mi300x_chunk9.csv`
**Cross-Validated Against**: `internal-auditor/fat_tail_autopsy_full.csv`, `benchmarking/k016_statusquo_comparison.csv`

## Top-5 Worst-Regret Shapes (ranked by K=3 lookup regret, descending)

| Rank | Shape (M, N, K) | K3 Tile Config (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) | K3 TFLOPS | Oracle Tile Config (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) | Oracle TFLOPS | TFLOPS Delta | Regret % |
|------|-----------------|------------------------------------------------------------------|-----------|----------------------------------------------------------------------|---------------|-------------|----------|
| 1 | (368, 14336, 4096) | 128, 128, 128, 8, 2 | 248.1788 [VERIFIED] | 128, 256, 64, 8, 2 | 370.3510 [VERIFIED] | 122.1722 | 32.99% |
| 2 | (384, 13312, 4096) | 128, 64, 128, 8, 2 | 237.1969 [VERIFIED] | 128, 256, 64, 8, 2 | 348.3275 [VERIFIED] | 111.1306 | 31.90% |
| 3 | (384, 13312, 3584) | 128, 128, 128, 8, 2 | 232.2665 [VERIFIED] | 128, 256, 64, 8, 2 | 340.9108 [VERIFIED] | 108.6443 | 31.87% |
| 4 | (384, 13312, 5120) | 128, 128, 128, 8, 2 | 241.7511 [VERIFIED] | 128, 256, 64, 8, 2 | 350.7831 [VERIFIED] | 109.0320 | 31.08% |
| 5 | (320, 14336, 4096) | 128, 64, 128, 8, 2 | 221.4631 [VERIFIED] | 64, 256, 64, 8, 2 | 318.4369 [VERIFIED] | 96.9738 | 30.45% |

## Per-Shape K=3 Tile Breakdown

For each shape, all three K=3 candidate tiles are shown with measured TFLOPS. The **bold** entry is the K=3 lookup winner (highest TFLOPS among the 3).

### Shape 1: (368, 14336, 4096) — Regret 32.99%

| K=3 Tile | TFLOPS | Source |
|----------|--------|--------|
| 128x64x128, warps=8, stages=2 | 244.6885 [VERIFIED] | results_mi300x_chunk9.csv, line M=368,N=14336,K=4096 |
| 16x64x128, warps=4, stages=2 | 111.8542 [VERIFIED] | results_mi300x_chunk9.csv, line M=368,N=14336,K=4096 |
| **128x128x128, warps=8, stages=2** | **248.1788** [VERIFIED] | results_mi300x_chunk9.csv, line M=368,N=14336,K=4096 |

Oracle: 128x256x64, warps=8, stages=2 → 370.3510 TFLOPS [VERIFIED] from results_mi300x_chunk9.csv

### Shape 2: (384, 13312, 4096) — Regret 31.90%

| K=3 Tile | TFLOPS | Source |
|----------|--------|--------|
| **128x64x128, warps=8, stages=2** | **237.1969** [VERIFIED] | results_mi300x_chunk9.csv, line M=384,N=13312,K=4096 |
| 16x64x128, warps=4, stages=2 | 111.9199 [VERIFIED] | results_mi300x_chunk9.csv, line M=384,N=13312,K=4096 |
| 128x128x128, warps=8, stages=2 | 236.1802 [VERIFIED] | results_mi300x_chunk9.csv, line M=384,N=13312,K=4096 |

Oracle: 128x256x64, warps=8, stages=2 → 348.3275 TFLOPS [VERIFIED] from results_mi300x_chunk9.csv

### Shape 3: (384, 13312, 3584) — Regret 31.87%

| K=3 Tile | TFLOPS | Source |
|----------|--------|--------|
| 128x64x128, warps=8, stages=2 | 231.2682 [VERIFIED] | results_mi300x_chunk9.csv, line M=384,N=13312,K=3584 |
| 16x64x128, warps=4, stages=2 | 114.9074 [VERIFIED] | results_mi300x_chunk9.csv, line M=384,N=13312,K=3584 |
| **128x128x128, warps=8, stages=2** | **232.2665** [VERIFIED] | results_mi300x_chunk9.csv, line M=384,N=13312,K=3584 |

Oracle: 128x256x64, warps=8, stages=2 → 340.9108 TFLOPS [VERIFIED] from results_mi300x_chunk9.csv

### Shape 4: (384, 13312, 5120) — Regret 31.08%

| K=3 Tile | TFLOPS | Source |
|----------|--------|--------|
| 128x64x128, warps=8, stages=2 | 237.9639 [VERIFIED] | results_mi300x_chunk9.csv, line M=384,N=13312,K=5120 |
| 16x64x128, warps=4, stages=2 | 107.3409 [VERIFIED] | results_mi300x_chunk9.csv, line M=384,N=13312,K=5120 |
| **128x128x128, warps=8, stages=2** | **241.7511** [VERIFIED] | results_mi300x_chunk9.csv, line M=384,N=13312,K=5120 |

Oracle: 128x256x64, warps=8, stages=2 → 350.7831 TFLOPS [VERIFIED] from results_mi300x_chunk9.csv

### Shape 5: (320, 14336, 4096) — Regret 30.45%

| K=3 Tile | TFLOPS | Source |
|----------|--------|--------|
| **128x64x128, warps=8, stages=2** | **221.4631** [VERIFIED] | results_mi300x_chunk8.csv, line M=320,N=14336,K=4096 |
| 16x64x128, warps=4, stages=2 | 116.8849 [VERIFIED] | results_mi300x_chunk8.csv, line M=320,N=14336,K=4096 |
| 128x128x128, warps=8, stages=2 | 221.2532 [VERIFIED] | results_mi300x_chunk8.csv, line M=320,N=14336,K=4096 |

Oracle: 64x256x64, warps=8, stages=2 → 318.4369 TFLOPS [VERIFIED] from results_mi300x_chunk8.csv

## Key Findings

1. **Oracle tile is missing from K=3 set in ALL 5 shapes**: The oracle tile for 4/5 shapes is `128x256x64_s2` and for 1/5 is `64x256x64_s2`. Neither tile is in the K=3 lookup set. The K=3 set contains no tile with BLOCK_N=256.

2. **BLOCK_N=256 is the critical gap**: The oracle tiles all use BLOCK_N=256, which enables wider data reuse for these prefill shapes with large N dimensions (13312–14336). The K=3 tiles max out at BLOCK_N=128.

3. **All 5 shapes are compute-bound prefill** with M in [320, 384], N in [13312, 14336], and K in [3584, 5120]. These are medium-M, large-N GEMM workloads typical of LLM prefill.

4. **Regret is consistently 30–33%** across all 5, indicating a systematic tile-selection failure mode, not noise.

5. **K=3 tile winner varies**: 128x128x128_s2 wins in 3/5 shapes; 128x64x128_s2 wins in 2/5 shapes. The 16x64x128_s2 tile is never competitive for these prefill shapes (< 50% of best K=3 tile).

## Data Provenance

All TFLOPS values sourced directly from MI300X GPU hardware sweep measurements:
- `benchmarking/results_mi300x_chunk8.csv` — shapes with M in [192, 368], 43 tile configs per shape
- `benchmarking/results_mi300x_chunk9.csv` — shapes with M in [368, 512], 43 tile configs per shape
- Hardware: AMD Instinct MI300X VF, gfx942 architecture, 304 CUs, bf16 dtype
- Commit: f9bae275 (K=3 analysis), 396eac8824 (exhaustive K=3 search cross-validation)
- All regret values recomputed from raw TFLOPS and cross-validated against `fat_tail_autopsy_full.csv` (deltas < 0.01%)
