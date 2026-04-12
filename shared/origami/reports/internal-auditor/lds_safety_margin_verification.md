# LDS Safety Margin Verification — Step 3

**Date**: 2026-04-12 | **Hardware**: MI300X (gfx942) | **Dtype**: bf16  
**Source**: `banff_data/tile_sweep_gpu{1,2,3}.csv` (17,686 rows total; 6,682 bf16 rows)  
**Method**: Remove boundary tiles (LDS = 65,536B, margin = 0B) from candidate set, re-compute per-shape oracle, compare regret

---

## 1. Boundary Tiles Identified

**Definition**: Tile configs where `lds_bytes == 65,536` (exactly at MI300X 64KB LDS capacity, 0 bytes margin).

| # | Tile Config | BLOCK_M | BLOCK_N | BLOCK_K | num_stages | LDS (bytes) | Margin (bytes) | Status |
|---|-------------|---------|---------|---------|------------|-------------|----------------|--------|
| 1 | `32x32x256_s3` | 32 | 32 | 256 | 3 | 65,536 | 0 | [VERIFIED] on MI300X gfx942 |
| 2 | `64x64x128_s3` | 64 | 64 | 128 | 3 | 65,536 | 0 | [VERIFIED] on MI300X gfx942 |
| 3 | `64x64x256_s2` | 64 | 64 | 256 | 2 | 65,536 | 0 | [VERIFIED] on MI300X gfx942 |
| 4 | `128x128x64_s3` | 128 | 128 | 64 | 3 | 65,536 | 0 | [VERIFIED] on MI300X gfx942 |
| 5 | `128x128x128_s2` | 128 | 128 | 128 | 2 | 65,536 | 0 | [VERIFIED] on MI300X gfx942 |
| 6 | `256x256x32_s3` | 256 | 256 | 32 | 3 | 65,536 | 0 | [VERIFIED] on MI300X gfx942 |
| 7 | `256x256x64_s2` | 256 | 256 | 64 | 2 | 65,536 | 0 | [VERIFIED] on MI300X gfx942 |
| 8 | `512x512x32_s2` | 512 | 512 | 32 | 2 | 65,536 | 0 | [VERIFIED] on MI300X gfx942 |

**Note**: Prior reports cited "5 boundary tiles" — the actual count is **8** when analyzing the full bf16 Banff sweep candidate set (104 unique tile configs). The prior count used a smaller tile inventory (9 tiles from `lds_budget_data.csv`); the Banff sweep contains 104 distinct bf16 tile configs.

**LDS formula verification**: For all 8 tiles, `LDS = (num_stages - 1) × (BLOCK_M × BLOCK_K + BLOCK_K × BLOCK_N) × 2` = 65,536 bytes. This matches the origami formula exactly (0% delta) as verified in `lds_budget_groundtruth.md` [VERIFIED].

---

## 2. Per-Shape Regret Comparison (23 Banff Evaluation Shapes)

**Note**: The Banff sweep dataset contains 23 bf16 shapes (not 16 as referenced in some prior reports). The original "16 shapes" counted bf16_NN only; bf16_TN and f8 data adds shapes to different GPU files. The analysis below covers all 23 unique (M,N,K) shapes present in the bf16 data.

| # | Shape | Oracle Tile (full) | Oracle TFLOPS | Filtered Best Tile | Filtered TFLOPS | Regret w/ Boundary | Regret w/o Boundary | Delta (pp) | Affected | Status |
|---|-------|-------------------|---------------|--------------------|-----------------|--------------------|---------------------|------------|----------|--------|
| 1 | 1×1024×1024 | `32x32x256_s3` | 0.2520 | `64x32x256_s2` | 0.2340 | 0.00% | 7.14% | +7.14 | YES | [VERIFIED] on MI300X gfx942 |
| 2 | 1×1024×2048 | `32x32x256_s3` | 0.3590 | `32x32x256_s2` | 0.3310 | 0.00% | 7.80% | +7.80 | YES | [VERIFIED] on MI300X gfx942 |
| 3 | 1×1024×4096 | `32x32x256_s3` | 0.4510 | `32x32x256_s2` | 0.4080 | 0.00% | 9.53% | +9.53 | YES | [VERIFIED] on MI300X gfx942 |
| 4 | 1×1024×8192 | `32x32x256_s3` | 0.5180 | `32x32x128_s3` | 0.4540 | 0.00% | 12.36% | +12.36 | YES | [VERIFIED] on MI300X gfx942 |
| 5 | 1×1024×14336 | `32x32x256_s3` | 0.5520 | `32x32x128_s3` | 0.4770 | 0.00% | 13.59% | +13.59 | YES | [VERIFIED] on MI300X gfx942 |
| 6 | 1×1536×5120 | `32x32x256_s3` | 0.6840 | `32x32x256_s2` | 0.6080 | 0.00% | 11.11% | +11.11 | YES | [VERIFIED] on MI300X gfx942 |
| 7 | 1×2048×1024 | `32x32x256_s3` | 0.4740 | `64x32x256_s2` | 0.4530 | 0.00% | 4.43% | +4.43 | YES | [VERIFIED] on MI300X gfx942 |
| 8 | 1×2048×2048 | `32x32x256_s3` | 0.6600 | `32x32x256_s2` | 0.6150 | 0.00% | 6.82% | +6.82 | YES | [VERIFIED] on MI300X gfx942 |
| 9 | 1×2048×4096 | `32x32x256_s3` | 0.8420 | `32x32x128_s3` | 0.7780 | 0.00% | 7.60% | +7.60 | YES | [VERIFIED] on MI300X gfx942 |
| 10 | 1×2048×7168 | `32x32x256_s3` | 0.9530 | `64x32x256_s2` | 0.8430 | 0.00% | 11.54% | +11.54 | YES | [VERIFIED] on MI300X gfx942 |
| 11 | 1×2048×8192 | `32x32x256_s3` | 0.9900 | `32x32x256_s2` | 0.8920 | 0.00% | 9.90% | +9.90 | YES | [VERIFIED] on MI300X gfx942 |
| 12 | 1×2048×14336 | `32x32x256_s3` | 1.0140 | `64x32x256_s2` | 0.8530 | 0.00% | 15.88% | +15.88 | YES | [VERIFIED] on MI300X gfx942 |
| 13 | 1×4096×1024 | `32x64x256_s2` | 0.8430 | `32x64x256_s2` | 0.8430 | 0.00% | 0.00% | 0.00 | no | [VERIFIED] on MI300X gfx942 |
| 14 | 1×4096×2048 | `32x32x256_s3` | 1.1420 | `32x64x256_s2` | 1.1350 | 0.00% | 0.61% | +0.61 | YES | [VERIFIED] on MI300X gfx942 |
| 15 | 1×4096×4096 | `32x32x256_s3` | 1.5470 | `32x32x128_s3` | 1.4280 | 0.00% | 7.69% | +7.69 | YES | [VERIFIED] on MI300X gfx942 |
| 16 | 1×4096×8192 | `32x32x256_s3` | 1.5760 | `64x32x256_s2` | 1.3640 | 0.00% | 13.45% | +13.45 | YES | [VERIFIED] on MI300X gfx942 |
| 17 | 1×4096×11008 | `32x32x256_s3` | 1.4140 | `64x32x256_s2` | 1.2640 | 0.00% | 10.61% | +10.61 | YES | [VERIFIED] on MI300X gfx942 |
| 18 | 1×4096×14336 | `32x32x256_s3` | 1.2690 | `64x32x256_s2` | 1.2290 | 0.00% | 3.15% | +3.15 | YES | [VERIFIED] on MI300X gfx942 |
| 19 | 1×5120×1536 | `32x32x256_s3` | 1.3360 | `64x32x256_s2` | 1.2890 | 0.00% | 3.52% | +3.52 | YES | [VERIFIED] on MI300X gfx942 |
| 20 | 1×5120×5120 | `32x32x256_s3` | 1.9220 | `64x32x256_s2` | 1.7430 | 0.00% | 9.31% | +9.31 | YES | [VERIFIED] on MI300X gfx942 |
| 21 | 1×5120×13824 | `32x32x256_s3` | 1.6100 | `64x32x256_s2` | 1.4500 | 0.00% | 9.94% | +9.94 | YES | [VERIFIED] on MI300X gfx942 |
| 22 | 1×5120×18432 | `32x32x256_s3` | 1.5710 | `64x32x256_s2` | 1.4080 | 0.00% | 10.38% | +10.38 | YES | [VERIFIED] on MI300X gfx942 |
| 23 | 1×7168×2048 | `32x32x32_s3` | 0.8030 | `32x32x32_s3` | 0.8030 | 0.00% | 0.00% | 0.00 | no | [VERIFIED] on MI300X gfx942 |

---

## 3. Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Banff shapes evaluated | 23 | [VERIFIED] on MI300X gfx942 |
| Boundary tiles removed | 8 | [VERIFIED] on MI300X gfx942 |
| Shapes affected (regret increases) | **21/23** | [VERIFIED] on MI300X gfx942 |
| Shapes unaffected | 2/23 (1×4096×1024, 1×7168×2048) | [VERIFIED] on MI300X gfx942 |
| Max regret increase | **15.88 pp** (shape 1×2048×14336) | [VERIFIED] on MI300X gfx942 |
| Mean regret increase | **8.10 pp** | [VERIFIED] on MI300X gfx942 |
| Median regret increase | **9.31 pp** | [VERIFIED] on MI300X gfx942 |
| Dominant boundary tile | `32x32x256_s3` (oracle for 21/23 shapes) | [VERIFIED] on MI300X gfx942 |

### Cross-Validation Spot Checks

| Shape | Tile | Reported TFLOPS | Raw File | Raw TFLOPS | Match |
|-------|------|-----------------|----------|------------|-------|
| 1×2048×14336 | `32x32x256_s3` | 1.0140 | `tile_sweep_gpu2.csv` | 1.0140 | ✓ [VERIFIED] |
| 1×2048×14336 | `64x32x256_s2` | 0.8530 | `tile_sweep_gpu2.csv` | 0.8530 | ✓ [VERIFIED] |
| 1×4096×1024 | `32x64x256_s2` | 0.8430 | `tile_sweep_gpu2.csv` | 0.8430 | ✓ [VERIFIED] |
| 1×4096×1024 | `32x32x256_s3` | N/A (0.8320) | `tile_sweep_gpu2.csv` | 0.8320 | ✓ [VERIFIED] — oracle correctly chosen as `32x64x256_s2` |

---

## 4. Conclusion & Safety Margin Recommendation

**Removing 8 boundary tiles changed regret on 21/23 shapes; max regret increase: 15.88 pp** [VERIFIED].

**The dominant boundary tile is `32x32x256_s3`** — it is the oracle tile for 21 of 23 Banff shapes. This tile uses exactly 65,536 bytes of LDS (`(3-1) × (32×256 + 256×32) × 2 = 2 × 32,768 = 65,536`). If a future Triton compiler change adds even 1 byte of additional LDS overhead (alignment padding, scratch buffers, etc.), this tile would crash on MI300X.

**Recommendation: Do NOT apply a 1KB safety margin for the Banff decode-shape regime.**

The prior report's recommendation of a 1KB margin was based on the (correct) observation that 0-margin tiles are fragile to compiler changes. However, the measured performance impact is severe: `32x32x256_s3` delivers 4–16% better performance than the best non-boundary alternative for decode shapes (M=1). Removing it would introduce a mean 8.10 pp regret across all Banff shapes.

**Instead, the recommended approach is**:
1. **Keep boundary tiles in production** — the origami LDS formula is exact (0% delta, verified from Triton source) so false-reject risk is zero today
2. **Add a compiler-version-pinned validation test** — if the Triton compiler LDS allocation changes, detect it immediately rather than adding a static margin
3. **Separately validate on MI355X/MI450** — the 64KB LDS limit may differ on future architectures

---

## 5. Evidence Files

- `internal-auditor/lds_safety_margin_verification.json` — Full per-shape results with source provenance
- `internal-auditor/lds_safety_margin_verification.md` — This report
- Source data: `banff_data/tile_sweep_gpu{1,2,3}.csv` (17,686 rows, MI300X gfx942 Banff sweep)
