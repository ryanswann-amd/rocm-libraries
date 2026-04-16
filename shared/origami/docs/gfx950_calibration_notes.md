# gfx950 (MI355X) Origami Calibration Notes

**Task**: K-045 | **Date**: 2026-04-16
**Hardware**: MI355X ES, mi355x-thor-2 (Rainier cluster)
**ROCm**: 7.2.0 | **Cross-validated**: 2 independent runs, <0.4% delta on verified constants

## Clock Correction

| Property | Placeholder | Measured | Source |
|---|---|---|---|
| compute_clock | 2.0 GHz (scripts) | 2.403 GHz | HIP runtime + rocm-smi sclk |
| memory_clock | — | 2000 MHz | rocm-smi mclk |

The origami C++ code reads `compute_clock_ghz` dynamically from
`properties.clockRate / 1e6` (hardware.cpp:73), so no C++ fix is needed
for the clock value itself. The bug was in calibration scripts that
hardcoded `clock_ghz=2.0`. All B/cyc constants below are corrected to
2.403 GHz actual boost clock.

## Calibrated Constants

### Layer 0: Pipeline Depth / Issue Rate

| Constant | Value | Status |
|---|---|---|
| issue_rate_bpc | 0.997 B/cyc @2.403 GHz | VERIFIED, 14-point sweep |
| pipeline_slope | 3.349 cyc/store | VERIFIED, 0.26% cross-run delta |
| fixed_overhead | ~18,870 cycles @2.403 GHz | UNVERIFIED, 14% cross-run variance |

### Layer 1: Single-CU HBM Store Bandwidth

| Constant | Placeholder | Measured | Status |
|---|---|---|---|
| cu_local_store_bpc | 33.0 | 56.78 B/cyc @2.403 GHz | VERIFIED, 0.32% delta |
| steady_state_bpc | 33.0 | 57.05 B/cyc @2.403 GHz | VERIFIED, 0.28% delta |
| Single-CU GB/s | — | 136.28 GB/s | VERIFIED, clock-independent |

### Applied Change: `mem_bw_per_wg_coefficients`

```
OLD: std::make_tuple(0, 0.008, 0)  // placeholder from gfx942
NEW: std::make_tuple(0, 0.019, 0)  // calibrated from MI355X measurement
```

**Derivation**:
- `hardware.mem3_perf_ratio = 1e9 * 6 / 2000000 = 3000` B/cyc (full chip)
- At 1 CU: `limited_mem_bw = 3000 * b * 1`
- Measured 1-CU BW = 56.78 B/cyc => `b = 56.78 / 3000 = 0.0189`, round to `0.019`

**Impact**: Only affects predictions at <53 CUs (new saturation point).
Full-chip (256 CU) predictions unchanged (bw_limited clamped to 1.0).

### Layers NOT Calibrated

| Layer | Reason | Status |
|---|---|---|
| Layer 1b (load BW) | Triton gfx950 codegen bug | BLOCKED |
| Layer 2 (remote store) | ES silicon XGMI peer disabled | BLOCKED |
| mem1_perf_ratio (L2) | No L2-specific microbenchmark | UNCHANGED (17) |
| mem2_perf_ratio (MALL) | No MALL-specific microbenchmark | UNCHANGED |
| mem3_perf_ratio (HBM) | No full-chip BW sweep | UNCHANGED (6) |

## Reproduction

```bash
# On MI355X node (Rainier cluster):
# 1. Run calibration
python3 slurm/calibrate_mi355x.py --layers 0,1a,1d --clock-ghz 2.403

# 2. Verify clock under load
rocm-smi --showclocks | grep sclk  # expect ~2400 MHz

# 3. Cross-validate: compare BPC deltas < 1%
```
