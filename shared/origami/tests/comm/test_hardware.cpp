/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "test_harness.hpp"

#include "origami/comm/hardware.hpp"

#include <cmath>

using namespace origami::comm;

// ─── Compile-time MI300X identity (reference constants) ─────────
TEST(mi300x_structural_constants) {
  static_assert(MI300X.num_cu == 304);
  static_assert(MI300X.num_xcd == 8);
  static_assert(MI300X.cu_per_xcd == 38);
  static_assert(MI300X.mshr_depth_per_wave == 12);
  static_assert(MI300X.waves_per_wg == 10);
  static_assert(MI300X.tcp_capacity_bytes == 32ULL * 1024ULL);
  static_assert(MI300X.l2_capacity_bytes == 4ULL * 1024ULL * 1024ULL);
  static_assert(MI300X.mall_capacity_bytes == 256ULL * 1024ULL * 1024ULL);
  static_assert(MI300X.hbm_capacity_bytes == 192ULL * 1024ULL * 1024ULL * 1024ULL);
  CHECK(true);  // static_assert above
}

TEST(mi300x_clock_and_conversions) {
  CHECK_NEAR(MI300X.clock_ghz, 2.0, 1e-12);
  CHECK_NEAR(MI300X.clock_hz(), 2.0e9, 1e-3);
  // 2000 cycles @ 2 GHz = 1000 ns = 1 µs.
  CHECK_NEAR(MI300X.cycles_to_ns(2000.0), 1000.0, 1e-12);
  CHECK_NEAR(MI300X.cycles_to_us(2000.0), 1.0, 1e-12);
  CHECK_NEAR(MI300X.ns_to_cycles(1000.0), 2000.0, 1e-12);
}

TEST(mi300x_xgmi_latency_660ns_round_trip) {
  // 660 ns @ 2 GHz = 1320 cycles.
  CHECK_NEAR(MI300X.xgmi_latency_cycles, 1320.0, 1e-9);
  CHECK_NEAR(MI300X.cycles_to_ns(MI300X.xgmi_latency_cycles), 660.0, 1e-9);
}

// ─── BW polynomial ──────────────────────────────────────────────
TEST(mi300x_bw_fraction_polynomial) {
  // mem_bw_coeffs = (0, 0.015, 0). fraction(N) = 0.015 N, clamp [0,1].
  CHECK_NEAR(MI300X.bw_fraction(0), 0.0, 1e-12);
  CHECK_NEAR(MI300X.bw_fraction(1), 0.015, 1e-12);
  CHECK_NEAR(MI300X.bw_fraction(66), 0.99, 1e-12);
  CHECK_NEAR(MI300X.bw_fraction(67), 1.0, 1e-12);   // clamped
  CHECK_NEAR(MI300X.bw_fraction(304), 1.0, 1e-12);  // clamped
}

TEST(mi300x_hbm_read_bw_per_cu_all_active) {
  // active_cus = num_cu = 304, fraction clamped to 1.0
  // → per-CU = hbm_read_bw / 304.
  const double expected = MI300X.hbm_read_bw / 304.0;
  CHECK_NEAR(MI300X.hbm_read_bw_per_cu(), expected, 1e-12);
}

TEST(mi300x_hbm_write_bw_per_cu_scaled) {
  // active = 10 CUs → fraction = 0.15, per-CU = hbm_write_bw * 0.15 / 10.
  const double expected = MI300X.hbm_write_bw * 0.15 / 10.0;
  CHECK_NEAR(MI300X.hbm_write_bw_per_cu(10), expected, 1e-9);
}

TEST(mi300x_l2_bw_per_cu_scaling) {
  // active_cus_on_xcd = cu_per_xcd → no scaling.
  CHECK_NEAR(MI300X.l2_bw_per_cu_scaled(MI300X.cu_per_xcd), MI300X.l2_bw_per_cu, 1e-12);
  // active = 1 → all of the XCD's L2 to a single CU (38× share).
  CHECK_NEAR(MI300X.l2_bw_per_cu_scaled(1),
             MI300X.l2_bw_per_cu * static_cast<double>(MI300X.cu_per_xcd),
             1e-12);
  // active > cu_per_xcd → clamp.
  CHECK_NEAR(MI300X.l2_bw_per_cu_scaled(1000), MI300X.l2_bw_per_cu, 1e-12);
}

// ─── comm_hardware_t MI300X_COMM ───────────────────────────────────
TEST(mi300x_comm_atomic_and_launch) {
  // 100 ns × 2 GHz = 200 cycles.
  CHECK_NEAR(MI300X_COMM.atomic_latency_cycles, 200.0, 1e-9);
  // 45 µs × 2 GHz = 90 000 cycles.
  CHECK_NEAR(MI300X_COMM.launch_overhead_cycles, 90000.0, 1e-9);
}

TEST(mi300x_comm_link_bw_matches_reference) {
  // link_bw = 49.1 * (1024**3) / 1e9 / 1.23 / 2.0
  //                 ≈ 49.1 GiB/s / 1.23 / 2 GHz ≈ 21.43 B/cycle.
  const double expected = 49.1 * (1024.0 * 1024.0 * 1024.0) / 1e9 / 1.23 / 2.0;
  CHECK_NEAR(MI300X_COMM.link_bw, expected, 1e-12);
}

TEST(mi300x_comm_link_bw_per_ns_is_link_payload_rate) {
  // bytes/cycle × 2 GHz = bytes/ns ≈ 42.86 B/ns per link.
  const double per_ns = MI300X_COMM.rate_per_ns(MI300X_COMM.link_bw);
  CHECK_NEAR(per_ns, 42.86, 0.05);  // ~42.86 B/ns from spec
}

ORIGAMI_TEST_MAIN()
