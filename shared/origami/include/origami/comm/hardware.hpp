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

// origami::comm — analytical communication cost model
//
// hardware_t constants — bottom of the cost model. All time-related fields
// are in **GPU cycles**; rates are in **per-cycle** units. The conversion
// to seconds happens once at the public API boundary (predict_row in
// collective.hpp).
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace origami::comm {

// ─── hardware_t (per-CU and per-XCD compute / memory) ──────────────
struct hardware_t {
  std::string_view arch;

  // Die structure
  int num_cu;
  int num_xcd;
  int cu_per_xcd;
  double clock_ghz;  // cycles per ns

  // Per-CU throughput ceilings
  double vmem_issue_rate;  // VMEM instructions per CU per cycle
  double valu_rate;        // VALU lane-elements per CU per cycle

  // TCP / vL1D (per CU)
  std::size_t tcp_capacity_bytes;
  double tcp_bw;  // bytes per CU per cycle

  // Outstanding request limits
  int mshr_depth_per_wave;
  int waves_per_wg;
  double xgmi_latency_cycles;  // round-trip latency for remote load

  // L2 / TCC (per XCD)
  std::size_t l2_capacity_bytes;
  double l2_bw_per_cu;  // bytes per CU per cycle

  // MALL / Infinity Cache (device-wide)
  std::size_t mall_capacity_bytes;
  double mall_bw;  // bytes per cycle aggregate

  // HBM
  double hbm_read_bw;   // bytes per cycle aggregate
  double hbm_write_bw;  // bytes per cycle aggregate
  std::size_t hbm_capacity_bytes;

  // BW scaling polynomial: fraction = a*N^2 + b*N + c, clamped [0,1]
  std::array<double, 3> mem_bw_coeffs = {0.0, 0.015, 0.0};

  // ── BW polynomial ──────────────────────────────────────────
  constexpr double bw_fraction(int active_cus) const noexcept {
    const double N = static_cast<double>(active_cus);
    const double f = mem_bw_coeffs[0] * N * N + mem_bw_coeffs[1] * N + mem_bw_coeffs[2];
    return std::clamp(f, 0.0, 1.0);
  }

  constexpr double hbm_read_bw_per_cu(int active_cus = -1) const noexcept {
    const int n    = (active_cus < 0) ? num_cu : active_cus;
    const double f = bw_fraction(n);
    return hbm_read_bw * f / static_cast<double>(n);
  }

  constexpr double hbm_write_bw_per_cu(int active_cus = -1) const noexcept {
    const int n    = (active_cus < 0) ? num_cu : active_cus;
    const double f = bw_fraction(n);
    return hbm_write_bw * f / static_cast<double>(n);
  }

  constexpr double l2_bw_per_cu_scaled(int active_cus_on_xcd = -1) const noexcept {
    int n = (active_cus_on_xcd < 0) ? cu_per_xcd : active_cus_on_xcd;
    n     = std::min(n, cu_per_xcd);
    return l2_bw_per_cu * (static_cast<double>(cu_per_xcd) / std::max(n, 1));
  }

  // ── Frequency / cycle ↔ time conversion ────────────────────
  constexpr double clock_hz() const noexcept { return clock_ghz * 1e9; }

  constexpr double cycles_to_seconds(double cycles) const noexcept { return cycles / clock_hz(); }
  constexpr double cycles_to_ns(double cycles) const noexcept {
    return cycles_to_seconds(cycles) * 1e9;
  }
  constexpr double cycles_to_us(double cycles) const noexcept {
    return cycles_to_seconds(cycles) * 1e6;
  }
  constexpr double seconds_to_cycles(double s) const noexcept { return s * clock_hz(); }
  constexpr double ns_to_cycles(double ns) const noexcept { return seconds_to_cycles(ns * 1e-9); }
  constexpr double us_to_cycles(double us) const noexcept { return seconds_to_cycles(us * 1e-6); }

  constexpr double rate_per_second(double per_cycle) const noexcept {
    return per_cycle * clock_hz();
  }
  constexpr double rate_per_ns(double per_cycle) const noexcept {
    return rate_per_second(per_cycle) * 1e-9;
  }

  constexpr double rate_per_cycle_from_per_second(double per_s) const noexcept {
    return per_s / clock_hz();
  }
  constexpr double rate_per_cycle_from_per_ns(double per_ns) const noexcept {
    return rate_per_cycle_from_per_second(per_ns * 1e9);
  }
};

// ─── comm_hardware_t (inter-GPU communication) ──────────────────────
struct comm_hardware_t {
  // xGMI link
  double link_bw;      // bytes per cycle per link, unidirectional
  int num_peer_links;  // links to other GPUs

  // SDMA
  int num_sdma_engines;
  double sdma_read_bw;   // bytes per cycle per link
  double sdma_write_bw;  // bytes per cycle per link

  // Protocol overhead
  double atomic_latency_cycles;
  double launch_overhead_cycles;

  double clock_ghz = 2.0;  // companion clock for ns display helpers

  constexpr double clock_hz() const noexcept { return clock_ghz * 1e9; }

  constexpr double cycles_to_seconds(double cycles) const noexcept { return cycles / clock_hz(); }
  constexpr double cycles_to_ns(double cycles) const noexcept {
    return cycles_to_seconds(cycles) * 1e9;
  }
  constexpr double cycles_to_us(double cycles) const noexcept {
    return cycles_to_seconds(cycles) * 1e6;
  }
  constexpr double seconds_to_cycles(double s) const noexcept { return s * clock_hz(); }
  constexpr double ns_to_cycles(double ns) const noexcept { return seconds_to_cycles(ns * 1e-9); }
  constexpr double us_to_cycles(double us) const noexcept { return seconds_to_cycles(us * 1e-6); }

  constexpr double rate_per_second(double per_cycle) const noexcept {
    return per_cycle * clock_hz();
  }
  constexpr double rate_per_ns(double per_cycle) const noexcept {
    return rate_per_second(per_cycle) * 1e-9;
  }
  constexpr double rate_per_cycle_from_per_ns(double per_ns) const noexcept {
    return per_ns * 1e9 / clock_hz();
  }
};

// ─── MI300X (CDNA3, gfx942) ──────────────────────────────────────
// Constants are derived from the MI300X (CDNA3) architecture; see the
// per-field commentary below for the relevant references.
inline constexpr double _MI300X_CLOCK_GHZ = 2.0;

inline constexpr hardware_t MI300X = {
    /* arch                 */ "gfx942",
    /* num_cu               */ 304,
    /* num_xcd              */ 8,
    /* cu_per_xcd           */ 38,
    /* clock_ghz            */ _MI300X_CLOCK_GHZ,
    /* vmem_issue_rate      */ 1.0,
    /* valu_rate            */ 2.10 * 64.0,
    /* tcp_capacity_bytes   */ 32ULL * 1024ULL,
    /* tcp_bw               */ 64.0,
    /* mshr_depth_per_wave  */ 12,
    /* waves_per_wg         */ 10,
    /* xgmi_latency_cycles  */ 660.0 * _MI300X_CLOCK_GHZ,
    /* l2_capacity_bytes    */ 4ULL * 1024ULL * 1024ULL,
    /* l2_bw_per_cu         */ 83.6 / _MI300X_CLOCK_GHZ,
    /* mall_capacity_bytes  */ 256ULL * 1024ULL * 1024ULL,
    /* mall_bw              */ 4730.0 / _MI300X_CLOCK_GHZ,
    /* hbm_read_bw          */ 4730.0 / _MI300X_CLOCK_GHZ,
    /* hbm_write_bw         */ 5140.0 / _MI300X_CLOCK_GHZ,
    /* hbm_capacity_bytes   */ 192ULL * 1024ULL * 1024ULL * 1024ULL,
    /* mem_bw_coeffs        */ {0.0, 0.015, 0.0},
};

inline constexpr comm_hardware_t MI300X_COMM = {
    /* link_bw                 */ 49.1 * (1024.0 * 1024.0 * 1024.0) / 1e9 / 1.23 /
        _MI300X_CLOCK_GHZ,
    /* num_peer_links          */ 7,
    /* num_sdma_engines        */ 14,
    /* sdma_read_bw            */ 49.5 / _MI300X_CLOCK_GHZ,
    /* sdma_write_bw           */ 23.6 / _MI300X_CLOCK_GHZ,
    /* atomic_latency_cycles   */ 100.0 * _MI300X_CLOCK_GHZ,
    /* launch_overhead_cycles  */ 45000.0 * _MI300X_CLOCK_GHZ,
    /* clock_ghz               */ _MI300X_CLOCK_GHZ,
};

}  // namespace origami::comm
