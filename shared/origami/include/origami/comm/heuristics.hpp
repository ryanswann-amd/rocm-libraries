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
// Empirical heuristic weights.
//
// Distinguish from neighbors:
//   hardware.hpp   — measured datasheet constants (facts about silicon)
//   heuristics.hpp — this file: empirical fits expected to drift with
//                    measurement coverage
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace origami::comm {

// ─── primitive_t enum + name table ─────────────────────────────────
// The canonical string keys for each primitive.
enum class primitive_t : std::uint8_t {
  all_gather,
  reduce_scatter,
  broadcast,
  all_reduce,
  all_to_all,
};

inline constexpr std::array<std::string_view, 5> PRIMITIVE_NAMES = {
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "all_reduce",
    "all_to_all",
};

constexpr std::string_view primitive_name(primitive_t p) noexcept {
  return PRIMITIVE_NAMES[static_cast<std::size_t>(p)];
}

// ─── framework_t enum + name table ─────────────────────────────────
enum class framework_t : std::uint8_t {
  raw,
  rccl,
  nccl,
  torch,
  jax,
  mpi,
};

inline constexpr std::array<std::string_view, 6> FRAMEWORK_NAMES = {
    "raw",
    "rccl",
    "nccl",
    "torch",
    "jax",
    "mpi",
};

constexpr std::string_view framework_name(framework_t f) noexcept {
  return FRAMEWORK_NAMES[static_cast<std::size_t>(f)];
}

// ─── heuristics_t ──────────────────────────────────────────────────
// The single home for empirical fudge factors. Override per-study by
// constructing a custom instance and passing it through
// `predict_tensor_collective`.
struct heuristics_t {
  // The MI300X clock used at C++ scope to convert the *_NS table
  // (host-meaningful units) into cycles (model-internal units). If
  // you ever override clock_ghz at study time, recompute these.
  static constexpr double MI300X_CLOCK_GHZ = 2.0;

  // ── WG-cap heuristic ────────────────────────────────────────
  // 16 KiB matches NCCL LL128 min chunk; fitted to rccl_master_sweep.
  int min_bytes_per_wg = 16'384;

  // ── framework_t overhead floor (nanoseconds, HOST wall time) ──
  // Indexed by framework_t enum. Only `torch` has a non-zero floor.
  std::array<double, 6> framework_overhead_ns = {
      0.0,        // raw
      0.0,        // rccl
      0.0,        // nccl
      400'000.0,  // torch (~400 µs MI300X floor — see calibration caveat)
      0.0,        // jax
      0.0,        // mpi
  };

  // ── Per-ring-step proxy/sync overhead (GPU cycles) ──────────
  // Defaults in ns: AG=10000, RS=4000, others=0. Stored directly in
  // cycles using MI300X_CLOCK_GHZ.
  std::array<double, 5> ring_step_overhead_cycles = {
      /* all_gather     */ 10'000.0 * MI300X_CLOCK_GHZ,
      /* reduce_scatter */ 4'000.0 * MI300X_CLOCK_GHZ,
      /* broadcast      */ 0.0,
      /* all_reduce     */ 0.0,
      /* all_to_all     */ 0.0,
  };

  // ── xGMI write concentration efficiency ─────────────────────
  // effective_link_util(wgs) = 1 - exp(-wgs / k).
  double xgmi_write_concentration_k_default                     = 4.0;
  std::array<double, 5> xgmi_write_concentration_k_by_primitive = {
      /* all_gather     */ 4.0,
      /* reduce_scatter */ 3.0,
      /* broadcast      */ 3.5,
      /* all_reduce     */ 6.0,
      /* all_to_all     */ 4.0,
  };

  constexpr double k_xgmi_write(primitive_t p) const noexcept {
    return xgmi_write_concentration_k_by_primitive[static_cast<std::size_t>(p)];
  }

  // String-keyed overload — used at the public API edge where the
  // caller passes a name (falls back to default on unknown).
  constexpr double k_xgmi_write(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < PRIMITIVE_NAMES.size(); ++i) {
      if (PRIMITIVE_NAMES[i] == name) { return xgmi_write_concentration_k_by_primitive[i]; }
    }
    return xgmi_write_concentration_k_default;
  }

  constexpr double ring_step_overhead(primitive_t p) const noexcept {
    return ring_step_overhead_cycles[static_cast<std::size_t>(p)];
  }
  constexpr double ring_step_overhead(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < PRIMITIVE_NAMES.size(); ++i) {
      if (PRIMITIVE_NAMES[i] == name) { return ring_step_overhead_cycles[i]; }
    }
    return 0.0;
  }

  constexpr double framework_overhead_us(framework_t f) const noexcept {
    return framework_overhead_ns[static_cast<std::size_t>(f)] / 1000.0;
  }
  constexpr double framework_overhead_us(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < FRAMEWORK_NAMES.size(); ++i) {
      if (FRAMEWORK_NAMES[i] == name) { return framework_overhead_ns[i] / 1000.0; }
    }
    return 0.0;
  }
};

inline constexpr heuristics_t DEFAULT_HEURISTICS{};

}  // namespace origami::comm
