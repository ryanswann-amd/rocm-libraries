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
// wg_tile latency computation. All times are in **GPU cycles**.
// Mirrors origami_comms/model/latency.py 1:1.
//
//     T_total = T_prologue + (num_iters - 1) × T_wlt + T_epilogue + T_sync
//
// The atom (T_wlt): cycles for one WG to transfer one iter_tile over
// one link, given all WGs contending for that link.
#pragma once

#include "origami/comm/hardware.hpp"
#include "origami/comm/heuristics.hpp"
#include "origami/comm/primitives.hpp"
#include "origami/comm/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace origami::comm {

// ─── iter_times_t: per-FU one-iteration cycle counts ───────────────
// Named fields instead of a string-keyed dict — matches Python's
// returned dict semantically; the bottleneck() helper resolves the
// FU with the max cycle count to a stable string.
struct iter_times_t {
  double vmem       = 0.0;
  double tcp        = 0.0;
  double l2         = 0.0;
  double mall       = 0.0;
  double hbm_read   = 0.0;
  double hbm_write  = 0.0;
  double xgmi_read  = 0.0;
  double xgmi_write = 0.0;
  double valu       = 0.0;

  constexpr double max_cycles() const noexcept {
    return std::max({vmem, tcp, l2, mall, hbm_read, hbm_write, xgmi_read, xgmi_write, valu});
  }

  // The FU name with the largest cycle count. Names match Python dict keys.
  constexpr std::string_view bottleneck() const noexcept {
    double best_v           = vmem;
    std::string_view best_k = "vmem";
    auto consider           = [&](double v, std::string_view k) {
      if (v > best_v) {
        best_v = v;
        best_k = k;
      }
    };
    consider(tcp, "tcp");
    consider(l2, "l2");
    consider(mall, "mall");
    consider(hbm_read, "hbm_read");
    consider(hbm_write, "hbm_write");
    consider(xgmi_read, "xgmi_read");
    consider(xgmi_write, "xgmi_write");
    consider(valu, "valu");
    return best_k;
  }
};

// ─── compute_iter_times ─────────────────────────────────────────
// Per-FU time (in cycles) for one inner-loop iteration. All hardware
// fields are cycle-based (bytes/cycle, cycles); the output is therefore
// in cycles too.
inline iter_times_t compute_iter_times(const functional_unit_work_t& work,
                                       const hardware_t& hw,
                                       const comm_hardware_t& comm_hw,
                                       double bw_per_wg,
                                       int active_cus,
                                       const heuristics_t& heur   = DEFAULT_HEURISTICS,
                                       std::string_view primitive = "") {
  constexpr double CL = static_cast<double>(CACHELINE_BYTES);
  iter_times_t t{};

  // ── VMEM ──
  const double vmem_total = static_cast<double>(work.vmem_read_instrs + work.vmem_write_instrs);
  if (hw.vmem_issue_rate > 0.0) t.vmem = vmem_total / hw.vmem_issue_rate;

  // ── TCP ──
  if (hw.tcp_bw > 0.0) {
    t.tcp = static_cast<double>(work.tcp_read_cl + work.tcp_write_cl) * CL / hw.tcp_bw;
  }

  // ── L2 (scaled by active CUs on this XCD) ──
  const int active_per_xcd =
      std::max<int>(static_cast<int>(std::ceil(static_cast<double>(active_cus) / hw.num_xcd)), 1);
  const double l2_bw = hw.l2_bw_per_cu_scaled(active_per_xcd);
  if (l2_bw > 0.0) { t.l2 = static_cast<double>(work.l2_read_cl + work.l2_write_cl) * CL / l2_bw; }

  // ── MALL ──
  if (hw.mall_bw > 0.0) {
    t.mall = static_cast<double>(work.mall_read_cl + work.mall_write_cl) * CL / hw.mall_bw;
  }

  // ── HBM (polynomial-scaled by active CUs) ──
  const double hbm_r_per_cu = hw.hbm_read_bw_per_cu(active_cus);
  const double hbm_w_per_cu = hw.hbm_write_bw_per_cu(active_cus);
  if (work.hbm_read_cl > 0 && hbm_r_per_cu > 0.0) {
    t.hbm_read = static_cast<double>(work.hbm_read_cl) * CL / hbm_r_per_cu;
  }
  if (work.hbm_write_cl > 0 && hbm_w_per_cu > 0.0) {
    t.hbm_write = static_cast<double>(work.hbm_write_cl) * CL / hbm_w_per_cu;
  }

  // ── xGMI read: min(link share, MSHR-limited) ──
  const double mshr_limited_bw =
      (static_cast<double>(hw.mshr_depth_per_wave) * hw.waves_per_wg * CL) / hw.xgmi_latency_cycles;
  const double effective_remote_read_bw = std::min(bw_per_wg, mshr_limited_bw);
  if (work.xgmi_read_cl > 0 && effective_remote_read_bw > 0.0) {
    t.xgmi_read = static_cast<double>(work.xgmi_read_cl) * CL / effective_remote_read_bw;
  }

  // ── xGMI write: concentration heuristic ──
  if (work.xgmi_write_cl > 0 && bw_per_wg > 0.0) {
    const double wgs_on_link         = std::max(comm_hw.link_bw / bw_per_wg, 1.0);
    const double k                   = heur.k_xgmi_write(primitive);
    const double util                = 1.0 - std::exp(-wgs_on_link / k);
    const double eff_link_bw         = comm_hw.link_bw * util;
    const double eff_write_bw_per_wg = eff_link_bw / wgs_on_link;
    t.xgmi_write = static_cast<double>(work.xgmi_write_cl) * CL / eff_write_bw_per_wg;
  }

  // ── VALU ──
  if (hw.valu_rate > 0.0 && work.valu_ops > 0) {
    t.valu = static_cast<double>(work.valu_ops) / hw.valu_rate;
  }

  return t;
}

// ─── iter_counts_from_tile ─────────────────────────────────────
// Resolve (num_iters, elements_per_iter) for one wg_tile.
//
//   contiguous=true : flat-byte iteration walk
//   contiguous=false: row-aware walk (per-row partial CL costs an iter)
//
// Returns {num_iters, elements_per_iter}. Both ≥ 1.
inline std::pair<std::size_t, std::size_t> iter_counts_from_tile(
    const std::optional<tile_shape_t>& wg_tile,
    std::size_t wg_tile_cachelines,
    std::size_t wg_tile_elements,
    std::size_t cl_per_iter) {
  if (wg_tile.has_value() && !wg_tile->contiguous) {
    const std::size_t iters_per_row =
        std::max<std::size_t>(ceil_div(wg_tile->cl_per_row(), cl_per_iter), 1);
    const std::size_t num_iters = std::max<std::size_t>(wg_tile->m * iters_per_row, 1);
    const std::size_t elements_per_iter =
        std::max<std::size_t>(ceil_div(wg_tile->n, iters_per_row), 1);
    return {num_iters, elements_per_iter};
  }
  const std::size_t num_iters = std::max<std::size_t>(ceil_div(wg_tile_cachelines, cl_per_iter), 1);
  const std::size_t elements_per_iter =
      std::max<std::size_t>(ceil_div(wg_tile_elements, num_iters), 1);
  return {num_iters, elements_per_iter};
}

// ─── compute_wg_tile_latency ────────────────────────────────────
// Full wg_tile transfer latency for one timestep, in cycles. Composes
// resolve_work_graph + compute_iter_times into the software-pipelined
// loop model.
inline wg_tile_latency_breakdown_t compute_wg_tile_latency(
    const std::vector<op_t>& work_graph,
    std::size_t wg_tile_cachelines,
    const comm_config_t& config,
    const hardware_t& hw,
    const comm_hardware_t& comm_hw,
    double bw_per_wg,
    std::size_t wg_tile_elements,
    std::optional<tile_shape_t> wg_tile = std::nullopt,
    std::optional<int> active_cus_opt   = std::nullopt,
    const heuristics_t& heur            = DEFAULT_HEURISTICS,
    std::string_view primitive          = "") {
  const std::size_t cl_per_iter = static_cast<std::size_t>(config.cl_per_iter());
  const int instrs_per_cl       = config.instrs_per_cl();

  auto [num_iters, elements_per_iter] =
      iter_counts_from_tile(wg_tile, wg_tile_cachelines, wg_tile_elements, cl_per_iter);

  const auto resolved = resolve_work_graph(
      work_graph,
      resolve_args_t{
          static_cast<int>(cl_per_iter), instrs_per_cl, static_cast<int>(elements_per_iter)});

  const int active_cus = active_cus_opt.value_or(config.num_wgs);
  const iter_times_t times =
      compute_iter_times(resolved.iter_work, hw, comm_hw, bw_per_wg, active_cus, heur, primitive);

  const double T_wlt = times.max_cycles();  // one iteration = one WLT

  // Prologue = max of READ path times that are >0.
  auto max_positive = [](std::initializer_list<double> xs) -> double {
    double best = 0.0;
    for (double v : xs)
      if (v > best) best = v;
    return best;
  };
  const double T_prologue = max_positive({times.hbm_read, times.xgmi_read, times.mall});
  const double T_epilogue = max_positive({times.hbm_write, times.xgmi_write});

  const double T_sync =
      static_cast<double>(resolved.sync_work.atomic_count) * comm_hw.atomic_latency_cycles;

  const double T_total = T_prologue +
                         std::max<double>(static_cast<double>(num_iters) - 1.0, 0.0) * T_wlt +
                         T_epilogue + T_sync;

  wg_tile_latency_breakdown_t out;
  out.T_total_cycles    = T_total;
  out.T_wlt_cycles      = T_wlt;
  out.T_prologue_cycles = T_prologue;
  out.T_epilogue_cycles = T_epilogue;
  out.T_sync_cycles     = T_sync;
  out.num_iters         = num_iters;

  out.T_vmem_cycles       = times.vmem;
  out.T_tcp_cycles        = times.tcp;
  out.T_l2_cycles         = times.l2;
  out.T_mall_cycles       = times.mall;
  out.T_hbm_read_cycles   = times.hbm_read;
  out.T_hbm_write_cycles  = times.hbm_write;
  out.T_xgmi_read_cycles  = times.xgmi_read;
  out.T_xgmi_write_cycles = times.xgmi_write;
  out.T_valu_cycles       = times.valu;
  out.bottleneck          = std::string{times.bottleneck()};

  out.clock_ghz = hw.clock_ghz;
  return out;
}

}  // namespace origami::comm
