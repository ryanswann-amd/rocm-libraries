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
// End-to-end collective latency. Mirrors origami_comms/model/collective.py.
//
// Two computation modes based on layout structure:
//   1. Sequential timesteps: each timestep may use a different link;
//      T = sum over timesteps of compute_wg_tile_latency(...).
//   2. Pipelined ring: all timesteps use the same link, multiple WGs
//      pipeline through ring steps; T = total_data/aggregate_throughput
//      + sync + per-step heuristic overhead.
#pragma once

#include "origami/comm/hardware.hpp"
#include "origami/comm/heuristics.hpp"
#include "origami/comm/latency.hpp"
#include "origami/comm/layouts.hpp"
#include "origami/comm/primitives.hpp"
#include "origami/comm/types.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace origami::comm {

// ─── ring-step heuristic ────────────────────────────────────────
inline double ring_step_overhead_cycles(std::string_view primitive,
                                        const collective_layout_t& layout,
                                        const heuristics_t& heur) {
  if (!layout.is_ring_class()) return 0.0;
  if (primitive.empty()) return 0.0;
  const double per_step = heur.ring_step_overhead(primitive);
  return per_step * static_cast<double>(layout.num_timesteps());
}

// ─── default layout factory ─────────────────────────────────────
// Mirrors Python COLLECTIVE_LAYOUTS dict.
inline std::unique_ptr<collective_layout_t> default_layout_for(std::string_view collective,
                                                               int num_gpus) {
  if (collective == "all_gather") return allgather_layout(num_gpus);
  if (collective == "reduce_scatter") return reduce_scatter_layout(num_gpus);
  if (collective == "all_reduce") return allreduce_two_shot_layout(num_gpus);
  if (collective == "all_to_all") return alltoall_layout(num_gpus);
  if (collective == "broadcast") return broadcast_layout(num_gpus);
  throw std::invalid_argument(std::string{"Unknown collective: "} + std::string{collective});
}

// ─── _compute_ring_latency ──────────────────────────────────────
inline double compute_ring_latency(const collective_layout_t& layout,
                                   const comm_problem_t& problem,
                                   const comm_config_t& config,
                                   const hardware_t& hw,
                                   const comm_hardware_t& comm_hw,
                                   int my_rank,
                                   const heuristics_t& heur,
                                   std::string_view primitive) {
  const int N             = problem.num_gpus;
  const int num_timesteps = layout.num_timesteps();
  const std::size_t CL    = CACHELINE_BYTES;

  // Per-timestep data crossing the link per GPU.
  const std::size_t gpu_timestep_tile_bytes =
      problem.gpu_tile_cachelines() * CL / static_cast<std::size_t>(layout.chunks_per_timestep());
  const std::size_t total_wire_bytes =
      gpu_timestep_tile_bytes * static_cast<std::size_t>(num_timesteps);

  const int eff_wgs = config.effective_num_wgs(gpu_timestep_tile_bytes);

  // Per-WG MSHR-limited throughput (bytes/cycle).
  const double mshr_bw_per_wg =
      (static_cast<double>(hw.mshr_depth_per_wave) * hw.waves_per_wg * static_cast<double>(CL)) /
      hw.xgmi_latency_cycles;

  // Aggregate throughput: physical link cap vs. CU-limited.
  const double aggregate_bw =
      std::min(comm_hw.link_bw, static_cast<double>(eff_wgs) * mshr_bw_per_wg);

  const double T_transfer = static_cast<double>(total_wire_bytes) / aggregate_bw;

  // Sync: count signal_t+wait_t ops in the work graph times atomic_latency_cycles.
  const auto entry = layout.link_of(/*pid=*/0, /*timestep=*/0, my_rank, N);
  int sync_ops     = 0;
  for (const auto& op : entry.work_graph) {
    std::visit(
        [&](const auto& concrete) {
          using T = std::decay_t<decltype(concrete)>;
          if constexpr (std::is_same_v<T, signal_t> || std::is_same_v<T, wait_t>) { ++sync_ops; }
        },
        op);
  }
  const double T_sync_per_step = static_cast<double>(sync_ops) * comm_hw.atomic_latency_cycles;
  const double T_sync_total    = static_cast<double>(num_timesteps) * T_sync_per_step;

  // Local HBM read/write also shares bandwidth — picks the bottleneck.
  const double hbm_bw_agg = hw.hbm_read_bw * hw.bw_fraction(eff_wgs);
  const double T_hbm      = static_cast<double>(total_wire_bytes) / hbm_bw_agg;

  const double T_transfer_total = std::max(T_transfer, T_hbm);

  const double T_step_overhead = ring_step_overhead_cycles(primitive, layout, heur);

  return comm_hw.launch_overhead_cycles + T_transfer_total + T_sync_total + T_step_overhead;
}

// ─── _compute_sequential_latency ────────────────────────────────
inline double compute_sequential_latency(const collective_layout_t& layout,
                                         const comm_problem_t& problem,
                                         const comm_config_t& config,
                                         const hardware_t& hw,
                                         const comm_hardware_t& comm_hw,
                                         int my_rank,
                                         const heuristics_t& heur,
                                         std::string_view primitive) {
  const int N                   = problem.num_gpus;
  const int chunks_per_timestep = layout.chunks_per_timestep();

  const tile_shape_t gpu_tile = problem.gpu_tile_shape();
  const tile_shape_t gpu_timestep_tile =
      gpu_tile.divide_byte_equal(static_cast<std::size_t>(chunks_per_timestep));

  const int eff_wgs = config.effective_num_wgs(gpu_timestep_tile.bytes());

  const tile_shape_t wg_tile =
      gpu_timestep_tile.divide_byte_equal(static_cast<std::size_t>(eff_wgs));
  const std::size_t wg_tile_cachelines = std::max<std::size_t>(wg_tile.cachelines(), 1);
  const std::size_t wg_tile_elements   = std::max<std::size_t>(wg_tile.elements(), 1);

  double T_timesteps = 0.0;
  for (int timestep = 0; timestep < layout.num_timesteps(); ++timestep) {
    const auto entry = layout.link_of(/*pid=*/0, timestep, my_rank, N);

    if (entry.is_self) {
      const double bw_per_wg = hw.hbm_read_bw_per_cu(eff_wgs);
      const auto breakdown =
          compute_wg_tile_latency(entry.work_graph,
                                  wg_tile_cachelines,
                                  config,
                                  hw,
                                  comm_hw,
                                  bw_per_wg,
                                  wg_tile_elements,
                                  /*wg_tile=*/std::optional<tile_shape_t>{wg_tile},
                                  /*active_cus=*/std::optional<int>{eff_wgs},
                                  heur,
                                  primitive);
      T_timesteps += breakdown.T_total_cycles;
    } else {
      const auto link_wg_counts = layout.active_links(timestep, eff_wgs, N);

      double T_link_max = 0.0;
      for (const auto& [link_id, wgs_on_link] : link_wg_counts) {
        const double bw_per_wg = comm_hw.link_bw / static_cast<double>(std::max(wgs_on_link, 1));

        const auto breakdown =
            compute_wg_tile_latency(entry.work_graph,
                                    wg_tile_cachelines,
                                    config,
                                    hw,
                                    comm_hw,
                                    bw_per_wg,
                                    wg_tile_elements,
                                    /*wg_tile=*/std::optional<tile_shape_t>{wg_tile},
                                    /*active_cus=*/std::optional<int>{eff_wgs},
                                    heur,
                                    primitive);
        T_link_max = std::max(T_link_max, breakdown.T_total_cycles);
      }
      T_timesteps += T_link_max;
    }
  }

  const double T_step_overhead = ring_step_overhead_cycles(primitive, layout, heur);
  return comm_hw.launch_overhead_cycles + T_timesteps + T_step_overhead;
}

// ─── compute_collective_latency ─────────────────────────────────
// Returns total predicted GPU cycles. Caller converts to seconds via
// hw.cycles_to_us at the public boundary.
inline double compute_collective_latency(std::string_view collective,
                                         const comm_problem_t& problem,
                                         const comm_config_t& config,
                                         const hardware_t& hw,
                                         const comm_hardware_t& comm_hw,
                                         const collective_layout_t* layout = nullptr,
                                         int my_rank                       = 0,
                                         const heuristics_t& heur          = DEFAULT_HEURISTICS) {
  std::unique_ptr<collective_layout_t> owned;
  const collective_layout_t* L = layout;
  if (!L) {
    owned = default_layout_for(collective, problem.num_gpus);
    L     = owned.get();
  }

  if (L->is_ring_pipeline()) {
    return compute_ring_latency(*L, problem, config, hw, comm_hw, my_rank, heur, collective);
  }
  return compute_sequential_latency(*L, problem, config, hw, comm_hw, my_rank, heur, collective);
}

// ─── predict_row ────────────────────────────────────────────────
// Predict latency in MICROSECONDS for one row of rccl_master_sweep.csv.
// Handles the AR/AG/BC/A2A vs RS msg_bytes convention.
inline double predict_row(std::string_view primitive,
                          std::size_t msg_bytes,
                          int world_size,
                          int nchannels,
                          const hardware_t& hw,
                          const comm_hardware_t& comm_hw,
                          std::size_t M            = 0,
                          std::size_t N            = 0,
                          int split_dim            = 0,
                          const heuristics_t& heur = DEFAULT_HEURISTICS) {
  constexpr data_type_t dtype = data_type_t::BFloat16;

  if (M == 0 || N == 0) {
    const std::size_t total_elements = msg_bytes / static_cast<std::size_t>(dtype_bytes(dtype));
    std::size_t per_rank_elements;
    if (primitive == "reduce_scatter") {
      per_rank_elements =
          std::max<std::size_t>(total_elements / static_cast<std::size_t>(world_size), 1);
    } else {
      per_rank_elements = total_elements;
    }
    M = 1;
    N = per_rank_elements;
  }

  comm_problem_t problem{M, N, world_size, dtype, split_dim};
  comm_config_t config{};
  config.num_wgs          = nchannels;
  config.load_width       = load_width_t::DWORDX16;
  config.vgprs_for_data   = 128;
  config.min_bytes_per_wg = heur.min_bytes_per_wg;

  const double T_cycles = compute_collective_latency(primitive,
                                                     problem,
                                                     config,
                                                     hw,
                                                     comm_hw,
                                                     /*layout=*/nullptr,
                                                     /*my_rank=*/0,
                                                     heur);

  return hw.cycles_to_us(T_cycles);
}

}  // namespace origami::comm
