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
// End-to-end collective latency: composes the per-WG-tile atom from
// latency.hpp over a collective's full communication schedule.
//
// The composition depends on how the schedule's timesteps relate in time,
// which is itself a property of the algorithm:
//
//   1. Sequential timesteps (latency-composed). The timesteps are data-
//      dependent — each must finish before the next begins (e.g. the two
//      phases of a two-shot all-reduce). They may use different links, so the
//      total is the *sum* of per-timestep tile latencies. Small messages live
//      here: the cost is a chain of fill/drain/handshake constants.
//
//   2. Pipelined ring (throughput-composed). All steps stream over the same
//      ring link and the steps overlap across WGs, so the ring behaves as one
//      long pipe. Latency does not add up step-by-step; instead total wire
//      bytes are divided by the *aggregate* sustainable throughput, then a
//      fixed sync + per-step overhead is added. Large messages live here: the
//      cost approaches bytes ÷ bandwidth.
//
// Every path also adds the once-per-launch kernel overhead, which is what
// dominates the sub-kilobyte regime.
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
inline double ring_step_overhead_cycles(primitive_t primitive,
                                        const collective_layout_t& layout,
                                        const heuristics_t& heur) {
  if (!layout.is_ring_class()) return 0.0;
  const double per_step = heur.ring_step_overhead(primitive);
  return per_step * static_cast<double>(layout.num_timesteps());
}

// ─── default layout factory ─────────────────────────────────────
// Maps each collective operation to its default layout (the implementation
// the engine uses when comm_config_t::layout is not overridden).
inline std::unique_ptr<collective_layout_t> default_layout_for(primitive_t collective,
                                                               int num_gpus) {
  switch (collective) {
    case primitive_t::all_gather: return allgather_layout(num_gpus);
    case primitive_t::reduce_scatter: return reduce_scatter_layout(num_gpus);
    case primitive_t::all_reduce: return allreduce_two_shot_layout(num_gpus);
    case primitive_t::all_to_all: return alltoall_layout(num_gpus);
    case primitive_t::broadcast: return broadcast_layout(num_gpus);
  }
  throw std::invalid_argument(std::string{"Unknown collective: "} +
                              std::string{primitive_name(collective)});
}

// ─── _compute_ring_latency ──────────────────────────────────────
inline double compute_ring_latency(const collective_layout_t& layout,
                                   const comm_problem_t& problem,
                                   const comm_config_t& config,
                                   const system_t& system,
                                   int my_rank,
                                   const heuristics_t& heur,
                                   primitive_t primitive) {
  const hardware_t& hw           = system.gpu;
  const comm_hardware_t& comm_hw = system.fabric;
  const int N                    = problem.num_gpus;
  const int num_timesteps        = layout.num_timesteps();
  const std::size_t CL           = CACHELINE_BYTES;

  // A ring moves one chunk per step; over num_timesteps steps each GPU pushes
  // num_timesteps such chunks across its outgoing link. That product is the
  // total bytes this rank puts on the wire — the numerator of the throughput
  // model.
  const std::size_t gpu_timestep_tile_bytes =
      problem.gpu_tile_cachelines() * CL / static_cast<std::size_t>(layout.chunks_per_timestep());
  const std::size_t total_wire_bytes =
      gpu_timestep_tile_bytes * static_cast<std::size_t>(num_timesteps);

  const int eff_wgs = config.effective_num_wgs(gpu_timestep_tile_bytes);

  // Per-WG remote throughput is the same latency cap as in latency.hpp:
  // outstanding misses (mshr_depth × waves × CL) drained every RTT.
  const double mshr_bw_per_wg =
      (static_cast<double>(hw.mshr_depth_per_wave) * hw.waves_per_wg * static_cast<double>(CL)) /
      hw.xgmi_latency_cycles;

  // The ring's sustainable rate is whichever ceiling binds first: the physical
  // link width, or the combined latency-limited throughput of the WGs feeding
  // it (eff_wgs × per-WG cap). Few WGs ⇒ CU-limited; many WGs ⇒ link-limited.
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
  // Each ring step needs its own handshake, and the steps are serialized by
  // the dependency chain, so sync cost accrues per step.
  const double T_sync_per_step = static_cast<double>(sync_ops) * comm_hw.atomic_latency_cycles;
  const double T_sync_total    = static_cast<double>(num_timesteps) * T_sync_per_step;

  // The same bytes that cross the fabric must also be read from / written to
  // local HBM, which has its own (CU-count-scaled) aggregate ceiling. The ring
  // can be bound by either resource, so take the slower of fabric and HBM.
  const double hbm_bw_agg = hw.hbm_read_bw * hw.bw_fraction(eff_wgs);
  const double T_hbm      = static_cast<double>(total_wire_bytes) / hbm_bw_agg;

  const double T_transfer_total = std::max(T_transfer, T_hbm);

  // Per-step proxy/handshake overhead the bandwidth model cannot see (CPU-
  // mediated; empirical, from heuristics).
  const double T_step_overhead = ring_step_overhead_cycles(primitive, layout, heur);

  // Fixed launch floor + the throughput-bound transfer + serial sync + per-step
  // overhead. Launch dominates tiny messages; transfer dominates large ones.
  return comm_hw.launch_overhead_cycles + T_transfer_total + T_sync_total + T_step_overhead;
}

// ─── _compute_sequential_latency ────────────────────────────────
inline double compute_sequential_latency(const collective_layout_t& layout,
                                         const comm_problem_t& problem,
                                         const comm_config_t& config,
                                         const system_t& system,
                                         int my_rank,
                                         const heuristics_t& heur,
                                         primitive_t primitive) {
  const hardware_t& hw           = system.gpu;
  const comm_hardware_t& comm_hw = system.fabric;
  const int N                    = problem.num_gpus;
  const int chunks_per_timestep  = layout.chunks_per_timestep();

  const tile_shape_t gpu_tile = problem.gpu_tile_shape();
  const tile_shape_t gpu_timestep_tile =
      gpu_tile.divide_byte_equal(static_cast<std::size_t>(chunks_per_timestep));

  const int eff_wgs = config.effective_num_wgs(gpu_timestep_tile.bytes());

  const tile_shape_t wg_tile =
      gpu_timestep_tile.divide_byte_equal(static_cast<std::size_t>(eff_wgs));
  const std::size_t wg_tile_cachelines = std::max<std::size_t>(wg_tile.cachelines(), 1);
  const std::size_t wg_tile_elements   = std::max<std::size_t>(wg_tile.elements(), 1);

  // Timesteps are data-dependent here, so their latencies add up. Within a
  // timestep, however, the links run in parallel — so a timestep costs the
  // *slowest* link, not their sum (T_link_max below).
  double T_timesteps = 0.0;
  for (int timestep = 0; timestep < layout.num_timesteps(); ++timestep) {
    const auto entry = layout.link_of(/*pid=*/0, timestep, my_rank, N);

    if (entry.is_self) {
      // A "self" step is a local copy (no peer): bound by local HBM, so the
      // per-WG budget is the HBM per-CU share rather than a link share.
      const double bw_per_wg = hw.hbm_read_bw_per_cu(eff_wgs);
      const auto breakdown =
          compute_wg_tile_latency(entry.work_graph,
                                  wg_tile_cachelines,
                                  config,
                                  system,
                                  bw_per_wg,
                                  wg_tile_elements,
                                  /*wg_tile=*/std::optional<tile_shape_t>{wg_tile},
                                  /*active_cus=*/std::optional<int>{eff_wgs},
                                  heur,
                                  primitive);
      T_timesteps += breakdown.T_total_cycles;
    } else {
      // A remote step may light up several links at once; active_links reports
      // how the eff_wgs workgroups are distributed over them. Each link's WGs
      // share that link's width evenly, and the timestep waits for the most
      // congested link to finish — hence the max over links.
      const auto link_wg_counts = layout.active_links(timestep, eff_wgs, N);

      double T_link_max = 0.0;
      for (const auto& [link_id, wgs_on_link] : link_wg_counts) {
        const double bw_per_wg = comm_hw.link_bw / static_cast<double>(std::max(wgs_on_link, 1));

        const auto breakdown =
            compute_wg_tile_latency(entry.work_graph,
                                    wg_tile_cachelines,
                                    config,
                                    system,
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

// ─── compute_collective_latency_for_rank ────────────────────────
// Predicted GPU cycles for *one* rank's timeline. This is the per-rank atom
// and the diagnostic entry point: call it directly to inspect whether ranks
// diverge. Caller converts cycles→µs at the public boundary.
//
// The operation comes from problem.collective (what to compute) and the
// implementation from config.layout (how) — null layout means use the default
// for the operation. This is the problem/config split: correctness inputs in
// the problem, performance inputs in the config.
inline double compute_collective_latency_for_rank(const comm_problem_t& problem,
                                                  const comm_config_t& config,
                                                  const system_t& system,
                                                  int my_rank,
                                                  const heuristics_t& heur = DEFAULT_HEURISTICS) {
  std::unique_ptr<collective_layout_t> owned;
  const collective_layout_t* L = config.layout;
  if (!L) {
    owned = default_layout_for(problem.collective, problem.num_gpus);
    L     = owned.get();
  }

  if (L->is_ring_pipeline()) {
    return compute_ring_latency(*L, problem, config, system, my_rank, heur, problem.collective);
  }
  return compute_sequential_latency(*L, problem, config, system, my_rank, heur, problem.collective);
}

// ─── compute_collective_latency ─────────────────────────────────
// Predicted GPU cycles for the whole collective. The operation completes only
// when its slowest participant does, so the cost is the *max* of every rank's
// timeline — this loop is where rank asymmetry, if any layout ever introduces
// it, would surface.
//
// Shortcut: with heur.assume_rank_symmetry the loop collapses to rank 0 alone
// (see heuristics_t — exact for the rank-symmetric layouts we ship today, an
// N× speedup). Default is the honest max so the engine stays correct for any
// future asymmetric layout without a flag change.
inline double compute_collective_latency(const comm_problem_t& problem,
                                         const comm_config_t& config,
                                         const system_t& system,
                                         const heuristics_t& heur = DEFAULT_HEURISTICS) {
  if (heur.assume_rank_symmetry) {
    return compute_collective_latency_for_rank(problem, config, system, /*my_rank=*/0, heur);
  }

  double T_max = 0.0;
  for (int rank = 0; rank < problem.num_gpus; ++rank) {
    T_max =
        std::max(T_max, compute_collective_latency_for_rank(problem, config, system, rank, heur));
  }
  return T_max;
}

// ─── predict_row ────────────────────────────────────────────────
// The byte-level public entry point: predict one collective call's latency in
// microseconds. Its job is to translate a benchmark row's conventions into a
// comm_problem_t/comm_config_t and then defer to the model above.
//
// The one subtlety it owns is the message-size convention: most collectives
// report msg_bytes as the per-rank buffer, but reduce_scatter reports the full
// pre-scatter buffer, so its per-rank share is msg_bytes / world_size. When no
// explicit [M,N] shape is given, the buffer is treated as a 1×N row of bf16
// elements. cl/sync contention, layout, and unit conversion are all delegated;
// the cycles→µs conversion happens here, at the boundary.
inline double predict_row(std::string_view primitive,
                          std::size_t msg_bytes,
                          int world_size,
                          int nchannels,
                          const system_t& system,
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
  problem.collective = primitive_from_name(primitive);  // string → enum at the edge
  comm_config_t config{};
  config.num_wgs          = nchannels;
  config.load_width       = load_width_t::DWORDX16;
  config.vgprs_for_data   = 128;
  config.min_bytes_per_wg = heur.min_bytes_per_wg;

  const double T_cycles = compute_collective_latency(problem, config, system, heur);

  return system.gpu.cycles_to_us(T_cycles);
}

}  // namespace origami::comm
