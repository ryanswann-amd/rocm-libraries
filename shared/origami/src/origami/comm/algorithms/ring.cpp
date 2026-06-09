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

#include "origami/comm/algorithms/ring.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

namespace origami::comm {

// Spread num_wgs over the ring links, conserving the total: each of the
// nrings=min(num_wgs, N-1) links gets floor(num_wgs/nrings), and the first
// `extra` links take one more so the counts sum back to num_wgs.
std::unordered_map<int, int> ring_distribute(int num_wgs, int num_gpus) {
  const int nrings = std::max(std::min(num_wgs, num_gpus - 1), 1);
  const int base   = num_wgs / nrings;
  const int extra  = num_wgs - base * nrings;
  std::unordered_map<int, int> out;
  for (int i = 0; i < nrings; ++i) { out[i] = base + (i < extra ? 1 : 0); }
  return out;
}

// Floored workgroups-per-ring-link share (clamped to >= 1) used to price per-link contention.
int ring_wgs_per_link(int num_wgs, int num_gpus) noexcept {
  const int nrings = std::max(std::min(num_wgs, num_gpus - 1), 1);
  return std::max(num_wgs / nrings, 1);
}

// ring_fixed: pipelined ring whose every hop crosses the fixed next-rank neighbour link,
// carrying a wait/signal producer->consumer dependency between adjacent ranks (the older
// ring all-reduce form).
//
// Store the communicator size and the per-hop work-graph closure (default: reduce-ring with
// signal/wait).
ring_fixed_algorithm_t::ring_fixed_algorithm_t(int num_gpus, work_graph_fn_t wg_fn)
    : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

// Every hop pushes to the fixed next-rank neighbour.
schedule_entry_t ring_fixed_algorithm_t::link_of(int /*pid*/, int /*timestep*/, int my_rank) const {
  const int next_rank = floor_mod(my_rank + 1, num_gpus_);
  auto work           = wg_fn_(next_rank, my_rank, num_gpus_, false);
  return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
}

// Floored ring share (see ring_wgs_per_link).
int ring_fixed_algorithm_t::wgs_on_link(int /*timestep*/, int num_wgs) const {
  return ring_wgs_per_link(num_wgs, num_gpus_);
}

// Workgroups distributed across the ring links (see ring_distribute).
std::unordered_map<int, int> ring_fixed_algorithm_t::active_links(int /*timestep*/,
                                                                  int num_wgs) const {
  return ring_distribute(num_wgs, num_gpus_);
}

// N-1 ring hops.
int ring_fixed_algorithm_t::num_timesteps() const { return num_gpus_ - 1; }
// Each hop moves 1/N of the buffer.
int ring_fixed_algorithm_t::chunks_per_timestep() const { return num_gpus_; }
// Ring-class: eligible for the per-step overhead heuristic.
bool ring_fixed_algorithm_t::is_ring_class() const { return true; }
// Pipelined ring: priced by the closed-form throughput model.
bool ring_fixed_algorithm_t::is_ring_pipeline() const { return true; }

// Reduce-ring hop: wait on prev, pull from prev, reduce, store, signal next.
std::vector<op_t> ring_fixed_algorithm_t::default_work_graph(int /*peer*/,
                                                             int my_rank,
                                                             int num_gpus,
                                                             bool /*is_self*/) {
  const int next_rank = floor_mod(my_rank + 1, num_gpus);
  const int prev_rank = floor_mod(my_rank - 1, num_gpus);
  return {
      load_t{},
      wait_t{prev_rank},
      pull_t{prev_rank},
      reduce_t{},
      store_t{},
      signal_t{next_rank},
  };
}

// ring_all_gather: all-gather as an N-1 hop ring, each step loading locally and forwarding
// its slice to the next rank.
//
// Store the communicator size.
ring_all_gather_algorithm_t::ring_all_gather_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

// Each step loads locally, stores, and pushes the slice to the next rank.
schedule_entry_t ring_all_gather_algorithm_t::link_of(int /*pid*/,
                                                      int /*timestep*/,
                                                      int my_rank) const {
  const int next_rank    = floor_mod(my_rank + 1, num_gpus_);
  std::vector<op_t> work = {
      load_t{},
      store_t{},
      push_t{next_rank},
  };
  return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
}

// Floored ring share (see ring_wgs_per_link).
int ring_all_gather_algorithm_t::wgs_on_link(int /*timestep*/, int num_wgs) const {
  return ring_wgs_per_link(num_wgs, num_gpus_);
}

// Workgroups distributed across the ring links (see ring_distribute).
std::unordered_map<int, int> ring_all_gather_algorithm_t::active_links(int /*timestep*/,
                                                                       int num_wgs) const {
  return ring_distribute(num_wgs, num_gpus_);
}

// N-1 ring hops.
int ring_all_gather_algorithm_t::num_timesteps() const { return num_gpus_ - 1; }
// One chunk per step (the per-rank send).
int ring_all_gather_algorithm_t::chunks_per_timestep() const { return 1; }
// Ring-class: eligible for the per-step overhead heuristic.
bool ring_all_gather_algorithm_t::is_ring_class() const { return true; }

// ring_reduce_scatter: the all-gather ring with a reduce on every hop, so each step
// loads, reduces, stores, and forwards the slice to the next rank.
//
// Store the communicator size.
ring_reduce_scatter_algorithm_t::ring_reduce_scatter_algorithm_t(int num_gpus)
    : num_gpus_{num_gpus} {}

// Each step loads, reduces, stores, and pushes the slice to the next rank.
schedule_entry_t ring_reduce_scatter_algorithm_t::link_of(int /*pid*/,
                                                          int /*timestep*/,
                                                          int my_rank) const {
  const int next_rank    = floor_mod(my_rank + 1, num_gpus_);
  std::vector<op_t> work = {
      load_t{},
      reduce_t{},
      store_t{},
      push_t{next_rank},
  };
  return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
}

// Floored ring share (see ring_wgs_per_link).
int ring_reduce_scatter_algorithm_t::wgs_on_link(int /*timestep*/, int num_wgs) const {
  return ring_wgs_per_link(num_wgs, num_gpus_);
}

// Workgroups distributed across the ring links (see ring_distribute).
std::unordered_map<int, int> ring_reduce_scatter_algorithm_t::active_links(int /*timestep*/,
                                                                           int num_wgs) const {
  return ring_distribute(num_wgs, num_gpus_);
}

// N-1 ring hops.
int ring_reduce_scatter_algorithm_t::num_timesteps() const { return num_gpus_ - 1; }
// One chunk per step (the per-rank slice).
int ring_reduce_scatter_algorithm_t::chunks_per_timestep() const { return 1; }
// Ring-class: eligible for the per-step overhead heuristic.
bool ring_reduce_scatter_algorithm_t::is_ring_class() const { return true; }

// ring_all_reduce: bandwidth-optimal all-reduce — a reduce-scatter ring then an all-gather
// ring, 2(N-1) hops all crossing the same neighbour link, making it a true pipelined ring
// priced by aggregate throughput rather than a sum of per-step latencies.
//
// Store the communicator size.
ring_all_reduce_algorithm_t::ring_all_reduce_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

// Reduce-scatter phase (pull+reduce) for the first N-1 steps, then all-gather (pull only);
// both phases cross the fixed prev->next neighbour link with a signal/wait dependency.
schedule_entry_t ring_all_reduce_algorithm_t::link_of(int /*pid*/,
                                                      int timestep,
                                                      int my_rank) const {
  const int next_rank = floor_mod(my_rank + 1, num_gpus_);
  const int prev_rank = floor_mod(my_rank - 1, num_gpus_);
  const int rs_visits = num_gpus_ - 1;

  std::vector<op_t> work;
  if (timestep < rs_visits) {
    work = {
        load_t{},
        wait_t{prev_rank},
        pull_t{prev_rank},
        reduce_t{},
        store_t{},
        signal_t{next_rank},
    };
  } else {
    work = {
        wait_t{prev_rank},
        pull_t{prev_rank},
        store_t{},
        signal_t{next_rank},
    };
  }
  return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
}

// Floored ring share (see ring_wgs_per_link).
int ring_all_reduce_algorithm_t::wgs_on_link(int /*timestep*/, int num_wgs) const {
  return ring_wgs_per_link(num_wgs, num_gpus_);
}

// Workgroups distributed across the ring links (see ring_distribute).
std::unordered_map<int, int> ring_all_reduce_algorithm_t::active_links(int /*timestep*/,
                                                                       int num_wgs) const {
  return ring_distribute(num_wgs, num_gpus_);
}

// 2(N-1) hops: a reduce-scatter ring followed by an all-gather ring.
int ring_all_reduce_algorithm_t::num_timesteps() const { return 2 * (num_gpus_ - 1); }
// Each hop moves 1/N of the buffer.
int ring_all_reduce_algorithm_t::chunks_per_timestep() const { return num_gpus_; }
// Ring-class: eligible for the per-step overhead heuristic.
bool ring_all_reduce_algorithm_t::is_ring_class() const { return true; }
// Pipelined ring: priced by the closed-form throughput model.
bool ring_all_reduce_algorithm_t::is_ring_pipeline() const { return true; }

// ring_broadcast: broadcast as an N-1 hop ring pipeline, each step forwarding the root's
// data to the next-rank neighbour.
//
// Store the communicator size.
ring_broadcast_algorithm_t::ring_broadcast_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

// Each step loads locally, stores, and pushes the data to the next rank.
schedule_entry_t ring_broadcast_algorithm_t::link_of(int /*pid*/,
                                                     int /*timestep*/,
                                                     int my_rank) const {
  const int next_rank    = floor_mod(my_rank + 1, num_gpus_);
  std::vector<op_t> work = {
      load_t{},
      store_t{},
      push_t{next_rank},
  };
  return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
}

// Floored ring share (see ring_wgs_per_link).
int ring_broadcast_algorithm_t::wgs_on_link(int /*timestep*/, int num_wgs) const {
  return ring_wgs_per_link(num_wgs, num_gpus_);
}

// Workgroups distributed across the ring links (see ring_distribute).
std::unordered_map<int, int> ring_broadcast_algorithm_t::active_links(int /*timestep*/,
                                                                      int num_wgs) const {
  return ring_distribute(num_wgs, num_gpus_);
}

// N-1 ring hops.
int ring_broadcast_algorithm_t::num_timesteps() const { return num_gpus_ - 1; }
// Each hop moves 1/N of the buffer.
int ring_broadcast_algorithm_t::chunks_per_timestep() const { return num_gpus_; }

}  // namespace origami::comm
