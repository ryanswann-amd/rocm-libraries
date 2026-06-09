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

#include "origami/comm/algorithms.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
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

// Store the communicator size and the per-hop work-graph closure (default: pull+store).
all_to_same_algorithm_t::all_to_same_algorithm_t(int num_gpus, work_graph_fn_t wg_fn)
    : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

// Target the next remote peer (my_rank + timestep + 1) and pull from it; load locally on a
// self-step.
schedule_entry_t all_to_same_algorithm_t::link_of(int /*pid*/, int timestep, int my_rank) const {
  const int peer     = floor_mod(my_rank + timestep + 1, num_gpus_);
  const bool is_self = (peer == my_rank);
  auto work          = wg_fn_(peer, my_rank, num_gpus_, is_self);
  return {is_self ? SELF_LINK : peer, peer, direction_t::PULL, std::move(work), is_self};
}

// One shared link per step, so every workgroup rides it.
int all_to_same_algorithm_t::wgs_on_link(int /*timestep*/, int num_wgs) const { return num_wgs; }

// Exactly one link is active this step, carrying all the workgroups.
std::unordered_map<int, int> all_to_same_algorithm_t::active_links(int timestep,
                                                                   int num_wgs) const {
  return {{timestep % (num_gpus_ - 1), num_wgs}};
}

// One round per remote peer.
int all_to_same_algorithm_t::num_timesteps() const { return num_gpus_ - 1; }

// Default hop: pull then store; load+store when the data is already local.
std::vector<op_t> all_to_same_algorithm_t::default_work_graph(int peer,
                                                              int /*my_rank*/,
                                                              int /*num_gpus*/,
                                                              bool is_self) {
  if (is_self) return {load_t{}, store_t{}};
  return {pull_t{peer}, store_t{}};
}

// Store the communicator size and the per-hop work-graph closure (default: pull+reduce).
pid_staggered_algorithm_t::pid_staggered_algorithm_t(int num_gpus, work_graph_fn_t wg_fn)
    : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

// Offset the starting peer by pid, walk peers in order, and pull each.
schedule_entry_t pid_staggered_algorithm_t::link_of(int pid, int timestep, int my_rank) const {
  const int start    = floor_mod(pid, num_gpus_);
  const int peer_idx = floor_mod(start + timestep, num_gpus_);
  const int peer     = floor_mod(my_rank + peer_idx, num_gpus_);
  const bool is_self = (peer == my_rank);
  auto work          = wg_fn_(peer, my_rank, num_gpus_, is_self);
  return {is_self ? SELF_LINK : peer, peer, direction_t::PULL, std::move(work), is_self};
}

// Spread the remote workgroups evenly over the N-1 links.
int pid_staggered_algorithm_t::wgs_on_link(int /*timestep*/, int num_wgs) const {
  const int num_links  = num_gpus_ - 1;
  const int remote_wgs = num_wgs * (num_gpus_ - 1) / num_gpus_;
  return std::max(remote_wgs / std::max(num_links, 1), 1);
}

// All N-1 links active each step, remote workgroups split evenly across them.
std::unordered_map<int, int> pid_staggered_algorithm_t::active_links(int /*timestep*/,
                                                                     int num_wgs) const {
  const int num_links  = num_gpus_ - 1;
  const int remote_wgs = num_wgs * (num_gpus_ - 1) / num_gpus_;
  const int per_link   = std::max(remote_wgs / std::max(num_links, 1), 1);
  std::unordered_map<int, int> out;
  for (int i = 0; i < num_links; ++i) out[i] = per_link;
  return out;
}

// N rounds, counting the self-step.
int pid_staggered_algorithm_t::num_timesteps() const { return num_gpus_; }
// Each step moves 1/N of the buffer.
int pid_staggered_algorithm_t::chunks_per_timestep() const { return num_gpus_; }

// Default hop: pull then reduce; load+reduce when local.
std::vector<op_t> pid_staggered_algorithm_t::default_work_graph(int peer,
                                                                int /*my_rank*/,
                                                                int /*num_gpus*/,
                                                                bool is_self) {
  if (is_self) return {load_t{}, reduce_t{}};
  return {pull_t{peer}, reduce_t{}};
}

// Store the communicator size and the per-hop work-graph closure (default: load+push).
pid_partitioned_algorithm_t::pid_partitioned_algorithm_t(int num_gpus, work_graph_fn_t wg_fn)
    : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

// Permanently bind the workgroup to the destination chosen by its pid and push there.
schedule_entry_t pid_partitioned_algorithm_t::link_of(int pid,
                                                      int /*timestep*/,
                                                      int my_rank) const {
  const int dest     = floor_mod(pid, num_gpus_);
  const int peer     = floor_mod(my_rank + dest, num_gpus_);
  const bool is_self = (peer == my_rank);
  auto work          = wg_fn_(peer, my_rank, num_gpus_, is_self);
  return {is_self ? SELF_LINK : peer, peer, direction_t::PUSH, std::move(work), is_self};
}

// Partition the workgroups evenly, one share per destination.
int pid_partitioned_algorithm_t::wgs_on_link(int /*timestep*/, int num_wgs) const {
  return std::max(num_wgs / num_gpus_, 1);
}

// All N-1 remote links active, workgroups partitioned evenly across them.
std::unordered_map<int, int> pid_partitioned_algorithm_t::active_links(int /*timestep*/,
                                                                       int num_wgs) const {
  const int per_link = std::max(num_wgs / num_gpus_, 1);
  std::unordered_map<int, int> out;
  for (int i = 0; i < num_gpus_ - 1; ++i) out[i] = per_link;
  return out;
}

// A single timestep: every destination is served at once.
int pid_partitioned_algorithm_t::num_timesteps() const { return 1; }

// Default hop: load then push; load+store when local.
std::vector<op_t> pid_partitioned_algorithm_t::default_work_graph(int peer,
                                                                  int /*my_rank*/,
                                                                  int /*num_gpus*/,
                                                                  bool is_self) {
  if (is_self) return {load_t{}, store_t{}};
  return {load_t{}, push_t{peer}};
}

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

// Store the communicator size.
two_shot_all_reduce_algorithm_t::two_shot_all_reduce_algorithm_t(int num_gpus)
    : num_gpus_{num_gpus} {}

// Reduce phase (step < N) pulls and sums each peer's slice; broadcast phase pushes
// the finished slice out, skipping self in the peer ordering.
schedule_entry_t two_shot_all_reduce_algorithm_t::link_of(int pid,
                                                          int timestep,
                                                          int my_rank) const {
  const int N     = num_gpus_;
  const int start = floor_mod(pid, N);
  if (is_reduce_phase_(timestep)) {
    const int peer_idx     = floor_mod(start + timestep, N);
    const int peer         = floor_mod(my_rank + peer_idx, N);
    const bool is_self     = (peer == my_rank);
    std::vector<op_t> work = is_self ? std::vector<op_t>{load_t{}, reduce_t{}}
                                     : std::vector<op_t>{pull_t{peer}, reduce_t{}};
    return {is_self ? SELF_LINK : peer, peer, direction_t::PULL, std::move(work), is_self};
  }
  // Broadcast phase. Skip self in peer ordering.
  const int bcast_idx    = timestep - N;
  const int peer_offset  = floor_mod(start + bcast_idx, N - 1) + 1;
  const int peer         = floor_mod(my_rank + peer_offset, N);
  std::vector<op_t> work = {
      load_t{},
      push_t{peer},
  };
  return {peer, peer, direction_t::PUSH, std::move(work), false};
}

// Remote workgroups spread over the N-1 links (fewer during the reduce phase's self-step).
int two_shot_all_reduce_algorithm_t::wgs_on_link(int timestep, int num_wgs) const {
  const int num_links = num_gpus_ - 1;
  const int remote_wgs =
      is_reduce_phase_(timestep) ? num_wgs * (num_gpus_ - 1) / num_gpus_ : num_wgs;
  return std::max(remote_wgs / std::max(num_links, 1), 1);
}

// All N-1 links active each step, workgroups spread evenly across them.
std::unordered_map<int, int> two_shot_all_reduce_algorithm_t::active_links(int timestep,
                                                                           int num_wgs) const {
  const int num_links = num_gpus_ - 1;
  const int remote_wgs =
      is_reduce_phase_(timestep) ? num_wgs * (num_gpus_ - 1) / num_gpus_ : num_wgs;
  const int per_link = std::max(remote_wgs / std::max(num_links, 1), 1);
  std::unordered_map<int, int> out;
  for (int i = 0; i < num_links; ++i) out[i] = per_link;
  return out;
}

// 2N-1 rounds: N reduce steps (incl. self) then N-1 broadcast steps.
int two_shot_all_reduce_algorithm_t::num_timesteps() const { return 2 * num_gpus_ - 1; }
// Each step moves 1/N of the buffer.
int two_shot_all_reduce_algorithm_t::chunks_per_timestep() const { return num_gpus_; }

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

// All-gather: ring.
std::unique_ptr<collective_algorithm_t> allgather_algorithm(int num_gpus) {
  return std::make_unique<ring_all_gather_algorithm_t>(num_gpus);
}

// Reduce-scatter: ring.
std::unique_ptr<collective_algorithm_t> reduce_scatter_algorithm(int num_gpus) {
  return std::make_unique<ring_reduce_scatter_algorithm_t>(num_gpus);
}

// Broadcast: ring.
std::unique_ptr<collective_algorithm_t> broadcast_algorithm(int num_gpus) {
  return std::make_unique<ring_broadcast_algorithm_t>(num_gpus);
}

// One-shot all-reduce: all-to-same schedule with a pull+reduce hop.
std::unique_ptr<collective_algorithm_t> allreduce_one_shot_algorithm(int num_gpus) {
  auto wg = [](int peer, int /*my_rank*/, int /*N*/, bool is_self) -> std::vector<op_t> {
    if (is_self) return {load_t{}, reduce_t{}};
    return {pull_t{peer}, reduce_t{}};
  };
  return std::make_unique<all_to_same_algorithm_t>(num_gpus, wg);
}

// Two-shot all-reduce: reduce-scatter shot then all-gather shot.
std::unique_ptr<collective_algorithm_t> allreduce_two_shot_algorithm(int num_gpus) {
  return std::make_unique<two_shot_all_reduce_algorithm_t>(num_gpus);
}

// Ring all-reduce: bandwidth-optimal pipelined ring.
std::unique_ptr<collective_algorithm_t> allreduce_ring_algorithm(int num_gpus) {
  return std::make_unique<ring_all_reduce_algorithm_t>(num_gpus);
}

// All-to-all: pid-staggered schedule with a load+push hop.
std::unique_ptr<collective_algorithm_t> alltoall_algorithm(int num_gpus) {
  auto wg = [](int peer, int /*my_rank*/, int /*N*/, bool is_self) -> std::vector<op_t> {
    if (is_self) return {load_t{}, store_t{}};
    return {load_t{}, push_t{peer}};
  };
  return std::make_unique<pid_staggered_algorithm_t>(num_gpus, wg);
}

// Map a (collective, algorithm) pair to a concrete algorithm; `automatic` picks each
// collective's canonical default, and any undefined pairing throws rather than costing nonsense.
std::unique_ptr<collective_algorithm_t> resolve_algorithm(primitive_t collective,
                                                          algorithm_t algorithm,
                                                          int num_gpus) {
  const bool automatic = (algorithm == algorithm_t::automatic);
  switch (collective) {
    case primitive_t::all_gather:
      if (automatic || algorithm == algorithm_t::ring) return allgather_algorithm(num_gpus);
      break;
    case primitive_t::reduce_scatter:
      if (automatic || algorithm == algorithm_t::ring) return reduce_scatter_algorithm(num_gpus);
      break;
    case primitive_t::broadcast:
      if (automatic || algorithm == algorithm_t::ring) return broadcast_algorithm(num_gpus);
      break;
    case primitive_t::all_reduce:
      switch (algorithm) {
        case algorithm_t::automatic:
        case algorithm_t::two_shot: return allreduce_two_shot_algorithm(num_gpus);
        case algorithm_t::one_shot: return allreduce_one_shot_algorithm(num_gpus);
        case algorithm_t::ring: return allreduce_ring_algorithm(num_gpus);
        default: break;
      }
      break;
    case primitive_t::all_to_all:
      if (automatic || algorithm == algorithm_t::direct) return alltoall_algorithm(num_gpus);
      break;
  }
  throw std::invalid_argument(std::string{"algorithm '"} + std::string{algorithm_name(algorithm)} +
                              "' is not a valid implementation of collective '" +
                              std::string{primitive_name(collective)} + "'");
}

// Engine-facing overload: collective + num_gpus come from the problem, algorithm from the config.
std::unique_ptr<collective_algorithm_t> resolve_algorithm(const comm_problem_t& problem,
                                                          const comm_config_t& config) {
  return resolve_algorithm(problem.collective, config.algorithm, problem.num_gpus);
}

}  // namespace origami::comm
