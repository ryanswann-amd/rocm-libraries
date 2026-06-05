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
// Collective algorithms: each algorithm emits the *schedule* of a collective,
// expressed as a pure function (pid, timestep) → which link a workgroup uses
// and what primitives it runs there. This is the bridge between an algorithm
// and the cost model: the model never hard-codes "all-gather costs X"; it asks
// the algorithm for the per-step work graph and lets latency.hpp /
// collective.hpp price it.
//
// Three quantities an algorithm exposes drive the whole cost, and each is a
// direct consequence of the algorithm's dataflow:
//   • num_timesteps()       — how many dependent communication rounds the
//     algorithm takes (e.g. a ring visits N-1 peers; two-shot does N reduce
//     rounds then N-1 broadcast rounds = 2N-1). More steps ⇒ more serial
//     handshakes and, for sequential algorithms, more added latency.
//   • chunks_per_timestep() — how finely each GPU's tile is sliced per step.
//     A ring sends 1/N of the buffer per hop (chunks = N); a whole-tile step
//     sends all of it (chunks = 1). This sets the per-step wire bytes.
//   • active_links()        — how the workgroups spread across the links lit
//     up this step, which sets per-link contention.
//
// Algorithms are closed-form functions over the rank topology, so they hold no
// per-WG state and fold cheaply; virtual dispatch happens once per collective.
//
// Self-timestep distinction: when a step maps a rank to itself, the data is
// already local, so the work graph uses load_t (local HBM) instead of pull_t
// (xGMI). Self-steps consume no fabric bandwidth and are not MSHR-limited —
// the model must not bill them as remote transfers.
#pragma once

#include "origami/comm/primitives.hpp"
#include "origami/comm/types.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace origami::comm {

inline constexpr int SELF_LINK = -1;  // sentinel: timestep is local

// ─── schedule_entry_t ───────────────────────────────────────────────
struct schedule_entry_t {
  int link_id;
  int peer_rank;
  direction_t direction;
  std::vector<op_t> work_graph;
  bool is_self = false;
};

// Work-graph closure: (peer, my_rank, num_gpus, is_self) -> [op_t].
using work_graph_fn_t =
    std::function<std::vector<op_t>(int peer, int my_rank, int num_gpus, bool is_self)>;

// Floored modulo: always returns a value in [0, n). Ring schedules index peers
// as my_rank ± offset, which can go negative or past N; this wraps those into
// a valid rank so a single closed form expresses "the neighbour k hops away".
constexpr int floor_mod(int a, int n) noexcept {
  const int r = a % n;
  return (r < 0) ? r + n : r;
}

// ─── Base ────────────────────────────────────────────────────────
class collective_algorithm_t {
 public:
  virtual ~collective_algorithm_t() = default;

  virtual schedule_entry_t link_of(int pid, int timestep, int my_rank, int num_gpus) const = 0;
  virtual int wgs_on_link(int timestep, int num_wgs, int num_gpus) const                   = 0;
  virtual std::unordered_map<int, int> active_links(int timestep,
                                                    int num_wgs,
                                                    int num_gpus) const                    = 0;
  virtual int num_timesteps() const                                                        = 0;

  // Default: each timestep moves the whole gpu_tile (= 1). Chunked
  // algorithms override (ring, two-shot, a2a → N).
  virtual int chunks_per_timestep() const { return 1; }

  // Algorithm-kind queries — the two checks the collective engine makes
  // to identify ring-class algorithms.
  //
  //   is_ring_class()    — eligible for per-step proxy/sync overhead
  //                        heuristic. Covers AG, RS, ring AR, ring fixed.
  //   is_ring_pipeline() — uses the closed-form pipelined-ring
  //                        throughput model rather than per-timestep
  //                        wg_tile sums. Covers ring AR + ring fixed
  //                        (the algorithms whose work graph carries
  //                        signal_t/wait_t ops).
  virtual bool is_ring_class() const { return false; }
  virtual bool is_ring_pipeline() const { return false; }
};

// ─── all_to_same_algorithm_t ────────────────────────────────────────────
// All WGs → same link each timestep. Used by one_shot AR + A2A
// (sequential). Skips self; visits N-1 remote peers.
class all_to_same_algorithm_t : public collective_algorithm_t {
 public:
  explicit all_to_same_algorithm_t(int num_gpus, work_graph_fn_t wg_fn = {})
      : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

  schedule_entry_t link_of(int /*pid*/, int timestep, int my_rank, int num_gpus) const override {
    const int peer     = floor_mod(my_rank + timestep + 1, num_gpus);
    const bool is_self = (peer == my_rank);
    auto work          = wg_fn_(peer, my_rank, num_gpus, is_self);
    return {is_self ? SELF_LINK : peer, peer, direction_t::PULL, std::move(work), is_self};
  }

  int wgs_on_link(int /*timestep*/, int num_wgs, int /*num_gpus*/) const override {
    return num_wgs;
  }

  std::unordered_map<int, int> active_links(int timestep,
                                            int num_wgs,
                                            int num_gpus) const override {
    return {{timestep % (num_gpus - 1), num_wgs}};
  }

  int num_timesteps() const override { return num_gpus_ - 1; }

 private:
  static std::vector<op_t> default_work_graph(int peer,
                                              int /*my_rank*/,
                                              int /*num_gpus*/,
                                              bool is_self) {
    if (is_self) return {load_t{}, store_t{}};
    return {pull_t{peer}, store_t{}};
  }

  int num_gpus_;
  work_graph_fn_t wg_fn_;
};

// ─── pid_staggered_algorithm_t ─────────────────────────────────────────
// pid % world_size offsets starting peer; WGs spread uniformly. Used
// by two-shot AR/RS and a2a. Includes a self-timestep.
class pid_staggered_algorithm_t : public collective_algorithm_t {
 public:
  explicit pid_staggered_algorithm_t(int num_gpus, work_graph_fn_t wg_fn = {})
      : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

  schedule_entry_t link_of(int pid, int timestep, int my_rank, int num_gpus) const override {
    const int start    = floor_mod(pid, num_gpus);
    const int peer_idx = floor_mod(start + timestep, num_gpus);
    const int peer     = floor_mod(my_rank + peer_idx, num_gpus);
    const bool is_self = (peer == my_rank);
    auto work          = wg_fn_(peer, my_rank, num_gpus, is_self);
    return {is_self ? SELF_LINK : peer, peer, direction_t::PULL, std::move(work), is_self};
  }

  int wgs_on_link(int /*timestep*/, int num_wgs, int num_gpus) const override {
    const int num_links  = num_gpus - 1;
    const int remote_wgs = num_wgs * (num_gpus - 1) / num_gpus;
    return std::max(remote_wgs / std::max(num_links, 1), 1);
  }

  std::unordered_map<int, int> active_links(int /*timestep*/,
                                            int num_wgs,
                                            int num_gpus) const override {
    const int num_links  = num_gpus - 1;
    const int remote_wgs = num_wgs * (num_gpus - 1) / num_gpus;
    const int per_link   = std::max(remote_wgs / std::max(num_links, 1), 1);
    std::unordered_map<int, int> out;
    for (int i = 0; i < num_links; ++i) out[i] = per_link;
    return out;
  }

  int num_timesteps() const override { return num_gpus_; }
  int chunks_per_timestep() const override { return num_gpus_; }

 private:
  static std::vector<op_t> default_work_graph(int peer,
                                              int /*my_rank*/,
                                              int /*num_gpus*/,
                                              bool is_self) {
    if (is_self) return {load_t{}, reduce_t{}};
    return {pull_t{peer}, reduce_t{}};
  }

  int num_gpus_;
  work_graph_fn_t wg_fn_;
};

// ─── pid_partitioned_algorithm_t ───────────────────────────────────────
// Each WG permanently assigned to one link (partitioned AG).
class pid_partitioned_algorithm_t : public collective_algorithm_t {
 public:
  explicit pid_partitioned_algorithm_t(int num_gpus, work_graph_fn_t wg_fn = {})
      : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

  schedule_entry_t link_of(int pid, int /*timestep*/, int my_rank, int num_gpus) const override {
    const int dest     = floor_mod(pid, num_gpus);
    const int peer     = floor_mod(my_rank + dest, num_gpus);
    const bool is_self = (peer == my_rank);
    auto work          = wg_fn_(peer, my_rank, num_gpus, is_self);
    return {is_self ? SELF_LINK : peer, peer, direction_t::PUSH, std::move(work), is_self};
  }

  int wgs_on_link(int /*timestep*/, int num_wgs, int num_gpus) const override {
    return std::max(num_wgs / num_gpus, 1);
  }

  std::unordered_map<int, int> active_links(int /*timestep*/,
                                            int num_wgs,
                                            int num_gpus) const override {
    const int per_link = std::max(num_wgs / num_gpus, 1);
    std::unordered_map<int, int> out;
    for (int i = 0; i < num_gpus - 1; ++i) out[i] = per_link;
    return out;
  }

  int num_timesteps() const override { return 1; }

 private:
  static std::vector<op_t> default_work_graph(int peer,
                                              int /*my_rank*/,
                                              int /*num_gpus*/,
                                              bool is_self) {
    if (is_self) return {load_t{}, store_t{}};
    return {load_t{}, push_t{peer}};
  }

  int num_gpus_;
  work_graph_fn_t wg_fn_;
};

// ─── Ring distribution helper ───────────────────────────────────
// Spread num_wgs workgroups over the available rings. The defining constraint
// is conservation: the WG counts returned must sum to exactly num_wgs — every
// launched workgroup is doing real work on some link, none invented, none
// dropped. (An earlier even-division form violated this, dropping WGs at high
// channel counts and fabricating them at low counts, mispricing contention.)
//
// There can be at most N-1 distinct ring links, so we use
// nrings = min(num_wgs, N-1) and hand each ring floor(num_wgs/nrings), giving
// the first `extra` rings one more so the remainder is absorbed and the total
// is preserved.
inline std::unordered_map<int, int> ring_distribute(int num_wgs, int num_gpus) {
  const int nrings = std::max(std::min(num_wgs, num_gpus - 1), 1);
  const int base   = num_wgs / nrings;
  const int extra  = num_wgs - base * nrings;
  std::unordered_map<int, int> out;
  for (int i = 0; i < nrings; ++i) { out[i] = base + (i < extra ? 1 : 0); }
  return out;
}

inline int ring_wgs_per_link(int num_wgs, int num_gpus) noexcept {
  const int nrings = std::max(std::min(num_wgs, num_gpus - 1), 1);
  return std::max(num_wgs / nrings, 1);
}

// ─── ring_fixed_algorithm_t ────────────────────────────────────────────
// All hops to next_rank. Used by ring AR (older form).
class ring_fixed_algorithm_t : public collective_algorithm_t {
 public:
  explicit ring_fixed_algorithm_t(int num_gpus, work_graph_fn_t wg_fn = {})
      : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

  schedule_entry_t link_of(int /*pid*/,
                           int /*timestep*/,
                           int my_rank,
                           int num_gpus) const override {
    const int next_rank = floor_mod(my_rank + 1, num_gpus);
    auto work           = wg_fn_(next_rank, my_rank, num_gpus, false);
    return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
  }

  int wgs_on_link(int /*timestep*/, int num_wgs, int num_gpus) const override {
    return ring_wgs_per_link(num_wgs, num_gpus);
  }

  std::unordered_map<int, int> active_links(int /*timestep*/,
                                            int num_wgs,
                                            int num_gpus) const override {
    return ring_distribute(num_wgs, num_gpus);
  }

  int num_timesteps() const override { return num_gpus_ - 1; }
  int chunks_per_timestep() const override { return num_gpus_; }
  bool is_ring_class() const override { return true; }
  bool is_ring_pipeline() const override { return true; }

 private:
  static std::vector<op_t> default_work_graph(int /*peer*/,
                                              int my_rank,
                                              int num_gpus,
                                              bool /*is_self*/) {
    const int next_rank = floor_mod(my_rank + 1, num_gpus);
    const int prev_rank = floor_mod(my_rank - 1, num_gpus);
    return {
        load_t{}, wait_t{prev_rank}, pull_t{prev_rank}, reduce_t{}, store_t{}, signal_t{next_rank}};
  }

  int num_gpus_;
  work_graph_fn_t wg_fn_;
};

// ─── ring_all_gather_algorithm_t ────────────────────────────────────────
// N-1 step ring. Each step: load_t (local) + store_t (local) + push_t (fwd).
class ring_all_gather_algorithm_t : public collective_algorithm_t {
 public:
  explicit ring_all_gather_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

  schedule_entry_t link_of(int /*pid*/,
                           int /*timestep*/,
                           int my_rank,
                           int num_gpus) const override {
    const int next_rank    = floor_mod(my_rank + 1, num_gpus);
    std::vector<op_t> work = {load_t{}, store_t{}, push_t{next_rank}};
    return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
  }

  int wgs_on_link(int /*timestep*/, int num_wgs, int num_gpus) const override {
    return ring_wgs_per_link(num_wgs, num_gpus);
  }

  std::unordered_map<int, int> active_links(int /*timestep*/,
                                            int num_wgs,
                                            int num_gpus) const override {
    return ring_distribute(num_wgs, num_gpus);
  }

  int num_timesteps() const override { return num_gpus_ - 1; }
  int chunks_per_timestep() const override { return 1; }  // AG convention: msg = per-rank send
  bool is_ring_class() const override { return true; }
  // Sequential-style throughput model (no pipelined ring).

 private:
  int num_gpus_;
};

// ─── ring_reduce_scatter_algorithm_t ────────────────────────────────────
// Structurally identical to AG ring + reduce_t.
class ring_reduce_scatter_algorithm_t : public collective_algorithm_t {
 public:
  explicit ring_reduce_scatter_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

  schedule_entry_t link_of(int /*pid*/,
                           int /*timestep*/,
                           int my_rank,
                           int num_gpus) const override {
    const int next_rank    = floor_mod(my_rank + 1, num_gpus);
    std::vector<op_t> work = {load_t{}, reduce_t{}, store_t{}, push_t{next_rank}};
    return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
  }

  int wgs_on_link(int /*timestep*/, int num_wgs, int num_gpus) const override {
    return ring_wgs_per_link(num_wgs, num_gpus);
  }

  std::unordered_map<int, int> active_links(int /*timestep*/,
                                            int num_wgs,
                                            int num_gpus) const override {
    return ring_distribute(num_wgs, num_gpus);
  }

  int num_timesteps() const override { return num_gpus_ - 1; }
  int chunks_per_timestep() const override { return 1; }
  bool is_ring_class() const override { return true; }

 private:
  int num_gpus_;
};

// ─── two_shot_all_reduce_algorithm_t ─────────────────────────────────────
// All-reduce factored as reduce-scatter then all-gather ("two shots"): each
// rank first pulls and sums every peer's slice (N reduce steps, including its
// own self-step), then pushes the finished slice out to all others (N-1
// broadcast steps). Hence num_timesteps = 2N-1 and each step moves 1/N of the
// buffer (chunks_per_timestep = N). is_reduce_phase_ just splits the timeline
// at step N into the two shots.
class two_shot_all_reduce_algorithm_t : public collective_algorithm_t {
 public:
  explicit two_shot_all_reduce_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

  schedule_entry_t link_of(int pid, int timestep, int my_rank, int num_gpus) const override {
    const int N     = num_gpus;
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
    std::vector<op_t> work = {load_t{}, push_t{peer}};
    return {peer, peer, direction_t::PUSH, std::move(work), false};
  }

  int wgs_on_link(int timestep, int num_wgs, int num_gpus) const override {
    const int num_links = num_gpus - 1;
    const int remote_wgs =
        is_reduce_phase_(timestep) ? num_wgs * (num_gpus - 1) / num_gpus : num_wgs;
    return std::max(remote_wgs / std::max(num_links, 1), 1);
  }

  std::unordered_map<int, int> active_links(int timestep,
                                            int num_wgs,
                                            int num_gpus) const override {
    const int num_links = num_gpus - 1;
    const int remote_wgs =
        is_reduce_phase_(timestep) ? num_wgs * (num_gpus - 1) / num_gpus : num_wgs;
    const int per_link = std::max(remote_wgs / std::max(num_links, 1), 1);
    std::unordered_map<int, int> out;
    for (int i = 0; i < num_links; ++i) out[i] = per_link;
    return out;
  }

  int num_timesteps() const override { return 2 * num_gpus_ - 1; }
  int chunks_per_timestep() const override { return num_gpus_; }

 private:
  constexpr bool is_reduce_phase_(int timestep) const noexcept { return timestep < num_gpus_; }
  int num_gpus_;
};

// ─── ring_all_reduce_algorithm_t ────────────────────────────────────────
// The bandwidth-optimal all-reduce: a reduce-scatter ring (N-1 steps, each
// pulls from prev, sums, signals next) followed by an all-gather ring (N-1
// steps, each pulls the finished slice and forwards it). 2(N-1) steps total,
// every step crossing the same neighbour link — which is why it is a true
// pipelined ring (is_ring_pipeline) priced by aggregate throughput, not a sum
// of per-step latencies. The wait_t/signal_t in the work graph are the
// producer→consumer dependency that serializes adjacent ranks within a step.
class ring_all_reduce_algorithm_t : public collective_algorithm_t {
 public:
  explicit ring_all_reduce_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

  schedule_entry_t link_of(int /*pid*/, int timestep, int my_rank, int num_gpus) const override {
    const int next_rank = floor_mod(my_rank + 1, num_gpus);
    const int prev_rank = floor_mod(my_rank - 1, num_gpus);
    const int rs_visits = num_gpus - 1;

    std::vector<op_t> work;
    if (timestep < rs_visits) {
      work = {load_t{},
              wait_t{prev_rank},
              pull_t{prev_rank},
              reduce_t{},
              store_t{},
              signal_t{next_rank}};
    } else {
      work = {wait_t{prev_rank}, pull_t{prev_rank}, store_t{}, signal_t{next_rank}};
    }
    return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
  }

  int wgs_on_link(int /*timestep*/, int num_wgs, int num_gpus) const override {
    return ring_wgs_per_link(num_wgs, num_gpus);
  }

  std::unordered_map<int, int> active_links(int /*timestep*/,
                                            int num_wgs,
                                            int num_gpus) const override {
    return ring_distribute(num_wgs, num_gpus);
  }

  int num_timesteps() const override { return 2 * (num_gpus_ - 1); }
  int chunks_per_timestep() const override { return num_gpus_; }
  bool is_ring_class() const override { return true; }
  bool is_ring_pipeline() const override { return true; }

 private:
  int num_gpus_;
};

// ─── ring_broadcast_algorithm_t ────────────────────────────────────────
// N-1 hop pipeline on 1 link. Each GPU: load_t + store_t + push_t.
class ring_broadcast_algorithm_t : public collective_algorithm_t {
 public:
  explicit ring_broadcast_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

  schedule_entry_t link_of(int /*pid*/,
                           int /*timestep*/,
                           int my_rank,
                           int num_gpus) const override {
    const int next_rank    = floor_mod(my_rank + 1, num_gpus);
    std::vector<op_t> work = {load_t{}, store_t{}, push_t{next_rank}};
    return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
  }

  int wgs_on_link(int /*timestep*/, int num_wgs, int num_gpus) const override {
    return ring_wgs_per_link(num_wgs, num_gpus);
  }

  std::unordered_map<int, int> active_links(int /*timestep*/,
                                            int num_wgs,
                                            int num_gpus) const override {
    return ring_distribute(num_wgs, num_gpus);
  }

  int num_timesteps() const override { return num_gpus_ - 1; }
  int chunks_per_timestep() const override { return num_gpus_; }

 private:
  int num_gpus_;
};

// ─── Standard collective constructors ───────────────────────────
// Free-function algorithm builders.

inline std::unique_ptr<collective_algorithm_t> allgather_algorithm(int num_gpus) {
  return std::make_unique<ring_all_gather_algorithm_t>(num_gpus);
}

inline std::unique_ptr<collective_algorithm_t> reduce_scatter_algorithm(int num_gpus) {
  return std::make_unique<ring_reduce_scatter_algorithm_t>(num_gpus);
}

inline std::unique_ptr<collective_algorithm_t> broadcast_algorithm(int num_gpus) {
  return std::make_unique<ring_broadcast_algorithm_t>(num_gpus);
}

inline std::unique_ptr<collective_algorithm_t> allreduce_one_shot_algorithm(int num_gpus) {
  auto wg = [](int peer, int /*my_rank*/, int /*N*/, bool is_self) -> std::vector<op_t> {
    if (is_self) return {load_t{}, reduce_t{}};
    return {pull_t{peer}, reduce_t{}};
  };
  return std::make_unique<all_to_same_algorithm_t>(num_gpus, wg);
}

inline std::unique_ptr<collective_algorithm_t> allreduce_two_shot_algorithm(int num_gpus) {
  return std::make_unique<two_shot_all_reduce_algorithm_t>(num_gpus);
}

inline std::unique_ptr<collective_algorithm_t> allreduce_ring_algorithm(int num_gpus) {
  return std::make_unique<ring_all_reduce_algorithm_t>(num_gpus);
}

inline std::unique_ptr<collective_algorithm_t> alltoall_algorithm(int num_gpus) {
  auto wg = [](int peer, int /*my_rank*/, int /*N*/, bool is_self) -> std::vector<op_t> {
    if (is_self) return {load_t{}, store_t{}};
    return {load_t{}, push_t{peer}};
  };
  return std::make_unique<pid_staggered_algorithm_t>(num_gpus, wg);
}

// ─── resolve_algorithm ───────────────────────────────────────────
// The (collective, algorithm) → implementation factory, and the single place
// the validity rule lives: an algorithm is a valid config only if it is
// defined for the problem's collective. `automatic` maps to each collective's
// canonical algorithm (preserving the historical defaults); a named value
// selects a specific one and any undefined (collective, algorithm) pair throws
// rather than silently costing a nonsense schedule.
//
//   all_gather / reduce_scatter / broadcast : ring only
//   all_reduce                              : two_shot (default), one_shot, ring
//   all_to_all                              : direct (pid-staggered) only
inline std::unique_ptr<collective_algorithm_t> resolve_algorithm(primitive_t collective,
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

}  // namespace origami::comm
