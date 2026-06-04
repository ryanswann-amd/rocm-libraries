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
// Collective layouts: (pid, timestep) → link.
// Mirrors origami_comms/model/layouts.py 1:1.
//
// Each layout is a closed-form function derived from actual Iris/RCCL
// kernel loop structures. The interface is polymorphic (virtual) — same
// shape as the Python base class. Layout objects are constructed once
// per collective call, so virtual dispatch overhead is negligible.
//
// Self-timestep distinction: when a timestep maps to my_rank the work
// graph uses load_t (local HBM) instead of pull_t (xGMI). Self-timesteps
// don't consume xGMI bandwidth or hit MSHR limits.
#pragma once

#include "origami/comm/primitives.hpp"
#include "origami/comm/types.hpp"

#include <algorithm>
#include <functional>
#include <memory>
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

// Work-graph closure. Mirrors Python `work_graph_fn(peer, my_rank, num_gpus, is_self) -> [op_t]`.
using work_graph_fn_t =
    std::function<std::vector<op_t>(int peer, int my_rank, int num_gpus, bool is_self)>;

// Python-style modulo: always returns a value in [0, n).
constexpr int py_mod(int a, int n) noexcept {
  const int r = a % n;
  return (r < 0) ? r + n : r;
}

// ─── Base ────────────────────────────────────────────────────────
class collective_layout_t {
 public:
  virtual ~collective_layout_t() = default;

  virtual schedule_entry_t link_of(int pid, int timestep, int my_rank, int num_gpus) const = 0;
  virtual int wgs_on_link(int timestep, int num_wgs, int num_gpus) const                   = 0;
  virtual std::unordered_map<int, int> active_links(int timestep,
                                                    int num_wgs,
                                                    int num_gpus) const                    = 0;
  virtual int num_timesteps() const                                                        = 0;

  // Default: each timestep moves the whole gpu_tile (= 1). Chunked
  // algorithms override (ring, two-shot, a2a → N).
  virtual int chunks_per_timestep() const { return 1; }

  // Layout-kind queries — match the two checks the collective engine
  // makes against Python's RING_LAYOUT_CLASSES / _is_ring_layout.
  //
  //   is_ring_class()    — eligible for per-step proxy/sync overhead
  //                        heuristic. Covers AG, RS, ring AR, ring fixed.
  //   is_ring_pipeline() — uses the closed-form pipelined-ring
  //                        throughput model rather than per-timestep
  //                        wg_tile sums. Covers ring AR + ring fixed
  //                        (the layouts whose work graph carries
  //                        signal_t/wait_t ops).
  virtual bool is_ring_class() const { return false; }
  virtual bool is_ring_pipeline() const { return false; }
};

// ─── all_to_same_layout_t ────────────────────────────────────────────
// All WGs → same link each timestep. Used by one_shot AR + A2A
// (sequential). Skips self; visits N-1 remote peers.
class all_to_same_layout_t : public collective_layout_t {
 public:
  explicit all_to_same_layout_t(int num_gpus, work_graph_fn_t wg_fn = {})
      : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

  schedule_entry_t link_of(int /*pid*/, int timestep, int my_rank, int num_gpus) const override {
    const int peer     = py_mod(my_rank + timestep + 1, num_gpus);
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

// ─── pid_staggered_layout_t ─────────────────────────────────────────
// pid % world_size offsets starting peer; WGs spread uniformly. Used
// by two-shot AR/RS and a2a. Includes a self-timestep.
class pid_staggered_layout_t : public collective_layout_t {
 public:
  explicit pid_staggered_layout_t(int num_gpus, work_graph_fn_t wg_fn = {})
      : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

  schedule_entry_t link_of(int pid, int timestep, int my_rank, int num_gpus) const override {
    const int start    = py_mod(pid, num_gpus);
    const int peer_idx = py_mod(start + timestep, num_gpus);
    const int peer     = py_mod(my_rank + peer_idx, num_gpus);
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

// ─── pid_partitioned_layout_t ───────────────────────────────────────
// Each WG permanently assigned to one link (partitioned AG).
class pid_partitioned_layout_t : public collective_layout_t {
 public:
  explicit pid_partitioned_layout_t(int num_gpus, work_graph_fn_t wg_fn = {})
      : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

  schedule_entry_t link_of(int pid, int /*timestep*/, int my_rank, int num_gpus) const override {
    const int dest     = py_mod(pid, num_gpus);
    const int peer     = py_mod(my_rank + dest, num_gpus);
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
// Distribute num_wgs across min(num_wgs, N-1) rings, first `extra`
// rings carry one more WG. Shared by every ring layout.
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

// ─── ring_fixed_layout_t ────────────────────────────────────────────
// All hops to next_rank. Used by ring AR (older form).
class ring_fixed_layout_t : public collective_layout_t {
 public:
  explicit ring_fixed_layout_t(int num_gpus, work_graph_fn_t wg_fn = {})
      : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

  schedule_entry_t link_of(int /*pid*/,
                           int /*timestep*/,
                           int my_rank,
                           int num_gpus) const override {
    const int next_rank = py_mod(my_rank + 1, num_gpus);
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
    const int next_rank = py_mod(my_rank + 1, num_gpus);
    const int prev_rank = py_mod(my_rank - 1, num_gpus);
    return {
        load_t{}, wait_t{prev_rank}, pull_t{prev_rank}, reduce_t{}, store_t{}, signal_t{next_rank}};
  }

  int num_gpus_;
  work_graph_fn_t wg_fn_;
};

// ─── ring_all_gather_layout_t ────────────────────────────────────────
// N-1 step ring. Each step: load_t (local) + store_t (local) + push_t (fwd).
class ring_all_gather_layout_t : public collective_layout_t {
 public:
  explicit ring_all_gather_layout_t(int num_gpus) : num_gpus_{num_gpus} {}

  schedule_entry_t link_of(int /*pid*/,
                           int /*timestep*/,
                           int my_rank,
                           int num_gpus) const override {
    const int next_rank    = py_mod(my_rank + 1, num_gpus);
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

// ─── ring_reduce_scatter_layout_t ────────────────────────────────────
// Structurally identical to AG ring + reduce_t.
class ring_reduce_scatter_layout_t : public collective_layout_t {
 public:
  explicit ring_reduce_scatter_layout_t(int num_gpus) : num_gpus_{num_gpus} {}

  schedule_entry_t link_of(int /*pid*/,
                           int /*timestep*/,
                           int my_rank,
                           int num_gpus) const override {
    const int next_rank    = py_mod(my_rank + 1, num_gpus);
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

// ─── two_shot_all_reduce_layout_t ─────────────────────────────────────
// reduce_t (N steps, pid-staggered) + Broadcast (N-1 steps, no self).
class two_shot_all_reduce_layout_t : public collective_layout_t {
 public:
  explicit two_shot_all_reduce_layout_t(int num_gpus) : num_gpus_{num_gpus} {}

  schedule_entry_t link_of(int pid, int timestep, int my_rank, int num_gpus) const override {
    const int N     = num_gpus;
    const int start = py_mod(pid, N);
    if (is_reduce_phase_(timestep)) {
      const int peer_idx     = py_mod(start + timestep, N);
      const int peer         = py_mod(my_rank + peer_idx, N);
      const bool is_self     = (peer == my_rank);
      std::vector<op_t> work = is_self ? std::vector<op_t>{load_t{}, reduce_t{}}
                                       : std::vector<op_t>{pull_t{peer}, reduce_t{}};
      return {is_self ? SELF_LINK : peer, peer, direction_t::PULL, std::move(work), is_self};
    }
    // Broadcast phase. Skip self in peer ordering.
    const int bcast_idx    = timestep - N;
    const int peer_offset  = py_mod(start + bcast_idx, N - 1) + 1;
    const int peer         = py_mod(my_rank + peer_offset, N);
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

// ─── ring_all_reduce_layout_t ────────────────────────────────────────
// RS phase (N-1 steps) + AG phase (N-1 steps).
class ring_all_reduce_layout_t : public collective_layout_t {
 public:
  explicit ring_all_reduce_layout_t(int num_gpus) : num_gpus_{num_gpus} {}

  schedule_entry_t link_of(int /*pid*/, int timestep, int my_rank, int num_gpus) const override {
    const int next_rank = py_mod(my_rank + 1, num_gpus);
    const int prev_rank = py_mod(my_rank - 1, num_gpus);
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

// ─── ring_broadcast_layout_t ────────────────────────────────────────
// N-1 hop pipeline on 1 link. Each GPU: load_t + store_t + push_t.
class ring_broadcast_layout_t : public collective_layout_t {
 public:
  explicit ring_broadcast_layout_t(int num_gpus) : num_gpus_{num_gpus} {}

  schedule_entry_t link_of(int /*pid*/,
                           int /*timestep*/,
                           int my_rank,
                           int num_gpus) const override {
    const int next_rank    = py_mod(my_rank + 1, num_gpus);
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
// Mirror the Python free-function builders.

inline std::unique_ptr<collective_layout_t> allgather_layout(int num_gpus) {
  return std::make_unique<ring_all_gather_layout_t>(num_gpus);
}

inline std::unique_ptr<collective_layout_t> reduce_scatter_layout(int num_gpus) {
  return std::make_unique<ring_reduce_scatter_layout_t>(num_gpus);
}

inline std::unique_ptr<collective_layout_t> broadcast_layout(int num_gpus) {
  return std::make_unique<ring_broadcast_layout_t>(num_gpus);
}

inline std::unique_ptr<collective_layout_t> allreduce_one_shot_layout(int num_gpus) {
  auto wg = [](int peer, int /*my_rank*/, int /*N*/, bool is_self) -> std::vector<op_t> {
    if (is_self) return {load_t{}, reduce_t{}};
    return {pull_t{peer}, reduce_t{}};
  };
  return std::make_unique<all_to_same_layout_t>(num_gpus, wg);
}

inline std::unique_ptr<collective_layout_t> allreduce_two_shot_layout(int num_gpus) {
  return std::make_unique<two_shot_all_reduce_layout_t>(num_gpus);
}

inline std::unique_ptr<collective_layout_t> allreduce_ring_layout(int num_gpus) {
  return std::make_unique<ring_all_reduce_layout_t>(num_gpus);
}

inline std::unique_ptr<collective_layout_t> alltoall_layout(int num_gpus) {
  auto wg = [](int peer, int /*my_rank*/, int /*N*/, bool is_self) -> std::vector<op_t> {
    if (is_self) return {load_t{}, store_t{}};
    return {load_t{}, push_t{peer}};
  };
  return std::make_unique<pid_staggered_layout_t>(num_gpus, wg);
}

}  // namespace origami::comm
