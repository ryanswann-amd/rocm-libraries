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

/// Sentinel link id: the timestep is local (a rank maps to itself, no fabric link is used).
inline constexpr int SELF_LINK = -1;

/**
 * @brief One workgroup's communication assignment for a single timestep.
 *
 * Returned by collective_algorithm_t::link_of: it names the fabric link to use,
 * the peer rank on the other end, the transfer direction, and the ordered work
 * graph of primitives to run there. When the step maps a rank to itself the
 * entry is flagged local (is_self) and link_id is SELF_LINK.
 */
struct schedule_entry_t {
  int link_id;                   ///< Fabric link id, or SELF_LINK when the step is local.
  int peer_rank;                 ///< Rank on the other end of the link (== own rank if self).
  direction_t direction;         ///< Whether data is pulled from or pushed to the peer.
  std::vector<op_t> work_graph;  ///< Ordered primitives the workgroup runs this timestep.
  bool is_self = false;          ///< True when the step is local (data already resident).
};

/**
 * @brief Closure that builds the per-timestep work graph for one workgroup.
 *
 * Signature: (peer, my_rank, num_gpus, is_self) -> [op_t]. Concrete algorithms
 * accept one to let callers customize the primitives emitted on each hop while
 * reusing the algorithm's link/timestep schedule.
 *
 * @param peer Peer rank for this hop.
 * @param my_rank Calling rank.
 * @param num_gpus Communicator size.
 * @param is_self True when the hop maps the rank to itself (data is local).
 * @return Ordered list of primitives to run for this hop.
 */
using work_graph_fn_t =
    std::function<std::vector<op_t>(int peer, int my_rank, int num_gpus, bool is_self)>;

/**
 * @brief Floored modulo: always returns a value in the half-open range [0, n).
 *
 * Ring schedules index peers as my_rank ± offset, which can go negative or past
 * N; this wraps those into a valid rank so a single closed form expresses "the
 * neighbour k hops away".
 *
 * @param a Possibly negative or out-of-range index.
 * @param n Modulus (communicator size); must be positive.
 * @return a reduced into the range [0, n).
 */
constexpr int floor_mod(int a, int n) noexcept {
  const int r = a % n;
  return (r < 0) ? r + n : r;
}

// ─── Base ────────────────────────────────────────────────────────
/**
 * @brief Abstract base for collective algorithms: the per-step schedule of a collective.
 *
 * Each algorithm emits a pure function of (pid, timestep) describing which link
 * a workgroup uses and what primitives it runs there, decoupling the cost model
 * from any hard-coded per-collective cost. num_gpus is fixed at construction
 * (num_gpus_ on each concrete algorithm), so the schedule queries below take
 * only the per-call coordinates (pid, timestep, my_rank, num_wgs) and read the
 * communicator size from the object.
 */
class collective_algorithm_t {
 public:
  /// @brief Virtual destructor for safe polymorphic deletion.
  virtual ~collective_algorithm_t() = default;

  /**
   * @brief Resolve the link, peer, and work graph for one workgroup at a timestep.
   *
   * @param pid Workgroup (partition) id within the launch.
   * @param timestep Communication round index (0-based).
   * @param my_rank Calling rank in the communicator.
   * @return schedule_entry_t naming the link, peer, direction, and primitives to run.
   */
  virtual schedule_entry_t link_of(int pid, int timestep, int my_rank) const = 0;

  /**
   * @brief Number of workgroups sharing a single active link this timestep.
   *
   * @param timestep Communication round index (0-based).
   * @param num_wgs Total workgroups participating in the collective.
   * @return Per-link workgroup count, used to price per-link contention.
   */
  virtual int wgs_on_link(int timestep, int num_wgs) const = 0;

  /**
   * @brief Map of the links lit up this timestep to the workgroups on each.
   *
   * @param timestep Communication round index (0-based).
   * @param num_wgs Total workgroups participating in the collective.
   * @return Map from link id to the number of workgroups assigned to that link.
   */
  virtual std::unordered_map<int, int> active_links(int timestep, int num_wgs) const = 0;

  /**
   * @brief Number of dependent communication rounds the algorithm takes.
   *
   * More steps mean more serial handshakes and, for sequential algorithms, more
   * added latency (e.g. a ring visits N-1 peers; two-shot does 2N-1 rounds).
   *
   * @return Count of timesteps in the schedule.
   */
  virtual int num_timesteps() const = 0;

  /**
   * @brief How finely each GPU's tile is sliced per timestep.
   *
   * Default: each timestep moves the whole gpu_tile (= 1). Chunked algorithms
   * override (ring, two-shot, a2a → N). This sets the per-step wire bytes.
   *
   * @return Number of chunks the per-GPU tile is split into per timestep.
   */
  virtual int chunks_per_timestep() const { return 1; }

  /**
   * @brief Whether the algorithm is ring-class for the per-step overhead heuristic.
   *
   * One of the two algorithm-kind checks the collective engine makes to identify
   * ring-class algorithms: a ring-class algorithm is eligible for the per-step
   * proxy/sync overhead heuristic. Covers AG, RS, ring AR, ring fixed.
   *
   * @return True for ring-class algorithms; false otherwise.
   */
  virtual bool is_ring_class() const { return false; }

  /**
   * @brief Whether to price the algorithm with the closed-form pipelined-ring model.
   *
   * The second algorithm-kind check: a pipelined ring uses the closed-form
   * pipelined-ring throughput model rather than per-timestep wg_tile sums.
   * Covers ring AR + ring fixed (the algorithms whose work graph carries
   * signal_t/wait_t ops).
   *
   * @return True for pipelined-ring algorithms; false otherwise.
   */
  virtual bool is_ring_pipeline() const { return false; }
};

/**
 * @brief Sequential collective where all workgroups target the same link each timestep.
 *
 * All WGs → same link each timestep. Used by one_shot AR + A2A (sequential).
 * Skips self; visits the N-1 remote peers one per round.
 */
class all_to_same_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size with an optional work-graph closure.
   *
   * @param num_gpus Communicator size (number of ranks).
   * @param wg_fn Per-hop work-graph builder; defaults to pull+store (load+store when local).
   */
  explicit all_to_same_algorithm_t(int num_gpus, work_graph_fn_t wg_fn = {})
      : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

  /// @brief Target the remote peer (my_rank + timestep + 1), pulling from it (load when local).
  schedule_entry_t link_of(int /*pid*/, int timestep, int my_rank) const override {
    const int peer     = floor_mod(my_rank + timestep + 1, num_gpus_);
    const bool is_self = (peer == my_rank);
    auto work          = wg_fn_(peer, my_rank, num_gpus_, is_self);
    return {is_self ? SELF_LINK : peer, peer, direction_t::PULL, std::move(work), is_self};
  }

  /// @brief All workgroups share the single active link, so the count is num_wgs.
  int wgs_on_link(int /*timestep*/, int num_wgs) const override { return num_wgs; }

  /// @brief Exactly one link is active per timestep, carrying all num_wgs workgroups.
  std::unordered_map<int, int> active_links(int timestep, int num_wgs) const override {
    return {{timestep % (num_gpus_ - 1), num_wgs}};
  }

  /// @brief N-1 rounds, one per remote peer.
  int num_timesteps() const override { return num_gpus_ - 1; }

 private:
  /**
   * @brief Default work graph: pull from the peer then store (load+store when local).
   *
   * @param peer Peer rank for this hop.
   * @param is_self True when the hop is local.
   * @return Ordered primitives for the hop.
   */
  static std::vector<op_t> default_work_graph(int peer,
                                              int /*my_rank*/,
                                              int /*num_gpus*/,
                                              bool is_self) {
    if (is_self) return {load_t{}, store_t{}};
    return {pull_t{peer}, store_t{}};
  }

  int num_gpus_;           ///< Communicator size.
  work_graph_fn_t wg_fn_;  ///< Per-hop work-graph builder.
};

/**
 * @brief Direct collective with each workgroup's starting peer staggered by pid.
 *
 * pid % world_size offsets the starting peer; workgroups spread uniformly over
 * the remote links. Used by two-shot AR/RS and a2a. Includes a self-timestep.
 */
class pid_staggered_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size with an optional work-graph closure.
   *
   * @param num_gpus Communicator size (number of ranks).
   * @param wg_fn Per-hop work-graph builder; defaults to pull+reduce (load+reduce when local).
   */
  explicit pid_staggered_algorithm_t(int num_gpus, work_graph_fn_t wg_fn = {})
      : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

  /// @brief Stagger the starting peer by pid, then step through peers, pulling each.
  schedule_entry_t link_of(int pid, int timestep, int my_rank) const override {
    const int start    = floor_mod(pid, num_gpus_);
    const int peer_idx = floor_mod(start + timestep, num_gpus_);
    const int peer     = floor_mod(my_rank + peer_idx, num_gpus_);
    const bool is_self = (peer == my_rank);
    auto work          = wg_fn_(peer, my_rank, num_gpus_, is_self);
    return {is_self ? SELF_LINK : peer, peer, direction_t::PULL, std::move(work), is_self};
  }

  /// @brief Spread the remote workgroups evenly across the N-1 links.
  int wgs_on_link(int /*timestep*/, int num_wgs) const override {
    const int num_links  = num_gpus_ - 1;
    const int remote_wgs = num_wgs * (num_gpus_ - 1) / num_gpus_;
    return std::max(remote_wgs / std::max(num_links, 1), 1);
  }

  /// @brief All N-1 links are active each timestep with the remote workgroups spread evenly.
  std::unordered_map<int, int> active_links(int /*timestep*/, int num_wgs) const override {
    const int num_links  = num_gpus_ - 1;
    const int remote_wgs = num_wgs * (num_gpus_ - 1) / num_gpus_;
    const int per_link   = std::max(remote_wgs / std::max(num_links, 1), 1);
    std::unordered_map<int, int> out;
    for (int i = 0; i < num_links; ++i) out[i] = per_link;
    return out;
  }

  /// @brief N rounds, including the self-timestep.
  int num_timesteps() const override { return num_gpus_; }
  /// @brief Each step moves 1/N of the buffer.
  int chunks_per_timestep() const override { return num_gpus_; }

 private:
  /**
   * @brief Default work graph: pull from the peer then reduce (load+reduce when local).
   *
   * @param peer Peer rank for this hop.
   * @param is_self True when the hop is local.
   * @return Ordered primitives for the hop.
   */
  static std::vector<op_t> default_work_graph(int peer,
                                              int /*my_rank*/,
                                              int /*num_gpus*/,
                                              bool is_self) {
    if (is_self) return {load_t{}, reduce_t{}};
    return {pull_t{peer}, reduce_t{}};
  }

  int num_gpus_;           ///< Communicator size.
  work_graph_fn_t wg_fn_;  ///< Per-hop work-graph builder.
};

/**
 * @brief Single-step collective with each workgroup permanently bound to one link.
 *
 * Each workgroup is permanently assigned to one link by its pid (partitioned
 * all-gather): every destination is pushed to in a single timestep.
 */
class pid_partitioned_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size with an optional work-graph closure.
   *
   * @param num_gpus Communicator size (number of ranks).
   * @param wg_fn Per-hop work-graph builder; defaults to load+push (load+store when local).
   */
  explicit pid_partitioned_algorithm_t(int num_gpus, work_graph_fn_t wg_fn = {})
      : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

  /// @brief Bind the workgroup to the destination chosen by its pid, pushing to that peer.
  schedule_entry_t link_of(int pid, int /*timestep*/, int my_rank) const override {
    const int dest     = floor_mod(pid, num_gpus_);
    const int peer     = floor_mod(my_rank + dest, num_gpus_);
    const bool is_self = (peer == my_rank);
    auto work          = wg_fn_(peer, my_rank, num_gpus_, is_self);
    return {is_self ? SELF_LINK : peer, peer, direction_t::PUSH, std::move(work), is_self};
  }

  /// @brief Workgroups split evenly across all links (one partition per destination).
  int wgs_on_link(int /*timestep*/, int num_wgs) const override {
    return std::max(num_wgs / num_gpus_, 1);
  }

  /// @brief All N-1 remote links are active with workgroups partitioned evenly across them.
  std::unordered_map<int, int> active_links(int /*timestep*/, int num_wgs) const override {
    const int per_link = std::max(num_wgs / num_gpus_, 1);
    std::unordered_map<int, int> out;
    for (int i = 0; i < num_gpus_ - 1; ++i) out[i] = per_link;
    return out;
  }

  /// @brief A single timestep: every destination is served at once.
  int num_timesteps() const override { return 1; }

 private:
  /**
   * @brief Default work graph: load locally then push to the peer (load+store when local).
   *
   * @param peer Peer rank for this hop.
   * @param is_self True when the hop is local.
   * @return Ordered primitives for the hop.
   */
  static std::vector<op_t> default_work_graph(int peer,
                                              int /*my_rank*/,
                                              int /*num_gpus*/,
                                              bool is_self) {
    if (is_self) return {load_t{}, store_t{}};
    return {load_t{}, push_t{peer}};
  }

  int num_gpus_;           ///< Communicator size.
  work_graph_fn_t wg_fn_;  ///< Per-hop work-graph builder.
};

// ─── Ring distribution helper ───────────────────────────────────
/**
 * @brief Spread num_wgs workgroups over the available ring links, conserving the total.
 *
 * The defining constraint is conservation: the WG counts returned must sum to
 * exactly num_wgs — every launched workgroup is doing real work on some link,
 * none invented, none dropped. (An earlier even-division form violated this,
 * dropping WGs at high channel counts and fabricating them at low counts,
 * mispricing contention.)
 *
 * There can be at most N-1 distinct ring links, so we use
 * nrings = min(num_wgs, N-1) and hand each ring floor(num_wgs/nrings), giving
 * the first `extra` rings one more so the remainder is absorbed and the total
 * is preserved.
 *
 * @param num_wgs Total workgroups to distribute.
 * @param num_gpus Communicator size (bounds the ring link count at N-1).
 * @return Map from ring link id to its workgroup count; the values sum to num_wgs.
 */
inline std::unordered_map<int, int> ring_distribute(int num_wgs, int num_gpus) {
  const int nrings = std::max(std::min(num_wgs, num_gpus - 1), 1);
  const int base   = num_wgs / nrings;
  const int extra  = num_wgs - base * nrings;
  std::unordered_map<int, int> out;
  for (int i = 0; i < nrings; ++i) { out[i] = base + (i < extra ? 1 : 0); }
  return out;
}

/**
 * @brief Workgroups per ring link, the floored share used to price per-link contention.
 *
 * @param num_wgs Total workgroups to distribute.
 * @param num_gpus Communicator size (bounds the ring link count at N-1).
 * @return floor(num_wgs / nrings), clamped to at least 1.
 */
inline int ring_wgs_per_link(int num_wgs, int num_gpus) noexcept {
  const int nrings = std::max(std::min(num_wgs, num_gpus - 1), 1);
  return std::max(num_wgs / nrings, 1);
}

/**
 * @brief Pipelined ring whose every hop crosses the fixed next_rank neighbour link.
 *
 * All hops go to next_rank. Used by ring all-reduce (older form); the work graph
 * carries the wait_t/signal_t producer→consumer dependency between adjacent ranks.
 */
class ring_fixed_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size with an optional work-graph closure.
   *
   * @param num_gpus Communicator size (number of ranks).
   * @param wg_fn Per-hop work-graph builder; defaults to the reduce-ring graph with signal/wait.
   */
  explicit ring_fixed_algorithm_t(int num_gpus, work_graph_fn_t wg_fn = {})
      : num_gpus_{num_gpus}, wg_fn_{wg_fn ? std::move(wg_fn) : default_work_graph} {}

  /// @brief Every hop pushes to the fixed next-rank neighbour.
  schedule_entry_t link_of(int /*pid*/, int /*timestep*/, int my_rank) const override {
    const int next_rank = floor_mod(my_rank + 1, num_gpus_);
    auto work           = wg_fn_(next_rank, my_rank, num_gpus_, false);
    return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
  }

  /// @brief Workgroups per ring link (see ring_wgs_per_link).
  int wgs_on_link(int /*timestep*/, int num_wgs) const override {
    return ring_wgs_per_link(num_wgs, num_gpus_);
  }

  /// @brief Workgroups distributed across the ring links (see ring_distribute).
  std::unordered_map<int, int> active_links(int /*timestep*/, int num_wgs) const override {
    return ring_distribute(num_wgs, num_gpus_);
  }

  /// @brief N-1 ring hops.
  int num_timesteps() const override { return num_gpus_ - 1; }
  /// @brief Each hop moves 1/N of the buffer.
  int chunks_per_timestep() const override { return num_gpus_; }
  /// @brief Ring-class for the per-step overhead heuristic.
  bool is_ring_class() const override { return true; }
  /// @brief Priced with the closed-form pipelined-ring throughput model.
  bool is_ring_pipeline() const override { return true; }

 private:
  /**
   * @brief Default reduce-ring work graph: wait on prev, pull, reduce, store, signal next.
   *
   * @param my_rank Calling rank (determines its prev/next neighbours).
   * @param num_gpus Communicator size.
   * @return Ordered primitives for the hop, including the signal/wait dependency.
   */
  static std::vector<op_t> default_work_graph(int /*peer*/,
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

  int num_gpus_;           ///< Communicator size.
  work_graph_fn_t wg_fn_;  ///< Per-hop work-graph builder.
};

/**
 * @brief All-gather ring: N-1 hops forwarding each rank's slice around the ring.
 *
 * N-1 step ring. Each step does load_t (local) + store_t (local) + push_t (forward to next).
 */
class ring_all_gather_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size.
   *
   * @param num_gpus Communicator size (number of ranks).
   */
  explicit ring_all_gather_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

  /// @brief Each step loads locally, stores, and pushes forward to the next rank.
  schedule_entry_t link_of(int /*pid*/, int /*timestep*/, int my_rank) const override {
    const int next_rank    = floor_mod(my_rank + 1, num_gpus_);
    std::vector<op_t> work = {
        load_t{},
        store_t{},
        push_t{next_rank},
    };
    return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
  }

  /// @brief Workgroups per ring link (see ring_wgs_per_link).
  int wgs_on_link(int /*timestep*/, int num_wgs) const override {
    return ring_wgs_per_link(num_wgs, num_gpus_);
  }

  /// @brief Workgroups distributed across the ring links (see ring_distribute).
  std::unordered_map<int, int> active_links(int /*timestep*/, int num_wgs) const override {
    return ring_distribute(num_wgs, num_gpus_);
  }

  /// @brief N-1 ring hops.
  int num_timesteps() const override { return num_gpus_ - 1; }
  /// @brief One chunk per step (AG convention: the message is the per-rank send).
  int chunks_per_timestep() const override { return 1; }  // AG convention: msg = per-rank send
  /// @brief Ring-class for the per-step overhead heuristic.
  bool is_ring_class() const override { return true; }
  // Sequential-style throughput model (no pipelined ring).

 private:
  int num_gpus_;  ///< Communicator size.
};

/**
 * @brief Reduce-scatter ring: the all-gather ring with a reduce on each hop.
 *
 * Structurally identical to the all-gather ring plus a reduce_t, so each step
 * loads, reduces, stores, and forwards the slice to the next rank.
 */
class ring_reduce_scatter_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size.
   *
   * @param num_gpus Communicator size (number of ranks).
   */
  explicit ring_reduce_scatter_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

  /// @brief Each step loads, reduces, stores, and pushes forward to the next rank.
  schedule_entry_t link_of(int /*pid*/, int /*timestep*/, int my_rank) const override {
    const int next_rank    = floor_mod(my_rank + 1, num_gpus_);
    std::vector<op_t> work = {
        load_t{},
        reduce_t{},
        store_t{},
        push_t{next_rank},
    };
    return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
  }

  /// @brief Workgroups per ring link (see ring_wgs_per_link).
  int wgs_on_link(int /*timestep*/, int num_wgs) const override {
    return ring_wgs_per_link(num_wgs, num_gpus_);
  }

  /// @brief Workgroups distributed across the ring links (see ring_distribute).
  std::unordered_map<int, int> active_links(int /*timestep*/, int num_wgs) const override {
    return ring_distribute(num_wgs, num_gpus_);
  }

  /// @brief N-1 ring hops.
  int num_timesteps() const override { return num_gpus_ - 1; }
  /// @brief One chunk per step (the per-rank slice).
  int chunks_per_timestep() const override { return 1; }
  /// @brief Ring-class for the per-step overhead heuristic.
  bool is_ring_class() const override { return true; }

 private:
  int num_gpus_;  ///< Communicator size.
};

/**
 * @brief All-reduce factored into a reduce-scatter shot then an all-gather shot.
 *
 * All-reduce factored as reduce-scatter then all-gather ("two shots"): each rank
 * first pulls and sums every peer's slice (N reduce steps, including its own
 * self-step), then pushes the finished slice out to all others (N-1 broadcast
 * steps). Hence num_timesteps = 2N-1 and each step moves 1/N of the buffer
 * (chunks_per_timestep = N). is_reduce_phase_ just splits the timeline at step N
 * into the two shots.
 */
class two_shot_all_reduce_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size.
   *
   * @param num_gpus Communicator size (number of ranks).
   */
  explicit two_shot_all_reduce_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

  /// @brief Reduce phase (steps < N) pulls and sums each slice; broadcast phase pushes it out.
  schedule_entry_t link_of(int pid, int timestep, int my_rank) const override {
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

  /// @brief Remote workgroups spread over the N-1 links (fewer in the reduce phase's self-step).
  int wgs_on_link(int timestep, int num_wgs) const override {
    const int num_links = num_gpus_ - 1;
    const int remote_wgs =
        is_reduce_phase_(timestep) ? num_wgs * (num_gpus_ - 1) / num_gpus_ : num_wgs;
    return std::max(remote_wgs / std::max(num_links, 1), 1);
  }

  /// @brief All N-1 links active each step, workgroups spread evenly across them.
  std::unordered_map<int, int> active_links(int timestep, int num_wgs) const override {
    const int num_links = num_gpus_ - 1;
    const int remote_wgs =
        is_reduce_phase_(timestep) ? num_wgs * (num_gpus_ - 1) / num_gpus_ : num_wgs;
    const int per_link = std::max(remote_wgs / std::max(num_links, 1), 1);
    std::unordered_map<int, int> out;
    for (int i = 0; i < num_links; ++i) out[i] = per_link;
    return out;
  }

  /// @brief 2N-1 rounds: N reduce steps (incl. self) then N-1 broadcast steps.
  int num_timesteps() const override { return 2 * num_gpus_ - 1; }
  /// @brief Each step moves 1/N of the buffer.
  int chunks_per_timestep() const override { return num_gpus_; }

 private:
  /// @brief True for the first N timesteps (the reduce-scatter shot); false in the broadcast shot.
  constexpr bool is_reduce_phase_(int timestep) const noexcept { return timestep < num_gpus_; }
  int num_gpus_;  ///< Communicator size.
};

/**
 * @brief Bandwidth-optimal all-reduce: a reduce-scatter ring then an all-gather ring.
 *
 * A reduce-scatter ring (N-1 steps, each pulls from prev, sums, signals next)
 * followed by an all-gather ring (N-1 steps, each pulls the finished slice and
 * forwards it). 2(N-1) steps total, every step crossing the same neighbour link
 * — which is why it is a true pipelined ring (is_ring_pipeline) priced by
 * aggregate throughput, not a sum of per-step latencies. The wait_t/signal_t in
 * the work graph are the producer→consumer dependency that serializes adjacent
 * ranks within a step.
 */
class ring_all_reduce_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size.
   *
   * @param num_gpus Communicator size (number of ranks).
   */
  explicit ring_all_reduce_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

  /// @brief Reduce-scatter phase (pull+reduce) for the first N-1 steps, then all-gather (pull).
  schedule_entry_t link_of(int /*pid*/, int timestep, int my_rank) const override {
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

  /// @brief Workgroups per ring link (see ring_wgs_per_link).
  int wgs_on_link(int /*timestep*/, int num_wgs) const override {
    return ring_wgs_per_link(num_wgs, num_gpus_);
  }

  /// @brief Workgroups distributed across the ring links (see ring_distribute).
  std::unordered_map<int, int> active_links(int /*timestep*/, int num_wgs) const override {
    return ring_distribute(num_wgs, num_gpus_);
  }

  /// @brief 2(N-1) hops: a reduce-scatter ring followed by an all-gather ring.
  int num_timesteps() const override { return 2 * (num_gpus_ - 1); }
  /// @brief Each hop moves 1/N of the buffer.
  int chunks_per_timestep() const override { return num_gpus_; }
  /// @brief Ring-class for the per-step overhead heuristic.
  bool is_ring_class() const override { return true; }
  /// @brief Priced with the closed-form pipelined-ring throughput model.
  bool is_ring_pipeline() const override { return true; }

 private:
  int num_gpus_;  ///< Communicator size.
};

/**
 * @brief Broadcast as an N-1 hop pipeline forwarding the root's data around the ring.
 *
 * N-1 hop pipeline on one link. Each GPU does load_t + store_t + push_t,
 * forwarding the data to its next-rank neighbour.
 */
class ring_broadcast_algorithm_t : public collective_algorithm_t {
 public:
  /**
   * @brief Construct for a fixed communicator size.
   *
   * @param num_gpus Communicator size (number of ranks).
   */
  explicit ring_broadcast_algorithm_t(int num_gpus) : num_gpus_{num_gpus} {}

  /// @brief Each step loads locally, stores, and pushes forward to the next rank.
  schedule_entry_t link_of(int /*pid*/, int /*timestep*/, int my_rank) const override {
    const int next_rank    = floor_mod(my_rank + 1, num_gpus_);
    std::vector<op_t> work = {
        load_t{},
        store_t{},
        push_t{next_rank},
    };
    return {next_rank, next_rank, direction_t::PUSH, std::move(work), false};
  }

  /// @brief Workgroups per ring link (see ring_wgs_per_link).
  int wgs_on_link(int /*timestep*/, int num_wgs) const override {
    return ring_wgs_per_link(num_wgs, num_gpus_);
  }

  /// @brief Workgroups distributed across the ring links (see ring_distribute).
  std::unordered_map<int, int> active_links(int /*timestep*/, int num_wgs) const override {
    return ring_distribute(num_wgs, num_gpus_);
  }

  /// @brief N-1 ring hops.
  int num_timesteps() const override { return num_gpus_ - 1; }
  /// @brief Each hop moves 1/N of the buffer.
  int chunks_per_timestep() const override { return num_gpus_; }

 private:
  int num_gpus_;  ///< Communicator size.
};

// ─── Standard collective constructors ───────────────────────────
// Free-function algorithm builders.

/**
 * @brief Build the all-gather algorithm (ring).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to a ring all-gather algorithm.
 */
inline std::unique_ptr<collective_algorithm_t> allgather_algorithm(int num_gpus) {
  return std::make_unique<ring_all_gather_algorithm_t>(num_gpus);
}

/**
 * @brief Build the reduce-scatter algorithm (ring).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to a ring reduce-scatter algorithm.
 */
inline std::unique_ptr<collective_algorithm_t> reduce_scatter_algorithm(int num_gpus) {
  return std::make_unique<ring_reduce_scatter_algorithm_t>(num_gpus);
}

/**
 * @brief Build the broadcast algorithm (ring).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to a ring broadcast algorithm.
 */
inline std::unique_ptr<collective_algorithm_t> broadcast_algorithm(int num_gpus) {
  return std::make_unique<ring_broadcast_algorithm_t>(num_gpus);
}

/**
 * @brief Build the one-shot all-reduce algorithm (all-to-same with pull+reduce work graph).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to an all-to-same algorithm configured for one-shot all-reduce.
 */
inline std::unique_ptr<collective_algorithm_t> allreduce_one_shot_algorithm(int num_gpus) {
  auto wg = [](int peer, int /*my_rank*/, int /*N*/, bool is_self) -> std::vector<op_t> {
    if (is_self) return {load_t{}, reduce_t{}};
    return {pull_t{peer}, reduce_t{}};
  };
  return std::make_unique<all_to_same_algorithm_t>(num_gpus, wg);
}

/**
 * @brief Build the two-shot all-reduce algorithm (reduce-scatter then all-gather).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to a two-shot all-reduce algorithm.
 */
inline std::unique_ptr<collective_algorithm_t> allreduce_two_shot_algorithm(int num_gpus) {
  return std::make_unique<two_shot_all_reduce_algorithm_t>(num_gpus);
}

/**
 * @brief Build the ring all-reduce algorithm (bandwidth-optimal pipelined ring).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to a ring all-reduce algorithm.
 */
inline std::unique_ptr<collective_algorithm_t> allreduce_ring_algorithm(int num_gpus) {
  return std::make_unique<ring_all_reduce_algorithm_t>(num_gpus);
}

/**
 * @brief Build the all-to-all algorithm (direct, pid-staggered with load+push work graph).
 *
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to a pid-staggered algorithm configured for all-to-all.
 */
inline std::unique_ptr<collective_algorithm_t> alltoall_algorithm(int num_gpus) {
  auto wg = [](int peer, int /*my_rank*/, int /*N*/, bool is_self) -> std::vector<op_t> {
    if (is_self) return {load_t{}, store_t{}};
    return {load_t{}, push_t{peer}};
  };
  return std::make_unique<pid_staggered_algorithm_t>(num_gpus, wg);
}

/**
 * @brief Map a (collective, algorithm) pair to a concrete algorithm implementation.
 *
 * The single place the validity rule lives: an algorithm is a valid config only
 * if it is defined for the problem's collective. `automatic` maps to each
 * collective's canonical algorithm (preserving the historical defaults); a named
 * value selects a specific one and any undefined (collective, algorithm) pair
 * throws rather than silently costing a nonsense schedule.
 *
 *   all_gather / reduce_scatter / broadcast : ring only
 *   all_reduce                              : two_shot (default), one_shot, ring
 *   all_to_all                              : direct (pid-staggered) only
 *
 * @param collective Collective primitive to implement.
 * @param algorithm Requested algorithm, or algorithm_t::automatic for the canonical default.
 * @param num_gpus Communicator size (number of ranks).
 * @return Owning pointer to the resolved algorithm.
 * @throws std::invalid_argument if the algorithm is not valid for the collective.
 */
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

/**
 * @brief Resolve the algorithm from a problem/config bundle (the engine-facing overload).
 *
 * The collective comes from the problem (correctness), the algorithm from the
 * config (performance), and num_gpus is the communicator size on the problem.
 * This is the form the latency engine calls; the scalar overload above stays for
 * unit tests that probe (collective, algorithm) pairs directly.
 *
 * @param problem Communication problem (provides the collective and num_gpus).
 * @param config Communication config (provides the requested algorithm).
 * @return Owning pointer to the resolved algorithm.
 * @throws std::invalid_argument if the algorithm is not valid for the collective.
 */
inline std::unique_ptr<collective_algorithm_t> resolve_algorithm(const comm_problem_t& problem,
                                                                 const comm_config_t& config) {
  return resolve_algorithm(problem.collective, config.algorithm, problem.num_gpus);
}

}  // namespace origami::comm
