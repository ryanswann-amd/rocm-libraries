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

/**
 * @file
 * @brief origami::graphs — integer-timestep runtimes.
 *
 * A graph says only what depends on what. A *runtime* says when each node is
 * dispatched, turning dataflow into a `schedule_t` that gives every node a start
 * timestep. The runtimes here charge every workgroup the same `wg_duration`, so
 * time is just workgroup counting; the continuous-time runtimes that ask a cost
 * model for a real duration per node live alongside these in cost_runtime.hpp.
 *
 * The hardware runs at most `lanes` workgroups at once — think CU partitions, or
 * occupancy. Leaving `lanes` unset means unlimited, which collapses a whole
 * kernel into one timestep and makes every policy below agree. The policies only
 * diverge once lanes are scarce, which is the interesting case:
 *
 *   - `breadth_first_runtime_t` runs one whole operation, syncs, then starts the
 *     next. Operations never overlap, so its makespan is the no-overlap baseline
 *     that the others are measured against.
 *   - `asap_runtime_t` is producer-greedy: ready nodes are taken in operator
 *     order, so producers drain first and consumers fill whatever lanes are left
 *     over. This is the throughput-maximising end of the range.
 *   - `depth_first_runtime_t` is chain-greedy: it reads the edges and prefers a
 *     node's consumer over a fresh sibling producer, pushing one tile the whole
 *     way through the pipeline early. Throughput can be worse than ASAP, but the
 *     pipeline fills sooner, which is what a fused producer-consumer kernel
 *     streaming a single tile end to end actually does.
 *
 * All three are the same greedy list scheduler under a different priority. That
 * is deliberate: the priority is the whole of the policy, so a new runtime is a
 * new ordering rather than a new scheduler.
 *
 * These answer the question analysis.hpp cannot. The critical path is the
 * unbounded-resource bound and ignores the fact that a machine has a finite
 * number of CUs; these runtimes take that number as input.
 */
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "origami/graphs/types.hpp"
#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

/** @brief Unlimited lanes, i.e. every ready node dispatches immediately. */
inline constexpr std::optional<int> unlimited_lanes = std::nullopt;

/** @brief The nodes dispatched at one timestep. */
struct timestep_t {
  index_t time = 0;              ///< the timestep
  std::vector<wg_node_t> nodes;  ///< nodes starting at it, in (operation, workgroup) order
};

/**
 * @brief A runtime's output: a start timestep for every node.
 *
 * The reference carries an `op_order` map from operation name to its position,
 * because its nodes name their operation. Here a node already stores the
 * operation *index*, which is that position, so the map would be the identity
 * and is left out.
 */
struct schedule_t {
  std::string runtime;           ///< policy that produced this
  index_t wg_duration = 1;       ///< timesteps charged per node
  std::optional<int> lanes;      ///< concurrency limit; unset means unlimited
  wg_node_map_t<index_t> start;  ///< start timestep per node
  std::vector<wg_node_t> order;  ///< nodes in the order they were dispatched

  /**
   * @brief Finish timestep of a node.
   *
   * @param node Node to query.
   * @return index_t start + wg_duration.
   * @throws std::out_of_range If the node is not in this schedule.
   */
  index_t finish(const wg_node_t& node) const;

  /** @brief Last finish time over every node; zero for an empty schedule. */
  index_t makespan() const;

  /**
   * @brief Nodes grouped by start timestep, in increasing time.
   *
   * Within a group nodes are sorted by (operation, workgroup) — note the absent
   * iteration, which the reference also omits. Ties therefore fall back on
   * dispatch order, which is why `order` is kept.
   *
   * @return std::vector<timestep_t> One entry per occupied timestep.
   */
  std::vector<timestep_t> timesteps() const;

  /** @brief Human-readable summary: node count, timesteps, makespan. */
  std::string summary() const;
};

/**
 * @brief A policy that turns a dataflow graph into a timed schedule.
 *
 * Takes `wg_graph_t`, so a policy is written once and applies to derived and
 * free graphs alike.
 */
class runtime_t {
 public:
  virtual ~runtime_t() = default;

  /** @brief Policy name, as it appears in a schedule. */
  virtual const std::string& name() const = 0;

  /**
   * @brief Assign every node a start timestep.
   *
   * @param graph Graph to schedule.
   * @return schedule_t The resulting schedule.
   */
  virtual schedule_t schedule(const wg_graph_t& graph) const = 0;
};

/**
 * @brief Whole operation, sync, next operation: the no-overlap baseline.
 *
 * Each operation runs to completion across `ceil(n / lanes)` waves before the
 * next starts. With unlimited lanes every operation is a single timestep, so
 * operation `i` runs at timestep `i`.
 *
 * This is the only runtime here that ignores the edges entirely. It does not
 * need them: running operations one at a time satisfies any dependency that
 * derivation could produce, since derived edges only ever point forwards.
 */
class breadth_first_runtime_t : public runtime_t {
 public:
  /**
   * @brief Construct the policy.
   *
   * @param lanes Concurrent workgroup limit; unset means unlimited.
   * @param wg_duration Timesteps charged per node.
   * @throws std::invalid_argument If lanes or wg_duration is below one.
   */
  explicit breadth_first_runtime_t(std::optional<int> lanes = unlimited_lanes,
                                   index_t wg_duration      = 1);

  const std::string& name() const override { return name_; }
  schedule_t schedule(const wg_graph_t& graph) const override;

 private:
  std::optional<int> lanes_;
  index_t wg_duration_;
  std::string name_ = "breadth-first";
};

/**
 * @brief Producer-greedy pipelining: drain producers, let consumers fill in.
 *
 * Ready nodes are taken in canonical (operation, workgroup, iteration) order, so
 * the earliest operation claims the lanes and a consumer only runs on what is
 * left. With unlimited lanes this is the as-soon-as-possible bound, where a
 * node's timestep is its longest dependency depth. With scarce lanes a consumer
 * slips into the gap left by a producer's ragged final wave, which is where the
 * overlap against breadth-first comes from.
 */
class asap_runtime_t : public runtime_t {
 public:
  /**
   * @brief Construct the policy.
   *
   * @param lanes Concurrent workgroup limit; unset means unlimited.
   * @param wg_duration Timesteps charged per node.
   * @throws std::invalid_argument If lanes or wg_duration is below one.
   */
  explicit asap_runtime_t(std::optional<int> lanes = unlimited_lanes, index_t wg_duration = 1);

  const std::string& name() const override { return name_; }
  schedule_t schedule(const wg_graph_t& graph) const override;

 private:
  std::optional<int> lanes_;
  index_t wg_duration_;
  std::string name_ = "asap (pipelined)";
};

/**
 * @brief Chain-greedy pipelining: dive down one producer-consumer chain.
 *
 * Priority is the visit order of a depth-first walk along the edges from the
 * source nodes, so a node's consumer ranks immediately after it and beats a
 * sibling producer that has not started. One result reaches the end of the
 * pipeline as early as timestep 1.
 */
class depth_first_runtime_t : public runtime_t {
 public:
  /**
   * @brief Construct the policy.
   *
   * @param lanes Concurrent workgroup limit; unset means unlimited.
   * @param wg_duration Timesteps charged per node.
   * @throws std::invalid_argument If lanes or wg_duration is below one.
   */
  explicit depth_first_runtime_t(std::optional<int> lanes = unlimited_lanes,
                                 index_t wg_duration      = 1);

  const std::string& name() const override { return name_; }
  schedule_t schedule(const wg_graph_t& graph) const override;

 private:
  std::optional<int> lanes_;
  index_t wg_duration_;
  std::string name_ = "depth-first";
};

/**
 * @brief Depth-first visit order over the edges, used as a scheduling priority.
 *
 * Exposed because it is the whole of the depth-first policy and is worth being
 * able to inspect and test on its own.
 *
 * The walk starts at the source nodes in canonical order and dives along
 * successors, also in canonical order. Nodes unreachable from any source — which
 * derivation cannot produce, but a free graph can — are given trailing ranks so
 * they stay schedulable.
 *
 * @note This is the one place the port deliberately departs from the reference,
 *       which sorts sources and successors by operation *name*. That is a slip
 *       rather than a decision: everywhere else, including the ASAP priority in
 *       the same file, orders by the operation's position in the operator list,
 *       and sorting by name makes the dive order depend on what the kernels were
 *       called. In the reference a graph declaring `z` before `b` dispatches `b`
 *       first, and renaming an operation silently reschedules it. Canonical
 *       order is used here instead. The two agree unless a node has consumers in
 *       two different operations whose names sort against their declaration
 *       order; oracle/compare.sh pins where they differ, and even there only the
 *       order changes, not the makespan.
 *
 * @param graph Graph to walk.
 * @return wg_node_map_t<std::size_t> A distinct rank per node, source-first.
 */
wg_node_map_t<std::size_t> depth_first_rank(const wg_graph_t& graph);

}  // namespace origami::graphs
