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
 * @brief origami::graphs — the shipped scheduling policies.
 *
 * `simulate()` in simulate.hpp owns the ready set, the clock and every cost
 * query; a `runtime_t` here is nothing but a policy over it — a priority, a
 * placement, and how readiness and rank trade off. None of the four below
 * carries a lane count, a per-node duration or a scheduler of its own: that is
 * simulate()'s job now, and asking the same node for both a policy and its cost
 * is exactly the confusion this split removes.
 *
 *   - `breadth_first_runtime_t` runs one whole operation, syncs, then starts
 *     the next. It is the only policy that needs the gate: with independent
 *     operations no priority can stop them overlapping, because both are ready
 *     at once.
 *   - `asap_runtime_t` is producer-greedy: ready nodes are taken in canonical
 *     operator order. `runtime_kind_t`'s `asap`, `roofline` and `event_driven`
 *     all name this one policy; what differs between those three is where a
 *     node's duration comes from, which is a cost model's business and not a
 *     policy's.
 *   - `depth_first_runtime_t` is chain-greedy: its priority is the topological
 *     depth-first visit order over the edges, `depth_first_rank`, taken
 *     rank-first rather than ready-first so the rank actually drives dispatch
 *     instead of only breaking ties nothing ever reaches.
 *   - `xcd_runtime_t` models chiplet dispatch: canonical launch order, taken
 *     rank-first rather than ready-first because a real dispatcher issues
 *     workgroups in grid order rather than picking whichever happens to be
 *     ready, placed by `xcd_placement_t`.
 *
 * These answer the question analysis.hpp cannot. The critical path is the
 * unbounded-resource bound and ignores the fact that a machine has a finite
 * number of CUs; simulate() with one of these policies takes that number as
 * input.
 *
 * `runtime_with_placement_t`, below the four, is not a fifth policy: it wraps
 * any of the above (or a caller's own) and swaps only the placement, for a
 * lane partition or other custom lane rule that composes with whichever
 * issuing order was already chosen.
 */
#pragma once

#include <cstddef>
#include <string>

#include "origami/graphs/placement.hpp"
#include "origami/graphs/simulate.hpp"
#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

/**
 * @brief Topological depth-first visit order over the edges, used as a
 *        scheduling priority.
 *
 * Exposed because it is the whole of the depth-first policy and is worth being
 * able to inspect and test on its own. Declared here, ahead of
 * `depth_first_runtime_t`, so that policy's inline `priority()` can call it.
 *
 * The walk starts at the source nodes in canonical order and dives along
 * successors, also in canonical order, but a node is only emitted once every
 * one of its predecessors already has a rank. That precondition is what makes
 * the rank topological (no edge ever runs backwards over it), which
 * `depth_first_runtime_t` needs: it dispatches rank-first
 * (`queue_order_t::rank_first`), and `simulate()` rejects a rank-first runtime
 * outright if any edge violates that order. A plain DFS that emits a node the
 * moment any single predecessor reaches it — the walk this function used to
 * be — does not have that property in general: with two predecessors reaching
 * a shared successor at different depths, it can rank the successor before
 * the predecessor still waiting its turn. Nodes unreachable from any source —
 * which derivation cannot produce, but a free graph can — are given trailing
 * ranks so they stay schedulable.
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
 *       order. oracle/compare.sh pins those differences as well as the newer
 *       rank-dominant depth-first differences, including cases where the
 *       makespan changes because the policy deliberately idles lanes to keep
 *       diving down a chain.
 *
 * @param graph Graph to walk.
 * @return wg_node_map_t<std::size_t> A distinct rank per node, source-first.
 */
wg_node_map_t<std::size_t> depth_first_rank(const wg_graph_t& graph);

/**
 * @brief Whole operation, sync, next operation: the no-overlap baseline.
 *
 * The only policy that needs the gate: with independent operations no priority
 * can stop them overlapping, because both are ready at once.
 */
class breadth_first_runtime_t : public runtime_t {
 public:
  const std::string& name() const override { return name_; }
  wg_node_map_t<std::size_t> priority(const wg_graph_t& graph) const override;
  const placement_t& placement() const override { return placement_; }
  bool gated(const wg_node_t& node, const sim_state_t& state) const override;

 private:
  earliest_free_placement_t placement_;
  std::string name_ = "breadth-first";
};

/**
 * @brief Producer-greedy: ready work in canonical order, earliest-ready first.
 *
 * This is also the policy `runtime_kind_t::roofline` and
 * `runtime_kind_t::event_driven` name: all three differ in where a node's
 * duration comes from, not in the order work is issued, which is the confusion
 * splitting policy from price removes.
 */
class asap_runtime_t : public runtime_t {
 public:
  const std::string& name() const override { return name_; }
  wg_node_map_t<std::size_t> priority(const wg_graph_t& graph) const override;
  const placement_t& placement() const override { return placement_; }

 private:
  earliest_free_placement_t placement_;
  std::string name_ = "asap (pipelined)";
};

/**
 * @brief Chain-greedy: dive down one producer-consumer chain.
 *
 * Rank-first (`queue_order_t::rank_first`), not ready-first: a source is
 * ready at cycle 0 exactly like its siblings, so under ready-first dispatch
 * rank only breaks a tie between equally-ready nodes, which two nodes on
 * different chains essentially never are. Dispatching by rank instead lets a
 * node further down a chain preempt a sibling source that happens to be
 * ready sooner, which is the entire point of the policy: a tile is pushed all
 * the way through the pipeline before the next tile is started, even if that
 * means leaving a lane idle for a moment rather than filling it with
 * whatever is ready.
 */
class depth_first_runtime_t : public runtime_t {
 public:
  const std::string& name() const override { return name_; }
  wg_node_map_t<std::size_t> priority(const wg_graph_t& graph) const override {
    return depth_first_rank(graph);
  }
  const placement_t& placement() const override { return placement_; }
  queue_order_t order() const override { return queue_order_t::rank_first; }

 private:
  earliest_free_placement_t placement_;
  std::string name_ = "depth-first";
};

/**
 * @brief A policy with its placement replaced, everything else forwarded.
 *
 * "Take a policy, override its placement" is a composition three separate
 * call sites wrote by hand before this shipped: a `graph_config_t::lane_pool`
 * needs it to wrap whichever policy `runtime` names in a `pooled_placement_t`,
 * the oracle's `priced_dump.cpp` needs the identical wrapper to reproduce that
 * behaviour outside the library, and a caller partitioning lanes between a
 * concurrent comm and GEMM launch from Python needs it too. None of the three
 * differ in any way that matters, so the wrapper belongs here rather than
 * being retyped a fourth time.
 *
 * Stores references rather than owned copies, matching every concrete policy
 * above, whose `placement()` also returns a reference to something that
 * outlives the call: both @p base and @p placement must outlive this adapter.
 */
class runtime_with_placement_t : public runtime_t {
 public:
  /**
   * @brief Wrap a policy, replacing its placement.
   * @param base Policy supplying name, priority, order and gate.
   * @param placement Placement to use instead of the base policy's own.
   */
  runtime_with_placement_t(const runtime_t& base, const placement_t& placement)
      : base_(base), placement_(placement) {}

  const std::string& name() const override { return base_.name(); }
  wg_node_map_t<std::size_t> priority(const wg_graph_t& graph) const override {
    return base_.priority(graph);
  }
  const placement_t& placement() const override { return placement_; }
  queue_order_t order() const override { return base_.order(); }
  bool gated(const wg_node_t& node, const sim_state_t& state) const override {
    return base_.gated(node, state);
  }

 private:
  const runtime_t& base_;
  const placement_t& placement_;
};

/** @brief Chiplet geometry: what `xcd_runtime_t` needs to know about the part. */
struct xcd_options_t {
  int num_xcds    = 8;   ///< accelerator dies
  int cus_per_xcd = 38;  ///< compute units per die

  /**
   * Operations whose name starts with this are treated as zero-work barriers:
   * they take no compute unit and do not consume a round-robin slot, they only
   * forward their ready time. A cheap way to express a whole-stage barrier with
   * a linear rather than quadratic number of edges. Passed straight through to
   * `simulate_options_t::skip_prefix`.
   */
  std::string skip_prefix = "sync";
};

/**
 * @brief Chiplet dispatch in launch order.
 *
 * Rank-first, because a real dispatcher issues workgroups in grid order rather
 * than picking whichever happens to be ready.
 *
 * @note Requires `simulate_options_t::lanes == lane_count()`. `simulate()`
 *       does not know this policy pins one lane per compute unit, and
 *       `xcd_placement_t::choose()` indexes the free-time array for the whole
 *       selected die before returning a lane. Passing too few lanes is therefore
 *       an out-of-bounds read, not a diagnosed placement error.
 */
class xcd_runtime_t : public runtime_t {
 public:
  /**
   * @brief Construct the policy.
   * @param num_xcds Accelerator dies.
   * @param cus_per_xcd Compute units per die.
   * @throws std::invalid_argument If either count is below one.
   */
  xcd_runtime_t(int num_xcds = 8, int cus_per_xcd = 38);

  const std::string& name() const override { return name_; }
  wg_node_map_t<std::size_t> priority(const wg_graph_t& graph) const override;
  const placement_t& placement() const override { return placement_; }
  queue_order_t order() const override { return queue_order_t::rank_first; }

  /** @brief Lanes this policy expects, i.e. dies times CUs per die. */
  int lane_count() const { return placement_.lane_count(); }

 private:
  xcd_placement_t placement_;
  std::string name_ = "xcd";
};

}  // namespace origami::graphs
