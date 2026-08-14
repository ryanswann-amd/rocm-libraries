// SPDX-License-Identifier: MIT
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file spec.hpp
 * @brief origami::graphs — reusable specifications, and the graphs they make.
 *
 * A `graph_t` is one problem at one configuration: concrete grids, concrete
 * edges. That is the right object to schedule and the wrong object to author,
 * because an autotuner wants one description of a kernel and many candidates
 * derived from it. This file supplies the missing half.
 *
 * A `graph_spec_t` holds operations whose workgroup counts and access patterns
 * are still symbolic, and `instantiate` binds a problem and a configuration to
 * produce the graph. Nothing here mentions hardware. That is deliberate: a
 * graph is device-independent, so one instantiation can be ranked against
 * several machines, and hardware arrives later, at ranking, where it is bound
 * into the cost context rather than baked into the structure.
 *
 * `expr_cost_t` is the other half of that story. It is the adapter that lets an
 * operation's deferred `node_cost` satisfy `cost_model_t`, the interface every
 * runtime speaks, so symbolic cost needed no scheduler changes when it landed.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "origami/comm/hardware.hpp"
#include "origami/graphs/core.hpp"
#include "origami/graphs/cost_expr.hpp"
#include "origami/graphs/cost_model.hpp"
#include "origami/graphs/types.hpp"

namespace origami::graphs {

/**
 * @brief Canonical hardware-scope bindings derived from a machine description.
 *
 * The names a cost expression may reference through `hardware_sym`. Origami's
 * `hardware_t` states most rates per cycle, which is the right unit for a
 * machine whose clock moves; the per-second spellings below are the
 * convenience forms, and both are published so a model can choose.
 *
 * **The bigger trap is not the clock symbol below, it is these seven names:**
 * `peak_flops_per_cu`, `hbm_bytes_per_second`, `hbm_read_bytes_per_second`,
 * `hbm_write_bytes_per_second`, `tcp_bytes_per_second_per_cu`,
 * `l2_bytes_per_second_per_cu`, `mall_bytes_per_second`. Dividing a
 * cycle-valued numerator (a byte count, a FLOP count) by any of these lands
 * back in **seconds**, silently, with no `clock_hz` anywhere in the
 * expression to catch a reviewer's eye. Prefer their per-cycle siblings
 * instead — `peak_flops_per_cu_per_cycle`, `hbm_read_bw`, `hbm_write_bw`,
 * `tcp_bw`, `l2_bw_per_cu`, `mall_bw` — which carry no such hazard because
 * there is no clock to divide back out. Note that `hbm_bytes_per_second` is
 * the unqualified spelling of "HBM bandwidth"; it is not the default,
 * `hbm_read_bw` is.
 *
 * `clock_hz`/`clock_ghz` themselves are a narrower, deliberate exception to
 * `node_cost` being cycle-valued end to end: some quantities a caller wants
 * to price — a spec-sheet matrix-core rate, say, which is unavoidably quoted
 * per second because no hardware struct here models it per cycle — cannot
 * reach cycles without one. `hardware_params` publishes the per-cycle name
 * for everything it can derive one for (`hbm_read_bw`, `tcp_bw`, `valu_rate`,
 * ...), so that path never needs the clock at all; reach for `clock_hz` only
 * when a rate you did not derive from `hardware_t` arrives already stated per
 * second, and prefer the per-cycle name otherwise. An expression that divides
 * by `clock_hz` to land back in seconds, rather than multiplying a per-second
 * rate up into cycles, has reopened exactly the hole this refactor closed —
 * and so, with less warning, does dividing by any of the seven names above.
 *
 * @param hardware Machine description.
 * @return param_map_t Bindings for the hardware scope.
 */
param_map_t hardware_params(const origami::comm::hardware_t& hardware);

/**
 * @brief A reusable, hardware-free description of a kernel.
 *
 * Ordered, like the operation list a graph is built from: dependencies flow
 * forwards, so a producer must appear before its consumer.
 */
class graph_spec_t {
 public:
  graph_spec_t() = default;

  /**
   * @brief Describe a kernel as an ordered operation list.
   *
   * @param operations Operations in topological order, still symbolic.
   */
  explicit graph_spec_t(std::vector<operation_t> operations);

  /** @brief The operations, in the order supplied. */
  const std::vector<operation_t>& operations() const { return operations_; }

  /** @brief Append an operation. */
  graph_spec_t& add(operation_t operation);

  /** @brief Number of operations. */
  std::size_t size() const { return operations_.size(); }

  /**
   * @brief Bind a problem and a configuration, and derive the graph.
   *
   * Returns a shared pointer rather than a value for a concrete reason: a cost
   * model holds a back-pointer to the graph it prices, so the graph's address
   * has to be stable from the moment it is priced. Returning by value would
   * move it out from under its own cost model.
   *
   * The graph comes back unpriced even when its operations carry `node_cost`
   * functions, because pricing needs hardware and hardware is not known here.
   * `attach_expr_cost` is the second step, and `rank_graphs` performs it for
   * you when handed a machine.
   *
   * @param problem Problem dimensions.
   * @param config Tuning parameters.
   * @param name Optional label, surfaced in ranking results.
   * @return std::shared_ptr<graph_t> The derived graph.
   * @throws std::out_of_range If a referenced symbol is unbound.
   * @throws std::invalid_argument If a grid resolves invalid.
   */
  std::shared_ptr<graph_t> instantiate(const problem_t& problem,
                                       const config_t& config,
                                       std::string name = "") const;

 private:
  std::vector<operation_t> operations_;
};

/**
 * @brief Price a specification-derived graph for one machine.
 *
 * The second half of instantiation, split out because hardware arrives later
 * than the problem and the configuration. Does nothing when no operation
 * carries a cost function, so it is safe to call on any graph.
 *
 * @param graph Graph to price; must outlive the cost model, which is why this
 *        takes the shared pointer that owns it.
 * @param hardware Hardware-scope bindings; see `hardware_params`.
 * @return bool True when a cost model was attached.
 */
bool attach_expr_cost(const std::shared_ptr<graph_t>& graph, param_map_t hardware);

/**
 * @brief Prices a graph from its operations' deferred cost expressions.
 *
 * The adapter between the two halves of the design. An operation says how one
 * of its nodes is priced, as an expression; a runtime asks how long a node
 * takes, as a number. This holds the graph's problem and config plus the
 * hardware supplied at ranking, and closes the gap at dispatch by filling in
 * the runtime scope and evaluating. The expression is expected to evaluate to
 * cycles — see `operation_t::node_cost` — so `node_cycles`, `node_cycles_at`
 * and `edge_cycles` are the whole of this class's interface, and the number
 * they return needs no conversion to be handed to `simulate()` or to
 * `wg_graph_t::set_cost`.
 *
 * Expressions are built once per node and cached, so repeated dispatch pays for
 * a tree walk rather than for rebuilding the tree.
 */
class expr_cost_t : public cost_model_t {
 public:
  /**
   * @brief Bind a graph's specification-owned costs to a machine.
   *
   * @param graph Graph whose operations carry `node_cost` functions.
   * @param hardware Hardware-scope bindings; see `hardware_params`.
   */
  expr_cost_t(const graph_t& graph, param_map_t hardware);

  /** @brief True when at least one operation carries a cost function. */
  static bool priced(const graph_t& graph);

  /**
   * @brief Price a node as if it ran alone.
   *
   * @param node Node to price.
   * @return double Cycles.
   */
  double node_cycles(const wg_node_t& node) const override;

  /**
   * @brief Price a node against live occupancy.
   *
   * Binds `runtime.active_cus` before evaluating, which is what makes a
   * contention term in a cost expression mean anything.
   *
   * @param node Node to price.
   * @param active_cus Lanes busy at dispatch, including this node.
   * @return double Cycles.
   */
  double node_cycles_at(const wg_node_t& node, int active_cus) const override;

  /**
   * @brief Hop latency; free, since a specification prices nodes only.
   *
   * @param edge Hop to price.
   * @return double Zero.
   */
  double edge_cycles(const edge_t& edge) const override;

 private:
  double evaluate(const wg_node_t& node, int active_cus) const;
  const cost_expr_t& expression_for(const wg_node_t& node) const;

  const graph_t* graph_;
  cost_context_t ctx_;
  mutable wg_node_map_t<cost_expr_t> cache_;
};

}  // namespace origami::graphs
