// SPDX-License-Identifier: MIT
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file spec.cpp
 * @brief Implementation of specifications, instantiation, and expression pricing.
 */

#include "origami/graphs/spec.hpp"

#include <stdexcept>
#include <utility>

namespace origami::graphs {

// ─── hardware bindings ────────────────────────────────────────────────

param_map_t hardware_params(const origami::comm::hardware_t& hardware) {
  param_map_t out;

  const double clock_hz = hardware.clock_ghz * 1e9;

  out.set("num_cus", static_cast<index_t>(hardware.num_cu));
  out.set("num_xcds", static_cast<index_t>(hardware.num_xcd));
  out.set("cu_per_xcd", static_cast<index_t>(hardware.cu_per_xcd));

  // Both spellings are published on purpose. Origami states rates per cycle,
  // which is the unit that survives a clock change; the per-second forms are
  // the convenience, and they bake in the nominal clock. A model that cares
  // about DVFS should prefer the per-cycle names and convert once, knowingly.
  out.set("clock_ghz", hardware.clock_ghz);
  out.set("clock_hz", clock_hz);

  out.set("valu_rate", hardware.valu_rate);
  out.set("vmem_issue_rate", hardware.vmem_issue_rate);

  // Two flops per lane-element, the fused multiply-add. This is the *vector*
  // peak: hardware_t carries no matrix-core rate, so a GEMM model that means
  // the MFMA peak must rebind this name rather than inherit it.
  out.set("peak_flops_per_cu", 2.0 * hardware.valu_rate * clock_hz);
  // The same VALU peak, per cycle: unlike the MFMA peak, this one needs no
  // instruction-shape knowledge to derive, so it is published rather than
  // left for every caller to hand-inline as `2.0 * valu_rate`.
  out.set("peak_flops_per_cu_per_cycle", 2.0 * hardware.valu_rate);

  out.set("tcp_bw", hardware.tcp_bw);
  out.set("tcp_bytes_per_second_per_cu", hardware.tcp_bw * clock_hz);
  out.set("l2_bw_per_cu", hardware.l2_bw_per_cu);
  out.set("l2_bytes_per_second_per_cu", hardware.l2_bw_per_cu * clock_hz);
  out.set("mall_bw", hardware.mall_bw);
  out.set("mall_bytes_per_second", hardware.mall_bw * clock_hz);

  out.set("hbm_read_bw", hardware.hbm_read_bw);
  out.set("hbm_write_bw", hardware.hbm_write_bw);
  out.set("hbm_read_bytes_per_second", hardware.hbm_read_bw * clock_hz);
  out.set("hbm_write_bytes_per_second", hardware.hbm_write_bw * clock_hz);
  // The unqualified name means read bandwidth, which is what a streaming model
  // is limited by.
  out.set("hbm_bytes_per_second", hardware.hbm_read_bw * clock_hz);

  out.set("xgmi_latency_cycles", hardware.xgmi_latency_cycles);
  out.set("mshr_depth_per_wave", static_cast<index_t>(hardware.mshr_depth_per_wave));
  out.set("waves_per_wg", static_cast<index_t>(hardware.waves_per_wg));

  out.set("tcp_capacity_bytes", static_cast<index_t>(hardware.tcp_capacity_bytes));
  out.set("l2_capacity_bytes", static_cast<index_t>(hardware.l2_capacity_bytes));
  out.set("mall_capacity_bytes", static_cast<index_t>(hardware.mall_capacity_bytes));
  out.set("hbm_capacity_bytes", static_cast<index_t>(hardware.hbm_capacity_bytes));

  return out;
}

// ─── graph_spec_t ─────────────────────────────────────────────────────

graph_spec_t::graph_spec_t(std::vector<operation_t> operations)
    : operations_(std::move(operations)) {}

graph_spec_t& graph_spec_t::add(operation_t operation) {
  operations_.push_back(std::move(operation));
  return *this;
}

std::shared_ptr<graph_t> graph_spec_t::instantiate(const problem_t& problem,
                                                   const config_t& config,
                                                   std::string name) const {
  if (operations_.empty()) {
    throw std::invalid_argument("graph_spec_t::instantiate: the specification has no operations");
  }
  return std::make_shared<graph_t>(operations_, eval_context_t{problem, config}, std::move(name));
}

bool attach_expr_cost(const std::shared_ptr<graph_t>& graph, param_map_t hardware) {
  if (!graph) throw std::invalid_argument("attach_expr_cost: the graph is null");
  if (!expr_cost_t::priced(*graph)) return false;

  graph->set_cost(std::make_shared<expr_cost_t>(*graph, std::move(hardware)));
  return true;
}

// ─── expr_cost_t ──────────────────────────────────────────────────────

bool expr_cost_t::priced(const graph_t& graph) {
  for (const operation_t& op : graph.operations()) {
    if (op.node_cost) return true;
  }
  return false;
}

expr_cost_t::expr_cost_t(const graph_t& graph, param_map_t hardware) : graph_(&graph) {
  ctx_.problem  = graph.problem();
  ctx_.config   = graph.config();
  ctx_.hardware = std::move(hardware);
}

const cost_expr_t& expr_cost_t::expression_for(const wg_node_t& node) const {
  const auto cached = cache_.find(node);
  if (cached != cache_.end()) return cached->second;

  const auto op = static_cast<std::size_t>(node.op);
  if (op >= graph_->operations().size()) {
    throw std::out_of_range("expr_cost_t: node names operation " + std::to_string(node.op) +
                            ", which this graph does not have");
  }

  const operation_t& operation = graph_->operations()[op];
  if (!operation.node_cost) {
    throw std::invalid_argument("expr_cost_t: operation '" + operation.name +
                                "' carries no node_cost, so its nodes cannot be priced; give "
                                "every operation a cost or price the graph another way");
  }
  return cache_.emplace(node, operation.node_cost(node)).first->second;
}

double expr_cost_t::evaluate(const wg_node_t& node, int active_cus) const {
  const cost_expr_t& expr = expression_for(node);

  // The runtime scope is the reason cost is deferred rather than a number: the
  // same node is worth a different amount depending on what else is resident.
  cost_context_t ctx = ctx_;
  ctx.runtime.set("active_cus", static_cast<index_t>(active_cus));
  return expr.eval(ctx);
}

double expr_cost_t::node_cycles(const wg_node_t& node) const {
  // No contention level to report means "as if alone".
  return evaluate(node, 1);
}

double expr_cost_t::node_cycles_at(const wg_node_t& node, int active_cus) const {
  return evaluate(node, active_cus);
}

double expr_cost_t::edge_cycles(const edge_t& edge) const {
  (void)edge;
  return 0.0;
}

}  // namespace origami::graphs
