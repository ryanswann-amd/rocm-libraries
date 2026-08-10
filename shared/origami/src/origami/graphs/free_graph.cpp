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

#include "origami/graphs/free_graph.hpp"

#include <stdexcept>
#include <utility>

namespace origami::graphs {

free_graph_t::free_graph_t(std::string name) : name_(std::move(name)) {}

int free_graph_t::add_op(std::string name, index_t num_workgroups, index_t num_iters) {
  if (num_workgroups < 0) {
    throw std::invalid_argument("operation '" + name + "' has a negative workgroup count " +
                                std::to_string(num_workgroups));
  }
  if (num_iters < 1) {
    throw std::invalid_argument("operation '" + name + "' has num_iters " +
                                std::to_string(num_iters) + "; at least 1 is required");
  }

  const int index = static_cast<int>(ops_.size());
  if (!op_index_.emplace(name, index).second) {
    throw std::invalid_argument("duplicate operation name '" + name + "'");
  }
  ops_.push_back(op_grid_t{std::move(name), num_workgroups, num_iters});
  return index;
}

std::string free_graph_t::label_of(const wg_node_t& node) const {
  return ops_[static_cast<std::size_t>(node.op)].name + "#" + std::to_string(node.wg) + "." +
         std::to_string(node.it);
}

void free_graph_t::validate_node(const wg_node_t& node, const char* which) const {
  if (node.op < 0 || node.op >= num_operations()) {
    throw std::invalid_argument(std::string{which} + " references operation index " +
                                std::to_string(node.op) + ", which is not declared");
  }
  const op_grid_t& g = ops_[static_cast<std::size_t>(node.op)];
  if (node.wg < 0 || node.wg >= g.num_workgroups) {
    throw std::invalid_argument(std::string{which} + " workgroup " + std::to_string(node.wg) +
                                " is outside operation '" + g.name + "' grid of " +
                                std::to_string(g.num_workgroups));
  }
  if (node.it < 0 || node.it >= g.num_iters) {
    throw std::invalid_argument(std::string{which} + " iteration " + std::to_string(node.it) +
                                " is outside operation '" + g.name + "' depth of " +
                                std::to_string(g.num_iters));
  }
}

void free_graph_t::add_edge(const wg_node_t& src,
                            const wg_node_t& dst,
                            std::string allocation,
                            index_t weight) {
  validate_node(src, "edge source");
  validate_node(dst, "edge destination");

  // Every edge must advance in the canonical (op, wg, it) node order. Because
  // that order is total, an acyclic graph is guaranteed by construction, which
  // is what the schedulers assume and what makes a topological sort always
  // succeed. The rule is looser than the derivation's, which can only ever pair
  // a strictly earlier operation with a later one: it additionally admits
  // dependencies *within* an operation, so an irregular kernel whose workgroup 5
  // waits on its own workgroup 3 is expressible here even though no access
  // pattern could produce it.
  if (!(src < dst)) {
    throw std::invalid_argument("edge " + label_of(src) + " -> " + label_of(dst) +
                                " does not advance in node order; edges must run forwards so the"
                                " graph stays acyclic");
  }

  const std::size_t index = edges_.size();
  edges_.push_back(edge_t{src, dst, std::move(allocation), weight});
  successors_[src].push_back(index);
  predecessors_[dst].push_back(index);
}

void free_graph_t::add_edge(const node_ref_t& src,
                            const node_ref_t& dst,
                            std::string allocation,
                            index_t weight) {
  add_edge(wg_node_t{op_index(src.op), src.wg, src.it},
           wg_node_t{op_index(dst.op), dst.wg, dst.it},
           std::move(allocation),
           weight);
}

free_graph_t free_graph_t::from_grid(std::vector<op_grid_t> ops, std::string name) {
  free_graph_t g(std::move(name));
  for (op_grid_t& o : ops) g.add_op(std::move(o.name), o.num_workgroups, o.num_iters);
  return g;
}

free_graph_t free_graph_t::from_edges(std::vector<op_grid_t> ops,
                                      const std::vector<edge_t>& edges,
                                      std::string name) {
  free_graph_t g = from_grid(std::move(ops), std::move(name));
  for (const edge_t& e : edges) g.add_edge(e.src, e.dst, e.allocation, e.weight);
  return g;
}

const std::string& free_graph_t::op_name(int op) const {
  return ops_.at(static_cast<std::size_t>(op)).name;
}

int free_graph_t::op_index(const std::string& name) const {
  const auto it = op_index_.find(name);
  if (it == op_index_.end()) throw std::out_of_range("no operation named '" + name + "'");
  return it->second;
}

const std::vector<std::size_t>& free_graph_t::successor_edges(const wg_node_t& node) const {
  const auto it = successors_.find(node);
  return it == successors_.end() ? no_edges() : it->second;
}

const std::vector<std::size_t>& free_graph_t::predecessor_edges(const wg_node_t& node) const {
  const auto it = predecessors_.find(node);
  return it == predecessors_.end() ? no_edges() : it->second;
}

}  // namespace origami::graphs
