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
 * @brief origami::graphs — the explicit-edge graph backend.
 *
 * The derivation in core.hpp produces edges by intersecting index sets, which
 * requires the access pattern to have a closed form. Plenty of real work does
 * not: an irregular gather, a sparse operator whose structure depends on the
 * data, a topology recovered from a measured trace, or a deliberately awkward
 * graph written to test a scheduler. For those there is nothing to intersect —
 * the edges are simply known — and `free_graph_t` states them outright.
 *
 * It implements the same `wg_graph_t` interface as the derived graph, so every
 * analysis, every runtime and the trace emitter accept it without knowing the
 * difference. Nothing downstream is duplicated for it.
 *
 * `graph_t::to_free()` snapshots a derived graph into one of these. That is
 * partly a convenience, and partly the acceptance gate for the whole split: if a
 * snapshot ever schedules differently from the graph it came from, the two
 * backends have diverged somewhere they should not have.
 *
 * Unlike `graph_t`, this type is mutable while being built — operations and
 * edges are added incrementally — and is expected to be left alone once it is
 * handed to a scheduler.
 */
#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "origami/graphs/types.hpp"
#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

/**
 * @brief One operation's grid, for declaring a free graph up front.
 */
struct op_grid_t {
  std::string name;            ///< operation identifier
  index_t num_workgroups = 0;  ///< workgroups launched
  index_t num_iters      = 1;  ///< internal iterations per workgroup
};

/**
 * @brief A node addressed by operation *name*, for authoring convenience.
 *
 * Callers building a graph by hand think in names; the graph stores interned
 * indices. This is the adapter, and it is what makes `{"fetch", 3}` work as an
 * edge endpoint.
 */
struct node_ref_t {
  std::string op;  ///< operation name
  int wg = 0;      ///< workgroup ordinal
  int it = 0;      ///< internal iteration ordinal
};

/**
 * @brief A workgroup graph whose edges are stated rather than derived.
 */
class free_graph_t : public wg_graph_t {
 public:
  /**
   * @brief Construct an empty graph.
   *
   * @param name Optional label, surfaced in ranking results.
   */
  explicit free_graph_t(std::string name = "");

  /**
   * @brief Declare an operation and its grid.
   *
   * @param name Operation identifier, unique within the graph.
   * @param num_workgroups Workgroups launched.
   * @param num_iters Internal iterations per workgroup; at least one.
   * @return int The operation's index, which is also its order.
   * @throws std::invalid_argument On a duplicate name or a non-positive grid.
   */
  int add_op(std::string name, index_t num_workgroups, index_t num_iters = 1);

  /**
   * @brief State a dependency between two nodes, addressed by operation name.
   *
   * @param src Producing node.
   * @param dst Consuming node.
   * @param allocation Buffer the dependency flows through; free-form here.
   * @param weight Elements moved over the hop.
   * @throws std::out_of_range If either operation is undeclared.
   * @throws std::invalid_argument If either ordinal is outside its grid, or the
   *         edge does not advance in (operation, workgroup, iteration) order.
   */
  void add_edge(const node_ref_t& src,
                const node_ref_t& dst,
                std::string allocation = "",
                index_t weight         = 1);

  /**
   * @brief State a dependency between two nodes, addressed by operation index.
   *
   * @param src Producing node.
   * @param dst Consuming node.
   * @param allocation Buffer the dependency flows through; free-form here.
   * @param weight Elements moved over the hop.
   * @throws std::invalid_argument If either node is outside its grid, or the
   *         edge does not advance in (operation, workgroup, iteration) order.
   *
   * Edges must strictly advance in the canonical node order, which keeps the
   * graph acyclic by construction. That admits dependencies within a single
   * operation — expressible here, but not by any access pattern.
   */
  void add_edge(const wg_node_t& src,
                const wg_node_t& dst,
                std::string allocation = "",
                index_t weight         = 1);

  /**
   * @brief Build a graph of declared grids with no edges yet.
   *
   * The starting point for a topology that will be filled in edge by edge, and
   * on its own the maximally parallel graph over those grids.
   *
   * @param ops Operation grids, in topological order.
   * @param name Optional label.
   * @return free_graph_t The graph.
   */
  static free_graph_t from_grid(std::vector<op_grid_t> ops, std::string name = "");

  /**
   * @brief Build a graph from declared grids and a complete edge list.
   *
   * @param ops Operation grids, in topological order.
   * @param edges Every dependency, referencing operations by index.
   * @param name Optional label.
   * @return free_graph_t The graph.
   * @throws std::invalid_argument If any edge is out of range or runs backwards.
   */
  static free_graph_t from_edges(std::vector<op_grid_t> ops,
                                 const std::vector<edge_t>& edges,
                                 std::string name = "");

  const std::string& name() const override { return name_; }

  const std::vector<edge_t>& edges() const override { return edges_; }

  int num_operations() const override { return static_cast<int>(ops_.size()); }

  index_t wg_count(int op) const override {
    return ops_.at(static_cast<std::size_t>(op)).num_workgroups;
  }

  index_t iter_count(int op) const override {
    return ops_.at(static_cast<std::size_t>(op)).num_iters;
  }

  const std::string& op_name(int op) const override;

  int op_index(const std::string& name) const override;

  const std::vector<std::size_t>& successor_edges(const wg_node_t& node) const override;

  const std::vector<std::size_t>& predecessor_edges(const wg_node_t& node) const override;

 private:
  std::string label_of(const wg_node_t& node) const;
  void validate_node(const wg_node_t& node, const char* which) const;

  std::string name_;
  std::vector<op_grid_t> ops_;
  std::unordered_map<std::string, int> op_index_;
  std::vector<edge_t> edges_;

  wg_node_map_t<std::vector<std::size_t>> successors_;
  wg_node_map_t<std::vector<std::size_t>> predecessors_;
};

}  // namespace origami::graphs
