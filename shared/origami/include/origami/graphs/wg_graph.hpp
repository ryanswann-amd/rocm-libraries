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
 * @brief origami::graphs — the workgroup graph vocabulary and backend interface.
 *
 * Everything downstream of the graph — topological order, critical path, all six
 * runtimes, the trace emitter — needs only four things: how many operations
 * there are, how large each one's grid is, what the edges are, and which edges
 * touch a given node. It does not need to know where those edges came from.
 *
 * That is what `wg_graph_t` captures. Two backends implement it today:
 *
 *   - `graph_t` (core.hpp) derives edges by intersecting symbolic access
 *     patterns. This works for strided and affine tile access, which covers
 *     essentially every GEMM and collective kernel.
 *   - `free_graph_t` (free_graph.hpp) states edges outright. Irregular
 *     gather/scatter, measured traces and adversarial test topologies have no
 *     clean access-pattern form, so there is nothing for a derivation to
 *     intersect; the edges are simply known.
 *
 * Trace import will be a third. Splitting the interface out is what stops the
 * scheduling layer from growing a second implementation per backend.
 *
 * The virtual calls here sit outside the expensive part of the work: derivation
 * is quadratic in workgroup pairs, whereas scheduling touches each node and edge
 * a constant number of times, so an indirect call per node is not measurable
 * against it. That is the opposite of the situation in symbolic.hpp, where
 * `index_set_t` is a variant precisely to avoid an indirect call.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "origami/graphs/types.hpp"

namespace origami::graphs {

class cost_model_t;

// ─── wg_node_t ────────────────────────────────────────────────────────

/**
 * @brief The schedulable atom: one iteration of one workgroup of one operation.
 *
 * `op` is the operation's index in its graph, not a string. Every runtime uses
 * nodes as map keys in hot loops, and interning makes hashing and equality three
 * integer compares. Because the interned id *is* the position in the operator
 * list, it doubles as the `op_order` that schedulers tie-break on, so the two
 * can never disagree.
 */
struct wg_node_t {
  int op = 0;  ///< operation index within the graph, also its order
  int wg = 0;  ///< workgroup ordinal
  int it = 0;  ///< internal iteration ordinal, zero for monolithic operations

  friend bool operator==(const wg_node_t& a, const wg_node_t& b) {
    return a.op == b.op && a.wg == b.wg && a.it == b.it;
  }
  friend bool operator!=(const wg_node_t& a, const wg_node_t& b) { return !(a == b); }

  /** @brief Lexicographic by (op, wg, it) — the canonical node order. */
  friend bool operator<(const wg_node_t& a, const wg_node_t& b) {
    if (a.op != b.op) return a.op < b.op;
    if (a.wg != b.wg) return a.wg < b.wg;
    return a.it < b.it;
  }
};

/**
 * @brief Hash functor for wg_node_t.
 *
 * Provided as a named type rather than relying solely on the `std::hash`
 * specialization at the foot of this header, because containers declared inside
 * this file would otherwise depend on a specialization declared after them.
 */
struct wg_node_hash_t {
  std::size_t operator()(const wg_node_t& n) const noexcept {
    // Three small integers packed into one word: op and it are bounded by the
    // operator count and pipeline depth, so collisions only arise for grids
    // beyond 2^32 workgroups.
    std::size_t h = static_cast<std::size_t>(static_cast<std::uint32_t>(n.wg));
    h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(n.op)) << 32;
    h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(n.it)) * 0x9E3779B97F4A7C15ULL;
    return h;
  }
};

/** @brief Convenience alias for a hash map keyed by workgroup-iteration node. */
template <typename T>
using wg_node_map_t = std::unordered_map<wg_node_t, T, wg_node_hash_t>;

// ─── edge_t ───────────────────────────────────────────────────────────

/**
 * @brief A producer-to-consumer dataflow edge.
 *
 * `weight` is the number of elements the producer wrote that the consumer reads,
 * which a cost model prices as the volume moved over this hop.
 */
struct edge_t {
  wg_node_t src;           ///< producing workgroup-iteration
  wg_node_t dst;           ///< consuming workgroup-iteration
  std::string allocation;  ///< buffer the dependency flows through
  index_t weight = 0;      ///< shared element count
};

// ─── wg_graph_t ───────────────────────────────────────────────────────

/**
 * @brief What a scheduler needs from a workgroup graph, independent of origin.
 *
 * Implementations are immutable once handed to a consumer: the analysis and
 * runtime layers assume grids and edges do not move under them.
 */
class wg_graph_t {
 public:
  virtual ~wg_graph_t() = default;

  // ─── cost ───────────────────────────────────────────────────────────

  /**
   * @brief Attach the cost model that prices this graph's nodes.
   *
   * Cost lives on the graph rather than beside it because the two travel
   * together: a candidate's tile shapes and its transfer sizes are properties of
   * the same configuration, and ranking several candidates at once means several
   * costs at once. `rank_graphs` cannot take one cost model for many graphs and
   * still be correct, which is what put this here.
   *
   * This is the one thing about a graph that may change after construction. The
   * structure is still immutable — attaching a price does not move an edge.
   *
   * @param cost Model to attach; shared so a caller need not outlive the graph.
   */
  void set_cost(std::shared_ptr<const cost_model_t> cost) { cost_ = std::move(cost); }

  /** @brief Attached cost model, or null when none has been set. */
  const cost_model_t* cost() const { return cost_.get(); }

  /** @brief Attached cost model as a shared pointer, for passing it on. */
  const std::shared_ptr<const cost_model_t>& cost_ptr() const { return cost_; }

  /** @brief Number of operations. */
  virtual int num_operations() const = 0;

  /**
   * @brief Name of an operation by index.
   *
   * @param op Operation index.
   * @return const std::string& The operation's name.
   */
  virtual const std::string& op_name(int op) const = 0;

  /**
   * @brief Index of an operation by name.
   *
   * @param name Operation name.
   * @return int The operation's index.
   * @throws std::out_of_range If no operation carries that name.
   */
  virtual int op_index(const std::string& name) const = 0;

  /**
   * @brief Workgroup count for an operation.
   *
   * @param op Operation index.
   * @return index_t Workgroup count.
   */
  virtual index_t wg_count(int op) const = 0;

  /**
   * @brief Internal iteration count for an operation.
   *
   * @param op Operation index.
   * @return index_t Iteration count, at least one.
   */
  virtual index_t iter_count(int op) const = 0;

  /** @brief Every edge in the graph. */
  virtual const std::vector<edge_t>& edges() const = 0;

  /**
   * @brief Indices of the edges leaving a node.
   *
   * @param node Producing node.
   * @return const std::vector<std::size_t>& Edge indices into edges().
   */
  virtual const std::vector<std::size_t>& successor_edges(const wg_node_t& node) const = 0;

  /**
   * @brief Indices of the edges arriving at a node.
   *
   * @param node Consuming node.
   * @return const std::vector<std::size_t>& Edge indices into edges().
   */
  virtual const std::vector<std::size_t>& predecessor_edges(const wg_node_t& node) const = 0;

  /** @brief Optional graph label, surfaced in ranking results. */
  virtual const std::string& name() const = 0;

  // ─── derived from the above, identical for every backend ────────────

  /**
   * @brief Every workgroup-iteration node, in (operation, workgroup, iteration) order.
   *
   * @return std::vector<wg_node_t> All nodes.
   */
  std::vector<wg_node_t> nodes() const;

  /** @brief Total node count, without materialising the node list. */
  std::size_t num_nodes() const;

  /**
   * @brief Render a node the way the reference implementation prints it.
   *
   * @param node Node to label.
   * @return std::string For example "gemm#12" or "gemm#12.3" when iterated.
   */
  std::string label(const wg_node_t& node) const;

  /**
   * @brief Human-readable description: operations, grids and edge count.
   *
   * @return std::string Multi-line summary.
   */
  virtual std::string summary() const;

 protected:
  /** @brief The empty edge-index list returned for nodes with no neighbours. */
  static const std::vector<std::size_t>& no_edges();

 private:
  std::shared_ptr<const cost_model_t> cost_;
};

}  // namespace origami::graphs

namespace std {

/** @brief Hash for wg_node_t, so nodes key a plain unordered container. */
template <>
struct hash<::origami::graphs::wg_node_t> : ::origami::graphs::wg_node_hash_t {};

}  // namespace std
