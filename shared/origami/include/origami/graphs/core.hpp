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
 * @brief origami::graphs — the four-object model and edge derivation.
 *
 * Four objects describe a pipeline, and the fifth thing — the dependency graph —
 * is *derived* rather than declared:
 *
 *   allocation_t     a logical buffer: a name and a possibly symbolic shape.
 *                    No data, only an index space.
 *   access_pattern_t one workgroup-indexed read or write onto an allocation.
 *   operation_t      a kernel-shaped unit of work: a symbolic workgroup count,
 *                    an iteration count, and the accesses it performs.
 *   graph_t          an ordered list of operations plus the bindings for their
 *                    symbolic dimensions.
 *
 * Constructing a graph_t resolves every operation's grid and then walks it: an
 * edge `(op_i, wg_p, it_p) -> (op_j, wg_c, it_c)` exists exactly when `op_i`
 * precedes `op_j` in the operator order and the indices the producer *writes* to
 * a shared allocation intersect the indices the consumer *reads* from it. The
 * edge's weight is the size of that intersection, which a cost model reads as
 * the volume travelling over the hop.
 *
 * Nothing here knows what a kernel computes. Only its access patterns and its
 * position in the operator order matter, which is what lets the same machinery
 * describe a GEMM, a collective, and a hand-written gather.
 *
 * Three properties of the model are easy to misread, so they are stated plainly:
 *
 *   - **Arbitrarily many operations, not just producer/consumer pairs.** The
 *     derivation considers every ordered pair `i < j`, not adjacent pairs only,
 *     so chains, fan-out, fan-in, diamonds and skip edges all fall out. A
 *     consumer that reads an early producer's buffer directly gets an edge from
 *     it even when intermediate stages sit between them.
 *   - **The operator order is a list, not a partial order.** Dependencies can
 *     only flow forwards through that list, so callers must supply a
 *     topologically sorted sequence. Independent operations are expressed by the
 *     *absence* of edges between them, not by their arrangement.
 *   - **Only Write-then-Read produces an edge.** Read-after-read is not a
 *     dependency. Write-after-write and write-after-read are ordering hazards
 *     that the operator sequence must respect, but they yield no fine-grained
 *     dataflow edge — so two operations writing the same allocation and feeding
 *     one reader are unordered with respect to each other. This mirrors the
 *     reference implementation; see the tests pinning the behaviour.
 */
#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "origami/graphs/cost_expr.hpp"
#include "origami/graphs/symbolic.hpp"
#include "origami/graphs/types.hpp"
#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

/**
 * @brief Explicit-edge graph backend (full definition in free_graph.hpp).
 *
 * Forward declared so graph_t can offer to_free() without including a header
 * that has no other reason to be part of the authoring model.
 */
class free_graph_t;

// ─── allocation_t ─────────────────────────────────────────────────────

/**
 * @brief A logical buffer: a name and a possibly symbolic shape, with no data.
 *
 * Shape entries may be literals or `scalar_expr_t` problem dimensions, and are
 * what an operator's workgroup count and access patterns are written against.
 *
 * There is deliberately no element type here. Everything the derivation
 * computes is measured in *elements* — an edge weight is a count of shared
 * indices — and converting that to bytes is the cost layer's job, where the
 * hardware is also known. Keeping dtype out of the IR is what lets this header
 * stay independent of the HIP-linked `origami::data_type_t`.
 */
class allocation_t {
 public:
  /**
   * @brief Declare a buffer.
   *
   * @param name Identifier; allocations are matched across operations by name.
   * @param shape Dimensions, each a literal or symbolic expression.
   */
  allocation_t(std::string name, std::vector<scalar_expr_t> shape);

  /** @brief The buffer's identifier. */
  const std::string& name() const { return name_; }

  /** @brief The declared, possibly symbolic, dimensions. */
  const std::vector<scalar_expr_t>& shape() const { return shape_; }

  /**
   * @brief Resolve every dimension against concrete bindings.
   *
   * @param env Dimension bindings.
   * @return std::vector<index_t> The concrete shape.
   */
  std::vector<index_t> resolved_shape(const eval_context_t& ctx) const;

  /**
   * @brief Total element count once dimensions are bound.
   *
   * @param env Dimension bindings.
   * @return index_t Product of the resolved dimensions; zero for a rank-0 shape.
   */
  index_t resolved_size(const eval_context_t& ctx) const;

  /**
   * @brief Render the declared shape for diagnostics.
   *
   * @return std::string A rendering such as "[M, 128]".
   */
  std::string shape_str() const;

 private:
  std::string name_;
  std::vector<scalar_expr_t> shape_;
};

// ─── access_pattern_t ─────────────────────────────────────────────────

/**
 * @brief One workgroup-indexed access onto an allocation.
 *
 * `indices` maps `(wg, it, env)` to the set of flattened allocation indices that
 * workgroup touches on that iteration. Use the `read()` / `write()` factories
 * rather than building this directly; they accept simpler callables too.
 */
struct access_pattern_t {
  allocation_t allocation;  ///< buffer this access targets
  role_t role;              ///< whether the access reads or writes
  index_fn_t indices;       ///< canonical (wg, it, env) index map
  std::string name;         ///< optional label, for diagnostics only
};

namespace detail {
template <typename>
inline constexpr bool always_false_v = false;
}  // namespace detail

/**
 * @brief Adapt any supported index-map arity to the canonical three-argument form.
 *
 * The reference implementation inspects a Python callable's signature at runtime
 * to accept `(wg)`, `(wg, env)` or `(wg, it, env)`. C++ has no equivalent, so
 * the same convenience is provided at compile time: whichever of the three forms
 * a caller supplies, this wraps it into an `index_fn_t`. Iteration-agnostic
 * patterns therefore need not mention `it` at all.
 *
 * @tparam F Callable type.
 * @param fn Index map in any supported arity.
 * @return index_fn_t The canonical index map.
 */
template <typename F>
index_fn_t make_index_fn(F fn) {
  if constexpr (std::is_invocable_r_v<index_set_t, F&, int, int, const eval_context_t&>) {
    return index_fn_t{std::move(fn)};
  } else if constexpr (std::is_invocable_r_v<index_set_t, F&, int, const eval_context_t&>) {
    return [fn = std::move(fn)](int wg, int, const eval_context_t& ctx) -> index_set_t {
      return fn(wg, ctx);
    };
  } else if constexpr (std::is_invocable_r_v<index_set_t, F&, int>) {
    return
        [fn = std::move(fn)](int wg, int, const eval_context_t&) -> index_set_t { return fn(wg); };
  } else {
    static_assert(detail::always_false_v<F>,
                  "index map must be callable as (wg), (wg, ctx) or (wg, it, ctx) "
                  "and return an index_set_t");
  }
}

/**
 * @brief Declare that an operation reads an allocation.
 *
 * @tparam F Index-map callable type, in any supported arity.
 * @param allocation Buffer being read.
 * @param indices Which indices each workgroup-iteration reads.
 * @param name Optional label for diagnostics.
 * @return access_pattern_t The read access.
 */
template <typename F>
access_pattern_t read(allocation_t allocation, F indices, std::string name = "") {
  return access_pattern_t{
      std::move(allocation), role_t::read, make_index_fn(std::move(indices)), std::move(name)};
}

/**
 * @brief Declare that an operation writes an allocation.
 *
 * @tparam F Index-map callable type, in any supported arity.
 * @param allocation Buffer being written.
 * @param indices Which indices each workgroup-iteration writes.
 * @param name Optional label for diagnostics.
 * @return access_pattern_t The write access.
 */
template <typename F>
access_pattern_t write(allocation_t allocation, F indices, std::string name = "") {
  return access_pattern_t{
      std::move(allocation), role_t::write, make_index_fn(std::move(indices)), std::move(name)};
}

// ─── operation_t ──────────────────────────────────────────────────────

/**
 * @brief A kernel-shaped unit of work.
 *
 * `num_workgroups` is symbolic — typically `grid_2d(M, N, BM, BN)` — and
 * resolves once the enclosing graph binds problem dimensions.
 *
 * `num_iters` is how many internal iterations each workgroup is pipelined into.
 * When greater than one, access patterns may use their `it` argument to expose
 * the sub-tile touched on each iteration, which is what makes *partial*
 * dependencies expressible: a consumer iteration then depends only on the
 * producer iterations that wrote what it reads, rather than on the producer
 * workgroup as a whole.
 */
struct operation_t {
  std::string name;                               ///< identifier, unique within a graph
  scalar_expr_t num_workgroups;                   ///< launch grid size, possibly symbolic
  std::vector<access_pattern_t> access_patterns;  ///< every buffer this operation touches
  scalar_expr_t num_iters = index_t{1};           ///< internal pipeline depth, at least 1

  /**
   * @brief Deferred price of one node of this operation, in cycles for one
   *        workgroup-iteration.
   *
   * Every node derived from the operation inherits this one function, and the
   * function receives the node, so boundary workgroups can be charged for the
   * tile they actually own rather than a full one. It returns an expression
   * rather than a number because hardware is not known here — a specification
   * is device-independent, and the expression is evaluated at dispatch, when
   * both the machine and the live occupancy are known.
   *
   * The expression should evaluate to cycles, matching `cost_model_t`.
   * `hardware_params` publishes both per-cycle hardware-scope symbols and a
   * handful of per-second convenience ones (`peak_flops_per_cu`,
   * `hbm_bytes_per_second`, and others); dividing by one of the per-second
   * names lands the expression back in seconds with no `clock_hz` in sight to
   * warn you, so prefer the per-cycle name and see that function's doc for
   * the full rule.
   *
   * Left empty when the graph is priced some other way, or not at all.
   */
  std::function<cost_expr_t(const wg_node_t& node)> node_cost;

  /**
   * @brief Declare an operation with an empty access list.
   *
   * @param name Identifier.
   * @param num_workgroups Launch grid size, literal or symbolic.
   */
  operation_t(std::string name, scalar_expr_t num_workgroups);

  /**
   * @brief Resolve the workgroup count.
   *
   * @param env Dimension bindings.
   * @return index_t Concrete workgroup count.
   * @throws std::invalid_argument If the count resolves negative.
   */
  index_t resolved_num_workgroups(const eval_context_t& ctx) const;

  /**
   * @brief Resolve the internal iteration count.
   *
   * @param env Dimension bindings.
   * @return index_t Concrete iteration count.
   * @throws std::invalid_argument If the count resolves below one.
   */
  index_t resolved_num_iters(const eval_context_t& ctx) const;

  /**
   * @brief Every access matching an allocation name and role.
   *
   * @param allocation_name Buffer name.
   * @param role Read or write.
   * @return std::vector<const access_pattern_t*> Matching accesses, in declaration order.
   */
  std::vector<const access_pattern_t*> patterns(const std::string& allocation_name,
                                                role_t role) const;

  /**
   * @brief The indices one workgroup-iteration touches on an allocation.
   *
   * Returns nothing when this operation has no access of that role on the
   * allocation, which is distinct from touching no indices. A single matching
   * pattern is returned untouched so the intersection stays O(1); several are
   * unioned, which materialises.
   *
   * @param allocation_name Buffer name.
   * @param role Read or write.
   * @param wg Workgroup ordinal.
   * @param it Iteration ordinal.
   * @param ctx Problem and config bindings.
   * @return std::optional<index_set_t> The indices, or nothing if no such access.
   */
  std::optional<index_set_t> resolve(const std::string& allocation_name,
                                     role_t role,
                                     int wg,
                                     int it,
                                     const eval_context_t& ctx) const;
};

// ─── graph_t ──────────────────────────────────────────────────────────

/**
 * @brief An operation sequence plus the workgroup graph derived from it.
 *
 * Construction resolves each operation's grid from `dims` and then derives every
 * edge, so a graph is immutable and complete once built. Adjacency is indexed at
 * the same time: the reference implementation rescans the whole edge list on
 * each `successors()` call, which is quadratic over a scheduling pass, so the
 * maps are built once here instead.
 */
class graph_t : public wg_graph_t {
 public:
  /**
   * @brief Derive the workgroup graph for an operation sequence.
   *
   * The graph keeps the problem and config it was built from, so a cost
   * expression can be evaluated against them later, when hardware arrives at
   * ranking. An `env_t` converts implicitly and binds the problem namespace.
   *
   * @param operations Operations in topological order; dependencies flow forwards only.
   * @param ctx Bindings for every symbolic name the operations reference.
   * @param name Optional label, surfaced in ranking results.
   * @throws std::invalid_argument On a duplicate operation name, an allocation
   *         declared with conflicting shapes, or a grid that resolves invalid.
   * @throws std::out_of_range If a referenced dimension is unbound.
   */
  explicit graph_t(std::vector<operation_t> operations,
                   eval_context_t ctx = {},
                   std::string name   = "");

  /** @brief The operations, in the order supplied. */
  const std::vector<operation_t>& operations() const { return operations_; }

  /** @brief The bindings this graph was resolved against. */
  const eval_context_t& context() const { return ctx_; }

  /** @brief The problem this graph was instantiated for. */
  const problem_t& problem() const { return ctx_.problem; }

  /** @brief The configuration this graph was instantiated for. */
  const config_t& config() const { return ctx_.config; }

  /** @brief The dimension bindings used to resolve this graph, both namespaces flattened. */
  const env_t& dims() const { return dims_; }

  /**
   * @brief Snapshot this derived graph as an explicit one.
   *
   * The result has the same grids and the same edges, but no access patterns and
   * no symbolic dimensions — it is the derivation's output frozen. Beyond being
   * useful for serialisation and for handing a derived topology to code that
   * only accepts explicit edges, this doubles as the acceptance gate for the
   * backend split: a snapshot must schedule identically to its origin.
   *
   * @return free_graph_t The equivalent explicit graph.
   */
  free_graph_t to_free() const;

  const std::string& name() const override { return name_; }

  const std::vector<edge_t>& edges() const override { return edges_; }

  int num_operations() const override { return static_cast<int>(operations_.size()); }

  index_t wg_count(int op) const override { return wg_count_.at(static_cast<std::size_t>(op)); }

  index_t iter_count(int op) const override { return iter_count_.at(static_cast<std::size_t>(op)); }

  const std::string& op_name(int op) const override;

  int op_index(const std::string& name) const override;

  const std::vector<std::size_t>& successor_edges(const wg_node_t& node) const override;

  const std::vector<std::size_t>& predecessor_edges(const wg_node_t& node) const override;

  /**
   * @brief Operations, grids and edge count, including the symbolic grid expressions.
   *
   * @return std::string Multi-line summary.
   */
  std::string summary() const override;

 private:
  void collect_allocations();
  void resolve_grids();
  void derive_edges();
  void index_adjacency();
  void edges_on(int producer, int consumer, const allocation_t& allocation);

  std::vector<operation_t> operations_;
  eval_context_t ctx_;
  env_t dims_;  // ctx_ flattened, kept for the dims() accessor and the summary
  std::string name_;

  std::unordered_map<std::string, int> op_index_;
  std::vector<allocation_t> allocations_;  // first-declaration order
  std::vector<index_t> wg_count_;
  std::vector<index_t> iter_count_;
  std::vector<edge_t> edges_;

  wg_node_map_t<std::vector<std::size_t>> successors_;
  wg_node_map_t<std::vector<std::size_t>> predecessors_;
};

}  // namespace origami::graphs
