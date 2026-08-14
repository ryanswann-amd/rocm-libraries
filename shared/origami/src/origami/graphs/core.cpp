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

#include "origami/graphs/core.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

#include "origami/graphs/free_graph.hpp"

namespace origami::graphs {

// ─── allocation_t ─────────────────────────────────────────────────────

allocation_t::allocation_t(std::string name, std::vector<scalar_expr_t> shape)
    : name_(std::move(name)), shape_(std::move(shape)) {}

std::vector<index_t> allocation_t::resolved_shape(const eval_context_t& ctx) const {
  std::vector<index_t> out;
  out.reserve(shape_.size());
  for (const scalar_expr_t& d : shape_) out.push_back(d.eval(ctx));
  return out;
}

index_t allocation_t::resolved_size(const eval_context_t& ctx) const {
  if (shape_.empty()) return 0;
  index_t total = 1;
  for (index_t d : resolved_shape(ctx)) total *= d;
  return total;
}

std::string allocation_t::shape_str() const {
  std::string out = "[";
  for (std::size_t i = 0; i < shape_.size(); ++i) {
    if (i > 0) out += ", ";
    out += shape_[i].str();
  }
  return out + "]";
}

// ─── operation_t ──────────────────────────────────────────────────────

operation_t::operation_t(std::string name, scalar_expr_t num_workgroups)
    : name(std::move(name)), num_workgroups(std::move(num_workgroups)) {}

index_t operation_t::resolved_num_workgroups(const eval_context_t& ctx) const {
  const index_t n = num_workgroups.eval(ctx);
  if (n < 0) {
    throw std::invalid_argument("operation '" + name + "' resolved to a negative workgroup count " +
                                std::to_string(n));
  }
  return n;
}

index_t operation_t::resolved_num_iters(const eval_context_t& ctx) const {
  const index_t n = num_iters.eval(ctx);
  if (n < 1) {
    throw std::invalid_argument("operation '" + name + "' resolved to num_iters " +
                                std::to_string(n) + "; at least 1 is required");
  }
  return n;
}

std::vector<const access_pattern_t*> operation_t::patterns(const std::string& allocation_name,
                                                           role_t role) const {
  std::vector<const access_pattern_t*> out;
  for (const access_pattern_t& ap : access_patterns) {
    if (ap.role == role && ap.allocation.name() == allocation_name) out.push_back(&ap);
  }
  return out;
}

std::optional<index_set_t> operation_t::resolve(const std::string& allocation_name,
                                                role_t role,
                                                int wg,
                                                int it,
                                                const eval_context_t& ctx) const {
  const std::vector<const access_pattern_t*> pats = patterns(allocation_name, role);
  if (pats.empty()) return std::nullopt;

  // One pattern is the overwhelmingly common case; return its set untouched so
  // an analytic progression survives and the intersection stays O(1).
  if (pats.size() == 1) return pats.front()->indices(wg, it, ctx);

  std::vector<index_t> merged;
  for (const access_pattern_t* ap : pats) {
    const std::vector<index_t> v = to_vector(ap->indices(wg, it, ctx));
    merged.insert(merged.end(), v.begin(), v.end());
  }
  return index_set_t{finite_set_t::of(std::move(merged))};
}

// ─── graph_t ──────────────────────────────────────────────────────────

graph_t::graph_t(std::vector<operation_t> operations, eval_context_t ctx, std::string name)
    : operations_(std::move(operations))
    , ctx_(std::move(ctx))
    , dims_(ctx_.to_env())
    , name_(std::move(name)) {
  collect_allocations();
  resolve_grids();
  derive_edges();
  index_adjacency();
}

void graph_t::collect_allocations() {
  std::map<std::string, std::string> shape_of;  // allocation name -> rendered shape

  for (std::size_t i = 0; i < operations_.size(); ++i) {
    const operation_t& op = operations_[i];

    // The reference implementation keys its grid tables by operation name, so a
    // repeated name silently collapses two operations into one. Rejecting it is
    // a deliberate divergence: kirigami keys by index, where the same mistake
    // would instead produce edges pointing at the wrong operation.
    if (!op_index_.emplace(op.name, static_cast<int>(i)).second) {
      throw std::invalid_argument("duplicate operation name '" + op.name + "'");
    }

    for (const access_pattern_t& ap : op.access_patterns) {
      const std::string& an     = ap.allocation.name();
      const std::string shape   = ap.allocation.shape_str();
      const auto [it, inserted] = shape_of.emplace(an, shape);
      if (inserted) {
        allocations_.push_back(ap.allocation);
      } else if (it->second != shape) {
        throw std::invalid_argument("allocation '" + an + "' declared with conflicting shapes " +
                                    it->second + " vs " + shape);
      }
    }
  }
}

void graph_t::resolve_grids() {
  constexpr index_t kMaxGrid = std::numeric_limits<int>::max();

  wg_count_.reserve(operations_.size());
  iter_count_.reserve(operations_.size());

  for (const operation_t& op : operations_) {
    index_t wgs   = 0;
    index_t iters = 0;
    try {
      wgs   = op.resolved_num_workgroups(ctx_);
      iters = op.resolved_num_iters(ctx_);
    } catch (const std::out_of_range& e) {
      throw std::out_of_range("operation '" + op.name + "' has an unbound dimension; bind it in " +
                              "the graph's env_t (" + e.what() + ")");
    }
    // Node ordinals are int, so an over-large grid would silently wrap.
    if (wgs > kMaxGrid || iters > kMaxGrid) {
      throw std::invalid_argument("operation '" + op.name + "' resolves to a grid too large to " +
                                  "index (" + std::to_string(wgs) + " x " + std::to_string(iters) +
                                  ")");
    }
    wg_count_.push_back(wgs);
    iter_count_.push_back(iters);
  }
}

void graph_t::derive_edges() {
  const int n = num_operations();
  for (int producer = 0; producer < n; ++producer) {
    // Every later operation, not merely the next one, so a consumer reading an
    // early producer's buffer gets its edge even across intervening stages.
    for (int consumer = producer + 1; consumer < n; ++consumer) {
      // Walking allocations in declaration order keeps edge order reproducible.
      // The reference intersects two Python sets of names here, whose iteration
      // order varies with the hash seed whenever a pair shares more than one
      // allocation.
      for (const allocation_t& alloc : allocations_) {
        const bool written   = !operations_[producer].patterns(alloc.name(), role_t::write).empty();
        const bool been_read = !operations_[consumer].patterns(alloc.name(), role_t::read).empty();
        if (written && been_read) edges_on(producer, consumer, alloc);
      }
    }
  }
}

void graph_t::edges_on(int producer, int consumer, const allocation_t& allocation) {
  const operation_t& p = operations_[producer];
  const operation_t& c = operations_[consumer];

  struct read_set_t {
    int wg = 0;
    int it = 0;
    index_set_t indices;
  };

  // Consumer reads are resolved once and reused across every producer node.
  std::vector<read_set_t> reads;
  const int c_wgs   = static_cast<int>(wg_count_[static_cast<std::size_t>(consumer)]);
  const int c_iters = static_cast<int>(iter_count_[static_cast<std::size_t>(consumer)]);
  for (int wc = 0; wc < c_wgs; ++wc) {
    for (int itc = 0; itc < c_iters; ++itc) {
      std::optional<index_set_t> rs = c.resolve(allocation.name(), role_t::read, wc, itc, ctx_);
      if (rs.has_value()) reads.push_back({wc, itc, std::move(*rs)});
    }
  }
  if (reads.empty()) return;

  const int p_wgs   = static_cast<int>(wg_count_[static_cast<std::size_t>(producer)]);
  const int p_iters = static_cast<int>(iter_count_[static_cast<std::size_t>(producer)]);
  for (int wp = 0; wp < p_wgs; ++wp) {
    for (int itp = 0; itp < p_iters; ++itp) {
      std::optional<index_set_t> ws = p.resolve(allocation.name(), role_t::write, wp, itp, ctx_);
      if (!ws.has_value() || is_empty(*ws)) continue;

      for (const read_set_t& r : reads) {
        const index_t weight = intersect_size(*ws, r.indices);
        if (weight > 0) {
          edges_.push_back(edge_t{wg_node_t{producer, wp, itp},
                                  wg_node_t{consumer, r.wg, r.it},
                                  allocation.name(),
                                  weight});
        }
      }
    }
  }
}

void graph_t::index_adjacency() {
  for (std::size_t i = 0; i < edges_.size(); ++i) {
    successors_[edges_[i].src].push_back(i);
    predecessors_[edges_[i].dst].push_back(i);
  }
}

const std::string& graph_t::op_name(int op) const {
  return operations_.at(static_cast<std::size_t>(op)).name;
}

int graph_t::op_index(const std::string& name) const {
  const auto it = op_index_.find(name);
  if (it == op_index_.end()) throw std::out_of_range("no operation named '" + name + "'");
  return it->second;
}

const std::vector<std::size_t>& graph_t::successor_edges(const wg_node_t& node) const {
  const auto it = successors_.find(node);
  return it == successors_.end() ? no_edges() : it->second;
}

const std::vector<std::size_t>& graph_t::predecessor_edges(const wg_node_t& node) const {
  const auto it = predecessors_.find(node);
  return it == predecessors_.end() ? no_edges() : it->second;
}

free_graph_t graph_t::to_free() const {
  std::vector<op_grid_t> grids;
  grids.reserve(operations_.size());
  for (int op = 0; op < num_operations(); ++op) {
    grids.push_back(op_grid_t{op_name(op), wg_count(op), iter_count(op)});
  }
  return free_graph_t::from_edges(std::move(grids), edges_, name_);
}

std::string graph_t::summary() const {
  std::string out = "Graph: " + std::to_string(operations_.size()) + " ops, " +
                    std::to_string(num_nodes()) + " workgroup-iterations, " +
                    std::to_string(edges_.size()) + " fine edges";

  if (!dims_.empty()) {
    // Sorted, so the rendering does not depend on hash-table order.
    std::map<std::string, index_t> sorted(dims_.begin(), dims_.end());
    out += "\n  dims: {";
    bool first = true;
    for (const auto& [k, v] : sorted) {
      if (!first) out += ", ";
      out += k + ": " + std::to_string(v);
      first = false;
    }
    out += "}";
  }

  for (int op = 0; op < num_operations(); ++op) {
    const index_t wgs   = wg_count_[static_cast<std::size_t>(op)];
    const index_t iters = iter_count_[static_cast<std::size_t>(op)];
    out += "\n  op '" + op_name(op) + "': " + std::to_string(wgs) + " wgs";
    if (iters > 1) out += " x " + std::to_string(iters) + " iters";
    const scalar_expr_t& expr = operations_[static_cast<std::size_t>(op)].num_workgroups;
    if (!expr.is_constant()) out += "   [= " + expr.str() + "]";
  }
  return out;
}

}  // namespace origami::graphs
