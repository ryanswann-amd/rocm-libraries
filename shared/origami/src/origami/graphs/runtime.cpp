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

#include "origami/graphs/runtime.hpp"

#include <algorithm>
#include <vector>

namespace origami::graphs {

namespace {

/** Position of each node in canonical (operation, workgroup, iteration) order. */
wg_node_map_t<std::size_t> canonical_rank(const wg_graph_t& graph) {
  wg_node_map_t<std::size_t> rank;
  const std::vector<wg_node_t> nodes = graph.nodes();
  rank.reserve(nodes.size());
  std::size_t i = 0;
  for (const wg_node_t& n : nodes) rank.emplace(n, i++);
  return rank;
}

}  // namespace

// ─── breadth-first ────────────────────────────────────────────────────

wg_node_map_t<std::size_t> breadth_first_runtime_t::priority(const wg_graph_t& graph) const {
  return canonical_rank(graph);
}

bool breadth_first_runtime_t::gated(const wg_node_t& node, const sim_state_t& state) const {
  // Hold anything that is not part of the earliest operation still unfinished.
  for (int op = 0; op < node.op; ++op) {
    const std::size_t i = static_cast<std::size_t>(op);
    if (state.retired_by_op[i] < state.total_by_op[i]) return true;
  }
  return false;
}

// ─── asap ─────────────────────────────────────────────────────────────

wg_node_map_t<std::size_t> asap_runtime_t::priority(const wg_graph_t& graph) const {
  return canonical_rank(graph);
}

// ─── depth-first ──────────────────────────────────────────────────────

// This rank is topological: a node is pushed onto the DFS stack only once
// every one of its predecessors has already been ranked (its indegree has
// been driven to 0 by the loop below), so popping and ranking it is always
// legal. A plain DFS that pushes a successor the moment any single
// predecessor reaches it is not topological in general -- with two
// predecessors reaching a shared successor at different depths, it can rank
// that successor before the predecessor still waiting its turn -- and
// depth_first_runtime_t dispatches in rank order (queue_order_t::rank_first),
// which requires a real topological rank or simulate() rejects it outright.
wg_node_map_t<std::size_t> depth_first_rank(const wg_graph_t& graph) {
  const std::vector<wg_node_t> nodes = graph.nodes();
  const std::vector<edge_t>& edges   = graph.edges();

  wg_node_map_t<std::vector<wg_node_t>> succ;
  wg_node_map_t<std::size_t> indegree;
  succ.reserve(nodes.size());
  indegree.reserve(nodes.size());
  for (const wg_node_t& n : nodes) {
    succ.emplace(n, std::vector<wg_node_t>{});
    indegree.emplace(n, 0);
  }
  for (const edge_t& e : edges) {
    succ[e.src].push_back(e.dst);
    ++indegree[e.dst];
  }
  for (auto& [node, list] : succ) {
    (void)node;
    std::sort(list.begin(), list.end());
  }

  wg_node_map_t<std::size_t> rank;
  rank.reserve(nodes.size());

  // nodes() is already canonically ordered, so the sources come out sorted.
  // Reversed onto the stack so the first source is the first one visited.
  std::vector<wg_node_t> stack;
  for (const wg_node_t& n : nodes) {
    if (indegree[n] == 0) stack.push_back(n);
  }
  std::reverse(stack.begin(), stack.end());

  while (!stack.empty()) {
    const wg_node_t n = stack.back();
    stack.pop_back();
    rank.emplace(n, rank.size());

    const std::vector<wg_node_t>& next = succ.at(n);
    for (auto it = next.rbegin(); it != next.rend(); ++it) {
      if (--indegree[*it] == 0) stack.push_back(*it);
    }
  }

  // A cycle (only possible in a hand-built free_graph_t; derivation cannot
  // produce one) leaves every node on it with indegree > 0 forever, so it
  // never reaches the stack above. Give those trailing ranks so the map stays
  // total; rank-first dispatch will fail validation before scheduling, with
  // simulate() reporting the non-topological rank rather than the cycle itself.
  for (const wg_node_t& n : nodes) {
    if (!rank.count(n)) rank.emplace(n, rank.size());
  }
  return rank;
}

// ─── xcd ──────────────────────────────────────────────────────────────

xcd_runtime_t::xcd_runtime_t(int num_xcds, int cus_per_xcd) : placement_(num_xcds, cus_per_xcd) {}

wg_node_map_t<std::size_t> xcd_runtime_t::priority(const wg_graph_t& graph) const {
  return canonical_rank(graph);
}

}  // namespace origami::graphs
