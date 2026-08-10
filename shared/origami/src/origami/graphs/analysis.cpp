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

#include "origami/graphs/analysis.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace origami::graphs {

namespace {

/** Unit node cost and free edges, matching the reference defaults. */
double cost_of(const node_cost_fn_t& fn, const wg_node_t& n) { return fn ? fn(n) : 1.0; }
double cost_of(const edge_cost_fn_t& fn, const edge_t& e) { return fn ? fn(e) : 0.0; }

}  // namespace

std::vector<wg_node_t> topological_order(const wg_graph_t& graph) {
  const std::vector<wg_node_t> all = graph.nodes();
  const std::vector<edge_t>& edges = graph.edges();

  wg_node_map_t<int> indegree;
  indegree.reserve(all.size());
  for (const wg_node_t& n : all) indegree.emplace(n, 0);
  for (const edge_t& e : edges) ++indegree[e.dst];

  // Seeded in canonical node order, then consumed from the back. The LIFO
  // discipline is deliberate: it does not change any makespan, but it decides
  // which of several equally long chains gets reported.
  std::vector<wg_node_t> ready;
  for (const wg_node_t& n : all) {
    if (indegree[n] == 0) ready.push_back(n);
  }

  std::vector<wg_node_t> order;
  order.reserve(all.size());
  while (!ready.empty()) {
    const wg_node_t n = ready.back();
    ready.pop_back();
    order.push_back(n);

    for (std::size_t ei : graph.successor_edges(n)) {
      const wg_node_t& dst = edges[ei].dst;
      if (--indegree[dst] == 0) ready.push_back(dst);
    }
  }

  if (order.size() != all.size()) {
    throw std::invalid_argument("cycle detected in workgroup graph (malformed input)");
  }
  return order;
}

bool is_acyclic(const wg_graph_t& graph) {
  try {
    topological_order(graph);
    return true;
  } catch (const std::invalid_argument&) { return false; }
}

critical_path_t critical_path(const wg_graph_t& graph,
                              const node_cost_fn_t& node_cost,
                              const edge_cost_fn_t& edge_cost) {
  const std::vector<wg_node_t> order = topological_order(graph);
  const std::vector<edge_t>& edges   = graph.edges();

  critical_path_t out;
  out.finish.reserve(order.size());
  wg_node_map_t<wg_node_t> came_from;
  wg_node_map_t<bool> has_pred;
  came_from.reserve(order.size());
  has_pred.reserve(order.size());

  for (const wg_node_t& n : order) {
    double best_start = 0.0;
    wg_node_t best_pred{};
    bool found = false;

    // Strictly greater, so the earliest predecessor in edge order wins a tie.
    for (std::size_t ei : graph.predecessor_edges(n)) {
      const edge_t& e      = edges[ei];
      const double arrival = out.finish.at(e.src) + cost_of(edge_cost, e);
      if (arrival > best_start) {
        best_start = arrival;
        best_pred  = e.src;
        found      = true;
      }
    }

    out.finish[n] = best_start + cost_of(node_cost, n);
    came_from[n]  = best_pred;
    has_pred[n]   = found;
  }

  if (out.finish.empty()) return out;

  // The reference takes max() over a dict whose insertion order is the
  // topological order, and Python's max returns the first maximal key. Walking
  // `order` with a strict comparison reproduces that tie-break; iterating the
  // hash map instead would pick an arbitrary winner among equals.
  wg_node_t sink   = order.front();
  double best_time = out.finish.at(sink);
  for (const wg_node_t& n : order) {
    const double f = out.finish.at(n);
    if (f > best_time) {
      best_time = f;
      sink      = n;
    }
  }

  wg_node_t cur = sink;
  while (true) {
    out.path.push_back(cur);
    if (!has_pred.at(cur)) break;
    cur = came_from.at(cur);
  }
  std::reverse(out.path.begin(), out.path.end());

  out.makespan = best_time;
  return out;
}

std::string overlap_report(const wg_graph_t& graph,
                           double serial_cost,
                           const node_cost_fn_t& node_cost,
                           const edge_cost_fn_t& edge_cost) {
  const critical_path_t cp = critical_path(graph, node_cost, edge_cost);
  const double saved       = serial_cost - cp.makespan;
  const double pct         = serial_cost != 0.0 ? 100.0 * saved / serial_cost : 0.0;

  std::string chain;
  for (std::size_t i = 0; i < cp.path.size(); ++i) {
    if (i > 0) chain += " -> ";
    chain += graph.label(cp.path[i]);
  }

  std::ostringstream os;
  os << std::fixed << std::setprecision(1);
  os << "serial (no overlap): " << serial_cost << "\n";
  os << "critical path      : " << cp.makespan << "\n";
  os << "overlap saving     : " << saved << " (" << std::setprecision(0) << pct << "%)\n";
  os << "critical chain     : " << chain;
  return os.str();
}

}  // namespace origami::graphs
