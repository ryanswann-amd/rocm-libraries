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
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace origami::graphs {

namespace {

/** Both knobs are counts of real things, so neither can sensibly be below one. */
void validate(const std::optional<int>& lanes, index_t wg_duration, const char* who) {
  if (lanes && *lanes < 1) {
    throw std::invalid_argument(std::string(who) + ": lanes must be at least 1 (got " +
                                std::to_string(*lanes) + "); leave it unset for unlimited");
  }
  if (wg_duration < 1) {
    throw std::invalid_argument(std::string(who) + ": wg_duration must be at least 1 (got " +
                                std::to_string(wg_duration) + ")");
  }
}

/**
 * Greedy list scheduler over the dataflow DAG.
 *
 * At each timestep the ready nodes are taken in `rank` order, up to `lanes` of
 * them. The rank is the entire difference between one pipelined policy and
 * another.
 *
 * The reference rebuilds the ready set every timestep by re-testing every
 * unscheduled node against all its predecessors. This walks the edges once
 * instead, which is the same schedule by a shorter route: every node dispatched
 * at `t` finishes at `t + wg_duration`, so a node whose predecessors were all
 * dispatched in earlier timesteps is exactly a node that is ready now. Counting
 * predecessors down to zero therefore recognises the same set the rescan would
 * have found, and both take it in rank order.
 */
schedule_t list_schedule(const wg_graph_t& graph,
                         std::string name,
                         const std::optional<int>& lanes,
                         index_t wg_duration,
                         const wg_node_map_t<std::size_t>& rank) {
  schedule_t out;
  out.runtime     = std::move(name);
  out.wg_duration = wg_duration;
  out.lanes       = lanes;

  const std::vector<wg_node_t> nodes = graph.nodes();
  const std::vector<edge_t>& edges   = graph.edges();
  if (nodes.empty()) return out;

  out.start.reserve(nodes.size());
  out.order.reserve(nodes.size());

  // Ranks are dense and unique, so they double as an index into the node list.
  std::vector<wg_node_t> node_of_rank(rank.size());
  for (const auto& [node, r] : rank) node_of_rank[r] = node;

  wg_node_map_t<std::size_t> unmet;
  unmet.reserve(nodes.size());
  for (const wg_node_t& n : nodes) unmet.emplace(n, 0);
  for (const edge_t& e : edges) ++unmet[e.dst];

  // Ordered by rank, which is what "take the highest-priority ready nodes"
  // needs, and cheap to add to and erase from as the wave front moves.
  std::set<std::size_t> ready;
  for (const wg_node_t& n : nodes) {
    if (unmet[n] == 0) ready.insert(rank.at(n));
  }

  const std::size_t width =
      lanes ? static_cast<std::size_t>(*lanes) : std::numeric_limits<std::size_t>::max();

  index_t t                 = 0;
  std::size_t num_scheduled = 0;
  std::vector<wg_node_t> taken;
  while (num_scheduled < nodes.size()) {
    taken.clear();
    for (auto it = ready.begin(); it != ready.end() && taken.size() < width;) {
      taken.push_back(node_of_rank[*it]);
      it = ready.erase(it);
    }

    for (const wg_node_t& n : taken) {
      out.start[n] = t;
      out.order.push_back(n);
      ++num_scheduled;
    }

    // Deferred to the end of the timestep on purpose: a node dispatched at `t`
    // is not finished until `t + wg_duration`, so its consumers become ready in
    // the next round, not this one.
    for (const wg_node_t& n : taken) {
      for (std::size_t ei : graph.successor_edges(n)) {
        const wg_node_t& dst = edges[ei].dst;
        if (--unmet[dst] == 0) ready.insert(rank.at(dst));
      }
    }

    if (taken.empty() && ready.empty() && num_scheduled < nodes.size()) {
      throw std::invalid_argument("cycle detected in workgroup graph (malformed input): " +
                                  std::to_string(nodes.size() - num_scheduled) +
                                  " node(s) never became ready");
    }
    t += wg_duration;
  }
  return out;
}

/** Canonical (operation, workgroup, iteration) order, as a dense rank. */
wg_node_map_t<std::size_t> canonical_rank(const wg_graph_t& graph) {
  const std::vector<wg_node_t> nodes = graph.nodes();
  wg_node_map_t<std::size_t> rank;
  rank.reserve(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) rank.emplace(nodes[i], i);
  return rank;
}

}  // namespace

// ─── schedule_t ───────────────────────────────────────────────────────

index_t schedule_t::finish(const wg_node_t& node) const { return start.at(node) + wg_duration; }

index_t schedule_t::makespan() const {
  index_t out = 0;
  for (const auto& [node, s] : start) {
    (void)node;
    out = std::max(out, s + wg_duration);
  }
  return out;
}

std::vector<timestep_t> schedule_t::timesteps() const {
  // Grouped by walking dispatch order, so that nodes tying on the sort key below
  // keep the order the runtime chose rather than a hash order.
  std::map<index_t, std::vector<wg_node_t>> groups;
  for (const wg_node_t& n : order) groups[start.at(n)].push_back(n);

  std::vector<timestep_t> out;
  out.reserve(groups.size());
  for (auto& [t, group] : groups) {
    // Iteration is deliberately not part of the key, matching the reference.
    std::stable_sort(group.begin(), group.end(), [](const wg_node_t& a, const wg_node_t& b) {
      if (a.op != b.op) return a.op < b.op;
      return a.wg < b.wg;
    });
    out.push_back(timestep_t{t, std::move(group)});
  }
  return out;
}

std::string schedule_t::summary() const {
  std::set<index_t> occupied;
  for (const auto& [node, s] : start) {
    (void)node;
    occupied.insert(s);
  }
  const std::string lanes_str = lanes ? std::to_string(*lanes) : "\u221e";
  return "runtime '" + runtime + "': " + std::to_string(start.size()) + " workgroups across " +
         std::to_string(occupied.size()) + " timesteps, makespan=" + std::to_string(makespan()) +
         "  (lanes=" + lanes_str + ", wg_duration=" + std::to_string(wg_duration) + ")";
}

// ─── breadth-first ────────────────────────────────────────────────────

breadth_first_runtime_t::breadth_first_runtime_t(std::optional<int> lanes, index_t wg_duration)
    : lanes_(lanes), wg_duration_(wg_duration) {
  validate(lanes_, wg_duration_, "breadth_first_runtime_t");
}

schedule_t breadth_first_runtime_t::schedule(const wg_graph_t& graph) const {
  schedule_t out;
  out.runtime     = name_;
  out.wg_duration = wg_duration_;
  out.lanes       = lanes_;

  index_t t = 0;
  for (int op = 0; op < graph.num_operations(); ++op) {
    std::vector<wg_node_t> subnodes;
    const int wgs   = static_cast<int>(graph.wg_count(op));
    const int iters = static_cast<int>(graph.iter_count(op));
    subnodes.reserve(static_cast<std::size_t>(wgs) * static_cast<std::size_t>(iters));
    for (int wg = 0; wg < wgs; ++wg) {
      for (int it = 0; it < iters; ++it) subnodes.push_back(wg_node_t{op, wg, it});
    }

    const std::size_t step =
        lanes_ ? static_cast<std::size_t>(*lanes_) : std::max<std::size_t>(subnodes.size(), 1);
    for (std::size_t i = 0; i < subnodes.size(); i += step) {
      for (std::size_t j = i; j < std::min(i + step, subnodes.size()); ++j) {
        out.start[subnodes[j]] = t;
        out.order.push_back(subnodes[j]);
      }
      t += wg_duration_;
    }
  }
  return out;
}

// ─── asap ─────────────────────────────────────────────────────────────

asap_runtime_t::asap_runtime_t(std::optional<int> lanes, index_t wg_duration)
    : lanes_(lanes), wg_duration_(wg_duration) {
  validate(lanes_, wg_duration_, "asap_runtime_t");
}

schedule_t asap_runtime_t::schedule(const wg_graph_t& graph) const {
  return list_schedule(graph, name_, lanes_, wg_duration_, canonical_rank(graph));
}

// ─── depth-first ──────────────────────────────────────────────────────

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
    if (rank.count(n)) continue;
    rank.emplace(n, rank.size());

    const std::vector<wg_node_t>& next = succ.at(n);
    for (auto it = next.rbegin(); it != next.rend(); ++it) {
      if (!rank.count(*it)) stack.push_back(*it);
    }
  }

  // A free graph can state edges that leave a node unreachable from every
  // source; derivation cannot. Give those trailing ranks so they still schedule.
  for (const wg_node_t& n : nodes) {
    if (!rank.count(n)) rank.emplace(n, rank.size());
  }
  return rank;
}

depth_first_runtime_t::depth_first_runtime_t(std::optional<int> lanes, index_t wg_duration)
    : lanes_(lanes), wg_duration_(wg_duration) {
  validate(lanes_, wg_duration_, "depth_first_runtime_t");
}

schedule_t depth_first_runtime_t::schedule(const wg_graph_t& graph) const {
  return list_schedule(graph, name_, lanes_, wg_duration_, depth_first_rank(graph));
}

}  // namespace origami::graphs
