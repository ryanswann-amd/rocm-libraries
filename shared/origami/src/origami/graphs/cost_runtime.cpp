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

#include "origami/graphs/cost_runtime.hpp"

#include <algorithm>
#include <cstdio>
#include <queue>
#include <stdexcept>
#include <vector>

namespace origami::graphs {

namespace {

void validate_scale(double scale, const char* who) {
  if (!(scale > 0.0)) {
    throw std::invalid_argument(std::string(who) + ": scale must be positive");
  }
}

void validate_lanes(const std::optional<int>& lanes, const char* who) {
  if (lanes && *lanes < 1) {
    throw std::invalid_argument(std::string(who) + ": lanes must be at least 1 (got " +
                                std::to_string(*lanes) + "); leave it unset for unlimited");
  }
}

/** Format a double the way Python's format spec does, for summary parity. */
std::string fixed(double v, int precision) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", precision, v);
  return std::string(buf);
}

/**
 * One entry in the dispatch queue.
 *
 * Ordered by ready time, then by canonical node order, then by push sequence.
 * The sequence is what the reference uses to keep nodes out of the comparison
 * entirely; it also makes the order total, so the schedule does not depend on
 * how ties happen to fall out of the heap.
 */
struct pending_t {
  double ready = 0.0;
  wg_node_t node;
  std::size_t seq = 0;

  /** Reversed, because std::priority_queue pops the largest. */
  bool operator<(const pending_t& other) const {
    if (ready != other.ready) return ready > other.ready;
    if (!(node == other.node)) return other.node < node;
    return seq > other.seq;
  }
};

/**
 * The scheduler behind both lane-pool runtimes.
 *
 * `contention_aware` selects dispatch-time pricing over static pricing, which is
 * the entire difference between the roofline and event-driven policies.
 */
cost_schedule_t lane_schedule(const wg_graph_t& graph,
                              const cost_model_t& cost,
                              const cost_runtime_options_t& opts,
                              std::string name,
                              bool contention_aware) {
  cost_schedule_t out;
  out.runtime = std::move(name);
  out.lanes   = opts.lanes;
  out.units   = opts.units;

  const std::vector<wg_node_t> nodes = graph.nodes();
  const std::vector<edge_t>& edges   = graph.edges();
  if (nodes.empty()) return out;

  wg_node_map_t<std::size_t> unmet;
  unmet.reserve(nodes.size());
  for (const wg_node_t& n : nodes) unmet.emplace(n, 0);
  for (const edge_t& e : edges) ++unmet[e.dst];

  // A workgroup occupies one compute unit for its whole lifetime, so its
  // iterations run in sequence on that unit. The graph cannot express this —
  // it is a property of the hardware, not of the dataflow — so it is added here
  // as an implicit precedence with lane affinity.
  wg_node_map_t<wg_node_t> next_iter;
  if (opts.serialize_wg_iters) {
    for (const wg_node_t& n : nodes) {
      if (n.it == 0) continue;
      const wg_node_t prev{n.op, n.wg, n.it - 1};
      next_iter.emplace(prev, n);
      ++unmet[n];
    }
  }

  wg_node_map_t<double> ready;
  ready.reserve(nodes.size());
  for (const wg_node_t& n : nodes) ready.emplace(n, 0.0);

  const int lane_count =
      opts.lanes ? *opts.lanes : static_cast<int>(std::max<std::size_t>(nodes.size(), 1));
  std::vector<double> lane_free(static_cast<std::size_t>(lane_count), 0.0);

  std::vector<int> all_lanes(static_cast<std::size_t>(lane_count));
  for (int i = 0; i < lane_count; ++i) all_lanes[static_cast<std::size_t>(i)] = i;

  // Lanes an operation may use: its pool clipped to the machine, or everything
  // when it has no pool or the pool falls entirely outside the machine.
  std::vector<std::vector<int>> allowed(static_cast<std::size_t>(graph.num_operations()));
  for (int op = 0; op < graph.num_operations(); ++op) {
    const auto it = opts.lane_pool.find(graph.op_name(op));
    std::vector<int> pool;
    if (it != opts.lane_pool.end()) {
      for (int li : it->second) {
        if (li >= 0 && li < lane_count) pool.push_back(li);
      }
    }
    allowed[static_cast<std::size_t>(op)] = pool.empty() ? all_lanes : pool;
  }

  std::priority_queue<pending_t> queue;
  std::size_t seq = 0;
  for (const wg_node_t& n : nodes) {
    if (unmet[n] == 0) queue.push(pending_t{ready[n], n, seq++});
  }

  wg_node_map_t<int> pinned;
  out.start.reserve(nodes.size());
  out.duration.reserve(nodes.size());
  out.lane.reserve(nodes.size());
  out.order.reserve(nodes.size());

  while (!queue.empty()) {
    const pending_t top = queue.top();
    queue.pop();
    const wg_node_t n = top.node;

    int chosen                     = -1;
    const auto pin                 = pinned.find(n);
    const std::vector<int>& choice = allowed[static_cast<std::size_t>(n.op)];
    if (pin != pinned.end()) {
      chosen = pin->second;
    } else {
      // First strict minimum, so an unused lane is preferred in index order.
      chosen = choice.front();
      for (int li : choice) {
        if (lane_free[static_cast<std::size_t>(li)] < lane_free[static_cast<std::size_t>(chosen)]) {
          chosen = li;
        }
      }
    }

    const double s = std::max(top.ready, lane_free[static_cast<std::size_t>(chosen)]);

    double d = 0.0;
    if (contention_aware) {
      int busy = 0;
      for (double free_at : lane_free) {
        if (free_at > s) ++busy;
      }
      d = cost.node_cost_at(n, busy + 1) * opts.scale;  // +1 counts this node
    } else {
      d = cost.node_cost(n) * opts.scale;
    }

    const double f  = s + d;
    out.start[n]    = s;
    out.duration[n] = d;
    out.lane[n]     = chosen;
    out.order.push_back(n);
    lane_free[static_cast<std::size_t>(chosen)] = f;

    for (std::size_t ei : graph.successor_edges(n)) {
      const edge_t& e      = edges[ei];
      const wg_node_t& dst = e.dst;
      ready[dst]           = std::max(ready[dst], f + cost.edge_cost(e) * opts.scale);
      if (--unmet[dst] == 0) queue.push(pending_t{ready[dst], dst, seq++});
    }

    const auto nxt = next_iter.find(n);
    if (nxt != next_iter.end()) {
      const wg_node_t& c = nxt->second;
      ready[c]           = std::max(ready[c], f);
      pinned[c]          = chosen;  // same compute unit for the workgroup's lifetime
      if (--unmet[c] == 0) queue.push(pending_t{ready[c], c, seq++});
    }
  }

  if (out.order.size() != nodes.size()) {
    throw std::invalid_argument("cycle detected in workgroup graph (malformed input): " +
                                std::to_string(nodes.size() - out.order.size()) +
                                " node(s) never became ready");
  }
  return out;
}

}  // namespace

// ─── cost_schedule_t ──────────────────────────────────────────────────

void cost_schedule_t::place(const wg_node_t& node, double start_time, double dur, int slot) {
  if (this->start.find(node) == this->start.end()) order.push_back(node);
  this->start[node]    = start_time;
  this->duration[node] = dur;
  this->lane[node]     = slot;
}

double cost_schedule_t::finish(const wg_node_t& node) const {
  return start.at(node) + duration.at(node);
}

double cost_schedule_t::makespan() const {
  double out = 0.0;
  for (const wg_node_t& n : order) out = std::max(out, finish(n));
  return out;
}

int cost_schedule_t::num_lanes() const {
  int highest = -1;
  for (const wg_node_t& n : order) highest = std::max(highest, lane.at(n));
  return highest + 1;
}

std::vector<double> cost_schedule_t::busy_by_op() const {
  int highest = -1;
  for (const wg_node_t& n : order) highest = std::max(highest, n.op);

  std::vector<double> out(static_cast<std::size_t>(highest + 1), 0.0);
  // Summed in dispatch order rather than hash order, so the floating-point
  // result does not depend on how the map happens to be laid out.
  for (const wg_node_t& n : order) out[static_cast<std::size_t>(n.op)] += duration.at(n);
  return out;
}

double cost_schedule_t::utilization() const {
  double total = 0.0;
  for (const wg_node_t& n : order) total += duration.at(n);
  const double capacity = makespan() * std::max(num_lanes(), 1);
  return capacity != 0.0 ? total / capacity : 0.0;
}

std::string cost_schedule_t::summary() const {
  const std::string lanes_str = lanes ? std::to_string(*lanes) : "\u221e";
  return "runtime '" + runtime + "': " + std::to_string(order.size()) + " atoms on " +
         std::to_string(num_lanes()) + " lanes, makespan=" + fixed(makespan(), 2) + units +
         ", utilisation=" + fixed(utilization() * 100.0, 0) + "%  (lanes=" + lanes_str + ")";
}

// ─── cost resolution ──────────────────────────────────────────────────

const cost_model_t& resolve_cost(const cost_model_t* override_cost, const wg_graph_t& graph) {
  if (override_cost != nullptr) return *override_cost;
  if (const cost_model_t* own = graph.cost()) return *own;
  throw std::invalid_argument("resolve_cost: graph '" + graph.name() +
                              "' has no cost model and the runtime was not given one; call "
                              "set_cost on the graph or pass a cost to the runtime");
}

// ─── roofline ─────────────────────────────────────────────────────────

roofline_runtime_t::roofline_runtime_t(cost_runtime_options_t options)
    : opts_(std::move(options)) {
  validate_lanes(opts_.lanes, "roofline_runtime_t");
  validate_scale(opts_.scale, "roofline_runtime_t");
}

roofline_runtime_t::roofline_runtime_t(const cost_model_t& cost, cost_runtime_options_t options)
    : roofline_runtime_t(std::move(options)) {
  cost_ = &cost;
}

cost_schedule_t roofline_runtime_t::schedule(const wg_graph_t& graph) const {
  return lane_schedule(graph, resolve_cost(cost_, graph), opts_, name_,
                       /*contention_aware=*/false);
}

// ─── event-driven ─────────────────────────────────────────────────────

event_driven_runtime_t::event_driven_runtime_t(cost_runtime_options_t options)
    : opts_(std::move(options)) {
  validate_lanes(opts_.lanes, "event_driven_runtime_t");
  validate_scale(opts_.scale, "event_driven_runtime_t");
}

event_driven_runtime_t::event_driven_runtime_t(const cost_model_t& cost,
                                               cost_runtime_options_t options)
    : event_driven_runtime_t(std::move(options)) {
  cost_ = &cost;
}

cost_schedule_t event_driven_runtime_t::schedule(const wg_graph_t& graph) const {
  return lane_schedule(graph, resolve_cost(cost_, graph), opts_, name_,
                       /*contention_aware=*/true);
}

// ─── xcd ──────────────────────────────────────────────────────────────

xcd_runtime_t::xcd_runtime_t(xcd_options_t options) : opts_(std::move(options)) {
  if (opts_.num_xcds < 1 || opts_.cus_per_xcd < 1) {
    throw std::invalid_argument("xcd_runtime_t: num_xcds and cus_per_xcd must be at least 1");
  }
  validate_scale(opts_.scale, "xcd_runtime_t");
}

xcd_runtime_t::xcd_runtime_t(const cost_model_t& cost, xcd_options_t options)
    : xcd_runtime_t(std::move(options)) {
  cost_ = &cost;
}

cost_schedule_t xcd_runtime_t::schedule(const wg_graph_t& graph) const {
  const cost_model_t& cost = resolve_cost(cost_, graph);
  const int num_cus        = opts_.num_xcds * opts_.cus_per_xcd;

  cost_schedule_t out;
  out.runtime = name_;
  out.lanes   = num_cus;
  out.units   = opts_.units;

  const std::vector<wg_node_t> nodes = graph.nodes();
  const std::vector<edge_t>& edges   = graph.edges();
  if (nodes.empty()) return out;

  std::vector<double> cu_free(static_cast<std::size_t>(num_cus), 0.0);
  wg_node_map_t<double> finish;
  finish.reserve(nodes.size());
  out.start.reserve(nodes.size());
  out.duration.reserve(nodes.size());
  out.lane.reserve(nodes.size());
  out.order.reserve(nodes.size());

  int slot = 0;  // round-robin position, advanced only for placed nodes
  for (const wg_node_t& n : nodes) {
    double ready = 0.0;
    for (std::size_t ei : graph.predecessor_edges(n)) {
      const auto it = finish.find(edges[ei].src);
      if (it == finish.end()) {
        // Launch order is canonical order, which both backends guarantee is
        // topological. A backend that broke that would silently drop the
        // dependency, so say so instead.
        throw std::invalid_argument("xcd_runtime_t: edge into " + graph.label(n) + " comes from " +
                                    graph.label(edges[ei].src) +
                                    ", which launches later; launch order is not topological");
      }
      ready = std::max(ready, it->second);
    }

    const std::string& op = graph.op_name(n.op);
    if (!opts_.skip_prefix.empty() && op.rfind(opts_.skip_prefix, 0) == 0) {
      out.start[n]    = ready;
      out.duration[n] = 0.0;
      out.lane[n]     = 0;
      out.order.push_back(n);
      finish[n] = ready;
      continue;
    }

    const int die   = slot % opts_.num_xcds;
    const int first = die * opts_.cus_per_xcd;
    int best        = first;
    for (int cu = first; cu < first + opts_.cus_per_xcd; ++cu) {
      if (cu_free[static_cast<std::size_t>(cu)] < cu_free[static_cast<std::size_t>(best)]) {
        best = cu;
      }
    }

    const double d = cost.node_cost(n) * opts_.scale;
    const double s = std::max(cu_free[static_cast<std::size_t>(best)], ready);

    out.start[n]    = s;
    out.duration[n] = d;
    out.lane[n]     = best;
    out.order.push_back(n);
    cu_free[static_cast<std::size_t>(best)] = s + d;
    finish[n]                               = s + d;
    ++slot;
  }
  return out;
}

}  // namespace origami::graphs
