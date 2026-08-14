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

#include "origami/graphs/simulate.hpp"

#include <algorithm>
#include <cstdio>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

// Only placement.hpp is needed for placement_t's full definition (required to
// call choose()). runtime.hpp is deliberately not included here: this file
// defines runtime_t, schedule_t and simulate() themselves, so runtime.hpp
// (which declares the four concrete policies over these types) would only add
// a dependency this file does not need, not resolve one.
#include "origami/graphs/placement.hpp"

namespace origami::graphs {

namespace {

/** Format a double the way Python's format spec does, for summary parity. */
std::string fixed(double v, int precision) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", precision, v);
  return std::string(buf);
}

/**
 * One entry in the dispatch queue.
 *
 * `seq` keeps the order total, so a schedule never depends on how ties fall out
 * of the heap. Comparison is reversed because std::priority_queue pops the
 * largest.
 */
struct pending_t {
  double ready     = 0.0;
  std::size_t rank = 0;
  wg_node_t node;
  std::size_t seq     = 0;
  bool rank_dominates = false;

  bool operator<(const pending_t& other) const {
    if (rank_dominates) {
      if (rank != other.rank) return rank > other.rank;
      if (ready != other.ready) return ready > other.ready;
    } else {
      if (ready != other.ready) return ready > other.ready;
      if (rank != other.rank) return rank > other.rank;
    }
    return seq > other.seq;
  }
};

}  // namespace

schedule_t simulate(const wg_graph_t& graph,
                    const runtime_t& runtime,
                    const cost_model_t& cost,
                    simulate_options_t options) {
  if (options.lanes && *options.lanes < 1) {
    throw std::invalid_argument("simulate: lanes must be at least 1 (got " +
                                std::to_string(*options.lanes) + "); leave it unset for unlimited");
  }

  schedule_t out;
  out.runtime = runtime.name();
  out.lanes   = options.lanes;

  const std::vector<wg_node_t> nodes = graph.nodes();
  const std::vector<edge_t>& edges   = graph.edges();
  if (nodes.empty()) return out;

  const wg_node_map_t<std::size_t> rank = runtime.priority(graph);

  // Under rank_first, a rank that runs backwards over some edge would dispatch
  // a node before its producer; check that once, up front, rather than
  // discovering it mid-schedule.
  const bool rank_dominates = runtime.order() == queue_order_t::rank_first;
  if (rank_dominates) {
    for (const edge_t& e : edges) {
      if (rank.at(e.src) >= rank.at(e.dst)) {
        throw std::invalid_argument(
            "runtime '" + runtime.name() + "' dispatches in rank order, but edge " +
            graph.label(e.src) + " -> " + graph.label(e.dst) +
            " runs backwards in that order, so the launch order is not topological");
      }
    }
  }

  wg_node_map_t<std::size_t> unmet;
  unmet.reserve(nodes.size());
  for (const wg_node_t& n : nodes) unmet.emplace(n, 0);
  for (const edge_t& e : edges) ++unmet[e.dst];

  // A workgroup occupies one compute unit for its whole lifetime, so its
  // iterations run in sequence on that unit. The graph cannot express this —
  // it is a property of the hardware, not of the dataflow — so it is added here
  // as an implicit precedence with lane affinity.
  wg_node_map_t<wg_node_t> next_iter;
  if (options.serialize_wg_iters) {
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
      options.lanes ? *options.lanes : static_cast<int>(std::max<std::size_t>(nodes.size(), 1));
  std::vector<double> lane_free(static_cast<std::size_t>(lane_count), 0.0);

  std::priority_queue<pending_t> queue;
  std::size_t seq = 0;
  for (const wg_node_t& n : nodes) {
    if (unmet[n] == 0) queue.push(pending_t{ready[n], rank.at(n), n, seq++, rank_dominates});
  }

  wg_node_map_t<int> pinned;
  out.start.reserve(nodes.size());
  out.duration.reserve(nodes.size());
  out.lane.reserve(nodes.size());
  out.order.reserve(nodes.size());

  // Per-operation progress, for sim_state_t. "Retired" means already placed
  // into the schedule and, in simulated time, actually finished by the point
  // the scheduler has reached; "in flight" is placed but not yet finished.
  const int num_ops = graph.num_operations();
  std::vector<int> retired_by_op(static_cast<std::size_t>(num_ops), 0);
  std::vector<int> in_flight_by_op(static_cast<std::size_t>(num_ops), 0);
  std::vector<int> total_by_op(static_cast<std::size_t>(num_ops), 0);
  for (const wg_node_t& n : nodes) ++total_by_op[static_cast<std::size_t>(n.op)];
  const sim_state_t state{retired_by_op, in_flight_by_op, total_by_op};

  // Nodes placed but not yet retired, kept with their finish time so a later
  // dispatch can tell whether simulated time has passed them.
  std::vector<std::pair<double, int>> in_flight;  // (finish, op)
  // Nodes withheld by runtime.gated(); replayed against the queue whenever a
  // retirement changes the state a gate might be watching.
  std::vector<pending_t> held;

  // Returns true when calling advance() released one or more held nodes back
  // into `queue`. The caller uses this to tell whether the node it just
  // popped now has fresh competition it has not yet been compared against.
  auto advance = [&](double now) {
    bool retired_any = false;
    for (std::size_t i = 0; i < in_flight.size();) {
      if (in_flight[i].first <= now) {
        const int op = in_flight[i].second;
        --in_flight_by_op[static_cast<std::size_t>(op)];
        ++retired_by_op[static_cast<std::size_t>(op)];
        in_flight[i] = in_flight.back();
        in_flight.pop_back();
        retired_any = true;
      } else {
        ++i;
      }
    }
    if (retired_any && !held.empty()) {
      // A held node keeps the `ready` time it had when the gate first caught
      // it, which can predate the retirement that just released it. Clamp to
      // `now` so a replayed node cannot dispatch before its release.
      for (pending_t p : held) {
        p.ready = std::max(p.ready, now);
        queue.push(p);
      }
      held.clear();
      return true;
    }
    return false;
  };

  std::size_t launch_index = 0;
  while (!queue.empty() || !held.empty()) {
    if (queue.empty()) {
      // Every remaining node is held by the gate, and nothing will replay it
      // on its own: advance() only replays on a retirement, and a retirement
      // only happens inside the pop loop below, which has nothing left to
      // pop. So time has to be pushed forward here instead, to the earliest
      // finish among nodes still in flight, which is the next moment a gate
      // could possibly see something new.
      if (in_flight.empty()) {
        throw std::invalid_argument(
            "runtime '" + runtime.name() +
            "' gate deadlocked: every remaining node is held and nothing is in "
            "flight to advance past, so the gate can never release them "
            "(not a cycle in the graph)");
      }
      double earliest = in_flight.front().first;
      for (const auto& [finish, op] : in_flight) {
        (void)op;
        earliest = std::min(earliest, finish);
      }
      const std::size_t held_before = held.size();
      advance(earliest);
      if (held.size() >= held_before) {
        throw std::invalid_argument("runtime '" + runtime.name() +
                                    "' gate deadlocked: replaying held node(s) made no progress "
                                    "(not a cycle in the graph)");
      }
      continue;
    }

    const pending_t top = queue.top();
    queue.pop();
    const wg_node_t n = top.node;

    if (advance(top.ready)) {
      // advance() just released held nodes into `queue` at or after `now`.
      // `top` was popped before that happened, so it has not yet been
      // compared against them; push it back and let the queue's ordering
      // decide who goes first. This cannot spin: `top` carries the same
      // `ready`/`rank` it had before, so replaying it retires nothing new on
      // the next `advance(top.ready)` call (everything with finish <= now
      // already retired on this call), `held` is now empty, and the queue
      // shrank by zero and grew by `held`'s prior size, so this path runs at
      // most once per retirement event.
      queue.push(top);
      continue;
    }
    if (runtime.gated(n, state)) {
      held.push_back(top);
      continue;
    }

    const std::string& op_name = graph.op_name(n.op);
    const bool skip = !options.skip_prefix.empty() && op_name.rfind(options.skip_prefix, 0) == 0;

    int chosen = -1;
    if (skip) {
      out.place(n, top.ready, 0.0, 0);
      in_flight.emplace_back(top.ready, n.op);
      ++in_flight_by_op[static_cast<std::size_t>(n.op)];
    } else {
      const auto pin = pinned.find(n);
      if (pin != pinned.end()) {
        chosen = pin->second;
      } else {
        const placement_context_t ctx{n, launch_index, lane_free};
        chosen = runtime.placement().choose(ctx);
      }
      if (chosen < 0 || chosen >= lane_count) {
        throw std::invalid_argument("placement '" + runtime.placement().name() +
                                    "' returned lane " + std::to_string(chosen) +
                                    " for a machine of " + std::to_string(lane_count) + " lanes");
      }

      const double s = std::max(top.ready, lane_free[static_cast<std::size_t>(chosen)]);

      double d = 0.0;
      if (options.reprice_on_dispatch) {
        int busy = 0;
        for (double free_at : lane_free) {
          if (free_at > s) ++busy;
        }
        d = cost.node_cycles_at(n, busy + 1);  // +1 counts this node
      } else {
        d = cost.node_cycles(n);
      }

      const double f = s + d;
      out.place(n, s, d, chosen);
      lane_free[static_cast<std::size_t>(chosen)] = f;
      in_flight.emplace_back(f, n.op);
      ++in_flight_by_op[static_cast<std::size_t>(n.op)];
      ++launch_index;
    }

    // Hoisted out of the skip/non-skip split above: `unmet[n]` counted this
    // node's own successor iteration regardless of whether it was skipped, so
    // that iteration's unblocking cannot stay conditional on skip without
    // wedging a skip-prefixed operation's later iterations forever. A skipped
    // node never took a lane, so its successor iteration is not pinned to one.
    const double f = out.finish(n);
    const auto nxt = next_iter.find(n);
    if (nxt != next_iter.end()) {
      const wg_node_t& c = nxt->second;
      ready[c]           = std::max(ready[c], f);
      if (!skip) pinned[c] = chosen;  // same compute unit for the workgroup's lifetime
      if (--unmet[c] == 0) queue.push(pending_t{ready[c], rank.at(c), c, seq++, rank_dominates});
    }

    for (std::size_t ei : graph.successor_edges(n)) {
      const edge_t& e      = edges[ei];
      const wg_node_t& dst = e.dst;
      ready[dst]           = std::max(ready[dst], f + cost.edge_cycles(e));
      if (--unmet[dst] == 0) {
        queue.push(pending_t{ready[dst], rank.at(dst), dst, seq++, rank_dominates});
      }
    }
  }

  if (out.order.size() != nodes.size()) {
    throw std::invalid_argument("cycle detected in workgroup graph (malformed input): " +
                                std::to_string(nodes.size() - out.order.size()) +
                                " node(s) never became ready");
  }
  return out;
}

// ─── schedule_t ───────────────────────────────────────────────────────
//
// A schedule is parallel node-keyed maps plus `order`, the dispatch sequence.
// Every query below walks `order` rather than the maps, so results — the
// floating-point sums especially — never depend on hash layout. The interface
// itself is documented in the header.

void schedule_t::place(const wg_node_t& node, double start_time, double dur, int lane_idx) {
  // Re-placing a node overwrites its times without appending to `order` twice.
  if (this->start.find(node) == this->start.end()) order.push_back(node);
  this->start[node]    = start_time;
  this->duration[node] = dur;
  this->lane[node]     = lane_idx;
}

double schedule_t::finish(const wg_node_t& node) const {
  return start.at(node) + duration.at(node);
}

double schedule_t::makespan() const {
  double out = 0.0;
  for (const wg_node_t& n : order) out = std::max(out, finish(n));
  return out;
}

int schedule_t::num_lanes() const {
  int highest = -1;
  for (const wg_node_t& n : order) highest = std::max(highest, lane.at(n));
  return highest + 1;
}

std::vector<double> schedule_t::busy_by_op() const {
  int highest = -1;
  for (const wg_node_t& n : order) highest = std::max(highest, n.op);

  std::vector<double> out(static_cast<std::size_t>(highest + 1), 0.0);
  // Summed in dispatch order rather than hash order, so the floating-point
  // result does not depend on how the map happens to be laid out.
  for (const wg_node_t& n : order) out[static_cast<std::size_t>(n.op)] += duration.at(n);
  return out;
}

double schedule_t::utilization() const {
  double total = 0.0;
  for (const wg_node_t& n : order) total += duration.at(n);
  const double capacity = makespan() * std::max(num_lanes(), 1);
  return capacity != 0.0 ? total / capacity : 0.0;
}

std::string schedule_t::summary() const {
  const std::string lanes_str = lanes ? std::to_string(*lanes) : "\u221e";
  return "runtime '" + runtime + "': " + std::to_string(order.size()) + " atoms on " +
         std::to_string(num_lanes()) + " lanes, makespan=" + fixed(makespan(), 2) +
         " cycles, utilisation=" + fixed(utilization() * 100.0, 0) + "%  (lanes=" + lanes_str + ")";
}

// ─── clock ──────────────────────────────────────────────────────────────

fixed_clock_t::fixed_clock_t(double ghz) : ghz_(ghz) {
  if (!(ghz > 0.0)) {
    throw std::invalid_argument("fixed_clock_t: frequency must be positive, so cycles can be "
                                "converted to seconds");
  }
}

timed_schedule_t to_seconds(const schedule_t& schedule,
                            const clock_t& clock,
                            double scale,
                            const std::string& units) {
  if (!(scale > 0.0)) {
    throw std::invalid_argument("to_seconds: scale must be positive, so a schedule cannot be "
                                "silently negated or collapsed to zero");
  }
  const double per_cycle = scale / (clock.ghz() * 1e9);
  timed_schedule_t out;
  out.runtime = schedule.runtime;
  out.units   = units;
  out.lane    = schedule.lane;
  out.order   = schedule.order;
  for (const wg_node_t& n : schedule.order) {
    out.start[n]    = schedule.start.at(n) * per_cycle;
    out.duration[n] = schedule.duration.at(n) * per_cycle;
  }
  return out;
}

double timed_schedule_t::finish(const wg_node_t& node) const {
  return start.at(node) + duration.at(node);
}

double timed_schedule_t::makespan() const {
  double out = 0.0;
  for (const wg_node_t& n : order) out = std::max(out, finish(n));
  return out;
}

std::string timed_schedule_t::summary() const {
  return "runtime '" + runtime + "': " + std::to_string(order.size()) +
         " atoms, makespan=" + fixed(makespan(), 2) + units;
}

}  // namespace origami::graphs
