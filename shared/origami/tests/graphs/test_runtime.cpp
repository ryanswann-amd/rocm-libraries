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
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "origami/graphs/analysis.hpp"
#include "origami/graphs/core.hpp"
#include "origami/graphs/free_graph.hpp"
#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/**
 * Render a schedule the way oracle/runtime_dump.cpp does: one group per
 * distinct start cycle, nodes within a group in canonical (op, wg) order —
 * the same grouping the old integer `schedule_t::timesteps()` produced, now
 * done by hand since a cycle-based `schedule_t` has no notion of a timestep.
 */
std::string render(const wg_graph_t& g, const schedule_t& s) {
  std::map<double, std::vector<wg_node_t>> groups;
  for (const wg_node_t& n : s.order) groups[s.start.at(n)].push_back(n);

  std::string out;
  bool first = true;
  for (auto& [t, group] : groups) {
    // Iteration is deliberately not part of the key, matching the reference.
    std::stable_sort(group.begin(), group.end(), [](const wg_node_t& a, const wg_node_t& b) {
      if (a.op != b.op) return a.op < b.op;
      return a.wg < b.wg;
    });
    if (!first) out += " | ";
    first = false;
    out += std::to_string(static_cast<long long>(t)) + ":";
    for (const wg_node_t& n : group) out += " " + g.label(n);
  }
  return out;
}

/** a(4 wgs) -> b(4 wgs), one edge per workgroup. */
graph_t one_to_one() {
  const allocation_t x("x", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4))};
  return graph_t({a, b});
}

/** Every dependency is respected: no node starts before a predecessor finishes. */
bool respects_dependencies(const wg_graph_t& g, const schedule_t& s) {
  for (const edge_t& e : g.edges()) {
    if (s.start.at(e.dst) < s.finish(e.src)) return false;
  }
  return true;
}

/**
 * Peak number of atoms simultaneously in flight, found by sweeping every
 * start and finish as an event rather than trusting the lane assignment: a
 * placement bug that put two atoms on the same lane at overlapping times
 * would still show up here even though `simulate()` never lets that
 * particular mistake happen by construction.
 */
int peak_concurrency(const schedule_t& s) {
  std::vector<std::pair<double, int>> events;
  events.reserve(s.order.size() * 2);
  for (const wg_node_t& n : s.order) {
    events.emplace_back(s.start.at(n), 1);
    events.emplace_back(s.finish(n), -1);
  }
  // A finish at the same instant as a start frees its lane first, so the two
  // intervals do not count as overlapping.
  std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) return a.first < b.first;
    return a.second < b.second;
  });

  int running = 0, peak = 0;
  for (const auto& [time, delta] : events) {
    running += delta;
    peak = std::max(peak, running);
  }
  return peak;
}

/** No more atoms run at once than the machine has lanes for. */
bool respects_lanes(const schedule_t& s) {
  if (!s.lanes) return true;
  return peak_concurrency(s) <= *s.lanes;
}

/** Fixed cost per node, no cost per edge -- the analogue of the old wg_duration knob. */
class flat_cost_t : public cost_model_t {
 public:
  explicit flat_cost_t(double cycles) : cycles_(cycles) {}
  double node_cycles(const wg_node_t&) const override { return cycles_; }
  double edge_cycles(const edge_t&) const override { return 0.0; }

 private:
  double cycles_;
};

}  // namespace

// ─── the properties every runtime must have ───────────────────────────

TEST(every_runtime_respects_dependencies_and_lanes) {
  const graph_t g = one_to_one();
  const unit_cost_t cost;

  for (int lanes = 1; lanes <= 5; ++lanes) {
    const breadth_first_runtime_t bf;
    const asap_runtime_t asap;
    const depth_first_runtime_t df;

    simulate_options_t opts;
    opts.lanes = lanes;
    for (const runtime_t* rt : {static_cast<const runtime_t*>(&bf),
                                static_cast<const runtime_t*>(&asap),
                                static_cast<const runtime_t*>(&df)}) {
      const schedule_t s = simulate(g, *rt, cost, opts);
      CHECK(s.start.size() == g.num_nodes());
      CHECK(s.order.size() == g.num_nodes());
      CHECK(respects_dependencies(g, s));
      CHECK(respects_lanes(s));
    }
  }
}

TEST(no_runtime_beats_the_critical_path) {
  // The critical path assumes unbounded hardware, so it is a lower bound on
  // anything a runtime can achieve, and unlimited lanes should reach it.
  const graph_t g    = one_to_one();
  const double bound = critical_path(g).makespan;
  const asap_runtime_t asap;
  const unit_cost_t cost;

  simulate_options_t narrow_opts;
  narrow_opts.lanes = 2;

  const schedule_t unlimited = simulate(g, asap, cost, simulate_options_t{});
  const schedule_t narrow    = simulate(g, asap, cost, narrow_opts);

  CHECK(unlimited.makespan() == bound);
  CHECK(narrow.makespan() >= bound);
}

TEST(breadth_first_is_the_no_overlap_baseline) {
  // Its makespan is the sum of each operation's waves, by construction. ASAP can
  // overlap ragged waves and should not exceed it; chain-greedy depth-first may
  // deliberately idle lanes and do worse than this baseline.
  const graph_t g = one_to_one();
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes = 3;

  const schedule_t bf   = simulate(g, breadth_first_runtime_t{}, cost, opts);
  const schedule_t asap = simulate(g, asap_runtime_t{}, cost, opts);

  CHECK(bf.makespan() == 4);  // two ragged waves per operation
  CHECK(asap.makespan() == 3);
  CHECK(asap.makespan() <= bf.makespan());
}

TEST(unlimited_lanes_make_every_policy_agree) {
  const graph_t g = one_to_one();
  const unit_cost_t cost;
  const std::string expected = "0: a#0 a#1 a#2 a#3 | 1: b#0 b#1 b#2 b#3";

  CHECK(render(g, simulate(g, breadth_first_runtime_t{}, cost, simulate_options_t{})) == expected);
  CHECK(render(g, simulate(g, asap_runtime_t{}, cost, simulate_options_t{})) == expected);
  CHECK(render(g, simulate(g, depth_first_runtime_t{}, cost, simulate_options_t{})) == expected);
}

// ─── what distinguishes the three policies ────────────────────────────

TEST(breadth_first_never_mixes_operations_within_a_wave) {
  const graph_t g = one_to_one();
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes = 2;

  const schedule_t s = simulate(g, breadth_first_runtime_t{}, cost, opts);
  CHECK(render(g, s) == "0: a#0 a#1 | 1: a#2 a#3 | 2: b#0 b#1 | 3: b#2 b#3");

  // No lane-full of simultaneous starts mixes operations.
  std::map<double, int> op_at_start;
  for (const wg_node_t& n : s.order) {
    const auto it = op_at_start.find(s.start.at(n));
    if (it == op_at_start.end()) {
      op_at_start.emplace(s.start.at(n), n.op);
    } else {
      CHECK(it->second == n.op);
    }
  }
}

TEST(asap_slips_consumers_into_a_ragged_producer_wave) {
  // With three lanes the producer's second wave uses one lane, leaving two for
  // consumers that are already unblocked. That gap is the whole overlap story.
  const graph_t g = one_to_one();
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes = 3;

  const schedule_t s = simulate(g, asap_runtime_t{}, cost, opts);
  CHECK(render(g, s) == "0: a#0 a#1 a#2 | 1: a#3 b#0 b#1 | 2: b#2 b#3");
  CHECK(s.makespan() == 3);
}

TEST(depth_first_pushes_one_tile_through_the_pipeline_first) {
  // Rank-dominant dispatch (queue_order_t::rank_first) is what lets this
  // policy live up to its name. depth_first_rank gives b#0 a lower rank than
  // a#1 -- the dive continues into a#0's consumer before returning to the next
  // sibling source -- and under rank-first that lower rank wins outright,
  // regardless of readiness: b#0, ready only once a#0 finishes, still cuts in
  // front of a#1, which was ready from cycle 0. asap has no such preference:
  // its queue is ready_first, rank only breaks a tie there, and a#1 is
  // strictly earlier-ready than b#0 can ever be, so asap drains every source
  // before touching a consumer. Same total work on one lane either way, so
  // the makespans agree; only the interleaving differs.
  const graph_t g = one_to_one();
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes = 1;

  const schedule_t s    = simulate(g, depth_first_runtime_t{}, cost, opts);
  const schedule_t asap = simulate(g, asap_runtime_t{}, cost, opts);
  CHECK(render(g, s) == "0: a#0 | 1: b#0 | 2: a#1 | 3: b#1 | 4: a#2 | 5: b#2 | 6: a#3 | 7: b#3");
  CHECK(render(g, asap) == "0: a#0 | 1: a#1 | 2: a#2 | 3: a#3 | 4: b#0 | 5: b#1 | 6: b#2 | 7: b#3");
  CHECK(s.start.at(wg_node_t{1, 0, 0}) == 1);
  CHECK(asap.start.at(wg_node_t{1, 0, 0}) == 4);
  CHECK(s.makespan() == asap.makespan());
}

// ─── depth-first ordering ─────────────────────────────────────────────

TEST(depth_first_rank_follows_the_chain_not_the_siblings) {
  const graph_t g                       = one_to_one();
  const wg_node_map_t<std::size_t> rank = depth_first_rank(g);

  CHECK(rank.size() == 8);
  // a#0 then its consumer b#0, before sibling a#1.
  CHECK(rank.at(wg_node_t{0, 0, 0}) == 0);
  CHECK(rank.at(wg_node_t{1, 0, 0}) == 1);
  CHECK(rank.at(wg_node_t{0, 1, 0}) == 2);
  CHECK(rank.at(wg_node_t{1, 1, 0}) == 3);
}

TEST(depth_first_orders_successors_by_operator_order_not_by_name) {
  // 'a' feeds 'z' (declared second) and 'b' (declared third). A reference that
  // sorts successors by name would dive into b first; ordering by operator
  // position dives into z. See the note on depth_first_rank.
  //
  // Each a#i has two successors this time, not one, but the rank-dominance
  // story is identical to depth_first_pushes_one_tile_through_the_pipeline_first:
  // depth_first_rank dives into a#0's first successor before returning to
  // a#1, so z#0 and b#0 both outrank a#1 and cut in front of it once a#0
  // retires, in operator order -- z#0 first, then b#0. Check relative
  // dispatch order rather than exact cycles, since that ordering, not the
  // timing, is what this test is pinning.
  const allocation_t x("x", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t z("z", 4);
  z.access_patterns = {read(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4))};
  const graph_t g({a, z, b});

  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes         = 1;
  const schedule_t s = simulate(g, depth_first_runtime_t{}, cost, opts);

  const auto dispatch_index = [&](const wg_node_t& n) {
    return std::find(s.order.begin(), s.order.end(), n) - s.order.begin();
  };
  CHECK(dispatch_index(wg_node_t{1, 0, 0}) < dispatch_index(wg_node_t{2, 0, 0}));  // z#0 before b#0

  // Only the order moves: the reference reports the same makespan here.
  CHECK(s.makespan() == 12);
}

TEST(depth_first_starts_at_the_first_declared_source) {
  // Two independent producers, 'z' declared before 'b'. Ordering sources by
  // name would start with b purely because of its spelling.
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});
  operation_t z("z", 4);
  z.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {write(y, contiguous(4))};
  operation_t c("c", 4);
  c.access_patterns = {read(x, contiguous(4)), read(y, contiguous(4))};
  const graph_t g({z, b, c});

  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes         = 2;
  const schedule_t s = simulate(g, depth_first_runtime_t{}, cost, opts);
  // z and b are unrelated sources, so depth_first_rank dives all the way down
  // z's chain (there is none -- c needs b too) before it can even reach b,
  // giving every z#i a lower rank than every b#i: z#0, z#1, z#2, z#3, b#0,
  // c#0, b#1, c#1, b#2, c#2, b#3, c#3. z therefore drains its two lanes'
  // worth first (cycles 0-1). Once b#0 dispatches alone at cycle 2 (both
  // lanes were still busy with z#2/z#3 at cycle 1), c#0 -- ranked right after
  // the b that unblocks it -- outranks the still-waiting b#1 for the lane
  // that frees next, so the two land in the same wave rather than b draining
  // first; the same rank-dominance repeats for every later (b#i, c#i-1) pair.
  CHECK(render(g, s) ==
        "0: z#0 z#1 | 1: z#2 z#3 | 2: b#0 | 3: b#1 c#0 | 4: b#2 c#1 | 5: b#3 c#2 | 6: c#3");

  // Both are independent sources ready at cycle 0, so z fully drains ahead of
  // b for the same reason a source always drains ahead of a not-yet-ready
  // consumer elsewhere in this file; the point being pinned is that it is z
  // that wins, not b, and not because of how the two names sort.
  const auto dispatch_index = [&](const wg_node_t& n) {
    return std::find(s.order.begin(), s.order.end(), n) - s.order.begin();
  };
  CHECK(dispatch_index(wg_node_t{0, 0, 0}) < dispatch_index(wg_node_t{1, 0, 0}));  // z#0 before b#0
}

TEST(unreachable_nodes_still_get_scheduled) {
  // A free graph can leave a node with no path from any source. Derivation
  // cannot, but the rank has to cope or the node would never be dispatched.
  free_graph_t g;
  g.add_op("a", 2);
  g.add_op("b", 2);
  g.add_edge(wg_node_t{0, 0, 0}, wg_node_t{1, 0, 0}, "x", 1);

  const wg_node_map_t<std::size_t> rank = depth_first_rank(g);
  CHECK(rank.size() == 4);

  simulate_options_t opts;
  opts.lanes         = 1;
  const schedule_t s = simulate(g, depth_first_runtime_t{}, unit_cost_t{}, opts);
  CHECK(s.start.size() == 4);
  CHECK(respects_dependencies(g, s));
}

// ─── iterated operations and both backends ────────────────────────────

TEST(iterations_are_scheduled_as_separate_atoms) {
  const allocation_t buf("buf", {index_t{32}});
  operation_t p("produce", 1);
  p.num_iters       = 4;
  p.access_patterns = {write(buf, streamed(32, 4))};
  operation_t c("consume", 1);
  c.num_iters       = 4;
  c.access_patterns = {read(buf, streamed(32, 4))};
  const graph_t g({p, c});

  // One workgroup each, but four atoms each, so a single lane takes 8 steps.
  // The graph itself does not serialise one workgroup's iterations against
  // each other (that is simulate_options_t::serialize_wg_iters, not asked for
  // here), so all four `produce` iterations are independent sources, ready at
  // cycle 0 -- but under rank-dominant dispatch that no longer means they all
  // drain first: depth_first_rank gives consume#0.i a lower rank than
  // produce#0.(i+1) (same dive-into-the-consumer-first shape as
  // depth_first_pushes_one_tile_through_the_pipeline_first), and that lower
  // rank wins the pop outright regardless of readiness, so each consume
  // iteration cuts in front of the next not-yet-ranked produce sibling.
  simulate_options_t opts;
  opts.lanes         = 1;
  const schedule_t s = simulate(g, depth_first_runtime_t{}, unit_cost_t{}, opts);
  CHECK(s.start.size() == 8);
  CHECK(s.makespan() == 8);
  CHECK(render(g, s) == "0: produce#0 | 1: consume#0 | 2: produce#0.1 | 3: consume#0.1 | "
                        "4: produce#0.2 | 5: consume#0.2 | 6: produce#0.3 | 7: consume#0.3");
}

TEST(a_snapshot_schedules_identically_to_the_graph_it_came_from) {
  const graph_t g         = one_to_one();
  const free_graph_t snap = g.to_free();
  const unit_cost_t cost;
  const asap_runtime_t asap;
  const depth_first_runtime_t df;

  for (int lanes = 1; lanes <= 4; ++lanes) {
    simulate_options_t opts;
    opts.lanes = lanes;
    CHECK(render(g, simulate(g, asap, cost, opts)) ==
          render(snap, simulate(snap, asap, cost, opts)));
    CHECK(render(g, simulate(g, df, cost, opts)) == render(snap, simulate(snap, df, cost, opts)));
  }
}

// ─── cost model scaling, empty graphs and validation ──────────────────

TEST(a_flat_cost_model_stretches_every_atom) {
  // wg_duration used to be a runtime constructor argument; now that duration
  // is entirely the cost model's concern, the same stretching is exercised
  // through a cost model instead of a runtime knob.
  const graph_t g = one_to_one();
  simulate_options_t opts;
  opts.lanes = 2;

  const schedule_t s = simulate(g, asap_runtime_t{}, flat_cost_t(10.0), opts);

  CHECK(s.makespan() == 40);
  CHECK(s.start.at(wg_node_t{0, 0, 0}) == 0);
  CHECK(s.start.at(wg_node_t{1, 0, 0}) == 20);
  CHECK(s.finish(wg_node_t{1, 0, 0}) == 30);
}

TEST(an_empty_graph_schedules_to_nothing) {
  const free_graph_t g;
  const unit_cost_t cost;
  const breadth_first_runtime_t bf;
  const asap_runtime_t asap;
  const depth_first_runtime_t df;
  for (const runtime_t* rt : {static_cast<const runtime_t*>(&bf),
                              static_cast<const runtime_t*>(&asap),
                              static_cast<const runtime_t*>(&df)}) {
    const schedule_t s = simulate(g, *rt, cost, simulate_options_t{});
    CHECK(s.start.empty());
    CHECK(s.makespan() == 0);
    CHECK(s.order.empty());
  }
}

TEST(non_positive_lanes_are_rejected) {
  // wg_duration validation used to live here too, but duration is no longer a
  // runtime constructor argument -- it is entirely the cost model's concern,
  // and simulate() places no constraint on what a cost model may return.
  const graph_t g = one_to_one();
  const unit_cost_t cost;
  const breadth_first_runtime_t bf;
  const asap_runtime_t asap;
  const depth_first_runtime_t df;
  const std::vector<const runtime_t*> runtimes{&bf, &asap, &df};

  for (int bad_lanes : {0, -1}) {
    simulate_options_t opts;
    opts.lanes = bad_lanes;

    for (const runtime_t* rt : runtimes) {
      bool threw = false;
      try {
        simulate(g, *rt, cost, opts);
      } catch (const std::invalid_argument&) { threw = true; }
      CHECK(threw);
    }
  }
}

TEST(schedule_summary_reports_the_shape_of_the_run) {
  const graph_t g = one_to_one();
  const unit_cost_t cost;

  const std::string wide = simulate(g, asap_runtime_t{}, cost, simulate_options_t{}).summary();
  CHECK(wide.find("runtime 'asap (pipelined)'") != std::string::npos);
  CHECK(wide.find("8 atoms") != std::string::npos);
  CHECK(wide.find("makespan=2") != std::string::npos);
  CHECK(wide.find("lanes=\u221e") != std::string::npos);

  simulate_options_t narrow_opts;
  narrow_opts.lanes        = 2;
  const std::string narrow = simulate(g, breadth_first_runtime_t{}, cost, narrow_opts).summary();
  CHECK(narrow.find("lanes=2") != std::string::npos);
}

TEST(earliest_free_placement_prefers_the_lane_that_frees_soonest) {
  const std::vector<double> free_at{7.0, 2.0, 5.0};
  const earliest_free_placement_t placement;
  const wg_node_t node{0, 0, 0};
  CHECK(placement.choose(placement_context_t{node, 0, free_at}) == 1);
}

TEST(earliest_free_placement_breaks_ties_towards_the_lower_lane) {
  const std::vector<double> free_at{2.0, 2.0, 2.0};
  const earliest_free_placement_t placement;
  const wg_node_t node{0, 0, 0};
  CHECK(placement.choose(placement_context_t{node, 0, free_at}) == 0);
}

TEST(xcd_placement_round_robins_dies_by_launch_index) {
  // Two dies, two CUs each: lanes 0,1 are die 0 and lanes 2,3 are die 1.
  const xcd_placement_t placement(2, 2);
  const wg_node_t node{0, 0, 0};

  const std::vector<double> idle{0.0, 0.0, 0.0, 0.0};
  CHECK(placement.choose(placement_context_t{node, 0, idle}) == 0);
  CHECK(placement.choose(placement_context_t{node, 1, idle}) == 2);

  // Launch 2 returns to die 0. In a real dispatch its first CU is busy by now,
  // which is what pushes the workgroup onto the die's other unit -- the
  // placement itself does not rotate CUs, it only reads free times.
  const std::vector<double> busy{4.0, 0.0, 4.0, 0.0};
  CHECK(placement.choose(placement_context_t{node, 2, busy}) == 1);
}

TEST(xcd_placement_prefers_the_lower_compute_unit_when_a_die_is_idle) {
  // Matches the reference dispatcher, which seeds its search at the die's
  // first CU and takes a strict minimum, so equal free times keep it there.
  const std::vector<double> idle{0.0, 0.0, 0.0, 0.0};
  const xcd_placement_t placement(2, 2);
  const wg_node_t node{0, 0, 0};
  CHECK(placement.choose(placement_context_t{node, 2, idle}) == 0);
}

TEST(xcd_placement_takes_the_soonest_compute_unit_within_its_die) {
  const std::vector<double> free_at{9.0, 1.0, 0.0, 0.0};
  const xcd_placement_t placement(2, 2);
  const wg_node_t node{0, 0, 0};
  // Launch index 0 is pinned to die 0, so lane 2 is not a candidate.
  CHECK(placement.choose(placement_context_t{node, 0, free_at}) == 1);
}

TEST(a_pool_confines_an_operation_and_a_stale_one_falls_back) {
  const graph_t g = one_to_one();  // a(4) -> b(4)

  // Confined: 'a' may only use lane 0 even though three are free, and 'b' may
  // only use lane 2, so a pool really does keep two operations apart.
  const pooled_placement_t confined({{"a", {0}}, {"b", {2}}}, g, 3);
  const std::vector<double> idle{0.0, 0.0, 0.0};
  CHECK(confined.choose(placement_context_t{wg_node_t{0, 0, 0}, 0, idle}) == 0);
  CHECK(confined.choose(placement_context_t{wg_node_t{1, 0, 0}, 1, idle}) == 2);

  // Within a pool it is still earliest-free, so a pool is a restriction on the
  // shipped rule rather than a different rule.
  const pooled_placement_t pair({{"a", {1, 2}}}, g, 3);
  const std::vector<double> busy{0.0, 5.0, 1.0};
  CHECK(pair.choose(placement_context_t{wg_node_t{0, 0, 0}, 0, busy}) == 2);

  // A pool naming only lanes the machine does not have degrades to every lane
  // rather than crashing, which is what makes a stale pool survivable: the
  // machine shrank, and a schedule that is merely unpartitioned is a better
  // answer than none. Ranking is stricter and rejects such a config outright,
  // since there a pool out of range is a mistake in the sweep.
  const pooled_placement_t stale({{"a", {7, 8}}}, g, 3);
  CHECK(stale.choose(placement_context_t{wg_node_t{0, 0, 0}, 0, busy}) == 0);

  // An operation the pool does not mention keeps the whole machine.
  CHECK(stale.choose(placement_context_t{wg_node_t{1, 0, 0}, 0, busy}) == 0);
}

TEST(runtime_with_placement_forwards_the_policy_and_swaps_only_the_placement) {
  const graph_t g = one_to_one();  // a(4) -> b(4), one edge per workgroup
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes = 2;

  const breadth_first_runtime_t base;
  const pooled_placement_t pool({{"a", {0}}, {"b", {1}}}, g, 2);
  const runtime_with_placement_t wrapped(base, pool);

  CHECK(wrapped.name() == base.name());
  CHECK(wrapped.order() == base.order());

  const schedule_t s = simulate(g, wrapped, cost, opts);
  CHECK(respects_dependencies(g, s));

  // The placement is the pool's, not breadth-first's own earliest-free: every
  // 'a' lands on lane 0 and every 'b' on lane 1, confining each operation to
  // one lane rather than spreading it across both, which are free the whole
  // time.
  for (const wg_node_t& n : s.order) CHECK(s.lane.at(n) == (n.op == 0 ? 0 : 1));

  // gated() also forwards from the base: breadth-first's whole-operation
  // barrier survives the wrap, so no 'b' starts until every 'a' has retired at
  // t=4, even though a per-edge dependency alone -- what asap_runtime_t relies
  // on -- would have let b#0 start the moment a#0 finished, at t=1.
  CHECK(render(g, s) == "0: a#0 | 1: a#1 | 2: a#2 | 3: a#3 | 4: b#0 | 5: b#1 | 6: b#2 | 7: b#3");
}

TEST(utilization_and_busy_by_op_measure_work_rather_than_wall_time) {
  // Every atom is one cycle, and a(4) -> b(4) over two lanes takes four cycles
  // of wall time with both lanes busy throughout, so eight lane-cycles are
  // available and eight are used.
  const graph_t g = one_to_one();
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes = 2;

  const schedule_t s = simulate(g, asap_runtime_t{}, cost, opts);
  CHECK_NEAR(s.makespan(), 4.0, 1e-12);
  CHECK_NEAR(s.utilization(), 1.0, 1e-12);

  const std::vector<double> busy = s.busy_by_op();
  CHECK(busy.size() == 2);
  CHECK_NEAR(busy[0], 4.0, 1e-12);
  CHECK_NEAR(busy[1], 4.0, 1e-12);

  // Idle lanes cost utilisation without changing the work done: the same eight
  // cycles of work now sit in a machine offering thirty-two.
  simulate_options_t wide;
  wide.lanes            = 8;
  const schedule_t airy = simulate(g, asap_runtime_t{}, cost, wide);
  CHECK_NEAR(airy.makespan(), 2.0, 1e-12);
  CHECK_NEAR(airy.utilization(), 8.0 / 16.0, 1e-12);
  CHECK_NEAR(airy.busy_by_op()[0], 4.0, 1e-12);
}

// ─── the four runtimes as policies over simulate() ────────────────────

TEST(asap_under_unit_cost_reproduces_the_dependency_depth) {
  free_graph_t g("chain");
  const int a = g.add_op("a", 1, 1);
  const int b = g.add_op("b", 1, 1);
  const int c = g.add_op("c", 1, 1);
  g.add_edge(wg_node_t{a, 0, 0}, wg_node_t{b, 0, 0}, "x", 1);
  g.add_edge(wg_node_t{b, 0, 0}, wg_node_t{c, 0, 0}, "x", 1);

  const asap_runtime_t runtime;
  const unit_cost_t cost;
  const schedule_t s = simulate(g, runtime, cost, simulate_options_t{});
  CHECK_NEAR(s.start.at(wg_node_t{a, 0, 0}), 0.0, 1e-12);
  CHECK_NEAR(s.start.at(wg_node_t{b, 0, 0}), 1.0, 1e-12);
  CHECK_NEAR(s.start.at(wg_node_t{c, 0, 0}), 2.0, 1e-12);
}

TEST(breadth_first_never_overlaps_two_operations) {
  free_graph_t g("independent");
  g.add_op("a", 2, 1);
  g.add_op("b", 2, 1);  // no edges: only the gate can separate them

  const breadth_first_runtime_t runtime;
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes         = 4;  // room to overlap, if the gate allowed it
  const schedule_t s = simulate(g, runtime, cost, opts);

  const double a_end   = std::max(s.finish(wg_node_t{0, 0, 0}), s.finish(wg_node_t{0, 1, 0}));
  const double b_start = std::min(s.start.at(wg_node_t{1, 0, 0}), s.start.at(wg_node_t{1, 1, 0}));
  CHECK(b_start >= a_end - 1e-12);
}

TEST(xcd_dispatches_in_launch_order_not_ready_order) {
  free_graph_t g("two_ops");
  g.add_op("a", 4, 1);
  const xcd_runtime_t runtime(2, 2);
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes         = 4;
  const schedule_t s = simulate(g, runtime, cost, opts);
  // Launch indices 0..3 alternate dies: lanes 0, 2, 1, 3.
  CHECK(s.lane.at(wg_node_t{0, 0, 0}) == 0);
  CHECK(s.lane.at(wg_node_t{0, 1, 0}) == 2);
  CHECK(s.lane.at(wg_node_t{0, 2, 0}) == 1);
  CHECK(s.lane.at(wg_node_t{0, 3, 0}) == 3);
}

// `trigger` unblocks `target` one cycle in; `sibling` is an independent
// source ready from cycle 0, but declared (and therefore ranked) after
// `target`. A rank-first runtime must still dispatch `target` first, even
// though it becomes ready strictly later than `sibling` -- unlike
// xcd_dispatches_in_launch_order_not_ready_order above, where every node is
// ready at cycle 0 and the test would pass under ready_first too, this graph
// actually exercises rank dominating readiness.

TEST(xcd_dispatches_by_rank_even_when_ready_last) {
  free_graph_t g("xcd_rank_dominance");
  const int trigger = g.add_op("trigger", 1, 1);
  const int target  = g.add_op("target", 1, 1);
  const int sibling = g.add_op("sibling", 1, 1);
  g.add_edge(wg_node_t{trigger, 0, 0}, wg_node_t{target, 0, 0}, "x", 1);

  const xcd_runtime_t runtime(1, 1);
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes         = 1;
  const schedule_t s = simulate(g, runtime, cost, opts);

  const auto dispatch_index = [&](const wg_node_t& n) {
    return std::find(s.order.begin(), s.order.end(), n) - s.order.begin();
  };
  // sibling is ready a full cycle before target (which waits on trigger), but
  // target's launch rank is lower, so rank_first dispatches it first anyway.
  CHECK(s.start.at(wg_node_t{sibling, 0, 0}) > s.start.at(wg_node_t{target, 0, 0}));
  CHECK(dispatch_index(wg_node_t{target, 0, 0}) < dispatch_index(wg_node_t{sibling, 0, 0}));
}

TEST(depth_first_dispatches_by_rank_even_when_ready_last) {
  // Same shape as the xcd test above: depth_first_rank assigns the same
  // ranks here as canonical order does (trigger, target and sibling are each
  // a lone source or a lone successor, so the DFS visits them in declaration
  // order), so depth_first_runtime_t -- the other rank_first policy -- is
  // exercised by the identical property.
  free_graph_t g("depth_first_rank_dominance");
  const int trigger = g.add_op("trigger", 1, 1);
  const int target  = g.add_op("target", 1, 1);
  const int sibling = g.add_op("sibling", 1, 1);
  g.add_edge(wg_node_t{trigger, 0, 0}, wg_node_t{target, 0, 0}, "x", 1);

  const depth_first_runtime_t runtime;
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes         = 1;
  const schedule_t s = simulate(g, runtime, cost, opts);

  const auto dispatch_index = [&](const wg_node_t& n) {
    return std::find(s.order.begin(), s.order.end(), n) - s.order.begin();
  };
  CHECK(s.start.at(wg_node_t{sibling, 0, 0}) > s.start.at(wg_node_t{target, 0, 0}));
  CHECK(dispatch_index(wg_node_t{target, 0, 0}) < dispatch_index(wg_node_t{sibling, 0, 0}));
}

TEST(xcd_throws_when_launch_order_is_not_topological) {
  // Neither backend can actually produce this graph: free_graph_t::add_edge
  // requires src < dst in canonical order, and graph_t's derivation is even
  // stricter (only ever a strictly earlier operation feeding a later one).
  // xcd's rank *is* canonical order, so simulate()'s topological-launch-order
  // check can never fire through either public backend -- forging a minimal
  // wg_graph_t, the way a_cycle_is_reported_rather_than_looping_forever does
  // for the cycle check in test_analysis.cpp, is the only way to reach it.
  struct backwards_edge_graph_t : wg_graph_t {
    std::vector<edge_t> e{edge_t{wg_node_t{1, 0, 0}, wg_node_t{0, 0, 0}, "x", 1}};
    std::string names[2] = {"a", "b"};
    std::string n;
    std::vector<std::size_t> from_op1{0}, none;

    int num_operations() const override { return 2; }
    const std::string& op_name(int op) const override { return names[op]; }
    int op_index(const std::string&) const override { return 0; }
    index_t wg_count(int) const override { return 1; }
    index_t iter_count(int) const override { return 1; }
    const std::vector<edge_t>& edges() const override { return e; }
    const std::string& name() const override { return n; }
    const std::vector<std::size_t>& successor_edges(const wg_node_t& node) const override {
      return node.op == 1 ? from_op1 : none;
    }
    const std::vector<std::size_t>& predecessor_edges(const wg_node_t& node) const override {
      return node.op == 0 ? from_op1 : none;
    }
  };

  backwards_edge_graph_t g;  // b (op 1) -> a (op 0): backwards in canonical order.
  const xcd_runtime_t runtime(1, 1);
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes = 1;

  bool threw = false;
  try {
    simulate(g, runtime, cost, opts);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(xcd_runtime_rejects_a_die_or_cu_count_below_one) {
  for (const auto& [num_xcds, cus_per_xcd] : std::vector<std::pair<int, int>>{
           {0, 4},
           {-1, 4},
           {4, 0},
           {4, -1},
       }) {
    bool threw = false;
    try {
      const xcd_runtime_t runtime(num_xcds, cus_per_xcd);
      (void)runtime;
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }
}

ORIGAMI_TEST_MAIN()
