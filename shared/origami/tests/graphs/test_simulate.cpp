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

#include <stdexcept>
#include <string>
#include <vector>

#include "origami/graphs/cost_model.hpp"
#include "origami/graphs/free_graph.hpp"
#include "origami/graphs/placement.hpp"
#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/** Canonical order, earliest-free lane: the simplest possible policy. */
class plain_runtime_t : public runtime_t {
 public:
  const std::string& name() const override { return name_; }
  wg_node_map_t<std::size_t> priority(const wg_graph_t& graph) const override {
    wg_node_map_t<std::size_t> rank;
    std::size_t i = 0;
    for (const wg_node_t& n : graph.nodes()) rank.emplace(n, i++);
    return rank;
  }
  const placement_t& placement() const override { return placement_; }

 private:
  earliest_free_placement_t placement_;
  std::string name_ = "plain";
};

/** Two operations of two workgroups; op 1 consumes op 0 workgroup for workgroup. */
free_graph_t chain_graph() {
  free_graph_t g("chain");
  const int a = g.add_op("a", 2, 1);
  const int b = g.add_op("b", 2, 1);
  g.add_edge(wg_node_t{a, 0, 0}, wg_node_t{b, 0, 0}, "x", 1);
  g.add_edge(wg_node_t{a, 1, 0}, wg_node_t{b, 1, 0}, "x", 1);
  return g;
}

/**
 * Canonical order, earliest-free lane, plus an operation barrier: a node in
 * `gate_op` is held back until every workgroup of `watch_op` has retired.
 *
 * This is what an operation barrier needs and a rank cannot express: `b#0`
 * becomes ready the moment its one producer (`a#0`) finishes, but the barrier
 * withholds it until *all* of `a` — including `a#1`, which `b#0` does not
 * depend on — has retired.
 */
class barrier_runtime_t : public runtime_t {
 public:
  barrier_runtime_t(int gate_op, int watch_op) : gate_op_(gate_op), watch_op_(watch_op) {}

  const std::string& name() const override { return name_; }
  wg_node_map_t<std::size_t> priority(const wg_graph_t& graph) const override {
    wg_node_map_t<std::size_t> rank;
    std::size_t i = 0;
    for (const wg_node_t& n : graph.nodes()) rank.emplace(n, i++);
    return rank;
  }
  const placement_t& placement() const override { return placement_; }

  bool gated(const wg_node_t& node, const sim_state_t& state) const override {
    if (node.op != gate_op_) return false;
    return state.retired_by_op[static_cast<std::size_t>(watch_op_)] <
           state.total_by_op[static_cast<std::size_t>(watch_op_)];
  }

 private:
  int gate_op_;
  int watch_op_;
  earliest_free_placement_t placement_;
  std::string name_ = "barrier";
};

/** One lane, operation 'a' with two workgroups, operation 'b' with one,
 * consuming only a#0, gated on all of 'a' retiring. */
free_graph_t barrier_graph() {
  free_graph_t g("barrier");
  const int a = g.add_op("a", 2, 1);
  const int b = g.add_op("b", 1, 1);
  g.add_edge(wg_node_t{a, 0, 0}, wg_node_t{b, 0, 0}, "x", 1);
  return g;
}

/** Always returns a lane outside the machine, to exercise the range check. */
class out_of_range_placement_t : public placement_t {
 public:
  const std::string& name() const override { return name_; }
  int choose(const placement_context_t& ctx) const override {
    return static_cast<int>(ctx.free_at.size());  // one past the last lane
  }

 private:
  std::string name_ = "out-of-range";
};

class misbehaving_runtime_t : public runtime_t {
 public:
  const std::string& name() const override { return name_; }
  wg_node_map_t<std::size_t> priority(const wg_graph_t& graph) const override {
    wg_node_map_t<std::size_t> rank;
    std::size_t i = 0;
    for (const wg_node_t& n : graph.nodes()) rank.emplace(n, i++);
    return rank;
  }
  const placement_t& placement() const override { return placement_; }

 private:
  out_of_range_placement_t placement_;
  std::string name_ = "misbehaving";
};

}  // namespace

TEST(simulate_with_unit_cost_gives_every_node_one_cycle) {
  const free_graph_t g = chain_graph();
  const plain_runtime_t runtime;
  const unit_cost_t cost;
  const schedule_t s = simulate(g, runtime, cost, simulate_options_t{});
  CHECK(s.order.size() == 4);
  CHECK_NEAR(s.duration.at(wg_node_t{0, 0, 0}), 1.0, 1e-12);
  CHECK_NEAR(s.makespan(), 2.0, 1e-12);
}

TEST(simulate_serialises_a_chain_when_only_one_lane_exists) {
  const free_graph_t g = chain_graph();
  const plain_runtime_t runtime;
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes         = 1;
  const schedule_t s = simulate(g, runtime, cost, opts);
  CHECK_NEAR(s.makespan(), 4.0, 1e-12);
  CHECK(s.num_lanes() == 1);
}

TEST(repricing_on_dispatch_charges_more_in_a_crowded_machine) {
  class busy_cost_t : public cost_model_t {
   public:
    double node_cycles(const wg_node_t&) const override { return 10.0; }
    double edge_cycles(const edge_t&) const override { return 0.0; }
    double node_cycles_at(const wg_node_t&, int active_cus) const override {
      return 10.0 * static_cast<double>(active_cus);
    }
  };

  free_graph_t g("wide");
  g.add_op("a", 2, 1);
  const plain_runtime_t runtime;
  const busy_cost_t cost;

  simulate_options_t opts;
  opts.lanes               = 2;
  const schedule_t flat    = simulate(g, runtime, cost, opts);
  opts.reprice_on_dispatch = true;
  const schedule_t priced  = simulate(g, runtime, cost, opts);

  CHECK_NEAR(flat.makespan(), 10.0, 1e-12);
  // Both nodes start at zero, so the second sees one lane already busy.
  CHECK_NEAR(priced.makespan(), 20.0, 1e-12);
}

// ─── gate deadlock (Critical) ──────────────────────────────────────────

TEST(a_gated_node_left_alone_in_the_queue_is_not_dropped) {
  // a#0 finishes at cycle 1 and a#1 at cycle 2; b#0 becomes the sole queue
  // entry with ready time 1 once a#0 finishes. Advancing to 1 retires only
  // a#0, so the barrier holds b#0 back, and the queue is then empty even
  // though b#0 is not yet placed. Before the fix this silently exits the
  // dispatch loop and the acyclic-graph guard misreports a cycle. This test
  // pins that the held node is not dropped and is eventually placed; the
  // single lane here, not the barrier, is what pins its start time at 2 (see
  // a_node_released_from_a_gate_cannot_start_before_its_release for the
  // barrier's own timing).
  const free_graph_t g = barrier_graph();
  const barrier_runtime_t runtime(/*gate_op=*/1, /*watch_op=*/0);
  const unit_cost_t cost;
  simulate_options_t opts;
  opts.lanes = 1;

  const schedule_t s = simulate(g, runtime, cost, opts);

  CHECK(s.order.size() == 3);
  CHECK_NEAR(s.start.at(wg_node_t{0, 0, 0}), 0.0, 1e-12);
  CHECK_NEAR(s.start.at(wg_node_t{0, 1, 0}), 1.0, 1e-12);
  // b#0 is not dropped: it is still placed, at cycle 2, even though its only
  // producer (a#0) finished at cycle 1 and the queue emptied out from under
  // it in between.
  CHECK_NEAR(s.start.at(wg_node_t{1, 0, 0}), 2.0, 1e-12);
  CHECK_NEAR(s.makespan(), 3.0, 1e-12);
}

// ─── gate release cannot precede its own barrier (Critical) ────────────

TEST(a_node_released_from_a_gate_cannot_start_before_its_release) {
  // Two lanes: a#0 costs 1, a#1 costs 5, and b#0 (gated on all of 'a'
  // retiring) depends only on a#0. b#0 becomes ready at 1, once a#0
  // finishes, and the gate holds it until a#1 also retires at 5. Before the
  // fix, the held entry kept ready=1 through the replay and dispatched at
  // start=1 — overlapping a#1's 0..5 span on the other lane, a barrier
  // violation. The fix clamps a replayed entry's ready to the release time,
  // so b#0 cannot start before 5.
  class uneven_a_cost_t : public cost_model_t {
   public:
    double node_cycles(const wg_node_t& n) const override {
      return (n.op == 0 && n.wg == 1) ? 5.0 : 1.0;
    }
    double edge_cycles(const edge_t&) const override { return 0.0; }
  };

  const free_graph_t g = barrier_graph();
  const barrier_runtime_t runtime(/*gate_op=*/1, /*watch_op=*/0);
  const uneven_a_cost_t cost;
  simulate_options_t opts;
  opts.lanes = 2;

  const schedule_t s = simulate(g, runtime, cost, opts);

  CHECK_NEAR(s.start.at(wg_node_t{1, 0, 0}), 5.0, 1e-12);
  CHECK_NEAR(s.makespan(), 6.0, 1e-12);
}

// ─── lane validation (Important 3) ──────────────────────────────────────

TEST(simulate_rejects_a_non_positive_lane_count) {
  const free_graph_t g = chain_graph();
  const plain_runtime_t runtime;
  const unit_cost_t cost;

  for (int bad_lanes : {0, -1}) {
    simulate_options_t opts;
    opts.lanes = bad_lanes;
    bool threw = false;
    try {
      simulate(g, runtime, cost, opts);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }
}

// ─── serialize_wg_iters + skip_prefix (Important 4) ─────────────────────

TEST(skip_prefixed_operations_still_release_their_next_iteration) {
  // A skip-prefixed operation with two iterations: unmet[] counts the
  // implicit precedence between the two iterations regardless of skip, so if
  // next_iter's unblocking stayed conditional on the non-skip branch, the
  // second iteration would never become ready and the acyclic graph would
  // misreport a cycle.
  free_graph_t g("skip-chain");
  const int barrier = g.add_op("barrier_wait", 1, 2);
  const plain_runtime_t runtime;
  const unit_cost_t cost;

  simulate_options_t opts;
  opts.serialize_wg_iters = true;
  opts.skip_prefix        = "barrier_";

  const schedule_t s = simulate(g, runtime, cost, opts);

  CHECK(s.order.size() == 2);
  CHECK_NEAR(s.duration.at(wg_node_t{barrier, 0, 0}), 0.0, 1e-12);
  CHECK_NEAR(s.duration.at(wg_node_t{barrier, 0, 1}), 0.0, 1e-12);
}

// ─── edge cost delays a consumer (Important 2) ──────────────────────────

TEST(a_non_zero_edge_cost_delays_the_consumer) {
  // simulate() delays a consumer by cost.edge_cycles(e); this was only
  // observable through the deleted test_cost_runtime.cpp. The surviving
  // edge_cycles tests check what a cost_table_t returns, not what the
  // scheduler does with it, and neither oracle suite uses a non-zero hop
  // cost, so nothing else pins this.
  free_graph_t g("hop");
  const int a = g.add_op("a", 1, 1);
  const int b = g.add_op("b", 1, 1);
  g.add_edge(wg_node_t{a, 0, 0}, wg_node_t{b, 0, 0}, "x", 1);

  class hop_cost_t : public cost_model_t {
   public:
    double node_cycles(const wg_node_t&) const override { return 1.0; }
    double edge_cycles(const edge_t&) const override { return 3.0; }
  };

  const plain_runtime_t runtime;
  const hop_cost_t cost;
  const schedule_t s = simulate(g, runtime, cost, simulate_options_t{});

  // a finishes at 1; the 3-cycle hop pushes b's ready time to 4, so b starts
  // at 4 and finishes at 5 -- a pinned number, not merely "later than a
  // zero-cost hop would give".
  CHECK_NEAR(s.start.at(wg_node_t{b, 0, 0}), 4.0, 1e-12);
  CHECK_NEAR(s.makespan(), 5.0, 1e-12);
}

// ─── serialize_wg_iters pins later iterations (Minor 2) ─────────────────

TEST(serialize_wg_iters_pins_a_later_iteration_to_its_first_iterations_lane) {
  // The skip-prefix test above exercises the branch that deliberately does
  // not pin; this is the pinning branch itself, which the deleted oracle
  // suite was the only thing covering. 'f' ties for lane 0 at time 0 (lower
  // index wins the tie), forcing a#0's cost-10 first iteration onto lane 1.
  // By the time a#0's second iteration is ready (cycle 10), lane 0 has been
  // free since cycle 1 and lane 1 only just freed -- earliest-free would pick
  // lane 0 -- so if the pin were lost, the second iteration would land there
  // instead of staying on lane 1 with the first.
  free_graph_t g("pin");
  const int f = g.add_op("f", 1, 1);
  const int a = g.add_op("a", 1, 2);

  class two_speed_cost_t : public cost_model_t {
   public:
    explicit two_speed_cost_t(int slow_op) : slow_op_(slow_op) {}
    double node_cycles(const wg_node_t& n) const override {
      return (n.op == slow_op_ && n.it == 0) ? 10.0 : 1.0;
    }
    double edge_cycles(const edge_t&) const override { return 0.0; }

   private:
    int slow_op_;
  };

  const plain_runtime_t runtime;
  const two_speed_cost_t cost(a);
  simulate_options_t opts;
  opts.lanes              = 2;
  opts.serialize_wg_iters = true;

  const schedule_t s = simulate(g, runtime, cost, opts);

  CHECK(s.lane.at(wg_node_t{f, 0, 0}) == 0);
  CHECK(s.lane.at(wg_node_t{a, 0, 0}) == 1);
  CHECK(s.lane.at(wg_node_t{a, 0, 1}) == 1);
}

// ─── placement out of range ──────────────────────────────────────────

TEST(a_placement_returning_an_out_of_range_lane_throws) {
  free_graph_t g("single");
  g.add_op("a", 1, 1);
  const misbehaving_runtime_t runtime;
  const unit_cost_t cost;

  bool threw = false;
  try {
    simulate(g, runtime, cost, simulate_options_t{});
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

// ─── fixed_clock_t and to_seconds (Important 5) ─────────────────────────

TEST(fixed_clock_rejects_a_non_positive_frequency) {
  for (double bad_ghz : {0.0, -1.0}) {
    bool threw = false;
    try {
      const fixed_clock_t clock(bad_ghz);
      (void)clock;
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }
}

TEST(to_seconds_converts_a_known_cycle_count_at_a_known_clock) {
  free_graph_t g("single");
  g.add_op("a", 1, 1);
  const plain_runtime_t runtime;

  class fixed_cost_t : public cost_model_t {
   public:
    double node_cycles(const wg_node_t&) const override { return 2100.0; }
    double edge_cycles(const edge_t&) const override { return 0.0; }
  };
  const fixed_cost_t cost;

  const schedule_t cycles = simulate(g, runtime, cost, simulate_options_t{});
  const fixed_clock_t clock(2.1);  // GHz
  const timed_schedule_t seconds = to_seconds(cycles, clock, /*scale=*/1e6, "us");

  // 2100 cycles at 2.1 GHz is one microsecond.
  CHECK_NEAR(seconds.duration.at(wg_node_t{0, 0, 0}), 1.0, 1e-9);
}

TEST(to_seconds_rejects_a_non_positive_scale) {
  free_graph_t g("single");
  g.add_op("a", 1, 1);
  const plain_runtime_t runtime;
  const unit_cost_t cost;
  const schedule_t cycles = simulate(g, runtime, cost, simulate_options_t{});
  const fixed_clock_t clock(1.0);

  for (double bad_scale : {0.0, -1.0}) {
    bool threw = false;
    try {
      to_seconds(cycles, clock, bad_scale);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }
}

ORIGAMI_TEST_MAIN()
