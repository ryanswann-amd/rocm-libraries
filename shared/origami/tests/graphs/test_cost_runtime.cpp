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

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "origami/graphs/core.hpp"
#include "origami/graphs/free_graph.hpp"
#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/** Every node costs its operation's entry; hops are free unless asked for. */
class flat_cost_t : public cost_model_t {
 public:
  explicit flat_cost_t(std::vector<double> per_op, double per_edge = 0.0)
      : per_op_(std::move(per_op)), per_edge_(per_edge) {}

  double node_cost(const wg_node_t& node) const override {
    return per_op_.at(static_cast<std::size_t>(node.op));
  }
  double edge_cost(const edge_t&) const override { return per_edge_; }

 private:
  std::vector<double> per_op_;
  double per_edge_;
};

/** Cost grows linearly with the number of busy lanes: a bandwidth-share model. */
class contended_cost_t : public cost_model_t {
 public:
  explicit contended_cost_t(double base) : base_(base) {}

  double node_cost(const wg_node_t&) const override { return base_; }
  double edge_cost(const edge_t&) const override { return 0.0; }
  double node_cost_at(const wg_node_t&, int active_cus) const override {
    return base_ * active_cus;
  }

 private:
  double base_;
};

/** a(4 wgs) -> b(2 wgs), one edge per producer. */
graph_t fan_in() {
  const allocation_t x("x", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 2);
  b.access_patterns = {read(x, contiguous(8))};
  return graph_t({a, b});
}

/** One workgroup each, four iterations, iteration i to iteration i. */
graph_t streamed_pair() {
  const allocation_t buf("buf", {index_t{32}});
  operation_t p("produce", 1);
  p.num_iters       = 4;
  p.access_patterns = {write(buf, streamed(32, 4))};
  operation_t c("consume", 1);
  c.num_iters       = 4;
  c.access_patterns = {read(buf, streamed(32, 4))};
  return graph_t({p, c});
}

bool respects_dependencies(const wg_graph_t& g, const cost_schedule_t& s, double tolerance) {
  for (const edge_t& e : g.edges()) {
    if (s.start.at(e.dst) < s.finish(e.src) - tolerance) return false;
  }
  return true;
}

/** No lane runs two atoms at once. */
bool lanes_are_exclusive(const cost_schedule_t& s) {
  for (const wg_node_t& a : s.order) {
    for (const wg_node_t& b : s.order) {
      if (a == b || s.lane.at(a) != s.lane.at(b)) continue;
      if (s.duration.at(a) == 0.0 || s.duration.at(b) == 0.0) continue;
      if (s.start.at(a) < s.finish(b) && s.start.at(b) < s.finish(a)) return false;
    }
  }
  return true;
}

}  // namespace

// ─── the properties every continuous runtime must have ────────────────

TEST(continuous_runtimes_respect_dependencies_and_lane_exclusivity) {
  const graph_t g = fan_in();
  const flat_cost_t cost({2.0, 5.0});

  for (int lanes = 1; lanes <= 4; ++lanes) {
    cost_runtime_options_t opts;
    opts.lanes = lanes;
    opts.scale = 1.0;

    const roofline_runtime_t roof(cost, opts);
    const event_driven_runtime_t event(cost, opts);
    for (const cost_runtime_t* rt :
         {static_cast<const cost_runtime_t*>(&roof), static_cast<const cost_runtime_t*>(&event)}) {
      const cost_schedule_t s = rt->schedule(g);
      CHECK(s.order.size() == g.num_nodes());
      CHECK(respects_dependencies(g, s, 1e-12));
      CHECK(lanes_are_exclusive(s));
      CHECK(s.num_lanes() <= lanes);
    }
  }
}

TEST(durations_come_from_the_cost_model) {
  const graph_t g = fan_in();
  const flat_cost_t cost({2.0, 5.0});
  cost_runtime_options_t opts;
  opts.scale = 1.0;

  const cost_schedule_t s = roofline_runtime_t(cost, opts).schedule(g);

  CHECK(s.duration.at(wg_node_t{0, 0, 0}) == 2.0);
  CHECK(s.duration.at(wg_node_t{1, 0, 0}) == 5.0);
  // Unlimited lanes: producers all start at 0, consumers at 2, finishing at 7.
  CHECK(s.start.at(wg_node_t{1, 0, 0}) == 2.0);
  CHECK(s.makespan() == 7.0);
}

// ─── cost on the graph ────────────────────────────────────────────────

TEST(a_runtime_without_a_cost_model_prices_the_graph_with_the_graphs_own) {
  graph_t g = fan_in();
  g.set_cost(std::make_shared<const flat_cost_t>(std::vector<double>{2.0, 5.0}));
  cost_runtime_options_t opts;
  opts.scale = 1.0;

  // Same numbers as durations_come_from_the_cost_model, from the graph's cost.
  const cost_schedule_t s = roofline_runtime_t(opts).schedule(g);

  CHECK(s.duration.at(wg_node_t{0, 0, 0}) == 2.0);
  CHECK(s.duration.at(wg_node_t{1, 0, 0}) == 5.0);
  CHECK(s.makespan() == 7.0);
}

TEST(two_graphs_can_carry_two_different_costs_under_one_runtime) {
  graph_t cheap = fan_in();
  graph_t dear  = fan_in();
  cheap.set_cost(std::make_shared<const flat_cost_t>(std::vector<double>{2.0, 5.0}));
  dear.set_cost(std::make_shared<const flat_cost_t>(std::vector<double>{20.0, 50.0}));

  cost_runtime_options_t opts;
  opts.scale = 1.0;
  const roofline_runtime_t runtime(opts);

  CHECK(runtime.schedule(cheap).makespan() == 7.0);
  CHECK(runtime.schedule(dear).makespan() == 70.0);
}

TEST(a_runtimes_own_cost_model_overrides_the_graphs) {
  graph_t g = fan_in();
  g.set_cost(std::make_shared<const flat_cost_t>(std::vector<double>{20.0, 50.0}));
  const flat_cost_t override_cost({2.0, 5.0});
  cost_runtime_options_t opts;
  opts.scale = 1.0;

  CHECK(roofline_runtime_t(override_cost, opts).schedule(g).makespan() == 7.0);
}

TEST(scheduling_a_graph_with_no_cost_at_all_throws) {
  const graph_t g = fan_in();  // no set_cost
  CHECK(g.cost() == nullptr);

  bool threw = false;
  try {
    roofline_runtime_t().schedule(g);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);

  threw = false;
  try {
    xcd_runtime_t().schedule(g);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(the_xcd_runtime_reads_the_graphs_cost_too) {
  graph_t pinned = fan_in();
  pinned.set_cost(std::make_shared<const flat_cost_t>(std::vector<double>{2.0, 5.0}));
  const flat_cost_t same({2.0, 5.0});

  xcd_options_t opts;
  opts.scale = 1.0;

  CHECK(xcd_runtime_t(opts).schedule(pinned).makespan() ==
        xcd_runtime_t(same, opts).schedule(pinned).makespan());
}

TEST(scale_converts_units_without_changing_the_order) {
  const graph_t g = fan_in();
  const flat_cost_t cost({2.0, 5.0});

  cost_runtime_options_t seconds;
  seconds.scale = 1.0;
  cost_runtime_options_t micros;
  micros.scale = 1e6;

  const cost_schedule_t a = roofline_runtime_t(cost, seconds).schedule(g);
  const cost_schedule_t b = roofline_runtime_t(cost, micros).schedule(g);

  CHECK(b.makespan() == a.makespan() * 1e6);
  CHECK(b.order == a.order);
  CHECK(b.lane.at(wg_node_t{1, 1, 0}) == a.lane.at(wg_node_t{1, 1, 0}));
}

TEST(edge_costs_delay_the_consumer) {
  const graph_t g = fan_in();
  const flat_cost_t free_hops({2.0, 5.0}, 0.0);
  const flat_cost_t slow_hops({2.0, 5.0}, 3.0);
  cost_runtime_options_t opts;
  opts.scale = 1.0;

  CHECK(roofline_runtime_t(free_hops, opts).schedule(g).makespan() == 7.0);
  CHECK(roofline_runtime_t(slow_hops, opts).schedule(g).makespan() == 10.0);
}

// ─── contention ───────────────────────────────────────────────────────

TEST(the_two_lane_pool_runtimes_agree_when_cost_ignores_contention) {
  // node_cost_at defaults to node_cost, so a model with no view on contention
  // makes the event-driven policy degenerate into the roofline one. Any
  // difference between them is therefore attributable to the model, not the
  // scheduler, which is the point of making it a defaulted virtual rather than
  // a feature-detected optional method.
  const graph_t g = fan_in();
  const flat_cost_t cost({2.0, 5.0});
  cost_runtime_options_t opts;
  opts.lanes = 3;
  opts.scale = 1.0;

  const cost_schedule_t roof  = roofline_runtime_t(cost, opts).schedule(g);
  const cost_schedule_t event = event_driven_runtime_t(cost, opts).schedule(g);

  CHECK(roof.order == event.order);
  CHECK(roof.makespan() == event.makespan());
  for (const wg_node_t& n : roof.order) {
    CHECK(roof.start.at(n) == event.start.at(n));
    CHECK(roof.duration.at(n) == event.duration.at(n));
    CHECK(roof.lane.at(n) == event.lane.at(n));
  }
}

TEST(event_driven_charges_more_when_the_machine_is_crowded) {
  const graph_t g = fan_in();
  const contended_cost_t cost(1.0);
  cost_runtime_options_t opts;
  opts.lanes = 4;
  opts.scale = 1.0;

  const cost_schedule_t s = event_driven_runtime_t(cost, opts).schedule(g);

  // All four producers dispatch at time 0. The first sees an idle machine and
  // is charged for one active lane; each subsequent one sees the previous still
  // running, so the charge climbs.
  CHECK(s.duration.at(wg_node_t{0, 0, 0}) == 1.0);
  CHECK(s.duration.at(wg_node_t{0, 1, 0}) == 2.0);
  CHECK(s.duration.at(wg_node_t{0, 2, 0}) == 3.0);
  CHECK(s.duration.at(wg_node_t{0, 3, 0}) == 4.0);

  // The roofline policy prices everything at the uncontended rate and so is
  // optimistic against it.
  const cost_schedule_t roof = roofline_runtime_t(cost, opts).schedule(g);
  CHECK(roof.makespan() < s.makespan());
}

// ─── lane pools ───────────────────────────────────────────────────────

TEST(a_lane_pool_partitions_the_machine_between_operations) {
  const graph_t g = fan_in();
  const flat_cost_t cost({2.0, 5.0});
  cost_runtime_options_t opts;
  opts.lanes     = 4;
  opts.scale     = 1.0;
  opts.lane_pool = {{"a", {0, 1}}, {"b", {2, 3}}};

  const cost_schedule_t s = roofline_runtime_t(cost, opts).schedule(g);

  for (const wg_node_t& n : s.order) {
    if (n.op == 0) CHECK(s.lane.at(n) <= 1);
    if (n.op == 1) CHECK(s.lane.at(n) >= 2);
  }
  // Four producers over two lanes is two rounds. b#0 needs only a#0 and a#1, so
  // it starts after the first round while the second is still running; b#1 needs
  // a#2 and a#3 and waits for both rounds. That staggering is the overlap a
  // partition buys, and it disappears if the pool is ignored.
  CHECK(s.start.at(wg_node_t{1, 0, 0}) == 2.0);
  CHECK(s.start.at(wg_node_t{1, 1, 0}) == 4.0);
}

TEST(a_pool_outside_the_machine_falls_back_to_every_lane) {
  const graph_t g = fan_in();
  const flat_cost_t cost({2.0, 5.0});
  cost_runtime_options_t opts;
  opts.lanes     = 2;
  opts.scale     = 1.0;
  opts.lane_pool = {{"a", {7, 8, 9}}};  // no such lanes on a 2-lane machine

  const cost_schedule_t s = roofline_runtime_t(cost, opts).schedule(g);
  CHECK(s.order.size() == 6);
  CHECK(s.num_lanes() <= 2);
}

// ─── iteration serialisation ──────────────────────────────────────────

TEST(serialising_iterations_pins_a_workgroup_to_one_lane) {
  // This is the gap the critical path leaves open: nothing in the graph says a
  // workgroup runs its own iterations in sequence, so without this flag all four
  // iterations run at once.
  const graph_t g = streamed_pair();
  const flat_cost_t cost({1.0, 1.0});

  cost_runtime_options_t parallel;
  parallel.scale = 1.0;
  cost_runtime_options_t serial;
  serial.scale              = 1.0;
  serial.serialize_wg_iters = true;

  const cost_schedule_t loose = roofline_runtime_t(cost, parallel).schedule(g);
  const cost_schedule_t tight = roofline_runtime_t(cost, serial).schedule(g);

  CHECK(loose.makespan() == 2.0);
  CHECK(tight.makespan() == 5.0);  // the four-deep pipeline finally shows up

  // Each workgroup's iterations share a lane, because a workgroup holds one
  // compute unit for its whole lifetime.
  for (int op = 0; op < 2; ++op) {
    const int first = tight.lane.at(wg_node_t{op, 0, 0});
    for (int it = 1; it < 4; ++it) CHECK(tight.lane.at(wg_node_t{op, 0, it}) == first);
  }
  CHECK(lanes_are_exclusive(tight));
}

// ─── xcd ──────────────────────────────────────────────────────────────

TEST(xcd_places_workgroups_round_robin_across_dies) {
  const graph_t g = fan_in();
  const flat_cost_t cost({2.0, 5.0});
  xcd_options_t opts;
  opts.num_xcds    = 4;
  opts.cus_per_xcd = 2;
  opts.scale       = 1.0;

  const cost_schedule_t s = xcd_runtime_t(cost, opts).schedule(g);

  // Launch order is a#0..a#3 then b#0..b#1, so slots 0..5 land on dies 0,1,2,3,
  // 0,1 — and the first compute unit of each die, all being idle.
  CHECK(s.lane.at(wg_node_t{0, 0, 0}) == 0);
  CHECK(s.lane.at(wg_node_t{0, 1, 0}) == 2);
  CHECK(s.lane.at(wg_node_t{0, 2, 0}) == 4);
  CHECK(s.lane.at(wg_node_t{0, 3, 0}) == 6);
  CHECK(s.lane.at(wg_node_t{1, 0, 0}) == 1);
  CHECK(s.lane.at(wg_node_t{1, 1, 0}) == 3);

  CHECK(s.lanes.has_value());
  CHECK(*s.lanes == 8);
  CHECK(respects_dependencies(g, s, 1e-12));
}

TEST(xcd_steals_the_compute_unit_that_frees_soonest) {
  // Eight workgroups on one die with two compute units: the second wave lands on
  // whichever unit is free, so the die stays busy.
  free_graph_t g;
  g.add_op("k", 8);
  const flat_cost_t cost({3.0});
  xcd_options_t opts;
  opts.num_xcds    = 1;
  opts.cus_per_xcd = 2;
  opts.scale       = 1.0;

  const cost_schedule_t s = xcd_runtime_t(cost, opts).schedule(g);

  CHECK(s.makespan() == 12.0);  // 8 atoms x 3 over 2 units
  CHECK(s.utilization() == 1.0);
  CHECK(lanes_are_exclusive(s));
}

TEST(xcd_skips_zero_work_barrier_operations) {
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});
  operation_t a("a", 2);
  a.access_patterns = {write(x, contiguous(8))};
  operation_t s("sync_all", 1);
  s.access_patterns = {read(x, contiguous(16)), write(y, contiguous(16))};
  operation_t b("b", 2);
  b.access_patterns = {read(y, contiguous(8))};
  const graph_t g({a, s, b});

  const flat_cost_t cost({4.0, 0.0, 4.0});
  xcd_options_t opts;
  opts.num_xcds    = 2;
  opts.cus_per_xcd = 1;
  opts.scale       = 1.0;

  const cost_schedule_t sched = xcd_runtime_t(cost, opts).schedule(g);

  // The barrier takes no time and no compute unit; it only forwards ready time.
  const wg_node_t barrier{1, 0, 0};
  CHECK(sched.duration.at(barrier) == 0.0);
  CHECK(sched.start.at(barrier) == 4.0);

  // Critically it does not consume a round-robin slot, so 'b' starts its own
  // pass over the dies where 'a' left off rather than one place further on.
  CHECK(sched.lane.at(wg_node_t{2, 0, 0}) == 0);
  CHECK(sched.lane.at(wg_node_t{2, 1, 0}) == 1);
  CHECK(sched.makespan() == 8.0);
}

// ─── schedule reporting ───────────────────────────────────────────────

TEST(busy_by_op_totals_work_not_wall_time) {
  const graph_t g = fan_in();
  const flat_cost_t cost({2.0, 5.0});
  cost_runtime_options_t opts;
  opts.scale = 1.0;

  const std::vector<double> busy = roofline_runtime_t(cost, opts).schedule(g).busy_by_op();

  CHECK(busy.size() == 2);
  CHECK(busy[0] == 8.0);   // four producers at 2
  CHECK(busy[1] == 10.0);  // two consumers at 5
}

TEST(utilization_is_work_over_available_lane_time) {
  // Two atoms of 2 on two lanes, then one atom of 5: 9 of 2 x 7 lane-seconds.
  const graph_t g = fan_in();
  const flat_cost_t cost({2.0, 5.0});
  cost_runtime_options_t opts;
  opts.lanes = 4;
  opts.scale = 1.0;

  const cost_schedule_t s = roofline_runtime_t(cost, opts).schedule(g);
  const double expected   = (4 * 2.0 + 2 * 5.0) / (s.makespan() * s.num_lanes());
  CHECK_NEAR(s.utilization(), expected, 1e-12);
}

TEST(cost_schedule_summary_reports_the_shape_of_the_run) {
  const graph_t g = fan_in();
  const flat_cost_t cost({2.0, 5.0});
  cost_runtime_options_t opts;
  opts.lanes = 4;
  opts.scale = 1.0;
  opts.units = "s";

  const std::string text = roofline_runtime_t(cost, opts).schedule(g).summary();
  CHECK(text.find("runtime 'roofline'") != std::string::npos);
  CHECK(text.find("6 atoms on 4 lanes") != std::string::npos);
  CHECK(text.find("makespan=7.00s") != std::string::npos);
  CHECK(text.find("(lanes=4)") != std::string::npos);
}

// ─── backends, empty graphs and validation ────────────────────────────

TEST(a_snapshot_has_the_same_continuous_time_schedule) {
  const graph_t g         = fan_in();
  const free_graph_t snap = g.to_free();
  const flat_cost_t cost({2.0, 5.0});
  cost_runtime_options_t opts;
  opts.lanes = 3;
  opts.scale = 1.0;

  const cost_schedule_t a = roofline_runtime_t(cost, opts).schedule(g);
  const cost_schedule_t b = roofline_runtime_t(cost, opts).schedule(snap);

  CHECK(a.order == b.order);
  CHECK(a.makespan() == b.makespan());
  for (const wg_node_t& n : a.order) {
    CHECK(a.start.at(n) == b.start.at(n));
    CHECK(a.lane.at(n) == b.lane.at(n));
  }
}

TEST(an_empty_graph_has_an_empty_continuous_time_schedule) {
  const free_graph_t g;
  const flat_cost_t cost({});

  for (const cost_schedule_t& s : {roofline_runtime_t(cost).schedule(g),
                                   event_driven_runtime_t(cost).schedule(g),
                                   xcd_runtime_t(cost).schedule(g)}) {
    CHECK(s.order.empty());
    CHECK(s.makespan() == 0.0);
    CHECK(s.num_lanes() == 0);
    CHECK(s.utilization() == 0.0);
  }
}

TEST(invalid_options_are_rejected) {
  const flat_cost_t cost({1.0});

  bool threw = false;
  try {
    cost_runtime_options_t bad;
    bad.lanes = 0;
    roofline_runtime_t(cost, bad);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);

  threw = false;
  try {
    cost_runtime_options_t bad;
    bad.scale = 0.0;
    event_driven_runtime_t(cost, bad);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);

  threw = false;
  try {
    xcd_options_t bad;
    bad.num_xcds = 0;
    xcd_runtime_t(cost, bad);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

ORIGAMI_TEST_MAIN()
