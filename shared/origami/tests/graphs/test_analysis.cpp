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

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "origami/graphs/core.hpp"
#include "origami/graphs/free_graph.hpp"
#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/** Position of a node within an ordering, for checking topological validity. */
std::size_t position_of(const std::vector<wg_node_t>& order, const wg_node_t& n) {
  for (std::size_t i = 0; i < order.size(); ++i) {
    if (order[i] == n) return i;
  }
  return order.size();
}

}  // namespace

// ─── topological order ────────────────────────────────────────────────

TEST(topological_order_lists_producers_before_consumers) {
  free_graph_t g;
  g.add_op("a", 3);
  g.add_op("b", 3);
  g.add_op("c", 3);
  for (int i = 0; i < 3; ++i) {
    g.add_edge(wg_node_t{0, i, 0}, wg_node_t{1, i, 0}, "x", 1);
    g.add_edge(wg_node_t{1, i, 0}, wg_node_t{2, i, 0}, "y", 1);
  }

  const std::vector<wg_node_t> order = topological_order(g);

  CHECK(order.size() == 9);
  for (const edge_t& e : g.edges()) {
    CHECK(position_of(order, e.src) < position_of(order, e.dst));
  }

  // Every node appears exactly once.
  const std::unordered_set<wg_node_t> unique(order.begin(), order.end());
  CHECK(unique.size() == 9);
}

TEST(a_graph_with_no_edges_orders_every_node) {
  const free_graph_t g = free_graph_t::from_grid({{"a", 4, 1}, {"b", 2, 2}});
  CHECK(topological_order(g).size() == 8);
}

TEST(an_empty_graph_has_an_empty_order) {
  const free_graph_t g;
  CHECK(topological_order(g).empty());
  CHECK(critical_path(g).makespan == 0.0);
  CHECK(critical_path(g).path.empty());
}

// ─── critical path with known answers ─────────────────────────────────
//
// These graphs have makespans that can be worked out by hand, so they test
// correctness rather than agreement with the reference.

TEST(a_chain_of_n_unit_nodes_has_makespan_n) {
  // One workgroup per operation, chained: the only path is the whole chain.
  free_graph_t g;
  for (int i = 0; i < 5; ++i) g.add_op("op" + std::to_string(i), 1);
  for (int i = 0; i < 4; ++i) { g.add_edge(wg_node_t{i, 0, 0}, wg_node_t{i + 1, 0, 0}, "buf", 1); }

  const critical_path_t cp = critical_path(g);

  CHECK(cp.makespan == 5.0);
  CHECK(cp.path.size() == 5);
  CHECK(cp.path.front() == (wg_node_t{0, 0, 0}));
  CHECK(cp.path.back() == (wg_node_t{4, 0, 0}));
}

TEST(independent_nodes_all_finish_at_their_own_cost) {
  // No edges at all, so the makespan is the single most expensive node rather
  // than the sum: this is the unbounded-resource bound.
  const free_graph_t g = free_graph_t::from_grid({{"wide", 100, 1}});

  const critical_path_t cp = critical_path(g);
  CHECK(cp.makespan == 1.0);
  CHECK(cp.path.size() == 1);
  CHECK(cp.finish.size() == 100);
}

TEST(the_longest_branch_of_a_diamond_wins) {
  // a -> b(slow) -> d and a -> c(fast) -> d. Costs: a=1, b=10, c=2, d=1.
  free_graph_t g;
  g.add_op("a", 1);
  g.add_op("b", 1);
  g.add_op("c", 1);
  g.add_op("d", 1);
  g.add_edge(wg_node_t{0, 0, 0}, wg_node_t{1, 0, 0}, "x", 1);
  g.add_edge(wg_node_t{0, 0, 0}, wg_node_t{2, 0, 0}, "x", 1);
  g.add_edge(wg_node_t{1, 0, 0}, wg_node_t{3, 0, 0}, "y", 1);
  g.add_edge(wg_node_t{2, 0, 0}, wg_node_t{3, 0, 0}, "z", 1);

  const node_cost_fn_t cost = [](const wg_node_t& n) -> double {
    switch (n.op) {
      case 0: return 1.0;
      case 1: return 10.0;
      case 2: return 2.0;
      default: return 1.0;
    }
  };

  const critical_path_t cp = critical_path(g, cost);

  CHECK(cp.makespan == 12.0);  // 1 + 10 + 1
  CHECK(cp.path.size() == 3);
  CHECK(cp.path[1].op == 1);  // through the slow branch
}

TEST(edge_costs_add_to_the_path) {
  free_graph_t g;
  g.add_op("a", 1);
  g.add_op("b", 1);
  g.add_edge(wg_node_t{0, 0, 0}, wg_node_t{1, 0, 0}, "x", 64);

  // A hop priced at one time unit per 32 elements.
  const edge_cost_fn_t hop = [](const edge_t& e) -> double {
    return static_cast<double>(e.weight) / 32.0;
  };

  CHECK(critical_path(g).makespan == 2.0);
  CHECK(critical_path(g, {}, hop).makespan == 4.0);  // 1 + 2 + 1
}

TEST(finish_times_are_reported_for_every_node) {
  free_graph_t g;
  g.add_op("a", 2);
  g.add_op("b", 2);
  g.add_edge(wg_node_t{0, 0, 0}, wg_node_t{1, 0, 0}, "x", 1);
  // b#1 has no predecessor, so it starts at zero.

  const critical_path_t cp = critical_path(g);

  CHECK(cp.finish.size() == 4);
  CHECK(cp.finish.at(wg_node_t{0, 0, 0}) == 1.0);
  CHECK(cp.finish.at(wg_node_t{0, 1, 0}) == 1.0);
  CHECK(cp.finish.at(wg_node_t{1, 0, 0}) == 2.0);
  CHECK(cp.finish.at(wg_node_t{1, 1, 0}) == 1.0);
  CHECK(cp.makespan == 2.0);
}

TEST(iterations_are_not_serialised_by_the_graph_alone) {
  const allocation_t buf("buf", {index_t{32}});

  operation_t produce("produce", 1);
  produce.num_iters       = 4;
  produce.access_patterns = {write(buf, streamed(32, 4))};

  operation_t consume("consume", 1);
  consume.num_iters       = 4;
  consume.access_patterns = {read(buf, streamed(32, 4))};

  const graph_t partial({produce, consume});

  // Consumer iteration i depends only on producer iteration i, giving four
  // independent two-node chains rather than one barrier.
  CHECK(partial.edges().size() == 4);

  // The makespan is still 2, not 5. Nothing in the graph says a workgroup runs
  // its own iterations one after another — that is a property of the hardware,
  // so it belongs to the runtimes rather than to this bound. The critical path
  // will happily run all four iterations at once.
  CHECK(critical_path(partial).makespan == 2.0);
}

TEST(serialising_iterations_by_hand_exposes_the_pipeline) {
  // The same shape as above, but with each workgroup's iterations chained, which
  // is what a single compute unit would actually do. Now the four-deep pipeline
  // shows up: the producer takes 4, and the last consumer iteration trails it by
  // one, for 5 rather than the 8 of running the two operations back to back.
  free_graph_t g;
  g.add_op("produce", 1, 4);
  g.add_op("consume", 1, 4);
  for (int i = 0; i < 4; ++i) {
    g.add_edge(wg_node_t{0, 0, i}, wg_node_t{1, 0, i}, "buf", 8);
    if (i > 0) {
      g.add_edge(wg_node_t{0, 0, i - 1}, wg_node_t{0, 0, i}, "", 0);
      g.add_edge(wg_node_t{1, 0, i - 1}, wg_node_t{1, 0, i}, "", 0);
    }
  }

  CHECK(critical_path(g).makespan == 5.0);
}

TEST(critical_path_works_through_the_backend_interface) {
  const allocation_t x("x", {index_t{16}});

  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 2);
  b.access_patterns = {read(x, contiguous(8))};

  const graph_t derived({a, b});
  const free_graph_t snap = derived.to_free();

  // The snapshot gate again, now at the level a caller actually cares about.
  CHECK(critical_path(derived).makespan == critical_path(snap).makespan);
  CHECK(critical_path(derived).path.size() == critical_path(snap).path.size());
}

// ─── cycles ───────────────────────────────────────────────────────────

TEST(a_cycle_is_reported_rather_than_looping_forever) {
  // Both backends refuse to build a cycle, so one has to be forged by hand to
  // prove the detection works at all.
  struct cyclic_graph_t : wg_graph_t {
    std::vector<edge_t> e;
    std::string n;
    std::vector<std::size_t> out0{0}, out1{1};

    int num_operations() const override { return 1; }
    const std::string& op_name(int) const override { return n; }
    int op_index(const std::string&) const override { return 0; }
    index_t wg_count(int) const override { return 2; }
    index_t iter_count(int) const override { return 1; }
    const std::vector<edge_t>& edges() const override { return e; }
    const std::string& name() const override { return n; }
    const std::vector<std::size_t>& successor_edges(const wg_node_t& node) const override {
      return node.wg == 0 ? out0 : out1;
    }
    const std::vector<std::size_t>& predecessor_edges(const wg_node_t& node) const override {
      return node.wg == 0 ? out1 : out0;
    }
  };

  cyclic_graph_t g;
  g.e.push_back(edge_t{wg_node_t{0, 0, 0}, wg_node_t{0, 1, 0}, "x", 1});
  g.e.push_back(edge_t{wg_node_t{0, 1, 0}, wg_node_t{0, 0, 0}, "x", 1});

  bool threw = false;
  try {
    topological_order(g);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

// ─── agreement with the reference implementation ──────────────────────
//
// Expected values produced by oracle/analysis_oracle.py against the Python
// wg_graphs library. Makespans are the easy part; the chains are the point.
// Several chains usually tie, and which one is reported falls out of the
// topological order and of Python's max() returning the first maximal key.
// fan_in and diamond below report *different* chains under unit and weighted
// costs, so they fail loudly if either tie-break drifts.

namespace {

/** The oracle's node cost: lopsided, so weighted runs cannot mimic unit ones. */
double oracle_node_cost(const wg_node_t& n) { return 1.0 + n.op * 2.0 + n.wg * 0.5; }

/** The oracle's edge cost: one time unit per eight elements. */
double oracle_edge_cost(const edge_t& e) { return static_cast<double>(e.weight) / 8.0; }

/** Render a critical chain the way the oracle prints it. */
std::string chain_of(const wg_graph_t& g, const critical_path_t& cp) {
  std::string s;
  for (std::size_t i = 0; i < cp.path.size(); ++i) {
    if (i > 0) s += " -> ";
    s += g.label(cp.path[i]);
  }
  return s;
}

}  // namespace

TEST(oracle_one_to_one) {
  const allocation_t x("x", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4))};
  const graph_t g({a, b});

  CHECK(g.edges().size() == 4);

  const critical_path_t unit = critical_path(g);
  CHECK(unit.makespan == 2.0);
  CHECK(chain_of(g, unit) == "a#3 -> b#3");

  const critical_path_t w = critical_path(g, oracle_node_cost, oracle_edge_cost);
  CHECK_NEAR(w.makespan, 7.5, 1e-9);
  CHECK(chain_of(g, w) == "a#3 -> b#3");
}

TEST(oracle_fan_in_reports_different_chains_per_cost_model) {
  const allocation_t x("x", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 2);
  b.access_patterns = {read(x, contiguous(8))};
  const graph_t g({a, b});

  CHECK(g.edges().size() == 4);

  const critical_path_t unit = critical_path(g);
  CHECK(unit.makespan == 2.0);
  CHECK(chain_of(g, unit) == "a#2 -> b#1");

  const critical_path_t w = critical_path(g, oracle_node_cost, oracle_edge_cost);
  CHECK_NEAR(w.makespan, 6.5, 1e-9);
  CHECK(chain_of(g, w) == "a#3 -> b#1");
}

TEST(oracle_three_stage_chain) {
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4)), write(y, contiguous(4))};
  operation_t c("c", 4);
  c.access_patterns = {read(y, contiguous(4))};
  const graph_t g({a, b, c});

  CHECK(g.edges().size() == 8);

  const critical_path_t unit = critical_path(g);
  CHECK(unit.makespan == 3.0);
  CHECK(chain_of(g, unit) == "a#3 -> b#3 -> c#3");

  const critical_path_t w = critical_path(g, oracle_node_cost, oracle_edge_cost);
  CHECK_NEAR(w.makespan, 14.5, 1e-9);
  CHECK(chain_of(g, w) == "a#3 -> b#3 -> c#3");
}

TEST(oracle_diamond_reports_different_branches_per_cost_model) {
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});
  const allocation_t z("z", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4)), write(y, contiguous(4))};
  operation_t c("c", 4);
  c.access_patterns = {read(x, contiguous(4)), write(z, contiguous(4))};
  operation_t d("d", 4);
  d.access_patterns = {read(y, contiguous(4)), read(z, contiguous(4))};
  const graph_t g({a, b, c, d});

  CHECK(g.edges().size() == 16);

  const critical_path_t unit = critical_path(g);
  CHECK(unit.makespan == 3.0);
  CHECK(chain_of(g, unit) == "a#3 -> b#3 -> d#3");

  const critical_path_t w = critical_path(g, oracle_node_cost, oracle_edge_cost);
  CHECK_NEAR(w.makespan, 18.5, 1e-9);
  CHECK(chain_of(g, w) == "a#3 -> c#3 -> d#3");
}

TEST(oracle_streamed_iterations) {
  const allocation_t buf("buf", {index_t{32}});
  operation_t p("produce", 1);
  p.num_iters       = 4;
  p.access_patterns = {write(buf, streamed(32, 4))};
  operation_t c("consume", 1);
  c.num_iters       = 4;
  c.access_patterns = {read(buf, streamed(32, 4))};
  const graph_t g({p, c});

  CHECK(g.edges().size() == 4);

  const critical_path_t unit = critical_path(g);
  CHECK(unit.makespan == 2.0);
  CHECK(chain_of(g, unit) == "produce#0.3 -> consume#0.3");

  const critical_path_t w = critical_path(g, oracle_node_cost, oracle_edge_cost);
  CHECK_NEAR(w.makespan, 5.0, 1e-9);
  CHECK(chain_of(g, w) == "produce#0.3 -> consume#0.3");
}

TEST(oracle_overlap_report_text_matches_verbatim) {
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4)), write(y, contiguous(4))};
  operation_t c("c", 4);
  c.access_patterns = {read(y, contiguous(4))};
  const graph_t g({a, b, c});

  // 5/8 is 62.5%, which both languages round to even. A divergence here would
  // mean the formatting rules differ, not the analysis.
  const std::string expected = "serial (no overlap): 8.0\n"
                               "critical path      : 3.0\n"
                               "overlap saving     : 5.0 (62%)\n"
                               "critical chain     : a#3 -> b#3 -> c#3";
  CHECK(overlap_report(g, 8.0) == expected);
}

// ─── overlap report ───────────────────────────────────────────────────

TEST(overlap_report_quantifies_the_saving) {
  free_graph_t g;
  g.add_op("a", 1);
  g.add_op("b", 1);
  g.add_edge(wg_node_t{0, 0, 0}, wg_node_t{1, 0, 0}, "x", 1);

  const std::string r = overlap_report(g, 4.0);

  CHECK(r.find("serial (no overlap): 4.0") != std::string::npos);
  CHECK(r.find("critical path      : 2.0") != std::string::npos);
  CHECK(r.find("overlap saving     : 2.0 (50%)") != std::string::npos);
  CHECK(r.find("a#0 -> b#0") != std::string::npos);
}

TEST(overlap_report_tolerates_a_zero_baseline) {
  const free_graph_t g;
  const std::string r = overlap_report(g, 0.0);
  CHECK(r.find("(0%)") != std::string::npos);
}

ORIGAMI_TEST_MAIN()
