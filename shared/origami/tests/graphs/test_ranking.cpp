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

#include "origami/graphs/ranking.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "origami/graphs/core.hpp"
#include "origami/graphs/cost.hpp"
#include "origami/graphs/free_graph.hpp"
#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/** Every atom costs the same, so lane count alone decides the makespan. */
class flat_cost_t : public cost_model_t {
 public:
  double node_cost(const wg_node_t&) const override { return 1.0; }
  double edge_cost(const edge_t&) const override { return 0.0; }
};

/** Flat, but at a price per graph: stands in for a candidate's own tile cost. */
class scaled_cost_t : public cost_model_t {
 public:
  explicit scaled_cost_t(double per_node) : per_node_(per_node) {}

  double node_cost(const wg_node_t&) const override { return per_node_; }
  double edge_cost(const edge_t&) const override { return 0.0; }

 private:
  double per_node_;
};

/** a(4 wgs) -> b(2 wgs). */
graph_t fan_in(const std::string& name = "fan_in") {
  const allocation_t x("x", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 2);
  b.access_patterns = {read(x, contiguous(8))};
  return graph_t({a, b}, {}, name);
}

/** Two nodes pointing at each other: a graph no shipped backend can build. */
class cyclic_graph_t : public wg_graph_t {
 public:
  cyclic_graph_t() {
    edges_.push_back(edge_t{wg_node_t{0, 0, 0}, wg_node_t{1, 0, 0}, "x", 1});
    edges_.push_back(edge_t{wg_node_t{1, 0, 0}, wg_node_t{0, 0, 0}, "x", 1});
    successors_[wg_node_t{0, 0, 0}]   = {0};
    successors_[wg_node_t{1, 0, 0}]   = {1};
    predecessors_[wg_node_t{1, 0, 0}] = {0};
    predecessors_[wg_node_t{0, 0, 0}] = {1};
  }

  int num_operations() const override { return 2; }
  const std::string& op_name(int op) const override { return names_[static_cast<std::size_t>(op)]; }
  int op_index(const std::string& name) const override {
    if (name == names_[0]) return 0;
    if (name == names_[1]) return 1;
    throw std::out_of_range("cyclic_graph_t: no operation named '" + name + "'");
  }
  index_t wg_count(int) const override { return 1; }
  index_t iter_count(int) const override { return 1; }
  const std::vector<edge_t>& edges() const override { return edges_; }
  const std::string& name() const override { return name_; }

  const std::vector<std::size_t>& successor_edges(const wg_node_t& node) const override {
    return lookup(successors_, node);
  }
  const std::vector<std::size_t>& predecessor_edges(const wg_node_t& node) const override {
    return lookup(predecessors_, node);
  }

 private:
  const std::vector<std::size_t>& lookup(const wg_node_map_t<std::vector<std::size_t>>& from,
                                         const wg_node_t& node) const {
    const auto it = from.find(node);
    return it != from.end() ? it->second : empty_;
  }

  std::string names_[2] = {"a", "b"};
  std::string name_     = "cyclic";
  std::vector<edge_t> edges_;
  wg_node_map_t<std::vector<std::size_t>> successors_;
  wg_node_map_t<std::vector<std::size_t>> predecessors_;
  std::vector<std::size_t> empty_;
};

graph_config_t at_lanes(runtime_kind_t kind, int lanes, std::string name = {}) {
  graph_config_t config;
  config.runtime = kind;
  config.lanes   = lanes;
  config.scale   = 1.0;
  config.name    = std::move(name);
  return config;
}

}  // namespace

// ─── runtime_kind_t ───────────────────────────────────────────────────

TEST(runtime_kind_names_round_trip) {
  for (runtime_kind_t k : {runtime_kind_t::breadth_first,
                           runtime_kind_t::asap,
                           runtime_kind_t::depth_first,
                           runtime_kind_t::roofline,
                           runtime_kind_t::event_driven,
                           runtime_kind_t::xcd}) {
    CHECK(runtime_kind_from_name(runtime_kind_name(k)) == k);
  }
  CHECK(is_continuous(runtime_kind_t::asap) == false);
  CHECK(is_continuous(runtime_kind_t::xcd) == true);

  bool threw = false;
  try {
    runtime_kind_from_name("nonsense");
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

// ─── predict_latency ──────────────────────────────────────────────────

TEST(more_lanes_never_lengthen_the_prediction) {
  const graph_t g = fan_in();
  const flat_cost_t cost;

  double previous = 1e30;
  for (int lanes : {1, 2, 3, 4, 8}) {
    const double now = predict_latency(g, at_lanes(runtime_kind_t::roofline, lanes), &cost);
    CHECK(now <= previous);
    previous = now;
  }
}

TEST(integer_runtimes_need_no_cost_model_and_continuous_ones_say_so) {
  const graph_t g = fan_in();

  // Counting workgroups needs no hardware, which is the point of the integer
  // family: a shape, available before anyone has calibrated anything.
  CHECK(predict_latency(g, at_lanes(runtime_kind_t::asap, 2)) == 3.0);

  bool threw = false;
  try {
    predict_latency(g, at_lanes(runtime_kind_t::roofline, 2));
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(breadth_first_is_never_faster_than_asap) {
  const graph_t g = fan_in();
  for (int lanes : {1, 2, 4}) {
    CHECK(predict_latency(g, at_lanes(runtime_kind_t::breadth_first, lanes)) >=
          predict_latency(g, at_lanes(runtime_kind_t::asap, lanes)));
  }
}

TEST(a_runtime_override_bypasses_the_enum) {
  const graph_t g = fan_in();
  const flat_cost_t cost;

  cost_runtime_options_t opts;
  opts.lanes = 1;
  opts.scale = 1.0;

  // Every field is ignored in favour of the supplied policy, so a config that
  // says 8 lanes still predicts the one-lane answer.
  graph_config_t config   = at_lanes(runtime_kind_t::asap, 8);
  config.runtime_override = std::make_shared<const roofline_runtime_t>(cost, opts);
  CHECK(predict_latency(g, config, &cost) == 6.0);
}

// ─── rejection ────────────────────────────────────────────────────────

TEST(infeasible_candidates_are_rejected_rather_than_thrown) {
  const graph_t g = fan_in();

  graph_config_t bad_lanes = at_lanes(runtime_kind_t::asap, 0);
  CHECK(rejection_reason(g, bad_lanes).empty() == false);
  CHECK(predict_latency(g, bad_lanes) == kRejectedLatency);

  // A pool naming a lane the machine does not have is a mistake in the sweep,
  // not an instruction to fall back to every lane.
  graph_config_t bad_pool = at_lanes(runtime_kind_t::roofline, 4);
  bad_pool.lane_pool      = {{"a", {0, 9}}};
  CHECK(rejection_reason(g, bad_pool).find("lane 9") != std::string::npos);

  graph_config_t bad_op = at_lanes(runtime_kind_t::roofline, 4);
  bad_op.lane_pool      = {{"nonexistent", {0}}};
  CHECK(rejection_reason(g, bad_op).find("nonexistent") != std::string::npos);

  graph_config_t fine = at_lanes(runtime_kind_t::roofline, 4);
  fine.lane_pool      = {{"a", {0, 1}}, {"b", {2, 3}}};
  CHECK(rejection_reason(g, fine).empty());
}

TEST(a_cyclic_graph_is_rejected_by_every_candidate) {
  // Neither shipped backend can produce this: graph_t derives forward edges only
  // and free_graph_t::add_edge refuses one that runs backwards. The guard exists
  // for a third-party wg_graph_t — a trace importer, say — which the interface
  // permits and which has no such invariant, so the check is exercised through a
  // deliberately malformed one rather than left unreachable and untested.
  const cyclic_graph_t g;
  CHECK(rejection_reason(g, at_lanes(runtime_kind_t::asap, 2)).find("cycle") != std::string::npos);
  CHECK(predict_latency(g, at_lanes(runtime_kind_t::asap, 2)) == kRejectedLatency);
}

// ─── rank_configs ─────────────────────────────────────────────────────

TEST(results_come_back_best_first) {
  const graph_t g = fan_in();
  const flat_cost_t cost;
  const std::vector<graph_config_t> configs = {at_lanes(runtime_kind_t::roofline, 1, "one"),
                                               at_lanes(runtime_kind_t::roofline, 4, "four"),
                                               at_lanes(runtime_kind_t::roofline, 2, "two")};

  const std::vector<prediction_result_t> ranked = rank_configs(g, configs, &cost);

  CHECK(ranked.size() == 3);
  for (std::size_t i = 1; i < ranked.size(); ++i) {
    CHECK(ranked[i - 1].latency <= ranked[i].latency);
  }
  CHECK(ranked.front().config.name == "four");
  CHECK(ranked.front().graph_name == "fan_in");
  CHECK(select_config(g, configs, &cost).config.name == "four");
}

TEST(equal_predictions_keep_the_order_they_were_offered) {
  // Two schedules with the same makespan really are equivalent to this model, so
  // a stable sort is the honest tie-break; inventing a preference between them
  // would make the winner depend on something the prediction does not measure.
  const graph_t g = fan_in();
  const flat_cost_t cost;
  const std::vector<graph_config_t> configs = {at_lanes(runtime_kind_t::roofline, 4, "first"),
                                               at_lanes(runtime_kind_t::roofline, 8, "second")};

  const std::vector<prediction_result_t> ranked = rank_configs(g, configs, &cost);
  CHECK(ranked[0].latency == ranked[1].latency);
  CHECK(ranked[0].config.name == "first");
}

TEST(feasible_candidates_displace_rejected_ones) {
  const graph_t g = fan_in();
  const flat_cost_t cost;
  const std::vector<graph_config_t> configs = {at_lanes(runtime_kind_t::roofline, 0, "broken"),
                                               at_lanes(runtime_kind_t::roofline, 2, "fine")};

  const std::vector<prediction_result_t> ranked = rank_configs(g, configs, &cost);
  CHECK(ranked.size() == 1);
  CHECK(ranked.front().config.name == "fine");
  CHECK(ranked.front().rejected == false);
}

TEST(an_all_rejected_sweep_is_reported_rather_than_thrown) {
  // The contract copied from origami::rank_configs. Reporting "all of these are
  // impossible, and here is why" is more useful than an exception, because the
  // rejection strings say what to change.
  const graph_t g = fan_in();
  const flat_cost_t cost;
  const std::vector<graph_config_t> configs = {at_lanes(runtime_kind_t::roofline, 0, "a"),
                                               at_lanes(runtime_kind_t::roofline, -3, "b")};

  const std::vector<prediction_result_t> ranked = rank_configs(g, configs, &cost);
  CHECK(ranked.size() == 2);
  for (const prediction_result_t& r : ranked) {
    CHECK(r.rejected);
    CHECK(r.latency == kRejectedLatency);
    CHECK(r.rejection.empty() == false);
  }
  CHECK(ranked[0].config.name == "a");  // order preserved
}

TEST(an_empty_candidate_list_throws) {
  const graph_t g = fan_in();
  bool threw      = false;
  try {
    rank_configs(g, {});
  } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
}

TEST(topk_keeps_the_best_few) {
  const graph_t g = fan_in();
  const flat_cost_t cost;
  const std::vector<graph_config_t> configs = {at_lanes(runtime_kind_t::roofline, 1),
                                               at_lanes(runtime_kind_t::roofline, 2),
                                               at_lanes(runtime_kind_t::roofline, 4)};

  CHECK(select_topk_configs(g, configs, 2, &cost).size() == 2);
  CHECK(select_topk_configs(g, configs, 99, &cost).size() == 3);
  CHECK(select_topk_configs(g, configs, 0, &cost).empty());
  CHECK(select_topk_configs(g, configs, 1, &cost).front().latency ==
        select_config(g, configs, &cost).latency);
}

// ─── across graphs ────────────────────────────────────────────────────

TEST(each_graph_is_ranked_with_its_own_cost_model) {
  // The case that put cost on the graph. Two candidates differ in tile shape, so
  // they differ in price as well as in structure; ranking them against one
  // shared cost model would compare the wrong thing. Here the structures are
  // identical and only the prices differ, so the cheaper graph must win, which
  // it cannot do if a single cost applies to both.
  graph_t cheap = fan_in("cheap");
  graph_t dear  = fan_in("dear");
  cheap.set_cost(std::make_shared<const scaled_cost_t>(1.0));
  dear.set_cost(std::make_shared<const scaled_cost_t>(10.0));

  const std::vector<const wg_graph_t*> graphs = {&dear, &cheap};  // worst offered first
  const std::vector<graph_config_t> configs   = {at_lanes(runtime_kind_t::roofline, 8)};

  const std::vector<prediction_result_t> ranked = rank_graphs(graphs, configs);

  CHECK(ranked.size() == 2);
  CHECK(ranked[0].graph_name == "cheap");
  CHECK(ranked[1].latency == ranked[0].latency * 10.0);
}

TEST(an_explicit_cost_model_overrides_every_graphs_own) {
  graph_t cheap = fan_in("cheap");
  graph_t dear  = fan_in("dear");
  cheap.set_cost(std::make_shared<const scaled_cost_t>(1.0));
  dear.set_cost(std::make_shared<const scaled_cost_t>(10.0));

  const std::vector<const wg_graph_t*> graphs = {&dear, &cheap};
  const std::vector<graph_config_t> configs   = {at_lanes(runtime_kind_t::roofline, 8)};
  const flat_cost_t uniform;

  const std::vector<prediction_result_t> ranked = rank_graphs(graphs, configs, &uniform);

  // Priced the same way, the two are indistinguishable and the order given holds.
  CHECK(ranked[0].latency == ranked[1].latency);
  CHECK(ranked[0].graph_name == "dear");
}

TEST(a_continuous_candidate_with_no_cost_anywhere_throws) {
  const graph_t g = fan_in();  // no set_cost
  bool threw      = false;
  try {
    predict_latency(g, at_lanes(runtime_kind_t::roofline, 8));
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(ranking_across_graphs_prices_the_whole_cross_product) {
  const graph_t wide = fan_in("wide");
  const flat_cost_t cost;

  // A second graph with the same operations but no dependency between them, so
  // it schedules strictly faster at any lane count.
  free_graph_t loose("loose");
  loose.add_op("a", 4);
  loose.add_op("b", 2);

  const std::vector<const wg_graph_t*> graphs = {&wide, &loose};
  const std::vector<graph_config_t> configs   = {at_lanes(runtime_kind_t::roofline, 2, "two"),
                                                 at_lanes(runtime_kind_t::roofline, 8, "eight")};

  const std::vector<prediction_result_t> ranked = rank_graphs(graphs, configs, &cost);

  CHECK(ranked.size() == 4);
  for (std::size_t i = 1; i < ranked.size(); ++i) {
    CHECK(ranked[i - 1].latency <= ranked[i].latency);
  }
  // The winner is a graph-and-schedule pair, which is the whole point: neither
  // choice can be made without the other.
  CHECK(ranked.front().graph_name == "loose");
  CHECK(ranked.front().config.name == "eight");
  CHECK(ranked.front().graph_index == 1);

  CHECK(select_graph(graphs, configs, &cost).graph_name == "loose");
  CHECK(select_topk_graphs(graphs, configs, 2, &cost).size() == 2);
}

TEST(the_best_schedule_can_differ_between_graphs) {
  // The reason a cross product is needed rather than two separate rankings.
  const graph_t serial = fan_in("serial");
  free_graph_t parallel("parallel");
  parallel.add_op("a", 6);

  const flat_cost_t cost;
  const std::vector<const wg_graph_t*> graphs = {&serial, &parallel};
  const std::vector<graph_config_t> configs   = {at_lanes(runtime_kind_t::roofline, 1, "one"),
                                                 at_lanes(runtime_kind_t::roofline, 6, "six")};

  const std::vector<prediction_result_t> ranked = rank_graphs(graphs, configs, &cost);
  CHECK(ranked.front().graph_name == "parallel");
  CHECK(ranked.front().latency == 1.0);  // six independent atoms on six lanes
  CHECK(ranked.back().latency == 6.0);   // everything on one lane
}

TEST(empty_and_null_inputs_are_rejected) {
  const graph_t g                             = fan_in();
  const std::vector<const wg_graph_t*> graphs = {&g};
  const std::vector<graph_config_t> configs   = {at_lanes(runtime_kind_t::asap, 2)};

  bool threw = false;
  try {
    rank_graphs({}, configs);
  } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);

  threw = false;
  try {
    rank_graphs(graphs, {});
  } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);

  threw = false;
  try {
    rank_graphs({nullptr}, configs);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

ORIGAMI_TEST_MAIN()
