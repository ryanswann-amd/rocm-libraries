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

#include "origami/graphs/free_graph.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "origami/graphs/core.hpp"
#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/**
 * Compare two graphs through the backend-agnostic interface.
 *
 * This is the acceptance gate for the backend split: a snapshot of a derived
 * graph must be indistinguishable from it through everything a scheduler can
 * see. Comparing via `wg_graph_t` rather than the concrete types is the point —
 * it is exactly the surface the runtimes consume.
 */
bool graphs_agree(const wg_graph_t& a, const wg_graph_t& b) {
  if (a.num_operations() != b.num_operations()) return false;
  if (a.num_nodes() != b.num_nodes()) return false;
  if (a.edges().size() != b.edges().size()) return false;

  for (int op = 0; op < a.num_operations(); ++op) {
    if (a.op_name(op) != b.op_name(op)) return false;
    if (a.wg_count(op) != b.wg_count(op)) return false;
    if (a.iter_count(op) != b.iter_count(op)) return false;
  }

  for (std::size_t i = 0; i < a.edges().size(); ++i) {
    const edge_t& x = a.edges()[i];
    const edge_t& y = b.edges()[i];
    if (x.src != y.src || x.dst != y.dst) return false;
    if (x.allocation != y.allocation || x.weight != y.weight) return false;
  }

  if (a.nodes() != b.nodes()) return false;
  for (const wg_node_t& n : a.nodes()) {
    if (a.successor_edges(n) != b.successor_edges(n)) return false;
    if (a.predecessor_edges(n) != b.predecessor_edges(n)) return false;
    if (a.label(n) != b.label(n)) return false;
  }
  return true;
}

/** A small derived graph used as the snapshot source. */
graph_t make_derived() {
  const allocation_t x("x", {index_t{32}});
  const allocation_t y("y", {index_t{32}});

  operation_t a("a", 8);
  a.access_patterns = {write(x, contiguous(4))};

  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(8)), write(y, contiguous(8))};

  operation_t c("c", 2);
  c.access_patterns = {read(y, contiguous(16))};

  return graph_t({a, b, c}, {}, "derived");
}

}  // namespace

// ─── authoring ────────────────────────────────────────────────────────

TEST(operations_are_indexed_in_declaration_order) {
  free_graph_t g("hand_built");

  CHECK(g.add_op("fetch", 64) == 0);
  CHECK(g.add_op("consume", 16, 4) == 1);

  CHECK(g.num_operations() == 2);
  CHECK(g.name() == "hand_built");
  CHECK(g.op_name(0) == "fetch");
  CHECK(g.op_index("consume") == 1);
  CHECK(g.wg_count(0) == 64);
  CHECK(g.iter_count(1) == 4);
  CHECK(g.num_nodes() == 64 + 64);
  CHECK(g.edges().empty());
}

TEST(edges_can_be_stated_by_operation_name) {
  free_graph_t g;
  g.add_op("fetch", 64);
  g.add_op("consume", 8);

  g.add_edge({"fetch", 3}, {"consume", 0}, "buf", 512);
  g.add_edge({"fetch", 7}, {"consume", 0}, "buf", 256);

  CHECK(g.edges().size() == 2);
  CHECK(g.edges()[0].weight == 512);
  CHECK(g.edges()[0].allocation == "buf");
  CHECK(g.edges()[0].src == (wg_node_t{0, 3, 0}));

  CHECK(g.predecessor_edges(wg_node_t{1, 0, 0}).size() == 2);
  CHECK(g.successor_edges(wg_node_t{0, 3, 0}).size() == 1);
  CHECK(g.successor_edges(wg_node_t{0, 4, 0}).empty());
}

TEST(an_irregular_gather_needs_no_closed_form) {
  // The motivating case: a scatter whose targets follow no analytic pattern, so
  // there is nothing for the derivation to intersect.
  const std::vector<int> targets = {2, 0, 2, 1, 0, 2};

  free_graph_t g("gather");
  g.add_op("scatter", static_cast<index_t>(targets.size()));
  g.add_op("reduce", 3);

  for (std::size_t i = 0; i < targets.size(); ++i) {
    g.add_edge(wg_node_t{0, static_cast<int>(i), 0}, wg_node_t{1, targets[i], 0}, "bins", 1);
  }

  CHECK(g.edges().size() == 6);
  CHECK(g.predecessor_edges(wg_node_t{1, 0, 0}).size() == 2);
  CHECK(g.predecessor_edges(wg_node_t{1, 1, 0}).size() == 1);
  CHECK(g.predecessor_edges(wg_node_t{1, 2, 0}).size() == 3);
}

TEST(dependencies_within_one_operation_are_expressible) {
  // No access pattern can produce this, but a work-stealing kernel where one
  // workgroup waits on another really does have this shape.
  free_graph_t g;
  g.add_op("steal", 4);

  g.add_edge(wg_node_t{0, 0, 0}, wg_node_t{0, 3, 0}, "queue", 1);

  CHECK(g.edges().size() == 1);
  CHECK(g.successor_edges(wg_node_t{0, 0, 0}).size() == 1);
  CHECK(g.predecessor_edges(wg_node_t{0, 3, 0}).size() == 1);
}

TEST(from_grid_declares_grids_without_edges) {
  const free_graph_t g = free_graph_t::from_grid({{"a", 4, 1}, {"b", 2, 3}}, "grid");

  CHECK(g.num_operations() == 2);
  CHECK(g.num_nodes() == 4 + 6);
  CHECK(g.edges().empty());
  CHECK(g.nodes().size() == 10);
}

TEST(from_edges_builds_a_complete_graph_in_one_call) {
  const std::vector<edge_t> edges = {
      edge_t{wg_node_t{0, 0, 0}, wg_node_t{1, 0, 0}, "buf", 8},
      edge_t{wg_node_t{0, 1, 0}, wg_node_t{1, 0, 0}, "buf", 8},
  };
  const free_graph_t g = free_graph_t::from_edges({{"p", 2, 1}, {"c", 1, 1}}, edges);

  CHECK(g.edges().size() == 2);
  CHECK(g.predecessor_edges(wg_node_t{1, 0, 0}).size() == 2);
}

// ─── validation ───────────────────────────────────────────────────────

TEST(edges_must_advance_in_node_order) {
  free_graph_t g;
  g.add_op("a", 4);
  g.add_op("b", 4);

  bool backwards = false;
  try {
    g.add_edge(wg_node_t{1, 0, 0}, wg_node_t{0, 0, 0}, "buf", 1);
  } catch (const std::invalid_argument&) { backwards = true; }
  CHECK(backwards);

  bool self_loop = false;
  try {
    g.add_edge(wg_node_t{0, 2, 0}, wg_node_t{0, 2, 0}, "buf", 1);
  } catch (const std::invalid_argument&) { self_loop = true; }
  CHECK(self_loop);
}

TEST(edges_outside_a_declared_grid_are_rejected) {
  free_graph_t g;
  g.add_op("a", 4);
  g.add_op("b", 4, 2);

  bool bad_wg = false;
  try {
    g.add_edge(wg_node_t{0, 9, 0}, wg_node_t{1, 0, 0}, "buf", 1);
  } catch (const std::invalid_argument&) { bad_wg = true; }
  CHECK(bad_wg);

  bool bad_iter = false;
  try {
    g.add_edge(wg_node_t{0, 0, 0}, wg_node_t{1, 0, 5}, "buf", 1);
  } catch (const std::invalid_argument&) { bad_iter = true; }
  CHECK(bad_iter);

  bool bad_op = false;
  try {
    g.add_edge(wg_node_t{0, 0, 0}, wg_node_t{7, 0, 0}, "buf", 1);
  } catch (const std::invalid_argument&) { bad_op = true; }
  CHECK(bad_op);
}

TEST(unknown_operation_names_are_rejected) {
  free_graph_t g;
  g.add_op("a", 4);

  bool threw = false;
  try {
    g.add_edge({"a", 0}, {"nope", 0}, "buf", 1);
  } catch (const std::out_of_range&) { threw = true; }
  CHECK(threw);
}

TEST(free_graphs_reject_duplicate_operation_names) {
  free_graph_t g;
  g.add_op("a", 4);

  bool threw = false;
  try {
    g.add_op("a", 8);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(non_positive_grids_are_rejected) {
  free_graph_t g;

  bool threw = false;
  try {
    g.add_op("a", 4, 0);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

// ─── the snapshot identity gate ───────────────────────────────────────

TEST(a_snapshot_is_indistinguishable_from_the_graph_it_came_from) {
  const graph_t derived   = make_derived();
  const free_graph_t snap = derived.to_free();

  CHECK(!derived.edges().empty());
  CHECK(graphs_agree(derived, snap));
  CHECK(snap.name() == "derived");
}

TEST(a_snapshot_of_a_partially_dependent_graph_also_agrees) {
  const allocation_t buf("buf", {index_t{32}});

  operation_t produce("produce", 2);
  produce.num_iters       = 4;
  produce.access_patterns = {write(buf, streamed(16, 4))};

  operation_t consume("consume", 2);
  consume.num_iters       = 4;
  consume.access_patterns = {read(buf, streamed(16, 4))};

  const graph_t derived({produce, consume});
  const free_graph_t snap = derived.to_free();

  CHECK(derived.edges().size() == 8);
  CHECK(graphs_agree(derived, snap));
}

TEST(both_backends_answer_the_same_questions_polymorphically) {
  const graph_t derived   = make_derived();
  const free_graph_t snap = derived.to_free();

  // What a runtime actually holds.
  const wg_graph_t& as_derived = derived;
  const wg_graph_t& as_free    = snap;

  CHECK(as_derived.num_nodes() == as_free.num_nodes());
  CHECK(as_derived.summary().find("fine edges") != std::string::npos);
  CHECK(as_free.summary().find("fine edges") != std::string::npos);
  CHECK(as_derived.label(wg_node_t{0, 3, 0}) == as_free.label(wg_node_t{0, 3, 0}));
  CHECK(as_free.op_index("b") == 1);
}

ORIGAMI_TEST_MAIN()
