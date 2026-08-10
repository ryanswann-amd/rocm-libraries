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

#include "origami/graphs/core.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/** Edges running from one named operation to another. */
int count_edges(const graph_t& g, const std::string& from, const std::string& to) {
  const int fi = g.op_index(from);
  const int ti = g.op_index(to);
  int n        = 0;
  for (const edge_t& e : g.edges()) {
    if (e.src.op == fi && e.dst.op == ti) ++n;
  }
  return n;
}

/** Total weight on the edges between two named operations. */
index_t total_weight(const graph_t& g, const std::string& from, const std::string& to) {
  const int fi = g.op_index(from);
  const int ti = g.op_index(to);
  index_t w    = 0;
  for (const edge_t& e : g.edges()) {
    if (e.src.op == fi && e.dst.op == ti) w += e.weight;
  }
  return w;
}

}  // namespace

// ─── derivation basics ────────────────────────────────────────────────

TEST(one_to_one_tiles_derive_one_edge_each) {
  const allocation_t buf("buf", {index_t{16}});

  operation_t produce("produce", 4);
  produce.access_patterns = {write(buf, contiguous(4))};

  operation_t consume("consume", 4);
  consume.access_patterns = {read(buf, contiguous(4))};

  const graph_t g({produce, consume});

  // Matching tilings pair each producer with exactly one consumer.
  CHECK(g.edges().size() == 4);
  CHECK(count_edges(g, "produce", "consume") == 4);
  CHECK(total_weight(g, "produce", "consume") == 16);
  for (const edge_t& e : g.edges()) {
    CHECK(e.weight == 4);
    CHECK(e.src.wg == e.dst.wg);
    CHECK(e.allocation == "buf");
  }
}

TEST(coarser_consumer_tiles_fan_in) {
  const allocation_t buf("buf", {index_t{16}});

  operation_t produce("produce", 4);
  produce.access_patterns = {write(buf, contiguous(4))};

  // Each consumer swallows two producer tiles.
  operation_t consume("consume", 2);
  consume.access_patterns = {read(buf, contiguous(8))};

  const graph_t g({produce, consume});

  CHECK(g.edges().size() == 4);
  CHECK(total_weight(g, "produce", "consume") == 16);
  CHECK(g.predecessor_edges(wg_node_t{1, 0, 0}).size() == 2);
  CHECK(g.successor_edges(wg_node_t{0, 0, 0}).size() == 1);
}

TEST(disjoint_accesses_derive_no_edges) {
  const allocation_t a("a", {index_t{16}});
  const allocation_t b("b", {index_t{16}});

  operation_t produce("produce", 4);
  produce.access_patterns = {write(a, contiguous(4))};

  operation_t consume("consume", 4);
  consume.access_patterns = {read(b, contiguous(4))};

  const graph_t g({produce, consume});
  CHECK(g.edges().empty());
}

TEST(read_after_read_is_not_a_dependency) {
  const allocation_t buf("buf", {index_t{16}});

  operation_t first("first", 4);
  first.access_patterns = {read(buf, contiguous(4))};

  operation_t second("second", 4);
  second.access_patterns = {read(buf, contiguous(4))};

  const graph_t g({first, second});
  CHECK(g.edges().empty());
}

TEST(backwards_dependencies_are_not_derived) {
  const allocation_t buf("buf", {index_t{16}});

  // The consumer is listed first, so nothing flows to it: the operator order is
  // a list, and dependencies only run forwards through it.
  operation_t consume("consume", 4);
  consume.access_patterns = {read(buf, contiguous(4))};

  operation_t produce("produce", 4);
  produce.access_patterns = {write(buf, contiguous(4))};

  const graph_t g({consume, produce});
  CHECK(g.edges().empty());
}

// ─── multi-operation graphs ───────────────────────────────────────────
//
// Every graph in the reference implementation has exactly two operations, so
// none of the shapes below were ever exercised there.

TEST(three_stage_chain_links_each_neighbour) {
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});

  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};

  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4)), write(y, contiguous(4))};

  operation_t c("c", 4);
  c.access_patterns = {read(y, contiguous(4))};

  const graph_t g({a, b, c});

  CHECK(count_edges(g, "a", "b") == 4);
  CHECK(count_edges(g, "b", "c") == 4);
  // 'c' never touches x, so the chain does not short-circuit.
  CHECK(count_edges(g, "a", "c") == 0);
  CHECK(g.edges().size() == 8);
}

TEST(diamond_fans_out_and_back_in) {
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

  CHECK(count_edges(g, "a", "b") == 4);
  CHECK(count_edges(g, "a", "c") == 4);
  CHECK(count_edges(g, "b", "d") == 4);
  CHECK(count_edges(g, "c", "d") == 4);
  CHECK(count_edges(g, "a", "d") == 0);
  CHECK(count_edges(g, "b", "c") == 0);
  CHECK(g.edges().size() == 16);

  // Each sink node waits on one node from each branch.
  CHECK(g.predecessor_edges(wg_node_t{g.op_index("d"), 0, 0}).size() == 2);
}

TEST(a_late_consumer_gets_a_skip_edge_from_an_early_producer) {
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});

  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};

  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4)), write(y, contiguous(4))};

  // 'c' reads the intermediate buffer *and* the original one.
  operation_t c("c", 4);
  c.access_patterns = {read(y, contiguous(4)), read(x, contiguous(4))};

  const graph_t g({a, b, c});

  CHECK(count_edges(g, "a", "b") == 4);
  CHECK(count_edges(g, "b", "c") == 4);
  CHECK(count_edges(g, "a", "c") == 4);  // the skip edge
  CHECK(g.edges().size() == 12);
}

TEST(write_after_write_leaves_the_two_writers_unordered) {
  const allocation_t buf("buf", {index_t{16}});

  operation_t first("first", 4);
  first.access_patterns = {write(buf, contiguous(4))};

  operation_t second("second", 4);
  second.access_patterns = {write(buf, contiguous(4))};

  operation_t reader("reader", 4);
  reader.access_patterns = {read(buf, contiguous(4))};

  const graph_t g({first, second, reader});

  // Both writers feed the reader...
  CHECK(count_edges(g, "first", "reader") == 4);
  CHECK(count_edges(g, "second", "reader") == 4);
  // ...but nothing orders them against each other. This is a real gap in the
  // model, pinned here so a future change to close it is a deliberate one.
  CHECK(count_edges(g, "first", "second") == 0);
}

TEST(independent_branches_share_no_edges) {
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});

  operation_t pa("pa", 4);
  pa.access_patterns = {write(x, contiguous(4))};

  operation_t pb("pb", 4);
  pb.access_patterns = {write(y, contiguous(4))};

  operation_t ca("ca", 4);
  ca.access_patterns = {read(x, contiguous(4))};

  operation_t cb("cb", 4);
  cb.access_patterns = {read(y, contiguous(4))};

  const graph_t g({pa, pb, ca, cb});

  CHECK(count_edges(g, "pa", "ca") == 4);
  CHECK(count_edges(g, "pb", "cb") == 4);
  // Parallelism shows up as absent edges, not as list structure.
  CHECK(count_edges(g, "pa", "cb") == 0);
  CHECK(count_edges(g, "pb", "ca") == 0);
  CHECK(count_edges(g, "pa", "pb") == 0);
}

// ─── partial dependencies ─────────────────────────────────────────────

TEST(streamed_iterations_give_partial_dependencies) {
  const allocation_t buf("buf", {index_t{32}});

  operation_t produce("produce", 2);
  produce.num_iters       = 4;
  produce.access_patterns = {write(buf, streamed(16, 4))};

  operation_t consume("consume", 2);
  consume.num_iters       = 4;
  consume.access_patterns = {read(buf, streamed(16, 4))};

  const graph_t g({produce, consume});

  // Iteration it of a workgroup depends only on the same iteration of its
  // producer, not on the whole producing workgroup: 2 wgs x 4 iters = 8 edges,
  // rather than the 32 a monolithic dependency would give.
  CHECK(g.edges().size() == 8);
  for (const edge_t& e : g.edges()) {
    CHECK(e.src.wg == e.dst.wg);
    CHECK(e.src.it == e.dst.it);
    CHECK(e.weight == 4);
  }
  CHECK(g.num_nodes() == 16);
}

TEST(monolithic_consumer_depends_on_every_producer_iteration) {
  const allocation_t buf("buf", {index_t{32}});

  operation_t produce("produce", 2);
  produce.num_iters       = 4;
  produce.access_patterns = {write(buf, streamed(16, 4))};

  // No iteration structure: reads its whole 16-element block at once.
  operation_t consume("consume", 2);
  consume.access_patterns = {read(buf, contiguous(16))};

  const graph_t g({produce, consume});

  CHECK(g.edges().size() == 8);
  CHECK(g.predecessor_edges(wg_node_t{1, 0, 0}).size() == 4);
}

// ─── access-pattern plumbing ──────────────────────────────────────────

TEST(index_maps_accept_all_three_arities) {
  const allocation_t buf("buf", {index_t{8}});

  operation_t produce("produce", 2);
  produce.access_patterns = {
      write(buf,
            [](int wg) {
              return range_t{wg * 4, 1, 4};
            }),
  };

  operation_t consume("consume", 2);
  consume.access_patterns = {
      read(buf,
           [](int wg, const env_t& env) {
             return range_t{wg * env.at("B"), 1, 4};
           }),
  };

  operation_t again("again", 2);
  again.access_patterns = {
      read(buf,
           [](int wg, int it, const env_t&) {
             return range_t{wg * 4 + it, 1, 4};
           }),
  };

  env_t dims;
  dims["B"] = 4;

  const graph_t g({produce, consume, again}, dims);
  CHECK(count_edges(g, "produce", "consume") == 2);
  CHECK(count_edges(g, "produce", "again") == 2);
}

TEST(several_patterns_on_one_allocation_union) {
  const allocation_t buf("buf", {index_t{16}});

  operation_t produce("produce", 1);
  produce.access_patterns = {write(buf, contiguous(4)), write(buf, contiguous(4, 8))};

  operation_t consume("consume", 4);
  consume.access_patterns = {read(buf, contiguous(4))};

  const graph_t g({produce, consume});

  // The single producer node writes [0,4) and [8,12), reaching consumers 0 and 2.
  CHECK(g.edges().size() == 2);
  CHECK(total_weight(g, "produce", "consume") == 8);
}

// ─── graph queries and validation ─────────────────────────────────────

TEST(symbolic_grids_resolve_from_bindings) {
  const scalar_expr_t m = sym("M");
  const allocation_t c("c", {m});

  operation_t gemm("gemm", ceil_div(m, 64));
  gemm.access_patterns = {write(c, contiguous(64))};

  operation_t reduce("reduce", ceil_div(m, 128));
  reduce.access_patterns = {read(c, contiguous(128))};

  env_t dims;
  dims["M"] = 512;

  const graph_t g({gemm, reduce}, dims);

  CHECK(g.wg_count(0) == 8);
  CHECK(g.wg_count(1) == 4);
  CHECK(g.num_nodes() == 12);
  CHECK(g.edges().size() == 8);  // each reduce tile covers two gemm tiles
}

TEST(nodes_are_listed_in_canonical_order) {
  const allocation_t buf("buf", {index_t{4}});

  operation_t a("a", 2);
  a.num_iters       = 2;
  a.access_patterns = {write(buf, contiguous(2))};

  operation_t b("b", 1);
  b.access_patterns = {read(buf, contiguous(4))};

  const graph_t g({a, b});
  const std::vector<wg_node_t> n = g.nodes();

  CHECK(n.size() == 5);
  CHECK(n[0] == (wg_node_t{0, 0, 0}));
  CHECK(n[1] == (wg_node_t{0, 0, 1}));
  CHECK(n[2] == (wg_node_t{0, 1, 0}));
  CHECK(n[3] == (wg_node_t{0, 1, 1}));
  CHECK(n[4] == (wg_node_t{1, 0, 0}));
}

TEST(node_labels_match_the_reference_rendering) {
  const allocation_t buf("buf", {index_t{4}});

  operation_t gemm("gemm", 2);
  gemm.num_iters       = 2;
  gemm.access_patterns = {write(buf, contiguous(2))};

  const graph_t g({gemm});

  CHECK(g.label(wg_node_t{0, 1, 0}) == "gemm#1");
  CHECK(g.label(wg_node_t{0, 1, 3}) == "gemm#1.3");
}

TEST(conflicting_allocation_shapes_are_rejected) {
  const allocation_t small("buf", {index_t{16}});
  const allocation_t large("buf", {index_t{32}});

  operation_t a("a", 1);
  a.access_patterns = {write(small, contiguous(16))};

  operation_t b("b", 1);
  b.access_patterns = {read(large, contiguous(32))};

  bool threw = false;
  try {
    const graph_t g({a, b});
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(duplicate_operation_names_are_rejected) {
  operation_t a("same", 1);
  operation_t b("same", 1);

  bool threw = false;
  try {
    const graph_t g({a, b});
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(unbound_dimensions_are_reported_with_the_operation) {
  operation_t a("gemm", sym("M"));

  bool threw = false;
  try {
    const graph_t g({a});
  } catch (const std::out_of_range& e) {
    threw = std::string{e.what()}.find("gemm") != std::string::npos;
  }
  CHECK(threw);
}

TEST(negative_workgroup_counts_are_rejected) {
  operation_t a("bad", index_t{0} - index_t{4});

  bool threw = false;
  try {
    const graph_t g({a});
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(summary_reports_grids_and_edges) {
  const allocation_t buf("buf", {sym("M")});

  operation_t gemm("gemm", ceil_div(sym("M"), 64));
  gemm.access_patterns = {write(buf, contiguous(64))};

  operation_t reduce("reduce", 2);
  reduce.num_iters       = 2;
  reduce.access_patterns = {read(buf, contiguous(128))};

  env_t dims;
  dims["M"] = 256;

  const graph_t g({gemm, reduce}, dims);
  const std::string s = g.summary();

  CHECK(s.find("2 ops") != std::string::npos);
  CHECK(s.find("M: 256") != std::string::npos);
  CHECK(s.find("'gemm': 4 wgs") != std::string::npos);
  CHECK(s.find("x 2 iters") != std::string::npos);
  CHECK(s.find("ceil_div(M, 64)") != std::string::npos);
}

ORIGAMI_TEST_MAIN()
