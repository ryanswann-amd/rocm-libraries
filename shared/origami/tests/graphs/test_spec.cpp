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

#include "origami/graphs/spec.hpp"

#include <cmath>
#include <stdexcept>

#include "origami/graphs/ranking.hpp"
#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/**
 * A tiled GEMM, described once and instantiated many times.
 *
 * Deliberately the doc's K-complete shape: one node per output tile, so the
 * cost of a node is the cost of a whole tile and the grid is small enough to
 * derive quickly.
 */
graph_spec_t tiled_gemm() {
  const scalar_expr_t M = problem_sym("M");
  const scalar_expr_t N = problem_sym("N");
  const scalar_expr_t K = problem_sym("K");

  const scalar_expr_t BM = config_sym("BM");
  const scalar_expr_t BN = config_sym("BN");

  const allocation_t a("A", {M, K});
  const allocation_t b("B", {K, N});
  const allocation_t c("C", {M, N});

  const scalar_expr_t n_tiles = ceil_div(M, BM) * ceil_div(N, BN);

  operation_t gemm("gemm", n_tiles);
  gemm.access_patterns = {
      read(a, rows(BM, K)),
      read(b, contiguous(BN * K)),
      write(c, rows(BM, N)),
  };

  // Priced as a roofline over the tile, in the doc's shape: exact integer
  // arithmetic for the tile's work, real arithmetic for the rates. Both rates
  // are the per-cycle spellings `hardware_params` publishes, so the result is
  // cycles with no `clock_hz` in sight — see that function's doc for why that
  // is the preferred form.
  const scalar_expr_t tiles_n = ceil_div(N, BN);
  gemm.node_cost              = [=](const wg_node_t& node) -> cost_expr_t {
    const scalar_expr_t tile_row = scalar_expr_t{index_t{node.wg}} / tiles_n;
    const scalar_expr_t tile_col = mod(scalar_expr_t{index_t{node.wg}}, tiles_n);

    // Boundary tiles are charged for what they actually own.
    const scalar_expr_t rows_owned = minimum(BM, M - tile_row * BM);
    const scalar_expr_t cols_owned = minimum(BN, N - tile_col * BN);

    const scalar_expr_t flops = scalar_expr_t{index_t{2}} * rows_owned * cols_owned * K;
    const scalar_expr_t bytes = (rows_owned * K + cols_owned * K + rows_owned * cols_owned) *
                                config_sym("bytes_per_element");

    // `peak_flops_per_cu_per_cycle` is the per-cycle sibling of
    // `peak_flops_per_cu` (both are `2.0 * valu_rate`, the fused-multiply-add
    // over lane-elements); this one carries no clock, matching the roofline
    // arm's compute roof.
    const cost_expr_t compute =
        cost_expr_t{flops} / (hardware_sym("peak_flops_per_cu_per_cycle") *
                              cost_sym(scope_t::config, "compute_efficiency"));
    const cost_expr_t memory =
        cost_expr_t{bytes} * runtime_sym("active_cus") / hardware_sym("hbm_read_bw");

    return maximum(compute, memory);
  };

  return graph_spec_t{{gemm}};
}

/**
 * A tile copy, priced purely by bandwidth.
 *
 * The GEMM above is compute-bound at every occupancy worth testing, so it
 * cannot show contention doing anything. This one is bandwidth-bound by
 * construction, which is what makes the runtime scope observable.
 */
graph_spec_t memory_bound_copy() {
  const scalar_expr_t N  = problem_sym("N");
  const scalar_expr_t BN = config_sym("BN");

  const allocation_t src("src", {N});
  const allocation_t dst("dst", {N});

  operation_t copy("copy", ceil_div(N, BN));
  copy.access_patterns = {read(src, contiguous(BN)), write(dst, contiguous(BN))};
  copy.node_cost       = [=](const wg_node_t&) -> cost_expr_t {
    const scalar_expr_t bytes = BN * config_sym("bytes_per_element") * scalar_expr_t{index_t{2}};
    // Per-cycle bandwidth again, so this is cycles with no clock involved.
    return cost_expr_t{bytes} * runtime_sym("active_cus") / hardware_sym("hbm_read_bw");
  };

  return graph_spec_t{{copy}};
}

problem_t square(index_t n) {
  problem_t p;
  p.set("M", n);
  p.set("N", n);
  p.set("K", n);
  return p;
}

config_t tile(index_t bm, index_t bn) {
  config_t c;
  c.set("BM", bm);
  c.set("BN", bn);
  c.set("bytes_per_element", index_t{2});
  c.set("compute_efficiency", 0.80);
  return c;
}

origami::comm::hardware_t mi300x() {
  origami::comm::gpu_topology_t topo;
  topo.arch              = origami::architecture_t::gfx942;
  topo.num_cu            = 304;
  topo.num_xcd           = 8;
  topo.cu_per_xcd        = 38;
  topo.l2_capacity_bytes = 4ULL * 1024ULL * 1024ULL;

  return origami::comm::make_system(
             origami::comm::get_arch_ceilings(origami::architecture_t::gfx942), topo, 2.1)
      .gpu;
}

bool near(double a, double b) { return std::abs(a - b) < 1e-9; }

/** The graph's own model, which `attach_expr_cost` is expected to have set. */
const cost_model_t& cycles_of(const graph_t& g) {
  const cost_model_t* cost = g.cost();
  if (cost == nullptr) throw std::logic_error("cycles_of: graph carries no cost model");
  return *cost;
}

}  // namespace

// ─── instantiation ────────────────────────────────────────────────────

TEST(one_spec_instantiates_to_many_graphs) {
  const graph_spec_t spec = tiled_gemm();

  const std::shared_ptr<graph_t> coarse = spec.instantiate(square(8192), tile(256, 256), "256x256");
  const std::shared_ptr<graph_t> fine   = spec.instantiate(square(8192), tile(128, 128), "128x128");

  // The same description, two grids: 32x32 tiles against 64x64.
  CHECK(coarse->wg_count(0) == 1024);
  CHECK(fine->wg_count(0) == 4096);
  CHECK(coarse->name() == "256x256");
  CHECK(fine->name() == "128x128");
}

TEST(an_instantiated_graph_remembers_what_it_was_built_from) {
  // The graph has to keep these: a cost expression is evaluated later, at
  // ranking, and by then the problem and config are otherwise long gone.
  const std::shared_ptr<graph_t> g = tiled_gemm().instantiate(square(4096), tile(128, 128));

  CHECK(g->problem().index_at("M", scope_t::problem) == 4096);
  CHECK(g->config().index_at("BM", scope_t::config) == 128);
  CHECK(near(g->config().number_at("compute_efficiency", scope_t::config), 0.80));
}

TEST(the_two_namespaces_do_not_collide) {
  // A problem dimension and a config parameter may share a name and stay
  // distinct, which is the point of scoping them.
  problem_t p;
  p.set("size", index_t{1024});
  config_t c;
  c.set("size", index_t{64});

  operation_t op("op", ceil_div(problem_sym("size"), config_sym("size")));
  const graph_spec_t spec{{op}};

  CHECK(spec.instantiate(p, c)->wg_count(0) == 16);
}

TEST(an_unbound_config_parameter_says_which_scope_it_wanted) {
  problem_t p;
  p.set("M", index_t{1024});

  operation_t op("op", ceil_div(problem_sym("M"), config_sym("BM")));
  const graph_spec_t spec{{op}};

  bool threw = false;
  try {
    spec.instantiate(p, config_t{});
  } catch (const std::out_of_range&) { threw = true; }
  CHECK(threw);
}

TEST(an_empty_spec_is_rejected) {
  bool threw = false;
  try {
    graph_spec_t{}.instantiate(square(64), tile(32, 32));
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

// ─── pricing ──────────────────────────────────────────────────────────

TEST(a_spec_carried_cost_prices_every_node) {
  const std::shared_ptr<graph_t> g = tiled_gemm().instantiate(square(4096), tile(128, 128));

  // Unpriced until hardware arrives: a graph is device-independent.
  CHECK(g->cost() == nullptr);
  CHECK(attach_expr_cost(g, hardware_params(mi300x())));
  CHECK(g->cost() != nullptr);

  const double alone = cycles_of(*g).node_cycles(wg_node_t{0, 0, 0});
  CHECK(alone > 0.0);
  // A few hundred thousand cycles (actual: ~6.2e5) for one 128x128x4096 tile
  // at mi300x()'s 2.1 GHz; a value up near a whole second's worth of cycles
  // (2.1e9) would mean the rate the expression divides by is wrong.
  CHECK(alone < 2.1e6);
}

TEST(a_node_costs_more_under_contention) {
  // The reason cost is an expression rather than a number: the same node is
  // worth a different amount depending on what else is resident.
  problem_t p;
  p.set("N", index_t{1 << 20});
  config_t c;
  c.set("BN", index_t{1024});
  c.set("bytes_per_element", index_t{2});

  const std::shared_ptr<graph_t> g = memory_bound_copy().instantiate(p, c);
  attach_expr_cost(g, hardware_params(mi300x()));

  const double alone   = cycles_of(*g).node_cycles_at(wg_node_t{0, 0, 0}, 1);
  const double crowded = cycles_of(*g).node_cycles_at(wg_node_t{0, 0, 0}, 256);

  CHECK(alone > 0.0);
  CHECK(near(crowded, alone * 256.0));
}

TEST(node_cost_without_an_occupancy_prices_the_node_alone) {
  problem_t p;
  p.set("N", index_t{1 << 20});
  config_t c;
  c.set("BN", index_t{1024});
  c.set("bytes_per_element", index_t{2});

  const std::shared_ptr<graph_t> g = memory_bound_copy().instantiate(p, c);
  attach_expr_cost(g, hardware_params(mi300x()));

  CHECK(near(cycles_of(*g).node_cycles(wg_node_t{0, 0, 0}),
             cycles_of(*g).node_cycles_at(wg_node_t{0, 0, 0}, 1)));
}

TEST(a_boundary_tile_is_charged_for_what_it_owns) {
  // 300 = two 128-tiles plus a 44-wide remainder, so the last tile in a row is
  // cheaper than a full one.
  problem_t p;
  p.set("M", index_t{300});
  p.set("N", index_t{300});
  p.set("K", index_t{256});

  const std::shared_ptr<graph_t> g = tiled_gemm().instantiate(p, tile(128, 128));
  attach_expr_cost(g, hardware_params(mi300x()));

  CHECK(g->wg_count(0) == 9);  // 3x3 tiles
  const double full     = cycles_of(*g).node_cycles(wg_node_t{0, 0, 0});
  const double boundary = cycles_of(*g).node_cycles(wg_node_t{0, 8, 0});
  CHECK(boundary < full);
}

TEST(an_unpriced_spec_attaches_nothing) {
  operation_t op("op", index_t{4});
  const std::shared_ptr<graph_t> g = graph_spec_t{{op}}.instantiate(problem_t{}, config_t{});

  CHECK(!attach_expr_cost(g, hardware_params(mi300x())));
  CHECK(g->cost() == nullptr);
}

// ─── ranking against a machine ────────────────────────────────────────

TEST(ranking_binds_hardware_and_prices_a_spec_without_a_cost_argument) {
  const graph_spec_t spec = tiled_gemm();

  const std::shared_ptr<graph_t> coarse = spec.instantiate(square(4096), tile(256, 256), "256x256");
  const std::shared_ptr<graph_t> fine   = spec.instantiate(square(4096), tile(128, 128), "128x128");

  graph_config_t schedule;
  schedule.runtime = runtime_kind_t::event_driven;
  schedule.name    = "event";

  const std::vector<const wg_graph_t*> graphs{coarse.get(), fine.get()};
  const std::vector<prediction_result_t> ranked = rank_graphs(graphs, {schedule}, mi300x());

  CHECK(ranked.size() == 2);
  CHECK(ranked[0].latency <= ranked[1].latency);
  // Every candidate was priced, which only happens if hardware reached the
  // specification's cost expressions.
  CHECK(ranked[0].latency > 0.0);
  CHECK(ranked[1].latency > 0.0);
}

TEST(ranking_fills_lanes_from_the_device) {
  const std::shared_ptr<graph_t> g = tiled_gemm().instantiate(square(2048), tile(256, 256));

  graph_config_t schedule;
  schedule.runtime = runtime_kind_t::event_driven;

  const std::vector<const wg_graph_t*> graphs{g.get()};
  const std::vector<prediction_result_t> ranked = rank_graphs(graphs, {schedule}, mi300x());

  CHECK(ranked.size() == 1);
  CHECK(ranked[0].config.lanes.has_value());
  CHECK(*ranked[0].config.lanes == 304);
}

TEST(an_explicit_lane_count_survives_the_device) {
  const std::shared_ptr<graph_t> g = tiled_gemm().instantiate(square(2048), tile(256, 256));

  graph_config_t schedule;
  schedule.runtime = runtime_kind_t::event_driven;
  schedule.lanes   = 8;

  const std::vector<const wg_graph_t*> graphs{g.get()};
  const std::vector<prediction_result_t> ranked = rank_graphs(graphs, {schedule}, mi300x());

  CHECK(*ranked[0].config.lanes == 8);
}

TEST(rank_graphs_prices_a_spec_to_an_exact_cycle_count) {
  // Every other ranking test here only asserts latency > 0 and relative
  // ordering, which is exactly the gap a real bug once hid behind:
  // prediction_result_t::latency reported cycles x 1e6 under a "microseconds"
  // label for several tasks before anyone noticed, because nothing pinned an
  // absolute value at this entry point. That mechanism is deleted now, but
  // this closes the loop with one hand-computed number.
  //
  // square(128) with a 128x128 tile makes a single, non-boundary tile (one
  // wg, no scheduling order to reason about), and lanes=1 pins active_cus to
  // 1, so the roofline formula in tiled_gemm() can be evaluated by hand:
  //   flops   = 2*BM*BN*K              = 2*128*128*128     = 4194304
  //   bytes   = (BM*K+BN*K+BM*BN)*2    = (16384+16384+16384)*2 = 98304
  //   compute = flops / (peak_flops_per_cu_per_cycle * compute_efficiency)
  //           = 4194304 / (2*2.10*64 * 0.80)              ~= 19504.76 cycles
  //   memory  = bytes / hbm_read_bw = bytes / (4730.0/2.1) ~=    43.64 cycles
  // Compute dominates, so the node's cost -- and, with one node and one lane,
  // the whole schedule's makespan -- is the compute figure above.
  const std::shared_ptr<graph_t> g = tiled_gemm().instantiate(square(128), tile(128, 128));

  graph_config_t schedule;
  schedule.runtime = runtime_kind_t::event_driven;
  schedule.lanes   = 1;

  const std::vector<const wg_graph_t*> graphs{g.get()};
  const std::vector<prediction_result_t> ranked = rank_graphs(graphs, {schedule}, mi300x());

  CHECK(ranked.size() == 1);
  CHECK_NEAR(ranked[0].latency, 19504.761904761905, 1e-3);
}

ORIGAMI_TEST_MAIN()
