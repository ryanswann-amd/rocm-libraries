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

// The GEMM bridge suite. Unlike the rest of tests/graphs, this one needs HIP
// headers and origami's analytical model, which is exactly why it is a separate
// runner: if these assertions could be folded into origami-graphs-tests, the
// graphs library would not be HIP-free.

#include "origami/graphs/cost_gemm.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "origami/graphs/core.hpp"
#include "origami/graphs/cost_runtime.hpp"
#include "origami/graphs/ranking.hpp"
#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/** A nominal full-die MI300X, built without asking for a device. */
origami::hardware_t nominal_mi300x() {
  return origami::hardware_t::get_hardware_for_arch(origami::architecture_t::gfx942,
                                                    /*N_CU=*/304,
                                                    /*lds_capacity=*/64 * 1024,
                                                    /*rf_capacity=*/512 * 1024,
                                                    /*L2_capacity=*/4 * 1024 * 1024,
                                                    /*compute_clock_khz=*/2'100'000);
}

gemm_spec_t square_gemm(std::size_t m, std::size_t n, std::size_t k, std::size_t tile = 128) {
  gemm_spec_t spec;
  spec.problem.size    = {m, n, k};
  spec.problem.a_dtype = origami::data_type_t::BFloat16;
  spec.problem.b_dtype = origami::data_type_t::BFloat16;
  spec.problem.c_dtype = origami::data_type_t::BFloat16;
  spec.problem.d_dtype = origami::data_type_t::BFloat16;

  spec.config.mt        = {tile, tile, 64};
  spec.config.mi        = {16, 16, 16};
  spec.config.occupancy = 2;
  return spec;
}

/** gemm(8 wgs) -> all_reduce(4 wgs). */
graph_t gemm_then_reduce() {
  const allocation_t c("c", {index_t{64}});
  const allocation_t out("out", {index_t{64}});
  operation_t gemm("gemm", 8);
  gemm.access_patterns = {write(c, contiguous(8))};
  operation_t ar("all_reduce", 4);
  ar.access_patterns = {read(c, contiguous(16)), write(out, contiguous(16))};
  return graph_t({gemm, ar});
}

}  // namespace

TEST(a_gemm_tile_is_priced_in_seconds) {
  const origami::hardware_t hw = nominal_mi300x();
  const double seconds         = gemm_seconds(square_gemm(4096, 4096, 4096), hw);

  CHECK(seconds > 0.0);
  // A single macro-tile is microseconds at most; anything near a second would
  // mean the cycles-to-seconds conversion went the wrong way.
  CHECK(seconds < 1e-3);
}

TEST(more_contraction_costs_more) {
  const origami::hardware_t hw = nominal_mi300x();
  CHECK(gemm_seconds(square_gemm(4096, 4096, 8192), hw) >
        gemm_seconds(square_gemm(4096, 4096, 1024), hw));
}

TEST(a_bigger_tile_takes_longer_per_workgroup) {
  // Per *workgroup*, not per GEMM: a 256-wide tile does four times the work of a
  // 128-wide one, and there are correspondingly fewer of them.
  const origami::hardware_t hw = nominal_mi300x();
  CHECK(gemm_seconds(square_gemm(4096, 4096, 4096, 256), hw) >
        gemm_seconds(square_gemm(4096, 4096, 4096, 128), hw));
}

TEST(an_invalid_config_is_rejected) {
  const origami::hardware_t hw = nominal_mi300x();
  gemm_spec_t spec             = square_gemm(4096, 4096, 4096);
  spec.config.occupancy        = 0;

  bool threw = false;
  try {
    gemm_seconds(spec, hw);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(a_stopped_clock_is_rejected_rather_than_dividing_by_zero) {
  origami::hardware_t hw = nominal_mi300x();
  hw.compute_clock_ghz   = 0.0;

  bool threw = false;
  try {
    gemm_seconds(square_gemm(4096, 4096, 4096), hw);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(the_bridge_reports_the_gemm_arm) {
  const graph_t g              = gemm_then_reduce();
  const origami::hardware_t hw = nominal_mi300x();

  cost_table_t table(g);
  table.set("gemm", gemm_cost(square_gemm(4096, 4096, 4096), hw));

  // Reporting the arm the caller chose, rather than a generic "custom", is the
  // reason op_cost_t::tagged exists.
  CHECK(table.kind_of(0) == cost_kind_t::gemm);
  CHECK(cost_kind_name(*table.kind_of(0)) == std::string("gemm"));
  CHECK(table.node_cost(wg_node_t{0, 0, 0}) == gemm_seconds(square_gemm(4096, 4096, 4096), hw));
}

TEST(gemm_pricing_ignores_instantaneous_contention) {
  // Documented, not accidental: origami's tile model takes an active-CU count as
  // a property of the launch, so the scheduler's moment-to-moment occupancy has
  // nowhere to go. Substituting one for the other would make an event-driven
  // result look contention-aware when it is not.
  const graph_t g              = gemm_then_reduce();
  const origami::hardware_t hw = nominal_mi300x();

  cost_table_t table(g);
  table.set("gemm", gemm_cost(square_gemm(4096, 4096, 4096), hw));

  const wg_node_t tile{0, 0, 0};
  CHECK(table.node_cost_at(tile, 1) == table.node_cost_at(tile, 304));
}

TEST(active_cus_is_a_property_of_the_launch) {
  const origami::hardware_t hw = nominal_mi300x();

  gemm_spec_t few  = square_gemm(4096, 4096, 4096);
  few.active_cus   = 8;
  gemm_spec_t many = square_gemm(4096, 4096, 4096);
  many.active_cus  = 304;

  // Fewer concurrent CUs means less pressure on shared bandwidth, so a tile
  // scheduled into a quiet machine is predicted to be no slower.
  CHECK(gemm_seconds(few, hw) <= gemm_seconds(many, hw));
}

TEST(the_bridge_drives_a_schedule_and_a_ranking) {
  const graph_t g              = gemm_then_reduce();
  const origami::hardware_t hw = nominal_mi300x();

  cost_table_t table(g);
  table.set("gemm", gemm_cost(square_gemm(4096, 4096, 4096), hw));
  table.set("all_reduce", op_cost_t::from_roofline(roofline_spec_t::comm_step(1 << 20, 2.0, 2.0)));

  cost_runtime_options_t opts;
  opts.lanes              = 8;
  const cost_schedule_t s = roofline_runtime_t(table, opts).schedule(g);
  CHECK(s.order.size() == 12);
  CHECK(s.makespan() > 0.0);

  graph_config_t config;
  config.runtime = runtime_kind_t::roofline;
  config.lanes   = 8;
  CHECK(predict_latency(g, config, &table) == s.makespan());
}

ORIGAMI_TEST_MAIN()
