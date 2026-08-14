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

#include "origami/graphs/cost_expr.hpp"

#include <cmath>
#include <stdexcept>

#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/** A context with one binding in each of the four scopes. */
cost_context_t four_scopes() {
  cost_context_t ctx;
  ctx.problem.set("K", index_t{8192});
  ctx.config.set("efficiency", 0.5);
  ctx.hardware.set("rate", 2.0);
  ctx.runtime.set("active_cus", index_t{16});
  return ctx;
}

bool near(double a, double b) { return std::abs(a - b) < 1e-9; }

}  // namespace

// ─── the four scopes ──────────────────────────────────────────────────

TEST(cost_expr_resolves_every_scope) {
  const cost_context_t ctx = four_scopes();

  CHECK(near(cost_expr_t{problem_sym("K")}.eval(ctx), 8192.0));
  CHECK(near(cost_sym(scope_t::config, "efficiency").eval(ctx), 0.5));
  CHECK(near(hardware_sym("rate").eval(ctx), 2.0));
  CHECK(near(runtime_sym("active_cus").eval(ctx), 16.0));
}

TEST(an_unbound_cost_symbol_names_its_scope) {
  const cost_context_t ctx = four_scopes();
  bool threw               = false;
  try {
    hardware_sym("peak_flops_per_cu").eval(ctx);
  } catch (const std::out_of_range&) { threw = true; }
  CHECK(threw);
}

TEST(index_expressions_cannot_reach_the_hardware_scope) {
  // Hardware is unknown when an access pattern is resolved, so the integer
  // expression must refuse the scope outright rather than fail later.
  bool threw = false;
  try {
    scoped_sym(scope_t::hardware, "rate");
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

// ─── arithmetic ───────────────────────────────────────────────────────

TEST(cost_expr_arithmetic_evaluates) {
  const cost_context_t ctx = four_scopes();
  const cost_expr_t rate   = hardware_sym("rate");

  CHECK(near((rate + 1.0).eval(ctx), 3.0));
  CHECK(near((rate - 0.5).eval(ctx), 1.5));
  CHECK(near((rate * 4.0).eval(ctx), 8.0));
  CHECK(near((rate / 4.0).eval(ctx), 0.5));
  CHECK(near(minimum(rate, 1.0).eval(ctx), 1.0));
  CHECK(near(maximum(rate, 1.0).eval(ctx), 2.0));
}

TEST(dividing_by_zero_throws_rather_than_returning_infinity) {
  // An infinity would survive a ranking as a merely very slow candidate; a
  // zero rate is a modelling mistake and should stop the run.
  const cost_context_t ctx = four_scopes();
  bool threw               = false;
  try {
    (hardware_sym("rate") / 0.0).eval(ctx);
  } catch (const std::domain_error&) { threw = true; }
  CHECK(threw);
}

// ─── the integer bridge ───────────────────────────────────────────────

TEST(an_embedded_index_expression_evaluates_exactly_then_widens) {
  cost_context_t ctx = four_scopes();
  ctx.problem.set("N", index_t{4096});
  ctx.config.set("BN", index_t{128});

  // ceil_div is integer arithmetic: 4096/128 is 32 exactly, not 31.999...
  const cost_expr_t tiles = ceil_div(problem_sym("N"), config_sym("BN"));
  CHECK(near(tiles.eval(ctx), 32.0));

  // And the rounding is the integer rounding, not what a double division gives.
  ctx.problem.set("N", index_t{4097});
  CHECK(near(tiles.eval(ctx), 33.0));
}

TEST(an_embedded_expression_reports_its_symbols) {
  const cost_expr_t e = cost_expr_t{problem_sym("M") * config_sym("BM")} * hardware_sym("rate");

  const std::set<scoped_symbol_t> s = e.scoped_symbols();
  CHECK(s.size() == 3);
  CHECK(s.count(scoped_symbol_t{scope_t::problem, "M"}) == 1);
  CHECK(s.count(scoped_symbol_t{scope_t::config, "BM"}) == 1);
  CHECK(s.count(scoped_symbol_t{scope_t::hardware, "rate"}) == 1);
}

// ─── the parameter maps ───────────────────────────────────────────────

TEST(an_integral_real_is_accepted_where_an_integer_is_required) {
  cost_context_t ctx = four_scopes();
  ctx.config.set("BM", 128.0);

  CHECK(config_sym("BM").eval(ctx.index_context()) == 128);
}

TEST(a_fractional_value_is_rejected_where_an_integer_is_required) {
  // The reason the two alternatives are kept apart: an efficiency knob reaching
  // index arithmetic would silently truncate a tile bound.
  cost_context_t ctx = four_scopes();
  ctx.config.set("BM", 127.5);

  bool threw = false;
  try {
    config_sym("BM").eval(ctx.index_context());
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(an_integer_binding_reads_as_a_real) {
  const cost_context_t ctx = four_scopes();
  CHECK(near(cost_sym(scope_t::problem, "K").eval(ctx), 8192.0));
}

TEST(a_lone_symbol_crossing_into_a_cost_expression_reads_as_a_real) {
  // Exactness is a property of arithmetic, and a lone symbol performs none, so
  // a fractional efficiency may be named with the same builder that names a
  // tile size. Only arithmetic done in the integer expression stays integral.
  cost_context_t ctx = four_scopes();
  ctx.config.set("compute_efficiency", 0.8);

  const cost_expr_t efficiency = config_sym("compute_efficiency");
  CHECK(near(efficiency.eval(ctx), 0.8));
}

TEST(a_compound_expression_crossing_into_a_cost_expression_stays_exact) {
  cost_context_t ctx = four_scopes();
  ctx.problem.set("N", index_t{7});
  ctx.config.set("BN", index_t{2});

  // Integer division truncates, and continues to after the promotion: 7/2 is
  // 3, not 3.5.
  const cost_expr_t tiles = floor_div(problem_sym("N"), config_sym("BN"));
  CHECK(near(tiles.eval(ctx), 3.0));
}

// ─── the doc's roofline, in miniature ─────────────────────────────────

TEST(a_roofline_prices_the_slower_of_two_arms) {
  cost_context_t ctx;
  ctx.problem.set("flops", index_t{1000});
  ctx.problem.set("bytes", index_t{100});
  // Compute is clock-free: FLOP/cycle is the rate origami states hardware in,
  // so no frequency enters this arm at all (see hardware_params's doc).
  ctx.hardware.set("flops_per_cycle", 1000.0);
  // Bandwidth is usually quoted per second, so reaching cycles from it needs
  // one multiplication by the clock — the legitimate use `hardware_sym`
  // keeps `clock_hz` around for.
  ctx.hardware.set("bytes_per_second", 50.0);
  ctx.hardware.set("clock_hz", 2.0);
  ctx.runtime.set("active_cus", index_t{1});

  const cost_expr_t compute = cost_expr_t{problem_sym("flops")} / hardware_sym("flops_per_cycle");
  const cost_expr_t memory  = cost_expr_t{problem_sym("bytes")} * runtime_sym("active_cus") /
                             hardware_sym("bytes_per_second") * hardware_sym("clock_hz");
  const cost_expr_t roofline = maximum(compute, memory);

  // Alone: compute is 1 cycle; memory is (100/50) s worth of bytes, times the
  // 2 Hz clock, i.e. 4 cycles — so memory is the slower arm.
  CHECK(near(roofline.eval(ctx), 4.0));

  // Contention is the whole reason cost is deferred: the same node costs more
  // when more CUs share the bus.
  ctx.runtime.set("active_cus", index_t{8});
  CHECK(near(roofline.eval(ctx), 32.0));
}

ORIGAMI_TEST_MAIN()
