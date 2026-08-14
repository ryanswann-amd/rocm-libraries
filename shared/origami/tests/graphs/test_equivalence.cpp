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

/**
 * @file
 * @brief origami::graphs — the regression net for the cycles refactor.
 *
 * The refactor made cycles the unit of cost throughout `simulate()`, on the
 * grounds that a frequency is not a property of a kernel: it moves with DVFS,
 * with the part, and with what else is resident. That move is only safe
 * because it did not silently change behaviour, and it replaced two prior
 * schedulers (an integer-timestep one, and a continuous cost-aware one) with
 * one, on the strength of four claims those old schedulers made only in
 * prose. Each test below pins one claim as an executable equivalence:
 *
 *  1. Unit cost reproduces the old integer timesteps — every start time is an
 *     exact integer when every node costs exactly one cycle, which is what
 *     made the deleted timestep scheduler's view of the world meaningful.
 *  2. Repricing equals static pricing for a contention-free model —
 *     `reprice_on_dispatch` is what absorbed the old event-driven runtime,
 *     and a model that never looks at `node_cycles_at`'s `active_cus` must be
 *     unaffected by turning it on.
 *  3. Unlimited lanes reduce ASAP to the dependency depth — with nothing to
 *     contend for, the makespan is the critical path and nothing else.
 *  4. `to_seconds` recovers the numbers the old seconds-era models returned —
 *     2100 cycles at 2.1 GHz is one microsecond, the boundary the cycles
 *     design rests on.
 *
 * A failure here does not mean a test is stale; it means one of the four
 * collapses this refactor made was unsound, and the two schedulers it
 * replaced were not actually equivalent to the one that replaced them.
 */

#include "origami/graphs/simulate.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "origami/graphs/cost_model.hpp"
#include "origami/graphs/free_graph.hpp"
#include "origami/graphs/runtime.hpp"
#include "origami/graphs/trace.hpp"
#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

// `flat_cost_t` and `contended_cost_t` were meant to move here from
// `tests/graphs/test_cost_runtime.cpp`, but that file is already gone from
// this tree. Neither survivor is reachable from here: ranking.cpp's
// `flat_cost_t` is a single-price-for-every-op anonymous-namespace type
// private to that translation unit, and test_ranking.cpp's is a different,
// no-argument-constructor type private to *its* translation unit. So both are
// redefined locally, matching the per-op-priced shape the deleted file gave
// them, with overrides renamed to the current `cost_model_t` interface
// (`node_cycles`, `edge_cycles`, `node_cycles_at`).

/** Every node costs its operation's entry; hops are free. */
class flat_cost_t : public cost_model_t {
 public:
  explicit flat_cost_t(std::vector<double> per_op) : per_op_(std::move(per_op)) {}

  double node_cycles(const wg_node_t& node) const override {
    return per_op_.at(static_cast<std::size_t>(node.op));
  }
  double edge_cycles(const edge_t&) const override { return 0.0; }

 private:
  std::vector<double> per_op_;
};

/** Cost grows linearly with the number of busy lanes: a bandwidth-share model. */
class contended_cost_t : public cost_model_t {
 public:
  explicit contended_cost_t(double base) : base_(base) {}

  double node_cycles(const wg_node_t&) const override { return base_; }
  double edge_cycles(const edge_t&) const override { return 0.0; }
  double node_cycles_at(const wg_node_t&, int active_cus) const override {
    return base_ * static_cast<double>(active_cus);
  }

 private:
  double base_;
};

}  // namespace

// ─── 1. unit cost reproduces the old integer timesteps ──────────────────

TEST(unit_cost_reproduces_the_old_integer_timesteps) {
  // Three operations chained wg-for-wg, five workgroups wide, on two lanes:
  // wide enough that lanes queue and every wave's ready time is itself the
  // sum of earlier max()es, so a stray division or averaging anywhere in the
  // dispatch loop would show up as a non-integral start. Under
  // `unit_cost_t` every term feeding those max()es is 1.0 or 0.0, so every
  // start staying an exact integer is what makes the deleted integer-
  // timestep scheduler's view of this schedule meaningful.
  free_graph_t g("pipeline");
  const int a = g.add_op("a", 5, 1);
  const int b = g.add_op("b", 5, 1);
  const int c = g.add_op("c", 5, 1);
  for (int w = 0; w < 5; ++w) {
    g.add_edge(wg_node_t{a, w, 0}, wg_node_t{b, w, 0}, "x", 1);
    g.add_edge(wg_node_t{b, w, 0}, wg_node_t{c, w, 0}, "y", 1);
  }

  simulate_options_t opts;
  opts.lanes = 2;
  const unit_cost_t cost;
  const schedule_t s = simulate(g, asap_runtime_t{}, cost, opts);

  CHECK(s.order.size() == 15);
  for (const wg_node_t& n : s.order) {
    const double start = s.start.at(n);
    CHECK_NEAR(start, std::round(start), 1e-12);
  }
}

// ─── 2. repricing equals static pricing for a contention-free model ─────

TEST(repricing_equals_static_pricing_for_a_contention_free_model) {
  // Eight workgroups on four lanes: two full waves, so the second wave
  // dispatches into a machine with lanes already busy and repricing has
  // something to react to. `flat_cost_t` never overrides `node_cycles_at`,
  // so `cost_model_t`'s default — ignore `active_cus` and return
  // `node_cycles` — applies regardless of `reprice_on_dispatch`, and the two
  // schedules must be identical, not merely equal in makespan.
  free_graph_t g("wide");
  g.add_op("a", 8, 1);
  const flat_cost_t cost({3.0});

  simulate_options_t opts;
  opts.lanes                     = 4;
  const schedule_t static_priced = simulate(g, asap_runtime_t{}, cost, opts);
  opts.reprice_on_dispatch       = true;
  const schedule_t repriced      = simulate(g, asap_runtime_t{}, cost, opts);

  CHECK(static_priced.order.size() == 8);
  CHECK_NEAR(static_priced.makespan(), 6.0, 1e-12);
  CHECK_NEAR(repriced.makespan(), 6.0, 1e-12);
  for (const wg_node_t& n : static_priced.order) {
    CHECK_NEAR(static_priced.start.at(n), repriced.start.at(n), 1e-12);
    CHECK_NEAR(static_priced.duration.at(n), repriced.duration.at(n), 1e-12);
  }

  // Converse: a model that *does* look at active_cus is not vacuously equal
  // under the same toggle, so the check above is proving repricing is a
  // no-op for this cost model specifically, not that simulate() ignores the
  // flag altogether.
  const contended_cost_t sensitive(3.0);
  opts.reprice_on_dispatch          = false;
  const schedule_t contended_flat   = simulate(g, asap_runtime_t{}, sensitive, opts);
  opts.reprice_on_dispatch          = true;
  const schedule_t contended_priced = simulate(g, asap_runtime_t{}, sensitive, opts);
  CHECK(contended_flat.makespan() != contended_priced.makespan());
}

// ─── 3. unlimited lanes reduce ASAP to the dependency depth ─────────────

TEST(unlimited_lanes_reduce_asap_to_the_dependency_depth) {
  // Five workgroups wide, chained one-to-one, and `simulate_options_t{}`
  // leaves `lanes` unset (unlimited): every producer dispatches at cycle 0
  // in its own lane regardless of width, so the makespan is the two-deep
  // dependency chain and nothing a lane limit would otherwise impose.
  free_graph_t g("chain");
  const int a = g.add_op("a", 5, 1);
  const int b = g.add_op("b", 5, 1);
  for (int w = 0; w < 5; ++w) g.add_edge(wg_node_t{a, w, 0}, wg_node_t{b, w, 0}, "x", 1);

  const unit_cost_t cost;
  const schedule_t s = simulate(g, asap_runtime_t{}, cost, simulate_options_t{});

  CHECK(s.order.size() == 10);
  for (int w = 0; w < 5; ++w) {
    CHECK_NEAR(s.start.at(wg_node_t{a, w, 0}), 0.0, 1e-12);
    CHECK_NEAR(s.start.at(wg_node_t{b, w, 0}), 1.0, 1e-12);
  }
  CHECK_NEAR(s.makespan(), 2.0, 1e-12);
}

// ─── 4. to_seconds recovers the numbers the old models returned ─────────

TEST(to_seconds_recovers_the_numbers_the_old_models_returned) {
  // 2100 cycles at 2.1 GHz is one microsecond; checked on the whole
  // conversion — per-node start and duration, and the unit label — not just
  // the makespan, since `to_seconds` is the one place a *schedule's* cycles
  // become a duration and every field it touches needs to agree.
  free_graph_t g("one");
  g.add_op("a", 1, 1);
  const flat_cost_t cost({2100.0});  // cycles
  const schedule_t s        = simulate(g, asap_runtime_t{}, cost, simulate_options_t{});
  const wg_node_t only_node = wg_node_t{0, 0, 0};

  CHECK_NEAR(s.start.at(only_node), 0.0, 1e-12);
  CHECK_NEAR(s.duration.at(only_node), 2100.0, 1e-12);

  const timed_schedule_t t = to_seconds(s, fixed_clock_t{2.1}, 1e6, "us");

  CHECK(t.units == "us");
  CHECK_NEAR(t.start.at(only_node), 0.0, 1e-9);
  CHECK_NEAR(t.duration.at(only_node), 1.0, 1e-9);
  CHECK_NEAR(t.makespan(), 1.0, 1e-9);
}

ORIGAMI_TEST_MAIN()
