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
 * @brief Dumps cost-model-priced schedules in the format of priced_oracle.py.
 *
 * The companion to runtime_dump.cpp, which prices every node identically and so
 * only compares dispatch order. Here pricing goes through the library's own
 * roofline arm (cost.hpp), so this comparison validates the shipped cost model
 * against the reference's cost.py rather than a copy of it: if roofline_cycles
 * ever drifts from the reference, these numbers move and the diff catches it.
 *
 * The reference prices nodes in seconds and displays microseconds, applying its
 * runtime's `scale` of 1e6 to every duration as it schedules. This library
 * schedules in cycles and applies both the clock and the scale once, at the end,
 * through `to_seconds` — the same two multiplications in the same direction,
 * moved to the boundary. Max and plus commute with a positive scalar, so the
 * printed values agree to well beyond the six decimals shown.
 */

#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "origami/graphs/core.hpp"
#include "origami/graphs/cost.hpp"
#include "origami/graphs/placement.hpp"
#include "origami/graphs/runtime.hpp"
#include "origami/graphs/simulate.hpp"

using namespace origami::graphs;

namespace {

struct case_t {
  std::string name;
  graph_t graph;
  std::unordered_map<std::string, roofline_spec_t> specs;
};

case_t gemm_then_reduce() {
  const allocation_t c("c", {index_t{64}});
  const allocation_t out("out", {index_t{64}});
  operation_t gemm("gemm", 8);
  gemm.access_patterns = {write(c, contiguous(8))};
  operation_t ar("all_reduce", 4);
  ar.access_patterns = {read(c, contiguous(16)), write(out, contiguous(16))};
  return {"gemm_reduce",
          graph_t({gemm, ar}),
          {{"gemm", roofline_spec_t::gemm_tile(128, 128, 4096)},
           {"all_reduce", roofline_spec_t::comm_step(1 << 20, 2.0, 2.0)}}};
}

case_t streamed_pair() {
  const allocation_t buf("buf", {index_t{32}});
  operation_t p("produce", 1);
  p.num_iters       = 4;
  p.access_patterns = {write(buf, streamed(32, 4))};
  operation_t q("consume", 1);
  q.num_iters       = 4;
  q.access_patterns = {read(buf, streamed(32, 4))};
  return {"streamed_pair",
          graph_t({p, q}),
          {{"produce", roofline_spec_t::gemm_tile(64, 64, 512)},
           {"consume", roofline_spec_t::comm_step(1 << 16)}}};
}

case_t with_barrier() {
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t s("sync_all", 1);
  s.access_patterns = {read(x, contiguous(16)), write(y, contiguous(16))};
  operation_t b("b", 4);
  b.access_patterns = {read(y, contiguous(4))};
  return {"with_barrier",
          graph_t({a, s, b}),
          {{"a", roofline_spec_t::gemm_tile(64, 64, 1024)},
           {"sync_all", roofline_spec_t{}},
           {"b", roofline_spec_t::comm_step(1 << 14)}}};
}

std::string fixed6(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6f", v);
  return std::string(buf);
}

std::string render(const wg_graph_t& g, const timed_schedule_t& s) {
  std::string out;
  bool first = true;
  for (const wg_node_t& n : g.nodes()) {
    if (!first) out += " ";
    first = false;
    out += g.label(n) + "@" + fixed6(s.start.at(n)) + "+" + fixed6(s.duration.at(n)) + "/L" +
           std::to_string(s.lane.at(n));
  }
  return out;
}

}  // namespace

int main() {
  const std::vector<case_t (*)()> cases = {gemm_then_reduce, streamed_pair, with_barrier};

  // Only the roofline arm reaches this file, so the clock that turns its cycles
  // back into time is that arm's own.
  const fixed_clock_t clock(roofline_hardware_t{}.compute_clock_ghz);
  constexpr double kMicroseconds = 1e6;  ///< the reference runtimes' own `scale`

  for (const auto& make : cases) {
    const case_t c = make();
    cost_table_t cost(c.graph);
    for (const auto& [op, spec] : c.specs) cost.set(op, op_cost_t::from_roofline(spec));

    const asap_runtime_t asap;

    // Tag, policy, and the machine to run it on. The seconds-era runtimes this
    // replaces each carried their own lane count and cost model; a policy now
    // carries neither, so what used to distinguish `roofline` from `event` is a
    // single simulate option, and what used to distinguish a pooled runtime is
    // a placement.
    std::vector<std::tuple<std::string, const runtime_t*, simulate_options_t>> configs;
    std::vector<std::unique_ptr<runtime_t>> owned;

    for (const std::optional<int>& lanes :
         {std::optional<int>{}, std::optional<int>{2}, std::optional<int>{8}}) {
      for (bool ser : {false, true}) {
        const std::string lane_str = lanes ? std::to_string(*lanes) : "None";
        const std::string suffix =
            "/lanes=" + lane_str + "/ser=" + (ser ? std::string("True") : std::string("False"));

        simulate_options_t opts;
        opts.lanes              = lanes;
        opts.serialize_wg_iters = ser;
        configs.emplace_back("roofline" + suffix, &asap, opts);

        simulate_options_t evt  = opts;
        evt.reprice_on_dispatch = true;
        configs.emplace_back("event" + suffix, &asap, evt);
      }
    }

    const lane_pool_t pool = {{"gemm", {0, 1, 2, 3, 4, 5}},
                              {"all_reduce", {6, 7}},
                              {"produce", {0, 1}},
                              {"consume", {2, 3}},
                              {"a", {0, 1, 2}},
                              {"b", {3}}};
    simulate_options_t pooled_opts;
    pooled_opts.lanes = 8;
    // The same "override the placement, forward everything else" composition
    // ranking.cpp builds for a `graph_config_t::lane_pool`, via the library's
    // own adapter rather than a local copy of it; the reference expresses the
    // same thing as a constructor argument to its roofline runtime. The
    // placement has to outlive the adapter, so it lives alongside `owned`
    // rather than inside it.
    const pooled_placement_t pool_placement(pool, c.graph, 8);
    owned.push_back(std::make_unique<runtime_with_placement_t>(asap, pool_placement));
    configs.emplace_back("roofline/pool", owned.back().get(), pooled_opts);

    for (const std::pair<int, int>& geometry : {std::pair<int, int>{2, 2}, {8, 38}}) {
      auto policy = std::make_unique<xcd_runtime_t>(geometry.first, geometry.second);
      simulate_options_t opts;
      // The chiplet placement indexes a whole die's free times, so the machine
      // has to be exactly as wide as the geometry; `sync` stages are the
      // zero-work barriers the reference's XCD runtime steps over.
      opts.lanes       = policy->lane_count();
      opts.skip_prefix = "sync";
      const std::string tag =
          "xcd/" + std::to_string(geometry.first) + "x" + std::to_string(geometry.second);
      owned.push_back(std::move(policy));
      configs.emplace_back(tag, owned.back().get(), opts);
    }

    for (const auto& [tag, rt, opts] : configs) {
      const schedule_t cycles  = simulate(c.graph, *rt, cost, opts);
      const timed_schedule_t s = to_seconds(cycles, clock, kMicroseconds, "us");
      const std::string head   = c.name + "/" + tag;
      std::printf("%s makespan=%s lanes=%d util=%s\n",
                  head.c_str(),
                  fixed6(s.makespan()).c_str(),
                  cycles.num_lanes(),
                  fixed6(cycles.utilization()).c_str());
      std::printf("%s atoms=%s\n", head.c_str(), render(c.graph, s).c_str());
    }
    std::printf("\n");
  }
  return 0;
}
