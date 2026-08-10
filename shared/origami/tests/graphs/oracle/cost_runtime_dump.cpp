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
 * @brief Dumps C++ continuous-time schedules in the format of cost_runtime_oracle.py.
 *
 * Pricing goes through the library's own roofline arm (cost.hpp), so this
 * comparison validates the shipped cost model rather than a copy of it: if
 * roofline_seconds ever drifts from the reference's cost.py, these numbers move
 * and the diff catches it.
 */

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "origami/graphs/core.hpp"
#include "origami/graphs/cost.hpp"
#include "origami/graphs/cost_runtime.hpp"

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

std::string render(const wg_graph_t& g, const cost_schedule_t& s) {
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

  for (const auto& make : cases) {
    const case_t c = make();
    cost_table_t cost(c.graph);
    for (const auto& [op, spec] : c.specs) cost.set(op, op_cost_t::from_roofline(spec));

    std::vector<std::pair<std::string, std::unique_ptr<cost_runtime_t>>> configs;
    for (const std::optional<int>& lanes :
         {std::optional<int>{}, std::optional<int>{2}, std::optional<int>{8}}) {
      for (bool ser : {false, true}) {
        const std::string lane_str = lanes ? std::to_string(*lanes) : "None";
        const std::string suffix =
            "/lanes=" + lane_str + "/ser=" + (ser ? std::string("True") : std::string("False"));

        cost_runtime_options_t opts;
        opts.lanes              = lanes;
        opts.serialize_wg_iters = ser;
        configs.emplace_back("roofline" + suffix, std::make_unique<roofline_runtime_t>(cost, opts));
        configs.emplace_back("event" + suffix,
                             std::make_unique<event_driven_runtime_t>(cost, opts));
      }
    }

    cost_runtime_options_t pooled;
    pooled.lanes     = 8;
    pooled.lane_pool = {{"gemm", {0, 1, 2, 3, 4, 5}},
                        {"all_reduce", {6, 7}},
                        {"produce", {0, 1}},
                        {"consume", {2, 3}},
                        {"a", {0, 1, 2}},
                        {"b", {3}}};
    configs.emplace_back("roofline/pool", std::make_unique<roofline_runtime_t>(cost, pooled));

    xcd_options_t small;
    small.num_xcds    = 2;
    small.cus_per_xcd = 2;
    configs.emplace_back("xcd/2x2", std::make_unique<xcd_runtime_t>(cost, small));
    configs.emplace_back("xcd/8x38", std::make_unique<xcd_runtime_t>(cost, xcd_options_t{}));

    for (const auto& [tag, rt] : configs) {
      const cost_schedule_t s = rt->schedule(c.graph);
      const std::string head  = c.name + "/" + tag;
      std::printf("%s makespan=%s lanes=%d util=%s\n",
                  head.c_str(),
                  fixed6(s.makespan()).c_str(),
                  s.num_lanes(),
                  fixed6(s.utilization()).c_str());
      std::printf("%s atoms=%s\n", head.c_str(), render(c.graph, s).c_str());
    }
    std::printf("\n");
  }
  return 0;
}
