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
 * @brief Dumps C++ schedules in the exact format of runtime_oracle.py.
 *
 * Paired with compare.sh, which runs this and the Python script and diffs them
 * line for line. Transcribing eighty schedules into assertions by hand would be
 * both tedious and unreliable; diffing the two dumps checks every node's
 * timestep, not just the makespan, and gives a readable diff when one moves.
 *
 * Cases and their names are kept in step with the Python side by hand. A case
 * present in one and missing from the other shows up in the diff.
 */

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "origami/graphs/core.hpp"
#include "origami/graphs/runtime.hpp"

using namespace origami::graphs;

namespace {

using case_t = std::pair<std::string, graph_t>;

case_t case_one_to_one() {
  const allocation_t x("x", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4))};
  return {"one_to_one", graph_t({a, b})};
}

case_t case_fan_in() {
  const allocation_t x("x", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 2);
  b.access_patterns = {read(x, contiguous(8))};
  return {"fan_in", graph_t({a, b})};
}

case_t case_chain3() {
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4)), write(y, contiguous(4))};
  operation_t c("c", 4);
  c.access_patterns = {read(y, contiguous(4))};
  return {"chain3", graph_t({a, b, c})};
}

case_t case_diamond() {
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
  return {"diamond", graph_t({a, b, c, d})};
}

case_t case_streamed() {
  const allocation_t buf("buf", {index_t{32}});
  operation_t p("produce", 1);
  p.num_iters       = 4;
  p.access_patterns = {write(buf, streamed(32, 4))};
  operation_t c("consume", 1);
  c.num_iters       = 4;
  c.access_patterns = {read(buf, streamed(32, 4))};
  return {"streamed", graph_t({p, c})};
}

case_t case_name_order() {
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});
  operation_t z("z", 4);
  z.access_patterns = {write(x, contiguous(4))};
  operation_t m("m", 4);
  m.access_patterns = {read(x, contiguous(4)), write(y, contiguous(4))};
  operation_t a("a", 4);
  a.access_patterns = {read(y, contiguous(4))};
  return {"name_order", graph_t({z, m, a})};
}

case_t case_split_named() {
  const allocation_t x("x", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t z("z", 4);
  z.access_patterns = {read(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4))};
  return {"split_named", graph_t({a, z, b})};
}

case_t case_two_sources() {
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});
  operation_t z("z", 4);
  z.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {write(y, contiguous(4))};
  operation_t c("c", 4);
  c.access_patterns = {read(x, contiguous(4)), read(y, contiguous(4))};
  return {"two_sources", graph_t({z, b, c})};
}

/**
 * One group per distinct start cycle, in canonical (op, wg) order within a
 * group -- what the old integer `schedule_t::timesteps()` produced, now done
 * by hand since a cycle-based `schedule_t` has no notion of a timestep. Under
 * `unit_cost_t` every start is an exact integer, so grouping by it reproduces
 * the same wave structure the reference's timestep dispatch produces.
 */
std::string render(const wg_graph_t& g, const schedule_t& s) {
  std::map<double, std::vector<wg_node_t>> groups;
  for (const wg_node_t& n : s.order) groups[s.start.at(n)].push_back(n);

  std::string out;
  bool first = true;
  for (auto& [t, group] : groups) {
    std::stable_sort(group.begin(), group.end(), [](const wg_node_t& a, const wg_node_t& b) {
      if (a.op != b.op) return a.op < b.op;
      return a.wg < b.wg;
    });
    if (!first) out += " | ";
    first = false;
    out += std::to_string(static_cast<long long>(t)) + ":";
    for (const wg_node_t& n : group) out += " " + g.label(n);
  }
  return out;
}

}  // namespace

int main() {
  const std::vector<case_t (*)()> cases             = {case_one_to_one,
                                                       case_fan_in,
                                                       case_chain3,
                                                       case_diamond,
                                                       case_streamed,
                                                       case_name_order,
                                                       case_split_named,
                                                       case_two_sources};
  const std::vector<std::optional<int>> lane_counts = {unlimited_lanes, 1, 2, 3};

  const unit_cost_t cost;

  for (const auto& make : cases) {
    const case_t c        = make();
    const std::string& nm = c.first;
    const graph_t& g      = c.second;

    for (const std::optional<int>& lanes : lane_counts) {
      std::vector<std::unique_ptr<runtime_t>> runtimes;
      runtimes.push_back(std::make_unique<breadth_first_runtime_t>());
      runtimes.push_back(std::make_unique<asap_runtime_t>());
      runtimes.push_back(std::make_unique<depth_first_runtime_t>());

      simulate_options_t opts;
      opts.lanes = lanes;

      const std::string lane_str = lanes ? std::to_string(*lanes) : "None";
      for (const auto& rt : runtimes) {
        const schedule_t s    = simulate(g, *rt, cost, opts);
        const std::string tag = nm + "/lanes=" + lane_str + "/" + rt->name();
        std::cout << tag << " makespan=" << static_cast<long long>(s.makespan()) << "\n";
        std::cout << tag << " steps=" << render(g, s) << "\n";
      }
    }
    std::cout << "\n";
  }
  return 0;
}
