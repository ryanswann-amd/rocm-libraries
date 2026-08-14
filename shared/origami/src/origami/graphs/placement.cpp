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

#include "origami/graphs/placement.hpp"

#include <stdexcept>

namespace origami::graphs {

namespace {

/** Lowest free time over `lanes`, ties going to the lower lane index. */
int soonest_of(const std::vector<int>& lanes, const std::vector<double>& free_at) {
  int best = lanes.front();
  for (int li : lanes) {
    if (free_at[static_cast<std::size_t>(li)] < free_at[static_cast<std::size_t>(best)]) best = li;
  }
  return best;
}

}  // namespace

int earliest_free_placement_t::choose(const placement_context_t& ctx) const {
  int best = 0;
  for (std::size_t i = 1; i < ctx.free_at.size(); ++i) {
    if (ctx.free_at[i] < ctx.free_at[static_cast<std::size_t>(best)]) best = static_cast<int>(i);
  }
  return best;
}

pooled_placement_t::pooled_placement_t(lane_pool_t pool, const wg_graph_t& graph, int lane_count) {
  std::vector<int> all(static_cast<std::size_t>(lane_count));
  for (int i = 0; i < lane_count; ++i) all[static_cast<std::size_t>(i)] = i;

  allowed_.resize(static_cast<std::size_t>(graph.num_operations()));
  for (int op = 0; op < graph.num_operations(); ++op) {
    const auto it = pool.find(graph.op_name(op));
    std::vector<int> clipped;
    if (it != pool.end()) {
      for (int li : it->second) {
        if (li >= 0 && li < lane_count) clipped.push_back(li);
      }
    }
    allowed_[static_cast<std::size_t>(op)] = clipped.empty() ? all : clipped;
  }
}

int pooled_placement_t::choose(const placement_context_t& ctx) const {
  return soonest_of(allowed_.at(static_cast<std::size_t>(ctx.node.op)), ctx.free_at);
}

xcd_placement_t::xcd_placement_t(int num_xcds, int cus_per_xcd)
    : num_xcds_(num_xcds), cus_per_xcd_(cus_per_xcd) {
  if (num_xcds < 1) throw std::invalid_argument("xcd_placement_t: num_xcds must be at least one");
  if (cus_per_xcd < 1) {
    throw std::invalid_argument("xcd_placement_t: cus_per_xcd must be at least one");
  }
}

int xcd_placement_t::choose(const placement_context_t& ctx) const {
  const int die   = static_cast<int>(ctx.launch_index % static_cast<std::size_t>(num_xcds_));
  const int first = die * cus_per_xcd_;
  int best        = first;
  for (int cu = first; cu < first + cus_per_xcd_; ++cu) {
    if (ctx.free_at[static_cast<std::size_t>(cu)] < ctx.free_at[static_cast<std::size_t>(best)]) {
      best = cu;
    }
  }
  return best;
}

}  // namespace origami::graphs
