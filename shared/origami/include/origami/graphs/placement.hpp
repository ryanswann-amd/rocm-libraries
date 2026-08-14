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
 * @brief origami::graphs — lane placement policies.
 *
 * A placement chooses which lane a dispatched node runs on, given the node,
 * its dispatch index and every lane's free time. It is deliberately not the
 * simulator: it cannot reorder work or read a cost model, so a machine's
 * physical layout (a pool of lanes, a chiplet's dies) stays a separate
 * concern from issuing order and from pricing.
 *
 * Split out of runtime.hpp into its own header so that simulate.hpp can get a
 * placement's full definition (needed to call `choose()`) without depending on
 * runtime.hpp itself, which declares the four concrete scheduling policies —
 * simulate.hpp only needs the placement interface those policies are built
 * from, not the policies themselves. runtime.hpp still includes this header,
 * so nothing that used to find a placement type there needs to change.
 */
#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

/** @brief Operation name to the lanes it may occupy. */
using lane_pool_t = std::unordered_map<std::string, std::vector<int>>;

/**
 * @brief What a placement is allowed to see.
 *
 * Deliberately not the simulator: a placement chooses a lane, it does not get
 * to reorder work or read the cost model.
 */
struct placement_context_t {
  const wg_node_t& node;               ///< node being dispatched
  std::size_t launch_index;            ///< dispatch counter, for round-robin
  const std::vector<double>& free_at;  ///< per-lane free time, in cycles
};

/** @brief Chooses which lane a dispatched node runs on. */
class placement_t {
 public:
  virtual ~placement_t() = default;

  /** @brief Placement name, for diagnostics. */
  virtual const std::string& name() const = 0;

  /**
   * @brief Lane for this node.
   *
   * @param ctx Node, launch index and per-lane free times.
   * @return int Lane index, which must be in range for the machine.
   */
  virtual int choose(const placement_context_t& ctx) const = 0;
};

/** @brief The lane that frees soonest, ties going to the lower index. */
class earliest_free_placement_t : public placement_t {
 public:
  const std::string& name() const override { return name_; }
  int choose(const placement_context_t& ctx) const override;

 private:
  std::string name_ = "earliest-free";
};

/**
 * @brief Earliest-free, but restricted to an operation's pool of lanes.
 *
 * How a concurrent comm and GEMM launch partitions the CUs so the two genuinely
 * overlap rather than time-slicing. A pool that falls entirely outside the
 * machine is ignored, so a stale pool degrades to the whole machine rather than
 * to a crash.
 */
class pooled_placement_t : public placement_t {
 public:
  /**
   * @brief Resolve pools against a graph's operations and a lane count.
   *
   * @param pool Operation name to lanes.
   * @param graph Graph whose operation names are resolved.
   * @param lane_count Lanes in the machine.
   */
  pooled_placement_t(lane_pool_t pool, const wg_graph_t& graph, int lane_count);

  const std::string& name() const override { return name_; }
  int choose(const placement_context_t& ctx) const override;

 private:
  std::vector<std::vector<int>> allowed_;  ///< per operation
  std::string name_ = "pooled";
};

/**
 * @brief Chiplet dispatch: a die by launch index, then a CU within that die.
 *
 * Lane `i` belongs to die `i / cus_per_xcd`. Workgroups land round-robin on
 * dies in launch order and take the compute unit in that die which frees
 * soonest, which is what a real chiplet part does with a persistent grid.
 */
class xcd_placement_t : public placement_t {
 public:
  /**
   * @brief Construct the rule.
   *
   * @param num_xcds Accelerator dies.
   * @param cus_per_xcd Compute units per die.
   * @throws std::invalid_argument If either count is below one.
   */
  xcd_placement_t(int num_xcds, int cus_per_xcd);

  const std::string& name() const override { return name_; }
  int choose(const placement_context_t& ctx) const override;

  /** @brief Lanes this rule expects the machine to have. */
  int lane_count() const { return num_xcds_ * cus_per_xcd_; }

 private:
  int num_xcds_;
  int cus_per_xcd_;
  std::string name_ = "xcd";
};

}  // namespace origami::graphs
