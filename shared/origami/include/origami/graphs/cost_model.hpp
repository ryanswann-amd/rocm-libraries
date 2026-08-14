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
 * @brief origami::graphs — what it costs to run a node, in cycles.
 *
 * Cycles rather than seconds because a frequency is not a property of a kernel:
 * it moves with DVFS, with the part, and with what else is resident. A model
 * that divides by a clock has folded a runtime condition into a static answer.
 * Seconds are recovered at the boundary by `to_seconds` in simulate.hpp, where
 * a caller supplies whatever clock they can defend.
 */
#pragma once

#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

/**
 * @brief Anything that can price a node and a hop, in cycles.
 *
 * This is the seam where hardware enters. A graph carries one
 * (`wg_graph_t::set_cost`), because a candidate's structure and its price come
 * from the same configuration.
 *
 * The methods keep their `_cycles` names: a caller reading `node_cycles` cannot
 * mistake the unit, which is the whole point of having moved off seconds.
 */
class cost_model_t {
 public:
  virtual ~cost_model_t() = default;

  /**
   * @brief Duration of one workgroup-iteration, in cycles.
   *
   * @param node Node to price.
   * @return double Cycles.
   */
  virtual double node_cycles(const wg_node_t& node) const = 0;

  /**
   * @brief Latency of one producer-to-consumer hop, in cycles.
   *
   * @param edge Hop to price.
   * @return double Cycles.
   */
  virtual double edge_cycles(const edge_t& edge) const = 0;

  /**
   * @brief Duration under contention, in cycles.
   *
   * Defaults to ignoring contention, which is what a model without a view on it
   * should say, so the simulator can call it unconditionally.
   *
   * @param node Node to price.
   * @param active_cus Lanes busy at dispatch, including this node.
   * @return double Cycles.
   */
  virtual double node_cycles_at(const wg_node_t& node, int active_cus) const {
    (void)active_cus;
    return node_cycles(node);
  }
};

/**
 * @brief One cycle per node, nothing per hop.
 *
 * Turns the simulator into the workgroup-counting scheduler that used to live
 * in runtime.hpp: every start time is an exact integer, so a schedule under
 * this model is a shape rather than a duration.
 */
class unit_cost_t : public cost_model_t {
 public:
  double node_cycles(const wg_node_t&) const override { return 1.0; }
  double edge_cycles(const edge_t&) const override { return 0.0; }
};

}  // namespace origami::graphs
