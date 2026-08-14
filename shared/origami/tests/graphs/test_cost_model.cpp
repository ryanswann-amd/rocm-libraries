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

#include "origami/graphs/cost_model.hpp"

#include "test_harness.hpp"

using namespace origami::graphs;

TEST(unit_cost_charges_one_cycle_per_node_and_nothing_per_edge) {
  const unit_cost_t cost;
  CHECK_NEAR(cost.node_cycles(wg_node_t{0, 0, 0}), 1.0, 1e-12);
  CHECK_NEAR(cost.node_cycles(wg_node_t{3, 7, 2}), 1.0, 1e-12);
  CHECK_NEAR(cost.edge_cycles(edge_t{wg_node_t{0, 0, 0}, wg_node_t{1, 0, 0}, "a", 1}), 0.0, 1e-12);
}

TEST(node_cycles_at_defaults_to_ignoring_contention) {
  const unit_cost_t cost;
  CHECK_NEAR(cost.node_cycles_at(wg_node_t{0, 0, 0}, 64), 1.0, 1e-12);
}

ORIGAMI_TEST_MAIN()
