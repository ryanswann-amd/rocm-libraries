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
 * @brief origami::graphs — critical-path and overlap analysis.
 *
 * The graph itself carries no cost. An external model supplies a duration per
 * node — `compute_wg_tile_latency` for a collective workgroup, origami for a
 * GEMM tile — and optionally a latency per edge, being the time for the
 * producer's output to become visible to the consumer. With those two hooks the
 * workgroup graph collapses to a textbook PERT problem: the longest
 * source-to-sink path is the end-to-end latency under unbounded parallelism, and
 * the difference against running the operations back to back is the overlap the
 * fused schedule buys.
 *
 * Keeping cost outside the graph is what makes this layer reusable. The same
 * topology can be priced for several hardware configurations without rebuilding
 * it, and the defaults — every node costs 1, every hop is free — turn the
 * makespan into the length of the longest dependency chain, which is a pure
 * structural property worth being able to ask for on its own.
 *
 * This is the *unbounded-resource* bound. It assumes every ready node runs the
 * instant its predecessors finish, so it is a lower bound on any real schedule.
 * The runtimes in runtime.hpp answer the complementary question of what happens
 * with a finite number of compute units.
 */
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

/**
 * @brief Duration of one workgroup-iteration.
 *
 * Defaults to 1 for every node when left empty, which makes the makespan count
 * dependency-chain hops.
 */
using node_cost_fn_t = std::function<double(const wg_node_t&)>;

/**
 * @brief Latency of one producer-to-consumer hop.
 *
 * Defaults to 0 for every edge when left empty, modelling a dependency that is
 * visible the moment the producer retires.
 */
using edge_cost_fn_t = std::function<double(const edge_t&)>;

/** @brief The longest path through a graph and the finish times that produced it. */
struct critical_path_t {
  double makespan = 0.0;         ///< finish time of the last node to complete
  std::vector<wg_node_t> path;   ///< the chain achieving it, source first
  wg_node_map_t<double> finish;  ///< finish time of every node
};

/**
 * @brief Kahn topological sort over the workgroup graph.
 *
 * The ready list is consumed from the back, which makes the traversal
 * depth-first in flavour. That choice does not affect any makespan, but it does
 * decide which chain gets reported when several tie, so it is preserved from the
 * reference implementation rather than replaced with a queue.
 *
 * @param graph Graph to order.
 * @return std::vector<wg_node_t> Every node, predecessors before successors.
 * @throws std::invalid_argument If the graph contains a cycle.
 */
std::vector<wg_node_t> topological_order(const wg_graph_t& graph);

/**
 * @brief Whether the graph can be ordered at all.
 *
 * Both backends make acyclic graphs by construction, so this is a check on
 * inputs that came from somewhere else — a hand-built `free_graph_t`, a trace
 * import, an adversarial test. Callers that want to report a bad graph rather
 * than fail on it need the question separated from the answer.
 *
 * @param graph Graph to check.
 * @return bool True when a topological order exists.
 */
bool is_acyclic(const wg_graph_t& graph);

/**
 * @brief Longest-path latency and the chain that achieves it.
 *
 * `finish[n] = start[n] + node_cost(n)`, where `start[n]` is the largest
 * `finish[src] + edge_cost(e)` over the edges arriving at `n`, or zero when `n`
 * has no predecessors.
 *
 * @param graph Graph to analyse.
 * @param node_cost Duration per node; unit cost when empty.
 * @param edge_cost Latency per hop; free when empty.
 * @return critical_path_t Makespan, critical chain, and all finish times.
 * @throws std::invalid_argument If the graph contains a cycle.
 */
critical_path_t critical_path(const wg_graph_t& graph,
                              const node_cost_fn_t& node_cost = {},
                              const edge_cost_fn_t& edge_cost = {});

/**
 * @brief Human-readable comparison of overlapped against serial execution.
 *
 * @param graph Graph to analyse.
 * @param serial_cost The no-overlap baseline, i.e. the sum of each operation's
 *        standalone time.
 * @param node_cost Duration per node; unit cost when empty.
 * @param edge_cost Latency per hop; free when empty.
 * @return std::string Four lines: serial, critical path, saving, and the chain.
 */
std::string overlap_report(const wg_graph_t& graph,
                           double serial_cost,
                           const node_cost_fn_t& node_cost = {},
                           const edge_cost_fn_t& edge_cost = {});

}  // namespace origami::graphs
