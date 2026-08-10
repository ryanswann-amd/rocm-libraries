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
 * @brief origami::graphs — continuous-time, cost-model-driven runtimes.
 *
 * The runtimes in runtime.hpp count workgroups: every atom is one timestep, so
 * the answer is a shape rather than a duration. These ask a cost model what each
 * atom actually costs and schedule in real time, so a fat GEMM tile and a thin
 * collective step occupy proportionally different widths and every node lands on
 * a concrete lane. That is the raw material for a Gantt chart, and for comparing
 * two candidate decompositions in units anyone can act on.
 *
 * Three policies, in increasing order of how much hardware they know about:
 *
 *   - `roofline_runtime_t` prices every node once, up front. Ready atoms go to
 *     the lane that frees soonest, earliest-ready first. `lane_pool` optionally
 *     pins an operation to a subset of lanes, which is how a concurrent comm and
 *     GEMM launch partitions the CUs so the two genuinely overlap rather than
 *     time-slicing.
 *   - `event_driven_runtime_t` is the same scheduler, but re-prices each node at
 *     dispatch time against the number of lanes currently busy. A node
 *     dispatched into a crowded machine is charged more than the same node
 *     dispatched into an idle one. Atoms already in flight are not re-priced
 *     when their neighbours retire; that would be a full fluid simulation.
 *   - `xcd_runtime_t` drops the lane pools and models chiplet dispatch directly:
 *     workgroups launch in grid order, land round-robin on one of `num_xcds`
 *     dies by launch index, and take the compute unit within that die that frees
 *     soonest. Every operation shares all CUs, which is what a fused kernel's
 *     producer and consumer workgroups actually do.
 *
 * `serialize_wg_iters` is worth calling out. Without it the iterations of one
 * workgroup can run concurrently, which is the gap analysis.hpp runs into: the
 * critical path through a streamed producer-consumer pair is two, not the depth
 * of the pipeline, because nothing says a workgroup runs its own iterations in
 * sequence. Setting it adds that precedence and pins the iterations to one lane,
 * since a workgroup occupies a CU for its whole lifetime.
 */
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "origami/graphs/types.hpp"
#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

/**
 * @brief Anything that can price a node and a hop, in seconds.
 *
 * This is the seam where hardware enters. Implementations are supplied by the
 * caller; the roofline model and the origami-backed comm and GEMM arms plug in
 * here. A graph carries one of these (`wg_graph_t::set_cost`), because a
 * candidate's structure and its price come from the same configuration and
 * ranking many candidates means many prices.
 */
class cost_model_t {
 public:
  virtual ~cost_model_t() = default;

  /**
   * @brief Duration of one workgroup-iteration, in seconds.
   *
   * @param node Node to price.
   * @return double Seconds.
   */
  virtual double node_cost(const wg_node_t& node) const = 0;

  /**
   * @brief Latency of one producer-to-consumer hop, in seconds.
   *
   * @param edge Hop to price.
   * @return double Seconds.
   */
  virtual double edge_cost(const edge_t& edge) const = 0;

  /**
   * @brief Duration under contention, in seconds.
   *
   * Defaults to ignoring contention, which is what a model without a view on it
   * should say. The reference feature-detects this method with `hasattr` and
   * silently falls back; a defaulted virtual says the same thing as a total
   * function, so `event_driven_runtime_t` can call it unconditionally and a
   * model that does model contention only has to override it.
   *
   * @param node Node to price.
   * @param active_cus Lanes busy at dispatch, including this node.
   * @return double Seconds.
   */
  virtual double node_cost_at(const wg_node_t& node, int active_cus) const {
    (void)active_cus;
    return node_cost(node);
  }
};

/**
 * @brief The cost model to price a graph with: the override if there is one,
 *        otherwise the graph's own.
 *
 * Every runtime resolves cost this way, so a runtime constructed without a cost
 * model prices whatever graph it is handed, and one constructed with a cost
 * model prices every graph the same way.
 *
 * @param override_cost Cost supplied to the runtime's constructor, or null.
 * @param graph Graph being scheduled.
 * @return const cost_model_t& The model to use.
 * @throws std::invalid_argument If neither is set.
 */
const cost_model_t& resolve_cost(const cost_model_t* override_cost, const wg_graph_t& graph);

/** @brief Operation name to the lanes it may occupy. */
using lane_pool_t = std::unordered_map<std::string, std::vector<int>>;

/**
 * @brief A continuous-time schedule: real start, duration and lane per atom.
 *
 * Unlike `schedule_t`, which has integer timesteps and one shared duration,
 * every node here carries its own duration and sits on a concrete lane, so the
 * result renders directly as a Gantt chart.
 */
struct cost_schedule_t {
  std::string runtime;             ///< policy that produced this
  wg_node_map_t<double> start;     ///< start time per node
  wg_node_map_t<double> duration;  ///< duration per node
  wg_node_map_t<int> lane;         ///< execution slot per node
  std::optional<int> lanes;        ///< slot limit as requested; unset means unlimited
  std::string units = "\u00b5s";   ///< display units, for summaries and traces
  std::vector<wg_node_t> order;    ///< nodes in dispatch order

  /**
   * @brief Record one atom: when it started, how long it ran, and where.
   *
   * The one mutator, so a runtime written outside this file — in particular one
   * written in Python — can build a schedule without reaching into four
   * containers and keeping them consistent by hand.
   *
   * Placing a node twice overwrites its times and leaves its position in
   * `order` alone, on the grounds that a runtime revising a placement has not
   * changed its mind about when it dispatched.
   *
   * @param node Atom to record.
   * @param start Start time, in this schedule's units.
   * @param duration Duration, in this schedule's units.
   * @param lane Execution slot.
   */
  void place(const wg_node_t& node, double start, double duration, int lane);

  /**
   * @brief Finish time of a node.
   *
   * @param node Node to query.
   * @return double start + duration.
   * @throws std::out_of_range If the node is not in this schedule.
   */
  double finish(const wg_node_t& node) const;

  /** @brief Last finish time over every node; zero for an empty schedule. */
  double makespan() const;

  /** @brief Highest occupied lane, plus one. */
  int num_lanes() const;

  /**
   * @brief Total busy time per operation: work done, not wall time.
   *
   * @return std::vector<double> Indexed by operation, summed in dispatch order.
   */
  std::vector<double> busy_by_op() const;

  /** @brief Fraction of available lane-time spent doing work. */
  double utilization() const;

  /** @brief Human-readable summary: atoms, lanes, makespan, utilisation. */
  std::string summary() const;
};

/** @brief Knobs shared by the lane-pool continuous runtimes. */
struct cost_runtime_options_t {
  std::optional<int> lanes;         ///< execution slots; unset means one per node
  double scale      = 1e6;          ///< seconds to display units; affects numbers, not order
  std::string units = "\u00b5s";    ///< display units
  lane_pool_t lane_pool;            ///< optional per-operation lane partition
  bool serialize_wg_iters = false;  ///< run a workgroup's iterations in sequence, on one lane
};

/**
 * @brief A policy that turns a dataflow graph into a continuous-time schedule.
 */
class cost_runtime_t {
 public:
  virtual ~cost_runtime_t() = default;

  /** @brief Policy name, as it appears in a schedule. */
  virtual const std::string& name() const = 0;

  /**
   * @brief Place every node in time and on a lane.
   *
   * @param graph Graph to schedule.
   * @return cost_schedule_t The resulting schedule.
   */
  virtual cost_schedule_t schedule(const wg_graph_t& graph) const = 0;
};

/**
 * @brief Greedy continuous-time scheduler with static pricing.
 *
 * An atom is ready once every producer has finished and that hop's latency has
 * elapsed; ready atoms dispatch earliest-first onto the lane that frees soonest,
 * so a producer and an independent consumer overlap whenever lanes allow.
 */
class roofline_runtime_t : public cost_runtime_t {
 public:
  /**
   * @brief Construct the policy, pricing each graph with its own cost model.
   *
   * @param options Lane count, scaling, lane pools and iteration serialisation.
   * @throws std::invalid_argument If lanes is below one or scale is not positive.
   */
  explicit roofline_runtime_t(cost_runtime_options_t options = {});

  /**
   * @brief Construct the policy with a cost model that overrides the graph's.
   *
   * @param cost Cost model, which must outlive this runtime.
   * @param options Lane count, scaling, lane pools and iteration serialisation.
   * @throws std::invalid_argument If lanes is below one or scale is not positive.
   */
  roofline_runtime_t(const cost_model_t& cost, cost_runtime_options_t options = {});

  const std::string& name() const override { return name_; }
  cost_schedule_t schedule(const wg_graph_t& graph) const override;

 private:
  const cost_model_t* cost_ = nullptr;
  cost_runtime_options_t opts_;
  std::string name_ = "roofline";
};

/**
 * @brief Greedy continuous-time scheduler that re-prices at dispatch time.
 *
 * Identical to `roofline_runtime_t` except that each node's duration comes from
 * `cost_model_t::node_cost_at` with the number of lanes busy at the moment it
 * starts. With a cost model that ignores contention the two agree exactly, which
 * is worth knowing: any difference between them is attributable to the model.
 */
class event_driven_runtime_t : public cost_runtime_t {
 public:
  /**
   * @brief Construct the policy, pricing each graph with its own cost model.
   *
   * @param options Lane count, scaling, lane pools and iteration serialisation.
   * @throws std::invalid_argument If lanes is below one or scale is not positive.
   */
  explicit event_driven_runtime_t(cost_runtime_options_t options = {});

  /**
   * @brief Construct the policy with a cost model that overrides the graph's.
   *
   * @param cost Cost model, which must outlive this runtime.
   * @param options Lane count, scaling, lane pools and iteration serialisation.
   * @throws std::invalid_argument If lanes is below one or scale is not positive.
   */
  event_driven_runtime_t(const cost_model_t& cost, cost_runtime_options_t options = {});

  const std::string& name() const override { return name_; }
  cost_schedule_t schedule(const wg_graph_t& graph) const override;

 private:
  const cost_model_t* cost_ = nullptr;
  cost_runtime_options_t opts_;
  std::string name_ = "event-driven";
};

/** @brief Knobs for the chiplet-aware runtime. */
struct xcd_options_t {
  int num_xcds      = 8;          ///< accelerator dies
  int cus_per_xcd   = 38;         ///< compute units per die
  double scale      = 1e6;        ///< seconds to display units
  std::string units = "\u00b5s";  ///< display units

  /**
   * Operations whose name starts with this are treated as zero-work barriers:
   * they take no compute unit and do not consume a round-robin slot, they only
   * forward their ready time. A cheap way to express a whole-stage barrier with
   * a linear rather than quadratic number of edges.
   */
  std::string skip_prefix = "sync";
};

/**
 * @brief Chiplet-aware, work-stealing, in-order scheduler.
 *
 * Models how a real GPU dispatches persistent workgroups on a chiplet part.
 * Nodes are visited in launch order — which is canonical node order, and a valid
 * topological order for anything either backend can build — placed round-robin
 * onto a die by launch index, and given the compute unit within that die that
 * frees soonest. There is no per-operation partition: every operation shares all
 * CUs, exactly as a fused kernel's producer and consumer workgroups do.
 */
class xcd_runtime_t : public cost_runtime_t {
 public:
  /**
   * @brief Construct the policy, pricing each graph with its own cost model.
   *
   * @param options Die and compute-unit counts, scaling, barrier prefix.
   * @throws std::invalid_argument If any count is below one or scale is not positive.
   */
  explicit xcd_runtime_t(xcd_options_t options = {});

  /**
   * @brief Construct the policy with a cost model that overrides the graph's.
   *
   * @param cost Cost model, which must outlive this runtime.
   * @param options Die and compute-unit counts, scaling, barrier prefix.
   * @throws std::invalid_argument If any count is below one or scale is not positive.
   */
  xcd_runtime_t(const cost_model_t& cost, xcd_options_t options = {});

  const std::string& name() const override { return name_; }

  /**
   * @copydoc cost_runtime_t::schedule
   * @throws std::invalid_argument If an edge runs backwards in canonical node
   *         order, which would make the launch order non-topological.
   */
  cost_schedule_t schedule(const wg_graph_t& graph) const override;

 private:
  const cost_model_t* cost_ = nullptr;
  xcd_options_t opts_;
  std::string name_ = "xcd";
};

}  // namespace origami::graphs
