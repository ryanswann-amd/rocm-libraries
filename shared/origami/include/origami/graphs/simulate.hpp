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
 * @brief origami::graphs — one scheduler, driven by a policy and a cost model.
 *
 * A graph says what depends on what. A runtime says in what order and where
 * work is issued. A cost model says how long each piece takes, in cycles. This
 * file owns the loop that combines the three, so a policy never has to know a
 * price and a price never has to know a policy.
 */
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "origami/graphs/cost_model.hpp"
#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

class placement_t;

/** @brief Unlimited lanes, i.e. every ready node dispatches immediately. */
inline constexpr std::optional<int> unlimited_lanes = std::nullopt;

/**
 * @brief A schedule in cycles: start, duration and lane for every atom.
 */
struct schedule_t {
  std::string runtime;             ///< policy that produced this
  std::optional<int> lanes;        ///< concurrency limit; unset means unlimited
  wg_node_map_t<double> start;     ///< start cycle per node
  wg_node_map_t<double> duration;  ///< cycles per node
  wg_node_map_t<int> lane;         ///< lane per node
  std::vector<wg_node_t> order;    ///< nodes in dispatch order

  /** @brief Record a node. Re-placing overwrites without re-appending. */
  void place(const wg_node_t& node, double start, double duration, int lane);

  /** @brief start + duration. @throws std::out_of_range If absent. */
  double finish(const wg_node_t& node) const;

  /** @brief Last finish over every node; zero when empty. */
  double makespan() const;

  /** @brief Highest occupied lane, plus one. */
  int num_lanes() const;

  /** @brief Total busy cycles per operation, summed in dispatch order. */
  std::vector<double> busy_by_op() const;

  /** @brief Fraction of available lane-time spent doing work. */
  double utilization() const;

  /** @brief Human-readable summary: atoms, lanes, makespan, utilisation. */
  std::string summary() const;
};

/** @brief What a gate may see: enough to sequence operations, no more. */
struct sim_state_t {
  const std::vector<int>& retired_by_op;    ///< nodes finished, per operation
  const std::vector<int>& in_flight_by_op;  ///< nodes dispatched but not retired
  const std::vector<int>& total_by_op;      ///< nodes that exist, per operation
};

/** @brief Whether readiness or the runtime's rank dominates dispatch order. */
enum class queue_order_t {
  ready_first,  ///< earliest-ready node wins; rank breaks ties

  /**
   * Lowest rank wins; a node waits for its turn.
   *
   * @note simulate() advances simulated time by calling `advance(top.ready)`
   * on the node it just popped, using that node's own ready time. Under
   * ready_first that time is non-decreasing across pops, so `advance` only
   * ever learns about the past. Under rank_first it is not: a low-rank node
   * with a late ready time can pop before a high-rank node with an earlier
   * one, and `advance` moves simulated time forward irreversibly (a retired
   * node never un-retires). The result is that by the time the
   * earlier-ready, later-popped node is considered, `sim_state_t` may already
   * reflect a point in time *after* that node's own ready time — a gate sees
   * a future state, never a past one, relative to the node it is judging.
   * A gate consulted under rank_first must therefore be safe to evaluate
   * against a state that is at least as advanced as the node's ready time,
   * never exactly at it. Gates that only ever ask "has all of operation X
   * retired" are fine, since that predicate is monotone; a gate that needs an
   * exact snapshot at the node's own ready time is not safe under this order.
   */
  rank_first
};

/**
 * @brief A scheduling policy: an order, a lane rule, and an optional gate.
 *
 * Deliberately says nothing about cost. Applying a runtime to a graph and then
 * applying a cost model is what produces a schedule.
 */
class runtime_t {
 public:
  virtual ~runtime_t() = default;

  /** @brief Policy name, as it appears in a schedule. */
  virtual const std::string& name() const = 0;

  /**
   * @brief Rank per node; the lowest-ranked ready node dispatches first.
   *
   * @param graph Graph being scheduled.
   * @return wg_node_map_t<std::size_t> A distinct rank per node.
   */
  virtual wg_node_map_t<std::size_t> priority(const wg_graph_t& graph) const = 0;

  /** @brief Which lane a dispatched node runs on. */
  virtual const placement_t& placement() const = 0;

  /** @brief Whether readiness or rank dominates. Defaults to readiness. */
  virtual queue_order_t order() const { return queue_order_t::ready_first; }

  /**
   * @brief Withhold a node whose producers have all finished.
   *
   * A rank orders the ready set but cannot keep a node out of it, which is
   * what an operation barrier needs. Defaults to withholding nothing.
   *
   * @note A held node is only replayed against the queue when some node
   * *retires* (its finish time is passed by `advance`), never on a mere
   * dispatch. A gate may legally depend on `retired_by_op` and
   * `total_by_op` — quantities that only change at retirement, which is
   * exactly when a held node gets another look — and on `in_flight_by_op`
   * *falling*, which also only happens at retirement. A gate that depends on
   * `in_flight_by_op` *rising* (a dispatch, not a retirement) will never be
   * re-evaluated for that change: nothing replays held nodes when a node is
   * merely dispatched, so such a gate can hold a node back forever.
   *
   * @param node Node that is otherwise ready.
   * @param state Per-operation progress counts.
   * @return bool True to hold the node back this round.
   */
  virtual bool gated(const wg_node_t& node, const sim_state_t& state) const {
    (void)node;
    (void)state;
    return false;
  }
};

/** @brief Knobs that are properties of the machine or the costing, not the policy. */
struct simulate_options_t {
  std::optional<int> lanes;          ///< execution slots; unset means one per node
  bool serialize_wg_iters  = false;  ///< a workgroup's iterations share one lane, in sequence
  bool reprice_on_dispatch = false;  ///< ask node_cycles_at with the busy lane count
  std::string skip_prefix;           ///< operations with this prefix are zero-work barriers
};

/**
 * @brief Schedule a graph under a policy, pricing it with a cost model.
 *
 * @param graph Graph to schedule.
 * @param runtime Policy supplying order, placement and gating.
 * @param cost Model supplying durations in cycles.
 * @param options Machine and costing knobs.
 * @return schedule_t Start, duration and lane per node, in cycles.
 * @throws std::invalid_argument If `options.lanes` is set below one, the
 *         graph has a cycle, a placement returns a lane outside the machine,
 *         or a runtime's gate deadlocks (holds a node with nothing left that
 *         could ever unblock it).
 */
schedule_t simulate(const wg_graph_t& graph,
                    const runtime_t& runtime,
                    const cost_model_t& cost,
                    simulate_options_t options = {});

/** @brief A clock, so seconds are a boundary concern rather than a model's. */
class clock_t {
 public:
  virtual ~clock_t() = default;
  /** @brief Frequency in GHz. */
  virtual double ghz() const = 0;
};

/** @brief A clock that does not vary. */
class fixed_clock_t : public clock_t {
 public:
  /**
   * @brief Construct the clock.
   * @param ghz Frequency.
   * @throws std::invalid_argument If ghz is not positive.
   */
  explicit fixed_clock_t(double ghz);
  double ghz() const override { return ghz_; }

 private:
  double ghz_;
};

/** @brief A schedule in seconds: the same layout, converted for display. */
struct timed_schedule_t {
  std::string runtime;
  std::string units = "s";
  wg_node_map_t<double> start;
  wg_node_map_t<double> duration;
  wg_node_map_t<int> lane;
  std::vector<wg_node_t> order;

  /** @brief start + duration. @throws std::out_of_range If absent. */
  double finish(const wg_node_t& node) const;

  double makespan() const;
  std::string summary() const;
};

/**
 * @brief Convert a schedule from cycles to seconds.
 *
 * The only place in the library that turns a *schedule's* cycles into a
 * duration; a cost model may separately apply a clock while deriving cycles
 * in the first place (the roofline arm's `compute_clock_ghz`, for instance),
 * but once a schedule exists this is the one boundary that gives it a wall-
 * clock unit. A DVFS-aware clock replaces `fixed_clock_t` here without
 * touching a cost model.
 *
 * @param schedule Schedule in cycles.
 * @param clock Clock to apply.
 * @param scale Multiplier on the result, e.g. 1e6 for microseconds.
 * @param units Name of the resulting unit, for display.
 * @return timed_schedule_t The converted schedule.
 * @throws std::invalid_argument If scale is not positive.
 */
timed_schedule_t to_seconds(const schedule_t& schedule,
                            const clock_t& clock,
                            double scale             = 1.0,
                            const std::string& units = "s");

}  // namespace origami::graphs
