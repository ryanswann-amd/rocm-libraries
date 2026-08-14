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
 * @brief origami::graphs — selection and ranking over schedules and graphs.
 *
 * The headline entry point is ranking, not a bare prediction, mirroring
 * `origami.hpp`. The correspondence is deliberate and one-for-one:
 *
 *   | origami                  | origami::graphs           | the input is        |
 *   |--------------------------|---------------------------|---------------------|
 *   | `problem_t`              | the `wg_graph_t`          | what depends on what|
 *   | `config_t` (tile shapes) | `graph_config_t`          | how it is scheduled |
 *   | `select_config`          | `select_config`           | best candidate      |
 *   | `rank_configs`           | `rank_configs`            | all, best first     |
 *   | `select_topk_configs`    | `select_topk_configs`     | top k               |
 *   | `compute_tile_latency`   | `predict_latency`         | one candidate       |
 *
 * What is tunable differs, which is the point. GEMM sweeps tile geometry;
 * kirigami sweeps the *resource mapping* — which runtime policy, how many lanes,
 * whether a workgroup's iterations serialise, and which lanes each operation may
 * use. That is the schedule tier, and it is the same enum-plus-override shape as
 * `comm_config_t`'s algorithm override and as `cost_kind_t`: one convention in
 * three places.
 *
 * A second trio takes several graphs at once, so a caller ranks a whole
 * graph-by-schedule cross product in one call rather than looping and merging.
 * That is the two-level selection an algorithm choice plus a schedule choice
 * needs: `two_shot` at 8 lanes may beat `ring` at 16, and neither ordering is
 * apparent from ranking each graph separately.
 *
 * Three contracts are copied from `rank_configs`:
 *   - results are ordered best first;
 *   - an empty candidate list throws `std::runtime_error`;
 *   - if *every* candidate is rejected, the rejected candidates are returned
 *     ranked at maximum latency rather than throwing, so a caller always has
 *     something to report.
 */
#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "origami/comm/hardware.hpp"
#include "origami/graphs/cost_model.hpp"
#include "origami/graphs/placement.hpp"
#include "origami/graphs/runtime.hpp"
#include "origami/graphs/wg_graph.hpp"

// runtime.hpp for `runtime_t` (what `runtime_override` holds) and
// `xcd_options_t`; placement.hpp for `lane_pool_t`, which runtime.hpp also
// pulls in but which this header names directly.

namespace origami::graphs {

/**
 * @brief Which scheduling policy a candidate uses, and how it is priced.
 *
 * Every kind runs through `simulate()`. What differs between them is the
 * dispatch policy and where the per-node duration comes from: the first three
 * charge a flat `wg_duration` cycles per node and so need no cost model, while
 * the last three price every node from one. `asap` and `roofline` are literally
 * the same policy under two names for that reason, which is a naming wart worth
 * knowing about: these values name a (policy, cost source) pair rather than a
 * policy alone.
 */
enum class runtime_kind_t {
  breadth_first,  ///< no overlap: one operation at a time, flat wg_duration
  asap,           ///< producer-greedy, flat wg_duration
  depth_first,    ///< chain-greedy, flat wg_duration
  roofline,       ///< producer-greedy, priced from the cost model up front
  event_driven,   ///< producer-greedy, re-priced against occupancy at dispatch
  xcd,            ///< chiplet round-robin in launch order, priced from the model
};

/**
 * @brief Name of a runtime policy.
 *
 * @param kind Policy to name.
 * @return const char* Lower-case name, e.g. "event_driven".
 */
const char* runtime_kind_name(runtime_kind_t kind);

/**
 * @brief Parse a runtime policy from its name.
 *
 * @param name One of the `runtime_kind_t` names.
 * @return runtime_kind_t The policy.
 * @throws std::invalid_argument If the name is not one of those.
 */
runtime_kind_t runtime_kind_from_name(const std::string& name);

/**
 * @brief Whether a kind needs a real cost model.
 *
 * `breadth_first`/`asap`/`depth_first` run through simulate() with a uniform
 * `wg_duration` cycles charged per node, so a candidate is schedulable with no
 * cost model at all — the answer is a shape, not a duration. The other three
 * price every node from a `cost_model_t` (the graph's own, or one supplied to
 * `predict_latency`/`rank_configs`), so they need one and fail without it.
 *
 * This predicate used to be spelled `is_continuous`, distinguishing integer
 * timesteps from real time. That distinction died when every policy moved onto
 * one cycle-valued simulator: both families now produce a cycle count, and the
 * only thing left to ask is where the cycles came from.
 *
 * @param kind Policy to ask about.
 * @return bool True when the kind prices every node from a `cost_model_t`;
 *         false when it charges a flat `wg_duration` instead.
 */
bool needs_cost_model(runtime_kind_t kind);

/**
 * @brief One candidate schedule: the performance input, as `config_t` is for GEMM.
 */
struct graph_config_t {
  runtime_kind_t runtime = runtime_kind_t::roofline;  ///< scheduling policy

  /**
   * Execution slots. Unset means unlimited, i.e. one per atom.
   *
   * `runtime_kind_t::xcd` overrides this with its own geometry's lane count
   * (`xcd.num_xcds * xcd.cus_per_xcd`) rather than honouring it, because
   * `xcd_placement_t` indexes the free-time array for a whole die and a
   * narrower machine would be an out-of-bounds read.
   */
  std::optional<int> lanes;

  /** Run each workgroup's iterations in sequence, on one lane. */
  bool serialize_wg_iters = false;

  /**
   * Per-operation lane partition. How a concurrent comm and GEMM launch splits
   * the compute units so the two genuinely overlap rather than time-slicing;
   * when set, it replaces the chosen policy's placement with a
   * `pooled_placement_t`. Ignored by `runtime_kind_t::xcd`, whose placement is
   * the chiplet geometry.
   */
  lane_pool_t lane_pool;

  /** Cycles charged per node by the kinds that use no cost model. */
  index_t wg_duration = 1;

  /** Chiplet geometry, used only by `runtime_kind_t::xcd`. */
  xcd_options_t xcd;

  /**
   * Escape hatch: when set, this policy is used verbatim in place of the one
   * `runtime` names. Mirrors `comm_config_t::algorithm_override` — an enum
   * covers the policies worth naming, and a pointer covers the one a caller
   * invented.
   *
   * Unlike the seconds-era override it replaces, this does *not* ignore the
   * other fields: a `runtime_t` is a policy and nothing more, so `lanes`,
   * `serialize_wg_iters` and the cost model still apply, exactly as they do to
   * a shipped policy. A caller who wants their own lane rule supplies it as the
   * policy's `placement()` — except when `lane_pool` is also set, in which case
   * `makespan_under` wraps *any* policy, override included, in a
   * `pooled_placement_t`, so the override's own placement is silently replaced
   * rather than composed with the pool.
   *
   * Shared rather than raw because a config is copied into every
   * `prediction_result_t` it produces, so the referent outlives the call that
   * used it. That is also what lets a runtime written in Python be passed here.
   */
  std::shared_ptr<const runtime_t> runtime_override;

  /** Optional label, so a ranking reads as a decision rather than an index. */
  std::string name;
};

/** @brief A candidate's predicted cost. */
struct prediction_result_t {
  /**
   * Makespan, in cycles — the unit every cost model in this library answers in,
   * and the unit `simulate()` schedules in, so no conversion happens between
   * the model and this field. A caller who wants seconds names a clock and
   * calls `to_seconds` on the schedule, which is the one place a *schedule's*
   * cycles become a duration; ranking deliberately does not do it for them,
   * because a graph is device-independent and `rank_configs` is handed no
   * machine to take a clock from.
   *
   * Comparable across every `runtime_kind_t`, but only meaningfully so within
   * one: a flat `wg_duration` cycle count and a cost-model cycle count are the
   * same unit describing different things, so a candidate list should stay
   * within one family.
   */
  double latency = 0.0;

  graph_config_t config;        ///< the candidate
  std::size_t graph_index = 0;  ///< which graph, for the multi-graph entry points
  std::string graph_name;       ///< that graph's name, so results read as names
  bool rejected = false;        ///< true when the candidate was infeasible

  /** @brief Why the candidate was rejected; empty when it was not. */
  std::string rejection;
};

/** @brief The latency assigned to a rejected candidate. */
inline constexpr double kRejectedLatency = std::numeric_limits<double>::max();

/**
 * @brief Reason a candidate cannot be scheduled, if any.
 *
 * Checked before scheduling so an infeasible candidate is reported rather than
 * throwing from inside a runtime.
 *
 * @param graph Graph to schedule.
 * @param config Candidate schedule.
 * @return std::string A description, or empty when the candidate is feasible.
 */
std::string rejection_reason(const wg_graph_t& graph, const graph_config_t& config);

/**
 * @brief Predicted makespan for one graph under one candidate schedule.
 *
 * The atom underneath the ranking, as `compute_tile_latency` is for GEMM.
 *
 * @param graph Graph to schedule.
 * @param config Candidate schedule.
 * @param cost Cost model, overriding the graph's own; unset uses the graph's.
 *        Required by the kinds `needs_cost_model` names and ignored by the rest,
 *        which charge a flat `wg_duration`.
 * @return double Makespan in cycles, or `kRejectedLatency` when the candidate is
 *         infeasible.
 * @throws std::invalid_argument If a kind needing a cost model is asked for and
 *         neither this nor the graph carries one.
 */
double predict_latency(const wg_graph_t& graph,
                       const graph_config_t& config,
                       const cost_model_t* cost = nullptr);

// ─── one graph, many schedules ────────────────────────────────────────

/**
 * @brief Rank candidate schedules for one graph, best first.
 *
 * @param graph Graph to schedule.
 * @param configs Candidate schedules.
 * @param cost Cost model for the kinds that need one, overriding each graph's
 *        own; unset uses the graph's.
 * @return std::vector<prediction_result_t> Results, lowest latency first. If
 *         every candidate was rejected they are all returned at
 *         `kRejectedLatency`, in the order given.
 * @throws std::runtime_error If `configs` is empty.
 */
std::vector<prediction_result_t> rank_configs(const wg_graph_t& graph,
                                              const std::vector<graph_config_t>& configs,
                                              const cost_model_t* cost = nullptr);

/**
 * @brief Best candidate schedule for one graph.
 *
 * @param graph Graph to schedule.
 * @param configs Candidate schedules.
 * @param cost Cost model for the kinds that need one, overriding each graph's
 *        own; unset uses the graph's.
 * @return prediction_result_t The first entry of `rank_configs`.
 * @throws std::runtime_error If `configs` is empty.
 */
prediction_result_t select_config(const wg_graph_t& graph,
                                  const std::vector<graph_config_t>& configs,
                                  const cost_model_t* cost = nullptr);

/**
 * @brief Best `topk` candidate schedules for one graph.
 *
 * @param graph Graph to schedule.
 * @param configs Candidate schedules.
 * @param topk How many to keep; more than there are keeps them all.
 * @param cost Cost model for the kinds that need one, overriding each graph's
 *        own; unset uses the graph's.
 * @return std::vector<prediction_result_t> Up to `topk` results, best first.
 * @throws std::runtime_error If `configs` is empty.
 */
std::vector<prediction_result_t> select_topk_configs(const wg_graph_t& graph,
                                                     const std::vector<graph_config_t>& configs,
                                                     std::size_t topk,
                                                     const cost_model_t* cost = nullptr);

// ─── many graphs, many schedules ──────────────────────────────────────

/**
 * @brief Rank the whole graph-by-schedule cross product, best first.
 *
 * @param graphs Candidate graphs; none may be null.
 * @param configs Candidate schedules, each tried against every graph.
 * @param cost Cost model for the kinds that need one, overriding each graph's
 *        own; unset uses the graph's.
 * @return std::vector<prediction_result_t> One result per pair, best first.
 * @throws std::runtime_error If either list is empty.
 * @throws std::invalid_argument If any graph pointer is null.
 */
std::vector<prediction_result_t> rank_graphs(const std::vector<const wg_graph_t*>& graphs,
                                             const std::vector<graph_config_t>& configs,
                                             const cost_model_t* cost = nullptr);

/**
 * @brief Rank against a machine, binding hardware at ranking rather than earlier.
 *
 * This is where hardware enters, and the reason it enters this late is that a
 * graph is device-independent: the same instantiation can be ranked for several
 * machines without re-deriving a single edge.
 *
 * Two things follow from being given a machine. Any graph that carries
 * specification-owned node costs and has no model of its own is priced with an
 * `expr_cost_t` bound to this hardware, which is what lets a caller rank a
 * specification without mentioning cost at all. And any scheduling resource the
 * caller left alone is filled in: an unset `lanes` becomes the device's compute
 * unit count, and chiplet geometry still at its defaults is replaced by the
 * device's. Anything set explicitly is left alone.
 *
 * @param graphs Candidate graphs; none may be null.
 * @param configs Candidate schedules, each tried against every graph.
 * @param hardware Machine to rank for.
 * @param cost Cost model overriding both the graphs' own and any derived from
 *        their specifications; unset lets those apply.
 * @return std::vector<prediction_result_t> One result per pair, best first.
 * @throws std::runtime_error If either list is empty.
 * @throws std::invalid_argument If any graph pointer is null.
 */
std::vector<prediction_result_t> rank_graphs(const std::vector<const wg_graph_t*>& graphs,
                                             const std::vector<graph_config_t>& configs,
                                             const origami::comm::hardware_t& hardware,
                                             const cost_model_t* cost = nullptr);

/**
 * @brief Rank against hardware bindings given directly.
 *
 * The general form, and the one to reach for when a model needs a name the
 * machine description does not carry. A peak matrix-core rate is the standing
 * example: it depends on the instruction shape a kernel chose, which is a
 * configuration property rather than a property of the part, so no honest
 * conversion from a hardware struct can supply it. Start from `hardware_params`
 * and add what your cost expressions reference.
 *
 * `num_cus`, `num_xcds` and `cu_per_xcd` are read from these bindings to fill
 * scheduling resources the caller left unset; any that are absent are left to
 * the config's own values.
 *
 * @param graphs Candidate graphs; none may be null.
 * @param configs Candidate schedules, each tried against every graph.
 * @param hardware Hardware-scope bindings.
 * @param cost Cost model overriding both the graphs' own and any derived from
 *        their specifications; unset lets those apply.
 * @return std::vector<prediction_result_t> One result per pair, best first.
 * @throws std::runtime_error If either list is empty.
 * @throws std::invalid_argument If any graph pointer is null.
 */
std::vector<prediction_result_t> rank_graphs(const std::vector<const wg_graph_t*>& graphs,
                                             const std::vector<graph_config_t>& configs,
                                             const param_map_t& hardware,
                                             const cost_model_t* cost = nullptr);

/**
 * @brief Best graph-and-schedule pair.
 *
 * @param graphs Candidate graphs; none may be null.
 * @param configs Candidate schedules.
 * @param cost Cost model for the kinds that need one, overriding each graph's
 *        own; unset uses the graph's.
 * @return prediction_result_t The first entry of `rank_graphs`.
 * @throws std::runtime_error If either list is empty.
 */
prediction_result_t select_graph(const std::vector<const wg_graph_t*>& graphs,
                                 const std::vector<graph_config_t>& configs,
                                 const cost_model_t* cost = nullptr);

/**
 * @brief Best `topk` graph-and-schedule pairs.
 *
 * @param graphs Candidate graphs; none may be null.
 * @param configs Candidate schedules.
 * @param topk How many to keep.
 * @param cost Cost model for the kinds that need one, overriding each graph's
 *        own; unset uses the graph's.
 * @return std::vector<prediction_result_t> Up to `topk` results, best first.
 * @throws std::runtime_error If either list is empty.
 */
std::vector<prediction_result_t> select_topk_graphs(const std::vector<const wg_graph_t*>& graphs,
                                                    const std::vector<graph_config_t>& configs,
                                                    std::size_t topk,
                                                    const cost_model_t* cost = nullptr);

}  // namespace origami::graphs
