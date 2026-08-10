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

#include "origami/graphs/cost_runtime.hpp"
#include "origami/graphs/runtime.hpp"
#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

/** @brief Which scheduling policy a candidate uses. */
enum class runtime_kind_t {
  breadth_first,  ///< no overlap: one operation at a time
  asap,           ///< producer-greedy, integer timesteps
  depth_first,    ///< chain-greedy, integer timesteps
  roofline,       ///< continuous time, priced once up front
  event_driven,   ///< continuous time, re-priced against contention at dispatch
  xcd,            ///< continuous time, chiplet round-robin with work stealing
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

/** @brief Whether a policy schedules in real time rather than integer timesteps. */
bool is_continuous(runtime_kind_t kind);

/**
 * @brief One candidate schedule: the performance input, as `config_t` is for GEMM.
 */
struct graph_config_t {
  runtime_kind_t runtime = runtime_kind_t::roofline;  ///< scheduling policy

  /** Execution slots. Unset means unlimited, i.e. one per atom. */
  std::optional<int> lanes;

  /** Run each workgroup's iterations in sequence, on one lane. */
  bool serialize_wg_iters = false;

  /** Per-operation lane partition; only meaningful for the continuous policies. */
  lane_pool_t lane_pool;

  /** Timestep length for the integer policies. */
  index_t wg_duration = 1;

  /** Chiplet geometry, used only by `runtime_kind_t::xcd`. */
  xcd_options_t xcd;

  /** Seconds to display units for the continuous policies. */
  double scale = 1e6;

  /**
   * Escape hatch: when set, this policy is used verbatim and every field above
   * is ignored. Mirrors `comm_config_t::algorithm_override` — an enum covers the
   * policies worth naming, and a pointer covers the one a caller invented.
   *
   * Shared rather than raw because a config is copied into every
   * `prediction_result_t` it produces, so the referent outlives the call that
   * used it. That is also what lets a runtime written in Python be passed here.
   */
  std::shared_ptr<const cost_runtime_t> runtime_override;

  /** Optional label, so a ranking reads as a decision rather than an index. */
  std::string name;
};

/** @brief A candidate's predicted cost, in the units its policy produced. */
struct prediction_result_t {
  /**
   * Makespan. Timesteps for the integer policies, `graph_config_t::scale` units
   * for the continuous ones — microseconds by default. Comparing across the two
   * families is meaningless, and `rank_configs` does not stop a caller mixing
   * them, so a candidate list should stay within one family.
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
 *        Required by the continuous policies and ignored by the integer ones,
 *        which count timesteps.
 * @return double Makespan, or `kRejectedLatency` when the candidate is infeasible.
 * @throws std::invalid_argument If a continuous policy is asked for and neither
 *         this nor the graph carries a cost model.
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
 * @param cost Cost model for the continuous policies, overriding each graph's
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
 * @param cost Cost model for the continuous policies, overriding each graph's
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
 * @param cost Cost model for the continuous policies, overriding each graph's
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
 * @param cost Cost model for the continuous policies, overriding each graph's
 *        own; unset uses the graph's.
 * @return std::vector<prediction_result_t> One result per pair, best first.
 * @throws std::runtime_error If either list is empty.
 * @throws std::invalid_argument If any graph pointer is null.
 */
std::vector<prediction_result_t> rank_graphs(const std::vector<const wg_graph_t*>& graphs,
                                             const std::vector<graph_config_t>& configs,
                                             const cost_model_t* cost = nullptr);

/**
 * @brief Best graph-and-schedule pair.
 *
 * @param graphs Candidate graphs; none may be null.
 * @param configs Candidate schedules.
 * @param cost Cost model for the continuous policies, overriding each graph's
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
 * @param cost Cost model for the continuous policies, overriding each graph's
 *        own; unset uses the graph's.
 * @return std::vector<prediction_result_t> Up to `topk` results, best first.
 * @throws std::runtime_error If either list is empty.
 */
std::vector<prediction_result_t> select_topk_graphs(const std::vector<const wg_graph_t*>& graphs,
                                                    const std::vector<graph_config_t>& configs,
                                                    std::size_t topk,
                                                    const cost_model_t* cost = nullptr);

}  // namespace origami::graphs
