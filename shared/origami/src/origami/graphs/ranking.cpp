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

#include "origami/graphs/ranking.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "origami/graphs/analysis.hpp"
#include "origami/graphs/runtime.hpp"
#include "origami/graphs/simulate.hpp"
#include "origami/graphs/spec.hpp"

namespace origami::graphs {

// ─── runtime_kind_t ───────────────────────────────────────────────────

const char* runtime_kind_name(runtime_kind_t kind) {
  switch (kind) {
    case runtime_kind_t::breadth_first: return "breadth_first";
    case runtime_kind_t::asap: return "asap";
    case runtime_kind_t::depth_first: return "depth_first";
    case runtime_kind_t::roofline: return "roofline";
    case runtime_kind_t::event_driven: return "event_driven";
    case runtime_kind_t::xcd: return "xcd";
  }
  throw std::invalid_argument("runtime_kind_name: unknown runtime kind");
}

runtime_kind_t runtime_kind_from_name(const std::string& name) {
  if (name == "breadth_first") return runtime_kind_t::breadth_first;
  if (name == "asap") return runtime_kind_t::asap;
  if (name == "depth_first") return runtime_kind_t::depth_first;
  if (name == "roofline") return runtime_kind_t::roofline;
  if (name == "event_driven") return runtime_kind_t::event_driven;
  if (name == "xcd") return runtime_kind_t::xcd;
  throw std::invalid_argument("runtime_kind_from_name: unknown runtime kind '" + name + "'");
}

bool needs_cost_model(runtime_kind_t kind) {
  switch (kind) {
    case runtime_kind_t::breadth_first:
    case runtime_kind_t::asap:
    case runtime_kind_t::depth_first: return false;
    case runtime_kind_t::roofline:
    case runtime_kind_t::event_driven:
    case runtime_kind_t::xcd: return true;
  }
  throw std::invalid_argument("needs_cost_model: unknown runtime kind");
}

// ─── feasibility ──────────────────────────────────────────────────────

std::string rejection_reason(const wg_graph_t& graph, const graph_config_t& config) {
  if (config.lanes && *config.lanes < 1) {
    return "lanes is " + std::to_string(*config.lanes) + ", which is below one";
  }

  // A pool naming lanes the machine does not have is almost always a mistake in
  // the sweep rather than an intent to fall back to every lane, so it is
  // rejected here even though the placement would quietly widen it.
  if (config.lanes) {
    for (const auto& [op, pool] : config.lane_pool) {
      for (int lane : pool) {
        if (lane < 0 || lane >= *config.lanes) {
          return "lane_pool for '" + op + "' names lane " + std::to_string(lane) +
                 ", outside the " + std::to_string(*config.lanes) + " lanes available";
        }
      }
    }
  }

  for (const auto& [op, pool] : config.lane_pool) {
    (void)pool;
    try {
      graph.op_index(op);
    } catch (const std::out_of_range&) {
      return "lane_pool names operation '" + op + "', which the graph does not have";
    }
  }

  // An override replaces only the policy `runtime` names, not the cost model
  // beneath it: `predict_accepted` still falls back to the flat `wg_duration`
  // charge whenever no cost model is available anywhere, override or not, so
  // that field is checked unconditionally rather than being skipped just
  // because a policy was supplied. The xcd geometry is different — it is read
  // only by `makespan_of`'s own xcd arm, which an override bypasses entirely —
  // so that check stays confined to the no-override case.
  if (config.wg_duration < 1 &&
      (config.runtime_override != nullptr || !needs_cost_model(config.runtime))) {
    return "wg_duration is " + std::to_string(config.wg_duration) + ", which is below one";
  }
  if (config.runtime_override == nullptr) {
    if (config.runtime == runtime_kind_t::xcd &&
        (config.xcd.num_xcds < 1 || config.xcd.cus_per_xcd < 1)) {
      return "xcd geometry has a die or compute-unit count below one";
    }
  }

  if (!is_acyclic(graph)) return "graph contains a cycle";
  return {};
}

// ─── prediction ───────────────────────────────────────────────────────

namespace {

/**
 * Prices every node at a fixed number of cycles and every hop at zero, which is
 * what `wg_duration` means: a ranking that counts workgroups rather than
 * modelling what each one really costs, and so needs no calibrated machine.
 */
class flat_cost_t : public cost_model_t {
 public:
  explicit flat_cost_t(double cycles) : cycles_(cycles) {}
  double node_cycles(const wg_node_t&) const override { return cycles_; }
  double edge_cycles(const edge_t&) const override { return 0.0; }

 private:
  double cycles_;
};

/**
 * Schedule under a policy, applying `lane_pool` if the candidate has one.
 *
 * The lane count is recomputed the way simulate() does, since a pool has to be
 * clipped against the machine the simulator will actually build.
 *
 * `lane_pool` says how the machine is divided between operations, which is a
 * property of the placement rather than of the issuing order, so it composes
 * with any policy via `runtime_with_placement_t` instead of being a policy of
 * its own. `pooled_placement_t` resolves operation names against the graph and
 * clips pools to the machine, so it has to be built per candidate rather than
 * held in a config.
 */
double makespan_under(const wg_graph_t& graph,
                      const graph_config_t& config,
                      const runtime_t& runtime,
                      const cost_model_t& cost,
                      const simulate_options_t& opts) {
  if (config.lane_pool.empty()) return simulate(graph, runtime, cost, opts).makespan();

  const int lane_count =
      opts.lanes ? *opts.lanes : static_cast<int>(std::max<std::size_t>(graph.num_nodes(), 1));
  const pooled_placement_t placement(config.lane_pool, graph, lane_count);
  const runtime_with_placement_t pooled(runtime, placement);
  return simulate(graph, pooled, cost, opts).makespan();
}

/** Schedule under the policy `config.runtime` names. */
double makespan_of(const wg_graph_t& graph,
                   const graph_config_t& config,
                   const cost_model_t& cost) {
  simulate_options_t opts;
  opts.lanes              = config.lanes;
  opts.serialize_wg_iters = config.serialize_wg_iters;
  // The whole of what `event_driven` adds to `roofline`: same policy, same
  // placement, each node re-priced against the lanes busy when it dispatches.
  opts.reprice_on_dispatch = config.runtime == runtime_kind_t::event_driven;

  switch (config.runtime) {
    case runtime_kind_t::breadth_first:
      return makespan_under(graph, config, breadth_first_runtime_t{}, cost, opts);
    case runtime_kind_t::depth_first:
      return makespan_under(graph, config, depth_first_runtime_t{}, cost, opts);
    case runtime_kind_t::asap:
    case runtime_kind_t::roofline:
    case runtime_kind_t::event_driven:
      return makespan_under(graph, config, asap_runtime_t{}, cost, opts);
    case runtime_kind_t::xcd: {
      const xcd_runtime_t runtime(config.xcd.num_xcds, config.xcd.cus_per_xcd);
      // The chiplet placement indexes the free-time array for a whole die, so
      // the machine has to be exactly as wide as the geometry says; a lane
      // count from anywhere else would be an out-of-bounds read. The pool is
      // ignored for the same reason: this policy's placement *is* the geometry.
      opts.lanes       = runtime.lane_count();
      opts.skip_prefix = config.xcd.skip_prefix;
      return simulate(graph, runtime, cost, opts).makespan();
    }
  }
  throw std::invalid_argument("predict_latency: unknown runtime kind");
}

/**
 * Predict for a candidate already known to be feasible.
 *
 * Split out because deciding feasibility is not free — it is a topological sort
 * — and a ranking has already made that decision by the time it wants a number.
 * Sharing the answer keeps a ranking to one `is_acyclic` per candidate rather
 * than two.
 */
double predict_accepted(const wg_graph_t& graph,
                        const graph_config_t& config,
                        const cost_model_t* cost) {
  if (cost == nullptr) cost = graph.cost();

  // An override is a policy and says nothing about price, so it takes whatever
  // cost model is available and falls back to the flat charge when there is
  // none — a caller's own policy works with or without a calibrated machine.
  const bool from_model =
      config.runtime_override != nullptr ? cost != nullptr : needs_cost_model(config.runtime);
  if (from_model && cost == nullptr) {
    throw std::invalid_argument(std::string("predict_latency: the ") +
                                runtime_kind_name(config.runtime) +
                                " candidate prices every node from a cost model and was given "
                                "none; call set_cost on the graph or pass one here, or choose a "
                                "kind that charges a flat wg_duration instead");
  }

  const flat_cost_t flat(static_cast<double>(config.wg_duration));
  const cost_model_t& priced = from_model ? *cost : static_cast<const cost_model_t&>(flat);

  if (config.runtime_override != nullptr) {
    simulate_options_t opts;
    opts.lanes              = config.lanes;
    opts.serialize_wg_iters = config.serialize_wg_iters;
    return makespan_under(graph, config, *config.runtime_override, priced, opts);
  }
  return makespan_of(graph, config, priced);
}

}  // namespace

double predict_latency(const wg_graph_t& graph,
                       const graph_config_t& config,
                       const cost_model_t* cost) {
  if (!rejection_reason(graph, config).empty()) return kRejectedLatency;
  return predict_accepted(graph, config, cost);
}

// ─── ranking ──────────────────────────────────────────────────────────

namespace {

/**
 * Sort best first, keeping the input order among equals.
 *
 * A stable sort is the tie-break: `origami::rank_configs` breaks ties on tile
 * geometry, but a schedule has no comparable notion of "bigger is better", so
 * the honest answer is that equal predictions stay in the order the caller
 * offered them. Two schedules with the same makespan really are equivalent to
 * this model, and pretending otherwise would invent a preference.
 */
void order_best_first(std::vector<prediction_result_t>& results) {
  std::stable_sort(results.begin(),
                   results.end(),
                   [](const prediction_result_t& a, const prediction_result_t& b) {
                     return a.latency < b.latency;
                   });
}

prediction_result_t evaluate(const wg_graph_t& graph,
                             const graph_config_t& config,
                             const cost_model_t* cost,
                             std::size_t graph_index) {
  prediction_result_t out;
  out.config      = config;
  out.graph_index = graph_index;
  out.graph_name  = graph.name();
  out.rejection   = rejection_reason(graph, config);
  out.rejected    = !out.rejection.empty();
  out.latency     = out.rejected ? kRejectedLatency : predict_accepted(graph, config, cost);
  return out;
}

/**
 * Drop rejected candidates, unless that would leave nothing.
 *
 * The contract copied from `rank_configs`: a caller that offered only
 * infeasible candidates still gets a ranking, at maximum latency, rather than an
 * exception. Reporting "all of these are impossible, here they are" is more
 * useful than throwing, because the rejection strings say why.
 */
void drop_rejected_unless_empty(std::vector<prediction_result_t>& results) {
  const bool any_feasible = std::any_of(
      results.begin(), results.end(), [](const prediction_result_t& r) { return !r.rejected; });
  if (!any_feasible) return;
  results.erase(
      std::remove_if(
          results.begin(), results.end(), [](const prediction_result_t& r) { return r.rejected; }),
      results.end());
}

std::vector<prediction_result_t> take(std::vector<prediction_result_t> results, std::size_t topk) {
  if (results.size() > topk) results.resize(topk);
  return results;
}

}  // namespace

std::vector<prediction_result_t> rank_configs(const wg_graph_t& graph,
                                              const std::vector<graph_config_t>& configs,
                                              const cost_model_t* cost) {
  if (configs.empty()) throw std::runtime_error("rank_configs: no configurations provided");

  std::vector<prediction_result_t> results;
  results.reserve(configs.size());
  for (const graph_config_t& config : configs) results.push_back(evaluate(graph, config, cost, 0));

  drop_rejected_unless_empty(results);
  order_best_first(results);
  return results;
}

prediction_result_t select_config(const wg_graph_t& graph,
                                  const std::vector<graph_config_t>& configs,
                                  const cost_model_t* cost) {
  return rank_configs(graph, configs, cost).front();
}

std::vector<prediction_result_t> select_topk_configs(const wg_graph_t& graph,
                                                     const std::vector<graph_config_t>& configs,
                                                     std::size_t topk,
                                                     const cost_model_t* cost) {
  return take(rank_configs(graph, configs, cost), topk);
}

std::vector<prediction_result_t> rank_graphs(const std::vector<const wg_graph_t*>& graphs,
                                             const std::vector<graph_config_t>& configs,
                                             const cost_model_t* cost) {
  if (graphs.empty()) throw std::runtime_error("rank_graphs: no graphs provided");
  if (configs.empty()) throw std::runtime_error("rank_graphs: no configurations provided");

  std::vector<prediction_result_t> results;
  results.reserve(graphs.size() * configs.size());
  for (std::size_t g = 0; g < graphs.size(); ++g) {
    if (graphs[g] == nullptr) {
      throw std::invalid_argument("rank_graphs: graph " + std::to_string(g) + " is null");
    }
    for (const graph_config_t& config : configs) {
      results.push_back(evaluate(*graphs[g], config, cost, g));
    }
  }

  drop_rejected_unless_empty(results);
  order_best_first(results);
  return results;
}

namespace {

/**
 * Fill scheduling resources the caller left alone from the machine.
 *
 * "Left alone" is unambiguous for `lanes`, which is optional. Chiplet geometry
 * has no unset state, so a config still carrying the struct's own defaults is
 * treated as unstated; anything else is the caller's decision and survives.
 */
std::vector<graph_config_t> resolve_against(const std::vector<graph_config_t>& configs,
                                            const param_map_t& hardware) {
  static const xcd_options_t kDefaults;

  const auto read = [&hardware](const char* name) -> std::optional<int> {
    if (!hardware.contains(name)) return std::nullopt;
    return static_cast<int>(hardware.index_at(name, scope_t::hardware));
  };

  const std::optional<int> num_cus    = read("num_cus");
  const std::optional<int> num_xcds   = read("num_xcds");
  const std::optional<int> cu_per_xcd = read("cu_per_xcd");

  std::vector<graph_config_t> out = configs;
  for (graph_config_t& config : out) {
    if (!config.lanes && num_cus) config.lanes = *num_cus;

    const bool geometry_untouched = config.xcd.num_xcds == kDefaults.num_xcds &&
                                    config.xcd.cus_per_xcd == kDefaults.cus_per_xcd;
    if (geometry_untouched && num_xcds && cu_per_xcd) {
      config.xcd.num_xcds    = *num_xcds;
      config.xcd.cus_per_xcd = *cu_per_xcd;
    }
  }
  return out;
}

}  // namespace

std::vector<prediction_result_t> rank_graphs(const std::vector<const wg_graph_t*>& graphs,
                                             const std::vector<graph_config_t>& configs,
                                             const origami::comm::hardware_t& hardware,
                                             const cost_model_t* cost) {
  return rank_graphs(graphs, configs, hardware_params(hardware), cost);
}

std::vector<prediction_result_t> rank_graphs(const std::vector<const wg_graph_t*>& graphs,
                                             const std::vector<graph_config_t>& configs,
                                             const param_map_t& hardware,
                                             const cost_model_t* cost) {
  if (graphs.empty()) throw std::runtime_error("rank_graphs: no graphs provided");
  if (configs.empty()) throw std::runtime_error("rank_graphs: no configurations provided");

  const param_map_t& hardware_scope          = hardware;
  const std::vector<graph_config_t> resolved = resolve_against(configs, hardware);

  // Derived models are owned here so they outlive every evaluation below; the
  // graphs are const, so they cannot be given the model to hold themselves.
  std::vector<std::shared_ptr<const expr_cost_t>> derived;

  std::vector<prediction_result_t> results;
  results.reserve(graphs.size() * resolved.size());

  for (std::size_t g = 0; g < graphs.size(); ++g) {
    if (graphs[g] == nullptr) {
      throw std::invalid_argument("rank_graphs: graph " + std::to_string(g) + " is null");
    }

    const cost_model_t* applied = cost;
    if (applied == nullptr && graphs[g]->cost() == nullptr) {
      const auto* structured = dynamic_cast<const graph_t*>(graphs[g]);
      if (structured != nullptr && expr_cost_t::priced(*structured)) {
        derived.push_back(std::make_shared<expr_cost_t>(*structured, hardware_scope));
        applied = derived.back().get();
      }
    }

    for (const graph_config_t& config : resolved) {
      results.push_back(evaluate(*graphs[g], config, applied, g));
    }
  }

  drop_rejected_unless_empty(results);
  order_best_first(results);
  return results;
}

prediction_result_t select_graph(const std::vector<const wg_graph_t*>& graphs,
                                 const std::vector<graph_config_t>& configs,
                                 const cost_model_t* cost) {
  return rank_graphs(graphs, configs, cost).front();
}

std::vector<prediction_result_t> select_topk_graphs(const std::vector<const wg_graph_t*>& graphs,
                                                    const std::vector<graph_config_t>& configs,
                                                    std::size_t topk,
                                                    const cost_model_t* cost) {
  return take(rank_graphs(graphs, configs, cost), topk);
}

}  // namespace origami::graphs
