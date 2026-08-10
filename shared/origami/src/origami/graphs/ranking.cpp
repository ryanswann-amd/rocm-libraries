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
#include <stdexcept>

#include "origami/graphs/analysis.hpp"

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

bool is_continuous(runtime_kind_t kind) {
  switch (kind) {
    case runtime_kind_t::breadth_first:
    case runtime_kind_t::asap:
    case runtime_kind_t::depth_first: return false;
    case runtime_kind_t::roofline:
    case runtime_kind_t::event_driven:
    case runtime_kind_t::xcd: return true;
  }
  throw std::invalid_argument("is_continuous: unknown runtime kind");
}

// ─── feasibility ──────────────────────────────────────────────────────

std::string rejection_reason(const wg_graph_t& graph, const graph_config_t& config) {
  if (config.runtime_override != nullptr) {
    // The caller supplied the policy, so the fields this would check are not the
    // ones that will be used. Only the graph itself can be rejected.
    return is_acyclic(graph) ? std::string() : std::string("graph contains a cycle");
  }

  if (config.lanes && *config.lanes < 1) {
    return "lanes is " + std::to_string(*config.lanes) + ", which is below one";
  }
  if (!is_continuous(config.runtime) && config.wg_duration < 1) {
    return "wg_duration is " + std::to_string(config.wg_duration) + ", which is below one";
  }
  if (config.scale <= 0.0) return "scale is not positive";
  if (config.runtime == runtime_kind_t::xcd &&
      (config.xcd.num_xcds < 1 || config.xcd.cus_per_xcd < 1)) {
    return "xcd geometry has a die or compute-unit count below one";
  }

  // A pool naming lanes the machine does not have is almost always a mistake in
  // the sweep rather than an intent to fall back to every lane, so it is
  // rejected here even though the runtime would quietly widen it.
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

  if (!is_acyclic(graph)) return "graph contains a cycle";
  return {};
}

// ─── prediction ───────────────────────────────────────────────────────

namespace {

cost_runtime_options_t continuous_options(const graph_config_t& config) {
  cost_runtime_options_t opts;
  opts.lanes              = config.lanes;
  opts.scale              = config.scale;
  opts.lane_pool          = config.lane_pool;
  opts.serialize_wg_iters = config.serialize_wg_iters;
  return opts;
}

double integer_makespan(const wg_graph_t& graph, const graph_config_t& config) {
  switch (config.runtime) {
    case runtime_kind_t::breadth_first:
      return static_cast<double>(
          breadth_first_runtime_t(config.lanes, config.wg_duration).schedule(graph).makespan());
    case runtime_kind_t::asap:
      return static_cast<double>(
          asap_runtime_t(config.lanes, config.wg_duration).schedule(graph).makespan());
    case runtime_kind_t::depth_first:
      return static_cast<double>(
          depth_first_runtime_t(config.lanes, config.wg_duration).schedule(graph).makespan());
    default: break;
  }
  throw std::invalid_argument("integer_makespan: not an integer-timestep runtime");
}

double continuous_makespan(const wg_graph_t& graph,
                           const graph_config_t& config,
                           const cost_model_t& cost) {
  switch (config.runtime) {
    case runtime_kind_t::roofline:
      return roofline_runtime_t(cost, continuous_options(config)).schedule(graph).makespan();
    case runtime_kind_t::event_driven:
      return event_driven_runtime_t(cost, continuous_options(config)).schedule(graph).makespan();
    case runtime_kind_t::xcd: {
      xcd_options_t opts = config.xcd;
      opts.scale         = config.scale;
      return xcd_runtime_t(cost, opts).schedule(graph).makespan();
    }
    default: break;
  }
  throw std::invalid_argument("continuous_makespan: not a continuous-time runtime");
}

/**
 * Predict for a candidate already known to be feasible.
 *
 * Split out because deciding feasibility is not free — with a runtime override
 * it is a topological sort — and a ranking has already made that decision by
 * the time it wants a number. Sharing the answer keeps a ranking to one
 * `is_acyclic` per candidate rather than two.
 */
double predict_accepted(const wg_graph_t& graph,
                        const graph_config_t& config,
                        const cost_model_t* cost) {
  if (config.runtime_override != nullptr) {
    return config.runtime_override->schedule(graph).makespan();
  }
  if (!is_continuous(config.runtime)) return integer_makespan(graph, config);

  if (cost == nullptr) cost = graph.cost();
  if (cost == nullptr) {
    throw std::invalid_argument(std::string("predict_latency: the ") +
                                runtime_kind_name(config.runtime) +
                                " runtime schedules in real time and needs a cost model; call "
                                "set_cost on the graph or pass one here, or choose an "
                                "integer-timestep runtime to count workgroups");
  }
  return continuous_makespan(graph, config, *cost);
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
