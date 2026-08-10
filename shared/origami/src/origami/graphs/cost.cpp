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

#include "origami/graphs/cost.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace origami::graphs {

// ─── cost_kind_t ──────────────────────────────────────────────────────

const char* cost_kind_name(cost_kind_t kind) {
  switch (kind) {
    case cost_kind_t::comm: return "comm";
    case cost_kind_t::gemm: return "gemm";
    case cost_kind_t::roofline: return "roofline";
    case cost_kind_t::custom: return "custom";
  }
  throw std::invalid_argument("cost_kind_name: unknown cost kind");
}

cost_kind_t cost_kind_from_name(const std::string& name) {
  if (name == "comm") return cost_kind_t::comm;
  if (name == "gemm") return cost_kind_t::gemm;
  if (name == "roofline") return cost_kind_t::roofline;
  if (name == "custom") return cost_kind_t::custom;
  throw std::invalid_argument("cost_kind_from_name: unknown cost kind '" + name +
                              "' (expected comm, gemm, roofline or custom)");
}

// ─── roofline arm ─────────────────────────────────────────────────────

double roofline_hardware_t::ridge_intensity() const {
  return hbm_bw != 0.0 ? peak_flops / hbm_bw : std::numeric_limits<double>::infinity();
}

roofline_spec_t roofline_spec_t::gemm_tile(double bm, double bn, double k, double dtype_bytes) {
  roofline_spec_t out;
  out.flops     = 2.0 * bm * bn * k;
  out.hbm_bytes = (bm * k + k * bn + bm * bn) * dtype_bytes;
  return out;
}

roofline_spec_t roofline_spec_t::comm_step(double elements, double dtype_bytes, double fan) {
  roofline_spec_t out;
  out.link_bytes = elements * dtype_bytes * fan;
  return out;
}

namespace {

/**
 * The three roofs, in seconds.
 *
 * A zero-work axis contributes zero rather than a division, which matters for
 * the link roof: charging its fixed per-hop latency to an operation that ships
 * nothing would make every compute node look like it touched the fabric.
 */
struct roofs_t {
  double compute = 0.0;
  double hbm     = 0.0;
  double link    = 0.0;
};

roofs_t roofs_of(const roofline_spec_t& spec, const roofline_hardware_t& hw) {
  roofs_t out;
  out.compute = spec.flops != 0.0 ? spec.flops / hw.peak_flops : 0.0;
  out.hbm     = spec.hbm_bytes != 0.0 ? spec.hbm_bytes / hw.hbm_bw : 0.0;
  out.link    = spec.link_bytes != 0.0 ? spec.link_bytes / hw.link_bw + hw.link_latency : 0.0;
  return out;
}

}  // namespace

double roofline_seconds(const roofline_spec_t& spec, const roofline_hardware_t& hardware) {
  const roofs_t r = roofs_of(spec, hardware);
  return std::max(r.compute, std::max(r.hbm, r.link));
}

const char* roofline_bound(const roofline_spec_t& spec, const roofline_hardware_t& hardware) {
  if (spec.flops == 0.0 && spec.hbm_bytes == 0.0 && spec.link_bytes == 0.0) return "none";
  const roofs_t r = roofs_of(spec, hardware);
  if (r.compute >= r.hbm && r.compute >= r.link) return "compute";
  return r.hbm >= r.link ? "hbm" : "link";
}

// ─── comm arm ─────────────────────────────────────────────────────────

comm::wg_tile_geometry_t comm_spec_t::geometry(std::size_t cacheline_bytes) const {
  if (tile) return comm::wg_tile_geometry_t::from_shape(*tile, cacheline_bytes);

  // Without a shape the tile is one flat byte run. Elements assume bf16, which
  // is what the reference adapter does; supply a tile to say otherwise. Both
  // counts are clamped, because zero of either is not a tile the model can walk.
  const std::size_t lines =
      cacheline_bytes != 0 ? (wg_tile_bytes + cacheline_bytes - 1) / cacheline_bytes : 0;
  return comm::wg_tile_geometry_t{
      std::max<std::size_t>(lines, 1), std::max<std::size_t>(wg_tile_bytes / 2, 1), std::nullopt};
}

comm::wg_tile_latency_breakdown_t comm_breakdown(const comm_spec_t& spec,
                                                 const comm::system_t& system,
                                                 std::optional<int> active_cus,
                                                 const comm::heuristics_t& heur) {
  if (spec.num_wgs < 1) {
    throw std::invalid_argument("comm_spec_t: num_wgs must be at least 1 (got " +
                                std::to_string(spec.num_wgs) + ")");
  }

  const comm::comm_config_t config{spec.num_wgs};
  const comm::latency_context_t ctx{config, system, heur, spec.primitive};

  // The reference defaults contention to the channel count, on the grounds that
  // the workgroups of one collective are the ones contending with each other.
  return comm::compute_wg_tile_latency(spec.work_graph,
                                       spec.geometry(system.gpu.cacheline_bytes),
                                       spec.bw_per_wg,
                                       active_cus.value_or(spec.num_wgs),
                                       ctx);
}

double comm_seconds(const comm_spec_t& spec,
                    const comm::system_t& system,
                    std::optional<int> active_cus,
                    const comm::heuristics_t& heur) {
  const comm::wg_tile_latency_breakdown_t breakdown =
      comm_breakdown(spec, system, active_cus, heur);
  return system.gpu.cycles_to_seconds(breakdown.T_total_cycles);
}

// ─── op_cost_t ────────────────────────────────────────────────────────

op_cost_t op_cost_t::from_roofline(roofline_spec_t spec) {
  op_cost_t out;
  out.kind_ = cost_kind_t::roofline;
  out.body_ = std::move(spec);
  return out;
}

op_cost_t op_cost_t::from_comm(comm_spec_t spec) {
  op_cost_t out;
  out.kind_ = cost_kind_t::comm;
  out.body_ = std::move(spec);
  return out;
}

op_cost_t op_cost_t::from_custom(custom_cost_fn_t fn) {
  return tagged(cost_kind_t::custom, std::move(fn));
}

op_cost_t op_cost_t::tagged(cost_kind_t kind, custom_cost_fn_t fn) {
  if (!fn) {
    throw std::invalid_argument(std::string("op_cost_t: the ") + cost_kind_name(kind) +
                                " arm needs a callable function");
  }
  op_cost_t out;
  out.kind_ = kind;
  out.body_ = std::move(fn);
  return out;
}

const roofline_spec_t& op_cost_t::roofline() const {
  if (kind_ != cost_kind_t::roofline) {
    throw std::invalid_argument(std::string("op_cost_t::roofline: this entry is priced by the ") +
                                cost_kind_name(kind_) + " arm");
  }
  return std::get<roofline_spec_t>(body_);
}

const comm_spec_t& op_cost_t::comm() const {
  if (kind_ != cost_kind_t::comm) {
    throw std::invalid_argument(std::string("op_cost_t::comm: this entry is priced by the ") +
                                cost_kind_name(kind_) + " arm");
  }
  return std::get<comm_spec_t>(body_);
}

// ─── cost_table_t ─────────────────────────────────────────────────────

cost_table_t::cost_table_t(const wg_graph_t& graph, cost_settings_t settings)
    : graph_(&graph)
    , settings_(std::move(settings))
    , by_op_(static_cast<std::size_t>(graph.num_operations())) {}

cost_table_t& cost_table_t::set(const std::string& op_name, op_cost_t cost) {
  // op_index throws for an unknown name, which is the point: a typo in an
  // operation name would otherwise silently leave it priced at the default.
  const int op                         = graph_->op_index(op_name);
  by_op_[static_cast<std::size_t>(op)] = std::move(cost);
  return *this;
}

bool cost_table_t::has(int op) const {
  if (op < 0 || static_cast<std::size_t>(op) >= by_op_.size()) return false;
  return by_op_[static_cast<std::size_t>(op)].has_value();
}

std::optional<cost_kind_t> cost_table_t::kind_of(int op) const {
  if (!has(op)) return std::nullopt;
  return by_op_[static_cast<std::size_t>(op)]->kind();
}

double cost_table_t::price(int op, const wg_node_t& node, std::optional<int> active_cus) const {
  if (!has(op)) return settings_.default_seconds;
  const op_cost_t& entry = *by_op_[static_cast<std::size_t>(op)];

  switch (entry.kind()) {
    case cost_kind_t::roofline:
      // Static by construction: a roofline has no view on who else is running.
      return roofline_seconds(std::get<roofline_spec_t>(entry.body_), settings_.roofline_hardware);

    case cost_kind_t::comm: {
      if (!settings_.comm_system) {
        throw std::invalid_argument(
            "cost_table_t: operation '" + graph_->op_name(op) +
            "' is priced by the comm arm, but cost_settings_t::comm_system is unset; build one "
            "with origami::comm::make_system or origami::comm::system_from_device");
      }
      return comm_seconds(std::get<comm_spec_t>(entry.body_),
                          *settings_.comm_system,
                          active_cus,
                          settings_.heuristics);
    }

    // Both of these carry a function. The gemm arm's comes from the HIP-linked
    // bridge target, which is why there is no gemm-specific case here.
    case cost_kind_t::custom:
    case cost_kind_t::gemm: {
      const custom_cost_fn_t& fn = std::get<custom_cost_fn_t>(entry.body_);
      // No contention level to pass on means "as if alone".
      return fn(node, active_cus.value_or(1));
    }
  }
  throw std::invalid_argument("cost_table_t: unknown cost kind");
}

double cost_table_t::node_cost(const wg_node_t& node) const {
  return price(node.op, node, std::nullopt);
}

double cost_table_t::node_cost_at(const wg_node_t& node, int active_cus) const {
  return price(node.op, node, active_cus);
}

double cost_table_t::edge_cost(const edge_t& edge) const {
  return settings_.edge_cost ? settings_.edge_cost(edge) : 0.0;
}

}  // namespace origami::graphs
