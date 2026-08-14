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
#include <string>
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

double roofline_hardware_t::peak_flops() const {
  if (!(compute_clock_ghz > 0.0)) {
    throw std::invalid_argument(
        "roofline_hardware_t::peak_flops: compute_clock_ghz must be positive, so the clock-free "
        "peak_flops_per_cycle can be expressed as a FLOP/s rate");
  }
  return peak_flops_per_cycle * compute_clock_ghz * 1e9;
}

double roofline_hardware_t::ridge_intensity() const {
  return hbm_bw != 0.0 ? peak_flops() / hbm_bw : std::numeric_limits<double>::infinity();
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

bool clocks_match(double lhs, double rhs) {
  // This check is meant to catch real configuration mistakes such as 2.0 vs
  // 2.1 GHz, not reject the last bit of two values computed by different
  // arithmetic paths.
  constexpr double rel_tol = 1e-9;
  const double scale       = std::max(1.0, std::max(std::abs(lhs), std::abs(rhs)));
  return std::abs(lhs - rhs) <= rel_tol * scale;
}

std::invalid_argument clock_mismatch_error(double comm_clock, double roof_clock) {
  return std::invalid_argument(
      "cost_table_t: cost_settings_t::comm_system's clock (" + std::to_string(comm_clock) +
      " GHz) disagrees with cost_settings_t::roofline_hardware.compute_clock_ghz (" +
      std::to_string(roof_clock) +
      " GHz); a table cannot sum cycles measured against two different clocks. Build "
      "comm_system at roofline_hardware.compute_clock_ghz (origami::comm::make_system's "
      "third argument), or set roofline_hardware.compute_clock_ghz to match comm_system's.");
}

/**
 * The three roofs, already in cycles.
 *
 * A zero-work axis contributes zero rather than a division, which matters for
 * the link roof: charging its fixed per-hop latency to an operation that ships
 * nothing would make every compute node look like it touched the fabric.
 *
 * Compute is `flops / peak_flops_per_cycle` directly — no clock involved, by
 * design (see roofline_hardware_t). HBM and link are continuous rates in
 * seconds, scaled into cycles by `hw.compute_clock_ghz` because those two
 * really do run in their own clock domain.
 */
struct roofs_t {
  double compute = 0.0;
  double hbm     = 0.0;
  double link    = 0.0;
};

roofs_t roofs_of(const roofline_spec_t& spec, const roofline_hardware_t& hw) {
  if (!(hw.compute_clock_ghz > 0.0)) {
    throw std::invalid_argument(
        "roofline_hardware_t: compute_clock_ghz must be positive, so the HBM and link roofs can "
        "be scaled from seconds into cycles (the compute roof needs no clock at all)");
  }
  const double clock_hz = hw.compute_clock_ghz * 1e9;
  roofs_t out;
  out.compute = spec.flops != 0.0 ? spec.flops / hw.peak_flops_per_cycle : 0.0;
  out.hbm     = spec.hbm_bytes != 0.0 ? (spec.hbm_bytes / hw.hbm_bw) * clock_hz : 0.0;
  out.link =
      spec.link_bytes != 0.0 ? (spec.link_bytes / hw.link_bw + hw.link_latency) * clock_hz : 0.0;
  return out;
}

}  // namespace

double roofline_cycles(const roofline_spec_t& spec, const roofline_hardware_t& hardware) {
  // No clock applied here: roofs_of already answers in cycles for all three
  // roofs, each by its own rule (see roofs_t).
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

double comm_cycles(const comm_spec_t& spec,
                   const comm::system_t& system,
                   std::optional<int> active_cus,
                   const comm::heuristics_t& heur) {
  const comm::wg_tile_latency_breakdown_t breakdown =
      comm_breakdown(spec, system, active_cus, heur);
  // No clock to delete here: compute_wg_tile_latency already answers in
  // cycles, and system.gpu.cycles_to_seconds used to be applied on the way
  // out. Returning T_total_cycles directly is the whole change.
  return breakdown.T_total_cycles;
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
  const int op = graph_->op_index(op_name);

  auto has_kind_after_set = [&](cost_kind_t kind) {
    for (std::size_t i = 0; i < by_op_.size(); ++i) {
      if (static_cast<int>(i) == op) {
        if (cost.kind() == kind) return true;
      } else if (by_op_[i] && by_op_[i]->kind() == kind) {
        return true;
      }
    }
    return false;
  };

  if (settings_.comm_system && has_kind_after_set(cost_kind_t::roofline) &&
      has_kind_after_set(cost_kind_t::comm)) {
    const double comm_clock = settings_.comm_system->gpu.clock_ghz;
    const double roof_clock = settings_.roofline_hardware.compute_clock_ghz;
    if (!clocks_match(comm_clock, roof_clock)) throw clock_mismatch_error(comm_clock, roof_clock);
  }

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
  if (!has(op)) return settings_.default_cycles;
  const op_cost_t& entry = *by_op_[static_cast<std::size_t>(op)];

  switch (entry.kind()) {
    case cost_kind_t::roofline:
      // Static by construction: a roofline has no view on who else is running.
      return roofline_cycles(std::get<roofline_spec_t>(entry.body_), settings_.roofline_hardware);

    case cost_kind_t::comm: {
      if (!settings_.comm_system) {
        throw std::invalid_argument(
            "cost_table_t: operation '" + graph_->op_name(op) +
            "' is priced by the comm arm, but cost_settings_t::comm_system is unset; build one "
            "with origami::comm::make_system or origami::comm::system_from_device");
      }
      return comm_cycles(std::get<comm_spec_t>(entry.body_),
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

double cost_table_t::node_cycles(const wg_node_t& node) const {
  return price(node.op, node, std::nullopt);
}

double cost_table_t::node_cycles_at(const wg_node_t& node, int active_cus) const {
  return price(node.op, node, active_cus);
}

double cost_table_t::edge_cycles(const edge_t& edge) const {
  return settings_.edge_cycles ? settings_.edge_cycles(edge) : 0.0;
}

}  // namespace origami::graphs
