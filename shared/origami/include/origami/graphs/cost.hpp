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
 * @brief origami::graphs — cost model selection and the roofline and comm arms.
 *
 * A graph carries no cost. This is where hardware enters, and the shape of it is
 * one entry per operation: an operation is priced by an arm, named by
 * `cost_kind_t`, carrying whatever that arm needs. `cost_table_t` collects those
 * entries into something the runtimes can call.
 *
 * Cost lives here rather than on `operation_t` deliberately. Keeping it out of
 * the graph is what lets one topology be priced for several machines without
 * being rebuilt, and it is what keeps `graph_t` from depending on the comm model.
 * The reference makes the same split: its `CombinedCostModel` is a dictionary
 * from operation name to spec, not a field on the operation.
 *
 * Every arm answers in cycles, per `cost_model_t`. The comm arm always
 * was cycle-native internally; the roofline arm is not — except for its
 * compute roof, which is expressed per cycle and needs no clock at all (a
 * compute-bound tile costs the same cycles regardless of DVFS). The HBM and
 * link roofs are continuous rates in their own clock domains, so
 * `roofline_hardware_t::compute_clock_ghz` is what scales *those* into
 * cycles.
 *
 * Three arms are available here, and a fourth is deliberately absent:
 *
 *   - `cost_kind_t::roofline` is the coarse first cut — the largest of the
 *     compute, HBM and link roofs. It ignores cache reuse and contention, so a
 *     deep-K GEMM tile looks more memory-bound than it is, but it is enough to
 *     give a GEMM tile and a collective step honestly different widths, and it
 *     needs no calibration.
 *   - `cost_kind_t::comm` calls `origami::comm::compute_wg_tile_latency`, the
 *     calibrated model, and is the one to use for anything that touches the
 *     fabric. It is genuinely contention-aware: `active_cus` reaches the model,
 *     so a schedule run with `simulate_options_t::reprice_on_dispatch` gets real
 *     numbers rather than a constant.
 *   - `cost_kind_t::custom` takes a function, for a measured trace or a model
 *     that does not exist yet.
 *   - `cost_kind_t::gemm` calls origami's analytical GEMM model, which is
 *     HIP-linked and so cannot live in this target. It arrives through the
 *     separate `origami-graphs-gemm` bridge (cost_gemm.hpp), which plugs in via
 *     `op_cost_t::tagged`.
 *
 * Everything in this header is HIP-free.
 */
#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "origami/comm/hardware.hpp"
#include "origami/comm/heuristics.hpp"
#include "origami/comm/latency.hpp"
#include "origami/comm/primitives.hpp"
#include "origami/comm/types.hpp"
#include "origami/graphs/cost_model.hpp"
#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

/** @brief Which arm prices an operation. */
enum class cost_kind_t {
  comm,      ///< origami::comm::compute_wg_tile_latency
  gemm,      ///< origami's GEMM model, via the HIP-linked bridge target
  roofline,  ///< compute / HBM / link roofs
  custom,    ///< a caller-supplied function
};

/**
 * @brief Name of a cost arm.
 *
 * @param kind Arm to name.
 * @return const char* Lower-case name, e.g. "roofline".
 */
const char* cost_kind_name(cost_kind_t kind);

/**
 * @brief Parse a cost arm from its name.
 *
 * @param name One of "comm", "gemm", "roofline", "custom".
 * @return cost_kind_t The arm.
 * @throws std::invalid_argument If the name is not one of those.
 */
cost_kind_t cost_kind_from_name(const std::string& name);

// ─── roofline arm ─────────────────────────────────────────────────────

/**
 * @brief Per-GPU roofline parameters.
 *
 * Separate from `origami::comm::hardware_t` on purpose. That one is the
 * calibrated, cycle-native machine description the comm model consumes; this is
 * the handful of headline rates a roofline needs, and it is what the reference
 * implementation's roofline is defined against. Mixing them would change the
 * roofline's numbers for no gain, since anything wanting accuracy should be on
 * the comm arm.
 *
 * Defaults are MI300X-ish: bf16 matrix peak, on-package HBM3, the xGMI egress
 * ceiling, and a ring-step latency.
 *
 * `hbm_bw`, `link_bw` and `link_latency` are stated the way a data sheet
 * states them — bytes/s, bytes/s, seconds — because HBM and xGMI run in
 * their own clock domains: a bandwidth-bound tile really does take fewer
 * engine cycles at a lower engine clock, so `compute_clock_ghz` is what
 * converts those two roofs' answer, which is in seconds, into cycles.
 *
 * The compute roof is different. Matrix throughput is not an independent
 * rate that happens to relate to the clock — it *is* CUs x FLOP/cycle/CU x
 * clock — so `peak_flops_per_cycle` states it clock-free (FLOP/cycle,
 * aggregate across the part) and the compute roof, in cycles, is simply
 * `flops / peak_flops_per_cycle`: the frequency cancels exactly, and a
 * compute-bound tile costs the same cycles no matter what `compute_clock_ghz`
 * is set to. `peak_flops()` recovers the spec-sheet FLOP/s a caller may want
 * to report, at whatever `compute_clock_ghz` currently is; it is a derived
 * accessor rather than a field precisely so the two can never be set to
 * disagree.
 */
struct roofline_hardware_t {
  /**
   * Aggregate matrix throughput, FLOP/cycle across the whole part — e.g.
   * MI300X's 304 CUs x 2048 bf16 FLOP/cycle/CU. Clock-free by construction:
   * multiplying by a clock and later dividing it back out would be a no-op at
   * best, and a caller who changes only one of the two would otherwise get a
   * silently wrong answer (see roofline_cycles's compute roof).
   */
  double peak_flops_per_cycle = 304.0 * 2048.0;
  double hbm_bw               = 5.3e12;  ///< bytes/s to local memory
  double link_bw              = 50e9;    ///< bytes/s off-GPU
  double link_latency         = 2.0e-6;  ///< fixed per-hop latency, seconds

  /**
   * Engine clock, GHz. Scales the HBM and link roofs from seconds into
   * cycles, because those two are continuous rates in their own clock
   * domains. Does **not** touch the compute roof: `peak_flops_per_cycle`
   * already is the per-cycle ceiling, so this field has no bearing on how
   * many cycles a compute-bound tile costs. Must be positive.
   */
  double compute_clock_ghz = 2.1;

  /**
   * @brief Spec-sheet compute peak, FLOP/s, at `compute_clock_ghz`.
   *
   * The data-sheet-friendly form of `peak_flops_per_cycle`, for a caller who
   * wants to report or compare against a FLOP/s figure. Always
   * `peak_flops_per_cycle * compute_clock_ghz * 1e9`; not an independently
   * settable field, so it cannot drift out of sync with the clock-free
   * ceiling the compute roof actually uses.
   *
   * @throws std::invalid_argument If compute_clock_ghz is not positive.
   */
  double peak_flops() const;

  /** @brief Arithmetic intensity, in FLOP/byte, at the compute/HBM ridge. */
  double ridge_intensity() const;
};

/**
 * @brief Per-workgroup work along the three roofline axes.
 *
 * One workgroup's floating-point work, local memory traffic and off-GPU traffic.
 */
struct roofline_spec_t {
  double flops      = 0.0;  ///< floating-point operations
  double hbm_bytes  = 0.0;  ///< bytes to and from local memory
  double link_bytes = 0.0;  ///< bytes off-GPU

  /**
   * @brief Work for one BM x BN output tile of a GEMM contracting over K.
   *
   * Reads the A and B panels and writes the C tile. No cache reuse is modelled,
   * so a deep-K tile looks memory-bound at this granularity.
   *
   * @param bm Tile rows.
   * @param bn Tile columns.
   * @param k Contraction length.
   * @param dtype_bytes Bytes per element.
   * @return roofline_spec_t The work.
   */
  static roofline_spec_t gemm_tile(double bm, double bn, double k, double dtype_bytes = 2.0);

  /**
   * @brief Work for one collective step moving elements off-GPU.
   *
   * Pure link traffic, so it lands on the inter-GPU roof and picks up the fixed
   * per-hop latency.
   *
   * @param elements Elements shipped.
   * @param dtype_bytes Bytes per element.
   * @param fan Peers egressed to.
   * @return roofline_spec_t The work.
   */
  static roofline_spec_t comm_step(double elements, double dtype_bytes = 2.0, double fan = 1.0);
};

/**
 * @brief Roofline duration for one workgroup, in cycles.
 *
 * The largest of the three roofs, each already in cycles: compute via
 * `hardware.peak_flops_per_cycle` directly, HBM and link scaled from seconds
 * by `hardware.compute_clock_ghz`.
 *
 * @param spec Per-workgroup work.
 * @param hardware Machine rates and clock.
 * @return double The largest of the three roofs, in cycles.
 * @throws std::invalid_argument If hardware.compute_clock_ghz is not positive.
 */
double roofline_cycles(const roofline_spec_t& spec, const roofline_hardware_t& hardware);

/**
 * @brief Which roof a workgroup sits on.
 *
 * @param spec Per-workgroup work.
 * @param hardware Machine rates.
 * @return const char* "compute", "hbm", "link", or "none" for no work at all.
 * @throws std::invalid_argument If hardware.compute_clock_ghz is not positive
 *         and spec has any work (an all-zero spec short-circuits to "none"
 *         before the clock is touched).
 */
const char* roofline_bound(const roofline_spec_t& spec, const roofline_hardware_t& hardware);

// ─── comm arm ─────────────────────────────────────────────────────────

/**
 * @brief Everything `origami::comm::compute_wg_tile_latency` needs for one
 *        workgroup of a communication operation.
 *
 * Bundled so a graph can be priced without the caller reassembling the comm
 * model's arguments per node.
 */
struct comm_spec_t {
  std::vector<comm::op_t> work_graph;  ///< primitives for one workgroup iteration
  std::size_t wg_tile_bytes = 0;       ///< bytes one workgroup moves per timestep
  int num_wgs               = 1;       ///< workgroups sharing the link, i.e. channels
  double bw_per_wg          = 0.0;     ///< this workgroup's share of link bandwidth, bytes/cycle

  /** Present for a strided tile, whose rows are walked separately. */
  std::optional<comm::tile_shape_t> tile;

  /** Collective context, which selects per-primitive heuristics. */
  std::optional<comm::primitive_t> primitive;

  /**
   * @brief Tile geometry as the latency model wants it.
   *
   * @param cacheline_bytes Hardware cache-line size.
   * @return comm::wg_tile_geometry_t Cache lines, elements and the optional shape.
   */
  comm::wg_tile_geometry_t geometry(std::size_t cacheline_bytes) const;
};

/**
 * @brief Full latency breakdown for one workgroup tile.
 *
 * @param spec Communication workload.
 * @param system Machine description.
 * @param active_cus Concurrently active compute units; the workgroup count when unset.
 * @param heur Tunable heuristics.
 * @return comm::wg_tile_latency_breakdown_t Per-stage and per-unit cycles.
 */
comm::wg_tile_latency_breakdown_t comm_breakdown(
    const comm_spec_t& spec,
    const comm::system_t& system,
    std::optional<int> active_cus  = std::nullopt,
    const comm::heuristics_t& heur = comm::DEFAULT_HEURISTICS);

/**
 * @brief Duration of one workgroup tile, in cycles.
 *
 * `origami::comm::compute_wg_tile_latency` already answers in cycles, so this
 * is `comm_breakdown(...).T_total_cycles` and nothing more: no clock enters
 * the comm arm at all, unlike the roofline arm.
 *
 * @param spec Communication workload.
 * @param system Machine description.
 * @param active_cus Concurrently active compute units; the workgroup count when unset.
 * @param heur Tunable heuristics.
 * @return double Cycles.
 */
double comm_cycles(const comm_spec_t& spec,
                   const comm::system_t& system,
                   std::optional<int> active_cus  = std::nullopt,
                   const comm::heuristics_t& heur = comm::DEFAULT_HEURISTICS);

// ─── custom arm ───────────────────────────────────────────────────────

/**
 * @brief A caller-supplied duration, in cycles.
 *
 * Receives the contention level so a measured model can use it; a function that
 * ignores the second argument is a constant-cost operation.
 */
using custom_cost_fn_t = std::function<double(const wg_node_t& node, int active_cus)>;

// ─── op_cost_t ────────────────────────────────────────────────────────

/**
 * @brief How one operation is priced: an arm plus that arm's inputs.
 */
class op_cost_t {
 public:
  /** @brief Price by the roofline arm. */
  static op_cost_t from_roofline(roofline_spec_t spec);

  /** @brief Price by the comm arm. */
  static op_cost_t from_comm(comm_spec_t spec);

  /** @brief Price by a caller-supplied function. */
  static op_cost_t from_custom(custom_cost_fn_t fn);

  /**
   * @brief Price by a function, but report a different arm.
   *
   * The seam the GEMM bridge plugs into. That arm has to live in a HIP-linked
   * target, so it cannot be a case in this file's switch; instead it supplies a
   * function and tags it `gemm`, which is what makes `kind_of` report the arm a
   * caller chose rather than a generic "custom". Being unable to link the bridge
   * is then a link error at the call site rather than a runtime throw from
   * somewhere deep in a scheduler.
   *
   * @param kind Arm to report.
   * @param fn Duration in cycles, given a node and a contention level.
   * @return op_cost_t The entry.
   * @throws std::invalid_argument If the function is empty.
   */
  static op_cost_t tagged(cost_kind_t kind, custom_cost_fn_t fn);

  /** @brief Which arm this uses. */
  cost_kind_t kind() const noexcept { return kind_; }

  /**
   * @brief The roofline work, for inspection.
   * @throws std::invalid_argument If this is not a roofline entry.
   */
  const roofline_spec_t& roofline() const;

  /**
   * @brief The communication workload, for inspection.
   * @throws std::invalid_argument If this is not a comm entry.
   */
  const comm_spec_t& comm() const;

 private:
  friend class cost_table_t;

  cost_kind_t kind_ = cost_kind_t::roofline;
  std::variant<roofline_spec_t, comm_spec_t, custom_cost_fn_t> body_;
};

// ─── cost_table_t ─────────────────────────────────────────────────────

/**
 * @brief Machine description and defaults shared by every entry in a table.
 *
 * @note When a table uses both the comm and roofline arms, `comm_system`'s
 * clock (`comm_system->gpu.clock_ghz`) must match
 * `roofline_hardware.compute_clock_ghz`. Both arms answer in cycles, and a
 * table's scheduler adds and maxes them across arms unconditionally, so cycles
 * measured against two different clocks would silently mix into one number
 * that is correct at neither. A table that uses only one arm does not need the
 * unused arm's clock to agree.
 */
struct cost_settings_t {
  roofline_hardware_t roofline_hardware;  ///< rates for the roofline arm

  /**
   * Machine description for the comm arm. Left unset on purpose: the comm
   * library stopped shipping a hardcoded MI300X, because the same part exposes
   * different CU and XCD counts under partitioning, so the machine has to come
   * from the caller. Build one with `origami::comm::make_system`, or from the
   * device with `origami::comm::system_from_device`. If the same table also
   * uses roofline-priced entries, build it at
   * `roofline_hardware.compute_clock_ghz` or set that field to the comm
   * system's clock.
   */
  std::optional<comm::system_t> comm_system;

  comm::heuristics_t heuristics = comm::DEFAULT_HEURISTICS;  ///< tunables for the comm arm

  /** Duration, in cycles, for an operation with no entry. Zero, as the reference does. */
  double default_cycles = 0.0;

  /** Hop latency, in cycles; free when unset, which is what both reference models say. */
  std::function<double(const edge_t&)> edge_cycles;
};

/**
 * @brief A cost model assembled from one entry per operation.
 *
 * Entries are keyed by operation name for the caller's convenience and resolved
 * to indices once, so pricing a node in a scheduler's inner loop is an array
 * lookup rather than a hash of a string.
 *
 * Contention reaches the arms that can use it. The comm arm forwards
 * `active_cus` to the latency model, which is what makes
 * `runtime_kind_t::event_driven` more than a relabelling of
 * `runtime_kind_t::roofline`; the roofline arm ignores it, being a static model.
 */
class cost_table_t : public cost_model_t {
 public:
  /**
   * @brief Build an empty table over a graph's operations.
   *
   * @param graph Graph whose operations will be priced; must outlive the table.
   * @param settings Machine description and defaults.
   */
  explicit cost_table_t(const wg_graph_t& graph, cost_settings_t settings = {});

  /**
   * @brief Price an operation.
   *
   * @param op_name Operation to price.
   * @param cost How to price it.
   * @return cost_table_t& This table, for chaining.
   * @throws std::out_of_range If the graph has no such operation.
   * @throws std::invalid_argument If this entry would make the table use both
   *         the comm and roofline arms while their clocks disagree (see
   *         `cost_settings_t`).
   */
  cost_table_t& set(const std::string& op_name, op_cost_t cost);

  /**
   * @brief Whether an operation has an entry.
   *
   * @param op Operation index.
   * @return bool True when priced explicitly rather than by the default.
   */
  bool has(int op) const;

  /**
   * @brief The arm pricing an operation.
   *
   * @param op Operation index.
   * @return std::optional<cost_kind_t> The arm, or nothing when unpriced.
   */
  std::optional<cost_kind_t> kind_of(int op) const;

  /** @brief Duration in cycles, at each arm's own default contention level. */
  double node_cycles(const wg_node_t& node) const override;

  /** @brief Duration in cycles under a stated contention level. */
  double node_cycles_at(const wg_node_t& node, int active_cus) const override;

  /** @brief Hop latency in cycles; free unless `cost_settings_t::edge_cycles` is set. */
  double edge_cycles(const edge_t& edge) const override;

 private:
  double price(int op, const wg_node_t& node, std::optional<int> active_cus) const;

  const wg_graph_t* graph_;
  cost_settings_t settings_;
  std::vector<std::optional<op_cost_t>> by_op_;
};

}  // namespace origami::graphs
