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

#include <stdexcept>
#include <string>
#include <vector>

#include "origami/graphs/core.hpp"
#include "origami/graphs/free_graph.hpp"
#include "test_harness.hpp"

using namespace origami::graphs;

// origami::graphs refers to the comm model as comm::, but that is only visible
// from inside the namespace; at file scope it has to be spelled out.
namespace comm = ::origami::comm;

namespace {

/**
 * The nominal full-die MI300X, built explicitly.
 *
 * The comm library stopped shipping a hardcoded machine because the same part
 * exposes different CU and XCD counts under partitioning, so a test that wants
 * stable numbers has to pin one. This mirrors the fixture the comm suite uses.
 */
comm::system_t nominal_mi300x() {
  constexpr comm::gpu_topology_t topology{
      ::origami::architecture_t::gfx942, 304, 8, 38, 4ULL * 1024ULL * 1024ULL};
  return comm::make_system(
      comm::get_arch_ceilings(::origami::architecture_t::gfx942), topology, 2.0);
}

/** An all-gather step: read locally, write locally, push to the next rank. */
comm_spec_t ag_step(int num_wgs, std::size_t bytes) {
  comm_spec_t spec;
  spec.work_graph    = {comm::load_t{}, comm::store_t{}, comm::push_t{/*peer=*/1}};
  spec.wg_tile_bytes = bytes;
  spec.num_wgs       = num_wgs;
  spec.bw_per_wg     = 8.0;
  spec.primitive     = comm::primitive_t::all_gather;
  return spec;
}

/** gemm(8 wgs) -> all_reduce(4 wgs): the motivating overlap case. */
graph_t gemm_then_reduce() {
  const allocation_t c("c", {index_t{64}});
  const allocation_t out("out", {index_t{64}});
  operation_t gemm("gemm", 8);
  gemm.access_patterns = {write(c, contiguous(8))};
  operation_t ar("all_reduce", 4);
  ar.access_patterns = {read(c, contiguous(16)), write(out, contiguous(16))};
  return graph_t({gemm, ar});
}

}  // namespace

// ─── cost_kind_t ──────────────────────────────────────────────────────

TEST(cost_kind_names_round_trip) {
  for (cost_kind_t k :
       {cost_kind_t::comm, cost_kind_t::gemm, cost_kind_t::roofline, cost_kind_t::custom}) {
    CHECK(cost_kind_from_name(cost_kind_name(k)) == k);
  }

  bool threw = false;
  try {
    cost_kind_from_name("nonsense");
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

// ─── roofline arm ─────────────────────────────────────────────────────

TEST(roofline_takes_the_slowest_of_the_three_roofs) {
  const roofline_hardware_t hw;

  // 2 x 128 x 128 x 4096 FLOP at 1.307e15 FLOP/s.
  const roofline_spec_t tile = roofline_spec_t::gemm_tile(128, 128, 4096);
  CHECK(tile.flops == 2.0 * 128 * 128 * 4096);
  CHECK(tile.hbm_bytes == (128.0 * 4096 + 4096.0 * 128 + 128.0 * 128) * 2);
  CHECK_NEAR(roofline_seconds(tile, hw), tile.hbm_bytes / hw.hbm_bw, 1e-18);
  // Deep K with no reuse modelled: the tile lands on the memory roof, which is
  // the known coarseness of a pure roofline rather than a surprise.
  CHECK(std::string(roofline_bound(tile, hw)) == "hbm");

  const roofline_spec_t step = roofline_spec_t::comm_step(1 << 20, 2.0, 2.0);
  CHECK(std::string(roofline_bound(step, hw)) == "link");
  CHECK_NEAR(roofline_seconds(step, hw), step.link_bytes / hw.link_bw + hw.link_latency, 1e-18);
}

TEST(a_shallow_tile_is_compute_bound) {
  const roofline_hardware_t hw;
  // Square and shallow: enough arithmetic per byte to clear the ridge.
  const roofline_spec_t tile = roofline_spec_t::gemm_tile(2048, 2048, 2048);
  CHECK(std::string(roofline_bound(tile, hw)) == "compute");
  CHECK(tile.flops / tile.hbm_bytes > hw.ridge_intensity());
}

TEST(an_empty_spec_costs_nothing_and_sits_on_no_roof) {
  const roofline_hardware_t hw;
  const roofline_spec_t empty;
  CHECK(roofline_seconds(empty, hw) == 0.0);
  CHECK(std::string(roofline_bound(empty, hw)) == "none");

  // A compute-only operation must not be charged the link's fixed latency, or
  // every GEMM tile would look as though it had touched the fabric.
  roofline_spec_t compute_only;
  compute_only.flops = 1.0e9;
  CHECK(roofline_seconds(compute_only, hw) < hw.link_latency);
}

// ─── comm arm ─────────────────────────────────────────────────────────

TEST(comm_cost_comes_from_the_calibrated_latency_model) {
  const comm::system_t machine = nominal_mi300x();
  const comm_spec_t spec       = ag_step(8, 1 << 16);

  const double seconds = comm_seconds(spec, machine);
  CHECK(seconds > 0.0);

  // The seconds must be exactly the model's cycles at the machine's clock, not
  // a re-derivation: this arm's whole value is that it defers to origami::comm.
  const comm::wg_tile_latency_breakdown_t breakdown = comm_breakdown(spec, machine);
  CHECK(seconds == machine.gpu.cycles_to_seconds(breakdown.T_total_cycles));
  CHECK(breakdown.T_total_cycles > 0.0);
}

TEST(comm_cost_rises_only_once_the_machine_is_genuinely_crowded) {
  const comm::system_t machine = nominal_mi300x();
  const comm_spec_t spec       = ag_step(8, 1 << 16);

  // A handful of channels do not contend: the model's memory contention scales
  // per XCD, and this part has 38 CUs on each of 8, so a few active CUs share
  // caches and HBM without measurably slowing each other.
  CHECK(comm_seconds(spec, machine, 8) == comm_seconds(spec, machine, 1));
  CHECK(comm_seconds(spec, machine, 64) == comm_seconds(spec, machine, 1));

  // A full die does.
  CHECK(comm_seconds(spec, machine, 304) > comm_seconds(spec, machine, 1));

  // Unset means the channel count, which is what the reference adapter defaults
  // to on the grounds that a collective's own workgroups contend with each other.
  CHECK(comm_seconds(spec, machine) == comm_seconds(spec, machine, spec.num_wgs));
}

TEST(comm_cost_rises_with_tile_size) {
  const comm::system_t machine = nominal_mi300x();
  CHECK(comm_seconds(ag_step(8, 1 << 18), machine) > comm_seconds(ag_step(8, 1 << 14), machine));
}

TEST(a_strided_tile_costs_more_than_a_contiguous_one) {
  const comm::system_t machine = nominal_mi300x();

  comm_spec_t flat = ag_step(8, 64 * 64 * 2);
  flat.tile        = comm::tile_shape_t{64, 64, comm::data_type_t::BFloat16, 0, true};

  comm_spec_t strided = flat;
  strided.tile        = comm::tile_shape_t{64, 64, comm::data_type_t::BFloat16, 0, false};

  // Each row of a strided tile is walked separately, so its partial final cache
  // line cannot merge with the next row.
  CHECK(comm_seconds(strided, machine) > comm_seconds(flat, machine));
}

TEST(a_non_positive_channel_count_is_rejected) {
  const comm::system_t machine = nominal_mi300x();
  comm_spec_t spec             = ag_step(8, 1 << 16);
  spec.num_wgs                 = 0;

  bool threw = false;
  try {
    comm_seconds(spec, machine);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

// ─── cost_table_t ─────────────────────────────────────────────────────

TEST(a_table_prices_each_operation_by_its_own_arm) {
  const graph_t g = gemm_then_reduce();

  cost_settings_t settings;
  settings.comm_system = nominal_mi300x();

  cost_table_t table(g, settings);
  table.set("gemm", op_cost_t::from_roofline(roofline_spec_t::gemm_tile(128, 128, 4096)))
      .set("all_reduce", op_cost_t::from_comm(ag_step(4, 1 << 16)));

  CHECK(table.kind_of(0) == cost_kind_t::roofline);
  CHECK(table.kind_of(1) == cost_kind_t::comm);

  const double gemm_cost = table.node_cost(wg_node_t{0, 0, 0});
  const double comm_cost = table.node_cost(wg_node_t{1, 0, 0});
  CHECK(gemm_cost > 0.0);
  CHECK(comm_cost > 0.0);
  CHECK(gemm_cost != comm_cost);

  // Each arm's own function is the authority; the table only dispatches.
  CHECK(gemm_cost ==
        roofline_seconds(roofline_spec_t::gemm_tile(128, 128, 4096), roofline_hardware_t{}));
  CHECK(comm_cost == comm_seconds(ag_step(4, 1 << 16), *settings.comm_system));
}

TEST(contention_reaches_the_comm_arm_but_not_the_roofline_arm) {
  const graph_t g = gemm_then_reduce();
  cost_settings_t settings;
  settings.comm_system = nominal_mi300x();

  cost_table_t table(g, settings);
  table.set("gemm", op_cost_t::from_roofline(roofline_spec_t::gemm_tile(128, 128, 4096)))
      .set("all_reduce", op_cost_t::from_comm(ag_step(4, 1 << 16)));

  const wg_node_t tile{0, 0, 0};
  const wg_node_t step{1, 0, 0};

  // A roofline has no view on who else is running, so it is flat by construction.
  CHECK(table.node_cost_at(tile, 1) == table.node_cost_at(tile, 304));
  // The comm model does, which is what makes event_driven_runtime_t worth having
  // over roofline_runtime_t: the contention level actually reaches the model.
  CHECK(table.node_cost_at(step, 304) > table.node_cost_at(step, 1));
}

TEST(unpriced_operations_fall_back_to_the_default) {
  const graph_t g = gemm_then_reduce();

  cost_table_t zeroed(g);
  CHECK(zeroed.has(0) == false);
  CHECK(zeroed.kind_of(0).has_value() == false);
  CHECK(zeroed.node_cost(wg_node_t{0, 0, 0}) == 0.0);

  cost_settings_t settings;
  settings.default_seconds = 1e-6;
  cost_table_t defaulted(g, settings);
  CHECK(defaulted.node_cost(wg_node_t{1, 0, 0}) == 1e-6);
}

TEST(a_custom_arm_takes_a_function) {
  const graph_t g = gemm_then_reduce();
  cost_table_t table(g);
  table.set("gemm", op_cost_t::from_custom([](const wg_node_t& n, int active) {
              return 1e-6 * (n.wg + 1) * active;
            }));

  CHECK(table.kind_of(0) == cost_kind_t::custom);
  CHECK_NEAR(table.node_cost(wg_node_t{0, 2, 0}), 3e-6, 1e-18);  // active defaults to 1
  CHECK_NEAR(table.node_cost_at(wg_node_t{0, 2, 0}, 4), 12e-6, 1e-18);

  bool threw = false;
  try {
    op_cost_t::from_custom(nullptr);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(edge_cost_is_free_unless_asked_for) {
  const graph_t g = gemm_then_reduce();
  const edge_t& e = g.edges().front();

  CHECK(cost_table_t(g).edge_cost(e) == 0.0);

  cost_settings_t settings;
  settings.edge_cost = [](const edge_t& edge) { return 1e-9 * static_cast<double>(edge.weight); };
  CHECK(cost_table_t(g, settings).edge_cost(e) == 1e-9 * static_cast<double>(e.weight));
}

TEST(a_misspelled_operation_name_is_rejected) {
  const graph_t g = gemm_then_reduce();
  cost_table_t table(g);

  // Silently accepting this would leave the operation priced at the default,
  // which is a plausible number and therefore a bad way to find out.
  bool threw = false;
  try {
    table.set("gemmm", op_cost_t::from_roofline(roofline_spec_t{}));
  } catch (const std::out_of_range&) { threw = true; }
  CHECK(threw);
}

TEST(the_comm_arm_reports_a_missing_machine_description) {
  const graph_t g = gemm_then_reduce();
  cost_table_t table(g);  // no comm_system
  table.set("all_reduce", op_cost_t::from_comm(ag_step(4, 1 << 16)));

  bool threw = false;
  try {
    table.node_cost(wg_node_t{1, 0, 0});
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(inspecting_an_entry_with_the_wrong_arm_is_rejected) {
  const op_cost_t roof = op_cost_t::from_roofline(roofline_spec_t::gemm_tile(64, 64, 64));
  CHECK(roof.roofline().flops == 2.0 * 64 * 64 * 64);

  bool threw = false;
  try {
    roof.comm();
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

// ─── driving a runtime with a real cost model ─────────────────────────

TEST(a_table_drives_the_continuous_time_runtimes) {
  const graph_t g = gemm_then_reduce();
  cost_settings_t settings;
  settings.comm_system = nominal_mi300x();

  cost_table_t table(g, settings);
  table.set("gemm", op_cost_t::from_roofline(roofline_spec_t::gemm_tile(128, 128, 4096)))
      .set("all_reduce", op_cost_t::from_comm(ag_step(4, 1 << 20)));

  cost_runtime_options_t opts;
  opts.lanes = 8;

  const cost_schedule_t s = roofline_runtime_t(table, opts).schedule(g);

  CHECK(s.order.size() == 12);
  CHECK(s.makespan() > 0.0);
  // The collective is far slower than a GEMM tile, so it dominates: exactly the
  // asymmetry a Gantt chart needs, and the reason a uniform timestep misleads.
  const std::vector<double> busy = s.busy_by_op();
  CHECK(busy[1] > busy[0]);

  // The two policies agree here, and the reason is worth pinning down rather
  // than assuming: re-pricing at dispatch only moves the answer if the model
  // charges for contention at the occupancy the schedule actually reaches. Eight
  // lanes is far below the point where the calibrated model starts charging, and
  // this collective is link-bound in any case, so the re-pricing is a no-op. A
  // divergence here would mean the machine got crowded enough to matter.
  const cost_schedule_t e = event_driven_runtime_t(table, opts).schedule(g);
  CHECK(e.makespan() == s.makespan());
}

ORIGAMI_TEST_MAIN()
