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

#include "origami/graphs/runtime.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "origami/graphs/analysis.hpp"
#include "origami/graphs/core.hpp"
#include "origami/graphs/free_graph.hpp"
#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/** Render a schedule the way oracle/runtime_dump.cpp does. */
std::string render(const wg_graph_t& g, const schedule_t& s) {
  std::string out;
  bool first = true;
  for (const timestep_t& ts : s.timesteps()) {
    if (!first) out += " | ";
    first = false;
    out += std::to_string(ts.time) + ":";
    for (const wg_node_t& n : ts.nodes) out += " " + g.label(n);
  }
  return out;
}

/** a(4 wgs) -> b(4 wgs), one edge per workgroup. */
graph_t one_to_one() {
  const allocation_t x("x", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4))};
  return graph_t({a, b});
}

/** Every dependency is respected: no node starts before a predecessor finishes. */
bool respects_dependencies(const wg_graph_t& g, const schedule_t& s) {
  for (const edge_t& e : g.edges()) {
    if (s.start.at(e.dst) < s.finish(e.src)) return false;
  }
  return true;
}

/** No timestep dispatches more than the lane limit. */
bool respects_lanes(const schedule_t& s) {
  if (!s.lanes) return true;
  for (const timestep_t& ts : s.timesteps()) {
    if (ts.nodes.size() > static_cast<std::size_t>(*s.lanes)) return false;
  }
  return true;
}

}  // namespace

// ─── the properties every runtime must have ───────────────────────────

TEST(every_runtime_respects_dependencies_and_lanes) {
  const graph_t g = one_to_one();

  for (int lanes = 1; lanes <= 5; ++lanes) {
    const breadth_first_runtime_t bf(lanes);
    const asap_runtime_t asap(lanes);
    const depth_first_runtime_t df(lanes);

    for (const runtime_t* rt : {static_cast<const runtime_t*>(&bf),
                                static_cast<const runtime_t*>(&asap),
                                static_cast<const runtime_t*>(&df)}) {
      const schedule_t s = rt->schedule(g);
      CHECK(s.start.size() == g.num_nodes());
      CHECK(s.order.size() == g.num_nodes());
      CHECK(respects_dependencies(g, s));
      CHECK(respects_lanes(s));
    }
  }
}

TEST(no_runtime_beats_the_critical_path) {
  // The critical path assumes unbounded hardware, so it is a lower bound on
  // anything a runtime can achieve, and unlimited lanes should reach it.
  const graph_t g            = one_to_one();
  const double bound         = critical_path(g).makespan;
  const schedule_t unlimited = asap_runtime_t().schedule(g);
  const schedule_t narrow    = asap_runtime_t(2).schedule(g);

  CHECK(static_cast<double>(unlimited.makespan()) == bound);
  CHECK(static_cast<double>(narrow.makespan()) >= bound);
}

TEST(breadth_first_is_the_no_overlap_baseline) {
  // Its makespan is the sum of each operation's waves, by construction, and no
  // pipelined policy can do worse.
  const graph_t g = one_to_one();

  const schedule_t bf   = breadth_first_runtime_t(3).schedule(g);
  const schedule_t asap = asap_runtime_t(3).schedule(g);

  CHECK(bf.makespan() == 4);  // two ragged waves per operation
  CHECK(asap.makespan() == 3);
  CHECK(asap.makespan() <= bf.makespan());
}

TEST(unlimited_lanes_make_every_policy_agree) {
  const graph_t g            = one_to_one();
  const std::string expected = "0: a#0 a#1 a#2 a#3 | 1: b#0 b#1 b#2 b#3";

  CHECK(render(g, breadth_first_runtime_t().schedule(g)) == expected);
  CHECK(render(g, asap_runtime_t().schedule(g)) == expected);
  CHECK(render(g, depth_first_runtime_t().schedule(g)) == expected);
}

// ─── what distinguishes the three policies ────────────────────────────

TEST(breadth_first_never_overlaps_two_operations) {
  const graph_t g    = one_to_one();
  const schedule_t s = breadth_first_runtime_t(2).schedule(g);

  CHECK(render(g, s) == "0: a#0 a#1 | 1: a#2 a#3 | 2: b#0 b#1 | 3: b#2 b#3");

  // No timestep mixes operations.
  for (const timestep_t& ts : s.timesteps()) {
    for (const wg_node_t& n : ts.nodes) CHECK(n.op == ts.nodes.front().op);
  }
}

TEST(asap_slips_consumers_into_a_ragged_producer_wave) {
  // With three lanes the producer's second wave uses one lane, leaving two for
  // consumers that are already unblocked. That gap is the whole overlap story.
  const graph_t g    = one_to_one();
  const schedule_t s = asap_runtime_t(3).schedule(g);

  CHECK(render(g, s) == "0: a#0 a#1 a#2 | 1: a#3 b#0 b#1 | 2: b#2 b#3");
  CHECK(s.makespan() == 3);
}

TEST(depth_first_pushes_one_tile_through_the_pipeline_first) {
  // A consumer runs at timestep 1 rather than after every producer, which is
  // what a fused kernel streaming one tile end to end does.
  const graph_t g    = one_to_one();
  const schedule_t s = depth_first_runtime_t(1).schedule(g);

  CHECK(render(g, s) == "0: a#0 | 1: b#0 | 2: a#1 | 3: b#1 | 4: a#2 | 5: b#2 | 6: a#3 | 7: b#3");

  // It reaches the last operation far sooner than ASAP, at equal makespan.
  const schedule_t asap = asap_runtime_t(1).schedule(g);
  CHECK(s.start.at(wg_node_t{1, 0, 0}) == 1);
  CHECK(asap.start.at(wg_node_t{1, 0, 0}) == 4);
  CHECK(s.makespan() == asap.makespan());
}

// ─── depth-first ordering ─────────────────────────────────────────────

TEST(depth_first_rank_follows_the_chain_not_the_siblings) {
  const graph_t g                       = one_to_one();
  const wg_node_map_t<std::size_t> rank = depth_first_rank(g);

  CHECK(rank.size() == 8);
  // a#0 then its consumer b#0, before sibling a#1.
  CHECK(rank.at(wg_node_t{0, 0, 0}) == 0);
  CHECK(rank.at(wg_node_t{1, 0, 0}) == 1);
  CHECK(rank.at(wg_node_t{0, 1, 0}) == 2);
  CHECK(rank.at(wg_node_t{1, 1, 0}) == 3);
}

TEST(depth_first_orders_successors_by_operator_order_not_by_name) {
  // 'a' feeds 'z' (declared second) and 'b' (declared third). The reference
  // sorts these successors by name and would dive into b first; ordering by
  // operator position dives into z. See the note on depth_first_rank.
  const allocation_t x("x", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t z("z", 4);
  z.access_patterns = {read(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {read(x, contiguous(4))};
  const graph_t g({a, z, b});

  const schedule_t s = depth_first_runtime_t(1).schedule(g);
  CHECK(render(g, s).substr(0, 24) == "0: a#0 | 1: z#0 | 2: b#0");

  // Only the order moves: the reference reports the same makespan here.
  CHECK(s.makespan() == 12);
}

TEST(depth_first_starts_at_the_first_declared_source) {
  // Two independent producers, 'z' declared before 'b'. Ordering sources by
  // name would start with b purely because of its spelling.
  const allocation_t x("x", {index_t{16}});
  const allocation_t y("y", {index_t{16}});
  operation_t z("z", 4);
  z.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 4);
  b.access_patterns = {write(y, contiguous(4))};
  operation_t c("c", 4);
  c.access_patterns = {read(x, contiguous(4)), read(y, contiguous(4))};
  const graph_t g({z, b, c});

  const schedule_t s = depth_first_runtime_t(2).schedule(g);
  CHECK(render(g, s) ==
        "0: z#0 z#1 | 1: z#2 z#3 | 2: b#0 b#1 | 3: c#0 c#1 | 4: b#2 b#3 | 5: c#2 c#3");
}

TEST(unreachable_nodes_still_get_scheduled) {
  // A free graph can leave a node with no path from any source. Derivation
  // cannot, but the rank has to cope or the node would never be dispatched.
  free_graph_t g;
  g.add_op("a", 2);
  g.add_op("b", 2);
  g.add_edge(wg_node_t{0, 0, 0}, wg_node_t{1, 0, 0}, "x", 1);

  const wg_node_map_t<std::size_t> rank = depth_first_rank(g);
  CHECK(rank.size() == 4);

  const schedule_t s = depth_first_runtime_t(1).schedule(g);
  CHECK(s.start.size() == 4);
  CHECK(respects_dependencies(g, s));
}

// ─── iterated operations and both backends ────────────────────────────

TEST(iterations_are_scheduled_as_separate_atoms) {
  const allocation_t buf("buf", {index_t{32}});
  operation_t p("produce", 1);
  p.num_iters       = 4;
  p.access_patterns = {write(buf, streamed(32, 4))};
  operation_t c("consume", 1);
  c.num_iters       = 4;
  c.access_patterns = {read(buf, streamed(32, 4))};
  const graph_t g({p, c});

  // One workgroup each, but four atoms each, so a single lane takes 8 steps.
  const schedule_t s = depth_first_runtime_t(1).schedule(g);
  CHECK(s.start.size() == 8);
  CHECK(s.makespan() == 8);
  CHECK(render(g, s) == "0: produce#0 | 1: consume#0 | 2: produce#0.1 | 3: consume#0.1 | "
                        "4: produce#0.2 | 5: consume#0.2 | 6: produce#0.3 | 7: consume#0.3");
}

TEST(a_snapshot_schedules_identically_to_the_graph_it_came_from) {
  const graph_t g         = one_to_one();
  const free_graph_t snap = g.to_free();

  for (int lanes = 1; lanes <= 4; ++lanes) {
    const asap_runtime_t asap(lanes);
    const depth_first_runtime_t df(lanes);
    CHECK(render(g, asap.schedule(g)) == render(snap, asap.schedule(snap)));
    CHECK(render(g, df.schedule(g)) == render(snap, df.schedule(snap)));
  }
}

// ─── wg_duration, empty graphs and validation ─────────────────────────

TEST(wg_duration_stretches_every_timestep) {
  const graph_t g    = one_to_one();
  const schedule_t s = asap_runtime_t(2, 10).schedule(g);

  CHECK(s.makespan() == 40);
  CHECK(s.start.at(wg_node_t{0, 0, 0}) == 0);
  CHECK(s.start.at(wg_node_t{1, 0, 0}) == 20);
  CHECK(s.finish(wg_node_t{1, 0, 0}) == 30);
}

TEST(an_empty_graph_schedules_to_nothing) {
  const free_graph_t g;
  for (const schedule_t& s : {breadth_first_runtime_t().schedule(g),
                              asap_runtime_t().schedule(g),
                              depth_first_runtime_t().schedule(g)}) {
    CHECK(s.start.empty());
    CHECK(s.makespan() == 0);
    CHECK(s.timesteps().empty());
  }
}

TEST(non_positive_lanes_and_durations_are_rejected) {
  bool threw = false;
  try {
    asap_runtime_t(0);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);

  threw = false;
  try {
    breadth_first_runtime_t(unlimited_lanes, 0);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);

  threw = false;
  try {
    depth_first_runtime_t(-1);
  } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
}

TEST(schedule_summary_reports_the_shape_of_the_run) {
  const graph_t g = one_to_one();

  const std::string wide = asap_runtime_t().schedule(g).summary();
  CHECK(wide.find("runtime 'asap (pipelined)'") != std::string::npos);
  CHECK(wide.find("8 workgroups across 2 timesteps") != std::string::npos);
  CHECK(wide.find("makespan=2") != std::string::npos);
  CHECK(wide.find("lanes=\u221e") != std::string::npos);

  const std::string narrow = breadth_first_runtime_t(2).schedule(g).summary();
  CHECK(narrow.find("lanes=2, wg_duration=1") != std::string::npos);
}

ORIGAMI_TEST_MAIN()
