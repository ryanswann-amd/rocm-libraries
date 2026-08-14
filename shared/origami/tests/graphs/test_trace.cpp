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

#include "origami/graphs/trace.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "origami/graphs/core.hpp"
#include "origami/graphs/cost.hpp"
#include "origami/graphs/free_graph.hpp"
#include "origami/graphs/runtime.hpp"
#include "test_harness.hpp"

using namespace origami::graphs;

namespace {

/** Occurrences of a substring; the cheap way to count events without a parser. */
std::size_t count(const std::string& haystack, const std::string& needle) {
  std::size_t n = 0;
  for (std::size_t at = haystack.find(needle); at != std::string::npos;
       at             = haystack.find(needle, at + needle.size())) {
    ++n;
  }
  return n;
}

/**
 * Enough of a JSON check to catch the mistakes this emitter can actually make:
 * unbalanced braces or brackets, and a trailing comma before a close. A real
 * parser would be better, but the graphs suite has no dependencies and a
 * hand-rolled one would be more code than the emitter.
 */
bool well_formed(const std::string& text) {
  int braces = 0, brackets = 0;
  bool in_string = false, escaped = false;
  char previous_significant = '\0';

  for (char c : text) {
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    switch (c) {
      case '"': in_string = true; break;
      case '{': ++braces; break;
      case '}':
        if (previous_significant == ',') return false;
        --braces;
        break;
      case '[': ++brackets; break;
      case ']':
        if (previous_significant == ',') return false;
        --brackets;
        break;
      default: break;
    }
    if (braces < 0 || brackets < 0) return false;
    if (c != ' ' && c != '\n' && c != '\t' && c != '\r') previous_significant = c;
  }
  return braces == 0 && brackets == 0 && !in_string;
}

/** a(4 wgs) -> b(2 wgs) through allocation x. */
graph_t fan_in() {
  const allocation_t x("x", {index_t{16}});
  operation_t a("a", 4);
  a.access_patterns = {write(x, contiguous(4))};
  operation_t b("b", 2);
  b.access_patterns = {read(x, contiguous(8))};
  return graph_t({a, b});
}

cost_table_t flat_table(const wg_graph_t& g) {
  cost_table_t table(g);
  for (int op = 0; op < g.num_operations(); ++op) {
    table.set(g.op_name(op), op_cost_t::from_custom([](const wg_node_t&, int) { return 2.0; }));
  }
  return table;
}

}  // namespace

TEST(a_priced_schedule_renders_one_slice_per_atom) {
  const graph_t g          = fan_in();
  const cost_table_t table = flat_table(g);
  simulate_options_t opts;
  opts.lanes = 3;

  const schedule_t s     = simulate(g, asap_runtime_t{}, table, opts);
  const std::string json = chrome_trace(g, s);

  CHECK(well_formed(json));
  CHECK(count(json, "\"ph\":\"X\"") == 6);    // one per node
  CHECK(count(json, "\"name\":\"a\"") == 4);  // named by operation
  CHECK(count(json, "\"name\":\"b\"") == 2);
  CHECK(json.find("\"traceEvents\"") != std::string::npos);
  CHECK(json.find("\"displayTimeUnit\"") != std::string::npos);
}

TEST(lanes_become_named_and_ordered_tracks) {
  const graph_t g          = fan_in();
  const cost_table_t table = flat_table(g);
  simulate_options_t opts;
  opts.lanes = 3;

  const std::string json = chrome_trace(g, simulate(g, asap_runtime_t{}, table, opts));

  CHECK(count(json, "\"name\":\"thread_name\"") == 3);
  // Without a sort index a viewer orders tracks by first appearance, which is
  // dispatch order rather than lane order.
  CHECK(count(json, "\"name\":\"thread_sort_index\"") == 3);
  CHECK(json.find("\"name\":\"lane 0\"") != std::string::npos);
  CHECK(json.find("\"name\":\"lane 2\"") != std::string::npos);
}

TEST(dependencies_become_flow_events_that_can_be_turned_off) {
  const graph_t g          = fan_in();
  const cost_table_t table = flat_table(g);
  const schedule_t s       = simulate(g, asap_runtime_t{}, table, simulate_options_t{});

  const std::string with = chrome_trace(g, s);
  CHECK(count(with, "\"ph\":\"s\"") == g.edges().size());
  CHECK(count(with, "\"ph\":\"f\"") == g.edges().size());
  // Named for the buffer the dependency flows through, so a viewer distinguishes
  // a handoff through one allocation from a handoff through another.
  CHECK(with.find("\"cat\":\"dependency\"") != std::string::npos);
  CHECK(with.find("\"name\":\"x\"") != std::string::npos);

  trace_options_t quiet;
  quiet.flow_edges       = false;
  const std::string none = chrome_trace(g, s, quiet);
  CHECK(count(none, "\"ph\":\"s\"") == 0);
  CHECK(well_formed(none));
}

TEST(timestamps_carry_the_schedules_own_numbers) {
  const graph_t g          = fan_in();
  const cost_table_t table = flat_table(g);
  const schedule_t s       = simulate(g, asap_runtime_t{}, table, simulate_options_t{});

  const std::string json = chrome_trace(g, s);
  CHECK(count(json, "\"dur\":2") == 6);  // every atom costs 2 in this model
  CHECK(json.find("\"ts\":0,") != std::string::npos);

  // scale is a bare multiplier on a cycle-valued schedule: it cannot supply a
  // clock, so it does not turn these bars into microseconds. This just pins
  // that the multiplier reaches every bar's rendered width unchanged, on an
  // axis whose units remain not meaningful — the route to a real time axis is
  // to_seconds(), exercised in the next test.
  trace_options_t big;
  big.scale = 1e6;
  CHECK(count(chrome_trace(g, s, big), "\"dur\":2000000") == 6);
}

TEST(a_timed_schedule_traces_in_the_unit_it_was_converted_to) {
  // The overload a viewer really wants: `to_seconds` is where a clock was
  // named, so the axis and the numbers on the bars finally mean the same thing,
  // and `trace_options_t::scale` has nothing left to do.
  const graph_t g          = fan_in();
  const cost_table_t table = flat_table(g);
  const schedule_t cycles  = simulate(g, asap_runtime_t{}, table, simulate_options_t{});

  // A gigahertz makes a cycle a nanosecond, so two cycles is 0.002 of a
  // microsecond, and the scale that gets there is the converter's own.
  const fixed_clock_t clock(1.0);
  const timed_schedule_t s = to_seconds(cycles, clock, 1e6, "us");
  CHECK(s.units == "us");

  const std::string json = chrome_trace(g, s);
  CHECK(well_formed(json));
  CHECK(count(json, "\"ph\":\"X\"") == 6);
  CHECK(count(json, "\"dur\":0.002") == 6);
  CHECK(count(json, "\"ph\":\"s\"") == g.edges().size());
}

TEST(a_cycle_schedule_renders_with_invented_tracks) {
  const graph_t g = fan_in();
  simulate_options_t opts;
  opts.lanes = 2;
  const asap_runtime_t runtime;
  const schedule_t s     = simulate(g, runtime, unit_cost_t{}, opts);
  const std::string json = chrome_trace(g, s);

  CHECK(well_formed(json));
  CHECK(count(json, "\"ph\":\"X\"") == 6);
  // Two lanes, so at most two atoms run at once and two tracks suffice.
  CHECK(count(json, "\"name\":\"thread_name\"") == 2);
  CHECK(count(json, "\"dur\":1") == 6);  // one cycle each, under the unit cost model
}

TEST(a_longer_cost_model_widens_every_bar) {
  const graph_t g = fan_in();
  simulate_options_t opts;
  opts.lanes = 2;

  class flat5_t : public cost_model_t {
   public:
    double node_cycles(const wg_node_t&) const override { return 5.0; }
    double edge_cycles(const edge_t&) const override { return 0.0; }
  };
  const asap_runtime_t runtime;
  const schedule_t s = simulate(g, runtime, flat5_t{}, opts);
  CHECK(count(chrome_trace(g, s), "\"dur\":5") == 6);
}

TEST(operation_names_are_escaped) {
  // A name with a quote in it would otherwise close the JSON string early and
  // produce a document no viewer can load.
  free_graph_t g;
  g.add_op("say \"hi\"\n\\path", 2);

  cost_table_t table(g);
  table.set("say \"hi\"\n\\path",
            op_cost_t::from_custom([](const wg_node_t&, int) { return 1.0; }));

  const std::string json =
      chrome_trace(g, simulate(g, asap_runtime_t{}, table, simulate_options_t{}));
  CHECK(well_formed(json));
  CHECK(json.find("say \\\"hi\\\"\\n\\\\path") != std::string::npos);
}

TEST(an_empty_schedule_still_renders_a_loadable_document) {
  const free_graph_t g;
  const cost_table_t table(g);
  const std::string json =
      chrome_trace(g, simulate(g, asap_runtime_t{}, table, simulate_options_t{}));

  CHECK(well_formed(json));
  CHECK(count(json, "\"ph\":\"X\"") == 0);
}

TEST(node_arguments_can_be_dropped) {
  const graph_t g          = fan_in();
  const cost_table_t table = flat_table(g);
  const schedule_t s       = simulate(g, asap_runtime_t{}, table, simulate_options_t{});

  CHECK(chrome_trace(g, s).find("\"args\":{\"wg\":") != std::string::npos);

  trace_options_t bare;
  bare.node_arguments    = false;
  const std::string json = chrome_trace(g, s, bare);
  CHECK(json.find("\"args\":{\"wg\":") == std::string::npos);
  CHECK(well_formed(json));
}

TEST(the_process_label_is_configurable) {
  const graph_t g          = fan_in();
  const cost_table_t table = flat_table(g);

  trace_options_t named;
  named.process_name = "two_shot at 8 lanes";
  const std::string json =
      chrome_trace(g, simulate(g, asap_runtime_t{}, table, simulate_options_t{}), named);
  CHECK(json.find("\"name\":\"two_shot at 8 lanes\"") != std::string::npos);
}

TEST(writing_to_a_file_round_trips) {
  const graph_t g          = fan_in();
  const cost_table_t table = flat_table(g);
  const schedule_t s       = simulate(g, asap_runtime_t{}, table, simulate_options_t{});

  const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                           "/origami-graphs-trace-test.json";
  write_chrome_trace(g, s, path);

  std::ifstream file(path);
  CHECK(file.good());
  std::stringstream buffer;
  buffer << file.rdbuf();
  CHECK(buffer.str() == chrome_trace(g, s));
  file.close();
  std::remove(path.c_str());

  bool threw = false;
  try {
    write_chrome_trace(g, s, "/nonexistent-directory-for-a-test/trace.json");
  } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
}

ORIGAMI_TEST_MAIN()
