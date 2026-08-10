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

TEST(a_continuous_schedule_renders_one_slice_per_atom) {
  const graph_t g          = fan_in();
  const cost_table_t table = flat_table(g);
  cost_runtime_options_t opts;
  opts.lanes = 3;
  opts.scale = 1.0;

  const cost_schedule_t s = roofline_runtime_t(table, opts).schedule(g);
  const std::string json  = chrome_trace(g, s);

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
  cost_runtime_options_t opts;
  opts.lanes = 3;
  opts.scale = 1.0;

  const std::string json = chrome_trace(g, roofline_runtime_t(table, opts).schedule(g));

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
  cost_runtime_options_t opts;
  opts.scale              = 1.0;
  const cost_schedule_t s = roofline_runtime_t(table, opts).schedule(g);

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
  cost_runtime_options_t opts;
  opts.scale              = 1.0;
  const cost_schedule_t s = roofline_runtime_t(table, opts).schedule(g);

  const std::string json = chrome_trace(g, s);
  CHECK(count(json, "\"dur\":2") == 6);  // every atom costs 2 in this model
  CHECK(json.find("\"ts\":0,") != std::string::npos);

  // Chrome reads microseconds, so a schedule left in seconds needs scaling or
  // every bar is a million times too narrow.
  trace_options_t micros;
  micros.scale = 1e6;
  CHECK(count(chrome_trace(g, s, micros), "\"dur\":2000000") == 6);
}

TEST(an_integer_schedule_renders_with_invented_tracks) {
  const graph_t g        = fan_in();
  const schedule_t s     = asap_runtime_t(std::optional<int>{2}).schedule(g);
  const std::string json = chrome_trace(g, s);

  CHECK(well_formed(json));
  CHECK(count(json, "\"ph\":\"X\"") == 6);
  // Two lanes, so at most two atoms share a timestep and two tracks suffice.
  CHECK(count(json, "\"name\":\"thread_name\"") == 2);
  CHECK(count(json, "\"dur\":1") == 6);  // one timestep each
}

TEST(a_longer_timestep_widens_every_bar) {
  const graph_t g    = fan_in();
  const schedule_t s = asap_runtime_t(std::optional<int>{2}, /*wg_duration=*/5).schedule(g);
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
  cost_runtime_options_t opts;
  opts.scale = 1.0;

  const std::string json = chrome_trace(g, roofline_runtime_t(table, opts).schedule(g));
  CHECK(well_formed(json));
  CHECK(json.find("say \\\"hi\\\"\\n\\\\path") != std::string::npos);
}

TEST(an_empty_schedule_still_renders_a_loadable_document) {
  const free_graph_t g;
  const cost_table_t table(g);
  const std::string json = chrome_trace(g, roofline_runtime_t(table).schedule(g));

  CHECK(well_formed(json));
  CHECK(count(json, "\"ph\":\"X\"") == 0);
}

TEST(node_arguments_can_be_dropped) {
  const graph_t g          = fan_in();
  const cost_table_t table = flat_table(g);
  cost_runtime_options_t opts;
  opts.scale              = 1.0;
  const cost_schedule_t s = roofline_runtime_t(table, opts).schedule(g);

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
  cost_runtime_options_t opts;
  opts.scale = 1.0;

  trace_options_t named;
  named.process_name     = "two_shot at 8 lanes";
  const std::string json = chrome_trace(g, roofline_runtime_t(table, opts).schedule(g), named);
  CHECK(json.find("\"name\":\"two_shot at 8 lanes\"") != std::string::npos);
}

TEST(writing_to_a_file_round_trips) {
  const graph_t g          = fan_in();
  const cost_table_t table = flat_table(g);
  cost_runtime_options_t opts;
  opts.scale              = 1.0;
  const cost_schedule_t s = roofline_runtime_t(table, opts).schedule(g);

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
