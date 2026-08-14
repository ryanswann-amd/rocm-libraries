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

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace origami::graphs {

namespace {

constexpr int kProcessId = 1;

/** JSON string escaping. Operation names are caller-supplied, so assume nothing. */
std::string quote(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 2);
  out += '"';
  for (char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  out += '"';
  return out;
}

/**
 * Nine significant digits, which is short enough to stay readable and long
 * enough that a schedule left in seconds does not round its bars away.
 */
std::string number(double value) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.9g", value);
  return std::string(buf);
}

/** Metadata events, which is how a viewer learns what to call a track. */
void emit_names(std::string& out, const trace_options_t& options, const std::vector<int>& lanes) {
  out += "  {\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,\"args\":{\"name\":" +
         quote(options.process_name) + "}},\n";
  for (int lane : lanes) {
    const std::string tid = std::to_string(lane);
    out += "  {\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" + tid +
           ",\"args\":{\"name\":\"lane " + tid + "\"}},\n";
    // Without this a viewer orders tracks by first appearance, which puts the
    // lanes in dispatch order rather than lane order.
    out += "  {\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":1,\"tid\":" + tid +
           ",\"args\":{\"sort_index\":" + tid + "}},\n";
  }
}

/** One complete event: a bar on a track. */
void emit_slice(std::string& out,
                const wg_graph_t& graph,
                const wg_node_t& node,
                int lane,
                double start,
                double duration,
                bool arguments) {
  const std::string& op = graph.op_name(node.op);
  out += "  {\"ph\":\"X\",\"pid\":1,\"tid\":" + std::to_string(lane) + ",\"name\":" + quote(op) +
         ",\"cat\":" + quote(op) + ",\"ts\":" + number(start) + ",\"dur\":" + number(duration);
  if (arguments) {
    out += ",\"args\":{\"wg\":" + std::to_string(node.wg) + ",\"it\":" + std::to_string(node.it) +
           ",\"op\":" + quote(op) + "}";
  }
  out += "},\n";
}

/**
 * A dependency arrow: one event where the producer ends, one where the consumer
 * begins. `bp:"e"` binds each end to the enclosing slice so the arrow lands on
 * the bar rather than floating at the timestamp.
 */
void emit_flow(std::string& out,
               std::size_t id,
               const std::string& allocation,
               int src_lane,
               double src_ts,
               int dst_lane,
               double dst_ts) {
  const std::string tag = quote(allocation);
  const std::string sid = std::to_string(id);
  out += "  {\"ph\":\"s\",\"pid\":1,\"tid\":" + std::to_string(src_lane) + ",\"id\":" + sid +
         ",\"name\":" + tag + ",\"cat\":\"dependency\",\"ts\":" + number(src_ts) + "},\n";
  out += "  {\"ph\":\"f\",\"bp\":\"e\",\"pid\":1,\"tid\":" + std::to_string(dst_lane) +
         ",\"id\":" + sid + ",\"name\":" + tag +
         ",\"cat\":\"dependency\",\"ts\":" + number(dst_ts) + "},\n";
}

/** Strip the trailing comma the emitters leave, and close the document. */
std::string close(std::string out) {
  const std::size_t last = out.find_last_of(',');
  if (last != std::string::npos && out.find_first_not_of(" \n", last + 1) == std::string::npos) {
    out.erase(last, 1);
  }
  out += "],\n \"displayTimeUnit\": \"ms\"\n}\n";
  return out;
}

std::string open_document() { return std::string("{\n \"traceEvents\": [\n"); }

void write_file(const std::string& path, const std::string& body) {
  std::ofstream file(path);
  if (!file) throw std::runtime_error("write_chrome_trace: cannot open '" + path + "' for writing");
  file << body;
  if (!file) throw std::runtime_error("write_chrome_trace: failed while writing '" + path + "'");
}

/**
 * Shared body for both schedule types. `schedule_t` and `timed_schedule_t`
 * differ only in the unit their numbers are in; both carry a real lane per
 * atom, so both render the same way.
 */
template <typename ScheduleT>
std::string chrome_trace_impl(const wg_graph_t& graph,
                              const ScheduleT& schedule,
                              const trace_options_t& options) {
  std::string out = open_document();

  std::vector<int> lanes;
  for (const wg_node_t& n : schedule.order) lanes.push_back(schedule.lane.at(n));
  std::sort(lanes.begin(), lanes.end());
  lanes.erase(std::unique(lanes.begin(), lanes.end()), lanes.end());
  emit_names(out, options, lanes);

  for (const wg_node_t& n : schedule.order) {
    emit_slice(out,
               graph,
               n,
               schedule.lane.at(n),
               schedule.start.at(n) * options.scale,
               schedule.duration.at(n) * options.scale,
               options.node_arguments);
  }

  if (options.flow_edges) {
    std::size_t id = 0;
    for (const edge_t& e : graph.edges()) {
      const auto src = schedule.start.find(e.src);
      const auto dst = schedule.start.find(e.dst);
      if (src == schedule.start.end() || dst == schedule.start.end()) continue;
      emit_flow(out,
                id++,
                e.allocation,
                schedule.lane.at(e.src),
                schedule.finish(e.src) * options.scale,
                schedule.lane.at(e.dst),
                dst->second * options.scale);
    }
  }
  return close(std::move(out));
}

}  // namespace

// ─── real-time schedules ──────────────────────────────────────────────

std::string chrome_trace(const wg_graph_t& graph,
                         const timed_schedule_t& schedule,
                         const trace_options_t& options) {
  return chrome_trace_impl(graph, schedule, options);
}

void write_chrome_trace(const wg_graph_t& graph,
                        const timed_schedule_t& schedule,
                        const std::string& path,
                        const trace_options_t& options) {
  write_file(path, chrome_trace(graph, schedule, options));
}

// ─── cycle-based schedules ─────────────────────────────────────────────

std::string chrome_trace(const wg_graph_t& graph,
                         const schedule_t& schedule,
                         const trace_options_t& options) {
  return chrome_trace_impl(graph, schedule, options);
}

void write_chrome_trace(const wg_graph_t& graph,
                        const schedule_t& schedule,
                        const std::string& path,
                        const trace_options_t& options) {
  write_file(path, chrome_trace(graph, schedule, options));
}

}  // namespace origami::graphs
