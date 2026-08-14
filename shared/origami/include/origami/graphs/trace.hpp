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
 * @brief origami::graphs — Chrome Trace Event JSON for a schedule.
 *
 * A schedule is a Gantt chart that has not been drawn yet. This writes it in the
 * Chrome Trace Event format, which `chrome://tracing`, Perfetto and every
 * profiler UI in the ROCm stack already read, so a predicted schedule can be
 * inspected with the same tools as a measured one — and, more usefully, laid
 * next to a real rocprof trace of the kernel it models.
 *
 * Each workgroup-iteration becomes one complete event on the lane it ran on, so
 * lanes read as tracks and the bars have the widths the cost model predicted.
 * Dependencies become flow events, the arrows a viewer draws from a producer to
 * its consumers, which is what makes a critical path visible rather than merely
 * computable.
 *
 * Chrome's timestamps are microseconds, and a `schedule_t` is in cycles, so the
 * schedule a viewer really wants is a `timed_schedule_t` from
 * `to_seconds(schedule, clock, 1e6, "us")`: a trace is a picture of time, and a
 * cycle count is only a time once someone has named a clock. Both overloads are
 * offered because rendering cycles is still useful for reading a schedule's
 * shape, but `trace_options_t::scale` is a bare multiplier and cannot supply
 * the missing clock — the only supported route to a time axis is `to_seconds`
 * followed by the `timed_schedule_t` overload, never `scale` applied to cycles.
 */
#pragma once

#include <cstddef>
#include <string>

#include "origami/graphs/simulate.hpp"
#include "origami/graphs/wg_graph.hpp"

// Only simulate.hpp is needed for schedule_t's and timed_schedule_t's full
// definitions, not runtime.hpp: this file renders whatever schedule it is
// handed and never constructs a policy itself, so runtime.hpp's four concrete
// policies would be an unused dependency.

namespace origami::graphs {

/** @brief Presentation choices for a trace. */
struct trace_options_t {
  /** Process label in the viewer; a good place for the case being compared. */
  std::string process_name = "kirigami";

  /**
   * Multiplier onto every timestamp, for a `timed_schedule_t` not already
   * scaled the way the caller wants (e.g. seconds versus microseconds).
   *
   * This is a bare multiplier and cannot carry a clock, so it has no honest
   * use on a cycle-valued `schedule_t`: cycles become a time only by naming a
   * frequency, which is what `to_seconds()` is for. Do not compute a
   * cycles-to-time factor by hand and pass it here — that applies a clock
   * outside `to_seconds()`, which is exactly what this library's cycles-as-cost
   * convention exists to prevent. Left at 1 for a `timed_schedule_t` that is
   * already in the unit its caller wants.
   */
  double scale = 1.0;

  /**
   * Emit dependency arrows. One pair of events per edge, so a graph with a
   * million edges makes a file a viewer will struggle with; turning this off
   * keeps the bars and drops the arrows.
   */
  bool flow_edges = true;

  /** Emit each node's operation, workgroup and iteration as event arguments. */
  bool node_arguments = true;
};

/**
 * @brief Render a schedule in real time as Chrome Trace Event JSON.
 *
 * The overload to prefer: a viewer's axis is time, and `to_seconds` is where a
 * clock was named, so the numbers on the bars mean what the axis says.
 *
 * @param graph Graph the schedule was built from, for names and dependencies.
 * @param schedule Schedule to render, in whatever unit `to_seconds` produced.
 * @param options Presentation choices.
 * @return std::string A complete JSON document.
 */
std::string chrome_trace(const wg_graph_t& graph,
                         const timed_schedule_t& schedule,
                         const trace_options_t& options = {});

/**
 * @brief Render a cycle-based schedule's shape as Chrome Trace Event JSON.
 *
 * Not a time axis: the bars are cycles wide, and `trace_options_t::scale` is a
 * bare multiplier that cannot supply the missing clock, so the numbers on
 * Chrome's microsecond axis are not meaningful as a duration here — only the
 * relative widths and ordering are. A caller who wants a real time axis names
 * a clock, calls `to_seconds()`, and uses the `timed_schedule_t` overload
 * above instead of reaching for `scale` on this one.
 *
 * @param graph Graph the schedule was built from.
 * @param schedule Schedule to render, in cycles.
 * @param options Presentation choices.
 * @return std::string A complete JSON document.
 */
std::string chrome_trace(const wg_graph_t& graph,
                         const schedule_t& schedule,
                         const trace_options_t& options = {});

/**
 * @brief Write a real-time schedule to a Chrome Trace Event file.
 *
 * @param graph Graph the schedule was built from.
 * @param schedule Schedule to render.
 * @param path Destination file.
 * @param options Presentation choices.
 * @throws std::runtime_error If the file cannot be written.
 */
void write_chrome_trace(const wg_graph_t& graph,
                        const timed_schedule_t& schedule,
                        const std::string& path,
                        const trace_options_t& options = {});

/**
 * @brief Write a cycle-based schedule to a Chrome Trace Event file.
 *
 * @param graph Graph the schedule was built from.
 * @param schedule Schedule to render.
 * @param path Destination file.
 * @param options Presentation choices.
 * @throws std::runtime_error If the file cannot be written.
 */
void write_chrome_trace(const wg_graph_t& graph,
                        const schedule_t& schedule,
                        const std::string& path,
                        const trace_options_t& options = {});

}  // namespace origami::graphs
