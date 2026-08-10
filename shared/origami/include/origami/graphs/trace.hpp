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
 * Chrome's timestamps are microseconds. `cost_schedule_t` is in whatever unit
 * its runtime's `scale` produced, which defaults to microseconds, so the default
 * is already right; a schedule built in seconds needs `trace_options_t::scale`
 * set to 1e6 or the bars will be a million times too narrow.
 */
#pragma once

#include <cstddef>
#include <string>

#include "origami/graphs/cost_runtime.hpp"
#include "origami/graphs/runtime.hpp"
#include "origami/graphs/wg_graph.hpp"

namespace origami::graphs {

/** @brief Presentation choices for a trace. */
struct trace_options_t {
  /** Process label in the viewer; a good place for the case being compared. */
  std::string process_name = "kirigami";

  /**
   * Multiplier onto every timestamp, for schedules not already in microseconds.
   * A `cost_schedule_t` from a default runtime is, so this is usually left at 1.
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
 * @brief Render a continuous-time schedule as Chrome Trace Event JSON.
 *
 * @param graph Graph the schedule was built from, for names and dependencies.
 * @param schedule Schedule to render.
 * @param options Presentation choices.
 * @return std::string A complete JSON document.
 */
std::string chrome_trace(const wg_graph_t& graph,
                         const cost_schedule_t& schedule,
                         const trace_options_t& options = {});

/**
 * @brief Render an integer-timestep schedule as Chrome Trace Event JSON.
 *
 * `schedule_t` records when each atom ran but not where, because an integer
 * runtime models lane *count* rather than lane identity. Atoms sharing a
 * timestep are therefore spread across tracks in dispatch order, which is a
 * faithful picture of the shape even though the specific track an atom lands on
 * carries no meaning.
 *
 * @param graph Graph the schedule was built from.
 * @param schedule Schedule to render.
 * @param options Presentation choices.
 * @return std::string A complete JSON document.
 */
std::string chrome_trace(const wg_graph_t& graph,
                         const schedule_t& schedule,
                         const trace_options_t& options = {});

/**
 * @brief Write a continuous-time schedule to a Chrome Trace Event file.
 *
 * @param graph Graph the schedule was built from.
 * @param schedule Schedule to render.
 * @param path Destination file.
 * @param options Presentation choices.
 * @throws std::runtime_error If the file cannot be written.
 */
void write_chrome_trace(const wg_graph_t& graph,
                        const cost_schedule_t& schedule,
                        const std::string& path,
                        const trace_options_t& options = {});

/**
 * @brief Write an integer-timestep schedule to a Chrome Trace Event file.
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
