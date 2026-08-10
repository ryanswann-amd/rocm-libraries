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
 * @brief origami::graphs — the GEMM cost arm, over origami's analytical model.
 *
 * @warning This header is **not** HIP-free, and it is the only part of
 * origami::graphs that is not. `origami/gemm.hpp` reaches `origami/hardware.hpp`,
 * which includes `<hip/hip_runtime.h>`, so anything including this file needs
 * HIP headers and must link `roc::origami-graphs-gemm` rather than plain
 * `roc::origami-graphs`.
 *
 * That split is the whole reason this is a separate target. The graphs core has
 * to stay buildable with nothing but a C++17 compiler — that is what lets it be
 * tested, fuzzed and used on a machine with no ROCm install — so the arm that
 * needs HIP is quarantined here rather than allowed to make the core need it
 * too. Callers who never price a GEMM never pay for it.
 *
 * The bridge is deliberately thin. `origami::gemm::compute_tile_latency` already
 * models one K-complete macro-tile on one compute unit, which is exactly one
 * workgroup of a GEMM operation, so the work here is unit conversion and
 * plumbing rather than modelling: origami reports **cycles**, the graphs cost
 * interface is in **seconds**.
 */
#pragma once

#include <optional>

#include "origami/gemm.hpp"
#include "origami/graphs/cost.hpp"
#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami::graphs {

/**
 * @brief Everything origami's GEMM model needs to price one workgroup.
 *
 * A `problem_t` and a `config_t` are what origami's selector already works in,
 * so a caller that ranked tile shapes with `origami::select_config` can pass the
 * winner straight through without restating it.
 */
struct gemm_spec_t {
  problem_t problem;  ///< global GEMM shape and data types
  config_t config;    ///< tile geometry and kernel configuration

  /**
   * Compute units the model should assume are active. Unset uses the hardware's
   * full complement, which is what origami's own whole-GEMM path does.
   */
  std::optional<std::size_t> active_cus;
};

/**
 * @brief Duration of one GEMM workgroup, in seconds.
 *
 * Wraps `origami::gemm::compute_tile_latency`, which prices one K-complete
 * macro-tile on one compute unit.
 *
 * @param spec GEMM shape and tile configuration.
 * @param hardware Machine description.
 * @return double Seconds, converted from the model's cycles at the machine clock.
 * @throws std::invalid_argument If the configuration is invalid, or if the
 *         hardware reports a non-positive clock.
 */
double gemm_seconds(const gemm_spec_t& spec, const hardware_t& hardware);

/**
 * @brief Price an operation with origami's analytical GEMM model.
 *
 * The result reports `cost_kind_t::gemm`, so `cost_table_t::kind_of` names the
 * arm the caller chose. Contention is not forwarded: origami's tile model takes
 * an active-CU count as a property of the launch rather than of the moment, so
 * `gemm_spec_t::active_cus` fixes it and the scheduler's instantaneous occupancy
 * is ignored. Saying so here is better than quietly substituting one for the
 * other, which would make an `event_driven_runtime_t` result look
 * contention-aware when it was not.
 *
 * @param spec GEMM shape and tile configuration.
 * @param hardware Machine description; copied, so it need not outlive the entry.
 * @return op_cost_t An entry for `cost_table_t::set`.
 */
op_cost_t gemm_cost(gemm_spec_t spec, const hardware_t& hardware);

}  // namespace origami::graphs
