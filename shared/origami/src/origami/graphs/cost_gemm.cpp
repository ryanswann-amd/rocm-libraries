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

#include "origami/graphs/cost_gemm.hpp"

#include <stdexcept>
#include <utility>

namespace origami::graphs {

double gemm_seconds(const gemm_spec_t& spec, const hardware_t& hardware) {
  if (!spec.config.is_valid()) {
    throw std::invalid_argument(
        "gemm_seconds: the config is not valid; macro-tile and MFMA dimensions must be positive "
        "and occupancy must be set");
  }
  if (!(hardware.compute_clock_ghz > 0.0)) {
    throw std::invalid_argument("gemm_seconds: the hardware reports a non-positive clock, so "
                                "cycles cannot be converted to seconds");
  }

  gemm::context_t context(spec.problem, hardware, spec.config);
  if (spec.active_cus) context.active_cus = *spec.active_cus;

  // compute_tile_latency prices one K-complete macro-tile on one compute unit,
  // which is precisely one workgroup of a GEMM operation. Its answer is cycles.
  const double cycles = gemm::compute_tile_latency(spec.problem, hardware, spec.config, context);
  return cycles / (hardware.compute_clock_ghz * 1e9);
}

op_cost_t gemm_cost(gemm_spec_t spec, const hardware_t& hardware) {
  // Priced once rather than per node: the model's answer depends on the problem
  // and the launch, not on which workgroup is asking, so calling it for every
  // atom of a large grid would repeat identical work.
  const double seconds = gemm_seconds(spec, hardware);
  return op_cost_t::tagged(cost_kind_t::gemm, [seconds](const wg_node_t&, int) { return seconds; });
}

}  // namespace origami::graphs
