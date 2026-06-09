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

// origami::comm — analytical communication cost model
//
// Tensor Collective layer — shape-aware frontend over predict_row.
//
// Top of the three-layer architecture:
//
//   Tensor Collective layer  (shape-aware: tensor + op + dim + W)
//     ↓ lowers via convention table
//   Collective layer          (byte-level: primitive + msg_bytes + W + NCH)
//     ↓ lowers via collective_algorithm_t
//   Workgroup layer
#pragma once

#include "origami/comm/collective.hpp"
#include "origami/comm/hardware.hpp"
#include "origami/comm/heuristics.hpp"
#include "origami/comm/types.hpp"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace origami::comm {

// ─── Wire-factor + msg-bytes convention ──────────────────────────
/**
 * @brief Fabric bytes a rank moves per byte of user buffer for a collective.
 *
 * These are the standard bus-bandwidth factors (matching rccl-tests): with N
 * ranks a ring touches each byte N-1 times, so AG and RS carry (N-1)×; an
 * all-reduce is an RS followed by an AG, hence 2(N-1)/N; an all-to-all keeps
 * 1/N locally and ships the rest, (N-1)/N; a broadcast sends each byte once.
 * Exposed for reporting (wire_bytes_per_rank); the cost model derives traffic
 * from the algorithm, so this is a cross-check, not the source of the prediction.
 *
 * @param op Collective name (all_reduce, all_gather, reduce_scatter, broadcast,
 *        all_to_all).
 * @param world_size Number of participating ranks.
 * @return double Wire-byte multiplier per byte of user buffer.
 * @throws std::invalid_argument If op is not a known collective.
 */
double wire_factor(std::string_view op, int world_size);

/**
 * @brief Convert a per-rank byte count into predict_row's msg_bytes convention.
 *
 * predict_row takes msg_bytes in the benchmark's convention, which differs by
 * op: every collective reports its per-rank buffer except reduce_scatter, whose
 * msg_bytes is the *aggregate* pre-scatter buffer = per_rank × N. This reverses
 * the per-rank division predict_row applies, so the two agree.
 *
 * @param op Collective name.
 * @param per_rank_bytes Per-rank buffer size in bytes.
 * @param world_size Number of participating ranks.
 * @return std::size_t msg_bytes in predict_row's convention.
 */
std::size_t msg_bytes_for_predict_row(std::string_view op,
                                      std::size_t per_rank_bytes,
                                      int world_size);

// ─── dtype normalization ─────────────────────────────────────────
/**
 * @brief Normalize a dtype name to a data_type_t.
 *
 * Accepts a data_type_t enum, a lowercase string alias, or a string with
 * torch./np./numpy. prefix (the enum case is handled by the overload below).
 *
 * @param dt Dtype name (e.g. "bf16", "torch.float32", "fp8").
 * @return data_type_t The matching enum value.
 * @throws std::invalid_argument If the dtype name is unsupported.
 */
data_type_t normalize_dtype(std::string_view dt);

/**
 * @brief Identity overload: a data_type_t is already normalized.
 *
 * @param dt The dtype enum value.
 * @return data_type_t The same value, unchanged.
 */
data_type_t normalize_dtype(data_type_t dt) noexcept;

// ─── Shape → (M_full, N_full, split_dim) lowering ───────────────
/**
 * @brief Full logical tensor extent and the sharded axis.
 *
 * The model reasons about the *full* logical tensor and a split axis, but the
 * caller supplies a *per-rank* shape (what each GPU holds). This reconstructs
 * the global [M,N]: collapse all-but-last dims into M (rows) and the last dim
 * into N (columns), then multiply whichever axis was sharded by world_size to
 * recover its full extent. split_dim records which axis that was, so the lower
 * layers re-derive each rank's tile by the inverse division.
 */
struct full_mn_t {
  /// Full row count (all-but-last dims collapsed).
  std::size_t M_full;
  /// Full column count (last dim).
  std::size_t N_full;
  /// Sharded axis: 0 = rows (M), 1 = columns (N).
  int split_dim;
};

/**
 * @brief Reconstruct the full [M,N] tensor extent and split axis from a per-rank
 *        shape.
 *
 * Collapses all-but-last dims into M and the last dim into N, then multiplies the
 * sharded axis by world_size to recover its full extent.
 *
 * @param shape Per-rank tensor shape (what each GPU holds); must be non-empty
 *        with all-positive entries.
 * @param dim Sharded axis; negative values index from the end (Python-style).
 * @param world_size Number of participating ranks (must be >= 1).
 * @return full_mn_t Full [M_full, N_full] extent with the normalized split_dim.
 * @throws std::invalid_argument On empty shape, non-positive entries,
 *         world_size < 1, or dim out of range.
 */
full_mn_t per_rank_shape_to_full_mn(const std::vector<std::size_t>& shape, int dim, int world_size);

/**
 * @brief Result of a shape-aware tensor collective prediction.
 *
 * Bundles the predicted latency with the inputs and the derived byte/tile
 * quantities used to produce it, plus framework-overhead bookkeeping.
 */
struct tensor_collective_prediction_t {
  /// Total predicted latency in µs, including framework overhead.
  double predicted_us;
  /// Collective name.
  std::string op;
  /// Per-rank input tensor shape.
  std::vector<std::size_t> input_shape;
  /// Sharded axis as supplied by the caller.
  int dim;
  /// Number of participating ranks.
  int world_size;
  /// Channels/workgroups driving the collective.
  int nchannels;
  /// Element data type.
  data_type_t dtype;
  /// Per-rank buffer size in bytes.
  std::size_t per_rank_bytes;
  /// Fabric bytes per rank (wire_factor cross-check).
  std::size_t wire_bytes_per_rank;
  /// msg_bytes in predict_row's convention.
  std::size_t msg_bytes;
  /// Per-GPU tile shape.
  tile_shape_t gpu_tile;
  /// Caller framework, used for the overhead lookup.
  std::string framework = "raw";
  /// Framework overhead included in predicted_us (µs).
  double framework_overhead_us = 0.0;

  /**
   * @brief Predicted latency with framework overhead removed.
   *
   * @return double Backend-only latency in microseconds.
   */
  constexpr double backend_us() const noexcept { return predicted_us - framework_overhead_us; }
};

// ─── Supported op set ───────────────────────────────────────────
/// Collective op names accepted by predict_tensor_collective.
inline constexpr std::string_view SUPPORTED_OPS[] = {"all_reduce",
                                                     "all_gather",
                                                     "reduce_scatter",
                                                     "broadcast",
                                                     "all_to_all"};

/**
 * @brief Check whether an op name is a supported collective.
 *
 * @param op Collective name to test.
 * @return bool True if op is in SUPPORTED_OPS, false otherwise.
 */
bool is_supported_op(std::string_view op);

/**
 * @brief Shape-aware tensor collective prediction (typed-dtype overload).
 *
 * Lowers a per-rank tensor shape + op into the byte-level predict_row model and
 * returns the predicted latency together with the derived byte/tile quantities.
 * W=1 is a no-op for every collective but still pays the framework overhead.
 *
 * @param op Collective name (must be a supported op).
 * @param input_shape Per-rank input tensor shape.
 * @param dtype Element data type.
 * @param world_size Number of participating ranks (must be >= 1).
 * @param dim Sharded axis (defaults to 0); negative values index from the end.
 * @param nchannels Channels/workgroups driving the collective (defaults to 32).
 * @param system GPU + fabric hardware description (defaults to MI300X_SYSTEM).
 * @param framework Caller framework for overhead accounting (defaults to "raw").
 * @param heur Tunable heuristic parameters (defaults to DEFAULT_HEURISTICS).
 * @return tensor_collective_prediction_t Prediction plus inputs and derived data.
 * @throws std::invalid_argument On unsupported op or world_size < 1.
 */
tensor_collective_prediction_t predict_tensor_collective(
    std::string_view op,
    const std::vector<std::size_t>& input_shape,
    data_type_t dtype,
    int world_size,
    int dim                    = 0,
    int nchannels              = 32,
    const system_t& system     = MI300X_SYSTEM,
    std::string_view framework = "raw",
    const heuristics_t& heur   = DEFAULT_HEURISTICS);

/**
 * @brief Shape-aware tensor collective prediction (string-dtype convenience
 *        overload).
 *
 * Normalizes dtype_name via normalize_dtype and forwards to the typed overload.
 *
 * @param op Collective name (must be a supported op).
 * @param input_shape Per-rank input tensor shape.
 * @param dtype_name Element data type as a string alias (e.g. "bf16").
 * @param world_size Number of participating ranks (must be >= 1).
 * @param dim Sharded axis (defaults to 0); negative values index from the end.
 * @param nchannels Channels/workgroups driving the collective (defaults to 32).
 * @param system GPU + fabric hardware description (defaults to MI300X_SYSTEM).
 * @param framework Caller framework for overhead accounting (defaults to "raw").
 * @param heur Tunable heuristic parameters (defaults to DEFAULT_HEURISTICS).
 * @return tensor_collective_prediction_t Prediction plus inputs and derived data.
 * @throws std::invalid_argument On unsupported op, world_size < 1, or unsupported
 *         dtype.
 */
tensor_collective_prediction_t predict_tensor_collective(
    std::string_view op,
    const std::vector<std::size_t>& input_shape,
    std::string_view dtype_name,
    int world_size,
    int dim                    = 0,
    int nchannels              = 32,
    const system_t& system     = MI300X_SYSTEM,
    std::string_view framework = "raw",
    const heuristics_t& heur   = DEFAULT_HEURISTICS);

}  // namespace origami::comm
