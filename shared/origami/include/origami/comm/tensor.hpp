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
inline double wire_factor(std::string_view op, int world_size) {
  const double n = static_cast<double>(world_size);
  if (op == "all_reduce") return 2.0 * (n - 1.0) / n;
  if (op == "all_gather") return (n - 1.0);
  if (op == "reduce_scatter") return (n - 1.0);
  if (op == "broadcast") return 1.0;
  if (op == "all_to_all") return (n - 1.0) / n;
  throw std::invalid_argument(std::string{"unknown op: "} + std::string{op});
}

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
inline std::size_t msg_bytes_for_predict_row(std::string_view op,
                                             std::size_t per_rank_bytes,
                                             int world_size) {
  if (op == "reduce_scatter") { return per_rank_bytes * static_cast<std::size_t>(world_size); }
  return per_rank_bytes;
}

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
inline data_type_t normalize_dtype(std::string_view dt) {
  auto strip = [](std::string s) {
    // Lowercase + strip common prefixes.
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    auto erase_prefix = [&](std::string_view p) {
      if (s.rfind(p, 0) == 0) s.erase(0, p.size());
    };
    erase_prefix("torch.");
    erase_prefix("np.");
    erase_prefix("numpy.");
    return s;
  };
  const std::string key = strip(std::string{dt});
  if (key == "bf16" || key == "bfloat16") return data_type_t::BFloat16;
  if (key == "fp16" || key == "float16" || key == "half") return data_type_t::Half;
  if (key == "fp32" || key == "float32" || key == "float") return data_type_t::Float;
  if (key == "fp64" || key == "float64" || key == "double") return data_type_t::Double;
  if (key == "fp8") return data_type_t::Float8;
  if (key == "int8") return data_type_t::Int8;
  throw std::invalid_argument(std::string{"unsupported dtype: "} + std::string{dt});
}

/**
 * @brief Identity overload: a data_type_t is already normalized.
 *
 * @param dt The dtype enum value.
 * @return data_type_t The same value, unchanged.
 */
inline data_type_t normalize_dtype(data_type_t dt) noexcept { return dt; }

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
inline full_mn_t per_rank_shape_to_full_mn(const std::vector<std::size_t>& shape,
                                           int dim,
                                           int world_size) {
  if (shape.empty()) {
    throw std::invalid_argument("input_shape must have at least one dimension");
  }
  for (auto d : shape) {
    if (d == 0) { throw std::invalid_argument("input_shape has non-positive entries"); }
  }
  if (world_size < 1) { throw std::invalid_argument("world_size must be >= 1"); }

  const int rank_ndim = static_cast<int>(shape.size());
  const int norm_dim  = (dim >= 0) ? dim : dim + rank_ndim;
  if (norm_dim < 0 || norm_dim >= rank_ndim) {
    throw std::invalid_argument("dim out of range for input_shape");
  }

  const std::size_t n_per_rank = shape.back();
  std::size_t m_per_rank       = 1;
  for (int i = 0; i + 1 < rank_ndim; ++i) m_per_rank *= shape[i];

  const int last_dim = rank_ndim - 1;
  if (norm_dim == last_dim) {
    return {m_per_rank, n_per_rank * static_cast<std::size_t>(world_size), 1};
  }
  return {m_per_rank * static_cast<std::size_t>(world_size), n_per_rank, 0};
}

// ─── tensor_collective_prediction_t ─────────────────────────────────
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
inline bool is_supported_op(std::string_view op) {
  for (const auto& s : SUPPORTED_OPS)
    if (s == op) return true;
  return false;
}

// ─── predict_tensor_collective ──────────────────────────────────
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
inline tensor_collective_prediction_t predict_tensor_collective(
    std::string_view op,
    const std::vector<std::size_t>& input_shape,
    data_type_t dtype,
    int world_size,
    int dim                    = 0,
    int nchannels              = 32,
    const system_t& system     = MI300X_SYSTEM,
    std::string_view framework = "raw",
    const heuristics_t& heur   = DEFAULT_HEURISTICS) {
  if (!is_supported_op(op)) {
    throw std::invalid_argument(std::string{"unsupported op: "} + std::string{op});
  }
  if (world_size < 1) { throw std::invalid_argument("world_size must be >= 1"); }

  const int eb = dtype_bytes(dtype);

  const double overhead_us = heur.framework_overhead_us(framework);

  // Degenerate: W=1 is a no-op for every collective, but the framework
  // still pays its overhead.
  if (world_size == 1) {
    std::size_t per_rank_elements = 1;
    for (auto d : input_shape) per_rank_elements *= d;
    const std::size_t per_rank_bytes_w1 = per_rank_elements * static_cast<std::size_t>(eb);

    tile_shape_t gpu_tile_w1;
    if (!input_shape.empty()) {
      const std::size_t n_last = input_shape.back();
      std::size_t outer        = 1;
      for (std::size_t i = 0; i + 1 < input_shape.size(); ++i) outer *= input_shape[i];
      gpu_tile_w1 = tile_shape_t{outer, n_last, dtype, /*split_dim=*/0, /*contiguous=*/true};
    }

    tensor_collective_prediction_t out{};
    out.predicted_us          = overhead_us;
    out.op                    = std::string{op};
    out.input_shape           = input_shape;
    out.dim                   = dim;
    out.world_size            = 1;
    out.nchannels             = nchannels;
    out.dtype                 = dtype;
    out.per_rank_bytes        = per_rank_bytes_w1;
    out.wire_bytes_per_rank   = 0;
    out.msg_bytes             = per_rank_bytes_w1;
    out.gpu_tile              = gpu_tile_w1;
    out.framework             = std::string{framework};
    out.framework_overhead_us = overhead_us;
    return out;
  }

  std::size_t per_rank_elements = 1;
  for (auto d : input_shape) per_rank_elements *= d;
  const std::size_t per_rank_bytes = per_rank_elements * static_cast<std::size_t>(eb);

  const std::size_t wire_bytes_per_rank =
      static_cast<std::size_t>(wire_factor(op, world_size) * static_cast<double>(per_rank_bytes));

  const auto full             = per_rank_shape_to_full_mn(input_shape, dim, world_size);
  const std::size_t msg_bytes = msg_bytes_for_predict_row(op, per_rank_bytes, world_size);

  const double backend_us = predict_row(
      op, msg_bytes, world_size, nchannels, system, full.M_full, full.N_full, full.split_dim, heur);

  const double predicted_us = backend_us + overhead_us;

  tile_shape_t gpu_tile{
      (full.split_dim == 0) ? full.M_full / static_cast<std::size_t>(world_size) : full.M_full,
      (full.split_dim == 1) ? full.N_full / static_cast<std::size_t>(world_size) : full.N_full,
      dtype,
      full.split_dim,
      /*contiguous=*/true};

  tensor_collective_prediction_t out{};
  out.predicted_us          = predicted_us;
  out.op                    = std::string{op};
  out.input_shape           = input_shape;
  out.dim                   = dim;
  out.world_size            = world_size;
  out.nchannels             = nchannels;
  out.dtype                 = dtype;
  out.per_rank_bytes        = per_rank_bytes;
  out.wire_bytes_per_rank   = wire_bytes_per_rank;
  out.msg_bytes             = msg_bytes;
  out.gpu_tile              = gpu_tile;
  out.framework             = std::string{framework};
  out.framework_overhead_us = overhead_us;
  return out;
}

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
inline tensor_collective_prediction_t predict_tensor_collective(
    std::string_view op,
    const std::vector<std::size_t>& input_shape,
    std::string_view dtype_name,
    int world_size,
    int dim                    = 0,
    int nchannels              = 32,
    const system_t& system     = MI300X_SYSTEM,
    std::string_view framework = "raw",
    const heuristics_t& heur   = DEFAULT_HEURISTICS) {
  return predict_tensor_collective(op,
                                   input_shape,
                                   normalize_dtype(dtype_name),
                                   world_size,
                                   dim,
                                   nchannels,
                                   system,
                                   framework,
                                   heur);
}

}  // namespace origami::comm
