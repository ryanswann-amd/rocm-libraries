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

#include "origami/comm/tensor.hpp"

namespace origami::comm {

double wire_factor(primitive_t op, int world_size) {
  const double n = static_cast<double>(world_size);
  switch (op) {
    case primitive_t::all_reduce: return 2.0 * (n - 1.0) / n;
    case primitive_t::all_gather: return (n - 1.0);
    case primitive_t::reduce_scatter: return (n - 1.0);
    case primitive_t::broadcast: return 1.0;
    case primitive_t::all_to_all: return (n - 1.0) / n;
  }
  throw std::invalid_argument(std::string{"unknown collective: "} +
                              std::string{primitive_name(op)});
}

std::size_t msg_bytes_for_predict_row(primitive_t op, std::size_t per_rank_bytes, int world_size) {
  if (op == primitive_t::reduce_scatter) {
    return per_rank_bytes * static_cast<std::size_t>(world_size);
  }
  return per_rank_bytes;
}

data_type_t normalize_dtype(std::string_view dt) {
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

data_type_t normalize_dtype(data_type_t dt) noexcept { return dt; }

full_mn_t per_rank_shape_to_full_mn(const std::vector<std::size_t>& shape,
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

tensor_collective_prediction_t predict_tensor_collective(
    std::string_view op,
    const std::vector<std::size_t>& input_shape,
    data_type_t dtype,
    int world_size,
    int dim,
    int nchannels,
    const system_t& system,
    std::string_view framework,
    const heuristics_t& heur) {
  const primitive_t prim = primitive_from_name(op);  // string -> enum at the edge
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
      static_cast<std::size_t>(wire_factor(prim, world_size) * static_cast<double>(per_rank_bytes));

  const auto full             = per_rank_shape_to_full_mn(input_shape, dim, world_size);
  const std::size_t msg_bytes = msg_bytes_for_predict_row(prim, per_rank_bytes, world_size);

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

tensor_collective_prediction_t predict_tensor_collective(
    std::string_view op,
    const std::vector<std::size_t>& input_shape,
    std::string_view dtype_name,
    int world_size,
    int dim,
    int nchannels,
    const system_t& system,
    std::string_view framework,
    const heuristics_t& heur) {
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
