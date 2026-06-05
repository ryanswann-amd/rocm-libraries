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
// Core type definitions for the communication cost model.
//
// Two ideas thread through every type here:
//   1. The cache line is the indivisible unit of memory traffic. The hardware
//      never moves a partial line, so the model counts work in *cache lines*,
//      not bytes, and tracks how efficiently a payload packs into them
//      (cacheline_efficiency). Wasted line bytes are real wasted bandwidth.
//   2. Work is decomposed top-down by even division: a [M,N] tensor is split
//      across GPUs, each GPU's tile is split across timesteps, each timestep
//      tile across workgroups, each WG tile streamed in fixed-size iterations.
//      Every divide_* below preserves total bytes while choosing an axis that
//      keeps rows contiguous (and therefore cache-line-aligned) when it can.
//
// Header-only; constexpr where possible so the whole division tree can fold
// at compile time.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

#include "origami/types.hpp"

namespace origami::comm {

// Forward declaration so comm_config_t can carry a layout-override pointer
// without including layouts.hpp (which depends on this header).
class collective_layout_t;

// The CDNA cache line is 64 B: the granularity at which TCP/L2/MALL/HBM and
// the xGMI fabric tag, fetch, and evict data. All traffic in the model is
// rounded up to whole lines because that is what the silicon actually moves.
inline constexpr std::size_t CACHELINE_BYTES = 64;

// ─── data_type_t ────────────────────────────────────────────────────
// Reuse the canonical Origami data-type enum rather than defining a second,
// incompatible one. `origami::data_type_t` resolves unqualified here via
// enclosing-namespace lookup (origami::comm is nested in origami).
using origami::data_type_t;

// Element size in bytes. Kept header-only on purpose: origami::data_type_to_bytes()
// is backed by datatype_to_bits() in the (HIP-linked) origami library, so calling
// it would force comm consumers to link roc::origami and pull in HIP. The comm
// model only needs byte sizes for the dtypes it supports, so we keep a local
// switch and preserve the header-only / HIP-free contract.
constexpr int dtype_bytes(data_type_t dt) noexcept {
  switch (dt) {
    case data_type_t::Float8:
    case data_type_t::Int8: return 1;
    case data_type_t::Half:
    case data_type_t::BFloat16: return 2;
    case data_type_t::Float: return 4;
    case data_type_t::Double: return 8;
    default: return 0;
  }
}

// ─── load_width_t ───────────────────────────────────────────────────
// How many bytes a single VMEM instruction moves. A wider load (dwordx16 =
// 64 B) fills a whole cache line in one instruction, so it minimizes the VMEM
// *issue* count — and VMEM issue (one instr/CU/cycle) is a distinct ceiling
// from memory bandwidth. Narrow loads can make issue, not bandwidth, the
// bottleneck. The enum value *is* the byte width, so casting recovers it.
enum class load_width_t : std::uint8_t {
  DWORD    = 4,
  DWORDX4  = 16,
  DWORDX16 = 64,
};

constexpr int load_width_bytes(load_width_t w) noexcept { return static_cast<int>(w); }

// Instructions needed to touch one cache line = 64 B / bytes-per-instr. This
// converts a cache-line count (the bandwidth view) into a VMEM-issue count
// (the issue-rate view) so latency.hpp can compare the two ceilings.
constexpr int instrs_per_cacheline(load_width_t w) noexcept {
  return static_cast<int>(CACHELINE_BYTES) / load_width_bytes(w);
}

// ─── direction_t / reduce_op_t (carried through op_t resolution) ───────
enum class direction_t : std::uint8_t { PUSH, PULL };
enum class reduce_op_t : std::uint8_t { SUM, MAX, MIN, PROD };

// ─── primitive_t: the collective OPERATION ─────────────────────────────
// Which collective is being performed. This is a property of the *problem*,
// not the config: it determines the correct result — an all-gather and a
// reduce-scatter of the same buffer produce different answers — whereas the
// layout that *implements* it (ring vs two-shot, …) is a performance choice
// and therefore lives in comm_config_t. Lives in types.hpp because both the
// problem type and the heuristics table are keyed by it.
enum class primitive_t : std::uint8_t {
  all_gather,
  reduce_scatter,
  broadcast,
  all_reduce,
  all_to_all,
};

inline constexpr std::array<std::string_view, 5> PRIMITIVE_NAMES = {
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "all_reduce",
    "all_to_all",
};

constexpr std::string_view primitive_name(primitive_t p) noexcept {
  return PRIMITIVE_NAMES[static_cast<std::size_t>(p)];
}

// Parse a canonical name into the enum. Used only at the public string edge
// (predict_row / predict_tensor_collective); throws on an unknown name, which
// preserves the original string-keyed layout factory's behaviour.
inline primitive_t primitive_from_name(std::string_view name) {
  for (std::size_t i = 0; i < PRIMITIVE_NAMES.size(); ++i) {
    if (PRIMITIVE_NAMES[i] == name) return static_cast<primitive_t>(i);
  }
  throw std::invalid_argument(std::string{"unknown collective: "} + std::string{name});
}

// ─── ceil_div: ceil(a/b) for positive integers ─────
template <typename A, typename B>
constexpr auto ceil_div(A a, B b) noexcept -> std::common_type_t<A, B> {
  using U = std::common_type_t<A, B>;
  return static_cast<U>((static_cast<U>(a) + static_cast<U>(b) - 1) / static_cast<U>(b));
}

// ─── tile_shape_t ───────────────────────────────────────────────────
// A 2D row-major tile (m × n × dtype) with a contiguity bit.
// The contiguous/non-contiguous regimes and their costs are described
// on the member accessors below.
struct tile_shape_t {
  std::size_t m     = 1;
  std::size_t n     = 1;
  data_type_t dtype = data_type_t::BFloat16;
  int split_dim     = 0;
  bool contiguous   = true;

  constexpr tile_shape_t() noexcept = default;
  constexpr tile_shape_t(std::size_t m_,
                         std::size_t n_,
                         data_type_t dt,
                         int split   = 0,
                         bool contig = true) noexcept
      : m{std::max<std::size_t>(m_, 1)}
      , n{std::max<std::size_t>(n_, 1)}
      , dtype{dt}
      , split_dim{split}
      , contiguous{contig} {}

  constexpr int element_bytes() const noexcept { return dtype_bytes(dtype); }
  constexpr std::size_t elements() const noexcept { return m * n; }
  constexpr std::size_t bytes() const noexcept { return elements() * element_bytes(); }

  // Cache lines per row, rounding up: a row whose byte length is not a
  // multiple of 64 still consumes a whole final line. This rounding is the
  // entire source of cache-line inefficiency for strided tiles.
  constexpr std::size_t cl_per_row() const noexcept {
    return ceil_div(n * static_cast<std::size_t>(element_bytes()), CACHELINE_BYTES);
  }

  // Total cache lines transferred. The contiguity bit captures a real memory-
  // layout fact: a contiguous tile is one flat byte run, so only the *single*
  // final line is partially wasted (ceil over total bytes). A non-contiguous
  // (column-striped) tile has m independent rows, each of which pays its own
  // partial-line rounding — m × cl_per_row — which can be far more traffic
  // than the same bytes laid out contiguously.
  constexpr std::size_t cachelines() const noexcept {
    if (contiguous) { return std::max<std::size_t>(ceil_div(bytes(), CACHELINE_BYTES), 1); }
    return m * cl_per_row();
  }

  // Logical 2D cacheline footprint.
  constexpr std::pair<std::size_t, std::size_t> cacheline_shape() const noexcept {
    if (contiguous) return {1, cachelines()};
    return {m, cl_per_row()};
  }

  // Useful bytes ÷ bytes actually moved across the line interface. 1.0 means
  // perfect packing; < 1.0 means bandwidth is spent on padding. Higher layers
  // pay for transferred-line bandwidth, so this ratio is the lever that makes
  // a poorly-aligned strided collective slower than its byte count suggests.
  constexpr double cacheline_efficiency() const noexcept {
    const std::size_t transferred = cachelines() * CACHELINE_BYTES;
    if (transferred == 0) return 1.0;
    return static_cast<double>(bytes()) / static_cast<double>(transferred);
  }

  // Split a tile into `factor` equal chunks along one axis.
  //   axis=0 row-stripe : fewer rows, same row length → stays contiguous and
  //                       cache-line-aligned (the cheap split).
  //   axis=1 column-stripe : shorter rows → each row re-pays partial-line
  //                       rounding, so the result is marked non-contiguous.
  constexpr tile_shape_t divide(std::size_t factor, int axis = 0) const noexcept {
    factor = std::max<std::size_t>(factor, 1);
    if (axis == 0) {
      return tile_shape_t{
          std::max<std::size_t>(ceil_div(m, factor), 1), n, dtype, split_dim, contiguous};
    }
    return tile_shape_t{
        m,
        std::max<std::size_t>(ceil_div(n, factor), 1),
        dtype,
        /*split_dim=*/1,
        /*contiguous=*/false,
    };
  }

  // Pick the cheaper split when possible: if there are at least `factor` rows,
  // row-stripe (axis 0) to keep contiguity; otherwise we are forced to cut
  // within rows (axis 1) and accept the per-row line padding. This is why a
  // tall tensor parallelizes across WGs more cheaply than a wide one.
  constexpr tile_shape_t divide_byte_equal(std::size_t factor) const noexcept {
    factor = std::max<std::size_t>(factor, 1);
    if (factor == 1) return *this;
    if (m >= factor) return divide(factor, /*axis=*/0);
    return divide(factor, /*axis=*/1);
  }
};

// ─── comm_problem_t ─────────────────────────────────────────────────
// 2D tensor [M, N] distributed across num_gpus, split along `split_dim`.
// split_dim selects which axis the data is partitioned on: split_dim=0 gives
// each rank a horizontal stripe of rows (ceil(M/num_gpus) × N, still dense and
// contiguous), split_dim=1 gives a vertical stripe of columns. The choice
// follows the collective's data-distribution convention and feeds the same
// contiguity reasoning as tile_shape_t.
struct comm_problem_t {
  std::size_t M;
  std::size_t N;
  int num_gpus;
  data_type_t dtype = data_type_t::BFloat16;
  int split_dim     = 0;
  // The operation to perform. Correctness-determining, hence part of the
  // problem (kept last so existing 5-field aggregate initializers still bind).
  primitive_t collective = primitive_t::all_reduce;

  constexpr int element_bytes() const noexcept { return dtype_bytes(dtype); }
  constexpr std::size_t message_bytes() const noexcept { return M * N * element_bytes(); }
  constexpr std::size_t total_elements() const noexcept { return M * N; }

  constexpr std::size_t gpu_tile_m() const noexcept {
    return (split_dim == 0) ? ceil_div(M, static_cast<std::size_t>(num_gpus)) : M;
  }
  constexpr std::size_t gpu_tile_n() const noexcept {
    return (split_dim == 1) ? ceil_div(N, static_cast<std::size_t>(num_gpus)) : N;
  }
  constexpr std::size_t gpu_tile_elements() const noexcept { return gpu_tile_m() * gpu_tile_n(); }
  constexpr std::size_t gpu_tile_bytes() const noexcept {
    return gpu_tile_elements() * element_bytes();
  }

  // Per-rank buffer is always dense → contiguous=true.
  constexpr tile_shape_t gpu_tile_shape() const noexcept {
    return tile_shape_t{gpu_tile_m(), gpu_tile_n(), dtype, split_dim, /*contiguous=*/true};
  }

  constexpr std::size_t gpu_tile_cachelines() const noexcept {
    return gpu_tile_shape().cachelines();
  }

  constexpr double cacheline_efficiency() const noexcept {
    const std::size_t transferred = gpu_tile_cachelines() * CACHELINE_BYTES;
    if (transferred == 0) return 1.0;
    return static_cast<double>(gpu_tile_bytes()) / static_cast<double>(transferred);
  }
};

// ─── comm_config_t ──────────────────────────────────────────────────
// Workgroup-level execution config. `min_bytes_per_wg` is sourced
// from heuristics::DEFAULT_HEURISTICS by the factory below — the
// constructor takes an explicit value to keep this header standalone.
struct comm_config_t {
  int num_wgs;
  load_width_t load_width = load_width_t::DWORDX16;
  int vgprs_for_data      = 128;
  int min_bytes_per_wg    = 16'384;  // default mirrors heuristics_t.min_bytes_per_wg

  // Optional implementation override for the problem's collective. null ⇒ the
  // engine uses default_layout_for(problem.collective). A performance choice
  // (same result, different cost), so it belongs in the config, not the
  // problem. Must implement problem.collective if supplied.
  const collective_layout_t* layout = nullptr;

  // Bytes a workgroup moves per software-pipelined iteration = the register
  // budget it dedicates to in-flight data (vgprs_for_data × 4 B/VGPR). This is
  // the depth of the load→compute→store pipeline the WG can keep outstanding;
  // it sets how the WG tile is chopped into iterations in latency.hpp.
  constexpr int bytes_per_iter() const noexcept { return vgprs_for_data * 4; }
  constexpr int cl_per_iter() const noexcept {
    return bytes_per_iter() / static_cast<int>(CACHELINE_BYTES);
  }
  constexpr int instrs_per_cl() const noexcept { return instrs_per_cacheline(load_width); }

  // Effective (not requested) workgroup count. A caller may launch many
  // channels, but if the per-timestep tile is small, splitting it across all
  // of them leaves each WG with a sliver of work. Below ~min_bytes_per_wg the
  // launch/sync cost per WG dominates and extra WGs do not add bandwidth, so
  // the model caps the count at floor(tile_bytes / min_bytes_per_wg) for all
  // bandwidth-scaling purposes. The default 16 KiB ≈ the NCCL LL128 minimum
  // chunk, the point below which more channels stop helping in practice.
  // (Launch + sync overheads still charge the full requested num_wgs.)
  constexpr int effective_num_wgs(std::size_t tile_bytes) const noexcept {
    if (min_bytes_per_wg <= 0) return num_wgs;
    const auto fit =
        std::max<std::size_t>(tile_bytes / static_cast<std::size_t>(min_bytes_per_wg), 1);
    return std::min(static_cast<std::size_t>(num_wgs), fit) == 0
               ? num_wgs
               : static_cast<int>(std::min(static_cast<std::size_t>(num_wgs), fit));
  }
};

// ─── functional_unit_work_t ──────────────────────────────────────────
// The work a single iteration imposes on each hardware functional unit,
// counted in its native unit: cache lines for the memory/fabric stages
// (TCP, L2, MALL, HBM, xGMI), issued instructions for VMEM, lane-ops for
// VALU, and discrete fabric atomics for sync. Keeping the units separate is
// the whole point: latency.hpp divides each count by that unit's independent
// throughput and takes the max, modelling the FUs as parallel contending
// resources rather than one fused "bandwidth". operator+ lets a work graph
// accumulate the contributions of its primitives. This is the atom that
// resolve_work_graph builds up.
struct functional_unit_work_t {
  std::int64_t vmem_read_instrs  = 0;
  std::int64_t vmem_write_instrs = 0;
  std::int64_t tcp_read_cl       = 0;
  std::int64_t tcp_write_cl      = 0;
  std::int64_t l2_read_cl        = 0;
  std::int64_t l2_write_cl       = 0;
  std::int64_t mall_read_cl      = 0;
  std::int64_t mall_write_cl     = 0;
  std::int64_t hbm_read_cl       = 0;
  std::int64_t hbm_write_cl      = 0;
  std::int64_t xgmi_read_cl      = 0;
  std::int64_t xgmi_write_cl     = 0;
  std::int64_t valu_ops          = 0;
  std::int64_t atomic_count      = 0;

  static constexpr functional_unit_work_t zero() noexcept { return {}; }

  constexpr functional_unit_work_t& operator+=(const functional_unit_work_t& o) noexcept {
    vmem_read_instrs += o.vmem_read_instrs;
    vmem_write_instrs += o.vmem_write_instrs;
    tcp_read_cl += o.tcp_read_cl;
    tcp_write_cl += o.tcp_write_cl;
    l2_read_cl += o.l2_read_cl;
    l2_write_cl += o.l2_write_cl;
    mall_read_cl += o.mall_read_cl;
    mall_write_cl += o.mall_write_cl;
    hbm_read_cl += o.hbm_read_cl;
    hbm_write_cl += o.hbm_write_cl;
    xgmi_read_cl += o.xgmi_read_cl;
    xgmi_write_cl += o.xgmi_write_cl;
    valu_ops += o.valu_ops;
    atomic_count += o.atomic_count;
    return *this;
  }

  friend constexpr functional_unit_work_t operator+(functional_unit_work_t a,
                                                    const functional_unit_work_t& b) noexcept {
    a += b;
    return a;
  }
};

// ─── wg_tile_latency_breakdown_t ──────────────────────────────────────
// Full wg_tile transfer latency in **GPU cycles**, plus per-FU breakdowns.
// ns helpers go through `clock_ghz` (defaulted to 2.0 = MI300X).
struct wg_tile_latency_breakdown_t {
  double T_total_cycles    = 0.0;
  double T_wlt_cycles      = 0.0;
  double T_prologue_cycles = 0.0;
  double T_epilogue_cycles = 0.0;
  double T_sync_cycles     = 0.0;
  std::size_t num_iters    = 0;

  double T_vmem_cycles       = 0.0;
  double T_tcp_cycles        = 0.0;
  double T_l2_cycles         = 0.0;
  double T_mall_cycles       = 0.0;
  double T_hbm_read_cycles   = 0.0;
  double T_hbm_write_cycles  = 0.0;
  double T_xgmi_read_cycles  = 0.0;
  double T_xgmi_write_cycles = 0.0;
  double T_valu_cycles       = 0.0;
  std::string bottleneck;

  double clock_ghz = 2.0;

  constexpr double clock_hz() const noexcept { return clock_ghz * 1e9; }

  constexpr double cycles_to_ns(double cycles) const noexcept { return cycles / clock_hz() * 1e9; }

  // ns accessors for display; the *_cycles fields are the source of truth.
  constexpr double T_total() const noexcept { return cycles_to_ns(T_total_cycles); }
  constexpr double T_wlt() const noexcept { return cycles_to_ns(T_wlt_cycles); }
  constexpr double T_prologue() const noexcept { return cycles_to_ns(T_prologue_cycles); }
  constexpr double T_epilogue() const noexcept { return cycles_to_ns(T_epilogue_cycles); }
  constexpr double T_sync() const noexcept { return cycles_to_ns(T_sync_cycles); }
  constexpr double T_vmem() const noexcept { return cycles_to_ns(T_vmem_cycles); }
  constexpr double T_tcp() const noexcept { return cycles_to_ns(T_tcp_cycles); }
  constexpr double T_l2() const noexcept { return cycles_to_ns(T_l2_cycles); }
  constexpr double T_mall() const noexcept { return cycles_to_ns(T_mall_cycles); }
  constexpr double T_hbm_read() const noexcept { return cycles_to_ns(T_hbm_read_cycles); }
  constexpr double T_hbm_write() const noexcept { return cycles_to_ns(T_hbm_write_cycles); }
  constexpr double T_xgmi_read() const noexcept { return cycles_to_ns(T_xgmi_read_cycles); }
  constexpr double T_xgmi_write() const noexcept { return cycles_to_ns(T_xgmi_write_cycles); }
  constexpr double T_valu() const noexcept { return cycles_to_ns(T_valu_cycles); }
};

}  // namespace origami::comm
