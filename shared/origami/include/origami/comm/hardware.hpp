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
// hardware_t constants — the bottom of the cost model: the physical ceilings
// every higher layer divides work by.
//
// Why everything is in cycles, not seconds:
//   The model reasons about contention between functional units (VMEM issue,
//   TCP, L2, MALL, HBM, xGMI, VALU) that all advance on the same GPU clock.
//   Expressing every rate as bytes-per-cycle and every latency as cycles lets
//   the bottleneck comparison in latency.hpp be a plain max() of like units,
//   and makes the model clock-portable: retargeting to a different frequency
//   (overclock studies, a future part) only changes clock_ghz, not the
//   per-cycle physics. The single cycles→seconds conversion happens once, at
//   the public API boundary (predict_row in collective.hpp).
//
// Unit identity used throughout: a rate quoted in GB/s equals
//   (GB/s) / clock_ghz  bytes-per-cycle,
// because bytes/cycle = (bytes/ns) / (cycles/ns) = (GB/s) / clock_ghz. That is
// why the MI300X table below writes peak aggregate rates as "<GB/s> / clock".
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace origami::comm {

// ─── hardware_t (per-CU and per-XCD compute / memory) ──────────────
/**
 * @brief Per-CU and per-XCD compute and memory ceilings for one GPU.
 *
 * The physical floor of the cost model: every higher layer divides its work by
 * these throughput ceilings and latencies. All bandwidths are expressed in
 * bytes-per-cycle and all latencies in cycles (see the unit identity at the top
 * of this file) so the bottleneck comparison in latency.hpp is a max() of like
 * units.
 */
struct hardware_t {
  /// Architecture identifier string (e.g. "gfx942").
  std::string_view arch;

  // Die structure
  int num_cu;        ///< Total compute units across the whole device.
  int num_xcd;       ///< Number of accelerator complex dies (XCDs).
  int cu_per_xcd;    ///< Compute units per XCD (num_cu / num_xcd).
  double clock_ghz;  ///< cycles per ns

  // Per-CU throughput ceilings
  double vmem_issue_rate;  ///< VMEM instructions per CU per cycle
  double valu_rate;        ///< VALU lane-elements per CU per cycle

  // TCP / vL1D (per CU)
  std::size_t tcp_capacity_bytes;  ///< Per-CU vL1D/TCP cache capacity in bytes.
  double tcp_bw;                   ///< bytes per CU per cycle

  // Outstanding request limits
  int mshr_depth_per_wave;     ///< Outstanding misses a wave issues before it stalls.
  int waves_per_wg;            ///< Waves co-resident per workgroup feeding misses.
  double xgmi_latency_cycles;  ///< round-trip latency for remote load

  // L2 / TCC (per XCD)
  std::size_t l2_capacity_bytes;  ///< Per-XCD L2/TCC capacity in bytes.
  double l2_bw_per_cu;            ///< bytes per CU per cycle

  // MALL / Infinity Cache (device-wide)
  std::size_t mall_capacity_bytes;  ///< Device-wide MALL/Infinity Cache capacity in bytes.
  double mall_bw;                   ///< bytes per cycle aggregate

  // HBM
  double hbm_read_bw;              ///< bytes per cycle aggregate
  double hbm_write_bw;             ///< bytes per cycle aggregate
  std::size_t hbm_capacity_bytes;  ///< Total HBM capacity in bytes.

  /// HBM bandwidth-utilization polynomial: fraction = a*N^2 + b*N + c of peak,
  /// clamped to [0,1], where N is the number of CUs concurrently streaming.
  ///
  /// First principle: a single CU cannot saturate HBM. Peak HBM bandwidth is
  /// only reached once enough independent CUs issue in parallel to keep every
  /// channel/arbiter busy. The fabric arbitrates per-request with no per-CU
  /// isolation, so sustained bandwidth ramps roughly linearly with active CUs
  /// until the channels saturate. The default {0,0.015,0} encodes
  /// fraction ≈ 0.015·N, i.e. each CU contributes ~1.5% of peak and the array
  /// reaches full peak near N≈67 CUs. (This linear fit is calibrated for the
  /// many-CU collective regime; it knowingly understates single-CU sustained
  /// bandwidth — see day2-atom-validation — which is irrelevant once dozens of
  /// workgroups stream a collective.)
  std::array<double, 3> mem_bw_coeffs = {0.0, 0.015, 0.0};

  // ── BW polynomial ──────────────────────────────────────────
  /**
   * @brief Fraction of peak HBM bandwidth sustainable with `active_cus` streaming.
   *
   * Evaluates the utilization polynomial a*N^2 + b*N + c (N = active CUs) and
   * clamps the result to [0, 1]. See @ref mem_bw_coeffs for the physical
   * rationale behind the coefficients.
   *
   * @param active_cus Number of CUs concurrently streaming HBM.
   * @return double Sustainable fraction of peak bandwidth in [0, 1].
   */
  constexpr double bw_fraction(int active_cus) const noexcept {
    const double N = static_cast<double>(active_cus);
    const double f = mem_bw_coeffs[0] * N * N + mem_bw_coeffs[1] * N + mem_bw_coeffs[2];
    return std::clamp(f, 0.0, 1.0);
  }

  /**
   * @brief Per-CU share of HBM read bandwidth at a given active-CU count.
   *
   * Per-CU share of HBM read bandwidth: take the aggregate ceiling, discount
   * it by how much of peak N active CUs can actually sustain (bw_fraction),
   * then split that sustained aggregate evenly across the N contenders. With
   * the linear polynomial the two N's partly cancel, so per-CU bandwidth is
   * nearly flat across the saturated range — the realistic behaviour of a
   * shared memory system once it is busy.
   *
   * @param active_cus Number of CUs concurrently streaming (-1 uses num_cu).
   * @return double Per-CU read bandwidth in bytes per cycle.
   */
  constexpr double hbm_read_bw_per_cu(int active_cus = -1) const noexcept {
    const int n    = (active_cus < 0) ? num_cu : active_cus;
    const double f = bw_fraction(n);
    return hbm_read_bw * f / static_cast<double>(n);
  }

  /**
   * @brief Per-CU share of HBM write bandwidth at a given active-CU count.
   *
   * The write-path analogue of @ref hbm_read_bw_per_cu: the aggregate write
   * ceiling is discounted by bw_fraction and split evenly across the N active
   * contenders.
   *
   * @param active_cus Number of CUs concurrently streaming (-1 uses num_cu).
   * @return double Per-CU write bandwidth in bytes per cycle.
   */
  constexpr double hbm_write_bw_per_cu(int active_cus = -1) const noexcept {
    const int n    = (active_cus < 0) ? num_cu : active_cus;
    const double f = bw_fraction(n);
    return hbm_write_bw * f / static_cast<double>(n);
  }

  /**
   * @brief Per-CU L2 bandwidth, scaled for partial XCD occupancy.
   *
   * Per-CU L2 bandwidth, scaled for partial XCD occupancy. l2_bw_per_cu is
   * calibrated for a fully-occupied XCD (all cu_per_xcd CUs sharing the TCC
   * crossbar). When only n < cu_per_xcd CUs are active, the same crossbar
   * bandwidth is shared among fewer consumers, so each active CU gets a larger
   * slice — hence the (cu_per_xcd / n) upscaling. This is the inverse of the
   * HBM polynomial: HBM under-delivers when under-subscribed, whereas the L2
   * crossbar is per-CU-bounded so an idle CU's share is reclaimed.
   *
   * @param active_cus_on_xcd CUs active on the XCD (-1 uses cu_per_xcd).
   * @return double Per-CU L2 bandwidth in bytes per cycle.
   */
  constexpr double l2_bw_per_cu_scaled(int active_cus_on_xcd = -1) const noexcept {
    int n = (active_cus_on_xcd < 0) ? cu_per_xcd : active_cus_on_xcd;
    n     = std::min(n, cu_per_xcd);
    return l2_bw_per_cu * (static_cast<double>(cu_per_xcd) / std::max(n, 1));
  }

  // ── Frequency / cycle ↔ time conversion ────────────────────
  /**
   * @brief Clock frequency in hertz.
   *
   * @return double Clock frequency in Hz (clock_ghz * 1e9).
   */
  constexpr double clock_hz() const noexcept { return clock_ghz * 1e9; }

  /**
   * @brief Convert a duration in cycles to seconds.
   *
   * @param cycles Duration in clock cycles.
   * @return double Equivalent duration in seconds.
   */
  constexpr double cycles_to_seconds(double cycles) const noexcept { return cycles / clock_hz(); }
  /**
   * @brief Convert a duration in cycles to nanoseconds.
   *
   * @param cycles Duration in clock cycles.
   * @return double Equivalent duration in nanoseconds.
   */
  constexpr double cycles_to_ns(double cycles) const noexcept {
    return cycles_to_seconds(cycles) * 1e9;
  }
  /**
   * @brief Convert a duration in cycles to microseconds.
   *
   * @param cycles Duration in clock cycles.
   * @return double Equivalent duration in microseconds.
   */
  constexpr double cycles_to_us(double cycles) const noexcept {
    return cycles_to_seconds(cycles) * 1e6;
  }
  /**
   * @brief Convert a duration in seconds to cycles.
   *
   * @param s Duration in seconds.
   * @return double Equivalent duration in clock cycles.
   */
  constexpr double seconds_to_cycles(double s) const noexcept { return s * clock_hz(); }
  /**
   * @brief Convert a duration in nanoseconds to cycles.
   *
   * @param ns Duration in nanoseconds.
   * @return double Equivalent duration in clock cycles.
   */
  constexpr double ns_to_cycles(double ns) const noexcept { return seconds_to_cycles(ns * 1e-9); }
  /**
   * @brief Convert a duration in microseconds to cycles.
   *
   * @param us Duration in microseconds.
   * @return double Equivalent duration in clock cycles.
   */
  constexpr double us_to_cycles(double us) const noexcept { return seconds_to_cycles(us * 1e-6); }

  /**
   * @brief Convert a per-cycle rate to a per-second rate.
   *
   * @param per_cycle Rate expressed per clock cycle.
   * @return double Equivalent rate per second.
   */
  constexpr double rate_per_second(double per_cycle) const noexcept {
    return per_cycle * clock_hz();
  }
  /**
   * @brief Convert a per-cycle rate to a per-nanosecond rate.
   *
   * @param per_cycle Rate expressed per clock cycle.
   * @return double Equivalent rate per nanosecond.
   */
  constexpr double rate_per_ns(double per_cycle) const noexcept {
    return rate_per_second(per_cycle) * 1e-9;
  }

  /**
   * @brief Convert a per-second rate to a per-cycle rate.
   *
   * @param per_s Rate expressed per second.
   * @return double Equivalent rate per clock cycle.
   */
  constexpr double rate_per_cycle_from_per_second(double per_s) const noexcept {
    return per_s / clock_hz();
  }
  /**
   * @brief Convert a per-nanosecond rate to a per-cycle rate.
   *
   * @param per_ns Rate expressed per nanosecond.
   * @return double Equivalent rate per clock cycle.
   */
  constexpr double rate_per_cycle_from_per_ns(double per_ns) const noexcept {
    return rate_per_cycle_from_per_second(per_ns * 1e9);
  }
};

// ─── comm_hardware_t (inter-GPU communication) ──────────────────────
/**
 * @brief Inter-GPU fabric ceilings: xGMI links, SDMA engines, and protocol overheads.
 *
 * Captures everything the cost model needs about the communication fabric that
 * joins one GPU to its peers. As with @ref hardware_t, bandwidths are in
 * bytes-per-cycle and latencies in cycles.
 */
struct comm_hardware_t {
  // xGMI link
  double link_bw;      ///< bytes per cycle per link, unidirectional
  int num_peer_links;  ///< links to other GPUs

  // SDMA
  int num_sdma_engines;  ///< Number of SDMA (DMA-copy) engines.
  double sdma_read_bw;   ///< bytes per cycle per link
  double sdma_write_bw;  ///< bytes per cycle per link

  // Protocol overhead
  double atomic_latency_cycles;   ///< Latency of one fabric atomic (signal/wait), in cycles.
  double launch_overhead_cycles;  ///< Fixed per-collective kernel launch/setup floor, in cycles.

  double clock_ghz = 2.0;  ///< companion clock for ns display helpers

  /**
   * @brief Clock frequency in hertz.
   *
   * @return double Clock frequency in Hz (clock_ghz * 1e9).
   */
  constexpr double clock_hz() const noexcept { return clock_ghz * 1e9; }

  /**
   * @brief Convert a duration in cycles to seconds.
   *
   * @param cycles Duration in clock cycles.
   * @return double Equivalent duration in seconds.
   */
  constexpr double cycles_to_seconds(double cycles) const noexcept { return cycles / clock_hz(); }
  /**
   * @brief Convert a duration in cycles to nanoseconds.
   *
   * @param cycles Duration in clock cycles.
   * @return double Equivalent duration in nanoseconds.
   */
  constexpr double cycles_to_ns(double cycles) const noexcept {
    return cycles_to_seconds(cycles) * 1e9;
  }
  /**
   * @brief Convert a duration in cycles to microseconds.
   *
   * @param cycles Duration in clock cycles.
   * @return double Equivalent duration in microseconds.
   */
  constexpr double cycles_to_us(double cycles) const noexcept {
    return cycles_to_seconds(cycles) * 1e6;
  }
  /**
   * @brief Convert a duration in seconds to cycles.
   *
   * @param s Duration in seconds.
   * @return double Equivalent duration in clock cycles.
   */
  constexpr double seconds_to_cycles(double s) const noexcept { return s * clock_hz(); }
  /**
   * @brief Convert a duration in nanoseconds to cycles.
   *
   * @param ns Duration in nanoseconds.
   * @return double Equivalent duration in clock cycles.
   */
  constexpr double ns_to_cycles(double ns) const noexcept { return seconds_to_cycles(ns * 1e-9); }
  /**
   * @brief Convert a duration in microseconds to cycles.
   *
   * @param us Duration in microseconds.
   * @return double Equivalent duration in clock cycles.
   */
  constexpr double us_to_cycles(double us) const noexcept { return seconds_to_cycles(us * 1e-6); }

  /**
   * @brief Convert a per-cycle rate to a per-second rate.
   *
   * @param per_cycle Rate expressed per clock cycle.
   * @return double Equivalent rate per second.
   */
  constexpr double rate_per_second(double per_cycle) const noexcept {
    return per_cycle * clock_hz();
  }
  /**
   * @brief Convert a per-cycle rate to a per-nanosecond rate.
   *
   * @param per_cycle Rate expressed per clock cycle.
   * @return double Equivalent rate per nanosecond.
   */
  constexpr double rate_per_ns(double per_cycle) const noexcept {
    return rate_per_second(per_cycle) * 1e-9;
  }
  /**
   * @brief Convert a per-nanosecond rate to a per-cycle rate.
   *
   * @param per_ns Rate expressed per nanosecond.
   * @return double Equivalent rate per clock cycle.
   */
  constexpr double rate_per_cycle_from_per_ns(double per_ns) const noexcept {
    return per_ns * 1e9 / clock_hz();
  }
};

// ─── system_t (the physical machine) ────────────────────────────────
/**
 * @brief The physical machine: a GPU plus the fabric that joins it to its peers.
 *
 * A GPU plus the fabric that joins it to its peers. The model is currently
 * homogeneous — every rank is assumed to be `gpu` — so a single hardware_t
 * suffices rather than one per GPU; the GPU *count* is the communicator size
 * and lives in comm_problem_t::num_gpus, not here. If heterogeneous nodes or a
 * non-uniform link topology ever need modelling, `gpu` grows into a per-rank
 * container and `fabric` into a bandwidth matrix; until then this stays the
 * minimal honest representation of what the cost model actually consumes.
 */
struct system_t {
  hardware_t gpu;          ///< one GPU's compute/memory ceilings
  comm_hardware_t fabric;  ///< the xGMI mesh between GPUs
};

// ─── MI300X (CDNA3, gfx942) ──────────────────────────────────────
/**
 * @brief Nominal CDNA3 compute-engine clock used to derive the MI300X tables.
 *
 * 2.0 GHz is the nominal CDNA3 compute-engine clock; every "X / clock" and
 * "X * clock" below converts a natively-measured rate or latency into the
 * model's cycle units (see the unit identity at the top of this file).
 */
inline constexpr double _MI300X_CLOCK_GHZ = 2.0;

/**
 * @brief MI300X (CDNA3, gfx942) hardware ceilings.
 *
 * Per-field derivations:
 *   num_cu/num_xcd/cu_per_xcd  : CDNA3 die layout — num_xcd XCDs, each with
 *                                cu_per_xcd compute units (their product).
 *   vmem_issue_rate = 1.0      : one VMEM instruction issued per CU per cycle
 *                                (the memory pipe accepts one address/cycle).
 *   valu_rate = 2.10 * 64.0    : 64 lanes per SIMD × ~2.10 elements/lane/cycle
 *                                sustained (dual-issue + occupancy factor).
 *   tcp_capacity_bytes = 32 KiB: the vL1D/TCP per-CU cache; the request FIFO
 *                                saturates around ~5 wide loads in flight.
 *   tcp_bw = 64.0              : one global_load_dwordx16 per cycle = 64 B/cycle
 *                                — the widest VMEM transaction the lane can move.
 *   mshr_depth_per_wave = 12   : 12 outstanding dwordx4 misses per wave before
 *                                the wave stalls (the measured N=13 cliff).
 *   waves_per_wg = 10          : waves co-resident per workgroup feeding misses.
 *   xgmi_latency_cycles        : 660 ns measured remote-load round trip × clock
 *                                = 1320 cycles. This is the latency the MSHR
 *                                depth must hide to sustain remote bandwidth.
 *   l2/mall/hbm *_bw           : measured aggregate peaks (GB/s) ÷ clock to get
 *                                bytes/cycle; HBM write peak (5140) > read
 *                                (4730) on this part.
 *   hbm_capacity 192 GiB       : 8 HBM3 stacks.
 *   mem_bw_coeffs              : HBM utilization-vs-active-CU fit (see above).
 */
inline constexpr hardware_t MI300X = {
    /* arch                 */ "gfx942",
    /* num_cu               */ 304,
    /* num_xcd              */ 8,
    /* cu_per_xcd           */ 38,
    /* clock_ghz            */ _MI300X_CLOCK_GHZ,
    /* vmem_issue_rate      */ 1.0,
    /* valu_rate            */ 2.10 * 64.0,
    /* tcp_capacity_bytes   */ 32ULL * 1024ULL,
    /* tcp_bw               */ 64.0,
    /* mshr_depth_per_wave  */ 12,
    /* waves_per_wg         */ 10,
    /* xgmi_latency_cycles  */ 660.0 * _MI300X_CLOCK_GHZ,
    /* l2_capacity_bytes    */ 4ULL * 1024ULL * 1024ULL,
    /* l2_bw_per_cu         */ 83.6 / _MI300X_CLOCK_GHZ,
    /* mall_capacity_bytes  */ 256ULL * 1024ULL * 1024ULL,
    /* mall_bw              */ 4730.0 / _MI300X_CLOCK_GHZ,
    /* hbm_read_bw          */ 4730.0 / _MI300X_CLOCK_GHZ,
    /* hbm_write_bw         */ 5140.0 / _MI300X_CLOCK_GHZ,
    /* hbm_capacity_bytes   */ 192ULL * 1024ULL * 1024ULL * 1024ULL,
    /* mem_bw_coeffs        */ {0.0, 0.015, 0.0},
};

/**
 * @brief MI300X inter-GPU xGMI fabric ceilings.
 *
 * Inter-GPU fabric. Per-field derivations:
 *   link_bw : 49.1 GiB/s measured per-link wire rate, converted to decimal
 *             bytes/s (×1024^3 / 1e9), discounted by the 1.23× wire-to-payload
 *             overhead (flit framing + ECC), then ÷ clock to get payload
 *             bytes/cycle. Net ≈ 42.9 GB/s of usable payload per link.
 *   num_peer_links = 7 : an MI300X reaches the other 7 GPUs over a fully
 *             connected single-hop xGMI mesh (one link per peer).
 *   num_sdma_engines / sdma_*_bw : DMA-copy engines and their measured one-way
 *             rates (read faster than write), ÷ clock to bytes/cycle.
 *   atomic_latency_cycles : ~100 ns per signal/wait fabric atomic × clock.
 *             This is the cost charged per producer/consumer handshake.
 *   launch_overhead_cycles : ~45 µs fixed kernel-launch + setup floor × clock,
 *             charged once per collective; it dominates the sub-kilobyte regime.
 */
inline constexpr comm_hardware_t MI300X_COMM = {
    /* link_bw                 */ 49.1 * (1024.0 * 1024.0 * 1024.0) / 1e9 / 1.23 /
        _MI300X_CLOCK_GHZ,
    /* num_peer_links          */ 7,
    /* num_sdma_engines        */ 14,
    /* sdma_read_bw            */ 49.5 / _MI300X_CLOCK_GHZ,
    /* sdma_write_bw           */ 23.6 / _MI300X_CLOCK_GHZ,
    /* atomic_latency_cycles   */ 100.0 * _MI300X_CLOCK_GHZ,
    /* launch_overhead_cycles  */ 45000.0 * _MI300X_CLOCK_GHZ,
    /* clock_ghz               */ _MI300X_CLOCK_GHZ,
};

/**
 * @brief The default machine: one MI300X GPU on the MI300X xGMI fabric.
 */
inline constexpr system_t MI300X_SYSTEM = {MI300X, MI300X_COMM};

}  // namespace origami::comm
