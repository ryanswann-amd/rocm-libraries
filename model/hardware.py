"""
MI300X / MI350X hardware constants for the communication cost model.

Every value traces to a measured architectural constant from the global wiki
(K-series experiments) or AMD documentation. See wiki/pages/mi300x-architecture.md.

UNIT CONVENTION (refactored 2026-06-01)
---------------------------------------
This module is the *bottom* of the cost model. All physical constants are
expressed in **GPU cycles** (or bytes/cycle for bandwidths), NOT in ns.
The conversion to seconds happens once, at the public API boundary
(`predict_row` in collective.py: divides by `clock_ghz * 1000` to produce
microseconds).

Rationale: the architectural source numbers are natively cycle-based
(e.g. "1 dwordx16 / cycle = 64 B/cycle" for VMEM, "660 ns RTT @ 2 GHz =
1320 cycles" for xGMI). Pre-multiplying by clock_ghz to expose ns values
obscures the underlying physics. Cycle units also let the same constants
serve multiple clocks (overclock studies, future architectures) by only
swapping `clock_ghz`.

Exception: framework_overhead in heuristics.py stays in **ns** — it's
host wall time and is unrelated to the GPU clock.
"""

from dataclasses import dataclass


@dataclass
class Hardware:
    """Per-CU and per-XCD compute/memory resources.

    All time-related fields are in **GPU cycles**; rates are in
    **per-cycle** units. Cycle ↔ ns conversion uses `clock_ghz` (cycles
    per ns), applied at the top of the model only.
    """

    arch: str

    # Die structure
    num_cu: int
    num_xcd: int
    cu_per_xcd: int
    clock_ghz: float                # cycles per ns (= 2.0 for MI300X at 2 GHz)

    # Per-CU throughput ceilings
    vmem_issue_rate: float          # VMEM instructions per CU per cycle
    valu_rate: float                # VALU lane-elements per CU per cycle (64-lane SIMD)

    # TCP / vL1D (per CU)
    tcp_capacity_bytes: int         # 32 KB on all MI-series
    tcp_bw: float                   # bytes per CU per cycle

    # Outstanding request limits (latency-bandwidth product)
    mshr_depth_per_wave: int        # outstanding global loads per wavefront (~12)
    waves_per_wg: int               # wavefronts per workgroup (threads / 64)
    xgmi_latency_cycles: float      # round-trip latency for remote load (~1320 cycles)

    # L2 / TCC (per XCD, shared across CUs)
    l2_capacity_bytes: int          # per XCD
    l2_bw_per_cu: float             # bytes per CU per cycle (share of XCD L2 BW)

    # MALL / Infinity Cache (device-wide)
    mall_capacity_bytes: int
    mall_bw: float                  # bytes per cycle aggregate

    # HBM
    hbm_read_bw: float              # bytes per cycle aggregate (measured peak)
    hbm_write_bw: float             # bytes per cycle aggregate (measured peak)
    hbm_capacity_bytes: int

    # BW scaling polynomial: fraction = a*N^2 + b*N + c, clamped [0,1]
    # From Origami's mem_bw_per_wg_coefficients (microbenchmark-calibrated)
    mem_bw_coeffs: tuple = (0.0, 0.015, 0.0)

    def hbm_read_bw_per_cu(self, active_cus: int = None):
        """HBM read BW per CU **in bytes/cycle**, scaled by active CU count."""
        if active_cus is None:
            active_cus = self.num_cu
        fraction = self._bw_fraction(active_cus)
        return self.hbm_read_bw * fraction / active_cus

    def hbm_write_bw_per_cu(self, active_cus: int = None):
        """HBM write BW per CU **in bytes/cycle**, scaled by active CU count."""
        if active_cus is None:
            active_cus = self.num_cu
        fraction = self._bw_fraction(active_cus)
        return self.hbm_write_bw * fraction / active_cus

    def hbm_read_bw_per_cu_ns(self, active_cus: int = None):
        """HBM read BW per CU in bytes/ns (display helper, via frequency)."""
        return self.rate_per_ns(self.hbm_read_bw_per_cu(active_cus))

    def hbm_write_bw_per_cu_ns(self, active_cus: int = None):
        """HBM write BW per CU in bytes/ns (display helper, via frequency)."""
        return self.rate_per_ns(self.hbm_write_bw_per_cu(active_cus))

    def l2_bw_per_cu_scaled(self, active_cus_on_xcd: int = None):
        """L2 BW per CU, accounting for how many CUs share this XCD's L2."""
        if active_cus_on_xcd is None:
            active_cus_on_xcd = self.cu_per_xcd
        active_cus_on_xcd = min(active_cus_on_xcd, self.cu_per_xcd)
        return self.l2_bw_per_cu * (self.cu_per_xcd / max(active_cus_on_xcd, 1))

    def _bw_fraction(self, active_cus: int) -> float:
        """Fraction of peak HBM BW achieved with N active CUs (Origami polynomial)."""
        a, b, c = self.mem_bw_coeffs
        fraction = a * active_cus * active_cus + b * active_cus + c
        return min(max(fraction, 0.0), 1.0)

    # ── Frequency and cycle ↔ time conversion ──────────────────────
    # The model operates entirely in cycles internally. The boundary
    # between cycles and wall-clock time is governed by one quantity —
    # the GPU clock frequency — applied via the helpers below. Every
    # ns/µs/s value displayed by the dashboard or returned at the
    # public API ultimately flows through `cycles_to_seconds`.
    @property
    def clock_hz(self) -> float:
        """GPU clock frequency in Hz (cycles per second)."""
        return self.clock_ghz * 1e9

    def cycles_to_seconds(self, cycles: float) -> float:
        """Convert GPU cycles to wall-clock seconds: cycles / frequency."""
        return cycles / self.clock_hz

    def cycles_to_ns(self, cycles: float) -> float:
        return self.cycles_to_seconds(cycles) * 1e9

    def cycles_to_us(self, cycles: float) -> float:
        return self.cycles_to_seconds(cycles) * 1e6

    def seconds_to_cycles(self, seconds: float) -> float:
        """Convert wall-clock seconds to GPU cycles: seconds × frequency."""
        return seconds * self.clock_hz

    def ns_to_cycles(self, ns: float) -> float:
        return self.seconds_to_cycles(ns * 1e-9)

    def us_to_cycles(self, us: float) -> float:
        return self.seconds_to_cycles(us * 1e-6)

    def rate_per_second(self, per_cycle: float) -> float:
        """Convert a per-cycle rate to a per-second rate: rate × frequency."""
        return per_cycle * self.clock_hz

    def rate_per_ns(self, per_cycle: float) -> float:
        return self.rate_per_second(per_cycle) * 1e-9

    def rate_per_cycle_from_per_second(self, per_second: float) -> float:
        """Convert a per-second rate to a per-cycle rate: rate / frequency."""
        return per_second / self.clock_hz

    def rate_per_cycle_from_per_ns(self, per_ns: float) -> float:
        return self.rate_per_cycle_from_per_second(per_ns * 1e9)

    # ── Backward-compat display helpers (cycles → ns / per-cycle → per-ns) ──
    # The model uses cycle-based units internally; these properties convert
    # via the GPU frequency for human-facing display in the dashboard.
    # Source of truth is the `*_cycles` / per-cycle fields above.
    @property
    def xgmi_latency_ns(self) -> float:
        return self.cycles_to_ns(self.xgmi_latency_cycles)

    @property
    def hbm_read_bw_ns(self) -> float:
        """HBM read BW in bytes/ns (peak aggregate)."""
        return self.rate_per_ns(self.hbm_read_bw)

    @property
    def hbm_write_bw_ns(self) -> float:
        return self.rate_per_ns(self.hbm_write_bw)

    @property
    def l2_bw_per_cu_ns(self) -> float:
        return self.rate_per_ns(self.l2_bw_per_cu)

    @property
    def mall_bw_ns(self) -> float:
        return self.rate_per_ns(self.mall_bw)

    @property
    def tcp_bw_ns(self) -> float:
        return self.rate_per_ns(self.tcp_bw)

    @property
    def vmem_issue_rate_ns(self) -> float:
        """VMEM issue rate in instructions/ns."""
        return self.rate_per_ns(self.vmem_issue_rate)

    @property
    def valu_rate_ns(self) -> float:
        return self.rate_per_ns(self.valu_rate)


@dataclass(frozen=True)
class CommHardware:
    """Inter-GPU communication resources.

    All time-related fields are in **GPU cycles**; rates are in
    **per-cycle** units. See module docstring for the unit convention.

    `clock_ghz` is kept here for the convenience accessors (
    `*_ns` / `*_per_ns` properties) so dashboard code can render cycles
    as nanoseconds without dragging a separate `Hardware` reference
    everywhere.
    """

    # xGMI link parameters (measured, first-principles)
    link_bw: float                  # bytes per cycle per link, unidirectional
    num_peer_links: int             # links to other GPUs
    # No aggregate egress ceiling — the per-link BW is the constraint.
    # Aggregate emerges from how many links are active (concurrent_links)
    # and how the DF arbitrates (captured by wgs_per_link).

    # SDMA engines
    num_sdma_engines: int
    sdma_read_bw: float             # bytes per cycle per link
    sdma_write_bw: float            # bytes per cycle per link

    # Protocol overhead (measured)
    atomic_latency_cycles: float    # per atomic operation (signal/wait)
    launch_overhead_cycles: float   # kernel dispatch cost

    # Companion clock for the *_ns display helpers below. Defaulted to
    # MI300X (2 GHz). Override if measuring at a different clock.
    clock_ghz: float = 2.0

    # ── Frequency and cycle ↔ time conversion ──────────────────────
    @property
    def clock_hz(self) -> float:
        """GPU clock frequency in Hz (cycles per second)."""
        return self.clock_ghz * 1e9

    def cycles_to_seconds(self, cycles: float) -> float:
        return cycles / self.clock_hz

    def cycles_to_ns(self, cycles: float) -> float:
        return self.cycles_to_seconds(cycles) * 1e9

    def cycles_to_us(self, cycles: float) -> float:
        return self.cycles_to_seconds(cycles) * 1e6

    def seconds_to_cycles(self, seconds: float) -> float:
        return seconds * self.clock_hz

    def ns_to_cycles(self, ns: float) -> float:
        return self.seconds_to_cycles(ns * 1e-9)

    def us_to_cycles(self, us: float) -> float:
        return self.seconds_to_cycles(us * 1e-6)

    def rate_per_second(self, per_cycle: float) -> float:
        return per_cycle * self.clock_hz

    def rate_per_ns(self, per_cycle: float) -> float:
        return self.rate_per_second(per_cycle) * 1e-9

    def rate_per_cycle_from_per_ns(self, per_ns: float) -> float:
        return per_ns * 1e9 / self.clock_hz

    # ── Backward-compat display helpers (cycles → ns via frequency) ──
    @property
    def atomic_latency_ns(self) -> float:
        return self.cycles_to_ns(self.atomic_latency_cycles)

    @property
    def launch_overhead_ns(self) -> float:
        return self.cycles_to_ns(self.launch_overhead_cycles)

    @property
    def link_bw_ns(self) -> float:
        """xGMI link BW in bytes/ns (per link, unidirectional)."""
        return self.rate_per_ns(self.link_bw)

    @property
    def sdma_read_bw_ns(self) -> float:
        return self.rate_per_ns(self.sdma_read_bw)

    @property
    def sdma_write_bw_ns(self) -> float:
        return self.rate_per_ns(self.sdma_write_bw)


# ─── MI300X (CDNA3, gfx942) ───
# All values from wiki/pages/mi300x-architecture.md, sourced from K-series experiments.

_MI300X_CLOCK_GHZ = 2.0  # cycles per ns — used below to derive cycle units

MI300X = Hardware(
    arch="gfx942",
    num_cu=304,
    num_xcd=8,
    cu_per_xcd=38,
    clock_ghz=_MI300X_CLOCK_GHZ,

    # Per-CU throughput — source numbers are natively per-cycle.
    # VMEM: 1 dwordx16 issue per cycle on global_load_dwordx16.
    vmem_issue_rate=1.0,            # instructions per CU per cycle
    # VALU: dual-issue confirmed, 2.10 wave-instrs/cycle/CU × 64 lanes/instr
    # = 134.4 elements/cycle/CU.
    valu_rate=2.10 * 64,            # 134.4 lane-elements per CU per cycle

    # TCP / vL1D: 32 KB, streaming BW limited by ~5 wide loads before stall.
    # 64 B/cycle per CU is the architectural peak.
    tcp_capacity_bytes=32 * 1024,
    tcp_bw=64.0,                    # bytes per CU per cycle

    # Outstanding request limits (from wiki K-7197: sharp cliff at N=13)
    mshr_depth_per_wave=12,         # outstanding global_load_dwordx4 per wavefront
    waves_per_wg=10,                # 640 threads / 64 lanes = 10 wavefronts (RCCL default)
    # xGMI RTT: 660 ns @ 2 GHz = 1320 cycles
    xgmi_latency_cycles=660.0 * _MI300X_CLOCK_GHZ,

    # L2 / TCC: 4 MB per XCD, 38 CUs share.
    # 25.4 TB/s aggregate read / 304 CUs = 83.6 GB/s/CU = 83.6 B/ns at 2 GHz
    # = 41.8 B/cycle per CU.
    l2_capacity_bytes=4 * 1024 * 1024,
    l2_bw_per_cu=83.6 / _MI300X_CLOCK_GHZ,

    # MALL: 256 MB, 304 ns per hit (from measured latency ladder).
    # Aggregate matches HBM read (MALL sits on the HBM path):
    # 4.73 TB/s = 4730 B/ns = 2365 B/cycle at 2 GHz.
    mall_capacity_bytes=256 * 1024 * 1024,
    mall_bw=4730.0 / _MI300X_CLOCK_GHZ,

    # HBM: measured peak (wiki K-series).
    hbm_read_bw=4730.0 / _MI300X_CLOCK_GHZ,    # 2365 B/cycle (4.73 TB/s @ 2 GHz)
    hbm_write_bw=5140.0 / _MI300X_CLOCK_GHZ,   # 2570 B/cycle (5.14 TB/s @ 2 GHz)
    hbm_capacity_bytes=192 * 1024 * 1024 * 1024,
)

MI300X_COMM = CommHardware(
    # xGMI: 49.1 GiB/s wire rate, 1.23× wire-to-payload overhead (framing,
    # ECC, flit). Payload rate = 49.1 GiB/s / 1.23 = 42.86 GB/s per link
    # = 42.86 B/ns @ 2 GHz = 21.43 B/cycle. Measured per-link payload
    # 43-47 GB/s consistent across W=2,4,8.
    link_bw=49.1 * (1024**3) / 1e9 / 1.23 / _MI300X_CLOCK_GHZ,
    num_peer_links=7,                                       # 8-GPU fully connected

    # SDMA
    num_sdma_engines=14,
    sdma_read_bw=49.5 / _MI300X_CLOCK_GHZ,                  # bytes per cycle per link
    sdma_write_bw=23.6 / _MI300X_CLOCK_GHZ,                 # bytes per cycle per link

    # 100 ns per atomic @ 2 GHz = 200 cycles
    atomic_latency_cycles=100.0 * _MI300X_CLOCK_GHZ,
    # 45 µs kernel launch + library overhead @ 2 GHz = 90 000 cycles
    launch_overhead_cycles=45000.0 * _MI300X_CLOCK_GHZ,
)
