"""Empirical heuristic weights — the calibration knobs.

This module is the single home for "weird constants" in the cost model:
fudge factors and empirically-tuned weights that we fit against measured
data when first-principles modeling has irreducible variance.

Distinguish from neighbors:
  - `model/hardware.py`   — measured/datasheet hardware constants
                            (clocks, BW peaks, MSHR depth, link rates).
                            These are *facts about the silicon*.
  - `model/heuristics.py`  — **this file** — empirical *fits* whose values
                            we expect to drift as measurement coverage
                            improves or new platforms land.
  - `model/types.py`        — pure model structure (TileShape, CommConfig).
                            Knobs that live there read their defaults
                            from this module so there is exactly one
                            source of truth per heuristic.

Every entry in `Heuristics` should declare:
  1. What physical/empirical effect it approximates.
  2. The calibration source (CSV file, microbenchmark, paper).
  3. Sensible alternative values so the knob is discoverable.

To override defaults for a study, instantiate a custom `Heuristics`
and pass it through the public APIs (e.g.
`predict_tensor_collective(..., heuristics=custom)`).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

# Default ring-step overheads in nanoseconds, converted to cycles via the
# MI300X clock at module import time. Kept as a module constant so the
# field default factory below is pickleable / introspectable.
_RING_STEP_OVERHEAD_NS = {
    "all_gather":     10_000.0,   # 10 µs per step (best fit after active_links fix)
    "reduce_scatter":  4_000.0,   #  4 µs per step
    "broadcast":           0.0,
    "all_reduce":          0.0,
    "all_to_all":          0.0,
}

# MI300X runs at 2.0 GHz; conversion factor cycles per ns.
# We store the cycles directly so the heuristic stays in the same unit as
# `hardware.launch_overhead_cycles` and the rest of the cycle-based model.
_MI300X_CLOCK_GHZ = 2.0


@dataclass(frozen=True)
class Heuristics:
    # ── WG-cap heuristic ────────────────────────────────────────────
    # Minimum bytes-per-WG below which a workgroup is considered "idle"
    # rather than a real consumer of a timestep's data. When the
    # per-timestep tile is small, the model treats only the fitting
    # number of WGs as active for the purposes of HBM/L2 bandwidth
    # polynomial scaling and xGMI link sharing.
    #
    # See `CommConfig.effective_num_wgs` for the cap site and
    # `model/collective.py` for the call sites.
    #
    # Calibration: rccl_master_sweep.csv (MI300X, W=8). 16 KiB ≈ NCCL
    # LL128 minimum chunk and minimizes byte-level MdAPE on the
    # 4 KiB–1 MiB regime where over-launch is most common.
    #
    # Alternatives:
    #   0       — disable the cap (pre-fix behavior, over-counts
    #             contention when channels >> work).
    #   64      — one cacheline; absolute physical floor.
    #   512     — one inner-loop iteration at vgprs_for_data=128.
    #   16_384  — NCCL LL128 minimum chunk (default).
    #   524_288 — NCCL Simple-protocol minimum chunk
    #             (NCCL_BUFFSIZE / NCCL_STEPS = 8 MiB / 16).
    min_bytes_per_wg: int = 16_384

    # ── Framework overhead floor ────────────────────────────────────
    # Constant per-call latency contributed by the host framework that
    # sits between user code and the collective backend (RCCL/NCCL).
    # Added at the Tensor Collective layer; the byte-level Collective
    # layer remains framework-neutral.
    #
    # UNIT: nanoseconds. Unlike the GPU model (which is internally in
    # cycles, converted to µs only at the API boundary), this is a
    # *host* wall-time constant that is unrelated to the GPU clock —
    # it would not scale with a different `clock_ghz`. Keeping it in
    # ns avoids conflating two clocks.
    #
    # ⚠ CALIBRATION CAVEAT (2026-06-04): the 400 µs torch value was fit
    # against `tensor_shapes_sweep.csv` (host e04u37) vs
    # `rccl_master_sweep.csv` (hosts a05u07/c09u13) — i.e., across
    # different MI300X clusters. It therefore absorbs both the true
    # framework overhead AND the inter-cluster hardware/RCCL-version
    # delta. The fit is only meaningful in the latency-bound regime
    # (per-rank ≲ 64 KiB); at large messages the actual gap grows
    # with message size, which a flat floor cannot capture. Re-fit
    # this against same-cluster data when it becomes available.
    # See wiki/log.md 2026-06-04 correction.
    #
    # Keyed by the framework string passed to
    # `predict_tensor_collective(..., framework=...)`. Add new
    # frameworks here as we measure them.
    framework_overhead_ns: Dict[str, float] = field(default_factory=lambda: {
        "raw":   0.0,            # rccl-tests, ncclbench — bare backend
        "rccl":  0.0,
        "nccl":  0.0,
        "torch": 400_000.0,      # torch.distributed (~400 µs MI300X floor)
        "jax":   0.0,            # TBD
        "mpi":   0.0,            # TBD
    })

    # ── Per-ring-step proxy/sync overhead ──────────────────────────
    # Each RCCL ring step incurs a CPU-mediated proxy thread send/recv
    # + barrier handshake. Charged per-step on top of the per-byte
    # transfer cost; pure latency-floor contribution that explains
    # the flat ~130 µs small-message gap observed for AG before this
    # heuristic was added (see wiki/pages/ag-outlier-investigation.md).
    #
    # UNIT: GPU cycles. Consistent with `hardware.launch_overhead_cycles`
    # and the bottom-up cycle convention. We store the defaults in
    # cycles (12 µs × 2 cycles/ns = 24 000 cycles for AG, etc.).
    #
    # Calibration: rccl_master_sweep.csv (MI300X, W=8). Fits min MdAPE
    # against per-primitive (nch, msg_bytes) grids. Drops AG MdAPE
    # 45.3% → 6.1% and RS 30.8% → 7.5% at W=8.
    #
    # Non-ring layouts get 0 — their per-timestep transfer model
    # already captures the relevant overheads.
    ring_step_overhead_cycles: Dict[str, float] = field(default_factory=lambda: {
        op: ns * _MI300X_CLOCK_GHZ for op, ns in _RING_STEP_OVERHEAD_NS.items()
    })

    # ── xGMI write concentration efficiency ────────────────────────
    # Per-WG xGMI write throughput scales with how many WGs share the
    # same outbound link. Empirical (rccl AG sweep, MI300X):
    #   1 WG/link → ~21% of payload rate
    #   2 WG/link → ~38%
    #   5 WG/link → ~65%
    #   9+ WG/link → ~83%
    # A clean fit: effective_link_util(wgs) = 1 - exp(-wgs / k).
    #
    # Replaces the MSHR-cap-for-writes formerly used by latency.py
    # (the MSHR cap is kept for *reads* — that's empirically right).
    #
    # Calibration: rccl_master_sweep.csv. k ≈ 3.5-5 across primitives.
    # The variation isn't pure hardware — it folds in per-collective
    # protocol choices RCCL makes (LL128 vs Simple, buffer commit
    # patterns, etc.). Per-primitive overrides via
    # `xgmi_write_concentration_k_by_primitive`; otherwise falls back
    # to `xgmi_write_concentration_k_default` (4.5, the AG/BC fit).
    xgmi_write_concentration_k_default: float = 4.0
    xgmi_write_concentration_k_by_primitive: Dict[str, float] = field(default_factory=lambda: {
        "all_gather":     4.0,
        "reduce_scatter": 3.0,
        "broadcast":      3.5,
        "all_reduce":     6.0,
        "all_to_all":     4.0,
    })

    def k_xgmi_write(self, primitive: Optional[str]) -> float:
        """Concentration constant for xGMI write; per-primitive when known."""
        if primitive is None:
            return self.xgmi_write_concentration_k_default
        return self.xgmi_write_concentration_k_by_primitive.get(
            primitive, self.xgmi_write_concentration_k_default
        )

    def framework_overhead_us(self, framework: str) -> float:
        """Per-call framework overhead in microseconds (0 if unknown)."""
        return self.framework_overhead_ns.get(framework, 0.0) / 1000.0


DEFAULT_HEURISTICS = Heuristics()
