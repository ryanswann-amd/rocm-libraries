"""Tests for the heuristic-weights module and how it wires through callers."""

import pytest

from model.heuristics import DEFAULT_HEURISTICS, Heuristics
from model.tensor_collective import predict_tensor_collective
from model.types import CommConfig


# ────────────────────────────────────────────────────────────────────
# Heuristics dataclass shape
# ────────────────────────────────────────────────────────────────────


def test_default_heuristics_values():
    """DEFAULT_HEURISTICS is the single source of truth — pin its values."""
    assert DEFAULT_HEURISTICS.min_bytes_per_wg == 16_384
    # All currently-known frameworks present.
    for key in ("raw", "rccl", "nccl", "torch", "jax", "mpi"):
        assert key in DEFAULT_HEURISTICS.framework_overhead_ns
    # Torch is the only non-zero default.
    assert DEFAULT_HEURISTICS.framework_overhead_ns["torch"] == 400_000.0
    assert DEFAULT_HEURISTICS.framework_overhead_ns["raw"] == 0.0


# ────────────────────────────────────────────────────────────────────
# Per-ring-step overhead heuristic
# ────────────────────────────────────────────────────────────────────


def test_ring_step_overhead_defaults():
    """AG and RS are the only ring primitives with a non-zero default."""
    h = DEFAULT_HEURISTICS
    # AG / RS calibrated to non-zero per-step floors (see ag-outlier-investigation).
    assert h.ring_step_overhead_cycles["all_gather"] > 0
    assert h.ring_step_overhead_cycles["reduce_scatter"] > 0
    # Non-ring layouts get 0.
    assert h.ring_step_overhead_cycles["broadcast"] == 0
    assert h.ring_step_overhead_cycles["all_reduce"] == 0
    assert h.ring_step_overhead_cycles["all_to_all"] == 0
    # AG should be larger than RS (measured: ~10 µs vs ~4 µs per step).
    assert h.ring_step_overhead_cycles["all_gather"] > h.ring_step_overhead_cycles["reduce_scatter"]


def test_ring_step_overhead_affects_ag_prediction():
    """Halving the AG per-step overhead must shrink AG's predicted latency."""
    from model.collective import predict_row
    from model.hardware import MI300X, MI300X_COMM
    base = predict_row("all_gather", 1024, 8, 8, MI300X, MI300X_COMM)
    halved = Heuristics(
        ring_step_overhead_cycles={
            "all_gather": DEFAULT_HEURISTICS.ring_step_overhead_cycles["all_gather"] / 2,
            "reduce_scatter": DEFAULT_HEURISTICS.ring_step_overhead_cycles["reduce_scatter"],
            "broadcast": 0, "all_reduce": 0, "all_to_all": 0,
        },
    )
    smaller = predict_row("all_gather", 1024, 8, 8, MI300X, MI300X_COMM, heuristics=halved)
    assert smaller < base
    # Difference should be ~ (N-1) × per_step_savings; with W=8 and ~5 µs per step
    # savings, expect roughly 35 µs (in µs). Allow a generous range to keep this
    # robust to other heuristic tweaks.
    assert 20 < base - smaller < 60


# ────────────────────────────────────────────────────────────────────
# xGMI write concentration heuristic
# ────────────────────────────────────────────────────────────────────


def test_xgmi_write_k_defaults_present():
    """All 5 RCCL primitives have a calibrated per-collective k."""
    h = DEFAULT_HEURISTICS
    for op in ("all_gather", "reduce_scatter", "broadcast", "all_reduce", "all_to_all"):
        assert op in h.xgmi_write_concentration_k_by_primitive
    # All k values within the empirical range (3-6).
    for op, k in h.xgmi_write_concentration_k_by_primitive.items():
        assert 2.5 <= k <= 6.5, f"k for {op} ({k}) outside calibrated range"


def test_k_xgmi_write_helper():
    """k_xgmi_write() returns per-primitive k when known, default otherwise."""
    h = DEFAULT_HEURISTICS
    assert h.k_xgmi_write("all_gather") == h.xgmi_write_concentration_k_by_primitive["all_gather"]
    # Unknown primitive falls back to default.
    assert h.k_xgmi_write("made-up-op") == h.xgmi_write_concentration_k_default
    # None also returns default.
    assert h.k_xgmi_write(None) == h.xgmi_write_concentration_k_default


def test_xgmi_write_k_monotonic_in_predicted_time():
    """k governs link-utilization ramp: util(W) = 1 - exp(-W/k).

    Larger k → slower saturation → lower per-WG effective rate →
    longer predicted transfer. Verify monotonicity on a large-message AG
    prediction (which is xGMI-write bound).
    """
    from model.collective import predict_row
    from model.hardware import MI300X, MI300X_COMM
    msg = 256 * 1024 * 1024
    times = []
    for k in (2.0, 4.0, 6.0, 8.0):
        h = Heuristics(
            xgmi_write_concentration_k_default=k,
            xgmi_write_concentration_k_by_primitive={
                "all_gather":     k,
                "reduce_scatter": 3.0,
                "broadcast":      3.5,
                "all_reduce":     6.0,
                "all_to_all":     4.0,
            },
        )
        times.append(predict_row("all_gather", msg, 8, 8, MI300X, MI300X_COMM, heuristics=h))
    # Strictly monotonic increase in k → strictly increasing predicted time.
    assert all(times[i] < times[i+1] for i in range(len(times)-1)), times


def test_framework_overhead_us_conversion():
    h = DEFAULT_HEURISTICS
    assert h.framework_overhead_us("torch") == 400.0
    assert h.framework_overhead_us("raw") == 0.0
    # Unknown frameworks return 0 (don't break old callers).
    assert h.framework_overhead_us("does-not-exist") == 0.0


def test_heuristics_is_frozen():
    """Heuristics is a frozen dataclass — must not be mutated in-place."""
    with pytest.raises(Exception):
        DEFAULT_HEURISTICS.min_bytes_per_wg = 0  # type: ignore[misc]


# ────────────────────────────────────────────────────────────────────
# CommConfig sources its default from Heuristics
# ────────────────────────────────────────────────────────────────────


def test_commconfig_min_bytes_default_from_heuristics():
    """CommConfig's default must equal DEFAULT_HEURISTICS.min_bytes_per_wg."""
    c = CommConfig(num_wgs=32)
    assert c.min_bytes_per_wg == DEFAULT_HEURISTICS.min_bytes_per_wg


def test_commconfig_min_bytes_explicit_override():
    """Explicitly passing a value still overrides the heuristic default."""
    c = CommConfig(num_wgs=32, min_bytes_per_wg=512)
    assert c.min_bytes_per_wg == 512
    c = CommConfig(num_wgs=32, min_bytes_per_wg=0)
    assert c.min_bytes_per_wg == 0


# ────────────────────────────────────────────────────────────────────
# Framework overhead flows through predict_tensor_collective
# ────────────────────────────────────────────────────────────────────


def test_predict_tensor_collective_default_is_raw():
    """Default framework='raw' adds zero overhead."""
    p = predict_tensor_collective(
        op="all_reduce", input_shape=(1024,), dtype="bf16",
        world_size=8, nchannels=16,
    )
    assert p.framework == "raw"
    assert p.framework_overhead_us == 0.0
    assert p.predicted_us == p.backend_us


def test_predict_tensor_collective_torch_adds_400us():
    """framework='torch' adds the calibrated 400 µs floor."""
    p_raw = predict_tensor_collective(
        op="all_reduce", input_shape=(1024,), dtype="bf16",
        world_size=8, nchannels=16, framework="raw",
    )
    p_torch = predict_tensor_collective(
        op="all_reduce", input_shape=(1024,), dtype="bf16",
        world_size=8, nchannels=16, framework="torch",
    )
    # Backend is identical across frameworks — overhead is purely additive.
    assert p_torch.backend_us == pytest.approx(p_raw.backend_us)
    # The visible prediction shifts by exactly the floor.
    assert p_torch.predicted_us - p_raw.predicted_us == pytest.approx(400.0)
    assert p_torch.framework_overhead_us == 400.0
    assert p_torch.framework == "torch"


def test_predict_tensor_collective_unknown_framework_zero_overhead():
    """Unknown frameworks add 0 µs (back-compat with old callers)."""
    p = predict_tensor_collective(
        op="all_gather", input_shape=(2048,), dtype="bf16",
        world_size=8, nchannels=16, framework="not-a-real-framework",
    )
    assert p.framework_overhead_us == 0.0
    assert p.predicted_us == p.backend_us


def test_predict_tensor_collective_custom_heuristics():
    """Custom Heuristics instances override the default floor."""
    custom = Heuristics(framework_overhead_ns={"torch": 1_000_000.0})
    p = predict_tensor_collective(
        op="all_reduce", input_shape=(1024,), dtype="bf16",
        world_size=8, nchannels=16, framework="torch", heuristics=custom,
    )
    assert p.framework_overhead_us == 1000.0
    p_default = predict_tensor_collective(
        op="all_reduce", input_shape=(1024,), dtype="bf16",
        world_size=8, nchannels=16, framework="torch",
    )
    # Custom heuristic adds 1000 µs instead of the default 400 µs:
    # so it should be 600 µs higher than the default-floor prediction.
    assert p.predicted_us - p_default.predicted_us == pytest.approx(600.0)


def test_predict_tensor_collective_world_size_1_still_pays_floor():
    """World size 1 is a no-op backend cost but still pays framework dispatch."""
    p = predict_tensor_collective(
        op="all_reduce", input_shape=(1024,), dtype="bf16",
        world_size=1, nchannels=16, framework="torch",
    )
    assert p.backend_us == 0.0
    assert p.predicted_us == 400.0
    assert p.framework_overhead_us == 400.0
