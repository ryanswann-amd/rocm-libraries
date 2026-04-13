# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for origami.comm — Communications Latency Model

Validates predict_comm_latency() against GPU-measured Iris all-reduce data
from Slurm jobs #17994 (ws=4) and #17995 (ws=8) on MI300X.
"""

import os
import sys
import types
import pytest

# Ensure the local origami package (with comm.py) is found before
# the system-installed one (which may lack comm and requires ROCm C++ bindings).
_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_src_dir))

# Remove any pre-loaded origami from the system site-packages
for key in list(sys.modules):
    if key == "origami" or key.startswith("origami."):
        del sys.modules[key]

# Create a lightweight origami package that skips C++ bindings
_pkg = types.ModuleType("origami")
_pkg.__path__ = [os.path.join(os.path.abspath(_src_dir), "origami")]
_pkg.__package__ = "origami"
sys.modules["origami"] = _pkg

from origami.comm import (
    predict_comm_latency,
    predict_comm_latency_ms,
    validate,
    list_hardware,
    list_profiles,
    get_profile,
    predict_batch,
    CommProfile,
    select_partition,
    compute_overlap_speedup,
)


class TestPredictCommLatency:
    """Core prediction API tests."""

    def test_all_reduce_ws8_returns_microseconds(self):
        """Prediction returns float in microseconds."""
        lat = predict_comm_latency("all_reduce", 16 * 1024**2, world_size=8)
        assert isinstance(lat, float)
        assert lat > 0

    def test_all_reduce_ws4(self):
        """ws=4 prediction works."""
        lat = predict_comm_latency("all_reduce", 16 * 1024**2, world_size=4)
        assert lat > 0

    def test_ms_wrapper(self):
        """predict_comm_latency_ms returns ms (1000x smaller than us)."""
        lat_us = predict_comm_latency("all_reduce", 1024**2, world_size=8)
        lat_ms = predict_comm_latency_ms("all_reduce", 1024**2, world_size=8)
        assert abs(lat_us / 1e3 - lat_ms) < 1e-10

    def test_cu_partitioning_increases_latency(self):
        """Fewer CUs → higher latency (less BW)."""
        lat_full = predict_comm_latency("all_reduce", 64 * 1024**2, world_size=8)
        lat_part = predict_comm_latency(
            "all_reduce", 64 * 1024**2, world_size=8, comm_cus=32
        )
        assert lat_part > lat_full

    def test_above_saturation_cus_same_as_full(self):
        """comm_cus >= min_comm_cus should give same result as None."""
        lat_full = predict_comm_latency("all_reduce", 64 * 1024**2, world_size=8)
        lat_sat = predict_comm_latency(
            "all_reduce", 64 * 1024**2, world_size=8, comm_cus=64
        )
        assert abs(lat_full - lat_sat) < 1e-6

    def test_all_gather_less_than_all_reduce(self):
        """all_gather moves half the data of all_reduce."""
        lat_ar = predict_comm_latency("all_reduce", 64 * 1024**2, world_size=8)
        lat_ag = predict_comm_latency("all_gather", 64 * 1024**2, world_size=8)
        assert lat_ag < lat_ar

    def test_reduce_scatter_same_as_all_gather(self):
        """reduce_scatter and all_gather have the same scale factor."""
        lat_ag = predict_comm_latency("all_gather", 64 * 1024**2, world_size=8)
        lat_rs = predict_comm_latency("reduce_scatter", 64 * 1024**2, world_size=8)
        assert abs(lat_ag - lat_rs) < 1e-6

    def test_unknown_collective_raises(self):
        with pytest.raises(ValueError, match="Unknown collective"):
            predict_comm_latency("scatter", 1024, world_size=8)

    def test_unknown_world_size_raises(self):
        with pytest.raises(ValueError, match="No calibrated"):
            predict_comm_latency("all_reduce", 1024, world_size=16)

    def test_zero_cus_raises(self):
        with pytest.raises(ValueError, match="comm_cus must be > 0"):
            predict_comm_latency("all_reduce", 1024, world_size=8, comm_cus=0)

    def test_latency_increases_with_message_size(self):
        """Larger messages → higher latency."""
        lat_small = predict_comm_latency("all_reduce", 1024, world_size=8)
        lat_large = predict_comm_latency("all_reduce", 256 * 1024**2, world_size=8)
        assert lat_large > lat_small


class TestValidation:
    """Validation against GPU-measured reference data."""

    def test_ws8_mape_under_10pct(self):
        """MAPE < 10% for ws=8 (Goal 1: <10% average error)."""
        v = validate("mi300x", 8)
        assert v["mape_pct"] < 10.0, f"ws=8 MAPE {v['mape_pct']}% >= 10%"

    def test_ws4_mape_under_10pct(self):
        """MAPE < 10% for ws=4 (Goal 1: <10% average error)."""
        v = validate("mi300x", 4)
        assert v["mape_pct"] < 10.0, f"ws=4 MAPE {v['mape_pct']}% >= 10%"

    def test_ws8_all_points_under_10pct(self):
        """Every individual point < 10% error for ws=8."""
        v = validate("mi300x", 8)
        for pt in v["points"]:
            assert pt["error_pct"] < 10.0, (
                f"ws=8 {pt['bytes']}B: {pt['error_pct']}% error"
            )

    def test_ws4_all_points_under_10pct(self):
        """Every individual point < 10% error for ws=4."""
        v = validate("mi300x", 4)
        for pt in v["points"]:
            assert pt["error_pct"] < 10.0, (
                f"ws=4 {pt['bytes']}B: {pt['error_pct']}% error"
            )

    def test_validation_returns_correct_structure(self):
        v = validate("mi300x", 8)
        assert "mape_pct" in v
        assert "max_error_pct" in v
        assert "n_points" in v
        assert "points" in v
        assert v["n_points"] == 8

    def test_invalid_validation_target(self):
        v = validate("mi300x", 16)
        assert "error" in v


class TestProfiles:
    """Profile listing and access."""

    def test_list_hardware(self):
        hw = list_hardware()
        assert "mi300x" in hw

    def test_list_profiles(self):
        profs = list_profiles()
        assert ("mi300x", 8) in profs
        assert ("mi300x", 4) in profs

    def test_get_profile(self):
        p = get_profile("mi300x", 8)
        assert isinstance(p, CommProfile)
        assert p.startup_ms == 0.2822
        assert p.effective_bw_gbps == 212.31
        assert p.min_comm_cus == 48

    def test_profile_is_frozen(self):
        p = get_profile("mi300x", 8)
        with pytest.raises(AttributeError):
            p.startup_ms = 999.0


class TestBatchPrediction:
    """Batch prediction API."""

    def test_batch_returns_list(self):
        batch = predict_batch(
            "all_reduce", [1024, 1024**2, 64 * 1024**2], world_size=8
        )
        assert len(batch) == 3
        for entry in batch:
            assert "message_bytes" in entry
            assert "predicted_us" in entry
            assert "predicted_ms" in entry

    def test_batch_ordering(self):
        batch = predict_batch(
            "all_reduce", [1024, 64 * 1024**2], world_size=8
        )
        assert batch[0]["predicted_us"] < batch[1]["predicted_us"]


# ── Five Reference Test Cases ──────────────────────────────────────────
# Required by sub-task spec: "at least 5 test cases matching verified data"

class TestVerifiedReferencePoints:
    """Validate against 5+ specific GPU-measured data points (Slurm #17995)."""

    @pytest.mark.parametrize("actual_bytes,measured_ms", [
        (65536, 0.2785),        # 64 KB
        (991232, 0.2867),       # ~1 MB
        (16588800, 0.3540),     # ~16 MB
        (67092480, 0.6010),     # ~64 MB
        (268378112, 1.5889),    # ~256 MB
    ])
    def test_ws8_reference_point(self, actual_bytes, measured_ms):
        """Each prediction within 10% of GPU-measured value [VERIFIED, Slurm #17995]."""
        predicted_ms = predict_comm_latency_ms(
            "all_reduce", actual_bytes, world_size=8, gpu="mi300x"
        )
        error_pct = abs(predicted_ms - measured_ms) / measured_ms * 100
        assert error_pct < 10.0, (
            f"{actual_bytes}B: predicted={predicted_ms:.4f}ms vs "
            f"measured={measured_ms:.4f}ms (error={error_pct:.1f}%)"
        )


# ── CU Partition Selection Tests ──────────────────────────────────────

class TestSelectPartition:
    """Tests for select_partition() CU allocation."""

    def test_returns_tuple(self):
        result = select_partition(4096, 4096, 4096, world_size=8)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cus_sum_to_total(self):
        gemm_cus, comm_cus = select_partition(4096, 4096, 4096)
        assert gemm_cus + comm_cus == 304

    def test_respects_xcd_boundaries(self):
        """CU allocations should be multiples of 38 (304/8 XCDs)."""
        gemm_cus, comm_cus = select_partition(4096, 4096, 4096)
        assert comm_cus % 38 == 0
        assert gemm_cus % 38 == 0

    def test_prefers_gemm_heavy_split(self):
        """GEMM should get >= 50% of CUs."""
        gemm_cus, comm_cus = select_partition(4096, 4096, 4096)
        assert gemm_cus >= comm_cus

    def test_different_world_sizes(self):
        """ws=4 should also produce valid partitions."""
        gemm_cus, comm_cus = select_partition(4096, 4096, 4096, world_size=4)
        assert gemm_cus + comm_cus == 304
        assert comm_cus > 0


class TestComputeOverlapSpeedup:
    """Tests for compute_overlap_speedup() analysis."""

    def test_returns_dict_with_expected_keys(self):
        result = compute_overlap_speedup(4096, 4096, 4096, 256, 48)
        expected_keys = {"t_sequential_ms", "t_overlap_ms", "speedup",
                         "gemm_ms", "comm_ms"}
        assert set(result.keys()) == expected_keys

    def test_speedup_greater_than_one(self):
        """Overlapped execution should be faster than sequential."""
        result = compute_overlap_speedup(4096, 4096, 4096, 256, 48)
        assert result["speedup"] > 1.0

    def test_overlap_time_less_than_sequential(self):
        result = compute_overlap_speedup(4096, 4096, 4096, 256, 48)
        assert result["t_overlap_ms"] < result["t_sequential_ms"]

    def test_overlap_is_max_of_gemm_and_comm(self):
        result = compute_overlap_speedup(4096, 4096, 4096, 256, 48)
        assert abs(result["t_overlap_ms"] -
                   max(result["gemm_ms"], result["comm_ms"])) < 1e-6
