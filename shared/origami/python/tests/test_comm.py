# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for origami.comm — Communications Latency Model

Validates predict_comm_latency() against GPU-measured Iris all-reduce data
from Slurm jobs #17994 (ws=4) and #17995 (ws=8) on MI300X.
"""

import importlib.util
import os
import sys
import pytest

# Import comm.py directly from the local source tree, bypassing any
# editable-installed origami packages that may shadow our local version.
_comm_path = os.path.join(os.path.dirname(__file__), "..", "src", "origami", "comm.py")
_comm_path = os.path.abspath(_comm_path)
_spec = importlib.util.spec_from_file_location("origami.comm", _comm_path)
_comm = importlib.util.module_from_spec(_spec)
sys.modules["origami.comm"] = _comm
_spec.loader.exec_module(_comm)

predict_comm_latency = _comm.predict_comm_latency
predict_comm_latency_ms = _comm.predict_comm_latency_ms
predict_overlap_latency = _comm.predict_overlap_latency
predict_for_tensor = _comm.predict_for_tensor
estimate_allreduce_ms = _comm.estimate_allreduce_ms
validate = _comm.validate
list_hardware = _comm.list_hardware
list_profiles = _comm.list_profiles
get_profile = _comm.get_profile
predict_batch = _comm.predict_batch
CommProfile = _comm.CommProfile
get_optimal_cu_partition = _comm.get_optimal_cu_partition
CUPartitionResult = _comm.CUPartitionResult
dtype_to_bytes = _comm.dtype_to_bytes


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


class TestOverlapPrediction:
    """Coexecution overlap model tests."""

    def test_overlap_returns_dict(self):
        r = predict_overlap_latency(0.5, 128 * 1024**2)
        assert isinstance(r, dict)
        assert "coexec_ms" in r
        assert "speedup" in r
        assert "bottleneck" in r

    def test_overlap_max_semantics(self):
        """Coexec time = max(gemm, comm)."""
        r = predict_overlap_latency(0.5, 128 * 1024**2, world_size=8)
        assert abs(r["coexec_ms"] - max(r["gemm_ms"], r["comm_ms"])) < 1e-6

    def test_overlap_speedup_over_sequential(self):
        """Speedup > 1 when both gemm and comm are nonzero."""
        r = predict_overlap_latency(0.5, 128 * 1024**2, world_size=8)
        assert r["speedup"] > 1.0

    def test_overlap_bottleneck_label(self):
        """Bottleneck is 'comm' when comm is slower."""
        # Use a tiny gemm latency so comm dominates
        r = predict_overlap_latency(0.001, 256 * 1024**2, world_size=8)
        assert r["bottleneck"] == "comm"

    def test_overlap_with_cu_partitioning(self):
        """CU partitioning increases comm latency in overlap."""
        r_full = predict_overlap_latency(0.5, 128 * 1024**2, world_size=8)
        r_part = predict_overlap_latency(0.5, 128 * 1024**2, world_size=8, comm_cus=24)
        assert r_part["comm_ms"] > r_full["comm_ms"]


class TestConvenienceAPIs:
    """Tests for predict_for_tensor() and estimate_allreduce_ms()."""

    def test_predict_for_tensor_shape_tuple(self):
        """predict_for_tensor with shape tuple matches manual calc."""
        lat = predict_for_tensor("all_reduce", (4096, 4096), dtype_bytes=2)
        expected = predict_comm_latency("all_reduce", 4096 * 4096 * 2, world_size=8)
        assert abs(lat - expected) < 1e-6

    def test_predict_for_tensor_different_dtypes(self):
        """FP32 (4 bytes) all-reduce takes ~2x longer than BF16 (2 bytes)."""
        lat_bf16 = predict_for_tensor("all_reduce", (1024, 1024), dtype_bytes=2)
        lat_fp32 = predict_for_tensor("all_reduce", (1024, 1024), dtype_bytes=4)
        assert lat_fp32 > lat_bf16

    def test_estimate_allreduce_ms_returns_float(self):
        lat = estimate_allreduce_ms(100_000_000)
        assert isinstance(lat, float)
        assert lat > 0

    def test_estimate_allreduce_ms_consistency(self):
        """estimate_allreduce_ms matches predict_comm_latency_ms."""
        lat1 = estimate_allreduce_ms(50_000_000, dtype_bytes=2, world_size=8)
        lat2 = predict_comm_latency_ms(
            "all_reduce", 50_000_000 * 2, world_size=8, gpu="mi300x"
        )
        assert abs(lat1 - lat2) < 1e-10


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


# ── CU Partition Optimizer Tests ──────────────────────────────────────────

class TestGetOptimalCUPartition:
    """Tests for get_optimal_cu_partition() — sub-task st-3 requirement (c)."""

    def test_returns_cu_partition_result(self):
        """Function returns a CUPartitionResult dataclass."""
        result = get_optimal_cu_partition(
            (4096, 4096, 4096),
            comm_spec={"collective": "all_reduce", "message_bytes": 32 * 1024**2},
        )
        assert isinstance(result, CUPartitionResult)

    def test_cu_split_sums_to_total(self):
        """gemm_cus + comm_cus == total_cus."""
        result = get_optimal_cu_partition(
            (4096, 4096, 4096),
            comm_spec={"collective": "all_reduce", "message_bytes": 32 * 1024**2},
        )
        assert result.gemm_cus + result.comm_cus == result.total_cus

    def test_coexec_is_max_of_gemm_and_comm(self):
        """Coexecution latency = max(gemm, comm)."""
        result = get_optimal_cu_partition(
            (4096, 4096, 4096),
            comm_spec={"collective": "all_reduce", "message_bytes": 64 * 1024**2},
        )
        expected = max(result.gemm_latency_ms, result.comm_latency_ms)
        assert abs(result.coexec_latency_ms - expected) < 1e-3

    def test_speedup_greater_than_one(self):
        """Coexec should always be faster than sequential with non-trivial sizes."""
        result = get_optimal_cu_partition(
            (4096, 4096, 4096),
            comm_spec={"collective": "all_reduce", "message_bytes": 64 * 1024**2},
        )
        assert result.speedup >= 1.0, f"Speedup {result.speedup} < 1.0"

    def test_bottleneck_label_valid(self):
        """Bottleneck is either 'gemm' or 'comm'."""
        result = get_optimal_cu_partition(
            (4096, 4096, 4096),
            comm_spec={"collective": "all_reduce", "message_bytes": 32 * 1024**2},
        )
        assert result.bottleneck in ("gemm", "comm")

    def test_all_splits_populated(self):
        """all_splits contains sweep data for plotting."""
        result = get_optimal_cu_partition(
            (4096, 4096, 4096),
            comm_spec={"collective": "all_reduce", "message_bytes": 32 * 1024**2},
            cu_step=16,
        )
        assert len(result.all_splits) > 0
        for split in result.all_splits:
            assert "gemm_cus" in split
            assert "comm_cus" in split
            assert "coexec_ms" in split

    def test_list_of_gemm_shapes(self):
        """Accepts a list of GEMM shapes (e.g., MLP block)."""
        result = get_optimal_cu_partition(
            [(4096, 11008, 4096), (11008, 4096, 4096)],
            comm_spec={"collective": "all_reduce", "message_bytes": 4096 * 4096 * 2},
        )
        assert isinstance(result, CUPartitionResult)
        assert result.gemm_cus > 0
        assert result.comm_cus > 0

    def test_default_comm_spec(self):
        """Without comm_spec, defaults to all_reduce with auto message_bytes."""
        result = get_optimal_cu_partition((4096, 4096, 4096))
        assert isinstance(result, CUPartitionResult)
        assert result.total_cus == 304  # MI300X

    def test_custom_gemm_latency_fn(self):
        """Custom GEMM latency function is used when provided."""
        call_log = []

        def mock_gemm(M, N, K, cus):
            call_log.append((M, N, K, cus))
            return 0.5  # constant 0.5 ms

        result = get_optimal_cu_partition(
            (4096, 4096, 4096),
            comm_spec={"collective": "all_reduce", "message_bytes": 64 * 1024**2},
            gemm_latency_fn=mock_gemm,
        )
        assert len(call_log) > 0, "Custom fn should have been called"
        # All gemm latencies should be 0.5 ms
        for split in result.all_splits:
            assert abs(split["gemm_ms"] - 0.5) < 1e-3

    def test_empty_shapes_raises(self):
        """Empty gemm_shapes raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            get_optimal_cu_partition([])

    def test_large_comm_favors_more_comm_cus(self):
        """Very large message should allocate more CUs to comm than tiny message."""
        result_large = get_optimal_cu_partition(
            (1024, 1024, 1024),
            comm_spec={"collective": "all_reduce", "message_bytes": 256 * 1024**2},
        )
        result_small = get_optimal_cu_partition(
            (1024, 1024, 1024),
            comm_spec={"collective": "all_reduce", "message_bytes": 64 * 1024},
        )
        # With large comm, should allocate at least as many or more CUs to comm
        assert result_large.comm_cus >= result_small.comm_cus

    def test_integration_with_torch_distributed_params(self):
        """API accepts torch.distributed-compatible parameters (Goal 3)."""
        # Simulate typical torch.distributed.all_reduce scenario:
        # gradient tensor of shape (4096, 4096), BF16
        tensor_size_bytes = 4096 * 4096 * 2
        result = get_optimal_cu_partition(
            gemm_shapes=(4096, 4096, 4096),
            comm_spec={
                "collective": "all_reduce",
                "message_bytes": tensor_size_bytes,
            },
            world_size=8,
            dtype_bytes=2,  # BF16
        )
        assert result.coexec_latency_ms > 0
        assert result.speedup >= 1.0


# ── dtype_to_bytes Tests ──────────────────────────────────────────────────

class TestDtypeToBytes:
    """Tests for the dtype_to_bytes() helper — torch.distributed compatibility."""

    def test_bf16_strings(self):
        assert dtype_to_bytes("bf16") == 2
        assert dtype_to_bytes("bfloat16") == 2

    def test_fp16_strings(self):
        assert dtype_to_bytes("fp16") == 2
        assert dtype_to_bytes("float16") == 2
        assert dtype_to_bytes("half") == 2

    def test_fp32_strings(self):
        assert dtype_to_bytes("fp32") == 4
        assert dtype_to_bytes("float32") == 4
        assert dtype_to_bytes("float") == 4

    def test_fp64_strings(self):
        assert dtype_to_bytes("fp64") == 8
        assert dtype_to_bytes("float64") == 8
        assert dtype_to_bytes("double") == 8

    def test_int_types(self):
        assert dtype_to_bytes("int8") == 1
        assert dtype_to_bytes("int16") == 2
        assert dtype_to_bytes("int32") == 4
        assert dtype_to_bytes("int64") == 8

    def test_int_passthrough(self):
        """Integer input is returned as-is (bytes-per-element)."""
        assert dtype_to_bytes(2) == 2
        assert dtype_to_bytes(4) == 4

    def test_case_insensitive(self):
        assert dtype_to_bytes("BF16") == 2
        assert dtype_to_bytes("Float32") == 4

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            dtype_to_bytes("complex128")


# ── dtype Parameter in predict_comm_latency Tests ──────────────────────────

class TestDtypeParameter:
    """Tests for the dtype parameter in predict_comm_latency() — Goal 3 compliance."""

    def test_dtype_accepted_without_error(self):
        """predict_comm_latency accepts dtype as optional parameter."""
        lat = predict_comm_latency(
            "all_reduce", 16 * 1024**2, world_size=8, dtype="bf16"
        )
        assert isinstance(lat, float)
        assert lat > 0

    def test_dtype_does_not_change_result(self):
        """dtype is informational — result depends only on message_bytes."""
        lat_no_dtype = predict_comm_latency("all_reduce", 16 * 1024**2, world_size=8)
        lat_bf16 = predict_comm_latency(
            "all_reduce", 16 * 1024**2, world_size=8, dtype="bf16"
        )
        lat_fp32 = predict_comm_latency(
            "all_reduce", 16 * 1024**2, world_size=8, dtype="fp32"
        )
        assert lat_no_dtype == lat_bf16 == lat_fp32

    def test_dtype_none_default(self):
        """dtype=None is the default and works."""
        lat = predict_comm_latency("all_reduce", 1024**2, world_size=8, dtype=None)
        assert lat > 0

    def test_torch_distributed_style_call(self):
        """Full torch.distributed-compatible calling convention."""
        # Simulates: predict_comm_latency(
        #     collective_op="all_reduce",
        #     tensor_size_bytes=grad.numel() * grad.element_size(),
        #     world_size=8, dtype="bf16")
        numel = 4096 * 4096
        elem_size = dtype_to_bytes("bf16")
        tensor_size_bytes = numel * elem_size

        lat = predict_comm_latency(
            collective="all_reduce",
            message_bytes=tensor_size_bytes,
            world_size=8,
            dtype="bf16",
        )
        assert lat > 0
        # Cross-check with predict_for_tensor
        lat2 = predict_for_tensor("all_reduce", (4096, 4096), dtype_bytes=2)
        assert abs(lat - lat2) < 1e-6
