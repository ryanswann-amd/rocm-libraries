# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for Triton-specific C++ bindings in origami."""

import pytest
import origami

from helpers import HARDWARE


class TestTargetT:
    """Tests for target_t enum and config_t.target field."""

    def test_target_enum_values_exist(self):
        assert origami.target_t.generic is not None
        assert origami.target_t.tensilelite is not None
        assert origami.target_t.rocroller is not None
        assert origami.target_t.triton is not None
        assert origami.target_t.composable_kernel is not None

    def test_config_default_target(self):
        config = origami.config_t()
        assert config.target == origami.target_t.tensilelite

    def test_config_target_roundtrip(self):
        config = origami.config_t()
        config.target = origami.target_t.triton
        assert config.target == origami.target_t.triton


class TestTritonLDS:
    """Tests for Triton LDS estimation functions.

    Uses the exact Python padded-encoding logic from tritonBLAS as the reference.
    """

    @staticmethod
    def _padded_size_pow2(unpadded, interval, padding):
        log2_i = (interval - 1).bit_length()
        log2_p = (padding - 1).bit_length() if padding else 0
        bp = (unpadded >> log2_i) << log2_p
        if unpadded % interval == 0 and bp >= padding:
            bp -= padding
        return unpadded + bp

    @classmethod
    def _python_estimate(cls, bm, bn, bk, bytes_a, bytes_b, num_stages=2):
        """Exact Python reference for estimate_triton_lds_bytes."""
        elem_a, elem_b = bm * bk, bk * bn
        pa = cls._padded_size_pow2(elem_a, 32, 4)
        pb = cls._padded_size_pow2(elem_b, 32, 4)
        if bk > 0 and (bk & (bk - 1)) == 0:
            alt = cls._padded_size_pow2(elem_a, bk, 8)
            if alt > pa:
                pa = alt
        if bn > 0 and (bn & (bn - 1)) == 0:
            alt = cls._padded_size_pow2(elem_b, bn, 8)
            if alt > pb:
                pb = alt
        return num_stages * int(pa * bytes_a + pb * bytes_b)

    @pytest.fixture
    def hw(self):
        return HARDWARE["gfx942"]

    def test_estimate_triton_lds_bytes_2stage(self):
        mt = origami.dim3_t(128, 128, 32)
        result = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 2)
        assert result == self._python_estimate(128, 128, 32, 2, 2, 2)

    def test_estimate_triton_lds_bytes_1stage(self):
        mt = origami.dim3_t(128, 128, 32)
        result = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 1)
        assert result == self._python_estimate(128, 128, 32, 2, 2, 1)

    def test_estimate_triton_lds_bytes_3stage(self):
        mt = origami.dim3_t(128, 128, 32)
        result = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 3)
        assert result == self._python_estimate(128, 128, 32, 2, 2, 3)

    def test_estimate_triton_lds_bytes_default_stages(self):
        mt = origami.dim3_t(256, 256, 64)
        result_default = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half)
        result_explicit = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 2)
        assert result_default == result_explicit

    def test_check_triton_lds_capacity_fits(self, hw):
        mt = origami.dim3_t(64, 64, 32)
        assert origami.check_triton_lds_capacity(hw, mt, origami.data_type_t.Half, origami.data_type_t.Half)

    def test_check_triton_lds_capacity_too_large(self, hw):
        mt = origami.dim3_t(512, 512, 128)
        assert not origami.check_triton_lds_capacity(hw, mt, origami.data_type_t.Half, origami.data_type_t.Half)

    def test_triton_lds_vs_standard_lds(self, hw):
        mt = origami.dim3_t(128, 128, 32)
        triton_2 = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 2)
        triton_3 = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 3)
        assert triton_3 > triton_2
        # Padded encoding should be larger than raw tile size
        raw = (128 * 32 * 2 + 32 * 128 * 2) * 2
        assert triton_2 > raw

    def test_estimate_matches_python_sweep(self):
        """Sweep a range of tile sizes and verify C++ matches Python exactly."""
        for bm in [16, 32, 64, 128, 256]:
            for bn in [16, 32, 64, 128, 256]:
                for bk in [16, 32, 64, 128, 256, 512]:
                    for ns in [1, 2, 3]:
                        mt = origami.dim3_t(bm, bn, bk)
                        cpp = origami.estimate_triton_lds_bytes(
                            mt, origami.data_type_t.Half, origami.data_type_t.Half, ns
                        )
                        py = self._python_estimate(bm, bn, bk, 2, 2, ns)
                        assert cpp == py, (
                            f"Mismatch at {bm}x{bn}x{bk} stages={ns}: C++={cpp} Python={py}"
                        )


class TestTritonWSParams:
    """Tests for Triton work-stealing parameter selection."""

    def test_few_tiles(self):
        result = origami.select_triton_ws_params(256, 256, 128, 128)
        assert result.counters_per_xcd == 1
        assert result.workgroup_mapping > 0

    def test_many_tiles(self):
        result = origami.select_triton_ws_params(16384, 16384, 128, 128)
        assert result.counters_per_xcd > 1
        assert result.workgroup_mapping > 0

    def test_struct_fields(self):
        result = origami.select_triton_ws_params(4096, 4096, 128, 128)
        assert hasattr(result, "counters_per_xcd")
        assert hasattr(result, "workgroup_mapping")


class TestTritonHierarchicalSplit:
    """Tests for Triton hierarchical split computation."""

    def test_basic_split(self):
        result = origami.compute_triton_hierarchical_split(4096, 4096, 128, 128, 8, 304, 38)
        assert result.local_per_xcd > 0
        assert result.local_per_xcd + result.global_tiles > 0

    def test_struct_fields(self):
        result = origami.compute_triton_hierarchical_split(2048, 2048, 128, 128, 8, 304, 38)
        assert hasattr(result, "local_per_xcd")
        assert hasattr(result, "global_tiles")


class TestTritonSKGrid:
    """Tests for Triton StreamK grid computation."""

    def test_data_parallel_case(self):
        # Enough tiles to cover all CUs without StreamK
        result = origami.compute_triton_sk_grid(16384, 16384, 4096, 128, 128, 64, 304, 16)
        assert result > 0

    def test_small_problem(self):
        result = origami.compute_triton_sk_grid(256, 256, 256, 128, 128, 64, 304, 16)
        assert result > 0
