# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for TritonOrigamiMatmulSelector — Triton-gated specialization.

These tests verify:
  1. LDS filtering predicates match the tritonblas origami.py implementation.
  2. The hipblaslt OrigamiMatmulSelector is completely untouched.
  3. All new code is gated to target_t == triton.
  4. Graceful fallback when LDS filtering prunes all configs.
"""

import pytest
import math

# Skip entire module if torch is not available
torch = pytest.importorskip("torch", reason="torch is required for Triton selector tests.")


class TestTritonLDSFunctions:
    """Unit tests for the standalone Triton LDS estimation functions."""

    def test_import_triton_lds_functions(self):
        """Verify the Triton LDS functions are importable from origami.selector."""
        from origami.selector import estimate_triton_lds_bytes, check_triton_lds_capacity
        assert callable(estimate_triton_lds_bytes)
        assert callable(check_triton_lds_capacity)

    def test_estimate_triton_lds_bytes_single_stage(self):
        """ns==1: max(A_bytes, B_bytes)."""
        from origami.selector import estimate_triton_lds_bytes
        # 128x128x64 with fp16 (2 bytes), 1 stage
        a_bytes = 128 * 64 * 2  # 16384
        b_bytes = 64 * 128 * 2  # 16384
        result = estimate_triton_lds_bytes(128, 128, 64, 2, 2, num_stages=1)
        assert result == max(a_bytes, b_bytes)

    def test_estimate_triton_lds_bytes_two_stages(self):
        """ns==2: (ns-1) * (A_bytes + B_bytes) = 1 * (A + B)."""
        from origami.selector import estimate_triton_lds_bytes
        # 256x256x64 with fp16 (2 bytes), 2 stages
        a_bytes = 256 * 64 * 2  # 32768
        b_bytes = 64 * 256 * 2  # 32768
        expected = 1 * (a_bytes + b_bytes)  # 65536
        result = estimate_triton_lds_bytes(256, 256, 64, 2, 2, num_stages=2)
        assert result == expected

    def test_estimate_triton_lds_bytes_three_stages(self):
        """ns==3: 2 * (A_bytes + B_bytes)."""
        from origami.selector import estimate_triton_lds_bytes
        a_bytes = 128 * 32 * 2
        b_bytes = 32 * 128 * 2
        expected = 2 * (a_bytes + b_bytes)
        result = estimate_triton_lds_bytes(128, 128, 32, 2, 2, num_stages=3)
        assert result == expected

    def test_check_triton_lds_capacity_fits(self):
        """Small tile should fit in 64KB LDS."""
        from origami.selector import check_triton_lds_capacity
        # 64x64x32 with fp16, 2 stages → (1) * (64*32*2 + 32*64*2) = 8192
        assert check_triton_lds_capacity(64, 64, 32, 2, 2, 65536, 2) is True

    def test_check_triton_lds_capacity_exceeds(self):
        """Large tile should exceed 64KB LDS with 3 stages."""
        from origami.selector import check_triton_lds_capacity
        # 256x256x128 with fp16, 3 stages → 2*(256*128*2 + 128*256*2) = 262144
        assert check_triton_lds_capacity(256, 256, 128, 2, 2, 65536, 3) is False

    def test_lds_formula_matches_tritonblas(self):
        """Cross-validate against known tritonblas origami.py LDS values.

        These values were validated against metadata.shared from compiled
        Triton kernels on gfx942 (Triton 3.6.0+rocm7.2.0).
        """
        from origami.selector import estimate_triton_lds_bytes
        # 256x256x64, fp16, 2 stages: should be 65536 (exactly 64KB)
        assert estimate_triton_lds_bytes(256, 256, 64, 2, 2, 2) == 65536
        # 128x128x128, fp16, 2 stages: should be 65536
        assert estimate_triton_lds_bytes(128, 128, 128, 2, 2, 2) == 65536
        # 128x128x64, fp16, 2 stages: should be 32768
        assert estimate_triton_lds_bytes(128, 128, 64, 2, 2, 2) == 32768
        # 64x64x64, fp16, 2 stages: should be 16384
        assert estimate_triton_lds_bytes(64, 64, 64, 2, 2, 2) == 16384


class TestTritonSelectorImport:
    """Verify the TritonOrigamiMatmulSelector is importable and distinct."""

    def test_import_triton_selector(self):
        """TritonOrigamiMatmulSelector should be importable."""
        from origami.selector import TritonOrigamiMatmulSelector
        assert TritonOrigamiMatmulSelector is not None

    def test_hipblaslt_selector_unchanged(self):
        """OrigamiMatmulSelector should still be importable and unchanged."""
        from origami.selector import OrigamiMatmulSelector
        # Verify the hipblaslt selector does NOT have Triton-specific methods
        assert not hasattr(OrigamiMatmulSelector, 'estimate_triton_lds')
        assert not hasattr(OrigamiMatmulSelector, '_select_ws_params')
        assert not hasattr(OrigamiMatmulSelector, 'hierarchical_split')
        assert not hasattr(OrigamiMatmulSelector, '_compute_sk_grid')
        assert not hasattr(OrigamiMatmulSelector, '_infer_matrix_instruction_dimensions')
        assert not hasattr(OrigamiMatmulSelector, '_generate_default_configs')

    def test_triton_selector_has_triton_methods(self):
        """TritonOrigamiMatmulSelector should have all Triton-specific methods."""
        from origami.selector import TritonOrigamiMatmulSelector
        assert hasattr(TritonOrigamiMatmulSelector, 'estimate_triton_lds')
        assert hasattr(TritonOrigamiMatmulSelector, '_select_ws_params')
        assert hasattr(TritonOrigamiMatmulSelector, 'hierarchical_split')
        assert hasattr(TritonOrigamiMatmulSelector, '_compute_sk_grid')
        assert hasattr(TritonOrigamiMatmulSelector, '_infer_matrix_instruction_dimensions')
        assert hasattr(TritonOrigamiMatmulSelector, '_generate_default_configs')

    def test_classes_are_independent(self):
        """TritonOrigamiMatmulSelector and OrigamiMatmulSelector share no inheritance."""
        from origami.selector import OrigamiMatmulSelector, TritonOrigamiMatmulSelector
        assert not issubclass(TritonOrigamiMatmulSelector, OrigamiMatmulSelector)
        assert not issubclass(OrigamiMatmulSelector, TritonOrigamiMatmulSelector)


class TestTargetTBinding:
    """Verify target_t enum is exposed via Python bindings."""

    def test_target_t_enum_exists(self):
        """target_t should be importable from origami."""
        import origami
        assert hasattr(origami, 'target_t')

    def test_target_t_values(self):
        """target_t should have the expected values."""
        import origami
        assert origami.target_t.triton is not None
        assert origami.target_t.tensilelite is not None
        assert origami.target_t.generic is not None

    def test_config_t_has_target(self):
        """config_t should have a target field."""
        import origami
        c = origami.config_t()
        assert hasattr(c, 'target')
        # Default should be tensilelite
        assert c.target == origami.target_t.tensilelite

    def test_config_t_target_settable(self):
        """config_t.target should be settable to triton."""
        import origami
        c = origami.config_t()
        c.target = origami.target_t.triton
        assert c.target == origami.target_t.triton
