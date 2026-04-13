# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for the Triton gating static analysis check.

Validates that the check_triton_gating.py script correctly:
1. Passes properly gated Triton code
2. Fails ungated Triton code with specific line references
3. Handles edge cases (comments, enum definitions, imports)
"""

import sys
from pathlib import Path

import pytest

# Add scripts dir to path so we can import the checker
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from check_triton_gating import (
    check_cpp_file,
    check_python_file,
    is_triton_reference_cpp,
    is_triton_reference_py,
    is_guard_line_cpp,
    is_guard_line_py,
)

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "tests" / "test_data"


class TestCppTritonDetection:
    """Test C++ triton reference detection."""

    def test_detects_target_t_triton(self):
        assert is_triton_reference_cpp("if (config.target == target_t::triton) {")

    def test_detects_triton_function(self):
        assert is_triton_reference_cpp("double triton_lds = compute_triton_lds(config);")

    def test_detects_tritonblas(self):
        assert is_triton_reference_cpp("apply_tritonblas_scoring(config);")

    def test_ignores_comments(self):
        assert not is_triton_reference_cpp("// This is a comment about triton")

    def test_ignores_enum_definition(self):
        assert not is_triton_reference_cpp("  triton = 3,  ///< Triton backend")

    def test_ignores_multiline_comment(self):
        assert not is_triton_reference_cpp("/* triton stuff */")


class TestCppGuardDetection:
    """Test C++ guard line detection."""

    def test_if_target_triton(self):
        assert is_guard_line_cpp("if (config.target == target_t::triton) {")

    def test_case_triton(self):
        assert is_guard_line_cpp("case target_t::triton:")

    def test_target_equals(self):
        assert is_guard_line_cpp("if (target == target_t::triton) {")

    def test_not_guard(self):
        assert not is_guard_line_cpp("double triton_lds = 0;")


class TestCppFileCheck:
    """Test full C++ file checking."""

    def test_gated_file_passes(self):
        filepath = str(TEST_DATA_DIR / "gated_triton_example.cpp")
        result = check_cpp_file(filepath)
        assert len(result.violations) == 0, (
            f"Properly gated file should pass, but got violations:\n"
            + "\n".join(f"  line {v.line}: {v.code}" for v in result.violations)
        )
        assert result.triton_references > 0, "Should have found triton references"
        assert result.gated_references > 0, "Should have found gated references"

    def test_ungated_file_fails(self):
        filepath = str(TEST_DATA_DIR / "ungated_triton_example.cpp")
        result = check_cpp_file(filepath)
        assert len(result.violations) > 0, (
            "Ungated file should have violations"
        )
        # Should catch the ungated triton references but not the gated ones
        violation_lines = {v.line for v in result.violations}
        # The ungated function has triton refs on specific lines
        assert result.gated_references > 0, (
            "Should have found some gated references in the properly_gated function"
        )
        # Verify violation messages have file/line info
        for v in result.violations:
            assert v.file == filepath
            assert v.line > 0
            assert len(v.code) > 0
            assert "triton" in v.reason.lower() or "Triton" in v.reason


class TestPythonTritonDetection:
    """Test Python triton reference detection."""

    def test_detects_target_t_triton(self):
        assert is_triton_reference_py("if config.target == origami.target_t.triton:")

    def test_detects_tritonblas(self):
        assert is_triton_reference_py("result = tritonblas_filter(config)")

    def test_ignores_comments(self):
        assert not is_triton_reference_py("# triton stuff here")

    def test_ignores_imports(self):
        assert not is_triton_reference_py("import triton")
        assert not is_triton_reference_py("from triton import language as tl")


class TestPythonGuardDetection:
    """Test Python guard line detection."""

    def test_if_target_triton(self):
        assert is_guard_line_py("if config.target == origami.target_t.triton:")

    def test_target_equals(self):
        assert is_guard_line_py("    if target == 'triton':")

    def test_not_guard(self):
        assert not is_guard_line_py("tritonblas_result = compute()")


class TestPythonFileCheck:
    """Test full Python file checking."""

    def test_gated_file_passes(self):
        filepath = str(TEST_DATA_DIR / "gated_triton_example.py")
        result = check_python_file(filepath)
        assert len(result.violations) == 0, (
            f"Properly gated file should pass, but got violations:\n"
            + "\n".join(f"  line {v.line}: {v.code}" for v in result.violations)
        )
        assert result.triton_references > 0

    def test_ungated_file_fails(self):
        filepath = str(TEST_DATA_DIR / "ungated_triton_example.py")
        result = check_python_file(filepath)
        assert len(result.violations) > 0, (
            "Ungated file should have violations"
        )
        assert result.gated_references > 0, (
            "Should still find gated references in properly_gated function"
        )
        for v in result.violations:
            assert v.file == filepath
            assert v.line > 0
            assert "triton" in v.reason.lower() or "Triton" in v.reason


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_file_with_no_triton(self):
        """File without any triton references should pass cleanly."""
        # Use an existing origami file that has no triton refs
        filepath = str(
            Path(__file__).parent.parent.parent / "src" / "origami" / "hardware.cpp"
        )
        result = check_cpp_file(filepath)
        assert len(result.violations) == 0
        assert result.triton_references == 0

    def test_nonexistent_file(self):
        """Nonexistent file should return empty result."""
        result = check_cpp_file("/nonexistent/file.cpp")
        assert len(result.violations) == 0
        assert result.triton_references == 0
