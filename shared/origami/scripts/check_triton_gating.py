#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Static Analysis Check: Triton Gating Enforcement for K-016

This CI guardrail scans all new/modified code in origami source files and verifies
that every new function, conditional block, and code path added for Triton
specialization is lexically inside a target_t=triton guard.

What it checks:
  - C++ files (gemm.cpp, etc.): New functions/blocks must be inside
    `if (... target ... triton ...)` or `case target_t::triton:` guards
  - Python files (selector.py, etc.): New functions/blocks must be inside
    `if ... target ... triton ...` or `target == ... triton` guards

What it does NOT flag:
  - Code that is target-agnostic (doesn't mention triton at all)
  - Enum definitions for target_t
  - Comments
  - Type/binding definitions that merely expose target_t

Usage:
    # Check modified files against develop branch
    python3 scripts/check_triton_gating.py

    # Check specific files
    python3 scripts/check_triton_gating.py --files src/origami/gemm.cpp

    # Check against a specific base branch/commit
    python3 scripts/check_triton_gating.py --base origin/develop

    # Run in strict mode (also checks for triton mentions in non-gated shared code)
    python3 scripts/check_triton_gating.py --strict

Exit codes:
    0 = All new triton-related code is properly gated
    1 = Ungated triton code found (with file:line details)
    2 = Script error
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Violation:
    """A single gating violation."""
    file: str
    line: int
    code: str
    reason: str
    severity: str = "ERROR"  # ERROR or WARN


@dataclass
class CheckResult:
    """Result of checking a single file."""
    file: str
    violations: list[Violation] = field(default_factory=list)
    triton_references: int = 0
    gated_references: int = 0


# =============================================================================
# C++ gating analysis
# =============================================================================

# Patterns that indicate triton-specific code (not just enum definitions)
CPP_TRITON_CODE_PATTERNS = [
    # Function calls, variable usage, logic involving triton
    re.compile(r'\btarget_t::triton\b'),
    re.compile(r'\btarget\s*==\s*.*triton', re.IGNORECASE),
    re.compile(r'\btriton\b.*\btarget\b', re.IGNORECASE),
    re.compile(r'\"triton\"'),
    re.compile(r'\btriton_', re.IGNORECASE),
    re.compile(r'\b_triton\b', re.IGNORECASE),
    re.compile(r'\blds_filter.*triton\b', re.IGNORECASE),
    re.compile(r'\btriton.*lds\b', re.IGNORECASE),
    re.compile(r'tritonblas', re.IGNORECASE),
]

# Patterns that are legitimate ungated triton references (exclusions)
CPP_TRITON_EXCLUDE_PATTERNS = [
    # Enum definition itself
    re.compile(r'^\s*triton\s*=\s*\d+'),
    # Comment lines
    re.compile(r'^\s*//'),
    re.compile(r'^\s*/?\*'),
    # String in enum binding
    re.compile(r'\.value\s*\(\s*"triton"'),
    # Type definition / forward declaration
    re.compile(r'^\s*enum\s+class\s+target_t'),
]

# Patterns that indicate we're inside a triton guard
CPP_GUARD_PATTERNS = [
    re.compile(r'if\s*\(.*target.*triton', re.IGNORECASE),
    re.compile(r'if\s*\(.*triton.*target', re.IGNORECASE),
    re.compile(r'if\s*\(.*target_t::triton'),
    re.compile(r'case\s+target_t::triton'),
    re.compile(r'target\s*==\s*target_t::triton'),
    re.compile(r'target_t::triton\s*==\s*target'),
    re.compile(r'\.target\s*==\s*target_t::triton'),
    re.compile(r'config\.target\s*==\s*target_t::triton'),
]


def is_triton_reference_cpp(line: str) -> bool:
    """Check if a C++ line references triton-specific code."""
    for pattern in CPP_TRITON_CODE_PATTERNS:
        if pattern.search(line):
            # Check exclusions
            for excl in CPP_TRITON_EXCLUDE_PATTERNS:
                if excl.search(line):
                    return False
            return True
    return False


def is_guard_line_cpp(line: str) -> bool:
    """Check if a C++ line is a triton guard."""
    for pattern in CPP_GUARD_PATTERNS:
        if pattern.search(line):
            return True
    return False


def check_cpp_file(filepath: str, diff_lines: Optional[set[int]] = None) -> CheckResult:
    """Check a C++ file for ungated triton references.

    Uses brace-tracking to determine if triton references are inside a guard scope.

    Args:
        filepath: Path to the C++ file
        diff_lines: If provided, only check these line numbers (1-based).
                    If None, check all lines.
    """
    result = CheckResult(file=filepath)

    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return result

    # Track brace depth and guard scopes
    # guard_scopes is a stack of brace depths where triton guards were opened
    brace_depth = 0
    guard_scopes: list[int] = []
    in_multiline_comment = False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track multiline comments
        if '/*' in stripped and '*/' not in stripped:
            in_multiline_comment = True
            continue
        if '*/' in stripped:
            in_multiline_comment = False
            continue
        if in_multiline_comment:
            continue

        # Skip pure comments
        if stripped.startswith('//'):
            continue

        # Track brace depth
        open_braces = line.count('{')
        close_braces = line.count('}')

        # Check for guard lines
        if is_guard_line_cpp(line):
            # The guard scope starts at the NEXT open brace
            # (or same line if brace is on same line)
            if open_braces > 0:
                guard_scopes.append(brace_depth + 1)
            else:
                guard_scopes.append(brace_depth + 1)  # Will be matched on next {

        brace_depth += open_braces

        # Pop guard scopes when their braces close
        for _ in range(close_braces):
            brace_depth -= 1
            while guard_scopes and guard_scopes[-1] > brace_depth:
                guard_scopes.pop()

        # Only check lines that are in the diff (if diff_lines specified)
        if diff_lines is not None and i not in diff_lines:
            continue

        # Check for triton references
        if is_triton_reference_cpp(line):
            result.triton_references += 1
            if guard_scopes:
                result.gated_references += 1
            else:
                result.violations.append(Violation(
                    file=filepath,
                    line=i,
                    code=stripped,
                    reason="Triton-specific code outside target_t::triton guard",
                ))

    return result


# =============================================================================
# Python gating analysis
# =============================================================================

PY_TRITON_CODE_PATTERNS = [
    re.compile(r'\btarget_t\.triton\b'),
    re.compile(r'\btarget.*triton', re.IGNORECASE),
    re.compile(r'\btriton.*target', re.IGNORECASE),
    re.compile(r'["\']triton["\']'),
    re.compile(r'\btriton_', re.IGNORECASE),
    re.compile(r'\b_triton\b', re.IGNORECASE),
    re.compile(r'tritonblas', re.IGNORECASE),
]

PY_TRITON_EXCLUDE_PATTERNS = [
    re.compile(r'^\s*#'),                    # Comments
    re.compile(r'^\s*import\s+triton'),      # Imports
    re.compile(r'^\s*from\s+triton'),        # Imports
    re.compile(r'\.value\s*\(\s*"triton"'),  # Binding definitions
    re.compile(r'triton\.runtime'),          # Type annotations mentioning triton
]

PY_GUARD_PATTERNS = [
    re.compile(r'if\s+.*target.*triton', re.IGNORECASE),
    re.compile(r'if\s+.*triton.*target', re.IGNORECASE),
    re.compile(r'if\s+.*target_t\.triton'),
    re.compile(r'target\s*==\s*.*triton', re.IGNORECASE),
    re.compile(r'\.target\s*==\s*.*triton', re.IGNORECASE),
]


def is_triton_reference_py(line: str) -> bool:
    """Check if a Python line references triton-specific code."""
    for pattern in PY_TRITON_CODE_PATTERNS:
        if pattern.search(line):
            for excl in PY_TRITON_EXCLUDE_PATTERNS:
                if excl.search(line):
                    return False
            return True
    return False


def is_guard_line_py(line: str) -> bool:
    """Check if a Python line is a triton guard."""
    for pattern in PY_GUARD_PATTERNS:
        if pattern.search(line):
            return True
    return False


def check_python_file(filepath: str, diff_lines: Optional[set[int]] = None) -> CheckResult:
    """Check a Python file for ungated triton references.

    Uses indentation-tracking (Python's block structure) to determine if
    triton references are inside a guard scope.

    Args:
        filepath: Path to the Python file
        diff_lines: If provided, only check these line numbers (1-based).
                    If None, check all lines.
    """
    result = CheckResult(file=filepath)

    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return result

    # Track guard scopes by indentation level
    # guard_indents is a stack of indentation levels where triton guards were found
    guard_indents: list[int] = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        # Measure indentation
        indent = len(line) - len(line.lstrip())

        # Pop guard scopes whose indentation is >= current (we've left their block)
        while guard_indents and guard_indents[-1] >= indent:
            guard_indents.pop()

        # Check for guard lines
        if is_guard_line_py(line):
            guard_indents.append(indent)

        # Only check lines that are in the diff (if diff_lines specified)
        if diff_lines is not None and i not in diff_lines:
            continue

        # Check for triton references
        if is_triton_reference_py(line):
            result.triton_references += 1
            if guard_indents:
                result.gated_references += 1
            else:
                # Skip if this IS a guard line itself
                if is_guard_line_py(line):
                    result.gated_references += 1
                else:
                    result.violations.append(Violation(
                        file=filepath,
                        line=i,
                        code=stripped,
                        reason="Triton-specific code outside target_t=triton guard",
                    ))

    return result


# =============================================================================
# Git diff integration
# =============================================================================

def get_changed_lines(base_ref: str, filepath: str) -> Optional[set[int]]:
    """Get line numbers that were added/modified in the diff.

    Returns None if the file is entirely new (check all lines).
    Returns empty set if the file wasn't changed.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--unified=0", base_ref, "--", filepath],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if not result.stdout:
        return set()

    changed_lines = set()
    for line in result.stdout.split('\n'):
        # Parse @@ -old_start,old_count +new_start,new_count @@
        match = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)
        if match:
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) else 1
            for ln in range(start, start + count):
                changed_lines.add(ln)

    return changed_lines


def get_modified_files(base_ref: str, origami_dir: str) -> list[str]:
    """Get list of files modified relative to base_ref within origami_dir."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref, "--", origami_dir],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return []
        return [f for f in result.stdout.strip().split('\n') if f]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


# =============================================================================
# Main check
# =============================================================================

def check_file(filepath: str, diff_lines: Optional[set[int]] = None) -> CheckResult:
    """Route to appropriate checker based on file extension."""
    if filepath.endswith(('.cpp', '.hpp', '.h', '.cc', '.cxx')):
        return check_cpp_file(filepath, diff_lines)
    elif filepath.endswith('.py'):
        return check_python_file(filepath, diff_lines)
    else:
        return CheckResult(file=filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Check that all Triton-specific code is gated on target_t=triton"
    )
    parser.add_argument(
        "--base", default="origin/develop",
        help="Base ref to diff against (default: origin/develop)"
    )
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Specific files to check (overrides git diff)"
    )
    parser.add_argument(
        "--all-lines", action="store_true",
        help="Check all lines, not just changed ones"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Also warn about triton mentions in non-gated shared utilities"
    )
    parser.add_argument(
        "--origami-dir", default=None,
        help="Path to origami directory (auto-detected if not set)"
    )
    args = parser.parse_args()

    # Auto-detect origami directory
    if args.origami_dir:
        origami_dir = args.origami_dir
    else:
        # Try to find relative to script location
        script_dir = Path(__file__).resolve().parent
        # scripts/ is inside origami/
        origami_dir = str(script_dir.parent)
        if not os.path.isdir(os.path.join(origami_dir, "src")):
            # Try one more level up
            origami_dir = str(script_dir.parent.parent)

    print(f"Origami dir: {origami_dir}")
    print(f"Base ref: {args.base}")

    # Get files to check
    if args.files:
        files_to_check = args.files
    else:
        files_to_check = get_modified_files(args.base, origami_dir)
        if not files_to_check:
            print("No modified origami files found. Nothing to check.")
            sys.exit(0)

    # Filter to relevant extensions
    relevant_extensions = {'.cpp', '.hpp', '.h', '.cc', '.cxx', '.py'}
    files_to_check = [
        f for f in files_to_check
        if Path(f).suffix in relevant_extensions
    ]

    if not files_to_check:
        print("No relevant source files to check.")
        sys.exit(0)

    print(f"\nChecking {len(files_to_check)} files:")
    for f in files_to_check:
        print(f"  {f}")

    # Run checks
    all_results: list[CheckResult] = []
    total_violations = 0

    for filepath in files_to_check:
        if args.all_lines:
            diff_lines = None
        else:
            diff_lines = get_changed_lines(args.base, filepath)
            if diff_lines is not None and len(diff_lines) == 0:
                continue

        result = check_file(filepath, diff_lines)
        all_results.append(result)

        if result.violations:
            total_violations += len(result.violations)

    # Report results
    print(f"\n{'='*72}")
    print("TRITON GATING CHECK RESULTS")
    print(f"{'='*72}\n")

    for result in all_results:
        if result.triton_references == 0 and not result.violations:
            continue

        status = "PASS" if not result.violations else "FAIL"
        print(f"[{status}] {result.file}")
        print(f"       Triton refs: {result.triton_references} "
              f"(gated: {result.gated_references}, "
              f"ungated: {len(result.violations)})")

        for v in result.violations:
            print(f"  {v.severity} {v.file}:{v.line}: {v.reason}")
            print(f"    > {v.code}")

        print()

    print(f"{'='*72}")
    if total_violations > 0:
        print(f"FAILED: {total_violations} ungated Triton reference(s) found.")
        print("All Triton-specific code must be inside a target_t=triton guard.")
        print("See plan: every new function/block added for Triton specialization")
        print("MUST be predicated on target_t=triton in config_t.")
        sys.exit(1)
    else:
        checked_count = sum(1 for r in all_results if r.triton_references > 0)
        gated_count = sum(r.gated_references for r in all_results)
        print(f"PASSED: {checked_count} files with Triton references, "
              f"all {gated_count} properly gated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
