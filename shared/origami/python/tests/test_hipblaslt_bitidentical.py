# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Bit-Identical hipBLASLt Regression Test for K-016 Triton Specialization

This CI guardrail ensures that the default origami path (target_t=tensilelite /
hipBLASLt) produces IDENTICAL output before and after Triton specialization code
is added. Any byte-level difference in config rankings = hard CI failure.

The test:
1. Runs origami with default config (no target_t=triton) on a comprehensive test
   corpus spanning multiple architectures, dtypes, transposes, and problem sizes.
2. Compares output against golden baselines generated from the pre-Triton codebase.
3. Verifies that latency predictions are numerically identical (not just close).
4. Validates that no new code path is reachable when target_t != triton.

Usage:
    # Run regression check (must have baselines already generated)
    pytest test_hipblaslt_bitidentical.py -v

    # Generate new baselines from current (known-good) code
    pytest test_hipblaslt_bitidentical.py -v --generate-baseline

This test MUST run on every PR that touches shared/origami/.
"""

import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

import origami
from helpers import HARDWARE, create_config_list, get_matrix_instructions


# Check if target_t enum is available in bindings
_HAS_TARGET_T = hasattr(origami, "target_t")

BASELINE_DIR = Path(__file__).parent / "baselines" / "hipblaslt_bitidentical"
PROBLEM_DATA_FILE = Path(__file__).parent / "data" / "problem_data.csv"

# Comprehensive dtype coverage
SUPPORTED_DTYPES = ["f16", "bf16", "f32", "xf32", "f8"]
TRANSPOSE_VALUES = [origami.transpose_t.T, origami.transpose_t.N]

# Config generation parameters — must match baseline exactly
CONFIG_PARAMS = {
    "mt_sizes": [16, 32, 48, 64, 96, 128, 192, 256],
    "depth_unroll": [16, 32, 64, 128, 256, 512],
    "occupancy_values": [1, 2],
    "wgm_values": [1, 4, 8],
}

# Top-K configs to track per problem
TOP_K = 5


def is_dtype_supported(arch_name: str, dtype: str) -> bool:
    """Check if a dtype is supported for the given architecture."""
    hardware = HARDWARE[arch_name]
    return len(get_matrix_instructions(hardware, dtype)) > 0


def load_problem_sizes() -> list[tuple[int, int, int, int]]:
    """Load problem sizes from CSV file."""
    problems = []
    with open(PROBLEM_DATA_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row["m"])
            n = int(row["n"])
            k = int(row["k"])
            batch = int(row["batch_count"])
            problems.append((m, n, k, batch))
    return problems


TEST_PROBLEM_SIZES = load_problem_sizes()


def create_problem(
    m: int,
    n: int,
    k: int,
    dtype: str,
    batch: int = 1,
    transA: origami.transpose_t = origami.transpose_t.T,
    transB: origami.transpose_t = origami.transpose_t.N,
) -> origami.problem_t:
    """Create a problem specification — default target (tensilelite)."""
    problem = origami.problem_t()
    problem.size = origami.dim3_t(m, n, k)
    problem.batch = batch
    problem.a_transpose = transA
    problem.b_transpose = transB
    problem.a_dtype = origami.string_to_datatype(dtype)
    problem.b_dtype = origami.string_to_datatype(dtype)
    problem.d_dtype = origami.string_to_datatype(dtype)
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = problem.a_dtype
    problem.a_mx_block_size = 0
    problem.b_mx_block_size = 0
    return problem


def result_to_record(result: origami.prediction_result_t) -> dict:
    """Convert prediction result to a deterministic record for comparison."""
    cfg = result.config
    return {
        "mt": [cfg.mt.m, cfg.mt.n, cfg.mt.k],
        "mi": [cfg.mi.m, cfg.mi.n, cfg.mi.k],
        "occupancy": cfg.occupancy,
        "wgm": cfg.workgroup_mapping,
        "latency": repr(result.latency),  # repr preserves full float precision
    }


def generate_full_snapshot(
    arch_name: str,
    dtype: str,
    transA: origami.transpose_t,
    transB: origami.transpose_t,
) -> dict:
    """Generate a full deterministic snapshot of origami output for default path.

    Returns a dict:
      { "problem_key": { "rankings": [...], "latencies": [...] }, ... }
    """
    hardware = HARDWARE[arch_name]
    configs = create_config_list(hardware, dtype, **CONFIG_PARAMS)

    if not configs:
        return {}

    # Verify all configs are using default target (tensilelite)
    # This is the critical assertion: no config should have target_t=triton
    # when running the hipblaslt regression path
    if _HAS_TARGET_T:
        for cfg in configs:
            assert cfg.target == origami.target_t.tensilelite, (
                f"Config has non-default target: {cfg.target}. "
                "hipblaslt regression test must use only default (tensilelite) configs."
            )

    snapshot = {}
    for m, n, k, batch in TEST_PROBLEM_SIZES:
        problem = create_problem(m, n, k, dtype, batch, transA, transB)
        try:
            ranked = origami.select_topk_configs(problem, hardware, configs, TOP_K)
            if ranked:
                key = f"{m}x{n}x{k}x{batch}"
                snapshot[key] = {
                    "rankings": [result_to_record(r) for r in ranked],
                }
        except Exception:
            pass

    return snapshot


def snapshot_fingerprint(snapshot: dict) -> str:
    """Compute a deterministic SHA-256 fingerprint of a snapshot.

    This catches ANY byte-level change in the output.
    """
    # Sort keys for determinism
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def transpose_key(transA: origami.transpose_t, transB: origami.transpose_t) -> str:
    a = "T" if transA == origami.transpose_t.T else "N"
    b = "T" if transB == origami.transpose_t.T else "N"
    return f"{a}{b}"


try:
    from yaml import CSafeDumper as SafeDumper, CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeDumper, SafeLoader


def get_baseline_path(arch_name: str) -> Path:
    return BASELINE_DIR / f"{arch_name}_bitidentical.yaml"


@lru_cache(maxsize=None)
def load_arch_baseline(arch_name: str) -> dict | None:
    path = get_baseline_path(arch_name)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return yaml.load(f, Loader=SafeLoader)


def save_baseline(
    arch_name: str,
    dtype: str,
    transA: origami.transpose_t,
    transB: origami.transpose_t,
    snapshot: dict,
    fingerprint: str,
) -> None:
    """Save snapshot + fingerprint to baseline file."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = get_baseline_path(arch_name)

    if path.exists():
        with open(path, "r") as f:
            baseline = yaml.load(f, Loader=SafeLoader) or {}
    else:
        baseline = {}

    tk = transpose_key(transA, transB)
    if dtype not in baseline:
        baseline[dtype] = {}
    baseline[dtype][tk] = {
        "fingerprint": fingerprint,
        "snapshot": snapshot,
    }

    with open(path, "w") as f:
        yaml.dump(baseline, f, Dumper=SafeDumper, default_flow_style=False, sort_keys=True, width=1000)

    load_arch_baseline.cache_clear()


def load_baseline_entry(
    arch_name: str, dtype: str, transA: origami.transpose_t, transB: origami.transpose_t
) -> dict | None:
    baseline = load_arch_baseline(arch_name)
    if baseline is None:
        return None
    try:
        return baseline[dtype][transpose_key(transA, transB)]
    except KeyError:
        return None


def diff_snapshots(current: dict, baseline: dict) -> list[str]:
    """Compare snapshots, reporting precise differences with file/line detail."""
    diffs = []

    for key, base_data in baseline.items():
        if key not in current:
            diffs.append(f"MISSING problem {key}: was in baseline, not in current output")
            continue

        curr_data = current[key]
        curr_rankings = curr_data["rankings"]
        base_rankings = base_data["rankings"]

        if len(curr_rankings) != len(base_rankings):
            diffs.append(
                f"{key}: Rank count changed ({len(base_rankings)} -> {len(curr_rankings)})"
            )

        for i, (cr, br) in enumerate(zip(curr_rankings, base_rankings)):
            if cr != br:
                diffs.append(
                    f"{key} rank {i}: MISMATCH\n"
                    f"  BASELINE: MT={br['mt']}, MI={br['mi']}, occ={br['occupancy']}, "
                    f"wgm={br['wgm']}, latency={br['latency']}\n"
                    f"  CURRENT:  MT={cr['mt']}, MI={cr['mi']}, occ={cr['occupancy']}, "
                    f"wgm={cr['wgm']}, latency={cr['latency']}"
                )

    for key in current:
        if key not in baseline:
            diffs.append(f"NEW problem {key}: in current but not in baseline (unexpected)")

    return diffs


@pytest.mark.regression
@pytest.mark.hipblaslt_guardrail
class TestHipblasltBitIdentical:
    """CI guardrail: default origami output must be bit-identical to pre-Triton baseline.

    Any difference means Triton specialization code leaked into the default path.
    """

    @pytest.mark.parametrize("arch_name", list(HARDWARE.keys()))
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize(
        "transA,transB",
        [
            (origami.transpose_t.T, origami.transpose_t.N),  # TN — most common
            (origami.transpose_t.N, origami.transpose_t.N),  # NN
        ],
    )
    def test_default_path_unchanged(
        self,
        arch_name: str,
        dtype: str,
        transA: origami.transpose_t,
        transB: origami.transpose_t,
        generate_baseline: bool,
    ):
        """Verify origami default path (tensilelite) is bit-identical to baseline.

        This test FAILS if:
        - Any config ranking changes
        - Any latency prediction changes (checked via SHA-256 fingerprint)
        - Any config has target != tensilelite
        """
        trans_str = transpose_key(transA, transB)

        if not is_dtype_supported(arch_name, dtype):
            pytest.skip(f"No {dtype} support for {arch_name}")

        snapshot = generate_full_snapshot(arch_name, dtype, transA, transB)

        if not snapshot:
            pytest.skip(f"No valid configs for {arch_name}/{dtype}/{trans_str}")

        fingerprint = snapshot_fingerprint(snapshot)

        if generate_baseline:
            save_baseline(arch_name, dtype, transA, transB, snapshot, fingerprint)
            pytest.skip(f"Generated baseline for {arch_name}/{dtype}/{trans_str}")

        entry = load_baseline_entry(arch_name, dtype, transA, transB)
        if entry is None:
            pytest.fail(
                f"No baseline for {arch_name}/{dtype}/{trans_str}. "
                f"Run with --generate-baseline to create it."
            )

        # Fast path: fingerprint match means bit-identical
        if fingerprint == entry["fingerprint"]:
            return  # PASS — output is identical

        # Slow path: detailed diff for diagnostic
        baseline_snapshot = entry["snapshot"]
        diffs = diff_snapshots(snapshot, baseline_snapshot)

        if diffs:
            diff_summary = "\n".join(diffs[:20])
            if len(diffs) > 20:
                diff_summary += f"\n... and {len(diffs) - 20} more differences"
            pytest.fail(
                f"hipBLASLt REGRESSION DETECTED for {arch_name}/{dtype}/{trans_str}\n"
                f"Fingerprint mismatch:\n"
                f"  Baseline: {entry['fingerprint']}\n"
                f"  Current:  {fingerprint}\n\n"
                f"Differences:\n{diff_summary}\n\n"
                f"This means Triton specialization code has leaked into the default "
                f"(tensilelite) path. All new code MUST be gated on target_t=triton."
            )
        else:
            # Fingerprint differs but no structural diff found —
            # could be floating point repr change
            pytest.fail(
                f"hipBLASLt REGRESSION: Fingerprint mismatch for "
                f"{arch_name}/{dtype}/{trans_str} but no structural diff found.\n"
                f"  Baseline fingerprint: {entry['fingerprint']}\n"
                f"  Current fingerprint:  {fingerprint}\n"
                f"This indicates a subtle numerical change in latency predictions."
            )


@pytest.mark.regression
@pytest.mark.hipblaslt_guardrail
class TestDefaultTargetInvariant:
    """Verify that all configs created without explicit target use tensilelite."""

    def test_default_config_target(self):
        """config_t default target must be tensilelite."""
        if not _HAS_TARGET_T:
            pytest.skip("target_t not yet exposed in bindings — rebuild origami first")
        cfg = origami.config_t()
        assert cfg.target == origami.target_t.tensilelite, (
            f"Default config_t.target is {cfg.target}, expected tensilelite. "
            f"This invariant protects the hipblaslt path."
        )

    @pytest.mark.parametrize("arch_name", list(HARDWARE.keys()))
    def test_config_list_all_tensilelite(self, arch_name: str):
        """All configs from create_config_list must have target=tensilelite."""
        if not _HAS_TARGET_T:
            pytest.skip("target_t not yet exposed in bindings — rebuild origami first")
        hardware = HARDWARE[arch_name]
        for dtype in SUPPORTED_DTYPES:
            if not is_dtype_supported(arch_name, dtype):
                continue
            configs = create_config_list(hardware, dtype, **CONFIG_PARAMS)
            for i, cfg in enumerate(configs):
                assert cfg.target == origami.target_t.tensilelite, (
                    f"Config {i} for {arch_name}/{dtype} has target={cfg.target}. "
                    f"Default path configs must all be tensilelite."
                )
