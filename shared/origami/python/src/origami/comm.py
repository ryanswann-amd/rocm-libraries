# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Origami Communications Latency Model
=====================================

Predicts collective communication latency (all-reduce, all-gather, reduce-scatter)
on AMD Instinct MI300X multi-GPU systems using a calibrated bandwidth-delay model.

The model is fitted to GPU-measured Iris all-reduce data on MI300X (XGMI topology):
  - World size 8: MAPE 2.1%, R² > 0.999 (Slurm #17995, 8 msg sizes × 50 iters)
  - World size 4: MAPE 2.6%, R² > 0.999 (Slurm #17994, 8 msg sizes × 50 iters)

Core equation::

    T_comm = T_startup + message_bytes / (effective_bw × bw_scale(comm_cus))

Where:
  - T_startup: fixed kernel-launch + barrier-sync cost (ms)
  - effective_bw: sustained bandwidth at saturated CU count (GB/s)
  - bw_scale: power-law ramp from 0→1 as comm_cus grows from 0→min_comm_cus,
    with exponent gamma=0.84 jointly fitted with min_comm_cus over 37 CU-sweep
    GPU-measured data points (MI300X, OCI cluster)

This module is **purely additive** — it does not modify any existing GEMM kernel
code, GEMM model code, or rocm-libraries source. It can be used standalone or
composed with the existing origami GEMM selector.

Usage Examples
--------------
Basic latency prediction::

    >>> from origami.comm import predict_comm_latency
    >>> # 16 MB all-reduce across 8 GPUs
    >>> lat = predict_comm_latency("all_reduce", 16 * 1024**2, world_size=8)
    >>> print(f"{lat:.3f} us")

With CU partitioning (for GEMM/comm coexecution)::

    >>> lat = predict_comm_latency("all_reduce", 64 * 1024**2,
    ...                            world_size=8, comm_cus=32)

Predefined hardware profiles::

    >>> from origami.comm import list_hardware
    >>> list_hardware()
    ['mi300x']

Data Provenance
---------------
All calibrated parameters are derived from GPU-measured Iris CCL sweeps on
AMD Instinct MI300X (OCI cluster, ROCm 7.0.2, non-exclusive nodes, min_ms
statistic). Sub-saturation BW scaling uses power-law exponent gamma=0.84,
jointly fitted with per-WS saturation thresholds over 37 CU-sweep points.

Combined validation MAPE: 3.37% across 53 all-reduce data points (msg sweeps
+ CU sweeps), 100% within 10% target. Max single-point error: 7.50%.

See Also
--------
- K-018 calibrated params: ``k018_calibrated_comm_params.json``
- K-020 coexecution model: ``coexecution_model.py``
- Iris CCL: AMD's GPU communication library (RCCL-compatible)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


# ── Calibrated Hardware Profiles ───────────────────────────────────────────
# Each profile contains GPU-measured parameters for a specific GPU + world_size.
# All numbers are from Slurm jobs on OCI MI300X, verified by the rigor team.

@dataclass(frozen=True)
class CommProfile:
    """Calibrated communication parameters for a GPU topology.

    These are **not** theoretical peaks — they are fitted to GPU-measured
    min-latency data from Iris all-reduce sweeps.

    Attributes:
        gpu: GPU model identifier (e.g., "mi300x").
        world_size: Number of GPUs in the collective.
        total_cus: Total CUs on the GPU.
        startup_ms: Fixed startup latency in milliseconds (kernel launch + barrier).
        effective_bw_gbps: Sustained BW in GB/s at saturated CU count.
        algo_bw_gbps: Algorithmic bandwidth (accounts for ring/tree overhead).
        min_comm_cus: CU count at which BW reaches >=95% of peak.
            Below this, ``bw_scale = (comm_cus / min_comm_cus) ** bw_gamma``.
        bw_gamma: Sub-saturation BW scaling exponent (default 0.84).
            Jointly fitted with min_comm_cus over 37 CU-sweep GPU-measured
            points (ws=4+ws=8) on MI300X (Slurm #17994/#17995, OCI cluster).
            gamma<1: XGMI protocol overhead amortizes at low CUs.
            Improvement over linear (gamma=1.0): CU sweep MAPE 3.81% vs 8.37%.
        mape_pct: Mean Absolute Percentage Error of the model vs measured data.
        slurm_job: Slurm job ID that produced the calibration data.
        data_quality: Provenance tag (e.g., "PROVISIONAL", "VERIFIED").
    """
    gpu: str
    world_size: int
    total_cus: int
    startup_ms: float
    effective_bw_gbps: float
    algo_bw_gbps: float
    min_comm_cus: int
    bw_gamma: float
    mape_pct: float
    slurm_job: int
    data_quality: str = "PROVISIONAL"


# Calibrated from K-018 Iris sweeps (OCI MI300X, ROCm 7.0.2)
# Source: k018_calibrated_comm_params.json, verified by rigor audit
# min_comm_cus jointly fitted with gamma=0.84 over 37 CU-sweep points
_PROFILES: Dict[Tuple[str, int], CommProfile] = {
    ("mi300x", 8): CommProfile(
        gpu="mi300x",
        world_size=8,
        total_cus=304,
        startup_ms=0.2822,
        effective_bw_gbps=212.31,
        algo_bw_gbps=371.53,
        min_comm_cus=40,          # Jointly fitted with gamma=0.84
        bw_gamma=0.84,
        mape_pct=2.11,
        slurm_job=17995,
        data_quality="PROVISIONAL — min_ms from non-exclusive nodes",
    ),
    ("mi300x", 4): CommProfile(
        gpu="mi300x",
        world_size=4,
        total_cus=304,
        startup_ms=0.2605,
        effective_bw_gbps=106.49,
        algo_bw_gbps=159.73,
        min_comm_cus=24,          # Jointly fitted with gamma=0.84
        bw_gamma=0.84,
        mape_pct=2.55,
        slurm_job=17994,
        data_quality="PROVISIONAL — min_ms from non-exclusive nodes",
    ),
    ("mi300x", 2): CommProfile(
        gpu="mi300x",
        world_size=2,
        total_cus=304,
        startup_ms=0.26,          # interpolated from ws=4 startup
        effective_bw_gbps=53.25,  # linear scaling from ws=4 (106.49 / 2)
        algo_bw_gbps=79.87,
        min_comm_cus=20,          # conservative: fewer links → fewer CUs needed
        bw_gamma=0.84,
        mape_pct=-1.0,            # not yet validated
        slurm_job=-1,
        data_quality="INTERPOLATED — derived from ws=4, not GPU-measured",
    ),
}


# ── Collective Scaling Functions ───────────────────────────────────────────

_SUPPORTED_COLLECTIVES = {"all_reduce", "all_gather", "reduce_scatter", "broadcast"}


def _validate_collective(collective: str) -> str:
    """Normalize and validate collective name."""
    coll = collective.lower().replace("-", "_")
    if coll not in _SUPPORTED_COLLECTIVES:
        raise ValueError(
            f"Unknown collective '{collective}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_COLLECTIVES))}"
        )
    return coll


def _collective_scale_factor(collective: str) -> float:
    """Return the data volume scale factor relative to all_reduce.

    The calibrated ``effective_bw_gbps`` was fitted to all_reduce as:
        ``T = T_startup + message_bytes / effective_bw``
    where ``message_bytes`` is the **raw tensor size** (not on-wire volume).
    The effective_bw already absorbs the ring algorithm's 2x(ws-1)/ws overhead.

    For other collectives, we scale the message bytes relative to all_reduce:
      - all_reduce: 1.0  (calibration baseline — no adjustment)
      - all_gather: 0.5  (half the data movement of all_reduce)
      - reduce_scatter: 0.5  (half the data movement of all_reduce)
      - broadcast: 0.5  (single direction, roughly half of all_reduce)

    These ratios are approximate and should be validated when per-collective
    GPU calibration data becomes available.
    """
    coll = _validate_collective(collective)
    if coll == "all_reduce":
        return 1.0
    elif coll in ("all_gather", "reduce_scatter", "broadcast"):
        return 0.5
    return 1.0


# ── Core Prediction API ───────────────────────────────────────────────────

def predict_comm_latency(
    collective: str,
    message_bytes: Union[int, float],
    world_size: int = 8,
    gpu: str = "mi300x",
    topology: str = "xgmi_ring",
    comm_cus: Optional[int] = None,
) -> float:
    """Predict collective communication latency in microseconds.

    Uses a calibrated bandwidth-delay product model fitted to GPU-measured
    Iris all-reduce data on MI300X.

    Parameters
    ----------
    collective : str
        Type of collective operation. One of:
        ``"all_reduce"``, ``"all_gather"``, ``"reduce_scatter"``, ``"broadcast"``.
    message_bytes : int or float
        Total message size in bytes **per GPU** (i.e., the tensor size before
        the collective, not the on-wire volume).
    world_size : int, optional
        Number of GPUs participating in the collective (default: 8).
    gpu : str, optional
        GPU model identifier (default: ``"mi300x"``).
    topology : str, optional
        Interconnect topology (default: ``"xgmi_ring"``). Currently only XGMI
        ring topology is calibrated.
    comm_cus : int or None, optional
        Number of CUs dedicated to communication (for CU-partitioned execution).
        If ``None``, assumes all CUs are available (full-chip BW).
        When < ``min_comm_cus``, bandwidth scales as power-law:
        ``bw_scale = (comm_cus / min_comm_cus) ** bw_gamma``.

    Returns
    -------
    float
        Predicted latency in **microseconds** (us).

    Raises
    ------
    ValueError
        If the collective type is unknown, or if the (gpu, world_size)
        combination has no calibrated profile.

    Notes
    -----
    The model equation:

    .. math::

        T_{comm} = T_{startup} + \\frac{\\text{message\\_bytes} \\times \\text{coll\\_scale}}{\\text{effective\\_bw} \\times \\text{bw\\_scale}}

    The ``effective_bw`` was calibrated against all_reduce using **raw message
    bytes** (not on-wire volume). It already absorbs the ring algorithm's
    2x(ws-1)/ws overhead. For non-all_reduce collectives, a ``coll_scale``
    factor adjusts the effective data volume (0.5 for all_gather/reduce_scatter).

    ``bw_scale`` accounts for sub-saturation CU allocation when ``comm_cus``
    is below the minimum required for peak bandwidth. Uses power-law exponent
    gamma=0.84 (jointly fitted with min_comm_cus over 37 CU-sweep points),
    which improved CU sweep MAPE from 8.37% (linear) to 3.81%.

    **Data provenance**: All calibrated parameters come from Iris CCL sweeps on
    MI300X (OCI cluster, ROCm 7.0.2). ws=8 data from Slurm #17995 (50 iters x
    8 message sizes), ws=4 from Slurm #17994.

    Examples
    --------
    >>> from origami.comm import predict_comm_latency

    # 16 MB all-reduce across 8 GPUs (full-chip bandwidth):
    >>> lat = predict_comm_latency("all_reduce", 16 * 1024**2, world_size=8)
    >>> print(f"{lat:.1f} us")

    # 128 MB all-gather across 4 GPUs:
    >>> lat = predict_comm_latency("all_gather", 128 * 1024**2, world_size=4)

    # 64 MB reduce-scatter with CU partitioning (32 CUs for comm):
    >>> lat = predict_comm_latency("reduce_scatter", 64 * 1024**2,
    ...                            world_size=8, comm_cus=32)
    """
    # Validate collective type
    coll = _validate_collective(collective)

    # Look up calibrated profile
    profile = _get_profile(gpu, world_size)

    # The calibrated effective_bw was fitted to all_reduce with raw message_bytes.
    # For other collectives, apply a scale factor to adjust the effective data volume.
    coll_scale = _collective_scale_factor(coll)
    scaled_bytes = float(message_bytes) * coll_scale

    # CU-dependent BW scaling: power-law ramp below saturation threshold
    # Exponent gamma=0.84 jointly fitted with per-WS min_comm_cus via grid
    # search over gamma=[0.40, 0.84], min_cus=[24..72] on 37 CU-sweep +
    # 16 msg-sweep GPU-measured data points, MI300X (Slurm #17994/17995, OCI):
    #   - Linear (gamma=1.0, old min_cus): CU sweep MAPE 8.37%, max 52.5%
    #   - Sublinear (gamma=0.84, fitted min_cus): CU sweep MAPE 3.81%, max 6.28%
    #   - All 53 all-reduce data points within 10% error (100% pass rate)
    #   - Combined MAPE: 3.37% across msg sweeps + CU sweeps
    # Physics: BW degrades sub-linearly with fewer CUs because XGMI links
    # saturate at low CU counts; fewer CUs still keep links partially busy.
    bw_scale = 1.0
    if comm_cus is not None:
        if comm_cus <= 0:
            raise ValueError("comm_cus must be > 0")
        if comm_cus < profile.min_comm_cus:
            bw_scale = (comm_cus / profile.min_comm_cus) ** profile.bw_gamma

    # BW-delay product: startup + transfer
    # effective_bw is in GB/s; convert to bytes/ms for ms arithmetic
    effective_bw_bytes_per_ms = profile.effective_bw_gbps * 1e6  # GB/s = 1e9 B/s = 1e6 B/ms
    if effective_bw_bytes_per_ms * bw_scale <= 0:
        return float("inf")

    t_transfer_ms = scaled_bytes / (effective_bw_bytes_per_ms * bw_scale)
    t_total_ms = profile.startup_ms + t_transfer_ms

    # Return in microseconds
    return t_total_ms * 1e3


def predict_comm_latency_ms(
    collective: str,
    message_bytes: Union[int, float],
    world_size: int = 8,
    gpu: str = "mi300x",
    topology: str = "xgmi_ring",
    comm_cus: Optional[int] = None,
) -> float:
    """Same as :func:`predict_comm_latency` but returns milliseconds.

    Convenience wrapper for integration with the GEMM model (which uses ms).

    See :func:`predict_comm_latency` for full documentation.
    """
    return predict_comm_latency(
        collective, message_bytes, world_size, gpu, topology, comm_cus
    ) / 1e3


# ── Profile Lookup ─────────────────────────────────────────────────────────

def _get_profile(gpu: str, world_size: int) -> CommProfile:
    """Retrieve a calibrated CommProfile.

    Raises ValueError with available options if the (gpu, world_size)
    pair is not calibrated.
    """
    key = (gpu.lower(), world_size)
    if key not in _PROFILES:
        available = sorted(_PROFILES.keys())
        raise ValueError(
            f"No calibrated comm profile for (gpu={gpu!r}, world_size={world_size}). "
            f"Available: {available}"
        )
    return _PROFILES[key]


def get_profile(gpu: str = "mi300x", world_size: int = 8) -> CommProfile:
    """Public accessor for calibrated communication profiles.

    Parameters
    ----------
    gpu : str
        GPU model identifier.
    world_size : int
        Number of GPUs.

    Returns
    -------
    CommProfile
        Frozen dataclass with all calibrated parameters.

    Examples
    --------
    >>> from origami.comm import get_profile
    >>> p = get_profile("mi300x", 8)
    >>> print(f"Startup: {p.startup_ms} ms, BW: {p.effective_bw_gbps} GB/s")
    Startup: 0.2822 ms, BW: 212.31 GB/s
    """
    return _get_profile(gpu, world_size)


def list_hardware() -> List[str]:
    """Return list of GPU models with calibrated comm profiles.

    Returns
    -------
    list of str
        GPU model identifiers (e.g., ``["mi300x"]``).
    """
    return sorted(set(gpu for gpu, _ in _PROFILES.keys()))


def list_profiles() -> List[Tuple[str, int]]:
    """Return list of all calibrated (gpu, world_size) combinations.

    Returns
    -------
    list of (str, int)
        Each entry is ``(gpu_model, world_size)``.
    """
    return sorted(_PROFILES.keys())


# ── Coexecution / Overlap Prediction ──────────────────────────────────

def predict_overlap_latency(
    gemm_latency_ms: float,
    message_bytes: Union[int, float],
    collective: str = "all_reduce",
    world_size: int = 8,
    gpu: str = "mi300x",
    comm_cus: Optional[int] = None,
) -> Dict:
    """Predict coexecution latency when GEMM and comm overlap on GPU.

    Computes the critical-path latency assuming GEMM and communication
    execute concurrently (CU-partitioned or stream-overlap):

    .. math::

        T_{coexec} = \\max(T_{gemm}, T_{comm})

    Parameters
    ----------
    gemm_latency_ms : float
        GEMM kernel latency in milliseconds (from origami.compute_total_latency
        or GPU-measured).
    message_bytes : int or float
        Communication message size in bytes.
    collective : str, optional
        Collective type (default: ``"all_reduce"``).
    world_size : int, optional
        Number of GPUs (default: 8).
    gpu : str, optional
        GPU model (default: ``"mi300x"``).
    comm_cus : int or None, optional
        CUs allocated to communication (None = full BW).

    Returns
    -------
    dict
        Keys: ``gemm_ms``, ``comm_ms``, ``coexec_ms``, ``sequential_ms``,
        ``speedup``, ``bottleneck``.

    Examples
    --------
    >>> from origami.comm import predict_overlap_latency
    >>> r = predict_overlap_latency(0.5, 128 * 1024**2, comm_cus=48)
    >>> print(f"Coexec: {r['coexec_ms']:.3f} ms, speedup: {r['speedup']:.2f}x")
    """
    comm_ms = predict_comm_latency_ms(
        collective, message_bytes, world_size, gpu, comm_cus=comm_cus
    )
    coexec_ms = max(gemm_latency_ms, comm_ms)
    sequential_ms = gemm_latency_ms + comm_ms
    speedup = sequential_ms / coexec_ms if coexec_ms > 0 else float("inf")

    return {
        "gemm_ms": gemm_latency_ms,
        "comm_ms": round(comm_ms, 4),
        "coexec_ms": round(coexec_ms, 4),
        "sequential_ms": round(sequential_ms, 4),
        "speedup": round(speedup, 4),
        "bottleneck": "gemm" if gemm_latency_ms >= comm_ms else "comm",
    }


# ── Batch Prediction ──────────────────────────────────────────────────────

def predict_batch(
    collective: str,
    message_bytes_list: List[Union[int, float]],
    world_size: int = 8,
    gpu: str = "mi300x",
    comm_cus: Optional[int] = None,
) -> List[Dict]:
    """Predict latency for multiple message sizes.

    Useful for building latency curves or validating the model against
    measured sweep data.

    Parameters
    ----------
    collective : str
        Collective type.
    message_bytes_list : list of int
        Message sizes in bytes.
    world_size : int
        Number of GPUs.
    gpu : str
        GPU model.
    comm_cus : int or None
        CUs for communication (None = full chip).

    Returns
    -------
    list of dict
        Each dict has keys: ``message_bytes``, ``predicted_us``,
        ``predicted_ms``, ``scaled_bytes``, ``collective``.
    """
    results = []
    for msg_bytes in message_bytes_list:
        lat_us = predict_comm_latency(
            collective, msg_bytes, world_size, gpu, comm_cus=comm_cus
        )
        coll_scale = _collective_scale_factor(collective)
        scaled = float(msg_bytes) * coll_scale
        results.append({
            "message_bytes": int(msg_bytes),
            "predicted_us": round(lat_us, 2),
            "predicted_ms": round(lat_us / 1e3, 4),
            "scaled_bytes": int(scaled),
            "collective": collective,
        })
    return results


# ── Validation Helper ─────────────────────────────────────────────────────

# GPU-measured reference data for self-test
# Source: Slurm #17995 (ws=8) and #17994 (ws=4), MI300X, min_ms statistic
_VALIDATION_DATA = {
    ("mi300x", 8): [
        # (actual_bytes, measured_min_ms)
        (65536, 0.2785),
        (245760, 0.2815),
        (991232, 0.2867),
        (4145152, 0.2836),
        (16588800, 0.3540),
        (67092480, 0.6010),
        (134217728, 0.9460),
        (268378112, 1.5889),
    ],
    ("mi300x", 4): [
        (65536, 0.2664),
        (245760, 0.2573),
        (991232, 0.2578),
        (4145152, 0.2856),
        (16588800, 0.4079),
        (67092480, 0.8958),
        (134217728, 1.5692),
        (268378112, 2.8106),
    ],
}


def validate(gpu: str = "mi300x", world_size: int = 8,
             verbose: bool = False) -> Dict:
    """Validate the model against GPU-measured reference data.

    Computes MAPE and per-point errors against Iris all-reduce sweep data
    embedded in this module.

    Parameters
    ----------
    gpu : str
        GPU model.
    world_size : int
        World size to validate.
    verbose : bool
        If True, print per-point comparison.

    Returns
    -------
    dict
        Validation results with keys: ``mape_pct``, ``max_error_pct``,
        ``n_points``, ``points`` (list of per-point dicts).

    Examples
    --------
    >>> from origami.comm import validate
    >>> v = validate("mi300x", 8)
    >>> assert v["mape_pct"] < 10.0, f"MAPE {v['mape_pct']}% exceeds 10% threshold"
    """
    key = (gpu.lower(), world_size)
    if key not in _VALIDATION_DATA:
        return {"error": f"No validation data for {key}"}

    ref_data = _VALIDATION_DATA[key]
    errors = []
    points = []

    for actual_bytes, measured_ms in ref_data:
        # The calibration data is from all_reduce, so validate with all_reduce
        predicted_ms = predict_comm_latency_ms(
            "all_reduce", actual_bytes, world_size, gpu
        )
        error_pct = abs(predicted_ms - measured_ms) / measured_ms * 100
        errors.append(error_pct)

        point = {
            "bytes": actual_bytes,
            "measured_ms": measured_ms,
            "predicted_ms": round(predicted_ms, 4),
            "error_pct": round(error_pct, 2),
        }
        points.append(point)

        if verbose:
            status = "OK" if error_pct < 10 else "!!"
            print(f"  {status} {actual_bytes:>12d} B  "
                  f"meas={measured_ms:.4f} ms  "
                  f"pred={predicted_ms:.4f} ms  "
                  f"err={error_pct:.1f}%")

    mape = sum(errors) / len(errors)
    return {
        "gpu": gpu,
        "world_size": world_size,
        "mape_pct": round(mape, 2),
        "max_error_pct": round(max(errors), 2),
        "n_points": len(errors),
        "points": points,
    }


# ── Torch-Compatible Convenience API ──────────────────────────────────────

def predict_for_tensor(
    collective: str,
    tensor_or_shape: "Union[object, Tuple[int, ...]]",
    dtype_bytes: int = 2,
    world_size: int = 8,
    gpu: str = "mi300x",
    comm_cus: Optional[int] = None,
) -> float:
    """Predict comm latency from a torch tensor or shape + dtype.

    Convenience wrapper that computes ``message_bytes`` automatically,
    mirroring the way users invoke ``torch.distributed`` collectives.

    Parameters
    ----------
    collective : str
        One of ``"all_reduce"``, ``"all_gather"``, ``"reduce_scatter"``, ``"broadcast"``.
    tensor_or_shape : torch.Tensor or tuple of int
        Either a torch tensor (its ``nbytes`` is used) or a shape tuple.
        If a shape tuple is given, ``dtype_bytes`` sets the element size.
    dtype_bytes : int, optional
        Bytes per element when ``tensor_or_shape`` is a shape tuple (default: 2
        for FP16/BF16). Ignored if a torch.Tensor is passed.
    world_size : int, optional
        Number of GPUs (default: 8).
    gpu : str, optional
        GPU model (default: ``"mi300x"``).
    comm_cus : int or None, optional
        CUs allocated to communication.

    Returns
    -------
    float
        Predicted latency in microseconds.

    Examples
    --------
    With a shape tuple (no torch dependency)::

        >>> from origami.comm import predict_for_tensor
        >>> # 4096x4096 BF16 all-reduce across 8 GPUs
        >>> lat = predict_for_tensor("all_reduce", (4096, 4096), dtype_bytes=2)
    """
    if isinstance(tensor_or_shape, tuple):
        numel = math.prod(tensor_or_shape)
        msg_bytes = numel * dtype_bytes
    else:
        # Assume torch.Tensor — use nbytes
        msg_bytes = tensor_or_shape.nelement() * tensor_or_shape.element_size()

    return predict_comm_latency(
        collective, msg_bytes, world_size, gpu, comm_cus=comm_cus
    )


def estimate_allreduce_ms(
    numel: int,
    dtype_bytes: int = 2,
    world_size: int = 8,
    gpu: str = "mi300x",
) -> float:
    """One-liner all-reduce latency estimator (milliseconds).

    The simplest possible API for the most common use case: predict
    all-reduce time for a gradient tensor in distributed training.

    Parameters
    ----------
    numel : int
        Number of elements in the tensor.
    dtype_bytes : int
        Bytes per element (2 for FP16/BF16, 4 for FP32).
    world_size : int
        Number of GPUs.
    gpu : str
        GPU model.

    Returns
    -------
    float
        Predicted all-reduce latency in milliseconds.

    Examples
    --------
    >>> from origami.comm import estimate_allreduce_ms
    >>> # How long will a 100M-param gradient allreduce take?
    >>> print(f"{estimate_allreduce_ms(100_000_000):.2f} ms")
    """
    return predict_comm_latency_ms(
        "all_reduce", numel * dtype_bytes, world_size, gpu
    )


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description="Origami Communications Latency Predictor"
    )
    sub = parser.add_subparsers(dest="cmd")

    # predict sub-command
    p_pred = sub.add_parser("predict", help="Predict latency for a single call")
    p_pred.add_argument("collective", choices=[
        "all_reduce", "all_gather", "reduce_scatter", "broadcast"
    ])
    p_pred.add_argument("message_bytes", type=int, help="Message size in bytes")
    p_pred.add_argument("--world-size", type=int, default=8)
    p_pred.add_argument("--gpu", default="mi300x")
    p_pred.add_argument("--comm-cus", type=int, default=None)

    # validate sub-command
    p_val = sub.add_parser("validate", help="Run model validation")
    p_val.add_argument("--gpu", default="mi300x")
    p_val.add_argument("--world-size", type=int, default=8)
    p_val.add_argument("--verbose", action="store_true")

    # profiles sub-command
    sub.add_parser("profiles", help="List calibrated profiles")

    args = parser.parse_args()

    if args.cmd == "predict":
        lat_us = predict_comm_latency(
            args.collective, args.message_bytes,
            args.world_size, args.gpu, comm_cus=args.comm_cus
        )
        print(f"{lat_us:.2f} us ({lat_us/1e3:.4f} ms)")

    elif args.cmd == "validate":
        result = validate(args.gpu, args.world_size, verbose=args.verbose)
        print(f"\nValidation: MAPE = {result['mape_pct']:.2f}% "
              f"(max {result['max_error_pct']:.2f}%) "
              f"over {result['n_points']} points")
        if result["mape_pct"] < 10.0:
            print("PASS: MAPE < 10% threshold")
        else:
            print("FAIL: MAPE >= 10% threshold")

    elif args.cmd == "profiles":
        for key in sorted(_PROFILES.keys()):
            p = _PROFILES[key]
            print(f"  {p.gpu} ws={p.world_size}: "
                  f"startup={p.startup_ms}ms  BW={p.effective_bw_gbps}GB/s  "
                  f"gamma={p.bw_gamma}  min_cus={p.min_comm_cus}  "
                  f"MAPE={p.mape_pct}%  [Job #{p.slurm_job}]  "
                  f"({p.data_quality})")

    else:
        parser.print_help()
