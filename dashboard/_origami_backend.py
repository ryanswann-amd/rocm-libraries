"""Backend selector for the dashboard.

Imports the C++ `origami_comm` pybind11 module if the build has been
done (and is importable from the current Python interpreter); falls
back transparently to the pure-Python `model/` package otherwise.

This module is the only place dashboard code should import
`predict_row` / `predict_tensor_collective` from. The page imports
look like:

    from _origami_backend import (
        predict_row, predict_tensor_collective,
        MI300X, MI300X_COMM, BACKEND_NAME,
    )

`BACKEND_NAME` is "cpp (origami_comm)" or "python (model/*)" and is
displayed in the sidebar so users can see which implementation served
the predictions.

The C++ bindings are bit-identical with Python (verified in
origami_comms_cpp by 34,599 IEEE-754-bit-identity assertions), so the
switch is purely a performance optimization — no numerical change.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WORKSPACE = _HERE.parent
_BUILD_DIR = _WORKSPACE / "origami_comms_cpp" / "build"

if str(_BUILD_DIR) not in sys.path and _BUILD_DIR.exists():
    sys.path.insert(0, str(_BUILD_DIR))

BACKEND_NAME: str
BACKEND_KIND: str           # "cpp" | "python"
_LOAD_ERROR: str | None = None

try:
    if os.environ.get("ORIGAMI_FORCE_PY_BACKEND"):
        raise ImportError("forced python backend via ORIGAMI_FORCE_PY_BACKEND")
    import origami_comm as _oc                            # type: ignore
    predict_row                = _oc.predict_row          # noqa: E305
    predict_tensor_collective  = _oc.predict_tensor_collective
    MI300X                     = _oc.MI300X
    MI300X_COMM                = _oc.MI300X_COMM
    DEFAULT_HEURISTICS         = _oc.DEFAULT_HEURISTICS
    DataType                   = _oc.DataType
    BACKEND_NAME = f"cpp (origami_comm {getattr(_oc, '__version__', '?')})"
    BACKEND_KIND = "cpp"
except Exception as _e:
    _LOAD_ERROR = repr(_e)
    from model.collective import predict_row                        # noqa: F401
    from model.tensor_collective import predict_tensor_collective    # noqa: F401
    from model.hardware import MI300X, MI300X_COMM                   # noqa: F401
    from model.heuristics import DEFAULT_HEURISTICS                  # noqa: F401
    from model.types import DataType                                 # noqa: F401
    BACKEND_NAME = "python (model/*)"
    BACKEND_KIND = "python"


def backend_caption() -> str:
    """Human-readable one-line backend status for sidebar display."""
    if BACKEND_KIND == "cpp":
        return f"backend  **{BACKEND_NAME}** — bit-identical with `model/`, ~Nx faster"
    msg = f"backend  **{BACKEND_NAME}**"
    if _LOAD_ERROR:
        msg += f"\n\nC++ bindings unavailable: `{_LOAD_ERROR}`"
        msg += ("\n\nTo enable: build `origami_comms_cpp/` "
                "(`cmake -B build && cmake --build build`), then "
                "`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "
                "streamlit run dashboard/Home.py`")
    return msg
