"""
Tensor Collective — predicted vs measured for shape-aware collectives.

Consumes `data/tensor_shapes_sweep.csv` (909 measurements on c42 MI300X
across 5 ops × W∈{2,4,8} × 45 BF16 shapes) and runs them through
`model.tensor_collective.predict_tensor_collective`.

What this page adds over the byte-level page:
  - Per-shape-family correlation (1d, transformer, aspect-ratio, sub-cacheline)
  - Shape-sensitivity panels: at fixed per-rank byte total, are measured
    and predicted latency invariant to (m, n) aspect ratio?
  - Per-row-width and per-ndim error breakdown
  - Framework-overhead diagnostic: this corpus is `torch.distributed` which
    sits ~400 µs above raw rccl-tests for small messages; the page calls
    that out explicitly per per-rank-bytes bucket.

The dataset is small enough (~900 points) that scatter plots are not laggy,
so most are rendered by default. Only the two heaviest scatters are gated.
"""

import sys
import os
_THIS = os.path.abspath(__file__)
_PAGES = os.path.dirname(_THIS)
_DASHBOARD = os.path.dirname(_PAGES)
_WORKSPACE = os.path.dirname(_DASHBOARD)
sys.path.insert(0, _WORKSPACE)
sys.path.insert(0, _DASHBOARD)  # for `_origami_backend`

from math import prod
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st

from _origami_backend import (                            # noqa: E402
    predict_tensor_collective, MI300X, MI300X_COMM, DataType,
    BACKEND_NAME, BACKEND_KIND, backend_caption,
)


SUPPORTED_OPS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast")
PALETTE = pc.qualitative.Plotly
FAMILY_PALETTE = pc.qualitative.Bold
TIGHT_MARGIN = dict(l=10, r=10, t=44, b=10)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _human_bytes(n: int) -> str:
    if n >= 1024**3: return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2: return f"{n / 1024**2:.1f} MB"
    if n >= 1024:    return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _shape_family(tag: str) -> str:
    """Group shape tags into families for color/aggregation."""
    if tag.startswith("1d_"):     return "1D flat"
    if tag.startswith("tf_"):     return "transformer (BS, H)"
    if tag.startswith("ar_"):     return "aspect-ratio (4 MiB)"
    if tag.startswith("subcl_"):  return "sub-cacheline (256 KiB)"
    return "other"


def _parse_shape(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in str(s).split("x"))


def _model_version() -> str:
    root = Path(__file__).resolve().parents[2] / "model"
    mtimes = []
    for p in sorted(root.glob("*.py")):
        try:
            mtimes.append((p.name, p.stat().st_mtime_ns))
        except OSError:
            continue
    return repr(mtimes)


# ────────────────────────────────────────────────────────────────────
# Data + predictions
# ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading tensor-shape sweep CSV…")
def load_csv() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[2] / "data" / "tensor_shapes_sweep.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["primitive"].isin(SUPPORTED_OPS)]
    df = df[(df["latency_us"] > 0) & (df["per_rank_bytes"] > 0)]
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Predicting all rows…")
def predict_all(df: pd.DataFrame, model_version: str, framework: str) -> pd.DataFrame:
    del model_version  # cache-key only
    preds, backend, overhead = [], [], []
    for _, r in df.iterrows():
        try:
            shape = _parse_shape(r["shape"])
            p = predict_tensor_collective(
                op=str(r["primitive"]),
                input_shape=shape,
                dtype=str(r["dtype"]),
                world_size=int(r["world_size"]),
                dim=int(r["dim"]),
                nchannels=32,
                hw=MI300X,
                comm_hw=MI300X_COMM,
                framework=framework,
            )
            preds.append(p.predicted_us)
            backend.append(p.backend_us)
            overhead.append(p.framework_overhead_us)
        except Exception:
            preds.append(float("nan"))
            backend.append(float("nan"))
            overhead.append(float("nan"))
    out = df.copy()
    out["predicted_us"] = preds
    out["backend_us"] = backend
    out["framework_overhead_us"] = overhead
    out["error_pct"] = (out["predicted_us"] - out["latency_us"]) / out["latency_us"] * 100
    out["abs_error_pct"] = out["error_pct"].abs()
    out["gap_us"] = out["latency_us"] - out["predicted_us"]
    out["shape_family"] = out["tag"].map(_shape_family)
    out["row_bytes"] = out["shape_n"] * 2  # BF16
    # m_lead and shape_n already in CSV; derive aspect ratio (clamped for log).
    out["aspect_ratio"] = (out["shape_m"] / out["shape_n"]).clip(lower=1e-6, upper=1e6)
    return out


# ────────────────────────────────────────────────────────────────────
# Plot factories
# ────────────────────────────────────────────────────────────────────

def figure_global_scatter(fdf: pd.DataFrame, color_by: str, log_axes: bool) -> go.Figure:
    keys = sorted(fdf[color_by].dropna().unique())
    fig = go.Figure()
    for i, k in enumerate(keys):
        sub = fdf[fdf[color_by] == k]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["latency_us"], y=sub["predicted_us"],
            mode="markers", name=str(k),
            marker=dict(size=5,
                        color=(FAMILY_PALETTE if color_by == "shape_family" else PALETTE)[i % 10],
                        opacity=0.7, line=dict(width=0)),
            customdata=np.stack([sub["shape"], sub["world_size"], sub["dim"],
                                  sub["per_rank_bytes"], sub["error_pct"],
                                  sub["tag"]], axis=1),
            hovertemplate=(
                f"<b>{k}</b><br>"
                "shape=%{customdata[0]} (tag=%{customdata[5]})<br>"
                "W=%{customdata[1]}, dim=%{customdata[2]}<br>"
                "per-rank=%{customdata[3]:,d} B<br>"
                "measured=%{x:.1f} µs  predicted=%{y:.1f} µs<br>"
                "err=%{customdata[4]:+.1f}%<extra></extra>"
            ),
        ))
    lo = float(min(fdf["latency_us"].min(), fdf["predicted_us"].min())) * 0.9
    hi = float(max(fdf["latency_us"].max(), fdf["predicted_us"].max())) * 1.1
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi],
                              line=dict(color="black", dash="dash", width=1),
                              name="y = x", mode="lines", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo * 1.5, hi * 1.5],
                              line=dict(color="gray", dash="dot", width=1),
                              name="±50%", mode="lines", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo / 1.5, hi / 1.5],
                              line=dict(color="gray", dash="dot", width=1),
                              name="", showlegend=False, mode="lines", hoverinfo="skip"))
    if log_axes:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    fig.update_layout(
        title=f"Predicted vs measured (color = {color_by})",
        xaxis_title="Measured latency (µs)",
        yaxis_title="Predicted latency (µs)",
        height=520, margin=TIGHT_MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def figure_shape_sensitivity(fdf: pd.DataFrame, family: str,
                              primitives: list) -> go.Figure:
    """For a fixed-byte-budget family (ar_ or subcl_), show measured vs
    predicted latency across all shapes at W=8 for each primitive.

    The whole point: model used to predict 50-175× variance across these
    same-byte shapes; after the dense-tile fix it predicts 1.0× (constant),
    matching measured 1.00-1.08×.
    """
    sub = fdf[fdf["tag"].str.startswith(f"{family}_")]
    sub = sub[sub["world_size"] == 8]
    sub = sub[sub["primitive"].isin(primitives)]
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data for family {family!r}",
            height=300, margin=TIGHT_MARGIN,
        )
        return fig

    fig = go.Figure()
    for i, p in enumerate(sorted(sub["primitive"].unique())):
        sp = sub[sub["primitive"] == p].sort_values("shape_m")
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=sp["shape"], y=sp["latency_us"], mode="markers+lines",
            name=f"meas {p}", legendgroup=p,
            marker=dict(symbol="circle-open", size=8, color=color, line=dict(width=2)),
            line=dict(color=color, width=1, dash="dot"),
            hovertemplate="meas %{y:.1f} µs<br>shape=%{x}<extra>" + p + "</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=sp["shape"], y=sp["predicted_us"], mode="markers+lines",
            name=f"pred {p}", legendgroup=p,
            marker=dict(symbol="x", size=8, color=color),
            line=dict(color=color, width=2),
            hovertemplate="pred %{y:.1f} µs<br>shape=%{x}<extra>" + p + "</extra>",
        ))
    fig.update_layout(
        title=(f"{family.upper()} family @ W=8 — measured (open ○) vs "
                "predicted (×) across same-byte shapes"),
        xaxis_title="shape (m × n)", yaxis_title="latency (µs)",
        height=380, margin=TIGHT_MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def figure_error_vs(fdf: pd.DataFrame, axis: str, log_x: bool,
                     color_by: str = "primitive") -> go.Figure:
    keys = sorted(fdf[color_by].dropna().unique())
    fig = go.Figure()
    for i, k in enumerate(keys):
        sub = fdf[fdf[color_by] == k]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[axis], y=sub["error_pct"],
            mode="markers", name=str(k),
            marker=dict(size=5, color=PALETTE[i % len(PALETTE)], opacity=0.65,
                        line=dict(width=0)),
            customdata=np.stack([sub["shape"], sub["world_size"], sub["tag"]], axis=1),
            hovertemplate=(
                f"<b>{k}</b><br>{axis}=" "%{x}<br>"
                "shape=%{customdata[0]}, W=%{customdata[1]}<br>"
                "tag=%{customdata[2]}, err=%{y:+.1f}%<extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.add_hline(y=50, line=dict(color="gray", dash="dot", width=1))
    fig.add_hline(y=-50, line=dict(color="gray", dash="dot", width=1))
    if log_x:
        fig.update_xaxes(type="log")
    fig.update_layout(
        title=f"Error % vs {axis}",
        xaxis_title=axis, yaxis_title="signed err %",
        height=340, margin=TIGHT_MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def figure_overhead_floor(fdf: pd.DataFrame, framework: str) -> go.Figure:
    """Per per-rank-bytes bucket: median measured / backend-only / predicted.

    The gap between `backend-only` and `predicted (with floor)` is the
    framework constant from `Heuristics.framework_overhead_ns`. The gap
    between `predicted` and `measured` is residual model error.
    """
    df = fdf.copy()
    bins = [0, 4*1024, 64*1024, 1024*1024, 16*1024*1024, 1024*1024*1024]
    labels = ["≤4 KB", "4-64 KB", "64 KB–1 MB", "1-16 MB", ">16 MB"]
    df["bucket"] = pd.cut(df["per_rank_bytes"], bins=bins, labels=labels)
    g = (df.groupby("bucket", observed=True)
            .agg(measured=("latency_us", "median"),
                 backend=("backend_us", "median"),
                 predicted=("predicted_us", "median"),
                 overhead=("framework_overhead_us", "median"),
                 rows=("latency_us", "count"))
            .reset_index())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=g["bucket"].astype(str), y=g["measured"],
        name=f"measured ({framework})",
        marker_color="#4C78A8",
        text=[f"{v:.0f} µs (n={n})" for v, n in zip(g["measured"], g["rows"])],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=g["bucket"].astype(str), y=g["backend"],
        name="backend-only (no framework floor)",
        marker_color="#9C8AC4",
        text=[f"{v:.0f} µs" for v in g["backend"]],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=g["bucket"].astype(str), y=g["predicted"],
        name=f"predicted (backend + {framework} floor)",
        marker_color="#F58518",
        text=[f"{v:.0f} µs" for v in g["predicted"]],
        textposition="outside",
    ))
    fig.update_yaxes(type="log", title="median latency (µs)")
    overhead_us = float(g["overhead"].dropna().iloc[0]) if not g["overhead"].dropna().empty else 0.0
    fig.update_layout(
        title=(f"Per per-rank-bytes bucket: measured vs backend-only "
               f"vs predicted ({framework} floor = {overhead_us:.0f} µs)"),
        height=380, margin=TIGHT_MARGIN, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ────────────────────────────────────────────────────────────────────
# Page body
# ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Tensor Correlation", layout="wide")
st.title("Tensor Collective — Predicted vs Measured (shape-aware)")
st.caption(
    "Validates `model.tensor_collective.predict_tensor_collective` against "
    "`data/tensor_shapes_sweep.csv` — torch.distributed measurements on "
    "c42 MI300X across 5 ops × W∈{2,4,8} × 45 BF16 shapes. The byte-level "
    "page (`1_Correlation.py`) covers the byte-level CCL layer; this page "
    "is for the shape-aware Tensor Collective layer above it. The sidebar "
    "**Framework** selector picks the host-overhead floor from "
    "`model.heuristics.DEFAULT_HEURISTICS`."
)

df_raw = load_csv()
if df_raw.empty:
    st.error(
        "Could not find `data/tensor_shapes_sweep.csv`. "
        "Run `bash microbench/run_torch_shapes.sh` to generate it."
    )
    st.stop()

# ────────── Sidebar (heuristics first so it parameterizes predictions) ──────────
st.sidebar.header("Heuristics")
framework = st.sidebar.selectbox(
    "Framework overhead floor",
    options=["torch", "raw", "nccl", "rccl", "mpi", "jax"],
    index=0,
    help=(
        "Pulls a per-call constant from `model.heuristics.DEFAULT_HEURISTICS"
        ".framework_overhead_ns`. The corpus is `torch.distributed` so the "
        "default is `torch` (~400 µs floor). Switch to `raw` to see the "
        "backend-only prediction (what rccl-tests would measure)."
    ),
)

st.sidebar.header("Cache")
if st.sidebar.button("Rebuild predictions",
                      help="Force re-run predict_tensor_collective on every row."):
    predict_all.clear()
    load_csv.clear()
    st.rerun()
st.sidebar.caption(f"model fingerprint  `{_model_version()[-12:-2]}`")
st.sidebar.caption(backend_caption())

df = predict_all(df_raw, _model_version(), framework)
df = df.dropna(subset=["predicted_us"])

st.sidebar.header("Filters")
primitives_all = sorted(df["primitive"].unique())
prims_sel = st.sidebar.multiselect("Collective", primitives_all, default=primitives_all)

worlds_all = sorted(df["world_size"].unique())
ws_sel = st.sidebar.multiselect("World size (W)", worlds_all, default=worlds_all)

families_all = sorted(df["shape_family"].unique())
fam_sel = st.sidebar.multiselect("Shape family", families_all, default=families_all)

ndims_all = sorted(df["shape_ndim"].unique())
ndim_sel = st.sidebar.multiselect("Shape ndim", ndims_all, default=ndims_all)

dtypes_all = sorted(df["dtype"].unique())
dt_sel = st.sidebar.multiselect("Dtype", dtypes_all, default=dtypes_all)

byte_options = sorted(df["per_rank_bytes"].unique().tolist())
byte_range = st.sidebar.select_slider(
    "Per-rank byte range",
    options=byte_options,
    value=(byte_options[0], byte_options[-1]),
    format_func=_human_bytes,
)

st.sidebar.header("Display")
log_axes = st.sidebar.checkbox("Log axes (global scatter)", value=True)
hide_outliers = st.sidebar.checkbox("Hide |err| > 500%", value=True)

mask = (
    df["primitive"].isin(prims_sel)
    & df["world_size"].isin(ws_sel)
    & df["shape_family"].isin(fam_sel)
    & df["shape_ndim"].isin(ndim_sel)
    & df["dtype"].isin(dt_sel)
    & (df["per_rank_bytes"] >= byte_range[0])
    & (df["per_rank_bytes"] <= byte_range[1])
)
if hide_outliers:
    mask &= df["abs_error_pct"] <= 500
fdf = df[mask].copy()

if fdf.empty:
    st.warning("No rows match the current filter selection.")
    st.stop()

# ────────── KPI row ──────────
kpis = st.columns(4)
kpis[0].metric("Filtered rows", f"{len(fdf):,}",
               delta=f"of {len(df):,} total", delta_color="off")
kpis[1].metric("Median |err|", f"{fdf['abs_error_pct'].median():.1f}%")
kpis[2].metric("Mean |err|", f"{fdf['abs_error_pct'].mean():.1f}%")
within20 = (fdf["abs_error_pct"] <= 20).mean() * 100
kpis[3].metric("Within ±20%", f"{within20:.1f}%",
               delta=f"±50%: {(fdf['abs_error_pct'] <= 50).mean()*100:.1f}%",
               delta_color="off")

# ─── Framework overhead diagnostic ───
st.info(
    "**Heads-up — this corpus is `torch.distributed`, not raw rccl-tests.** "
    "torch.distributed adds ~400 µs of Python/dispatch overhead per "
    "collective call vs raw rccl-tests; the model is calibrated against "
    "rccl-tests, so it under-predicts torch by ~400 µs for messages "
    "below 16 MB. Above 16 MB the corpora converge. The overhead floor "
    "panel below makes the gap concrete; the bias should be read as "
    "'model vs raw RCCL is fine, model vs torch needs a framework floor.'"
)

with st.expander("Per-collective stats (current filters)", expanded=True):
    g = fdf.groupby("primitive").agg(
        rows=("error_pct", "count"),
        md_ape=("abs_error_pct", "median"),
        mean_ape=("abs_error_pct", "mean"),
        max_ape=("abs_error_pct", "max"),
        bias_pct=("error_pct", "median"),
    ).round(1).reset_index()
    g.columns = ["primitive", "rows", "MdAPE %", "MeanAPE %", "MaxAPE %", "median bias %"]
    st.dataframe(g, use_container_width=True, hide_index=True)

st.divider()

# ────────── Framework overhead floor ──────────
st.subheader("Framework overhead floor")
st.caption(
    f"Median measured (`torch.distributed`) vs backend-only vs predicted "
    f"(backend + `{framework}` floor) per per-rank-bytes bucket. The "
    f"`{framework}` floor is pulled from `model.heuristics.DEFAULT_HEURISTICS"
    f".framework_overhead_ns`. Switch the framework in the sidebar to "
    f"toggle the floor on/off."
)
st.plotly_chart(figure_overhead_floor(fdf, framework), use_container_width=True)

st.divider()

# ────────── Global predicted-vs-measured ──────────
st.subheader("Predicted vs Measured")

color_choice = st.radio(
    "Color points by",
    options=["primitive", "shape_family", "world_size", "shape_ndim"],
    horizontal=True, index=0, key="global_scatter_color",
)
st.plotly_chart(figure_global_scatter(fdf, color_choice, log_axes),
                use_container_width=True)

st.divider()

# ────────── Shape sensitivity ──────────
st.subheader("Shape sensitivity at fixed per-rank byte total")
st.caption(
    "Pre-dense-tile-fix the model predicted 50–175× variance across shapes "
    "with identical byte totals. The empirical measurement is "
    "1.00–1.08× — i.e., RCCL is essentially shape-agnostic at fixed total "
    "bytes. Both panels below should now show measured and predicted "
    "essentially flat across shapes within a family."
)

fam_cols = st.columns(2)
fam_prims = sorted(fdf["primitive"].unique())
with fam_cols[0]:
    st.plotly_chart(figure_shape_sensitivity(fdf, "ar", fam_prims),
                    use_container_width=True)
with fam_cols[1]:
    st.plotly_chart(figure_shape_sensitivity(fdf, "subcl", fam_prims),
                    use_container_width=True)

# Spread comparison table
spread_rows = []
for fam_pref, fam_label in [("ar_", "aspect-ratio (4 MiB)"),
                              ("subcl_", "sub-cacheline (256 KiB)"),
                              ("tf_", "transformer (mixed bytes)"),
                              ("1d_", "1D flat (mixed bytes)")]:
    g = fdf[fdf["tag"].str.startswith(fam_pref)]
    if g.empty:
        continue
    sm = g.groupby(["primitive", "world_size", "per_rank_bytes"])["latency_us"]\
            .agg(["min", "max"]).assign(r=lambda x: x["max"] / x["min"])
    sp = g.groupby(["primitive", "world_size", "per_rank_bytes"])["predicted_us"]\
            .agg(["min", "max"]).assign(r=lambda x: x["max"] / x["min"])
    spread_rows.append({
        "family": fam_label,
        "n groups": len(sm),
        "measured spread (max/min)": f"{sm['r'].min():.2f} – {sm['r'].max():.2f}×",
        "model spread (max/min)": f"{sp['r'].min():.2f} – {sp['r'].max():.2f}×",
    })

if spread_rows:
    st.dataframe(pd.DataFrame(spread_rows), use_container_width=True, hide_index=True)
    st.caption(
        "Spread = ratio of max to min latency across all shapes in the "
        "family with identical (primitive, W, per_rank_bytes). 1.0× = "
        "perfectly shape-invariant."
    )

st.divider()

# ────────── Error vs axis ──────────
st.subheader("Error % vs problem axis")

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(figure_error_vs(fdf, "per_rank_bytes", log_x=True),
                    use_container_width=True)
with c2:
    st.plotly_chart(figure_error_vs(fdf, "world_size", log_x=False),
                    use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(figure_error_vs(fdf, "row_bytes", log_x=True,
                                     color_by="shape_family"),
                    use_container_width=True)
    st.caption("row_bytes = shape_n × element_bytes (BF16=2). "
               "Below 64 B = sub-cacheline rows; should NOT correlate with "
               "error after the dense-tile fix.")
with c4:
    st.plotly_chart(figure_error_vs(fdf, "shape_ndim", log_x=False,
                                     color_by="primitive"),
                    use_container_width=True)
    st.caption("ndim=1 (flat) vs ndim=2 (matrix). Difference here would "
               "signal a 2D modeling gap.")

st.divider()

# ────────── Per-shape-family tabs ──────────
st.subheader("Per-shape-family detail")
fam_tabs = st.tabs(sorted(fdf["shape_family"].unique()))
for tab, fam in zip(fam_tabs, sorted(fdf["shape_family"].unique())):
    with tab:
        sub = fdf[fdf["shape_family"] == fam]
        if sub.empty:
            st.info("No rows for this family.")
            continue
        kc = st.columns(4)
        kc[0].metric("rows", f"{len(sub):,}")
        kc[1].metric("Median |err|", f"{sub['abs_error_pct'].median():.1f}%")
        kc[2].metric("Mean |err|", f"{sub['abs_error_pct'].mean():.1f}%")
        kc[3].metric("Median bias", f"{sub['error_pct'].median():+.1f}%")

        # Per-tag stats inside the family
        per_tag = sub.groupby(["tag", "primitive"]).agg(
            rows=("error_pct", "count"),
            shape=("shape", "first"),
            per_rank_B=("per_rank_bytes", "first"),
            med_meas=("latency_us", "median"),
            med_pred=("predicted_us", "median"),
            MdAPE=("abs_error_pct", "median"),
            bias=("error_pct", "median"),
        ).round(1).reset_index()
        per_tag["per_rank_B"] = per_tag["per_rank_B"].map(_human_bytes)
        st.dataframe(per_tag, use_container_width=True, hide_index=True)

st.divider()

# ────────── Raw rows ──────────
with st.expander(f"Filtered rows ({len(fdf):,})", expanded=False):
    show_cols = ["primitive", "world_size", "shape", "dim", "dtype",
                 "shape_family", "per_rank_bytes",
                 "latency_us", "predicted_us", "error_pct", "tag"]
    sample = fdf[show_cols].sort_values(
        ["primitive", "world_size", "per_rank_bytes", "tag"]
    )
    st.dataframe(sample, use_container_width=True, hide_index=True)
