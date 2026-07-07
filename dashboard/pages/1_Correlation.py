"""
Predicted vs Measured — broad correlation across message sizes and channel counts.

Layout (mirrors ROCm/demystify's legacy dashboard style):
  - Top KPI row (rows, MdAPE, Mean APE, % within ±20%)
  - Global scatter: predicted vs measured (log-log, y=x diagonal, ±50% bands)
  - "Error % vs axis" trio: vs msg_bytes, vs world_size, vs nchannels
  - Per-collective tabs with abs-error and signed-bias heatmaps over (msg, NCH)
  - Latency curves vs message size, overlaid measured (markers) and predicted (lines)
"""

import sys
import os
_THIS = os.path.abspath(__file__)
_PAGES = os.path.dirname(_THIS)
_DASHBOARD = os.path.dirname(_PAGES)
_WORKSPACE = os.path.dirname(_DASHBOARD)
sys.path.insert(0, _WORKSPACE)
sys.path.insert(0, _DASHBOARD)  # for `_origami_backend`

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st

from _origami_backend import (                            # noqa: E402
    predict_row, MI300X, MI300X_COMM,
    BACKEND_NAME, BACKEND_KIND, backend_caption,
)


st.set_page_config(page_title="Correlation", layout="wide")
st.title("Predicted vs Measured — Broad Correlation")
st.caption(
    "Sweeps the `data/rccl_master_sweep.csv` benchmark corpus and compares "
    "Origami's per-row prediction against the RCCL measurement, "
    "across message size, world size, and number of channels."
)

SUPPORTED = {"all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast"}
PALETTE = pc.qualitative.Plotly
TIGHT_MARGIN = dict(l=10, r=10, t=44, b=10)


# ────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────

def _model_version() -> str:
    """A fingerprint of the model code on disk.

    Used as an extra cache key so that edits to model/* invalidate the cached
    predictions even though the call signature of predict_all is unchanged.
    """
    root = Path(__file__).resolve().parents[2] / "model"
    mtimes = []
    for p in sorted(root.glob("*.py")):
        try:
            mtimes.append((p.name, p.stat().st_mtime_ns))
        except OSError:
            continue
    return repr(mtimes)


@st.cache_data(show_spinner="Loading benchmark CSV…")
def load_csv() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[2] / "data" / "rccl_master_sweep.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["primitive"].isin(SUPPORTED)]
    df = df[(df["latency_us"] > 0) & (df["msg_bytes"] > 0)]
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Predicting all rows…")
def predict_all(df: pd.DataFrame, model_version: str) -> pd.DataFrame:
    # `model_version` is just a cache key — it isn't read, but its hash is part
    # of @st.cache_data's signature so any change to model/*.py mtimes
    # invalidates the cached prediction frame.
    del model_version
    keys = df[["primitive", "msg_bytes", "world_size", "nchannels"]].drop_duplicates()
    rows = []
    for _, k in keys.iterrows():
        try:
            pred = predict_row(
                k["primitive"], int(k["msg_bytes"]), int(k["world_size"]),
                int(k["nchannels"]), MI300X, MI300X_COMM,
            )
        except Exception:
            pred = float("nan")
        rows.append({**k.to_dict(), "predicted_us": pred})
    pred_df = pd.DataFrame(rows)
    out = df.merge(pred_df, on=["primitive", "msg_bytes", "world_size", "nchannels"], how="left")
    out["error_pct"] = (out["predicted_us"] - out["latency_us"]) / out["latency_us"] * 100
    out["abs_error_pct"] = out["error_pct"].abs()
    out["log2_msg"] = np.log2(out["msg_bytes"]).round().astype(int)
    out = _annotate_regimes(out)
    return out


# ────────────────────────────────────────────────────────────────────
# Regime classification
# ────────────────────────────────────────────────────────────────────

# Per-collective wire-volume factor used to derive an "ideal" bandwidth-bound
# latency (lower bound). Matches the rccl-tests busbw formulas.
_WIRE_FACTOR = {
    "all_reduce":      lambda n: 2 * (n - 1) / n,
    "all_gather":      lambda n: (n - 1),
    "reduce_scatter":  lambda n: (n - 1) / n,
    "all_to_all":      lambda n: (n - 1) / n,
    "broadcast":       lambda n: 1.0,
}

# us — kernel launch + library overhead floor (matches MI300X_COMM)
_LAUNCH_FLOOR_US = MI300X_COMM.launch_overhead_ns / 1000.0
# Aggregate xGMI BW available when (W-1) links saturated at link_bw (GB/s).
# Used as a conservative lower bound on transfer time.
_LINK_BW_GBPS = MI300X_COMM.link_bw_ns  # bytes/ns == GB/s (cycles→ns via clock_ghz)


def _ideal_bw_us(row) -> float:
    """Ideal bandwidth-bound transfer time (us) assuming (W-1) links saturated."""
    prim = row["primitive"]
    n = max(int(row["world_size"]), 2)
    factor_fn = _WIRE_FACTOR.get(prim)
    if factor_fn is None:
        return float("nan")
    wire_bytes = factor_fn(n) * float(row["msg_bytes"])
    agg_bw_gbps = max(n - 1, 1) * _LINK_BW_GBPS
    return (wire_bytes / agg_bw_gbps) / 1e3  # ns→us


def _annotate_regimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ideal_bw_us"] = df.apply(_ideal_bw_us, axis=1)

    # Regime by predicted latency vs the launch floor and the bw lower bound.
    # latency-bound: floor dominates — predicted within 1.5× of launch floor
    # bw-bound:      transfer time ≫ floor and within 3× of the ideal bw bound
    # transition:    everything in between
    pred = df["predicted_us"]
    floor_ratio = pred / _LAUNCH_FLOOR_US
    bw_ratio = pred / df["ideal_bw_us"]

    def classify(fr, br):
        if not np.isfinite(fr) or not np.isfinite(br):
            return "unknown"
        if fr <= 1.5:
            return "latency-bound"
        if br <= 3.0 and fr >= 3.0:
            return "bandwidth-bound"
        return "transition"

    df["regime"] = [classify(fr, br) for fr, br in zip(floor_ratio, bw_ratio)]

    # Coarse buckets that are useful in their own right
    def msg_bucket(b):
        if b <= 64 * 1024:           return "small (≤64 KB)"
        if b <= 1024 * 1024:         return "medium (64 KB–1 MB)"
        if b <= 64 * 1024 * 1024:    return "large (1–64 MB)"
        return "huge (>64 MB)"

    def nch_bucket(n):
        if n <= 2:   return "low (≤2)"
        if n <= 16:  return "mid (4–16)"
        return "high (32+)"

    df["msg_bucket"] = df["msg_bytes"].map(msg_bucket)
    df["nch_bucket"] = df["nchannels"].map(nch_bucket)
    return df


# ────────────────────────────────────────────────────────────────────
# Plot factories
# ────────────────────────────────────────────────────────────────────

def _color_for_primitive(prim: str, primitives: List[str]) -> str:
    return PALETTE[primitives.index(prim) % len(PALETTE)]


def figure_global_scatter(fdf: pd.DataFrame, log_axes: bool) -> go.Figure:
    primitives = sorted(fdf["primitive"].unique())
    fig = go.Figure()
    for p in primitives:
        sub = fdf[fdf["primitive"] == p]
        fig.add_trace(go.Scatter(
            x=sub["latency_us"], y=sub["predicted_us"],
            mode="markers", name=p,
            marker=dict(
                size=np.clip(np.log2(sub["nchannels"].to_numpy() + 1) + 2, 2, 7),
                color=_color_for_primitive(p, primitives),
                opacity=0.5,
                line=dict(width=0),
            ),
            customdata=np.stack([sub["msg_bytes"], sub["world_size"],
                                  sub["nchannels"], sub["error_pct"]], axis=1),
            hovertemplate=(
                f"<b>{p}</b><br>"
                "msg=%{customdata[0]:,d} B<br>"
                "W=%{customdata[1]}, NCH=%{customdata[2]}<br>"
                "measured=%{x:.1f} µs<br>predicted=%{y:.1f} µs<br>"
                "err=%{customdata[3]:+.1f}%<extra></extra>"
            ),
        ))
    lo = float(min(fdf["latency_us"].min(), fdf["predicted_us"].min())) * 0.9
    hi = float(max(fdf["latency_us"].max(), fdf["predicted_us"].max())) * 1.1
    diag_kwargs = dict(mode="lines", hoverinfo="skip", showlegend=True)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi],
                              line=dict(color="black", dash="dash", width=1),
                              name="y = x", **diag_kwargs))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo * 1.5, hi * 1.5],
                              line=dict(color="gray", dash="dot", width=1),
                              name="±50%", **diag_kwargs))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo / 1.5, hi / 1.5],
                              line=dict(color="gray", dash="dot", width=1),
                              name="", showlegend=False, mode="lines", hoverinfo="skip"))
    if log_axes:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    fig.update_layout(
        title="Predicted vs Measured (marker size ∝ NCH)",
        xaxis_title="Measured latency (µs)",
        yaxis_title="Predicted latency (µs)",
        height=540, margin=TIGHT_MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def figure_vs_axis(fdf: pd.DataFrame, axis: str, log_x: bool) -> go.Figure:
    """error_pct (signed) vs one of msg_bytes/world_size/nchannels, colored by primitive."""
    primitives = sorted(fdf["primitive"].unique())
    fig = go.Figure()
    for p in primitives:
        sub = fdf[fdf["primitive"] == p]
        fig.add_trace(go.Scatter(
            x=sub[axis], y=sub["error_pct"],
            mode="markers",
            name=p,
            marker=dict(size=3, color=_color_for_primitive(p, primitives), opacity=0.4),
            customdata=np.stack([sub["world_size"], sub["nchannels"]], axis=1),
            hovertemplate=(
                f"<b>{p}</b><br>"
                f"{axis}=" "%{x}" "<br>W=%{customdata[0]}, NCH=%{customdata[1]}"
                "<br>err=%{y:+.1f}%<extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.add_hline(y=50, line=dict(color="gray", dash="dot", width=1))
    fig.add_hline(y=-50, line=dict(color="gray", dash="dot", width=1))
    if log_x:
        fig.update_xaxes(type="log")
    title_map = {"msg_bytes": "Error % vs message size",
                 "world_size": "Error % vs world size",
                 "nchannels": "Error % vs channels"}
    fig.update_layout(
        title=title_map.get(axis, f"Error % vs {axis}"),
        xaxis_title=axis, yaxis_title="signed err %",
        height=340, margin=TIGHT_MARGIN, showlegend=False,
    )
    return fig


def figure_heatmap(sub: pd.DataFrame, value_col: str, *, signed: bool,
                   title: str) -> go.Figure:
    pivot = (sub.groupby(["log2_msg", "nchannels"])[value_col]
                .median().reset_index())
    mat = pivot.pivot(index="nchannels", columns="log2_msg",
                      values=value_col).sort_index(ascending=False)
    x_labels = [_human_bytes(2**c) for c in mat.columns]
    if signed:
        lim = float(np.nanmax(np.abs(mat.values))) if mat.size else 1.0
        zmin, zmax = -min(200, lim), min(200, lim)
        colorscale = "RdBu"
        colorbar_title = "bias %"
        hover_fmt = "%{z:+.1f}%"
    else:
        zmin = 0
        zmax = min(200, float(np.nanmax(mat.values))) if mat.size else 100
        colorscale = "RdYlGn_r"
        colorbar_title = "|err| %"
        hover_fmt = "%{z:.1f}%"
    fig = go.Figure(data=go.Heatmap(
        z=mat.values, x=x_labels, y=[str(n) for n in mat.index],
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar=dict(title=colorbar_title),
        hovertemplate="msg=%{x}<br>NCH=%{y}<br>" + hover_fmt + "<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="message size", yaxis_title="channels (NCH)",
        height=400, margin=TIGHT_MARGIN,
    )
    return fig


def figure_latency_curves(cdf: pd.DataFrame, primitive: str, world: int) -> go.Figure:
    agg = (cdf.groupby(["nchannels", "msg_bytes"])
              .agg(measured=("latency_us", "median"),
                   predicted=("predicted_us", "median"))
              .reset_index()
              .sort_values(["nchannels", "msg_bytes"]))
    fig = go.Figure()
    for i, nch in enumerate(sorted(agg["nchannels"].unique())):
        color = PALETTE[i % len(PALETTE)]
        sub = agg[agg["nchannels"] == nch]
        fig.add_trace(go.Scatter(
            x=sub["msg_bytes"], y=sub["measured"], mode="markers",
            marker=dict(size=4, color=color, symbol="circle-open", line=dict(width=1.5)),
            name=f"meas NCH={nch}", legendgroup=str(nch),
        ))
        fig.add_trace(go.Scatter(
            x=sub["msg_bytes"], y=sub["predicted"], mode="lines",
            line=dict(color=color, width=2),
            name=f"pred NCH={nch}", legendgroup=str(nch),
        ))
    fig.update_xaxes(type="log", title="msg_bytes")
    fig.update_yaxes(type="log", title="latency (µs)")
    fig.update_layout(
        title=f"{primitive} @ W={world} — measured (markers) vs predicted (lines)",
        height=520, margin=TIGHT_MARGIN, legend=dict(orientation="v"),
    )
    return fig


def _human_bytes(n: int) -> str:
    if n >= 1024**3: return f"{n // 1024**3} GB"
    if n >= 1024**2: return f"{n // 1024**2} MB"
    if n >= 1024:    return f"{n // 1024} KB"
    return f"{n} B"


# ────────────────────────────────────────────────────────────────────
# Per-regime plot factories
# ────────────────────────────────────────────────────────────────────

def figure_regime_heatmap(fdf: pd.DataFrame, regime_col: str,
                          regime_order: List[str], metric: str = "abs_error_pct",
                          stat: str = "median") -> go.Figure:
    """Heatmap (regime × collective) → MdAPE / MeanAPE / signed-bias."""
    agg = (fdf.groupby([regime_col, "primitive"])[metric]
                .agg(stat).reset_index())
    mat = agg.pivot(index=regime_col, columns="primitive", values=metric)
    mat = mat.reindex(regime_order)
    if metric == "error_pct":
        lim = float(np.nanmax(np.abs(mat.values))) if mat.size else 1.0
        zmin, zmax = -min(200, lim), min(200, lim)
        colorscale, cbar = "RdBu", "median bias %"
        hover_fmt = "%{z:+.1f}%"
    else:
        zmin = 0
        zmax = min(150, float(np.nanmax(mat.values))) if mat.size else 80
        colorscale, cbar = "RdYlGn_r", f"{stat} |err| %"
        hover_fmt = "%{z:.1f}%"
    fig = go.Figure(data=go.Heatmap(
        z=mat.values,
        x=list(mat.columns),
        y=list(mat.index),
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=cbar),
        text=[[f"{v:.0f}" if np.isfinite(v) else "" for v in row] for row in mat.values],
        texttemplate="%{text}",
        hovertemplate="regime=%{y}<br>primitive=%{x}<br>" + hover_fmt + "<extra></extra>",
    ))
    fig.update_layout(
        height=320, margin=TIGHT_MARGIN,
        xaxis_title="collective", yaxis_title="regime",
    )
    return fig


def figure_regime_scatter(fdf: pd.DataFrame, regime_col: str,
                           regime_order: List[str], log_axes: bool) -> go.Figure:
    """Predicted vs measured colored by regime (one trace per regime)."""
    fig = go.Figure()
    palette = pc.qualitative.Bold
    for i, r in enumerate(regime_order):
        sub = fdf[fdf[regime_col] == r]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["latency_us"], y=sub["predicted_us"],
            mode="markers", name=f"{r} (n={len(sub):,})",
            marker=dict(size=3, color=palette[i % len(palette)], opacity=0.55,
                        line=dict(width=0)),
            customdata=np.stack([sub["primitive"], sub["world_size"],
                                  sub["nchannels"], sub["msg_bytes"],
                                  sub["error_pct"]], axis=1),
            hovertemplate=(
                "%{customdata[0]} W=%{customdata[1]} NCH=%{customdata[2]}<br>"
                "msg=%{customdata[3]:,d} B<br>"
                "meas=%{x:.1f} µs  pred=%{y:.1f} µs<br>"
                "err=%{customdata[4]:+.1f}%<extra></extra>"
            ),
        ))
    lo = float(min(fdf["latency_us"].min(), fdf["predicted_us"].min())) * 0.9
    hi = float(max(fdf["latency_us"].max(), fdf["predicted_us"].max())) * 1.1
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                              line=dict(color="black", dash="dash", width=1),
                              name="y = x", hoverinfo="skip"))
    if log_axes:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    fig.update_layout(
        title=f"Predicted vs measured by {regime_col}",
        xaxis_title="measured µs", yaxis_title="predicted µs",
        height=440, margin=TIGHT_MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def regime_stats_table(fdf: pd.DataFrame, regime_col: str,
                       regime_order: List[str]) -> pd.DataFrame:
    g = (fdf.groupby([regime_col, "primitive"])
            .agg(rows=("error_pct", "count"),
                 md_ape=("abs_error_pct", "median"),
                 mean_ape=("abs_error_pct", "mean"),
                 within20=("abs_error_pct", lambda s: (s <= 20).mean() * 100),
                 bias=("error_pct", "median"))
            .reset_index())
    # Add an "all" row per regime
    all_g = (fdf.groupby(regime_col)
                .agg(rows=("error_pct", "count"),
                     md_ape=("abs_error_pct", "median"),
                     mean_ape=("abs_error_pct", "mean"),
                     within20=("abs_error_pct", lambda s: (s <= 20).mean() * 100),
                     bias=("error_pct", "median"))
                .reset_index())
    all_g["primitive"] = "— all —"
    g = pd.concat([g, all_g], ignore_index=True)
    g[regime_col] = pd.Categorical(g[regime_col], regime_order, ordered=True)
    g = g.sort_values([regime_col, "primitive"])
    g = g.rename(columns={
        "md_ape": "MdAPE %", "mean_ape": "MeanAPE %",
        "within20": "within ±20 %", "bias": "median bias %",
    })
    for c in ("MdAPE %", "MeanAPE %", "within ±20 %", "median bias %"):
        g[c] = g[c].round(1)
    return g


# ────────────────────────────────────────────────────────────────────
# Page body
# ────────────────────────────────────────────────────────────────────

df_raw = load_csv()
if df_raw.empty:
    st.error("Could not find `data/rccl_master_sweep.csv`. "
             "Place the benchmark CSV there and reload.")
    st.stop()

df = predict_all(df_raw, _model_version())

st.sidebar.header("Cache")
if st.sidebar.button("Rebuild predictions",
                      help="Force re-run predict_row() across every CSV row "
                           "(use this if you edited model code and the "
                           "dashboard still shows stale numbers)."):
    predict_all.clear()
    load_csv.clear()
    st.rerun()
st.sidebar.caption(f"model fingerprint  `{_model_version()[-12:-2]}`")
st.sidebar.caption(backend_caption())

st.sidebar.header("Filters")
primitives_all = sorted(df["primitive"].unique())
prims_sel = st.sidebar.multiselect("Collective", primitives_all, default=primitives_all)

worlds_all = sorted(df["world_size"].unique())
ws_sel = st.sidebar.multiselect(
    "World size (W)", worlds_all,
    default=[w for w in worlds_all if w >= 2],
)

nchs_all = sorted(df["nchannels"].unique())
nch_sel = st.sidebar.multiselect("Channels (NCH)", nchs_all, default=nchs_all)

msg_options = sorted(df["msg_bytes"].unique().tolist())
msg_range = st.sidebar.select_slider(
    "Msg size range",
    options=msg_options,
    value=(msg_options[0], msg_options[-1]),
    format_func=_human_bytes,
)

st.sidebar.header("Display")
log_axes = st.sidebar.checkbox("Log axes (global scatter)", value=True)
log_x_vs_msg = st.sidebar.checkbox("Log x on “error vs msg” plot", value=True)
hide_outliers = st.sidebar.checkbox("Hide |err| > 500%", value=True)

st.sidebar.header("Heavy plots")
st.sidebar.caption(
    "Non-aggregated scatter plots draw one marker per CSV row "
    "(~30k points), which is slow to render and makes panning/filtering "
    "laggy. They're off by default — toggle individually below or "
    "use the master switch."
)
show_all_scatter = st.sidebar.checkbox(
    "Render all non-aggregated scatters", value=False,
    help="Master switch — overrides the individual scatter toggles below.",
)
show_global_scatter = (
    show_all_scatter
    or st.sidebar.checkbox("Global predicted-vs-measured", value=False)
)
show_err_vs_axis = (
    show_all_scatter
    or st.sidebar.checkbox("Error % vs (msg / W / NCH) trio", value=False)
)
show_regime_scatter = (
    show_all_scatter
    or st.sidebar.checkbox("Predicted-vs-measured by regime", value=False)
)

mask = (
    df["primitive"].isin(prims_sel)
    & df["world_size"].isin(ws_sel)
    & df["nchannels"].isin(nch_sel)
    & (df["msg_bytes"] >= msg_range[0])
    & (df["msg_bytes"] <= msg_range[1])
)
if hide_outliers:
    mask &= df["abs_error_pct"] <= 500
fdf = df[mask].copy().dropna(subset=["predicted_us"])

if fdf.empty:
    st.warning("No rows match the current filter selection.")
    st.stop()

# ─── Top KPI row ───
kpis = st.columns(4)
kpis[0].metric("Filtered rows", f"{len(fdf):,}",
               delta=f"of {len(df):,} total", delta_color="off")
kpis[1].metric("Median |err|", f"{fdf['abs_error_pct'].median():.1f}%")
kpis[2].metric("Mean |err|", f"{fdf['abs_error_pct'].mean():.1f}%")
within20 = (fdf["abs_error_pct"] <= 20).mean() * 100
kpis[3].metric("Within ±20%", f"{within20:.1f}%",
               delta=f"±50%: {(fdf['abs_error_pct'] <= 50).mean()*100:.1f}%",
               delta_color="off")

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

# ─── Global scatter (gated — ~30k points is slow to render) ───
st.subheader("Global predicted-vs-measured scatter")
if show_global_scatter:
    st.plotly_chart(figure_global_scatter(fdf, log_axes), use_container_width=True)
else:
    st.info(
        f"Hidden by default ({len(fdf):,} non-aggregated points). "
        "Use the **Heavy plots** section in the sidebar to render it, "
        "or click below."
    )
    if st.button("Show graph (global scatter)", key="show_global_btn"):
        st.session_state["show_global_via_btn"] = True
    if st.session_state.get("show_global_via_btn"):
        st.plotly_chart(figure_global_scatter(fdf, log_axes), use_container_width=True)

st.divider()

# ─── Per-regime correlation ───
st.subheader("Correlation per regime")
st.caption(
    "Three orthogonal ways to slice the corpus. "
    "**Latency / transition / bandwidth-bound** is model-driven: "
    f"latency-bound when predicted ≤ 1.5× launch floor ({_LAUNCH_FLOOR_US:.0f} µs), "
    "bandwidth-bound when predicted is within 3× of the ideal (N-1)-link "
    "saturation lower bound. Bucket regimes by msg size and by NCH are coarse "
    "but commonly the most actionable."
)

REGIME_TABS = {
    "By model regime (latency / bw / transition)": (
        "regime",
        ["latency-bound", "transition", "bandwidth-bound", "unknown"],
    ),
    "By message size": (
        "msg_bucket",
        ["small (≤64 KB)", "medium (64 KB–1 MB)", "large (1–64 MB)", "huge (>64 MB)"],
    ),
    "By channels (NCH)": (
        "nch_bucket",
        ["low (≤2)", "mid (4–16)", "high (32+)"],
    ),
}

regime_tab_objs = st.tabs(list(REGIME_TABS.keys()))
for tab, (label, (col, order)) in zip(regime_tab_objs, REGIME_TABS.items()):
    with tab:
        present = [r for r in order if r in fdf[col].unique()]
        if not present:
            st.info("No rows in any of these regimes under current filters.")
            continue

        # Two-up: MdAPE heatmap + signed-bias heatmap
        h1, h2 = st.columns(2)
        with h1:
            st.plotly_chart(
                figure_regime_heatmap(fdf, col, present,
                                      metric="abs_error_pct", stat="median"),
                use_container_width=True,
            )
            st.caption("Median |error %| — green = accurate, red = inaccurate")
        with h2:
            st.plotly_chart(
                figure_regime_heatmap(fdf, col, present,
                                      metric="error_pct", stat="median"),
                use_container_width=True,
            )
            st.caption("Median signed bias — blue = model under-predicts, "
                       "red = over-predicts")

        # Stats table + colored scatter (scatter gated — ~30k points)
        st.dataframe(regime_stats_table(fdf, col, present),
                     use_container_width=True, hide_index=True)
        if show_regime_scatter:
            st.plotly_chart(
                figure_regime_scatter(fdf, col, present, log_axes=log_axes),
                use_container_width=True,
            )
        else:
            btn_key = f"show_regime_btn_{col}"
            state_key = f"show_regime_via_btn_{col}"
            st.caption(
                "Predicted-vs-measured scatter hidden "
                f"({len(fdf):,} non-aggregated points). "
                "Use the **Heavy plots** sidebar section or click below."
            )
            if st.button("Show graph (regime scatter)", key=btn_key):
                st.session_state[state_key] = True
            if st.session_state.get(state_key):
                st.plotly_chart(
                    figure_regime_scatter(fdf, col, present, log_axes=log_axes),
                    use_container_width=True,
                )

st.divider()

# ─── Error vs axis trio (gated — three ~30k-point scatters) ───
st.subheader("Error % vs problem axis")

def _render_error_vs_axis_trio() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(figure_vs_axis(fdf, "msg_bytes", log_x_vs_msg),
                        use_container_width=True)
    with c2:
        st.plotly_chart(figure_vs_axis(fdf, "world_size", False),
                        use_container_width=True)
    with c3:
        st.plotly_chart(figure_vs_axis(fdf, "nchannels", False),
                        use_container_width=True)


if show_err_vs_axis:
    _render_error_vs_axis_trio()
else:
    st.info(
        f"Three non-aggregated scatters hidden ({len(fdf):,} points each). "
        "Use the **Heavy plots** sidebar section or click below."
    )
    if st.button("Show graphs (error vs axis trio)", key="show_err_axis_btn"):
        st.session_state["show_err_axis_via_btn"] = True
    if st.session_state.get("show_err_axis_via_btn"):
        _render_error_vs_axis_trio()

st.divider()

# ─── Per-collective heatmaps ───
st.subheader("Per-collective heatmaps over (msg, NCH)")
st.caption("Median across the world-size dimension; current filters applied.")

prim_tabs = st.tabs(sorted(fdf["primitive"].unique()))
for tab, p in zip(prim_tabs, sorted(fdf["primitive"].unique())):
    with tab:
        sub = fdf[fdf["primitive"] == p]
        if sub.empty:
            st.info("No rows for this collective under current filters.")
            continue
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                figure_heatmap(sub, "abs_error_pct", signed=False,
                               title=f"{p} — median |error %|"),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                figure_heatmap(sub, "error_pct", signed=True,
                               title=f"{p} — signed bias (red over-, blue under-)"),
                use_container_width=True,
            )

st.divider()

# ─── Latency curves ───
st.subheader("Latency vs message size")
st.caption("Markers = measured (median across replicates); lines = predicted.")

curve_cols = st.columns([2, 1])
with curve_cols[0]:
    curve_prim = st.selectbox("Collective", sorted(fdf["primitive"].unique()),
                               key="curve_prim")
with curve_cols[1]:
    valid_ws = sorted(fdf[fdf["primitive"] == curve_prim]["world_size"].unique())
    curve_w = st.selectbox("World size", valid_ws, index=0, key="curve_w")

cdf = fdf[(fdf["primitive"] == curve_prim) & (fdf["world_size"] == curve_w)]
if cdf.empty:
    st.info("No data for this selection.")
else:
    st.plotly_chart(figure_latency_curves(cdf, curve_prim, int(curve_w)),
                     use_container_width=True)

st.divider()

# ─── Raw filtered table ───
with st.expander(f"Filtered rows ({len(fdf):,})", expanded=False):
    show_cols = ["primitive", "world_size", "nchannels", "msg_bytes",
                 "latency_us", "predicted_us", "error_pct"]
    sample = fdf[show_cols].sort_values(
        ["primitive", "world_size", "nchannels", "msg_bytes"]
    )
    st.dataframe(sample.head(2000), use_container_width=True, hide_index=True)
    if len(sample) > 2000:
        st.caption(f"Showing first 2,000 of {len(sample):,} rows.")
