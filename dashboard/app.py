"""
Interactive comm cost model explorer.
Run: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import csv
from math import ceil
from collections import defaultdict

from model.hardware import MI300X, MI300X_COMM, Hardware, CommHardware
from model.types import CommProblem, CommConfig, CACHELINE_BYTES, dtype_bytes, DataType
from model.collective import compute_collective_latency, COLLECTIVE_LAYOUTS, _is_ring_layout
from model.latency import compute_wg_tile_latency, resolve_work_graph, compute_iter_times
from model.primitives import Signal, Wait

st.set_page_config(page_title="Comm Cost Model Explorer", layout="wide")
st.title("Communication Cost Model — Step-by-Step Explorer")

# ─── Sidebar: knobs ───
st.sidebar.header("Parameters")

collective = st.sidebar.selectbox("Collective", list(COLLECTIVE_LAYOUTS.keys()))
world_size = st.sidebar.selectbox("World Size (N)", [2, 4, 8], index=2)
num_channels = st.sidebar.select_slider("Channels (NCH)", options=[1,2,4,8,16,32,64,128], value=32)

st.sidebar.header("Tensor Shape")
use_2d = st.sidebar.checkbox("2D Tensor", value=False)
if use_2d:
    tensor_m = st.sidebar.number_input("M (rows)", min_value=1, value=4096, step=256)
    tensor_n = st.sidebar.number_input("N (columns, contiguous)", min_value=1, value=4096, step=256)
    split_dim = st.sidebar.radio("Split dimension", [0, 1],
                                  format_func=lambda x: f"M (rows → [M/{world_size}, N])" if x == 0
                                  else f"N (cols → [M, N/{world_size}])")
    msg_bytes = tensor_m * tensor_n * dtype_bytes(DataType.BF16)
    st.sidebar.caption(f"Total: {msg_bytes/(1024**2):.1f} MB ({tensor_m}×{tensor_n} BF16)")
else:
    msg_bytes = st.sidebar.select_slider("Message Size",
        options=[1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824],
        value=67108864,
        format_func=lambda x: f"{x/(1024**2):.0f} MB" if x >= 1048576 else f"{x/1024:.0f} KB")
    tensor_m = 1
    tensor_n = msg_bytes // dtype_bytes(DataType.BF16)
    split_dim = 0

st.sidebar.header("Hardware Knobs")
# The model is cycle-based internally; sliders accept human-friendly
# ns/µs/GB-per-s units and we convert through the GPU frequency to
# cycles / bytes-per-cycle for the model.
link_bw_gbps_ns = st.sidebar.slider(
    "Link payload BW (GB/s)", 10.0, 60.0,
    MI300X_COMM.link_bw_ns, 1.0,
)
mshr_depth = st.sidebar.slider("MSHR depth per wave", 4, 24, MI300X.mshr_depth_per_wave)
waves_per_wg = st.sidebar.slider("Waves per WG", 1, 16, MI300X.waves_per_wg)
xgmi_latency_ns = st.sidebar.slider(
    "xGMI RTT (ns)", 100, 2000, int(MI300X.xgmi_latency_ns), 50,
)
launch_overhead_us = st.sidebar.slider(
    "Launch overhead (µs)", 0, 200,
    int(MI300X_COMM.launch_overhead_ns / 1000), 5,
)

# Build hardware with knobs. The GPU frequency (MI300X.clock_hz) governs
# every ns/µs/GB/s ↔ cycle/per-cycle conversion below.
hw = Hardware(
    arch=MI300X.arch, num_cu=MI300X.num_cu, num_xcd=MI300X.num_xcd,
    cu_per_xcd=MI300X.cu_per_xcd, clock_ghz=MI300X.clock_ghz,
    vmem_issue_rate=MI300X.vmem_issue_rate, valu_rate=MI300X.valu_rate,
    tcp_capacity_bytes=MI300X.tcp_capacity_bytes, tcp_bw=MI300X.tcp_bw,
    mshr_depth_per_wave=mshr_depth, waves_per_wg=waves_per_wg,
    xgmi_latency_cycles=MI300X.ns_to_cycles(float(xgmi_latency_ns)),
    l2_capacity_bytes=MI300X.l2_capacity_bytes, l2_bw_per_cu=MI300X.l2_bw_per_cu,
    mall_capacity_bytes=MI300X.mall_capacity_bytes, mall_bw=MI300X.mall_bw,
    hbm_read_bw=MI300X.hbm_read_bw, hbm_write_bw=MI300X.hbm_write_bw,
    hbm_capacity_bytes=MI300X.hbm_capacity_bytes,
)
comm_hw = CommHardware(
    link_bw=hw.rate_per_cycle_from_per_ns(link_bw_gbps_ns),
    num_peer_links=world_size - 1,
    num_sdma_engines=14,
    sdma_read_bw=hw.rate_per_cycle_from_per_ns(49.5),
    sdma_write_bw=hw.rate_per_cycle_from_per_ns(23.6),
    atomic_latency_cycles=MI300X_COMM.atomic_latency_cycles,
    launch_overhead_cycles=hw.us_to_cycles(float(launch_overhead_us)),
    clock_ghz=hw.clock_ghz,
)

problem = CommProblem(M=tensor_m, N=tensor_n, num_gpus=world_size, split_dim=split_dim)
config = CommConfig(num_wgs=num_channels)

# ─── Compute prediction ───
# Model is in cycles; convert once here for the dashboard's ns/µs displays.
# Cycle → time is mediated by the GPU frequency (hw.clock_hz).
layout_fn = COLLECTIVE_LAYOUTS[collective]
layout = layout_fn(world_size)
pred_cycles = compute_collective_latency(collective, problem, config, hw, comm_hw)
pred_ns = hw.cycles_to_ns(pred_cycles)
pred_us = hw.cycles_to_us(pred_cycles)

# ─── Step 0: Hardware Topology ───
st.header("0. Hardware — MI300X Physical Picture")

st.markdown(
    "Every bandwidth (`B/ns` ≡ `GB/s`) and latency in the cost model traces to one of "
    "these constants. Aggregate rates show the device-wide total; per-CU rates show what "
    "one workgroup sees once you account for sharing."
)

hw_mem_tab, hw_top_tab, hw_spec_tab = st.tabs(
    ["Memory hierarchy", "xGMI topology", "All constants"]
)

# ── Memory hierarchy: layered stack from registers → HBM, with xGMI/SDMA side-arrows ──
with hw_mem_tab:
    fig_mem = go.Figure()

    # Layered horizontal bars, top to bottom: VGPRs, TCP, L2, MALL, HBM
    levels = [
        {
            "name": "VGPRs / Registers",
            "fill": "#FFEBEE",
            "edge": "#C62828",
            "lines": [
                f"<b>{hw.num_cu} CUs</b> across {hw.num_xcd} XCDs ({hw.cu_per_xcd}/XCD)",
                f"@ {hw.clock_ghz} GHz, 64-lane SIMD",
                f"VMEM issue: <b>{hw.vmem_issue_rate_ns} instr/ns/CU</b>  •  "
                f"VALU: <b>{hw.valu_rate_ns:.1f} elts/ns/CU</b>",
            ],
        },
        {
            "name": "TCP / vL1D",
            "fill": "#FFF3E0",
            "edge": "#EF6C00",
            "lines": [
                f"{hw.tcp_capacity_bytes//1024} KB per CU  •  "
                f"<b>{hw.tcp_bw_ns} GB/s per CU</b>",
                f"= aggregate <b>{hw.tcp_bw_ns * hw.num_cu / 1000:.1f} TB/s</b> "
                f"(if all {hw.num_cu} CUs active)",
            ],
        },
        {
            "name": "L2 / TCC",
            "fill": "#FFF9C4",
            "edge": "#F9A825",
            "lines": [
                f"{hw.l2_capacity_bytes//(1024*1024)} MB per XCD × {hw.num_xcd} XCDs "
                f"= <b>{hw.l2_capacity_bytes*hw.num_xcd//(1024*1024)} MB</b> total",
                f"<b>{hw.l2_bw_per_cu_ns} GB/s per CU</b>  •  "
                f"aggregate <b>{hw.l2_bw_per_cu_ns * hw.num_cu / 1000:.1f} TB/s</b>",
            ],
        },
        {
            "name": "MALL (Infinity Cache)",
            "fill": "#E8F5E9",
            "edge": "#2E7D32",
            "lines": [
                f"<b>{hw.mall_capacity_bytes//(1024*1024)} MB</b> device-wide",
                f"BW <b>{hw.mall_bw_ns} GB/s</b>  ≈ HBM read rate (on the path)",
            ],
        },
        {
            "name": "HBM3",
            "fill": "#E3F2FD",
            "edge": "#1565C0",
            "lines": [
                f"<b>{hw.hbm_capacity_bytes//(1024**3)} GB</b>",
                f"Read <b>{hw.hbm_read_bw_ns} GB/s ({hw.hbm_read_bw_ns/1000:.2f} TB/s)</b>  •  "
                f"Write <b>{hw.hbm_write_bw_ns} GB/s ({hw.hbm_write_bw_ns/1000:.2f} TB/s)</b>",
            ],
        },
    ]

    box_w = 8.0
    box_h = 1.2
    gap = 0.35
    n = len(levels)
    total_h = n * box_h + (n - 1) * gap

    for i, lvl in enumerate(levels):
        y_top = total_h - i * (box_h + gap)
        y_bot = y_top - box_h
        fig_mem.add_shape(
            type="rect", x0=0, y0=y_bot, x1=box_w, y1=y_top,
            line=dict(color=lvl["edge"], width=2),
            fillcolor=lvl["fill"], layer="below",
        )
        label = f"<b>{lvl['name']}</b><br>" + "<br>".join(lvl["lines"])
        fig_mem.add_annotation(
            x=box_w / 2, y=(y_top + y_bot) / 2,
            text=label, showarrow=False,
            font=dict(size=12, color="#222"),
            align="center", xanchor="center", yanchor="middle",
        )
        # Vertical connector between layers
        if i < n - 1:
            mid_y = y_bot - gap / 2
            fig_mem.add_annotation(
                x=box_w / 2, y=mid_y,
                text="<b>↕</b>", showarrow=False,
                font=dict(size=18, color="#555"),
            )

    # xGMI side-arrow
    xgmi_label = (
        f"<b>xGMI</b><br>"
        f"{comm_hw.num_peer_links} links × <b>{comm_hw.link_bw_ns:.1f} GB/s</b> payload<br>"
        f"= <b>{comm_hw.num_peer_links * comm_hw.link_bw_ns:.0f} GB/s</b> egress<br>"
        f"RTT <b>{hw.xgmi_latency_ns:.0f} ns</b>  •  "
        f"atomic <b>{comm_hw.atomic_latency_ns:.0f} ns</b>"
    )
    # Arrow from HBM box edge to xGMI label
    hbm_y_mid = box_h / 2
    fig_mem.add_annotation(
        x=box_w + 2.6, y=hbm_y_mid + 0.6,
        ax=box_w, ay=hbm_y_mid + 0.6,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#9C27B0",
    )
    fig_mem.add_shape(
        type="rect", x0=box_w + 1.2, y0=hbm_y_mid - 0.55,
        x1=box_w + 4.5, y1=hbm_y_mid + 1.7,
        line=dict(color="#9C27B0", width=2),
        fillcolor="#F3E5F5", layer="below",
    )
    fig_mem.add_annotation(
        x=box_w + 2.85, y=hbm_y_mid + 0.6,
        text=xgmi_label, showarrow=False,
        font=dict(size=11, color="#222"),
        align="center", xanchor="center", yanchor="middle",
    )

    # SDMA side-arrow at L2 level for visual separation
    sdma_label = (
        f"<b>SDMA</b><br>"
        f"{comm_hw.num_sdma_engines} engines<br>"
        f"Read <b>{comm_hw.sdma_read_bw_ns:.1f} GB/s/link</b><br>"
        f"Write <b>{comm_hw.sdma_write_bw_ns:.1f} GB/s/link</b>"
    )
    sdma_y = total_h - 2 * (box_h + gap) + box_h / 2  # MALL level approx
    fig_mem.add_annotation(
        x=box_w + 2.6, y=sdma_y,
        ax=box_w, ay=sdma_y,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#00897B",
    )
    fig_mem.add_shape(
        type="rect", x0=box_w + 1.2, y0=sdma_y - 0.9,
        x1=box_w + 4.5, y1=sdma_y + 0.9,
        line=dict(color="#00897B", width=2),
        fillcolor="#E0F2F1", layer="below",
    )
    fig_mem.add_annotation(
        x=box_w + 2.85, y=sdma_y,
        text=sdma_label, showarrow=False,
        font=dict(size=11, color="#222"),
        align="center", xanchor="center", yanchor="middle",
    )

    # MSHR/xGMI BW cap callout
    mshr_bw_global = (hw.mshr_depth_per_wave * hw.waves_per_wg * CACHELINE_BYTES) / hw.xgmi_latency_ns
    cap_label = (
        f"<b>per-WG remote-read cap</b><br>"
        f"MSHR × waves × CL / RTT<br>"
        f"= {hw.mshr_depth_per_wave} × {hw.waves_per_wg} × {CACHELINE_BYTES} / {hw.xgmi_latency_ns:.0f}<br>"
        f"= <b>{mshr_bw_global:.2f} GB/s</b> per WG"
    )
    cap_y = total_h - 4 * (box_h + gap) + box_h / 2  # HBM level approx
    fig_mem.add_shape(
        type="rect", x0=box_w + 1.2, y0=cap_y - 1.1,
        x1=box_w + 4.5, y1=cap_y + 1.1,
        line=dict(color="#FB8C00", width=2, dash="dot"),
        fillcolor="#FFF8E1", layer="below",
    )
    fig_mem.add_annotation(
        x=box_w + 2.85, y=cap_y,
        text=cap_label, showarrow=False,
        font=dict(size=11, color="#222"),
        align="center", xanchor="center", yanchor="middle",
    )

    fig_mem.update_xaxes(visible=False, range=[-0.3, box_w + 4.9])
    fig_mem.update_yaxes(visible=False, range=[-0.3, total_h + 0.4])
    fig_mem.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white",
        title="Per-GPU memory hierarchy (all rates in GB/s)",
    )
    st.plotly_chart(fig_mem, use_container_width=True)

    st.caption(
        "Each layer shows the resource the cost model debits when a primitive crosses it. "
        "Per-CU rates govern individual workgroup throughput; aggregates govern the device "
        "ceiling. The MSHR-derived per-WG remote-read cap is the limit that often binds in "
        "the model — see how it compares with the per-link xGMI rate."
    )

# ── xGMI topology: N GPUs in a ring/clique, links labeled ──
with hw_top_tab:
    fig_top = go.Figure()
    N = world_size
    radius = 1.0
    # Place GPUs on a circle
    angles = [2 * np.pi * i / N - np.pi / 2 for i in range(N)]
    xs = [radius * np.cos(a) for a in angles]
    ys = [radius * np.sin(a) for a in angles]

    # Draw all-to-all edges (MI300X has fully-connected 8-GPU xGMI: 7 peers per rank)
    for i in range(N):
        for j in range(i + 1, N):
            fig_top.add_trace(go.Scatter(
                x=[xs[i], xs[j]], y=[ys[i], ys[j]],
                mode="lines",
                line=dict(color="#9C27B0", width=2),
                hoverinfo="skip", showlegend=False,
            ))
            # Label one link with the BW (only annotate every link if W<=4 to avoid clutter)
            if N <= 4 or (i == 0 and j == 1):
                mx, my = (xs[i] + xs[j]) / 2, (ys[i] + ys[j]) / 2
                fig_top.add_annotation(
                    x=mx, y=my,
                    text=f"<b>{comm_hw.link_bw_ns:.1f} GB/s</b>",
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#9C27B0",
                    font=dict(size=10),
                )

    # GPU nodes
    fig_top.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(size=70, color="#1565C0",
                    line=dict(color="#0D47A1", width=3)),
        text=[f"GPU{i}" for i in range(N)],
        textfont=dict(size=12, color="white"),
        textposition="middle center",
        hovertext=[f"Rank {i}<br>{comm_hw.num_peer_links} xGMI peers" for i in range(N)],
        hoverinfo="text",
        showlegend=False,
    ))

    edges = N * (N - 1) // 2
    aggregate_egress = (N - 1) * comm_hw.link_bw_ns  # per GPU, GB/s
    bisection = (N // 2) * (N // 2) * comm_hw.link_bw_ns if N >= 2 else 0

    fig_top.update_xaxes(visible=False, range=[-1.6, 1.6])
    fig_top.update_yaxes(visible=False, range=[-1.6, 1.6], scaleanchor="x", scaleratio=1)
    fig_top.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        title=(
            f"xGMI topology @ W={N} — fully-connected, {edges} bidirectional links<br>"
            f"<sub>per-GPU egress = (N−1) × link_bw = {aggregate_egress:.0f} GB/s   •   "
            f"bisection ≈ {bisection:.0f} GB/s</sub>"
        ),
    )
    st.plotly_chart(fig_top, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("xGMI peers / GPU", f"{N-1}",
                 f"of {comm_hw.num_peer_links} max")
    col_b.metric("Per-link payload BW", f"{comm_hw.link_bw_ns:.1f} GB/s",
                 f"{comm_hw.link_bw_ns*1.23:.1f} GB/s wire (×1.23 framing)")
    col_c.metric("Per-GPU aggregate egress", f"{aggregate_egress:.0f} GB/s",
                 f"if all {N-1} links saturated")

# ── Spec table ──
with hw_spec_tab:
    import pandas as pd
    on_chip = pd.DataFrame([
        {"resource": "CUs", "count": hw.num_cu, "value": f"{hw.cu_per_xcd}/XCD × {hw.num_xcd} XCDs",
         "rate / spec": f"@ {hw.clock_ghz} GHz"},
        {"resource": "VMEM issue", "count": "per CU", "value": f"{hw.vmem_issue_rate_ns} instr/ns",
         "rate / spec": f"aggr {hw.vmem_issue_rate_ns*hw.num_cu:.0f} instr/ns"},
        {"resource": "VALU", "count": "per CU", "value": f"{hw.valu_rate_ns:.1f} elts/ns",
         "rate / spec": f"aggr {hw.valu_rate_ns*hw.num_cu/1000:.1f} Telts/ns"},
        {"resource": "TCP/vL1D", "count": "per CU",
         "value": f"{hw.tcp_capacity_bytes//1024} KB cap",
         "rate / spec": f"{hw.tcp_bw_ns} GB/s per CU"},
        {"resource": "L2/TCC", "count": f"{hw.num_xcd} XCDs",
         "value": f"{hw.l2_capacity_bytes//(1024*1024)} MB/XCD ({hw.l2_capacity_bytes*hw.num_xcd//(1024*1024)} MB total)",
         "rate / spec": f"{hw.l2_bw_per_cu_ns} GB/s per CU"},
        {"resource": "MALL (Infinity Cache)", "count": "device-wide",
         "value": f"{hw.mall_capacity_bytes//(1024*1024)} MB",
         "rate / spec": f"{hw.mall_bw_ns} GB/s"},
        {"resource": "HBM3", "count": "device-wide",
         "value": f"{hw.hbm_capacity_bytes//(1024**3)} GB",
         "rate / spec": f"R {hw.hbm_read_bw_ns} / W {hw.hbm_write_bw_ns} GB/s"},
        {"resource": "MSHR depth / wave", "count": "per wave",
         "value": str(hw.mshr_depth_per_wave),
         "rate / spec": f"{hw.waves_per_wg} waves/WG, RTT {hw.xgmi_latency_ns:.0f} ns"},
        {"resource": "Per-WG remote-read cap", "count": "derived",
         "value": f"{mshr_bw_global:.2f} GB/s",
         "rate / spec": "= MSHR×waves×CL/RTT"},
    ])

    inter = pd.DataFrame([
        {"resource": "xGMI peer links", "count": comm_hw.num_peer_links,
         "value": f"{comm_hw.link_bw_ns:.2f} GB/s payload",
         "rate / spec": f"wire {comm_hw.link_bw_ns*1.23:.1f} GB/s (×1.23)"},
        {"resource": "xGMI RTT", "count": "—",
         "value": f"{hw.xgmi_latency_ns:.0f} ns",
         "rate / spec": "remote load round-trip"},
        {"resource": "atomic latency", "count": "—",
         "value": f"{comm_hw.atomic_latency_ns:.0f} ns",
         "rate / spec": "per Signal/Wait"},
        {"resource": "SDMA engines", "count": comm_hw.num_sdma_engines,
         "value": f"R {comm_hw.sdma_read_bw_ns:.1f} / W {comm_hw.sdma_write_bw_ns:.1f} GB/s",
         "rate / spec": "per link"},
        {"resource": "launch overhead", "count": "—",
         "value": f"{comm_hw.launch_overhead_ns/1000:.1f} µs",
         "rate / spec": "kernel dispatch floor"},
    ])

    st.subheader("On-chip resources")
    st.dataframe(on_chip, use_container_width=True, hide_index=True)
    st.subheader("Inter-GPU resources")
    st.dataframe(inter, use_container_width=True, hide_index=True)

st.divider()

# ─── Step 1: Layout ───
st.header("1. Layout: How work maps to links")

num_ts = layout.num_timesteps
st.markdown(f"""
- **Collective**: `{collective}`
- **Layout class**: `{type(layout).__name__}`
- **Timesteps**: `{num_ts}`
- **World size**: `{world_size}` GPUs, `{world_size - 1}` xGMI links
- **Channels (WGs)**: `{num_channels}`
""")

# Show the (pid, timestep) → link table for a few WGs
st.subheader("Layout table: (pid, timestep) → link")
cols_to_show = min(num_ts, 8)
wgs_to_show = min(num_channels, 8)

table_data = []
for pid in range(wgs_to_show):
    row_data = {}
    for ts in range(cols_to_show):
        entry = layout.link_of(pid, ts, 0, world_size)
        if entry.is_self:
            row_data[f"ts={ts}"] = "SELF"
        else:
            row_data[f"ts={ts}"] = f"L{entry.link_id}"
    table_data.append(row_data)

st.dataframe(table_data, use_container_width=True)

# Show contention
st.subheader("Contention per timestep")
contention_data = []
for ts in range(num_ts):
    entry = layout.link_of(0, ts, 0, world_size)
    if entry.is_self:
        contention_data.append({"timestep": ts, "type": "SELF", "wgs_on_link": "-",
                                "concurrent_links": "-", "bw_per_wg": "-"})
    else:
        links = layout.active_links(ts, num_channels, world_size)
        wgs = list(links.values())[0] if links else 0
        n_links = len(links)
        bpw = comm_hw.link_bw / max(wgs, 1)
        contention_data.append({"timestep": ts, "type": "REMOTE",
                                "wgs_on_link": wgs, "concurrent_links": n_links,
                                "bw_per_wg (B/ns)": f"{bpw:.2f}"})
st.dataframe(contention_data[:12], use_container_width=True)

# ─── Step 2: Per-WG tile ───
st.header("2. Per-GPU Tile (2D)")

N = world_size
split_label = f"M/{N}" if split_dim == 0 else f"N/{N}"
st.markdown(f"""
- **Tensor shape**: `[{tensor_m}, {tensor_n}]` ({problem.message_bytes/(1024**2):.1f} MB total)
- **Split dimension**: `{split_dim}` → per-GPU tile = `[{problem.gpu_tile_m}, {problem.gpu_tile_n}]`
- **Per-GPU tile bytes**: `{problem.gpu_tile_bytes:,}` ({problem.gpu_tile_bytes/1024:.1f} KB)
- **Per-GPU cache lines**: `{problem.gpu_tile_cachelines:,}` ({problem.gpu_tile_cachelines * 64:,} bytes transferred)
- **Cacheline efficiency**: `{problem.cacheline_efficiency:.1%}` {"✓" if problem.cacheline_efficiency > 0.95 else "⚠️ padding waste"}
""")

gpu_tile_shape = problem.gpu_tile_shape
gpu_timestep_tile = gpu_tile_shape.divide_byte_equal(layout.chunks_per_timestep)
wg_tile = gpu_timestep_tile.divide_byte_equal(num_channels)

gpu_timestep_tile_cachelines = max(gpu_timestep_tile.cachelines, 1)
gpu_timestep_tile_elements = max(gpu_timestep_tile.elements, 1)
wg_tile_cachelines = max(wg_tile.cachelines, 1)
wg_tile_elements = max(wg_tile.elements, 1)

st.markdown(f"""
- **Per-WG cache lines**: `{wg_tile_cachelines:,}` ({wg_tile_cachelines * 64:,} bytes)
- **Per-WG elements**: `{wg_tile_elements:,}`
- **Iterations**: `{ceil(wg_tile_cachelines / config.cl_per_iter):,}` (cl_per_iter = {config.cl_per_iter})
""")

# ─── Step 3: Work graph ───
st.header("3. Work Graph per Peer Visit")

for ts in range(min(num_ts, 3)):
    entry = layout.link_of(0, ts, 0, world_size)
    ops_str = " → ".join(type(op).__name__ for op in entry.work_graph)
    prefix = "SELF" if entry.is_self else f"Link {entry.link_id}"
    st.markdown(f"**Timestep {ts}** ({prefix}): `{ops_str}`")

if num_ts > 3:
    st.markdown(f"*... {num_ts - 3} more timesteps*")

# ─── Step 4: WLT cost breakdown ───
st.header("4. WLT Cost Breakdown (per iteration)")

entry = layout.link_of(0, 0, 0, world_size)
if entry.is_self:
    entry = layout.link_of(0, 1, 0, world_size) if num_ts > 1 else entry

links = layout.active_links(0, num_channels, world_size)
wgs = list(links.values())[0] if links else num_channels
bw_per_wg = comm_hw.link_bw / max(wgs, 1)

mshr_bw = (hw.mshr_depth_per_wave * hw.waves_per_wg * CACHELINE_BYTES) / hw.xgmi_latency_ns

breakdown = compute_wg_tile_latency(
    entry.work_graph, wg_tile_cachelines, config, hw, comm_hw, bw_per_wg, wg_tile_elements,
    wg_tile=wg_tile,
)

times = {
    "VMEM issue": breakdown.T_vmem,
    "TCP (vL1D)": breakdown.T_tcp,
    "L2 (TCC)": breakdown.T_l2,
    "MALL": breakdown.T_mall,
    "HBM read": breakdown.T_hbm_read,
    "HBM write": breakdown.T_hbm_write,
    "xGMI read": breakdown.T_xgmi_read,
    "xGMI write": breakdown.T_xgmi_write,
    "VALU": breakdown.T_valu,
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Per-iteration times (ns)")
    fig_bar = go.Figure()
    names = list(times.keys())
    vals = list(times.values())
    colors = ['#E53935' if v == max(vals) and v > 0 else '#90CAF9' for v in vals]
    fig_bar.add_trace(go.Bar(x=names, y=vals, marker_color=colors))
    fig_bar.update_layout(yaxis_title="Time (ns)", height=350)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown(f"**Bottleneck**: `{breakdown.bottleneck}` at `{max(vals):.2f}` ns")

with col2:
    st.subheader("WLT total")
    st.markdown(f"""
    ```
    T_prologue:     {breakdown.T_prologue:>10.1f} ns
    T_wlt (iter):   {breakdown.T_wlt:>10.2f} ns × {max(breakdown.num_iters-1,0)} iters
    T_epilogue:     {breakdown.T_epilogue:>10.1f} ns
    T_sync:         {breakdown.T_sync:>10.1f} ns
    ─────────────────────────────────
    T_total:        {breakdown.T_total:>10.0f} ns ({breakdown.T_total/1000:.1f} µs)
    ```
    """)
    # bw_per_wg is in B/cycle (matches model); convert to B/ns for display
    # via the GPU frequency.
    bw_per_wg_ns = hw.rate_per_ns(bw_per_wg)
    mshr_bw_ns = mshr_bw  # mshr_bw above was already built using xgmi_latency_ns → B/ns
    st.markdown(f"""
    **Key rates:**
    - MSHR-limited BW: `{mshr_bw_ns:.2f}` B/ns per WG
    - Link share BW: `{bw_per_wg_ns:.2f}` B/ns per WG (`{comm_hw.link_bw_ns:.1f} / {wgs}`)
    - Effective remote read: `{min(bw_per_wg_ns, mshr_bw_ns):.2f}` B/ns
    - HBM read per CU: `{hw.hbm_read_bw_per_cu_ns(num_channels):.2f}` B/ns
    """)

# ─── Step 5: Playground — Trace & Graph ───
st.header("5. Playground — Trace the Analytical Computation")

st.markdown(
    "Every line of the cost model, with formula, inputs, and the resulting value. "
    "Adjust knobs in the sidebar to see the trace update live."
)

is_ring = _is_ring_layout(layout)

def _fmt_ns(x):
    if x >= 1_000_000:
        return f"{x/1000:.0f} µs"
    if x >= 1000:
        return f"{x/1000:.1f} µs"
    return f"{x:.1f} ns"

def _fmt_bw(x):
    return f"{x:.2f} B/ns"

def _node(name, label, fill="#FFFFFF", shape="box"):
    label_esc = label.replace('"', '\\"')
    return f'  {name} [label="{label_esc}", shape={shape}, style="filled,rounded", fillcolor="{fill}"];'


# ─── Pre-compute everything the trace needs (shared between Trace + Graph tabs) ───
def _build_trace_rows():
    """Return a list of (section, quantity, formula, inputs, value, unit, note) tuples
    matching the order of operations in model/collective.py + model/latency.py."""
    rows = []
    N_ = world_size
    NCH_ = num_channels
    CL = CACHELINE_BYTES
    dt_bytes = problem.element_bytes

    # ─ Inputs ─
    rows.append(("Inputs", "msg_bytes", "M × N × dtype_bytes",
                 f"M={problem.M}, N={problem.N}, dtype={problem.dtype.name} ({dt_bytes} B)",
                 f"{problem.message_bytes:,}", "B",
                 f"{problem.message_bytes/(1024**2):.2f} MiB"))
    rows.append(("Inputs", "world_size", "—", "—", str(N_), "GPUs",
                 f"{N_ - 1} xGMI peers per rank"))
    rows.append(("Inputs", "num_wgs (NCH)", "—", "—", str(NCH_), "CTAs", ""))
    rows.append(("Inputs", "link_bw", "—", "—", f"{comm_hw.link_bw_ns:.2f}",
                 "B/ns", f"{comm_hw.link_bw_ns*1e9/(1024**3):.1f} GiB/s payload"))
    rows.append(("Inputs", "mshr_depth × waves × CL / xGMI_RTT",
                 f"{hw.mshr_depth_per_wave} × {hw.waves_per_wg} × {CL} / {hw.xgmi_latency_ns:.0f}",
                 "from hw spec",
                 f"{(hw.mshr_depth_per_wave*hw.waves_per_wg*CL)/hw.xgmi_latency_ns:.2f}",
                 "B/ns", "per-WG MSHR-limited remote-read BW"))
    rows.append(("Inputs", "launch_overhead", "—", "—",
                 f"{comm_hw.launch_overhead_ns/1000:.1f}", "µs", "kernel dispatch floor"))

    # ─ Tile sizing ─
    # Per-rank local tile is dense in memory regardless of split_dim, so
    # cache-line count is the flat-byte total. (The torch shape sweep
    # confirms RCCL collectives are shape-agnostic at fixed byte total.)
    rows.append(("Tile", "gpu_tile_bytes",
                 "gpu_tile_m × gpu_tile_n × dtype",
                 f"gpu_tile_m={problem.gpu_tile_m}, "
                 f"gpu_tile_n={problem.gpu_tile_n}, dtype={dt_bytes}",
                 f"{problem.gpu_tile_bytes:,}", "B",
                 f"per-rank local tile is dense (split_dim={problem.split_dim} "
                 "controls wire-volume only)"))
    rows.append(("Tile", "gpu_tile_cachelines",
                 "ceil(gpu_tile_bytes / CL)",
                 f"gpu_tile_bytes={problem.gpu_tile_bytes:,}, CL={CL}",
                 f"{problem.gpu_tile_cachelines:,}", "cl",
                 f"{problem.cacheline_efficiency:.1%} efficient "
                 f"({problem.gpu_tile_bytes/1024:.1f} KiB)"))

    chunks_pt = layout.chunks_per_timestep
    rows.append(("Tile", "chunks_per_timestep", "from layout class",
                 type(layout).__name__,
                 str(chunks_pt), "",
                 "= N for chunked algos (ring, two-shot, a2a), 1 otherwise"))
    rows.append(("Tile", "gpu_timestep_tile_cachelines",
                 "gpu_tile_cachelines // chunks_per_timestep",
                 f"gpu_tile_cachelines={problem.gpu_tile_cachelines:,}, "
                 f"chunks_per_timestep={chunks_pt}",
                 f"{gpu_timestep_tile_cachelines:,}", "cl",
                 f"{gpu_timestep_tile_cachelines*CL/1024:.1f} KiB per timestep per rank"))
    rows.append(("Tile", "wg_tile_cachelines",
                 "ceil(gpu_timestep_tile_cachelines / NCH)",
                 f"gpu_timestep_tile_cachelines={gpu_timestep_tile_cachelines:,}, "
                 f"NCH={NCH_}",
                 f"{wg_tile_cachelines:,}", "cl",
                 f"{wg_tile_cachelines*CL/1024:.1f} KiB per WG per timestep"))
    rows.append(("Tile", "cl_per_iter", "vgprs_for_data × 4 / CL",
                 f"vgprs={config.vgprs_for_data}, CL={CL}",
                 str(config.cl_per_iter), "cl",
                 f"{config.bytes_per_iter} B per iter"))
    num_iters_val = max(ceil(wg_tile_cachelines / config.cl_per_iter), 1)
    rows.append(("Tile", "num_iters", "ceil(wg_tile_cachelines / cl_per_iter)",
                 f"wg_tile_cachelines={wg_tile_cachelines:,}, cl_per_iter={config.cl_per_iter}",
                 f"{num_iters_val:,}", "iters", ""))
    rows.append(("Tile", "iter_tile_elements",
                 "ceil(wg_tile_elements / num_iters)",
                 f"wg_tile_elements={wg_tile_elements:,}, num_iters={num_iters_val:,}",
                 f"{max(ceil(wg_tile_elements/num_iters_val),1):,}", "elts",
                 "feeds VALU work per iter"))

    # ─ Layout / contention ─
    rows.append(("Layout", "num_timesteps", "from layout class",
                 type(layout).__name__,
                 str(num_ts), "timesteps",
                 f"{layout.num_timesteps} for {collective}"))

    # Pick first non-self timestep for FU breakdown
    show_idx = 0
    for ts in range(num_ts):
        e_ = layout.link_of(0, ts, 0, world_size)
        if not e_.is_self:
            show_idx = ts
            break
    show_entry = layout.link_of(0, show_idx, 0, world_size)
    show_links = layout.active_links(show_idx, num_channels, world_size)
    show_wgs = list(show_links.values())[0] if show_links else num_channels
    # show_bw_per_wg in B/cycle (model input); ns variants used for display.
    show_bw_per_wg = comm_hw.link_bw / max(show_wgs, 1)                   # B/cycle
    show_bw_per_wg_ns_trace = hw.rate_per_ns(show_bw_per_wg)               # B/ns (display)
    mshr_bw_val = (hw.mshr_depth_per_wave * hw.waves_per_wg * CL) / hw.xgmi_latency_ns  # B/ns
    eff_xgmi_read_ns = min(show_bw_per_wg_ns_trace, mshr_bw_val)          # B/ns
    eff_note = "**MSHR-bound**" if mshr_bw_val < show_bw_per_wg_ns_trace else "link-share-bound"

    rows.append(("Layout", f"wgs_on_link (timestep {show_idx})",
                 "from layout.active_links()",
                 f"NCH={NCH_}, N={N_}",
                 str(show_wgs), "WGs",
                 f"{len(show_links)} concurrent links"))
    rows.append(("Layout", "bw_per_wg", "link_bw / wgs_on_link",
                 f"link_bw={comm_hw.link_bw_ns:.2f}, wgs={show_wgs}",
                 f"{hw.rate_per_ns(show_bw_per_wg):.2f}", "B/ns",
                 "before MSHR cap"))
    rows.append(("Layout", "eff_xgmi_read", "min(bw_per_wg, mshr_bw)  [in B/ns]",
                 f"bw_per_wg={show_bw_per_wg:.2f}, mshr_bw={mshr_bw_val:.2f}",
                 f"{eff_xgmi_read_ns:.2f}", "B/ns", eff_note))

    # ─ Work graph (per-iter FU work for the timestep shown) ─
    iter_work, sync_work = resolve_work_graph(
        show_entry.work_graph,
        config.cl_per_iter, config.instrs_per_cl,
        max(ceil(wg_tile_elements/num_iters_val), 1),
    )
    wg_str = " → ".join(type(op).__name__ for op in show_entry.work_graph)
    rows.append(("Work Graph",
                 f"primitives (timestep {show_idx})", wg_str,
                 f"peer={show_entry.peer_rank}, dir={show_entry.direction}",
                 "", "",
                 "vmem_r={}, tcp_r={}, l2_r={}, mall_r={}, hbm_r={}, xgmi_r={}".format(
                     iter_work.vmem_read_instrs, iter_work.tcp_read_cl,
                     iter_work.l2_read_cl, iter_work.mall_read_cl,
                     iter_work.hbm_read_cl, iter_work.xgmi_read_cl)))
    rows.append(("Work Graph", "writes (per iter)", "",
                 "—",
                 "", "",
                 "vmem_w={}, tcp_w={}, l2_w={}, mall_w={}, hbm_w={}, xgmi_w={}".format(
                     iter_work.vmem_write_instrs, iter_work.tcp_write_cl,
                     iter_work.l2_write_cl, iter_work.mall_write_cl,
                     iter_work.hbm_write_cl, iter_work.xgmi_write_cl)))
    rows.append(("Work Graph", "atomics (sync)", "Σ Signal/Wait",
                 "—", str(sync_work.atomic_count), "",
                 f"atomic_lat={comm_hw.atomic_latency_ns:.0f} ns"))
    rows.append(("Work Graph", "valu_ops (per iter)", "elements_per_iter",
                 f"elements={max(ceil(wg_tile_elements/num_iters_val),1):,}",
                 str(iter_work.valu_ops), "ops", ""))

    # ─ Per-iter FU times ─
    breakdown_show = compute_wg_tile_latency(
        show_entry.work_graph, wg_tile_cachelines, config, hw, comm_hw,
        show_bw_per_wg, wg_tile_elements, wg_tile=wg_tile,
    )
    bnk = breakdown_show.bottleneck

    # All inline rate/bw values shown in ns-domain units for display consistency.
    fu = [
        ("T_vmem", "vmem_instrs / vmem_issue_rate",
         f"vmem={iter_work.vmem_read_instrs + iter_work.vmem_write_instrs}, rate={hw.vmem_issue_rate_ns} instr/ns",
         breakdown_show.T_vmem),
        ("T_tcp", "(tcp_r + tcp_w) × CL / tcp_bw",
         f"cl={iter_work.tcp_read_cl + iter_work.tcp_write_cl}, tcp_bw={hw.tcp_bw_ns} B/ns",
         breakdown_show.T_tcp),
        ("T_l2", "(l2_r + l2_w) × CL / l2_bw_per_cu(active_per_xcd)",
         f"cl={iter_work.l2_read_cl + iter_work.l2_write_cl}, active_per_xcd={max(ceil(NCH_/hw.num_xcd),1)}",
         breakdown_show.T_l2),
        ("T_mall", "(mall_r + mall_w) × CL / mall_bw",
         f"cl={iter_work.mall_read_cl + iter_work.mall_write_cl}, mall_bw={hw.mall_bw_ns} B/ns",
         breakdown_show.T_mall),
        ("T_hbm_read", "hbm_r_cl × CL / hbm_read_bw_per_cu(NCH)",
         f"cl={iter_work.hbm_read_cl}, hbm_r/CU={hw.hbm_read_bw_per_cu_ns(NCH_):.2f} B/ns",
         breakdown_show.T_hbm_read),
        ("T_hbm_write", "hbm_w_cl × CL / hbm_write_bw_per_cu(NCH)",
         f"cl={iter_work.hbm_write_cl}, hbm_w/CU={hw.hbm_write_bw_per_cu_ns(NCH_):.2f} B/ns",
         breakdown_show.T_hbm_write),
        ("T_xgmi_read", "xgmi_r_cl × CL / eff_xgmi_read",
         f"cl={iter_work.xgmi_read_cl}, eff={eff_xgmi_read_ns:.2f} B/ns",
         breakdown_show.T_xgmi_read),
        ("T_xgmi_write", "xgmi_w_cl × CL / bw_per_wg",
         f"cl={iter_work.xgmi_write_cl}, bw_per_wg={hw.rate_per_ns(show_bw_per_wg):.2f} B/ns",
         breakdown_show.T_xgmi_write),
        ("T_valu", "valu_ops / valu_rate",
         f"ops={iter_work.valu_ops}, rate={hw.valu_rate_ns:.1f} elts/ns",
         breakdown_show.T_valu),
    ]
    bottleneck_key = max(fu, key=lambda r: r[3])[0] if any(v[3] > 0 for v in fu) else ""
    for name, formula, inputs, val in fu:
        note = "← **bottleneck**" if name == bottleneck_key and val > 0 else ""
        rows.append(("Per-iter (1 WLT)", name, formula, inputs,
                     f"{val:.2f}" if val > 0 else "0",
                     "ns", note))

    rows.append(("Per-iter (1 WLT)", "T_wlt", "max(FU times)",
                 f"argmax = {bnk}",
                 f"{breakdown_show.T_wlt:.2f}", "ns",
                 "one WLT atom"))

    # ─ Per-timestep composition ─
    rows.append(("Per-timestep", "T_prologue", "max(hbm_r, xgmi_r, mall) — ramp-up",
                 "first cache lines arrive",
                 f"{breakdown_show.T_prologue:.1f}", "ns", ""))
    rows.append(("Per-timestep", "T_epilogue", "max(hbm_w, xgmi_w) — drain",
                 "last writes commit",
                 f"{breakdown_show.T_epilogue:.1f}", "ns", ""))
    rows.append(("Per-timestep", "T_sync",
                 "atomic_count × atomic_latency",
                 f"atomics={sync_work.atomic_count}, lat={comm_hw.atomic_latency_ns:.0f} ns",
                 f"{breakdown_show.T_sync:.1f}", "ns",
                 "Signal/Wait roundtrips"))
    rows.append(("Per-timestep", "T_timestep",
                 "T_prologue + (num_iters−1)×T_wlt + T_epilogue + T_sync",
                 f"prol+{num_iters_val-1}×{breakdown_show.T_wlt:.1f}+epi+sync",
                 f"{breakdown_show.T_total:,.0f}", "ns",
                 f"= {breakdown_show.T_total/1000:.1f} µs"))

    # ─ Sum across timesteps ─
    ts_times_ns = []
    for ts in range(num_ts):
        e_ = layout.link_of(0, ts, 0, world_size)
        if e_.is_self:
            bw_pw_ = hw.hbm_read_bw_per_cu(num_channels)
            bd_ = compute_wg_tile_latency(e_.work_graph, wg_tile_cachelines, config, hw, comm_hw, bw_pw_, wg_tile_elements, wg_tile=wg_tile)
            ts_times_ns.append(("SELF", bd_.T_total))
        else:
            t_per_link = []
            for _lid, _wgs in layout.active_links(ts, num_channels, world_size).items():
                bw_pw_ = comm_hw.link_bw / max(_wgs, 1)
                bd_ = compute_wg_tile_latency(e_.work_graph, wg_tile_cachelines, config, hw, comm_hw, bw_pw_, wg_tile_elements, wg_tile=wg_tile)
                t_per_link.append(bd_.T_total)
            ts_times_ns.append((f"L{e_.link_id}", max(t_per_link) if t_per_link else 0.0))
    sum_ts_ns = sum(t for _, t in ts_times_ns)

    rows.append(("Total", "T_timesteps",
                 "Σ over timesteps of max-across-active-links of T_timestep",
                 f"{num_ts} timesteps ({sum(1 for tag,_ in ts_times_ns if tag=='SELF')} self)",
                 f"{sum_ts_ns:,.0f}", "ns",
                 f"= {sum_ts_ns/1000:.1f} µs"))
    rows.append(("Total", "T_total",
                 "launch_overhead + T_timesteps",
                 f"launch={comm_hw.launch_overhead_ns:,.0f}, timesteps={sum_ts_ns:,.0f}",
                 f"{pred_ns:,.0f}", "ns",
                 f"**= {pred_us:.1f} µs**"))

    return rows, ts_times_ns, show_idx, breakdown_show, mshr_bw_val, show_bw_per_wg


# Build once so the snapshot block at the bottom can reuse it
trace_rows, ts_times, show_idx, show_bd, mshr_bw_for_show, bw_pw_for_show = _build_trace_rows()

tab_trace, tab_graph = st.tabs(["Trace (table)", "Graph (DAG)"])

with tab_trace:
    if is_ring:
        st.info(
            "Ring layouts use the `_compute_ring_latency` aggregate-throughput model "
            "(not the per-timestep wg_tile path). Use the Graph tab for the ring-specific DAG. "
            "Below is what the wg_tile path *would* produce if invoked — provided as context."
        )

    # Group by section using expanders
    import pandas as pd
    sections = []
    for s, *_ in trace_rows:
        if s not in sections:
            sections.append(s)

    for sec in sections:
        sec_rows = [r for r in trace_rows if r[0] == sec]
        with st.expander(f"**{sec}** ({len(sec_rows)} quantities)",
                         expanded=(sec in ("Per-iter (1 WLT)", "Per-timestep", "Total"))):
            df = pd.DataFrame(
                [{
                    "quantity": r[1],
                    "formula": r[2],
                    "inputs": r[3],
                    "value": r[4],
                    "unit": r[5],
                    "note": r[6],
                } for r in sec_rows]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

    # Per-timestep breakdown chart
    st.subheader("Per-timestep T_timestep (sequential sum)")
    labels = [f"t{i} ({tag})" for i, (tag, _) in enumerate(ts_times)]
    ys = [t/1000 for _, t in ts_times]
    colors = ['#90A4AE' if 'SELF' in lbl else '#42A5F5' for lbl in labels]
    fig_pv = go.Figure(go.Bar(x=labels, y=ys, marker_color=colors,
                               text=[f"{y:.1f}" for y in ys], textposition="auto"))
    fig_pv.update_layout(yaxis_title="µs per timestep", height=320,
                         title=f"{num_ts} timesteps — Σ = {sum(ys):.1f} µs")
    st.plotly_chart(fig_pv, use_container_width=True)

    # Waterfall: launch → timesteps → total
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Launch", "Σ timesteps", "T_total"],
        y=[comm_hw.launch_overhead_ns/1000, sum(ys), pred_us],
        text=[f"{comm_hw.launch_overhead_ns/1000:.1f}",
              f"+{sum(ys):.1f}",
              f"={pred_us:.1f}"],
        textposition="outside",
        connector={"line": {"color": "#888"}},
    ))
    fig_wf.update_layout(yaxis_title="µs", height=340,
                         title="T_total = launch + Σ T_timestep_i")
    st.plotly_chart(fig_wf, use_container_width=True)

def _render_ring_dag():
    # Local re-creation of the ring math for display purposes; works in
    # nanoseconds (B/ns and ns) to match the dashboard's display labels.
    # The actual model runs in cycles internally — see model/collective.py.
    N = world_size
    num_wgs = num_channels
    num_timesteps = layout.num_timesteps
    chunks_pt = layout.chunks_per_timestep
    gpu_timestep_tile_bytes = problem.gpu_tile_cachelines * CACHELINE_BYTES // chunks_pt
    total_wire_bytes = gpu_timestep_tile_bytes * num_timesteps
    CL = CACHELINE_BYTES
    mshr_bw_per_wg = (hw.mshr_depth_per_wave * hw.waves_per_wg * CL) / hw.xgmi_latency_ns   # B/ns
    aggregate_bw = min(comm_hw.link_bw_ns, num_wgs * mshr_bw_per_wg)                        # B/ns
    T_transfer = total_wire_bytes / aggregate_bw                                            # ns
    sync_entry = layout.link_of(0, 0, 0, world_size)
    sync_ops = sum(1 for op in sync_entry.work_graph if isinstance(op, (Signal, Wait)))
    T_sync_per_step = sync_ops * comm_hw.atomic_latency_ns                                  # ns
    T_sync_total = num_timesteps * T_sync_per_step                                          # ns
    hbm_bw_agg = hw.hbm_read_bw_ns * hw._bw_fraction(num_wgs)                               # B/ns
    T_hbm = total_wire_bytes / hbm_bw_agg                                                   # ns
    T_transfer_total = max(T_transfer, T_hbm)
    transfer_winner = "xGMI" if T_transfer >= T_hbm else "HBM"

    INPUT = "#E3F2FD"
    DERIVED = "#FFF9C4"
    TIME = "#C8E6C9"
    BOTTLE = "#FFCDD2"
    TOTAL = "#B39DDB"

    dot_lines = ["digraph G {", '  rankdir=TB; node [fontname="Helvetica", fontsize=10];',
                 '  edge [fontname="Helvetica", fontsize=8, color="#666"];']

    dot_lines.append(_node("msg", f"msg_bytes = {problem.message_bytes:,}\\n({problem.message_bytes/(1024**2):.2f} MB)", INPUT))
    dot_lines.append(_node("W", f"world_size N = {N}", INPUT))
    dot_lines.append(_node("NCH", f"num_wgs = {num_wgs}", INPUT))
    dot_lines.append(_node("link_bw", f"link_bw = {_fmt_bw(comm_hw.link_bw_ns)}", INPUT))
    dot_lines.append(_node("hbm_bw", f"hbm_read_bw = {_fmt_bw(hw.hbm_read_bw_ns)}", INPUT))
    dot_lines.append(_node("mshr_in", f"MSHR × waves × CL / xGMI_RTT\\n= {hw.mshr_depth_per_wave}×{hw.waves_per_wg}×{CL} / {hw.xgmi_latency_ns:.0f}", INPUT))
    dot_lines.append(_node("launch", f"launch_overhead = {comm_hw.launch_overhead_ns/1000:.1f} µs", INPUT))

    dot_lines.append(_node("gpu_tile", f"gpu_tile_cachelines\\n= {problem.gpu_tile_cachelines:,}", DERIVED))
    dot_lines.append(_node("nts", f"num_timesteps = 2(N-1) | (N-1)\\n= {num_timesteps}", DERIVED))
    dot_lines.append(_node("cpt", f"chunks_per_timestep\\n= {chunks_pt}", DERIVED))
    dot_lines.append(_node("ts_tile", f"gpu_timestep_tile_bytes\\n= gpu_tile_cachelines × CL / chunks_per_timestep\\n= {gpu_timestep_tile_bytes:,} B", DERIVED))
    dot_lines.append(_node("wire", f"total_wire_bytes\\n= gpu_timestep_tile_bytes × num_timesteps\\n= {total_wire_bytes:,} B", DERIVED))
    dot_lines.append(_node("mshr_bw", f"mshr_bw_per_wg\\n= {_fmt_bw(mshr_bw_per_wg)}", DERIVED))
    dot_lines.append(_node("agg_bw", f"aggregate_bw\\n= min(link_bw, NCH × mshr_bw)\\n= {_fmt_bw(aggregate_bw)}", DERIVED))
    dot_lines.append(_node("hbm_frac", f"hbm_bw_agg = hbm × frac(NCH)\\n= {_fmt_bw(hbm_bw_agg)}", DERIVED))

    transfer_color = BOTTLE if transfer_winner == "xGMI" else TIME
    hbm_color = BOTTLE if transfer_winner == "HBM" else TIME
    dot_lines.append(_node("T_xgmi", f"T_transfer (xGMI)\\n= wire / agg_bw\\n= {_fmt_ns(T_transfer)}", transfer_color))
    dot_lines.append(_node("T_hbm", f"T_hbm = wire / hbm_agg\\n= {_fmt_ns(T_hbm)}", hbm_color))
    dot_lines.append(_node("T_xfer", f"T_transfer_total\\n= max(T_xgmi, T_hbm)\\nwinner: {transfer_winner}\\n= {_fmt_ns(T_transfer_total)}", TIME))
    dot_lines.append(_node("T_sync", f"T_sync = num_timesteps × {sync_ops}×atomic_lat\\n= {_fmt_ns(T_sync_total)}", TIME))

    dot_lines.append(_node("T_total", f"T_total = launch + T_xfer + T_sync\\n= {_fmt_ns(pred_ns)}", TOTAL))

    edges = [
        ("msg", "gpu_tile"), ("W", "gpu_tile"),
        ("W", "nts"),
        ("gpu_tile", "ts_tile"), ("cpt", "ts_tile"),
        ("ts_tile", "wire"), ("nts", "wire"),
        ("mshr_in", "mshr_bw"),
        ("mshr_bw", "agg_bw"), ("link_bw", "agg_bw"), ("NCH", "agg_bw"),
        ("hbm_bw", "hbm_frac"), ("NCH", "hbm_frac"),
        ("wire", "T_xgmi"), ("agg_bw", "T_xgmi"),
        ("wire", "T_hbm"), ("hbm_frac", "T_hbm"),
        ("T_xgmi", "T_xfer"), ("T_hbm", "T_xfer"),
        ("nts", "T_sync"),
        ("launch", "T_total"), ("T_xfer", "T_total"), ("T_sync", "T_total"),
    ]
    for s, d in edges:
        dot_lines.append(f"  {s} -> {d};")
    dot_lines.append("}")

    st.graphviz_chart("\n".join(dot_lines), use_container_width=True)

    # Waterfall: launch + transfer + sync
    st.subheader("Where the time goes")
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Launch", "Transfer", "Sync", "T_total"],
        y=[comm_hw.launch_overhead_ns/1000, T_transfer_total/1000, T_sync_total/1000, pred_us],
        text=[f"{comm_hw.launch_overhead_ns/1000:.1f}",
              f"+{T_transfer_total/1000:.1f}",
              f"+{T_sync_total/1000:.1f}",
              f"={pred_us:.1f}"],
        textposition="outside",
        connector={"line": {"color": "#888"}},
    ))
    fig_wf.update_layout(yaxis_title="µs", height=380,
                         title=f"Ring path: T = launch + max(T_xGMI, T_HBM) + T_sync — winner: {transfer_winner}")
    st.plotly_chart(fig_wf, use_container_width=True)


def _render_seq_dag():
    INPUT = "#E3F2FD"
    DERIVED = "#FFF9C4"
    TIME = "#C8E6C9"
    BOTTLE = "#FFCDD2"
    TOTAL = "#B39DDB"

    # Recompute per-timestep times to identify bottleneck + per-timestep T values
    ts_times = []
    for ts in range(num_ts):
        e = layout.link_of(0, ts, 0, world_size)
        if e.is_self:
            bw_pw = hw.hbm_read_bw_per_cu(num_channels)
            bd = compute_wg_tile_latency(e.work_graph, wg_tile_cachelines, config, hw, comm_hw, bw_pw, wg_tile_elements, wg_tile=wg_tile)
            ts_times.append(("SELF", bd.T_total, bd.bottleneck))
        else:
            link_wgs = layout.active_links(ts, num_channels, world_size)
            t_links = []
            for _lid, _wgs in link_wgs.items():
                bw_pw = comm_hw.link_bw / max(_wgs, 1)
                bd = compute_wg_tile_latency(e.work_graph, wg_tile_cachelines, config, hw, comm_hw, bw_pw, wg_tile_elements, wg_tile=wg_tile)
                t_links.append((bd.T_total, bd.bottleneck))
            t_max = max(t_links, key=lambda x: x[0])
            ts_times.append((f"L{e.link_id}", t_max[0], t_max[1]))

    T_timesteps_ns = sum(t for _, t, _ in ts_times)

    # Use the first remote timestep for the per-timestep FU breakdown
    show_idx = next((i for i, e in enumerate([layout.link_of(0, ts, 0, world_size) for ts in range(num_ts)])
                     if not e.is_self), 0)
    show_entry = layout.link_of(0, show_idx, 0, world_size)
    show_links = layout.active_links(show_idx, num_channels, world_size)
    show_wgs = list(show_links.values())[0] if show_links else num_channels
    # bw_per_wg is fed into the model in B/cycle; the corresponding
    # display value (in B/ns) is computed below for the graphviz labels.
    show_bw_per_wg = comm_hw.link_bw / max(show_wgs, 1)                      # B/cycle (model input)
    show_bw_per_wg_ns = hw.rate_per_ns(show_bw_per_wg)                        # B/ns (display)
    show_bd = compute_wg_tile_latency(show_entry.work_graph, wg_tile_cachelines, config, hw, comm_hw, show_bw_per_wg, wg_tile_elements, wg_tile=wg_tile)
    mshr_bw_val = (hw.mshr_depth_per_wave * hw.waves_per_wg * CACHELINE_BYTES) / hw.xgmi_latency_ns  # B/ns

    work_graph_str = " → ".join(type(op).__name__ for op in show_entry.work_graph)

    dot_lines = ["digraph G {", '  rankdir=TB; node [fontname="Helvetica", fontsize=10];',
                 '  edge [fontname="Helvetica", fontsize=8, color="#666"];']

    # Inputs
    dot_lines.append(_node("msg", f"msg_bytes = {problem.message_bytes:,}\\n({problem.message_bytes/(1024**2):.2f} MB)", INPUT))
    dot_lines.append(_node("W", f"world_size N = {world_size}", INPUT))
    dot_lines.append(_node("NCH", f"num_wgs = {num_channels}", INPUT))
    dot_lines.append(_node("link_bw", f"link_bw = {_fmt_bw(comm_hw.link_bw_ns)}", INPUT))
    dot_lines.append(_node("mshr_in", f"MSHR × waves × CL / xGMI_RTT\\n= {hw.mshr_depth_per_wave}×{hw.waves_per_wg}×{CACHELINE_BYTES} / {hw.xgmi_latency_ns:.0f}", INPUT))
    dot_lines.append(_node("hbm_bw", f"hbm_read_bw = {_fmt_bw(hw.hbm_read_bw_ns)}", INPUT))
    dot_lines.append(_node("launch", f"launch_overhead = {comm_hw.launch_overhead_ns/1000:.1f} µs", INPUT))
    dot_lines.append(_node("wg_graph", f"work_graph (timestep {show_idx})\\n{work_graph_str}", INPUT))

    # Derived
    dot_lines.append(_node("gpu_tile", f"gpu_tile_cachelines = {problem.gpu_tile_cachelines:,}", DERIVED))
    dot_lines.append(_node("cpt", f"chunks_per_timestep = {layout.chunks_per_timestep}", DERIVED))
    dot_lines.append(_node("ts_tile", f"gpu_timestep_tile_cachelines\\n= gpu_tile_cachelines // chunks_per_timestep\\n= {gpu_timestep_tile_cachelines:,}", DERIVED))
    dot_lines.append(_node("wg_tile", f"wg_tile_cachelines\\n= ceil(gpu_timestep_tile_cachelines / NCH)\\n= {wg_tile_cachelines:,}", DERIVED))
    dot_lines.append(_node("num_iters", f"num_iters = ceil(wg_tile_cachelines / cl_per_iter)\\n= {ceil(wg_tile_cachelines / config.cl_per_iter):,}", DERIVED))
    dot_lines.append(_node("wgs_link", f"wgs_on_link (timestep {show_idx}) = {show_wgs}", DERIVED))
    dot_lines.append(_node("bw_pw", f"bw_per_wg = link_bw / wgs_on_link\\n= {_fmt_bw(show_bw_per_wg_ns)}", DERIVED))
    dot_lines.append(_node("mshr_bw", f"mshr_bw = {_fmt_bw(mshr_bw_val)}", DERIVED))
    dot_lines.append(_node("eff_xgmi_r", f"eff_xgmi_read = min(bw_pw, mshr_bw)\\n= {_fmt_bw(min(show_bw_per_wg_ns, mshr_bw_val))}", DERIVED))
    dot_lines.append(_node("npv", f"num_timesteps = {num_ts}", DERIVED))

    # Functional unit times (per iter)
    fu_times = {
        "T_vmem": show_bd.T_vmem,
        "T_tcp": show_bd.T_tcp,
        "T_l2": show_bd.T_l2,
        "T_mall": show_bd.T_mall,
        "T_hbm_r": show_bd.T_hbm_read,
        "T_hbm_w": show_bd.T_hbm_write,
        "T_xgmi_r": show_bd.T_xgmi_read,
        "T_xgmi_w": show_bd.T_xgmi_write,
        "T_valu": show_bd.T_valu,
    }
    max_fu = max(fu_times.values()) if any(v > 0 for v in fu_times.values()) else 0
    bottleneck_name = max(fu_times, key=fu_times.get) if max_fu > 0 else ""
    fu_pretty = {
        "T_vmem": "VMEM", "T_tcp": "TCP", "T_l2": "L2", "T_mall": "MALL",
        "T_hbm_r": "HBM_read", "T_hbm_w": "HBM_write",
        "T_xgmi_r": "xGMI_read", "T_xgmi_w": "xGMI_write", "T_valu": "VALU",
    }
    for k, v in fu_times.items():
        color = BOTTLE if k == bottleneck_name and v > 0 else (TIME if v > 0 else "#EEEEEE")
        dot_lines.append(_node(k, f"{fu_pretty[k]}\\n{_fmt_ns(v)}", color))

    # T_wlt = max
    dot_lines.append(_node("T_wlt", f"T_wlt = max(FU times)\\nbottleneck: {fu_pretty.get(bottleneck_name, '—')}\\n= {_fmt_ns(show_bd.T_wlt)}", BOTTLE if max_fu > 0 else TIME))
    dot_lines.append(_node("T_prologue", f"T_prologue = max(read times)\\n= {_fmt_ns(show_bd.T_prologue)}", TIME))
    dot_lines.append(_node("T_epilogue", f"T_epilogue = max(write times)\\n= {_fmt_ns(show_bd.T_epilogue)}", TIME))
    dot_lines.append(_node("T_sync", f"T_sync = atomics × atomic_lat\\n= {_fmt_ns(show_bd.T_sync)}", TIME))

    dot_lines.append(_node("T_timestep", f"T_timestep = T_prologue\\n  + (num_iters-1) × T_wlt\\n  + T_epilogue + T_sync\\n= {_fmt_ns(show_bd.T_total)}", TIME))
    dot_lines.append(_node("T_timesteps", f"T_timesteps = Σ T_timestep_i\\n({num_ts} timesteps)\\n= {_fmt_ns(T_timesteps_ns)}", TIME))
    dot_lines.append(_node("T_total", f"T_total = launch + T_timesteps\\n= {_fmt_ns(pred_ns)}", TOTAL))

    edges = [
        ("msg", "gpu_tile"), ("W", "gpu_tile"),
        ("gpu_tile", "ts_tile"), ("cpt", "ts_tile"),
        ("ts_tile", "wg_tile"), ("NCH", "wg_tile"),
        ("wg_tile", "num_iters"),
        ("NCH", "wgs_link"), ("W", "wgs_link"),
        ("link_bw", "bw_pw"), ("wgs_link", "bw_pw"),
        ("mshr_in", "mshr_bw"),
        ("bw_pw", "eff_xgmi_r"), ("mshr_bw", "eff_xgmi_r"),
        ("W", "npv"),
        # work graph + per-iter inputs feed each FU time
        ("wg_graph", "T_vmem"),
        ("wg_graph", "T_tcp"),
        ("wg_graph", "T_l2"),
        ("wg_graph", "T_mall"),
        ("wg_graph", "T_hbm_r"), ("hbm_bw", "T_hbm_r"),
        ("wg_graph", "T_hbm_w"), ("hbm_bw", "T_hbm_w"),
        ("wg_graph", "T_xgmi_r"), ("eff_xgmi_r", "T_xgmi_r"),
        ("wg_graph", "T_xgmi_w"), ("bw_pw", "T_xgmi_w"),
        ("wg_graph", "T_valu"),
    ]
    for k in fu_times:
        edges.append((k, "T_wlt"))
    edges += [
        ("T_hbm_r", "T_prologue"), ("T_xgmi_r", "T_prologue"), ("T_mall", "T_prologue"),
        ("T_hbm_w", "T_epilogue"), ("T_xgmi_w", "T_epilogue"),
        ("wg_graph", "T_sync"),
        ("T_wlt", "T_timestep"), ("num_iters", "T_timestep"),
        ("T_prologue", "T_timestep"), ("T_epilogue", "T_timestep"), ("T_sync", "T_timestep"),
        ("T_timestep", "T_timesteps"), ("npv", "T_timesteps"),
        ("launch", "T_total"), ("T_timesteps", "T_total"),
    ]
    for s, d in edges:
        dot_lines.append(f"  {s} -> {d};")
    dot_lines.append("}")

    st.graphviz_chart("\n".join(dot_lines), use_container_width=True)

    # Waterfall + per-timestep bar
    st.subheader("Where the time goes")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Launch", "Σ timesteps", "T_total"],
            y=[comm_hw.launch_overhead_ns/1000, T_timesteps_ns/1000, pred_us],
            text=[f"{comm_hw.launch_overhead_ns/1000:.1f}",
                  f"+{T_timesteps_ns/1000:.1f}",
                  f"={pred_us:.1f}"],
            textposition="outside",
            connector={"line": {"color": "#888"}},
        ))
        fig_wf.update_layout(yaxis_title="µs", height=360,
                             title="Sequential path: T = launch + Σ T_timestep_i")
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_b:
        labels = [f"t{i} ({tag})" for i, (tag, _, _) in enumerate(ts_times)]
        ys = [t/1000 for _, t, _ in ts_times]
        bottlenecks = [b for _, _, b in ts_times]
        colors = ['#90A4AE' if 'SELF' in lbl else '#42A5F5' for lbl in labels]
        fig_pv = go.Figure(go.Bar(
            x=labels, y=ys, marker_color=colors,
            hovertext=[f"bottleneck: {b}" for b in bottlenecks],
        ))
        fig_pv.update_layout(yaxis_title="µs per timestep", height=360,
                             title=f"Per-timestep time ({num_ts} timesteps)")
        st.plotly_chart(fig_pv, use_container_width=True)


with tab_graph:
    st.markdown(
        "Nodes are quantities, edges are dependencies. Red = bottleneck functional unit. "
        "Use this view to see structural dependencies; use the Trace tab for formulas + values."
    )
    if is_ring:
        _render_ring_dag()
    else:
        _render_seq_dag()

# ─── Step 6: Total latency ───
st.header("6. Total Collective Latency")

st.markdown(f"""
```
T_launch:       {comm_hw.launch_overhead_ns/1000:>8.1f} µs
T_timesteps:  {(pred_ns - comm_hw.launch_overhead_ns)/1000:>8.1f} µs  ({num_ts} timesteps)
──────────────────────────────
T_total:        {pred_us:>8.1f} µs
```
""")

# ─── Step 7: Comparison with measured data ───
st.header("7. Comparison with Measured Data")

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "rccl_master_sweep.csv")

if os.path.exists(data_path):
    measured = defaultdict(list)
    with open(data_path) as f:
        for row in csv.DictReader(f):
            if row["primitive"] != collective: continue
            if int(row["world_size"]) != world_size: continue
            if int(row["nchannels"]) != num_channels: continue
            mb = int(row["msg_bytes"])
            if mb == 0: continue
            meas = float(row["latency_us"])
            if meas <= 0: continue
            measured[mb].append(meas)

    if measured:
        AMP = {
            "all_reduce": lambda N: 2 * (N-1) / N,
            "reduce_scatter": lambda N: (N-1) / N,
            "all_gather": lambda N: (N-1) / N,
            "all_to_all": lambda N: (N-1) / N,
            "broadcast": lambda N: 1.0,
        }
        amp = AMP[collective](world_size)

        msgs = sorted(measured.keys())
        meas_lat = [np.median(measured[m]) for m in msgs]
        meas_bw = [m * amp / (lat * 1000) for m, lat in zip(msgs, meas_lat)]

        pred_lats = []
        pred_bws = []
        for m in msgs:
            p = CommProblem(M=1, N=m // dtype_bytes(DataType.BF16), num_gpus=world_size)
            c = CommConfig(num_wgs=num_channels)
            t = hw.cycles_to_us(compute_collective_latency(collective, p, c, hw, comm_hw))
            pred_lats.append(t)
            pred_bws.append(m * amp / (t * 1000) if t > 0 else 0)

        col1, col2 = st.columns(2)

        with col1:
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Scatter(x=msgs, y=meas_lat, mode='lines+markers',
                                         name='Measured', line=dict(color='#1E88E5', width=2),
                                         marker=dict(size=5)))
            fig_lat.add_trace(go.Scatter(x=msgs, y=pred_lats, mode='lines+markers',
                                         name='Predicted', line=dict(color='red', width=2, dash='dash'),
                                         marker=dict(size=5, symbol='square')))
            fig_lat.update_layout(title="Latency vs Message Size",
                                  xaxis_title="Message Size (bytes)",                                   yaxis_title="Latency (µs)",                                   height=400)
            st.plotly_chart(fig_lat, use_container_width=True)

        with col2:
            fig_bw = go.Figure()
            fig_bw.add_trace(go.Scatter(x=msgs, y=meas_bw, mode='lines+markers',
                                        name='Measured', line=dict(color='#1E88E5', width=2),
                                        marker=dict(size=5)))
            fig_bw.add_trace(go.Scatter(x=msgs, y=pred_bws, mode='lines+markers',
                                        name='Predicted', line=dict(color='red', width=2, dash='dash'),
                                        marker=dict(size=5, symbol='square')))
            fig_bw.update_layout(title="Bus Bandwidth vs Message Size",
                                  xaxis_title="Message Size (bytes)",                                   yaxis_title="Bus BW (GB/s)",
                                  height=400)
            st.plotly_chart(fig_bw, use_container_width=True)

        # Current point
        if msg_bytes in measured:
            meas_val = np.median(measured[msg_bytes])
            err = (pred_us - meas_val) / meas_val * 100
            st.metric("Error at selected message size",
                      f"{err:+.1f}%",
                      f"Predicted {pred_us:.0f} µs vs Measured {meas_val:.0f} µs")

        # ─── NCH sweep at fixed message size ───
        st.subheader(f"Channel Sweep at {msg_bytes/(1024**2):.0f} MB" if msg_bytes >= 1048576
                     else f"Channel Sweep at {msg_bytes/1024:.0f} KB")

        nch_values = [1, 2, 4, 8, 16, 32, 64, 128]
        nch_meas_lat = []
        nch_pred_lat = []
        nch_meas_bw = []
        nch_pred_bw = []
        nch_valid = []

        for nch_sweep in nch_values:
            # Measured: find this (collective, W, NCH, msg_bytes)
            meas_vals = []
            for row in csv.DictReader(open(data_path)):
                if row["primitive"] != collective: continue
                if int(row["world_size"]) != world_size: continue
                if int(row["nchannels"]) != nch_sweep: continue
                if int(row["msg_bytes"]) != msg_bytes: continue
                m = float(row["latency_us"])
                if m > 0:
                    meas_vals.append(m)

            # Predicted (always compute, even without measured data)
            p = CommProblem(M=tensor_m, N=tensor_n, num_gpus=world_size, split_dim=split_dim)
            c = CommConfig(num_wgs=nch_sweep)
            try:
                pred_t = hw.cycles_to_us(compute_collective_latency(collective, p, c, hw, comm_hw))
            except:
                pred_t = 0

            if pred_t > 0:
                nch_pred_lat.append(pred_t)
                nch_pred_bw.append(msg_bytes * amp / (pred_t * 1000))
                nch_valid.append(nch_sweep)

                if meas_vals:
                    meas_t = np.median(meas_vals)
                    nch_meas_lat.append(meas_t)
                    nch_meas_bw.append(msg_bytes * amp / (meas_t * 1000))
                else:
                    nch_meas_lat.append(None)
                    nch_meas_bw.append(None)

        if nch_valid:
            # Filter measured to only non-None points
            meas_nch = [n for n, v in zip(nch_valid, nch_meas_lat) if v is not None]
            meas_lat_filt = [v for v in nch_meas_lat if v is not None]
            meas_bw_filt = [v for v in nch_meas_bw if v is not None]

            col1, col2 = st.columns(2)

            with col1:
                fig_nch_lat = go.Figure()
                if meas_lat_filt:
                    fig_nch_lat.add_trace(go.Scatter(x=meas_nch, y=meas_lat_filt, mode='lines+markers',
                                                      name='Measured', line=dict(color='#1E88E5', width=2),
                                                      marker=dict(size=6)))
                fig_nch_lat.add_trace(go.Scatter(x=nch_valid, y=nch_pred_lat, mode='lines+markers',
                                                  name='Predicted', line=dict(color='red', width=2, dash='dash'),
                                                  marker=dict(size=6, symbol='square')))
                fig_nch_lat.update_layout(title=f"Latency vs Channels (msg={msg_bytes/(1024**2):.0f}MB)"
                                          if msg_bytes >= 1048576 else
                                          f"Latency vs Channels (msg={msg_bytes/1024:.0f}KB)",
                                          xaxis_title="Channels (NCH)",                                           yaxis_title="Latency (µs)",                                           height=350)
                st.plotly_chart(fig_nch_lat, use_container_width=True)

            with col2:
                fig_nch_bw = go.Figure()
                if meas_bw_filt:
                    fig_nch_bw.add_trace(go.Scatter(x=meas_nch, y=meas_bw_filt, mode='lines+markers',
                                                     name='Measured', line=dict(color='#1E88E5', width=2),
                                                     marker=dict(size=6)))
                fig_nch_bw.add_trace(go.Scatter(x=nch_valid, y=nch_pred_bw, mode='lines+markers',
                                                 name='Predicted', line=dict(color='red', width=2, dash='dash'),
                                                 marker=dict(size=6, symbol='square')))
                fig_nch_bw.update_layout(title=f"Bus BW vs Channels (msg={msg_bytes/(1024**2):.0f}MB)"
                                         if msg_bytes >= 1048576 else
                                         f"Bus BW vs Channels (msg={msg_bytes/1024:.0f}KB)",
                                         xaxis_title="Channels (NCH)",                                          yaxis_title="Bus BW (GB/s)",
                                         height=350)
                st.plotly_chart(fig_nch_bw, use_container_width=True)

    else:
        st.info("No measured data for this (collective, W, NCH) combination")
else:
    st.warning(f"Data file not found: {data_path}")


# ─── Step 8: Shareable snapshot ───
st.divider()
st.header("8. Copy snapshot to share")

st.markdown(
    "If something looks wrong, click the copy icon in the top-right of the block "
    "below and paste it into chat — it bundles every input, the full trace, and "
    "the measured-vs-predicted error for this exact configuration."
)

def _build_snapshot() -> str:
    lines = []
    lines.append("## Origami-comms dashboard snapshot")
    lines.append("")
    lines.append("### Configuration")
    lines.append("")
    lines.append(f"- collective: `{collective}`")
    lines.append(f"- world_size (N): `{world_size}`")
    lines.append(f"- num_channels (NCH): `{num_channels}`")
    lines.append(f"- tensor: `[{tensor_m}, {tensor_n}]` {problem.dtype.name}, "
                 f"split_dim={split_dim}")
    lines.append(f"- msg_bytes: `{problem.message_bytes:,}` "
                 f"({problem.message_bytes/(1024**2):.3f} MiB)")
    lines.append(f"- per-GPU tile: `[{problem.gpu_tile_m}, {problem.gpu_tile_n}]`  "
                 f"({problem.gpu_tile_bytes:,} B, {problem.gpu_tile_cachelines:,} CLs, "
                 f"{problem.cacheline_efficiency:.1%} efficient)")
    lines.append(f"- layout class: `{type(layout).__name__}`  "
                 f"(num_timesteps = {num_ts}, ring path = {is_ring})")
    lines.append("")

    lines.append("### Hardware (sidebar-adjusted)")
    lines.append("")
    lines.append(f"- link_bw: `{comm_hw.link_bw_ns:.2f} GB/s` "
                 f"(default {MI300X_COMM.link_bw_ns:.2f})")
    lines.append(f"- mshr_depth_per_wave: `{hw.mshr_depth_per_wave}` "
                 f"(default {MI300X.mshr_depth_per_wave})")
    lines.append(f"- waves_per_wg: `{hw.waves_per_wg}` "
                 f"(default {MI300X.waves_per_wg})")
    lines.append(f"- xgmi_latency_ns: `{hw.xgmi_latency_ns:.0f}` "
                 f"(default {MI300X.xgmi_latency_ns:.0f})")
    lines.append(f"- launch_overhead_ns: `{comm_hw.launch_overhead_ns:.0f}` "
                 f"(default {MI300X_COMM.launch_overhead_ns:.0f})")
    lines.append(f"- derived per-WG MSHR cap: "
                 f"`{(hw.mshr_depth_per_wave * hw.waves_per_wg * CACHELINE_BYTES) / hw.xgmi_latency_ns:.2f} GB/s`")
    lines.append("")

    # Measured lookup for this exact (collective, W, NCH, msg_bytes)
    meas_summary = "(no measurement file)"
    err_str = ""
    snap_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "data", "rccl_master_sweep.csv")
    if os.path.exists(snap_data_path):
        vals = []
        with open(snap_data_path) as fh:
            for row in csv.DictReader(fh):
                if row["primitive"] != collective: continue
                if int(row["world_size"]) != world_size: continue
                if int(row["nchannels"]) != num_channels: continue
                if int(row["msg_bytes"]) != problem.message_bytes: continue
                try:
                    v = float(row["latency_us"])
                    if v > 0:
                        vals.append(v)
                except Exception:
                    pass
        if vals:
            arr = sorted(vals)
            med = arr[len(arr)//2]
            mn, mx = arr[0], arr[-1]
            err = (pred_us - med) / med * 100
            meas_summary = (f"median {med:.2f} µs, min {mn:.2f}, max {mx:.2f} "
                            f"(n={len(vals)})")
            err_str = f" → error {err:+.1f}%"
        else:
            meas_summary = "(no rows for this exact config)"

    lines.append("### Prediction vs measured")
    lines.append("")
    lines.append(f"- **predicted T_total**: `{pred_us:.2f} µs`")
    lines.append(f"- **measured latency_us**: {meas_summary}{err_str}")
    lines.append("")

    # Trace table as markdown
    lines.append("### Trace")
    lines.append("")
    sec_order = []
    for s, *_ in trace_rows:
        if s not in sec_order:
            sec_order.append(s)
    for sec in sec_order:
        lines.append(f"#### {sec}")
        lines.append("")
        lines.append("| quantity | formula | inputs | value | unit | note |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in trace_rows:
            if r[0] != sec:
                continue
            # Escape pipe chars so they don't break the markdown table
            cells = [(c if c is not None else "") for c in r[1:]]
            cells = [str(c).replace("|", "\\|") for c in cells]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Per-timestep time vector
    if ts_times:
        lines.append("### Per-timestep T_timestep (µs)")
        lines.append("")
        for i, (tag, t_ns) in enumerate(ts_times):
            lines.append(f"- t{i} ({tag}): `{t_ns/1000:.2f}` µs")
        lines.append(f"- **Σ timesteps**: `{sum(t for _, t in ts_times)/1000:.2f}` µs")
        lines.append(f"- launch_overhead: `{comm_hw.launch_overhead_ns/1000:.2f}` µs")
        lines.append(f"- **T_total**: `{pred_us:.2f}` µs")
        lines.append("")

    lines.append(f"_Generated by `dashboard/app.py` for collective `{collective}`, "
                 f"W={world_size}, NCH={num_channels}, msg={problem.message_bytes} B._")

    return "\n".join(lines)

snapshot_md = _build_snapshot()

# st.code() renders a built-in copy icon in the top-right corner
st.code(snapshot_md, language="markdown")

with st.expander("Preview rendered snapshot"):
    st.markdown(snapshot_md)
