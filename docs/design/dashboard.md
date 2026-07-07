# Comm Cost Model — Validation Dashboard

Python prototype (Streamlit + Plotly). Two core visualizations: a multi-layer throughput roofline and a normalized latency scaling curve.

## Visualization 1: Multi-Layer Throughput Roofline

One roofline per SoC level, each plotting measured throughput ceilings as horizontal lines. The communication workload's achieved throughput is plotted against these ceilings to show which hardware resource is binding.

### Per-CU Level — Functional Units

The per-CU roofline shows the parallel functional units and how much work each must do per workgroup. For a comm WG the work is: read N_cl cache lines (VMEM), optionally stage through LDS, write N_cl cache lines to xGMI. The **slowest functional unit** determines the WG's throughput.

| Functional Unit | Throughput | Unit | Load Width Impact |
|----------------|-----------|------|-------------------|
| VMEM read (coalesced) | 110 GB/s | per CU | dwordx16: 1 instr/cl; dwordx4: 4 instr/cl |
| VMEM read (scatter) | 3.3 GB/s | per CU | 33× penalty from non-coalesced |
| LDS bandwidth | 128 B/cycle (MI300X), 256 B/cycle (MI350X) | per CU | — |
| vL1D (TCP) | 32 KB capacity, ~5 wide loads before stall | per CU | Wider loads fill TCP faster |

X-axis: cache lines per workgroup (N_cl = ceil(chunk_bytes_per_wg / 64)).
Y-axis: achieved throughput per CU (cachelines/µs).

Key insight: load width selection changes where the per-CU roofline binds. Narrow loads (dwordx4) hit VMEM instruction issue rate before link BW; wide loads (dwordx16) pass through the CU quickly and bind on link BW or L2 BW instead.

### Per-XCD Level — Cache-Line Throughput

Above the CU, all traffic is denominated in **cache lines** (64 B each). A request for < 64 B is padded to a full cache line.

| Resource | Throughput | Unit |
|----------|-----------|------|
| L2 read bandwidth | measured | cachelines/ns per XCD |
| L2 write bandwidth | measured | cachelines/ns per XCD |
| L2 capacity | 65,536 cl (MI300X), 131,072 cl (MI350X) | cachelines per XCD |
| DF port bandwidth | request-proportional share | cachelines/ns per XCD |

Aggregate XCD traffic = N_comm_cus × N_cl_per_wg cache lines.
Shows whether L2 bandwidth or DF port becomes the bottleneck as more comm WGs share an XCD.

### Per-Device Level — Cache-Line Throughput

| Resource | Throughput | Unit |
|----------|-----------|------|
| HBM read bandwidth | 73.9M cachelines/ms (4.73 TB/s / 64 B) | per GPU |
| HBM write bandwidth | 80.3M cachelines/ms (5.14 TB/s / 64 B) | per GPU |
| MALL bandwidth | 304 ns/hit | per GPU |
| Per-GPU xGMI egress | 769K cachelines/ms (49.2 GB/s / 64 B) | per GPU |

The egress ceiling (769K cl/ms) is ~96× lower than HBM read (73.9M cl/ms) — for standalone communication, xGMI egress always binds before HBM. Under concurrent GEMM, HBM headroom shrinks.

### Per-Fabric Level — Cache-Line Throughput

| Resource | Throughput | Unit |
|----------|-----------|------|
| Per-link bandwidth | 767K cachelines/ms (49.1 GiB/s / 64 B) | per link |
| Fabric aggregate | 5.2M cachelines/ms (334 GB/s / 64 B) | 8-GPU total |
| Per-GPU egress | 769K cachelines/ms | per GPU |

Shows how the collective algorithm's step structure maps to cache-line transfer rates across links.

---

## Visualization 2: Normalized Latency vs Workgroup Count

The primary validation of the model's correctness. Tests whether the model correctly predicts how latency scales as workgroup count increases.

### What it shows

- X-axis: number of workgroups (1 to N_CU, log scale)
- Y-axis: normalized latency (T / T_single_wg)
- Predicted curve (from model)
- Measured points (from `rccl_master_sweep.csv` and `rccl_cu_occupancy.csv`)

### Why this matters

The model claims `T_comm` is determined by the slowest per-CU functional unit scaled by how many CUs execute in parallel. If the model is correct:

- **Below CU saturation**: adding WGs should reduce latency ~linearly (more parallelism, fixed total work)
- **At saturation**: adding WGs provides zero benefit (link bandwidth is the ceiling)
- **Above saturation**: adding WGs may degrade latency (DF contention, L2 pollution)

The normalized latency curve exposes whether the model captures these regimes correctly:

```
    T / T_1wg │
              │
         1.0  ┤ ●                                  ← 1 WG baseline
              │   ●
              │     ●
              │       ●                             ← linear scaling regime
              │         ●                              (parallelism helps)
              │           ●
              │             ●─────●─────●─────●     ← saturation plateau
              │                                        (link BW ceiling)
              │                              ●  ●   ← potential degradation
              │                                        (contention)
              │
              └──────────────────────────────────── num_workgroups
                  1    4    8   16   32   64  128  256
```

### Correlation test

For each (collective, msg_size, world_size) triple:

```python
# Measured: latency at each nchannels (proxy for WG count)
measured = master.query("primitive==coll & msg_bytes==size & world_size==ws")
                  .sort_values("nchannels")

# Predicted: model output at each WG count
predicted = [predict(coll, size, ws, nch) for nch in measured.nchannels]

# Normalize both to single-channel baseline
measured_norm = measured.latency_us / measured.latency_us.iloc[0]
predicted_norm = predicted / predicted[0]

# Correlation: does the model capture the shape?
r = np.corrcoef(measured_norm, predicted_norm)[0, 1]
```

The dashboard reports:
- **Per-collective Pearson r**: correlation between predicted and measured normalized scaling
- **Saturation knee accuracy**: does the predicted knee (WG count where adding more stops helping) match measured?
- **Regime classification accuracy**: for each point, does the model correctly identify latency-bound vs bandwidth-bound?

---

## Dashboard Pages

### Page 1: Multi-Layer Roofline

Four stacked roofline plots (CU → XCD → Device → Fabric), each with:
- Horizontal lines: throughput ceilings in cachelines/µs for each resource at that level
- Predicted operating point for the selected config
- Binding constraint highlighted (which ceiling is active)

Interactive controls:
- Collective selector
- World size (2, 4, 8)
- Message size slider (1 KB – 1 GiB)
- Load width selector (dword, dwordx4, dwordx16) — shows impact on per-CU roofline
- Architecture (MI300X / MI350X)

### Page 2: Normalized Latency vs Workgroup Count (Primary Validation)

The core correlation test. For a selected (collective, msg_size, world_size):

- X-axis: number of workgroups (controlled via nchannels as proxy)
- Y-axis: T(N_wg) / T(1_wg) — normalized to single-WG baseline
- **Predicted curve** (from model): should show linear scaling → saturation knee → plateau
- **Measured scatter** (from `rccl_master_sweep.csv`)
- Pearson r between predicted and measured normalized curves displayed prominently
- Saturation knee annotated: predicted vs measured WG count where adding more WGs stops helping
- Color: which roofline ceiling (CU functional unit, L2 BW, egress, link BW) is binding at each point

This page answers: **does the model correctly predict the shape of latency scaling, not just absolute values?**

### Page 3: Correlation Heatmap

Rows = (collective, world_size), columns = message size buckets (1KB, 64KB, 1MB, 64MB, 1GB).
Cell color = Pearson r between predicted and measured normalized scaling across WG counts.
Red cells = model breaks. Green cells = strong correlation.

This gives an at-a-glance view of where in the parameter space the model is trustworthy.

### Page 4: Absolute Accuracy

Scatter plot of predicted vs measured latency for all 33,690 rows.
45° line = perfect prediction. Colored by binding constraint from the roofline.
Error distribution histogram. Separate panels per collective.

---

## Data Sources

| File | Rows | Primary Use |
|------|------|-------------|
| `~/comm_data/comm_only/rccl_master_sweep.csv` | 33,690 | Scaling curves, absolute validation |
| `~/comm_data/comm_only/rccl_channel_sweep.csv` | 5,760 | Channel (WG) scaling validation |
| `~/comm_data/comm_only/rccl_cu_occupancy.csv` | 1,009 | CU saturation point validation |

## File Structure

```
dashboard/
  app.py                  # Streamlit entry point
  model/
    __init__.py
    types.py              # comm_problem_t, comm_config_t, comm_hardware_t
    hardware.py           # MI300X / MI350X architectural constants
    latency.py            # compute_comm_latency() hierarchy
  viz/
    roofline.py           # Multi-layer throughput roofline
    scaling.py            # Normalized latency vs WG count
    correlation.py        # Sweep correlation heatmap
    scatter.py            # Predicted vs measured absolute
  data/
    loader.py             # Load from ~/comm_data/comm_only/
```
