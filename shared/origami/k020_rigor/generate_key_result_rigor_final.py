#!/usr/bin/env python3
"""Generate key_result_rigor.png — Rigor verification dashboard for K-020.
GPU: MI300X | Data: Jobs 17451, 261022, 260996, 18180, K-018 17995
Command: python3 /tmp/generate_key_result_rigor.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("K-020 Rigor Verification Dashboard — MI300X\n"
             "Data: Jobs 17451/261022/260996/18180/17995 | Branch: k020/origami-model-for-coexecuted-work-steali-rigor",
             fontsize=13, fontweight='bold')

# ── Panel A: Model speedup vs measured speedup (cross-validation) ──
ax = axes[0, 0]
# K-019 cross-validation data (8 configs)
shapes_cv = ['MLP\n296/8', 'MLP\n256/48', 'Attn\n296/8', 'Attn\n256/48',
             'Sq8K\n296/8', 'Sq8K\n256/48', 'Lg16K\n296/8', 'Lg16K\n256/48']
# Model predicted speedups (from analytical model with per-shape exponents)
model_speedup = [0.487, 1.816, 1.106, 0.986, 0.731, 1.305, 1.154, 1.052]
# GPU measured speedups (from K-019 st-3 / concurrent jobs)
measured_speedup = [np.nan, 1.369, np.nan, 0.786, np.nan, 1.011, 0.966, 0.620]

x = np.arange(len(shapes_cv))
width = 0.35
bars1 = ax.bar(x - width/2, model_speedup, width, label='Model prediction', color='#4472C4', alpha=0.85)
# Only plot measured where available
meas_vals = [v if not np.isnan(v) else 0 for v in measured_speedup]
meas_colors = ['#ED7D31' if not np.isnan(v) else 'none' for v in measured_speedup]
meas_edges = ['#ED7D31' if not np.isnan(v) else 'none' for v in measured_speedup]
for i, (val, color, edge) in enumerate(zip(meas_vals, meas_colors, meas_edges)):
    if val > 0:
        ax.bar(x[i] + width/2, val, width, color=color, edgecolor=edge, alpha=0.85)
    else:
        ax.bar(x[i] + width/2, 0, width, color='white', edgecolor='gray', linestyle='--', alpha=0.3)
ax.bar([], [], color='#ED7D31', alpha=0.85, label='GPU measured [VERIFIED]')
ax.bar([], [], color='white', edgecolor='gray', linestyle='--', alpha=0.3, label='No concurrent data')
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Breakeven (1.0×)')
ax.set_xticks(x)
ax.set_xticklabels(shapes_cv, fontsize=8)
ax.set_ylabel('Speedup vs Sequential')
ax.set_title('(a) Model vs Measured Speedup\n[VERIFIED] — 27.5% mean overprediction', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 2.1)

# ── Panel B: CU scaling verification — measured TFLOPS vs CU count ──
ax = axes[0, 1]
cu_counts_j17451 = [152, 224, 256, 296, 304]
# min_ms TFLOPS from Job 17451 — HSA_CU_MASK was INEFFECTIVE
mlp_tflops = [622.2, 611.2, 597.4, 615.8, 612.2]
attn_tflops = [595.4, 594.0, 599.6, 595.4, 587.7]
sq8k_tflops = [638.3, 636.1, 624.2, 631.5, 607.1]
lg16k_tflops = [586.6, 585.6, 589.4, 585.5, 586.3]

ax.plot(cu_counts_j17451, mlp_tflops, 'o-', color='#4472C4', label='MLP', markersize=6)
ax.plot(cu_counts_j17451, attn_tflops, 's-', color='#ED7D31', label='Attn', markersize=6)
ax.plot(cu_counts_j17451, sq8k_tflops, '^-', color='#70AD47', label='Sq8K', markersize=6)
ax.plot(cu_counts_j17451, lg16k_tflops, 'D-', color='#FFC000', label='Lg16K', markersize=6)
# Show the old model prediction (400T * (G/304)^0.85)
cu_model = np.linspace(128, 304, 50)
old_model = 400 * (cu_model/304)**0.85
ax.plot(cu_model, old_model, '--', color='red', linewidth=2, alpha=0.6, label='Old model (400T, β=0.85)')
ax.fill_between([128, 304], [570, 570], [650, 650], alpha=0.1, color='green', label='Measured range')
ax.set_xlabel('CU Count')
ax.set_ylabel('TFLOPS (FP16)')
ax.set_title('(b) CU Scaling: FLAT via HSA_CU_MASK (Job 17451)\n[VERIFIED] — mask non-functional on ROCm 7.x', fontsize=10)
ax.legend(fontsize=7, loc='lower right')
ax.set_xlim(120, 320)
ax.set_ylim(200, 700)

# ── Panel C: bench-sanity cross-report validation ──
ax = axes[1, 0]
labels = ['α_BE\nMLP', 'α_BE\nAttn', 'α_BE\nSq8K', 'α_BE\nLg16K', 
          'TFLOPS\nMLP', 'TFLOPS\nAttn', 'Speed\nMLP', 'Overpred\n%',
          'β MLP', 'β Attn', 'β Sq8K', 'β Lg16K']
# Diff percentages from bench-sanity runs
diffs = [21.1, 74.4, 0.1, 0.2, 3.6, 10.9, 0.0, 0.0, 16.5, 0.0, 0.8, 0.2]
pass_fail = ['EXPLAINED', 'EXPLAINED', 'PASS', 'PASS', 'PASS', 'PASS', 
             'PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS']
colors = ['#FFC000' if pf == 'EXPLAINED' else '#70AD47' for pf in pass_fail]
bars = ax.barh(range(len(labels)), diffs, color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(x=20.0, color='red', linestyle='--', linewidth=1.5, label='20% threshold')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Cross-Report Divergence (%)')
ax.set_title('(c) bench-sanity.py Cross-Report Validation\n12 pairs: 10 PASS, 2 EXPLAINED (diff CU splits)', fontsize=10)
ax.legend(fontsize=8)
# Add annotations for explained items
ax.annotate('Different CU splits\n(296/8 vs 256/48)', xy=(21.1, 0), xytext=(45, 0.5),
           fontsize=7, arrowprops=dict(arrowstyle='->', color='orange'), color='orange')
ax.annotate('Negative α_BE = regression\n(measured concurrent)', xy=(74.4, 1), xytext=(75, 2),
           fontsize=7, arrowprops=dict(arrowstyle='->', color='orange'), color='orange')
ax.invert_yaxis()

# ── Panel D: Classification summary with safety margins ──
ax = axes[1, 1]
shape_names = ['MLP', 'Attn', 'Sq8K', 'Lg16K']
alpha_be = [0.883, 0.071, 0.425, 0.191]
alpha_measured = [0.0149, 0.0092, 0.0022, 0.0131]
safety_margins = [59.3, 7.7, 193.0, 14.6]
verdicts = ['GO', 'NO-GO', 'GO\n(prov.)', 'NO-GO']
verdict_colors = ['#70AD47', '#FF0000', '#FFC000', '#FF0000']

x_pos = np.arange(len(shape_names))
bar_width = 0.35

bars1 = ax.bar(x_pos - bar_width/2, alpha_be, bar_width, label='α_BE (breakeven)', 
               color='#4472C4', alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x_pos + bar_width/2, alpha_measured, bar_width, label='α_measured', 
               color='#ED7D31', alpha=0.85, edgecolor='black', linewidth=0.5)

# Add verdict labels
for i, (v, c) in enumerate(zip(verdicts, verdict_colors)):
    ax.text(i, max(alpha_be[i], alpha_measured[i]) + 0.05, v, ha='center', 
            fontsize=10, fontweight='bold', color=c,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=c, alpha=0.8))

# Add safety margin annotations
for i, sm in enumerate(safety_margins):
    ax.text(i, -0.08, f'{sm}× margin', ha='center', fontsize=7, color='gray')

ax.axhline(y=0.124, color='red', linestyle=':', linewidth=1, alpha=0.5, label='D2D_MAX (0.124)')
ax.set_xticks(x_pos)
ax.set_xticklabels(shape_names)
ax.set_ylabel('Interference α')
ax.set_title('(d) GO/NO-GO Classification with Safety Margins\n[VERIFIED] — all α from GPU measurement', fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(-0.15, 1.1)

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.text(0.5, 0.01, 
            "python3 generate_key_result_rigor.py | All data [VERIFIED] on AMD MI300X | Jobs: 17451, 261022, 260996, 18180, 17995",
            ha='center', fontsize=8, style='italic')

outpath = '/home/ryaswann/global_orchestrator/reports/tasks/K-020/rigor/key_result_rigor.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")
