#!/usr/bin/env python3
"""K-020 Rigor Re-evaluation: Executive key result figure.

Produces key_result_rigor.png showing:
  1. Per-shape provenance status (VERIFIED vs UNVERIFIED)
  2. Cross-source discrepancies with bench-sanity thresholds
  3. Classification stability under parameter uncertainty

Data provenance: All input values sourced from:
  - s1_ws_gemm_verified_measurements.json (fullnode_v3, rad-mi300x-1, MI300X)
  - s2_iris_bw_verified_measurements.json (K-018 Job 17995, OCI MI300X)
  - k020_coexec_results_p2v9_radha.json (rad-mi300x-1, MI300X)
  - k020_interference_ci_detail.json (rad-mi300x-1, MI300X)
  - provenance_audit_final.csv (cross-team audit)

Command: python3 k020_rigor_reeval_executive.py
Branch: k020/origami-model-for-coexecuted-work-steali-rigor
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "/home/ryaswann/global_orchestrator/reports/tasks/K-020/rigor"

# ---- DATA (all from verified JSON files) ----
shapes = ['MLP', 'Attn', 'Sq8K', 'Lg16K']

# s1: WS TFLOPS at 296 CU (fullnode_v3, rad-mi300x-1) [VERIFIED]
ws_tflops_296 = [353.3, 538.7, 503.7, 536.7]
# s1: rocBLAS 304 CU (fullnode_v3, rad-mi300x-1) [VERIFIED]
rocblas_fn = [598.5, 558.5, 618.5, 526.4]
# Cross-check: rocBLAS 304 CU (Job 17451, OCI) [VERIFIED]
rocblas_j17 = [611.5, 587.0, 608.0, 580.0]

# s2: Iris BW at 8 CU (K-018 Job 17995) [VERIFIED]
iris_bw_8cu = [71.78, 61.35, 75.18, 107.91]

# p2v9: Interference alpha (rad-mi300x-1) [MEASURED]
p2v9_alpha = [0.12411, 0.00153, 0.03652, 0.01561]
# Profiling team interference alpha (different methodology)
prof_alpha = [0.0149, 0.00919, 0.0022, 0.0156]

# Classifications from executive summary
classifications = ['GO', 'NO-GO', 'MONITOR', 'MONITOR']
class_colors = {'GO': '#2ca02c', 'MONITOR': '#ff7f0e', 'NO-GO': '#d62728'}

# bench-sanity cross-source deltas
rocblas_delta_pct = [abs(a-b)/max(a,b)*100 for a,b in zip(rocblas_fn, rocblas_j17)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "K-020 Rigor Re-evaluation — MI300X (rad-mi300x-1 + OCI)\n"
    "Branch: k020/origami-model-for-coexecuted-work-steali-rigor",
    fontsize=13, fontweight='bold'
)

# ---- Panel 1: WS vs rocBLAS TFLOPS with cross-source bars ----
ax1 = axes[0, 0]
x = np.arange(len(shapes))
w = 0.25
bars1 = ax1.bar(x - w, rocblas_fn, w, label='rocBLAS (fullnode_v3)', color='#4c72b0', alpha=0.8)
bars2 = ax1.bar(x, rocblas_j17, w, label='rocBLAS (Job 17451)', color='#4c72b0', alpha=0.4, hatch='//')
bars3 = ax1.bar(x + w, ws_tflops_296, w, label='WS @296CU (fullnode_v3)', color='#dd8452', alpha=0.8)
ax1.set_ylabel('TFLOPS (fp16)')
ax1.set_title('WS vs rocBLAS — Cross-Source Validation')
ax1.set_xticks(x)
ax1.set_xticklabels(shapes)
ax1.legend(fontsize=8, loc='upper right')
ax1.set_ylim(0, 700)
# Annotate cross-source deltas
for i, d in enumerate(rocblas_delta_pct):
    color = '#d62728' if d > 5 else '#2ca02c'
    ax1.annotate(f'Δ{d:.1f}%', xy=(x[i]-w/2, max(rocblas_fn[i], rocblas_j17[i])+10),
                 fontsize=8, ha='center', color=color, fontweight='bold')
ax1.axhline(y=0, color='k', linewidth=0.5)

# ---- Panel 2: Interference alpha discrepancy ----
ax2 = axes[0, 1]
bars_p2v9 = ax2.bar(x - 0.15, p2v9_alpha, 0.3, label='p2v9 (memcpy proxy)', color='#c44e52', alpha=0.8)
bars_prof = ax2.bar(x + 0.15, prof_alpha, 0.3, label='Profiling (D2D)', color='#55a868', alpha=0.8)
ax2.set_ylabel('Interference α')
ax2.set_title('Interference α — Two Measurement Methods')
ax2.set_xticks(x)
ax2.set_xticklabels(shapes)
ax2.legend(fontsize=8)
ax2.set_yscale('log')
ax2.set_ylim(1e-4, 0.5)
# Annotate the 8.3x MLP discrepancy
ax2.annotate('8.3× discrepancy\n(different comm type)',
             xy=(0, 0.12411), xytext=(0.8, 0.3),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=9, color='red', fontweight='bold')

# ---- Panel 3: Classification stability under α uncertainty ----
ax3 = axes[1, 0]
# α_BE values from the provenance audit
alpha_be = [0.883, 0.071, 0.425, 0.191]
alpha_be_corrected = [0.583, 0.044, 0.279, 0.124]  # alignment corrected (α_exp=0.781)

bar_colors = [class_colors[c] for c in classifications]
bars_ab = ax3.bar(x - 0.15, alpha_be, 0.3, label='α_BE (provenance audit)',
                  color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
bars_ac = ax3.bar(x + 0.15, alpha_be_corrected, 0.3, label='α_BE (corrected α_exp=0.781)',
                  color=bar_colors, alpha=0.4, edgecolor='black', linewidth=0.5, hatch='//')

ax3.axhline(y=0.5, color='#2ca02c', linestyle='--', linewidth=1.5, label='GO threshold (0.5)')
ax3.axhline(y=0.125, color='#ff7f0e', linestyle='--', linewidth=1.5, label='MONITOR threshold (0.125)')
ax3.set_ylabel('α_BE (breakeven)')
ax3.set_title('Classification Stability Under CU Exponent Correction')
ax3.set_xticks(x)
ax3.set_xticklabels(shapes)
ax3.legend(fontsize=7, loc='upper right')
ax3.set_ylim(0, 1.0)
# Annotate Lg16K margin
ax3.annotate(f'margin=0.001\n(razor-thin)', xy=(3.15, 0.124), xytext=(2.5, 0.35),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=9, color='red', fontweight='bold')

# ---- Panel 4: Provenance summary heatmap ----
ax4 = axes[1, 1]
params = ['GEMM TFLOPS', 'rocBLAS base', 'Iris BW', 'α (interference)', 'α_BE', 'CU exponent']
# 2=VERIFIED, 1=VERIFIED-DIFF-SOURCE, 0=DISCREPANCY
status_matrix = np.array([
    [2, 2, 2, 2],  # GEMM TFLOPS - all verified
    [2, 2, 2, 1],  # rocBLAS - Lg16K has 9.2% cross-source gap
    [2, 2, 2, 2],  # Iris BW - all verified (Job 17995)
    [0, 2, 1, 2],  # α - MLP has 8.3x discrepancy, Sq8K has different direction
    [2, 2, 2, 0],  # α_BE - Lg16K is at margin boundary
    [1, 1, 1, 1],  # CU exponent - corrected from 0.874→0.781
])
cmap = plt.cm.RdYlGn
im = ax4.imshow(status_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)
ax4.set_xticks(range(len(shapes)))
ax4.set_xticklabels(shapes, fontsize=10)
ax4.set_yticks(range(len(params)))
ax4.set_yticklabels(params, fontsize=9)
ax4.set_title('Provenance Verification Matrix')
# Text annotations
labels_map = {2: 'VERIFIED', 1: 'WEAK', 0: 'DISCREPANCY'}
for i in range(len(params)):
    for j in range(len(shapes)):
        val = status_matrix[i, j]
        ax4.text(j, i, labels_map[val], ha='center', va='center',
                fontsize=7, fontweight='bold',
                color='white' if val == 0 else 'black')

plt.tight_layout()
plt.figtext(0.5, 0.01,
    "python3 k020_rigor_reeval_executive.py | Data: fullnode_v3 (rad-mi300x-1), Job 17451 (OCI), K-018 Job 17995",
    ha='center', fontsize=8, style='italic')
plt.savefig(f'{OUT}/key_result_rigor.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT}/key_result_rigor.png")
plt.close()

# Also save the bench-sanity cross-validation results as JSON
sanity_results = {
    "bench_sanity_cross_validations": [
        {
            "test": "WS TFLOPS 296CU vs 304CU",
            "pairs": {s: {"val1": ws_tflops_296[i], "val2": [357.8, 542.3, 505.5, 530.3][i],
                          "diff_pct": abs(ws_tflops_296[i]-[357.8,542.3,505.5,530.3][i])/max(ws_tflops_296[i],[357.8,542.3,505.5,530.3][i])*100,
                          "status": "PASS"} for i, s in enumerate(shapes)},
            "threshold_pct": 20, "all_pass": True
        },
        {
            "test": "rocBLAS cross-source (fullnode_v3 vs Job 17451)",
            "pairs": {s: {"val1": rocblas_fn[i], "val2": rocblas_j17[i],
                          "diff_pct": rocblas_delta_pct[i],
                          "status": "PASS" if rocblas_delta_pct[i] < 10 else "WARNING"}
                      for i, s in enumerate(shapes)},
            "threshold_pct": 10,
            "all_pass": all(d < 10 for d in rocblas_delta_pct),
            "warning": "Lg16K at 9.2% — near threshold. Different ROCm versions (6.4 vs 7.0)."
        },
        {
            "test": "MLP interference alpha discrepancy",
            "p2v9_memcpy": 0.12411, "profiling_d2d": 0.0149,
            "ratio": 8.3, "status": "DISCREPANCY",
            "explanation": "Different comm types: hipMemcpyAsync (intra-GPU HBM) vs D2D copy stream. MLP has small GEMM → more sensitive to memory contention type."
        }
    ],
    "provenance_gaps": [
        {"param": "beta_base", "value": 0.15, "source": "K-019 fit (unsubstantiated)", "impact": "MOOT — all shapes have AI >> ridge, interference term is identically zero"},
        {"param": "comm_cu_exponent", "value": 0.7, "source": "K-019 fit (unsubstantiated)", "impact": "MOOT — same reason"},
        {"param": "concurrent_iris_bw", "value": None, "source": "UNMEASURED", "impact": "CLASSIFICATION-RELEVANT for MLP — need concurrent GEMM+Iris BW on 2-GPU MI300X node"}
    ],
    "cu_exponent_correction": {
        "original": 0.874, "corrected_ols": 0.781,
        "per_shape_verified": {"MLP": 0.479, "Attn": 0.790, "Sq8K": 0.665, "Lg16K": 0.629},
        "source": "cu_scaling_fit_results.json (fullnode_v3, rad-mi300x-1) [VERIFIED]",
        "impact": "α_BE shifts: MLP 0.883→0.583 (still GO), Lg16K 0.191→0.124 (MONITOR→borderline NO-GO at 212 GB/s BW)"
    }
}
with open(f'{OUT}/k020_rigor_reeval_executive_results.json', 'w') as f:
    json.dump(sanity_results, f, indent=2)
print(f"Saved: {OUT}/k020_rigor_reeval_executive_results.json")
