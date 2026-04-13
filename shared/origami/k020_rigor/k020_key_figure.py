#!/usr/bin/env python3
"""Generate key_result_rigor.png — classification sensitivity to kernel choice"""
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("matplotlib not available, skipping figure generation")
    sys.exit(0)

REPORT_DIR = "/home/ryaswann/global_orchestrator/reports/tasks/K-020"
OUT_DIR = "/home/ryaswann/global_orchestrator/reports/tasks/K-020/rigor"

# Data from the analysis
shapes = ['MLP', 'Attn', 'Sq8K', 'Lg16K']
alpha_be_rocblas = [0.883187, 0.070707, 0.424648, 0.191449]
alpha_be_ws = [0.402167, 0.019164, 0.192680, 0.082996]

D2D_MAX = 0.124
GO_THRESHOLD = 0.248

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 2]})

# ─── Panel 1: Alpha_BE comparison ───────────────────────────────────────
x = np.arange(len(shapes))
width = 0.35

bars_roc = ax1.bar(x - width/2, alpha_be_rocblas, width, label='rocBLAS (Job 17451, 612 TF)', color='#2196F3', edgecolor='black', linewidth=0.5)
bars_ws = ax1.bar(x + width/2, alpha_be_ws, width, label='WS kernel (Job 261022, ~288 TF)', color='#FF9800', edgecolor='black', linewidth=0.5)

# Threshold lines
ax1.axhline(y=GO_THRESHOLD, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'GO threshold ({GO_THRESHOLD})')
ax1.axhline(y=D2D_MAX, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'MONITOR/NO-GO ({D2D_MAX})')

# Annotations for flips
for i, s in enumerate(shapes):
    roc_cls = 'GO' if alpha_be_rocblas[i] > GO_THRESHOLD else ('MONITOR' if alpha_be_rocblas[i] >= D2D_MAX else 'NO-GO')
    ws_cls = 'GO' if alpha_be_ws[i] > GO_THRESHOLD else ('MONITOR' if alpha_be_ws[i] >= D2D_MAX else 'NO-GO')
    if roc_cls != ws_cls:
        ax1.annotate(f'FLIP!\n{roc_cls}→{ws_cls}', xy=(i + width/2, alpha_be_ws[i]),
                    xytext=(i + width/2 + 0.3, alpha_be_ws[i] + 0.05),
                    fontsize=9, fontweight='bold', color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax1.set_ylabel('α_BE (breakeven interference tolerance)', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(shapes, fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.legend(loc='upper left', fontsize=8)
ax1.set_title('Classification Sensitivity to Kernel Choice\nrocBLAS vs WS-Realistic TFLOPS', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# ─── Panel 2: BW margin table ──────────────────────────────────────────
ax2.axis('off')
table_data = [
    ['Shape', 'α_BE\n(rocBLAS)', 'α_BE\n(WS)', 'Class\n(rocBLAS)', 'Class\n(WS)', 'BW margin\nto flip'],
    ['MLP', '0.883', '0.402', 'GO', 'GO', '-232% ↓'],
    ['Attn', '0.071', '0.019', 'NO-GO', 'NO-GO', 'always'],
    ['Sq8K', '0.425', '0.193', 'GO', 'MONITOR', '-64% ↓'],
    ['Lg16K', '0.191', '0.083', 'MONITOR', 'NO-GO', '-45% ↓'],
]

colors_cells = [['#E0E0E0'] * 6]
for i, row in enumerate(table_data[1:]):
    row_colors = []
    for j, cell in enumerate(row):
        if 'GO' == cell and j >= 3:
            row_colors.append('#C8E6C9')
        elif 'MONITOR' == cell and j >= 3:
            row_colors.append('#FFF9C4')
        elif 'NO-GO' == cell and j >= 3:
            row_colors.append('#FFCDD2')
        else:
            row_colors.append('white')
    colors_cells.append(row_colors)

table = ax2.table(cellText=table_data, cellColours=colors_cells,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.8)

# Header styling
for j in range(6):
    table[0, j].set_facecolor('#424242')
    table[0, j].set_text_props(color='white', fontweight='bold')

ax2.set_title('Classification Summary\n& BW Safety Margins (exp=1.0)', fontsize=12, fontweight='bold')

plt.suptitle('K-020 Rigor: α_BE Stress Test — MI300X (OCI Job 17451 + Alola Job 261022)',
            fontsize=11, y=0.02, fontweight='normal', color='gray')
plt.figtext(0.5, -0.02, 
    'python3 k020_rigor_analysis.py | Branch: k020/origami-model-for-coexecuted-work-steali-rigor',
    ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'key_result_rigor.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/key_result_rigor.png")

