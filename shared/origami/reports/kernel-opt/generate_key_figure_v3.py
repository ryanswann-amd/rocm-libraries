#!/usr/bin/env python3
"""Generate key result figure for K-016 kernel-opt root-cause analysis.
Data source: regret_decomposition_summary.csv + top5_root_cause.csv (MI300X Tensile sweep)
Branch: k016/triton-specialization-in-origami-kernel-opt @ 45bfabb1e2
"""
import csv
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
K016_DIR = os.path.dirname(REPORT_DIR)
MI300X_CUS = 304

# Read top-5 root cause data
shapes = []
with open(os.path.join(REPORT_DIR, 'top5_root_cause.csv')) as f:
    reader = csv.DictReader(f)
    for row in reader:
        shapes.append(row)

# Read k3 per-shape regret for histogram
k3_regrets = []
with open(os.path.join(K016_DIR, 'benchmarking', 'k3_per_shape_regret.csv')) as f:
    reader = csv.DictReader(f)
    for row in reader:
        k3_regrets.append(float(row['k3_regret_pct']))

# Compute wave quantization data
for s in shapes:
    name = s['shape']
    dims = name.replace('_bf16','').replace('_f16','').split('x')
    M, N, K = int(dims[0]), int(dims[1]), int(dims[2])
    ot = s['oracle_tile'].split('x')
    pt = s['picked_tile'].split('x')
    oM, oN = int(ot[0]), int(ot[1])
    pM, pN = int(pt[0]), int(pt[1])
    oracle_grid = math.ceil(M/oM) * math.ceil(N/oN)
    picked_grid = math.ceil(M/pM) * math.ceil(N/pN)
    oracle_waves = math.ceil(oracle_grid / MI300X_CUS)
    picked_waves = math.ceil(picked_grid / MI300X_CUS)
    s['oracle_eff'] = oracle_grid / (oracle_waves * MI300X_CUS)
    s['picked_eff'] = picked_grid / (picked_waves * MI300X_CUS)
    s['oracle_grid'] = oracle_grid
    s['picked_grid'] = picked_grid

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle(
    'K-016 Top-5 Regret Root-Cause — MI300X (gfx942) Tensile sweep\n'
    'branch k016/triton-specialization-in-origami-kernel-opt @ 45bfabb1e2',
    fontsize=11, fontweight='bold'
)

# Panel 1: Regret bar chart with oracle vs picked TFLOPS
ax1 = axes[0]
short_names = []
regrets = []
oracle_vals = []
picked_vals = []
for s in shapes:
    name = s['shape'].replace('_bf16','').replace('_f16','(f16)')
    short_names.append(name.replace('x','\n', 1))
    regrets.append(float(s['regret_pct']))
    oracle_vals.append(float(s['oracle_tflops']))
    picked_vals.append(float(s['picked_tflops']))

x = np.arange(len(shapes))
width = 0.35
bars1 = ax1.bar(x - width/2, oracle_vals, width, label='Oracle TFLOPS', color='#2ecc71', alpha=0.85)
bars2 = ax1.bar(x + width/2, picked_vals, width, label='Picked TFLOPS', color='#e74c3c', alpha=0.85)

# Add regret % labels
for i, r in enumerate(regrets):
    ax1.annotate(f'{r:.1f}%', (x[i], max(oracle_vals[i], picked_vals[i])),
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#c0392b')

ax1.set_ylabel('TFLOPS [VERIFIED]')
ax1.set_xticks(x)
ax1.set_xticklabels([s['shape'].replace('_bf16','').replace('_f16','(f16)') for s in shapes],
                    fontsize=6, rotation=30, ha='right')
ax1.legend(fontsize=8)
ax1.set_title('Oracle vs Picked TFLOPS', fontsize=10)

# Panel 2: Wave efficiency comparison
ax2 = axes[1]
oracle_effs = [s['oracle_eff'] for s in shapes]
picked_effs = [s['picked_eff'] for s in shapes]
bars3 = ax2.bar(x - width/2, oracle_effs, width, label='Oracle wave eff', color='#3498db', alpha=0.85)
bars4 = ax2.bar(x + width/2, picked_effs, width, label='Picked wave eff', color='#e67e22', alpha=0.85)
ax2.set_ylabel('Wave Efficiency (grid/CUs)')
ax2.set_xticks(x)
ax2.set_xticklabels([s['shape'].replace('_bf16','').replace('_f16','(f16)') for s in shapes],
                    fontsize=6, rotation=30, ha='right')
ax2.legend(fontsize=8)
ax2.set_title('Wave Quantization Efficiency', fontsize=10)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

# Panel 3: Root cause taxonomy
ax3 = axes[2]
causes = {}
for s in shapes:
    cause = s['root_cause'].replace('cost_model_', 'CM: ').replace('tile_handoff_bug_', 'Bug: ').replace('severe_', '')
    causes[cause] = causes.get(cause, 0) + float(s['tflops_gap'])

cause_names = list(causes.keys())
cause_gaps = list(causes.values())
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6']
bars5 = ax3.barh(cause_names, cause_gaps, color=colors[:len(cause_names)], alpha=0.85)
ax3.set_xlabel('Aggregate TFLOPS Gap [VERIFIED]')
ax3.set_title('Root Cause → TFLOPS Left on Table', fontsize=10)
for bar, val in zip(bars5, cause_gaps):
    ax3.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
fig.text(0.5, 0.01,
         'Data: regret_decomposition_summary.csv + top5_root_cause.csv | MI300X Tensile sweep, 3-run avg',
         ha='center', fontsize=7, style='italic')

outpath = os.path.join(REPORT_DIR, 'key_result_kernel-opt.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved: {outpath} ({os.path.getsize(outpath)} bytes)')
plt.close()
