#!/usr/bin/env python3
"""Generate the key result figure for K-016 kernel-opt root-cause analysis.
All data sourced from [VERIFIED] hardware measurements in regret_decomposition_summary.csv
and k3_per_shape_regret.csv (MI300X, OCI/Banff sweeps).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

REPORT_DIR = "/home/ryaswann/global_orchestrator/reports/tasks/K-016/kernel-opt"
K3_CSV = "/home/ryaswann/global_orchestrator/reports/tasks/K-016/benchmarking/k3_per_shape_regret.csv"
DECOMP_CSV = "/home/ryaswann/global_orchestrator/reports/tasks/K-016/regret_decomposition_summary.csv"
BANFF_CSV = "/home/ryaswann/global_orchestrator/reports/tasks/K-016/regret_decomposition_summary.csv"

# --- Panel 1: K=3 regret distribution histogram ---
regrets = []
with open(K3_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        regrets.append(float(row['k3_regret_pct']))
regrets = np.array(regrets)

# --- Panel 2: Top-5 Banff shapes root-cause breakdown ---
# Data from regret_decomposition_summary.csv (16-shape Banff correlation, MI300X Tensile sweep)
banff_top5 = [
    {'shape': '2048×4096×5376\nbf16', 'regret': 35.7, 'oracle_tflops': 1014.45, 'pick_tflops': 652.4,
     'oracle_tile': '256×128×64', 'pick_tile': '256×224×64', 'proxy_rank': 14, 'category': 'BLOCK_N misrank'},
    {'shape': '1×16384×16384\nbf16', 'regret': 29.3, 'oracle_tflops': 6.576, 'pick_tflops': 4.648,
     'oracle_tile': '16×32×512', 'pick_tile': '16×256×128', 'proxy_rank': 4, 'category': 'BLOCK_K misrank'},
    {'shape': '1×13312×16384\nbf16', 'regret': 29.1, 'oracle_tflops': 6.342, 'pick_tflops': 4.497,
     'oracle_tile': '64×64×256', 'pick_tile': '16×256×128', 'proxy_rank': 73, 'category': 'Severe misrank'},
    {'shape': '3600×4096×4096\nbf16', 'regret': 28.7, 'oracle_tflops': 1092.68, 'pick_tflops': 778.876,
     'oracle_tile': '256×256×64', 'pick_tile': '192×256×64', 'proxy_rank': 1, 'category': 'Tile handoff bug'},
    {'shape': '1536×3584×3584\nf16', 'regret': 24.0, 'oracle_tflops': 743.126, 'pick_tflops': 564.925,
     'oracle_tile': '128×192×64', 'pick_tile': '192×96×128', 'proxy_rank': 35, 'category': 'BLOCK_N misrank'},
]

# --- Create figure ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 1.3]})

# Panel 1: Regret distribution
ax1 = axes[0]
bins = np.arange(0, 36, 1)
counts, _, patches = ax1.hist(regrets, bins=bins, color='#4DBEEE', edgecolor='white', linewidth=0.5, alpha=0.85)
# Color fat tail
for i, patch in enumerate(patches):
    if bins[i] >= 25:
        patch.set_facecolor('#D95319')
        patch.set_alpha(1.0)
ax1.axvline(np.mean(regrets), color='#EDB120', linewidth=2, linestyle='--', label=f'Mean={np.mean(regrets):.2f}%')
ax1.axvline(np.percentile(regrets, 95), color='#A2142F', linewidth=2, linestyle=':', label=f'P95={np.percentile(regrets, 95):.1f}%')
fat_tail_n = np.sum(regrets >= 25)
fat_tail_pct = fat_tail_n / len(regrets) * 100
ax1.text(27, max(counts)*0.8, f'Fat tail\n{fat_tail_n} shapes\n({fat_tail_pct:.1f}%)',
         fontsize=9, color='#D95319', fontweight='bold', ha='center')
ax1.set_xlabel('Regret (%)', fontsize=11)
ax1.set_ylabel('Number of shapes', fontsize=11)
ax1.set_title('K=3 Regret Distribution (1,863 shapes)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.set_xlim(-0.5, 35)

# Panel 2: Top-5 shapes waterfall with root cause
ax2 = axes[1]
y_pos = np.arange(len(banff_top5))
colors_root = {'BLOCK_N misrank': '#0072BD', 'BLOCK_K misrank': '#77AC30',
               'Severe misrank': '#A2142F', 'Tile handoff bug': '#D95319'}
bar_colors = [colors_root.get(s['category'], '#999') for s in banff_top5]

bars = ax2.barh(y_pos, [s['regret'] for s in banff_top5], color=bar_colors,
                edgecolor='white', linewidth=0.5, height=0.6, alpha=0.9)

for i, s in enumerate(banff_top5):
    # Label with root cause
    label = f"Oracle rank #{s['proxy_rank']} → {s['category']}"
    ax2.text(s['regret'] + 0.5, i, label, va='center', fontsize=8, color='#333')

ax2.set_yticks(y_pos)
ax2.set_yticklabels([s['shape'] for s in banff_top5], fontsize=9)
ax2.set_xlabel('Regret (%)', fontsize=11)
ax2.set_title('Top-5 Banff Shapes: Root Cause', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 55)
ax2.invert_yaxis()

# Add legend for root causes
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l) for l, c in colors_root.items()]
ax2.legend(handles=legend_elements, fontsize=8, loc='lower right')

fig.suptitle("K-016 Root-Cause Analysis — MI300X (gfx942) — branch k016/triton-specialization-in-origami-kernel-opt",
             fontsize=10, fontweight='bold', y=0.98)
fig.text(0.5, 0.01,
         "Data: regret_decomposition_summary.csv (16 Banff shapes, Tensile sweep) + k3_per_shape_regret.csv (1863 shapes, MI300X GPU-measured)",
         ha='center', fontsize=7, color='#666')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
outpath = os.path.join(REPORT_DIR, 'key_result_kernel-opt.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")

# --- Also generate regret distribution K3 standalone ---
fig2, ax = plt.subplots(figsize=(8, 4.5))
counts, _, patches = ax.hist(regrets, bins=np.arange(0, 36, 1), color='#4DBEEE', edgecolor='white', linewidth=0.5, alpha=0.85)
for i, patch in enumerate(patches):
    if np.arange(0, 36, 1)[i] >= 25:
        patch.set_facecolor('#D95319')
        patch.set_alpha(1.0)
ax.axvline(np.mean(regrets), color='#EDB120', linewidth=2, linestyle='--', label=f'Mean={np.mean(regrets):.2f}%')
ax.axvline(np.percentile(regrets, 95), color='#A2142F', linewidth=2, linestyle=':', label=f'P95={np.percentile(regrets, 95):.1f}%')
ax.axvline(np.percentile(regrets, 99), color='#7E2F8E', linewidth=2, linestyle='-.', label=f'P99={np.percentile(regrets, 99):.1f}%')
ax.text(27, max(counts)*0.8, f'{fat_tail_n} shapes\n≥25% regret', fontsize=9, color='#D95319', fontweight='bold', ha='center')
ax.set_xlabel('Regret (%)', fontsize=11)
ax.set_ylabel('Number of shapes', fontsize=11)
ax.set_title('K=3 Regret Distribution — MI300X (1,863 shapes)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
fig2.text(0.5, 0.01, "Source: k3_per_shape_regret.csv (MI300X GPU-measured, commit f9bae275)", ha='center', fontsize=7, color='#666')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
fig2.savefig(os.path.join(REPORT_DIR, 'regret_distribution_k3.png'), dpi=150, bbox_inches='tight')
print(f"Saved: regret_distribution_k3.png")

# Print verified stats
print(f"\n=== VERIFIED K=3 Corpus Statistics (1,863 shapes, MI300X) ===")
print(f"Mean regret: {np.mean(regrets):.4f}%")
print(f"Std regret: {np.std(regrets):.4f}%")
print(f"P50 (median): {np.median(regrets):.4f}%")
print(f"P95: {np.percentile(regrets, 95):.4f}%")
print(f"P99: {np.percentile(regrets, 99):.4f}%")
print(f"Max: {np.max(regrets):.4f}%")
print(f"Zero regret: {np.sum(regrets == 0)}/{len(regrets)} ({np.sum(regrets==0)/len(regrets)*100:.1f}%)")
print(f"Fat tail (≥25%): {fat_tail_n}/{len(regrets)} ({fat_tail_pct:.1f}%)")
