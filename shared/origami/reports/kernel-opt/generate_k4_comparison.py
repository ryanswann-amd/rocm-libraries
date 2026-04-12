#!/usr/bin/env python3
"""Generate K=3 vs K=4 regret comparison figure.
All data from merged_full_corpus.csv (MI300X GPU-measured, commit f9bae275).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict

corpus = '/home/ryaswann/global_orchestrator/reports/tasks/K-016/slurm/merged_full_corpus.csv'
k3_tiles = {'128x64x128', '128x128x128', '16x64x128'}
k4_tiles = k3_tiles | {'128x256x64'}

shape_data = defaultdict(dict)
with open(corpus) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['source'] == 'mi300x_scaled_chunks':
            shape = row['shape']
            tile = row['tile_config']
            tflops = float(row['measured_tflops'])
            shape_data[shape][tile] = tflops

k3_regrets = []
k4_regrets = []
for shape, tiles in sorted(shape_data.items()):
    oracle = max(tiles.values())
    if oracle <= 0:
        continue
    k3_best = max((tiles.get(t, 0) for t in k3_tiles), default=0)
    k4_best = max((tiles.get(t, 0) for t in k4_tiles), default=0)
    if k3_best == 0:
        continue
    k3_regrets.append((oracle - k3_best) / oracle * 100)
    k4_regrets.append((oracle - k4_best) / oracle * 100)

k3_r = np.array(k3_regrets)
k4_r = np.array(k4_regrets)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Overlaid histograms
ax1 = axes[0]
bins = np.arange(0, 36, 1)
ax1.hist(k3_r, bins=bins, alpha=0.6, color='#D95319', label=f'K=3 (mean={np.mean(k3_r):.2f}%)', edgecolor='white')
ax1.hist(k4_r, bins=bins, alpha=0.6, color='#0072BD', label=f'K=4 (mean={np.mean(k4_r):.2f}%)', edgecolor='white')
ax1.axvline(25, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax1.text(27, ax1.get_ylim()[1]*0.5 if ax1.get_ylim()[1] > 0 else 50,
         f'K=3: {np.sum(k3_r>=25)} shapes\nK=4: {np.sum(k4_r>=25)} shapes',
         fontsize=9, color='red', fontweight='bold')
ax1.set_xlabel('Regret (%)')
ax1.set_ylabel('Number of shapes')
ax1.set_title('Regret Distribution: K=3 vs K=4', fontweight='bold')
ax1.legend(fontsize=9)

# Panel 2: Scatter — K=3 vs K=4 per-shape
ax2 = axes[1]
ax2.scatter(k3_r, k4_r, alpha=0.3, s=10, c='#0072BD')
ax2.plot([0, 35], [0, 35], 'k--', alpha=0.4, label='No change')
ax2.set_xlabel('K=3 Regret (%)')
ax2.set_ylabel('K=4 Regret (%)')
ax2.set_title('Per-Shape Regret: K=3 vs K=4', fontweight='bold')
ax2.set_xlim(-1, 35)
ax2.set_ylim(-1, 35)

# Highlight improved shapes
improved = (k3_r - k4_r) > 0.1
ax2.scatter(k3_r[improved], k4_r[improved], alpha=0.5, s=15, c='#D95319',
            label=f'{np.sum(improved)} improved', zorder=5)
ax2.legend(fontsize=9)

fig.suptitle("K=4 Tile Set Impact — Adding 128x256x64 — MI300X (1,863 shapes, commit f9bae275)",
             fontsize=10, fontweight='bold')
fig.text(0.5, 0.01,
         "K=3: {128x64x128, 128x128x128, 16x64x128} | K=4: +128x256x64 | Source: merged_full_corpus.csv",
         ha='center', fontsize=7, color='#666')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

outpath = '/home/ryaswann/global_orchestrator/reports/tasks/K-016/kernel-opt/k4_tile_impact.png'
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")

# Summary stats
print(f"\n=== K=3 vs K=4 Summary ===")
print(f"K=3: mean={np.mean(k3_r):.4f}%, P95={np.percentile(k3_r,95):.2f}%, worst={np.max(k3_r):.2f}%, fat_tail={np.sum(k3_r>=25)}")
print(f"K=4: mean={np.mean(k4_r):.4f}%, P95={np.percentile(k4_r,95):.2f}%, worst={np.max(k4_r):.2f}%, fat_tail={np.sum(k4_r>=25)}")
