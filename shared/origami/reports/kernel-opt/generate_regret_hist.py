#!/usr/bin/env python3
"""Generate K=3 regret distribution histogram for K-016.
Data source: benchmarking/k3_per_shape_regret.csv (MI300X corpus, 1,863 shapes)
Branch: k016/triton-specialization-in-origami-kernel-opt @ 45bfabb1e2
"""
import csv, os, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
K016_DIR = os.path.dirname(REPORT_DIR)

regrets = []
with open(os.path.join(K016_DIR, 'benchmarking', 'k3_per_shape_regret.csv')) as f:
    reader = csv.DictReader(f)
    for row in reader:
        regrets.append(float(row['k3_regret_pct']))

mean_r = statistics.mean(regrets)
p95 = sorted(regrets)[int(len(regrets)*0.95)]
fat_tail = sum(1 for r in regrets if r >= 25)

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle(
    'K=3 Regret Distribution — 1,863-shape corpus on MI300X\n'
    'branch k016/triton-specialization-in-origami-kernel-opt @ 45bfabb1e2',
    fontsize=11, fontweight='bold'
)

# Plot histogram
bins = np.arange(0, 36, 1)
n, bins_out, patches = ax.hist(regrets, bins=bins, color='#3498db', alpha=0.8, edgecolor='white')

# Color fat tail red
for i, patch in enumerate(patches):
    if bins_out[i] >= 25:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.9)

ax.axvline(mean_r, color='#2c3e50', linestyle='--', linewidth=2, label=f'Mean: {mean_r:.2f}%')
ax.axvline(p95, color='#e67e22', linestyle='--', linewidth=2, label=f'P95: {p95:.2f}%')

ax.set_xlabel('Regret (%)')
ax.set_ylabel('Number of Shapes')
ax.legend(fontsize=10)

# Annotate fat tail
ax.annotate(f'{fat_tail} shapes ≥25%\n(fat tail)',
           xy=(27, max(n[25:])*1.5 if any(n[25:]) else 5), fontsize=9,
           color='#e74c3c', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8'))

stats_text = (f'N = {len(regrets)} shapes\n'
              f'Mean = {mean_r:.2f}%\n'
              f'Median = {statistics.median(regrets):.2f}%\n'
              f'P95 = {p95:.2f}%\n'
              f'Max = {max(regrets):.2f}%\n'
              f'Zero-regret = {sum(1 for r in regrets if r == 0)}\n'
              f'Fat tail (≥25%) = {fat_tail}')
ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.text(0.5, 0.01,
         'Data: benchmarking/k3_per_shape_regret.csv | MI300X Tensile sweep, 3-run avg per tile',
         ha='center', fontsize=7, style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
outpath = os.path.join(REPORT_DIR, 'regret_distribution_k3.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved: {outpath} ({os.path.getsize(outpath)} bytes)')
plt.close()
