#!/usr/bin/env python3
"""Generate cycle 4 key_result_internal-auditor.png — 3-panel audit dashboard."""
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load verified results
with open(os.path.join(OUT_DIR, 'k5_regret_verified_1863shapes.json')) as f:
    stats = json.load(f)

# Load per-shape K=3 data and compute K=5 from corpus
bench = {}
with open(os.path.join(BASE, 'benchmarking', 'k3_per_shape_regret.csv')) as f:
    for row in csv.DictReader(f):
        bench[row['shape']] = {
            'k3_regret': float(row['k3_regret_pct']),
            'oracle': float(row['oracle_tflops']),
            'k3_tflops': float(row['k3_tflops']),
            'category': row['category']
        }

K5_TILES = {'128x64x128', '16x64x128', '128x128x128', '16x16x256', '128x256x64'}
shape_k5_best = {}
with open(os.path.join(BASE, 'slurm', 'merged_full_corpus.csv')) as f:
    for row in csv.DictReader(f):
        shape = row['shape']
        if shape not in bench:
            continue
        tile = row['tile_config']
        if tile not in K5_TILES:
            continue
        try:
            tflops = float(row['measured_tflops'])
        except:
            continue
        shape_k5_best[shape] = max(shape_k5_best.get(shape, 0), tflops)

# Build arrays
shapes = sorted(bench.keys())
k3r, k5r, flops, cats = [], [], [], []
for s in shapes:
    k3r.append(bench[s]['k3_regret'])
    oracle = bench[s]['oracle']
    k5_best = min(shape_k5_best.get(s, bench[s]['k3_tflops']), oracle)
    k5_regret = (oracle - k5_best) / oracle * 100 if k5_best > 0 else 100.0
    k5r.append(k5_regret)
    parts = s.split('_')[0].split('x')
    flops.append(2.0 * int(parts[0]) * int(parts[1]) * int(parts[2]))
    cats.append(bench[s]['category'])

k3r = np.array(k3r)
k5r = np.array(k5r)
flops = np.array(flops)
weights = flops / flops.sum()
cats = np.array(cats)

# K-scaling data from k5_greedy_result.json
k_scaling = [
    (1, 17.8267), (2, 5.1945), (3, 3.0615), (4, 1.8075), (5, 1.0575)
]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle('K-016 Internal Audit — K=3 vs K=5 Regret on MI300X (gfx942, OCI cluster)\n'
             'Branch: k016/triton-specialization-in-origami-internal-auditor | Data: 1,863 shapes, 80K+ MI300X measurements',
             fontsize=11, fontweight='bold')

# Panel 1: K-scaling curve
ax = axes[0]
ks = [d[0] for d in k_scaling]
means = [d[1] for d in k_scaling]
ax.plot(ks, means, 'bo-', linewidth=2, markersize=8, label='Mean regret')
ax.axhline(y=3.0615, color='red', linestyle='--', alpha=0.6, label=f'K=3 = {3.0615:.2f}%')
ax.axhline(y=1.0575, color='green', linestyle='--', alpha=0.6, label=f'K=5 = {1.0575:.2f}%')
ax.fill_between([3, 5], [0, 0], [3.0615, 3.0615], alpha=0.1, color='green')
ax.set_xlabel('K (tile count)', fontsize=11)
ax.set_ylabel('Mean regret (%)', fontsize=11)
ax.set_title('K-Scaling Curve [VERIFIED]', fontsize=11)
ax.set_xticks(ks)
ax.set_ylim(0, max(means) * 1.1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
# Annotate K=3→K=5 delta
ax.annotate(f'Δ = 2.00pp\n(65.5% reduction)',
           xy=(4, 2.0), fontsize=9, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='orange', alpha=0.8))

# Panel 2: Regret distribution histogram (K=3 vs K=5)
ax = axes[1]
bins = np.arange(0, 36, 1)
ax.hist(k3r, bins=bins, alpha=0.5, color='red', label=f'K=3 (mean={k3r.mean():.2f}%)', density=False)
ax.hist(k5r, bins=bins, alpha=0.5, color='green', label=f'K=5 (mean={k5r.mean():.2f}%)', density=False)
# Add P95 lines
k3_p95 = np.percentile(k3r, 95)
k5_p95 = np.percentile(k5r, 95)
ax.axvline(k3_p95, color='red', linestyle=':', alpha=0.8, label=f'K=3 P95={k3_p95:.1f}%')
ax.axvline(k5_p95, color='green', linestyle=':', alpha=0.8, label=f'K=5 P95={k5_p95:.1f}%')
ax.set_xlabel('Regret (%)', fontsize=11)
ax.set_ylabel('Shape count', fontsize=11)
ax.set_title('Regret Distribution [VERIFIED]', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Panel 3: Per-category FLOP-weighted regret
ax = axes[2]
unique_cats = ['decode', 'small_batch', 'medium_batch', 'compute_bound']
cat_labels = ['Decode\n(M=1)', 'Small Batch\n(M=2-32)', 'Medium Batch\n(M=33-256)', 'Compute Bound\n(M>256)']
k3_cat_means = []
k5_cat_means = []
cat_flop_pcts = []
for cat in unique_cats:
    mask = cats == cat
    if mask.sum() > 0:
        cat_weights = weights[mask] / weights[mask].sum() if weights[mask].sum() > 0 else np.ones(mask.sum()) / mask.sum()
        k3_cat_means.append(np.average(k3r[mask], weights=cat_weights))
        k5_cat_means.append(np.average(k5r[mask], weights=cat_weights))
        cat_flop_pcts.append(weights[mask].sum() * 100)
    else:
        k3_cat_means.append(0)
        k5_cat_means.append(0)
        cat_flop_pcts.append(0)

x = np.arange(len(unique_cats))
width = 0.35
bars1 = ax.bar(x - width/2, k3_cat_means, width, label='K=3', color='salmon', edgecolor='darkred', alpha=0.8)
bars2 = ax.bar(x + width/2, k5_cat_means, width, label='K=5', color='lightgreen', edgecolor='darkgreen', alpha=0.8)

# Add FLOP% labels on top
for i, (b1, b2, fp) in enumerate(zip(bars1, bars2, cat_flop_pcts)):
    ax.text(i, max(b1.get_height(), b2.get_height()) + 0.3,
           f'{fp:.1f}% FLOPs', ha='center', fontsize=8, color='gray')

ax.set_ylabel('FLOP-weighted mean regret (%)', fontsize=10)
ax.set_title('Per-Category Regret [VERIFIED]', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0.04, 1, 0.93])
fig.text(0.5, 0.01,
         'Source: slurm/merged_full_corpus.csv + benchmarking/k3_per_shape_regret.csv | '
         'python3 internal-auditor/cycle4_figure.py',
         ha='center', fontsize=8, color='gray')

out_path = os.path.join(OUT_DIR, 'key_result_internal-auditor.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close()
