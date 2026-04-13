#!/usr/bin/env python3
"""
K-020 Rigor — Key result figure: bench-sanity cross-reference audit
Shows measurement discrepancies across data sources for each shape.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# === Data from verified sources ===
shapes = ['MLP\n4K×14K×4K', 'Sq8K\n8K³', 'Attn\n8K×4.6K×37K', 'Lg16K\n16K³']

# Speedup measurements from different sources, same shape/split
# Source 1: K-019 st-3 (rad-mi300x-1, ROCm 6.4, tritonBLAS WS + torch.copy_)
# Source 2: Jobs 18180/18254/18267 (OCI, ROCm 7.0, WS + Iris XGMI)
k019_256_48 = [1.084, 1.011, 0.710, 0.955]
cycle8_256_48 = [1.369, None, 0.786, 0.620]  # Sq8K not measured in cycle 8

# bench-sanity threshold
threshold = 0.20

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('K-020 Rigor Audit: Cross-Source Measurement Discrepancies\n'
             'MI300X — K-019 st-3 (ROCm 6.4) vs Jobs 18180/18254/18267 (ROCm 7.0)',
             fontsize=13, fontweight='bold')

colors_k019 = '#2196F3'
colors_c8 = '#FF5722'

# Panel 1: Speedup comparison bar chart
ax = axes[0, 0]
x = np.arange(len(shapes))
width = 0.35
bars1 = ax.bar(x - width/2, k019_256_48, width, label='K-019 st-3 (ROCm 6.4)', color=colors_k019, alpha=0.8)
c8_vals = [v if v is not None else 0 for v in cycle8_256_48]
c8_colors = [colors_c8 if v is not None else 'none' for v in cycle8_256_48]
bars2 = ax.bar(x + width/2, c8_vals, width, label='Cycle 8 (ROCm 7.0)', color=colors_c8, alpha=0.8)
# No data hatching for Sq8K cycle 8
if cycle8_256_48[1] is None:
    bars2[1].set_facecolor('lightgray')
    bars2[1].set_hatch('///')
    bars2[1].set_edgecolor('gray')

ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Breakeven')
ax.set_ylabel('Measured Speedup')
ax.set_title('A) Speedup at 256/48 CU Split')
ax.set_xticks(x)
ax.set_xticklabels(shapes, fontsize=9)
ax.legend(fontsize=8)
ax.set_ylim(0, 1.6)

# Panel 2: bench-sanity pass/fail
ax = axes[0, 1]
diffs = []
labels = []
colors = []
for i, shape in enumerate(['MLP', 'Sq8K', 'Attn', 'Lg16K']):
    if cycle8_256_48[i] is not None:
        diff = abs(k019_256_48[i] - cycle8_256_48[i]) / max(abs(k019_256_48[i]), abs(cycle8_256_48[i]))
        diffs.append(diff * 100)
        labels.append(shape)
        colors.append('#F44336' if diff > threshold else '#4CAF50')
    else:
        diffs.append(0)
        labels.append(shape + '\n(no data)')
        colors.append('lightgray')

bars = ax.bar(range(len(labels)), diffs, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=20.0, color='red', linestyle='--', linewidth=2, label='bench-sanity threshold (20%)')
ax.set_ylabel('Discrepancy (%)')
ax.set_title('B) bench-sanity.py Validation')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=10)
ax.legend(fontsize=8)

# Annotate PASS/FAIL
for i, (bar, d) in enumerate(zip(bars, diffs)):
    if d > 0:
        status = 'FAIL' if d > 20 else 'PASS'
        color = 'red' if d > 20 else 'green'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{status}\n{d:.1f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=color)

# Panel 3: CU scaling exponent spread across sources
ax = axes[1, 0]
shapes_alpha = ['MLP', 'Attn', 'Sq8K', 'Lg16K']
# Method A (benchmarking c4): Alola, ROCm 7.2
method_a = [0.400, 0.818, 0.598, 0.603]
# K-019 (research c8): rad-mi300x-1, ROCm 6.4
k019_alpha = [0.479, 0.790, 0.665, 0.629]
# Profiling: HSA corrected
profiling = [0.823, 0.851, 0.831, 0.774]
# Benchmarking: 8-pt Alola fit
benchmarking = [0.854, 0.823, 0.849, 0.847]

x = np.arange(len(shapes_alpha))
w = 0.2
ax.bar(x - 1.5*w, method_a, w, label='Method A (ROCm 7.2)', color='#FF9800', alpha=0.8)
ax.bar(x - 0.5*w, k019_alpha, w, label='K-019 (ROCm 6.4)', color='#2196F3', alpha=0.8)
ax.bar(x + 0.5*w, profiling, w, label='Profiling (HSA corr.)', color='#9C27B0', alpha=0.8)
ax.bar(x + 1.5*w, benchmarking, w, label='Benchmarking (8-pt Alola)', color='#4CAF50', alpha=0.8)

ax.set_ylabel('CU Scaling Exponent α')
ax.set_title('C) α Exponent Spread Across Sources')
ax.set_xticks(x)
ax.set_xticklabels(shapes_alpha, fontsize=10)
ax.legend(fontsize=7, loc='lower right')
ax.set_ylim(0, 1.1)

# Annotate MLP spread
mlp_spread = max(method_a[0], k019_alpha[0], profiling[0], benchmarking[0]) - min(method_a[0], k019_alpha[0], profiling[0], benchmarking[0])
ax.annotate(f'MLP spread: {mlp_spread:.3f}\n(106% relative)',
            xy=(0, 0.854), xytext=(0.8, 1.0),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=8, color='red', fontweight='bold')

# Panel 4: Model vs Measured — systematic overprediction
ax = axes[1, 1]
model_speedups = [1.62, 1.81, 1.25, 1.40, 0.96, 1.07, 1.06, 1.19]
measured_speedups = [1.084, 0.982, 1.011, 0.994, 0.710, 0.660, 0.955, 0.966]
config_labels = ['MLP\n256/48', 'MLP\n296/8', 'Sq8K\n256/48', 'Sq8K\n296/8',
                 'Attn\n256/48', 'Attn\n296/8', 'Lg16K\n256/48', 'Lg16K\n296/8']

# Plot perfect line
ax.plot([0.5, 2.0], [0.5, 2.0], 'k--', linewidth=1, label='Perfect prediction')

# Color by shape
shape_colors = {'MLP': '#2196F3', 'Sq8K': '#4CAF50', 'Attn': '#F44336', 'Lg16K': '#FF9800'}
for i, (m, meas, label) in enumerate(zip(model_speedups, measured_speedups, config_labels)):
    shape = label.split('\n')[0]
    ax.scatter(meas, m, color=shape_colors[shape], s=100, zorder=5, edgecolor='black')
    ax.annotate(label, (meas, m), fontsize=6, textcoords="offset points", xytext=(5, 5))

ax.set_xlabel('Measured Speedup')
ax.set_ylabel('Model-Predicted Speedup')
ax.set_title('D) Model vs Measured (27.5% mean overshoot)')
ax.set_xlim(0.5, 1.5)
ax.set_ylim(0.5, 2.0)

# Shade overprediction zone
ax.fill_between([0.5, 2.0], [0.5, 2.0], [2.0, 2.0], alpha=0.08, color='red', label='Overprediction zone')
ax.legend(fontsize=8)

plt.tight_layout()

fig.text(0.5, 0.005,
         'python rigor/generate_key_result_rigor.py | Data: K-019 st-3 (rad-mi300x-1, ROCm 6.4) + '
         'Jobs 18180/18254/18267 (OCI, ROCm 7.0) | bench-sanity.py threshold=20%',
         ha='center', fontsize=7, style='italic', color='gray')

plt.savefig('/home/ryaswann/global_orchestrator/reports/tasks/K-020/rigor/key_result_rigor.png',
            dpi=200, bbox_inches='tight')
print("Saved key_result_rigor.png")
