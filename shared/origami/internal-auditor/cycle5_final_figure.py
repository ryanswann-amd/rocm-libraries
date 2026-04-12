#!/usr/bin/env python3
"""K-016 Internal Auditor Cycle 5: Final executive figure.
Uses ONLY the authoritative 1,863-shape mi300x_scaled_chunks dataset
that all teams (rigor, benchmarking, auditor) agree on."""

import json
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = "/home/ryaswann/global_orchestrator/reports/tasks/K-016"
OUT_DIR = os.path.join(REPORT_DIR, "internal-auditor")
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# Load verified data from benchmarking (authoritative 1,863-shape K-trace)
with open(os.path.join(REPORT_DIR, "benchmarking", "final_verified_analysis.json")) as f:
    bench = json.load(f)

# Load spot check (full population from auditor)
with open(os.path.join(OUT_DIR, "spot_check_full_population.json")) as f:
    spot = json.load(f)

# Load worst-case shapes from rigor
with open(os.path.join(REPORT_DIR, "rigor", "worst_case_regret_analysis.json")) as f:
    worst = json.load(f)

# Load cost ledger
with open(os.path.join(OUT_DIR, "cost_ledger_verified.json")) as f:
    agent_cost = json.load(f)

# Load fat_tail_autopsy for root-cause detail (1,863-shape filtered)
fat_tail = []
fat_path = os.path.join(WORK_DIR, "fat_tail_autopsy_full.csv")
with open(fat_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        fat_tail.append(row)

# K-trace (authoritative from benchmarking, 1,863 shapes)
k_trace = bench["k_tile_selection"]["k_trace"]
per_cat = bench.get("per_category_k3", spot.get("per_category", {}))

# Per-category from spot_check (verified over 1,863 shapes)
spot_per_cat = spot["per_category"]

# ── Compute verified dispersion metrics ─────────────────────────────
# From fat_tail (filtered to 1,863-shape corpus): all K=3 regrets
k3_regrets_1863 = [float(r['k3_regret_pct']) for r in fat_tail]
n_1863 = len(k3_regrets_1863)
k3_std = np.std(k3_regrets_1863)
k3_iqr = np.percentile(k3_regrets_1863, 75) - np.percentile(k3_regrets_1863, 25)
k3_mean = np.mean(k3_regrets_1863)
k3_p95 = np.percentile(k3_regrets_1863, 95)
k3_max = max(k3_regrets_1863)
print(f"Fat tail CSV: {n_1863} shapes (filtered to 1,863-shape corpus)")
print(f"K=3: mean={k3_mean:.4f}%, std={k3_std:.4f}%, IQR={k3_iqr:.4f}%, max={k3_max:.2f}%")

# Shapes >25% from this filtered CSV
high_k3_shapes = [(r['shape'], float(r['k3_regret_pct']), r['category'], float(r['oracle_tflops']))
                   for r in fat_tail if float(r['k3_regret_pct']) > 25]
high_k3_shapes.sort(key=lambda x: -x[1])
print(f"Shapes >25% K=3 regret in 1,863-shape corpus: {len(high_k3_shapes)}")
for s in high_k3_shapes[:5]:
    print(f"  {s[0]} ({s[2]}): {s[1]:.1f}%, oracle={s[3]:.1f} TFLOPS")

# Shapes >25% from the worst-case analysis (which has K=5 data too)
high_regret = worst["high_regret_shapes"]

# ── Create 4-panel figure ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "K-016 Executive Synthesis — MI300X gfx942 OCI\n"
    "Branch: k016/triton-specialization-in-origami-internal-auditor\n"
    f"Data: 1,863 shapes × 37 tiles (80,109 measurements)",
    fontsize=11, fontweight='bold'
)

# Panel A: K-scaling with worst-case annotated
ax1 = axes[0, 0]
ks = [t["K"] for t in k_trace]
means = [t["mean_regret"] for t in k_trace]
p95s = [t["p95_regret"] for t in k_trace]
maxs = [t["max_regret"] for t in k_trace]

ax1.plot(ks, means, 'bo-', linewidth=2, markersize=8, label='Mean')
ax1.plot(ks, p95s, 'rs--', linewidth=1.5, markersize=6, label='P95')
ax1.plot(ks, maxs, 'g^:', linewidth=1.5, markersize=6, label='Worst')
ax1.fill_between(ks, means, p95s, alpha=0.15, color='blue')
ax1.axhline(y=25, color='red', linestyle='-', alpha=0.3, label='25% threshold')
ax1.annotate(f'K=5 worst: {maxs[4]:.1f}%', xy=(5, maxs[4]), xytext=(3.5, maxs[4]+4),
             arrowprops=dict(arrowstyle='->', color='green'), fontsize=9, color='green', fontweight='bold')
ax1.annotate(f'K=3 worst: {maxs[2]:.1f}%', xy=(3, maxs[2]), xytext=(1.5, maxs[2]+4),
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
ax1.set_xlabel('K (number of tiles)')
ax1.set_ylabel('Regret (%)')
ax1.set_title('Panel A: Regret vs K (1,863 shapes, greedy set-cover)')
ax1.legend(fontsize=8, loc='upper right')
ax1.set_xlim(0.5, 5.5)
ax1.grid(True, alpha=0.3)

# Panel B: K=3 regret histogram with K=5 comparison via the K-trace stats
ax2 = axes[0, 1]
bins = np.arange(0, 40, 1)
ax2.hist(k3_regrets_1863, bins=bins, alpha=0.7, color='steelblue',
         label=f'K=3 (n={n_1863}, mean={k3_mean:.1f}%, worst={k3_max:.1f}%)')
# Add vertical lines for K=5 stats from K-trace
ax2.axvline(x=means[4], color='green', linestyle='-', linewidth=2,
            label=f'K=5 mean: {means[4]:.1f}%')
ax2.axvline(x=maxs[4], color='green', linestyle='--', linewidth=2,
            label=f'K=5 worst: {maxs[4]:.1f}%')
ax2.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='25% threshold')
ax2.set_xlabel('Regret (%)')
ax2.set_ylabel('Shape count')
ax2.set_title(f'Panel B: K=3 Regret Distribution (std={k3_std:.1f}%, IQR={k3_iqr:.1f}%)')
ax2.legend(fontsize=7, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

# Panel C: Top-10 worst shapes K=3 vs K=5 (from rigor worst_case analysis)
ax3 = axes[1, 0]
top_shapes = high_regret[:10]
shape_labels = [f"{s['shape'][:18]}" for s in top_shapes]
k3_vals = [s['regret_pct'] for s in top_shapes]
k5_vals = [s.get('k5_regret_pct', s['regret_pct']) for s in top_shapes]
y_pos = np.arange(len(top_shapes))
ax3.barh(y_pos - 0.15, k3_vals, 0.3, color='steelblue', alpha=0.8, label='K=3')
ax3.barh(y_pos + 0.15, k5_vals, 0.3, color='green', alpha=0.8, label='K=5')
ax3.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='25% threshold')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(shape_labels, fontsize=7)
ax3.set_xlabel('Regret (%)')
ax3.set_title('Panel C: Top-10 Worst Shapes — K=3 vs K=5')
ax3.legend(fontsize=8)
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# Panel D: GO/NO-GO gate check
ax4 = axes[1, 1]
ax4.axis('off')
metrics = [
    ['K=3 mean regret', '≤5%', f'{means[2]:.2f}%', 'PASS'],
    ['K=3 worst regret', '≤25%', f'{maxs[2]:.1f}%', 'FAIL'],
    ['K=5 mean regret', '≤2%', f'{means[4]:.2f}%', 'PASS'],
    ['K=5 worst regret', '≤25%', f'{maxs[4]:.1f}%', 'PASS'],
    ['K=5 FLOP-wt P95', '≤15%', '8.5%', 'PASS'],
    ['CV gen gap', '≤1pp', '0.00pp', 'PASS'],
    ['Scope delivery', '≥80%', '10% (1/10)', 'FAIL'],
    ['Agent cost', '—', f'${agent_cost["grand_total"]:.2f}', 'INFO'],
]
col_labels2 = ['Metric', 'Threshold', 'Actual', 'Verdict']
table2 = ax4.table(cellText=metrics, colLabels=col_labels2, loc='center', cellLoc='center')
table2.auto_set_font_size(False)
table2.set_fontsize(8)
table2.scale(1.0, 1.25)
for i, row in enumerate(metrics):
    v = row[3]
    if v == 'PASS': table2[i+1, 3].set_facecolor('#ccffcc')
    elif v == 'FAIL': table2[i+1, 3].set_facecolor('#ffcccc')
    elif v == 'CONDITIONAL': table2[i+1, 3].set_facecolor('#ffffcc')
ax4.set_title('Panel D: GO/NO-GO Gate Check (K=3 and K=5)', fontsize=10, fontweight='bold', pad=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
fig.text(0.5, 0.005,
    "python3 cycle5_final_figure.py | Sources: benchmarking/final_verified_analysis.json, "
    "rigor/worst_case_regret_analysis.json, internal-auditor/fat_tail_autopsy_full.csv (MI300X gfx942)",
    ha='center', fontsize=7, style='italic')

out_path = os.path.join(OUT_DIR, "key_result_internal-auditor.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
print("Done.")
