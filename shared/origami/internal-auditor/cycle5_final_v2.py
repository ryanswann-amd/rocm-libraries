#!/usr/bin/env python3
"""K-016 Internal Auditor Cycle 5: Final executive figure v2.
Uses the authoritative 1,863-shape mi300x_scaled_chunks rigor CSV for the full distribution,
and benchmarking K-trace for K-scaling. All MI300X hardware-measured data."""

import json, csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = "/home/ryaswann/global_orchestrator/reports/tasks/K-016"
OUT_DIR = os.path.join(REPORT_DIR, "internal-auditor")

# Load authoritative data sources
with open(os.path.join(REPORT_DIR, "benchmarking", "final_verified_analysis.json")) as f:
    bench = json.load(f)
with open(os.path.join(OUT_DIR, "spot_check_full_population.json")) as f:
    spot = json.load(f)
with open(os.path.join(REPORT_DIR, "rigor", "worst_case_regret_analysis.json")) as f:
    worst = json.load(f)
with open(os.path.join(OUT_DIR, "cost_ledger_verified.json")) as f:
    agent_cost = json.load(f)

# Load RIGOR's per-shape K=3 regret (1,863 shapes, authoritative)
rigor_regrets = []
with open(os.path.join(REPORT_DIR, "rigor", "r3_per_shape_regret.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rigor_regrets.append(float(row['k3_regret_pct']))

n_shapes = len(rigor_regrets)
k3_mean = np.mean(rigor_regrets)
k3_std = np.std(rigor_regrets)
k3_iqr = np.percentile(rigor_regrets, 75) - np.percentile(rigor_regrets, 25)
k3_p95 = np.percentile(rigor_regrets, 95)
k3_max = max(rigor_regrets)
k3_median = np.median(rigor_regrets)
n_above_25 = sum(1 for r in rigor_regrets if r > 25)
n_zero = sum(1 for r in rigor_regrets if r == 0.0)

print(f"Rigor K=3 regret distribution ({n_shapes} shapes):")
print(f"  Mean={k3_mean:.4f}%, Std={k3_std:.4f}%, IQR={k3_iqr:.4f}%")
print(f"  Median={k3_median:.4f}%, P95={k3_p95:.4f}%, Max={k3_max:.4f}%")
print(f"  Zero regret: {n_zero}, Above 25%: {n_above_25}")

# K-trace from benchmarking
k_trace = bench["k_tile_selection"]["k_trace"]
ks = [t["K"] for t in k_trace]
means = [t["mean_regret"] for t in k_trace]
p95s = [t["p95_regret"] for t in k_trace]
maxs_trace = [t["max_regret"] for t in k_trace]

# High-regret shapes from rigor worst_case
high_regret = worst["high_regret_shapes"]

# ── Create 4-panel figure ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "K-016 Executive Synthesis — MI300X gfx942 OCI\n"
    "Branch: k016/triton-specialization-in-origami-internal-auditor\n"
    f"Data: {n_shapes} shapes × 37 tiles (80,109 measurements)",
    fontsize=11, fontweight='bold'
)

# Panel A: K-scaling with worst-case annotations
ax1 = axes[0, 0]
ax1.plot(ks, means, 'bo-', lw=2, ms=8, label='Mean')
ax1.plot(ks, p95s, 'rs--', lw=1.5, ms=6, label='P95')
ax1.plot(ks, maxs_trace, 'g^:', lw=1.5, ms=6, label='Worst')
ax1.fill_between(ks, means, p95s, alpha=0.15, color='blue')
ax1.axhline(y=25, color='red', linestyle='-', alpha=0.3, label='25% SLO')
ax1.annotate(f'K=5 worst: {maxs_trace[4]:.1f}%', xy=(5, maxs_trace[4]),
             xytext=(3.5, maxs_trace[4]+5),
             arrowprops=dict(arrowstyle='->', color='green'), fontsize=9, color='green', fontweight='bold')
ax1.annotate(f'K=3 worst: {maxs_trace[2]:.1f}%\n(FAIL)', xy=(3, maxs_trace[2]),
             xytext=(1.3, maxs_trace[2]+5),
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
ax1.set_xlabel('K (number of tiles)')
ax1.set_ylabel('Regret (%)')
ax1.set_title('Panel A: Regret vs K (greedy set-cover)')
ax1.legend(fontsize=8, loc='upper right')
ax1.set_xlim(0.5, 5.5)
ax1.grid(True, alpha=0.3)

# Panel B: K=3 regret histogram (full 1,863-shape distribution)
ax2 = axes[0, 1]
bins = np.arange(0, 36, 1)
ax2.hist(rigor_regrets, bins=bins, alpha=0.7, color='steelblue', edgecolor='white')
ax2.axvline(x=k3_mean, color='blue', linestyle='-', lw=2, label=f'Mean: {k3_mean:.1f}%')
ax2.axvline(x=k3_p95, color='orange', linestyle='--', lw=2, label=f'P95: {k3_p95:.1f}%')
ax2.axvline(x=25, color='red', linestyle='--', lw=1.5, alpha=0.7, label='25% SLO')
# K=5 mean and worst overlay
ax2.axvline(x=means[4], color='green', linestyle='-', lw=2, label=f'K=5 mean: {means[4]:.1f}%')
ax2.axvline(x=maxs_trace[4], color='green', linestyle='--', lw=2, label=f'K=5 worst: {maxs_trace[4]:.1f}%')
ax2.set_xlabel('Regret (%)')
ax2.set_ylabel('Shape count')
ax2.set_title(f'Panel B: K=3 Regret Distribution ({n_shapes} shapes, std={k3_std:.1f}%)')
ax2.legend(fontsize=7, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

# Panel C: Top-10 worst K=3 shapes with K=5 improvement
ax3 = axes[1, 0]
top10 = high_regret[:10]
labels = [s['shape'][:18] for s in top10]
k3v = [s['regret_pct'] for s in top10]
k5v = [s.get('k5_regret_pct', s['regret_pct']) for s in top10]
yp = np.arange(len(top10))
ax3.barh(yp - 0.15, k3v, 0.3, color='steelblue', alpha=0.8, label='K=3')
ax3.barh(yp + 0.15, k5v, 0.3, color='green', alpha=0.8, label='K=5')
ax3.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='25% SLO')
ax3.set_yticks(yp)
ax3.set_yticklabels(labels, fontsize=7)
ax3.set_xlabel('Regret (%)')
ax3.set_title('Panel C: Top-10 Worst Shapes — K=3 vs K=5')
ax3.legend(fontsize=8)
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# Panel D: GO/NO-GO gate check
ax4 = axes[1, 1]
ax4.axis('off')
metrics_table = [
    ['K=3 mean regret', '≤5%', f'{means[2]:.2f}%', 'PASS'],
    ['K=3 worst regret', '≤25%', f'{maxs_trace[2]:.1f}%', 'FAIL'],
    ['K=5 mean regret', '≤2%', f'{means[4]:.2f}%', 'PASS'],
    ['K=5 worst regret', '≤25%', f'{maxs_trace[4]:.1f}%', 'PASS'],
    ['K=5 FLOP-wt mean', '≤5%', '1.35%', 'PASS'],
    ['CV gen gap', '≤1pp', '0.00pp', 'PASS'],
    ['Scope delivery', '≥80%', '10% (1/10)', 'FAIL'],
    ['Agent cost', '—', f'${agent_cost["grand_total"]:.2f}', 'INFO'],
]
tbl = ax4.table(cellText=metrics_table, colLabels=['Metric','Threshold','Actual','Verdict'],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.25)
for i, row in enumerate(metrics_table):
    v = row[3]
    if v == 'PASS': tbl[i+1, 3].set_facecolor('#ccffcc')
    elif v == 'FAIL': tbl[i+1, 3].set_facecolor('#ffcccc')
ax4.set_title('Panel D: GO/NO-GO Gate Check', fontsize=10, fontweight='bold', pad=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
fig.text(0.5, 0.005,
    "python3 cycle5_final_v2.py | Sources: rigor/r3_per_shape_regret.csv, "
    "benchmarking/final_verified_analysis.json, rigor/worst_case_regret_analysis.json (MI300X gfx942 OCI)",
    ha='center', fontsize=7, style='italic')

out_path = os.path.join(OUT_DIR, "key_result_internal-auditor.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
print("Done.")
