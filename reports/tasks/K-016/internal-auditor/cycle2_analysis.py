#!/usr/bin/env python3
"""K-016 Cycle 2: Deep fat-tail analysis and root-cause decomposition.

Data source: banff_data/correlation_results.json (MI300X gfx942, Banff cluster)
Code source: origami.cpp:530-676, gemm.cpp:570-896 (cost model + tie-breaking)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load data
with open('/home/ryaswann/global_orchestrator/reports/tasks/K-016/banff_data/correlation_results.json') as f:
    data = json.load(f)

shapes = data['metrics']
shapes_sorted = sorted(shapes, key=lambda x: x['regret_pct'], reverse=True)

# === ANALYSIS 1: Full 16-shape TFLOPS gap table ===
print("=" * 80)
print("TABLE 1: All 16 shapes — TFLOPS gap analysis")
print("=" * 80)
print(f"{'Shape':<35} {'Pick TFLOPS':>12} {'Oracle TFLOPS':>14} {'Gap TFLOPS':>11} {'Regret%':>8}")
print("-" * 80)

total_gap = 0
total_picked = 0
total_oracle = 0
for s in shapes_sorted:
    pick_t = s['picked_gflops'] / 1000
    oracle_t = s['oracle_gflops'] / 1000
    gap_t = oracle_t - pick_t
    total_gap += gap_t
    total_picked += pick_t
    total_oracle += oracle_t
    print(f"{s['shape']:<35} {pick_t:>12.1f} {oracle_t:>14.1f} {gap_t:>11.1f} {s['regret_pct']:>7.1f}%")

print("-" * 80)
print(f"{'TOTAL':<35} {total_picked:>12.1f} {total_oracle:>14.1f} {total_gap:>11.1f} {data['aggregate']['avg_regret']:>7.1f}%")

# === ANALYSIS 2: Root cause classification ===
print("\n" + "=" * 80)
print("TABLE 2: Root cause classification")
print("=" * 80)

root_causes = []
for s in shapes_sorted:
    if s['regret_pct'] == 0:
        continue
    pt = s['picked_tile'].split('x')
    ot = s['oracle_tile'].split('x')
    pm, pn, pk = int(pt[0]), int(pt[1]), int(pt[2])
    om, on, ok = int(ot[0]), int(ot[1]), int(ot[2])

    # Primary root cause
    if pk < ok * 0.5 and pn > on * 1.5:
        cause = "BLOCK_N_over + BLOCK_K_under"
    elif pn > on * 1.5:
        cause = "BLOCK_N_overvaluation"
    elif pm * pn > om * on * 1.5:
        cause = "tile_area_overvaluation"
    elif pm != om and pn == on and pk == ok:
        cause = "BLOCK_M_misrank"
    else:
        cause = "general_misrank"

    root_causes.append({
        'shape': s['shape'],
        'regret': s['regret_pct'],
        'gap_tflops': (s['oracle_gflops'] - s['picked_gflops']) / 1000,
        'cause': cause,
        'picked': s['picked_tile'],
        'oracle': s['oracle_tile']
    })
    print(f"{s['shape']:<35} regret={s['regret_pct']:>5.1f}% cause={cause}")

# === ANALYSIS 3: N/K ratio bias ===
print("\n" + "=" * 80)
print("TABLE 3: N/K ratio bias in tile selection")
print("=" * 80)

nk_deltas = []
regrets = []
for s in shapes_sorted:
    pt = s['picked_tile'].split('x')
    ot = s['oracle_tile'].split('x')
    pick_nk = int(pt[1]) / int(pt[2])
    oracle_nk = int(ot[1]) / int(ot[2])
    delta = pick_nk - oracle_nk
    nk_deltas.append(delta)
    regrets.append(s['regret_pct'])

nk_deltas = np.array(nk_deltas)
regrets_arr = np.array(regrets)
corr = np.corrcoef(nk_deltas, regrets_arr)[0, 1]
n_higher = np.sum(nk_deltas > 0.1)
n_same = np.sum(np.abs(nk_deltas) <= 0.1)
n_lower = np.sum(nk_deltas < -0.1)

print(f"Pearson correlation (N/K delta vs regret): {corr:.4f}")
print(f"Model picks higher N/K tile: {n_higher}/16 shapes")
print(f"Model picks same N/K tile: {n_same}/16 shapes")
print(f"Model picks lower N/K tile: {n_lower}/16 shapes")

# === ANALYSIS 4: Arithmetic intensity tie-breaker impact ===
print("\n" + "=" * 80)
print("TABLE 4: Arithmetic intensity tie-breaker analysis")
print("=" * 80)
print("(origami.cpp:570-579: AI = 2*M*N*K / (M*K + N*K + M*N))")
print()
for s in shapes_sorted[:7]:
    pt = s['picked_tile'].split('x')
    ot = s['oracle_tile'].split('x')
    pm, pn, pk = int(pt[0]), int(pt[1]), int(pt[2])
    om, on, ok = int(ot[0]), int(ot[1]), int(ot[2])

    pick_ai = (2*pm*pn*pk) / (pm*pk + pn*pk + pm*pn)
    oracle_ai = (2*om*on*ok) / (om*ok + on*ok + om*on)

    print(f"{s['shape']:<35} pick_AI={pick_ai:>6.2f} oracle_AI={oracle_ai:>6.2f} delta={pick_ai-oracle_ai:>+6.2f}")

# === FIGURE: Key result dashboard ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("K-016 Cycle 2: Fat-Tail Regret Decomposition — MI300X (Banff gfx942)\n"
             "Branch: k016/triton-specialization-in-origami-internal-auditor\n"
             "Data: correlation_results.json, 16 shapes × 259 tile configs",
             fontsize=11, fontweight='bold')

# Panel 1: Regret waterfall (all shapes)
ax1 = axes[0, 0]
names = [s['shape'].replace('_bf16_r','').replace('_f16_r','(f16)') for s in shapes_sorted]
regret_vals = [s['regret_pct'] for s in shapes_sorted]
colors = ['#d32f2f' if r > 25 else '#f57c00' if r > 15 else '#1976d2' for r in regret_vals]
bars = ax1.barh(range(len(names)), regret_vals, color=colors)
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names, fontsize=7)
ax1.set_xlabel('Regret (%)')
ax1.set_title('Per-Shape Regret (>25% = red)', fontsize=10)
ax1.axvline(x=25, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.invert_yaxis()

# Panel 2: TFLOPS gap bar chart (top 7)
ax2 = axes[0, 1]
top7 = shapes_sorted[:7]
gaps = [(s['oracle_gflops'] - s['picked_gflops'])/1000 for s in top7]
names7 = [s['shape'].replace('_bf16_r','').replace('_f16_r','(f16)') for s in top7]
bars2 = ax2.barh(range(len(names7)), gaps, color=['#d32f2f' if g > 200 else '#f57c00' for g in gaps])
ax2.set_yticks(range(len(names7)))
ax2.set_yticklabels(names7, fontsize=8)
ax2.set_xlabel('TFLOPS Gap (Oracle - Picked)')
ax2.set_title('Absolute TFLOPS Left on Table (Top 7)', fontsize=10)
ax2.invert_yaxis()
for i, g in enumerate(gaps):
    ax2.text(g + 5, i, f'{g:.0f}T', va='center', fontsize=8)

# Panel 3: N/K ratio bias scatter
ax3 = axes[1, 0]
for i, s in enumerate(shapes_sorted):
    pt = s['picked_tile'].split('x')
    ot = s['oracle_tile'].split('x')
    pick_nk = int(pt[1]) / int(pt[2])
    oracle_nk = int(ot[1]) / int(ot[2])
    delta = pick_nk - oracle_nk
    color = '#d32f2f' if s['regret_pct'] > 25 else '#f57c00' if s['regret_pct'] > 15 else '#1976d2'
    ax3.scatter(delta, s['regret_pct'], c=color, s=60, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Picked N/K ratio − Oracle N/K ratio')
ax3.set_ylabel('Regret (%)')
ax3.set_title(f'N/K Ratio Bias vs Regret (r={corr:.2f})', fontsize=10)
ax3.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
ax3.axhline(y=25, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

# Panel 4: Root cause pie chart
ax4 = axes[1, 1]
cause_counts = {}
cause_regret = {}
for rc in root_causes:
    c = rc['cause']
    cause_counts[c] = cause_counts.get(c, 0) + 1
    cause_regret[c] = cause_regret.get(c, 0) + rc['regret']

labels_pie = list(cause_counts.keys())
sizes = [cause_regret[c] for c in labels_pie]
pie_colors = ['#d32f2f', '#f57c00', '#1976d2', '#388e3c', '#7b1fa2'][:len(labels_pie)]
wedges, texts, autotexts = ax4.pie(sizes, labels=None, autopct='%1.0f%%',
                                    colors=pie_colors, startangle=90)
ax4.legend(wedges, [f'{l}\n({cause_counts[l]} shapes, {cause_regret[l]:.0f}pp)' for l in labels_pie],
           loc='center left', bbox_to_anchor=(0.85, 0.5), fontsize=7)
ax4.set_title('Root Cause by Regret Weight', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
fig.text(0.5, 0.01,
         "python3 cycle2_analysis.py | Data: banff_data/correlation_results.json | origami.cpp:530-676 + gemm.cpp:570-896",
         ha='center', fontsize=7, style='italic')

outpath = '/home/ryaswann/global_orchestrator/reports/tasks/K-016/internal-auditor/key_result_internal-auditor.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {outpath}")

# Save detailed data
output_data = {
    'total_tflops_gap': total_gap,
    'fat_tail_count': sum(1 for s in shapes_sorted if s['regret_pct'] > 25),
    'fat_tail_tflops_gap': sum((s['oracle_gflops']-s['picked_gflops'])/1000 for s in shapes_sorted if s['regret_pct'] > 25),
    'nk_ratio_pearson_correlation': corr,
    'model_picks_higher_nk': int(n_higher),
    'root_causes': {c: {'count': cause_counts[c], 'total_regret_pp': cause_regret[c]} for c in cause_counts},
    'source_file': 'banff_data/correlation_results.json',
    'hardware': 'MI300X gfx942 (Banff cluster)',
    'cost_model_scoring': 'gemm.cpp:862: L_tile_single = max(L_compute*w_compute, L_mem*w_memory)',
    'tie_breaker': 'origami.cpp:570-676: arithmetic_intensity then problem-dimension heuristic'
}
with open('/home/ryaswann/global_orchestrator/reports/tasks/K-016/internal-auditor/cycle2_analysis.json', 'w') as f:
    json.dump(output_data, f, indent=2)
print("Analysis data saved: cycle2_analysis.json")
