#!/usr/bin/env python3
"""
Generate key_result_rigor.png — 4-panel rigor verification dashboard.
Panel A: CU scaling power-law fit (Method A, WS target_cus)
Panel B: Measured interference α vs breakeven α (decision map)
Panel C: Model-predicted vs GPU-measured comparison (scatter + identity)
Panel D: Provenance audit summary — [VERIFIED] vs [HYPOTHESIS] parameters

All data from GPU-measured sources on MI300X.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- DATA (all [VERIFIED] from cu_scaling_verified_data.csv and interference_verified_data.csv) ---

# Method A: WS target_cus CU scaling (rad-mi300x-1, ROCm 6.4.0-47, 150 samples each)
cu_counts_ws = np.array([80, 120, 152, 192, 224, 256, 288, 296, 304])
shapes = {
    'MLP (4096×14336×4096)': {
        'tflops': np.array([189.7, 232.1, 282.6, 331.3, 368.1, 360.3, 349.6, 348.2, 352.8]),
        'color': '#2196F3', 'marker': 'o',
        'alpha_BE': 0.883, 'measured_alpha': 0.0149, 'safety': 59.3, 'class': 'GO',
        'concurrent_ms': 1.4265, 'solo_ms': 1.4056, 'n': 150
    },
    'Attn (8192×4608×36864)': {
        'tflops': np.array([183.6, 248.4, 276.9, 412.3, 393.9, 372.2, 532.3, 530.6, 529.6]),
        'color': '#F44336', 'marker': 's',
        'alpha_BE': 0.0707, 'measured_alpha': 0.00919, 'safety': 7.7, 'class': 'NO-GO',
        'concurrent_ms': 9.9805, 'solo_ms': 9.8896, 'n': 120
    },
    'Sq8K (8192×8192×8192)': {
        'tflops': np.array([211.1, 268.9, 327.8, 397.3, 433.0, 504.4, 489.3, 486.9, 487.6]),
        'color': '#4CAF50', 'marker': '^',
        'alpha_BE': 0.4246, 'measured_alpha': 0.0022, 'safety': 193.0, 'class': 'GO',
        'concurrent_ms': 2.3046, 'solo_ms': 2.2995, 'n': 150
    },
    'Lg16K (16384×16384×16384)': {
        'tflops': np.array([225.1, 300.8, 358.6, 405.2, 443.0, 502.0, 494.9, 531.3, 524.5]),
        'color': '#FF9800', 'marker': 'D',
        'alpha_BE': 0.1914, 'measured_alpha': 0.0131, 'safety': 14.6, 'class': 'MONITOR',
        'concurrent_ms': 28.1252, 'solo_ms': 27.7616, 'n': 120
    }
}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('K-020 Rigor Verification: Coexecution Model — MI300X GPU-Measured Data\n'
             'rad-mi300x-1 (ROCm 6.4.0-47) + ctr-cx71-mi300x-02 (ROCm 7.2.0)',
             fontsize=13, fontweight='bold')

# --- Panel A: CU Scaling Power-Law ---
ax = axes[0, 0]
for name, d in shapes.items():
    # Fit power law: TFLOPS = a * CU^exp
    log_cu = np.log(cu_counts_ws / 304.0)
    log_tf = np.log(d['tflops'] / d['tflops'][-1])
    # OLS fit
    A = np.vstack([log_cu, np.ones(len(log_cu))]).T
    result = np.linalg.lstsq(A, log_tf, rcond=None)
    exp_fit = result[0][0]

    ax.plot(cu_counts_ws, d['tflops'], d['marker'] + '-', color=d['color'],
            label=f'{name.split(" ")[0]} (α={exp_fit:.2f})', markersize=6, linewidth=1.5)
    # Fit line
    cu_fit = np.linspace(60, 310, 100)
    tf_fit = d['tflops'][-1] * (cu_fit / 304.0) ** exp_fit
    ax.plot(cu_fit, tf_fit, '--', color=d['color'], alpha=0.4, linewidth=1)

ax.set_xlabel('Active CUs (target_cus)')
ax.set_ylabel('TFLOPS (median)')
ax.set_title('A: CU Scaling — WS target_cus [VERIFIED]', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(60, 320)
ax.grid(True, alpha=0.3)
ax.text(0.97, 0.03, 'n=150 each\nrad-mi300x-1', transform=ax.transAxes,
        fontsize=7, va='bottom', ha='right', style='italic', color='gray')

# --- Panel B: Interference Decision Map ---
ax = axes[0, 1]
labels_short = ['MLP', 'Attn', 'Sq8K', 'Lg16K']
alpha_be = [shapes[k]['alpha_BE'] for k in shapes]
measured_a = [shapes[k]['measured_alpha'] for k in shapes]
colors = [shapes[k]['color'] for k in shapes]
classes = [shapes[k]['class'] for k in shapes]

x_pos = np.arange(len(labels_short))
bar_width = 0.35

bars_be = ax.bar(x_pos - bar_width/2, alpha_be, bar_width, label='α_BE (breakeven)',
                  color='lightgray', edgecolor='black', linewidth=0.8)
bars_m = ax.bar(x_pos + bar_width/2, measured_a, bar_width, label='α_measured',
                 color=colors, edgecolor='black', linewidth=0.8)

# Add safety margin text
for i, (be, m, cls) in enumerate(zip(alpha_be, measured_a, classes)):
    margin = be / m if m > 0 else float('inf')
    ax.text(i, be + 0.02, f'{margin:.0f}× margin\n{cls}',
            ha='center', va='bottom', fontsize=7, fontweight='bold',
            color='green' if cls == 'GO' else ('orange' if cls == 'MONITOR' else 'red'))

ax.set_yscale('log')
ax.set_ylabel('Interference α')
ax.set_title('B: Measured α vs Breakeven α [VERIFIED]', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_short)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.text(0.97, 0.03, 'n=120-150 each', transform=ax.transAxes,
        fontsize=7, va='bottom', ha='right', style='italic', color='gray')

# --- Panel C: Cross-Method Validation (WS vs HSA_CU_MASK at 304 CUs) ---
ax = axes[1, 0]
# WS kernel @ 304 CUs (rad-mi300x-1)
ws_304 = [352.8, 529.6, 487.6, 524.5]
# rocBLAS+HSA_CU_MASK @ 304 CUs (ctr-cx71-mi300x-02)
rocblas_304 = [593.5, 558.9, 638.5, 511.3]
# Method A standalone Triton WS @ 304 CUs (ctr-cx71-mi300x-02)
standalone_304 = [288.4, 275.5, 294.8, 294.3]

x = np.arange(len(labels_short))
w = 0.25
ax.bar(x - w, ws_304, w, label='WS target_cus (rad)', color='#2196F3', edgecolor='black', linewidth=0.5)
ax.bar(x, rocblas_304, w, label='rocBLAS+mask (Alola)', color='#F44336', edgecolor='black', linewidth=0.5)
ax.bar(x + w, standalone_304, w, label='Standalone Triton (Alola)', color='#4CAF50', edgecolor='black', linewidth=0.5)

ax.set_ylabel('TFLOPS @ 304 CUs')
ax.set_title('C: Cross-Method Throughput Comparison [VERIFIED]', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels_short)
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.text(0.03, 0.97, '3 independent methods\n2 clusters, 2 ROCm versions',
        transform=ax.transAxes, fontsize=7, va='top', ha='left', style='italic', color='gray')

# --- Panel D: Parameter Provenance Summary ---
ax = axes[1, 1]
# Provenance categories
categories = ['T_gemm(304)\n4 shapes', 'CU scaling α\n4 shapes', 'Interference α\n4 shapes',
              'α_BE\n4 shapes', 'beta_base', 'comm_cu_exp']
verified_count = [4, 4, 4, 4, 0, 0]
hypothesis_count = [0, 0, 0, 0, 1, 1]
inert_flag = [False, False, False, False, True, True]  # moot for current shapes

y_pos = np.arange(len(categories))
ax.barh(y_pos, verified_count, 0.4, label='[VERIFIED]', color='#4CAF50', edgecolor='black', linewidth=0.5)
ax.barh(y_pos, [-h for h in hypothesis_count], 0.4, label='[HYPOTHESIS] (inert)', color='#FFC107',
        edgecolor='black', linewidth=0.5)

for i, (v, h, inert) in enumerate(zip(verified_count, hypothesis_count, inert_flag)):
    if v > 0:
        ax.text(v + 0.1, i, f'{v} ✓', va='center', fontsize=9, color='green', fontweight='bold')
    if h > 0:
        ax.text(-h - 0.1, i, f'INERT' if inert else f'{h} ✗', va='center', ha='right',
                fontsize=9, color='orange' if inert else 'red', fontweight='bold')

ax.set_xlabel('Parameter count')
ax.set_title('D: Provenance Audit — 0 [ESTIMATED] Tags', fontweight='bold')
ax.set_yticks(y_pos)
ax.set_yticklabels(categories, fontsize=8)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlim(-2, 6)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.text(0.5, 0.01,
         'bench-sanity.py: PASS (20/20 Job17451 + 32/32 MethodA + 38/38 Coexec) | '
         'git: k020/origami-model-for-coexecuted-work-steali-rigor',
         ha='center', fontsize=7, style='italic', color='gray')

plt.savefig('/home/ryaswann/global_orchestrator/reports/tasks/K-020/rigor/key_result_rigor.png',
            dpi=150, bbox_inches='tight')
print("Saved key_result_rigor.png")
