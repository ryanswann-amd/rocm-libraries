#!/usr/bin/env python3
"""Head-to-head comparison: trained model vs origami on fresh shapes.

For each shape:
1. Benchmark all solutions with hipblaslt-bench
2. Extract features + origami predictions
3. Model picks best kernel, origami picks best kernel
4. Compare actual GFLOPS of each pick vs oracle

Usage:
  python3 compare.py --bench /path/to/hipblaslt-bench --checkpoint checkpoints/best.pt --n-shapes 20 --gpu 0
"""

import argparse
import math
import sys
import time

import numpy as np
import torch

from bench import run_hipblaslt_bench
from buffer import ReplayBuffer
from evaluate import MIN_ORACLE_GFLOPS
from features import (
    NUM_FEATURES, extract_features_for_shape, get_clock_mhz, make_hardware,
)
from model import CorrectionMLP
from train import sample_shape, load_config


def run_comparison(cfg, bench_path, checkpoint_path, n_shapes, gpu_id):
    hw = make_hardware(cfg['arch'])
    clock_mhz = get_clock_mhz(cfg['arch'])
    s = cfg['sampling']
    device = torch.device('cpu')

    # Load model
    model = CorrectionMLP(
        input_dim=cfg['model']['input_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        n_blocks=cfg['model']['n_blocks'],
        dropout=cfg['model']['dropout'],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path} (step {ckpt['step']})")
    print(f"Checkpoint metrics: {ckpt.get('metrics', {})}")
    print()

    # Results
    results = []

    print(f"{'Shape':>20s}  {'#Sol':>5s}  {'Oracle':>10s}  {'Model':>10s}  {'Origami':>10s}  "
          f"{'M_reg':>7s}  {'O_reg':>7s}  {'M_tile':>16s}  {'O_tile':>16s}  {'Best_tile':>16s}")
    print("-" * 140)

    for i in range(n_shapes):
        M, N, K = sample_shape(cfg)
        shape_str = f"{M}x{N}x{K}"

        # Benchmark
        solutions = run_hipblaslt_bench(
            bench_path, M, N, K,
            transA=s['transA'], transB=s['transB'], dtype=s['dtype'],
            iters=cfg['bench']['iters'], cold_iters=cfg['bench']['cold_iters'],
            gpu_id=gpu_id, best_per_tile=True,
        )

        if not solutions or len(solutions) < 2:
            print(f"{shape_str:>20s}  {'SKIP':>5s}  (no solutions)")
            continue

        # Extract features
        feat_results = extract_features_for_shape(
            hw, M, N, K, solutions,
            dtype=s['dtype'], transA=s['transA'], transB=s['transB'],
            n_cu=cfg['n_cu'], clock_mhz=clock_mhz,
        )

        if not feat_results:
            print(f"{shape_str:>20s}  {'SKIP':>5s}  (no features)")
            continue

        # Oracle: best actual GFLOPS
        gflops_arr = np.array([r['gflops'] for r in feat_results])
        oracle_idx = np.argmax(gflops_arr)
        oracle_gflops = gflops_arr[oracle_idx]

        if oracle_gflops < MIN_ORACLE_GFLOPS:
            print(f"{shape_str:>20s}  {len(feat_results):>5d}  {oracle_gflops:>10.0f}  {'(too small)':>10s}")
            continue

        # Model prediction: pick kernel with highest predicted percentile
        features = np.stack([r['features'] for r in feat_results])
        with torch.no_grad():
            preds = model(torch.from_numpy(features).float()).numpy()
        model_idx = np.argmax(preds)  # highest predicted percentile = fastest
        model_gflops = gflops_arr[model_idx]

        # Origami prediction: pick kernel with lowest origami latency
        log_origami_us = features[:, 5]  # feature index 5 = log(origami_us)
        origami_idx = np.argmin(log_origami_us)
        origami_gflops = gflops_arr[origami_idx]

        model_regret = 1.0 - model_gflops / oracle_gflops
        origami_regret = 1.0 - origami_gflops / oracle_gflops

        # Tile names
        model_tile = feat_results[model_idx].get('kernel_sig', '')
        origami_tile = feat_results[origami_idx].get('kernel_sig', '')
        oracle_tile = feat_results[oracle_idx].get('kernel_sig', '')

        # Extract MT from kernel sig
        import re
        def get_mt(sig):
            m = re.search(r'MT(\d+x\d+x\d+)', sig)
            return m.group(1) if m else '?'

        model_mt = get_mt(model_tile)
        origami_mt = get_mt(origami_tile)
        oracle_mt = get_mt(oracle_tile)

        results.append({
            'shape': shape_str, 'M': M, 'N': N, 'K': K,
            'n_solutions': len(feat_results),
            'oracle_gflops': oracle_gflops,
            'model_gflops': model_gflops,
            'origami_gflops': origami_gflops,
            'model_regret': model_regret,
            'origami_regret': origami_regret,
            'model_tile': model_mt,
            'origami_tile': origami_mt,
            'oracle_tile': oracle_mt,
        })

        winner = "MODEL" if model_gflops > origami_gflops else ("TIE" if model_gflops == origami_gflops else "ORIG")
        print(f"{shape_str:>20s}  {len(feat_results):>5d}  {oracle_gflops:>10.0f}  {model_gflops:>10.0f}  {origami_gflops:>10.0f}  "
              f"{model_regret:>6.1%}  {origami_regret:>6.1%}  {model_mt:>16s}  {origami_mt:>16s}  {oracle_mt:>16s}  {winner}")
        sys.stdout.flush()

    if not results:
        print("\nNo results collected!")
        return

    # Summary
    print()
    print("=" * 140)
    print("SUMMARY")
    print("=" * 140)
    n = len(results)
    model_wins = sum(1 for r in results if r['model_gflops'] > r['origami_gflops'])
    origami_wins = sum(1 for r in results if r['origami_gflops'] > r['model_gflops'])
    ties = n - model_wins - origami_wins

    avg_model_regret = np.mean([r['model_regret'] for r in results])
    avg_origami_regret = np.mean([r['origami_regret'] for r in results])

    avg_model_gflops = np.mean([r['model_gflops'] for r in results])
    avg_origami_gflops = np.mean([r['origami_gflops'] for r in results])
    avg_oracle_gflops = np.mean([r['oracle_gflops'] for r in results])

    model_top1 = sum(1 for r in results if r['model_regret'] == 0) / n
    origami_top1 = sum(1 for r in results if r['origami_regret'] == 0) / n

    print(f"  Shapes evaluated:       {n}")
    print(f"  Model wins:             {model_wins}/{n} ({model_wins/n:.0%})")
    print(f"  Origami wins:           {origami_wins}/{n} ({origami_wins/n:.0%})")
    print(f"  Ties:                   {ties}/{n}")
    print(f"  Model avg regret:       {avg_model_regret:.2%}")
    print(f"  Origami avg regret:     {avg_origami_regret:.2%}")
    print(f"  Model Top-1:            {model_top1:.1%}")
    print(f"  Origami Top-1:          {origami_top1:.1%}")
    print(f"  Avg GFLOPS (model):     {avg_model_gflops:,.0f}")
    print(f"  Avg GFLOPS (origami):   {avg_origami_gflops:,.0f}")
    print(f"  Avg GFLOPS (oracle):    {avg_oracle_gflops:,.0f}")
    print(f"  Model efficiency:       {avg_model_gflops/avg_oracle_gflops:.1%} of oracle")
    print(f"  Origami efficiency:     {avg_origami_gflops/avg_oracle_gflops:.1%} of oracle")


def main():
    parser = argparse.ArgumentParser(description='Model vs Origami comparison')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--bench', required=True, help='Path to hipblaslt-bench')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt', help='Model checkpoint')
    parser.add_argument('--n-shapes', type=int, default=30, help='Number of shapes to test')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg['bench']['path'] = args.bench
    run_comparison(cfg, args.bench, args.checkpoint, args.n_shapes, args.gpu)


if __name__ == '__main__':
    main()
