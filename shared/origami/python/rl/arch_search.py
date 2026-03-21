#!/usr/bin/env python3
"""Architecture search on collected data. Runs on CPU while GPUs collect more data.

Tests different model configs with 5-fold shape-level cross-validation.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Reuse existing modules
from buffer import ReplayBuffer
from evaluate import evaluate_model, MIN_ORACLE_GFLOPS
from model import CorrectionMLP


def load_data(parquet_path: str = 'data/gfx942/bf16_r_TN/dataset.parquet'):
    """Load data from parquet into arrays."""
    buf = ReplayBuffer(max_size=500000)
    buf.load_parquet(Path(parquet_path))
    return buf.get_all()


def cross_validate(data, model_fn, n_folds=5, train_steps=2000, lr=1e-3,
                   weight_decay=1e-5, batch_size=512):
    """Shape-level k-fold cross-validation.

    Returns dict with mean metrics across folds.
    """
    shape_ids = data['shape_ids']
    unique_shapes = np.unique(shape_ids)
    np.random.shuffle(unique_shapes)

    fold_size = len(unique_shapes) // n_folds
    fold_metrics = []

    for fold in range(n_folds):
        # Split shapes into train/test
        test_shapes = set(unique_shapes[fold * fold_size:(fold + 1) * fold_size])
        test_mask = np.array([sid in test_shapes for sid in shape_ids])
        train_mask = ~test_mask

        if train_mask.sum() < 100 or test_mask.sum() < 100:
            continue

        # Build train/test datasets
        train_features = data['features'][train_mask]
        train_targets = data['log_correction'][train_mask]
        train_shape_ids = data['shape_ids'][train_mask]

        test_data = {
            'features': data['features'][test_mask],
            'log_correction': data['log_correction'][test_mask],
            'shape_ids': data['shape_ids'][test_mask],
            'gflops': data['gflops'][test_mask],
            'M': data['M'][test_mask],
            'N': data['N'][test_mask],
            'K': data['K'][test_mask],
        }

        # Train
        model = model_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        n_train = len(train_features)
        for step in range(train_steps):
            idx = np.random.randint(0, n_train, size=min(batch_size, n_train))
            features_t = torch.from_numpy(train_features[idx]).float()
            targets_t = torch.from_numpy(train_targets[idx]).float()

            model.train()
            preds = model(features_t)
            loss = (preds - targets_t).abs().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on test set
        metrics = evaluate_model(model, test_data)
        fold_metrics.append(metrics)

    if not fold_metrics:
        return None

    # Average across folds
    avg = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics if isinstance(m[key], (int, float))]
        if vals:
            avg[key] = sum(vals) / len(vals)
    avg['n_folds'] = len(fold_metrics)
    return avg


def run_experiment(name, data, model_fn, **kwargs):
    """Run a single experiment and return results."""
    t0 = time.time()
    metrics = cross_validate(data, model_fn, **kwargs)
    elapsed = time.time() - t0
    if metrics:
        metrics['name'] = name
        metrics['time_s'] = elapsed
        n_params = sum(p.numel() for p in model_fn().parameters())
        metrics['n_params'] = n_params
    return metrics


class WiderMLP(nn.Module):
    """Wider MLP variant."""
    def __init__(self, input_dim=92, hidden_dim=512, n_blocks=2, dropout=0.1):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        ) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_norm(x)
        h = self.input_proj(h)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h).squeeze(-1)


class DeeperMLP(nn.Module):
    """Deeper MLP variant."""
    def __init__(self, input_dim=92, hidden_dim=256, n_blocks=4, dropout=0.1):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        ) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_norm(x)
        h = self.input_proj(h)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h).squeeze(-1)


class BottleneckMLP(nn.Module):
    """Bottleneck MLP: wide -> narrow -> wide."""
    def __init__(self, input_dim=92, hidden_dim=256, bottleneck=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[nn.Sequential(
            nn.Linear(hidden_dim, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        ) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_norm(x)
        h = self.input_proj(h)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h).squeeze(-1)


class LayerNormMLP(nn.Module):
    """MLP with LayerNorm instead of BatchNorm."""
    def __init__(self, input_dim=92, hidden_dim=256, n_blocks=2, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        ) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_norm(x)
        h = self.input_proj(h)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h).squeeze(-1)


class TwoHeadMLP(nn.Module):
    """Two-head: shared backbone, separate heads for score + confidence."""
    def __init__(self, input_dim=92, hidden_dim=256, n_blocks=2, dropout=0.1):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        ) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_norm(x)
        h = self.input_proj(h)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h).squeeze(-1)


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data['features'])} entries, {len(np.unique(data['shape_ids']))} shapes")

    # Compute within-shape percentile targets (same as train.py)
    shape_ids = data['shape_ids']
    gflops = data['gflops']
    targets = np.zeros(len(gflops), dtype=np.float32)
    for sid in np.unique(shape_ids):
        mask = shape_ids == sid
        g = gflops[mask]
        ranks = np.argsort(np.argsort(g))
        targets[mask] = ranks / max(len(ranks) - 1, 1)
    data['log_correction'] = targets

    experiments = [
        # Baseline (current model)
        ("baseline_256x2", lambda: CorrectionMLP(92, 256, 2, 0.1)),

        # Width variations
        ("wide_512x2", lambda: WiderMLP(92, 512, 2, 0.1)),
        ("narrow_128x2", lambda: CorrectionMLP(92, 128, 2, 0.1)),
        ("very_wide_1024x2", lambda: WiderMLP(92, 1024, 2, 0.1)),

        # Depth variations
        ("deep_256x4", lambda: DeeperMLP(92, 256, 4, 0.1)),
        ("deep_256x6", lambda: DeeperMLP(92, 256, 6, 0.1)),
        ("shallow_256x1", lambda: CorrectionMLP(92, 256, 1, 0.1)),

        # Dropout variations
        ("nodrop_256x2", lambda: CorrectionMLP(92, 256, 2, 0.0)),
        ("highdrop_256x2", lambda: CorrectionMLP(92, 256, 2, 0.3)),

        # Bottleneck
        ("bottleneck_256_64x2", lambda: BottleneckMLP(92, 256, 64, 2, 0.1)),

        # LayerNorm
        ("layernorm_256x2", lambda: LayerNormMLP(92, 256, 2, 0.1)),

        # Learning rate variations (tested via kwargs)
        ("baseline_lr3e-4", lambda: CorrectionMLP(92, 256, 2, 0.1)),
        ("baseline_lr3e-3", lambda: CorrectionMLP(92, 256, 2, 0.1)),
    ]

    lr_overrides = {
        "baseline_lr3e-4": 3e-4,
        "baseline_lr3e-3": 3e-3,
    }

    results = []
    for name, model_fn in experiments:
        lr = lr_overrides.get(name, 1e-3)
        print(f"\n{'='*60}")
        print(f"Experiment: {name} (lr={lr})")
        print(f"{'='*60}")

        metrics = run_experiment(name, data, model_fn, lr=lr, train_steps=3000)
        if metrics:
            print(f"  Regret: {metrics['regret_pct']:.2f}%  Top-1: {metrics['top1_acc']:.1%}  "
                  f"Top-5: {metrics['top5_acc']:.1%}  MAE: {metrics['mae']:.4f}  "
                  f"Params: {metrics['n_params']:,}  Time: {metrics['time_s']:.1f}s")
            results.append(metrics)
        else:
            print(f"  FAILED")

    # Summary table
    print(f"\n{'='*80}")
    print(f"ARCHITECTURE SEARCH RESULTS (5-fold CV, {len(np.unique(shape_ids))} shapes)")
    print(f"{'='*80}")
    print(f"{'Name':30s} {'Regret':>8s} {'Top-1':>7s} {'Top-5':>7s} {'MAE':>8s} {'Params':>8s}")
    print(f"{'-'*80}")

    results.sort(key=lambda r: r['regret_pct'])
    for r in results:
        print(f"{r['name']:30s} {r['regret_pct']:7.2f}% {r['top1_acc']:6.1%} {r['top5_acc']:6.1%} "
              f"{r['mae']:8.4f} {r['n_params']:>8,}")

    # Save results
    with open('arch_search_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to arch_search_results.json")


if __name__ == '__main__':
    main()
