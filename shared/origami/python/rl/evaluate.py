#!/usr/bin/env python3
"""Evaluation metrics for the correction model.

Computes:
- MAE on log-correction predictions
- Selection regret: 1 - picked_gflops / oracle_gflops
- Top-1 accuracy: fraction of shapes where predicted best = actual best
- Top-5 accuracy
"""

import logging
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    buffer_data: dict,
    device: torch.device = torch.device('cpu'),
) -> dict:
    """Evaluate model on all buffer data.

    Args:
        model: CorrectionMLP
        buffer_data: dict from buffer.get_all() with keys:
            features, log_correction, shape_ids, gflops, M, N, K

    Returns:
        dict with: mae, regret_mean, top1_acc, top5_acc, n_shapes, n_entries
    """
    if buffer_data is None or len(buffer_data['features']) == 0:
        return {'mae': float('inf'), 'regret_mean': 1.0, 'top1_acc': 0.0,
                'top5_acc': 0.0, 'n_shapes': 0, 'n_entries': 0}

    model.eval()
    with torch.no_grad():
        features_t = torch.from_numpy(buffer_data['features']).float().to(device)
        targets = buffer_data['log_correction']
        preds = model(features_t).cpu().numpy()

    # Global MAE
    mae = np.abs(preds - targets).mean()

    # Per-shape metrics
    shape_ids = buffer_data['shape_ids']
    gflops = buffer_data['gflops']
    unique_shapes = np.unique(shape_ids)

    top1_hits = 0
    top5_hits = 0
    regrets = []

    for sid in unique_shapes:
        mask = shape_ids == sid
        s_preds = preds[mask]
        s_gflops = gflops[mask]

        if len(s_preds) < 2:
            continue

        # The model predicts log(actual/origami). Lower predicted correction
        # means the kernel is faster than origami expected. But we need to rank
        # by predicted actual latency = origami_us * exp(pred_correction).
        # Since we want highest gflops, and gflops ~ 1/latency, we want
        # lowest predicted correction (most negative = fastest actual).
        pred_ranking = np.argsort(s_preds)  # ascending correction = fastest first
        actual_ranking = np.argsort(-s_gflops)  # descending gflops = fastest first

        best_actual_idx = actual_ranking[0]
        best_pred_idx = pred_ranking[0]

        # Top-1: did we pick the actual fastest kernel?
        if best_pred_idx == best_actual_idx:
            top1_hits += 1

        # Top-5: is the actual fastest in our top-5 picks?
        top5_set = set(pred_ranking[:5])
        if best_actual_idx in top5_set:
            top5_hits += 1

        # Regret: how much gflops do we lose vs oracle?
        oracle_gflops = s_gflops[best_actual_idx]
        picked_gflops = s_gflops[best_pred_idx]
        regret = 1.0 - (picked_gflops / oracle_gflops) if oracle_gflops > 0 else 0.0
        regrets.append(regret)

    n_shapes = len(regrets)
    metrics = {
        'mae': float(mae),
        'regret_mean': float(np.mean(regrets)) if regrets else 1.0,
        'regret_pct': float(np.mean(regrets) * 100) if regrets else 100.0,
        'top1_acc': float(top1_hits / n_shapes) if n_shapes > 0 else 0.0,
        'top5_acc': float(top5_hits / n_shapes) if n_shapes > 0 else 0.0,
        'n_shapes': n_shapes,
        'n_entries': len(preds),
    }

    return metrics


def format_metrics(metrics: dict) -> str:
    """Format metrics dict as a human-readable string."""
    return (
        f"MAE={metrics['mae']:.4f}  "
        f"Regret={metrics['regret_pct']:.2f}%  "
        f"Top-1={metrics['top1_acc']:.1%}  "
        f"Top-5={metrics['top5_acc']:.1%}  "
        f"({metrics['n_entries']} entries, {metrics['n_shapes']} shapes)"
    )
