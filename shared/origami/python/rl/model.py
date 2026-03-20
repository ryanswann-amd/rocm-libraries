#!/usr/bin/env python3
"""Residual MLP for latency correction prediction.

Predicts log(actual_latency / origami_latency) as a correction factor.
Physics-informed: the model learns residuals on top of origami's analytical
predictions, not absolute latencies.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A single residual block: Linear -> GELU -> Dropout + skip connection."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CorrectionMLP(nn.Module):
    """MLP that predicts log-correction factor for origami latency.

    Architecture: Linear(in -> hidden) -> GELU -> ResidualBlock x N -> Linear(hidden -> 1)

    Input:  92 features per (problem, kernel) pair
    Output: scalar log_correction = log(actual_us / origami_us)
    """

    def __init__(
        self,
        input_dim: int = 92,
        hidden_dim: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns (batch,) shaped predictions."""
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.head(h).squeeze(-1)


def ranking_loss(
    pred_corrections: torch.Tensor,
    actual_corrections: torch.Tensor,
    shape_ids: torch.Tensor,
) -> torch.Tensor:
    """Pairwise ranking loss within each shape.

    For pairs (i, j) within the same shape, penalize inversions:
    if actual_i < actual_j (i is faster) but pred_i > pred_j (model says i is slower).

    We use log-corrections, so lower correction = relatively faster than origami predicted.
    We want the model to rank kernels by actual speed within a shape.
    """
    unique_shapes = shape_ids.unique()
    total_loss = torch.tensor(0.0, device=pred_corrections.device)
    n_pairs = 0

    for sid in unique_shapes:
        mask = shape_ids == sid
        if mask.sum() < 2:
            continue
        pred = pred_corrections[mask]
        actual = actual_corrections[mask]

        # All pairs within this shape
        n = pred.size(0)
        # Sample pairs to avoid O(n^2) for large shapes
        max_pairs = min(n * (n - 1) // 2, 256)
        if n > 23:  # 23*22/2 = 253, close to 256
            idx = torch.randperm(n, device=pred.device)[:23]
            pred = pred[idx]
            actual = actual[idx]
            n = pred.size(0)

        # Pairwise differences
        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)   # (n, n)
        actual_diff = actual.unsqueeze(0) - actual.unsqueeze(1)  # (n, n)

        # Upper triangle only (avoid double counting)
        triu_mask = torch.triu(torch.ones(n, n, device=pred.device, dtype=torch.bool), diagonal=1)
        pred_d = pred_diff[triu_mask]
        actual_d = actual_diff[triu_mask]

        # Margin ranking loss: if actual_d > 0 (i slower than j), pred_d should be > 0
        # loss = max(0, -sign(actual_d) * pred_d + margin)
        margin = 0.1
        signs = actual_d.sign()
        pair_loss = torch.clamp(margin - signs * pred_d, min=0)

        total_loss = total_loss + pair_loss.sum()
        n_pairs += pair_loss.numel()

    if n_pairs == 0:
        return torch.tensor(0.0, device=pred_corrections.device)
    return total_loss / n_pairs


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    shape_ids: torch.Tensor,
    ranking_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Combined loss: MAE + ranking_weight * ranking_loss.

    Returns (loss, metrics_dict).
    """
    mae = (pred - target).abs().mean()
    rloss = ranking_loss(pred, target, shape_ids)
    total = mae + ranking_weight * rloss
    return total, {
        'mae': mae.item(),
        'ranking_loss': rloss.item(),
        'total_loss': total.item(),
    }
