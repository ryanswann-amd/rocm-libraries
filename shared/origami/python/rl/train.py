#!/usr/bin/env python3
"""Origami RL Trainer — Continuous learning loop for GEMM kernel selection.

Autoresearch-style: autonomous loop, clear metric, experiment tracking, never stops.

Core loop:
1. Sample random GEMM shapes (log-uniform M/N/K)
2. Benchmark each with hipblaslt-bench --algo_method all (parallel across GPUs)
3. Extract 92 features per (shape, kernel) pair using origami analytical model
4. Add to replay buffer
5. Train residual MLP on mini-batches from buffer
6. Evaluate: MAE on log-correction + selection regret
7. If regret improved -> checkpoint
8. Loop forever

Usage:
  # Validation mode (quick sanity check)
  python3 train.py --validate --bench /path/to/hipblaslt-bench --gpus 0

  # Full training
  python3 train.py --bench /path/to/hipblaslt-bench --gpus 0 1 2 3
"""

import argparse
import datetime
import json
import logging
import math
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml

from bench import run_hipblaslt_bench, save_bench_csv
from buffer import ReplayBuffer
from evaluate import evaluate_model, format_metrics
from features import (
    NUM_FEATURES,
    extract_features_for_shape,
    get_clock_mhz,
    make_hardware,
)
from model import CorrectionMLP, compute_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('train')


def sample_shape(cfg: dict) -> tuple[int, int, int]:
    """Sample a random GEMM shape with log-uniform dimensions."""
    s = cfg['sampling']
    lo_m, hi_m = s['log2_m']
    lo_n, hi_n = s['log2_n']
    lo_k, hi_k = s['log2_k']

    M = max(16, round(2 ** np.random.uniform(lo_m, hi_m) / 16) * 16)
    N = max(16, round(2 ** np.random.uniform(lo_n, hi_n) / 16) * 16)
    K = max(16, round(2 ** np.random.uniform(lo_k, hi_k) / 16) * 16)
    return M, N, K


def benchmark_shape(bench_path: str, M: int, N: int, K: int,
                    gpu_id: int, cfg: dict) -> list[dict]:
    """Benchmark a single shape on a single GPU."""
    bench_cfg = cfg['bench']
    return run_hipblaslt_bench(
        bench_path, M, N, K,
        transA=cfg['sampling']['transA'],
        transB=cfg['sampling']['transB'],
        dtype=cfg['sampling']['dtype'],
        iters=bench_cfg['iters'],
        cold_iters=bench_cfg['cold_iters'],
        gpu_id=gpu_id,
        timeout=bench_cfg['timeout'],
    )


def process_shape(hw, shape_id: int, M: int, N: int, K: int,
                  solutions: list[dict], cfg: dict) -> list[dict]:
    """Extract features and build buffer entries for a shape.

    Target is within-shape gflops percentile (0=worst, 1=best).
    This normalizes away between-shape variation entirely.
    """
    s = cfg['sampling']
    feat_results = extract_features_for_shape(
        hw, M, N, K, solutions,
        dtype=s['dtype'],
        transA=s['transA'],
        transB=s['transB'],
        n_cu=cfg['n_cu'],
        clock_mhz=get_clock_mhz(cfg['arch']),
    )

    if not feat_results:
        return []

    # Compute within-shape percentile targets (higher = faster)
    gflops_arr = np.array([r['gflops'] for r in feat_results])
    ranks = np.argsort(np.argsort(gflops_arr))  # rank indices
    percentiles = ranks / max(len(ranks) - 1, 1)  # 0=worst, 1=best

    entries = []
    for r, pct in zip(feat_results, percentiles):
        entries.append({
            'shape_id': shape_id,
            'M': M, 'N': N, 'K': K,
            'kernel_sig': r['kernel_sig'],
            'features': r['features'],
            'log_correction': float(pct),  # target: within-shape percentile
            'gflops': r['gflops'],
            'us': r['us'],
        })
    return entries


def train_step(model, optimizer, buffer, cfg, device) -> dict:
    """Single training step on a mini-batch from the buffer."""
    batch = buffer.sample(cfg['training']['batch_size'])
    if batch is None:
        return {'total_loss': 0, 'mae': 0, 'ranking_loss': 0}

    features = torch.from_numpy(batch['features']).float().to(device)
    targets = torch.from_numpy(batch['log_correction']).float().to(device)
    shape_ids = torch.from_numpy(batch['shape_ids']).long().to(device)

    model.train()
    preds = model(features)
    loss, metrics = compute_loss(preds, targets, shape_ids,
                                 ranking_weight=cfg['training']['ranking_weight'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return metrics


def save_checkpoint(model, optimizer, step, metrics, path: Path):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)


def load_checkpoint(model, optimizer, path: Path, device) -> int:
    """Load model checkpoint. Returns the step number."""
    if not path.exists():
        return 0
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    log.info("Loaded checkpoint from step %d: %s", ckpt['step'], ckpt.get('metrics', {}))
    return ckpt['step']


def append_results_tsv(path: Path, step: int, metrics: dict, action: str):
    """Append a row to results.tsv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    with open(path, 'a') as f:
        if header_needed:
            f.write("timestamp\tstep\tmae\tregret_pct\ttop1_acc\ttop5_acc\tn_shapes\tn_entries\taction\n")
        f.write(
            f"{datetime.datetime.now().isoformat()}\t"
            f"{step}\t"
            f"{metrics['mae']:.6f}\t"
            f"{metrics['regret_pct']:.4f}\t"
            f"{metrics['top1_acc']:.4f}\t"
            f"{metrics['top5_acc']:.4f}\t"
            f"{metrics['n_shapes']}\t"
            f"{metrics['n_entries']}\t"
            f"{action}\n"
        )


def append_training_log(path: Path, entry: dict):
    """Append a JSON line to the training log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(entry, default=str) + '\n')


def load_config(config_path: str) -> dict:
    """Load config from YAML, with defaults."""
    default_cfg = {
        'bench': {'path': '/opt/rocm/bin/hipblaslt-bench', 'iters': 20,
                  'cold_iters': 3, 'timeout': 300},
        'gpus': [0],
        'arch': 'gfx942',
        'n_cu': 304,
        'sampling': {
            'dtype': 'bf16_r', 'transA': 'T', 'transB': 'N',
            'log2_m': [0, 15], 'log2_n': [0, 15], 'log2_k': [0, 15],
        },
        'model': {'input_dim': 92, 'hidden_dim': 256, 'n_blocks': 2, 'dropout': 0.1},
        'training': {
            'lr': 0.001, 'weight_decay': 1e-5, 'ranking_weight': 0.1,
            'batch_size': 512, 'train_steps_per_shape': 20, 'buffer_size': 100000,
        },
        'eval_interval': 50,
        'data_dir': './data',
        'checkpoint_dir': './checkpoints',
    }

    if os.path.exists(config_path):
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        # Shallow merge top-level, deep merge dicts
        for key, val in file_cfg.items():
            if isinstance(val, dict) and key in default_cfg and isinstance(default_cfg[key], dict):
                default_cfg[key].update(val)
            else:
                default_cfg[key] = val

    return default_cfg


def run_validation(cfg: dict, bench_path: str, gpus: list[int]):
    """Validation mode: benchmark 5 shapes, train briefly, report metrics."""
    log.info("=== VALIDATION MODE ===")
    device = torch.device('cpu')  # validation is CPU-only for simplicity
    hw = make_hardware(cfg['arch'])

    buffer = ReplayBuffer(max_size=10000)
    model = CorrectionMLP(
        input_dim=cfg['model']['input_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        n_blocks=cfg['model']['n_blocks'],
        dropout=cfg['model']['dropout'],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )

    # Benchmark 5 shapes
    n_val_shapes = 5
    gpu = gpus[0]
    total_solutions = 0

    for i in range(n_val_shapes):
        M, N, K = sample_shape(cfg)
        log.info("Validate shape %d/%d: %dx%dx%d on GPU %d", i + 1, n_val_shapes, M, N, K, gpu)

        solutions = benchmark_shape(bench_path, M, N, K, gpu, cfg)
        if not solutions:
            log.warning("No solutions for %dx%dx%d, skipping", M, N, K)
            continue

        entries = process_shape(hw, i, M, N, K, solutions, cfg)
        if entries:
            buffer.add_shape(entries)
            total_solutions += len(entries)
            log.info("  -> %d solutions, %d with features", len(solutions), len(entries))

    if total_solutions == 0:
        log.error("VALIDATION FAILED: no data collected")
        return False

    log.info("Collected %d entries across %d shapes", len(buffer), buffer.n_shapes)

    # Train for 30 seconds
    t0 = time.monotonic()
    step = 0
    while time.monotonic() - t0 < 30:
        metrics = train_step(model, optimizer, buffer, cfg, device)
        step += 1
        if step % 50 == 0:
            log.info("  Train step %d: MAE=%.4f", step, metrics['mae'])

    # Evaluate
    buf_data = buffer.get_all()
    eval_metrics = evaluate_model(model, buf_data, device)
    log.info("=== VALIDATION RESULTS ===")
    log.info("  %s", format_metrics(eval_metrics))
    log.info("  Training steps: %d in %.1fs", step, time.monotonic() - t0)

    # Validation passes if: data pipeline works + model trains + MAE improves
    ok = eval_metrics['n_shapes'] > 0 and eval_metrics['mae'] < float('inf')
    log.info("  Status: %s", "PASS" if ok else "FAIL")
    return ok


def run_training(cfg: dict, bench_path: str, gpus: list[int]):
    """Main training loop. Runs forever until interrupted."""
    log.info("=== ORIGAMI RL TRAINER ===")
    log.info("GPUs: %s, Arch: %s, Dtype: %s", gpus, cfg['arch'], cfg['sampling']['dtype'])

    device = torch.device('cpu')  # MLP is small, CPU is fine
    hw = make_hardware(cfg['arch'])

    # Paths
    data_dir = Path(cfg['data_dir']) / cfg['arch'] / f"{cfg['sampling']['dtype']}_{cfg['sampling']['transA']}{cfg['sampling']['transB']}"
    bench_dir = data_dir / 'benchmarks'
    dataset_path = data_dir / 'dataset.parquet'
    ckpt_dir = Path(cfg['checkpoint_dir'])
    results_path = Path('results.tsv')
    training_log_path = data_dir / 'training_log.jsonl'

    # Initialize
    buffer = ReplayBuffer(max_size=cfg['training']['buffer_size'])
    model = CorrectionMLP(
        input_dim=cfg['model']['input_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        n_blocks=cfg['model']['n_blocks'],
        dropout=cfg['model']['dropout'],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )

    # Resume from checkpoint
    best_ckpt = ckpt_dir / 'best.pt'
    start_step = load_checkpoint(model, optimizer, best_ckpt, device)

    # Load existing dataset
    n_loaded = buffer.load_parquet(dataset_path)
    if n_loaded > 0:
        log.info("Resumed with %d entries from existing dataset", n_loaded)

    # Tracking
    best_regret = float('inf')
    shape_counter = buffer.n_shapes
    total_train_steps = start_step
    shapes_since_eval = 0

    # Graceful shutdown
    shutdown = False
    def handle_signal(signum, frame):
        nonlocal shutdown
        log.info("Shutdown signal received, finishing current batch...")
        shutdown = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log.info("Starting training loop (buffer: %d entries, %d shapes)", len(buffer), buffer.n_shapes)

    while not shutdown:
        # Sample shapes for this batch (one per GPU)
        batch_shapes = []
        for _ in gpus:
            M, N, K = sample_shape(cfg)
            batch_shapes.append((M, N, K))

        # Benchmark in parallel across GPUs
        futures = {}
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            for gpu_id, (M, N, K) in zip(gpus, batch_shapes):
                fut = executor.submit(benchmark_shape, bench_path, M, N, K, gpu_id, cfg)
                futures[fut] = (M, N, K, gpu_id)

            for fut in as_completed(futures):
                M, N, K, gpu_id = futures[fut]
                try:
                    solutions = fut.result()
                except Exception as e:
                    log.error("Bench failed for %dx%dx%d on GPU %d: %s", M, N, K, gpu_id, e)
                    continue

                if not solutions:
                    log.warning("No solutions for %dx%dx%d", M, N, K)
                    continue

                shape_counter += 1
                shape_id = shape_counter

                # Save raw benchmark CSV
                csv_path = bench_dir / f"{M}x{N}x{K}_{int(time.time())}.csv"
                save_bench_csv(solutions, csv_path, M, N, K)

                # Extract features
                entries = process_shape(hw, shape_id, M, N, K, solutions, cfg)
                if entries:
                    buffer.add_shape(entries)
                    shapes_since_eval += 1
                    log.info(
                        "Shape %d: %dx%dx%d -> %d/%d solutions with features (buffer: %d)",
                        shape_id, M, N, K, len(entries), len(solutions), len(buffer),
                    )

        # Train on buffer
        if len(buffer) > 0:
            steps = cfg['training']['train_steps_per_shape'] * len(gpus)
            for _ in range(steps):
                step_metrics = train_step(model, optimizer, buffer, cfg, device)
                total_train_steps += 1

                if total_train_steps % 100 == 0:
                    append_training_log(training_log_path, {
                        'step': total_train_steps,
                        'timestamp': datetime.datetime.now().isoformat(),
                        **step_metrics,
                    })

        # Evaluate periodically
        if shapes_since_eval >= cfg['eval_interval']:
            shapes_since_eval = 0
            buf_data = buffer.get_all()
            eval_metrics = evaluate_model(model, buf_data, device)
            log.info("EVAL [step %d, %d shapes]: %s",
                     total_train_steps, buffer.n_shapes, format_metrics(eval_metrics))

            action = 'eval'
            if eval_metrics['regret_mean'] < best_regret:
                best_regret = eval_metrics['regret_mean']
                save_checkpoint(model, optimizer, total_train_steps, eval_metrics,
                                ckpt_dir / 'best.pt')
                action = 'keep'
                log.info("  -> New best regret: %.4f%%, saved checkpoint", eval_metrics['regret_pct'])

            # Periodic checkpoint
            save_checkpoint(model, optimizer, total_train_steps, eval_metrics,
                            ckpt_dir / f'step_{total_train_steps}.pt')

            append_results_tsv(results_path, total_train_steps, eval_metrics, action)

            # Save dataset
            buffer.save_parquet(dataset_path)

    # Shutdown: final save
    log.info("Shutting down...")
    buffer.save_parquet(dataset_path)
    buf_data = buffer.get_all()
    if buf_data is not None:
        eval_metrics = evaluate_model(model, buf_data, device)
        save_checkpoint(model, optimizer, total_train_steps, eval_metrics, ckpt_dir / 'final.pt')
        append_results_tsv(results_path, total_train_steps, eval_metrics, 'shutdown')
        log.info("Final: %s", format_metrics(eval_metrics))

    log.info("Done. %d shapes, %d train steps.", buffer.n_shapes, total_train_steps)


def main():
    parser = argparse.ArgumentParser(description='Origami RL Trainer')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--bench', type=str, help='Path to hipblaslt-bench binary')
    parser.add_argument('--gpus', type=int, nargs='+', help='GPU IDs to use')
    parser.add_argument('--arch', type=str, help='GPU architecture (gfx942, gfx950)')
    parser.add_argument('--dtype', type=str, help='Data type (bf16_r, f16_r)')
    parser.add_argument('--transpose', type=str, help='Transpose mode (TN, NN, NT, TT)')
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    parser.add_argument('--eval-interval', type=int, help='Evaluate every N shapes')
    parser.add_argument('--validate', action='store_true', help='Run validation mode')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Override from CLI
    if args.bench:
        cfg['bench']['path'] = args.bench
    if args.gpus:
        cfg['gpus'] = args.gpus
    if args.arch:
        cfg['arch'] = args.arch
    if args.dtype:
        cfg['sampling']['dtype'] = args.dtype
    if args.transpose:
        cfg['sampling']['transA'] = args.transpose[0]
        cfg['sampling']['transB'] = args.transpose[1]
    if args.checkpoint_dir:
        cfg['checkpoint_dir'] = args.checkpoint_dir
    if args.eval_interval:
        cfg['eval_interval'] = args.eval_interval

    bench_path = cfg['bench']['path']
    gpus = cfg['gpus']

    if args.validate:
        ok = run_validation(cfg, bench_path, gpus)
        sys.exit(0 if ok else 1)
    else:
        run_training(cfg, bench_path, gpus)


if __name__ == '__main__':
    main()
