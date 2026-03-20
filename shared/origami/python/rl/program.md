# Origami RL Trainer — Agent Instructions

## What This Is

Continuous training loop that learns a latency correction model on top of
origami's analytical GEMM predictions. Uses hipblaslt-bench as ground truth.

## How to Run

### Validation (quick sanity check, ~3 min)
```bash
cd shared/origami/python/rl
python3 train.py --validate --bench /path/to/hipblaslt-bench --gpus 0
```

### Full 24h Training
```bash
cd shared/origami/python/rl
python3 train.py \
  --bench /path/to/hipblaslt-bench \
  --gpus 0 1 2 3 \
  --arch gfx942 \
  --dtype bf16_r \
  --checkpoint-dir ./checkpoints \
  --eval-interval 50
```

## Monitoring

- **results.tsv**: Experiment log (step, MAE, regret, top-1, timestamp)
- **stdout**: Real-time progress
- **checkpoints/**: Model weights (best + periodic)

## Key Metrics

- **MAE**: Mean absolute error on log(actual/origami) prediction
- **Regret**: % gflops lost vs oracle selection (lower = better)
- **Top-1**: Fraction of shapes where we pick the actual best kernel

## Baseline (GBR reranker on 16 shapes)
- Top-1: 43.8%, Top-5: 81.2%, Regret: 3.08%
- Target: beat this with more data + neural net

## Architecture

Physics-informed residual MLP: predicts correction factor on origami's
analytical latency. 92 features per (problem, kernel) pair.
See train.py docstring for full details.
