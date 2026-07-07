#!/bin/bash
# Run the tensor-shape collective sweep across world sizes.
#
# Produces / appends to data/tensor_shapes_sweep.csv. Safe to re-run; the
# script appends and dashboards dedup on (timestamp, primitive, shape, ...).
#
# Usage:
#   bash microbench/run_torch_shapes.sh           # W=2,4,8 default
#   bash microbench/run_torch_shapes.sh 8         # only W=8
#   bash microbench/run_torch_shapes.sh "2 8"     # W=2 and 8

set -e
cd "$(dirname "$0")/.."

WORLD_SIZES=${1:-"2 4 8"}
OUT=data/tensor_shapes_sweep.csv
mkdir -p "$(dirname "$OUT")"

NGPUS_AVAIL=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "# Detected $NGPUS_AVAIL GPUs."

for W in $WORLD_SIZES; do
    if [ "$W" -gt "$NGPUS_AVAIL" ]; then
        echo "# Skipping W=$W (only $NGPUS_AVAIL GPUs available)."
        continue
    fi
    echo ""
    echo "=========================================="
    echo "  Tensor shape sweep  W=$W"
    echo "=========================================="
    PORT=$((29500 + W))
    torchrun --nproc_per_node=$W --master_port=$PORT \
        microbench/sweep_torch_shapes.py \
        --out "$OUT" \
        --warmup 10 --iters 30 \
        --max-per-rank-mb 128
done

echo ""
echo "# Wrote: $OUT"
python3 -c "
import pandas as pd
df = pd.read_csv('$OUT')
print(f'Total rows: {len(df):,}')
print(df.groupby(['primitive','world_size']).size().unstack(fill_value=0))
"
