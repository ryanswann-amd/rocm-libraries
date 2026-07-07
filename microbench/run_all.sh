#!/bin/bash
# Run all microbenchmarks inside the container.
# Usage: docker exec <container> bash /path/to/run_all.sh

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "Microbenchmark 1: P2P bandwidth vs WG count"
echo "============================================"
python3 microbench/p2p_bw_vs_wgs.py 2>&1 | tee microbench/output_p2p_bw.txt

echo ""
echo "============================================"
echo "Microbenchmark 3+4: Collective overhead"
echo "============================================"
# W=2 (2 GPUs)
torchrun --nproc_per_node=2 --master_port=29500 \
    microbench/collective_overhead.py 2>&1 | tee microbench/output_overhead_w2.txt

echo ""
# W=8 (8 GPUs)
torchrun --nproc_per_node=8 --master_port=29501 \
    microbench/collective_overhead.py 2>&1 | tee microbench/output_overhead_w8.txt

echo ""
echo "Done. Results in microbench/output_*.txt"
