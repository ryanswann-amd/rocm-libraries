#!/bin/bash
# Fresh collective sweep using rccl-tests (gold standard benchmarks).
# Run inside container on MI300X 8-GPU node.
set -e

OUTDIR=/home/ryaswann/mc2-workspaces/origami-comms-bench/results
mkdir -p $OUTDIR

# Install rccl-tests if not present
if ! command -v /opt/rocm/bin/all_reduce_perf &>/dev/null; then
    echo "Installing rccl-tests..."
    cd /tmp
    git clone https://github.com/ROCm/rccl-tests.git 2>/dev/null || true
    cd rccl-tests
    make MPI=0 HIP_HOME=/opt/rocm RCCL_HOME=/opt/rocm 2>&1 | tail -3
    export PATH=/tmp/rccl-tests/build:$PATH
fi

# Find the perf binaries
PERF_DIR=""
for d in /tmp/rccl-tests/build /opt/rocm/bin /usr/local/bin; do
    if [ -f "$d/all_reduce_perf" ]; then
        PERF_DIR=$d
        break
    fi
done

if [ -z "$PERF_DIR" ]; then
    echo "rccl-tests not found, falling back to torch.distributed"
    python3 /home/ryaswann/mc2-workspaces/origami-comms-bench/microbench/sweep_torch.py
    exit 0
fi

echo "Using rccl-tests from $PERF_DIR"
echo "GPUs: $(rocm-smi --showid 2>/dev/null | grep -c GPU || echo 8)"

# Sweep parameters
NGPUS=8
WARMUP=20
ITERS=100
# Message sizes: 1K to 1G in powers of 4
SIZES="-b 1K -e 1G -f 4"

for COLL in all_reduce all_gather reduce_scatter alltoall broadcast; do
    PERF=$PERF_DIR/${COLL}_perf
    if [ ! -f "$PERF" ]; then
        echo "Skipping $COLL (binary not found)"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  $COLL  (8 GPUs, $ITERS iters)"
    echo "=========================================="

    # Default channels (auto)
    echo "--- NCH=auto ---"
    $PERF -g $NGPUS $SIZES -w $WARMUP -n $ITERS -d bfloat16 2>&1 | tee $OUTDIR/${COLL}_auto.txt

    # Explicit channel sweeps
    for NCH in 1 4 8 16 32 64; do
        echo "--- NCH=$NCH ---"
        NCCL_MIN_NCHANNELS=$NCH NCCL_MAX_NCHANNELS=$NCH \
            $PERF -g $NGPUS $SIZES -w $WARMUP -n $ITERS -d bfloat16 2>&1 | tee $OUTDIR/${COLL}_nch${NCH}.txt
    done
done

echo ""
echo "Results saved to $OUTDIR/"
ls -la $OUTDIR/
