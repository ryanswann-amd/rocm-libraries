#!/bin/bash
# Setup script for origami RL trainer
# Installs Python dependencies and builds hipblaslt-bench from source
#
# Prerequisites: ROCm 7.2+, PyTorch with ROCm, Python 3.12+
#
# Usage:
#   ./setup.sh                    # install deps only
#   ./setup.sh --build-bench      # also build hipblaslt-bench

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROCM_LIBS_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "==> Installing Python dependencies..."
pip install pyarrow pyyaml scipy pandas numpy torch 2>/dev/null || \
    pip install --user pyarrow pyyaml scipy pandas numpy torch

echo "==> Installing origami from source..."
cd "$ROCM_LIBS_ROOT/shared/origami/python"
pip install -e . 2>/dev/null || pip install --user -e .

if [[ "${1:-}" == "--build-bench" ]]; then
    echo "==> Building hipblaslt-bench from source..."
    echo "    This requires: cmake, gfortran (or patch to skip), msgpack-cxx"

    # Ensure cmake is available
    if ! command -v cmake &>/dev/null; then
        pip install cmake 2>/dev/null || pip install --user cmake
        export PATH="$(python3 -c 'import cmake; print(cmake.CMAKE_BIN_DIR)'):$PATH"
    fi

    # Build hipblaslt-bench
    ARCH="${GPU_ARCH:-gfx942}"
    BUILD_DIR="$ROCM_LIBS_ROOT/build-bench"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake "$ROCM_LIBS_ROOT/projects/hipblaslt" \
        -DCMAKE_PREFIX_PATH="/opt/rocm;/usr/local" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGPU_TARGETS="$ARCH" \
        -DBUILD_TESTING=OFF \
        -DHIPBLASLT_BUILD_TESTING=OFF \
        -DHIPBLASLT_ENABLE_ROCROLLER=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -Wno-dev

    cmake --build . --target hipblaslt-bench -j"$(nproc)"

    BENCH_PATH="$BUILD_DIR/clients/hipblaslt-bench"
    echo "==> hipblaslt-bench built at: $BENCH_PATH"
    echo "    Run trainer with: python3 train.py --bench $BENCH_PATH"
fi

echo "==> Setup complete."
echo "    Run validation: python3 train.py --validate --bench /path/to/hipblaslt-bench --gpus 0"
