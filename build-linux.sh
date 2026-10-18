#!/usr/bin/env bash
set -euo pipefail
cargo build --release --locked
if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA Toolkit/nvcc not found: CPU CLI built; GPU requires CUDA >= 12.8." >&2
  exit 0
fi
nvcc -O3 -std=c++17 -arch=sm_120 cuda/uuid_gpu.cu -o target/release/uuid-gpu
echo "Built target/release/uuid-v7-bench and target/release/uuid-gpu"