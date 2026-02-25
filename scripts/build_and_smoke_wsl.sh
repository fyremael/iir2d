#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v cmake >/dev/null 2>&1 || { echo "cmake not found in PATH"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found in PATH"; exit 1; }
command -v nvcc >/dev/null 2>&1 || { echo "nvcc not found in PATH"; exit 1; }

cmake -S . -B build_wsl -DCMAKE_BUILD_TYPE=Release
cmake --build build_wsl -j

cp -f build_wsl/libiir2d_jax.so python/iir2d_jax/libiir2d_jax.so

PYTHONPATH="$ROOT/python" python3 "$ROOT/scripts/smoke_core_status.py"
PYTHONPATH="$ROOT/python" python3 "$ROOT/smoke_jax.py"
