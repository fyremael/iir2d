#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v cmake >/dev/null 2>&1 || { echo "cmake not found in PATH"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found in PATH"; exit 1; }

cmake -S . -B build_macos -DCMAKE_BUILD_TYPE=Release -DIIR2D_BUILD_CPU_STUB=ON
cmake --build build_macos -j

cp -f build_macos/libiir2d_jax.dylib python/iir2d_jax/libiir2d_jax.dylib

PYTHONPATH="$ROOT/python" python3 "$ROOT/scripts/smoke_core_status.py"
