#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v cmake >/dev/null 2>&1 || { echo "cmake not found in PATH"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found in PATH"; exit 1; }

cmake_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DIIR2D_BUILD_CPU_STUB=ON
)
if [[ -n "${IIR2D_MACOS_ARCH:-}" ]]; then
  cmake_args+=("-DCMAKE_OSX_ARCHITECTURES=${IIR2D_MACOS_ARCH}")
fi

cmake -S . -B build_macos "${cmake_args[@]}"
cmake --build build_macos -j

cp -f build_macos/libiir2d_jax.dylib python/iir2d_jax/libiir2d_jax.dylib

if [[ "${IIR2D_SKIP_RUNTIME_SMOKE:-0}" == "1" ]]; then
  echo "Skipping runtime smoke for cross-compiled artifact."
else
  PYTHONPATH="$ROOT/python" python3 "$ROOT/scripts/smoke_core_status.py"
fi
