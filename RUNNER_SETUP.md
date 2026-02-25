# Self-Hosted CUDA Runner Setup

This document defines the required setup for GitHub Actions self-hosted runners used by:
`/.github/workflows/cuda-cross-platform.yml`

## Required Runner Labels
## Linux runner
1. `self-hosted`
2. `linux`
3. `x64`
4. `gpu`
5. `cuda`

## Windows runner
1. `self-hosted`
2. `windows`
3. `x64`
4. `gpu`
5. `cuda`

## Linux Runner Prerequisites
1. NVIDIA driver installed and GPU visible (`nvidia-smi` works).
2. CUDA toolkit installed and `nvcc` on `PATH`.
3. `cmake` available on `PATH`.
4. Python 3.10+ available as `python3`.
5. Build tools for CMake/Ninja available.
6. JAX + jaxlib with CUDA support installed for smoke:
   1. `jax==0.4.38`
   2. `jaxlib==0.4.38` (CUDA-enabled build)

Validation commands:
```bash
nvidia-smi
nvcc --version
cmake --version
python3 -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__); print(jax.devices())"
```

## Windows Runner Prerequisites
1. NVIDIA driver installed and GPU visible (`nvidia-smi` works).
2. CUDA toolkit installed and `nvcc` on `PATH`.
3. Minimum CUDA version: `11.8` (recommended `12.x`).
4. Visual Studio Build Tools with C++ toolset installed.
5. `cmake` available on `PATH`.
6. Python 3.10+ available as `python`.
7. Optional for full local GPU smoke only (not required for CI Windows path):
   1. `jax==0.4.38`
   2. `jaxlib==0.4.38` (CUDA-enabled build)

Validation commands (PowerShell):
```powershell
nvidia-smi
nvcc --version
cmake --version
```
Optional (full local GPU smoke readiness):
```powershell
python -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__); print(jax.devices())"
```

## Repository Runtime Validation
Run from repo root (`iir2d_op`):

Linux:
```bash
bash scripts/build_and_smoke_wsl.sh
```
Benchmark smoke (core C API):
```bash
python3 scripts/benchmark_core_cuda.py \
  --sizes 512x512 --filter_ids 1 --border_modes mirror --precisions f32 \
  --warmup 3 --iters 8 \
  --out_csv /tmp/iir2d_core_bench_smoke.csv
```

Windows:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_and_smoke_windows.ps1 -SkipGpuSmoke
```

Optional full local GPU smoke:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_and_smoke_windows.ps1
```

## CI Bring-Up Checklist
1. Register one Linux runner and one Windows runner with required labels.
2. Run validation commands on each host.
3. Trigger `cuda-cross-platform` workflow via `workflow_dispatch`.
4. Confirm both jobs pass.
5. Run core benchmark harness and archive CSV artifact.
6. Mark `ENG-006` done in `task_board.md`.

## Common Failures
1. `No CUDA toolset found`:
   1. CUDA toolkit missing or `nvcc` not on `PATH`.
2. `Unsupported CUDA toolkit version X.Y. Require CUDA >= 11.8 on Windows`:
   1. Upgrade CUDA toolkit to a supported version for current MSVC toolchain.
3. `No GPU device visible to JAX` (optional full Windows GPU smoke path):
   1. Driver issue or non-CUDA jaxlib installed.
4. Missing DLL/SO after build:
   1. Build failed earlier; inspect CMake build log first.
