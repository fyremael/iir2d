# Getting Started (<30 Minutes)

## Goal
Build the CUDA core, run contract smoke, and produce one benchmark CSV.

## Linux/WSL Path
1. Build and smoke:
```bash
cd /mnt/d/_codex/iir2d_op
bash scripts/build_and_smoke_wsl.sh
```
2. Benchmark:
```bash
python3 scripts/benchmark_core_cuda.py \
  --sizes 512x512 \
  --filter_ids 1 \
  --border_modes mirror \
  --precisions f32 \
  --warmup 3 \
  --iters 8 \
  --out_csv /tmp/iir2d_quickstart.csv
```

## Windows Path (status-contract validation)
1. Build and smoke:
```powershell
cd D:\_codex\iir2d_op
powershell -ExecutionPolicy Bypass -File .\scripts\build_and_smoke_windows.ps1 -SkipGpuSmoke
```

## Expected Outputs
1. `PASS: iir2d_status_string contract` from `scripts/smoke_core_status.py`.
2. On Linux/WSL benchmark run: `Wrote benchmark CSV: ...`.
3. CSV contains `latency_ms_p50`, `latency_ms_p95`, and throughput fields.

## Common Fast Fixes
1. `nvcc not found`: add CUDA toolkit to `PATH`.
2. `cmake not found`: install CMake.
3. Version mismatch errors on Windows: ensure CUDA `>=11.8`.
