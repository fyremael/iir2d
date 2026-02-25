# Troubleshooting Guide

## Diagnostic Checklist
1. Confirm toolchain:
   1. `cmake --version`
   2. `nvcc --version`
2. Confirm GPU visibility:
   1. `nvidia-smi`
3. Run contract smoke:
   1. `python scripts/smoke_core_status.py`

## Top Failure Modes
1. `nvcc not found`
   1. Cause: CUDA toolkit not installed or not on `PATH`.
   2. Fix: install CUDA and update `PATH`.
2. `Unsupported CUDA toolkit version` on Windows
   1. Cause: version below minimum floor.
   2. Fix: upgrade CUDA to supported version (`>=11.8`).
3. Missing `iir2d_jax` shared library
   1. Cause: build artifact not copied into `python/iir2d_jax`.
   2. Fix: rebuild and copy via platform smoke script.
4. Non-zero status from `iir2d_forward_cuda`
   1. Cause: invalid params/pointers or runtime CUDA errors.
   2. Fix: decode with `iir2d_status_string` and validate inputs.
5. CI run passes but CUDA jobs skipped
   1. Cause: hosted fallback mode active.
   2. Fix: set repo variable `IIR2D_USE_SELF_HOSTED=true` and ensure runners are online.
6. Benchmark result variance too high
   1. Cause: insufficient warmup/iters or shared GPU contention.
   2. Fix: increase `--warmup`, `--iters`, and rerun on dedicated runner.
7. Windows full GPU smoke shows CPU-only JAX
   1. Cause: non-CUDA jaxlib or host runtime mismatch.
   2. Fix: keep CI path on status-only smoke and validate GPU smoke on Linux gate.
8. Packaging artifacts missing expected files
   1. Cause: incomplete staging or copy errors.
   2. Fix: rerun packaging scripts and verify output manifest.
9. Regression alarm in nightly benchmark
   1. Cause: true performance drop or environment drift.
   2. Fix: compare metadata columns first (driver/runtime/GPU), then inspect kernel changes.
10. Ambiguous support tickets
   1. Cause: missing build/version metadata.
   2. Fix: include `iir2d_api_version_*` and `iir2d_build_fingerprint` in ticket template.
