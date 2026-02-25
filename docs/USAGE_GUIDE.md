# IIR2D Usage Guide

This guide focuses on shipping the CUDA core with reproducible evidence.

## Intuition First
1. Think of the C API as the product, wrappers as adapters.
   Annotation: lock integration contracts to `csrc/iir2d_core.h`.
2. Think in release loops, not ad-hoc runs.
   Annotation: `Build -> Smoke -> Benchmark -> Evidence Link -> Sign-off`.
3. Think in reliability percentiles, not one-off timings.
   Annotation: use `p50` for typical experience and `p95` for risk budgeting.

## Quick Paths
### Linux/WSL Build + Smoke
```bash
cd /mnt/d/_codex/iir2d_op
bash scripts/build_and_smoke_wsl.sh
```
Annotation: this validates both status-contract smoke and JAX smoke on Linux.

### Windows Build + Status Smoke
```powershell
cd D:\_codex\iir2d_op
powershell -ExecutionPolicy Bypass -File .\scripts\build_and_smoke_windows.ps1 -SkipGpuSmoke
```
Annotation: `-SkipGpuSmoke` keeps Windows on deterministic status-contract validation.

## Core Benchmark Harness
```bash
cd /mnt/d/_codex/iir2d_op
python3 scripts/benchmark_core_cuda.py \
  --sizes 512x512,1024x1024 \
  --filter_ids 1,4,8 \
  --border_modes mirror \
  --precisions f32 \
  --warmup 10 \
  --iters 50 \
  --out_csv /tmp/iir2d_core_bench.csv
```
Annotation: benchmark rows include environment metadata, which is required for defensible performance claims.

### How to Read Output
1. `latency_ms_p50`: typical latency per call.
2. `latency_ms_p95`: tail latency under mild variance.
3. `throughput_mpix_per_s_p50`: image-scale throughput for planning capacity.
4. `throughput_gb_per_s_p50`: memory-traffic proxy for bottleneck intuition.

Intuition pump: if `p50` improves but `p95` regresses, you improved demos but hurt production predictability.

## CI Operating Modes
1. Self-hosted mode (`IIR2D_USE_SELF_HOSTED=true`):
   Annotation: real CUDA jobs run on Linux + Windows self-hosted runners.
2. Hosted fallback mode (default when variable is unset):
   Annotation: control-plane validation only; useful for continuity, not final performance evidence.

Runner details are in `RUNNER_SETUP.md`.

## C API Integration Notes
Reference: `csrc/iir2d_core.h`

1. Call `iir2d_forward_cuda(const void* in, void* out, const IIR2D_Params* params)`.
2. `in/out` must be device pointers to contiguous row-major `HxW` tensors.
3. `filter_id` must be `1..8`; unsupported values fail with stable status codes.
4. On failure, always map via `iir2d_status_string(code)`.

Intuition pump: treat status codes like SQLSTATE errors: program against them, do not parse ad-hoc logs.

## Evidence Hygiene (for commercialization)
For each reported benchmark or release claim, keep:
1. Exact command line.
2. Raw CSV artifact.
3. CI run URL(s).
4. Artifact URL.
5. Reviewer sign-off entry.

If any one is missing, treat the claim as draft.
