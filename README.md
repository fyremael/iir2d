# IIR2D Native Hooks (PyTorch + JAX)

This folder provides native CUDA hooks for the IIR2D filters:
- **PyTorch**: C++/CUDA extension (`iir2d_torch_ext`)
- **JAX**: CUDA custom call (`iir2d_jax`)

The CUDA core uses a simple, per-row recursive implementation (correctness-first).

Usage guide:
1. `docs/USAGE_GUIDE.md` (annotated operational runbook with intuition pumps)

Commercialization docs:
1. `docs/PRODUCT_ONE_PAGER.md`
2. `docs/GETTING_STARTED_30MIN.md`
3. `docs/API_REFERENCE.md`
4. `docs/TROUBLESHOOTING.md`
5. `docs/COMPATIBILITY_MATRIX.md`
6. `docs/GTM_ICP_RUBRIC.md`
7. `docs/DESIGN_PARTNER_PILOT_TEMPLATE.md`
8. `docs/PRICING_AND_PACKAGING.md`
9. `docs/PILOT_TO_PAID_PLAYBOOK.md`
10. `docs/PACKAGING_LINUX.md`
11. `docs/PACKAGING_WINDOWS.md`
12. `docs/PILOT_WAVE1_EXECUTION.md`
13. `visual_showcase/index.html`

## Build (Linux)

### PyTorch
```bash
cd iir2d_op
python setup.py install
```
For packaging-only wheel builds (includes `iir2d_video` + `scripts`, skips CUDA extension build):
```bash
cd iir2d_op
python3 -m pip install --upgrade build
IIR2D_SKIP_EXT=1 python3 -m build --wheel --outdir dist
```

### JAX
```bash
cd iir2d_op
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```
Copy the resulting shared library next to `python/iir2d_jax/__init__.py`.

## Cross-Platform Build + Smoke
Use the platform scripts to keep Linux/WSL and Windows build paths healthy:

### Linux / WSL
```bash
cd iir2d_op
bash scripts/build_and_smoke_wsl.sh
```

### Windows (PowerShell)
```powershell
cd iir2d_op
powershell -ExecutionPolicy Bypass -File .\scripts\build_and_smoke_windows.ps1
```
Optional:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_and_smoke_windows.ps1 -SkipGpuSmoke
```

`scripts/smoke_core_status.py` validates stable C API status-code mappings without requiring a full model run.

Prerequisites:
1. `cmake`
2. CUDA toolkit with `nvcc` on `PATH`
3. Python (`python3` on Linux/WSL, `python` on Windows)

CI:
1. `.github/workflows/cuda-cross-platform.yml` enforces self-hosted CUDA mode on protected push paths (`IIR2D_USE_SELF_HOSTED=true`), with hosted contributor fallback when unavailable.
2. CI policy:
   1. Linux job runs full smoke (`scripts/build_and_smoke_wsl.sh`).
   2. Linux job runs CUDA-vs-CPU parity matrix over filter IDs `1..8`.
   3. Linux job runs multi-case GPU video E2E smoke (decode -> CUDA IIR2D -> encode), then computes PSNR/SSIM/temporal quality metrics.
   4. Windows job runs status-only smoke (`scripts/build_and_smoke_windows.ps1 -SkipGpuSmoke`).
   5. Hosted fallback runs lint + unit tests for contributor feedback loops.
3. Required runner labels:
   1. Linux: `self-hosted`, `linux`, `x64`, `gpu`, `cuda`
   2. Windows: `self-hosted`, `windows`, `x64`, `gpu`, `cuda`
4. Runner provisioning and validation checklist: `docs/RUNNER_SETUP.md`
5. Nightly full-matrix perf regression:
   1. `.github/workflows/nightly-perf-regression.yml`
   2. Runs CUDA-vs-CPU parity matrix over filter IDs `1..8` before benchmarking.
   3. Generates an all-8-filter benchmark matrix artifact (`/tmp/iir2d_core_bench_nightly_all8.csv`).
   4. Runs regression checks against full baseline `release_records/artifacts/benchmark_baselines/core_protocol_v2_all8.csv`.
   5. Uploads benchmark CSV and markdown trend report artifacts.
6. Python quality gates:
   1. `.github/workflows/quality-gates.yml` runs ruff lint + pytest coverage on core harness modules.
   2. Local run:
```bash
python3 -m pip install -r requirements-dev.txt
python3 scripts/check_asset_sizes.py --max_mb 25
python3 -m ruff check scripts/core_harness.py scripts/benchmark_core_cuda.py scripts/benchmark_video_cuda_pipeline.py scripts/video_quality_metrics.py scripts/build_video_report_pack.py scripts/iir2d_cpu_reference.py scripts/validate_cuda_cpu_matrix.py scripts/build_benchmark_claims_packet.py scripts/check_perf_regression.py scripts/check_perf_regression_matrix.py scripts/check_asset_sizes.py scripts/video_demo_cuda_pipeline.py iir2d_video tests
python3 -m pytest tests \
  --cov=scripts.core_harness \
  --cov=scripts.iir2d_cpu_reference \
  --cov=scripts.validate_cuda_cpu_matrix \
  --cov=scripts.build_benchmark_claims_packet \
  --cov=scripts.check_perf_regression \
  --cov=scripts.check_perf_regression_matrix \
  --cov=scripts.check_asset_sizes \
  --cov=scripts.benchmark_video_cuda_pipeline \
  --cov=scripts.video_demo_cuda_pipeline \
  --cov-report=term-missing \
  --cov-fail-under=85
```
7. Release artifact workflow:
   1. `.github/workflows/release-artifacts.yml`
   2. On `workflow_dispatch` or `v*` tags, builds a wheel (`IIR2D_SKIP_EXT=1`) and uploads a standardized video report-pack artifact.

## Core Benchmark Harness (Commercialization Baseline)
Use the C API benchmark harness to produce reproducible p50/p95 latency and throughput, with environment metadata attached to every row:

```bash
python3 scripts/benchmark_core_cuda.py \
  --sizes 512x512,1024x1024,2048x2048 \
  --filter_ids 1,2,3,4,5,6,7,8 \
  --border_modes mirror \
  --precisions f32,mixed \
  --warmup 10 \
  --iters 50 \
  --out_csv /tmp/iir2d_core_bench.csv
```

CSV columns include:
1. Case parameters: width/height/filter/border/precision.
2. Metrics: `latency_ms_p50`, `latency_ms_p95`, `throughput_mpix_per_s_p50`, `throughput_gb_per_s_p50`.
3. Environment metadata: host/platform/python/CUDA runtime + driver/GPU model/`nvcc` release/library path.

Claim publication protocol:
1. `docs/BENCHMARK_PROTOCOL.md`
2. Claims packet generator: `scripts/build_benchmark_claims_packet.py`

Build a publishable claims packet from a benchmark CSV:
```bash
python3 scripts/build_benchmark_claims_packet.py \
  --in_csv /tmp/iir2d_core_bench.csv \
  --out_md /tmp/iir2d_claims_packet.md \
  --benchmark_command "python3 scripts/benchmark_core_cuda.py --sizes 512x512,1024x1024,2048x2048 --filter_ids 1,2,3,4,5,6,7,8 --border_modes mirror --precisions f32,mixed --warmup 10 --iters 50 --out_csv /tmp/iir2d_core_bench.csv"
```

Nightly full-matrix trend check (same workload matrix):
```bash
python3 scripts/check_perf_regression_matrix.py \
  --current_csv /tmp/iir2d_core_bench.csv \
  --baseline_csv release_records/artifacts/benchmark_baselines/core_protocol_v2_all8.csv \
  --metric latency_ms_p50 \
  --direction lower_is_better \
  --max_regression_pct 25.0 \
  --out_report /tmp/iir2d_core_bench_trend_report.md
```

## CPU Parity Contract + Validator
Canonical CPU parity contract:
1. `docs/CPU_REFERENCE_DECISION.md`

Run CUDA-vs-CPU matrix validation:
```bash
python3 scripts/validate_cuda_cpu_matrix.py \
  --sizes 63x47 \
  --filter_ids 1,2,3,4,5,6,7,8 \
  --border_modes clamp,mirror,wrap,constant \
  --precisions f32,mixed,f64 \
  --border_const 0.125
```

Release governance:
1. `docs/RELEASE_GATE_POLICY.md`
2. `docs/RELEASE_CHECKLIST.md`
3. Executed records: `release_records/`

## Visual Showcase
Run a polished static demo page with side-by-side filter reveals and benchmark charts:

```bash
python3 -m http.server 8080
# open http://localhost:8080/visual_showcase/
```

Tracked showcase image assets are policy-gated at `<=25 MiB` per file (`scripts/check_asset_sizes.py`).
Committed sample video artifacts for showcase/demo live under `visual_showcase/assets/video_demo/`.

## Video Demo (CUDA C API)
Run decode -> CUDA IIR2D -> encode (default tuned for natural video: luma filtering + adaptive temporal blend):

```bash
python3 scripts/video_demo_cuda_pipeline.py \
  --in_video /path/to/input.mp4 \
  --out_video /path/to/output.mp4 \
  --filter_id 1 \
  --border_mode mirror \
  --precision f32 \
  --color_mode luma \
  --strength 0.65 \
  --temporal_mode adaptive \
  --codec libx264 \
  --preset medium \
  --crf 18
```

Notes:
1. Requires `ffmpeg` + `ffprobe` on `PATH`.
2. For strict fixed-mode temporal behavior, use `--temporal_mode fixed --temporal_ema_alpha 1.0` to disable temporal smoothing.
3. For smoke runs, set `--max_frames` (for example `--max_frames 120`).
4. For NVIDIA encoder output, set `--codec h264_nvenc`.

Benchmark decode -> CUDA IIR2D -> encode throughput:

```bash
python3 scripts/benchmark_video_cuda_pipeline.py \
  --in_video /path/to/input.mp4 \
  --out_csv /tmp/iir2d_video_bench.csv \
  --filter_id 1 \
  --border_mode mirror \
  --precision f32 \
  --color_mode luma \
  --strength 0.65 \
  --temporal_mode adaptive \
  --mode full \
  --codec libx264 \
  --encode_sink null \
  --warmup_frames 24 \
  --timed_frames 240
```

Key outputs in CSV:
1. Per-frame loop latency (`loop_ms_p50`, `loop_ms_p95`).
2. Stage means (`decode_ms_mean`, `process_ms_mean`, `encode_ms_mean`).
3. Throughput (`timed_fps`, `timed_mpix_per_s`).

Objective video quality metrics (PSNR/SSIM/temporal motion delta):

```bash
python3 scripts/video_quality_metrics.py \
  --reference_video /path/to/input.mp4 \
  --test_video /path/to/output.mp4 \
  --out_csv /tmp/iir2d_video_quality.csv \
  --min_psnr_mean 15.0 \
  --min_ssim_mean 0.35 \
  --max_temporal_delta_mean 0.08
```

Python API wrappers (typed configs):

```python
from iir2d_video import (
    VideoBenchmarkConfig,
    VideoProcessConfig,
    VideoQualityConfig,
    benchmark_video,
    evaluate_video_quality,
    process_video,
)

process_video(VideoProcessConfig(in_video="input.mp4", out_video="output.mp4"))
benchmark_video(VideoBenchmarkConfig(in_video="input.mp4", out_csv="bench.csv"))
evaluate_video_quality(VideoQualityConfig(reference_video="input.mp4", test_video="output.mp4", out_csv="quality.csv"))
```

Build a partner-ready markdown report pack from generated artifacts:

```bash
python3 scripts/build_video_report_pack.py \
  --benchmark_csvs /tmp/iir2d_video_bench.csv \
  --quality_csvs /tmp/iir2d_video_quality.csv \
  --clips visual_showcase/assets/video_demo/compare_portrait_original_f1_f4_f8.mp4,visual_showcase/assets/video_demo/output_portrait_zoom_iir2d_f1_natural.mp4 \
  --out_md /tmp/iir2d_video_report_pack.md
```

## Usage

### PyTorch
```python
import torch
from iir2d_torch import iir2d

x = torch.randn(512, 512, device="cuda", dtype=torch.float32)
y = iir2d(x, filter_id=4, border="mirror", precision="f32")
```

### JAX
```python
import jax
import jax.numpy as jnp
from iir2d_jax import iir2d

x = jnp.ones((512, 512), dtype=jnp.float32)
y = iir2d(x, filter_id=4, border="mirror", precision="f32")
```

## Notes
- `precision=f64` requires float64 inputs.
- JAX custom call runs on **GPU only** and expects the shared library to be present.
- The core path is correctness-first; performance can be improved by porting the scan-based kernels.
- JAX requires `jax.jit` on GPU (eager mode will raise).
- Autograd/VJP uses the same operator as an approximate adjoint.
- Batched tensors are supported by vmap (JAX) and CHW/NCHW loops (PyTorch).
- CUDA C API returns stable `IIR2D_Status` codes:
  - `0`: `IIR2D_STATUS_OK`
  - `-1`: `IIR2D_STATUS_INVALID_ARGUMENT`
  - `-2`: `IIR2D_STATUS_INVALID_DIMENSION`
  - `-3`: `IIR2D_STATUS_INVALID_FILTER_ID`
  - `-4`: `IIR2D_STATUS_INVALID_BORDER_MODE`
  - `-5`: `IIR2D_STATUS_INVALID_PRECISION`
  - `-6`: `IIR2D_STATUS_NULL_POINTER`
  - `-7`: `IIR2D_STATUS_WORKSPACE_ERROR`
  - `-8`: `IIR2D_STATUS_CUDA_ERROR`
  - use `iir2d_status_string(code)` for human-readable diagnostics.
- Public API version macros are in `csrc/iir2d_core.h`:
  - `IIR2D_API_VERSION_MAJOR`
  - `IIR2D_API_VERSION_MINOR`
  - `IIR2D_API_VERSION_PATCH`
- API/ABI compatibility policy is defined in `docs/ABI_POLICY.md`.

## ML Engineer Demo Pack (JAX)

### Optional deps (WSL/Linux)
```bash
python3 -m pip install --user "jax==0.4.38" "jaxlib==0.4.38" "optax"
python3 -m pip install --user "jax[cuda12]==0.4.38"
```
`scripts/ml_engineer_demo.py` will use Flax if import-compatible in your environment; otherwise it falls back to a pure-JAX model path automatically.

### 1) Throughput + quality benchmark (`iir2d` vs separable conv)
```bash
PYTHONPATH=/mnt/d/_codex/iir2d_op/python \
python3 /mnt/d/_codex/iir2d_op/scripts/benchmark_iir2d_vs_conv_jax.py \
  --batch 8 --height 1024 --width 1024 --channels 3 --iters 20 \
  --out_csv /tmp/iir2d_vs_conv.csv
```

### 2) Flax integration module
See:
- `python/iir2d_jax/flax_layer.py`

Includes:
- `IIR2DResidual`: learnable residual blend over one IIR filter.
- `IIR2DBank`: learnable mixture over multiple fixed filter IDs.
- `IIRDenoiseStem`: practical Conv -> IIR bank -> Conv residual block.

### 3) Notebook-style training demo
```bash
PYTHONPATH=/mnt/d/_codex/iir2d_op/python \
python3 /mnt/d/_codex/iir2d_op/scripts/ml_engineer_demo.py \
  --steps 60 --batch 8 --size 128 \
  --out_csv /tmp/iir2d_demo_metrics.csv \
  --out_history_csv /tmp/iir2d_demo_history.csv
```
For a full control comparison including backprop through IIR, add:
```bash
  --with_trainable_iir
```

This prints:
- runtime snapshot (GPU/JAX + iir latency),
- side-by-side tiny denoise training metrics,
- PSNR gains for baseline conv vs conv+iir-frozen (and optional trainable-iir control).

### 4) Plot the demo metrics
```bash
python3 /mnt/d/_codex/iir2d_op/scripts/plot_demo_metrics.py \
  --summary_csv /tmp/iir2d_demo_metrics.csv \
  --history_csv /tmp/iir2d_demo_history.csv \
  --out_dir /tmp/iir2d_plots
```
Generates:
- `psnr_gain_bar.png`
- `train_time_bar.png`
- `loss_curves.png` (when history CSV is provided)

### 5) Real-image dataset evaluation path
Run on a folder of real images (PNG/JPG/etc), with train/val split and noisy restoration task:
```bash
PYTHONPATH=/mnt/d/_codex/iir2d_op/python \
python3 /mnt/d/_codex/iir2d_op/scripts/evaluate_real_images.py \
  --data_dir /mnt/d/_codex/showcase \
  --steps 240 --batch 8 --patch 128 --noise_sigma 0.08 --channels 1 \
  --out_csv /tmp/iir2d_real_metrics.csv \
  --out_history_csv /tmp/iir2d_real_history.csv
```
Then plot:
```bash
python3 /mnt/d/_codex/iir2d_op/scripts/plot_demo_metrics.py \
  --summary_csv /tmp/iir2d_real_metrics.csv \
  --history_csv /tmp/iir2d_real_history.csv \
  --out_dir /tmp/iir2d_real_plots
```

### 6) Hugging Face dataset training run (non-toy)
Example with `food101` (50k train / 5k val subset):
```bash
python3 -m pip install --user datasets pyarrow

PYTHONPATH=/mnt/d/_codex/iir2d_op/python \
python3 /mnt/d/_codex/iir2d_op/scripts/evaluate_hf_dataset.py \
  --dataset food101 \
  --train_split train --val_split validation \
  --train_max_samples 50000 --val_max_samples 5000 \
  --steps 5000 --batch 8 --patch 128 --channels 3 --noise_sigma 0.08 \
  --out_csv /tmp/iir2d_hf_metrics.csv \
  --out_history_csv /tmp/iir2d_hf_history.csv

python3 /mnt/d/_codex/iir2d_op/scripts/plot_demo_metrics.py \
  --summary_csv /tmp/iir2d_hf_metrics.csv \
  --history_csv /tmp/iir2d_hf_history.csv \
  --out_dir /tmp/iir2d_hf_plots
```
