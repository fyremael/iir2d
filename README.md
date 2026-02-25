# IIR2D Native Hooks (PyTorch + JAX)

This folder provides native CUDA hooks for the IIR2D filters:
- **PyTorch**: C++/CUDA extension (`iir2d_torch_ext`)
- **JAX**: CUDA custom call (`iir2d_jax`)

The CUDA core uses a simple, per-row recursive implementation (correctness-first).

Usage guide:
1. `USAGE_GUIDE.md` (annotated operational runbook with intuition pumps)

## Build (Linux)

### PyTorch
```bash
cd iir2d_op
python setup.py install
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
1. `.github/workflows/cuda-cross-platform.yml` supports two modes:
   1. Self-hosted CUDA mode (`IIR2D_USE_SELF_HOSTED=true`): Linux + Windows CUDA jobs run.
   2. Hosted fallback mode (default): control-plane validation only; self-hosted CUDA jobs are skipped.
2. CI policy:
   1. Linux job runs full smoke (`scripts/build_and_smoke_wsl.sh`).
   2. Windows job runs status-only smoke (`scripts/build_and_smoke_windows.ps1 -SkipGpuSmoke`).
   3. Hosted fallback uploads a benchmark artifact from repository evidence path for audit continuity.
3. Required runner labels:
   1. Linux: `self-hosted`, `linux`, `x64`, `gpu`, `cuda`
   2. Windows: `self-hosted`, `windows`, `x64`, `gpu`, `cuda`
4. Runner provisioning and validation checklist: `RUNNER_SETUP.md`

## Core Benchmark Harness (Commercialization Baseline)
Use the C API benchmark harness to produce reproducible p50/p95 latency and throughput, with environment metadata attached to every row:

```bash
python3 scripts/benchmark_core_cuda.py \
  --sizes 1024x1024,2048x2048 \
  --filter_ids 1,4,8 \
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
1. `BENCHMARK_PROTOCOL.md`

Release governance:
1. `RELEASE_GATE_POLICY.md`
2. `RELEASE_CHECKLIST.md`
3. Executed records: `release_records/`

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
- API/ABI compatibility policy is defined in `ABI_POLICY.md`.

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
