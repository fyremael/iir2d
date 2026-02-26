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

## Extended Appendix: Filter Teaching Notes

### Filter 1 (`F1 EMA`)
- Qualities: first-order exponential smoother (`alpha=0.85`), low compute, predictable blur profile.
- Advantages: very stable, fast, easy to tune mentally, good default for noise suppression.
- Drawbacks: can look soft on fine detail; directional pass structure can leave mild anisotropy in edge-heavy scenes.
- Practical uses: baseline denoise pass, pre-smoothing before segmentation/thresholding, real-time preview pipelines.

### Filter 2 (`F2 SOS`)
- Qualities: cascaded first-order sections (`a=0.75`, `b=0.25`) for stronger smoothing than F1 with controlled response.
- Advantages: cleaner suppression of high-frequency noise than F1, still robust and easy to deploy.
- Drawbacks: more blur than F1 at default settings; can flatten texture if overused.
- Practical uses: artifact cleanup for noisy sources, stable "reference look" for demos, temporal preconditioning in video workflows.

### Filter 3 (`F3 Biquad`)
- Qualities: second-order recursive form (biquad-scan contract) with richer response shape than first-order filters.
- Advantages: can preserve macro-structure while reducing fine noise; useful when F1/F2 are too plain.
- Drawbacks: more sensitive to boundary and block-scan implementation details; historically most likely to reveal block artifacts when contract math is wrong.
- Practical uses: controlled stylization, structural smoothing where second-order response is desired, advanced experimentation.

### Filter 4 (`F4 SOS`)
- Qualities: cascaded biquad sections for stronger second-order shaping than F3.
- Advantages: aggressive smoothing/stylization with limited kernel footprint and good runtime throughput.
- Drawbacks: highest risk of over-smoothing and artifact visibility in challenging boundary conditions; needs careful QA on representative media.
- Practical uses: heavy denoise/style looks, low-detail background cleanup, creative/post-processing variants.

### Filter 5 (`F5 FB First`)
- Qualities: forward-backward first-order pass (`zero-phase-like` behavior) reducing directional lag.
- Advantages: better symmetry than one-way filters; often cleaner on edges where phase lag is noticeable.
- Drawbacks: double pass cost; can still soften detail.
- Practical uses: edge-aware smoothing where directional phase bias is undesirable, pre-processing for downstream measurement tasks.

### Filter 6 (`F6 Deriche-ish`)
- Qualities: recursive Deriche-inspired formulation with efficient large-scale smoothing behavior.
- Advantages: strong smoothing with favorable compute characteristics versus large spatial kernels.
- Drawbacks: parameter intuition is less obvious than F1/F2; not ideal as a first "safe default" for new users.
- Practical uses: large-radius-like smoothing intent at low cost, pipelines that need high throughput on big frames.

### Filter 7 (`F7 Sharper EMA`)
- Qualities: contract-compatible first-order EMA variant intended for a crisper profile family.
- Advantages: simple drop-in option when teams want a separate API slot for EMA-family tuning/experiments.
- Drawbacks: current stable contract coefficients align with F1; expected output is effectively the same today.
- Practical uses: compatibility placeholder for product tiers/presets, future-proofing around stable `1..8` API IDs.

### Filter 8 (`F8 State`)
- Qualities: state-space slot in the stable API, currently mapped to the same shipped biquad-scan contract behavior as F3.
- Advantages: preserves contract space for a distinct state-space evolution without breaking public IDs.
- Drawbacks: current output equivalence with F3 can confuse users expecting a visibly distinct look.
- Practical uses: compatibility and migration planning, controlled experimentation behind a stable public identifier.

### Selection Heuristic (Practical)
- Start with F2 for default production smoothing.
- Use F1 when you need maximum simplicity and speed.
- Use F5 when phase symmetry matters.
- Use F6 for strong smoothing at large image sizes.
- Use F3/F4/F8 only with explicit visual QA on your target media.
