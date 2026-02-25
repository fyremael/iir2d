# IIR2D Core Benchmark Protocol (v1.2)

## Purpose
This protocol defines how to generate external-facing performance claims for the CUDA core API.
It is designed to be reproducible, auditable, and conservative.

## Scope
1. API under test: `iir2d_forward_cuda` from `csrc/iir2d_core.h`.
2. Precision modes: `f32`, `mixed`, `f64` (when supported by workload).
3. Filter IDs: `1..8`.
4. Border modes: `clamp`, `mirror`, `wrap`, `constant`.

## Reporting Rules
1. Always report `latency_ms_p50` and `latency_ms_p95`.
2. Always report throughput (`throughput_mpix_per_s_p50` and/or `throughput_gb_per_s_p50`).
3. Publish exact benchmark command line(s).
4. Publish environment metadata:
   1. GPU model
   2. Driver version
   3. CUDA runtime version
   4. CUDA toolkit (`nvcc`) release
   5. OS and Python version
5. Do not publish single-run numbers as final claims.
6. Do not compare against baselines with mismatched precision or workload shape.

## Standard Workload Matrix
1. Sizes:
   1. `512x512`
   2. `1024x1024`
   3. `2048x2048`
2. Filters:
   1. `1`
   2. `4`
   3. `8`
3. Border mode:
   1. `mirror`
4. Precision:
   1. `f32`
   2. `mixed`

## Benchmark Command
Run from repository root (`iir2d_op`):

```bash
python3 scripts/benchmark_core_cuda.py \
  --sizes 512x512,1024x1024,2048x2048 \
  --filter_ids 1,4,8 \
  --border_modes mirror \
  --precisions f32,mixed \
  --warmup 10 \
  --iters 50 \
  --out_csv /tmp/iir2d_core_benchmark_v1.csv
```

Windows (PowerShell):

```powershell
python scripts\benchmark_core_cuda.py `
  --sizes 512x512,1024x1024,2048x2048 `
  --filter_ids 1,4,8 `
  --border_modes mirror `
  --precisions f32,mixed `
  --warmup 10 `
  --iters 50 `
  --out_csv $env:TEMP\iir2d_core_benchmark_v1.csv
```

## Output Artifact Requirements
1. CSV artifact from benchmark harness.
2. Commit SHA / release tag used for run.
3. Timestamp in UTC.
4. Optional: raw logs from command execution.
5. Publishable claims packet in markdown form.
6. Trend report comparing against approved full-matrix baseline.

## Claims Packet Build Step
Use the packet builder to produce an externally shareable summary from a benchmark CSV:

```bash
python3 scripts/build_benchmark_claims_packet.py \
  --in_csv /tmp/iir2d_core_benchmark_v1.csv \
  --out_md /tmp/iir2d_claims_packet_v1.md \
  --benchmark_command "python3 scripts/benchmark_core_cuda.py --sizes 512x512,1024x1024,2048x2048 --filter_ids 1,4,8 --border_modes mirror --precisions f32,mixed --warmup 10 --iters 50 --out_csv /tmp/iir2d_core_benchmark_v1.csv"
```

Required output:
1. Environment metadata block.
2. Case-by-case results table.
3. Explicit claim hygiene checklist.
4. Approval placeholders for Product/GTM/QA.

## Baseline and Trend Comparison
Approved baseline artifact:
1. `release_records/artifacts/benchmark_baselines/core_protocol_v1.csv`

Trend comparison command:
```bash
python3 scripts/check_perf_regression_matrix.py \
  --current_csv /tmp/iir2d_core_benchmark_v1.csv \
  --baseline_csv release_records/artifacts/benchmark_baselines/core_protocol_v1.csv \
  --metric latency_ms_p50 \
  --direction lower_is_better \
  --max_regression_pct 25.0 \
  --out_report /tmp/iir2d_core_benchmark_v1_trend_report.md
```

Pass condition:
1. Worst-case per-scenario latency regression is <= `25.0%`.
2. Current and baseline matrices have identical case keys.

## CI Smoke Variant
For CI validation (not final claim generation), run a reduced sweep:
1. Size: `512x512`
2. Filter: `1`
3. Precision: `f32`
4. Warmup: `3`
5. Iterations: `8`

## Nightly Full-Matrix Automation
1. Workflow: `.github/workflows/nightly-perf-regression.yml`
2. Runs standard workload matrix nightly on self-hosted Linux CUDA runner.
3. Uploads both:
   1. full benchmark CSV
   2. trend report markdown

## Claim Hygiene Checklist
1. Precision parity between compared systems confirmed.
2. Same workload matrix used for all compared systems.
3. No cherry-picking of fastest case only.
4. Outlier handling disclosed (if applied).
5. Full CSV attached for review.

## Sign-off Workflow
1. QA verifies command line + artifact integrity.
2. Product Lead approves claim wording scope.
3. GTM Lead approves external messaging context.
4. Only signed packets may be used in external collateral.
