# IIR2D Product One-Pager

## Problem
Teams with CUDA pipelines need a predictable, low-latency 2D recursive filter primitive that can be embedded in existing inference/training code without rewriting their stack.

## Core Offering
1. Stable C API (`csrc/iir2d_core.h`) for CUDA device-pointer execution.
2. Deterministic status/error contract for operational safety.
3. Reproducible benchmark harness and CI evidence chain for performance claims.

## Scope
1. CUDA kernel execution path for filter IDs `1..8`.
2. Border modes: `clamp`, `mirror`, `wrap`, `constant`.
3. Precision modes: `f32`, `mixed`, `f64`.
4. Self-hosted CI release gating with benchmark artifacts.

## Non-Goals
1. CPU production runtime replacement.
2. Turnkey managed service.
3. Broad model-zoo integration in RC scope.

## Buyer Value
1. Faster integration via ABI-stable native entry points.
2. Lower operational risk via explicit status codes and smoke tests.
3. Commercially defensible performance evidence (run links + CSV artifacts).

## Success Criteria (RC -> GA)
1. Pilot teams integrate in less than one week.
2. Two consecutive self-hosted CI passes remain green per release.
3. Benchmark protocol runs are reproducible by external reviewers.
