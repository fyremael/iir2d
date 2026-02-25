# IIR2D Release Gate Policy (REL-001)

## Purpose
Define objective release gates and decision rules for promoting builds from development to Release Candidate (RC) and then to General Availability (GA).

## Scope
This policy applies to the CUDA core SDK deliverable and its release artifacts.

## Release Stages
1. `dev`: Continuous integration outputs; not customer-promotable.
2. `rc`: Candidate build evaluated against release gates.
3. `ga`: Approved release for external distribution.

## Required Evidence Artifacts
1. CI run links/logs for required jobs.
2. Benchmark CSV artifact(s) from `scripts/benchmark_core_cuda.py`.
3. Compatibility and environment metadata used for validation.
4. Open issues list with severity and disposition.

## Gate Definitions
### Gate A: Technical Confidence
1. Linux full smoke passes (`scripts/build_and_smoke_wsl.sh`).
2. Windows status smoke passes (`scripts/build_and_smoke_windows.ps1 -SkipGpuSmoke`).
3. C API status-code contract validation passes (`scripts/smoke_core_status.py`).
4. No unresolved `P0` defects; any `P1` requires explicit waiver.

### Gate B: Operability and Reproducibility
1. CI workflow has two consecutive green runs on target branch.
2. Rebuild from clean workspace reproduces core build artifacts.
3. Benchmark smoke artifact is attached in CI (`linux-core-benchmark-smoke`).

### Gate C: Performance Evidence
1. Core benchmark harness (`scripts/benchmark_core_cuda.py`) executed on reference hardware matrix.
2. Report includes `latency_ms_p50`, `latency_ms_p95`, and throughput metrics.
3. Comparison baseline and command lines are published in release notes.

### Gate D: Commercial Readiness
1. Packaging/install docs are current for release scope.
2. Legal/security requirements for release stage are satisfied or formally waived.
3. Owner sign-offs are recorded in the RC checklist.

## Decision Rules
1. RC promotion requires Gates A-C pass and Gate D not blocked by `P0`.
2. GA promotion requires Gates A-D pass with no unresolved `P0`/`P1`.
3. Any waiver must include:
   1. scope,
   2. risk statement,
   3. owner,
   4. expiry date.

## Severity Policy
1. `P0`: release blocker, cannot promote.
2. `P1`: promotion blocked unless approved waiver exists.
3. `P2+`: may proceed with documented follow-up.

## Sign-off Roles
1. Core Kernel Engineer
2. Platform Engineer
3. QA Engineer
4. Product Lead

## Process
1. Start from `RELEASE_CHECKLIST.md` for each RC candidate and save records under `release_records/`.
2. Attach required evidence links/artifacts.
3. Collect sign-offs from required roles.
4. Record promotion decision and timestamp.
