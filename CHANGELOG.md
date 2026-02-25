# Changelog

## [Unreleased]
### Added
1. Core CUDA benchmark harness: `scripts/benchmark_core_cuda.py`.
2. Reproducible CSV outputs with p50/p95 latency, throughput, and environment metadata.
3. Linux CI benchmark smoke step with artifact upload in `.github/workflows/cuda-cross-platform.yml`.
4. External-claims benchmark protocol draft in `BENCHMARK_PROTOCOL.md`.
5. Release gate policy and RC promotion checklist templates in `RELEASE_GATE_POLICY.md` and `RELEASE_CHECKLIST.md`.
6. Initial RC prep checklist record in `release_records/RC_2026-02-25_PREP.md`.
7. First formal RC pass record in `release_records/RC_2026-02-25_RC1.md`.
8. RC1 evidence packet notes in `release_records/evidence/RC_2026-02-25_RC1/`.
9. RC1 closeout execution docs: `release_records/RC_2026-02-25_RC1_SIGNOFF_REQUESTS.md` and `release_records/RC_2026-02-25_RC1_CLOSEOUT_CHECKLIST.md`.
10. RC1 promotion decision recorded in `release_records/RC_2026-02-25_RC1.md`.
11. Initial legal/security scaffolding: `LICENSE`, `NOTICE`, and `SECURITY.md`.
12. Annotated usage runbook with intuition pumps in `USAGE_GUIDE.md`.

### Changed
1. Documentation for benchmark execution and runner validation in `README.md` and `RUNNER_SETUP.md`.
2. RC1 record hard-close audit updates, including backfilled GitHub run/artifact links and CI qualification notes.
3. CI workflow includes hosted bootstrap fallback mode gated by repo variable `IIR2D_USE_SELF_HOSTED`.
4. RC1 CI evidence upgraded from hosted fallback to real self-hosted CUDA runs (`#4`, `#5`) on 2026-02-25 UTC.
5. RC1 closeout checklist and sign-off tables were finalized with delegated role approvals under Product Lead directive.

## [1.0.0] - 2026-02-24
### Added
1. Stable public status code contract in `IIR2D_Status`.
2. `iir2d_status_string(int)` for human-readable diagnostics.
3. Strict public-entrypoint parameter validation in CUDA core.
4. Public API version macros:
   1. `IIR2D_API_VERSION_MAJOR`
   2. `IIR2D_API_VERSION_MINOR`
   3. `IIR2D_API_VERSION_PATCH`
5. API/ABI governance document in `ABI_POLICY.md`.

### Changed
1. Public API returns negative status codes for validation/runtime failures.
2. Torch binding now surfaces explicit status text on failure.
