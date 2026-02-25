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
13. Commercialization collateral docs:
   1. `PRODUCT_ONE_PAGER.md`
   2. `GETTING_STARTED_30MIN.md`
   3. `API_REFERENCE.md`
   4. `TROUBLESHOOTING.md`
   5. `COMPATIBILITY_MATRIX.md`
   6. `GTM_ICP_RUBRIC.md`
   7. `DESIGN_PARTNER_PILOT_TEMPLATE.md`
   8. `PRICING_AND_PACKAGING.md`
   9. `PILOT_TO_PAID_PLAYBOOK.md`
   10. `PACKAGING_LINUX.md`
   11. `PACKAGING_WINDOWS.md`
14. Security/dependency policy tooling:
   1. `.github/workflows/dependency-license-scan.yml`
   2. `scripts/check_license_policy.py`
   3. `requirements-security.txt`
15. Nightly performance regression automation:
   1. `.github/workflows/nightly-perf-regression.yml`
   2. `scripts/check_perf_regression.py`
16. Canonical CPU reference contract decision in `CPU_REFERENCE_DECISION.md`.
17. Canonical CPU reference implementation for filters `1..8` in `scripts/iir2d_cpu_reference.py`.
18. CUDA-vs-CPU matrix validator in `scripts/validate_cuda_cpu_matrix.py`.
19. Benchmark claims packet builder in `scripts/build_benchmark_claims_packet.py`.

### Changed
1. Documentation for benchmark execution and runner validation in `README.md` and `RUNNER_SETUP.md`.
2. RC1 record hard-close audit updates, including backfilled GitHub run/artifact links and CI qualification notes.
3. CI workflow includes hosted bootstrap fallback mode gated by repo variable `IIR2D_USE_SELF_HOSTED`.
4. RC1 CI evidence upgraded from hosted fallback to real self-hosted CUDA runs (`#4`, `#5`) on 2026-02-25 UTC.
5. RC1 closeout checklist and sign-off tables were finalized with delegated role approvals under Product Lead directive.
6. Core runtime API now exposes version and build fingerprint query functions, validated in `scripts/smoke_core_status.py`.
7. Linux self-hosted CI now enforces CUDA-vs-CPU parity matrix validation in `.github/workflows/cuda-cross-platform.yml`.
8. Benchmark protocol upgraded to v1.1 with claim packet generation and sign-off workflow in `BENCHMARK_PROTOCOL.md`.

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
