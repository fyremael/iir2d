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
20. Wave 1 pilot launch execution kit:
   1. `PILOT_WAVE1_EXECUTION.md`
   2. `release_records/pilot_wave1/OUTREACH_TEMPLATES.md`
   3. `release_records/pilot_wave1/PILOT_BRIEF_A_VISION_PLATFORM.md`
   4. `release_records/pilot_wave1/PILOT_BRIEF_B_ML_INFRA.md`
   5. `release_records/pilot_wave1/PILOT_BRIEF_C_REGULATED_IMAGING.md`
21. Python quality toolchain files:
   1. `requirements-dev.txt`
   2. `pyproject.toml`
22. Python unit tests:
   1. `tests/test_cpu_reference.py`
   2. `tests/test_claims_packet.py`
   3. `tests/test_perf_regression.py`
23. GitHub quality gate workflow: `.github/workflows/quality-gates.yml`.
24. Full-matrix regression comparator: `scripts/check_perf_regression_matrix.py`.
25. Approved full protocol benchmark baseline artifact: `release_records/artifacts/benchmark_baselines/core_protocol_v1.csv`.
26. Shared harness runtime module: `scripts/core_harness.py`.
27. Harness-focused unit tests:
   1. `tests/test_core_harness.py`
   2. `tests/test_validate_cuda_cpu_matrix.py`
28. Visual demo surface in `visual_showcase/index.html` with curated filter outputs and benchmark charts.
29. Asset size policy gate script and tests:
   1. `scripts/check_asset_sizes.py`
   2. `tests/test_check_asset_sizes.py`

### Changed
1. Documentation for benchmark execution and runner validation in `README.md` and `RUNNER_SETUP.md`.
2. RC1 record hard-close audit updates, including backfilled GitHub run/artifact links and CI qualification notes.
3. CI workflow includes hosted bootstrap fallback mode gated by repo variable `IIR2D_USE_SELF_HOSTED`.
4. RC1 CI evidence upgraded from hosted fallback to real self-hosted CUDA runs (`#4`, `#5`) on 2026-02-25 UTC.
5. RC1 closeout checklist and sign-off tables were finalized with delegated role approvals under Product Lead directive.
6. Core runtime API now exposes version and build fingerprint query functions, validated in `scripts/smoke_core_status.py`.
7. Linux self-hosted CI now enforces CUDA-vs-CPU parity matrix validation in `.github/workflows/cuda-cross-platform.yml`.
8. Benchmark protocol upgraded to v1.1 with claim packet generation and sign-off workflow in `BENCHMARK_PROTOCOL.md`.
9. Legal posture finalized to MIT license in `LICENSE`; `NOTICE` now declares project license explicitly.
10. `GTM-004` moved to in-progress with wave-launch artifacts tracked in `task_board.md`.
11. `README.md` now includes local lint + coverage commands for harness hardening.
12. Nightly performance workflow now runs full benchmark protocol matrix and emits trend report artifact using `scripts/check_perf_regression_matrix.py`.
13. Benchmark protocol upgraded to v1.2 with baseline + trend-comparison contract.
14. `scripts/benchmark_core_cuda.py` and `scripts/validate_cuda_cpu_matrix.py` now consume a shared harness contract from `scripts/core_harness.py`.
15. Quality gates now lint and enforce coverage on `scripts.core_harness` and `scripts.validate_cuda_cpu_matrix`.
16. `README.md` now includes a `Visual Showcase` launch path for live commercialization demos.
17. Repository markdown docs moved under `docs/` with `README.md` restored at repo root for GitHub landing-page rendering.
18. Quality gates now enforce tracked image asset size policy at `<=25 MiB` per file.
19. `visual_showcase/assets/` switched from PNG-heavy payloads to compressed WebP assets for lighter demo delivery.

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
