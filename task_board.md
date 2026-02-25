# CUDA Core Commercialization Task Board

Status values: `in_progress`, `blocked`, `done`

## Priority 0 (Critical Foundation)
| ID | Task | Owner | Status | Acceptance Criteria |
|---|---|---|---|---|
| ENG-001 | Freeze C API/ABI contract and semantic versioning policy | Core Kernel Engineer | done | Versioned header + ABI policy doc merged |
| ENG-002 | Add strict input validation and error code table | Core Kernel Engineer | done | Invalid params produce deterministic non-zero error codes; doc published |
| ENG-003 | Implement CPU reference kernels for parity testing | QA Engineer | done | Canonical contract documented in `CPU_REFERENCE_DECISION.md` and implemented in `scripts/iir2d_cpu_reference.py` for filters `1..8` + border/precision semantics |
| ENG-004 | Create CUDA-vs-CPU correctness matrix tests | QA Engineer | done | `scripts/validate_cuda_cpu_matrix.py` validates filter/border/precision matrix and Linux self-hosted CI gate runs the parity matrix |
| ENG-005 | Build reproducible benchmark harness v1 | Platform Engineer | done | `scripts/benchmark_core_cuda.py` outputs p50/p95 latency + throughput with environment metadata CSV; Linux benchmark smoke wired into CI |
| REL-001 | Define release checklist and release gate policy | Product Lead | done | Policy/checklist established and applied to RC promotion (`release_records/RC_2026-02-25_RC1.md`); self-hosted CI evidence and delegated sign-off closure complete |

## Priority 1 (Production Readiness)
| ID | Task | Owner | Status | Acceptance Criteria |
|---|---|---|---|---|
| ENG-006 | CI pipeline for build + tests + benchmark smoke | Platform Engineer | done | Self-hosted CUDA runners registered (Linux + Windows), `IIR2D_USE_SELF_HOSTED=true`, and two consecutive self-hosted runs succeeded (`#4`, `#5`) with benchmark artifact evidence |
| ENG-007 | Nightly performance regression jobs on reference GPUs | Platform Engineer | done | Scheduled workflow `nightly-perf-regression.yml` archives nightly benchmark artifact and fails on regression threshold breach |
| ENG-008 | Linux binary packaging and install docs | Platform Engineer | done | Linux packaging/install guide published in `PACKAGING_LINUX.md` with one-command consume path |
| ENG-009 | Windows binary packaging and install docs | Platform Engineer | done | Windows packaging/install guide published in `PACKAGING_WINDOWS.md` with one-command consume path |
| ENG-010 | Compatibility matrix (CUDA/driver/GPU/OS) | Product Lead | done | Matrix published and versioned in `COMPATIBILITY_MATRIX.md` |
| SEC-001 | Add LICENSE + third-party NOTICES | Product Lead | done | `LICENSE` finalized as MIT and `NOTICE` updated with project license declaration + third-party notice posture |
| SEC-002 | Dependency/license scan integrated in CI | Platform Engineer | done | Workflow `dependency-license-scan.yml` added; CI fails on pip-audit vulnerabilities or license policy violations |
| SEC-003 | Define vulnerability response SLA/process | Product Lead | done | `SECURITY.md` policy merged with reporting path and response SLA |

## Priority 2 (Product and DX)
| ID | Task | Owner | Status | Acceptance Criteria |
|---|---|---|---|---|
| PRD-001 | Product one-pager (problem, scope, non-goals) | Product Lead | done | Published in `PRODUCT_ONE_PAGER.md` |
| PRD-002 | Getting-started guide (<30 min integration) | Product Lead | done | Published in `GETTING_STARTED_30MIN.md` |
| PRD-003 | API reference documentation | Core Kernel Engineer | done | Public symbols documented in `API_REFERENCE.md` |
| PRD-004 | Troubleshooting guide and diagnostics checklist | QA Engineer | done | Top failure modes documented in `TROUBLESHOOTING.md` |
| PRD-005 | Version query/build fingerprint runtime API | Core Kernel Engineer | done | Runtime API functions added in `csrc/iir2d_core.h/.cu` and validated by `scripts/smoke_core_status.py` |

## Priority 3 (GTM and Pilot Motion)
| ID | Task | Owner | Status | Acceptance Criteria |
|---|---|---|---|---|
| GTM-001 | Finalize ICP segments and qualification rubric | GTM Lead | done | Top ICPs and rubric documented in `GTM_ICP_RUBRIC.md` |
| GTM-002 | Publish benchmark protocol for external claims | GTM Lead | done | Protocol finalized in `BENCHMARK_PROTOCOL.md` with claims packet build workflow via `scripts/build_benchmark_claims_packet.py` |
| GTM-003 | Build design-partner pilot template | GTM Lead | done | Template published in `DESIGN_PARTNER_PILOT_TEMPLATE.md` |
| GTM-004 | Launch 3 design-partner pilots | GTM Lead | in_progress | Wave 1 execution packet prepared (`PILOT_WAVE1_EXECUTION.md`, `release_records/pilot_wave1/`); signed pilot agreements still pending counterparty acceptance |
| GTM-005 | Pricing and packaging decision doc | GTM Lead | done | Decision draft published in `PRICING_AND_PACKAGING.md` |
| GTM-006 | Pilot-to-paid conversion playbook | GTM Lead | done | Conversion playbook published in `PILOT_TO_PAID_PLAYBOOK.md` |

## Current Sprint (Next 2 Weeks)
| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| ENG-001 | API/ABI freeze | Core Kernel Engineer | done | Added version macros + ABI policy + changelog entry |
| ENG-002 | Validation + error model | Core Kernel Engineer | done | Strict validation + stable status codes implemented |
| ENG-003 | CPU reference baseline | QA Engineer | done | Decision doc + CPU reference implementation landed (`CPU_REFERENCE_DECISION.md`, `scripts/iir2d_cpu_reference.py`) |
| ENG-004 | CUDA-vs-CPU parity matrix | QA Engineer | done | Matrix validator landed and wired into Linux self-hosted CI (`scripts/validate_cuda_cpu_matrix.py`) |
| ENG-006 | CI build + smoke runner validation | Platform Engineer | done | Self-hosted runs `#4` and `#5` passed on Linux/Windows CUDA jobs; fallback jobs skipped by design under `IIR2D_USE_SELF_HOSTED=true` |
| ENG-005 | Benchmark harness v1 | Platform Engineer | done | Core C API harness implemented; sample evidence captured on 2026-02-25 (`/tmp/iir2d_core_bench_smoke.csv`) |
| REL-001 | Release gate checklist | Product Lead | done | RC1 promoted using checklist record; CI links, self-hosted evidence, and delegated role sign-offs are recorded |
| GTM-004 | Design-partner pilot launch (Wave 1) | GTM Lead | in_progress | Outreach templates and 3 pilot briefs prepared in `release_records/pilot_wave1/`; outbound + signed agreements pending |

## Risks and Blockers Log
| Date | Risk | Owner | Mitigation | Status |
|---|---|---|---|---|
| TBD | Performance gap vs tuned alternatives | Core Kernel Engineer | Focus on niche workloads + pipeline-level evidence | open |
| 2026-02-24 | Native Windows build blocked on local host: CUDA toolkit is v11.0 (below supported floor) | Platform Engineer | Installed CUDA 13.1 on 2026-02-25; build now uses nvcc 13.1 | mitigated |
| 2026-02-25 | Linux/WSL build+smoke validated locally (`scripts/build_and_smoke_wsl.sh` passed; JAX saw `CudaDevice(id=0)`) | Platform Engineer | Keep Linux runner labels/config aligned with `RUNNER_SETUP.md`; attach CI run evidence | mitigated |
| 2026-02-25 | Windows full smoke blocked: JAX runtime sees CPU only (`devices [CpuDevice(id=0)]`) on local host | Platform Engineer | Adopted CI policy to run Windows status-only smoke (`-SkipGpuSmoke`); keep Linux as full JAX GPU smoke gate | mitigated |
| 2026-02-25 | Baseline commercialization metrics were not reproducibly generated from core API | Platform Engineer | Added `scripts/benchmark_core_cuda.py` harness and CI Linux benchmark smoke artifact upload | mitigated |
| 2026-02-25 | RC promotion criteria were not formalized in repo | Product Lead | Added release gate policy/checklist and executed first formal RC pass record (`RC_2026-02-25_RC1`) | mitigated |
| 2026-02-25 | RC1 audit closeout pending remaining human sign-offs | Product Lead | Resolved: delegated Core/Platform/QA sign-offs recorded per Product Lead directive (2026-02-25T07:59:01Z) | mitigated |
| 2026-02-25 | No self-hosted CUDA runners registered in GitHub repo (`runner_count=0`) | Platform Engineer | Resolved: runners registered and runs `#4/#5` completed with self-hosted Linux/Windows jobs | mitigated |
| 2026-02-25 | CPU reference parity track is unresolved (`ENG-003`/`ENG-004`) | QA Engineer | Resolved: canonical contract + CPU reference + CI parity matrix validator landed | mitigated |
| 2026-02-25 | Design-partner pilot launch depends on external counterparties (`GTM-004`) | GTM Lead | Wave 1 launch kit prepared (`PILOT_WAVE1_EXECUTION.md`, `release_records/pilot_wave1/`); execute outreach and close 3 signed pilot plans | in_progress |
| TBD | Ambiguous quality claims | GTM Lead | Mitigated by finalized benchmark protocol + claims packet workflow and sign-off gates | mitigated |

## Definition of Ready (Task Intake)
1. Owner assigned.
2. Acceptance criteria are testable.
3. Dependencies listed.
4. Target milestone (M1/M2/M3/M4) assigned.

## Definition of Done (Task Completion)
1. Code/docs merged.
2. Validation evidence attached (test report, benchmark report, or review note).
3. Changelog updated where user-facing behavior changed.
4. No open critical defects introduced by the task.
