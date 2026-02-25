# CUDA Core Commercialization Task Board

Status values: `todo`, `in_progress`, `blocked`, `done`

## Priority 0 (Critical Foundation)
| ID | Task | Owner | Status | Acceptance Criteria |
|---|---|---|---|---|
| ENG-001 | Freeze C API/ABI contract and semantic versioning policy | Core Kernel Engineer | done | Versioned header + ABI policy doc merged |
| ENG-002 | Add strict input validation and error code table | Core Kernel Engineer | done | Invalid params produce deterministic non-zero error codes; doc published |
| ENG-003 | Implement CPU reference kernels for parity testing | QA Engineer | todo | CPU ref covers all filter IDs and border modes |
| ENG-004 | Create CUDA-vs-CPU correctness matrix tests | QA Engineer | todo | Matrix runs in CI with documented tolerances |
| ENG-005 | Build reproducible benchmark harness v1 | Platform Engineer | done | `scripts/benchmark_core_cuda.py` outputs p50/p95 latency + throughput with environment metadata CSV; Linux benchmark smoke wired into CI |
| REL-001 | Define release checklist and release gate policy | Product Lead | done | Policy/checklist established and applied to RC promotion (`release_records/RC_2026-02-25_RC1.md`); CI link backfill complete, remaining signatures + self-hosted evidence hardening tracked |

## Priority 1 (Production Readiness)
| ID | Task | Owner | Status | Acceptance Criteria |
|---|---|---|---|---|
| ENG-006 | CI pipeline for build + tests + benchmark smoke | Platform Engineer | in_progress | GitHub Actions run evidence exists (runs `#2`, `#3`) via hosted fallback mode; self-hosted CUDA runner registration + `IIR2D_USE_SELF_HOSTED=true` required for full gate completion |
| ENG-007 | Nightly performance regression jobs on reference GPUs | Platform Engineer | todo | Nightly report archived; alert on threshold breach |
| ENG-008 | Linux binary packaging and install docs | Platform Engineer | todo | One-command consume path validated |
| ENG-009 | Windows binary packaging and install docs | Platform Engineer | todo | One-command consume path validated |
| ENG-010 | Compatibility matrix (CUDA/driver/GPU/OS) | Product Lead | todo | Matrix published and versioned per release |
| SEC-001 | Add LICENSE + third-party NOTICES | Product Lead | in_progress | `LICENSE` + `NOTICE` drafts added; legal review/final notices for distributed artifacts pending |
| SEC-002 | Dependency/license scan integrated in CI | Platform Engineer | todo | CI fails on policy violations |
| SEC-003 | Define vulnerability response SLA/process | Product Lead | done | `SECURITY.md` policy merged with reporting path and response SLA |

## Priority 2 (Product and DX)
| ID | Task | Owner | Status | Acceptance Criteria |
|---|---|---|---|---|
| PRD-001 | Product one-pager (problem, scope, non-goals) | Product Lead | todo | Approved narrative for internal/external use |
| PRD-002 | Getting-started guide (<30 min integration) | Product Lead | todo | New engineer demo validated by dry run |
| PRD-003 | API reference documentation | Core Kernel Engineer | todo | All public symbols documented |
| PRD-004 | Troubleshooting guide and diagnostics checklist | QA Engineer | todo | Top 10 expected failures documented with fixes |
| PRD-005 | Version query/build fingerprint runtime API | Core Kernel Engineer | todo | Runtime returns version/build metadata |

## Priority 3 (GTM and Pilot Motion)
| ID | Task | Owner | Status | Acceptance Criteria |
|---|---|---|---|---|
| GTM-001 | Finalize ICP segments and qualification rubric | GTM Lead | todo | Top 2 ICPs documented with qualification checklist |
| GTM-002 | Publish benchmark protocol for external claims | GTM Lead | in_progress | Protocol draft added in `BENCHMARK_PROTOCOL.md`; pending GTM sign-off |
| GTM-003 | Build design-partner pilot template | GTM Lead | todo | Template includes baseline, success criteria, timeline |
| GTM-004 | Launch 3 design-partner pilots | GTM Lead | todo | 3 active pilots with signed success criteria |
| GTM-005 | Pricing and packaging decision doc | GTM Lead | todo | Approved pricing sheet + eval/production terms |
| GTM-006 | Pilot-to-paid conversion playbook | GTM Lead | todo | Defined conversion criteria and decision points |

## Current Sprint (Next 2 Weeks)
| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| ENG-001 | API/ABI freeze | Core Kernel Engineer | done | Added version macros + ABI policy + changelog entry |
| ENG-002 | Validation + error model | Core Kernel Engineer | done | Strict validation + stable status codes implemented |
| ENG-003 | CPU reference baseline | QA Engineer | todo | Enables matrix tests |
| ENG-006 | CI build + smoke runner validation | Platform Engineer | in_progress | CI runs `#2` and `#3` are green via hosted fallback; self-hosted GPU runners and repo variable enablement are pending for full CUDA job execution |
| ENG-005 | Benchmark harness v1 | Platform Engineer | done | Core C API harness implemented; sample evidence captured on 2026-02-25 (`/tmp/iir2d_core_bench_smoke.csv`) |
| REL-001 | Release gate checklist | Product Lead | done | RC1 promoted using checklist record; CI links backfilled; remaining audit items are role signatures and self-hosted CUDA evidence hardening |

## Risks and Blockers Log
| Date | Risk | Owner | Mitigation | Status |
|---|---|---|---|---|
| TBD | Performance gap vs tuned alternatives | Core Kernel Engineer | Focus on niche workloads + pipeline-level evidence | open |
| 2026-02-24 | Native Windows build blocked on local host: CUDA toolkit is v11.0 (below supported floor) | Platform Engineer | Installed CUDA 13.1 on 2026-02-25; build now uses nvcc 13.1 | mitigated |
| 2026-02-25 | Linux/WSL build+smoke validated locally (`scripts/build_and_smoke_wsl.sh` passed; JAX saw `CudaDevice(id=0)`) | Platform Engineer | Keep Linux runner labels/config aligned with `RUNNER_SETUP.md`; attach CI run evidence | mitigated |
| 2026-02-25 | Windows full smoke blocked: JAX runtime sees CPU only (`devices [CpuDevice(id=0)]`) on local host | Platform Engineer | Adopted CI policy to run Windows status-only smoke (`-SkipGpuSmoke`); keep Linux as full JAX GPU smoke gate | mitigated |
| 2026-02-25 | Baseline commercialization metrics were not reproducibly generated from core API | Platform Engineer | Added `scripts/benchmark_core_cuda.py` harness and CI Linux benchmark smoke artifact upload | mitigated |
| 2026-02-25 | RC promotion criteria were not formalized in repo | Product Lead | Added release gate policy/checklist and executed first formal RC pass record (`RC_2026-02-25_RC1`) | mitigated |
| 2026-02-25 | RC1 audit closeout pending remaining human sign-offs | Product Lead | CI links/artifacts are now backfilled; collect Core/Platform/QA signatures for completeness | in_progress |
| 2026-02-25 | No self-hosted CUDA runners registered in GitHub repo (`runner_count=0`) | Platform Engineer | Register Linux/Windows runners, set `IIR2D_USE_SELF_HOSTED=true`, then re-run two consecutive CI passes with active self-hosted jobs | open |
| TBD | Ambiguous quality claims | GTM Lead | Protocol drafted (`BENCHMARK_PROTOCOL.md`) and core harness artifacts available; pending GTM sign-off | in_progress |

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
