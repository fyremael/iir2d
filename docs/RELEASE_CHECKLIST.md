# IIR2D RC Promotion Checklist

Checklist owner: Product Lead
Policy reference: `RELEASE_GATE_POLICY.md`
Record location: `release_records/`

## Candidate Metadata
1. Candidate ID:
2. Target stage (`rc` or `ga`):
3. Commit SHA:
4. Build timestamp (UTC):
5. Evaluated by:

## Gate A: Technical Confidence
- [ ] Linux full smoke pass evidence attached.
- [ ] Windows status smoke pass evidence attached.
- [ ] C API status-contract smoke pass evidence attached.
- [ ] No unresolved `P0` defects.
- [ ] All `P1` defects resolved or waiver attached.

## Gate B: Operability and Reproducibility
- [ ] Two consecutive green CI runs on target branch.
- [ ] Clean rebuild reproducibility verified.
- [ ] Benchmark smoke artifact attached (`linux-core-benchmark-smoke`).

## Gate C: Performance Evidence
- [ ] Core benchmark run executed with `scripts/benchmark_core_cuda.py`.
- [ ] Benchmark CSV artifact attached.
- [ ] p50/p95 and throughput metrics reviewed.
- [ ] Command lines + environment metadata documented.

## Gate D: Commercial Readiness
- [ ] Packaging/install docs validated for this stage.
- [ ] Legal/security obligations for this stage reviewed.
- [ ] Open issue list triaged and accepted.

## Waivers (if any)
1. Waiver ID:
2. Scope:
3. Risk:
4. Owner:
5. Expiry:
6. Approval:

## Sign-off
| Role | Name | Decision (`approve`/`reject`) | Date (UTC) | Notes |
|---|---|---|---|---|
| Core Kernel Engineer |  |  |  |  |
| Platform Engineer |  |  |  |  |
| QA Engineer |  |  |  |  |
| Product Lead |  |  |  |  |

## Promotion Decision
1. Final decision (`promote`/`hold`):
2. Decision timestamp (UTC):
3. Decision owner:
4. Follow-ups (if hold):
