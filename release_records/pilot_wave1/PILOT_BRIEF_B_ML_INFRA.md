# Pilot Brief B - Applied ML Infra Team

Date Opened: 2026-02-25  
Status: outreach_ready

## ICP Qualification (0-2 each)
1. CUDA maturity: 1
2. Pain severity: 2
3. Integration readiness: 2
4. Commercial urgency: 1
Total: 6 (qualified)

## Pain Hypothesis
The team needs deterministic preprocessing/postprocessing in GPU pipelines and requires artifact-backed performance claims before internal adoption.

## Proposed Pilot Scope
1. Filter IDs: `2`, `5`, `6`
2. Border modes tested: `mirror`, `constant`
3. Precision modes: `f32`, `mixed`
4. Validation gate: CUDA-vs-CPU parity matrix (`scripts/validate_cuda_cpu_matrix.py`)

## Success Criteria
1. Technical: parity matrix + smoke scripts run green in their environment.
2. Operational: clear diagnostics path from status codes and troubleshooting guide.
3. Economic: measurable pipeline-stage improvement tied to target workload.

## Pilot Timeline (3 weeks)
1. Week 1: environment bring-up and status-contract validation.
2. Week 2: benchmark protocol execution with claims packet output.
3. Week 3: production-readiness gap review and commercial checkpoint.

## Next Actions
1. Send Day 0 outreach with `docs/GETTING_STARTED_30MIN.md` and `docs/API_REFERENCE.md`.
2. Prepare benchmark command variant matching their representative batch/image shapes.
3. Pre-schedule week-3 decision call at pilot kickoff.
