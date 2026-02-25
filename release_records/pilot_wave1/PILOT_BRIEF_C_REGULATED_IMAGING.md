# Pilot Brief C - Regulated Imaging / Edge CUDA Team

Date Opened: 2026-02-25  
Status: outreach_ready

## ICP Qualification (0-2 each)
1. CUDA maturity: 2
2. Pain severity: 2
3. Integration readiness: 1
4. Commercial urgency: 1
Total: 6 (qualified)

## Pain Hypothesis
The team needs deterministic behavior, reproducible evidence, and explicit error handling for deployment in regulated or high-assurance imaging workflows.

## Proposed Pilot Scope
1. Filter IDs: `1`, `3`, `6`, `8`
2. Border modes: full sweep (`clamp`, `mirror`, `wrap`, `constant`)
3. Precision focus: `f32` production, `f64` verification
4. Evidence set: benchmark CSV + claims packet + CI parity logs

## Success Criteria
1. Correctness: parity matrix passes within tolerance on target GPU.
2. Traceability: benchmark claims packet generated and approved internally.
3. Delivery: pilot ends with explicit production recommendation.

## Pilot Timeline (3 weeks)
1. Week 1: install path and CI smoke on customer-like environment.
2. Week 2: correctness + performance evidence capture.
3. Week 3: compliance/operability review + decision.

## Next Actions
1. Send Day 0 outreach with `COMPATIBILITY_MATRIX.md`, `SECURITY.md`, and one-pager.
2. Predefine evidence folders for audit retention.
3. Require named technical approver and procurement approver at kickoff.
