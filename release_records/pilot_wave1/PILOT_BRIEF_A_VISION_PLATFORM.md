# Pilot Brief A - Vision Platform Team

Date Opened: 2026-02-25  
Status: outreach_ready

## ICP Qualification (0-2 each)
1. CUDA maturity: 2
2. Pain severity: 2
3. Integration readiness: 1
4. Commercial urgency: 1
Total: 6 (qualified)

## Pain Hypothesis
The team has a latency-sensitive image pre/post-processing stage and needs a deterministic CUDA primitive with clear operational contracts.

## Proposed Pilot Scope
1. Filter IDs: `1`, `4`, `8`
2. Border mode baseline: `mirror`
3. Precision baseline: `f32`
4. Data sizes: `512x512`, `1024x1024`, `2048x2048`

## Success Criteria
1. Performance: meet or beat current stage p50 latency on one production-representative path.
2. Reliability: no critical defects in CI-gated path for 2 consecutive cycles.
3. Integration effort: first meaningful output in <= 1 engineering week.

## Pilot Timeline (3 weeks)
1. Week 1: integration + smoke + baseline benchmark command alignment.
2. Week 2: benchmark protocol run + evidence review.
3. Week 3: decision checkpoint (`proceed`, `extend`, `stop`).

## Next Actions
1. Send Day 0 outreach with `docs/PRODUCT_ONE_PAGER.md` and 30-minute discovery request.
2. Pre-build target-specific claims packet from benchmark harness output.
3. Lock decision date before pilot start.
