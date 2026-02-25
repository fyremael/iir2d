# Commercialization Hardening Specification
## Scope
This specification applies to the original CUDA-native filter core (`csrc/iir2d_core.cu` and C/C++ API), not the JAX integration layer.

## Objectives
1. Productize the CUDA core as a reliable, supportable SDK component.
2. Establish measurable technical and business value for target customers.
3. Build a repeatable go-to-market motion for pilots and paid adoption.

## Target Outcomes
1. Release-ready SDK for Linux and Windows with documented compatibility.
2. Benchmark-backed positioning versus customer-relevant alternatives.
3. At least 3 design-partner pilots with clear success metrics.

## Non-Goals
1. Positioning as a general-purpose deep learning training accelerator.
2. Shipping framework wrappers (JAX/PyTorch) as the primary product.

## Workstreams

## 1) Engineering Hardening
### 1.1 API and ABI Stability
1. Define and publish stable C API surface (`iir2d_forward_cuda`, `iir2d_forward_cuda_stream`, params struct, enums).
2. Introduce semantic versioning for API/ABI compatibility.
3. Add deprecation policy and compatibility guarantees per minor/major version.
4. Freeze parameter semantics (`filter_id`, border modes, precision behavior).

Acceptance Criteria:
1. Versioned header with explicit API stability contract.
2. Changelog includes compatibility notes.
3. ABI check performed in CI for release branches.

### 1.2 Validation and Error Model
1. Add strict runtime validation for all public inputs:
   1. Width/height > 0.
   2. Valid `filter_id` range and behavior.
   3. Valid precision enum and dtype expectations.
   4. Valid border mode enum.
2. Replace ambiguous failures with explicit status codes and messages.
3. Publish error code table in docs.

Acceptance Criteria:
1. Invalid inputs return deterministic error codes.
2. No silent fallback/default behavior for malformed params.

### 1.3 Correctness and Numeric Quality
1. Create CPU reference implementation for each filter mode.
2. Add parity tests (CUDA vs CPU) across:
   1. Shapes (small, odd/even, large).
   2. Border modes.
   3. Precision modes (`f32`, `mixed`, `f64`).
3. Define per-filter tolerances (absolute/relative) and document expected drift.
4. Add determinism tests across repeated runs and streams.

Acceptance Criteria:
1. Test matrix passes on target GPUs.
2. Numeric tolerances and deterministic guarantees are documented.

### 1.4 Performance and Reliability
1. Build standardized benchmark harness:
   1. Throughput/latency by shape/filter/precision.
   2. Warmup, steady-state, variance stats (p50/p95).
2. Add stress tests:
   1. Long-duration runs.
   2. Multi-stream concurrency.
   3. OOM behavior and recovery.
3. Profile and optimize kernel hotspots with Nsight traces archived per release.
4. Define SLOs for key profiles (for example: 1080p single frame latency targets).

Acceptance Criteria:
1. Performance baseline recorded and regression-gated in CI.
2. Stress tests run clean for a defined burn-in window (for example 1-4 hours).

### 1.5 Build, Packaging, and Distribution
1. Produce release artifacts:
   1. Linux `.so` and Windows `.dll`.
   2. Versioned headers.
   3. Sample apps and benchmark binaries.
2. Publish package channels:
   1. Source release tarball.
   2. Binary release bundle per OS/CUDA version.
3. Document support matrix:
   1. CUDA toolkit/runtime versions.
   2. GPU architectures.
   3. Driver minimums.
4. Add installer/consumption examples for CMake and Bazel.

Acceptance Criteria:
1. One-command install/consume path works for Linux and Windows reference environments.
2. Release notes include full compatibility matrix.

### 1.6 CI/CD and Release Process
1. Set up CI pipelines for:
   1. Build.
   2. Unit/integration tests.
   3. Benchmark smoke.
2. Set up nightly performance jobs on representative GPUs.
3. Define release gates and sign-off checklist.
4. Add artifact provenance and reproducible build metadata.

Acceptance Criteria:
1. No manual-only release steps.
2. Release candidate promoted only after all gates pass.

### 1.7 Security, Compliance, and Legal Hygiene
1. Add license file and third-party notices.
2. Run dependency/license scanning for build/runtime deps.
3. Add secure coding checks (compiler flags, static analysis where applicable).
4. Define vulnerability disclosure and patch SLAs.

Acceptance Criteria:
1. Licensing and notices complete for commercial distribution.
2. Security checklist attached to each release.

## 2) Product Hardening
### 2.1 Product Definition
1. Define SKU:
   1. Core CUDA SDK (primary).
   2. Optional enterprise support package.
2. Define supported use cases:
   1. Real-time imaging/video pre-processing.
   2. Industrial/medical/scientific filtering pipelines.
3. Define out-of-scope use cases to avoid mis-positioning.

Acceptance Criteria:
1. Product one-pager with crisp problem/solution statement.
2. Messaging avoids unsupported ML superiority claims.

### 2.2 Documentation and Developer Experience
1. Publish docs:
   1. Getting started in <30 minutes.
   2. API reference.
   3. Performance tuning guide.
   4. Troubleshooting and FAQ.
2. Provide examples:
   1. Minimal C++ example.
   2. Streaming pipeline example.
   3. Batch processing benchmark example.
3. Add migration guide between major versions.

Acceptance Criteria:
1. New engineer can integrate and run first filter in one session.
2. Support tickets for setup issues trend down after doc release.

### 2.3 Quality Claims and Benchmark Protocol
1. Define authoritative benchmark protocol:
   1. Hardware/software disclosure.
   2. Input datasets and resolutions.
   3. Exact commands and run count.
2. Report both performance and quality metrics where relevant.
3. Include competitor/comparator baselines that customers actually use.

Acceptance Criteria:
1. All external claims are reproducible from published protocol.
2. Marketing material references benchmark IDs and run dates.

### 2.4 Support and Operability
1. Define support tiers, response SLAs, escalation flow.
2. Add runtime diagnostics:
   1. Version query API.
   2. Build fingerprint.
   3. Optional debug logging hooks.
3. Publish known issues and workaround registry.

Acceptance Criteria:
1. Customers can self-diagnose common integration failures.
2. Support can triage incidents with standardized diagnostics.

## 3) GTM Hardening
### 3.1 Ideal Customer Profile (ICP) and Segmentation
1. Prioritize segments with immediate pain:
   1. Video analytics vendors.
   2. Machine vision OEMs.
   3. Imaging SaaS platforms with GPU-heavy preprocessing costs.
2. Define buyer persona:
   1. Engineering lead (technical champion).
   2. Product owner (business sponsor).
3. Create qualification criteria (GPU footprint, latency sensitivity, integration maturity).

Acceptance Criteria:
1. Top 2 ICP segments selected with quantified TAM/SAM assumptions.
2. Lead qualification rubric used by all outreach.

### 3.2 Value Proposition and Positioning
1. Primary value message:
   1. Deterministic, high-throughput recursive filtering as a drop-in CUDA component.
2. Proof points:
   1. Measured latency/throughput improvements in representative pipelines.
   2. Operational stability under sustained load.
3. Objection handling:
   1. Build complexity.
   2. Precision correctness.
   3. Vendor lock-in concerns.

Acceptance Criteria:
1. Sales deck and technical brief share consistent claims and metrics.
2. Each claim maps to a benchmark artifact or pilot evidence.

### 3.3 Pilot Program
1. Create design-partner pilot template:
   1. Baseline measurement.
   2. Integration plan.
   3. Success thresholds (latency, cost, quality).
   4. Timeline and owners.
2. Offer pilot enablement package:
   1. Integration support.
   2. Weekly performance review.
   3. Joint final report.
3. Define conversion criteria from pilot to paid.

Acceptance Criteria:
1. 3 pilots executed with signed success criteria.
2. At least 1 pilot converts to paid engagement.

### 3.4 Pricing and Packaging
1. Evaluate pricing models:
   1. Per-deployment license.
   2. Usage-based (GPU-hour or throughput tiers).
   3. Subscription + support.
2. Create commercial terms:
   1. Evaluation license.
   2. Production license.
   3. Enterprise support add-on.
3. Define discount policy and non-standard terms policy.

Acceptance Criteria:
1. Pricing sheet approved and tested in pilot conversations.
2. Time-to-quote under 48 hours.

### 3.5 Revenue Readiness
1. Build pipeline model:
   1. Target accounts.
   2. Stage definitions.
   3. Conversion assumptions.
2. Track core KPIs:
   1. Pilot win rate.
   2. Pilot-to-paid conversion.
   3. Gross margin per deployment profile.
3. Build reference cases and testimonials from first customers.

Acceptance Criteria:
1. Monthly GTM review with KPI dashboard.
2. First repeatable reference architecture published.

## Delivery Plan
## Phase 1 (0-6 weeks): Foundation
1. API/ABI stabilization.
2. Validation/error model.
3. Correctness test harness.
4. Initial benchmark protocol.

Exit Criteria:
1. Internal release candidate with passing core test matrix.

## Phase 2 (6-12 weeks): Production Readiness
1. CI/CD and nightly perf gates.
2. Packaging for Linux/Windows.
3. Documentation and examples.
4. Security/legal hygiene completion.

Exit Criteria:
1. External beta SDK ready for design partners.

## Phase 3 (12-20 weeks): Commercial Launch Prep
1. Design-partner pilots.
2. Pricing/package finalization.
3. Case studies and benchmark-backed collateral.

Exit Criteria:
1. GA launch decision with at least one paid conversion path.

## Risks and Mitigations
1. Risk: Performance not compelling versus tuned alternatives.
   Mitigation: Focus on specific workload niches; add pipeline-level benchmarks, not microbench-only claims.
2. Risk: Integration friction reduces adoption.
   Mitigation: Ship robust binaries, clear docs, and reference integrations.
3. Risk: Quality/correctness concerns block production use.
   Mitigation: Publish reference parity tests and strict acceptance tolerances.
4. Risk: Scope drift into framework wrappers.
   Mitigation: Keep CUDA SDK as the product core; wrappers remain optional adapters.

## Definition of Done (Commercialization Readiness)
1. Engineering:
   1. Stable API/ABI and validated release pipeline.
   2. Comprehensive correctness + reliability + performance tests.
2. Product:
   1. Complete docs, support model, reproducible benchmark protocol.
3. GTM:
   1. Defined ICP, pilot playbook, pricing, and reference evidence.
4. Business:
   1. At least one validated paid adoption path.
