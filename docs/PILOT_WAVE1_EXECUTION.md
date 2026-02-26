# Pilot Wave 1 Execution Plan

Date: 2026-02-25 UTC  
Owner: GTM Lead

## Objective
Secure 3 signed design-partner pilot agreements by 2026-03-12 using the existing commercialization packet and a strict outreach cadence.

## Target Slots (Wave 1)
1. Slot A: Vision/imaging platform team (ICP-1 target score >= 6).
2. Slot B: Applied ML infra team (ICP-2 target score >= 6).
3. Slot C: Regulated imaging or edge-CUDA team (ICP-1 target score >= 6).

## Required Assets (already in repo)
1. Product framing: `PRODUCT_ONE_PAGER.md`
2. Quick integration path: `GETTING_STARTED_30MIN.md`
3. API + compatibility: `API_REFERENCE.md`, `COMPATIBILITY_MATRIX.md`
4. Claims evidence process: `BENCHMARK_PROTOCOL.md`
5. Claims packet generator: `scripts/build_benchmark_claims_packet.py`
6. Pilot structure template: `DESIGN_PARTNER_PILOT_TEMPLATE.md`
7. Conversion path: `PILOT_TO_PAID_PLAYBOOK.md`
8. Outreach tracker: `release_records/pilot_wave1/OUTREACH_TRACKER.md`
9. Per-target readiness checklist: `release_records/pilot_wave1/PILOT_ACCEPTANCE_CHECKLIST.md`

## Outreach Cadence (per target)
1. Day 0: Intro email with one-pager + 20-minute technical discovery ask.
2. Day 2: Follow-up with benchmark protocol + claims packet promise scoped to their workload.
3. Day 5: Short final nudge with explicit pilot window (3 weeks) and success criteria structure.

## Discovery Call Agenda (20-30 minutes)
1. Current pipeline bottleneck and where recursive filtering sits.
2. Runtime constraints: GPU class, CUDA/toolchain, deployment model.
3. Success metrics: p50/p95 latency, throughput, integration time, reliability gates.
4. Pilot terms: start date, 3-week timeline, decision checkpoint.

## Pilot Acceptance Gates
1. Technical fit:
   1. CUDA deployment path exists.
   2. Team can run benchmark commands and share reproducible output.
2. Business fit:
   1. Named owner and decision deadline.
   2. Path from pilot to paid decision is defined.

## Weekly Operating Rhythm (Wave 1)
1. Monday: pipeline review (target status + next action owner).
2. Wednesday: technical prep review (demo, benchmark, claims packet readiness).
3. Friday: conversion forecast (best case / commit / risk).

## Execution Checklist
1. Create one pilot brief per target in `release_records/pilot_wave1/`.
2. Produce one benchmark claims packet tailored to each target workload.
3. Capture every interaction and next step directly in each pilot brief.
4. Require a dated go/no-go checkpoint for each target by week 3.
5. Update `OUTREACH_TRACKER.md` on Monday/Wednesday/Friday cadence.
6. Complete `PILOT_ACCEPTANCE_CHECKLIST.md` before each proposal handoff.
