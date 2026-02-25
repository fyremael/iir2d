# ICP Segments and Qualification Rubric

## ICP 1: Vision/Imaging Platform Teams (GPU-heavy)
### Qualification
1. Uses CUDA in production.
2. Has latency-sensitive image filtering/transformation stage.
3. Can run self-hosted CI or controlled GPU benchmark environment.
4. Has owner for native integration (C++/Python).

## ICP 2: Applied ML Infrastructure Teams
### Qualification
1. Needs deterministic preprocessing/postprocessing in model pipelines.
2. Requires audit-grade benchmark evidence for internal adoption.
3. Accepts native dependency in exchange for performance/operability control.
4. Has defined pilot success criteria (latency, throughput, reliability).

## Disqualifiers
1. No GPU deployment path.
2. No ability to evaluate native binary artifacts.
3. Needs fully managed hosted service only.

## Scoring (0-2 each, max 8)
1. CUDA maturity
2. Pain severity
3. Integration readiness
4. Commercial urgency

Target pilots: score `>=6`.
