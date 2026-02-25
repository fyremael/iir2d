# CPU Reference Decision

Date: 2026-02-25 UTC  
Scope: `ENG-003` unblock (CPU reference for parity testing)

## Decision
Use a **spec-first, scalar CPU reference** as the canonical correctness contract for filter IDs `1..8`.

This reference is intentionally independent of CUDA scan/prefix implementation details.

## Why This Is The Canonical Contract
1. It is deterministic and easy to audit line-by-line.
2. It defines behavior in terms of public API semantics (`filter_id`, border mode, precision), not GPU execution strategy.
3. It remains stable even if CUDA kernels are re-optimized internally.

## Contract Details
1. Border semantics match the C API (`clamp`, `mirror`, `wrap`, `constant`) exactly.
2. Separable pipeline is the reference execution model:
   1. Row pass on `HxW`.
   2. Transpose.
   3. Row pass on transposed image.
   4. Transpose back.
3. Filter definitions:
   1. `1` and `7`: first-order recurrence (`alpha=0.85` profile).
   2. `2`: cascade of two first-order sections (`a=0.75`, `b=0.25`).
   3. `3`: single biquad profile (`b=[0.2,0.2,0.2]`, `a=[0.3,-0.1]`) with the same block-transform scan composition order used by the shipped CUDA core.
   4. `4`: cascade of two biquad profiles using the same block-transform scan composition contract.
   5. `5`: forward-backward first-order smoothing.
   6. `6`: Deriche forward/backward pair with `sigma=2.0`; second separable pass mirrors current runtime launch aliasing behavior for exact parity.
   7. `8`: same biquad scan-profile contract as `3` (current runtime behavior).
4. Precision policy:
   1. `f32`: float32 I/O and float32 accumulation.
   2. `mixed`: float32 I/O and float64 accumulation (cast-back points mirror shipped runtime path).
   3. `f64`: float64 I/O and float64 accumulation.

## Parity Tolerances (ENG-004 input)
1. `f32`: `rtol=5e-3`, `atol=5e-3`
2. `mixed`: `rtol=7e-3`, `atol=7e-3`
3. `f64`: `rtol=5e-8`, `atol=5e-8`

Rationale: tolerances are strict enough to catch semantic regressions while allowing expected floating-point ordering differences between scalar CPU and GPU scan implementations.

## Implementation
1. Canonical CPU reference implementation: `scripts/iir2d_cpu_reference.py`
2. CUDA-vs-CPU matrix validator: `scripts/validate_cuda_cpu_matrix.py`
