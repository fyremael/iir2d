# IIR2D CUDA Core API/ABI Policy

## Scope
This policy applies to the C API declared in `csrc/iir2d_core.h`:
1. `iir2d_forward_cuda`
2. `iir2d_forward_cuda_stream`
3. `iir2d_status_string`
4. Public enums and `IIR2D_Params`

## Versioning
1. The public API uses semantic versioning via header macros:
   1. `IIR2D_API_VERSION_MAJOR`
   2. `IIR2D_API_VERSION_MINOR`
   3. `IIR2D_API_VERSION_PATCH`
2. Version increments:
   1. `PATCH`: internal fixes/perf changes with no API/ABI break.
   2. `MINOR`: backward-compatible API additions.
   3. `MAJOR`: breaking API/ABI changes.

## ABI Compatibility Contract
1. ABI is guaranteed stable across releases with the same `MAJOR`.
2. ABI-breaking changes are only allowed in a `MAJOR` release.
3. Existing enum numeric values and status code numeric values are stable once released.
4. Field order and meaning in `IIR2D_Params` are stable within a major version.

## Error Code Stability
1. Status codes are part of the public contract.
2. Existing codes must not be renumbered or redefined.
3. New status codes may be added in minor releases, preserving existing values.

## Deprecation Policy
1. API symbols are deprecated only in minor releases.
2. Deprecated symbols remain available for at least one full major cycle.
3. Removal happens only in the next major release.

## Change Control Requirements
1. Any public API/ABI change must include:
   1. Header update.
   2. `CHANGELOG.md` entry.
   3. Rationale in PR description.
2. Any proposed ABI break requires explicit major-version plan.

## Release Checklist (API/ABI)
1. Confirm header version macros are updated appropriately.
2. Confirm status code values are unchanged unless adding new values.
3. Confirm `IIR2D_Params` binary layout compatibility.
4. Confirm docs/examples updated.
