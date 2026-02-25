# RC1 Legal/Security Review Note

Date (UTC): 2026-02-25
Reviewer: Codex session (pre-sign-off review; refreshed during closeout)

## Current State
1. `SEC-001` (LICENSE/NOTICES) is `in_progress`:
   1. Draft `LICENSE` and `NOTICE` files are in repo.
   2. Final legal wording/sign-off remains open before external distribution.
2. `SEC-002` (dependency/license scan in CI) is `done`:
   1. Workflow `.github/workflows/dependency-license-scan.yml` enforces vulnerability and license policy checks.
3. `SEC-003` (vulnerability response process) is `done`:
   1. Public policy file `SECURITY.md` is present.

## RC Recommendation
1. Treat legal/security items as controlled RC waivers, not GA-complete.
2. Require Product Lead approval with explicit expiry.
3. Block GA promotion until SEC items are completed.

## Proposed Waiver
1. Waiver ID: `WAIVER-RC1-SEC-001-003`
2. Scope: RC stage only.
3. Risk: incomplete legal/security operationalization for external distribution.
4. Owner: Product Lead.
5. Expiry: 2026-03-31 UTC.
6. Condition: no GA promotion without SEC closure.
