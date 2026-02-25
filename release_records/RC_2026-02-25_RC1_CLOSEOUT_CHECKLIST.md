# RC1 Closeout Checklist (Promotion-Ready)

Use this list to move `RC_2026-02-25_RC1` from `hold` to `promote`.

## Required Inputs
- [x] Add CI run link #1 to `release_records/RC_2026-02-25_RC1.md` (Evidence placeholder #1).
- [x] Add CI run link #2 to `release_records/RC_2026-02-25_RC1.md` (Evidence placeholder #2).
- [x] Add CI benchmark artifact link to `release_records/RC_2026-02-25_RC1.md` (Evidence placeholder #3).
- [x] Confirm self-hosted CUDA jobs (Linux + Windows) succeed on the linked runs.
- [x] Product Lead approves or rejects `WAIVER-RC1-SEC-001-003`.
- [x] Core Kernel Engineer fills sign-off row.
- [x] Platform Engineer fills sign-off row.
- [x] QA Engineer fills sign-off row.
- [x] Product Lead fills sign-off row.

## Finalization
- [x] Set `Final decision` in `release_records/RC_2026-02-25_RC1.md` to `promote` or `hold`.
- [x] Update decision timestamp (UTC).
- [x] Update decision owner.
- [x] If promoted, update `task_board.md` status notes for `REL-001`.
- [x] If held, keep blockers listed under follow-ups with owner/date. (`N/A - promoted`)

## Fast Path
1. Send role requests from `release_records/RC_2026-02-25_RC1_SIGNOFF_REQUESTS.md`.
2. Collect replies.
3. Paste decisions directly into `release_records/RC_2026-02-25_RC1.md`.
