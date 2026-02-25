# RC1 Sign-off Requests (Ready to Send)

Reference record: `release_records/RC_2026-02-25_RC1.md`

## Request 1: Core Kernel Engineer
Subject: `RC1 sign-off request (Core Kernel Engineer) - IIR2D`

Message:
1. Please review `release_records/RC_2026-02-25_RC1.md`.
2. Verify technical gate evidence refs: `A1`, `A2`, `A3`, `A4`.
3. Confirm decision as `approve` or `reject` with brief note.
4. Fill your row in the sign-off table.

Evidence shortcuts:
1. Linux smoke logs: `release_records/evidence/RC_2026-02-25_RC1/pass1_linux_smoke.txt`, `release_records/evidence/RC_2026-02-25_RC1/pass2_linux_smoke.txt`
2. Windows status smoke logs: `release_records/evidence/RC_2026-02-25_RC1/pass1_windows_status_smoke.txt`, `release_records/evidence/RC_2026-02-25_RC1/pass2_windows_status_smoke.txt`
3. Issue triage: `release_records/evidence/RC_2026-02-25_RC1/issue_triage_rc1.md`

## Request 2: Platform Engineer
Subject: `RC1 sign-off request (Platform Engineer) - IIR2D`

Message:
1. Please review `release_records/RC_2026-02-25_RC1.md`.
2. Verify operability evidence refs: `B1`, `B2`, `B3`.
3. Add CI run links/artifact links when available.
4. Confirm decision as `approve` or `reject` and fill your row.

Evidence shortcuts:
1. Linux benchmark logs: `release_records/evidence/RC_2026-02-25_RC1/pass1_linux_benchmark_smoke.txt`, `release_records/evidence/RC_2026-02-25_RC1/pass2_linux_benchmark_smoke.txt`
2. Benchmark artifacts: `release_records/artifacts/RC_2026-02-25_RC1/pass1_linux_core_benchmark_smoke.csv`, `release_records/artifacts/RC_2026-02-25_RC1/pass2_linux_core_benchmark_smoke.csv`

## Request 3: QA Engineer
Subject: `RC1 sign-off request (QA Engineer) - IIR2D`

Message:
1. Please review `release_records/RC_2026-02-25_RC1.md`.
2. Verify test and triage evidence refs: `A1`, `A2`, `A3`, `A5`.
3. Confirm issue acceptance state and decision (`approve`/`reject`).
4. Fill your row in the sign-off table.

Evidence shortcuts:
1. Smoke logs in `release_records/evidence/RC_2026-02-25_RC1/`
2. Issue triage: `release_records/evidence/RC_2026-02-25_RC1/issue_triage_rc1.md`

## Request 4: Product Lead
Subject: `RC1 promotion decision request (Product Lead) - IIR2D`

Message:
1. Please review `release_records/RC_2026-02-25_RC1.md`.
2. Decide waiver `WAIVER-RC1-SEC-001-003` as `approve` or `reject`.
3. Verify commercial readiness refs: `D1`, `D2`, `D3`.
4. Fill Product Lead sign-off row.
5. Update final promotion decision (`promote` or `hold`).

Evidence shortcuts:
1. Packaging note: `release_records/evidence/RC_2026-02-25_RC1/packaging_validation_note.md`
2. Legal/security note: `release_records/evidence/RC_2026-02-25_RC1/legal_security_review_note.md`
3. Triage note: `release_records/evidence/RC_2026-02-25_RC1/issue_triage_rc1.md`
