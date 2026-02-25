# Compatibility Matrix (RC1)

Updated: 2026-02-25 UTC  
Release context: `release_records/RC_2026-02-25_RC1.md`

| OS | Runner/Host | GPU | Driver | CUDA Toolkit | CI Mode | Status |
|---|---|---|---|---|---|---|
| Linux (WSL/runner) | `iir2d-linux-selfhosted-01` | NVIDIA GeForce RTX 2080 | 591.44 | 13.1 (runtime observed 12.6 in local evidence) | self-hosted | validated |
| Windows | `iir2d-win-selfhosted-01` | NVIDIA GeForce RTX 2080 | 591.44 | 13.1 | self-hosted | validated (status-smoke policy) |

## Notes
1. Linux path is full smoke + benchmark artifact gate.
2. Windows path uses status-only smoke in CI policy (`-SkipGpuSmoke`).
3. Hosted fallback mode exists for control-plane continuity but is not primary release evidence.
