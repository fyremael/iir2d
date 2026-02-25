# Windows Packaging and Install Guide

## Package Contents
1. `iir2d_jax.dll`
2. `csrc/iir2d_core.h`
3. `scripts/smoke_core_status.py`
4. `BENCHMARK_PROTOCOL.md`
5. `LICENSE`, `NOTICE`, `SECURITY.md`

## Build + Stage
```powershell
cd D:\_codex\iir2d_op
powershell -ExecutionPolicy Bypass -File .\scripts\build_and_smoke_windows.ps1 -SkipGpuSmoke
New-Item -ItemType Directory -Force dist\windows | Out-Null
Copy-Item -Force .\python\iir2d_jax\iir2d_jax.dll .\dist\windows\
Copy-Item -Force .\csrc\iir2d_core.h .\dist\windows\
```

## One-Command Consume Path (local)
```powershell
cd D:\_codex\iir2d_op
powershell -ExecutionPolicy Bypass -File .\scripts\build_and_smoke_windows.ps1 -SkipGpuSmoke
```

## Validation
1. `scripts/smoke_core_status.py` prints `PASS`.
2. DLL exists in `python\iir2d_jax\` after build.
