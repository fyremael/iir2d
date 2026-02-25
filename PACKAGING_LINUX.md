# Linux Packaging and Install Guide

## Package Contents
1. `libiir2d_jax.so`
2. `csrc/iir2d_core.h`
3. `scripts/smoke_core_status.py`
4. `BENCHMARK_PROTOCOL.md`
5. `LICENSE`, `NOTICE`, `SECURITY.md`

## Build + Stage
```bash
cd /mnt/d/_codex/iir2d_op
bash scripts/build_and_smoke_wsl.sh
mkdir -p dist/linux
cp -f build_wsl/libiir2d_jax.so dist/linux/
cp -f csrc/iir2d_core.h dist/linux/
```

## One-Command Consume Path (local)
```bash
cd /mnt/d/_codex/iir2d_op
bash scripts/build_and_smoke_wsl.sh
```

## Validation
1. `scripts/smoke_core_status.py` prints `PASS`.
2. Benchmark harness can load staged library and emit CSV.
