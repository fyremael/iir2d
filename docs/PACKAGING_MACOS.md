# macOS Packaging (x86_64 + arm64)

This package provides a macOS compatibility binary for API/ABI integration checks on non-CUDA hosts.

Contents:
1. `libiir2d_jax.dylib` (CPU stub build of the C API symbols)
2. `iir2d_core.h`
3. `LICENSE`
4. `NOTICE`
5. `SHA256SUMS.txt`

Notes:
1. macOS bundles are CPU-stub binaries and return `IIR2D_STATUS_CUDA_ERROR` for CUDA execution entrypoints.
2. Use these artifacts for contract validation and packaging integration; production filtering remains the CUDA Linux/Windows path.
3. Cross-compiling `x86_64` on Apple Silicon may require `IIR2D_SKIP_RUNTIME_SMOKE=1` because host runtime cannot load the foreign-arch dylib.

Build + package locally:
```bash
cd iir2d_op
# optional: set target arch to x86_64 or arm64
# export IIR2D_MACOS_ARCH=x86_64
bash scripts/build_and_smoke_macos.sh
mkdir -p dist/macos
cp -f build_macos/libiir2d_jax.dylib dist/macos/
cp -f csrc/iir2d_core.h dist/macos/
cp -f LICENSE NOTICE dist/macos/
(cd dist/macos && shasum -a 256 libiir2d_jax.dylib > SHA256SUMS.txt)
tar -czf dist/iir2d-macos-<arch>-dev.tar.gz -C dist/macos .
```
