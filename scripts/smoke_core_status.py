import ctypes
import os
import sys
from pathlib import Path


EXPECTED = {
    0: "ok",
    -1: "invalid_argument",
    -2: "invalid_dimension",
    -3: "invalid_filter_id",
    -4: "invalid_border_mode",
    -5: "invalid_precision",
    -6: "null_pointer",
    -7: "workspace_error",
    -8: "cuda_error",
}


def candidate_libs(repo_root: Path) -> list[Path]:
    pkg = repo_root / "python" / "iir2d_jax"
    if os.name == "nt":
        return [
            pkg / "iir2d_jax.dll",
            repo_root / "build_win_ninja" / "iir2d_jax.dll",
            repo_root / "build_win_vs2019" / "Release" / "iir2d_jax.dll",
            repo_root / "build" / "iir2d_jax.dll",
            repo_root / "build" / "Release" / "iir2d_jax.dll",
        ]
    return [
        pkg / "libiir2d_jax.so",
        pkg / "iir2d_jax.so",
        repo_root / "build_wsl" / "libiir2d_jax.so",
    ]


def load_lib(paths: list[Path]) -> ctypes.CDLL:
    for p in paths:
        if p.exists():
            return ctypes.CDLL(str(p))
    raise RuntimeError("No iir2d shared library found in expected locations.")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    lib = load_lib(candidate_libs(repo_root))
    lib.iir2d_status_string.restype = ctypes.c_char_p
    lib.iir2d_status_string.argtypes = [ctypes.c_int]

    errors = []
    for code, expected in EXPECTED.items():
        got = lib.iir2d_status_string(code)
        text = got.decode("utf-8") if got is not None else "<null>"
        if text != expected:
            errors.append((code, expected, text))

    if errors:
        print("FAIL: status-string mismatch")
        for code, expected, got in errors:
            print(f"  code={code}: expected={expected!r} got={got!r}")
        return 1

    print("PASS: iir2d_status_string contract")
    print(f"library={getattr(lib, '_name', '<unknown>')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
