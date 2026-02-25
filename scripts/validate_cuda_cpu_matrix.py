#!/usr/bin/env python3
"""Validate CUDA core output against canonical CPU reference across a case matrix."""

from __future__ import annotations

import argparse
import ctypes
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from iir2d_cpu_reference import BORDER_MAP, PRECISION_MAP, TOLERANCE_MAP, iir2d_cpu_reference

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


class IIR2D_Params(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("filter_id", ctypes.c_int),
        ("border_mode", ctypes.c_int),
        ("border_const", ctypes.c_float),
        ("precision", ctypes.c_int),
    ]


@dataclass(frozen=True)
class Case:
    width: int
    height: int
    filter_id: int
    border_mode: str
    precision: str


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_str_list(value: str) -> list[str]:
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def parse_sizes(value: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for tok in value.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)x(\d+)", tok)
        if not m:
            raise ValueError(f"Invalid size token {tok!r}; expected WxH like 63x47")
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def build_cases(
    sizes: Iterable[tuple[int, int]],
    filter_ids: Iterable[int],
    border_modes: Iterable[str],
    precisions: Iterable[str],
) -> list[Case]:
    out: list[Case] = []
    for width, height in sizes:
        for filter_id in filter_ids:
            for border_mode in border_modes:
                for precision in precisions:
                    out.append(
                        Case(
                            width=width,
                            height=height,
                            filter_id=filter_id,
                            border_mode=border_mode,
                            precision=precision,
                        )
                    )
    return out


def candidate_core_libraries(repo_root: Path) -> list[Path]:
    pkg = repo_root / "python" / "iir2d_jax"
    if os.name == "nt":
        return [
            pkg / "iir2d_jax.dll",
            repo_root / "build_win_ninja" / "iir2d_jax.dll",
            repo_root / "build_win_vs2019" / "Release" / "iir2d_jax.dll",
            pkg / "libiir2d_jax.so",
            pkg / "iir2d_jax.so",
            repo_root / "build_wsl" / "libiir2d_jax.so",
        ]
    return [
        pkg / "libiir2d_jax.so",
        pkg / "iir2d_jax.so",
        repo_root / "build_wsl" / "libiir2d_jax.so",
        pkg / "iir2d_jax.dll",
        repo_root / "build_win_ninja" / "iir2d_jax.dll",
        repo_root / "build_win_vs2019" / "Release" / "iir2d_jax.dll",
    ]


def load_core_library(repo_root: Path) -> ctypes.CDLL:
    errors: list[str] = []
    for path in candidate_core_libraries(repo_root):
        if not path.exists():
            continue
        try:
            return ctypes.CDLL(str(path))
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        joined = "\n".join(errors)
        raise RuntimeError(f"Could not load iir2d shared library from candidates:\n{joined}")
    raise FileNotFoundError("Could not locate iir2d shared library in expected paths.")


def find_cudart() -> ctypes.CDLL:
    if os.name == "nt":
        search_dirs: list[Path] = []
        cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if cuda_root.exists():
            versions = sorted(
                [p for p in cuda_root.iterdir() if p.is_dir() and re.fullmatch(r"v\d+\.\d+", p.name)],
                key=lambda p: tuple(int(x) for x in p.name[1:].split(".")),
                reverse=True,
            )
            search_dirs.extend([p / "bin" for p in versions])
        for key, val in os.environ.items():
            if key.startswith("CUDA_PATH") and val:
                search_dirs.append(Path(val) / "bin")
        for seg in os.environ.get("PATH", "").split(os.pathsep):
            if seg:
                search_dirs.append(Path(seg))
        seen: set[str] = set()
        for d in search_dirs:
            if not d.exists():
                continue
            d_key = str(d).lower()
            if d_key in seen:
                continue
            seen.add(d_key)
            for dll in sorted(d.glob("cudart64_*.dll"), reverse=True):
                try:
                    return ctypes.CDLL(str(dll))
                except OSError:
                    continue
        raise RuntimeError("Failed to load cudart64_*.dll from CUDA/PATH locations.")

    for name in (
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "/usr/local/cuda/lib64/libcudart.so",
    ):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("Failed to load libcudart.so")


def configure_cudart(cudart: ctypes.CDLL) -> None:
    cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
    cudart.cudaGetErrorString.restype = ctypes.c_char_p
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemcpy.restype = ctypes.c_int
    cudart.cudaDeviceSynchronize.argtypes = []
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int


def configure_core_lib(core_lib: ctypes.CDLL) -> None:
    core_lib.iir2d_forward_cuda.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(IIR2D_Params)]
    core_lib.iir2d_forward_cuda.restype = ctypes.c_int
    core_lib.iir2d_status_string.argtypes = [ctypes.c_int]
    core_lib.iir2d_status_string.restype = ctypes.c_char_p


def cuda_check(cudart: ctypes.CDLL, code: int, context: str) -> None:
    if code == 0:
        return
    err = cudart.cudaGetErrorString(code)
    msg = err.decode("utf-8") if err else f"CUDA error {code}"
    raise RuntimeError(f"{context} failed: {msg} ({code})")


def run_cuda_forward(
    cudart: ctypes.CDLL,
    core_lib: ctypes.CDLL,
    host_in: np.ndarray,
    params: IIR2D_Params,
) -> np.ndarray:
    host_in = np.ascontiguousarray(host_in)
    host_out = np.empty_like(host_in)
    bytes_total = host_in.nbytes
    d_in = ctypes.c_void_p()
    d_out = ctypes.c_void_p()
    try:
        cuda_check(cudart, cudart.cudaMalloc(ctypes.byref(d_in), bytes_total), "cudaMalloc(d_in)")
        cuda_check(cudart, cudart.cudaMalloc(ctypes.byref(d_out), bytes_total), "cudaMalloc(d_out)")
        cuda_check(
            cudart,
            cudart.cudaMemcpy(
                d_in,
                host_in.ctypes.data_as(ctypes.c_void_p),
                bytes_total,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy(H2D)",
        )

        status = core_lib.iir2d_forward_cuda(d_in, d_out, ctypes.byref(params))
        if status != 0:
            text = core_lib.iir2d_status_string(status)
            msg = text.decode("utf-8") if text else str(status)
            raise RuntimeError(f"iir2d_forward_cuda failed: {msg} ({status})")
        cuda_check(cudart, cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
        cuda_check(
            cudart,
            cudart.cudaMemcpy(
                host_out.ctypes.data_as(ctypes.c_void_p),
                d_out,
                bytes_total,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy(D2H)",
        )
        return host_out
    finally:
        if d_in.value:
            cudart.cudaFree(d_in)
        if d_out.value:
            cudart.cudaFree(d_out)


def max_diff_summary(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float, tuple[int, int], float, float]:
    abs_diff = np.abs(actual - expected)
    flat_idx = int(np.argmax(abs_diff))
    idx = np.unravel_index(flat_idx, abs_diff.shape)
    max_abs = float(abs_diff[idx])
    exp_v = float(expected[idx])
    got_v = float(actual[idx])
    denom = max(abs(exp_v), 1e-30)
    max_rel = abs(got_v - exp_v) / denom
    return max_abs, float(max_rel), (int(idx[0]), int(idx[1])), exp_v, got_v


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate CUDA output against canonical CPU reference.")
    ap.add_argument("--sizes", default="63x47", help="Comma list, e.g. 63x47,128x96")
    ap.add_argument("--filter_ids", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--border_modes", default="clamp,mirror,wrap,constant")
    ap.add_argument("--precisions", default="f32,mixed,f64")
    ap.add_argument("--border_const", type=float, default=0.125)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rtol_f32", type=float, default=TOLERANCE_MAP["f32"][0])
    ap.add_argument("--atol_f32", type=float, default=TOLERANCE_MAP["f32"][1])
    ap.add_argument("--rtol_mixed", type=float, default=TOLERANCE_MAP["mixed"][0])
    ap.add_argument("--atol_mixed", type=float, default=TOLERANCE_MAP["mixed"][1])
    ap.add_argument("--rtol_f64", type=float, default=TOLERANCE_MAP["f64"][0])
    ap.add_argument("--atol_f64", type=float, default=TOLERANCE_MAP["f64"][1])
    ap.add_argument("--max_failures", type=int, default=8)
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes)
    filter_ids = parse_int_list(args.filter_ids)
    border_modes = parse_str_list(args.border_modes)
    precisions = parse_str_list(args.precisions)

    for fid in filter_ids:
        if fid < 1 or fid > 8:
            raise ValueError(f"Invalid filter_id {fid}; expected 1..8")
    for mode in border_modes:
        if mode not in BORDER_MAP:
            raise ValueError(f"Invalid border mode {mode!r}; expected one of {sorted(BORDER_MAP)}")
    for precision in precisions:
        if precision not in PRECISION_MAP:
            raise ValueError(f"Invalid precision {precision!r}; expected one of {sorted(PRECISION_MAP)}")

    tolerances = {
        "f32": (args.rtol_f32, args.atol_f32),
        "mixed": (args.rtol_mixed, args.atol_mixed),
        "f64": (args.rtol_f64, args.atol_f64),
    }

    repo_root = Path(__file__).resolve().parents[1]
    core_lib = load_core_library(repo_root)
    cudart = find_cudart()
    configure_core_lib(core_lib)
    configure_cudart(cudart)

    cases = build_cases(sizes, filter_ids, border_modes, precisions)
    failures = 0

    for idx, case in enumerate(cases, 1):
        if case.precision == "f64":
            dtype = np.float64
        else:
            dtype = np.float32
        seed_case = (
            args.seed
            + case.filter_id * 1009
            + case.width * 917
            + case.height * 613
            + BORDER_MAP[case.border_mode] * 101
            + PRECISION_MAP[case.precision] * 29
        )
        rng = np.random.default_rng(seed_case)
        host_in = (rng.random((case.height, case.width), dtype=dtype) * dtype(2.0)) - dtype(1.0)
        params = IIR2D_Params(
            width=case.width,
            height=case.height,
            filter_id=case.filter_id,
            border_mode=BORDER_MAP[case.border_mode],
            border_const=float(args.border_const),
            precision=PRECISION_MAP[case.precision],
        )
        expected = iir2d_cpu_reference(
            host_in,
            filter_id=case.filter_id,
            border_mode=case.border_mode,
            border_const=float(args.border_const),
            precision=case.precision,
        )
        got = run_cuda_forward(cudart, core_lib, host_in, params)
        rtol, atol = tolerances[case.precision]
        ok = np.allclose(got, expected, rtol=rtol, atol=atol)
        max_abs, max_rel, loc, exp_v, got_v = max_diff_summary(expected, got)
        status = "PASS" if ok else "FAIL"
        print(
            f"[{idx}/{len(cases)}] {status} "
            f"{case.width}x{case.height} f{case.filter_id} {case.border_mode}/{case.precision} "
            f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
        )
        if not ok:
            failures += 1
            print(
                f"  detail: loc={loc} expected={exp_v:.9e} got={got_v:.9e} "
                f"tol(rtol={rtol:.3e}, atol={atol:.3e})"
            )
            if failures >= args.max_failures:
                print(f"Stopping after {failures} failures (max_failures={args.max_failures}).")
                break

    if failures:
        print(f"FAIL: {failures} case(s) exceeded tolerance.")
        return 1
    print(f"PASS: all {len(cases)} cases matched CPU reference contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
