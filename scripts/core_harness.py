#!/usr/bin/env python3
"""Shared runtime and matrix utilities for CUDA core harness scripts."""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2

BORDER_MAP = {
    "clamp": 0,
    "mirror": 1,
    "wrap": 2,
    "constant": 3,
}

PRECISION_MAP = {
    "f32": 0,
    "mixed": 1,
    "f64": 2,
}


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
            raise ValueError(f"Invalid size token {tok!r}; expected WxH like 1024x1024")
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


def validate_case_axes(
    filter_ids: Iterable[int],
    border_modes: Iterable[str],
    precisions: Iterable[str],
) -> None:
    for filter_id in filter_ids:
        if filter_id < 1 or filter_id > 8:
            raise ValueError(f"Invalid filter_id {filter_id}; expected 1..8")
    for border_mode in border_modes:
        if border_mode not in BORDER_MAP:
            raise ValueError(f"Invalid border mode {border_mode!r}; expected one of {sorted(BORDER_MAP)}")
    for precision in precisions:
        if precision not in PRECISION_MAP:
            raise ValueError(f"Invalid precision {precision!r}; expected one of {sorted(PRECISION_MAP)}")


def parse_case_matrix(
    sizes_arg: str,
    filter_ids_arg: str,
    border_modes_arg: str,
    precisions_arg: str,
) -> list[Case]:
    sizes = parse_sizes(sizes_arg)
    filter_ids = parse_int_list(filter_ids_arg)
    border_modes = parse_str_list(border_modes_arg)
    precisions = parse_str_list(precisions_arg)
    validate_case_axes(filter_ids, border_modes, precisions)
    return build_cases(sizes, filter_ids, border_modes, precisions)


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


def load_core_library(repo_root: Path) -> tuple[ctypes.CDLL, Path]:
    errors: list[str] = []
    for path in candidate_core_libraries(repo_root):
        if not path.exists():
            continue
        try:
            return ctypes.CDLL(str(path)), path
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        details = "\n".join(errors)
        raise RuntimeError(f"Could not load iir2d shared library from candidates:\n{details}")
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
        for directory in search_dirs:
            if not directory.exists():
                continue
            d_key = str(directory).lower()
            if d_key in seen:
                continue
            seen.add(d_key)
            for dll in sorted(directory.glob("cudart64_*.dll"), reverse=True):
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
    cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    cudart.cudaMemset.restype = ctypes.c_int
    cudart.cudaDeviceSynchronize.argtypes = []
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int
    cudart.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudart.cudaRuntimeGetVersion.restype = ctypes.c_int
    cudart.cudaDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudart.cudaDriverGetVersion.restype = ctypes.c_int
    cudart.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudart.cudaGetDevice.restype = ctypes.c_int

    cudart.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudart.cudaEventCreate.restype = ctypes.c_int
    cudart.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaEventDestroy.restype = ctypes.c_int
    cudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cudart.cudaEventRecord.restype = ctypes.c_int
    cudart.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
    cudart.cudaEventSynchronize.restype = ctypes.c_int
    cudart.cudaEventElapsedTime.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    cudart.cudaEventElapsedTime.restype = ctypes.c_int


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


def decode_cuda_version(v: int) -> str:
    if v <= 0:
        return "unknown"
    major = v // 1000
    minor = (v % 1000) // 10
    return f"{major}.{minor}"


def parse_nvcc_release(text: str) -> str:
    m = re.search(r"release\s+(\d+\.\d+)", text)
    return m.group(1) if m else "unknown"


def run_cmd(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()
    except Exception:
        return ""


def get_gpu_name_and_driver() -> tuple[str, str]:
    out = run_cmd(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if not out:
        return ("unknown", "unknown")
    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return (line.strip(), "unknown")
    return (parts[0], parts[1])
