#!/usr/bin/env python3
"""Reproducible benchmark harness for the IIR2D CUDA core C API."""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import platform
import re
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


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
            raise ValueError(f"Invalid size token: {tok!r}. Expected WxH, e.g. 1024x1024")
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def find_core_library(repo_root: Path) -> Path:
    pkg = repo_root / "python" / "iir2d_jax"
    candidates = [
        pkg / "libiir2d_jax.so",
        pkg / "iir2d_jax.so",
        pkg / "iir2d_jax.dll",
        repo_root / "build_wsl" / "libiir2d_jax.so",
        repo_root / "build_win_ninja" / "iir2d_jax.dll",
        repo_root / "build_win_vs2019" / "Release" / "iir2d_jax.dll",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not locate iir2d_jax shared library in expected paths.")


def _run_cmd(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()
    except Exception:
        return ""


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
            key = str(d).lower()
            if key in seen:
                continue
            seen.add(key)
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


def decode_cuda_version(v: int) -> str:
    if v <= 0:
        return "unknown"
    major = v // 1000
    minor = (v % 1000) // 10
    return f"{major}.{minor}"


def parse_nvcc_release(text: str) -> str:
    m = re.search(r"release\s+(\d+\.\d+)", text)
    return m.group(1) if m else "unknown"


def get_gpu_name_and_driver() -> tuple[str, str]:
    out = _run_cmd(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if not out:
        return ("unknown", "unknown")
    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return (line.strip(), "unknown")
    return (parts[0], parts[1])


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


def run_case(
    cudart: ctypes.CDLL,
    core_lib: ctypes.CDLL,
    case: Case,
    border_const: float,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, str | int | float]:
    n = case.width * case.height
    if case.precision == "f64":
        dtype = np.float64
        elem_size = 8
    else:
        dtype = np.float32
        elem_size = 4
    bytes_total = n * elem_size

    rng = np.random.default_rng(seed + case.filter_id * 17 + case.width * 3 + case.height)
    host_in = rng.random(n, dtype=np.float32)
    if dtype is np.float64:
        host_in = host_in.astype(np.float64)
    host_in = host_in.reshape((case.height, case.width))

    d_in = ctypes.c_void_p()
    d_out = ctypes.c_void_p()
    start_evt = ctypes.c_void_p()
    end_evt = ctypes.c_void_p()
    lat_ms: list[float] = []

    params = IIR2D_Params(
        width=case.width,
        height=case.height,
        filter_id=case.filter_id,
        border_mode=BORDER_MAP[case.border_mode],
        border_const=float(border_const),
        precision=PRECISION_MAP[case.precision],
    )

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
        cuda_check(cudart, cudart.cudaMemset(d_out, 0, bytes_total), "cudaMemset(d_out)")
        cuda_check(cudart, cudart.cudaEventCreate(ctypes.byref(start_evt)), "cudaEventCreate(start)")
        cuda_check(cudart, cudart.cudaEventCreate(ctypes.byref(end_evt)), "cudaEventCreate(end)")

        for _ in range(warmup):
            status = core_lib.iir2d_forward_cuda(d_in, d_out, ctypes.byref(params))
            if status != 0:
                text = core_lib.iir2d_status_string(status)
                msg = text.decode("utf-8") if text else str(status)
                raise RuntimeError(f"Warmup iir2d_forward_cuda failed: {msg} ({status})")
        cuda_check(cudart, cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)")

        for _ in range(iters):
            cuda_check(cudart, cudart.cudaEventRecord(start_evt, None), "cudaEventRecord(start)")
            status = core_lib.iir2d_forward_cuda(d_in, d_out, ctypes.byref(params))
            if status != 0:
                text = core_lib.iir2d_status_string(status)
                msg = text.decode("utf-8") if text else str(status)
                raise RuntimeError(f"Timed iir2d_forward_cuda failed: {msg} ({status})")
            cuda_check(cudart, cudart.cudaEventRecord(end_evt, None), "cudaEventRecord(end)")
            cuda_check(cudart, cudart.cudaEventSynchronize(end_evt), "cudaEventSynchronize(end)")
            elapsed = ctypes.c_float()
            cuda_check(
                cudart,
                cudart.cudaEventElapsedTime(ctypes.byref(elapsed), start_evt, end_evt),
                "cudaEventElapsedTime",
            )
            lat_ms.append(float(elapsed.value))

        arr = np.asarray(lat_ms, dtype=np.float64)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        mean = float(np.mean(arr))
        tput_mpix = (n / 1e6) / (p50 / 1000.0)
        tput_gb_s = (2.0 * bytes_total / 1e9) / (p50 / 1000.0)
        return {
            "width": case.width,
            "height": case.height,
            "pixels": n,
            "filter_id": case.filter_id,
            "border_mode": case.border_mode,
            "precision": case.precision,
            "latency_ms_min": float(np.min(arr)),
            "latency_ms_p50": p50,
            "latency_ms_p95": p95,
            "latency_ms_mean": mean,
            "latency_ms_max": float(np.max(arr)),
            "throughput_mpix_per_s_p50": tput_mpix,
            "throughput_gb_per_s_p50": tput_gb_s,
            "warmup_iters": warmup,
            "timed_iters": iters,
        }
    finally:
        if start_evt.value:
            cudart.cudaEventDestroy(start_evt)
        if end_evt.value:
            cudart.cudaEventDestroy(end_evt)
        if d_in.value:
            cudart.cudaFree(d_in)
        if d_out.value:
            cudart.cudaFree(d_out)


def build_cases(
    sizes: Iterable[tuple[int, int]],
    filter_ids: Iterable[int],
    border_modes: Iterable[str],
    precisions: Iterable[str],
) -> list[Case]:
    out: list[Case] = []
    for (w, h) in sizes:
        for fid in filter_ids:
            for b in border_modes:
                for p in precisions:
                    out.append(Case(width=w, height=h, filter_id=fid, border_mode=b, precision=p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark harness for iir2d_forward_cuda.")
    ap.add_argument("--sizes", default="1024x1024", help="Comma list, e.g. 1024x1024,2048x2048")
    ap.add_argument("--filter_ids", default="1,4,8", help="Comma list of filter IDs (1..8)")
    ap.add_argument("--border_modes", default="mirror", help="Comma list: clamp,mirror,wrap,constant")
    ap.add_argument("--precisions", default="f32", help="Comma list: f32,mixed,f64")
    ap.add_argument("--border_const", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    lib_path = find_core_library(repo_root)
    core_lib = ctypes.CDLL(str(lib_path))
    cudart = find_cudart()
    configure_core_lib(core_lib)
    configure_cudart(cudart)

    runtime_v = ctypes.c_int()
    driver_v = ctypes.c_int()
    device_id = ctypes.c_int(-1)
    cuda_check(cudart, cudart.cudaRuntimeGetVersion(ctypes.byref(runtime_v)), "cudaRuntimeGetVersion")
    cuda_check(cudart, cudart.cudaDriverGetVersion(ctypes.byref(driver_v)), "cudaDriverGetVersion")
    cuda_check(cudart, cudart.cudaGetDevice(ctypes.byref(device_id)), "cudaGetDevice")

    gpu_name, gpu_driver = get_gpu_name_and_driver()
    nvcc_text = _run_cmd(["nvcc", "--version"])
    nvcc_release = parse_nvcc_release(nvcc_text)

    sizes = parse_sizes(args.sizes)
    filter_ids = parse_int_list(args.filter_ids)
    border_modes = parse_str_list(args.border_modes)
    precisions = parse_str_list(args.precisions)

    for fid in filter_ids:
        if fid < 1 or fid > 8:
            raise ValueError(f"Invalid filter_id {fid}; expected 1..8")
    for b in border_modes:
        if b not in BORDER_MAP:
            raise ValueError(f"Invalid border mode {b!r}")
    for p in precisions:
        if p not in PRECISION_MAP:
            raise ValueError(f"Invalid precision {p!r}")

    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "lib_path": str(lib_path),
        "cuda_runtime_version": decode_cuda_version(runtime_v.value),
        "cuda_driver_version": decode_cuda_version(driver_v.value),
        "cuda_device_id": device_id.value,
        "gpu_name": gpu_name,
        "gpu_driver": gpu_driver,
        "nvcc_release": nvcc_release,
    }

    rows: list[dict[str, str | int | float]] = []
    cases = build_cases(sizes, filter_ids, border_modes, precisions)
    for i, case in enumerate(cases, 1):
        result = run_case(
            cudart=cudart,
            core_lib=core_lib,
            case=case,
            border_const=args.border_const,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        )
        row = dict(env)
        row["case_index"] = i
        row["case_total"] = len(cases)
        row.update(result)
        rows.append(row)
        print(
            f"[{i}/{len(cases)}] {case.width}x{case.height} "
            f"f{case.filter_id} {case.border_mode}/{case.precision}: "
            f"p50={result['latency_ms_p50']:.3f} ms p95={result['latency_ms_p95']:.3f} ms "
            f"tput={result['throughput_mpix_per_s_p50']:.1f} MPix/s"
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote benchmark CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
