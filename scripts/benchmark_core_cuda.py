#!/usr/bin/env python3
"""Reproducible benchmark harness for the IIR2D CUDA core C API."""

from __future__ import annotations

import argparse
import csv
import ctypes
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__:
    from .core_harness import (
        BORDER_MAP,
        CUDA_MEMCPY_HOST_TO_DEVICE,
        PRECISION_MAP,
        Case,
        IIR2D_Params,
        configure_core_lib,
        configure_cudart,
        cuda_check,
        decode_cuda_version,
        find_cudart,
        get_gpu_name_and_driver,
        load_core_library,
        parse_case_matrix,
        parse_nvcc_release,
        run_cmd,
    )
else:
    from core_harness import (
        BORDER_MAP,
        CUDA_MEMCPY_HOST_TO_DEVICE,
        PRECISION_MAP,
        Case,
        IIR2D_Params,
        configure_core_lib,
        configure_cudart,
        cuda_check,
        decode_cuda_version,
        find_cudart,
        get_gpu_name_and_driver,
        load_core_library,
        parse_case_matrix,
        parse_nvcc_release,
        run_cmd,
    )


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
    core_lib, lib_path = load_core_library(repo_root)
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
    nvcc_release = parse_nvcc_release(run_cmd(["nvcc", "--version"]))

    cases = parse_case_matrix(
        sizes_arg=args.sizes,
        filter_ids_arg=args.filter_ids,
        border_modes_arg=args.border_modes,
        precisions_arg=args.precisions,
    )

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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
