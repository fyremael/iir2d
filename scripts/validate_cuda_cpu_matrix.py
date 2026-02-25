#!/usr/bin/env python3
"""Validate CUDA core output against canonical CPU reference across a case matrix."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np

if __package__:
    from .core_harness import (
        BORDER_MAP,
        CUDA_MEMCPY_DEVICE_TO_HOST,
        CUDA_MEMCPY_HOST_TO_DEVICE,
        PRECISION_MAP,
        IIR2D_Params,
        configure_core_lib,
        configure_cudart,
        cuda_check,
        find_cudart,
        load_core_library,
        parse_case_matrix,
    )
    from .iir2d_cpu_reference import TOLERANCE_MAP, iir2d_cpu_reference
else:
    from core_harness import (
        BORDER_MAP,
        CUDA_MEMCPY_DEVICE_TO_HOST,
        CUDA_MEMCPY_HOST_TO_DEVICE,
        PRECISION_MAP,
        IIR2D_Params,
        configure_core_lib,
        configure_cudart,
        cuda_check,
        find_cudart,
        load_core_library,
        parse_case_matrix,
    )
    from iir2d_cpu_reference import TOLERANCE_MAP, iir2d_cpu_reference


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

    cases = parse_case_matrix(
        sizes_arg=args.sizes,
        filter_ids_arg=args.filter_ids,
        border_modes_arg=args.border_modes,
        precisions_arg=args.precisions,
    )
    tolerances = {
        "f32": (args.rtol_f32, args.atol_f32),
        "mixed": (args.rtol_mixed, args.atol_mixed),
        "f64": (args.rtol_f64, args.atol_f64),
    }

    repo_root = Path(__file__).resolve().parents[1]
    core_lib, _ = load_core_library(repo_root)
    cudart = find_cudart()
    configure_core_lib(core_lib)
    configure_cudart(cudart)

    failures = 0
    for idx, case in enumerate(cases, 1):
        dtype = np.float64 if case.precision == "f64" else np.float32
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
