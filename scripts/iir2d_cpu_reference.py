#!/usr/bin/env python3
"""Canonical scalar CPU reference for IIR2D filter semantics."""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np


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

TOLERANCE_MAP = {
    "f32": (5e-3, 5e-3),
    "mixed": (7e-3, 7e-3),
    "f64": (5e-8, 5e-8),
}


def _normalize_border_mode(border_mode: int | str) -> int:
    if isinstance(border_mode, str):
        key = border_mode.strip().lower()
        if key not in BORDER_MAP:
            raise ValueError(f"Invalid border mode {border_mode!r}; expected one of {sorted(BORDER_MAP)}")
        return BORDER_MAP[key]
    if border_mode not in (0, 1, 2, 3):
        raise ValueError(f"Invalid border mode {border_mode}; expected 0..3")
    return int(border_mode)


def _normalize_precision(precision: int | str) -> str:
    if isinstance(precision, str):
        key = precision.strip().lower()
        if key not in PRECISION_MAP:
            raise ValueError(f"Invalid precision {precision!r}; expected one of {sorted(PRECISION_MAP)}")
        return key
    inv = {v: k for k, v in PRECISION_MAP.items()}
    if precision not in inv:
        raise ValueError(f"Invalid precision {precision}; expected 0..2")
    return inv[int(precision)]


def _border_sample(row: np.ndarray, idx: int, border_mode: int, border_const: np.generic) -> np.generic:
    n = int(row.shape[0])
    if 0 <= idx < n:
        return row[idx]
    if border_mode == BORDER_MAP["constant"]:
        return border_const
    if border_mode == BORDER_MAP["clamp"]:
        return row[0 if idx < 0 else (n - 1)]
    if border_mode == BORDER_MAP["wrap"]:
        m = idx % n
        if m < 0:
            m += n
        return row[m]
    period = n * 2
    m = idx % period
    if m < 0:
        m += period
    if m >= n:
        m = period - 1 - m
    return row[m]


def _row_first(
    in_row: np.ndarray,
    b0: np.generic,
    b1: np.generic,
    a1: np.generic,
    border_mode: int,
    border_const: np.generic,
    acc: Callable[[float | np.generic], np.generic],
    out_dtype: np.dtype,
) -> np.ndarray:
    x = in_row.astype(acc, copy=False)
    out = np.empty_like(in_row, dtype=out_dtype)
    xm1 = _border_sample(x, -1, border_mode, border_const)
    xm2 = _border_sample(x, -2, border_mode, border_const)
    y_prev = b0 * xm1 + b1 * xm2
    x_prev = xm1
    for i in range(x.shape[0]):
        xi = x[i]
        yi = b0 * xi + b1 * x_prev + a1 * y_prev
        out[i] = yi
        x_prev = xi
        y_prev = yi
    return out


def _row_biquad(
    in_row: np.ndarray,
    b0: np.generic,
    b1: np.generic,
    b2: np.generic,
    a1: np.generic,
    a2: np.generic,
    border_mode: int,
    border_const: np.generic,
    acc: Callable[[float | np.generic], np.generic],
    out_dtype: np.dtype,
) -> np.ndarray:
    x = in_row.astype(acc, copy=False)
    out = np.empty_like(in_row, dtype=out_dtype)
    xm1 = _border_sample(x, -1, border_mode, border_const)
    xm2 = _border_sample(x, -2, border_mode, border_const)
    xm3 = _border_sample(x, -3, border_mode, border_const)
    xm4 = _border_sample(x, -4, border_mode, border_const)
    y2 = b0 * xm2 + b1 * xm3 + b2 * xm4
    y1 = b0 * xm1 + b1 * xm2 + b2 * xm3 + a1 * y2
    x1 = xm1
    x2 = xm2
    for i in range(x.shape[0]):
        xi = x[i]
        yi = b0 * xi + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2
        out[i] = yi
        x2 = x1
        x1 = xi
        y2 = y1
        y1 = yi
    return out


def _row_biquad_scan_contract(
    in_row: np.ndarray,
    b0: np.generic,
    b1: np.generic,
    b2: np.generic,
    a1: np.generic,
    a2: np.generic,
    border_mode: int,
    border_const: np.generic,
    acc: Callable[[float | np.generic], np.generic],
    out_dtype: np.dtype,
    block_width: int = 256,
) -> np.ndarray:
    """Mirror the shipped block-transform scan contract used by CUDA for f3/f4/f8."""
    x = in_row.astype(acc, copy=False)
    n = int(x.shape[0])
    out = np.empty_like(in_row, dtype=out_dtype)

    xm1 = _border_sample(x, -1, border_mode, border_const)
    xm2 = _border_sample(x, -2, border_mode, border_const)
    xm3 = _border_sample(x, -3, border_mode, border_const)
    xm4 = _border_sample(x, -4, border_mode, border_const)
    y2 = b0 * xm2 + b1 * xm3 + b2 * xm4
    y1 = b0 * xm1 + b1 * xm2 + b2 * xm3 + a1 * y2
    state = np.array([xm1, xm2, y1, y2], dtype=x.dtype)

    zero = acc(0.0)
    one = acc(1.0)
    for start in range(0, n, block_width):
        end = min(start + block_width, n)
        scan_A: list[np.ndarray] = []
        scan_b: list[np.ndarray] = []
        for idx in range(start, end):
            xi = x[idx]
            A = np.array(
                [
                    [zero, zero, zero, zero],
                    [one, zero, zero, zero],
                    [b1, b2, a1, a2],
                    [zero, zero, one, zero],
                ],
                dtype=x.dtype,
            )
            b = np.array([xi, zero, b0 * xi, zero], dtype=x.dtype)
            if scan_A:
                left_A = scan_A[-1]
                left_b = scan_b[-1]
                out_A = left_A @ A
                out_b = left_A @ b + left_b
            else:
                out_A = A
                out_b = b
            scan_A.append(out_A)
            scan_b.append(out_b)

        block_in = state
        for j in range(end - start):
            y_vec = scan_A[j] @ block_in + scan_b[j]
            out[start + j] = y_vec[2]
        state = scan_A[-1] @ state + scan_b[-1]
    return out


def _row_fwd_bwd_first(
    in_row: np.ndarray,
    b0: np.generic,
    b1: np.generic,
    a1: np.generic,
    border_mode: int,
    border_const: np.generic,
    acc: Callable[[float | np.generic], np.generic],
    out_dtype: np.dtype,
) -> np.ndarray:
    x = in_row.astype(acc, copy=False)
    out = np.empty_like(in_row, dtype=out_dtype)
    xm1 = _border_sample(x, -1, border_mode, border_const)
    xm2 = _border_sample(x, -2, border_mode, border_const)
    y_prev = b0 * xm1 + b1 * xm2
    x_prev = xm1
    for i in range(x.shape[0]):
        xi = x[i]
        yi = b0 * xi + b1 * x_prev + a1 * y_prev
        out[i] = yi
        x_prev = xi
        y_prev = yi

    y = out.astype(acc, copy=False)
    xp1 = _border_sample(y, y.shape[0], border_mode, border_const)
    xp2 = _border_sample(y, y.shape[0] + 1, border_mode, border_const)
    y_prev = b0 * xp1 + b1 * xp2
    x_prev = xp1
    for i in range(y.shape[0] - 1, -1, -1):
        xi = y[i]
        yi = b0 * xi + b1 * x_prev + a1 * y_prev
        y[i] = yi
        x_prev = xi
        y_prev = yi
    return y.astype(out_dtype, copy=False)


def _row_statespace(
    in_row: np.ndarray,
    b0: np.generic,
    b1: np.generic,
    b2: np.generic,
    a1: np.generic,
    a2: np.generic,
    border_mode: int,
    border_const: np.generic,
    acc: Callable[[float | np.generic], np.generic],
    out_dtype: np.dtype,
) -> np.ndarray:
    x = in_row.astype(acc, copy=False)
    out = np.empty_like(in_row, dtype=out_dtype)
    xm2 = _border_sample(x, -2, border_mode, border_const)
    xm1 = _border_sample(x, -1, border_mode, border_const)
    z1 = acc(0.0)
    z2 = acc(0.0)
    for xi in (xm2, xm1):
        yi = b0 * xi + z1
        z1 = b1 * xi + z2 + a1 * yi
        z2 = b2 * xi + a2 * yi
    for i in range(x.shape[0]):
        xi = x[i]
        yi = b0 * xi + z1
        z1 = b1 * xi + z2 + a1 * yi
        z2 = b2 * xi + a2 * yi
        out[i] = yi
    return out


def _row_deriche(
    in_row: np.ndarray,
    a0: np.generic,
    a1: np.generic,
    a2: np.generic,
    a3: np.generic,
    b1: np.generic,
    b2: np.generic,
    c1: np.generic,
    border_mode: int,
    border_const: np.generic,
    acc: Callable[[float | np.generic], np.generic],
    out_dtype: np.dtype,
    backward_on_forward: bool = False,
) -> np.ndarray:
    x = in_row.astype(acc, copy=False)
    yp = np.empty_like(in_row, dtype=out_dtype)
    yn = np.empty_like(in_row, dtype=out_dtype)

    xm1 = _border_sample(x, -1, border_mode, border_const)
    xm2 = _border_sample(x, -2, border_mode, border_const)
    xm3 = _border_sample(x, -3, border_mode, border_const)
    ym2 = a0 * xm2 + a1 * xm3
    ym1 = a0 * xm1 + a1 * xm2 + b1 * ym2
    for i in range(x.shape[0]):
        xi = x[i]
        yi = a0 * xi + a1 * xm1 + b1 * ym1 + b2 * ym2
        yp[i] = yi
        xm1 = xi
        ym2 = ym1
        ym1 = yi

    backward_src = yp.astype(acc, copy=False) if backward_on_forward else x
    xp1 = _border_sample(backward_src, backward_src.shape[0], border_mode, border_const)
    xp2 = _border_sample(backward_src, backward_src.shape[0] + 1, border_mode, border_const)
    xp3 = _border_sample(backward_src, backward_src.shape[0] + 2, border_mode, border_const)
    yn2 = a2 * xp2 + a3 * xp3
    yn1 = a2 * xp1 + a3 * xp2 + b1 * yn2
    for i in range(backward_src.shape[0] - 1, -1, -1):
        xi = backward_src[i]
        yi = a2 * xp1 + a3 * xp2 + b1 * yn1 + b2 * yn2
        yn[i] = yi
        xp2 = xp1
        xp1 = xi
        yn2 = yn1
        yn1 = yi

    yp_acc = yp.astype(acc, copy=False)
    yn_acc = yn.astype(acc, copy=False)
    out = c1 * (yp_acc + yn_acc)
    return out.astype(out_dtype, copy=False)


def _apply_rows(
    image: np.ndarray,
    filter_id: int,
    border_mode: int,
    border_const: np.generic,
    acc: Callable[[float | np.generic], np.generic],
    out_dtype: np.dtype,
    deriche_backward_on_forward: bool = False,
) -> np.ndarray:
    out = np.empty_like(image, dtype=out_dtype)
    if filter_id in (1, 7):
        alpha = acc(0.85)
        b0 = acc(1.0) - alpha
        b1 = acc(0.0)
        a1 = alpha
        for r in range(image.shape[0]):
            out[r] = _row_first(image[r], b0, b1, a1, border_mode, border_const, acc, out_dtype)
        return out

    if filter_id == 2:
        a = acc(0.75)
        b = acc(0.25)
        tmp = np.empty_like(image, dtype=out_dtype)
        for r in range(image.shape[0]):
            tmp[r] = _row_first(image[r], b, acc(0.0), a, border_mode, border_const, acc, out_dtype)
        for r in range(image.shape[0]):
            out[r] = _row_first(tmp[r], b, acc(0.0), a, border_mode, border_const, acc, out_dtype)
        return out

    if filter_id == 3:
        b0 = acc(0.2)
        b1 = acc(0.2)
        b2 = acc(0.2)
        a1 = acc(0.3)
        a2 = acc(-0.1)
        for r in range(image.shape[0]):
            out[r] = _row_biquad_scan_contract(
                image[r], b0, b1, b2, a1, a2, border_mode, border_const, acc, out_dtype
            )
        return out

    if filter_id == 4:
        tmp = np.empty_like(image, dtype=out_dtype)
        for r in range(image.shape[0]):
            tmp[r] = _row_biquad_scan_contract(
                image[r],
                acc(0.2),
                acc(0.2),
                acc(0.2),
                acc(0.3),
                acc(-0.1),
                border_mode,
                border_const,
                acc,
                out_dtype,
            )
        for r in range(image.shape[0]):
            out[r] = _row_biquad_scan_contract(
                tmp[r],
                acc(0.3),
                acc(0.1),
                acc(0.1),
                acc(0.2),
                acc(-0.05),
                border_mode,
                border_const,
                acc,
                out_dtype,
            )
        return out

    if filter_id == 5:
        b0 = acc(0.4)
        b1 = acc(0.0)
        a1 = acc(0.6)
        for r in range(image.shape[0]):
            out[r] = _row_fwd_bwd_first(image[r], b0, b1, a1, border_mode, border_const, acc, out_dtype)
        return out

    if filter_id == 6:
        der_sigma = acc(2.0)
        alpha = acc(1.695) / der_sigma
        ema = acc(np.exp(-float(alpha)))
        ema2 = acc(np.exp(-2.0 * float(alpha)))
        k = (acc(1.0) - ema) * (acc(1.0) - ema) / (acc(1.0) + acc(2.0) * alpha * ema - ema2)
        a0 = k
        a1 = k * (alpha - acc(1.0)) * ema
        a2 = k * (alpha + acc(1.0)) * ema
        a3 = -k * ema2
        b1 = acc(2.0) * ema
        b2 = -ema2
        c1 = acc(1.0)
        for r in range(image.shape[0]):
            out[r] = _row_deriche(
                image[r],
                a0,
                a1,
                a2,
                a3,
                b1,
                b2,
                c1,
                border_mode,
                border_const,
                acc,
                out_dtype,
                backward_on_forward=deriche_backward_on_forward,
            )
        return out

    if filter_id == 8:
        for r in range(image.shape[0]):
            out[r] = _row_biquad_scan_contract(
                image[r],
                acc(0.2),
                acc(0.2),
                acc(0.2),
                acc(0.3),
                acc(-0.1),
                border_mode,
                border_const,
                acc,
                out_dtype,
            )
        return out

    raise ValueError(f"Invalid filter_id {filter_id}; expected 1..8")


def iir2d_cpu_reference(
    image: np.ndarray,
    filter_id: int,
    border_mode: int | str = "mirror",
    border_const: float = 0.0,
    precision: int | str = "f32",
) -> np.ndarray:
    """Run canonical CPU reference over a 2D image."""
    x = np.asarray(image)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D input (H, W), got shape {x.shape}")
    if x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError(f"Expected positive dimensions, got {x.shape}")
    if int(filter_id) < 1 or int(filter_id) > 8:
        raise ValueError(f"Invalid filter_id {filter_id}; expected 1..8")

    border_mode_i = _normalize_border_mode(border_mode)
    precision_s = _normalize_precision(precision)

    if precision_s == "f64":
        io_dtype = np.float64
        acc = np.float64
    elif precision_s == "mixed":
        io_dtype = np.float32
        acc = np.float64
    else:
        io_dtype = np.float32
        acc = np.float32

    x_io = np.asarray(x, dtype=io_dtype, order="C")
    bconst = acc(border_const)

    row_pass = _apply_rows(
        x_io,
        int(filter_id),
        border_mode_i,
        bconst,
        acc,
        io_dtype,
        deriche_backward_on_forward=False,
    )
    transposed = np.ascontiguousarray(row_pass.T)
    col_pass_t = _apply_rows(
        transposed,
        int(filter_id),
        border_mode_i,
        bconst,
        acc,
        io_dtype,
        deriche_backward_on_forward=(int(filter_id) == 6),
    )
    return np.ascontiguousarray(col_pass_t.T)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run canonical IIR2D CPU reference on random input.")
    ap.add_argument("--size", default="32x32", help="Input image size WxH, e.g. 63x47")
    ap.add_argument("--filter_id", type=int, default=4)
    ap.add_argument("--border_mode", default="mirror", choices=sorted(BORDER_MAP))
    ap.add_argument("--precision", default="f32", choices=sorted(PRECISION_MAP))
    ap.add_argument("--border_const", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    try:
        w_str, h_str = args.size.lower().split("x")
        width = int(w_str)
        height = int(h_str)
    except Exception as exc:
        raise ValueError(f"Invalid --size {args.size!r}; expected WxH") from exc

    rng = np.random.default_rng(args.seed)
    if args.precision == "f64":
        x = rng.random((height, width), dtype=np.float64)
    else:
        x = rng.random((height, width), dtype=np.float32)
    y = iir2d_cpu_reference(
        x,
        filter_id=args.filter_id,
        border_mode=args.border_mode,
        border_const=args.border_const,
        precision=args.precision,
    )
    rtol, atol = TOLERANCE_MAP[args.precision]
    print(
        f"ok: shape={y.shape} dtype={y.dtype} filter={args.filter_id} "
        f"border={args.border_mode} precision={args.precision} "
        f"tol(rtol={rtol}, atol={atol})"
    )
    print(f"stats: min={float(np.min(y)):.6f} max={float(np.max(y)):.6f} mean={float(np.mean(y)):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
