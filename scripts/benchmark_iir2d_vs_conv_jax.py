import argparse
import csv
import time
from typing import Callable, Dict, List

import jax
import jax.numpy as jnp
from jax import lax

from iir2d_jax import iir2d


def gaussian_kernel_1d(size: int, sigma: float, dtype=jnp.float32) -> jnp.ndarray:
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    radius = size // 2
    x = jnp.arange(-radius, radius + 1, dtype=dtype)
    k = jnp.exp(-(x**2) / (2.0 * sigma * sigma))
    return k / jnp.sum(k)


def separable_conv_nhwc(x: jnp.ndarray, k1d: jnp.ndarray, repeats: int = 1) -> jnp.ndarray:
    c = x.shape[-1]
    kh = k1d.shape[0]
    # Depthwise kernels in HWIO format: input-channels-per-group=1, output-channels=channels.
    kx = jnp.tile(k1d.reshape((1, kh, 1, 1)), (1, 1, 1, c))
    ky = jnp.tile(k1d.reshape((kh, 1, 1, 1)), (1, 1, 1, c))
    dn = ("NHWC", "HWIO", "NHWC")
    y = x
    for _ in range(repeats):
        y = lax.conv_general_dilated(
            y,
            kx,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=dn,
            feature_group_count=c,
        )
        y = lax.conv_general_dilated(
            y,
            ky,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=dn,
            feature_group_count=c,
        )
    return y


def iir_nhwc(x: jnp.ndarray, filter_id: int = 4, border: str = "mirror", precision: str = "f32") -> jnp.ndarray:
    n, h, w, c = x.shape
    x2d = jnp.transpose(x, (0, 3, 1, 2)).reshape((n * c, h, w))
    y2d = iir2d(x2d, filter_id=filter_id, border=border, precision=precision)
    return jnp.transpose(y2d.reshape((n, c, h, w)), (0, 2, 3, 1))


def benchmark(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    jit_fn = jax.jit(fn)
    y = jit_fn(x)
    y.block_until_ready()
    for _ in range(warmup):
        y = jit_fn(x)
        y.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = jit_fn(x)
        y.block_until_ready()
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1000.0 / iters
    pix = int(x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])
    return {
        "latency_ms": ms,
        "mpps": pix / (ms * 1e3),
        "mean": float(jnp.mean(y)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark iir2d custom call vs separable conv baselines")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="")
    args = parser.parse_args()

    print("jax", jax.__version__)
    print("devices", jax.devices())

    key = jax.random.PRNGKey(args.seed)
    x = jax.random.uniform(
        key,
        shape=(args.batch, args.height, args.width, args.channels),
        dtype=jnp.float32,
    )

    ref_kernel = gaussian_kernel_1d(15, sigma=3.0)
    ref_fn = lambda t: separable_conv_nhwc(t, ref_kernel, repeats=1)
    ref = jax.jit(ref_fn)(x)
    ref.block_until_ready()

    methods: Dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
        "iir_filter4": lambda t: iir_nhwc(t, filter_id=4, border="mirror", precision="f32"),
        "sepconv5x5_x4": lambda t: separable_conv_nhwc(t, gaussian_kernel_1d(5, 1.1), repeats=4),
        "sepconv9x9_x2": lambda t: separable_conv_nhwc(t, gaussian_kernel_1d(9, 2.2), repeats=2),
    }

    rows: List[Dict[str, float]] = []
    for name, fn in methods.items():
        stats = benchmark(fn, x, args.warmup, args.iters)
        y = jax.jit(fn)(x)
        y.block_until_ready()
        mse = float(jnp.mean((y - ref) ** 2))
        row = {
            "method": name,
            "latency_ms": stats["latency_ms"],
            "mpps": stats["mpps"],
            "mse_vs_ref": mse,
            "output_mean": stats["mean"],
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: r["latency_ms"])
    print("\nmethod,latency_ms,mpps,mse_vs_ref,output_mean")
    for r in rows:
        print(f"{r['method']},{r['latency_ms']:.3f},{r['mpps']:.3f},{r['mse_vs_ref']:.8f},{r['output_mean']:.8f}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["method", "latency_ms", "mpps", "mse_vs_ref", "output_mean"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
