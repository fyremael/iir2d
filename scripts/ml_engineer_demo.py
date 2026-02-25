import argparse
import csv
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import optax

from iir2d_jax import iir2d, iir2d_vjp

_HAS_FLAX = False
_FLAX_IMPORT_ERROR = ""
try:
    if not hasattr(jax.config, "define_bool_state"):
        def _define_bool_state(name: str, default: bool, help_str: str):
            del help_str
            setattr(jax.config, name, default)
            return default
        jax.config.define_bool_state = _define_bool_state
    if not hasattr(jax, "linear_util"):
        from jax._src import linear_util as _linear_util
        jax.linear_util = _linear_util

    import flax.linen as nn  # type: ignore
    from flax.training import train_state  # type: ignore
    from iir2d_jax.flax_layer import IIRDenoiseStem  # type: ignore
    _HAS_FLAX = True
except Exception as exc:  # pragma: no cover - environment dependent
    _FLAX_IMPORT_ERROR = str(exc)


def iir_nhwc(x: jnp.ndarray, filter_id: int = 4, differentiable: bool = False) -> jnp.ndarray:
    n, h, w, c = x.shape
    x2d = jnp.transpose(x, (0, 3, 1, 2)).reshape((n * c, h, w))
    op = iir2d_vjp if differentiable else iir2d
    y2d = op(x2d, filter_id=filter_id, border="mirror", precision="f32")
    return jnp.transpose(y2d.reshape((n, c, h, w)), (0, 2, 3, 1))


def conv2d_nhwc(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    return lax.conv_general_dilated(
        x,
        w,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


def psnr(x: jnp.ndarray, y: jnp.ndarray, peak: float = 1.0) -> jnp.ndarray:
    mse = jnp.mean((x - y) ** 2)
    return 20.0 * jnp.log10(peak) - 10.0 * jnp.log10(jnp.maximum(mse, 1e-12))


@dataclass
class Config:
    steps: int
    batch: int
    size: int
    lr: float


def synthetic_batch(key: jnp.ndarray, batch: int, size: int) -> Dict[str, jnp.ndarray]:
    k1, k2 = jax.random.split(key)
    clean = jax.random.uniform(k1, (batch, size, size, 1), minval=0.0, maxval=1.0, dtype=jnp.float32)
    clean = iir_nhwc(clean, filter_id=4, differentiable=False)
    noise = 0.08 * jax.random.normal(k2, clean.shape, dtype=jnp.float32)
    noisy = jnp.clip(clean + noise, 0.0, 1.0)
    return {"noisy": noisy, "clean": clean}


# -----------------------
# Flax backend
# -----------------------
if _HAS_FLAX:
    class BaselineConv(nn.Module):
        channels: int = 32

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            h = nn.Conv(self.channels, (3, 3), padding="SAME", use_bias=False)(x)
            h = nn.gelu(h)
            h = nn.Conv(x.shape[-1], (3, 3), padding="SAME", use_bias=False)(h)
            return x + h


def train_model_flax(model, cfg: Config, seed: int):
        rng = jax.random.PRNGKey(seed)
        rng, init_key = jax.random.split(rng)
        params = model.init(init_key, jnp.zeros((cfg.batch, cfg.size, cfg.size, 1), jnp.float32))["params"]
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.adamw(learning_rate=cfg.lr, weight_decay=1e-5),
        )

        @jax.jit
        def step_fn(state, batch):
            def loss_fn(params):
                pred = state.apply_fn({"params": params}, batch["noisy"])
                return jnp.mean((pred - batch["clean"]) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        losses: List[float] = []
        t0 = time.perf_counter()
        for _ in range(cfg.steps):
            rng, bk = jax.random.split(rng)
            batch = synthetic_batch(bk, cfg.batch, cfg.size)
            state, loss = step_fn(state, batch)
            losses.append(float(loss))
        elapsed = time.perf_counter() - t0

        rng, ek = jax.random.split(rng)
        eval_batch = synthetic_batch(ek, cfg.batch, cfg.size)
        pred = jax.jit(lambda p, x: model.apply({"params": p}, x))(state.params, eval_batch["noisy"])
        pred.block_until_ready()

        return {
            "loss_start": losses[0],
            "loss_end": losses[-1],
            "train_time_s": elapsed,
            "psnr_noisy": float(psnr(eval_batch["clean"], eval_batch["noisy"])),
            "psnr_pred": float(psnr(eval_batch["clean"], pred)),
            "losses": losses,
        }


# -----------------------
# Pure JAX fallback backend
# -----------------------
def init_baseline_params(key: jnp.ndarray, hidden: int = 32) -> Dict[str, jnp.ndarray]:
    k1, k2 = jax.random.split(key)
    w1 = 0.05 * jax.random.normal(k1, (3, 3, 1, hidden), dtype=jnp.float32)
    w2 = 0.05 * jax.random.normal(k2, (3, 3, hidden, 1), dtype=jnp.float32)
    return {"w1": w1, "w2": w2}


def apply_baseline(params: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    h = conv2d_nhwc(x, params["w1"])
    h = jax.nn.gelu(h)
    h = conv2d_nhwc(h, params["w2"])
    return x + h


def init_iir_params(key: jnp.ndarray, hidden: int = 32, num_filters: int = 4) -> Dict[str, jnp.ndarray]:
    k1, k2, k3 = jax.random.split(key, 3)
    w1 = 0.05 * jax.random.normal(k1, (3, 3, 1, hidden), dtype=jnp.float32)
    w2 = 0.05 * jax.random.normal(k2, (3, 3, hidden, 1), dtype=jnp.float32)
    logits = 0.01 * jax.random.normal(k3, (num_filters, hidden), dtype=jnp.float32)
    return {"w1": w1, "w2": w2, "logits": logits}


def init_iir_frozen_params(key: jnp.ndarray, hidden: int = 32, num_filters: int = 4) -> Dict[str, jnp.ndarray]:
    k1, k2, k3 = jax.random.split(key, 3)
    w1 = 0.05 * jax.random.normal(k1, (3, 3, 1, hidden), dtype=jnp.float32)
    w2 = 0.05 * jax.random.normal(k2, (3, 3, hidden, 1), dtype=jnp.float32)
    # Frozen extractor mixes filter responses on single-channel input.
    logits = 0.01 * jax.random.normal(k3, (num_filters, 1), dtype=jnp.float32)
    return {"w1": w1, "w2": w2, "logits": logits}


def apply_iir_bank(params: Dict[str, jnp.ndarray], x: jnp.ndarray, filter_ids: Tuple[int, ...] = (1, 3, 4, 8)) -> jnp.ndarray:
    h = conv2d_nhwc(x, params["w1"])
    h = jax.nn.gelu(h)

    ys = [iir_nhwc(h, filter_id=int(fid), differentiable=True) for fid in filter_ids]
    stack = jnp.stack(ys, axis=0)  # [F, N, H, W, C]
    w = jax.nn.softmax(params["logits"], axis=0).reshape((len(filter_ids), 1, 1, 1, -1))
    h = jnp.sum(stack * w, axis=0)

    h = conv2d_nhwc(h, params["w2"])
    return x + h


def apply_iir_frozen_extractor(
    params: Dict[str, jnp.ndarray],
    x: jnp.ndarray,
    filter_ids: Tuple[int, ...] = (1, 3, 4, 8),
) -> jnp.ndarray:
    # Frozen feature extractor: no gradient through iir custom call.
    ys = [iir_nhwc(x, filter_id=int(fid), differentiable=False) for fid in filter_ids]
    stack = jnp.stack(ys, axis=0)
    w = jax.nn.softmax(params["logits"], axis=0).reshape((len(filter_ids), 1, 1, 1, -1))
    feat = jnp.sum(stack * w, axis=0)
    feat = lax.stop_gradient(feat)

    h = conv2d_nhwc(feat, params["w1"])
    h = jax.nn.gelu(h)
    h = conv2d_nhwc(h, params["w2"])
    return x + h


def train_model_jax(apply_fn, init_fn, cfg: Config, seed: int):
    rng = jax.random.PRNGKey(seed)
    rng, ik = jax.random.split(rng)
    params = init_fn(ik)
    opt = optax.adamw(cfg.lr, weight_decay=1e-5)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        def loss_fn(p):
            pred = apply_fn(p, batch["noisy"])
            return jnp.mean((pred - batch["clean"]) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses: List[float] = []
    t0 = time.perf_counter()
    for _ in range(cfg.steps):
        rng, bk = jax.random.split(rng)
        batch = synthetic_batch(bk, cfg.batch, cfg.size)
        params, opt_state, loss = step(params, opt_state, batch)
        losses.append(float(loss))
    elapsed = time.perf_counter() - t0

    rng, ek = jax.random.split(rng)
    eval_batch = synthetic_batch(ek, cfg.batch, cfg.size)
    pred = jax.jit(apply_fn)(params, eval_batch["noisy"])
    pred.block_until_ready()

    return {
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "train_time_s": elapsed,
        "psnr_noisy": float(psnr(eval_batch["clean"], eval_batch["noisy"])),
        "psnr_pred": float(psnr(eval_batch["clean"], pred)),
        "losses": losses,
    }


def latency_ms(fn, x, warmup=3, iters=10):
    jf = jax.jit(fn)
    y = jf(x)
    y.block_until_ready()
    for _ in range(warmup):
        y = jf(x)
        y.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = jf(x)
        y.block_until_ready()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def main() -> int:
    parser = argparse.ArgumentParser(description="Notebook-style demo for iir2d + model integration")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_history_csv", type=str, default="")
    parser.add_argument("--with_trainable_iir", action="store_true")
    args = parser.parse_args()

    cfg = Config(steps=args.steps, batch=args.batch, size=args.size, lr=args.lr)

    print("# Section 1: Runtime Snapshot")
    print("jax", jax.__version__)
    print("devices", jax.devices())

    key = jax.random.PRNGKey(args.seed)
    x = jax.random.normal(key, (cfg.batch, cfg.size, cfg.size, 1), dtype=jnp.float32)
    iir_lat = latency_ms(lambda t: iir_nhwc(t, filter_id=4), x)
    print(f"iir2d_latency_ms={iir_lat:.3f} @ shape={tuple(x.shape)}")

    print("\n# Section 2: Tiny Denoise Training (Conv vs Conv+IIR)")
    if _HAS_FLAX:
        backend = "flax"
        b = train_model_flax(BaselineConv(channels=32), cfg, seed=args.seed + 1)
        m = train_model_flax(IIRDenoiseStem(channels=32, filter_ids=(1, 3, 4, 8), precision="f32"), cfg, seed=args.seed + 2)
        frozen_result = train_model_flax(IIRDenoiseStem(channels=32, filter_ids=(1, 3, 4, 8), precision="f32"), cfg, seed=args.seed + 3)
    else:
        backend = f"pure_jax_fallback ({_FLAX_IMPORT_ERROR})"
        b = train_model_jax(apply_baseline, init_baseline_params, cfg, seed=args.seed + 1)
        m = train_model_jax(apply_iir_bank, init_iir_params, cfg, seed=args.seed + 2) if args.with_trainable_iir else None
        frozen_result = train_model_jax(apply_iir_frozen_extractor, init_iir_frozen_params, cfg, seed=args.seed + 3)
    print("backend", backend)

    rows = [
        {
            "model": "baseline_conv",
            "loss_start": b["loss_start"],
            "loss_end": b["loss_end"],
            "train_time_s": b["train_time_s"],
            "psnr_noisy": b["psnr_noisy"],
            "psnr_pred": b["psnr_pred"],
            "psnr_gain": b["psnr_pred"] - b["psnr_noisy"],
        },
        {
            "model": "conv_plus_iir_frozen",
            "loss_start": frozen_result["loss_start"],
            "loss_end": frozen_result["loss_end"],
            "train_time_s": frozen_result["train_time_s"],
            "psnr_noisy": frozen_result["psnr_noisy"],
            "psnr_pred": frozen_result["psnr_pred"],
            "psnr_gain": frozen_result["psnr_pred"] - frozen_result["psnr_noisy"],
        },
    ]
    if m is not None:
        rows.append(
            {
                "model": "conv_plus_iir_bank",
                "loss_start": m["loss_start"],
                "loss_end": m["loss_end"],
                "train_time_s": m["train_time_s"],
                "psnr_noisy": m["psnr_noisy"],
                "psnr_pred": m["psnr_pred"],
                "psnr_gain": m["psnr_pred"] - m["psnr_noisy"],
            }
        )

    print("model,loss_start,loss_end,train_time_s,psnr_noisy,psnr_pred,psnr_gain")
    for r in rows:
        print(
            f"{r['model']},{r['loss_start']:.6f},{r['loss_end']:.6f},"
            f"{r['train_time_s']:.3f},{r['psnr_noisy']:.3f},{r['psnr_pred']:.3f},{r['psnr_gain']:.3f}"
        )

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as out_file:
            writer = csv.DictWriter(
                out_file,
                fieldnames=[
                    "model",
                    "loss_start",
                    "loss_end",
                    "train_time_s",
                    "psnr_noisy",
                    "psnr_pred",
                    "psnr_gain",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {args.out_csv}")

    if args.out_history_csv:
        history_rows = []
        for step, loss in enumerate(b["losses"]):
            history_rows.append({"model": "baseline_conv", "step": step, "loss": float(loss)})
        for step, loss in enumerate(frozen_result["losses"]):
            history_rows.append({"model": "conv_plus_iir_frozen", "step": step, "loss": float(loss)})
        if m is not None:
            for step, loss in enumerate(m["losses"]):
                history_rows.append({"model": "conv_plus_iir_bank", "step": step, "loss": float(loss)})

        with open(args.out_history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "step", "loss"])
            writer.writeheader()
            writer.writerows(history_rows)
        print(f"Saved history CSV: {args.out_history_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
