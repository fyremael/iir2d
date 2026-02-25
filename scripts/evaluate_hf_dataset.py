import argparse
import csv
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import optax
from datasets import load_dataset

from iir2d_jax import iir2d, iir2d_vjp


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


def to_np_image(sample, image_key: str, channels: int) -> np.ndarray:
    img = sample[image_key]
    if channels == 1:
        img = img.convert("L")
        arr = np.asarray(img, dtype=np.float32)[..., None]
    else:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
    return arr / 255.0


def random_crop(rng: np.random.Generator, img: np.ndarray, patch: int) -> np.ndarray:
    h, w, c = img.shape
    if h < patch or w < patch:
        scale = max(patch / h, patch / w)
        nh, nw = int(np.ceil(h * scale)), int(np.ceil(w * scale))
        arr = np.clip(img.squeeze(-1) if c == 1 else img, 0.0, 1.0)
        arr_u8 = (arr * 255.0).astype(np.uint8)
        from PIL import Image
        pil = Image.fromarray(arr_u8)
        pil = pil.resize((nw, nh), Image.BILINEAR)
        img = np.asarray(pil, dtype=np.float32) / 255.0
        if c == 1:
            img = img[..., None]
        h, w, c = img.shape
    y0 = rng.integers(0, h - patch + 1)
    x0 = rng.integers(0, w - patch + 1)
    return img[y0:y0 + patch, x0:x0 + patch, :]


def make_noisy(clean: np.ndarray, sigma: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    noise = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
    noisy = np.clip(clean + noise, 0.0, 1.0)
    return noisy, clean


@dataclass
class Config:
    steps: int
    batch: int
    patch: int
    lr: float
    noise_sigma: float
    channels: int


def init_baseline_params(key: jnp.ndarray, in_ch: int, hidden: int = 32) -> Dict[str, jnp.ndarray]:
    k1, k2 = jax.random.split(key)
    w1 = 0.05 * jax.random.normal(k1, (3, 3, in_ch, hidden), dtype=jnp.float32)
    w2 = 0.05 * jax.random.normal(k2, (3, 3, hidden, in_ch), dtype=jnp.float32)
    return {"w1": w1, "w2": w2}


def apply_baseline(params: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    h = conv2d_nhwc(x, params["w1"])
    h = jax.nn.gelu(h)
    h = conv2d_nhwc(h, params["w2"])
    return x + h


def init_iir_frozen_params(key: jnp.ndarray, in_ch: int, hidden: int = 32, num_filters: int = 4) -> Dict[str, jnp.ndarray]:
    k1, k2, k3 = jax.random.split(key, 3)
    w1 = 0.05 * jax.random.normal(k1, (3, 3, in_ch, hidden), dtype=jnp.float32)
    w2 = 0.05 * jax.random.normal(k2, (3, 3, hidden, in_ch), dtype=jnp.float32)
    logits = 0.01 * jax.random.normal(k3, (num_filters, in_ch), dtype=jnp.float32)
    return {"w1": w1, "w2": w2, "logits": logits}


def init_iir_residual_params(
    key: jnp.ndarray,
    in_ch: int,
    hidden: int = 32,
    num_filters: int = 4,
) -> Dict[str, jnp.ndarray]:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    w1 = 0.05 * jax.random.normal(k1, (3, 3, in_ch, hidden), dtype=jnp.float32)
    w2 = 0.05 * jax.random.normal(k2, (3, 3, hidden, in_ch), dtype=jnp.float32)
    # Mixture over fixed stable iir kernels.
    logits = 0.01 * jax.random.normal(k3, (num_filters, in_ch), dtype=jnp.float32)
    # 1x1 projection on iir branch to adapt channel responses.
    w_proj = 0.05 * jax.random.normal(k4, (1, 1, in_ch, in_ch), dtype=jnp.float32)
    # Learnable residual gain, constrained by sigmoid in forward.
    gain_logit = jnp.zeros((in_ch,), dtype=jnp.float32)
    return {"w1": w1, "w2": w2, "logits": logits, "w_proj": w_proj, "gain_logit": gain_logit}


def apply_iir_frozen(params: Dict[str, jnp.ndarray], x: jnp.ndarray, filter_ids: Tuple[int, ...] = (1, 3, 4, 8)) -> jnp.ndarray:
    ys = [iir_nhwc(x, filter_id=int(fid), differentiable=False) for fid in filter_ids]
    stack = jnp.stack(ys, axis=0)
    w = jax.nn.softmax(params["logits"], axis=0).reshape((len(filter_ids), 1, 1, 1, -1))
    feat = jnp.sum(stack * w, axis=0)
    feat = lax.stop_gradient(feat)
    h = conv2d_nhwc(feat, params["w1"])
    h = jax.nn.gelu(h)
    h = conv2d_nhwc(h, params["w2"])
    return x + h


def apply_iir_residual(
    params: Dict[str, jnp.ndarray],
    x: jnp.ndarray,
    filter_ids: Tuple[int, ...] = (1, 3, 4, 8),
) -> jnp.ndarray:
    # Main learned conv trunk.
    trunk = conv2d_nhwc(x, params["w1"])
    trunk = jax.nn.gelu(trunk)
    trunk = conv2d_nhwc(trunk, params["w2"])

    # Auxiliary iir branch: fixed stable filters, learnable mixture + projection + gated residual.
    ys = [iir_nhwc(x, filter_id=int(fid), differentiable=False) for fid in filter_ids]
    stack = jnp.stack(ys, axis=0)
    w = jax.nn.softmax(params["logits"], axis=0).reshape((len(filter_ids), 1, 1, 1, -1))
    iir_mix = jnp.sum(stack * w, axis=0)
    iir_feat = conv2d_nhwc(iir_mix, params["w_proj"])
    gain = jax.nn.sigmoid(params["gain_logit"]).reshape((1, 1, 1, -1))
    return x + trunk + gain * iir_feat


def build_batch(ds, image_key: str, cfg: Config, rng: np.random.Generator) -> Dict[str, jnp.ndarray]:
    noisy_batch = []
    clean_batch = []
    n = len(ds)
    for _ in range(cfg.batch):
        idx = int(rng.integers(0, n))
        img = to_np_image(ds[idx], image_key=image_key, channels=cfg.channels)
        crop = random_crop(rng, img, cfg.patch)
        noisy, clean = make_noisy(crop, cfg.noise_sigma, rng)
        noisy_batch.append(noisy)
        clean_batch.append(clean)
    noisy = jnp.asarray(np.stack(noisy_batch, axis=0), dtype=jnp.float32)
    clean = jnp.asarray(np.stack(clean_batch, axis=0), dtype=jnp.float32)
    return {"noisy": noisy, "clean": clean}


def train_model(apply_fn, params, cfg: Config, train_ds, val_ds, image_key: str, seed: int):
    opt = optax.adamw(cfg.lr, weight_decay=1e-5)
    opt_state = opt.init(params)
    npr = np.random.default_rng(seed)

    @jax.jit
    def step_fn(params, opt_state, batch):
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
        batch = build_batch(train_ds, image_key=image_key, cfg=cfg, rng=npr)
        params, opt_state, loss = step_fn(params, opt_state, batch)
        losses.append(float(loss))
    elapsed = time.perf_counter() - t0

    val_batch = build_batch(val_ds, image_key=image_key, cfg=cfg, rng=npr)
    pred = jax.jit(apply_fn)(params, val_batch["noisy"])
    pred.block_until_ready()

    return {
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "train_time_s": elapsed,
        "psnr_noisy": float(psnr(val_batch["clean"], val_batch["noisy"])),
        "psnr_pred": float(psnr(val_batch["clean"], pred)),
        "losses": losses,
    }


def infer_image_key(ds) -> str:
    for k in ds.column_names:
        v = ds[0][k]
        if hasattr(v, "convert"):
            return k
    raise RuntimeError(f"Could not infer image column from columns: {ds.column_names}")


def main() -> int:
    parser = argparse.ArgumentParser(description="HF dataset training run for baseline vs frozen-iir")
    parser.add_argument("--dataset", default="food101")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--train_max_samples", type=int, default=0)
    parser.add_argument("--val_max_samples", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--noise_sigma", type=float, default=0.08)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--out_history_csv", default="")
    args = parser.parse_args()

    print("Loading dataset", args.dataset)
    train_ds = load_dataset(args.dataset, split=args.train_split)
    val_ds = load_dataset(args.dataset, split=args.val_split)

    if args.train_max_samples > 0:
        train_ds = train_ds.select(range(min(args.train_max_samples, len(train_ds))))
    if args.val_max_samples > 0:
        val_ds = val_ds.select(range(min(args.val_max_samples, len(val_ds))))

    image_key = infer_image_key(train_ds)

    cfg = Config(
        steps=args.steps,
        batch=args.batch,
        patch=args.patch,
        lr=args.lr,
        noise_sigma=args.noise_sigma,
        channels=args.channels,
    )

    print("jax", jax.__version__)
    print("devices", jax.devices())
    print("train_samples", len(train_ds), "val_samples", len(val_ds), "image_key", image_key)

    key = jax.random.PRNGKey(args.seed)
    k1, k2 = jax.random.split(key)
    k3 = jax.random.PRNGKey(args.seed + 999)
    base_params = init_baseline_params(k1, in_ch=cfg.channels)
    frozen_params = init_iir_frozen_params(k2, in_ch=cfg.channels)
    residual_params = init_iir_residual_params(k3, in_ch=cfg.channels)

    base = train_model(apply_baseline, base_params, cfg, train_ds, val_ds, image_key, seed=args.seed + 1)
    frozen = train_model(apply_iir_frozen, frozen_params, cfg, train_ds, val_ds, image_key, seed=args.seed + 2)
    residual = train_model(apply_iir_residual, residual_params, cfg, train_ds, val_ds, image_key, seed=args.seed + 3)

    rows = [
        {
            "model": "baseline_conv",
            "loss_start": base["loss_start"],
            "loss_end": base["loss_end"],
            "train_time_s": base["train_time_s"],
            "psnr_noisy": base["psnr_noisy"],
            "psnr_pred": base["psnr_pred"],
            "psnr_gain": base["psnr_pred"] - base["psnr_noisy"],
        },
        {
            "model": "conv_plus_iir_frozen",
            "loss_start": frozen["loss_start"],
            "loss_end": frozen["loss_end"],
            "train_time_s": frozen["train_time_s"],
            "psnr_noisy": frozen["psnr_noisy"],
            "psnr_pred": frozen["psnr_pred"],
            "psnr_gain": frozen["psnr_pred"] - frozen["psnr_noisy"],
        },
        {
            "model": "conv_plus_iir_residual",
            "loss_start": residual["loss_start"],
            "loss_end": residual["loss_end"],
            "train_time_s": residual["train_time_s"],
            "psnr_noisy": residual["psnr_noisy"],
            "psnr_pred": residual["psnr_pred"],
            "psnr_gain": residual["psnr_pred"] - residual["psnr_noisy"],
        },
    ]

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
                fieldnames=["model", "loss_start", "loss_end", "train_time_s", "psnr_noisy", "psnr_pred", "psnr_gain"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {args.out_csv}")

    if args.out_history_csv:
        history_rows = []
        for i, l in enumerate(base["losses"]):
            history_rows.append({"model": "baseline_conv", "step": i, "loss": float(l)})
        for i, l in enumerate(frozen["losses"]):
            history_rows.append({"model": "conv_plus_iir_frozen", "step": i, "loss": float(l)})
        for i, l in enumerate(residual["losses"]):
            history_rows.append({"model": "conv_plus_iir_residual", "step": i, "loss": float(l)})
        with open(args.out_history_csv, "w", newline="", encoding="utf-8") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=["model", "step", "loss"])
            writer.writeheader()
            writer.writerows(history_rows)
        print(f"Saved history CSV: {args.out_history_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
