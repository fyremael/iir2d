import argparse
import csv
import glob
import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from iir2d_jax import iir2d, iir2d_vjp
from jax import lax
from PIL import Image


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


def list_image_files(data_dir: str) -> list[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files: list[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    files = sorted(set(files))
    if not files:
        raise RuntimeError(f"No image files found in {data_dir}")
    return files


def load_image(path: str, channels: int) -> np.ndarray:
    with Image.open(path) as img:
        if channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    if channels == 1:
        arr = arr[..., None]
    return arr


def random_crop(rng: np.random.Generator, img: np.ndarray, patch: int) -> np.ndarray:
    h, w, c = img.shape
    if h < patch or w < patch:
        scale = max(patch / h, patch / w)
        nh, nw = int(np.ceil(h * scale)), int(np.ceil(w * scale))
        arr = np.clip(img.squeeze(-1) if c == 1 else img, 0.0, 1.0)
        arr_u8 = (arr * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr_u8)
        pil = pil.resize((nw, nh), Image.BILINEAR)
        img = np.asarray(pil, dtype=np.float32) / 255.0
        if c == 1:
            img = img[..., None]
        h, w, c = img.shape
    y0 = rng.integers(0, h - patch + 1)
    x0 = rng.integers(0, w - patch + 1)
    return img[y0:y0 + patch, x0:x0 + patch, :]


def make_noisy(clean: np.ndarray, sigma: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
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


def init_baseline_params(key: jnp.ndarray, in_ch: int, hidden: int = 32) -> dict[str, jnp.ndarray]:
    k1, k2 = jax.random.split(key)
    w1 = 0.05 * jax.random.normal(k1, (3, 3, in_ch, hidden), dtype=jnp.float32)
    w2 = 0.05 * jax.random.normal(k2, (3, 3, hidden, in_ch), dtype=jnp.float32)
    return {"w1": w1, "w2": w2}


def apply_baseline(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    h = conv2d_nhwc(x, params["w1"])
    h = jax.nn.gelu(h)
    h = conv2d_nhwc(h, params["w2"])
    return x + h


def init_iir_frozen_params(key: jnp.ndarray, in_ch: int, hidden: int = 32, num_filters: int = 4) -> dict[str, jnp.ndarray]:
    k1, k2, k3 = jax.random.split(key, 3)
    w1 = 0.05 * jax.random.normal(k1, (3, 3, in_ch, hidden), dtype=jnp.float32)
    w2 = 0.05 * jax.random.normal(k2, (3, 3, hidden, in_ch), dtype=jnp.float32)
    logits = 0.01 * jax.random.normal(k3, (num_filters, in_ch), dtype=jnp.float32)
    return {"w1": w1, "w2": w2, "logits": logits}


def apply_iir_frozen(params: dict[str, jnp.ndarray], x: jnp.ndarray, filter_ids: tuple[int, ...] = (1, 3, 4, 8)) -> jnp.ndarray:
    ys = [iir_nhwc(x, filter_id=int(fid), differentiable=False) for fid in filter_ids]
    stack = jnp.stack(ys, axis=0)
    w = jax.nn.softmax(params["logits"], axis=0).reshape((len(filter_ids), 1, 1, 1, -1))
    feat = jnp.sum(stack * w, axis=0)
    feat = lax.stop_gradient(feat)
    h = conv2d_nhwc(feat, params["w1"])
    h = jax.nn.gelu(h)
    h = conv2d_nhwc(h, params["w2"])
    return x + h


def build_batch(paths: Sequence[str], cfg: Config, rng: np.random.Generator) -> dict[str, jnp.ndarray]:
    noisy_batch = []
    clean_batch = []
    for _ in range(cfg.batch):
        p = paths[int(rng.integers(0, len(paths)))]
        img = load_image(p, cfg.channels)
        crop = random_crop(rng, img, cfg.patch)
        noisy, clean = make_noisy(crop, cfg.noise_sigma, rng)
        noisy_batch.append(noisy)
        clean_batch.append(clean)
    noisy = jnp.asarray(np.stack(noisy_batch, axis=0), dtype=jnp.float32)
    clean = jnp.asarray(np.stack(clean_batch, axis=0), dtype=jnp.float32)
    return {"noisy": noisy, "clean": clean}


def train_model(apply_fn, params, cfg: Config, train_paths: Sequence[str], val_paths: Sequence[str], seed: int):
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

    losses: list[float] = []
    t0 = time.perf_counter()
    for _ in range(cfg.steps):
        batch = build_batch(train_paths, cfg, npr)
        params, opt_state, loss = step_fn(params, opt_state, batch)
        losses.append(float(loss))
    elapsed = time.perf_counter() - t0

    val_batch = build_batch(val_paths, cfg, npr)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-image dataset evaluation for iir2d model variants")
    parser.add_argument("--data_dir", required=True, help="Directory containing images")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--noise_sigma", type=float, default=0.08)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--out_history_csv", default="")
    args = parser.parse_args()

    cfg = Config(
        steps=args.steps,
        batch=args.batch,
        patch=args.patch,
        lr=args.lr,
        noise_sigma=args.noise_sigma,
        channels=args.channels,
    )

    all_paths = list_image_files(args.data_dir)
    rnd = random.Random(args.seed)
    rnd.shuffle(all_paths)
    split = max(1, int(len(all_paths) * args.train_ratio))
    train_paths = all_paths[:split]
    val_paths = all_paths[split:] if split < len(all_paths) else all_paths[-max(1, len(all_paths)//5):]

    print("jax", jax.__version__)
    print("devices", jax.devices())
    print("images_total", len(all_paths), "train", len(train_paths), "val", len(val_paths))

    k = jax.random.PRNGKey(args.seed)
    k1, k2 = jax.random.split(k)
    base_params = init_baseline_params(k1, in_ch=cfg.channels)
    frozen_params = init_iir_frozen_params(k2, in_ch=cfg.channels)

    base = train_model(apply_baseline, base_params, cfg, train_paths, val_paths, seed=args.seed + 1)
    frozen = train_model(apply_iir_frozen, frozen_params, cfg, train_paths, val_paths, seed=args.seed + 2)

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
        for i, loss_value in enumerate(base["losses"]):
            history_rows.append({"model": "baseline_conv", "step": i, "loss": float(loss_value)})
        for i, loss_value in enumerate(frozen["losses"]):
            history_rows.append({"model": "conv_plus_iir_frozen", "step": i, "loss": float(loss_value)})
        with open(args.out_history_csv, "w", newline="", encoding="utf-8") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=["model", "step", "loss"])
            writer.writeheader()
            writer.writerows(history_rows)
        print(f"Saved history CSV: {args.out_history_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
