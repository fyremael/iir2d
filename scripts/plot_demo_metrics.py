import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_summary(summary_rows, out_dir):
    models = [r["model"] for r in summary_rows]
    psnr_gain = [float(r["psnr_gain"]) for r in summary_rows]
    train_time = [float(r["train_time_s"]) for r in summary_rows]
    colors = plt.cm.tab10(range(len(models)))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(models, psnr_gain, color=colors)
    ax.set_ylabel("PSNR Gain (dB)")
    ax.set_title("Denoise Quality Gain by Model")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    out_path = os.path.join(out_dir, "psnr_gain_bar.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(models, train_time, color=colors)
    ax.set_ylabel("Train Time (s)")
    ax.set_title("Training Time by Model")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h, f"{h:.2f}s", ha="center", va="bottom", fontsize=9)
    out_path2 = os.path.join(out_dir, "train_time_bar.png")
    fig.tight_layout()
    fig.savefig(out_path2, dpi=160)
    plt.close(fig)

    return [out_path, out_path2]


def plot_history(history_rows, out_dir):
    grouped = defaultdict(lambda: {"step": [], "loss": []})
    for r in history_rows:
        m = r["model"]
        grouped[m]["step"].append(int(r["step"]))
        grouped[m]["loss"].append(float(r["loss"]))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model, vals in grouped.items():
        paired = sorted(zip(vals["step"], vals["loss"], strict=False))
        xs = [p[0] for p in paired]
        ys = [p[1] for p in paired]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.8, label=model)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    out_path = os.path.join(out_dir, "loss_curves.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return [out_path]


def main():
    parser = argparse.ArgumentParser(description="Plot iir2d demo metrics from CSV outputs")
    parser.add_argument("--summary_csv", required=True, help="Path to summary CSV from ml_engineer_demo.py")
    parser.add_argument("--history_csv", default="", help="Optional path to history CSV from ml_engineer_demo.py")
    parser.add_argument("--out_dir", default="/tmp/iir2d_plots", help="Output directory for generated PNG files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary_rows = _read_csv(args.summary_csv)
    if not summary_rows:
        raise RuntimeError(f"No rows found in summary CSV: {args.summary_csv}")

    outputs = []
    outputs.extend(plot_summary(summary_rows, args.out_dir))

    if args.history_csv:
        history_rows = _read_csv(args.history_csv)
        if history_rows:
            outputs.extend(plot_history(history_rows, args.out_dir))

    print("Generated plots:")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
