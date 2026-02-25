#!/usr/bin/env python3
"""Compute objective video quality metrics for processed outputs."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

if __package__:
    from .video_demo_cuda_pipeline import VideoSpec, probe_video
else:
    from video_demo_cuda_pipeline import VideoSpec, probe_video


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Expected at least one value.")
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def frame_luma_bt709(frame_rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * frame_rgb[:, :, 0]) + (0.7152 * frame_rgb[:, :, 1]) + (0.0722 * frame_rgb[:, :, 2])


def compute_frame_psnr(reference: np.ndarray, test: np.ndarray) -> float:
    mse = float(np.mean((reference - test) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_frame_ssim_luma(reference: np.ndarray, test: np.ndarray) -> float:
    x = frame_luma_bt709(reference).astype(np.float64, copy=False)
    y = frame_luma_bt709(test).astype(np.float64, copy=False)
    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))

    c1 = (0.01**2)
    c2 = (0.03**2)
    numer = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denom = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if abs(denom) < 1e-12:
        return 1.0
    return float(numer / denom)


def build_decode_command(ffmpeg: str, in_video: Path, spec: VideoSpec, max_frames: int) -> list[str]:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(in_video),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "0",
    ]
    if max_frames > 0:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.append("-")
    return cmd


def evaluate_quality(args: argparse.Namespace) -> dict[str, str | int | float]:
    reference_video = Path(args.reference_video).resolve()
    test_video = Path(args.test_video).resolve()
    if not reference_video.exists():
        raise FileNotFoundError(f"Reference video not found: {reference_video}")
    if not test_video.exists():
        raise FileNotFoundError(f"Test video not found: {test_video}")
    if args.max_frames < 0:
        raise ValueError("Expected --max_frames >= 0.")

    reference_spec = probe_video(args.ffprobe, reference_video)
    test_spec = probe_video(args.ffprobe, test_video)
    if (reference_spec.width, reference_spec.height) != (test_spec.width, test_spec.height):
        raise ValueError(
            "Resolution mismatch: "
            f"reference={reference_spec.width}x{reference_spec.height}, "
            f"test={test_spec.width}x{test_spec.height}"
        )

    frame_bytes = reference_spec.width * reference_spec.height * 3
    ref_cmd = build_decode_command(args.ffmpeg, reference_video, reference_spec, args.max_frames)
    test_cmd = build_decode_command(args.ffmpeg, test_video, test_spec, args.max_frames)
    ref_proc = subprocess.Popen(ref_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    test_proc = subprocess.Popen(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    temporal_delta_values: list[float] = []
    prev_ref: np.ndarray | None = None
    prev_test: np.ndarray | None = None

    try:
        if ref_proc.stdout is None or test_proc.stdout is None:
            raise RuntimeError("Failed to open ffmpeg decode pipes.")
        while True:
            ref_blob = ref_proc.stdout.read(frame_bytes)
            test_blob = test_proc.stdout.read(frame_bytes)
            if not ref_blob and not test_blob:
                break
            if not ref_blob or not test_blob:
                raise RuntimeError("Reference and test videos have different frame counts.")
            if len(ref_blob) != frame_bytes or len(test_blob) != frame_bytes:
                raise RuntimeError("Incomplete frame payload while decoding videos.")

            ref_frame = np.frombuffer(ref_blob, dtype=np.uint8).reshape((reference_spec.height, reference_spec.width, 3))
            test_frame = np.frombuffer(test_blob, dtype=np.uint8).reshape((reference_spec.height, reference_spec.width, 3))
            ref_norm = ref_frame.astype(np.float32) / 255.0
            test_norm = test_frame.astype(np.float32) / 255.0

            psnr_values.append(compute_frame_psnr(ref_norm, test_norm))
            ssim_values.append(compute_frame_ssim_luma(ref_norm, test_norm))

            if prev_ref is not None and prev_test is not None:
                ref_motion = float(np.mean(np.abs(ref_norm - prev_ref)))
                test_motion = float(np.mean(np.abs(test_norm - prev_test)))
                temporal_delta_values.append(abs(test_motion - ref_motion))
            prev_ref = ref_norm
            prev_test = test_norm
    finally:
        if ref_proc.stdout is not None:
            ref_proc.stdout.close()
        if test_proc.stdout is not None:
            test_proc.stdout.close()

    ref_rc = ref_proc.wait()
    test_rc = test_proc.wait()
    ref_err = ref_proc.stderr.read().decode("utf-8", errors="replace") if ref_proc.stderr else ""
    test_err = test_proc.stderr.read().decode("utf-8", errors="replace") if test_proc.stderr else ""
    if ref_proc.stderr is not None:
        ref_proc.stderr.close()
    if test_proc.stderr is not None:
        test_proc.stderr.close()
    if ref_rc != 0:
        raise RuntimeError(f"Reference decode failed ({ref_rc}): {ref_err.strip()}")
    if test_rc != 0:
        raise RuntimeError(f"Test decode failed ({test_rc}): {test_err.strip()}")
    if not psnr_values:
        raise RuntimeError("No frames decoded for quality evaluation.")

    psnr_stats = summarize(psnr_values)
    ssim_stats = summarize(ssim_values)
    temporal_stats = summarize(temporal_delta_values) if temporal_delta_values else {
        "min": 0.0,
        "p50": 0.0,
        "p95": 0.0,
        "mean": 0.0,
        "max": 0.0,
    }

    return {
        "reference_video": str(reference_video),
        "test_video": str(test_video),
        "width": reference_spec.width,
        "height": reference_spec.height,
        "reference_fps": reference_spec.fps,
        "test_fps": test_spec.fps,
        "frames": len(psnr_values),
        "psnr_db_min": psnr_stats["min"],
        "psnr_db_p50": psnr_stats["p50"],
        "psnr_db_p95": psnr_stats["p95"],
        "psnr_db_mean": psnr_stats["mean"],
        "psnr_db_max": psnr_stats["max"],
        "ssim_min": ssim_stats["min"],
        "ssim_p50": ssim_stats["p50"],
        "ssim_p95": ssim_stats["p95"],
        "ssim_mean": ssim_stats["mean"],
        "ssim_max": ssim_stats["max"],
        "temporal_delta_min": temporal_stats["min"],
        "temporal_delta_p50": temporal_stats["p50"],
        "temporal_delta_p95": temporal_stats["p95"],
        "temporal_delta_mean": temporal_stats["mean"],
        "temporal_delta_max": temporal_stats["max"],
    }


def evaluate_thresholds(row: dict[str, str | int | float], args: argparse.Namespace) -> list[str]:
    violations: list[str] = []
    psnr_mean = float(row["psnr_db_mean"])
    ssim_mean = float(row["ssim_mean"])
    temporal_mean = float(row["temporal_delta_mean"])
    if args.min_psnr_mean is not None and psnr_mean < args.min_psnr_mean:
        violations.append(f"psnr_db_mean={psnr_mean:.4f} below min {args.min_psnr_mean:.4f}")
    if args.min_ssim_mean is not None and ssim_mean < args.min_ssim_mean:
        violations.append(f"ssim_mean={ssim_mean:.4f} below min {args.min_ssim_mean:.4f}")
    if args.max_temporal_delta_mean is not None and temporal_mean > args.max_temporal_delta_mean:
        violations.append(f"temporal_delta_mean={temporal_mean:.6f} above max {args.max_temporal_delta_mean:.6f}")
    return violations


def write_csv(path: Path, row: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute PSNR/SSIM/temporal-delta metrics between two videos.")
    ap.add_argument("--reference_video", required=True)
    ap.add_argument("--test_video", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--min_psnr_mean", type=float, default=None)
    ap.add_argument("--min_ssim_mean", type=float, default=None)
    ap.add_argument("--max_temporal_delta_mean", type=float, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    row = evaluate_quality(args)
    out_csv = Path(args.out_csv).resolve()
    write_csv(out_csv, row)
    violations = evaluate_thresholds(row, args)
    print(
        f"video_quality: frames={row['frames']} psnr_mean={row['psnr_db_mean']:.3f} "
        f"ssim_mean={row['ssim_mean']:.4f} temporal_delta_mean={row['temporal_delta_mean']:.6f} "
        f"csv={out_csv}"
    )
    if violations:
        for msg in violations:
            print(f"threshold_violation: {msg}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
