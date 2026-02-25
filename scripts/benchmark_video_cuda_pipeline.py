#!/usr/bin/env python3
"""Benchmark harness for decode -> CUDA IIR2D -> encode video pipelines."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__:
    from .core_harness import BORDER_MAP, PRECISION_MAP
    from .video_demo_cuda_pipeline import CudaFrameFilter, VideoSpec, apply_temporal_ema, probe_video
else:
    from core_harness import BORDER_MAP, PRECISION_MAP
    from video_demo_cuda_pipeline import CudaFrameFilter, VideoSpec, apply_temporal_ema, probe_video


def summarize_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("Expected at least one timed sample.")
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def build_decode_command(ffmpeg: str, in_video: Path) -> list[str]:
    return [
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
        "-",
    ]


def build_encode_command(
    ffmpeg: str,
    spec: VideoSpec,
    codec: str,
    preset: str,
    crf: int,
    sink: str,
    out_video: Path | None,
) -> list[str]:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{spec.width}x{spec.height}",
        "-r",
        f"{spec.fps:.12g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        codec,
    ]
    if codec == "libx264":
        cmd.extend(["-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"])
    elif codec == "h264_nvenc":
        cmd.extend(["-preset", preset, "-rc", "vbr", "-cq", str(crf), "-pix_fmt", "yuv420p"])
    else:
        cmd.extend(["-pix_fmt", "yuv420p"])

    if sink == "null":
        cmd.extend(["-f", "null", "-"])
    elif sink == "file":
        if out_video is None:
            raise ValueError("Expected out_video when sink='file'.")
        out_video.parent.mkdir(parents=True, exist_ok=True)
        cmd.append(str(out_video))
    else:
        raise ValueError(f"Unsupported sink {sink!r}.")
    return cmd


def make_row(
    args: argparse.Namespace,
    spec: VideoSpec,
    timed_frames: int,
    loop_ms: list[float],
    decode_ms: list[float],
    process_ms: list[float],
    encode_ms: list[float],
    wall_ms_total: float,
) -> dict[str, str | int | float]:
    loop_stats = summarize_ms(loop_ms)
    decode_stats = summarize_ms(decode_ms)
    process_stats = summarize_ms(process_ms)
    encode_stats = summarize_ms(encode_ms) if encode_ms else {"min": 0.0, "p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0}

    timed_seconds = float(np.sum(np.asarray(loop_ms, dtype=np.float64))) / 1000.0
    timed_fps = float(timed_frames / timed_seconds) if timed_seconds > 0 else 0.0
    timed_mpix_per_s = float((timed_frames * spec.width * spec.height) / 1e6 / timed_seconds) if timed_seconds > 0 else 0.0

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_video": str(Path(args.in_video).resolve()),
        "width": spec.width,
        "height": spec.height,
        "source_fps": spec.fps,
        "filter_id": args.filter_id,
        "border_mode": args.border_mode,
        "precision": args.precision,
        "temporal_ema_alpha": float(args.temporal_ema_alpha),
        "mode": args.mode,
        "codec": args.codec,
        "preset": args.preset,
        "crf": args.crf,
        "encode_sink": args.encode_sink,
        "out_video": str(Path(args.out_video).resolve()) if args.out_video else "",
        "warmup_frames": args.warmup_frames,
        "timed_frames": timed_frames,
        "loop_ms_min": loop_stats["min"],
        "loop_ms_p50": loop_stats["p50"],
        "loop_ms_p95": loop_stats["p95"],
        "loop_ms_mean": loop_stats["mean"],
        "loop_ms_max": loop_stats["max"],
        "decode_ms_mean": decode_stats["mean"],
        "process_ms_mean": process_stats["mean"],
        "encode_ms_mean": encode_stats["mean"],
        "timed_fps": timed_fps,
        "timed_mpix_per_s": timed_mpix_per_s,
        "pipeline_wall_ms_total": wall_ms_total,
    }


def write_rows(out_csv: Path, rows: list[dict[str, str | int | float]], append: bool) -> None:
    if not rows:
        raise ValueError("Expected at least one row to write.")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    write_header = (not append) or (not out_csv.exists())
    with out_csv.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args: argparse.Namespace) -> int:
    in_video = Path(args.in_video).resolve()
    out_video = Path(args.out_video).resolve() if args.out_video else None
    if not in_video.exists():
        raise FileNotFoundError(f"Input video not found: {in_video}")
    if args.filter_id < 1 or args.filter_id > 8:
        raise ValueError(f"Invalid --filter_id {args.filter_id}; expected 1..8")
    if args.temporal_ema_alpha <= 0.0 or args.temporal_ema_alpha > 1.0:
        raise ValueError("Expected --temporal_ema_alpha in (0, 1].")
    if args.warmup_frames < 0 or args.timed_frames <= 0:
        raise ValueError("Expected --warmup_frames >= 0 and --timed_frames > 0.")
    if args.mode == "filter_only" and args.encode_sink == "file":
        raise ValueError("--encode_sink file is only supported in --mode full.")
    if args.encode_sink == "file" and out_video is None:
        raise ValueError("Expected --out_video when --encode_sink=file.")

    spec = probe_video(args.ffprobe, in_video)
    frame_bytes = spec.width * spec.height * 3
    decode_cmd = build_decode_command(args.ffmpeg, in_video)
    encode_cmd = None
    if args.mode == "full":
        encode_cmd = build_encode_command(
            ffmpeg=args.ffmpeg,
            spec=spec,
            codec=args.codec,
            preset=args.preset,
            crf=args.crf,
            sink=args.encode_sink,
            out_video=out_video,
        )

    decode_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encode_proc = None
    if encode_cmd is not None:
        encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    loops: list[float] = []
    decode_times: list[float] = []
    process_times: list[float] = []
    encode_times: list[float] = []
    processed_total = 0
    timed_total = 0
    target_total = args.warmup_frames + args.timed_frames
    stopped_early = False
    ema_prev: np.ndarray | None = None

    wall_start = time.perf_counter()
    try:
        if decode_proc.stdout is None:
            raise RuntimeError("Failed to open decode stdout pipe.")
        if encode_proc is not None and encode_proc.stdin is None:
            raise RuntimeError("Failed to open encode stdin pipe.")
        with CudaFrameFilter(
            width=spec.width,
            height=spec.height,
            filter_id=args.filter_id,
            border_mode=args.border_mode,
            border_const=args.border_const,
            precision=args.precision,
        ) as runner:
            while timed_total < args.timed_frames:
                loop_t0 = time.perf_counter()
                decode_t0 = loop_t0
                blob = decode_proc.stdout.read(frame_bytes)
                decode_t1 = time.perf_counter()
                if not blob:
                    break
                if len(blob) != frame_bytes:
                    raise RuntimeError("Incomplete frame payload from decoder.")

                frame_u8 = np.frombuffer(blob, dtype=np.uint8).reshape((spec.height, spec.width, 3))
                process_t0 = time.perf_counter()
                x = frame_u8.astype(np.float32) / 255.0
                channels = [runner.forward_gray(x[:, :, c]) for c in range(3)]
                filtered = np.stack(channels, axis=2).astype(np.float32)
                smoothed = apply_temporal_ema(filtered, ema_prev, args.temporal_ema_alpha)
                ema_prev = smoothed
                out_u8 = np.clip(np.rint(smoothed * 255.0), 0, 255).astype(np.uint8)
                process_t1 = time.perf_counter()

                encode_t1 = process_t1
                if encode_proc is not None and encode_proc.stdin is not None:
                    encode_proc.stdin.write(out_u8.tobytes())
                    encode_t1 = time.perf_counter()

                loop_t1 = encode_t1
                processed_total += 1
                if processed_total > args.warmup_frames:
                    loops.append((loop_t1 - loop_t0) * 1000.0)
                    decode_times.append((decode_t1 - decode_t0) * 1000.0)
                    process_times.append((process_t1 - process_t0) * 1000.0)
                    if encode_proc is not None:
                        encode_times.append((encode_t1 - process_t1) * 1000.0)
                    timed_total += 1
                    if args.report_every > 0 and timed_total % args.report_every == 0:
                        print(f"timed {timed_total}/{args.timed_frames} frames")

                if processed_total >= target_total:
                    stopped_early = True
                    break
    finally:
        if encode_proc is not None and encode_proc.stdin is not None:
            encode_proc.stdin.close()
        if decode_proc.stdout is not None:
            decode_proc.stdout.close()
        if stopped_early and decode_proc.poll() is None:
            decode_proc.terminate()

    decode_rc = decode_proc.wait()
    decode_stderr = decode_proc.stderr.read().decode("utf-8", errors="replace") if decode_proc.stderr else ""
    if decode_proc.stderr is not None:
        decode_proc.stderr.close()
    if decode_rc != 0 and not stopped_early:
        raise RuntimeError(f"Decode process failed ({decode_rc}): {decode_stderr.strip()}")

    if encode_proc is not None:
        encode_rc = encode_proc.wait()
        encode_stderr = encode_proc.stderr.read().decode("utf-8", errors="replace") if encode_proc.stderr else ""
        if encode_proc.stderr is not None:
            encode_proc.stderr.close()
        if encode_rc != 0:
            raise RuntimeError(f"Encode process failed ({encode_rc}): {encode_stderr.strip()}")

    if timed_total <= 0:
        raise RuntimeError("No timed frames were processed. Increase video length or reduce warmup/timed frames.")

    wall_ms_total = (time.perf_counter() - wall_start) * 1000.0
    row = make_row(
        args=args,
        spec=spec,
        timed_frames=timed_total,
        loop_ms=loops,
        decode_ms=decode_times,
        process_ms=process_times,
        encode_ms=encode_times,
        wall_ms_total=wall_ms_total,
    )
    out_csv = Path(args.out_csv).resolve()
    write_rows(out_csv=out_csv, rows=[row], append=args.append)
    print(
        f"done timed_frames={timed_total} loop_p50={row['loop_ms_p50']:.3f} ms "
        f"fps={row['timed_fps']:.2f} mpix_per_s={row['timed_mpix_per_s']:.2f} "
        f"csv={out_csv}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark decode -> CUDA IIR2D -> encode video pipeline.")
    ap.add_argument("--in_video", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--filter_id", type=int, default=4)
    ap.add_argument("--border_mode", default="mirror", choices=sorted(BORDER_MAP))
    ap.add_argument("--border_const", type=float, default=0.0)
    ap.add_argument("--precision", default="f32", choices=sorted(PRECISION_MAP))
    ap.add_argument("--temporal_ema_alpha", type=float, default=1.0)
    ap.add_argument("--mode", default="full", choices=["full", "filter_only"])
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--codec", default="libx264")
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--encode_sink", default="null", choices=["null", "file"])
    ap.add_argument("--out_video", default="")
    ap.add_argument("--warmup_frames", type=int, default=24)
    ap.add_argument("--timed_frames", type=int, default=240)
    ap.add_argument("--report_every", type=int, default=60)
    ap.add_argument("--append", action="store_true")
    return ap.parse_args()


def main() -> int:
    return run_benchmark(parse_args())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
