#!/usr/bin/env python3
"""Minimal decode -> CUDA IIR2D -> encode video demo pipeline."""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
from dataclasses import dataclass
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
    )
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
    )


@dataclass(frozen=True)
class VideoSpec:
    width: int
    height: int
    fps: float


def parse_fps(rate_text: str) -> float:
    if "/" in rate_text:
        num_text, den_text = rate_text.split("/", 1)
        num = float(num_text)
        den = float(den_text)
        if num <= 0 or den <= 0:
            raise ValueError(f"Invalid frame rate {rate_text!r}")
        return num / den
    value = float(rate_text)
    if value <= 0:
        raise ValueError(f"Invalid frame rate {rate_text!r}")
    return value


def probe_video(ffprobe: str, in_video: Path) -> VideoSpec:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(in_video),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {in_video}: {proc.stderr.strip()}")
    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {in_video}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = parse_fps(str(stream["r_frame_rate"]))
    return VideoSpec(width=width, height=height, fps=fps)


def build_encode_command(
    ffmpeg: str,
    out_video: Path,
    spec: VideoSpec,
    codec: str,
    preset: str,
    crf: int,
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
    cmd.append(str(out_video))
    return cmd


def apply_temporal_ema(
    current: np.ndarray,
    previous: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    if previous is None:
        return current
    return (alpha * current) + ((1.0 - alpha) * previous)


class CudaFrameFilter:
    def __init__(
        self,
        width: int,
        height: int,
        filter_id: int,
        border_mode: str,
        border_const: float,
        precision: str,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        self.core_lib, _ = load_core_library(repo_root)
        self.cudart = find_cudart()
        configure_core_lib(self.core_lib)
        configure_cudart(self.cudart)

        if precision == "f64":
            self.dtype = np.float64
        else:
            self.dtype = np.float32
        self.width = width
        self.height = height
        self.n = width * height
        self.elem_size = 8 if self.dtype is np.float64 else 4
        self.byte_count = self.n * self.elem_size
        self.params = IIR2D_Params(
            width=width,
            height=height,
            filter_id=filter_id,
            border_mode=BORDER_MAP[border_mode],
            border_const=float(border_const),
            precision=PRECISION_MAP[precision],
        )
        self.d_in = ctypes.c_void_p()
        self.d_out = ctypes.c_void_p()
        cuda_check(
            self.cudart,
            self.cudart.cudaMalloc(ctypes.byref(self.d_in), self.byte_count),
            "cudaMalloc(d_in)",
        )
        cuda_check(
            self.cudart,
            self.cudart.cudaMalloc(ctypes.byref(self.d_out), self.byte_count),
            "cudaMalloc(d_out)",
        )

    def close(self) -> None:
        if self.d_in.value:
            self.cudart.cudaFree(self.d_in)
            self.d_in = ctypes.c_void_p()
        if self.d_out.value:
            self.cudart.cudaFree(self.d_out)
            self.d_out = ctypes.c_void_p()

    def __enter__(self) -> CudaFrameFilter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def forward_gray(self, frame_gray: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(frame_gray, dtype=self.dtype)
        if x.shape != (self.height, self.width):
            raise ValueError(f"Expected frame shape {(self.height, self.width)}, got {x.shape}")
        out = np.empty_like(x)
        cuda_check(
            self.cudart,
            self.cudart.cudaMemcpy(
                self.d_in,
                x.ctypes.data_as(ctypes.c_void_p),
                self.byte_count,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy(H2D)",
        )
        status = self.core_lib.iir2d_forward_cuda(self.d_in, self.d_out, ctypes.byref(self.params))
        if status != 0:
            text = self.core_lib.iir2d_status_string(status)
            msg = text.decode("utf-8") if text else str(status)
            raise RuntimeError(f"iir2d_forward_cuda failed: {msg} ({status})")
        cuda_check(self.cudart, self.cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
        cuda_check(
            self.cudart,
            self.cudart.cudaMemcpy(
                out.ctypes.data_as(ctypes.c_void_p),
                self.d_out,
                self.byte_count,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy(D2H)",
        )
        return out


def run_pipeline(args: argparse.Namespace) -> int:
    in_video = Path(args.in_video).resolve()
    out_video = Path(args.out_video).resolve()
    if not in_video.exists():
        raise FileNotFoundError(f"Input video not found: {in_video}")
    if args.filter_id < 1 or args.filter_id > 8:
        raise ValueError(f"Invalid --filter_id {args.filter_id}; expected 1..8")
    if args.border_mode not in BORDER_MAP:
        raise ValueError(f"Invalid --border_mode {args.border_mode!r}; expected one of {sorted(BORDER_MAP)}")
    if args.precision not in PRECISION_MAP:
        raise ValueError(f"Invalid --precision {args.precision!r}; expected one of {sorted(PRECISION_MAP)}")
    if args.temporal_ema_alpha <= 0.0 or args.temporal_ema_alpha > 1.0:
        raise ValueError("Expected --temporal_ema_alpha in (0, 1].")
    if args.max_frames < 0:
        raise ValueError("Expected --max_frames >= 0.")

    spec = probe_video(args.ffprobe, in_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    frame_bytes = spec.width * spec.height * 3
    decode_cmd = [
        args.ffmpeg,
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
    encode_cmd = build_encode_command(
        ffmpeg=args.ffmpeg,
        out_video=out_video,
        spec=spec,
        codec=args.codec,
        preset=args.preset,
        crf=args.crf,
    )

    decode_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    processed = 0
    frame_cap_hit = False
    ema_prev: np.ndarray | None = None
    try:
        if decode_proc.stdout is None or encode_proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg pipes.")
        with CudaFrameFilter(
            width=spec.width,
            height=spec.height,
            filter_id=args.filter_id,
            border_mode=args.border_mode,
            border_const=args.border_const,
            precision=args.precision,
        ) as runner:
            while True:
                blob = decode_proc.stdout.read(frame_bytes)
                if not blob:
                    break
                if len(blob) != frame_bytes:
                    raise RuntimeError("Incomplete frame payload from decoder.")
                frame_u8 = np.frombuffer(blob, dtype=np.uint8).reshape((spec.height, spec.width, 3))
                x = frame_u8.astype(np.float32) / 255.0
                channels: list[np.ndarray] = []
                for c in range(3):
                    channels.append(runner.forward_gray(x[:, :, c]))
                filtered = np.stack(channels, axis=2).astype(np.float32)
                smoothed = apply_temporal_ema(filtered, ema_prev, args.temporal_ema_alpha)
                ema_prev = smoothed
                out_u8 = np.clip(np.rint(smoothed * 255.0), 0, 255).astype(np.uint8)
                encode_proc.stdin.write(out_u8.tobytes())
                processed += 1
                if args.max_frames > 0 and processed >= args.max_frames:
                    frame_cap_hit = True
                    break
    finally:
        if encode_proc.stdin is not None:
            encode_proc.stdin.close()
        if decode_proc.stdout is not None:
            decode_proc.stdout.close()
        if frame_cap_hit and decode_proc.poll() is None:
            decode_proc.terminate()

    decode_rc = decode_proc.wait()
    encode_rc = encode_proc.wait()
    decode_stderr = decode_proc.stderr.read().decode("utf-8", errors="replace") if decode_proc.stderr else ""
    encode_stderr = encode_proc.stderr.read().decode("utf-8", errors="replace") if encode_proc.stderr else ""
    if decode_proc.stderr is not None:
        decode_proc.stderr.close()
    if encode_proc.stderr is not None:
        encode_proc.stderr.close()
    if decode_rc != 0 and not frame_cap_hit:
        raise RuntimeError(f"Decode process failed ({decode_rc}): {decode_stderr.strip()}")
    if encode_rc != 0:
        raise RuntimeError(f"Encode process failed ({encode_rc}): {encode_stderr.strip()}")

    print(
        f"Done: frames={processed} "
        f"filter=f{args.filter_id} border={args.border_mode} precision={args.precision} "
        f"temporal_ema_alpha={args.temporal_ema_alpha:.3f} out={out_video}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Decode video, run CUDA IIR2D per frame, then encode output.")
    ap.add_argument("--in_video", required=True)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--filter_id", type=int, default=4)
    ap.add_argument("--border_mode", default="mirror", choices=sorted(BORDER_MAP))
    ap.add_argument("--border_const", type=float, default=0.0)
    ap.add_argument("--precision", default="f32", choices=sorted(PRECISION_MAP))
    ap.add_argument(
        "--temporal_ema_alpha",
        type=float,
        default=1.0,
        help="Temporal EMA blend alpha in (0,1], where 1.0 disables temporal smoothing.",
    )
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--codec", default="libx264", help="Video codec for output (e.g. libx264, h264_nvenc).")
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--max_frames", type=int, default=0, help="Optional frame cap for smoke runs.")
    return ap.parse_args()


def main() -> int:
    return run_pipeline(parse_args())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
