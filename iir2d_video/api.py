"""Programmatic wrappers for IIR2D video scripts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts import benchmark_video_cuda_pipeline, video_demo_cuda_pipeline, video_quality_metrics


@dataclass(frozen=True)
class VideoProcessConfig:
    in_video: str
    out_video: str
    filter_id: int = 1
    border_mode: str = "mirror"
    border_const: float = 0.0
    precision: str = "f32"
    color_mode: str = "luma"
    strength: float = 0.65
    temporal_mode: str = "adaptive"
    temporal_ema_alpha: float = 0.9
    temporal_alpha_min: float = 0.10
    temporal_alpha_max: float = 0.95
    temporal_motion_threshold: float = 0.08
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    codec: str = "libx264"
    preset: str = "medium"
    crf: int = 18
    max_frames: int = 0


@dataclass(frozen=True)
class VideoBenchmarkConfig:
    in_video: str
    out_csv: str
    filter_id: int = 1
    border_mode: str = "mirror"
    border_const: float = 0.0
    precision: str = "f32"
    color_mode: str = "luma"
    strength: float = 0.65
    temporal_mode: str = "adaptive"
    temporal_ema_alpha: float = 0.9
    temporal_alpha_min: float = 0.10
    temporal_alpha_max: float = 0.95
    temporal_motion_threshold: float = 0.08
    mode: str = "full"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    codec: str = "libx264"
    preset: str = "medium"
    crf: int = 18
    encode_sink: str = "null"
    out_video: str = ""
    warmup_frames: int = 24
    timed_frames: int = 240
    report_every: int = 60
    append: bool = False


@dataclass(frozen=True)
class VideoQualityConfig:
    reference_video: str
    test_video: str
    out_csv: str
    max_frames: int = 0
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    min_psnr_mean: float | None = None
    min_ssim_mean: float | None = None
    max_temporal_delta_mean: float | None = None


def _to_namespace(obj: object) -> argparse.Namespace:
    return argparse.Namespace(**asdict(obj))


def process_video(config: VideoProcessConfig) -> int:
    """Run decode -> CUDA IIR2D -> encode using a typed config object."""
    return video_demo_cuda_pipeline.run_pipeline(_to_namespace(config))


def benchmark_video(config: VideoBenchmarkConfig) -> int:
    """Benchmark decode/process/encode throughput using a typed config object."""
    return benchmark_video_cuda_pipeline.run_benchmark(_to_namespace(config))


def evaluate_video_quality(config: VideoQualityConfig) -> int:
    """Compute objective quality metrics between reference and processed videos."""
    args = _to_namespace(config)
    row = video_quality_metrics.evaluate_quality(args)
    video_quality_metrics.write_csv(path=Path(args.out_csv).resolve(), row=row)
    violations = video_quality_metrics.evaluate_thresholds(row, args)
    if violations:
        return 1
    return 0
