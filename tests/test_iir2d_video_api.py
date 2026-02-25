from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import iir2d_video.api as api


def test_process_video_wrapper(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen: dict[str, object] = {}

    def fake_run_pipeline(args: argparse.Namespace) -> int:
        seen["args"] = args
        return 0

    monkeypatch.setattr(api.video_demo_cuda_pipeline, "run_pipeline", fake_run_pipeline)
    rc = api.process_video(api.VideoProcessConfig(in_video="in.mp4", out_video="out.mp4"))
    assert rc == 0
    assert isinstance(seen["args"], argparse.Namespace)


def test_benchmark_video_wrapper(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen: dict[str, object] = {}

    def fake_run_benchmark(args: argparse.Namespace) -> int:
        seen["args"] = args
        return 0

    monkeypatch.setattr(api.benchmark_video_cuda_pipeline, "run_benchmark", fake_run_benchmark)
    rc = api.benchmark_video(api.VideoBenchmarkConfig(in_video="in.mp4", out_csv="out.csv"))
    assert rc == 0
    assert isinstance(seen["args"], argparse.Namespace)


def test_quality_wrapper(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        api.video_quality_metrics,
        "evaluate_quality",
        lambda args: {"psnr_db_mean": 20.0, "ssim_mean": 0.8, "temporal_delta_mean": 0.01},
    )
    monkeypatch.setattr(api.video_quality_metrics, "write_csv", lambda path, row: None)
    monkeypatch.setattr(api.video_quality_metrics, "evaluate_thresholds", lambda row, args: [])
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        out_csv = Path(td) / "quality.csv"
        rc = api.evaluate_video_quality(
            api.VideoQualityConfig(reference_video="ref.mp4", test_video="test.mp4", out_csv=str(out_csv))
        )
        assert rc == 0
