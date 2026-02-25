from __future__ import annotations

import argparse

import numpy as np
import pytest

import scripts.video_quality_metrics as quality


def test_summarize_stats() -> None:
    stats = quality.summarize([1.0, 2.0, 3.0])
    assert stats["min"] == pytest.approx(1.0)
    assert stats["p50"] == pytest.approx(2.0)
    assert stats["mean"] == pytest.approx(2.0)


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError):
        quality.summarize([])


def test_psnr_and_ssim_identical_frames() -> None:
    frame = np.ones((8, 8, 3), dtype=np.float32) * 0.35
    psnr = quality.compute_frame_psnr(frame, frame)
    ssim = quality.compute_frame_ssim_luma(frame, frame)
    assert psnr == pytest.approx(99.0)
    assert ssim == pytest.approx(1.0)


def test_psnr_and_ssim_detect_difference() -> None:
    a = np.zeros((8, 8, 3), dtype=np.float32)
    b = np.ones((8, 8, 3), dtype=np.float32) * 0.5
    psnr = quality.compute_frame_psnr(a, b)
    ssim = quality.compute_frame_ssim_luma(a, b)
    assert psnr < 10.0
    assert ssim < 0.5


def test_evaluate_thresholds() -> None:
    row = {
        "psnr_db_mean": 20.0,
        "ssim_mean": 0.7,
        "temporal_delta_mean": 0.02,
    }
    args_ok = argparse.Namespace(min_psnr_mean=18.0, min_ssim_mean=0.6, max_temporal_delta_mean=0.05)
    assert quality.evaluate_thresholds(row, args_ok) == []

    args_bad = argparse.Namespace(min_psnr_mean=22.0, min_ssim_mean=0.8, max_temporal_delta_mean=0.01)
    violations = quality.evaluate_thresholds(row, args_bad)
    assert len(violations) == 3
