from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import scripts.build_video_report_pack as pack


def test_parse_csv_path_list() -> None:
    out = pack.parse_csv_path_list(" a.csv , b.csv ,,")
    assert out == [Path("a.csv"), Path("b.csv")]


def test_read_first_row_and_build_report() -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        root = Path(td)
        bench = root / "bench.csv"
        with bench.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timed_frames", "loop_ms_p50", "loop_ms_p95", "timed_fps", "timed_mpix_per_s"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "timed_frames": "32",
                    "loop_ms_p50": "1.2",
                    "loop_ms_p95": "1.8",
                    "timed_fps": "24.0",
                    "timed_mpix_per_s": "320.0",
                }
            )

        quality = root / "quality.csv"
        with quality.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frames", "psnr_db_mean", "ssim_mean", "temporal_delta_mean"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "frames": "32",
                    "psnr_db_mean": "18.0",
                    "ssim_mean": "0.62",
                    "temporal_delta_mean": "0.02",
                }
            )

        clip = root / "clip.mp4"
        clip.write_bytes(b"x" * 4096)

        report = pack.build_report(
            benchmark_rows=[(bench, pack.read_first_row(bench))],
            quality_rows=[(quality, pack.read_first_row(quality))],
            clips=[clip],
        )
        assert "Benchmark Summary" in report
        assert "Quality Summary" in report
        assert "clip.mp4" in report
