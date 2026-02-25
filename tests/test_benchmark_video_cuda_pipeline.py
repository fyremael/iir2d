from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

import scripts.benchmark_video_cuda_pipeline as bench_video


class _FakeReader:
    def __init__(self, payloads: list[bytes]) -> None:
        self._payloads = list(payloads)
        self.closed = False

    def read(self, n: int) -> bytes:  # noqa: ARG002
        if not self._payloads:
            return b""
        return self._payloads.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeWriter:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.closed = False

    def write(self, data: bytes) -> int:
        self.buffer.extend(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class _FakeErr:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.closed = False

    def read(self) -> bytes:
        return self.text.encode("utf-8")

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    def __init__(
        self,
        *,
        stdout: _FakeReader | None = None,
        stdin: _FakeWriter | None = None,
        stderr_text: str = "",
        wait_rc: int = 0,
    ) -> None:
        self.stdout = stdout
        self.stdin = stdin
        self.stderr = _FakeErr(stderr_text)
        self._wait_rc = wait_rc
        self.terminated = False

    def poll(self) -> int | None:
        return None if not self.terminated else self._wait_rc

    def terminate(self) -> None:
        self.terminated = True

    def wait(self) -> int:
        return self._wait_rc


class _FakeCudaFrameFilter:
    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.kwargs = kwargs

    def __enter__(self) -> _FakeCudaFrameFilter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def forward_gray(self, frame_gray: np.ndarray) -> np.ndarray:
        return frame_gray


def _make_args(work_dir: Path, **overrides) -> argparse.Namespace:
    args = {
        "in_video": str(work_dir / "in.mp4"),
        "out_csv": str(work_dir / "metrics.csv"),
        "filter_id": 4,
        "border_mode": "mirror",
        "border_const": 0.0,
        "precision": "f32",
        "temporal_ema_alpha": 1.0,
        "mode": "full",
        "ffmpeg": "ffmpeg",
        "ffprobe": "ffprobe",
        "codec": "libx264",
        "preset": "medium",
        "crf": 18,
        "encode_sink": "null",
        "out_video": "",
        "warmup_frames": 0,
        "timed_frames": 2,
        "report_every": 0,
        "append": False,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def test_summarize_ms() -> None:
    stats = bench_video.summarize_ms([1.0, 2.0, 3.0])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["p50"] == pytest.approx(2.0)


def test_summarize_ms_rejects_empty() -> None:
    with pytest.raises(ValueError):
        bench_video.summarize_ms([])


def test_build_decode_command() -> None:
    cmd = bench_video.build_decode_command("ffmpeg", Path("in.mp4"))
    assert cmd[0] == "ffmpeg"
    assert cmd[-1] == "-"


def test_build_encode_command_null_and_file() -> None:
    spec = bench_video.VideoSpec(width=1280, height=720, fps=30.0)
    cmd_null = bench_video.build_encode_command(
        ffmpeg="ffmpeg",
        spec=spec,
        codec="libx264",
        preset="slow",
        crf=19,
        sink="null",
        out_video=None,
    )
    assert cmd_null[-2:] == ["null", "-"]

    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        out = Path(td) / "out.mp4"
        cmd_file = bench_video.build_encode_command(
            ffmpeg="ffmpeg",
            spec=spec,
            codec="h264_nvenc",
            preset="p4",
            crf=23,
            sink="file",
            out_video=out,
        )
        assert cmd_file[-1] == str(out)


def test_write_rows_overwrite_then_append() -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        csv_path = Path(td) / "rows.csv"
        row1 = {"a": 1, "b": 2}
        row2 = {"a": 3, "b": 4}
        bench_video.write_rows(csv_path, [row1], append=False)
        bench_video.write_rows(csv_path, [row2], append=True)
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["a"] == "1"
        assert rows[1]["a"] == "3"


def test_run_benchmark_filter_only_path(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        work_dir = Path(td)
        in_video = work_dir / "in.mp4"
        in_video.write_bytes(b"stub")
        frame = bytes([10, 20, 30])
        decode_proc = _FakeProc(stdout=_FakeReader([frame, frame, b""]))
        monkeypatch.setattr(bench_video, "probe_video", lambda ffprobe, p: bench_video.VideoSpec(1, 1, 24.0))
        monkeypatch.setattr(bench_video, "CudaFrameFilter", _FakeCudaFrameFilter)
        monkeypatch.setattr(bench_video.subprocess, "Popen", lambda *args, **kwargs: decode_proc)

        rc = bench_video.run_benchmark(_make_args(work_dir, mode="filter_only", timed_frames=2))
        assert rc == 0
        out_csv = work_dir / "metrics.csv"
        with out_csv.open("r", encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f))
        assert int(row["timed_frames"]) == 2
        assert float(row["encode_ms_mean"]) == pytest.approx(0.0)


def test_run_benchmark_full_mode_stops_early_without_decode_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        work_dir = Path(td)
        in_video = work_dir / "in.mp4"
        in_video.write_bytes(b"stub")
        frame = bytes([7, 8, 9])
        decode_proc = _FakeProc(stdout=_FakeReader([frame, frame, frame, b""]), wait_rc=1)
        encode_writer = _FakeWriter()
        encode_proc = _FakeProc(stdin=encode_writer)
        procs = [decode_proc, encode_proc]
        monkeypatch.setattr(bench_video, "probe_video", lambda ffprobe, p: bench_video.VideoSpec(1, 1, 24.0))
        monkeypatch.setattr(bench_video, "CudaFrameFilter", _FakeCudaFrameFilter)
        monkeypatch.setattr(bench_video.subprocess, "Popen", lambda *args, **kwargs: procs.pop(0))

        rc = bench_video.run_benchmark(_make_args(work_dir, mode="full", timed_frames=1))
        assert rc == 0
        assert decode_proc.terminated
        assert bytes(encode_writer.buffer) == frame


def test_run_benchmark_raises_when_not_enough_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        work_dir = Path(td)
        in_video = work_dir / "in.mp4"
        in_video.write_bytes(b"stub")
        decode_proc = _FakeProc(stdout=_FakeReader([b""]))
        monkeypatch.setattr(bench_video, "probe_video", lambda ffprobe, p: bench_video.VideoSpec(1, 1, 24.0))
        monkeypatch.setattr(bench_video, "CudaFrameFilter", _FakeCudaFrameFilter)
        monkeypatch.setattr(bench_video.subprocess, "Popen", lambda *args, **kwargs: decode_proc)

        with pytest.raises(RuntimeError, match="No timed frames"):
            bench_video.run_benchmark(_make_args(work_dir, mode="filter_only", timed_frames=2))
