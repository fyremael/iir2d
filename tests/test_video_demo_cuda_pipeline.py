from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.video_demo_cuda_pipeline as video_demo


def test_parse_fps_fraction_and_decimal() -> None:
    assert video_demo.parse_fps("30000/1001") == pytest.approx(29.97002997, rel=1e-7)
    assert video_demo.parse_fps("24") == pytest.approx(24.0)


@pytest.mark.parametrize("value", ["0", "-5", "25/0", "0/1001", "-24000/1001"])
def test_parse_fps_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        video_demo.parse_fps(value)


def test_build_encode_command_libx264() -> None:
    spec = video_demo.VideoSpec(width=1920, height=1080, fps=59.94)
    cmd = video_demo.build_encode_command(
        ffmpeg="ffmpeg",
        out_video=Path("out.mp4"),
        spec=spec,
        codec="libx264",
        preset="slow",
        crf=20,
    )
    assert cmd[:2] == ["ffmpeg", "-hide_banner"]
    assert "-crf" in cmd
    assert "20" in cmd
    assert cmd[-1] == "out.mp4"


def test_build_encode_command_nvenc() -> None:
    spec = video_demo.VideoSpec(width=1280, height=720, fps=30.0)
    cmd = video_demo.build_encode_command(
        ffmpeg="ffmpeg",
        out_video=Path("nvenc.mp4"),
        spec=spec,
        codec="h264_nvenc",
        preset="p4",
        crf=23,
    )
    assert "-rc" in cmd
    assert "vbr" in cmd
    assert "-cq" in cmd
    assert "23" in cmd


def test_apply_temporal_ema_first_frame() -> None:
    current = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    out = video_demo.apply_temporal_ema(current, previous=None, alpha=0.2)
    np.testing.assert_allclose(out, current)


def test_apply_temporal_ema_blend_scalar_and_map_alpha() -> None:
    current = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    previous = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
    out_scalar = video_demo.apply_temporal_ema(current, previous=previous, alpha=0.25)
    np.testing.assert_allclose(out_scalar, np.array([[[0.25, 0.75, 0.0]]], dtype=np.float32))

    alpha_map = np.array([[[0.5]]], dtype=np.float32)
    out_map = video_demo.apply_temporal_ema(current, previous=previous, alpha=alpha_map)
    np.testing.assert_allclose(out_map, np.array([[[0.5, 0.5, 0.0]]], dtype=np.float32))


def test_rgb_ycbcr_roundtrip() -> None:
    rgb = np.array(
        [
            [[0.1, 0.2, 0.3], [0.9, 0.5, 0.2]],
            [[0.8, 0.1, 0.7], [0.3, 0.9, 0.6]],
        ],
        dtype=np.float32,
    )
    y, cb, cr = video_demo.rgb_to_ycbcr_bt709(rgb)
    recon = video_demo.ycbcr_to_rgb_bt709(y, cb, cr)
    np.testing.assert_allclose(recon, rgb, atol=3e-6, rtol=0.0)


def test_resolve_temporal_alpha_modes() -> None:
    current = np.array([[[0.8, 0.8, 0.8]]], dtype=np.float32)
    prev = np.array([[[0.2, 0.2, 0.2]]], dtype=np.float32)
    fixed = video_demo.resolve_temporal_alpha(
        current=current,
        previous=prev,
        temporal_mode="fixed",
        temporal_ema_alpha=0.33,
        temporal_alpha_min=0.1,
        temporal_alpha_max=0.9,
        temporal_motion_threshold=0.1,
    )
    assert fixed == pytest.approx(0.33)

    adaptive = video_demo.resolve_temporal_alpha(
        current=current,
        previous=prev,
        temporal_mode="adaptive",
        temporal_ema_alpha=0.33,
        temporal_alpha_min=0.1,
        temporal_alpha_max=0.9,
        temporal_motion_threshold=0.2,
    )
    assert isinstance(adaptive, np.ndarray)
    assert adaptive.shape == (1, 1, 1)
    assert float(adaptive[0, 0, 0]) == pytest.approx(0.9)


class _IdentityRunner:
    def forward_gray(self, frame_gray: np.ndarray) -> np.ndarray:
        return frame_gray


def test_filter_frame_modes() -> None:
    frame = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    identity = _IdentityRunner()

    rgb_out = video_demo.filter_frame(identity, frame, color_mode="rgb", strength=1.0)
    np.testing.assert_allclose(rgb_out, frame, atol=0.0, rtol=0.0)

    luma_out = video_demo.filter_frame(identity, frame, color_mode="luma", strength=1.0)
    np.testing.assert_allclose(luma_out, frame, atol=3e-6, rtol=0.0)


def test_probe_video_parses_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"streams": [{"width": 640, "height": 360, "r_frame_rate": "30000/1001"}]}
    monkeypatch.setattr(
        video_demo.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=str(payload).replace("'", '"'), stderr=""),
    )
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        spec = video_demo.probe_video("ffprobe", Path(td) / "in.mp4")
    assert spec.width == 640
    assert spec.height == 360
    assert spec.fps == pytest.approx(29.97002997, rel=1e-7)


def test_probe_video_raises_when_ffprobe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        video_demo.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="bad stream"),
    )
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        with pytest.raises(RuntimeError, match="ffprobe failed"):
            video_demo.probe_video("ffprobe", Path(td) / "in.mp4")


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
        "out_video": str(work_dir / "out.mp4"),
        "filter_id": 4,
        "border_mode": "mirror",
        "border_const": 0.0,
        "precision": "f32",
        "color_mode": "rgb",
        "strength": 1.0,
        "temporal_mode": "fixed",
        "temporal_ema_alpha": 1.0,
        "temporal_alpha_min": 0.1,
        "temporal_alpha_max": 0.95,
        "temporal_motion_threshold": 0.08,
        "ffmpeg": "ffmpeg",
        "ffprobe": "ffprobe",
        "codec": "libx264",
        "preset": "medium",
        "crf": 18,
        "max_frames": 0,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def test_run_pipeline_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        work_dir = Path(td)
        in_video = work_dir / "in.mp4"
        in_video.write_bytes(b"stub")
        frame = bytes([10, 20, 30, 40, 50, 60])
        decode_proc = _FakeProc(stdout=_FakeReader([frame, b""]))
        encode_writer = _FakeWriter()
        encode_proc = _FakeProc(stdin=encode_writer)
        procs = [decode_proc, encode_proc]
        monkeypatch.setattr(video_demo, "probe_video", lambda ffprobe, p: video_demo.VideoSpec(2, 1, 30.0))
        monkeypatch.setattr(video_demo, "CudaFrameFilter", _FakeCudaFrameFilter)
        monkeypatch.setattr(video_demo.subprocess, "Popen", lambda *args, **kwargs: procs.pop(0))

        rc = video_demo.run_pipeline(_make_args(work_dir))
        assert rc == 0
        assert bytes(encode_writer.buffer) == frame
        assert encode_writer.closed
        assert decode_proc.stdout is not None and decode_proc.stdout.closed
        out = capsys.readouterr().out
        assert "Done: frames=1" in out


def test_run_pipeline_frame_cap_allows_decode_termination(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        work_dir = Path(td)
        in_video = work_dir / "in.mp4"
        in_video.write_bytes(b"stub")
        frame = bytes([9, 8, 7])
        decode_proc = _FakeProc(stdout=_FakeReader([frame, frame, b""]), wait_rc=1)
        encode_writer = _FakeWriter()
        encode_proc = _FakeProc(stdin=encode_writer)
        procs = [decode_proc, encode_proc]
        monkeypatch.setattr(video_demo, "probe_video", lambda ffprobe, p: video_demo.VideoSpec(1, 1, 24.0))
        monkeypatch.setattr(video_demo, "CudaFrameFilter", _FakeCudaFrameFilter)
        monkeypatch.setattr(video_demo.subprocess, "Popen", lambda *args, **kwargs: procs.pop(0))

        rc = video_demo.run_pipeline(_make_args(work_dir, max_frames=1))
        assert rc == 0
        assert decode_proc.terminated
        assert bytes(encode_writer.buffer) == frame


def test_run_pipeline_raises_on_decode_error_without_frame_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        work_dir = Path(td)
        in_video = work_dir / "in.mp4"
        in_video.write_bytes(b"stub")
        frame = bytes([10, 20, 30])
        decode_proc = _FakeProc(stdout=_FakeReader([frame, b""]), stderr_text="decode failed", wait_rc=2)
        encode_proc = _FakeProc(stdin=_FakeWriter())
        procs = [decode_proc, encode_proc]
        monkeypatch.setattr(video_demo, "probe_video", lambda ffprobe, p: video_demo.VideoSpec(1, 1, 30.0))
        monkeypatch.setattr(video_demo, "CudaFrameFilter", _FakeCudaFrameFilter)
        monkeypatch.setattr(video_demo.subprocess, "Popen", lambda *args, **kwargs: procs.pop(0))

        with pytest.raises(RuntimeError, match="Decode process failed"):
            video_demo.run_pipeline(_make_args(work_dir))


def test_run_pipeline_validates_strength(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        work_dir = Path(td)
        (work_dir / "in.mp4").write_bytes(b"stub")
        with pytest.raises(ValueError, match="--strength"):
            video_demo.run_pipeline(_make_args(work_dir, strength=1.2))
