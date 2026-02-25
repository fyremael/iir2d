from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

import scripts.build_benchmark_claims_packet as claims
from scripts.build_benchmark_claims_packet import build_packet


def test_build_packet_contains_command_env_and_table() -> None:
    rows = [
        {
            "width": "512",
            "height": "512",
            "filter_id": "1",
            "border_mode": "mirror",
            "precision": "f32",
            "latency_ms_p50": "0.320",
            "latency_ms_p95": "0.410",
            "throughput_mpix_per_s_p50": "819.2",
            "throughput_gb_per_s_p50": "6.55",
            "host": "hostA",
            "platform": "Linux",
            "python_version": "3.12",
            "gpu_name": "GPU-X",
            "gpu_driver": "570.00",
            "cuda_runtime_version": "13.0",
            "cuda_driver_version": "13.0",
            "nvcc_release": "13.0",
            "lib_path": "python/iir2d_jax/libiir2d_jax.so",
        },
        {
            "width": "1024",
            "height": "1024",
            "filter_id": "4",
            "border_mode": "mirror",
            "precision": "mixed",
            "latency_ms_p50": "1.280",
            "latency_ms_p95": "1.550",
            "throughput_mpix_per_s_p50": "819.2",
            "throughput_gb_per_s_p50": "6.55",
            "host": "hostA",
            "platform": "Linux",
            "python_version": "3.12",
            "gpu_name": "GPU-X",
            "gpu_driver": "570.00",
            "cuda_runtime_version": "13.0",
            "cuda_driver_version": "13.0",
            "nvcc_release": "13.0",
            "lib_path": "python/iir2d_jax/libiir2d_jax.so",
        },
    ]
    packet = build_packet(rows, Path("input.csv"), "python3 scripts/benchmark_core_cuda.py --sizes 512x512")

    assert "# Benchmark Claims Packet" in packet
    assert "## Benchmark Command" in packet
    assert "python3 scripts/benchmark_core_cuda.py --sizes 512x512" in packet
    assert "- GPU: `GPU-X`" in packet
    assert "| 1024x1024 | 4 | mirror | mixed | 1.280 | 1.550 | 819.2 | 6.550 |" in packet


def test_build_packet_raises_on_empty_rows() -> None:
    with pytest.raises(ValueError):
        build_packet([], Path("input.csv"), "python3 scripts/benchmark_core_cuda.py")


def test_load_rows_reads_csv_via_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = "width,height,filter_id,border_mode,precision\n512,512,1,mirror,f32\n"

    class FakeFile:
        def __enter__(self) -> io.StringIO:
            return io.StringIO(payload)

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_open(self: Path, mode: str, encoding: str, newline: str) -> FakeFile:
        assert mode == "r"
        return FakeFile()

    monkeypatch.setattr(Path, "open", fake_open)
    rows = claims.load_rows(Path("dummy.csv"))
    assert rows[0]["filter_id"] == "1"


def test_main_writes_output(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_benchmark_claims_packet.py",
            "--in_csv",
            "in.csv",
            "--out_md",
            "out.md",
            "--benchmark_command",
            "python3 scripts/benchmark_core_cuda.py --sizes 512x512",
        ],
    )
    monkeypatch.setattr(claims, "load_rows", lambda _p: [{"width": "1", "height": "1", "filter_id": "1", "border_mode": "mirror", "precision": "f32", "latency_ms_p50": "1", "latency_ms_p95": "1", "throughput_mpix_per_s_p50": "1", "throughput_gb_per_s_p50": "1"}])  # noqa: E501
    monkeypatch.setattr(claims, "build_packet", lambda *_args: "packet")

    def fake_mkdir(self: Path, parents: bool, exist_ok: bool) -> None:
        assert parents
        assert exist_ok

    def fake_write_text(self: Path, text: str, encoding: str) -> int:
        captured["path"] = str(self)
        captured["text"] = text
        assert encoding == "utf-8"
        return len(text)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
    monkeypatch.setattr(Path, "write_text", fake_write_text)

    assert claims.main() == 0
    assert captured["path"].endswith("out.md")
    assert captured["text"] == "packet\n"
