from __future__ import annotations

from pathlib import Path

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
