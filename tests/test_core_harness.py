from __future__ import annotations

import ctypes
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.core_harness as harness


def test_parse_case_matrix_cross_product() -> None:
    cases = harness.parse_case_matrix(
        sizes_arg="64x32,32x16",
        filter_ids_arg="1,2",
        border_modes_arg="mirror,wrap",
        precisions_arg="f32",
    )
    assert len(cases) == 8
    assert cases[0] == harness.Case(width=64, height=32, filter_id=1, border_mode="mirror", precision="f32")
    assert cases[-1] == harness.Case(width=32, height=16, filter_id=2, border_mode="wrap", precision="f32")


def test_parse_case_matrix_rejects_invalid_axes() -> None:
    with pytest.raises(ValueError, match="Invalid filter_id"):
        harness.parse_case_matrix(
            sizes_arg="64x32",
            filter_ids_arg="9",
            border_modes_arg="mirror",
            precisions_arg="f32",
        )
    with pytest.raises(ValueError, match="Invalid border mode"):
        harness.parse_case_matrix(
            sizes_arg="64x32",
            filter_ids_arg="1",
            border_modes_arg="bad",
            precisions_arg="f32",
        )
    with pytest.raises(ValueError, match="Invalid precision"):
        harness.parse_case_matrix(
            sizes_arg="64x32",
            filter_ids_arg="1",
            border_modes_arg="mirror",
            precisions_arg="bad",
        )


def test_parse_sizes_rejects_bad_token() -> None:
    with pytest.raises(ValueError, match="expected WxH"):
        harness.parse_sizes("64by32")


def test_candidate_core_libraries_for_posix_and_nt(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path("repo")
    monkeypatch.setattr(harness.os, "name", "posix")
    posix_paths = harness.candidate_core_libraries(repo_root)
    assert posix_paths[0].name == "libiir2d_jax.so"

    monkeypatch.setattr(harness.os, "name", "nt")
    nt_paths = harness.candidate_core_libraries(repo_root)
    assert nt_paths[0].name == "iir2d_jax.dll"


def test_load_core_library_returns_first_loadable(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = Path("bad.so")
    good = Path("good.so")
    monkeypatch.setattr(harness, "candidate_core_libraries", lambda _: [bad, good])
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: self in {bad, good},
    )

    class FakeCDLL:
        pass

    def fake_cdll(path: str) -> FakeCDLL:
        if Path(path) == bad:
            raise OSError("bad load")
        return FakeCDLL()

    monkeypatch.setattr(harness.ctypes, "CDLL", fake_cdll)
    lib, lib_path = harness.load_core_library(Path("unused"))
    assert isinstance(lib, FakeCDLL)
    assert lib_path == good


def test_load_core_library_reports_errors_or_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = Path("bad.so")
    monkeypatch.setattr(harness, "candidate_core_libraries", lambda _: [bad])
    monkeypatch.setattr(Path, "exists", lambda self: self == bad)
    monkeypatch.setattr(harness.ctypes, "CDLL", lambda _: (_ for _ in ()).throw(OSError("bad load")))
    with pytest.raises(RuntimeError, match="Could not load iir2d shared library"):
        harness.load_core_library(Path("unused"))

    monkeypatch.setattr(harness, "candidate_core_libraries", lambda _: [Path("missing.so")])
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(FileNotFoundError):
        harness.load_core_library(Path("unused"))


def test_find_cudart_posix_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness.os, "name", "posix")
    seen: list[str] = []

    class FakeCDLL:
        pass

    def fake_cdll(name: str) -> FakeCDLL:
        seen.append(name)
        if name == "libcudart.so.12":
            return FakeCDLL()
        raise OSError("missing")

    monkeypatch.setattr(harness.ctypes, "CDLL", fake_cdll)
    lib = harness.find_cudart()
    assert isinstance(lib, FakeCDLL)
    assert "libcudart.so" in seen

    monkeypatch.setattr(harness.ctypes, "CDLL", lambda _: (_ for _ in ()).throw(OSError("missing")))
    with pytest.raises(RuntimeError, match="Failed to load libcudart.so"):
        harness.find_cudart()


def test_find_cudart_windows_prefers_cuda_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness.os, "name", "nt")

    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
        fake_root = Path(tmp) / "cuda_root"
        dll_path = fake_root / "v13.1" / "bin" / "cudart64_130.dll"
        dll_path.parent.mkdir(parents=True, exist_ok=True)
        dll_path.write_text("", encoding="utf-8")

        real_path_type = type(fake_root)

        def fake_path(value: str) -> Path:
            if value == r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA":
                return fake_root
            return real_path_type(value)

        class FakeCDLL:
            pass

        monkeypatch.setattr(harness, "Path", fake_path)
        monkeypatch.setattr(harness.ctypes, "CDLL", lambda _: FakeCDLL())
        monkeypatch.setattr(harness.os, "environ", {"PATH": "", "CUDA_PATH": ""})
        lib = harness.find_cudart()
        assert isinstance(lib, FakeCDLL)


def test_configure_and_error_helpers() -> None:
    cudart = SimpleNamespace(
        cudaGetErrorString=lambda code: b"err",
        cudaMalloc=lambda ptr, size: 0,
        cudaFree=lambda ptr: 0,
        cudaMemcpy=lambda dst, src, n, kind: 0,
        cudaMemset=lambda ptr, v, n: 0,
        cudaDeviceSynchronize=lambda: 0,
        cudaRuntimeGetVersion=lambda out: 0,
        cudaDriverGetVersion=lambda out: 0,
        cudaGetDevice=lambda out: 0,
        cudaEventCreate=lambda out: 0,
        cudaEventDestroy=lambda evt: 0,
        cudaEventRecord=lambda evt, stream: 0,
        cudaEventSynchronize=lambda evt: 0,
        cudaEventElapsedTime=lambda ms, s, e: 0,
    )
    harness.configure_cudart(cudart)
    assert cudart.cudaMalloc.restype is ctypes.c_int

    core_lib = SimpleNamespace(
        iir2d_forward_cuda=lambda d_in, d_out, params: 0,
        iir2d_status_string=lambda code: b"ok",
    )
    harness.configure_core_lib(core_lib)
    assert core_lib.iir2d_status_string.restype is ctypes.c_char_p

    harness.cuda_check(cudart, 0, "noop")
    with pytest.raises(RuntimeError, match="failed: err"):
        harness.cuda_check(cudart, 3, "ctx")


def test_decode_nvcc_run_cmd_and_gpu_name(monkeypatch: pytest.MonkeyPatch) -> None:
    assert harness.decode_cuda_version(12080) == "12.8"
    assert harness.decode_cuda_version(0) == "unknown"
    assert harness.parse_nvcc_release("Cuda compilation tools, release 13.1, V13.1.2") == "13.1"
    assert harness.parse_nvcc_release("no release marker") == "unknown"

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda args, check, capture_output, text: SimpleNamespace(returncode=0, stdout="ok\n"),
    )
    assert harness.run_cmd(["dummy"]) == "ok"
    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda args, check, capture_output, text: SimpleNamespace(returncode=1, stdout=""),
    )
    assert harness.run_cmd(["dummy"]) == ""
    monkeypatch.setattr(harness.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()))
    assert harness.run_cmd(["dummy"]) == ""

    monkeypatch.setattr(harness, "run_cmd", lambda args: "GPU 1, 555.12")
    assert harness.get_gpu_name_and_driver() == ("GPU 1", "555.12")
    monkeypatch.setattr(harness, "run_cmd", lambda args: "")
    assert harness.get_gpu_name_and_driver() == ("unknown", "unknown")
    monkeypatch.setattr(harness, "run_cmd", lambda args: "just_name")
    assert harness.get_gpu_name_and_driver() == ("just_name", "unknown")
