from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

import scripts.core_harness as harness
import scripts.validate_cuda_cpu_matrix as validate


def test_max_diff_summary() -> None:
    expected = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    actual = np.array([[0.0, 1.0], [2.0, 3.1]], dtype=np.float32)
    max_abs, max_rel, loc, exp_v, got_v = validate.max_diff_summary(expected, actual)
    assert max_abs == pytest.approx(0.1, rel=0, abs=1e-6)
    assert max_rel == pytest.approx(0.1 / 3.0, rel=0, abs=1e-6)
    assert loc == (1, 1)
    assert exp_v == pytest.approx(3.0)
    assert got_v == pytest.approx(3.1)


def test_main_pass_and_fail_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    case = harness.Case(width=8, height=4, filter_id=1, border_mode="mirror", precision="f32")
    monkeypatch.setattr(validate, "parse_case_matrix", lambda **_: [case])
    monkeypatch.setattr(validate, "load_core_library", lambda _: (object(), Path("fake.dll")))
    monkeypatch.setattr(validate, "find_cudart", lambda: object())
    monkeypatch.setattr(validate, "configure_core_lib", lambda _: None)
    monkeypatch.setattr(validate, "configure_cudart", lambda _: None)

    def fake_ref(
        image: np.ndarray,
        filter_id: int,
        border_mode: str,
        border_const: float,
        precision: str,
    ) -> np.ndarray:
        assert filter_id == case.filter_id
        assert border_mode == case.border_mode
        assert border_const == pytest.approx(0.125)
        assert precision == case.precision
        return np.array(image, copy=True)

    monkeypatch.setattr(validate, "iir2d_cpu_reference", fake_ref)
    monkeypatch.setattr(
        validate,
        "run_cuda_forward",
        lambda cudart, core_lib, host_in, params: np.array(host_in, copy=True),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_cuda_cpu_matrix.py",
            "--sizes",
            "8x4",
            "--filter_ids",
            "1",
            "--border_modes",
            "mirror",
            "--precisions",
            "f32",
        ],
    )
    assert validate.main() == 0

    monkeypatch.setattr(
        validate,
        "run_cuda_forward",
        lambda cudart, core_lib, host_in, params: np.array(host_in, copy=True) + np.float32(1.0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_cuda_cpu_matrix.py",
            "--sizes",
            "8x4",
            "--filter_ids",
            "1",
            "--border_modes",
            "mirror",
            "--precisions",
            "f32",
            "--max_failures",
            "1",
        ],
    )
    assert validate.main() == 1


def test_run_cuda_forward_success_and_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validate, "cuda_check", lambda cudart, code, context: None)

    class FakeCudart:
        def __init__(self) -> None:
            self._alloc_counter = 100
            self.freed: list[int] = []

        def cudaMalloc(self, ptr: ctypes.c_void_p, size: int) -> int:
            self._alloc_counter += 1
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(self._alloc_counter)
            return 0

        def cudaMemcpy(self, dst: ctypes.c_void_p, src: ctypes.c_void_p, size: int, kind: int) -> int:
            return 0

        def cudaDeviceSynchronize(self) -> int:
            return 0

        def cudaFree(self, ptr: ctypes.c_void_p) -> int:
            self.freed.append(int(ptr.value))
            return 0

    class FakeCore:
        def __init__(self, status: int) -> None:
            self._status = status

        def iir2d_forward_cuda(self, d_in: ctypes.c_void_p, d_out: ctypes.c_void_p, params: ctypes.c_void_p) -> int:
            return self._status

        @staticmethod
        def iir2d_status_string(code: int) -> bytes:
            return b"core fail"

    host_in = np.ones((4, 8), dtype=np.float32)
    params = harness.IIR2D_Params(
        width=8,
        height=4,
        filter_id=1,
        border_mode=harness.BORDER_MAP["mirror"],
        border_const=0.0,
        precision=harness.PRECISION_MAP["f32"],
    )
    cudart = FakeCudart()

    out = validate.run_cuda_forward(cudart, FakeCore(status=0), host_in, params)
    assert out.shape == host_in.shape
    assert len(cudart.freed) == 2

    with pytest.raises(RuntimeError, match="iir2d_forward_cuda failed"):
        validate.run_cuda_forward(FakeCudart(), FakeCore(status=-3), host_in, params)
