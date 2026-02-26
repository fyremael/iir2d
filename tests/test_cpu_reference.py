from __future__ import annotations

import sys

import numpy as np
import pytest

import scripts.iir2d_cpu_reference as cpu_ref
from scripts.iir2d_cpu_reference import iir2d_cpu_reference


@pytest.mark.parametrize("filter_id", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize(
    ("precision", "dtype"),
    [("f32", np.float32), ("mixed", np.float32), ("f64", np.float64)],
)
def test_reference_shape_dtype_and_finite(filter_id: int, precision: str, dtype: np.dtype) -> None:
    rng = np.random.default_rng(42 + filter_id)
    x = rng.random((9, 11), dtype=np.float64 if precision == "f64" else np.float32)
    y = iir2d_cpu_reference(
        x,
        filter_id=filter_id,
        border_mode="mirror",
        border_const=0.125,
        precision=precision,
    )
    assert y.shape == x.shape
    assert y.dtype == dtype
    assert np.isfinite(y).all()


def test_border_mode_string_and_int_are_equivalent() -> None:
    rng = np.random.default_rng(123)
    x = rng.random((7, 13), dtype=np.float32)
    y_text = iir2d_cpu_reference(x, filter_id=4, border_mode="mirror", precision="f32")
    y_int = iir2d_cpu_reference(x, filter_id=4, border_mode=1, precision="f32")
    assert np.allclose(y_text, y_int, rtol=0.0, atol=0.0)


def test_filter3_and_filter8_match_current_runtime_contract() -> None:
    rng = np.random.default_rng(7)
    x = rng.random((8, 10), dtype=np.float32)
    y3 = iir2d_cpu_reference(x, filter_id=3, border_mode="wrap", precision="f32")
    y8 = iir2d_cpu_reference(x, filter_id=8, border_mode="wrap", precision="f32")
    assert np.allclose(y3, y8, rtol=0.0, atol=0.0)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        iir2d_cpu_reference(np.zeros((2, 3, 4), dtype=np.float32), filter_id=1)
    with pytest.raises(ValueError):
        iir2d_cpu_reference(np.zeros((4, 4), dtype=np.float32), filter_id=99)
    with pytest.raises(ValueError):
        iir2d_cpu_reference(np.zeros((4, 4), dtype=np.float32), filter_id=1, border_mode="bad")
    with pytest.raises(ValueError):
        iir2d_cpu_reference(np.zeros((4, 4), dtype=np.float32), filter_id=1, precision="bad")


def test_normalize_border_and_precision_invalid_integer_branches() -> None:
    with pytest.raises(ValueError):
        cpu_ref._normalize_border_mode(99)  # noqa: SLF001
    with pytest.raises(ValueError):
        cpu_ref._normalize_precision(99)  # noqa: SLF001


def test_border_sample_all_modes_and_out_of_bounds() -> None:
    row = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    assert cpu_ref._border_sample(row, 1, cpu_ref.BORDER_MAP["mirror"], np.float32(7.0)) == np.float32(20.0)  # noqa: SLF001
    assert cpu_ref._border_sample(row, -1, cpu_ref.BORDER_MAP["constant"], np.float32(7.0)) == np.float32(7.0)  # noqa: SLF001
    assert cpu_ref._border_sample(row, -7, cpu_ref.BORDER_MAP["clamp"], np.float32(7.0)) == np.float32(10.0)  # noqa: SLF001
    assert cpu_ref._border_sample(row, -4, cpu_ref.BORDER_MAP["wrap"], np.float32(7.0)) == np.float32(30.0)  # noqa: SLF001
    assert cpu_ref._border_sample(row, -1, cpu_ref.BORDER_MAP["mirror"], np.float32(7.0)) == np.float32(10.0)  # noqa: SLF001


def test_internal_row_biquad_and_statespace_helpers_return_expected_shape() -> None:
    x = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    y_bq = cpu_ref._row_biquad(  # noqa: SLF001
        x,
        np.float32(0.2),
        np.float32(0.2),
        np.float32(0.2),
        np.float32(0.3),
        np.float32(-0.1),
        cpu_ref.BORDER_MAP["mirror"],
        np.float32(0.0),
        np.float32,
        np.dtype(np.float32),
    )
    y_ss = cpu_ref._row_statespace(  # noqa: SLF001
        x,
        np.float32(0.2),
        np.float32(0.2),
        np.float32(0.2),
        np.float32(0.3),
        np.float32(-0.1),
        cpu_ref.BORDER_MAP["mirror"],
        np.float32(0.0),
        np.float32,
        np.dtype(np.float32),
    )
    assert y_bq.shape == x.shape
    assert y_ss.shape == x.shape
    assert np.isfinite(y_bq).all()
    assert np.isfinite(y_ss).all()


def test_scan_contract_matches_scalar_biquad_across_block_boundaries() -> None:
    rng = np.random.default_rng(19)
    x = rng.random(521, dtype=np.float64).astype(np.float32)
    y_scan = cpu_ref._row_biquad_scan_contract(  # noqa: SLF001
        x,
        np.float32(0.2),
        np.float32(0.2),
        np.float32(0.2),
        np.float32(0.3),
        np.float32(-0.1),
        cpu_ref.BORDER_MAP["mirror"],
        np.float32(0.0),
        np.float32,
        np.dtype(np.float32),
        block_width=256,
    )
    y_ref = cpu_ref._row_biquad(  # noqa: SLF001
        x,
        np.float32(0.2),
        np.float32(0.2),
        np.float32(0.2),
        np.float32(0.3),
        np.float32(-0.1),
        cpu_ref.BORDER_MAP["mirror"],
        np.float32(0.0),
        np.float32,
        np.dtype(np.float32),
    )
    np.testing.assert_allclose(y_scan, y_ref, rtol=1e-6, atol=1e-6)


def test_apply_rows_invalid_filter_branch_raises() -> None:
    with pytest.raises(ValueError):
        cpu_ref._apply_rows(  # noqa: SLF001
            np.ones((2, 3), dtype=np.float32),
            99,
            cpu_ref.BORDER_MAP["mirror"],
            np.float32(0.0),
            np.float32,
            np.dtype(np.float32),
        )


def test_reference_rejects_empty_shape() -> None:
    with pytest.raises(ValueError):
        iir2d_cpu_reference(np.zeros((0, 4), dtype=np.float32), filter_id=1)


def test_cli_main_valid_and_invalid_size(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "iir2d_cpu_reference.py",
            "--size",
            "7x5",
            "--filter_id",
            "6",
            "--border_mode",
            "mirror",
            "--precision",
            "f64",
            "--seed",
            "2",
        ],
    )
    assert cpu_ref.main() == 0
    out = capsys.readouterr().out
    assert "ok: shape=(5, 7)" in out

    monkeypatch.setattr(sys, "argv", ["iir2d_cpu_reference.py", "--size", "bad"])
    with pytest.raises(ValueError):
        cpu_ref.main()
