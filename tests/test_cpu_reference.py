from __future__ import annotations

import numpy as np
import pytest

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
