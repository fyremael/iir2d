from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from scripts import check_perf_regression


def test_read_first_parses_first_row(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = "latency_ms_p50\n1.234000\n2.000000\n"

    class FakeFile:
        def __enter__(self) -> io.StringIO:
            return io.StringIO(payload)

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_open(self: Path, mode: str, encoding: str, newline: str) -> FakeFile:
        assert mode == "r"
        return FakeFile()

    monkeypatch.setattr(Path, "open", fake_open)
    row = check_perf_regression.read_first(Path("dummy.csv"))
    assert row["latency_ms_p50"] == "1.234000"


def test_read_first_raises_on_empty_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = "latency_ms_p50\n"

    class FakeFile:
        def __enter__(self) -> io.StringIO:
            return io.StringIO(payload)

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: FakeFile())
    with pytest.raises(RuntimeError):
        check_perf_regression.read_first(Path("dummy.csv"))


def test_main_passes_when_regression_within_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter([{"latency_ms_p50": "1.100000"}, {"latency_ms_p50": "1.000000"}])
    monkeypatch.setattr(check_perf_regression, "read_first", lambda _path: next(values))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_perf_regression.py",
            "--current_csv",
            "current.csv",
            "--baseline_csv",
            "baseline.csv",
            "--metric",
            "latency_ms_p50",
            "--max_regression_pct",
            "15",
        ],
    )
    assert check_perf_regression.main() == 0


def test_main_fails_when_regression_exceeds_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter([{"latency_ms_p50": "1.300000"}, {"latency_ms_p50": "1.000000"}])
    monkeypatch.setattr(check_perf_regression, "read_first", lambda _path: next(values))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_perf_regression.py",
            "--current_csv",
            "current.csv",
            "--baseline_csv",
            "baseline.csv",
            "--metric",
            "latency_ms_p50",
            "--max_regression_pct",
            "15",
        ],
    )
    with pytest.raises(SystemExit):
        check_perf_regression.main()


def test_main_fails_when_baseline_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter([{"latency_ms_p50": "1.300000"}, {"latency_ms_p50": "0.000000"}])
    monkeypatch.setattr(check_perf_regression, "read_first", lambda _path: next(values))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_perf_regression.py",
            "--current_csv",
            "current.csv",
            "--baseline_csv",
            "baseline.csv",
            "--metric",
            "latency_ms_p50",
            "--max_regression_pct",
            "15",
        ],
    )
    with pytest.raises(RuntimeError):
        check_perf_regression.main()
