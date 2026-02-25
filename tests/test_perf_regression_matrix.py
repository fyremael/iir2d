from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scripts.check_perf_regression_matrix as matrix


def _row(
    width: int,
    height: int,
    filter_id: int,
    border_mode: str,
    precision: str,
    metric_value: float,
) -> dict[str, str]:
    return {
        "width": str(width),
        "height": str(height),
        "filter_id": str(filter_id),
        "border_mode": border_mode,
        "precision": precision,
        "latency_ms_p50": f"{metric_value:.6f}",
        "throughput_mpix_per_s_p50": f"{metric_value:.6f}",
    }


def test_compare_lower_is_better() -> None:
    base = [_row(512, 512, 1, "mirror", "f32", 1.0)]
    cur = [_row(512, 512, 1, "mirror", "f32", 1.1)]
    rows = matrix.compare(cur, base, metric="latency_ms_p50", direction="lower_is_better")
    assert len(rows) == 1
    assert rows[0].regression_pct == pytest.approx(10.0)


def test_compare_higher_is_better() -> None:
    base = [_row(512, 512, 1, "mirror", "f32", 100.0)]
    cur = [_row(512, 512, 1, "mirror", "f32", 90.0)]
    rows = matrix.compare(cur, base, metric="throughput_mpix_per_s_p50", direction="higher_is_better")
    assert rows[0].regression_pct == pytest.approx(10.0)


def test_compare_mismatched_case_keys_raises() -> None:
    base = [_row(512, 512, 1, "mirror", "f32", 1.0)]
    cur = [_row(1024, 1024, 1, "mirror", "f32", 1.0)]
    with pytest.raises(RuntimeError):
        matrix.compare(cur, base, metric="latency_ms_p50", direction="lower_is_better")


def test_report_contains_case_and_worst_regression() -> None:
    base = [_row(512, 512, 1, "mirror", "f32", 1.0)]
    cur = [_row(512, 512, 1, "mirror", "f32", 1.05)]
    rows = matrix.compare(cur, base, metric="latency_ms_p50", direction="lower_is_better")
    report = matrix.build_report(rows, metric="latency_ms_p50", threshold=20.0, direction="lower_is_better")
    assert "Benchmark Trend Report" in report
    assert "512x512 f1 mirror/f32" in report
    assert "Worst regression" in report


def test_main_pass_and_fail_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    base = [_row(512, 512, 1, "mirror", "f32", 1.0)]
    cur_pass = [_row(512, 512, 1, "mirror", "f32", 1.1)]
    cur_fail = [_row(512, 512, 1, "mirror", "f32", 1.4)]

    monkeypatch.setattr(matrix, "load_rows", lambda p: cur_pass if "cur" in str(p) else base)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_perf_regression_matrix.py",
            "--current_csv",
            "cur.csv",
            "--baseline_csv",
            "base.csv",
            "--metric",
            "latency_ms_p50",
            "--max_regression_pct",
            "15",
            "--out_report",
            "",
        ],
    )
    assert matrix.main() == 0

    monkeypatch.setattr(matrix, "load_rows", lambda p: cur_fail if "cur" in str(p) else base)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_perf_regression_matrix.py",
            "--current_csv",
            "cur.csv",
            "--baseline_csv",
            "base.csv",
            "--metric",
            "latency_ms_p50",
            "--max_regression_pct",
            "15",
            "--out_report",
            "",
        ],
    )
    with pytest.raises(SystemExit):
        matrix.main()


def test_main_writes_report(monkeypatch: pytest.MonkeyPatch) -> None:
    base = [_row(512, 512, 1, "mirror", "f32", 1.0)]
    cur = [_row(512, 512, 1, "mirror", "f32", 1.1)]
    monkeypatch.setattr(matrix, "load_rows", lambda p: cur if "cur" in str(p) else base)

    captured: dict[str, str] = {}

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
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_perf_regression_matrix.py",
            "--current_csv",
            "cur.csv",
            "--baseline_csv",
            "base.csv",
            "--metric",
            "latency_ms_p50",
            "--max_regression_pct",
            "15",
            "--out_report",
            "report.md",
        ],
    )
    assert matrix.main() == 0
    assert captured["path"].endswith("report.md")
    assert "Benchmark Trend Report" in captured["text"]
