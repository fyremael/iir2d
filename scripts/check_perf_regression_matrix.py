#!/usr/bin/env python3
"""Compare full benchmark matrices and fail on regressions over threshold."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaseKey:
    width: int
    height: int
    filter_id: int
    border_mode: str
    precision: str


@dataclass(frozen=True)
class ComparisonRow:
    key: CaseKey
    baseline: float
    current: float
    regression_pct: float


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_key(row: dict[str, str]) -> CaseKey:
    return CaseKey(
        width=int(row["width"]),
        height=int(row["height"]),
        filter_id=int(row["filter_id"]),
        border_mode=row["border_mode"],
        precision=row["precision"],
    )


def index_rows(rows: list[dict[str, str]]) -> dict[CaseKey, dict[str, str]]:
    out: dict[CaseKey, dict[str, str]] = {}
    for row in rows:
        key = make_key(row)
        if key in out:
            raise RuntimeError(f"Duplicate case in CSV: {key}")
        out[key] = row
    return out


def compare(
    current_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    metric: str,
    direction: str,
) -> list[ComparisonRow]:
    current = index_rows(current_rows)
    baseline = index_rows(baseline_rows)
    if set(current) != set(baseline):
        missing_in_current = sorted(set(baseline) - set(current), key=lambda k: (k.width, k.height, k.filter_id, k.border_mode, k.precision))
        missing_in_baseline = sorted(set(current) - set(baseline), key=lambda k: (k.width, k.height, k.filter_id, k.border_mode, k.precision))
        raise RuntimeError(
            "Case-key mismatch between current and baseline matrices.\n"
            f"missing_in_current={missing_in_current}\n"
            f"missing_in_baseline={missing_in_baseline}"
        )

    out: list[ComparisonRow] = []
    for key in sorted(current, key=lambda k: (k.width, k.height, k.filter_id, k.border_mode, k.precision)):
        c = float(current[key][metric])
        b = float(baseline[key][metric])
        if b <= 0.0:
            raise RuntimeError(f"Baseline metric must be > 0 for case {key} metric={metric}, got {b}")
        raw = ((c - b) / b) * 100.0
        regression_pct = raw if direction == "lower_is_better" else -raw
        out.append(ComparisonRow(key=key, baseline=b, current=c, regression_pct=regression_pct))
    return out


def build_report(
    rows: list[ComparisonRow],
    metric: str,
    threshold: float,
    direction: str,
) -> str:
    lines = [
        "# Benchmark Trend Report",
        "",
        f"- Metric: `{metric}`",
        f"- Direction: `{direction}`",
        f"- Regression threshold: `{threshold:.2f}%`",
        "",
        "| Case | Baseline | Current | Regression % |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        k = row.key
        case = f"{k.width}x{k.height} f{k.filter_id} {k.border_mode}/{k.precision}"
        lines.append(
            f"| {case} | {row.baseline:.6f} | {row.current:.6f} | {row.regression_pct:.2f} |"
        )
    worst = max(rows, key=lambda r: r.regression_pct)
    lines += [
        "",
        f"- Worst regression: `{worst.regression_pct:.2f}%` at `{worst.key}`",
        f"- Pass condition: worst regression <= `{threshold:.2f}%`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current_csv", required=True)
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--metric", default="latency_ms_p50")
    ap.add_argument(
        "--direction",
        default="lower_is_better",
        choices=("lower_is_better", "higher_is_better"),
        help="Whether larger metric values are regressions (lower_is_better) or improvements (higher_is_better).",
    )
    ap.add_argument("--max_regression_pct", type=float, default=20.0)
    ap.add_argument("--out_report", default="")
    args = ap.parse_args()

    current_rows = load_rows(Path(args.current_csv))
    baseline_rows = load_rows(Path(args.baseline_csv))
    comp = compare(
        current_rows=current_rows,
        baseline_rows=baseline_rows,
        metric=args.metric,
        direction=args.direction,
    )
    worst = max(comp, key=lambda r: r.regression_pct)
    print(
        f"metric={args.metric} direction={args.direction} "
        f"worst_regression_pct={worst.regression_pct:.2f} threshold={args.max_regression_pct:.2f}"
    )
    if args.out_report:
        report = build_report(comp, args.metric, args.max_regression_pct, args.direction)
        out = Path(args.out_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"wrote_report={out}")

    if worst.regression_pct > args.max_regression_pct:
        raise SystemExit(
            f"Regression exceeded threshold: worst={worst.regression_pct:.2f}% > {args.max_regression_pct:.2f}%"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
