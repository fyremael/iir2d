#!/usr/bin/env python3
"""Fail CI if benchmark regression exceeds threshold."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_first(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        row = next(r, None)
        if row is None:
            raise RuntimeError(f"No rows in CSV: {path}")
        return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current_csv", required=True)
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--metric", default="latency_ms_p50")
    ap.add_argument("--max_regression_pct", type=float, default=20.0)
    args = ap.parse_args()

    current = read_first(Path(args.current_csv))
    baseline = read_first(Path(args.baseline_csv))

    c = float(current[args.metric])
    b = float(baseline[args.metric])
    if b <= 0:
        raise RuntimeError("Baseline metric must be > 0")

    regression_pct = ((c - b) / b) * 100.0
    print(
        f"metric={args.metric} baseline={b:.6f} current={c:.6f} "
        f"regression_pct={regression_pct:.2f} threshold={args.max_regression_pct:.2f}"
    )
    if regression_pct > args.max_regression_pct:
        raise SystemExit(
            f"Regression exceeded threshold: {regression_pct:.2f}% > {args.max_regression_pct:.2f}%"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
