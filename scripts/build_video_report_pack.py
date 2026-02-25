#!/usr/bin/env python3
"""Build a partner-friendly markdown report from video benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


def parse_csv_path_list(raw: str) -> list[Path]:
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def read_first_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV had no rows: {path}")
    return rows[0]


def file_size_mib(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def build_report(
    benchmark_rows: list[tuple[Path, dict[str, str]]],
    quality_rows: list[tuple[Path, dict[str, str]]],
    clips: list[Path],
) -> str:
    lines: list[str] = []
    lines.append("# Video Report Pack")
    lines.append("")
    lines.append(f"- generated_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    lines.append("## Benchmark Summary")
    if not benchmark_rows:
        lines.append("- no benchmark CSV supplied")
    else:
        for path, row in benchmark_rows:
            lines.append(f"- source: `{path}`")
            lines.append(
                "  "
                + ", ".join(
                    [
                        f"timed_frames={row.get('timed_frames', 'n/a')}",
                        f"loop_ms_p50={row.get('loop_ms_p50', 'n/a')}",
                        f"loop_ms_p95={row.get('loop_ms_p95', 'n/a')}",
                        f"timed_fps={row.get('timed_fps', 'n/a')}",
                        f"timed_mpix_per_s={row.get('timed_mpix_per_s', 'n/a')}",
                    ]
                )
            )
    lines.append("")

    lines.append("## Quality Summary")
    if not quality_rows:
        lines.append("- no quality CSV supplied")
    else:
        for path, row in quality_rows:
            lines.append(f"- source: `{path}`")
            lines.append(
                "  "
                + ", ".join(
                    [
                        f"frames={row.get('frames', 'n/a')}",
                        f"psnr_db_mean={row.get('psnr_db_mean', 'n/a')}",
                        f"ssim_mean={row.get('ssim_mean', 'n/a')}",
                        f"temporal_delta_mean={row.get('temporal_delta_mean', 'n/a')}",
                    ]
                )
            )
    lines.append("")

    lines.append("## Media Assets")
    if not clips:
        lines.append("- no clips supplied")
    else:
        for clip in clips:
            lines.append(f"- `{clip}` ({file_size_mib(clip):.2f} MiB)")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build markdown report pack from benchmark/quality/video artifacts.")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--benchmark_csvs", default="")
    ap.add_argument("--quality_csvs", default="")
    ap.add_argument("--clips", default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_csvs = parse_csv_path_list(args.benchmark_csvs)
    quality_csvs = parse_csv_path_list(args.quality_csvs)
    clips = parse_csv_path_list(args.clips)

    for path in [*benchmark_csvs, *quality_csvs, *clips]:
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

    benchmark_rows = [(path, read_first_row(path)) for path in benchmark_csvs]
    quality_rows = [(path, read_first_row(path)) for path in quality_csvs]
    report = build_report(benchmark_rows=benchmark_rows, quality_rows=quality_rows, clips=clips)

    out_path = Path(args.out_md).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote report pack: {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
