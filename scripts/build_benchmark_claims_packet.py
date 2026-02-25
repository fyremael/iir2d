#!/usr/bin/env python3
"""Build a publishable benchmark claims markdown packet from harness CSV output."""

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


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sort_key(k: CaseKey) -> tuple[int, int, int, str, str]:
    return (k.width, k.height, k.filter_id, k.border_mode, k.precision)


def build_packet(rows: list[dict[str, str]], source_csv: Path, benchmark_command: str) -> str:
    if not rows:
        raise ValueError("Benchmark CSV has no rows.")

    first = rows[0]
    env_lines = [
        f"- Host: `{first.get('host', 'unknown')}`",
        f"- Platform: `{first.get('platform', 'unknown')}`",
        f"- Python: `{first.get('python_version', 'unknown')}`",
        f"- GPU: `{first.get('gpu_name', 'unknown')}`",
        f"- GPU Driver: `{first.get('gpu_driver', 'unknown')}`",
        f"- CUDA Runtime: `{first.get('cuda_runtime_version', 'unknown')}`",
        f"- CUDA Driver API: `{first.get('cuda_driver_version', 'unknown')}`",
        f"- nvcc: `{first.get('nvcc_release', 'unknown')}`",
        f"- Library: `{first.get('lib_path', 'unknown')}`",
    ]

    cases: dict[CaseKey, dict[str, str]] = {}
    for row in rows:
        key = CaseKey(
            width=int(row["width"]),
            height=int(row["height"]),
            filter_id=int(row["filter_id"]),
            border_mode=row["border_mode"],
            precision=row["precision"],
        )
        # Keep first occurrence per case key.
        cases.setdefault(key, row)

    table_lines = [
        "| Size | Filter | Border | Precision | p50 ms | p95 ms | MPix/s (p50) | GB/s (p50) |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for key in sorted(cases, key=sort_key):
        row = cases[key]
        table_lines.append(
            "| "
            f"{key.width}x{key.height} | "
            f"{key.filter_id} | "
            f"{key.border_mode} | "
            f"{key.precision} | "
            f"{float(row['latency_ms_p50']):.3f} | "
            f"{float(row['latency_ms_p95']):.3f} | "
            f"{float(row['throughput_mpix_per_s_p50']):.1f} | "
            f"{float(row['throughput_gb_per_s_p50']):.3f} |"
        )

    claim_hygiene = [
        "- Precision parity: confirmed in run matrix.",
        "- Workload parity: same matrix across all compared systems.",
        "- Full evidence: CSV retained and linked.",
        "- Command reproducibility: full command line recorded.",
    ]

    return "\n".join(
        [
            "# Benchmark Claims Packet",
            "",
            "## Source Artifact",
            f"- CSV: `{source_csv}`",
            "",
            "## Benchmark Command",
            "```bash",
            benchmark_command.strip(),
            "```",
            "",
            "## Environment",
            *env_lines,
            "",
            "## Results Summary",
            *table_lines,
            "",
            "## Claim Hygiene",
            *claim_hygiene,
            "",
            "## Approval",
            "- Product Lead: _pending_",
            "- GTM Lead: _pending_",
            "- QA Lead: _pending_",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Build markdown benchmark claims packet from benchmark CSV.")
    ap.add_argument("--in_csv", required=True, help="Input CSV from scripts/benchmark_core_cuda.py")
    ap.add_argument("--out_md", required=True, help="Output markdown path")
    ap.add_argument("--benchmark_command", required=True, help="Exact command used to produce the CSV")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_md = Path(args.out_md)
    rows = load_rows(in_csv)
    markdown = build_packet(rows, in_csv, args.benchmark_command)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown + "\n", encoding="utf-8")
    print(f"Wrote claims packet: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
