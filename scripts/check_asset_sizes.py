#!/usr/bin/env python3
"""Fail when tracked image assets exceed a maximum size."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_MB = 25.0
DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class AssetEntry:
    path: Path
    size_bytes: int


def list_tracked_files(repo_root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("Failed to list tracked files via git ls-files.")
    raw = proc.stdout.decode("utf-8", errors="strict")
    rel_paths = [part for part in raw.split("\0") if part]
    return [repo_root / rel for rel in rel_paths]


def collect_image_assets(tracked_files: list[Path], extensions: tuple[str, ...]) -> list[AssetEntry]:
    ext_set = {ext.lower() for ext in extensions}
    assets: list[AssetEntry] = []
    for path in tracked_files:
        if path.suffix.lower() not in ext_set:
            continue
        if not path.exists():
            continue
        assets.append(AssetEntry(path=path, size_bytes=path.stat().st_size))
    return assets


def find_violations(assets: list[AssetEntry], max_bytes: int) -> list[AssetEntry]:
    return [asset for asset in assets if asset.size_bytes > max_bytes]


def format_mib(size_bytes: int) -> str:
    return f"{size_bytes / (1024.0 * 1024.0):.2f} MiB"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate tracked image assets stay below a size threshold.")
    ap.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--max_mb", type=float, default=DEFAULT_MAX_MB)
    ap.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated list of image extensions to enforce.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    max_bytes = int(args.max_mb * 1024.0 * 1024.0)
    extensions = tuple(ext.strip().lower() for ext in args.extensions.split(",") if ext.strip())
    tracked_files = list_tracked_files(repo_root)
    assets = collect_image_assets(tracked_files, extensions)
    violations = sorted(find_violations(assets, max_bytes), key=lambda a: a.size_bytes, reverse=True)

    print(
        f"Asset size policy: max={args.max_mb:.2f} MiB, "
        f"tracked_images={len(assets)}, violations={len(violations)}"
    )
    if not violations:
        return 0

    for asset in violations:
        rel = asset.path.relative_to(repo_root)
        print(f"  {rel} -> {format_mib(asset.size_bytes)}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
